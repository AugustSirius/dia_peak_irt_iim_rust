mod utils;
mod cache;
mod processing;

use std::sync::{Arc, Mutex};
use std::io::BufWriter;

use cache::CacheManager;
use utils::{
    read_timstof_data, build_indexed_data, read_parquet_with_polars,
    library_records_to_dataframe, merge_library_and_report, get_unique_precursor_ids, 
    process_library_fast, create_rt_im_dicts, build_lib_matrix, build_precursors_matrix_step1, 
    build_precursors_matrix_step2, build_range_matrix_step3, build_precursors_matrix_step3, 
    build_frag_info, LibCols, PrecursorLibData, prepare_precursor_lib_data,
    extract_unique_rt_im_values, save_unique_values_to_files, UniqueValues
};
use processing::{
    FastChunkFinder, build_rt_intensity_matrix_optimized, prepare_precursor_features,
    calculate_mz_range, extract_ms2_data, build_mask_matrices, extract_aligned_rt_values,
    reshape_and_combine_matrices, process_single_precursor_rsm
};

use rayon::prelude::*;
use std::{error::Error, path::Path, time::Instant, env, fs::File};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis};
use polars::prelude::*;
use ndarray_npy::{NpzWriter, write_npy};

// New struct to hold RSM results
#[derive(Debug)]
pub struct RSMPrecursorResults {
    pub index: usize,
    pub precursor_id: String,
    pub rsm_matrix: Array4<f32>,  // Shape: [1, 5, 72, 396]
    pub all_rt: Vec<f32>,          // 396 RT values
}

fn main() -> Result<(), Box<dyn Error>> {
    // Fixed parameters
    let batch_size = 1000;
    let parallel_threads = 32;
    let output_dir = "output";
    
    // let d_folder = "/Users/augustsirius/Desktop/raw_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    // let report_file_path = "/Users/augustsirius/Desktop/dia_peak_irt_iim_rust/20250730_v5.3_TPHPlib_frag1025_swissprot_final_all_from_Yueliang.parquet";
    // let lib_file_path = "/Users/augustsirius/Desktop/raw_data/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";

    let d_folder = "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    let report_file_path = "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/dia_peak_irt_iim_rust/raw_data/20250730_v5.3_TPHPlib_frag1025_swissprot_final_all_from_Yueliang.parquet";
    let lib_file_path = "/storage/guotiannanLab/wangshuaiyao/006.DIABERT_TimsTOF_Rust/dia_peak_irt_iim_rust/raw_data/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(parallel_threads)
        .build_global()
        .unwrap();
    
    println!("Initialized parallel processing with {} threads", parallel_threads);
    println!("Batch size: {}", batch_size);
    println!("Output directory: {}", output_dir);
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // ================================ DATA LOADING AND INDEXING ================================
    let cache_manager = CacheManager::new();
    
    println!("\n========== DATA PREPARATION PHASE ==========");
    let total_start = Instant::now();
        
    let (ms1_indexed, ms2_indexed_pairs) = if cache_manager.is_cache_valid(d_path) {
        println!("Found valid cache, loading indexed data directly...");
        let cache_load_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = cache_manager.load_indexed_data(d_path)?;
        println!("Cache loading time: {:.5} seconds", cache_load_start.elapsed().as_secs_f32());
        (ms1_indexed, ms2_indexed_pairs)
    } else {
        println!("Cache invalid or non-existent, reading TimsTOF data...");
        
        let raw_data_start = Instant::now();
        let raw_data = read_timstof_data(d_path)?;
        println!("Raw data reading time: {:.5} seconds", raw_data_start.elapsed().as_secs_f32());
        println!("  - MS1 data points: {}", raw_data.ms1_data.mz_values.len());
        println!("  - MS2 windows: {}", raw_data.ms2_windows.len());
        
        println!("\nBuilding indexed data structures...");
        let index_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = build_indexed_data(raw_data)?;
        println!("Index building time: {:.5} seconds", index_start.elapsed().as_secs_f32());
        
        let cache_save_start = Instant::now();
        cache_manager.save_indexed_data(d_path, &ms1_indexed, &ms2_indexed_pairs)?;
        println!("Cache saving time: {:.5} seconds", cache_save_start.elapsed().as_secs_f32());
        
        (ms1_indexed, ms2_indexed_pairs)
    };
    
    println!("Total data preparation time: {:.5} seconds", total_start.elapsed().as_secs_f32());
    
    let finder = FastChunkFinder::new(ms2_indexed_pairs)?;
    
    // ================================ LIBRARY AND REPORT LOADING ================================
    println!("\n========== LIBRARY AND REPORT PROCESSING ==========");
    let lib_processing_start = Instant::now();

    let library_records = process_library_fast(lib_file_path)?;
    let library_df = library_records_to_dataframe(library_records.clone())?;

    let report_df = read_parquet_with_polars(report_file_path)?;
    let diann_result = merge_library_and_report(library_df, report_df)?;
    
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    println!("Library and report processing time: {:.5} seconds", lib_processing_start.elapsed().as_secs_f32());
    
    let device = "cpu";
    let frag_repeat_num = 5;
    
    // ================================ BATCH PRECURSOR PROCESSING ================================
    println!("\n========== BATCH PRECURSOR PROCESSING ==========");
    
    println!("\n[Step 1] Preparing library data for all precursors");
    let prep_start = Instant::now();
    
    let unique_precursor_ids: Vec<String> = diann_precursor_id_all
        .column("transition_group_id")?
        .str()?
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_string()))
        .collect();

    let total_unique_precursors = unique_precursor_ids.len();
    println!("\n========== LIBRARY STATISTICS ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    
    let lib_cols = LibCols::default();
    
    // Process all precursors (no max_precursors limit)
    let precursor_lib_data_list = prepare_precursor_lib_data(
        &library_records,
        &unique_precursor_ids,
        &assay_rt_kept_dict,
        &assay_im_kept_dict,
        &lib_cols,
        total_unique_precursors,  // Process all precursors
    )?;
    
    println!("  - Prepared data for {} precursors", precursor_lib_data_list.len());
    println!("  - Preparation time: {:.5} seconds", prep_start.elapsed().as_secs_f32());
    
    drop(library_records);
    println!("  - Released library_records from memory");
    
    // Process in batches
    let total_batches = (precursor_lib_data_list.len() + batch_size - 1) / batch_size;
    println!("\n[Step 2] Processing {} precursors in {} batches", 
             precursor_lib_data_list.len(), total_batches);
    
    for batch_idx in 0..total_batches {
        let batch_start_idx = batch_idx * batch_size;
        let batch_end_idx = ((batch_idx + 1) * batch_size).min(precursor_lib_data_list.len());
        let batch_precursors = &precursor_lib_data_list[batch_start_idx..batch_end_idx];
        
        println!("\n========== Processing Batch {}/{} ==========", batch_idx + 1, total_batches);
        println!("Precursors {} to {} (total: {})", 
                 batch_start_idx + 1, batch_end_idx, batch_precursors.len());
        
        let batch_start = Instant::now();
        
        use std::sync::atomic::{AtomicUsize, Ordering};
        let processed_count = Arc::new(AtomicUsize::new(0));
        let batch_count = batch_precursors.len();
        
        let results_mutex = Arc::new(Mutex::new(Vec::new()));
        
        // Process batch in parallel
        batch_precursors
            .par_iter()
            .enumerate()
            .for_each(|(batch_internal_idx, precursor_data)| {
                let global_index = batch_start_idx + batch_internal_idx;
                
                let result = process_single_precursor_rsm(
                    precursor_data,
                    &ms1_indexed,
                    &finder,
                    frag_repeat_num,
                    device,
                );
                
                let current = processed_count.fetch_add(1, Ordering::SeqCst) + 1;
                
                match result {
                    Ok((precursor_id, rsm_matrix, all_rt)) => {
                        println!("[Batch {} - {}/{}] ✓ Successfully processed: {} (global index: {})", 
                                 batch_idx + 1, current, batch_count, precursor_id, global_index);
                        
                        let rsm_result = RSMPrecursorResults {
                            index: batch_internal_idx,
                            precursor_id: precursor_id.clone(),
                            rsm_matrix,
                            all_rt,
                        };
                        
                        let mut results = results_mutex.lock().unwrap();
                        results.push(rsm_result);
                    },
                    Err(e) => {
                        eprintln!("[Batch {} - {}/{}] ✗ Error processing {} (global index: {}): {}", 
                                  batch_idx + 1, current, batch_count, 
                                  precursor_data.precursor_id, global_index, e);
                    }
                }
            });
        
        let batch_elapsed = batch_start.elapsed();
        
        println!("\n========== SAVING BATCH {} RESULTS ==========", batch_idx + 1);
        let save_start = Instant::now();
        
        let mut results = Arc::try_unwrap(results_mutex).unwrap().into_inner().unwrap();
        
        // Sort results by original index to restore order
        results.sort_by_key(|r| r.index);
        
        // Save batch results
        save_batch_results_as_npy(&results, batch_precursors, batch_idx, &output_dir)?;
        
        println!("Batch {} save time: {:.5} seconds", batch_idx + 1, save_start.elapsed().as_secs_f32());
        
        println!("\n========== BATCH {} SUMMARY ==========", batch_idx + 1);
        println!("Processed: {} precursors", results.len());
        println!("Failed: {} precursors", batch_count - results.len());
        println!("Batch processing time: {:.5} seconds", batch_elapsed.as_secs_f32());
        println!("Average time per precursor: {:.5} seconds", 
                 batch_elapsed.as_secs_f32() / batch_count as f32);
    }
    
    println!("\n========== OVERALL PROCESSING SUMMARY ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    println!("Total processed: {} precursors", precursor_lib_data_list.len());
    println!("Processing mode: Parallel ({} threads)", parallel_threads);
    println!("Batch size: {}", batch_size);
    println!("Total batches: {}", total_batches);
    println!("Output directory: {}", output_dir);
    
    Ok(())
}

fn save_batch_results_as_npy(
    results: &[RSMPrecursorResults], 
    original_precursor_list: &[PrecursorLibData],
    batch_idx: usize,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    
    if results.is_empty() {
        return Err("No results to save".into());
    }
    
    // Create batch-specific filenames with simple numbering
    let batch_name = format!("batch_{}", batch_idx);
    let rsm_filename = format!("{}/{}_rsm.npy", output_dir, batch_name);
    let rt_filename = format!("{}/{}_rt_values.npy", output_dir, batch_name);
    let index_filename = format!("{}/{}_index.txt", output_dir, batch_name);
    
    // Create a map of index to result for quick lookup
    let mut result_map = std::collections::HashMap::new();
    for result in results {
        result_map.insert(result.index, result);
    }
    
    // Initialize the combined RSM matrix and RT values based on original order
    let n_original = original_precursor_list.len();
    let frag_repeat_num = 5;
    let n_fragments = 72; // MS1 + MS2 fragments
    let n_scans = 396;
    
    // Initialize with zeros
    let mut all_rsm_matrix = Array4::<f32>::zeros((n_original, frag_repeat_num, n_fragments, n_scans));
    let mut all_rt_values = Array2::<f32>::zeros((n_original, n_scans));
    let mut precursor_ids = Vec::with_capacity(n_original);
    let mut status_list = Vec::with_capacity(n_original);
    
    // Fill matrices in original order
    for (i, precursor_data) in original_precursor_list.iter().enumerate() {
        precursor_ids.push(precursor_data.precursor_id.clone());
        
        if let Some(result) = result_map.get(&i) {
            // Successfully processed - copy the RSM matrix
            all_rsm_matrix.slice_mut(s![i, .., .., ..]).assign(&result.rsm_matrix.slice(s![0, .., .., ..]));
            
            // Copy RT values
            for (j, &rt_val) in result.all_rt.iter().enumerate() {
                if j < n_scans {
                    all_rt_values[[i, j]] = rt_val;
                }
            }
            status_list.push("SUCCESS");
        } else {
            // Failed to process - keep as zeros
            status_list.push("FAILED");
        }
    }
    
    // Save RSM matrix
    println!("Saving RSM matrix to: {}", rsm_filename);
    println!("  Shape: [{}, {}, {}, {}]", n_original, frag_repeat_num, n_fragments, n_scans);
    write_npy(&rsm_filename, &all_rsm_matrix)?;
    
    // Save RT values
    println!("Saving RT values to: {}", rt_filename);
    println!("  Shape: [{}, {}]", n_original, n_scans);
    write_npy(&rt_filename, &all_rt_values)?;
    
    // Save index file
    println!("Saving index file to: {}", index_filename);
    let mut id_file = File::create(&index_filename)?;
    writeln!(id_file, "# Index file for RSM matrices and RT values - Batch {}", batch_idx)?;
    writeln!(id_file, "# Total precursors in batch: {}", n_original)?;
    writeln!(id_file, "# Successfully processed: {}", results.len())?;
    writeln!(id_file, "# Failed: {}", n_original - results.len())?;
    writeln!(id_file, "# RSM matrix shape: [{}, {}, {}, {}]", n_original, frag_repeat_num, n_fragments, n_scans)?;
    writeln!(id_file, "# RT values shape: [{}, {}]", n_original, n_scans)?;
    writeln!(id_file, "# Row_Index\tPrecursor_ID\tStatus")?;
    
    for (i, (id, status)) in precursor_ids.iter().zip(status_list.iter()).enumerate() {
        writeln!(id_file, "{}\t{}\t{}", i, id, status)?;
    }
    
    println!("\nSuccessfully saved batch {} files:", batch_idx);
    println!("  - RSM matrix: {}", rsm_filename);
    println!("  - RT values: {}", rt_filename);
    println!("  - Index file: {}", index_filename);
    
    Ok(())
}