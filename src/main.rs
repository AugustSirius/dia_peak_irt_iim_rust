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
    reshape_and_combine_matrices, process_single_precursor_compressed
};

use rayon::prelude::*;
use std::{error::Error, path::Path, time::Instant, env, fs::File};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis};
use polars::prelude::*;
use ndarray_npy::{NpzWriter, write_npy};

// Struct to hold compressed results with index for ordering
#[derive(Debug)]
pub struct CompressedPrecursorResults {
    pub index: usize,  // Add index to maintain order
    pub precursor_id: String,
    pub rt_counts: Array1<f32>,
    pub im_counts: Array1<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_precursors = 4; // Can be adjusted as needed
    let parallel_threads = 16; // Hardcoded parameter - change this to control thread count
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(parallel_threads)
        .build_global()
        .unwrap();
    
    println!("Initialized parallel processing with {} threads", parallel_threads);
    
    let args: Vec<String> = env::args().collect();
    
    if let Some(arg) = args.get(1) {
        match arg.as_str() {
            "--clear-cache" => {
                CacheManager::new().clear_cache()?;
                return Ok(());
            }
            "--cache-info" => {
                let cache_manager = CacheManager::new();
                let info = cache_manager.get_cache_info()?;
                if info.is_empty() {
                    println!("Cache is empty");
                } else {
                    println!("Cache files:");
                    for (name, _, size_str) in info {
                        println!("  {} - {}", name, size_str);
                    }
                }
                return Ok(());
            }
            _ => {}
        }
    }
    
    let d_folder = args.get(1).cloned().unwrap_or_else(|| {
        "/Users/augustsirius/Desktop/raw_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d".to_string()
    });
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // ================================ DATA LOADING AND INDEXING ================================
    let cache_manager = CacheManager::new();
    
    println!("\n========== DATA PREPARATION PHASE ==========");
    let total_start = Instant::now();
    
    let (ms1_indexed, ms2_indexed_pairs, ms2_rt_values, all_im_values) = if cache_manager.is_cache_valid(d_path) {
        println!("Found valid cache, loading indexed data directly...");
        let cache_load_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = cache_manager.load_indexed_data(d_path)?;
        println!("Cache loading time: {:.5} seconds", cache_load_start.elapsed().as_secs_f32());
        
        println!("\nReading raw data to extract unique RT and IM values...");
        let raw_data = read_timstof_data(d_path)?;
        let extract_start = Instant::now();
        let unique_values = extract_unique_rt_im_values(&raw_data);
        println!("Extraction time: {:.5} seconds", extract_start.elapsed().as_secs_f32());
        
        println!("\n========== UNIQUE VALUE STATISTICS ==========");
        println!("Unique MS2 RT values: {}", unique_values.ms2_rt_values.len());
        println!("Unique IM values (all): {}", unique_values.all_im_values.len());
        
        let unique_values_dir = "unique_values_output";
        println!("\nSaving unique values to '{}'...", unique_values_dir);
        save_unique_values_to_files(&unique_values, unique_values_dir)?;
        println!("Unique values saved successfully.");
        
        let ms2_rt_values = unique_values.ms2_rt_values;
        let all_im_values = unique_values.all_im_values;

        (ms1_indexed, ms2_indexed_pairs, ms2_rt_values, all_im_values)
    } else {
        println!("Cache invalid or non-existent, reading TimsTOF data...");
        
        let raw_data_start = Instant::now();
        let raw_data = read_timstof_data(d_path)?;
        println!("Raw data reading time: {:.5} seconds", raw_data_start.elapsed().as_secs_f32());
        println!("  - MS1 data points: {}", raw_data.ms1_data.mz_values.len());
        println!("  - MS2 windows: {}", raw_data.ms2_windows.len());
        
        println!("\nExtracting unique RT and IM values...");
        let extract_start = Instant::now();
        let unique_values = extract_unique_rt_im_values(&raw_data);
        println!("Extraction time: {:.5} seconds", extract_start.elapsed().as_secs_f32());
        
        println!("\n========== UNIQUE VALUE STATISTICS ==========");
        println!("Unique MS2 RT values: {}", unique_values.ms2_rt_values.len());
        println!("Unique IM values (all): {}", unique_values.all_im_values.len());
        
        let unique_values_dir = "unique_values_output";
        println!("\nSaving unique values to '{}'...", unique_values_dir);
        save_unique_values_to_files(&unique_values, unique_values_dir)?;
        println!("Unique values saved successfully.");
        
        let ms2_rt_values = unique_values.ms2_rt_values;
        let all_im_values = unique_values.all_im_values;
        
        println!("\nBuilding indexed data structures...");
        let index_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = build_indexed_data(raw_data)?;
        println!("Index building time: {:.5} seconds", index_start.elapsed().as_secs_f32());
        
        let cache_save_start = Instant::now();
        cache_manager.save_indexed_data(d_path, &ms1_indexed, &ms2_indexed_pairs)?;
        println!("Cache saving time: {:.5} seconds", cache_save_start.elapsed().as_secs_f32());
        
        (ms1_indexed, ms2_indexed_pairs, ms2_rt_values, all_im_values)
    };
    
    println!("Total data preparation time: {:.5} seconds", total_start.elapsed().as_secs_f32());
    
    let finder = FastChunkFinder::new(ms2_indexed_pairs)?;
    
    // ================================ LIBRARY AND REPORT LOADING ================================
    println!("\n========== LIBRARY AND REPORT PROCESSING ==========");
    let lib_processing_start = Instant::now();
    
    let lib_file_path = "/Users/augustsirius/Desktop/rust_for_iRT_iIM/fitter_rt_lib.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    
    let diann_result = library_records_to_dataframe(library_records.clone())?;
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    println!("Library and report processing time: {:.5} seconds", lib_processing_start.elapsed().as_secs_f32());
    
    let device = "cpu";
    let frag_repeat_num = 5;
    
    // ================================ BATCH PRECURSOR PROCESSING ================================
    println!("\n========== BATCH PRECURSOR PROCESSING ==========");
    
    println!("\n[Step 1] Preparing library data for batch processing");
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
    
    let precursor_lib_data_list = prepare_precursor_lib_data(
        &library_records,
        &unique_precursor_ids,
        &assay_rt_kept_dict,
        &assay_im_kept_dict,
        &lib_cols,
        max_precursors,
    )?;
    
    println!("  - Prepared data for {} precursors", precursor_lib_data_list.len());
    println!("  - Preparation time: {:.5} seconds", prep_start.elapsed().as_secs_f32());
    
    drop(library_records);
    println!("  - Released library_records from memory");
    
    println!("\n[Step 2] Processing individual precursors");
    
    let batch_start = Instant::now();
    
    use std::sync::atomic::{AtomicUsize, Ordering};
    let processed_count = Arc::new(AtomicUsize::new(0));
    let total_count = precursor_lib_data_list.len();
    
    let results_mutex = Arc::new(Mutex::new(Vec::new()));
    
    // Process with enumeration to maintain order
    precursor_lib_data_list
        .par_iter()
        .enumerate()  // Add enumeration to track original index
        .for_each(|(original_index, precursor_data)| {
            let result = process_single_precursor_compressed(
                precursor_data,
                &ms1_indexed,
                &finder,
                frag_repeat_num,
                device,
                &ms2_rt_values,
                &all_im_values,
            );
            
            let current = processed_count.fetch_add(1, Ordering::SeqCst) + 1;
            
            match result {
                Ok((precursor_id, rt_counts, im_counts)) => {
                    println!("[{}/{}] ✓ Successfully processed: {} (index: {})", 
                             current, total_count, precursor_id, original_index);
                    
                    let compressed_result = CompressedPrecursorResults {
                        index: original_index,  // Store original index
                        precursor_id: precursor_id.clone(),
                        rt_counts,
                        im_counts,
                    };
                    
                    let mut results = results_mutex.lock().unwrap();
                    results.push(compressed_result);
                },
                Err(e) => {
                    eprintln!("[{}/{}] ✗ Error processing {} (index: {}): {}", 
                              current, total_count, precursor_data.precursor_id, original_index, e);
                }
            }
        });
    
    let batch_elapsed = batch_start.elapsed();
    
    println!("\n========== SAVING RESULTS TO SEPARATE FILES ==========");
    let save_start = Instant::now();
    
    let mut results = Arc::try_unwrap(results_mutex).unwrap().into_inner().unwrap();
    
    // Sort results by original index to restore order
    results.sort_by_key(|r| r.index);
    
    // Save results as three separate files
    save_results_as_separate_files(&results, &precursor_lib_data_list)?;
    
    println!("Save time: {:.5} seconds", save_start.elapsed().as_secs_f32());
    
    println!("\n========== PROCESSING SUMMARY ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    println!("Successfully processed: {} precursors", results.len());
    println!("Processing mode: Parallel ({} threads)", parallel_threads);
    println!("Total batch processing time: {:.5} seconds", batch_elapsed.as_secs_f32());
    println!("Average time per precursor: {:.5} seconds", 
             batch_elapsed.as_secs_f32() / precursor_lib_data_list.len() as f32);
    
    Ok(())
}

fn save_results_as_separate_files(
    results: &[CompressedPrecursorResults], 
    original_precursor_list: &[PrecursorLibData]
) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    
    if results.is_empty() {
        return Err("No results to save".into());
    }
    
    // Get dimensions from successfully processed results
    let rt_len = results[0].rt_counts.len();
    let im_len = results[0].im_counts.len();
    
    // Verify all results have the same shape
    for result in results {
        if result.rt_counts.len() != rt_len {
            return Err("Inconsistent RT count lengths across precursors".into());
        }
        if result.im_counts.len() != im_len {
            return Err("Inconsistent IM count lengths across precursors".into());
        }
    }
    
    // Create a map of index to result for quick lookup
    let mut result_map = std::collections::HashMap::new();
    for result in results {
        result_map.insert(result.index, result);
    }
    
    // Initialize matrices and ID list based on original order
    let n_original = original_precursor_list.len();
    let mut all_rt_matrix = Array2::<f32>::zeros((n_original, rt_len));
    let mut all_im_matrix = Array2::<f32>::zeros((n_original, im_len));
    let mut precursor_ids = Vec::with_capacity(n_original);
    let mut status_list = Vec::with_capacity(n_original);
    
    // Fill matrices in original order
    for (i, precursor_data) in original_precursor_list.iter().enumerate() {
        precursor_ids.push(precursor_data.precursor_id.clone());
        
        if let Some(result) = result_map.get(&i) {
            // Successfully processed
            all_rt_matrix.slice_mut(s![i, ..]).assign(&result.rt_counts);
            all_im_matrix.slice_mut(s![i, ..]).assign(&result.im_counts);
            status_list.push("SUCCESS");
        } else {
            // Failed to process - keep as zeros
            status_list.push("FAILED");
        }
    }
    
    // Save RT matrix
    println!("Saving RT matrix to: all_rt_matrix.npy");
    write_npy("all_rt_matrix.npy", &all_rt_matrix)?;
    
    // Save IM matrix
    println!("Saving IM matrix to: all_im_matrix.npy");
    write_npy("all_im_matrix.npy", &all_im_matrix)?;
    
    // Save precursor IDs with status
    println!("Saving precursor IDs to: precursor_ids.txt");
    let mut id_file = File::create("precursor_ids.txt")?;
    writeln!(id_file, "# Precursor IDs corresponding to rows in all_rt_matrix.npy and all_im_matrix.npy")?;
    writeln!(id_file, "# Total precursors: {}", n_original)?;
    writeln!(id_file, "# Successfully processed: {}", results.len())?;
    writeln!(id_file, "# Failed: {}", n_original - results.len())?;
    writeln!(id_file, "# RT matrix shape: ({}, {})", n_original, rt_len)?;
    writeln!(id_file, "# IM matrix shape: ({}, {})", n_original, im_len)?;
    writeln!(id_file, "# Row_Index\tPrecursor_ID\tStatus")?;
    
    for (i, (id, status)) in precursor_ids.iter().zip(status_list.iter()).enumerate() {
        writeln!(id_file, "{}\t{}\t{}", i, id, status)?;
    }
    
    println!("\nSuccessfully saved:");
    println!("  - RT matrix: all_rt_matrix.npy (shape: {} x {})", n_original, rt_len);
    println!("  - IM matrix: all_im_matrix.npy (shape: {} x {})", n_original, im_len);
    println!("  - Precursor IDs: precursor_ids.txt ({} entries)", n_original);
    println!("  - Successfully processed: {} precursors", results.len());
    println!("  - Failed: {} precursors", n_original - results.len());
    
    Ok(())
}