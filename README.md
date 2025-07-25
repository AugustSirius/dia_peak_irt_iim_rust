# TimsTOF Data Processing Tool

A high-performance Rust application for processing TimsTOF mass spectrometry data, specifically designed for retention time (RT) and ion mobility (iM) analysis.

## Features

- **Fast TimsTOF Data Processing**: Efficiently reads and processes Bruker TimsTOF `.d` files
- **Parallel Processing**: Multi-threaded processing using Rayon for optimal performance
- **Smart Caching**: Intelligent caching system to avoid reprocessing data
- **RT/iM Matrix Generation**: Generates retention time and ion mobility count matrices
- **Library Integration**: Processes spectral libraries for precursor analysis
- **Memory Efficient**: Optimized memory usage for large datasets

## Dependencies

This project uses several key Rust crates:
- `timsrust`: For reading TimsTOF data files
- `rayon`: For parallel processing
- `polars`: For fast data manipulation
- `ndarray`: For numerical array operations
- `ndarray-npy`: For NumPy-compatible file output

## Usage

### Basic Usage

```bash
# Process a specific TimsTOF .d folder
cargo run --release -- /path/to/data.d

# Clear cache
cargo run --release -- --clear-cache

# View cache information
cargo run --release -- --cache-info
```

### Build Profiles

The project includes several optimized build profiles:

- `dev`: Standard development build with debug info
- `dev-opt`: Development build with level 2 optimizations
- `fast-release`: Fast release builds for testing
- `release`: Standard release build (balanced)
- `max-perf`: Maximum performance build for production

```bash
# Build with maximum performance profile
cargo build --profile max-perf
```

## Output

The application generates:
- `all_rt_matrix.npy`: Retention time count matrix
- `all_im_matrix.npy`: Ion mobility count matrix  
- `precursor_ids.txt`: List of precursor IDs with processing status
- `unique_values_output/`: Directory containing unique RT and iM values

## Configuration

Key parameters can be adjusted in `main.rs`:
- `max_precursors`: Maximum number of precursors to process
- `parallel_threads`: Number of parallel processing threads
- `frag_repeat_num`: Fragment repeat number for analysis

## Architecture

The codebase is organized into several modules:

- `main.rs`: Main application logic and orchestration
- `utils.rs`: Utility functions for data processing and I/O
- `processing.rs`: Core processing algorithms and data structures  
- `cache.rs`: Caching system for performance optimization

## Requirements

- Rust 1.70+ (2021 edition)
- TimsTOF `.d` data files
- Spectral library file (TSV format)

## Performance

This tool is optimized for:
- Large-scale TimsTOF datasets
- Multi-core systems (automatically detects available cores)
- Memory-efficient processing of spectral data
- Fast I/O operations with caching

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 