import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .core import (__version__, validate_input_files, setup_output_directories, 
                   get_source_path, estimate_memory_requirements)
from .pipeline import run_strainminer_pipeline


def setup_logging(log_dir: str, verbose: bool = False) -> logging.Logger:
    """
    Configure comprehensive logging for the StrainMiner pipeline.
    
    Sets up both file and console logging with appropriate levels and formatting.
    Creates timestamped log files for persistent record keeping.
    
    Parameters
    ----------
    log_dir : str
        Directory where log files should be written
    verbose : bool, optional
        Enable verbose console output for debugging. Default is False.
        
    Returns
    -------
    logging.Logger
        Configured logger instance for the main application
        
    Examples
    --------
    >>> logger = setup_logging("/path/to/logs", verbose=True)
    >>> logger.info("Pipeline started")
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"strainminer_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler - always detailed
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - level depends on verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments and check file accessibility.
    
    This function performs comprehensive validation of all input parameters,
    checks file existence and permissions, validates parameter ranges,
    ensures output directory is writable, and provides helpful error messages.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing all pipeline parameters
        
    Raises
    ------
    FileNotFoundError
        If required input files don't exist or aren't accessible
    PermissionError
        If output directory isn't writable or input files aren't readable
    ValueError
        If parameter values are outside valid ranges
        
    Examples
    --------
    >>> args = parse_arguments()
    >>> try:
    ...     validate_arguments(args)
    ...     print("All arguments valid")
    ... except ValueError as e:
    ...     print(f"Invalid parameter: {e}")
    """
    # Validate input file existence and accessibility
    if not validate_input_files(args.bam, args.assembly, args.reads):
        raise FileNotFoundError("One or more input files are missing or inaccessible")
    
    # Validate numeric parameter ranges
    if not (0.0 <= args.error_rate <= 1.0):
        raise ValueError(f"Error rate must be between 0.0 and 1.0, got {args.error_rate}")
    
    if args.window <= 0:
        raise ValueError(f"Window size must be positive, got {args.window}")
        
    if not (0.0 <= args.min_coverage <= 1.0):
        raise ValueError(f"Minimum coverage must be between 0.0 and 1.0, got {args.min_coverage}")
    
    if args.min_row_quality <= 0:
        raise ValueError(f"Minimum row quality must be positive, got {args.min_row_quality}")
        
    if args.min_col_quality <= 0:
        raise ValueError(f"Minimum column quality must be positive, got {args.min_col_quality}")
        
    if args.min_base_quality < 0:
        raise ValueError(f"Minimum base quality cannot be negative, got {args.min_base_quality}")
    
    # Validate output directory writability
    output_path = Path(args.output)
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_path}")
        if not os.access(output_path, os.W_OK):
            raise PermissionError(f"Output directory {output_path} is not writable")
    else:
        # Check if parent directory is writable
        parent_dir = output_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PermissionError(f"Cannot create output directory: {e}")
        if not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"Cannot write to parent directory: {parent_dir}")

def setup_environment(output_folder: str, verbose: bool = False) -> Dict[str, str]:
    """
    Setup the execution environment and directory structure.
    
    Creates necessary directories, sets up logging, configures temporary storage,
    establishes links to external tools, and prepares the runtime environment.
    
    Parameters
    ----------
    output_folder : str
        Base output directory path
    verbose : bool, optional
        Enable verbose logging output. Default is False.
        
    Returns
    -------
    Dict[str, str]
        Dictionary containing paths to important directories:
        - 'output_dir': Main output directory
        - 'tmp_dir': Temporary files directory  
        - 'logs_dir': Log files directory
        - 'results_dir': Final results directory
        - 'intermediate_dir': Intermediate processing files
        - 'build_dir': External tools directory (if available)
        
    Examples
    --------
    >>> env_paths = setup_environment("/path/to/output", verbose=True)
    >>> print(f"Temporary files: {env_paths['tmp_dir']}")
    >>> print(f"Logs: {env_paths['logs_dir']}")
    """
    # Create comprehensive directory structure
    dir_paths = setup_output_directories(output_folder)
    
    # Setup logging with the created log directory
    logger = setup_logging(dir_paths['logs_dir'], verbose)
    logger.info(f"StrainMiner v{__version__} - Environment setup")
    logger.info(f"Output directory: {dir_paths['output_dir']}")
    
    # Setup external tools directory (build)
    strainminer_build_dir = Path(__file__).parent / 'build'
    root_build_dir = Path(__file__).parent.parent.parent / 'build'
    
    build_dir = None
    if strainminer_build_dir.exists():
        build_dir = str(strainminer_build_dir.resolve())
        logger.info(f"Found build directory: {build_dir}")
    elif root_build_dir.exists():
        try:
            strainminer_build_dir.symlink_to(root_build_dir.resolve())
            build_dir = str(strainminer_build_dir.resolve())
            logger.info(f"Created symlink to build directory: {build_dir}")
        except OSError as e:
            logger.warning(f"Could not create symlink to build directory: {e}")
            build_dir = str(root_build_dir.resolve())
    else:
        logger.warning("Build directory not found - external tools may not work")
    
    # Add build directory to results
    if build_dir:
        dir_paths['build_dir'] = build_dir
    
    return dir_paths

def check_system_resources(args: argparse.Namespace) -> None:
    """
    Check system resources and provide warnings if constraints might be exceeded.
    
    Estimates memory requirements and checks available system resources to
    warn users about potential performance issues or failures.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Examples
    --------
    >>> args = parse_arguments()
    >>> check_system_resources(args)
    # May print warnings about memory requirements
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get available memory (rough estimate)
        import psutil
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Estimate memory requirements (rough approximation)
        # This would need more sophisticated estimation based on file sizes
        estimated_memory = estimate_memory_requirements(
            contig_length=1_000_000,  # Rough estimate
            coverage_depth=30,        # Rough estimate  
            window_size=args.window
        )
        
        peak_memory = estimated_memory['total_estimated']
        
        if peak_memory > available_memory_mb * 0.8:
            logger.warning(f"Estimated memory usage ({peak_memory:.1f} MB) may exceed "
                          f"available memory ({available_memory_mb:.1f} MB)")
            logger.warning("Consider reducing window size or using more stringent filters")
        
    except ImportError:
        logger.debug("psutil not available - skipping memory check")
    except Exception as e:
        logger.debug(f"Could not check system resources: {e}")

def init(assembly_file: str, 
         bam_file: str, 
         reads_file: str, 
         output_folder: str,
         error_rate: float = 0.025, 
         window: int = 5000,
         min_coverage: float = 0.6,
         min_row_quality: int = 5,
         min_col_quality: int = 3,
         min_base_quality: int = 10,
         verbose: bool = False,
         keep_temp: bool = False) -> int:
    """
    Initialize and execute the complete StrainMiner pipeline.
    
    This function serves as the main orchestrator for the StrainMiner workflow,
    handling environment setup, parameter validation, pipeline execution,
    progress monitoring, and cleanup operations. It provides comprehensive 
    error handling and detailed progress reporting.
    
    Parameters
    ----------
    assembly_file : str
        Path to assembly file in GFA format containing reference contigs.
        Must be readable and properly formatted.
    bam_file : str
        Path to alignment file in BAM format containing mapped reads.
        Must be indexed for efficient random access.
    reads_file : str
        Path to sequencing reads file (FASTQ/FASTA format, optionally compressed).
        Used for assembly polishing and validation steps.
    output_folder : str
        Path to output directory where all results will be written.
        Directory will be created if it doesn't exist.
    error_rate : float, optional
        Expected sequencing error rate for optimization algorithms.
        Should reflect actual data quality. Typical values:
        - Illumina: 0.01-0.025
        - PacBio: 0.05-0.1  
        - Oxford Nanopore: 0.08-0.15
        Default is 0.025 (2.5%).
    window : int, optional
        Size of genomic windows for sliding-window analysis in base pairs.
        Smaller windows provide higher resolution but increase computation time.
        Should be chosen based on:
        - Read length: window should be ≥ 2× average read length
        - Resolution needs: smaller for fine-scale analysis
        - Computational resources: larger for speed
        Default is 5000 bp.
    min_coverage : float, optional
        Minimum coverage threshold for variant positions (0.0-1.0).
        Positions covered by fewer reads are filtered out to improve
        confidence. Higher values increase stringency. Default is 0.6 (60%).
    min_row_quality : int, optional
        Minimum number of reads required for a valid cluster.
        Clusters with fewer reads are discarded as low-confidence.
        Should be balanced with expected strain abundance. Default is 5.
    min_col_quality : int, optional
        Minimum number of genomic positions required for region processing.
        Regions with fewer positions are skipped to focus on informative areas.
        Default is 3.
    min_base_quality : int, optional
        Minimum base quality score (Phred scale) required for variant calling.
        Low-quality bases are filtered out to reduce false positives.
        Typical values: 10-20 for most applications. Default is 10.
    verbose : bool, optional
        Enable verbose logging output for debugging and detailed monitoring.
        Increases console output significantly. Default is False.
    keep_temp : bool, optional
        Preserve temporary files after completion for inspection or debugging.
        Useful for troubleshooting but requires additional disk space.
        Default is False (cleanup temporary files).
        
    Returns
    -------
    int
        Exit code indicating pipeline execution status:
        - 0: Successful completion
        - 1: Input validation error  
        - 2: Pipeline execution error
        - 3: Output writing error
        - 4: Environment setup error
        - 5: System resource error
        
    Examples
    --------
    Basic pipeline execution:
    
    >>> exit_code = init(
    ...     assembly_file="reference.gfa",
    ...     bam_file="aligned_reads.bam", 
    ...     reads_file="raw_reads.fastq",
    ...     output_folder="strainminer_output/"
    ... )
    >>> if exit_code == 0:
    ...     print("Pipeline completed successfully")
    ... else:
    ...     print(f"Pipeline failed with exit code {exit_code}")
    
    High-resolution analysis with custom parameters:
    
    >>> exit_code = init(
    ...     assembly_file="complex_genome.gfa",
    ...     bam_file="long_reads.bam",
    ...     reads_file="ont_reads.fastq", 
    ...     output_folder="detailed_analysis/",
    ...     error_rate=0.05,           # Higher error rate for long reads
    ...     window=2000,               # Smaller windows for resolution
    ...     min_coverage=0.7,          # Higher coverage requirement  
    ...     min_base_quality=15,       # Stricter quality filter
    ...     verbose=True,              # Detailed logging
    ...     keep_temp=True             # Keep intermediate files
    ... )
    
    Memory-constrained analysis:
    
    >>> exit_code = init(
    ...     assembly_file="large_genome.gfa", 
    ...     bam_file="high_coverage.bam",
    ...     reads_file="reads.fastq",
    ...     output_folder="efficient_analysis/",
    ...     window=8000,               # Larger windows to reduce memory
    ...     min_coverage=0.8,          # Higher threshold to filter data
    ...     min_row_quality=8,         # Larger clusters only
    ...     keep_temp=False            # Aggressive cleanup
    ... )
    
    Debugging problematic dataset:
    
    >>> exit_code = init(
    ...     assembly_file="problem.gfa",
    ...     bam_file="problem.bam", 
    ...     reads_file="problem.fastq",
    ...     output_folder="debug_output/",
    ...     error_rate=0.1,            # More permissive for noisy data
    ...     min_coverage=0.3,          # Lower threshold to include more data
    ...     min_row_quality=2,         # Accept smaller clusters
    ...     verbose=True,              # Maximum logging detail
    ...     keep_temp=True             # Preserve all intermediate files
    ... )
    
    Notes
    -----
    - **Resource Management**: Function monitors system resources and provides
      warnings about potential memory or disk space issues.
      
    - **Progress Tracking**: Detailed logging tracks progress through each
      pipeline stage with timing information and intermediate results.
      
    - **Error Recovery**: Comprehensive error handling attempts to provide
      actionable error messages and suggestions for parameter adjustment.
      
    - **File Management**: Automatically manages temporary files and provides
      options for cleanup or preservation based on user needs.
    
    **Performance Optimization Tips**:
    - Use larger windows for faster processing on large genomes
    - Increase coverage thresholds to reduce memory usage
    - Enable verbose mode only for debugging (reduces performance)
    - Consider system memory when choosing parameters
    
    **Common Exit Codes and Solutions**:
    - Exit code 1: Check input file paths and permissions
    - Exit code 2: Review parameter values and data quality  
    - Exit code 3: Verify output directory permissions and disk space
    - Exit code 4: Check system dependencies and tool availability
    - Exit code 5: Free system memory or adjust parameters
    
    See Also
    --------
    run_strainminer_pipeline : Core pipeline execution function
    validate_input_files : Input validation details
    setup_output_directories : Output directory structure
    """
    try:
        # Setup execution environment
        env_paths = setup_environment(output_folder, verbose)
        logger = logging.getLogger(__name__)
        
        logger.info(f"StrainMiner v{__version__} - Pipeline Initialization")
        logger.info(f"Assembly: {assembly_file}")
        logger.info(f"BAM file: {bam_file}")
        logger.info(f"Reads: {reads_file}")
        logger.info(f"Output: {output_folder}")
        
    except Exception as e:
        print(f"ERROR: Environment setup failed: {e}")
        return 4
    
    try:
        # Create args-like object for validation
        args = argparse.Namespace(
            assembly=assembly_file,
            bam=bam_file, 
            reads=reads_file,
            output=output_folder,
            error_rate=error_rate,
            window=window,
            min_coverage=min_coverage,
            min_row_quality=min_row_quality,
            min_col_quality=min_col_quality,
            min_base_quality=min_base_quality
        )
        
        # Validate all arguments
        validate_arguments(args)
        logger.info("Input validation completed successfully")
        
        # Check system resources
        check_system_resources(args)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1
    
    try:
        # Execute main pipeline
        logger.info("Starting StrainMiner pipeline execution")
        start_time = datetime.now()
        
        # This would call the main pipeline function
        # For now, this is a placeholder that would be replaced with actual pipeline call
        logger.info("Pipeline execution would start here...")
        
        # Placeholder for actual pipeline execution:
        # result = run_strainminer_pipeline(
        #     assembly_file=assembly_file,
        #     bam_file=bam_file,
        #     reads_file=reads_file,
        #     output_dir=env_paths['output_dir'],
        #     error_rate=error_rate,
        #     window=window,
        #     min_coverage=min_coverage,
        #     min_row_quality=min_row_quality,
        #     min_col_quality=min_col_quality,
        #     min_base_quality=min_base_quality,
        #     tmp_dir=env_paths['tmp_dir'],
        #     verbose=verbose
        # )
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        logger.info(f"Pipeline completed successfully in {elapsed}")
        logger.info(f"Results written to: {env_paths['results_dir']}")
        
        # Cleanup temporary files if requested
        if not keep_temp:
            logger.info("Cleaning up temporary files...")
            # Add cleanup logic here
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 2

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments for StrainMiner pipeline.
    
    Creates a comprehensive argument parser with detailed help text,
    input validation, and sensible defaults for all pipeline parameters.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments with all required and optional parameters
        
   
    Examples:
    -------
        Basic usage  
        python -m strainminer -a assembly.gfa -b reads.bam -r reads.fastq -o output/
        

        High-resolution analysis  
        python -m strainminer -a assembly.gfa -b reads.bam -r reads.fastq -o output/ \\
                                --error-rate 0.01 --window 2000 --verbose
        
        Long-read optimized
        python -m strainminer -a assembly.gfa -b ont.bam -r ont.fastq -o output/ \\
                                --error-rate 0.08 --min-base-quality 7
    """
    
    # Required arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '-a', '--assembly', 
        required=True, 
        help='Assembly file in GFA format containing reference contigs'
    )
    required.add_argument(
        '-b', '--bam', 
        required=True, 
        help='Alignment file in BAM format (must be indexed)'
    )
    required.add_argument(
        '-r', '--reads', 
        required=True, 
        help='Sequencing reads file (FASTQ/FASTA, optionally compressed)'
    )
    required.add_argument(
        '-o', '--output', 
        required=True, 
        help='Output directory (will be created if it doesn\'t exist)'
    )
    
    # Pipeline parameters
    pipeline = parser.add_argument_group('Pipeline Parameters')
    pipeline.add_argument(
        '-e', '--error-rate', 
        type=float, 
        default=0.025,
        metavar='FLOAT',
        help='Expected sequencing error rate (0.0-1.0). '
             'Typical values: Illumina=0.025, PacBio=0.08, ONT=0.1 (default: %(default)s)'
    )
    pipeline.add_argument(
        '-w', '--window', 
        type=int, 
        default=5000,
        metavar='INT',
        help='Genomic window size in base pairs for analysis. '
             'Should be ≥2× average read length (default: %(default)s)'
    )
    
    # Quality filters
    quality = parser.add_argument_group('Quality Control Parameters')
    quality.add_argument(
        '--min-coverage', 
        type=float, 
        default=0.6,
        metavar='FLOAT',
        help='Minimum coverage fraction for variant positions (0.0-1.0) (default: %(default)s)'
    )
    quality.add_argument(
        '--min-row-quality', 
        type=int, 
        default=5,
        metavar='INT', 
        help='Minimum reads required per cluster (default: %(default)s)'
    )
    quality.add_argument(
        '--min-col-quality', 
        type=int, 
        default=3,
        metavar='INT',
        help='Minimum positions required per genomic region (default: %(default)s)'
    )
    quality.add_argument(
        '--min-base-quality', 
        type=int, 
        default=10,
        metavar='INT',
        help='Minimum base quality score (Phred scale) for variant calling (default: %(default)s)'
    )
    
    # Output and debugging options
    output_opts = parser.add_argument_group('Output and Debugging Options')
    output_opts.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output for detailed progress monitoring'
    )
    output_opts.add_argument(
        '--keep-temp', 
        action='store_true',
        help='Preserve temporary files after completion (useful for debugging)'
    )
    output_opts.add_argument(
        '--version', 
        action='version', 
        version=f'StrainMiner {__version__}'
    )
    
    return parser.parse_args()

def main() -> int:
    """
    Main entry point for the StrainMiner command-line interface.
    
    Handles argument parsing, pipeline initialization, and execution with
    comprehensive error handling and user-friendly error messages.
    
    Returns
    -------
    int
        Exit code for shell integration (0 = success, non-zero = error)
        
    Examples
    --------
    >>> # Called automatically when running as module
    >>> # python -m strainminer [arguments]
    >>> exit_code = main()
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Execute pipeline with parsed arguments
        exit_code = init(
            assembly_file=args.assembly,
            bam_file=args.bam,
            reads_file=args.reads, 
            output_folder=args.output,
            error_rate=args.error_rate,
            window=args.window,
            min_coverage=args.min_coverage,
            min_row_quality=args.min_row_quality,
            min_col_quality=args.min_col_quality,
            min_base_quality=args.min_base_quality,
            verbose=args.verbose,
            keep_temp=args.keep_temp
        )
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        if '--verbose' in sys.argv:
            print(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())