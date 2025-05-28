import os
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pysam as ps

# Package version information
__version__ = "2.0.0"

# Configure module-level logging
logger = logging.getLogger(__name__)


def get_data(input_file: ps.AlignmentFile, 
             contig_name: str, 
             start_pos: int, 
             end_pos: int,
             min_coverage: int = 5,
             min_base_quality: int = 10,
             max_major_allele_freq: float = 0.95) -> Dict[int, Dict[str, int]]:
    """
    Extract suspicious genomic positions with variant alleles from alignment data.
    
    This function performs pileup analysis on a specified genomic region to identify
    positions where multiple alleles are present with sufficient frequency to suggest
    genuine variants (as opposed to sequencing errors). It's the primary data extraction
    function for downstream strain separation analysis.
    
    The algorithm identifies heterozygous positions by examining allele frequencies
    at each covered position, filtering for positions where the major allele frequency
    is below a specified threshold, indicating potential strain variants or SNPs.
    
    Parameters
    ----------
    input_file : ps.AlignmentFile
        Opened pysam AlignmentFile object (BAM/SAM) containing mapped sequencing reads.
        File must be properly indexed for efficient region-based access.
    contig_name : str
        Name of the reference contig/chromosome to analyze. Must match header names
        in the alignment file exactly (case-sensitive).
    start_pos : int
        Starting genomic position for analysis (0-based coordinate system).
        Must be non-negative and less than contig length.
    end_pos : int
        Ending genomic position for analysis (0-based, exclusive).
        Must be greater than start_pos and within contig bounds.
    min_coverage : int, optional
        Minimum number of reads required to cover a position for variant calling.
        Positions with fewer reads are skipped to avoid low-confidence calls.
        Default is 5 reads.
    min_base_quality : int, optional
        Minimum base quality score (Phred scale) required for reads to contribute
        to variant calling. Low-quality bases are filtered out. Default is 10.
    max_major_allele_freq : float, optional
        Maximum frequency allowed for the most common allele at a position.
        Positions where the major allele exceeds this frequency are considered
        monomorphic and excluded. Default is 0.95 (95%).
        
    Returns
    -------
    Dict[int, Dict[str, int]]
        Nested dictionary mapping genomic positions to read variant calls:
        - Outer key: genomic position (int)
        - Inner key: read name (str)  
        - Inner value: allele call (int)
          - 0: read has minor allele (variant)
          - 1: read has major allele (reference-like)
          - Missing: read doesn't cover position or filtered out
          
        Only positions with evidence for multiple alleles are included.
        
    Algorithm Details
    -----------------
    1. **Pileup Generation**: 
       - Use pysam pileup with specified quality and coverage filters
       - Process each position in the specified genomic window
       
    2. **Allele Frequency Analysis**:
       - Extract all base calls at each position
       - Calculate frequency of each unique base
       - Sort alleles by decreasing frequency
       
    3. **Variant Position Detection**:
       - Skip positions where major allele frequency ≥ max_major_allele_freq
       - Require at least 2 distinct alleles with sufficient reads
       
    4. **Read Assignment**:
       - Assign binary codes based on allele identity:
         * 1: read carries the least frequent (minor) allele
         * 0: read carries the second most frequent allele  
         * Filter: reads with other alleles or quality issues
    
    Raises
    ------
    ValueError
        If start_pos >= end_pos or coordinates are invalid
    KeyError  
        If contig_name is not found in alignment file header
    IOError
        If alignment file is not properly indexed or accessible
    MemoryError
        If region is too large for available system memory
        
    See Also
    --------
    create_matrix : Converts variant dictionary to numerical matrix format
    pre_processing : Downstream processing of variant matrices  
    pysam.AlignmentFile.pileup : Underlying pileup generation function
    
    """
    # Input validation
    if start_pos >= end_pos:
        raise ValueError(f"start_pos ({start_pos}) must be less than end_pos ({end_pos})")
    
    if start_pos < 0:
        raise ValueError(f"start_pos ({start_pos}) must be non-negative")
        
    # Check if contig exists in alignment file
    if not hasattr(input_file, 'references') or input_file.references is None:
        raise ValueError("Alignment file does not contain reference information")
        
    if contig_name not in input_file.references:
        available_contigs = list(input_file.references)
        if len(available_contigs) == 0:
            raise ValueError("No contigs found in alignment file")
        
        # Show first 5 contigs in error message
        contig_preview = ', '.join(available_contigs[:5])
        if len(available_contigs) > 5:
            contig_preview += f"... (and {len(available_contigs) - 5} more)"
            
        raise ValueError(f"Contig '{contig_name}' not found in alignment file. "
                        f"Available contigs: {contig_preview}")
    
    logger.debug(f"Extracting variants from {contig_name}:{start_pos}-{end_pos}")
    
    # Initialize results dictionary
    suspicious_positions = {}
    
    try:
        # Generate pileup over specified region with quality filters
        pileup_iter = input_file.pileup(
            contig=contig_name,
            start=start_pos, 
            stop=end_pos,
            truncate=True,
            min_base_quality=min_base_quality
        )
        
        positions_processed = 0
        positions_with_variants = 0
        
        # Process each position in the pileup
        for pileupcolumn in pileup_iter:
            if pileupcolumn.nsegments >=5:
                
                tmp_dict = {}
                sequence = np.char.upper(np.array(pileupcolumn.get_query_sequences()))
                bases, freq = np.unique(np.array(sequence),return_counts=True)
        
                if bases[0] =='':    
                    ratio = freq[1:]/sum(freq[1:])
                    bases = bases[1:]
                else:
                    ratio = freq/sum(freq)
                
                if len(ratio) >0:
                    idx_sort = np.argsort(ratio)
                    
                    if ratio[idx_sort[-1]]<0.95:
                        for pileupread in pileupcolumn.pileups:        
                            if not pileupread.is_del and not pileupread.is_refskip:
                                # query position is None if is_del or is_refskip is set.
                                if pileupread.alignment.query_sequence[pileupread.query_position] == bases[idx_sort[-1]]:
                                    tmp_dict[pileupread.alignment.query_name] =  1
                                elif pileupread.alignment.query_sequence[pileupread.query_position] == bases[idx_sort[-2]]:
                                    tmp_dict[pileupread.alignment.query_name] =  0
                                else :
                                    tmp_dict[pileupread.alignment.query_name] = np.nan
                    
                        suspicious_positions[pileupcolumn.reference_pos] = tmp_dict        

        
        logger.info(f"Processed {positions_processed} positions, "
                   f"found variants at {positions_with_variants} positions")
                   
    except Exception as e:
        logger.error(f"Error during pileup processing for {contig_name}:{start_pos}-{end_pos}: {e}")
        raise RuntimeError(f"Pileup processing failed: {e}") from e
    
    return suspicious_positions


def validate_input_files(bam_file: str, 
                        assembly_file: str, 
                        reads_file: str) -> bool:
    """
    Validate input file existence, accessibility, and format compatibility.
    
    This function performs comprehensive validation of all required input files
    for the StrainMiner pipeline, checking file existence, read permissions,
    basic format validation, and associated index files where required.
    
    Parameters
    ----------
    bam_file : str
        Path to BAM alignment file. Must be indexed (*.bai file present).
    assembly_file : str  
        Path to assembly file in GFA format.
    reads_file : str
        Path to sequencing reads file (FASTQ, FASTA, or compressed variants).
        
    Returns
    -------
    bool
        True if all files pass validation, False otherwise.
        Detailed error messages are logged for any validation failures.
        
    Raises
    ------
    FileNotFoundError
        If any required file is missing
    PermissionError  
        If files are not readable
    """
    validation_passed = True
    
    # Check BAM file
    bam_path = Path(bam_file)
    if not bam_path.exists():
        logger.error(f"BAM file not found: {bam_file}")
        validation_passed = False
    elif not bam_path.is_file():
        logger.error(f"BAM path is not a file: {bam_file}")
        validation_passed = False
    elif not os.access(bam_path, os.R_OK):
        logger.error(f"BAM file not readable: {bam_file}")
        validation_passed = False
    else:
        # Check for BAM index
        index_candidates = [
            bam_path.with_suffix('.bam.bai'),
            bam_path.with_suffix('.bai'),
            Path(str(bam_path) + '.bai')
        ]
        if not any(idx.exists() for idx in index_candidates):
            logger.warning(f"BAM index not found for {bam_file}. "
                          "This may cause performance issues.")
    
    # Check assembly file
    assembly_path = Path(assembly_file)
    if not assembly_path.exists():
        logger.error(f"Assembly file not found: {assembly_file}")
        validation_passed = False
    elif not assembly_path.is_file():
        logger.error(f"Assembly path is not a file: {assembly_file}")
        validation_passed = False
    elif not os.access(assembly_path, os.R_OK):
        logger.error(f"Assembly file not readable: {assembly_file}")
        validation_passed = False
    elif not assembly_path.suffix.lower() in ['.gfa', '.gfa1', '.gfa2']:
        logger.warning(f"Assembly file extension not recognized as GFA: {assembly_file}")
    
    # Check reads file
    reads_path = Path(reads_file)
    if not reads_path.exists():
        logger.error(f"Reads file not found: {reads_file}")
        validation_passed = False
    elif not reads_path.is_file():
        logger.error(f"Reads path is not a file: {reads_file}")
        validation_passed = False
    elif not os.access(reads_path, os.R_OK):
        logger.error(f"Reads file not readable: {reads_file}")
        validation_passed = False
    
    return validation_passed


def setup_output_directories(output_path: str) -> Dict[str, str]:
    """
    Create comprehensive output directory structure for StrainMiner pipeline.
    
    Sets up all necessary subdirectories for organized output storage,
    including temporary files, logs, results, and intermediate data.
    
    Parameters
    ----------
    output_path : str
        Base output directory path
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping directory purposes to their absolute paths:
        - 'output_dir': Main output directory
        - 'tmp_dir': Temporary files  
        - 'logs_dir': Log files
    """
    output_dir = Path(output_path).resolve()
    
    # Create main output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define subdirectory structure
    subdirs = {
        'tmp_dir': output_dir / 'tmp',
        'logs_dir': output_dir / 'logs'}
    
    # Create all subdirectories
    for purpose, dir_path in subdirs.items():
        dir_path.mkdir(exist_ok=True)
        logger.debug(f"Created {purpose}: {dir_path}")
    
    # Add main output directory to results
    result_dirs = {'output_dir': str(output_dir)}
    result_dirs.update({k: str(v) for k, v in subdirs.items()})
    
    return result_dirs


def get_source_path() -> str:
    """
    Determine the source path for StrainMiner external tools and dependencies.
    
    This function resolves the path to external binaries and scripts required
    by the StrainMiner pipeline, handling various installation scenarios.
    
    Returns
    -------
    str
        Absolute path to the source directory containing external tools
        
    Examples
    --------
    >>> src_path = get_source_path()
    >>> build_dir = os.path.join(src_path, "build")
    >>> print(f"External tools located at: {build_dir}")
    """
    # Get the directory containing this module
    module_dir = Path(__file__).parent
    
    # Check common locations for source directory
    candidates = [
        module_dir,                    # Same directory as module
        module_dir.parent,             # Parent directory  
        module_dir.parent.parent,      # Two levels up
        Path.cwd(),                    # Current working directory
    ]
    
    for candidate in candidates:
        if (candidate / 'build').exists():
            logger.debug(f"Found source path: {candidate}")
            return str(candidate.resolve())
    
    # Default fallback
    logger.warning("Could not locate source directory with build tools")
    return str(module_dir.resolve())


def get_contig_info(alignment_file: ps.AlignmentFile) -> List[Tuple[str, int]]:
    """
    Extract contig names and lengths from alignment file header.
    
    Parameters
    ----------
    alignment_file : ps.AlignmentFile
        Opened alignment file
        
    Returns
    -------
    List[Tuple[str, int]]
        List of (contig_name, contig_length) tuples
        
    Examples
    --------
    >>> with pysam.AlignmentFile("data.bam", "rb") as bam:
    ...     contigs = get_contig_info(bam)
    ...     for name, length in contigs:
    ...         print(f"{name}: {length:,} bp")
    """
    try:
        header_dict = alignment_file.header.to_dict()
        sequences = header_dict.get('SQ', [])
        
        contig_info = []
        for seq_record in sequences:
            name = seq_record.get('SN', 'Unknown')
            length = seq_record.get('LN', 0)
            contig_info.append((name, length))
            
        return contig_info
        
    except Exception as e:
        logger.error(f"Error extracting contig information: {e}")
        return []


def estimate_memory_requirements(contig_length: int, 
                               coverage_depth: int,
                               window_size: int = 5000) -> Dict[str, float]:
    """
    Estimate memory requirements for processing a genomic region.
    
    Provides rough estimates of memory usage for different pipeline stages
    to help with resource planning and parameter optimization.
    
    Parameters
    ----------
    contig_length : int
        Length of contig in base pairs
    coverage_depth : int  
        Expected read coverage depth
    window_size : int, optional
        Window size for sliding window analysis
        
    Returns
    -------
    Dict[str, float]
        Dictionary with memory estimates in MB for different operations:
        - 'pileup_memory': Memory for pileup data structure
        - 'matrix_memory': Memory for variant matrix storage
        - 'clustering_memory': Memory for clustering operations
        - 'total_estimated': Total estimated peak memory usage
        
    Examples
    --------
    >>> # Estimate for 1Mb contig at 50x coverage
    >>> memory_est = estimate_memory_requirements(1_000_000, 50)
    >>> print(f"Estimated peak memory: {memory_est['total_estimated']:.1f} MB")
    """
    # Rough estimates based on empirical observations
    
    # Pileup memory: roughly 8 bytes per read per position with variants
    variant_fraction = 0.001  # Assume 0.1% positions have variants
    pileup_positions = contig_length * variant_fraction
    pileup_memory = (pileup_positions * coverage_depth * 8) / (1024 * 1024)
    
    # Matrix memory: variant positions × reads × 4 bytes (float32)
    num_windows = (contig_length + window_size - 1) // window_size
    avg_variants_per_window = max(1, pileup_positions / num_windows)
    matrix_memory = (avg_variants_per_window * coverage_depth * 4) / (1024 * 1024)
    
    # Clustering memory: roughly 2x matrix memory for intermediate structures
    clustering_memory = matrix_memory * 2
    
    # Total with safety factor
    total_estimated = (pileup_memory + matrix_memory + clustering_memory) * 1.5
    
    return {
        'pileup_memory': pileup_memory,
        'matrix_memory': matrix_memory, 
        'clustering_memory': clustering_memory,
        'total_estimated': total_estimated
    }

