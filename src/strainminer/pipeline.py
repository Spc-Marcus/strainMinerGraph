import os
import json
import logging
import numpy as np
import pandas as pd
import pysam as ps
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from .core import get_data, setup_output_directories
from .matrix import create_matrix, pre_processing
from .cluster import clustering_full_matrix
from .postprocess import post_processing
from .assembly import setup_assembly_pipeline, run_create_new_contigs, run_graph_unzip, convert_gfa_to_fasta
from .io_utils import write_reads_to_file, write_haplotype_groups, write_contig_header

logger = logging.getLogger(__name__)


class StrainMinerPipeline:
    """
    Main StrainMiner pipeline orchestrator for microbial strain separation.
    
    Parameters
    ----------
    error_rate : float
        Expected sequencing error rate
    window_size : int
        Size of genomic windows for analysis
    min_row_quality : int
        Minimum reads required per cluster
    min_col_quality : int
        Minimum positions required per region
    """
    
    def __init__(self, error_rate: float = 0.025, window_size: int = 5000, 
                 min_row_quality: int = 5, min_col_quality: int = 3):
        self.error_rate = error_rate
        self.window_size = window_size
        self.min_row_quality = min_row_quality
        self.min_col_quality = min_col_quality
        self.filtered_col_threshold = 0.6
    
    def process_genomic_window(self, alignment_file: ps.AlignmentFile, contig_name: str, 
                             start_pos: int, end_pos: int) -> Tuple[List[np.ndarray], List[str]]:
        """
        Process complete genomic window through StrainMiner pipeline.
        
        Parameters
        ----------
        alignment_file : ps.AlignmentFile
            Opened BAM alignment file
        contig_name : str
            Name of the contig to process
        start_pos : int
            Start position of the window
        end_pos : int
            End position of the window
            
        Returns
        -------
        Tuple[List[np.ndarray], List[str]]
            Tuple of (clusters, read_names) where clusters is list of strain assignments
            and read_names is list of corresponding read identifiers
            
        Raises
        ------
        RuntimeError
            If window processing fails
        """
        try:
            logger.debug(f"Processing window {contig_name}:{start_pos}-{end_pos}")
            
            # Extract variant data from alignment
            dict_of_sus_pos = get_data(alignment_file, contig_name, start_pos, end_pos)
            
            if not dict_of_sus_pos:
                logger.debug(f"No variants found in window {contig_name}:{start_pos}-{end_pos}")
                return [], []
            
            # Create matrix from variant dictionary
            matrix, reads = create_matrix(dict_of_sus_pos, self.filtered_col_threshold)
            
            if matrix.size == 0:
                logger.debug(f"Empty matrix for window {contig_name}:{start_pos}-{end_pos}")
                return [], reads
            
            # Preprocessing: imputation and binarization
            processed_matrix, regions, steps = pre_processing(matrix, self.min_col_quality)
            
            # Clustering to identify strain groups
            final_steps = clustering_full_matrix(
                processed_matrix, regions, steps, 
                self.min_row_quality, self.min_col_quality, self.error_rate
            )
            
            # Post-processing to finalize clusters
            clusters = post_processing(processed_matrix, final_steps, reads, distance_thresh=0.05)
            
            logger.info(f"Window {contig_name}:{start_pos}-{end_pos}: found {len(clusters)} strain groups")
            return clusters, reads
            
        except Exception as e:
            logger.error(f"Failed to process window {contig_name}:{start_pos}-{end_pos}: {e}")
            raise RuntimeError(f"Window processing failed: {e}") from e
    
    def process_single_contig(self, file: ps.AlignmentFile, sol_file, contig: Dict[str, Any], 
                            window: int, tmp_dir: str, error_rate: float) -> Tuple[List[str], Dict[str, Tuple]]:
        """
        Process a single contig through the StrainMiner pipeline.
        
        Parameters
        ----------
        file : ps.AlignmentFile
            Opened BAM alignment file
        sol_file : TextIO
            Output file handle
        contig : Dict[str, Any]
            Contig information dictionary with 'SN' (name) and 'LN' (length) keys
        window : int
            Window size for sliding window analysis
        tmp_dir : str
            Temporary directory for intermediate files
        error_rate : float
            Expected sequencing error rate
            
        Returns
        -------
        Tuple[List[str], Dict[str, Tuple]]
            Tuple of (list_of_reads, read_positions) for the processed contig
            
        Raises
        ------
        RuntimeError
            If contig processing fails
        """
        try:
            contig_name = contig['SN']
            contig_length = contig['LN']
            
            logger.info(f"Processing contig {contig_name} (length: {contig_length:,} bp)")
            
            list_of_reads = []
            index_of_reads = {}
            haplotypes = []
            
            # Process contig in sliding windows
            for start_pos in range(0, contig_length, window):
                end_pos = min(start_pos + window, contig_length)
                
                logger.debug(f"Processing window {start_pos}-{end_pos}")
                
                # Extract variant data
                dict_of_sus_pos = get_data(file, contig_name, start_pos, end_pos)
                
                # Save intermediate data
                output_file = Path(tmp_dir) / f"dict_of_sus_pos_{contig_name}_{start_pos}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(dict_of_sus_pos, f, ensure_ascii=False, indent=2)
                
                haplotypes_here = {}
                
                if dict_of_sus_pos:
                    # Create and filter matrix
                    df = pd.DataFrame(dict_of_sus_pos)
                    
                    # Filter reads spanning the window
                    if len(df.columns) >= 3:
                        tmp_idx = df.iloc[:, :len(df.columns)//3].dropna(axis=0, how='all')
                        df = df.loc[tmp_idx.index, :]
                        tmp_idx = df.iloc[:, 2*len(df.columns)//3:].dropna(axis=0, how='all')
                        df = df.loc[tmp_idx.index, :]
                    
                    df = df.dropna(axis=1, thresh=self.filtered_col_threshold * len(df.index))
                    reads = list(df.index)
                    
                    # Clustering if sufficient data
                    if not df.empty:
                        X_matrix = df.to_numpy()
                        matrix, regions, steps = pre_processing(X_matrix, self.min_col_quality)
                        steps = clustering_full_matrix(
                            matrix, regions, steps, 
                            self.min_row_quality, self.min_col_quality, error_rate
                        )
                        clusters = post_processing(matrix, steps, reads, distance_thresh=0.05)
                        
                        # Assign haplotype labels
                        if len(clusters) > 1:
                            for idx, cluster in enumerate(clusters):
                                for read in cluster:
                                    if read not in index_of_reads:
                                        index_of_reads[read] = len(list_of_reads)
                                        list_of_reads.append(read)
                                    haplotypes_here[index_of_reads[read]] = idx
                            logger.info(f"Found {len(clusters)} strain groups in window {start_pos}-{end_pos}")
                        else:
                            # Single group or no clustering
                            for read in reads:
                                if read not in index_of_reads:
                                    index_of_reads[read] = len(list_of_reads)
                                    list_of_reads.append(read)
                                haplotypes_here[index_of_reads[read]] = -1
                    else:
                        logger.debug(f"No valid reads in window {start_pos}-{end_pos}")
                else:
                    logger.debug(f"No variants found in window {start_pos}-{end_pos}")
                
                haplotypes.append(haplotypes_here)
            
            # Write contig header
            write_contig_header(sol_file, contig_name, contig_length)
            
            # Collect read positions
            read_positions = {}
            for read in file.fetch():
                read_name = read.query_name
                if read_name in list_of_reads:
                    start_read = read.query_alignment_start or 0
                    end_read = read.query_alignment_end or read.query_length
                    start_contig = read.reference_start
                    end_contig = read.reference_end
                    strand = 0 if read.is_reverse else 1
                    read_positions[read_name] = (start_read, end_read, start_contig, end_contig, strand)
            
            # Write reads and haplotype groups
            write_reads_to_file(sol_file, list_of_reads, read_positions)
            write_haplotype_groups(sol_file, haplotypes, window, contig_length, list_of_reads)
            
            logger.info(f"Completed processing contig {contig_name}: {len(list_of_reads)} reads processed")
            return list_of_reads, read_positions
            
        except Exception as e:
            logger.error(f"Failed to process contig {contig_name}: {e}")
            raise RuntimeError(f"Contig processing failed: {e}") from e


def run_strainminer_pipeline(bam_file: str, assembly_file: str, reads_file: str, 
                           output_dir: str, error_rate: float = 0.025, 
                           window_size: int = 5000, print_mode: Optional[str] = None, **kwargs) -> str:
    """
    Main entry point function for StrainMiner pipeline execution.
    
    Parameters
    ----------
    bam_file : str
        Path to BAM alignment file
    assembly_file : str
        Path to assembly file in GFA format
    reads_file : str
        Path to sequencing reads file
    output_dir : str
        Output directory path
    error_rate : float
        Expected sequencing error rate
    window_size : int
        Window size for analysis
    print_mode : Optional[str]
        Debug print mode ('start', 'end', 'step', 'all')
    **kwargs
        Additional pipeline parameters
        
    Returns
    -------
    str
        Path to final assembly file
        
    Raises
    ------
    RuntimeError
        If pipeline execution fails
    """
    try:
        logger.info("Starting StrainMiner pipeline execution")
        
        # Configure print mode if specified
        if print_mode:
            from ..decorateur.perf import set_print_mode
            set_print_mode(print_mode)
            logger.info(f"Debug print mode configured: {print_mode}")
        
        # Setup directories
        dir_paths = setup_output_directories(output_dir)
        tmp_dir = dir_paths['tmp_dir']
        
        # Initialize pipeline
        pipeline = StrainMinerPipeline(error_rate, window_size)
        
        # Open BAM file
        with ps.AlignmentFile(bam_file, 'rb') as file:
            contigs = file.header.to_dict()['SQ']
            
            if not contigs:
                raise RuntimeError("No contigs found in BAM file")
            
            # Process each contig
            with open(Path(tmp_dir) / 'reads_haplo.gro', 'w') as sol_file:
                for contig in contigs:
                    pipeline.process_single_contig(
                        file, sol_file, contig, window_size, tmp_dir, error_rate
                    )
        
        # Setup assembly pipeline
        path_to_src = str(Path(__file__).parent.parent.parent)
        gaffile, zipped_GFA, samFile = setup_assembly_pipeline(
            tmp_dir, path_to_src, assembly_file, reads_file, error_rate, bam_file
        )
        
        # Run assembly steps
        run_create_new_contigs(
            path_to_src, assembly_file, reads_file, error_rate, 
            tmp_dir, samFile, gaffile, zipped_GFA
        )
        
        outfile = run_graph_unzip(path_to_src, gaffile, zipped_GFA, output_dir, tmp_dir)
        fasta_file = convert_gfa_to_fasta(path_to_src, outfile)
        
        logger.info(f"Pipeline completed successfully. Final assembly: {fasta_file}")
        return fasta_file
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise RuntimeError(f"StrainMiner pipeline failed: {e}") from e