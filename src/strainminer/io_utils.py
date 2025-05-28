import argparse
import logging
from typing import List, Dict, Tuple, TextIO

logger = logging.getLogger(__name__)


def write_reads_to_file(sol_file: TextIO, list_of_reads: List[str], 
                       read_positions: Dict[str, Tuple[int, int, int, int, int]]) -> None:
    """
    Write read information to output file with position details.
    
    Parameters
    ----------
    sol_file : TextIO
        Open file handle for writing output
    list_of_reads : List[str]
        List of read names to write
    read_positions : Dict[str, Tuple[int, int, int, int, int]]
        Dictionary mapping read names to position tuples (start_read, end_read, start_contig, end_contig, strand)
        
    Raises
    ------
    IOError
        If file writing fails
    """
    try:
        for read_name in list_of_reads:
            if read_name in read_positions:
                start_read, end_read, start_contig, end_contig, strand = read_positions[read_name]
                sol_file.write(f'READ\t{read_name}\t{start_read}\t{end_read}\t{start_contig}\t{end_contig}\t{strand}\n')
            else:
                sol_file.write(f'READ\t{read_name}\t-1\t-1\t-1\t-1\t-1\n')
                logger.warning(f"Read {read_name} not found in alignment positions")
    except Exception as e:
        logger.error(f"Failed to write reads to file: {e}")
        raise IOError(f"Could not write reads to file: {e}") from e


def write_haplotype_groups(sol_file: TextIO, haplotypes: List[Dict[int, int]], 
                          window: int, contig_length: int, list_of_reads: List[str]) -> None:
    """
    Write haplotype group information to output file.
    
    Parameters
    ----------
    sol_file : TextIO
        Open file handle for writing output
    haplotypes : List[Dict[int, int]]
        List of haplotype dictionaries for each window
    window : int
        Window size used for analysis
    contig_length : int
        Total length of the contig
    list_of_reads : List[str]
        List of all read names
        
    Raises
    ------
    IOError
        If file writing fails
    """
    try:
        for w in range(len(haplotypes)):
            start_pos = w * window
            end_pos = min(contig_length, (w + 1) * window)
            
            sol_file.write(f'GROUP\t{start_pos}\t{end_pos}\t')
            
            # Initialize haplotype assignments for all reads
            haplotypes_list = [-2 for _ in range(len(list_of_reads))]
            
            # Fill in known haplotype assignments
            for read_idx in haplotypes[w].keys():
                haplotypes_list[read_idx] = haplotypes[w][read_idx]
            
            # Write read indices and assignments
            read_indices = []
            haplo_assignments = []
            for h in range(len(haplotypes_list)):
                if haplotypes_list[h] != -2:
                    read_indices.append(str(h))
                    haplo_assignments.append(str(haplotypes_list[h]))
            
            sol_file.write(','.join(read_indices) + '\t')
            sol_file.write(','.join(haplo_assignments) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to write haplotype groups: {e}")
        raise IOError(f"Could not write haplotype groups: {e}") from e


def write_contig_header(sol_file: TextIO, contig_name: str, contig_length: int) -> None:
    """
    Write contig header information to output file.
    
    Parameters
    ----------
    sol_file : TextIO
        Open file handle for writing output
    contig_name : str
        Name of the contig
    contig_length : int
        Length of the contig in base pairs
        
    Raises
    ------
    IOError
        If file writing fails
    """
    try:
        sol_file.write(f'CONTIG\t{contig_name}\t{contig_length}\t1\n')
        logger.debug(f"Wrote header for contig {contig_name} (length: {contig_length})")
    except Exception as e:
        logger.error(f"Failed to write contig header: {e}")
        raise IOError(f"Could not write contig header: {e}") from e

