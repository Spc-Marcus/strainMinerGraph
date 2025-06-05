import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def setup_assembly_pipeline(tmp_dir: str, path_to_src: str, originalAssembly: str, 
                           readsFile: str, error_rate: float, file_path: str) -> Tuple[str, str, str]:
    """
    Setup and prepare files for assembly pipeline.
    
    Parameters
    ----------
    tmp_dir : str
        Temporary directory path
    path_to_src : str
        Path to source directory containing build tools
    originalAssembly : str
        Path to original assembly file
    readsFile : str
        Path to reads file
    error_rate : float
        Expected sequencing error rate
    file_path : str
        Path to BAM file
        
    Returns
    -------
    Tuple[str, str, str]
        Tuple of (gaffile, zipped_GFA, samFile) paths
        
    Raises
    ------
    RuntimeError
        If setup fails
    """
    try:
        gaffile = os.path.join(tmp_dir, "reads_on_new_contig.gaf")
        zipped_GFA = os.path.join(tmp_dir, "zipped_assembly.gfa")
        samFile = os.path.join(tmp_dir, "reads_on_asm.sam")
        
        # Convert BAM to SAM
        convert_bam_to_sam(file_path, samFile)
        
        logger.debug(f"Assembly pipeline setup completed")
        return gaffile, zipped_GFA, samFile
        
    except Exception as e:
        logger.error(f"Assembly pipeline setup failed: {e}")
        raise RuntimeError(f"Could not setup assembly pipeline: {e}") from e


def run_create_new_contigs(path_to_src: str, originalAssembly: str, readsFile: str, 
                          error_rate: float, tmp_dir: str, samFile: str, 
                          gaffile: str, zipped_GFA: str) -> None:
    """
    Execute create_new_contigs command.
    
    Parameters
    ----------
    path_to_src : str
        Path to source directory
    originalAssembly : str
        Path to original assembly file
    readsFile : str
        Path to reads file
    error_rate : float
        Expected sequencing error rate
    tmp_dir : str
        Temporary directory path
    samFile : str
        Path to SAM file
    gaffile : str
        Path to output GAF file
    zipped_GFA : str
        Path to output zipped GFA file
        
    Raises
    ------
    RuntimeError
        If create_new_contigs execution fails
    """
    try:
        polish_everything = "0"
        polisher = "racon"
        technology = "ont"
        nb_threads = "1"
        
        build_path = Path(path_to_src) / "build" / "create_new_contigs"
        reads_haplo = os.path.join(tmp_dir, "reads_haplo.gro")
        
        command = [
            str(build_path),
            originalAssembly,
            readsFile,
            str(error_rate),
            reads_haplo,
            samFile,
            tmp_dir,
            nb_threads,
            technology,
            zipped_GFA,
            gaffile,
            polisher,
            polish_everything,
            "minimap2", "racon", "medaka", "samtools", "python", "1"
        ]
        
        logger.debug(f"Running create_new_contigs: {' '.join(command)}")
        
        with open(os.path.join(tmp_dir, "create_new_contigs.log"), "w") as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
        
        if result.returncode != 0:
            raise RuntimeError(f"create_new_contigs failed with exit code {result.returncode}")
            
        logger.debug("create_new_contigs completed successfully")
        
    except Exception as e:
        logger.critical(f"create_new_contigs execution failed: {e}")
        raise RuntimeError(f"Could not run create_new_contigs: {e}") from e


def run_graph_unzip(path_to_src: str, gaffile: str, zipped_GFA: str, 
                   out: str, tmp_dir: str) -> str:
    """
    Execute GraphUnzip command.
    
    Parameters
    ----------
    path_to_src : str
        Path to source directory
    gaffile : str
        Path to GAF file
    zipped_GFA : str
        Path to zipped GFA file
    out : str
        Output directory
    tmp_dir : str
        Temporary directory path
        
    Returns
    -------
    str
        Path to output GFA file
        
    Raises
    ------
    RuntimeError
        If GraphUnzip execution fails
    """
    try:
        outfile = os.path.join(out.rstrip('/'), "strainminer_final_assembly.gfa")
        graphunzip_script = os.path.join(path_to_src, "GraphUnzip", "graphunzip.py")
        
        command = [
            "python", graphunzip_script, "unzip",
            "-l", gaffile,
            "-g", zipped_GFA,
            "-o", outfile
        ]
        
        logger.debug(f"Running GraphUnzip: {' '.join(command)}")
        
        with open(os.path.join(tmp_dir, "logGraphUnzip.txt"), "w") as log_file:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log_file)
        
        if result.returncode != 0:
            raise RuntimeError(f"GraphUnzip failed with exit code {result.returncode}")
            
        logger.debug("GraphUnzip completed successfully")
        return outfile
        
    except Exception as e:
        logger.critical(f"GraphUnzip execution failed: {e}")
        raise RuntimeError(f"Could not run GraphUnzip: {e}") from e


def convert_gfa_to_fasta(path_to_src: str, outfile: str) -> str:
    """
    Convert GFA to FASTA format.
    
    Parameters
    ----------
    path_to_src : str
        Path to source directory
    outfile : str
        Path to input GFA file
        
    Returns
    -------
    str
        Path to output FASTA file
        
    Raises
    ------
    RuntimeError
        If conversion fails
    """
    try:
        fasta_name = outfile[:-4] + ".fasta"
        gfa2fa_path = os.path.join(path_to_src, "build", "gfa2fa")
        
        command = [gfa2fa_path, outfile]
        
        logger.debug(f"Converting GFA to FASTA: {' '.join(command)}")
        
        with open(fasta_name, "w") as output_file:
            result = subprocess.run(command, stdout=output_file, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"gfa2fa failed with exit code {result.returncode}")
            
        logger.debug(f"FASTA file created: {fasta_name}")
        return fasta_name
        
    except Exception as e:
        logger.critical(f"GFA to FASTA conversion failed: {e}")
        raise RuntimeError(f"Could not convert GFA to FASTA: {e}") from e


def convert_bam_to_sam(input_bam: str, output_sam: str) -> None:
    """
    Convert BAM file to SAM format using samtools.
    
    Parameters
    ----------
    input_bam : str
        Path to input BAM file
    output_sam : str
        Path to output SAM file
        
    Raises
    ------
    RuntimeError
        If conversion fails
    """
    try:
        command = ["samtools", "view", "-h", "-o", output_sam, input_bam]
        
        logger.debug(f"Converting BAM to SAM: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"samtools view failed: {result.stderr}")
            
        logger.debug(f"SAM file created: {output_sam}")
        
    except Exception as e:
        logger.error(f"BAM to SAM conversion failed: {e}")
        raise RuntimeError(f"Could not convert BAM to SAM: {e}") from e