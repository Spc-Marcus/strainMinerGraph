#!/usr/bin/env python3
"""
Extract reads from BAM file to FASTQ format for StrainMiner
"""

import argparse
import subprocess
from pathlib import Path

def extract_reads_from_bam(bam_file: str, fastq_file: str) -> None:
    """
    Extract reads from BAM to FASTQ format using samtools.
    
    Parameters
    ----------
    bam_file : str
        Path to input BAM file
    fastq_file : str
        Path to output FASTQ file
    """
    print(f"Extracting reads from {bam_file} to {fastq_file}")
    
    try:
        # Use samtools to convert BAM to FASTQ
        command = ["samtools", "fastq", bam_file]
        
        with open(fastq_file, 'w') as f_out:
            result = subprocess.run(command, stdout=f_out, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error running samtools: {result.stderr}")
            return False
        
        print(f"Reads extracted successfully: {fastq_file}")
        return True
        
    except FileNotFoundError:
        print("Error: samtools not found. Please install samtools.")
        return False
    except Exception as e:
        print(f"Error extracting reads: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract reads from BAM to FASTQ")
    parser.add_argument("bam", help="Input BAM file")
    parser.add_argument("fastq", help="Output FASTQ file")
    
    args = parser.parse_args()
    
    if not Path(args.bam).exists():
        print(f"Error: BAM file {args.bam} not found")
        return 1
    
    success = extract_reads_from_bam(args.bam, args.fastq)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
