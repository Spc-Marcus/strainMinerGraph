#!/usr/bin/env python3
"""
Convert FASTA assembly to GFA format for StrainMiner
"""

import argparse
from pathlib import Path

def fasta_to_gfa(fasta_file: str, gfa_file: str) -> None:
    """
    Convert FASTA file to simple GFA format.
    
    Parameters
    ----------
    fasta_file : str
        Path to input FASTA file
    gfa_file : str
        Path to output GFA file
    """
    print(f"Converting {fasta_file} to {gfa_file}")
    
    with open(fasta_file, 'r') as f_in, open(gfa_file, 'w') as f_out:
        # Write GFA header
        f_out.write("H\tVN:Z:1.0\n")
        
        sequence = ""
        contig_name = None
        contig_id = 1
        
        for line in f_in:
            line = line.strip()
            if line.startswith('>'):
                # Write previous contig if exists
                if contig_name and sequence:
                    f_out.write(f"S\t{contig_id}\t{sequence}\n")
                    contig_id += 1
                
                # Start new contig
                contig_name = line[1:].split()[0]  # Take only first part of header
                sequence = ""
            else:
                sequence += line
        
        # Write last contig
        if contig_name and sequence:
            f_out.write(f"S\t{contig_id}\t{sequence}\n")
    
    print(f"Conversion complete: {gfa_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert FASTA to GFA format")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("gfa", help="Output GFA file")
    
    args = parser.parse_args()
    
    if not Path(args.fasta).exists():
        print(f"Error: FASTA file {args.fasta} not found")
        return 1
    
    fasta_to_gfa(args.fasta, args.gfa)
    return 0

if __name__ == "__main__":
    exit(main())
