import pandas as pd 
import numpy as np
import gurobipy as grb
import pysam as ps
#TODO: Importer les fonctions nécessaires pour le traitement des données
# Version du package
__version__ = "1.1.0"  # À remplacer par la version actuelle

def file_path(path=None):
    """
    Gestion des chemins de fichiers
    """
    # Implémentation de la fonction
    pass

def get_data(input_file: ps.AlignmentFile, contig_name: str, start_pos: int, end_pos: int) -> dict:
    """
    Retrieve and process data from an alignment file.

    Parameters:
    -----------
    input_file : ps.AlignmentFile
        Alignment file containing the data to process.
    contig_name : str
        Name of the contig to extract.
    start_pos : int
        Start position for extraction.
    end_pos : int
        End position for extraction.

    Returns:
    --------
    dict
        Dictionary containing the extracted data from input_file.
    """
    intercol = input_file.pileup(contig=contig_name, start=start_pos, end=end_pos, truncate=True, min_base_quality=10)
    list_of_sus_pos = {}
    # Iterate over pileup columns
    for pileupcolumn in intercol:
        # Check if the pileup column has at least 5 segments
        if pileupcolumn.nsegments >= 5:
            temp_dict = {}  # Temporary dictionary to store column data
            sequence = np.char.upper(np.array(pileupcolumn.get_query_sequences()))  # Convert sequences to uppercase
            bases, freq = np.unique(np.array(sequence), return_counts=True)  # Get unique bases and their frequencies

            if bases[0] == '':
                ratio = freq[1:] / sum(freq[1:])  # Calculate base ratios (ignore empty base)
                bases = bases[1:]  # Ignore empty base
            else:
                ratio = freq / sum(freq)  # Calculate base ratios

            if len(ratio) > 0:  # Check if ratio is not empty
                idx_sort = np.argsort(ratio)[::-1]  # Get indices to sort bases by decreasing ratio

                if ratio[idx_sort[0]] < 0.95:  # If the most frequent base ratio is less than 0.95
                    for pileupread in pileupcolumn.pileups:  # Iterate over pileup reads
                        if not pileupread.is_del and not pileupread.is_refskip:
                            # Check if the read base matches the least frequent base
                            value = (1 if pileupread.alignment.query_sequence[pileupread.query_position] == bases[idx_sort[-1]] else 0)
                            temp_dict[pileupread.alignment.query_name] = value  # Add value to temporary dictionary
                    list_of_sus_pos[pileupcolumn.pos] = temp_dict  # Add temporary dictionary to main dictionary with position as key
    # Return the dictionary containing suspicious positions
    return list_of_sus_pos


