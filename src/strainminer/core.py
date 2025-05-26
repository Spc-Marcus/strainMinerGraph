import json
import pandas as pd 
import numpy as np
import sys
import os

import gurobipy as grb
import pysam as ps

from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances

#TODO: Importer les fonctions nécessaires pour le traitement des données
# Version du package
__version__ = "1.1.0"  # À remplacer par la version actuelle

def file_path(path=None):
    """
    Gestion des chemins de fichiers
    """
    # Implémentation de la fonction
    pass

def get_data(input_file: ps.AlignmentFile, contig_name : str, start_pos : int,end_pos :int) -> dict:
    """
    Récupération et traitement des données
    Parameters:
    -----------
    input_file : ps.AlignmentFile
        Fichier d'alignement contenant les données à traiter.
    contig_name : str
        Nom du contig à extraire.
    start_pos : int
        Position de début de l'extraction.
    end_pos : int
        Position de fin de l'extraction.
    Returns: 
    --------
    dict
        Dictionnaire contenant les données extraite de input_file.
    """
    intercol = input_file.pileup(contig = contig_name, start =start_pos,end =  end_pos,truncate=True, min_base_quality=10)
    list_of_sus_pos ={}
    # Parcourir les colonnes de pileup
    for pileupcolumn in intercol:
        # Vérifier si la colonne de pileup a au moins 5 segments
        if pileupcolumn.nsegments >= 5:

            temp_dict = {}  # Dictionnaire temporaire pour stocker les données de la colonne
            sequence = np.char.upper(np.array(pileupcolumn.get_query_sequences()))  # Convertir les séquences en majuscules
            bases,freq = np.unique(np.array(sequence), return_counts=True) # Obtenir les bases uniques et leurs fréquences

            if bases[0] == '':
                ratio = freq[1:] / sum(freq[1:]) # Calculer le ratio des bases (ignorer la base vide)
                bases = bases[1:] # Ignorer la base vide
            else:
                ratio = freq / sum(freq) # Calculer le ratio des bases
            
            if len(ratio) > 0:  # Vérifier si le ratio n'est pas vide
                idx_sort = np.argsort(ratio)[::-1]  # Obtenir les indices de tri des bases par ratio décroissant

                if ratio[idx_sort[0]] < 0.95: # Si le ratio de la base la plus fréquente est inférieur à 0.95
                    for pileupread in pileupcolumn.pileups: # Parcourir les lectures de la pileup
                        if not pileupread.is_del and not pileupread.is_refskip:
                            # Vérifier si la base de la lecture correspond à la base 
                            value = ( 1 if pileupread.alignment.query_sequence[pileupread.query_position] == bases[idx_sort[-1]] else 0) 
                            temp_dict[pileupread.alignment.query_name] = value # Ajouter la valeur au dictionnaire temporaire
                    list_of_sus_pos[pileupcolumn.pos] = temp_dict # Ajouter le dictionnaire temporaire au dictionnaire principal avec la position comme clé
    # Retourner le dictionnaire contenant les positions suspectes
    return list_of_sus_pos


def pre_processing(data, params=None):
    """
    Pré-traitement des données
    """
    # Implémentation de la fonction
    pass

def biclustering_full_matrix(matrix, params=None):
    """
    Application de l'algorithme de biclustering
    """
    # Implémentation de la fonction
    pass

def post_processing(results, output_path=None):
    """
    Post-traitement des résultats
    """
    # Implémentation de la fonction
    pass

def process_data(input_file, output_path=None, params=None):
    pass