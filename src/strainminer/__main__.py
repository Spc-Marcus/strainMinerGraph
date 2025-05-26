import os
import sys
from .core import __version__, file_path, get_data, pre_processing, biclustering_full_matrix, post_processing

def init(assembly_file, bam_file, reads_file, output_folder, 
         error_rate=0.025, window=5000):
    """
    Initialise et exécute le pipeline StrainMiner
    
    Parameters:
    -----------
    assembly_file : str
        Fichier d'assemblage au format GFA
    bam_file : str
        Fichier d'alignement au format BAM
    reads_file : str
        Fichier de lectures
    output_folder : str
        Dossier de sortie
    error_rate : float, optional
        Taux d'erreur estimé des données (défaut: 0.025)
    window : int, optional
        Taille de la fenêtre pour la séparation des lectures (défaut: 5000)
    """
    print('StrainMiner version', __version__)
    
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Créer le dossier temporaire
    tmp_dir = os.path.join(output_folder, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # Créer un lien symbolique vers le répertoire build s'il n'existe pas
    strainminer_build_dir = os.path.join(os.path.dirname(__file__), 'build')
    root_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
    
    if not os.path.exists(strainminer_build_dir) and os.path.exists(root_build_dir):
        os.symlink(root_build_dir, strainminer_build_dir)
    
    # Assembler la commande et exécuter le pipeline
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'strainminer.py'),
        '-a', assembly_file,
        '-b', bam_file,
        '-r', reads_file,
        '-e', str(error_rate),
        '-o', output_folder,
        '--window', str(window)
    ]
    
    return os.system(' '.join(cmd))

# Exemple d'utilisation reste inchangé
if __name__ == "__main__":
    # Paramètres d'exemple - à remplacer par vos valeurs
    init(
        assembly_file="test_datasets/test_1/target.gfa",
        bam_file="test_datasets/test_1/alignment.bam", 
        reads_file="test_datasets/test_1/reads.fastq",
        output_folder="output_strainminer",
        error_rate=0.025,
        window=5000
    )