import os
import sys
from .core import __version__
import argparse

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

def main():
    """
    Point d'entrée pour la ligne de commande.
    Parse les arguments et appelle la fonction init().
    """
    parser = argparse.ArgumentParser(description=f'StrainMiner version {__version__}')
    parser.add_argument('-a', '--assembly', required=True, help='Fichier d\'assemblage au format GFA')
    parser.add_argument('-b', '--bam', required=True, help='Fichier d\'alignement au format BAM')
    parser.add_argument('-r', '--reads', required=True, help='Fichier de lectures')
    parser.add_argument('-o', '--output', required=True, help='Dossier de sortie')
    parser.add_argument('-e', '--error-rate', type=float, default=0.025, help='Taux d\'erreur (défaut: 0.025)')
    parser.add_argument('-w', '--window', type=int, default=5000, help='Taille de la fenêtre (défaut: 5000)')
    
    args = parser.parse_args()
    
    return init(
        assembly_file=args.assembly,
        bam_file=args.bam,
        reads_file=args.reads,
        output_folder=args.output,
        error_rate=args.error_rate,
        window=args.window
    )

if __name__ == "__main__":
    sys.exit(main())