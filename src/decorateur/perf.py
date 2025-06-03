import functools
import logging
from typing import Callable, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Variable globale pour stocker le mode d'impression
_print_mode: Optional[str] = None

def set_print_mode(mode: str) -> None:
    """Configure le mode d'impression global."""
    global _print_mode
    _print_mode = mode
    logger.debug(f"Print mode set to: {mode}")

def get_print_mode() -> Optional[str]:
    """Récupère le mode d'impression actuel."""
    return _print_mode

def debug_matrix_start(matrix: np.ndarray) -> None:
    """Fonction appelée au début avec seulement la matrice.
    Affiche la forme de la matrice et sa distribution."""
    
    # Statistiques de la matrice
    total_elements = matrix.size
    ones = np.sum(matrix == 1)
    zeros = np.sum(matrix == 0)
    
    print(f"  Matrix density: {ones/total_elements*100:.2f}% ones, {zeros/total_elements*100:.2f}% zeros")
    
    # Visualisation seulement si la matrice n'est pas trop grande
    if matrix.size <= 10000:  # Limite pour éviter les graphiques trop lourds
        try:
            plt.figure(figsize=(15, 8))
            
            # Trier la matrice pour la visualisation
            # Trier les lignes par nombre de 1s (densité décroissante)
            row_sums = np.sum(matrix, axis=1)
            row_order = np.argsort(row_sums)[::-1]  # ordre décroissant
            
            # Trier les colonnes par nombre de 1s (densité décroissante)
            col_sums = np.sum(matrix, axis=0)
            col_order = np.argsort(col_sums)[::-1]  # ordre décroissant
            
            # Appliquer le tri
            sorted_matrix = matrix[np.ix_(row_order, col_order)]
            
            plt.imshow(sorted_matrix, cmap='gray_r', interpolation='nearest', 
                      vmin=0, vmax=1, aspect='auto', extent=[0, matrix.shape[1], 0, matrix.shape[0]])
            
            # Titre avec les statistiques
            title = (f'Matrix Visualization - Shape: {matrix.shape}\n'
                    f'Total: {total_elements} | '
                    f'Ones: {ones} ({ones/total_elements*100:.1f}%) | '
                    f'Zeros: {zeros} ({zeros/total_elements*100:.1f}%)')
            
            plt.title(title, fontsize=10)
            cbar = plt.colorbar(label='Value')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['0', '1'])
            plt.xlabel('Genomic Positions')
            plt.ylabel('Reads')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available for visualization")
        except Exception as e:
            print(f"Could not create visualization: {e}")
    else:
        print(f"Matrix too large ({matrix.size} elements) for visualization - skipping plot")

def debug_preprocessing(matrix: np.ndarray, steps: Any) -> None:
    """Fonction appelée pour le preprocessing avec matrice et steps."""
    print(f"###################################################### Debugging Preprocessing - Matrix Shape: {matrix.shape}, Steps: {steps} ######################################################")

def debug_postprocessing(matrix: np.ndarray, steps: Any) -> None:
    """Fonction appelée pour le postprocessing avec matrice et steps."""
    print(f"###################################################### Debugging Postprocessing - Matrix Shape: {matrix.shape}, Steps: {steps} ######################################################")

def debug_find_quasi_biclique(result_tuple: tuple) -> None:
    """Fonction appelée avec le tuple de retour de find_quasi_biclique."""
    print(f"###################################################### Debugging Find Quasi Biclique - Result: {result_tuple} ######################################################")

def create_matrix_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour create_matrix - désactivé car appelé manuellement."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Ne plus appeler automatiquement debug_matrix_start
        result = func(*args, **kwargs)
        return result
    return wrapper

def preprocessing_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour pre_processing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_print_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler debug_preprocessing si mode all
        if mode == 'all':
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            if matrix is not None and hasattr(result, '__len__') and len(result) >= 2:
                debug_preprocessing(matrix, result[1])  # steps
        
        return result
    return wrapper

def postprocessing_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour post_processing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_print_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler debug_postprocessing si mode end ou all
        if mode in ['end', 'all']:
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            steps = args[1] if len(args) >= 2 else None
            if matrix is not None and steps is not None:
                debug_postprocessing(matrix, steps)
        
        return result
    return wrapper

def find_quasi_biclique_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour find_quasi_biclique."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_print_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler debug_find_quasi_biclique si mode all
        if mode == 'all':
            debug_find_quasi_biclique(result)
        
        return result
    return wrapper

def matrix_decorator(func: Callable) -> Callable:
    """Décorateur générique pour les autres fonctions de matrice."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Pas de debug spécifique pour les autres fonctions de matrice
        return func(*args, **kwargs)
    return wrapper

def clustering_decorator(func: Callable) -> Callable:
    """Décorateur générique pour les autres fonctions de clustering."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Pas de debug spécifique pour les autres fonctions de clustering
        return func(*args, **kwargs)
    return wrapper

# Décorateur principal qui dispatche vers les décorateurs spécialisés
def print_decorator(debug_type: str):
    """
    Décorateur principal qui dispatche vers les décorateurs spécialisés.
    
    Parameters
    ----------
    debug_type : str
        Type de debug: 'matrix', 'preprocessing', 'clustering', 'postprocessing'
    """
    def decorator(func: Callable) -> Callable:
        # Dispatcher vers le bon décorateur selon le nom de la fonction
        if func.__name__ == 'pre_processing':
            return preprocessing_decorator(func)
        elif func.__name__ == 'post_processing':
            return postprocessing_decorator(func)
        elif func.__name__ == 'find_quasi_biclique':
            return find_quasi_biclique_decorator(func)
        elif debug_type == 'matrix':
            return matrix_decorator(func)
        elif debug_type == 'clustering':
            return clustering_decorator(func)
        else:
            # Pas de décorateur spécialisé, retourner la fonction inchangée
            return func
    
    return decorator
