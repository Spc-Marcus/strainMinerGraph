import functools
import logging
from typing import Callable, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Import the C bitarray sorting module
BITARRAY_AVAILABLE = False
bitarray_heapsort = None

try:
    import bitarray_heapsort
    BITARRAY_AVAILABLE = True
except ImportError:
    try:
        from ..utils import bitarray_heapsort
        BITARRAY_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Variable globale pour stocker le mode d'affichage
_display_mode: Optional[str] = None

def set_display_mode(mode: str) -> None:
    """Configure le mode d'affichage global."""
    global _display_mode
    _display_mode = mode
    logger.debug(f"Display mode set to: {mode}")

def get_display_mode() -> Optional[str]:
    """Récupère le mode d'affichage actuel."""
    return _display_mode

# Backward compatibility for old print_mode interface
def set_print_mode(mode: str) -> None:
    """Backward compatibility function for set_print_mode.
    
    Deprecated: Use set_display_mode instead.
    """
    logger.warning("set_print_mode is deprecated, use set_display_mode instead")
    set_display_mode(mode)

def get_print_mode() -> Optional[str]:
    """Backward compatibility function for get_print_mode.
    
    Deprecated: Use get_display_mode instead.
    """
    logger.warning("get_print_mode is deprecated, use get_display_mode instead")
    return get_display_mode()

def display_matrix_start(matrix: np.ndarray) -> None:
    """Fonction pour afficher la matrice avec tri bitarray.
    Doit être appelée manuellement."""
    
    # Statistiques de la matrice
    total_elements = matrix.size
    ones = np.sum(matrix == 1)
    zeros = np.sum(matrix == 0)
    
    logger.info(f"Matrix density: {ones/total_elements*100:.2f}% ones, {zeros/total_elements*100:.2f}% zeros")
    
    # Visualisation seulement si la matrice n'est pas trop grande
    if matrix.size <= 10000:  # Limite pour éviter les graphiques trop lourds
        try:
            plt.figure(figsize=(15, 8))
            logger.debug(f"Matrix properties: dtype={matrix.dtype}, shape={matrix.shape}, C_CONTIGUOUS={matrix.flags['C_CONTIGUOUS']}")
            
            # Utiliser uniquement le tri bitarray
            if BITARRAY_AVAILABLE and bitarray_heapsort is not None:
                logger.debug("Using bitarray element-by-element sorting (1 > 0)")
                # Convertir en uint8 et s'assurer que c'est C-contiguous
                if matrix.dtype != np.uint8:
                    matrix_uint8 = matrix.astype(np.uint8)
                else:
                    matrix_uint8 = matrix
                
                # S'assurer que la matrice est C-contiguous
                if not matrix_uint8.flags['C_CONTIGUOUS']:
                    matrix_uint8 = np.ascontiguousarray(matrix_uint8)
                
                logger.debug(f"Converted matrix properties: dtype={matrix_uint8.dtype}, C_CONTIGUOUS={matrix_uint8.flags['C_CONTIGUOUS']}")
                
                sorted_matrix = bitarray_heapsort.sort_numpy(matrix_uint8)
                logger.debug("Matrix sorted using bitarray_heapsort C module with element-by-element comparison")
            else:
                # Message de debug et pas de tri
                logger.warning("bitarray_heapsort module not available - showing unsorted matrix")
                sorted_matrix = matrix
                
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
            logger.warning("matplotlib not available for visualization")
        except Exception as e:
            logger.error(f"Could not create visualization: {e}")
    else:
        logger.info(f"Matrix too large ({matrix.size} elements) for visualization - skipping plot")

def display_preprocessing(matrix: np.ndarray, steps: Any) -> None:
    """Fonction pour afficher les informations de preprocessing."""
    logger.info(f"Displaying Preprocessing - Matrix Shape: {matrix.shape}, Steps: {steps}")

def display_postprocessing(matrix: np.ndarray, steps: Any) -> None:
    """Fonction pour afficher les informations de postprocessing."""
    logger.info(f"Displaying Postprocessing - Matrix Shape: {matrix.shape}, Steps: {steps}")

def display_find_quasi_biclique(result_tuple: tuple) -> None:
    """Fonction pour afficher les résultats de find_quasi_biclique."""
    logger.info(f"Displaying Find Quasi Biclique - Result: {result_tuple}")

def create_matrix_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour create_matrix - passif."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Exécuter la fonction sans affichage automatique
        result = func(*args, **kwargs)
        return result
    return wrapper

def preprocessing_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour pre_processing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_display_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler display_matrix_start si mode start ou all (après imputation)
        if mode in ['start', 'all']:
            # result contient (matrix, regions, steps) - matrix est après imputation
            if hasattr(result, '__len__') and len(result) >= 1:
                matrix = result[0] if hasattr(result[0], 'shape') else None
                if matrix is not None and matrix.size > 0:
                    logger.info("Calling display_matrix_start from preprocessing_decorator (after imputation)")
                    display_matrix_start(matrix)
        
        # Appeler display_preprocessing si mode all
        if mode == 'all':
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            if matrix is not None and hasattr(result, '__len__') and len(result) >= 2:
                display_preprocessing(matrix, result[1])  # steps
        
        return result
    return wrapper

def postprocessing_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour post_processing."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_display_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler display_postprocessing si mode end ou all
        if mode in ['end', 'all']:
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            steps = args[1] if len(args) >= 2 else None
            if matrix is not None and steps is not None:
                display_postprocessing(matrix, steps)
        
        return result
    return wrapper

def find_quasi_biclique_decorator(func: Callable) -> Callable:
    """Décorateur spécialisé pour find_quasi_biclique."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_display_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler display_find_quasi_biclique si mode all
        if mode == 'all':
            display_find_quasi_biclique(result)
        
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
def print_decorator(display_type: str):
    """
    Décorateur principal qui dispatche vers les décorateurs spécialisés.
    
    Parameters
    ----------
    display_type : str
        Type d'affichage: 'matrix', 'preprocessing', 'clustering', 'postprocessing'
    """
    def decorator(func: Callable) -> Callable:
        logger.debug(f"Applying decorator for function '{func.__name__}' with type '{display_type}'")
        # Dispatcher vers le bon décorateur selon le nom de la fonction
        if func.__name__ == 'create_matrix':
            logger.debug("Applying create_matrix_decorator")
            return create_matrix_decorator(func)
        elif func.__name__ == 'pre_processing':
            return preprocessing_decorator(func)
        elif func.__name__ == 'post_processing':
            return postprocessing_decorator(func)
        elif func.__name__ == 'find_quasi_biclique':
            return find_quasi_biclique_decorator(func)
        elif display_type == 'matrix':
            return matrix_decorator(func)
        elif display_type == 'clustering':
            return clustering_decorator(func)
        else:
            # Pas de décorateur spécialisé, retourner la fonction inchangée
            return func
    
    return decorator
