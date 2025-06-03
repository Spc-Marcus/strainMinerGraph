import functools
import logging
from typing import Callable, Any, Optional
import numpy as np

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
    """Fonction appelée au début avec seulement la matrice."""
    print(f"###################################################### Debugging Start - Matrix Shape: {matrix.shape} ######################################################")

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
    """Décorateur spécialisé pour create_matrix."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_print_mode()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Appeler debug_matrix_start si mode start ou all
        if mode in ['start', 'all']:
            # result est un tuple (matrix, reads)
            if isinstance(result, (tuple, list)) and len(result) >= 1:
                matrix = result[0]
                if hasattr(matrix, 'shape') and matrix.size > 0:
                    debug_matrix_start(matrix)
        
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
        if func.__name__ == 'create_matrix':
            return create_matrix_decorator(func)
        elif func.__name__ == 'pre_processing':
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
