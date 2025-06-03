"""
Performance and Display Decorators Module

This module provides decorators for monitoring pipeline performance and controlling
visualization display modes throughout the StrainMiner pipeline execution.

The decorators integrate with the display_matrix utilities to provide configurable
visualization of matrices, clustering steps, and processing results at different
stages of the pipeline.

Key Features:
- Configurable display modes (start, end, all)
- Function-specific decorators for different pipeline stages
- Matrix visualization integration
- Backward compatibility with legacy print_mode interface

Display Modes:
- 'start': Show matrices after initial creation and preprocessing
- 'end': Show final results after post-processing
- 'all': Show comprehensive debugging output at all stages

Example Usage:
    # Configure display mode globally
    set_display_mode('all')
    
    # Functions decorated with @print_decorator will automatically
    # display matrices based on the configured mode
    
    @print_decorator('matrix')
    def create_matrix(...):
        # Matrix creation logic
        return matrix, reads
"""

import functools
import logging
from typing import Callable, Any, List, Optional, Tuple
import numpy as np

from ..utils.display_matrix import display_matrix, display_matrix_step, display_matrix_steps, display_clustered_matrix

logger = logging.getLogger(__name__)

# Global variable to store the current display mode
_display_mode: Optional[str] = None

def set_display_mode(mode: str) -> None:
    """
    Configure the global display mode for matrix visualizations.
    
    This function sets the visualization behavior for all decorated functions
    throughout the pipeline execution. The mode determines which matrices
    and clustering steps are displayed during processing.
    
    Parameters
    ----------
    mode : str
        Display mode to activate. Valid options:
        - 'start': Display matrices after creation and preprocessing
        - 'end': Display final clustered results after post-processing
        - 'all': Display comprehensive output at all pipeline stages
        - None: Disable all automatic visualizations
    """
    global _display_mode
    _display_mode = mode
    logger.debug(f"Display mode set to: {mode}")

def get_display_mode() -> Optional[str]:
    """
    Retrieve the current display mode setting.
    
    Returns
    -------
    Optional[str]
        Current display mode ('start', 'end', 'all', or None)
    
    Examples
    --------
    >>> current_mode = get_display_mode()
    >>> if current_mode == 'all':
    ...     print("Comprehensive debugging is enabled")
    """
    return _display_mode

# Backward compatibility functions for legacy print_mode interface
def set_print_mode(mode: str) -> None:
    """
    Backward compatibility function for legacy print_mode interface.
    
    Deprecated: Use set_display_mode instead.
    
    Parameters
    ----------
    mode : str
        Display mode (same as set_display_mode)
    """
    logger.warning("set_print_mode is deprecated, use set_display_mode instead")
    set_display_mode(mode)

def get_print_mode() -> Optional[str]:
    """
    Backward compatibility function for legacy print_mode interface.
    
    Deprecated: Use get_display_mode instead.
    
    Returns
    -------
    Optional[str]
        Current display mode
    """
    logger.warning("get_print_mode is deprecated, use get_display_mode instead")
    return get_display_mode()

def display_matrix_start(matrix: np.ndarray) -> None:
    """
    Display matrix with bitarray sorting for initial pipeline stages.
    
    This function shows the matrix after creation and preprocessing,
    with reads sorted using the bitarray heapsort algorithm to reveal
    patterns and data quality.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix to display
    """
    display_matrix(matrix, sort=True, title="Matrix After Creation")

def display_preprocessing(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]]) -> None:
    """
    Display preprocessing information and statistics.
    
    Shows summary information about the preprocessing steps including
    region identification, inhomogeneous region detection, and
    preliminary clustering decisions.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix that was preprocessed
    steps : List[Tuple[List[int], List[int], List[int]]]
        List of preprocessing steps, where each tuple contains:
        - reads1: List of read indices for cluster 1
        - reads0: List of read indices for cluster 0
        - cols: List of column indices for the region
    """
    # Show all cluster
    display_matrix_steps(matrix, steps, title="pre-processing Steps")

def display_postprocessing(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]], read_names: List[str]) -> None:
    """
    Display comprehensive post-processing results including final clustered matrix.
    
    This function shows both statistical summaries of the clustering steps
    and a visual representation of the final clustered matrix with reads
    organized by strain groups and cluster boundaries marked.
    
    Parameters
    ----------
    matrix : np.ndarray
        Original binary variant matrix
    steps : List[Tuple[List[int], List[int], List[int]]]
        Complete list of biclustering steps applied during processing
    read_names : List[str]
        List of read names corresponding to matrix rows
    """
    # Show all cluster
    display_matrix_steps(matrix, steps, title="Post-processing Steps")
    
    # Show visual representation of final clustered matrix
    display_clustered_matrix(matrix, steps, read_names, title="Final Clustered Matrix")

def display_clustering_step(matrix: np.ndarray, rw1: list, rw0: list, cols: list) -> None:
    """
    Display results of a single clustering step during biclustering.
    
    Shows the matrix reorganized according to a specific clustering decision,
    with visual separation lines indicating how reads and positions were
    divided into different clusters.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix
    rw1 : list
        List of read indices assigned to cluster 1
    rw0 : list
        List of read indices assigned to cluster 0
    cols : list
        List of genomic position indices used for this clustering step
    """
    display_matrix_step(matrix, rw1, rw0, cols)

def create_matrix_decorator(func: Callable) -> Callable:
    """
    Specialized decorator for create_matrix function - passive monitoring.
    
    This decorator wraps the create_matrix function but does not trigger
    automatic visualizations. The matrix display is handled by the
    preprocessing decorator to show the matrix after imputation.
    
    Parameters
    ----------
    func : Callable
        The create_matrix function to decorate
    
    Returns
    -------
    Callable
        Wrapped function with identical behavior
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Execute function without automatic display
        # Matrix visualization is handled by preprocessing decorator
        result = func(*args, **kwargs)
        return result
    return wrapper

def preprocessing_decorator(func: Callable) -> Callable:
    """
    Specialized decorator for pre_processing function.
    
    Handles visualization of matrices after preprocessing steps including
    imputation and binarization. Shows the matrix at the 'start' display
    mode and preprocessing steps in 'all' mode.
    
    Parameters
    ----------
    func : Callable
        The pre_processing function to decorate
    
    Returns
    -------
    Callable
        Wrapped function with conditional visualization
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_display_mode()
        
        # Execute the preprocessing function
        result = func(*args, **kwargs)
        
        # Display matrix after imputation/binarization if mode is 'start' or 'all'
        if mode in ['start', 'all']:
            # result contains (matrix, regions, steps) - matrix is after imputation
            if hasattr(result, '__len__') and len(result) >= 1:
                matrix = result[0] if hasattr(result[0], 'shape') else None
                if matrix is not None and matrix.size > 0:
                    logger.info("Displaying matrix after preprocessing (imputation and binarization)")
                    display_matrix_start(matrix)
        
        # Display preprocessing steps summary if mode is 'all'
        if mode == 'all':
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            if matrix is not None and hasattr(result, '__len__') and len(result) >= 3:
                # result = (matrix, regions, steps) - steps is at index 2
                logger.info("Displaying preprocessing steps summary")
                display_preprocessing(matrix, result[2])
        
        return result
    return wrapper

def postprocessing_decorator(func: Callable) -> Callable:
    """
    Specialized decorator for post_processing function.
    
    Handles visualization of final clustering results including strain
    separation and cluster organization. Shows comprehensive results
    in 'end' and 'all' display modes.
    
    Parameters
    ----------
    func : Callable
        The post_processing function to decorate
    
    Returns
    -------
    Callable
        Wrapped function with conditional visualization
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_display_mode()
        
        # Execute the post-processing function
        result = func(*args, **kwargs)
        
        # Display post-processing results if mode is 'end' or 'all'
        if mode in ['end', 'all']:
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            steps = args[1] if len(args) >= 2 else None
            read_names = args[2] if len(args) >= 3 else None
            if matrix is not None and steps is not None and read_names is not None:
                logger.info("Displaying post-processing results and final clustered matrix")
                display_postprocessing(matrix, steps, read_names)
        
        return result
    return wrapper

def clustering_step_decorator(func: Callable) -> Callable:
    """
    Specialized decorator for individual clustering step functions.
    
    Handles visualization of individual biclustering decisions during
    the iterative clustering process. Shows step-by-step matrix
    reorganization in 'all' display mode only.
    
    Parameters
    ----------
    func : Callable
        A clustering step function to decorate
    
    Returns
    -------
    Callable
        Wrapped function with conditional step visualization
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mode = get_display_mode()
        
        # Execute the clustering step function
        result = func(*args, **kwargs)
        
        # Display clustering step only in 'all' mode
        if mode == 'all':
            matrix = args[0] if args and hasattr(args[0], 'shape') else None
            if matrix is not None and len(result) == 3:
                rw1, rw0, cols = result
                logger.info("Displaying individual clustering step results")
                display_clustering_step(matrix, rw1, rw0, cols)
        
        return result
    return wrapper

def print_decorator(display_type: str):
    """
    Main decorator factory that dispatches to specialized decorators.
    
    This is the primary decorator used throughout the StrainMiner pipeline.
    It automatically selects the appropriate specialized decorator based on
    the function name and display type.
    
    Parameters
    ----------
    display_type : str
        Type of display for the decorated function. Valid options:
        - 'matrix': Matrix creation and manipulation functions
        - 'preprocessing': Data preprocessing and filtering functions
        - 'clustering': Biclustering algorithm functions
        - 'postprocessing': Final clustering and strain separation functions
    
    Returns
    -------
    Callable
        Decorator function that applies appropriate visualization behavior
    
    Function Dispatch Logic
    ----------------------
    - 'create_matrix': Uses create_matrix_decorator (passive)
    - 'pre_processing': Uses preprocessing_decorator (start/all modes)
    - 'post_processing': Uses postprocessing_decorator (end/all modes)
    - 'clustering_step': Uses clustering_step_decorator (all mode only)
    - Other functions: No special behavior (passthrough)
    
   
    Notes
    -----
    - Decorator selection is based on exact function name matching
    - Unrecognized functions are returned unchanged (no decoration)
    - All decorators preserve original function signatures and behavior
    - Visualization behavior depends on current display_mode setting
    """
    def decorator(func: Callable) -> Callable:
        logger.debug(f"Applying decorator for function '{func.__name__}' with type '{display_type}'")
        
        # Dispatch to appropriate specialized decorator based on function name
        if func.__name__ == 'create_matrix':
            logger.debug("Applying create_matrix_decorator")
            return create_matrix_decorator(func)
        elif func.__name__ == 'pre_processing':
            logger.debug("Applying preprocessing_decorator")
            return preprocessing_decorator(func)
        elif func.__name__ == 'post_processing':
            logger.debug("Applying postprocessing_decorator")
            return postprocessing_decorator(func)
        elif func.__name__ == 'clustering_step':
            logger.debug("Applying clustering_step_decorator")
            return clustering_step_decorator(func)
        else:
            # No specialized decorator available, return function unchanged
            logger.debug(f"No specialized decorator for '{func.__name__}', returning unchanged")
            return func
    
    return decorator
