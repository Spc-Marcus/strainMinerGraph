import pandas as pd
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import KNNImputer
import logging
from ..decorateur.perf import print_decorator

logger = logging.getLogger(__name__)



def create_matrix(dict_of_sus_pos : dict, min_coverage_threshold :int =0.6 ) -> tuple:
    """
    Preprocess a dictionary of suspicious positions to create a filtered matrix.
    
    This function converts a dictionary of genomic positions and their read values
    into a filtered pandas DataFrame and numpy matrix. It applies filtering to ensure
    reads span the entire window and positions have sufficient coverage.
    
    Parameters:
    -----------
    dict_of_sus_pos : dict
        Dictionary containing suspicious positions, where keys are genomic positions
        and values are dictionaries mapping {read_name: binary_value}
        Example: {1000: {'read1': 1, 'read2': 0}, 1001: {'read1': 0, 'read2': 1}}
    min_coverage_threshold : float, optional
        Minimum coverage threshold for columns (0.0-1.0). Columns with coverage
        below this threshold will be removed. Default is 0.6 (60% coverage).
    
    Returns:
    --------
    tuple
        A tuple containing:
        - matrix (numpy.ndarray): Filtered matrix of genomic variants with shape
          (n_reads, n_positions). Empty array if no data passes filtering.
        - reads (list): List of read names corresponding to matrix rows after
          filtering. Empty list if no reads pass filtering.
    
    Notes:
    ------
    The function applies three main filtering steps:
    1. Removes reads that have no data in the first third of positions
    2. Removes reads that have no data in the last third of positions  
    3. Removes positions (columns) with insufficient coverage
    
    This ensures that retained reads span the entire genomic window and that
    positions have enough data for reliable variant calling.
    """
    if not dict_of_sus_pos:
        return np.array([]), []
    
    # Create DataFrame from dictionary
    df = pd.DataFrame(dict_of_sus_pos)
    
    if df.empty:
        return np.array([]), []
    
    logger.debug(f"Processing {df.shape[1]} positions with {df.shape[0]} reads")
    
    variant_matrix = df.copy()
    
    for col in df.columns:
        position_data = df[col].dropna()  # Ignorer les NaN
        
        if len(position_data) == 0:
            continue
            
        allele_counts = position_data.value_counts()
        
        if len(allele_counts) == 0:
            continue
        elif len(allele_counts) == 1:
            # Position monomorphe - tous les reads ont la même valeur
            variant_matrix[col] = 1  # Tous identiques à la référence = 1
        else:
            majority_allele = allele_counts.index[0]  # Plus fréquent (référence)
            
            # CORRECTION: 1 = identique à la référence (majority), 0 = variant
            variant_matrix[col] = (df[col] == majority_allele).astype(int)
            
            # Conserver les NaN pour les positions manquantes
            variant_matrix.loc[df[col].isna(), col] = np.nan
            
        logger.debug(f"Position {col}: majority={majority_allele}, "
                    f"reference_calls={(variant_matrix[col] == 1).sum()}, "
                    f"variant_calls={(variant_matrix[col] == 0).sum()}")
    
    # Filter reads that span the window (existing logic)
    if len(variant_matrix.columns) >= 3:
        tmp_idx = variant_matrix.iloc[:, :len(variant_matrix.columns)//3].dropna(axis=0, how='all')
        variant_matrix = variant_matrix.loc[tmp_idx.index, :]
        
        tmp_idx = variant_matrix.iloc[:, 2*len(variant_matrix.columns)//3:].dropna(axis=0, how='all')
        variant_matrix = variant_matrix.loc[tmp_idx.index, :]
    
    # Filter columns with insufficient coverage
    if not variant_matrix.empty:
        variant_matrix = variant_matrix.dropna(axis=1, thresh=min_coverage_threshold * len(variant_matrix.index))
    
    # Extract filtered reads
    reads = list(variant_matrix.index)
    
    # Convert to numpy matrix
    matrix = variant_matrix.to_numpy() if not variant_matrix.empty else np.array([])
    
    logger.debug(f"Matrix created: {matrix.shape} with {len(reads)} reads")
    
    return matrix, reads

@print_decorator('preprocessing')
def pre_processing(input_matrix:np.ndarray ,min_col_quality :int = 3, default :int =0,certitude:float = 0.3)-> tuple:
    """
    Pre-processes the input matrix by imputing missing values, binarizing data, and identifying inhomogeneous regions.
    Steps performed:
    1. Fills missing values in the input matrix using K-Nearest Neighbors (KNN) imputation.
    2. Binarizes the matrix based on upper and lower thresholds (values >= 1 - certitude set to 1, <= certitude set to 0, others set to default).
    3. Splits the matrix into regions (groups of columns) using hierarchical clustering (FeatureAgglomeration) based on Hamming distance.
    4. Further splits large or noisy regions and identifies inhomogeneous regions based on the distribution of values.
    5. Optionally, splits homogeneous regions into two clusters using AgglomerativeClustering.
    Parameters
    ----------
    input_matrix : numpy.ndarray
        The input matrix to be pre-processed, with shape (m, n).
    min_col_quality : int, optional
        Minimum number of columns required for a region to be considered significant (default is 3).
    default : int, optional
        Default value for uncertain entries (default is 0).
    certitude : float, optional
        Certainty threshold for binarization (default is 0.3).
    Returns
    -------
    matrix : numpy.ndarray
        The processed matrix after imputation and binarization.
    inhomogenious_regions : list of list of int
        List of regions (each a list of column indices) identified as inhomogeneous.
    steps : list of tuple
        List of tuples describing the steps taken to split homogeneous regions. Each tuple contains:
            (cluster1_indices, cluster0_indices, region_columns)
    """
    
    m,n = input_matrix.shape
    if m > 0 and n > 0:
        if certitude <0 or certitude>=0.5:
            raise ValueError("Certitude must be in the range [0, 0.5).")
        # Impute missing values using KNN
        matrix = impute_missing_values(input_matrix, n_neighbors=10)
        matrix = binarize_matrix(matrix, certitude=certitude, default=default)
    else:
        matrix = input_matrix

    # Init regions and steps
    regions = [list(range(matrix.shape[1]))]
    inhomogenious_regions = [list(range(matrix.shape[1]))]
    steps = []

    if m>5 and n > 15:
        # Initial clustering of columns using FeatureAgglomeration
        agglo=FeatureAgglomeration(n_clusters=None, metric='hamming', linkage='complete', distance_threshold=0.35)
        agglo.fit(matrix)
        labels = agglo.labels_
        splitted_cols = {}
    
        # Group columns by their cluster labels
        for idx, label in enumerate(labels):
            if label in splitted_cols:
                splitted_cols[label].append(idx)
            else:
                splitted_cols[label] = [idx]
        
        regions = []        
        for cols in splitted_cols.values():
            
            # If the region is too large, remove the noisiest reads by splitting them with a strict distance
            # threshold, then take only the significant clusters
            if len(cols) > 15:
                matrix_reg = matrix[:,cols].copy()
                agglo=FeatureAgglomeration(n_clusters=None, metric='hamming', linkage='complete', distance_threshold=0.025)
                agglo.fit(matrix_reg)
                labels = agglo.labels_
                groups = {}

                for idx, label in enumerate(labels):
                    if label in groups:
                        groups[label].append(cols[idx])
                    else:
                        groups[label] = [cols[idx]]

                for c in groups.values():
                    if len(c) >= min_col_quality:
                        regions.append(c)
            elif len(cols) >= min_col_quality:
                regions.append(cols)
                
        # Identify inhomogeneous regions        
        inhomogenious_regions = []        
        for region in regions:
            thres = (min_col_quality/len(region))
            matrix_reg = matrix[:,region].copy()
            matrix_reg[matrix_reg==-1] = 0  # Treat uncertain values as 0
            x_matrix = (matrix_reg.sum(axis = 1))/len(region)  # Proportion of variants per read
            
            # Check if region is heterogeneous (many reads with intermediate proportions)
            if len(x_matrix[(x_matrix>=thres)*(x_matrix<=1-thres)]) > 10:
                inhomogenious_regions.append(region)
            else:
                # Cut into 2 regions of 1 and 0 for homogeneous regions
                agglo = AgglomerativeClustering(n_clusters=2, metric='hamming', linkage='complete')
                agglo.fit(matrix_reg)  
                labels = agglo.labels_  
                cluster1, cluster0 = [], []
                
                for idx, label in enumerate(labels):
                    if label == 0:
                        cluster0.append(idx)
                    else:
                        cluster1.append(idx)
                
                # Calculate cluster averages
                mat_cl1 = matrix_reg[cluster1,:].sum()/(len(cluster1)*len(region))
                mat_cl0 = matrix_reg[cluster0,:].sum()/(len(cluster0)*len(region))

                # Only keep well-separated clusters (>90% or <10%)
                if (mat_cl0 > 0.9 or mat_cl0 < 0.1) and (mat_cl1 > 0.9 or mat_cl1 < 0.1):
                    steps.append((cluster1, cluster0, region))
    # Log matrix statistics for debugging
    input_ones = (input_matrix == 1).sum()
    input_zeros = (input_matrix == 0).sum()
    input_nans = np.isnan(input_matrix).sum()
    input_total = input_matrix.size
    
    output_ones = (matrix == 1).sum()
    output_zeros = (matrix == 0).sum()
    output_total = matrix.size
    
    # Log debug info instead of saving to file
    logger.debug(f"Matrix Processing Debug Info:")
    logger.debug(f"INPUT MATRIX {input_matrix.shape}: "
               f"Total={input_total}, "
               f"Ones={input_ones} ({input_ones/input_total*100:.2f}%), "
               f"Zeros={input_zeros} ({input_zeros/input_total*100:.2f}%), "
               f"NaNs={input_nans} ({input_nans/input_total*100:.2f}%)")
    
    logger.debug(f"OUTPUT MATRIX {matrix.shape}: "
               f"Total={output_total}, "
               f"Ones={output_ones} ({output_ones/output_total*100:.2f}%), "
               f"Zeros={output_zeros} ({output_zeros/output_total*100:.2f}%)")
    
    logger.debug(f"REGIONS: Total={len(regions)}, "
               f"Inhomogeneous={len(inhomogenious_regions)}, "
               f"Steps={len(steps)}")
    return matrix, inhomogenious_regions, steps

@print_decorator('matrix')
def impute_missing_values(matrix: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """
    Impute missing values in a genomic variant matrix using K-Nearest Neighbors.
    
    This function fills missing or uncertain values in a variant matrix by leveraging
    patterns from similar reads. It uses the K-Nearest Neighbors algorithm to identify
    reads with similar variant patterns and imputes missing values based on the
    consensus of these neighbors.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix with shape (n_reads, n_positions) containing variant calls.
        Values can include:
        - 1: Reference allele or major variant
        - 0: Alternative allele or minor variant  
        - np.nan: Missing or uncertain values to be imputed
        - Other numerical values: Intermediate probabilities or quality scores
        
        Missing values (np.nan, None) will be replaced with estimated values
        based on K nearest neighbors.
        
    n_neighbors : int, optional
        Number of nearest neighbors to use for imputation. Must be positive
        and less than the number of reads with complete data.
        Default is 10.
        
    Returns
    -------
    np.ndarray
        Matrix with same shape as input but with missing values imputed.
        All np.nan values will be replaced with estimated values based on
        the K-nearest neighbors algorithm. Original non-missing values are
        preserved unchanged.
        
        **Value range**: Imputed values will be in the range of existing
        non-missing values in the dataset (typically 0-1 for binary variants).

    Raises
    ------
    ValueError
        If n_neighbors is not positive or exceeds the number of complete reads
    MemoryError
        If matrix is too large for available system memory
    RuntimeError
        If imputation fails due to insufficient non-missing data
        
    See Also
    --------
    binarize_matrix : Often applied after imputation to ensure binary values
    pre_processing : Comprehensive preprocessing pipeline including imputation
    sklearn.impute.KNNImputer : Underlying scikit-learn implementation
    """
    # Input validation
    if matrix.size == 0:
        return matrix.copy()
        
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
    
    # Check if there are any missing values to impute
    if not np.isnan(matrix).any():
        logger.debug("No missing values found, returning original matrix")
        return matrix.copy()
    
    # Check for sufficient non-missing data
    n_reads = matrix.shape[0]
    if n_neighbors >= n_reads:
        logger.warning(f"n_neighbors ({n_neighbors}) >= n_reads ({n_reads}), using {n_reads-1}")
        n_neighbors = max(1, n_reads - 1)
    
    try:
        # Initialize KNN imputer with specified parameters
        imputer = KNNImputer(
            n_neighbors=n_neighbors)
        
        # Perform imputation
        logger.debug(f"Imputing missing values using {n_neighbors} neighbors")
        imputed_matrix = imputer.fit_transform(matrix)
        
        # Log imputation statistics
        original_missing = np.isnan(matrix).sum()
        remaining_missing = np.isnan(imputed_matrix).sum()
        success_rate = (original_missing - remaining_missing) / max(1, original_missing) * 100
        
        logger.debug(f"Imputation completed: {original_missing} → {remaining_missing} missing values "
                   f"({success_rate:.1f}% success rate)")
        
        return imputed_matrix
        
    except Exception as e:
        logger.error(f"Imputation failed: {e}")
        raise RuntimeError(f"Could not impute missing values: {e}") from e


@print_decorator('matrix')
def binarize_matrix(matrix: np.ndarray, certitude: float = 0.3, default: int = 0) -> np.ndarray:
    """
    Binarize a continuous variant matrix using certainty-based thresholds.
    
    This function converts continuous or probabilistic variant values into binary
    classifications (0/1) based on confidence thresholds. Values with high certainty
    are assigned definitive binary values, while uncertain values receive a default
    assignment. This is essential for clustering algorithms that require binary input.
    
    The binarization strategy uses symmetric thresholds around 0.5 to classify
    values as definitely 0, definitely 1, or uncertain. This approach preserves
    high-confidence variant calls while handling ambiguous cases consistently.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix with shape (n_reads, n_positions) containing continuous
        variant values. Typically contains:
        - Values near 0: Strong evidence for reference allele
        - Values near 1: Strong evidence for alternative allele  
        - Values near 0.5: Uncertain/ambiguous evidence
        - Values between 0-1: Probabilistic variant calls
        
        **Expected input range**: Most algorithms assume values in [0, 1] range,
        though the function handles arbitrary numeric values.
        
    certitude : float, optional
        Certainty threshold for binary classification (0.0 ≤ certitude < 0.5).
        Default is 0.3 (30% certainty requirement).
        
    default : int, optional
        Default value assigned to uncertain entries that fall between thresholds
        Default is 0 (conservative approach).
        
    Returns
    -------
    np.ndarray
        Binarized matrix with same shape as input. All values converted to:
        - 0: Reference allele or confident non-variant
        - 1: Alternative allele or confident variant
        - default: Uncertain values (as specified by default parameter)
        
        **Data type**: Returns integer array for efficient processing in
        clustering algorithms.
        
    Raises
    ------
    ValueError
        If certitude is not in valid range [0, 0.5)
    TypeError
        If matrix contains non-numeric values
        
    See Also
    --------
    impute_missing_values : Often applied before binarization
    pre_processing : Complete preprocessing pipeline including binarization
    """
    # Input validation
    if not 0.0 <= certitude < 0.5:
        raise ValueError(f"Certitude must be in range [0, 0.5), got {certitude}")
    
    if matrix.size == 0:
        return matrix.astype(int)
    
    # Check for non-numeric values
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("Matrix must contain numeric values")
    
    # Calculate thresholds
    lower_threshold = certitude
    upper_threshold = 1.0 - certitude
    
    logger.debug(f"Binarizing matrix with thresholds: ≤{lower_threshold:.3f} → 0, "
                f"≥{upper_threshold:.3f} → 1, between → {default}")
    
    # Apply three-way classification using numpy vectorized operations
    # This is equivalent to nested np.where but more readable
    result = np.full_like(matrix, default, dtype=int)
    result[matrix <= lower_threshold] = 0
    result[matrix >= upper_threshold] = 1
    
    # Log binarization statistics
    n_zeros = (result == 0).sum()
    n_ones = (result == 1).sum()
    n_uncertain = (result == default).sum()
    n_total = result.size
    
    logger.debug(f"Binarization completed: {n_zeros} zeros ({n_zeros/n_total*100:.1f}%), "
               f"{n_ones} ones ({n_ones/n_total*100:.1f}%), "
               f"{n_uncertain} uncertain ({n_uncertain/n_total*100:.1f}%)")
    
    return result