import pandas as pd
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import KNNImputer


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
    # Create DataFrame from dictionary
    df = pd.DataFrame(dict_of_sus_pos)
    
    if df.empty:
        return np.array([]), []
    
    # Filter reads that have no data in the first third of columns
    if len(df.columns) >= 3:  # Check sufficient columns for thirds division
        tmp_idx = df.iloc[:, :len(df.columns)//3].dropna(axis=0, how='all')
        df = df.loc[tmp_idx.index, :]
        
        # Filter reads that have no data in the last third of columns
        tmp_idx = df.iloc[:, 2*len(df.columns)//3:].dropna(axis=0, how='all')
        df = df.loc[tmp_idx.index, :]
    
    # Filter columns with insufficient data coverage
    if not df.empty:
        df = df.dropna(axis=1, thresh=min_coverage_threshold * len(df.index))
    
    # Extract list of reads after filtering
    reads = list(df.index)
    
    # Convert to numpy matrix for return
    matrix = df.to_numpy() if not df.empty else np.array([])
    
    return matrix, reads

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
        upper,lower = 1 - certitude, certitude
        # Impute missing values using KNN
        imputer = KNNImputer(n_neighbors=10)
        matrix = imputer.fit_transform(input_matrix)
        # Binarize the matrix
        matrix = np.where(matrix >= upper, 1, np.where(matrix <= lower, 0, default))
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

    return matrix, inhomogenious_regions, steps