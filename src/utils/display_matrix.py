from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging

# Import the C bitarray sorting module (optional performance enhancement)
BITARRAY_AVAILABLE = False
bitarray_heapsort = None

try:
    import bitarray_heapsort
    BITARRAY_AVAILABLE = True
except ImportError:
    try:
        from . import bitarray_heapsort
        BITARRAY_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)

def display_matrix(matrix: np.ndarray, sort: bool = False, title: str = None) -> None:
    """
    Display a binary variant matrix with optional bitarray sorting.
    
    This function provides the primary visualization for binary variant matrices
    throughout the StrainMiner pipeline. It can optionally apply bitarray-based
    sorting to reveal patterns in the data and shows comprehensive statistics
    about matrix density and composition.
    
    The function automatically handles large matrices by showing only statistical
    summaries to prevent performance issues with visualization.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix with shape (n_reads, n_positions).
        Expected values are 0 (variant), 1 (reference), or NaN (missing).
    sort : bool, optional
        Whether to apply bitarray heapsort to reveal patterns in the matrix.
        Uses element-by-element comparison (1 > 0) if bitarray module is available.
        Default is False.
    title : str, optional
        Custom title for the visualization. If None, generates automatic title
        with matrix dimensions and statistics. Default is None.
    
    Notes
    -----
    **Matrix Size Limitations**:
    - Only visualizes matrices with ≤10,000 total elements
    - Large matrices show statistical summaries in logs instead
    - This prevents memory issues and ensures responsive performance
    
    **Bitarray Sorting**:
    - Requires optional bitarray_heapsort C module for best performance
    - Falls back to unsorted display if module unavailable
    - Sorting reveals clustering patterns and data quality issues
    
    **Color Scheme**:
    - Black pixels: Reference alleles (value = 1)
    - White pixels: Variant alleles (value = 0)
    - Gray scale shows intermediate/uncertain values
    
    
    """
    # Calculate comprehensive matrix statistics
    total_elements = matrix.size
    ones = np.sum(matrix == 1)       # Reference alleles
    zeros = np.sum(matrix == 0)      # Variant alleles
    
    # Log density statistics for all matrices
    logger.info(f"Matrix density: {ones/total_elements*100:.2f}% ones, {zeros/total_elements*100:.2f}% zeros")
    
    # Only create visualizations for reasonably sized matrices
    if matrix.size <= 10000:  # Limit to prevent heavy graphics
        try:
            plt.figure(figsize=(15, 8))
            logger.debug(f"Matrix properties: dtype={matrix.dtype}, shape={matrix.shape}, C_CONTIGUOUS={matrix.flags['C_CONTIGUOUS']}")
            
            # Apply bitarray sorting if requested and available
            if sort and BITARRAY_AVAILABLE and bitarray_heapsort is not None:
                logger.debug("Using bitarray element-by-element sorting (1 > 0)")
                
                # Convert to uint8 and ensure C-contiguous layout for optimal performance
                if matrix.dtype != np.uint8:
                    matrix_uint8 = matrix.astype(np.uint8)
                else:
                    matrix_uint8 = matrix
                
                # Ensure C-contiguous memory layout for C module compatibility
                if not matrix_uint8.flags['C_CONTIGUOUS']:
                    matrix_uint8 = np.ascontiguousarray(matrix_uint8)
                
                logger.debug(f"Converted matrix properties: dtype={matrix_uint8.dtype}, C_CONTIGUOUS={matrix_uint8.flags['C_CONTIGUOUS']}")
                
                # Apply bitarray heapsort algorithm
                sorted_matrix = bitarray_heapsort.sort_numpy(matrix_uint8)
                logger.debug("Matrix sorted using bitarray_heapsort C module with element-by-element comparison")
            else:
                if sort:
                    logger.warning("bitarray_heapsort module not available - showing unsorted matrix")
                sorted_matrix = matrix
                
            # Create matrix visualization with proper coordinate system
            plt.imshow(sorted_matrix, cmap='gray_r', interpolation='nearest', 
                      vmin=0, vmax=1, aspect='auto', extent=[0, matrix.shape[1], 0, matrix.shape[0]])
            
            # Generate comprehensive title with statistics
            if title is None:
                title = f'Matrix Visualization - Shape: {matrix.shape}\n'
            else:
                title = f'{title} - Shape: {matrix.shape}\n'
                
            # Add detailed statistics to title
            title += (f'Total: {total_elements} | '
                     f'Ones: {ones} ({ones/total_elements*100:.1f}%) | '
                     f'Zeros: {zeros} ({zeros/total_elements*100:.1f}%)')
            
            # Finalize plot appearance
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

def display_matrix_step(matrix: np.ndarray, rw1: list, rw0: list, cols: list) -> None:
    """
    Display the results of a single clustering step.
    
    This function visualizes the outcome of one step in the clustering process,
    showing how reads and columns are organized after applying the clustering
    algorithm. It highlights the separation between different clusters and
    provides insights into the clustering progression.
    
    The visualization is similar to the raw matrix display, but only includes
    the reads and columns selected in this step. It also shows separation lines
    between rw1, rw0, and remaining reads/columns.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix with shape (n_reads, n_positions).
        Expected values are 0 (variant), 1 (reference), or NaN (missing).
    rw1 : list
        List of read indices belonging to cluster 1 (rw1) in this step.
    rw0 : list
        List of read indices belonging to cluster 0 (rw0) in this step.
    cols : list
        List of column indices selected in this step.
    """
    if matrix.size > 10000:
        logger.info("Matrix too large for step visualization")
        return
    
    try:
        # Reorganize rows: rw1, then rw0, then the rest
        all_rows = list(range(matrix.shape[0]))
        remaining_rows = [i for i in all_rows if i not in rw1 and i not in rw0]
        new_row_order = rw1 + rw0 + remaining_rows
        
        # Reorganize columns: selected cols first, then the rest
        all_cols = list(range(matrix.shape[1]))
        remaining_cols = [i for i in all_cols if i not in cols]
        new_col_order = cols + remaining_cols
        
        # Reorganize the matrix
        reordered_matrix = matrix[np.ix_(new_row_order, new_col_order)]
        
        plt.figure(figsize=(15, 8))
        # Use extent for precise coordinate control
        plt.imshow(reordered_matrix, cmap='gray_r', interpolation='nearest', 
                  vmin=0, vmax=1, aspect='auto', 
                  extent=[0, reordered_matrix.shape[1], 0, reordered_matrix.shape[0]])
        
        # Calculate correct positions for separation lines
        # Correction for inverted Y axis: use total height - position
        matrix_height = reordered_matrix.shape[0]
        
        # Calculate positions from bottom (as in extent)
        rw1_sep = matrix_height - len(rw1) if len(rw1) > 0 else None
        rw0_sep = matrix_height - (len(rw1) + len(rw0)) if len(rw0) > 0 else None
        cols_sep = len(cols) if len(cols) > 0 else None
        
        # Draw vertical separation line across full height
        if cols_sep is not None and len(cols) > 0:
            plt.axvline(x=cols_sep, color='green', linewidth=2, linestyle='-', label='cols separation')
        
        # Draw horizontal separation lines that stop at the vertical line
        if rw1_sep is not None and len(rw1) > 0:
            if cols_sep is not None:
                # Horizontal line that stops at vertical line
                plt.plot([0, cols_sep], [rw1_sep, rw1_sep], 
                        color='red', linewidth=2, linestyle='-', label='rw1/rw0 separation')
            else:
                # Horizontal line across full width if no vertical line
                plt.axhline(y=rw1_sep, color='red', linewidth=2, linestyle='-', label='rw1/rw0 separation')
        
        if rw0_sep is not None and len(rw0) > 0:
            if cols_sep is not None:
                # Horizontal line that stops at vertical line
                plt.plot([0, cols_sep], [rw0_sep, rw0_sep], 
                        color='blue', linewidth=2, linestyle='-', label='rw0/remaining separation')
            else:
                # Horizontal line across full width if no vertical line
                plt.axhline(y=rw0_sep, color='blue', linewidth=2, linestyle='-', label='rw0/remaining separation')
        
        # Title with information
        title = f'Clustering Step - rw1: {len(rw1)}, rw0: {len(rw0)}, cols: {len(cols)}'
        plt.title(title)
        plt.xlabel('Columns (selected | remaining)')
        plt.ylabel('Rows (rw1 | rw0 | remaining)')
        
        # Add legend if lines were drawn
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.colorbar(label='Value')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Could not create step visualization: {e}")

def display_matrix_steps(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]], title: str = None) -> None:
    pass

def display_clustered_matrix(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]], 
                           read_names: List[str], title: str = None) -> None:
    """
    Display the matrix with rows sorted according to clusters obtained from post-processing.
    
    This function reproduces the clustering logic from post-processing to visually reorganize
    the matrix and show how reads are grouped into clusters. It provides a clear visualization
    of the final clustering result, including boundaries between different strains with
    strain labels and annotations.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix with shape (n_reads, n_positions).
        Expected values are 0 (variant), 1 (reference), or NaN (missing).
    steps : List[Tuple[List[int], List[int], List[int]]]
        List of clustering steps from biclustering.
    read_names : List[str]
        List of read names corresponding to the matrix rows.
    title : str, optional
        Custom title for the display. Default is None.
    
    Notes
    -----
    **Enhanced Strain Visualization**:
    - Shows strain boundaries with colored separation lines
    - Annotates each strain cluster with "Strain N" labels
    - Displays strain statistics (read count, density)
    - Highlights orphaned reads separately from main strains
    """
    if matrix.size > 10000:
        logger.info("Matrix too large for clustered visualization")
        return
    
    if not steps:
        logger.warning("No clustering steps provided")
        display_matrix(matrix, sort=False, title=title or "Matrix (unclustered)")
        return
    
    try:
        # Reproduce clustering logic from post_processing
        clusters = [list(range(len(read_names)))]
        
        # Apply each clustering step
        for step_idx, step in enumerate(steps):
            reads1, reads0, cols = step
            
            # Convert to lists if necessary
            if hasattr(reads1, '__iter__') and not isinstance(reads1, (list, tuple, np.ndarray)):
                reads1 = list(reads1)
            if hasattr(reads0, '__iter__') and not isinstance(reads0, (list, tuple, np.ndarray)):
                reads0 = list(reads0)
            
            if len(reads1) == 0 or len(reads0) == 0:
                continue
                
            new_clusters = []
            
            # Split each existing cluster according to current step
            for cluster in clusters:
                clust1 = [c for c in cluster if c in reads1]
                clust0 = [c for c in cluster if c in reads0]
                
                if len(clust1) > 0:
                    new_clusters.append(clust1)
                if len(clust0) > 0:
                    new_clusters.append(clust0)
            
            clusters = new_clusters
        
        # Filter small clusters (as in post_processing)
        min_cluster_size = 5
        large_clusters = []
        orphaned_reads = []
        
        for cluster in clusters:
            if len(cluster) > min_cluster_size:
                large_clusters.append(cluster)
            else:
                orphaned_reads.extend(cluster)
        
        logger.info(f"Found {len(large_clusters)} large clusters, {len(orphaned_reads)} orphaned reads")
        
        # Create new row order: clusters first, then orphans
        new_row_order = []
        cluster_boundaries = []
        current_pos = 0
        
        for i, cluster in enumerate(large_clusters):
            new_row_order.extend(cluster)
            current_pos += len(cluster)
            cluster_boundaries.append(current_pos - 0.5)  # Separation position
        
        # Add orphaned reads at the end
        new_row_order.extend(orphaned_reads)
        
        # Reorganize matrix according to clusters
        if new_row_order:
            clustered_matrix = matrix[new_row_order, :]
        else:
            clustered_matrix = matrix
        
        # Display clustered matrix
        plt.figure(figsize=(15, 10))
        plt.imshow(clustered_matrix, cmap='gray_r', interpolation='nearest', 
                  vmin=0, vmax=1, aspect='auto',
                  extent=[0, clustered_matrix.shape[1], 0, clustered_matrix.shape[0]])
        
        # COMPLETE CORRECTION: Recalculate separation line positions
        matrix_height = clustered_matrix.shape[0]
        
        # Draw separation lines between clusters with strain annotations
        strain_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        strain_positions = []  # Store central positions of each strain
        
        for i, boundary in enumerate(cluster_boundaries[:-1]):  # No line after last cluster
            # Correction must be: matrix_height - boundary (without additional -0.5)
            corrected_y = matrix_height - boundary - 0.5  # Add missing -0.5
            color = strain_colors[i % len(strain_colors)]
            plt.axhline(y=corrected_y, color=color, 
                       linewidth=2, linestyle='-', alpha=0.8)
        
        # Calculate central positions for strain labels
        current_y = matrix_height
        for i, cluster in enumerate(large_clusters):
            cluster_height = len(cluster)
            strain_center_y = current_y - cluster_height / 2
            strain_positions.append((strain_center_y, cluster_height, len(cluster)))
            current_y -= cluster_height
        
        # Add strain labels
        for i, (center_y, height, read_count) in enumerate(strain_positions):
            strain_label = f"Strain {i+1}\n({read_count} reads)"
            color = strain_colors[i % len(strain_colors)]
            
            # Place label to the right of matrix
            plt.text(clustered_matrix.shape[1] + clustered_matrix.shape[1] * 0.02, 
                    center_y, strain_label, 
                    verticalalignment='center', 
                    horizontalalignment='left',
                    fontsize=10, 
                    color=color,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                             edgecolor=color, alpha=0.8))
        
        # Draw line to separate orphans if any
        if orphaned_reads and large_clusters:
            orphan_boundary = len(new_row_order) - len(orphaned_reads) - 0.5
            # Apply same Y axis correction with additional -0.5
            corrected_orphan_y = matrix_height - orphan_boundary - 0.5  # Add missing -0.5
            plt.axhline(y=corrected_orphan_y, color='orange', 
                       linewidth=2, linestyle='--', alpha=0.8)
            
            # Add label for orphans
            if len(orphaned_reads) > 0:
                orphan_center_y = corrected_orphan_y - len(orphaned_reads) / 2
                orphan_label = f"Orphaned\n({len(orphaned_reads)} reads)"
                plt.text(clustered_matrix.shape[1] + clustered_matrix.shape[1] * 0.02, 
                        orphan_center_y, orphan_label,
                        verticalalignment='center', 
                        horizontalalignment='left',
                        fontsize=10, 
                        color='orange',
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                 edgecolor='orange', alpha=0.8))
        
        # Titre avec informations sur les clusters
        if title is None:
            title = f'Clustered Matrix - {len(large_clusters)} strains, {len(orphaned_reads)} orphans'
        else:
            title = f'{title} - {len(large_clusters)} strains, {len(orphaned_reads)} orphans'
        
        title += f'\nShape: {clustered_matrix.shape}'
        
        # Statistiques de la matrice
        ones = np.sum(clustered_matrix == 1)
        zeros = np.sum(clustered_matrix == 0)
        total = clustered_matrix.size
        title += f' | Ones: {ones} ({ones/total*100:.1f}%) | Zeros: {zeros} ({zeros/total*100:.1f}%)'
        
        plt.title(title, fontsize=10)
        plt.xlabel('Genomic Positions')
        plt.ylabel('Reads (clustered by strain)')
        
        # Ajouter une colorbar
        cbar = plt.colorbar(label='Value')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0', '1'])
        
        # Ajouter une légende pour les lignes de séparation et souches
        if cluster_boundaries or orphaned_reads:
            from matplotlib.lines import Line2D
            legend_elements = []
            
            # Ajouter les souches à la légende
            for i in range(len(large_clusters)):
                color = strain_colors[i % len(strain_colors)]
                legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                            label=f'Strain {i+1} boundary'))
            
            # Ajouter les orphelins à la légende
            if orphaned_reads:
                legend_elements.append(Line2D([0], [0], color='orange', lw=2, 
                                            linestyle='--', label='Orphaned reads'))
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Adjust margins to leave space for labels
        plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.show()

        
        # Logger des informations détaillées sur les clusters
        logger.info(f"=== CLUSTERED MATRIX DISPLAY ===")
        for i, cluster in enumerate(large_clusters):
            cluster_matrix = clustered_matrix[sum(len(c) for c in large_clusters[:i]):
                                            sum(len(c) for c in large_clusters[:i+1]), :]
            if cluster_matrix.size > 0:
                strain_density = np.mean(cluster_matrix == 1) * 100
                logger.info(f"Strain {i+1}: {len(cluster)} reads, density: {strain_density:.1f}% ones")
            else:
                logger.info(f"Strain {i+1}: {len(cluster)} reads")
        if orphaned_reads:
            logger.info(f"Orphaned reads: {len(orphaned_reads)} reads")
        logger.info(f"=== END CLUSTERED MATRIX DISPLAY ===")
        
    except Exception as e:
        logger.error(f"Could not create clustered matrix visualization: {e}")
        # Fallback vers affichage normal
        display_matrix(matrix, sort=False, title=title or "Matrix (clustering failed)")