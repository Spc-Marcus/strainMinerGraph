import numpy as np
import logging
import time
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
from .stats import timed_matrix_operation

# Fix import path - decorateur is at src level, not within strainminer package
try:
    from ..decorateur.perf import print_decorator
except ImportError:
    # Fallback if decorateur module is not available
    def print_decorator(name):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@print_decorator('postprocessing')
@timed_matrix_operation('postprocessing')
def post_processing(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]], 
                   read_names: List[str], distance_thresh: float = 0.1) -> List[np.ndarray]:
    """
    Post-process clustering results and create final strain clusters.
    
    This function takes the output of the biclustering algorithm and creates final
    read clusters representing different microbial strains. It processes hierarchical
    clustering steps, filters small clusters, assigns orphaned reads, and merges
    similar clusters to produce biologically meaningful strain groups.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix with shape (n_reads, n_positions) containing
        processed variant calls (0, 1, or -1 for uncertain)
    steps : List[Tuple[List[int], List[int], List[int]]]
        List of clustering steps from biclustering algorithm. Each step contains:
        - reads1: List of read indices assigned to cluster 1
        - reads0: List of read indices assigned to cluster 0  
        - cols: List of column indices used for this clustering decision
    read_names : List[str]
        List of read names corresponding to matrix rows
    distance_thresh : float, optional
        Distance threshold for merging similar clusters using Hamming distance.
        Default is 0.1 (10% difference threshold)
        
    Returns
    -------
    List[np.ndarray]
        List of strain clusters, where each cluster is a numpy array of read names
        belonging to that strain. Empty list if no valid clusters found.
        
    Raises
    ------
    ValueError
        If matrix dimensions don't match read_names length
    RuntimeError
        If clustering post-processing fails
        
    See Also
    --------
    clustering_full_matrix : Previous step in the pipeline
    """
    start_time = time.time()
    
    try:
        if hasattr(steps, '__iter__') and not isinstance(steps, (list, tuple)):
            steps = list(steps)
            
        # Input validation
        if len(read_names) != matrix.shape[0]:
            raise ValueError(f"Matrix rows ({matrix.shape[0]}) don't match read_names length ({len(read_names)})")
        
        if matrix.size == 0 or len(read_names) == 0:
            logger.warning("Empty matrix or read names, returning empty clusters")
            return []
        
        logger.debug(f"Starting post-processing: {matrix.shape[0]} reads, {len(steps)} clustering steps")
        
        # Begin with all reads in the same group
        clusters = [list(range(len(read_names)))]
        
        # Go through each step and separate reads based on clustering decisions
        for step_idx, step in enumerate(steps):
            reads1, reads0, cols = step
            if hasattr(reads1, '__iter__') and not isinstance(reads1, (list, tuple, np.ndarray)):
                reads1 = list(reads1)
            if hasattr(reads0, '__iter__') and not isinstance(reads0, (list, tuple, np.ndarray)):
                reads0 = list(reads0)
            
            if len(reads1) == 0 or len(reads0) == 0:
                logger.debug(f"Skipping step {step_idx}: empty group (reads1={len(reads1)}, reads0={len(reads0)})")
                continue
                
            logger.debug(f"Processing step {step_idx}: {len(reads1)} vs {len(reads0)} reads, {len(cols)} columns")
            new_clusters = []
            
            # For each existing cluster, split it based on current step
            for cluster_idx, cluster in enumerate(clusters):
                clust1 = [c for c in cluster if c in reads1]
                clust0 = [c for c in cluster if c in reads0]
                
                # Only keep non-empty clusters
                if len(clust1) > 0:
                    new_clusters.append(clust1)
                if len(clust0) > 0:
                    new_clusters.append(clust0)
            
            clusters = new_clusters
        
        logger.debug(f"After hierarchical splitting: {len(clusters)} clusters")
        
        # Remove small clusters and collect orphaned reads
        min_cluster_size = 5
        orphaned_reads = []
        large_clusters = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) <= min_cluster_size:
                orphaned_reads.extend(cluster)
                logger.debug(f"Orphaning small cluster {cluster_idx} with {len(cluster)} reads")
            else:
                large_clusters.append(cluster)
        
        clusters = large_clusters
        logger.info(f"Kept {len(clusters)} large clusters, orphaned {len(orphaned_reads)} reads")
        
        if len(clusters) == 0:
            logger.warning("No large clusters found after filtering")
            return []
        
        logger.debug("Calculating cluster means...")
        
        # Calculate mean vectors for each cluster
        cluster_means = []
        for cluster_idx, cluster in enumerate(clusters):
            logger.debug(f"Calculating mean for cluster {cluster_idx} with {len(cluster)} reads")
            
            # Create mean vector ignoring uncertain values (-1)
            cluster_matrix = matrix[cluster]
            cluster_mean = np.zeros(cluster_matrix.shape[1])
            
            for col in range(cluster_matrix.shape[1]):
                col_data = cluster_matrix[:, col]
                valid_data = col_data[col_data != -1]  # Exclude uncertain values
                if len(valid_data) > 0:
                    cluster_mean[col] = np.round(np.mean(valid_data))
                else:
                    cluster_mean[col] = 0  # Default for columns with only uncertain values
            
            cluster_means.append(cluster_mean)
        
        logger.debug("Assigning orphaned reads...")
        
        # Assign orphaned reads to closest clusters
        reassigned_count = 0
        for read_idx in orphaned_reads:
            if read_idx >= matrix.shape[0]:
                continue
                
            read_vector = matrix[read_idx]
            
            # Calculate distances to all cluster means
            distances = []
            for cluster_mean in cluster_means:
                # Only consider positions where read has definite calls (not -1)
                valid_positions = read_vector != -1
                if np.any(valid_positions):
                    read_valid = read_vector[valid_positions]
                    mean_valid = cluster_mean[valid_positions]
                    if len(read_valid) > 0:
                        dist = np.mean(read_valid != mean_valid)  # Hamming distance
                    else:
                        dist = 1.0  # Maximum distance if no valid positions
                else:
                    dist = 1.0  # Maximum distance if all uncertain
                distances.append(dist)
            
            # Assign to closest cluster if distance is reasonable
            if distances:
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < 0.3:  # 30% difference threshold
                    clusters[min_dist_idx].append(read_idx)
                    reassigned_count += 1
        
        logger.info(f"Reassigned {reassigned_count} orphaned reads to existing clusters")
        
        logger.debug("Merging similar clusters...")
        
        # Merge similar clusters based on their mean vectors
        if len(clusters) > 1:
            # Recalculate means after reassignment
            final_cluster_means = []
            for cluster in clusters:
                cluster_matrix = matrix[cluster]
                cluster_mean = np.zeros(cluster_matrix.shape[1])
                
                for col in range(cluster_matrix.shape[1]):
                    col_data = cluster_matrix[:, col]
                    valid_data = col_data[col_data != -1]
                    if len(valid_data) > 0:
                        cluster_mean[col] = np.round(np.mean(valid_data))
                    else:
                        cluster_mean[col] = 0
                
                final_cluster_means.append(cluster_mean)
            
            # Hierarchical clustering of cluster means
            if len(final_cluster_means) > 1:
                agglo_cl = AgglomerativeClustering(
                    n_clusters=None,
                    metric='hamming',
                    linkage='complete',
                    distance_threshold=distance_thresh
                )
                cluster_labels = agglo_cl.fit_predict(final_cluster_means)
                
                # Merge clusters with same labels
                merged_clusters = {}
                for idx, label in enumerate(cluster_labels):
                    if label in merged_clusters:
                        merged_clusters[label].extend(clusters[idx])
                    else:
                        merged_clusters[label] = list(clusters[idx])
                
                clusters = list(merged_clusters.values())
                logger.info(f"Merged similar clusters to {len(clusters)} final strain groups")
        
        logger.debug("Converting to read names...")
        
        # Convert read indices to read names and sort
        result_clusters = []
        for cluster_idx, cluster in enumerate(clusters):
            logger.debug(f"Converting cluster {cluster_idx} with {len(cluster)} reads")
            
            if len(cluster) > 0:  # Only include non-empty clusters
                # CORRECTION: Ensure cluster is a list
                cluster_list = list(cluster) if not isinstance(cluster, list) else cluster
                
                # Filter valid indices and convert to read names
                valid_indices = [r for r in cluster_list if isinstance(r, (int, np.integer)) and 0 <= r < len(read_names)]
                
                if len(valid_indices) > 0:
                    cluster_names = [read_names[r] for r in valid_indices]
                    cluster_names_sorted = np.array(sorted(cluster_names))
                    result_clusters.append(cluster_names_sorted)
                    logger.debug(f"Final cluster {cluster_idx}: {len(cluster_names)} reads")
        
        logger.info(f"Post-processing completed: {len(result_clusters)} distinct strain clusters identified")
        
        # Calculer les métriques détaillées pour la visualisation
        processing_time = time.time() - start_time
        orphaned_count = len(orphaned_reads) if 'orphaned_reads' in locals() else 0
        total_reads_in_clusters = sum(len(cluster) for cluster in result_clusters)
        cluster_sizes = [len(cluster) for cluster in result_clusters]
        reassigned = reassigned_count if 'reassigned_count' in locals() else 0
        
        # Créer seulement la visualisation des steps et matrice simplifiée
        from datetime import datetime
        from pathlib import Path
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_dir = Path("stats")
        stats_dir.mkdir(exist_ok=True)
        
        try:
            simplified_matrix = create_step_visualization_matrix(matrix, steps, result_clusters, read_names)
            
            # Sauvegarder la matrice simplifiée
            matrix_file = stats_dir / f"simplified_matrix_{timestamp}.txt"
            
            with open(matrix_file, 'w', encoding='utf-8') as f:
                f.write("=== SIMPLIFIED CLUSTERING MATRIX ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Matrix shape: {simplified_matrix.shape[0]} clusters x {simplified_matrix.shape[1]} key positions\n\n")
                
                f.write("--- CLUSTERING STEPS SUMMARY ---\n")
                for step_idx, (reads1, reads0, cols) in enumerate(steps):
                    f.write(f"Step {step_idx + 1}: Split {len(reads1)} vs {len(reads0)} reads using {len(cols)} positions\n")
                    f.write(f"  Positions: {cols[:10]}{'...' if len(cols) > 10 else ''}\n")
                
                f.write("\n--- SIMPLIFIED MATRIX ---\n")
                f.write("Clusters (rows) vs Key Positions (columns)\n")
                
                # Identifier les colonnes importantes utilisées dans les steps
                important_cols = set()
                for _, _, cols in steps:
                    important_cols.update(cols)
                important_cols = sorted(list(important_cols))
                
                # En-tête des colonnes
                f.write("Cluster\\Position")
                for col in important_cols[:20]:  # Limiter à 20 colonnes pour la lisibilité
                    f.write(f"\t{col}")
                if len(important_cols) > 20:
                    f.write("\t...")
                f.write("\n")
                
                # Données de la matrice
                for cluster_idx in range(simplified_matrix.shape[0]):
                    f.write(f"Cluster_{cluster_idx + 1}")
                    for col_idx in range(min(20, simplified_matrix.shape[1])):
                        f.write(f"\t{int(simplified_matrix[cluster_idx, col_idx])}")
                    if simplified_matrix.shape[1] > 20:
                        f.write("\t...")
                    f.write(f"\t({len(result_clusters[cluster_idx])} reads)")
                    f.write("\n")
                
                f.write("\n--- CLUSTER DETAILS ---\n")
                for cluster_idx, cluster in enumerate(result_clusters):
                    f.write(f"Cluster_{cluster_idx + 1}: {len(cluster)} reads\n")
                    f.write(f"  Representative reads: {', '.join(cluster[:5])}\n")
                    if len(cluster) > 5:
                        f.write(f"  ... and {len(cluster) - 5} more\n")
                
                f.write("\n--- STEP-BY-STEP VISUALIZATION ---\n")
                f.write("Hierarchical clustering progression:\n")
                
                # Visualiser la progression du clustering
                current_groups = [list(range(len(read_names)))]
                f.write(f"Initial: 1 group with {len(read_names)} reads\n")
                
                for step_idx, (reads1, reads0, cols) in enumerate(steps):
                    new_groups = []
                    for group in current_groups:
                        group1 = [r for r in group if r in reads1]
                        group0 = [r for r in group if r in reads0]
                        if len(group1) > 0:
                            new_groups.append(group1)
                        if len(group0) > 0:
                            new_groups.append(group0)
                    
                    current_groups = new_groups
                    f.write(f"Step {step_idx + 1}: {len(current_groups)} groups - sizes: {[len(g) for g in current_groups]}\n")
                
                f.write(f"\nFinal: {len(result_clusters)} strain clusters after post-processing\n")
                f.write("=== END VISUALIZATION ===\n")
            
            logger.info(f"Simplified matrix visualization saved to {matrix_file}")
            
            # Sauvegarder aussi en format CSV pour analyse
            csv_file = stats_dir / f"simplified_matrix_{timestamp}.csv"
            if simplified_matrix.size > 0:
                np.savetxt(csv_file, simplified_matrix, delimiter=',', fmt='%d', 
                          header=f"Simplified matrix: {simplified_matrix.shape[0]} clusters x {simplified_matrix.shape[1]} positions")
                logger.info(f"Simplified matrix CSV saved to {csv_file}")
                
        except Exception as e:
            logger.warning(f"Failed to create step visualization: {e}")
        
        # Enregistrer les statistiques de post-processing
        processing_time = time.time() - start_time
        from .stats import get_stats
        stats = get_stats()
        if stats and stats.enabled:
            # Calculer les variables locales si elles n'existent pas
            orphaned_count = len(orphaned_reads) if 'orphaned_reads' in locals() else 0
            
            stats.record_matrix_processing(
                matrix_shape=matrix.shape,
                processing_time=processing_time,
                stage='postprocessing',
                input_steps=len(steps),
                output_clusters=len(result_clusters),
                orphaned_reads=orphaned_count,
                # Ajouter des champs optionnels avec des valeurs par défaut
                contig_name='',
                genomic_region='',
                positions_with_variants=0,
                contig_length=0,
                windows_processed=0,
                total_variants=0,
                reads_processed=matrix.shape[0]
            )
            
            # AJOUTER: Enregistrer AUSSI les statistiques de clustering pour le post-processing
            stats.record_clustering_step(
                matrix_shape=matrix.shape,
                execution_time=processing_time,
                iterations=1,  # Une seule itération de post-processing
                patterns_found=len(result_clusters),
                ilp_calls=0,   # Pas d'ILP dans le post-processing
                positive_reads=sum(len(cluster) for cluster in result_clusters),
                negative_reads=orphaned_count,
                final_cols=matrix.shape[1],
                region_id=-2,  # -2 pour indiquer le post-processing
                initial_cols=matrix.shape[1],
                error_rate=0.0,  # Pas applicable
                min_row_quality=5,  # Valeur par défaut
                min_col_quality=3   # Valeur par défaut
            )
        
        return result_clusters
        
    except Exception as e:
        logger.critical(f"Post-processing failed: {e}")
        raise RuntimeError(f"Could not complete post-processing: {e}") from e


def create_step_visualization_matrix(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]], 
                                   result_clusters: List[np.ndarray], read_names: List[str]) -> np.ndarray:
    """
    Créer une matrice de visualisation simplifiée montrant les étapes de clustering.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrice binaire originale
    steps : List[Tuple[List[int], List[int], List[int]]]
        Étapes de clustering
    result_clusters : List[np.ndarray]
        Clusters finaux
    read_names : List[str]
        Noms des reads
        
    Returns
    -------
    np.ndarray
        Matrice simplifiée avec lignes et colonnes fusionnées
    """
    # Créer un mapping des reads vers leurs clusters finaux
    read_to_cluster = {}
    for cluster_idx, cluster in enumerate(result_clusters):
        for read_name in cluster:
            read_idx = read_names.index(read_name) if read_name in read_names else -1
            if read_idx >= 0:
                read_to_cluster[read_idx] = cluster_idx
    
    # Identifier les colonnes importantes utilisées dans les steps
    important_cols = set()
    for _, _, cols in steps:
        important_cols.update(cols)
    important_cols = sorted(list(important_cols))
    
    # Créer la matrice simplifiée
    n_clusters = len(result_clusters)
    n_important_cols = len(important_cols)
    
    if n_clusters == 0 or n_important_cols == 0:
        return np.array([])
    
    simplified_matrix = np.zeros((n_clusters, n_important_cols))
    
    # Calculer les valeurs moyennes pour chaque cluster sur les colonnes importantes
    for cluster_idx, cluster in enumerate(result_clusters):
        read_indices = []
        for read_name in cluster:
            if read_name in read_names:
                read_idx = read_names.index(read_name)
                read_indices.append(read_idx)
        
        if read_indices:
            cluster_data = matrix[read_indices]
            for col_idx, original_col in enumerate(important_cols):
                if original_col < matrix.shape[1]:
                    col_data = cluster_data[:, original_col]
                    valid_data = col_data[col_data != -1]
                    if len(valid_data) > 0:
                        simplified_matrix[cluster_idx, col_idx] = np.round(np.mean(valid_data))
                    else:
                        simplified_matrix[cluster_idx, col_idx] = 0
    
    return simplified_matrix