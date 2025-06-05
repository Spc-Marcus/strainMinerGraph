import time
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import functools

logger = logging.getLogger(__name__)

class PerformanceStats:
    """
    Collecteur de statistiques de performance pour StrainMiner.
    
    Collecte des métriques sur les temps d'exécution, tailles de matrices,
    nombre d'appels aux solveurs ILP, etc.
    """
    
    def __init__(self, output_dir: str, enabled: bool = False):
        self.enabled = enabled
        self.output_dir = Path(output_dir) / "stats"
        self.data_lock = threading.Lock()
        
        # Dictionnaires pour stocker les données
        self.clustering_stats = []
        self.ilp_stats = []
        self.matrix_stats = []
        self.preprocessing_stats = []
        self.benchmark_stats = []  # Nouvelles statistiques de benchmark
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Performance statistics enabled, output dir: {self.output_dir}")
    
    def record_benchmark_data(self, stage: str, matrix_shape: tuple, execution_time: float, **kwargs):
        """Enregistrer les données de benchmark complètes."""
        if not self.enabled:
            return
            
        with self.data_lock:
            record = {
                'timestamp': datetime.now().isoformat(),
                'stage': stage,
                'matrix_rows': matrix_shape[0],
                'matrix_cols': matrix_shape[1],
                'matrix_size': matrix_shape[0] * matrix_shape[1],
                'execution_time_sec': execution_time,
                **kwargs
            }
            self.benchmark_stats.append(record)
            logger.debug(f"Benchmark recorded: {stage} - {matrix_shape} - {execution_time:.3f}s")
    
    def record_matrix_processing(self, matrix_shape: tuple, processing_time: float, 
                                stage: str, **kwargs):
        """Enregistrer les statistiques de traitement de matrice."""
        if not self.enabled:
            return
            
        with self.data_lock:
            record = {
                'timestamp': datetime.now().isoformat(),
                'stage': stage,
                'matrix_rows': matrix_shape[0],
                'matrix_cols': matrix_shape[1],
                'matrix_size': matrix_shape[0] * matrix_shape[1],
                'processing_time_sec': processing_time,
                **kwargs
            }
            self.matrix_stats.append(record)
            logger.debug(f"Matrix processing recorded: {stage} - {matrix_shape}")
    
    def record_clustering_step(self, matrix_shape: tuple, execution_time: float,
                              iterations: int, patterns_found: int, **kwargs):
        """Enregistrer les statistiques d'une étape de clustering."""
        if not self.enabled:
            return
            
        with self.data_lock:
            record = {
                'timestamp': datetime.now().isoformat(),
                'matrix_rows': matrix_shape[0],
                'matrix_cols': matrix_shape[1],
                'matrix_size': matrix_shape[0] * matrix_shape[1],
                'execution_time_sec': execution_time,
                'iterations': iterations,
                'patterns_found': patterns_found,
                **kwargs
            }
            self.clustering_stats.append(record)
            logger.debug(f"Clustering step recorded: {matrix_shape} - {iterations} iterations - {patterns_found} patterns")
    
    def record_ilp_call(self, matrix_shape: tuple, execution_time: float,
                       solver_status: str, objective_value: float,
                       phase: str, **kwargs):
        """Enregistrer les statistiques d'un appel au solveur ILP."""
        if not self.enabled:
            return
            
        with self.data_lock:
            record = {
                'timestamp': datetime.now().isoformat(),
                'matrix_rows': matrix_shape[0],
                'matrix_cols': matrix_shape[1],
                'matrix_size': matrix_shape[0] * matrix_shape[1],
                'execution_time_sec': execution_time,
                'solver_status': solver_status,
                'objective_value': objective_value,
                'phase': phase,
                **kwargs
            }
            self.ilp_stats.append(record)
            logger.debug(f"ILP call recorded: {phase} - {solver_status} - {execution_time:.3f}s")
    
    def record_preprocessing(self, input_shape: tuple, output_shape: tuple,
                           execution_time: float, **kwargs):
        """Enregistrer les statistiques de préprocessing."""
        if not self.enabled:
            return
            
        with self.data_lock:
            record = {
                'timestamp': datetime.now().isoformat(),
                'input_rows': input_shape[0],
                'input_cols': input_shape[1],
                'input_size': input_shape[0] * input_shape[1],
                'output_rows': output_shape[0],
                'output_cols': output_shape[1],
                'output_size': output_shape[0] * output_shape[1],
                'execution_time_sec': execution_time,
                **kwargs
            }
            self.preprocessing_stats.append(record)
            logger.debug(f"Preprocessing recorded: {input_shape} -> {output_shape}")
    
    def save_all_stats(self):
        """Sauvegarder toutes les statistiques dans des fichiers CSV."""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les statistiques de clustering
        if self.clustering_stats:
            clustering_file = self.output_dir / f"clustering_stats_{timestamp}.csv"
            self._save_to_csv_flexible(self.clustering_stats, clustering_file)
            logger.info(f"Clustering statistics saved to {clustering_file}")
        
        # Sauvegarder les statistiques ILP
        if self.ilp_stats:
            ilp_file = self.output_dir / f"ilp_stats_{timestamp}.csv"
            self._save_to_csv_flexible(self.ilp_stats, ilp_file)
            logger.info(f"ILP statistics saved to {ilp_file}")
        
        # Sauvegarder les statistiques de matrices
        if self.matrix_stats:
            matrix_file = self.output_dir / f"matrix_stats_{timestamp}.csv"
            self._save_to_csv_flexible(self.matrix_stats, matrix_file)
            logger.info(f"Matrix statistics saved to {matrix_file}")
        
        # Sauvegarder les statistiques de préprocessing
        if self.preprocessing_stats:
            preprocessing_file = self.output_dir / f"preprocessing_stats_{timestamp}.csv"
            self._save_to_csv_flexible(self.preprocessing_stats, preprocessing_file)
            logger.info(f"Preprocessing statistics saved to {preprocessing_file}")
        
        # Sauvegarder les statistiques de benchmark
        if self.benchmark_stats:
            benchmark_file = self.output_dir / f"benchmark_stats_{timestamp}.csv"
            self._save_to_csv_flexible(self.benchmark_stats, benchmark_file)
            logger.info(f"Benchmark statistics saved to {benchmark_file}")
    
    def _save_to_csv_flexible(self, data: List[Dict], filename: Path):
        """Sauvegarder une liste de dictionnaires dans un fichier CSV avec gestion flexible des champs."""
        if not data:
            return
            
        try:
            # Collecter tous les champs possibles de tous les enregistrements
            all_fieldnames = set()
            for record in data:
                all_fieldnames.update(record.keys())
            
            # Trier les champs pour avoir un ordre cohérent
            fieldnames = sorted(list(all_fieldnames))
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                # Écrire chaque enregistrement en remplissant les champs manquants avec des valeurs par défaut
                for record in data:
                    complete_record = {}
                    for field in fieldnames:
                        complete_record[field] = record.get(field, '')  # Valeur par défaut vide
                    writer.writerow(complete_record)
                    
        except Exception as e:
            logger.error(f"Failed to save statistics to {filename}: {e}")

    def _save_to_csv(self, data: List[Dict], filename: Path):
        """Ancienne méthode de sauvegarde (conservée pour compatibilité)."""
        # Utiliser la nouvelle méthode flexible
        self._save_to_csv_flexible(data, filename)

# Instance globale du collecteur de statistiques
_global_stats: Optional[PerformanceStats] = None

def initialize_stats(output_dir: str, enabled: bool = False):
    """Initialiser le collecteur de statistiques global."""
    global _global_stats
    _global_stats = PerformanceStats(output_dir, enabled)

def get_stats() -> Optional[PerformanceStats]:
    """Obtenir l'instance globale du collecteur de statistiques."""
    return _global_stats

def timed_matrix_operation(stage: str, **extra_kwargs):
    """
    Décorateur pour mesurer le temps d'exécution des opérations sur matrices.
    
    Parameters
    ----------
    stage : str
        Nom de l'étape (e.g., 'imputation', 'binarization', 'clustering')
    **extra_kwargs
        Arguments supplémentaires à enregistrer
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stats = get_stats()
            if not stats or not stats.enabled:
                return func(*args, **kwargs)
            
            # Essayer de détecter la matrice dans les arguments
            matrix = None
            if args and hasattr(args[0], 'shape'):
                matrix = args[0]
            elif 'matrix' in kwargs and hasattr(kwargs['matrix'], 'shape'):
                matrix = kwargs['matrix']
            elif 'input_matrix' in kwargs and hasattr(kwargs['input_matrix'], 'shape'):
                matrix = kwargs['input_matrix']
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if matrix is not None:
                stats.record_matrix_processing(
                    matrix_shape=matrix.shape,
                    processing_time=execution_time,
                    stage=stage,
                    **extra_kwargs
                )
            
            return result
        return wrapper
    return decorator

def timed_ilp_call(phase: str):
    """
    Décorateur spécialisé pour mesurer les appels ILP.
    
    Parameters
    ----------
    phase : str
        Phase de l'optimisation ('seeding', 'row_extension', 'col_extension')
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stats = get_stats()
            if not stats or not stats.enabled:
                return func(*args, **kwargs)
            
            # Détecter la matrice d'entrée
            matrix = None
            if args and hasattr(args[0], 'shape'):
                matrix = args[0]
            elif 'input_matrix' in kwargs and hasattr(kwargs['input_matrix'], 'shape'):
                matrix = kwargs['input_matrix']
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if matrix is not None:
                # Extraire des informations du résultat
                rows, cols, success = result if isinstance(result, tuple) and len(result) == 3 else ([], [], False)
                objective_value = len(rows) * len(cols) if success else 0
                solver_status = 'optimal' if success else 'failed'
                
                stats.record_ilp_call(
                    matrix_shape=matrix.shape,
                    execution_time=execution_time,
                    solver_status=solver_status,
                    objective_value=objective_value,
                    phase=phase,
                    rows_selected=len(rows),
                    cols_selected=len(cols)
                )
            
            return result
        return wrapper
    return decorator
