from typing import List, Tuple
import numpy as np
import gurobipy as grb
import logging
import os
import sys
import time
import contextlib
from io import StringIO
from .stats import get_stats, timed_matrix_operation, timed_ilp_call

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

# Suppress ALL Gurobi output
os.environ['GRB_LICENSE_FILE'] = ''  # Prevent license file messages
gurobi_logger = logging.getLogger('gurobipy')
gurobi_logger.setLevel(logging.CRITICAL)
gurobi_logger.propagate = False

options = {
	"WLSACCESSID":"af4b8280-70cd-47bc-aeef-69ecf14ecd10",
	"WLSSECRET":"04da6102-8eb3-4e38-ba06-660ea8f87bf2",
	"LICENSEID":2669217
}

@contextlib.contextmanager
def suppress_gurobi_output():
    """Context manager to completely suppress Gurobi output"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

@print_decorator('clustering')
@timed_matrix_operation('full_clustering')
def clustering_full_matrix(
        input_matrix:np.ndarray, 
        regions :List[List[int]],
        steps : List[Tuple[List[int], List[int], List[int]]], 
        min_row_quality:int=5,
        min_col_quality:int = 3,
        error_rate : float = 0.025
        ) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Perform exhaustive iterative biclustering on a binary matrix to extract all significant patterns.
    
    This function systematically processes predefined regions of a binary matrix to identify 
    all possible separations between rows based on their column patterns. It applies binary 
    clustering iteratively until no more significant patterns can be found.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix with values (0, 1) where rows and columns represent 
        data points and features respectively. Values indicate feature presence (1), 
        absence (0).
    regions : List[List[int]]
        List of column index groups to process separately. Each region contains 
        column indices that should be analyzed together as a coherent unit.
    steps : List[Tuple[List[int], List[int], List[int]]]
        Pre-existing clustering results to preserve. Each tuple contains
        (row_indices_group1, row_indices_group2, column_indices).
    min_row_quality : int, optional
        Minimum number of rows required for a cluster to be considered valid.
        Default is 5.
    min_col_quality : int, optional
        Minimum number of columns required for a region to be processed.
        Regions with fewer columns are skipped. Default is 3.
    error_rate : float, optional
        Tolerance level for pattern detection, allowing for noise and imperfections
        in the binary patterns. Default is 0.025 (2.5%).
    
    Returns
    -------
    List[Tuple[List[int], List[int], List[int]]]
        Complete list of all valid clustering steps found. Each tuple contains:
        - [0] : List[int] - Row indices in first group (pattern match)
        - [1] : List[int] - Row indices in second group (pattern opposite)  
        - [2] : List[int] - Column indices where this separation is significant
        
        Only returns steps where both groups are non-empty and column count
        meets minimum quality threshold.
    
    Algorithm
    ---------
    1. **Initialization**: Start with existing clustering steps
    
    2. **Region Iteration**: Process each column region independently:
       - Skip regions with insufficient columns (< min_col_quality)
       - Initialize remaining columns for processing
       
    3. **Exhaustive Pattern Extraction**: For each region:
       - Apply binary clustering to find one significant row separation
       - Convert local column indices to global matrix coordinates
       - Save valid separations to results
       - Remove processed columns from remaining set
       - Continue until no significant patterns remain
    
    4. **Result Filtering**: Return only clustering steps that satisfy:
       - Both row groups contain at least one element
       - Column set meets minimum quality requirements
    
    See Also
    --------
    clustering_step : Single iteration of binary pattern detection
    find_quasi_biclique : Core optimization algorithm for dense pattern detection  
    pre_processing : Preprocessing step that identifies initial regions and steps
    post_processing : Final step that converts clustering steps to read groups
    
    """
    # Initialize result list with existing steps
    steps_result = steps.copy() if steps else []
    total_clustering_steps = 0
    total_ilp_calls = 0
    start_time = time.time()
    
    # Process each region if any regions are provided
    if len(regions) > 0:
        for idx, region in enumerate(regions):   
            # Initialize remaining columns for this region
            remain_cols = region
            status = True
            region_start_time = time.time()
            region_steps = 0
            region_ilp_calls = 0
            
            # Only process regions that meet minimum quality threshold
            if len(remain_cols) >= min_col_quality:
                logger.debug(f"Processing region {idx} with {len(remain_cols)} columns")  # GARDER CE LOG
                
                # Iteratively extract patterns until no more significant ones found
                while len(remain_cols) >= min_col_quality and status:
                    # Apply clustering to current remaining columns
                    sub_matrix = input_matrix[:, remain_cols]
                    reads1, reads0, cols, ilp_calls = clustering_step_with_stats(
                        sub_matrix, 
                        error_rate=error_rate,
                        min_row_quality=min_row_quality, 
                        min_col_quality=min_col_quality
                    )
                    
                    region_ilp_calls += ilp_calls
                    
                    # Convert local column indices back to global matrix coordinates
                    cols = [remain_cols[c] for c in cols]
                    
                    # Check if valid pattern was found
                    if len(cols) == 0:
                        status = False  # No more patterns, stop processing this region
                    else:
                        # Save valid clustering step
                        steps_result.append((reads1, reads0, cols))
                        region_steps += 1
                        logger.debug(f"Found clustering step: {len(reads1)} vs {len(reads0)} reads, {len(cols)} columns")  # GARDER CE LOG
                        # Remove processed columns from remaining set
                        remain_cols = [c for c in remain_cols if c not in cols]
            else:
                logger.debug(f"Skipping region {idx}: insufficient columns ({len(remain_cols)} < {min_col_quality})")  # GARDER CE LOG
            
            # AJOUTER les statistiques EN PLUS des logs existants
            region_time = time.time() - region_start_time
            stats = get_stats()
            if stats and stats.enabled:
                # Enregistrer dans benchmark_stats
                stats.record_benchmark_data(
                    stage='region_clustering',
                    matrix_shape=(input_matrix.shape[0], len(region)),
                    execution_time=region_time,
                    region_id=idx,
                    initial_cols=len(region),
                    patterns_found=region_steps,
                    ilp_calls_total=region_ilp_calls,
                    min_row_quality=min_row_quality,
                    min_col_quality=min_col_quality,
                    error_rate=error_rate
                )
                
                # AJOUTER: Enregistrer AUSSI dans clustering_stats pour chaque région
                stats.record_clustering_step(
                    matrix_shape=(input_matrix.shape[0], len(region)),
                    execution_time=region_time,
                    iterations=region_steps,
                    patterns_found=region_steps,
                    ilp_calls=region_ilp_calls,
                    positive_reads=0,  # Non applicable au niveau région
                    negative_reads=0,  # Non applicable au niveau région
                    final_cols=len(remain_cols),
                    region_id=idx,
                    initial_cols=len(region),
                    error_rate=error_rate,
                    min_row_quality=min_row_quality,
                    min_col_quality=min_col_quality
                )
                
            total_clustering_steps += region_steps
            total_ilp_calls += region_ilp_calls
    
    total_time = time.time() - start_time
    
    # AJOUTER les statistiques globales EN PLUS des logs existants
    stats = get_stats()
    if stats and stats.enabled:
        valid_steps_count = len([step for step in steps_result if len(step[0]) > 0 and len(step[1]) > 0])
        
        # Enregistrer dans benchmark_stats
        stats.record_benchmark_data(
            stage='full_clustering_summary',
            matrix_shape=input_matrix.shape,
            execution_time=total_time,
            total_regions=len(regions),
            total_clustering_steps=total_clustering_steps,
            total_ilp_calls=total_ilp_calls,
            valid_steps=valid_steps_count,
            avg_ilp_per_region=total_ilp_calls / max(1, len(regions)),
            avg_time_per_region=total_time / max(1, len(regions))
        )
        
        # AJOUTER: Enregistrer AUSSI un résumé dans clustering_stats
        stats.record_clustering_step(
            matrix_shape=input_matrix.shape,
            execution_time=total_time,
            iterations=total_clustering_steps,
            patterns_found=valid_steps_count,
            ilp_calls=total_ilp_calls,
            positive_reads=0,  # Résumé global
            negative_reads=0,  # Résumé global
            final_cols=0,      # Résumé global
            region_id=-1,      # -1 pour indiquer un résumé global
            initial_cols=input_matrix.shape[1],
            error_rate=error_rate,
            min_row_quality=min_row_quality,
            min_col_quality=min_col_quality
        )
    
    # Log clustering results for debugging (GARDER CES LOGS)
    logger.debug(f"Clustering completed: {len(steps_result)} total steps found")
    
    # Filter and return only valid steps with non-empty groups and sufficient columns
    valid_steps = [step for step in steps_result if len(step[0]) > 0 and len(step[1]) > 0 and len(step[2]) >= min_col_quality]
    logger.info(f"Found {len(valid_steps)} valid clustering steps")  # GARDER CE LOG
    return valid_steps

def clustering_step_with_stats(input_matrix: np.ndarray,
                             error_rate: float = 0.025,
                             min_row_quality: int = 5,
                             min_col_quality: int = 3,
                             ) -> Tuple[List[int], List[int], List[int], int]:
    """Version de clustering_step qui retourne aussi le nombre d'appels ILP."""
    ilp_calls = 0
    
    # Create binary matrices for pattern detection
    matrix1 = input_matrix.copy()
    matrix1[matrix1 == -1] = 0  # Convert missing values to 0 for positive patterns
    
    matrix0 = input_matrix.copy()
    matrix0[matrix0 == -1] = 1
    matrix0 = (matrix0 - 1) * -1  # Invert matrix for negative pattern detection
    
    logger.debug(f"Starting clustering step on {matrix1.shape[0]} rows, {matrix1.shape[1]} columns")  # GARDER CE LOG
    
    # Initialize tracking variables for iterative clustering
    remain_rows = range(matrix1.shape[0])  # All rows initially available
    current_cols = range(matrix1.shape[1])  # All columns initially available
    clustering_1 = True  # Alternate between positive (True) and negative (False) patterns
    status = True  # Continue while valid patterns are found
    rw1, rw0 = [], []  # Accumulate rows for positive and negative groups
    
    # Iteratively extract patterns until insufficient data remains
    iteration = 0
    start_time = time.time()
    while len(remain_rows) >= min_row_quality and len(current_cols) >= min_col_quality and status:
        iteration += 1
        logger.debug(f"Clustering iteration {iteration}: {len(remain_rows)} rows, {len(current_cols)} columns, pattern={'positive' if clustering_1 else 'negative'}")
        
        # Apply quasi-biclique detection on appropriate matrix
        if clustering_1:
            # Search for positive patterns (dense regions of 1s)
            rw, cl, status = find_quasi_biclique(matrix1[remain_rows][:, current_cols], error_rate)
        else:
            # Search for negative patterns (dense regions of 0s in original)
            rw, cl, status = find_quasi_biclique(matrix0[remain_rows][:, current_cols], error_rate)
        
        if status:
            ilp_calls += 1
             
        # Convert local indices back to global matrix coordinates
        rw = [remain_rows[r] for r in rw]  # Map row indices to original matrix
        cl = [current_cols[c] for c in cl]  # Map column indices to original matrix
        
        current_cols = cl  # Update working column set to detected significant columns
        
        # Accumulate rows into appropriate pattern group if valid pattern found
        if status and len(cl) > 0:
            if clustering_1:
                rw1.extend(rw)  # Add to positive pattern group
                logger.debug(f"Added {len(rw)} rows to positive group")
            else:
                rw0.extend(rw)  # Add to negative pattern group
                logger.debug(f"Added {len(rw)} rows to negative group")
        else:
            logger.debug(f"No valid pattern found in iteration {iteration}")
                
        # Remove processed rows from remaining set for next iteration
        remain_rows = [r for r in remain_rows if r not in rw]
        # Alternate pattern detection type for next iteration
        clustering_1 = not clustering_1
    
    execution_time = time.time() - start_time
    
    # AJOUTER les statistiques EN PLUS des logs existants
    stats = get_stats()
    if stats and stats.enabled:
        patterns_found = 1 if (len(rw1) > 0 or len(rw0) > 0) else 0
        
        # Enregistrer dans benchmark_stats
        stats.record_benchmark_data(
            stage='clustering_step_detailed',
            matrix_shape=input_matrix.shape,
            execution_time=execution_time,
            iterations=iteration,
            ilp_calls=ilp_calls,
            positive_reads=len(rw1),
            negative_reads=len(rw0),
            final_cols=len(current_cols),
            patterns_found=patterns_found,
            error_rate=error_rate,
            min_row_quality=min_row_quality,
            min_col_quality=min_col_quality,
            avg_time_per_iteration=execution_time / max(1, iteration),
            avg_time_per_ilp=execution_time / max(1, ilp_calls)
        )
        
        # AJOUTER: Enregistrer AUSSI dans clustering_stats
        stats.record_clustering_step(
            matrix_shape=input_matrix.shape,
            execution_time=execution_time,
            iterations=iteration,
            patterns_found=patterns_found,
            ilp_calls=ilp_calls,
            positive_reads=len(rw1),
            negative_reads=len(rw0),
            final_cols=len(current_cols),
            # Ajouter des champs optionnels pour cohérence
            region_id=0,
            initial_cols=input_matrix.shape[1],
            error_rate=error_rate,
            min_row_quality=min_row_quality,
            min_col_quality=min_col_quality
        )
        
    # Log final clustering statistics (GARDER CE LOG)
    logger.debug(f"Clustering step completed: {len(rw1)} positive reads, {len(rw0)} negative reads, {len(current_cols)} columns")
    return rw1, rw0, current_cols, ilp_calls

@print_decorator('clustering')
@timed_ilp_call('complete_optimization')
def find_quasi_biclique(
    input_matrix: np.ndarray,
    error_rate: float = 0.025
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using integer linear programming optimization.
    
    This function identifies the largest dense sub-matrix where most elements are 1s,
    with tolerance for noise defined by the error rate. It uses a three-phase approach:
    seeding with a high-density region, then iteratively extending by rows and columns
    to maximize the objective while maintaining density constraints.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix with values (0, 1) where rows and columns represent 
        data points and features respectively. Values indicate feature presence (1) 
        or absence (0).
    error_rate : float, optional
        Maximum fraction of 0s allowed in the quasi-biclique, defining noise tolerance.
        A value of 0.025 means up to 2.5% of cells can be 0s. Default is 0.025.
    
    Returns
    -------
    Tuple[List[int], List[int], bool]
        Triple containing the quasi-biclique results:
        - [0] : List[int] - Row indices included in the quasi-biclique
        - [1] : List[int] - Column indices included in the quasi-biclique  
        - [2] : bool - Success status (True if valid solution found, False otherwise)
        
        Empty lists are returned when no significant quasi-biclique is found or
        optimization fails.
    
    Algorithm
    ---------
    1. **Preprocessing and Sorting**:
       - Sort rows and columns by decreasing number of 1s
       - Handle edge cases (empty matrix)
       
    2. **Seed Region Selection**:
       - Start with top 1/3 of rows and columns by density
       - Search for largest sub-region with >99% density of 1s
       - Use this as initial seed for optimization
       
    3. **Phase 1 - Seeding Optimization**:
       - Create ILP model with binary variables for rows, columns, and cells
       - Maximize sum of selected 1s subject to density constraint
       - Ensure cells are selected only if both row and column are selected
       
    4. **Phase 2 - Row Extension**:
       - Identify remaining rows with >50% compatibility with selected columns
       - Add compatible rows to optimization model
       - Re-optimize to find improved quasi-biclique
       
    5. **Phase 3 - Column Extension**:
       - Identify remaining columns with >90% compatibility with selected rows
       - Add compatible columns to optimization model
       - Perform final optimization
       
    6. **Result Extraction**:
       - Extract selected rows and columns from optimal solution
       - Return success status based on optimization outcome
    
    Mathematical Formulation
    ------------------------
    **Variables**:
    - `r_i ∈ {0,1}` : Binary indicator for row i selection
    - `c_j ∈ {0,1}` : Binary indicator for column j selection  
    - `x_ij ∈ {0,1}` : Binary indicator for cell (i,j) selection
    
    **Objective**:
    - Maximize: `Σ_{i,j} M[i,j] × x_ij` (sum of 1s in selected cells)
    
    **Constraints**:
    - `x_ij ≤ r_i` : Cell selected only if row selected
    - `x_ij ≤ c_j` : Cell selected only if column selected
    - `x_ij ≥ r_i + c_j - 1` : Cell selected if both row and column selected
    - `Σ_{i,j} x_ij × (1-M[i,j]) ≤ error_rate × Σ_{i,j} x_ij` : Density constraint
    
    Raises
    ------
    gurobipy.GurobiError
        If Gurobi optimization fails due to licensing or solver issues
    numpy.linalg.LinAlgError  
        If matrix operations fail due to invalid dimensions or data
    MemoryError
        If insufficient memory for optimization model creation

    See Also
    --------
    setup_gurobi_model : Model configuration and parameter setup
    clustering_step : Uses quasi-biclique detection for alternating patterns
    clustering_full_matrix : Applies quasi-biclique detection across regions
    
    """
    # Copy input matrix to avoid modifying original
    X_problem = input_matrix.copy()

    # Sort rows and columns by decreasing number of 1s (highest density first)
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]

    m = len(rows_sorted)
    n = len(cols_sorted)

    # Handle edge case: empty matrix
    if m == 0 or n == 0:
        logger.debug("Empty matrix provided to quasi-biclique detection")
        return [], [], False

    logger.debug(f"Starting quasi-biclique detection on {m}x{n} matrix")  # GARDER CE LOG
    
    optimization_phases = []

    # Phase 1: Seed region selection - find dense area to start optimization
    seed_rows = m // 3
    seed_cols = n // 3

    # Adjust step size based on matrix width
    if n > 50:
        step_n = 10
    else:
        step_n = 2

    # Search for the largest sub-region with >99% density of 1s
    for x in range(m // 3, m, 10):
        for y in range(seed_cols, n, step_n):
            nb_of_ones = 0
            for row in rows_sorted[:x]:
                for col in cols_sorted[:y]:
                    nb_of_ones += X_problem[row, col]
            ratio_ones = nb_of_ones / (x * y)
            if ratio_ones > 0.99:
                seed_rows = x
                seed_cols = y

    logger.debug(f"Using seed region: {seed_rows} rows x {seed_cols} columns")

    # Initialize Gurobi optimization environment with complete output suppression
    try:
        with suppress_gurobi_output():
            env = grb.Env(params=options)
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.setParam('LogFile', "")
            model = grb.Model('max_model', env=env)          
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0
            model.Params.LogFile = ""
            model.Params.MIPGAP = 0.05
            model.Params.TimeLimit = 20
    except Exception as e:
        logger.warning(f"Failed to initialize Gurobi environment: {e}")
        return [], [], False

    # Phase 2: Create initial optimization model with seed region
    # Decision variables for rows, columns, and cells in seed region
    lpRows = model.addVars(rows_sorted[:seed_rows], lb=0, ub=1, vtype=grb.GRB.INTEGER, name='rw')
    lpCols = model.addVars(cols_sorted[:seed_cols], lb=0, ub=1, vtype=grb.GRB.INTEGER, name='cl')
    lpCells = model.addVars([(r, c) for r in rows_sorted[:seed_rows] for c in cols_sorted[:seed_cols]], 
                        lb=0, ub=1, vtype=grb.GRB.INTEGER, name='ce')

    # Objective: maximize sum of 1s in selected cells
    model.setObjective(grb.quicksum([(X_problem[c[0]][c[1]]) * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)

    # Constraints: cells can only be selected if both row and column are selected
    for cell in lpCells:
        model.addConstr(lpCells[cell] <= lpRows[cell[0]], name=f'row_{cell}')      # Cell needs row
        model.addConstr(lpCells[cell] <= lpCols[cell[1]], name=f'col_{cell}')      # Cell needs column
        model.addConstr(lpCells[cell] >= lpRows[cell[0]] + lpCols[cell[1]] - 1, name=f'both_{cell}')  # Force selection if both selected

    # Density constraint: limit fraction of 0s in selected region
    model.addConstr(error_rate * lpCells.sum() >= grb.quicksum(
        [lpCells[coord] * (1 - X_problem[coord[0]][coord[1]]) for coord in lpCells]), 'err_thrshld')

    total_solve_time = 0
    phases_executed = 0
    
    # Solve initial seeding optimization
    seed_start = time.time()
    with suppress_gurobi_output():
        model.optimize()
    seed_time = time.time() - seed_start
    total_solve_time += seed_time
    phases_executed += 1
    
    # AJOUTER les statistiques EN PLUS des logs existants
    stats = get_stats()
    if stats and stats.enabled:
        status_str = 'optimal' if model.Status == grb.GRB.OPTIMAL else f'status_{model.Status}'
        obj_val = model.ObjVal if hasattr(model, 'ObjVal') else 0
        gap = model.MIPGap if hasattr(model, 'MIPGap') else 0
        
        stats.record_benchmark_data(
            stage='ilp_seeding',
            matrix_shape=(seed_rows, seed_cols),
            execution_time=seed_time,
            solver_status=status_str,
            objective_value=obj_val,
            variables=model.NumVars,
            constraints=model.NumConstrs,
            gap=gap,
            error_rate=error_rate,
            # Ajouter des champs optionnels avec des valeurs par défaut
            rows_added=0,
            cols_added=0,
            phases_executed=1
        )
        
        stats.record_ilp_call(
            matrix_shape=(seed_rows, seed_cols),
            execution_time=seed_time,
            solver_status=status_str,
            objective_value=obj_val,
            phase='seeding',
            variables=model.NumVars,
            constraints=model.NumConstrs,
            # Ajouter des champs optionnels avec des valeurs par défaut
            rows_added=0,
            cols_added=0,
            gap=gap,
            error_rate=error_rate
        )

    # Extract seed solution results
    rw = []
    cl = []
    for var in model.getVars():
        if var.VarName.startswith('rw') and var.X > 0.5:
            rw.append(int(var.VarName.split('[')[1].split(']')[0]))
        elif var.VarName.startswith('cl') and var.X > 0.5:
            cl.append(int(var.VarName.split('[')[1].split(']')[0]))

    logger.debug(f"Initial seed solution: {len(rw)} rows, {len(cl)} columns")  # GARDER CE LOG

    # Phase 3: Row extension - add compatible rows
    rem_rows = [r for r in rows_sorted if r not in lpRows.keys()]
    if len(cl) > 0:
        # Find rows with >50% compatibility with selected columns
        rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
        potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
    else:
        potential_rows = []

    # Add compatible rows to model and re-optimize
    if potential_rows:
        logger.debug(f"Extending with {len(potential_rows)} compatible rows")
        lpRows.update(model.addVars(potential_rows, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='rw'))
        new_cells = model.addVars([(r, c) for r in potential_rows for c in cl], 
                                lb=0, ub=1, vtype=grb.GRB.INTEGER, name='ce')
        lpCells.update(new_cells)
        
        # Update objective with new cells
        model.setObjective(grb.quicksum([(X_problem[c[0]][c[1]]) * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)
        
        # Add constraints for new cells
        for cell in new_cells:
            model.addConstr(lpCells[cell] <= lpRows[cell[0]], name=f'row_{cell}')
            model.addConstr(lpCells[cell] <= lpCols[cell[1]], name=f'col_{cell}')
            model.addConstr(lpCells[cell] >= lpRows[cell[0]] + lpCols[cell[1]] - 1, name=f'both_{cell}')
        
        # Update density constraint
        model.addConstr(error_rate * lpCells.sum() >= grb.quicksum(
            [lpCells[coord] * (1 - X_problem[coord[0]][coord[1]]) for coord in lpCells]), 'err_thrshld_row')
        
        row_start = time.time()
        with suppress_gurobi_output():
            model.optimize()
        row_time = time.time() - row_start
        total_solve_time += row_time
        phases_executed += 1
        
        # Enregistrer les statistiques de l'extension des lignes
        if stats and stats.enabled:
            status_str = 'optimal' if model.Status == grb.GRB.OPTIMAL else f'status_{model.Status}'
            obj_val = model.ObjVal if hasattr(model, 'ObjVal') else 0
            gap = model.MIPGap if hasattr(model, 'MIPGap') else 0
            
            stats.record_benchmark_data(
                stage='ilp_row_extension',
                matrix_shape=(len(rw) + len(potential_rows), len(cl)),
                execution_time=row_time,
                solver_status=status_str,
                objective_value=obj_val,
                variables=model.NumVars,
                constraints=model.NumConstrs,
                rows_added=len(potential_rows),
                gap=gap,
                error_rate=error_rate,
                # Ajouter des champs optionnels avec des valeurs par défaut
                cols_added=0,
                phases_executed=2
            )
            
            stats.record_ilp_call(
                matrix_shape=(len(rw) + len(potential_rows), len(cl)),
                execution_time=row_time,
                solver_status=status_str,
                objective_value=obj_val,
                phase='row_extension',
                rows_added=len(potential_rows),
                variables=model.NumVars,
                constraints=model.NumConstrs,
                gap=gap,
                error_rate=error_rate,
                # Ajouter des champs optionnels avec des valeurs par défaut
                cols_added=0
            )

    # Extract results after row extension
    rw = []
    cl = []
    for var in model.getVars():
        if var.VarName.startswith('rw') and var.X > 0.5:
            rw.append(int(var.VarName.split('[')[1].split(']')[0]))
        elif var.VarName.startswith('cl') and var.X > 0.5:
            cl.append(int(var.VarName.split('[')[1].split(']')[0]))
                
    # Phase 4: Column extension - add compatible columns
    rem_cols = [c for c in cols_sorted if c not in lpCols.keys()]
    if len(rw) > 0:
        # Find columns with >90% compatibility with selected rows
        rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
        potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
    else:
        potential_cols = []

    # Add compatible columns to model and perform final optimization
    if potential_cols:
        logger.debug(f"Extending with {len(potential_cols)} compatible columns")
        lpCols.update(model.addVars(potential_cols, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='cl'))
        new_cells = model.addVars([(r, c) for r in rw for c in potential_cols], 
                                lb=0, ub=1, vtype=grb.GRB.INTEGER, name='ce')
        lpCells.update(new_cells)
        
        # Update objective with all cells
        model.setObjective(grb.quicksum([(X_problem[c[0]][c[1]]) * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)
        
        # Add constraints for new cells
        for cell in new_cells:
            model.addConstr(lpCells[cell] <= lpRows[cell[0]], name=f'row_{cell}')
            model.addConstr(lpCells[cell] <= lpCols[cell[1]], name=f'col_{cell}')
            model.addConstr(lpCells[cell] >= lpRows[cell[0]] + lpCols[cell[1]] - 1, name=f'both_{cell}')
        
        # Final density constraint
        model.addConstr(error_rate * lpCells.sum() >= grb.quicksum(
            [lpCells[coord] * (1 - X_problem[coord[0]][coord[1]]) for coord in lpCells]), 'err_thrshld_col')
        
        col_start = time.time()
        with suppress_gurobi_output():
            model.optimize()
        col_time = time.time() - col_start
        total_solve_time += col_time
        phases_executed += 1
        
        # Enregistrer les statistiques de l'extension des colonnes
        if stats and stats.enabled:
            status_str = 'optimal' if model.Status == grb.GRB.OPTIMAL else f'status_{model.Status}'
            obj_val = model.ObjVal if hasattr(model, 'ObjVal') else 0
            gap = model.MIPGap if hasattr(model, 'MIPGap') else 0
            
            stats.record_benchmark_data(
                stage='ilp_column_extension',
                matrix_shape=(len(rw), len(cl) + len(potential_cols)),
                execution_time=col_time,
                solver_status=status_str,
                objective_value=obj_val,
                variables=model.NumVars,
                constraints=model.NumConstrs,
                cols_added=len(potential_cols),
                gap=gap,
                error_rate=error_rate,
                # Ajouter des champs optionnels avec des valeurs par défaut
                rows_added=0,
                phases_executed=3
            )
            
            stats.record_ilp_call(
                matrix_shape=(len(rw), len(cl) + len(potential_cols)),
                execution_time=col_time,
                solver_status=status_str,
                objective_value=obj_val,
                phase='column_extension',
                cols_added=len(potential_cols),
                variables=model.NumVars,
                constraints=model.NumConstrs,
                gap=gap,
                error_rate=error_rate,
                # Ajouter des champs optionnels avec des valeurs par défaut
                rows_added=0
            )
    # Extract final results after column extension
    rw = []
    cl = []
    for var in model.getVars():
        if var.VarName.startswith('rw') and var.X > 0.5:
            rw.append(int(var.VarName.split('[')[1].split(']')[0]))
        elif var.VarName.startswith('cl') and var.X > 0.5:
            cl.append(int(var.VarName.split('[')[1].split(']')[0]))
    logger.debug(f"Final solution after column extension: {len(rw)} rows, {len(cl)} columns")  # GARDER CE LOG
    # AJOUTER un résumé final EN PLUS des logs existants
    if stats and stats.enabled:
        final_status = 'optimal' if model.Status == grb.GRB.OPTIMAL else f'status_{model.Status}'
        final_obj_val = model.ObjVal if hasattr(model, 'ObjVal') else 0
        final_gap = model.MIPGap if hasattr(model, 'MIPGap') else 0
        
        stats.record_benchmark_data(
            stage='ilp_complete_summary',
            matrix_shape=input_matrix.shape,
            execution_time=total_solve_time,
            solver_status=final_status,
            objective_value=final_obj_val,
            phases_executed=phases_executed,
            final_rows=len(rw),
            final_cols=len(cl),
            solution_density=len(rw) * len(cl) / max(1, input_matrix.size),
            avg_time_per_phase=total_solve_time / max(1, phases_executed),
            # Ajouter tous les champs potentiels
            variables=model.NumVars,
            constraints=model.NumConstrs,
            gap=final_gap,
            error_rate=error_rate,
            rows_added=0,
            cols_added=0
        )

    # Check optimization status and return appropriate results
    status = model.Status

    if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
        logger.warning("Optimization problem is infeasible or unbounded")
        return [], [], False
    elif status == grb.GRB.TIME_LIMIT:
        logger.warning("Optimization hit time limit, returning partial solution")
        return rw, cl, True  # Return solution even with time limit
    elif status != grb.GRB.OPTIMAL:
        logger.warning(f"Optimization terminated with non-optimal status: {status}")
        return [], [], False
    
    logger.debug(f"Quasi-biclique found: {len(rw)} rows, {len(cl)} columns")  # GARDER CE LOG        
    return rw, cl, True
