from typing import List, Tuple
import numpy as np

import gurobipy as grb

options = {
	"WLSACCESSID":"af4b8280-70cd-47bc-aeef-69ecf14ecd10",
	"WLSSECRET":"04da6102-8eb3-4e38-ba06-660ea8f87bf2",
	"LICENSEID":2669217
}

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
    # Initialize results with pre-existing clustering steps
    results = steps.copy()
    
    # Handle edge case: empty matrix
    if len(input_matrix) == 0 or len(input_matrix[0]) == 0:
        return results
    
    # Process each column region independently
    for region in regions:
        # Initialize remaining columns to process in this region
        remaining_columns = region.copy()
        status = True
        
        # Continue clustering until no more patterns found or insufficient columns remain
        while status and len(remaining_columns) >= min_col_quality:
            # Perform single clustering step on current column subset
            read1, read0, columns = clustering_step(
                input_matrix[:, remaining_columns],  # Extract submatrix for current columns
                error_rate=error_rate,
                min_row_quality=min_row_quality,
                min_col_quality=min_col_quality
            )

            # Check if clustering found significant patterns
            if len(columns) == 0:
                # No more patterns found, stop processing this region
                status = False
            else:
                # Store valid clustering result
                results.append((
                    read1,    # Rows with positive pattern
                    read0,    # Rows with negative pattern  
                    columns   # Columns where separation is significant
                ))
                
                # Remove processed columns from remaining set
                remaining_columns = [
                    col for col in remaining_columns if col not in columns
                ]

    # Filter results to return only valid clusters with non-empty groups
    return (rep for rep in results if len(rep[0]) > 0 and len(rep[1]) > 0 and len(rep[2]) >= min_col_quality)

def clustering_step(input_matrix: np.ndarray,
                    error_rate: float = 0.025,
                    min_row_quality: int = 5,
                    min_col_quality: int = 3,
                    ) -> Tuple[List[int], List[int], List[int]]:
    """
    Perform a single binary clustering step on a matrix to identify one significant row separation.
    
    This function applies alternating quasi-biclique detection to find groups of rows with 
    similar patterns across columns. It processes both positive (1s) and negative (0s) patterns 
    iteratively until a stable separation is found or no significant patterns remain.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix with values (0, 1) where rows represent data points 
        and columns represent features. Values indicate feature presence (1), 
        absence (0)
    error_rate : float, optional
        Tolerance level for quasi-biclique detection, allowing for noise and 
        imperfections in binary patterns. Default is 0.025 (2.5%).
    min_row_quality : int, optional
        Minimum number of rows required to continue processing. Algorithm stops 
        when fewer rows remain unprocessed. Default is 5.
    min_col_quality : int, optional
        Minimum number of columns required to continue processing. Algorithm stops 
        when fewer significant columns remain. Default is 3.
    
    Returns
    -------
    Tuple[List[int], List[int], List[int]]
        Triple containing the results of binary clustering:
        - [0] : List[int] - Row indices with positive pattern (predominantly 1s)
        - [1] : List[int] - Row indices with negative pattern (predominantly 0s)  
        - [2] : List[int] - Column indices where separation is most significant
        
        Empty lists are returned for categories where no significant patterns are found.
    
    Algorithm
    ---------
    1. **Matrix Preparation**:
       - Create negative matrix: invert values to detect 0-patterns
       
    2. **Alternating Pattern Detection**:
       - Start with all rows and columns available for processing
       - Alternate between searching for 1-patterns and 0-patterns
       - Apply quasi-biclique optimization to find dense sub-matrices
       
    3. **Noise Filtering**:
       - For each detected pattern, calculate column homogeneity
       - Retain only columns with homogeneity > 5 × error_rate
       - Remove extremely noisy or inconsistent columns
       
    4. **Iterative Refinement**:
       - Accumulate rows into positive or negative groups based on current pattern
       - Remove processed rows from remaining set
       - Continue until insufficient rows or columns remain
       
    5. **Termination Conditions**:
       - Stop when fewer than min_row_quality rows remain
       - Stop when fewer than min_col_quality columns remain  
       - Stop when no significant patterns are detected
    
    See Also
    --------
    clustering_full_matrix : Applies quasi-biclique detection across regions
    find_quasi_biclique : Core optimization algorithm for dense pattern detection
       
    """
    # Initialize matrices: original and inverted for detecting 0-patterns
    matrix1 = input_matrix.copy()  # Matrix for detecting 1-patterns (positive patterns)
    matrix0 = np.where(input_matrix == -1, 1, 1 - input_matrix)  # Inverted matrix for detecting 0-patterns (negative patterns)

    # Initialize row and column tracking variables
    remain_rows = range(matrix1.shape[0])  # Rows still available for processing
    current_cols = range(matrix1.shape[1])  # Columns currently being analyzed
    clustering_1 = True  # Flag to alternate between 1-pattern and 0-pattern detection
    status = True  # Flag to continue processing while patterns are found
    rw1, rw0 = [], []  # Accumulate rows with positive and negative patterns respectively
    
    # Main clustering loop: continue until insufficient rows/columns or no patterns found
    while len(remain_rows) >= min_row_quality and len(current_cols) >= min_col_quality and status:
        # Alternate between searching for 1-patterns and 0-patterns
        if clustering_1:
            # Search for quasi-biclique in original matrix (1-patterns)
            rw, cl, status = find_quasi_biclique(matrix1[remain_rows][:, current_cols], error_rate)
        else:
            # Search for quasi-biclique in inverted matrix (0-patterns)
            rw, cl, status = find_quasi_biclique(matrix0[remain_rows][:, current_cols], error_rate)
             
        # Convert local indices back to global matrix coordinates
        rw = [remain_rows[r] for r in rw]  # Map row indices to original matrix
        cl = [current_cols[c] for c in cl]  # Map column indices to original matrix
        
        # Filter out extremely noisy columns based on pattern homogeneity
        if len(cl) > 0:
            if len(rw) == 0:
                # No rows found, clear columns to stop processing
                current_cols = []
            else:
                # Calculate column homogeneity for quality filtering
                if clustering_1:
                    # For 1-patterns: calculate density of 1s in selected rows/columns
                    col_homogeneity = matrix1[rw][:, cl].sum(axis=0) / len(rw)
                else:
                    # For 0-patterns: calculate density in inverted matrix
                    col_homogeneity = matrix0[rw][:, cl].sum(axis=0) / len(rw)
                
                # Keep only columns with homogeneity above threshold (5× error rate)
                current_cols = [c for idx, c in enumerate(cl) if col_homogeneity[idx] > 5 * error_rate]
        else:
            # No significant columns found, stop processing
            status = False   
            
        # Accumulate rows into appropriate pattern groups
        if status:
            if clustering_1:
                # Add rows showing positive patterns (predominantly 1s)
                rw1 = rw1 + [r for r in rw]
            else:
                # Add rows showing negative patterns (predominantly 0s)
                rw0 = rw0 + [r for r in rw]  
                
        # Remove processed rows from remaining set for next iteration
        remain_rows = [r for r in remain_rows if r not in rw]
        
        # Switch pattern detection mode for next iteration
        clustering_1 = not clustering_1
    
    # Return final row groups and significant columns
    return rw1, rw0, current_cols

        



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
    matrix = input_matrix.copy()
    
    # Sort columns and rows by sum in descending order (highest density first)
    cols_sorted = np.argsort(matrix.sum(axis=0))[::-1]
    rows_sorted = np.argsort(matrix.sum(axis=1))[::-1]

    m = len(rows_sorted)
    n = len(cols_sorted)
    
    # Handle edge case: empty matrix
    if m == 0 or n == 0:
        return [], [], False

    # Phase 1: Seed Region Selection
    # Start with top 1/3 of rows and columns by density
    seed_rows = m // 3
    seed_cols = n // 3
    step_n = (10 if n > 50 else 2)  # Adaptive step size based on matrix size

    # Find the largest sub-region with >99% density of 1s as seed
    for i in range(seed_rows, m, 10):
        for j in range(seed_cols, n, step_n):
            nb_of_ones = 0
            # Count 1s in current sub-region
            for row in rows_sorted[:i]:
                for col in cols_sorted[:j]:
                    nb_of_ones += matrix[row, col]
            
            ratio_ones = nb_of_ones / (i * j)
            if ratio_ones > 0.99:
                # Found a seed region with sufficient density
                seed_rows = i
                seed_cols = j

    # Initialize Gurobi optimization environment
    try:
        model = setup_gurobi_model(error_rate=error_rate)
    except Exception as e:
        print(f"Warning: Failed to setup Gurobi model: {e}")
        return [], [], False

    # Phase 1: Seeding Optimization
    # Create binary variables for initial seed region
    lpRows = model.addVars(rows_sorted[:seed_rows], lb=0, ub=1, vtype=grb.GRB.BINARY, name='rw')
    lpCols = model.addVars(cols_sorted[:seed_cols], lb=0, ub=1, vtype=grb.GRB.BINARY, name='cl')
    lpCells = model.addVars([(r, c) for r in rows_sorted[:seed_rows] for c in cols_sorted[:seed_cols]], 
                           lb=0, ub=1, vtype=grb.GRB.BINARY, name='ce')

    # Objective: Maximize sum of selected 1s
    model.setObjective(grb.quicksum([matrix[c[0]][c[1]] * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)

    # Constraints: Cell selected only if both row and column are selected
    for cell in lpCells:
        model.addConstr(lpCells[cell] <= lpRows[cell[0]], f'{cell}_row_constraint')
        model.addConstr(lpCells[cell] <= lpCols[cell[1]], f'{cell}_col_constraint')
        model.addConstr(lpCells[cell] >= lpRows[cell[0]] + lpCols[cell[1]] - 1, f'{cell}_both_constraint')

    # Density constraint: fraction of 0s ≤ error_rate
    model.addConstr(grb.quicksum([lpCells[coord] * (1 - matrix[coord[0]][coord[1]]) for coord in lpCells]) 
                   <= error_rate * lpCells.sum(), 'density_constraint')

    # Solve initial optimization
    model.optimize()

    # Extract selected rows and columns from solution
    selected_rows = [r for r in lpRows.keys() if lpRows[r].X > 0.5]
    selected_cols = [c for c in lpCols.keys() if lpCols[c].X > 0.5]

    # Phase 2: Row Extension
    # Find remaining rows with >50% compatibility with selected columns
    rem_rows = [r for r in rows_sorted if r not in lpRows.keys()]
    if len(selected_cols) > 0:  # Only if we have selected columns
        rem_rows_sum = matrix[rem_rows][:, selected_cols].sum(axis=1)
        potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(selected_cols)]
    else:
        potential_rows = []

    # Add compatible rows to optimization
    if potential_rows:
        lpRows.update(model.addVars(potential_rows, lb=0, ub=1, vtype=grb.GRB.BINARY, name='rw'))
        new_cells = model.addVars([(r, c) for r in potential_rows for c in selected_cols], 
                                 lb=0, ub=1, vtype=grb.GRB.BINARY, name='ce')
        lpCells.update(new_cells)

        # Update objective function
        model.setObjective(grb.quicksum([matrix[c[0]][c[1]] * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)

        # Add constraints for new cells
        for cell in new_cells:
            model.addConstr(lpCells[cell] <= lpRows[cell[0]], f'{cell}_row_constraint')
            model.addConstr(lpCells[cell] <= lpCols[cell[1]], f'{cell}_col_constraint')
            model.addConstr(lpCells[cell] >= lpRows[cell[0]] + lpCols[cell[1]] - 1, f'{cell}_both_constraint')

        # Update density constraint
        model.addConstr(grb.quicksum([lpCells[coord] * (1 - matrix[coord[0]][coord[1]]) for coord in lpCells]) 
                       <= error_rate * lpCells.sum(), 'density_constraint_phase2')

        # Re-optimize with extended rows
        model.optimize()

    # Extract updated selected rows and columns
    selected_rows = [r for r in lpRows.keys() if lpRows[r].X > 0.5]
    selected_cols = [c for c in lpCols.keys() if lpCols[c].X > 0.5]

    # Phase 3: Column Extension
    # Find remaining columns with >90% compatibility with selected rows
    rem_cols = [c for c in cols_sorted if c not in lpCols.keys()]
    if len(selected_rows) > 0:  # Only if we have selected rows
        rem_cols_sum = matrix[selected_rows][:, rem_cols].sum(axis=0)
        potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(selected_rows)]
    else:
        potential_cols = []

    # Add compatible columns to optimization
    if potential_cols:
        lpCols.update(model.addVars(potential_cols, lb=0, ub=1, vtype=grb.GRB.BINARY, name='cl'))
        new_cells = model.addVars([(r, c) for r in selected_rows for c in potential_cols], 
                                 lb=0, ub=1, vtype=grb.GRB.BINARY, name='ce')
        lpCells.update(new_cells)

        # Update objective function
        model.setObjective(grb.quicksum([matrix[c[0]][c[1]] * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)

        # Add constraints for new cells
        for cell in new_cells:
            model.addConstr(lpCells[cell] <= lpRows[cell[0]], f'{cell}_row_constraint')
            model.addConstr(lpCells[cell] <= lpCols[cell[1]], f'{cell}_col_constraint')
            model.addConstr(lpCells[cell] >= lpRows[cell[0]] + lpCols[cell[1]] - 1, f'{cell}_both_constraint')

        # Update density constraint
        model.addConstr(grb.quicksum([lpCells[coord] * (1 - matrix[coord[0]][coord[1]]) for coord in lpCells]) 
                       <= error_rate * lpCells.sum(), 'density_constraint_phase3')

        # Final optimization
        model.optimize()

    # Extract final results
    final_rows = [r for r in lpRows.keys() if lpRows[r].X > 0.5]
    final_cols = [c for c in lpCols.keys() if lpCols[c].X > 0.5]

    # Check optimization status and return appropriate result
    status = model.Status
    
    if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
        return [], [], False  # Optimization failed
    elif status == grb.GRB.TIME_LIMIT:
        return final_rows, final_cols, True  # Time limit reached but solution available
    elif status != grb.GRB.OPTIMAL:
        return [], [], False  # Other optimization issues
    
    # Successful optimization
    return final_rows, final_cols, True


def setup_gurobi_model(error_rate: float = 0.025, time_limit: int = 20, mip_gap: float = 0.05) -> grb.Model:
    """
    Setup Gurobi optimization model with standard parameters for quasi-biclique detection.
    
    This function creates and configures a Gurobi optimization environment with 
    predefined parameters optimized for quasi-biclique detection problems. It handles
    licensing, logging, and solver parameter configuration.
    
    Parameters
    ----------
    error_rate : float, optional
        Error tolerance for quasi-biclique detection. This parameter is stored
        for reference but doesn't directly affect model setup. Default is 0.025.
    time_limit : int, optional
        Maximum time in seconds allowed for optimization. Default is 20 seconds.
    mip_gap : float, optional
        Mixed Integer Programming gap tolerance. Optimization stops when gap
        between best solution and best bound is below this threshold. Default is 0.05 (5%).
    
    Returns
    -------
    grb.Model
        Configured Gurobi optimization model ready for variable and constraint addition.
        The model has the following pre-configured parameters:
        - OutputFlag = 0 (suppress console output)
        - TimeLimit = time_limit
        - MIPGAP = mip_gap

    Raises
    ------
    gurobipy.GurobiError
        If Gurobi license is invalid or environment setup fails

    See Also
    --------
    find_quasi_biclique : Main function using this model setup
    clustering_step : Uses quasi-biclique detection with these models
    """
    try:
        # Create Gurobi environment with license configuration
        env = grb.Env(params=options, logToConsole=0)
        
        # Create optimization model
        model = grb.Model('StrainMiner_QuasiBiclique', env=env)
        
        # Configure solver parameters
        model.Params.OutputFlag = 0        # Suppress console output
        model.Params.TimeLimit = time_limit # Set time limit in seconds
        model.Params.MIPGAP = mip_gap      # Set MIP gap tolerance
        
        # Optional: Add additional performance parameters
        model.Params.Threads = 1           # Single-threaded for consistency
        model.Params.Presolve = 2          # Aggressive presolve
        model.Params.Cuts = 2              # Aggressive cutting planes
        
        return model
        
    except grb.GurobiError as e:
        raise grb.GurobiError(f"Failed to setup Gurobi model: {e}")