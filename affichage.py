import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors




data_str = """
1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1
0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1
1	1	1	0	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1
1	1	1	1	0	0	0	0	0	1	1	1	1	1	1	0	0	1	0	0	0	1	1	0	1	1	1	1	0	1	1	1	1	1	1	1	1	0	0	0	0	0	1	1	1	1	1	1
1	1	1	1	0	0	0	0	0	1	1	1	1	1	1	0	0	0	0	0	0	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	0	0	0	0	0	1	1	1	1	1	1
1	1	1	1	0	0	0	0	0	1	1	1	1	1	1	0	0	0	0	0	0	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	0	0	0	0	0	1	1	1	1	1	1
1	0	1	1	0	1	0	0	0	1	1	1	1	1	1	0	0	1	0	0	0	1	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	0	0	1	0	0	1	1	0	1	1	1
1	1	1	1	0	0	0	0	0	1	1	1	1	1	1	0	0	0	0	0	0	1	1	1	0	1	1	1	0	1	1	1	1	1	1	1	1	0	0	0	0	0	1	1	1	1	1	1
1	1	0	0	0	0	1	1	1	1	1	1	0	0	0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	0	1	0	0	1	1	1	1	1	1	1	1	1	1	0	1	1	1	1
1	1	0	0	0	0	1	1	1	1	1	1	0	0	0	0	0	1	1	1	0	1	1	1	1	1	1	1	1	0	0	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	0	0	0	0	1	1	1	0	1	1	0	0	0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
1	1	0	0	0	0	1	1	0	1	1	1	0	0	0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
"""

matrix = np.array([list(map(int, line.split())) for line in data_str.strip().split('\n')])
regions1 = [
    (1,[(1,19)]),
    (3,[(1,15),(0,19)]),
    (5,[(1,10),(0,19)]),
    (8,[(1,10),(0,15),(1,19)])
]
regions2=[
    (1,[],[1]),
    (3,[15],[1,0]),
    (5,[10],[1,0]),
    (8,[10,15],[1,0,1]),
    (11,[],[1]),
    (14,[15],[1,0]),
    (16,[10],[1,0]),
    (20,[10,15],[1,0,1])
]
def plot_matrix(matrix, regions):
    plt.figure(figsize=(12, 8))
    cmap = mcolors.ListedColormap(['white', 'black'])
    plt.imshow(matrix, cmap=cmap, interpolation='none', vmin=0, vmax=1, 
               extent=[0, matrix.shape[1], 0, matrix.shape[0]])  # Fixed extent order
    row_count = matrix.shape[0] -1

    last_c = 0
    for c, rows,v in regions:
        # Draw horizontal lines for each row
        c=c+1
        for r in rows:
            # Horizontal line from last column to current column at row r
            plt.plot([last_c, c], [row_count-r, row_count-r], color='red', linewidth=2)
        
        
        plt.plot([c, c], [0, matrix.shape[0]], color='blue', linewidth=2)
        
        last_c = c

    # Set axis limits
    plt.xlim(0, matrix.shape[1])
    plt.ylim(0, matrix.shape[0])  # Normal orientation (not inverted)
    
    # Add axis labels
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    plt.grid(False)
    plt.tight_layout()
    plt.show()
def create_masque(matrix, regions):
    mask = matrix.copy()
    last_col = 0
    for region in regions:
        col_max = region[0]
        last_row = 0
        if col_max < last_col:
            raise ValueError("Les colonnes doivent être triées par ordre croissant.")
        if col_max >= matrix.shape[1]:
            raise ValueError("La colonne maximale dépasse la taille de la matrice.")
        for subregion in region[1]:
            value = subregion[0]
            row = subregion[1]
            if row < last_row:
                raise ValueError("Les lignes doivent être triées par ordre croissant.")
            if row >= matrix.shape[0]:
                raise ValueError("La ligne maximale dépasse la taille de la matrice.")
            mask[last_row:row+1, last_col:col_max+1] = value
            last_row = row+1
        last_col = col_max+1
    return mask


#m = create_masque(matrix, regions1)

#plot_matrix(m, regions2)

