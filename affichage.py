import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import bitarray_heapsort




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
    plt.xlabel('Sus-pos')
    plt.ylabel('Reads')
    
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


def get_matrix_from_json(file_path):
    """
    Convertit les données JSON en une matrice NumPy.
    Les positions sont en abscisse (colonnes), les reads en ordonnée (lignes).
    Si une valeur est absente pour un read, elle sera -1.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extraire tous les identifiants de reads uniques
    read_ids = set()
    for position_data in data.values():
        read_ids.update(position_data.keys())
    
    # Trier les reads et les positions pour une matrice cohérente
    sorted_read_ids = sorted(read_ids, key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    sorted_positions = sorted(data.keys(), key=int)
    
    # Créer la matrice vide (lignes: reads, colonnes: positions), initialisée à -1
    matrix = np.full((len(sorted_read_ids), len(sorted_positions)), 0, dtype=np.uint8)
    
    # Remplir la matrice
    for j, position in enumerate(sorted_positions):
        position_data = data[position]
        for i, read_id in enumerate(sorted_read_ids):
            if read_id in position_data:
                matrix[i, j] = position_data[read_id]
    
    # Créer un dictionnaire de mapping pour faciliter l'interprétation
    position_mapping = {j: position for j, position in enumerate(sorted_positions)}
    read_mapping = {i: read_id for i, read_id in enumerate(sorted_read_ids)}
    
    return matrix, position_mapping, read_mapping

# Charger et convertir les données
matrix, positions, reads = get_matrix_from_json('/udd/mfoin/Dev/strainMinerGraph/output_strainminer/tmp/dict_of_sus_pos_ctg0_0.json')

print("\nMatrice convertie depuis JSON:")
print(f"Forme: {matrix.shape} (positions × reads)")
print(f"Nombre de positions: {len(positions)}")
print(f"Nombre de reads: {len(reads)}")
# Afficher la matrice
print(matrix)
matrix = bitarray_heapsort.sort_numpy(matrix)

plot_matrix(matrix,[])