# Explication de la fonction `biclustering_full_matrix`

## Vue d'ensemble

Cette fonction effectue un **clustering itératif** sur l'ensemble de la matrice génomique pour identifier tous les sous-groupes de reads (haplotypes) possibles dans chaque région identifiée.

## Étapes détaillées

### 1. Initialisation
```python
steps_result = steps  # Conserve les étapes déjà trouvées lors du préprocessing
```

### 2. Traitement de chaque région
```python
for idx,region in enumerate(regions):   
    remain_cols = region  # Colonnes restantes à traiter dans cette région
```

Pour chaque région identifiée lors du préprocessing, la fonction va extraire tous les clusters possibles.

### 3. Clustering itératif par région
```python
while len(remain_cols)>=min_col_quality and status:
    reads1,reads0,cols = binary_clustering_step(X_matrix[:,remain_cols], 
                                               error_rate=error_rate,
                                               min_row_quality=5, 
                                               min_col_quality=3)
```

**Processus itératif :**
- Applique `binary_clustering_step` sur les colonnes restantes
- Cette fonction trouve un cluster de reads avec pattern similaire
- Retourne :
  - `reads1` : reads avec variants (pattern "1")
  - `reads0` : reads sans variants (pattern "0") 
  - `cols` : colonnes où ce pattern est significatif

### 4. Mise à jour et continuation
```python
cols = [remain_cols[c] for c in cols]  # Conversion indices locaux → indices globaux

if len(cols) == 0:
    status = False  # Arrêt si aucune colonne significative
else:
    steps_result.append((reads1,reads0,cols))  # Sauvegarde du cluster trouvé
    remain_cols = [c for c in remain_cols if c not in cols]  # Retire les colonnes traitées
```

### 5. Filtrage final
```python
return [step for step in steps_result if len(step[0])>0 and len(step[1])>0 and len(step[2])>=min_col_quality]
```

Ne garde que les clusters valides avec :
- Au moins 1 read dans chaque groupe (reads1 et reads0)
- Au moins `min_col_quality` positions génomiques

## Exemple concret

```python
# Région initiale avec 10 colonnes [0,1,2,3,4,5,6,7,8,9]
region = [0,1,2,3,4,5,6,7,8,9]

# Itération 1 : trouve un cluster sur colonnes [0,1,2]
reads1=[0,1,2], reads0=[3,4], cols=[0,1,2]
steps_result.append(([0,1,2], [3,4], [0,1,2]))
remain_cols = [3,4,5,6,7,8,9]  # Retire [0,1,2]

# Itération 2 : trouve un cluster sur colonnes [5,6]
reads1=[1,5], reads0=[0,3], cols=[5,6]  
steps_result.append(([1,5], [0,3], [5,6]))
remain_cols = [3,4,7,8,9]  # Retire [5,6]

# Itération 3 : pas assez de colonnes significatives
reads1=[], reads0=[], cols=[]
status = False  # Arrêt
```

## Objectif biologique

Cette fonction permet de :

1. **Extraire tous les haplotypes** présents dans chaque région génomique
2. **Maximiser l'information** en trouvant tous les patterns de variants possibles
3. **Séparer les souches microbiennes** de manière exhaustive
4. **Préparer les données** pour le post-processing qui assignera chaque read à un haplotype final

## Résultat

La fonction retourne une **liste complète d'étapes de clustering** qui sera utilisée par `post_processing()` pour construire l'arbre de décision final permettant de séparer tous les reads en groupes d'haplotypes distincts.

C'est l'étape centrale de **StrainMiner** qui identifie tous les variants informatifs pour la séparation des souches.



# Explication de la fonction `binary_clustering_step`

## Vue d'ensemble

Cette fonction effectue **une étape de clustering binaire** sur une matrice génomique pour identifier un groupe de reads ayant un pattern de variants similaire. Elle utilise l'algorithme de **quasi-biclique** pour trouver des sous-matrices cohérentes.

## Étapes détaillées

### 1. Préparation des matrices
```python
X_problem_1 = X_problem.copy()
X_problem_1[X_problem_1 == -1] = 0  # Matrice pour chercher les "1" (variants présents)

X_problem_0 = X_problem.copy()
X_problem_0[X_problem_0 == -1] = 1
X_problem_0 = (X_problem_0-1)*-1    # Matrice inversée pour chercher les "0" (variants absents)
```

**Objectif** : Créer deux matrices pour pouvoir chercher alternativement :
- Les groupes de reads **avec variants** (matrice 1)
- Les groupes de reads **sans variants** (matrice 0 inversée)

### 2. Boucle de clustering alterné
```python
while len(remain_rows)>=min_row_quality and len(current_cols)>=min_col_quality and status:
    if clustering_1:
        rw,cl,status = quasibiclique(X_problem_1[remain_rows][:,current_cols],error_rate)
    else:
        rw,cl,status = quasibiclique(X_problem_0[remain_rows][:,current_cols],error_rate)
```

**Processus alterné** :
- **Étape 1** : Cherche un quasi-biclique de "1" (reads avec variants)
- **Étape 2** : Cherche un quasi-biclique de "0" (reads sans variants)
- **Répète** jusqu'à ne plus trouver de patterns significatifs

### 3. Validation et filtrage des colonnes
```python
if clustering_1:
    col_homogeneity = X_problem_1[rw][:,cl].sum(axis=0)/len(rw)
else:
    col_homogeneity = X_problem_0[rw][:,cl].sum(axis=0)/len(rw)
current_cols = [c for idx,c in enumerate(cl) if col_homogeneity[idx] > 5*error_rate]
```

**Filtrage du bruit** :
- Calcule l'homogénéité de chaque colonne dans le cluster trouvé
- Ne garde que les colonnes avec >5× le taux d'erreur (≥12.5% par défaut)
- Élimine les colonnes trop bruyantes

### 4. Accumulation et suppression
```python
if clustering_1:
    rw1 = rw1 + [r for r in rw]  # Accumule les reads avec variants
else:
    rw0 = rw0 + [r for r in rw]  # Accumule les reads sans variants

remain_rows = [r for r in remain_rows if r not in rw]  # Retire les reads traités
clustering_1 = not clustering_1  # Alterne le type de clustering
```

## Exemple concret

```python
# Matrice d'entrée (4 reads × 6 positions)
X_matrix = [
    [1, 1, 0, 0, -1, 1],  # read 0
    [1, 1, 0, 0, 1, 1],   # read 1  
    [0, 0, 1, 1, 0, 0],   # read 2
    [0, 0, 1, 1, -1, 0]   # read 3
]

# Itération 1 (clustering_1=True) : cherche patterns de "1"
# → Trouve reads [0,1] sur colonnes [0,1,5] (pattern de variants)
# → rw1 = [0,1], current_cols = [0,1,5], remain_rows = [2,3]

# Itération 2 (clustering_1=False) : cherche patterns de "0" 
# → Trouve reads [2,3] sur colonnes [0,1] (absence de variants)
# → rw0 = [2,3], current_cols = [0,1], remain_rows = []

# Résultat final :
# rw1 = [0,1]      # Reads avec variants
# rw0 = [2,3]      # Reads sans variants  
# current_cols = [0,1]  # Positions informatives pour cette séparation
```

## Objectif biologique

Cette fonction identifie **une séparation significative** entre deux groupes de reads :

1. **Groupe 1** (`rw1`) : Reads qui **ont des variants** aux positions identifiées
2. **Groupe 0** (`rw0`) : Reads qui **n'ont pas de variants** aux mêmes positions
3. **Positions** (`current_cols`) : Positions génomiques qui permettent cette séparation

## Utilisation dans le pipeline

- Appelée répétitivement par `biclustering_full_matrix`
- Chaque appel trouve **une nouvelle séparation** dans les données restantes
- Les résultats s'accumulent pour créer un **arbre de décision** complet
- Permet de séparer progressivement tous les haplotypes présents

Cette fonction est le **cœur algorithmique** de StrainMiner qui identifie les variants discriminants pour séparer les souches microbiennes.

# Explication de la fonction `quasibiclique`

## Vue d'ensemble

Cette fonction trouve un **quasi-biclique** dans une matrice binaire en utilisant la **programmation linéaire entière** avec Gurobi. Un quasi-biclique est une sous-matrice où la plupart des éléments sont des "1", avec une tolérance d'erreur définie.

## Étapes détaillées

### 1. Préparation et tri
```python
cols_sorted = np.argsort(X_problem.sum(axis = 0))[::-1]  # Colonnes triées par nb de 1s (décroissant)
rows_sorted = np.argsort(X_problem.sum(axis = 1))[::-1]  # Lignes triées par nb de 1s (décroissant)
```

**Stratégie** : Commencer par les lignes/colonnes ayant le plus de "1" pour maximiser les chances de trouver un biclique dense.

### 2. Sélection d'une région de départ (seeding)
```python
seed_rows = m//3
seed_cols = n//3
# Recherche d'une région avec >99% de 1s
for x in range(m//3,m,10):
    for y in range(n//3,n,step_n):
        ratio_of_1 = nb_of_1_in_seed/(x*y)
        if ratio_of_1 > 0.99 and x*y>seed_rows*seed_cols:
            seed_rows = x
            seed_cols = y
```

**Objectif** : Trouver la plus grande sous-matrice initiale avec presque 100% de "1" pour démarrer l'optimisation.

### 3. Modélisation en programmation linéaire

#### Variables de décision
```python
lpRows = model.addVars(rows_sorted[:seed_rows], vtype=grb.GRB.INTEGER, name='rw')    # 1 si ligne sélectionnée
lpCols = model.addVars(cols_sorted[:seed_cols], vtype=grb.GRB.INTEGER, name='cl')    # 1 si colonne sélectionnée  
lpCells = model.addVars([(r,c) for r,c in product], vtype=grb.GRB.INTEGER, name='ce') # 1 si cellule sélectionnée
```

#### Fonction objectif
```python
model.setObjective(grb.quicksum([(X_problem[c[0]][c[1]]) * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)
```
**Maximise la somme des valeurs** dans les cellules sélectionnées.

#### Contraintes principales
```python
# Une cellule ne peut être sélectionnée que si sa ligne ET sa colonne sont sélectionnées
model.addConstr(1 - lpRows[cell[0]] >= lpCells[cell], f'{cell}_cr')
model.addConstr(1 - lpCols[cell[1]] >= lpCells[cell], f'{cell}_cc')
model.addConstr(1 - lpRows[cell[0]] - lpCols[cell[1]] <= lpCells[cell], f'{cell}_ccr')

# Contrainte de taux d'erreur : nb de 0s ≤ error_rate × nb total de cellules
model.addConstr(error_rate*lpCells.sum() >= grb.quicksum([lpCells[coord]*(1-X_problem[coord[0]][coord[1]]) for coord in lpCells]))
```

### 4. Extension itérative

#### Extension par lignes
```python
rem_rows_sum = X_problem[rem_rows][:,cl].sum(axis=1)
potential_rows = [r for idx,r in enumerate(rem_rows) if rem_rows_sum[idx]>0.5*len(cl)]
```
Ajoute les lignes qui ont >50% de "1" dans les colonnes déjà sélectionnées.

#### Extension par colonnes  
```python
rem_cols_sum = X_problem[rw][:,rem_cols].sum(axis=0)
potential_cols = [c for idx,c in enumerate(rem_cols) if rem_cols_sum[idx]>0.9*len(rw)]
```
Ajoute les colonnes qui ont >90% de "1" dans les lignes déjà sélectionnées.

### 5. Extraction du résultat
```python
for var in model.getVars():
    if var.X == 0:  # Variables NON sélectionnées
        if name[0:2] == 'rw':
            rw += [int(name[3:-1])]
        elif name[0:2] == 'cl':
            cl += [int(name[3:-1])]
```

⚠️ **Note importante** : Le code récupère les variables avec `var.X == 0`, ce qui semble contre-intuitif. Cela pourrait être une inversion logique dans l'implémentation.

## Exemple concret

```python
# Matrice d'entrée
X_matrix = [
    [1, 1, 0, 1],  # ligne 0: 3/4 = 75% de 1s
    [1, 1, 1, 1],  # ligne 1: 4/4 = 100% de 1s  
    [0, 1, 1, 1],  # ligne 2: 3/4 = 75% de 1s
    [1, 0, 1, 0]   # ligne 3: 2/4 = 50% de 1s
]

# Après tri : rows_sorted = [1, 0, 2, 3], cols_sorted = [1, 2, 0, 3]

# Seeding : trouve région [1,0] × [1,2] avec 100% de 1s
# Extension : ajoute ligne 2 (compatible)
# Résultat : lignes [0,1,2], colonnes [1,2] forment un quasi-biclique dense
```

## Objectif biologique

Dans le contexte génomique :
- **Lignes** = reads (séquences)
- **Colonnes** = positions génomiques  
- **Quasi-biclique** = groupe de reads ayant des variants similaires aux mêmes positions
- **Tolérance d'erreur** = prise en compte du bruit de séquençage

Cette fonction identifie des **groupes cohérents de reads** partageant des patterns de variants similaires, ce qui est essentiel pour séparer les différentes souches microbiennes.