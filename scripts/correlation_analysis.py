import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy import stats
from pathlib import Path

def analyze_matrix_size_correlations(stats_dir: str):
    """
    Analyse spécifique des corrélations entre taille de matrice et performance.
    """
    print("🔍 Analyse des corrélations taille de matrice...")
    
    # Charger les données benchmark
    benchmark_files = list(Path(stats_dir).glob('*benchmark_stats_*.csv'))
    if not benchmark_files:
        print("❌ Fichiers benchmark non trouvés")
        return
    
    df = pd.read_csv(benchmark_files[0])
    
    # Calculer des métriques de taille
    df['matrix_area'] = df['matrix_rows'] * df['matrix_cols']
    df['matrix_density'] = df['matrix_area'] / (df['matrix_rows'] + df['matrix_cols'])
    
    # Filtrer les données valides
    valid_data = df[(df['matrix_area'] > 0) & (df['execution_time_sec'] > 0)]
    
    correlations = {}
    
    # 1. Corrélation Pearson (linéaire)
    if len(valid_data) > 3:
        pearson_corr, pearson_p = pearsonr(valid_data['matrix_area'], valid_data['execution_time_sec'])
        spearman_corr, spearman_p = spearmanr(valid_data['matrix_area'], valid_data['execution_time_sec'])
        
        correlations['matrix_area_vs_time'] = {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
        }
        
        print(f"📊 Corrélation taille matrice vs temps:")
        print(f"   • Pearson: {pearson_corr:.3f} (p={pearson_p:.3f})")
        print(f"   • Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
    
    # 2. Analyse par étape
    stage_correlations = {}
    for stage in valid_data['stage'].unique():
        stage_data = valid_data[valid_data['stage'] == stage]
        if len(stage_data) > 3:
            corr, p = spearmanr(stage_data['matrix_area'], stage_data['execution_time_sec'])
            stage_correlations[stage] = {'correlation': corr, 'p_value': p}
            print(f"   • {stage}: {corr:.3f} (p={p:.3f})")
    
    correlations['by_stage'] = stage_correlations
    
    return correlations

def analyze_regions_ilp_correlations(stats_dir: str):
    """
    Analyse des corrélations entre régions et utilisation ILP.
    """
    print("\n🎯 Analyse régions vs ILP...")
    
    # Charger les données
    benchmark_files = list(Path(stats_dir).glob('*benchmark_stats_*.csv'))
    ilp_files = list(Path(stats_dir).glob('*ilp_stats_*.csv'))
    
    if not benchmark_files or not ilp_files:
        print("❌ Fichiers nécessaires non trouvés")
        return
    
    benchmark_df = pd.read_csv(benchmark_files[0])
    ilp_df = pd.read_csv(ilp_files[0])
    
    # Données de régions depuis benchmark
    region_data = benchmark_df[benchmark_df['stage'] == 'region_clustering'].copy()
    
    if len(region_data) == 0:
        print("❌ Pas de données de régions trouvées")
        return
    
    # Analyser les corrélations dans les régions
    correlations = {}
    
    # 1. Colonnes initiales vs appels ILP
    if 'initial_cols' in region_data.columns and 'ilp_calls_total' in region_data.columns:
        valid_regions = region_data.dropna(subset=['initial_cols', 'ilp_calls_total'])
        if len(valid_regions) > 3:
            corr, p = spearmanr(valid_regions['initial_cols'], valid_regions['ilp_calls_total'])
            correlations['cols_vs_ilp_calls'] = {'correlation': corr, 'p_value': p}
            print(f"📊 Colonnes initiales vs appels ILP: {corr:.3f} (p={p:.3f})")
    
    # 2. Taille matrice région vs temps ILP total
    region_data.loc[:, 'region_area'] = region_data['matrix_rows'] * region_data['matrix_cols']
    if len(region_data) > 3:
        corr, p = spearmanr(region_data['region_area'], region_data['execution_time_sec'])
        correlations['region_area_vs_time'] = {'correlation': corr, 'p_value': p}
        print(f"📊 Taille région vs temps: {corr:.3f} (p={p:.3f})")
    
    # 3. Analyser les données ILP par phase
    ilp_phase_stats = ilp_df.groupby('phase').agg({
        'execution_time_sec': ['mean', 'sum', 'count'],
        'variables': 'mean',
        'constraints': 'mean',
        'gap': 'mean'
    })
    
    print(f"\n📈 Statistiques ILP par phase:")
    print(ilp_phase_stats.round(3))
    
    return correlations, ilp_phase_stats

def create_advanced_correlation_plots(stats_dir: str, save_path: str = None):
    """
    Crée des graphiques de corrélation avancés.
    """
    print("\n📈 Création des graphiques de corrélation avancés...")
    
    # Configuration du style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # Charger les données
    files = {
        'benchmark': list(Path(stats_dir).glob('*benchmark_stats_*.csv')),
        'ilp': list(Path(stats_dir).glob('*ilp_stats_*.csv')),
        'clustering': list(Path(stats_dir).glob('*clustering_stats_*.csv'))
    }
    
    data = {}
    for file_type, file_list in files.items():
        if file_list:
            data[file_type] = pd.read_csv(file_list[0])
    
    # 1. Analyse de scalabilité par composant (subplot 1)
    plt.subplot(3, 3, 1)
    if 'benchmark' in data:
        df = data['benchmark'].copy()
        
        # Calculer la taille de la matrice
        df['matrix_area'] = df['matrix_rows'] * df['matrix_cols']
        
        # Identifier les étapes critiques
        critical_stages = df.groupby('stage')['execution_time_sec'].sum().nlargest(3)
        
        # Créer le graphique de scalabilité
        colors = plt.cm.Set1(np.linspace(0, 1, len(critical_stages)))
        
        for i, stage in enumerate(critical_stages.index):
            stage_data = df[df['stage'] == stage]
            
            if len(stage_data) > 3:
                # Calculer la pente de scalabilité
                valid_data = stage_data[(stage_data['matrix_area'] > 0) & 
                                      (stage_data['execution_time_sec'] > 0)]
                
                if len(valid_data) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        np.log10(valid_data['matrix_area'] + 1),
                        np.log10(valid_data['execution_time_sec'] + 0.001)
                    )
                    
                    # Tracer les points et la ligne de tendance
                    plt.scatter(valid_data['matrix_area'], valid_data['execution_time_sec'], 
                              alpha=0.7, c=[colors[i]], label=f'{stage[:15]} (slope={slope:.2f})')
                    
                    # Ligne de tendance
                    x_trend = np.logspace(np.log10(valid_data['matrix_area'].min()), 
                                        np.log10(valid_data['matrix_area'].max()), 50)
                    y_trend = 10**(slope * np.log10(x_trend + 1) + intercept)
                    plt.plot(x_trend, y_trend, '--', color=colors[i], alpha=0.8)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Taille matrice (log)')
        plt.ylabel('Temps exécution (log)')
        plt.title('Analyse de Scalabilité par Composant')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2. Taille matrice vs temps (log-log) (subplot 2)
    plt.subplot(3, 3, 2)
    if 'benchmark' in data:
        df = data['benchmark']
        df['matrix_area'] = df['matrix_rows'] * df['matrix_cols']
        valid_data = df[(df['matrix_area'] > 0) & (df['execution_time_sec'] > 0)]
        
        # Points colorés par stage
        stages = valid_data['stage'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
        
        for i, stage in enumerate(stages):
            stage_data = valid_data[valid_data['stage'] == stage]
            plt.scatter(stage_data['matrix_area'], stage_data['execution_time_sec'], 
                       alpha=0.7, c=[colors[i]], label=stage[:15])
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Taille matrice (log)')
        plt.ylabel('Temps exécution (log)')
        plt.title('Taille vs Temps par Étape')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Distribution des temps par étape (subplot 3)
    plt.subplot(3, 3, 3)
    if 'benchmark' in data:
        time_by_stage = data['benchmark'].groupby('stage')['execution_time_sec'].sum().sort_values()
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_by_stage)))
        
        bars = plt.barh(range(len(time_by_stage)), time_by_stage.values, color=colors)
        plt.yticks(range(len(time_by_stage)), 
                   [s[:20] + '...' if len(s) > 20 else s for s in time_by_stage.index])
        plt.xlabel('Temps total (sec)')
        plt.title('Distribution du Temps')
        
        # Ajouter les valeurs sur les barres
        for i, (bar, value) in enumerate(zip(bars, time_by_stage.values)):
            plt.text(value + max(time_by_stage.values) * 0.01, i, f'{value:.1f}s', 
                    va='center', fontsize=8)
    
    # 4. Performance ILP par phase (subplot 4)
    plt.subplot(3, 3, 4)
    if 'ilp' in data:
        ilp_df = data['ilp']
        
        # Temps moyen par phase
        phase_times = ilp_df.groupby('phase')['execution_time_sec'].mean()
        plt.bar(range(len(phase_times)), phase_times.values)
        plt.xticks(range(len(phase_times)), phase_times.index, rotation=45)
        plt.ylabel('Temps moyen (sec)')
        plt.title('Temps ILP par Phase')
    
    # 5. Complexité ILP vs performance (subplot 5)
    plt.subplot(3, 3, 5)
    if 'ilp' in data:
        ilp_df = data['ilp'].copy()
        ilp_df['complexity'] = ilp_df['variables'] * ilp_df['constraints']
        
        # Colorer par phase
        phases = ilp_df['phase'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
        
        for i, phase in enumerate(phases):
            phase_data = ilp_df[ilp_df['phase'] == phase]
            plt.scatter(phase_data['complexity'], phase_data['execution_time_sec'], 
                       alpha=0.7, c=[colors[i]], label=phase)
        
        plt.xscale('log')
        plt.xlabel('Complexité ILP (log)')
        plt.ylabel('Temps exécution (sec)')
        plt.title('Complexité vs Performance ILP')
        plt.legend()
    
    # 6. Taux de succès ILP (subplot 6)
    plt.subplot(3, 3, 6)
    if 'ilp' in data:
        success_by_phase = ilp_df.groupby('phase').apply(
            lambda x: (x['solver_status'] == 'optimal').mean() * 100
        )
        
        bars = plt.bar(range(len(success_by_phase)), success_by_phase.values)
        plt.xticks(range(len(success_by_phase)), success_by_phase.index, rotation=45)
        plt.ylabel('Taux de succès (%)')
        plt.title('Taux de Succès ILP')
        plt.ylim(0, 105)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, success_by_phase.values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value:.1f}%', 
                    ha='center', va='bottom')
    
    # 7. Régions vs appels ILP (subplot 7)
    plt.subplot(3, 3, 7)
    if 'benchmark' in data:
        region_data = data['benchmark'][data['benchmark']['stage'] == 'region_clustering']
        if len(region_data) > 0 and 'initial_cols' in region_data.columns:
            plt.scatter(region_data['initial_cols'], region_data['ilp_calls_total'], alpha=0.7)
            plt.xlabel('Colonnes initiales')
            plt.ylabel('Appels ILP total')
            plt.title('Taille Région vs Appels ILP')
            
            # Ajouter ligne de tendance
            if len(region_data) > 2:
                z = np.polyfit(region_data['initial_cols'], region_data['ilp_calls_total'], 1)
                p = np.poly1d(z)
                plt.plot(region_data['initial_cols'], p(region_data['initial_cols']), "r--", alpha=0.8)
    
    # 8. Distribution des gaps ILP (subplot 8)
    plt.subplot(3, 3, 8)
    if 'ilp' in data:
        gaps = ilp_df['gap'].dropna()
        if len(gaps) > 0:
            plt.hist(gaps, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(gaps.mean(), color='red', linestyle='--', label=f'Moyenne: {gaps.mean():.3f}')
            plt.axvline(0.025, color='orange', linestyle='--', label='Seuil acceptable (2.5%)')
            plt.xlabel('Gap d\'optimalité')
            plt.ylabel('Fréquence')
            plt.title('Distribution des Gaps ILP')
            plt.legend()
    
    # 9. Patterns trouvés vs temps (subplot 9)
    plt.subplot(3, 3, 9)
    if 'clustering' in data:
        clustering_df = data['clustering']
        valid_clustering = clustering_df[(clustering_df['patterns_found'] >= 0) & 
                                        (clustering_df['execution_time_sec'] > 0)]
        
        if len(valid_clustering) > 0:
            plt.scatter(valid_clustering['patterns_found'], valid_clustering['execution_time_sec'], alpha=0.7)
            plt.xlabel('Patterns trouvés')
            plt.ylabel('Temps exécution (sec)')
            plt.title('Patterns vs Temps de Clustering')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Graphiques sauvegardés: {save_path}")
    
    plt.show()

def main():
    """Exemple d'utilisation du script de corrélation."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python correlation_analysis.py <stats_directory>")
        sys.exit(1)
    
    stats_dir = sys.argv[1]
    
    # Analyses de corrélation
    matrix_corr = analyze_matrix_size_correlations(stats_dir)
    regions_corr, ilp_stats = analyze_regions_ilp_correlations(stats_dir)
    
    # Graphiques
    plot_path = Path(stats_dir) / "advanced_correlations.png"
    create_advanced_correlation_plots(stats_dir, str(plot_path))
    
    print("\n✅ Analyse de corrélation terminée!")

if __name__ == "__main__":
    main()
