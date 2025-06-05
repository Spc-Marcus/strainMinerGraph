import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

class StrainMinerPerformanceAnalyzer:
    """
    Analyseur de performance pour les données StrainMiner.
    
    Cette classe charge et analyse les fichiers CSV de statistiques pour identifier
    les corrélations entre taille des matrices, régions détectées et performance ILP.
    """
    
    def __init__(self, stats_directory: str):
        """
        Initialise l'analyseur avec un répertoire de statistiques.
        
        Parameters
        ----------
        stats_directory : str
            Chemin vers le répertoire contenant les fichiers CSV de statistiques
        """
        self.stats_dir = Path(stats_directory)
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Charge tous les fichiers CSV de statistiques."""
        print("📊 Chargement des données de performance...")
        
        # Patterns de fichiers
        file_patterns = {
            'benchmark': '*benchmark_stats_*.csv',
            'clustering': '*clustering_stats_*.csv', 
            'ilp': '*ilp_stats_*.csv',
            'matrix': '*matrix_stats_*.csv',
            'preprocessing': '*preprocessing_stats_*.csv'
        }
        
        for data_type, pattern in file_patterns.items():
            files = list(self.stats_dir.glob(pattern))
            if files:
                # Prendre le fichier le plus récent
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                try:
                    df = pd.read_csv(latest_file)
                    self.data[data_type] = df
                    print(f"✅ {data_type}: {len(df)} enregistrements chargés")
                except Exception as e:
                    print(f"❌ Erreur lors du chargement de {data_type}: {e}")
            else:
                print(f"⚠️  Aucun fichier trouvé pour {data_type}")
    
    def analyze_matrix_complexity(self):
        """
        Analyse la complexité des matrices et leur impact sur les performances.
        """
        print("\n🔍 Analyse de la complexité des matrices...")
        
        if 'benchmark' not in self.data:
            print("❌ Données benchmark non disponibles")
            return
        
        df = self.data['benchmark'].copy()
        
        # Calculer des métriques de complexité
        df['matrix_area'] = df['matrix_rows'] * df['matrix_cols']
        df['complexity_score'] = np.log10(df['matrix_area'] + 1)
        
        # Analyser par stage
        complexity_by_stage = df.groupby('stage').agg({
            'matrix_rows': ['mean', 'std', 'min', 'max'],
            'matrix_cols': ['mean', 'std', 'min', 'max'],
            'matrix_area': ['mean', 'std', 'min', 'max'],
            'execution_time_sec': ['mean', 'std', 'sum']
        }).round(2)
        
        print("\n📈 Complexité des matrices par étape:")
        print(complexity_by_stage)
        
        self.results['matrix_complexity'] = complexity_by_stage
        return complexity_by_stage
    
    def analyze_regions_vs_performance(self):
        """
        Analyse la corrélation entre le nombre de régions et les performances.
        """
        print("\n🎯 Analyse des régions vs performance...")
        
        correlations = {}
        
        # 1. Analyse du preprocessing
        if 'preprocessing' in self.data:
            prep_df = self.data['preprocessing']
            
            if len(prep_df) > 1:
                # Corrélations pour le preprocessing
                prep_corr = prep_df[['regions_found', 'inhomogeneous_regions', 
                                   'preprocessing_steps', 'execution_time_sec']].corr()
                print("\n📊 Corrélations dans le preprocessing:")
                print(prep_corr.round(3))
                correlations['preprocessing'] = prep_corr
        
        # 2. Analyse des benchmarks avec régions
        if 'benchmark' in self.data:
            bench_df = self.data['benchmark']
            
            # Filtrer les données avec informations de régions
            region_data = bench_df[bench_df['stage'].str.contains('region|preprocessing', na=False)]
            
            if len(region_data) > 1:
                region_cols = ['initial_cols', 'patterns_found', 'ilp_calls_total', 
                              'execution_time_sec', 'region_id']
                available_cols = [col for col in region_cols if col in region_data.columns]
                
                if len(available_cols) > 2:
                    region_corr = region_data[available_cols].corr()
                    print("\n📊 Corrélations par région:")
                    print(region_corr.round(3))
                    correlations['regions'] = region_corr
        
        self.results['correlations'] = correlations
        return correlations
    
    def analyze_ilp_performance(self):
        """
        Analyse détaillée des performances ILP.
        """
        print("\n⚡ Analyse des performances ILP...")
        
        if 'ilp' not in self.data:
            print("❌ Données ILP non disponibles")
            return
        
        ilp_df = self.data['ilp'].copy()
        
        # Calculer des métriques ILP
        ilp_df['complexity_score'] = ilp_df['variables'] * ilp_df['constraints'] / 1000000
        ilp_df['success'] = (ilp_df['solver_status'] == 'optimal').astype(int)
        
        # Analyse par phase
        phase_analysis = ilp_df.groupby('phase').agg({
            'execution_time_sec': ['mean', 'std', 'max'],
            'gap': ['mean', 'std'],
            'success': 'mean',
            'variables': 'mean',
            'constraints': 'mean',
            'complexity_score': 'mean'
        }).round(3)
        
        print("\n📊 Performance ILP par phase:")
        print(phase_analysis)
        
        # Corrélations ILP
        numeric_cols = ['matrix_rows', 'matrix_cols', 'execution_time_sec', 
                       'variables', 'constraints', 'gap', 'complexity_score']
        available_cols = [col for col in numeric_cols if col in ilp_df.columns]
        
        if len(available_cols) > 2:
            ilp_corr = ilp_df[available_cols].corr()
            print("\n📊 Corrélations dans les performances ILP:")
            print(ilp_corr.round(3))
            
            self.results['ilp_analysis'] = {
                'phase_analysis': phase_analysis,
                'correlations': ilp_corr
            }
        
        return phase_analysis, ilp_corr
    
    def find_performance_bottlenecks(self):
        """
        Identifie les goulets d'étranglement de performance.
        """
        print("\n🚧 Identification des goulets d'étranglement...")
        
        bottlenecks = {}
        
        if 'benchmark' in self.data:
            df = self.data['benchmark']
            
            # Temps par étape
            time_by_stage = df.groupby('stage')['execution_time_sec'].agg(['sum', 'mean', 'count'])
            time_by_stage['percentage'] = (time_by_stage['sum'] / time_by_stage['sum'].sum()) * 100
            time_by_stage = time_by_stage.sort_values('percentage', ascending=False)
            
            print("\n⏱️  Répartition du temps par étape:")
            print(time_by_stage.round(2))
            
            # Identifier les étapes lentes
            slow_stages = time_by_stage[time_by_stage['percentage'] > 20]
            if len(slow_stages) > 0:
                print(f"\n⚠️  Étapes consommant >20% du temps:")
                for stage, data in slow_stages.iterrows():
                    print(f"   • {stage}: {data['percentage']:.1f}% ({data['sum']:.1f}s)")
            
            bottlenecks['time_distribution'] = time_by_stage
        
        # Analyser les échecs ILP
        if 'ilp' in self.data:
            ilp_df = self.data['ilp']
            
            # Taux de succès par phase
            success_rate = ilp_df.groupby('phase').agg({
                'solver_status': lambda x: (x == 'optimal').mean()
            })
            success_rate.columns = ['success_rate']
            success_rate['success_rate'] *= 100
            
            print("\n✅ Taux de succès ILP par phase:")
            print(success_rate.round(1))
            
            # Solutions sous-optimales
            suboptimal = ilp_df[ilp_df['gap'] > 0.05]
            if len(suboptimal) > 0:
                print(f"\n⚠️  Solutions sous-optimales (gap > 5%): {len(suboptimal)}/{len(ilp_df)} "
                      f"({len(suboptimal)/len(ilp_df)*100:.1f}%)")
            
            bottlenecks['ilp_success'] = success_rate
        
        self.results['bottlenecks'] = bottlenecks
        return bottlenecks
    
    def create_correlation_plots(self, save_path: str = None):
        """
        Crée des graphiques de corrélation.
        """
        print("\n📈 Création des graphiques de corrélation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analyse des Corrélations StrainMiner', fontsize=16)
        
        # 1. Taille matrice vs temps d'exécution
        if 'benchmark' in self.data:
            df = self.data['benchmark']
            df['matrix_area'] = df['matrix_rows'] * df['matrix_cols']
            
            # Filtrer les valeurs aberrantes
            df_clean = df[(df['matrix_area'] > 0) & (df['execution_time_sec'] > 0)]
            
            axes[0,0].scatter(df_clean['matrix_area'], df_clean['execution_time_sec'], alpha=0.6)
            axes[0,0].set_xlabel('Taille matrice (lignes × colonnes)')
            axes[0,0].set_ylabel('Temps d\'exécution (sec)')
            axes[0,0].set_title('Taille matrice vs Temps d\'exécution')
            axes[0,0].set_xscale('log')
            axes[0,0].set_yscale('log')
            
            # Ajouter ligne de tendance
            if len(df_clean) > 1:
                z = np.polyfit(np.log10(df_clean['matrix_area']), 
                              np.log10(df_clean['execution_time_sec']), 1)
                p = np.poly1d(z)
                axes[0,0].plot(df_clean['matrix_area'], 
                              10**p(np.log10(df_clean['matrix_area'])), "r--", alpha=0.8)
        
        # 2. Régions vs appels ILP
        if 'benchmark' in self.data:
            region_data = df[df['stage'] == 'region_clustering']
            if len(region_data) > 0:
                axes[0,1].scatter(region_data['initial_cols'], region_data['ilp_calls_total'], alpha=0.6)
                axes[0,1].set_xlabel('Colonnes initiales par région')
                axes[0,1].set_ylabel('Appels ILP total')
                axes[0,1].set_title('Taille région vs Appels ILP')
        
        # 3. Complexité ILP vs temps
        if 'ilp' in self.data:
            ilp_df = self.data['ilp']
            ilp_df['complexity'] = ilp_df['variables'] * ilp_df['constraints']
            
            axes[1,0].scatter(ilp_df['complexity'], ilp_df['execution_time_sec'], alpha=0.6)
            axes[1,0].set_xlabel('Complexité ILP (variables × contraintes)')
            axes[1,0].set_ylabel('Temps d\'exécution (sec)')
            axes[1,0].set_title('Complexité ILP vs Temps')
            axes[1,0].set_xscale('log')
        
        # 4. Distribution des temps par étape
        if 'benchmark' in self.data:
            time_by_stage = df.groupby('stage')['execution_time_sec'].sum().sort_values(ascending=True)
            axes[1,1].barh(range(len(time_by_stage)), time_by_stage.values)
            axes[1,1].set_yticks(range(len(time_by_stage)))
            axes[1,1].set_yticklabels([s[:15] + '...' if len(s) > 15 else s for s in time_by_stage.index])
            axes[1,1].set_xlabel('Temps total (sec)')
            axes[1,1].set_title('Temps par étape du pipeline')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graphiques sauvegardés: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self):
        """
        Génère un rapport complet de performance.
        """
        print("\n📋 Génération du rapport de performance...")
        
        report = []
        report.append("# Rapport d'Analyse de Performance StrainMiner\n")
        
        # 1. Résumé des données
        report.append("## 1. Résumé des données\n")
        for data_type, df in self.data.items():
            report.append(f"- **{data_type}**: {len(df)} enregistrements")
        report.append("")
        
        # 2. Complexité des matrices
        if 'matrix_complexity' in self.results:
            report.append("## 2. Complexité des matrices\n")
            complexity = self.results['matrix_complexity']
            for stage in complexity.index:
                avg_area = complexity.loc[stage, ('matrix_area', 'mean')]
                avg_time = complexity.loc[stage, ('execution_time_sec', 'mean')]
                report.append(f"- **{stage}**: {avg_area:.0f} cellules moyenne, {avg_time:.2f}s moyenne")
            report.append("")
        
        # 3. Corrélations importantes
        if 'correlations' in self.results:
            report.append("## 3. Corrélations importantes\n")
            
            if 'regions' in self.results['correlations']:
                corr_matrix = self.results['correlations']['regions']
                if 'execution_time_sec' in corr_matrix.columns:
                    time_corrs = corr_matrix['execution_time_sec'].abs().sort_values(ascending=False)
                    report.append("**Facteurs corrélés au temps d'exécution:**")
                    for factor, corr in time_corrs.items():
                        if factor != 'execution_time_sec' and abs(corr) > 0.3:
                            direction = "positive" if corr > 0 else "négative"
                            report.append(f"- {factor}: {corr:.3f} (corrélation {direction})")
            report.append("")
        
        # 4. Goulets d'étranglement
        if 'bottlenecks' in self.results:
            report.append("## 4. Goulets d'étranglement identifiés\n")
            
            if 'time_distribution' in self.results['bottlenecks']:
                time_dist = self.results['bottlenecks']['time_distribution']
                slow_stages = time_dist[time_dist['percentage'] > 15]
                
                report.append("**Étapes consommant le plus de temps:**")
                for stage, data in slow_stages.iterrows():
                    report.append(f"- {stage}: {data['percentage']:.1f}% du temps total")
            report.append("")
        
        # 5. Recommandations
        report.append("## 5. Recommandations d'optimisation\n")
        
        # Analyser les données pour recommandations
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")
        
        report_text = "\n".join(report)
        
        # Sauvegarder le rapport
        report_path = self.stats_dir / "performance_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📋 Rapport généré: {report_path}")
        print("\n" + "="*60)
        print(report_text)
        print("="*60)
        
        return report_text
    
    def _generate_recommendations(self):
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []
        
        # Analyser les performances ILP
        if 'ilp' in self.data:
            ilp_df = self.data['ilp']
            avg_gap = ilp_df['gap'].mean()
            failure_rate = (ilp_df['solver_status'] != 'optimal').mean()
            
            if avg_gap > 0.05:
                recommendations.append("Réduire error_rate pour améliorer la qualité des solutions ILP")
            
            if failure_rate > 0.1:
                recommendations.append("Augmenter min_col_quality pour réduire la complexité des problèmes ILP")
        
        # Analyser les temps de traitement
        if 'benchmark' in self.data:
            df = self.data['benchmark']
            total_time = df['execution_time_sec'].sum()
            
            if total_time > 1800:  # > 30 minutes
                recommendations.append("Augmenter window_size pour réduire le nombre de fenêtres traitées")
                recommendations.append("Augmenter min_coverage pour filtrer plus agressivement les données")
        
        # Analyser la complexité des matrices
        if 'matrix_complexity' in self.results:
            complexity = self.results['matrix_complexity']
            if ('matrix_area', 'mean') in complexity.columns:
                avg_complexity = complexity[('matrix_area', 'mean')].mean()
                if avg_complexity > 100000:  # Matrices très grandes
                    recommendations.append("Utiliser des filtres plus stricts (min_row_quality, min_col_quality)")
        
        if not recommendations:
            recommendations.append("Les performances semblent optimales avec les paramètres actuels")
        
        return recommendations
    
    def run_complete_analysis(self):
        """
        Lance une analyse complète de performance.
        """
        print("🚀 Lancement de l'analyse complète de performance StrainMiner\n")
        
        # 1. Charger les données
        self.load_data()
        
        if not self.data:
            print("❌ Aucune donnée chargée. Vérifiez le répertoire de statistiques.")
            return
        
        # 2. Analyses individuelles
        self.analyze_matrix_complexity()
        self.analyze_regions_vs_performance()
        self.analyze_ilp_performance()
        self.find_performance_bottlenecks()
        
        # 3. Créer les graphiques
        plot_path = self.stats_dir / "correlation_analysis.png"
        self.create_correlation_plots(str(plot_path))
        
        # 4. Générer le rapport
        self.generate_performance_report()
        
        print("\n✅ Analyse complète terminée!")
        return self.results

def main():
    """Exemple d'utilisation du script d'analyse."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_performance.py <stats_directory>")
        print("Exemple: python analyze_performance.py output_strainminer/stats/")
        sys.exit(1)
    
    stats_dir = sys.argv[1]
    
    # Créer l'analyseur et lancer l'analyse
    analyzer = StrainMinerPerformanceAnalyzer(stats_dir)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()
