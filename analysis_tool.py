 #!/usr/bin/env python3
"""
Ferramenta de análise pós-treinamento para comparar resultados de diferentes estratégias
e gerar relatórios consolidados.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FLAnalyzer:
    """Analisador de resultados de Aprendizado Federado."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.metrics_dir = Path("metrics")
        self.strategies_data = {}
        self.client_data = {}
        
    def load_server_metrics(self):
        """Carrega métricas do servidor para todas as estratégias."""
        print("Carregando métricas do servidor...")
        
        for csv_file in self.results_dir.glob("detailed_metrics_*.csv"):
            strategy = csv_file.stem.replace("detailed_metrics_", "")
            df = pd.read_csv(csv_file)
            self.strategies_data[strategy] = df
            print(f"  - Carregado: {strategy}")
    
    def load_client_metrics(self):
        """Carrega métricas individuais dos clientes."""
        print("Carregando métricas dos clientes...")
        
        for client_dir in self.metrics_dir.glob("client_*"):
            client_id = int(client_dir.name.split("_")[1])
            metrics_file = client_dir / "metrics_history.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.client_data[client_id] = json.load(f)
                print(f"  - Cliente {client_id} carregado")
    
    def generate_comparative_analysis(self):
        """Gera análise comparativa entre estratégias."""
        if not self.strategies_data:
            print("Nenhum dado de estratégia encontrado!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Comparação de convergência entre estratégias
        ax1 = axes[0, 0]
        for strategy, df in self.strategies_data.items():
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty and 'global_eval_loss' in eval_df.columns:
                ax1.plot(eval_df['rounds'], eval_df['global_eval_loss'], 
                        marker='o', label=strategy.upper(), linewidth=2)
        
        ax1.set_title('Comparação de Convergência entre Estratégias', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Rodada')
        ax1.set_ylabel('Perda de Validação Global')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Taxa de convergência
        ax2 = axes[0, 1]
        convergence_rates = {}
        
        for strategy, df in self.strategies_data.items():
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty and 'global_eval_loss' in eval_df.columns:
                losses = eval_df['global_eval_loss'].values
                if len(losses) > 1:
                    # Taxa de melhoria por rodada
                    improvements = -np.diff(losses)
                    convergence_rates[strategy] = improvements
                    ax2.plot(eval_df['rounds'].values[1:], improvements, 
                            marker='s', label=strategy.upper(), linewidth=1.5, alpha=0.7)
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Taxa de Melhoria por Rodada', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rodada')
        ax2.set_ylabel('Redução na Perda')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Boxplot de desempenho final dos clientes
        ax3 = axes[1, 0]
        final_performances = []
        labels = []
        
        for strategy, df in self.strategies_data.items():
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty:
                last_round = eval_df['rounds'].max()
                last_round_data = eval_df[eval_df['rounds'] == last_round]
                
                client_losses = []
                for i in range(1, 4):
                    col = f'client_{i}_eval_loss'
                    if col in last_round_data.columns:
                        value = last_round_data[col].values
                        if len(value) > 0 and not pd.isna(value[0]):
                            client_losses.append(value[0])
                
                if client_losses:
                    final_performances.append(client_losses)
                    labels.append(strategy.upper())
        
        if final_performances:
            bp = ax3.boxplot(final_performances, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(labels))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_title('Distribuição Final de Desempenho dos Clientes', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Estratégia')
            ax3.set_ylabel('Perda Final de Validação')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Métricas de eficiência
        ax4 = axes[1, 1]
        strategies = []
        metrics_data = {
            'Convergência': [],
            'Estabilidade': [],
            'Heterogeneidade': []
        }
        
        for strategy, df in self.strategies_data.items():
            eval_df = df[df['phase'] == 'eval']
            if not eval_df.empty and 'global_eval_loss' in eval_df.columns:
                strategies.append(strategy.upper())
                
                # Convergência: melhoria total
                losses = eval_df['global_eval_loss'].values
                if len(losses) > 1:
                    convergence = (losses[0] - losses[-1]) / losses[0]
                    metrics_data['Convergência'].append(convergence)
                else:
                    metrics_data['Convergência'].append(0)
                
                # Estabilidade: desvio padrão das mudanças
                if len(losses) > 1:
                    changes = np.diff(losses)
                    stability = 1 / (1 + np.std(changes))  # Invertido para que maior = mais estável
                    metrics_data['Estabilidade'].append(stability)
                else:
                    metrics_data['Estabilidade'].append(0)
                
                # Heterogeneidade: variação média entre clientes
                client_cols = [f'client_{i}_eval_loss' for i in range(1, 4)]
                last_round = eval_df['rounds'].max()
                last_round_data = eval_df[eval_df['rounds'] == last_round]
                
                client_values = []
                for col in client_cols:
                    if col in last_round_data.columns:
                        val = last_round_data[col].values
                        if len(val) > 0 and not pd.isna(val[0]):
                            client_values.append(val[0])
                
                if len(client_values) > 1:
                    heterogeneity = 1 / (1 + np.std(client_values))  # Invertido
                    metrics_data['Heterogeneidade'].append(heterogeneity)
                else:
                    metrics_data['Heterogeneidade'].append(0)
        
        if strategies:
            x = np.arange(len(strategies))
            width = 0.25
            
            for i, (metric, values) in enumerate(metrics_data.items()):
                ax4.bar(x + i * width, values, width, label=metric, alpha=0.8)
            
            ax4.set_xlabel('Estratégia')
            ax4.set_ylabel('Score Normalizado')
            ax4.set_title('Métricas de Eficiência Comparativas', fontsize=14, fontweight='bold')
            ax4.set_xticks(x + width)
            ax4.set_xticklabels(strategies)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Análise Comparativa de Estratégias de Aprendizado Federado', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comparative_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Análise comparativa salva em: {self.results_dir / 'comparative_analysis.pdf'}")
    
    def generate_client_evolution_analysis(self):
        """Analisa a evolução individual dos clientes."""
        if not self.client_data:
            print("Nenhum dado de cliente encontrado!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Evolução do treinamento por cliente
        ax1 = axes[0, 0]
        for client_id, data in self.client_data.items():
            if 'rounds' in data and 'train_losses' in data:
                ax1.plot(data['rounds'], data['train_losses'], 
                        marker='o', label=f'Cliente {client_id}', linewidth=2)
        
        ax1.set_title('Evolução do Treinamento por Cliente', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Rodada')
        ax1.set_ylabel('Perda de Treinamento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Evolução da validação por cliente
        ax2 = axes[0, 1]
        for client_id, data in self.client_data.items():
            if 'eval_losses' in data and data['eval_losses']:
                rounds = list(range(1, len(data['eval_losses']) + 1))
                ax2.plot(rounds, data['eval_losses'], 
                        marker='s', label=f'Cliente {client_id}', linewidth=2)
        
        ax2.set_title('Evolução da Validação por Cliente', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rodada')
        ax2.set_ylabel('Perda de Validação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Mudanças nos parâmetros do modelo
        ax3 = axes[1, 0]
        for client_id, data in self.client_data.items():
            if 'model_updates' in data and data['model_updates']:
                rounds = list(range(1, len(data['model_updates']) + 1))
                ax3.plot(rounds, data['model_updates'], 
                        marker='^', label=f'Cliente {client_id}', linewidth=1.5, alpha=0.7)
        
        ax3.set_title('Magnitude das Atualizações do Modelo', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Rodada')
        ax3.set_ylabel('Mudança Média nos Parâmetros')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Tempo de convergência relativo
        ax4 = axes[1, 1]
        convergence_times = []
        client_ids = []
        
        for client_id, data in self.client_data.items():
            if 'train_losses' in data and len(data['train_losses']) > 1:
                losses = np.array(data['train_losses'])
                # Encontrar quando a perda estabiliza (mudança < 1%)
                for i in range(1, len(losses)):
                    if abs(losses[i] - losses[i-1]) / losses[i-1] < 0.01:
                        convergence_times.append(data['rounds'][i])
                        client_ids.append(f'Cliente {client_id}')
                        break
                else:
                    convergence_times.append(data['rounds'][-1])
                    client_ids.append(f'Cliente {client_id}')
        
        if convergence_times:
            colors = sns.color_palette("husl", len(client_ids))
            bars = ax4.bar(client_ids, convergence_times, color=colors, alpha=0.7)
            ax4.set_title('Tempo de Convergência por Cliente', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Cliente')
            ax4.set_ylabel('Rodada de Convergência')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, convergence_times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value}', ha='center', va='bottom')
        
        plt.suptitle('Análise de Evolução Individual dos Clientes', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'client_evolution_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Análise de evolução dos clientes salva em: {self.results_dir / 'client_evolution_analysis.pdf'}")
    
    def generate_summary_report(self):
        """Gera relatório resumido em formato texto e JSON."""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "strategies_analyzed": list(self.strategies_data.keys()),
            "clients_analyzed": list(self.client_data.keys()),
            "summary": {}
        }
        
        # Análise por estratégia
        for strategy, df in self.strategies_data.items():
            eval_df = df[df['phase'] == 'eval']
            train_df = df[df['phase'] == 'train']
            
            strategy_summary = {
                "total_rounds": int(eval_df['rounds'].max()) if not eval_df.empty else 0,
                "final_global_eval_loss": float(eval_df['global_eval_loss'].iloc[-1]) if not eval_df.empty else None,
                "final_global_train_loss": float(train_df['global_train_loss'].iloc[-1]) if not train_df.empty else None,
            }
            
            # Calcular métricas de melhoria
            if not eval_df.empty and len(eval_df) > 1:
                initial = eval_df['global_eval_loss'].iloc[0]
                final = eval_df['global_eval_loss'].iloc[-1]
                strategy_summary["improvement_percentage"] = float((initial - final) / initial * 100)
                strategy_summary["convergence_rate"] = float(np.mean(np.diff(eval_df['global_eval_loss'].values)))
            
            # Métricas por cliente
            client_metrics = {}
            for i in range(1, 4):
                col = f'client_{i}_eval_loss'
                if col in eval_df.columns:
                    client_losses = eval_df[col].dropna()
                    if not client_losses.empty:
                        client_metrics[f'client_{i}'] = {
                            "final_loss": float(client_losses.iloc[-1]),
                            "best_loss": float(client_losses.min()),
                            "worst_loss": float(client_losses.max()),
                            "std_loss": float(client_losses.std())
                        }
            
            strategy_summary["client_metrics"] = client_metrics
            report["summary"][strategy] = strategy_summary
        
        # Salvar relatório
        report_file = self.results_dir / "summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nRelatório resumido salvo em: {report_file}")
        
        # Imprimir resumo no console
        print("\n" + "="*60)
        print("RESUMO DA ANÁLISE")
        print("="*60)
        
        for strategy, summary in report["summary"].items():
            print(f"\nEstratégia: {strategy.upper()}")
            print(f"  - Rodadas totais: {summary['total_rounds']}")
            print(f"  - Perda final (validação): {summary['final_global_eval_loss']:.6f}" if summary['final_global_eval_loss'] else "  - Perda final: N/A")
            print(f"  - Melhoria: {summary.get('improvement_percentage', 0):.2f}%")
            
            if summary['client_metrics']:
                print("  - Desempenho por cliente:")
                for client, metrics in summary['client_metrics'].items():
                    print(f"    {client}: Final={metrics['final_loss']:.6f}, Melhor={metrics['best_loss']:.6f}")
    
    def run_full_analysis(self):
        """Executa análise completa."""
        print("\n" + "="*60)
        print("INICIANDO ANÁLISE COMPLETA DOS RESULTADOS")
        print("="*60 + "\n")
        
        self.load_server_metrics()
        self.load_client_metrics()
        
        if self.strategies_data:
            self.generate_comparative_analysis()
        
        if self.client_data:
            self.generate_client_evolution_analysis()
        
        if self.strategies_data or self.client_data:
            self.generate_summary_report()
        
        print("\n" + "="*60)
        print("ANÁLISE COMPLETA FINALIZADA")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Análise de Resultados FL")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Diretório com os resultados")
    args = parser.parse_args()
    
    analyzer = FLAnalyzer(args.results_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()