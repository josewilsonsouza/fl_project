#!/usr/bin/env python3
"""
Análise Comparativa de Estratégias de Agregação em Aprendizado Federado.
Gera visualizações profissionais em PDF a partir de ficheiros de métricas consolidadas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')

# Configuração de estilo para apresentação (baseado no seu trajectory_analysis.py)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Paleta de cores para estratégias e clientes
STRATEGY_COLORS = {
    'fedavg': '#1f77b4',
    'fedadam': '#ff7f0e',
    'fedyogi': '#2ca02c',
    'fedadagrad': '#d62728',
}
CLIENT_COLORS = {
    1: '#2E7D32', # Verde escuro
    2: '#1565C0', # Azul
    3: '#C62828', # Vermelho escuro
    'global': '#000000' # Preto
}
INMETRO_BLUE = '#3D538D'

class StrategyAnalyzer:
    def __init__(self, results_dir='results_comparison_120', output_dir='comparison_graphics_120'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.strategy_data = {}

    def load_all_strategy_data(self):
        """Carrega os ficheiros de métricas de todas as estratégias."""
        print("Carregando resultados das estratégias...")
        
        # Regex para extrair o nome da estratégia do nome do ficheiro
        file_pattern = re.compile(r"metrics_(.+)\.csv")
        
        for csv_file in self.results_dir.glob("metrics_*.csv"):
            match = file_pattern.match(csv_file.name)
            if match:
                strategy_name = match.group(1)
                df = pd.read_csv(csv_file)
                self.strategy_data[strategy_name] = df
                print(f"  ✓ Estratégia '{strategy_name}' carregada com {len(df)} registos.")
    
    def plot_global_convergence(self):
        """Compara a convergência da perda de validação global entre estratégias."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for name, data in self.strategy_data.items():
            ax.plot(data['rounds'], data['global_eval_loss'], 
                    marker='o', linestyle='-',
                    label=name.upper(), color=STRATEGY_COLORS.get(name, 'gray'),
                    alpha=0.8, markersize=5)

        ax.set_xlabel('Rodada de Treinamento', fontsize=16)
        ax.set_ylabel('Perda de Validação Global (MSE)', fontsize=16)
        ax.set_title('Comparativo de Convergência Global', 
                     fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        output_file = self.output_dir / 'global_convergence.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")

    def plot_training_vs_validation(self):
        """Plota a perda de treino vs. validação para cada estratégia."""
        for name, data in self.strategy_data.items():
            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(data['rounds'], data['global_train_loss'], 
                    marker='s', linestyle='--', label='Treinamento Global', 
                    color=CLIENT_COLORS.get(2))

            ax.plot(data['rounds'], data['global_eval_loss'], 
                    marker='o', linestyle='-', label='Validação Global', 
                    color=CLIENT_COLORS.get(3))

            ax.set_xlabel('Rodada de Treinamento', fontsize=16)
            ax.set_ylabel('Perda Média (MSE)', fontsize=16)
            ax.set_title(f'Treino vs. Validação - Estratégia {name.upper()}', 
                         fontsize=18, fontweight='bold', color=INMETRO_BLUE)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

            output_file = self.output_dir / f'train_vs_eval_{name}.pdf'
            plt.savefig(output_file, format='pdf')
            plt.close()
            print(f"  ✓ Salvo: {output_file}")

    def plot_client_performance(self):
        """Gera um gráfico de desempenho por cliente para cada estratégia."""
        for name, data in self.strategy_data.items():
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plota desempenho de cada cliente
            for client_id in [1, 2, 3]:
                col_name = f'client_{client_id}_eval_loss'
                if col_name in data.columns:
                    ax.plot(data['rounds'], data[col_name], marker='.', linestyle='--', 
                            label=f'Cliente {client_id}', color=CLIENT_COLORS.get(client_id))
            
            # Plota a média global para referência
            ax.plot(data['rounds'], data['global_eval_loss'], marker='o', linestyle='-',
                    label='Média Global', color=CLIENT_COLORS.get('global'), linewidth=2.5)

            ax.set_title(f'Desempenho por Cliente - Estratégia {name.upper()}', fontsize=18)
            ax.set_xlabel('Rodada', fontsize=14)
            ax.set_ylabel('Perda de Validação (MSE)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            
            output_file = self.output_dir / f'client_performance_{name}.pdf'
            plt.savefig(output_file, format='pdf')
            plt.close()
            print(f"  ✓ Salvo: {output_file}")

    def generate_all_plots(self):
        """Executa todas as funções de plotagem."""
        print("\n=== Gerando Análise Comparativa de Estratégias ===\n")
        self.load_all_strategy_data()

        if not self.strategy_data:
            print("❌ Nenhum dado de estratégia foi carregado. Verifique os ficheiros CSV na pasta de resultados.")
            return

        print("\nGerando visualizações na pasta 'comparison_graphics/'...")
        self.plot_global_convergence()
        self.plot_training_vs_validation()
        self.plot_client_performance()
        
        print(f"\n✅ Todas as visualizações foram geradas com sucesso!")
        print(f"📁 Ficheiros salvos em: {self.output_dir}/")

def main():
    analyzer = StrategyAnalyzer(results_dir='results_comparison_120', output_dir='comparison_graphics_120')
    analyzer.generate_all_plots()

if __name__ == "__main__":
    main()