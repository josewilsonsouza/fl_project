#!/usr/bin/env python3
"""
Análise Estatística dos Dados de Trajetos para Apresentação
Gera visualizações profissionais em PDF para slides - Um gráfico por arquivo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Configuração de estilo para apresentação
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

# Cores do Inmetro e JIC
INMETRO_BLUE = '#3D538D'
JIC_GOLD = '#D2A11D'
COLORS = {
    'client_1': '#2E7D32',  # Verde escuro
    'client_2': '#1565C0',  # Azul
    'client_3': '#E65100',  # Laranja
}

class TrajectoryAnalyzer:
    def __init__(self, data_dir='data', output_dir='graphics'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.clients_data = {}
        self.statistics = {}
        
    def load_all_data(self):
        """Carrega dados de todos os clientes e trajetos"""
        print("Carregando dados dos clientes...")
        
        for client_id in [1, 2, 3]:
            client_dir = self.data_dir / f'client_{client_id}'
            if not client_dir.exists():
                print(f"  ⚠ Diretório {client_dir} não encontrado")
                continue
                
            trajectories = []
            for csv_file in sorted(client_dir.glob('*.csv')):
                df = pd.read_csv(csv_file)
                df['trajectory_id'] = csv_file.stem
                df['client_id'] = client_id
                trajectories.append(df)
                
            if trajectories:
                self.clients_data[client_id] = pd.concat(trajectories, ignore_index=True)
                print(f"  ✓ Cliente {client_id}: {len(trajectories)} trajetos carregados")
    
    def calculate_statistics(self):
        """Calcula estatísticas por cliente e trajeto"""
        features = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y', 'P_kW']
        
        for client_id, data in self.clients_data.items():
            client_stats = {
                'total_samples': len(data),
                'trajectories': data['trajectory_id'].nunique(),
                'features': {}
            }
            
            # Estatísticas por feature
            for feature in features:
                if feature in data.columns:
                    client_stats['features'][feature] = {
                        'mean': data[feature].mean(),
                        'std': data[feature].std(),
                        'min': data[feature].min(),
                        'max': data[feature].max(),
                        'q25': data[feature].quantile(0.25),
                        'q50': data[feature].quantile(0.50),
                        'q75': data[feature].quantile(0.75),
                    }
            
            # Estatísticas por trajeto
            trajectory_stats = []
            for traj_id in data['trajectory_id'].unique():
                traj_data = data[data['trajectory_id'] == traj_id]
                traj_stat = {
                    'trajectory_id': traj_id,
                    'samples': len(traj_data),
                    'duration_min': len(traj_data) * 0.1 / 60,  # assumindo 10Hz
                }
                
                for feature in features:
                    if feature in traj_data.columns:
                        traj_stat[f'{feature}_mean'] = traj_data[feature].mean()
                        traj_stat[f'{feature}_std'] = traj_data[feature].std()
                
                trajectory_stats.append(traj_stat)
            
            client_stats['trajectory_details'] = pd.DataFrame(trajectory_stats)
            self.statistics[client_id] = client_stats
    
    def plot_data_volume(self):
        """Gráfico de volume de dados por cliente"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        clients = []
        samples = []
        trajectories = []
        
        for client_id, stats in self.statistics.items():
            clients.append(f'Cliente {client_id}')
            samples.append(stats['total_samples'])
            trajectories.append(stats['trajectories'])
        
        x = np.arange(len(clients))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [s/1000 for s in samples], width, 
                      label='Amostras (×1000)', color=INMETRO_BLUE, alpha=0.8)
        
        # Adicionar valores nas barras
        for bar, val in zip(bars1, samples):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val/1000:.1f}k', ha='center', va='bottom', fontsize=12)
        
        # Eixo secundário para trajetos
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, trajectories, width, 
                       label='Trajetos', color=JIC_GOLD, alpha=0.8)
        
        for bar, val in zip(bars2, trajectories):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=12)
        
        ax.set_xlabel('Cliente', fontsize=16)
        ax.set_ylabel('Quantidade de Amostras (×1000)', fontsize=16)
        ax.set_title('Volume de Dados por Cliente', fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(clients)
        ax.grid(True, alpha=0.3)
        
        ax2.set_ylabel('Número de Trajetos', color=JIC_GOLD, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=JIC_GOLD)
        
        # Legenda combinada
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)
        
        plt.tight_layout()
        output_file = self.output_dir / 'data_volume.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_trajectory_durations(self):
        """Gráfico de duração dos trajetos"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        all_durations = []
        all_labels = []
        colors_list = []
        
        for client_id, stats in self.statistics.items():
            if 'trajectory_details' in stats:
                durations = stats['trajectory_details']['duration_min'].values
                all_durations.extend(durations)
                all_labels.extend([f'C{client_id}'] * len(durations))
                colors_list.extend([COLORS[f'client_{client_id}']] * len(durations))
        
        bars = ax.bar(range(len(all_durations)), all_durations, color=colors_list, alpha=0.7)
        
        ax.set_xlabel('Identificação do Trajeto', fontsize=16)
        ax.set_ylabel('Duração (minutos)', fontsize=16)
        ax.set_title('Duração de Cada Trajeto Coletado', fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar separadores entre clientes
        prev_label = all_labels[0]
        for i, label in enumerate(all_labels[1:], 1):
            if label != prev_label:
                ax.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.5)
                prev_label = label
        
        # Legenda customizada
        legend_elements = [mpatches.Patch(color=COLORS[f'client_{i}'], 
                                         label=f'Cliente {i}', alpha=0.7) 
                          for i in [1, 2, 3]]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        plt.tight_layout()
        output_file = self.output_dir / 'trajectory_durations.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_speed_distribution(self):
        """Distribuição de velocidade média por trajeto"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for client_id, stats in self.statistics.items():
            if 'trajectory_details' in stats:
                speeds = stats['trajectory_details']['vehicle_speed_mean'].dropna()
                ax.hist(speeds, bins=15, alpha=0.6, 
                       label=f'Cliente {client_id}',
                       color=COLORS[f'client_{client_id}'], edgecolor='black')
        
        ax.set_xlabel('Velocidade Média (km/h)', fontsize=16)
        ax.set_ylabel('Frequência', fontsize=16)
        ax.set_title('Distribuição de Velocidade Média dos Trajetos', 
                    fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'speed_distribution.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_energy_consumption(self):
        """Boxplot do consumo energético"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        consumption_data = []
        client_labels = []
        
        for client_id in [1, 2, 3]:
            if client_id in self.statistics:
                stats = self.statistics[client_id]
                if 'features' in stats and 'P_kW' in stats['features']:
                    # Coletar dados brutos para boxplot
                    data = self.clients_data[client_id]['P_kW'].dropna()
                    # Filtrar outliers extremos
                    q1 = data.quantile(0.01)
                    q99 = data.quantile(0.99)
                    filtered_data = data[(data >= q1) & (data <= q99)]
                    consumption_data.append(filtered_data.values)
                    client_labels.append(f'Cliente {client_id}')
        
        bp = ax.boxplot(consumption_data, labels=client_labels, 
                       patch_artist=True, showmeans=True)
        
        # Colorir os boxes
        for patch, client_id in zip(bp['boxes'], [1, 2, 3]):
            patch.set_facecolor(COLORS[f'client_{client_id}'])
            patch.set_alpha(0.6)
        
        # Customizar elementos do boxplot
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        plt.setp(bp['means'], marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8)
        
        ax.set_xlabel('Cliente', fontsize=16)
        ax.set_ylabel('Potência (kW)', fontsize=16)
        ax.set_title('Distribuição de Consumo Energético por Cliente', 
                    fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar legenda para a média
        ax.plot([], [], 'rD', markersize=8, label='Média')
        ax.legend(loc='upper right', fontsize=14)
        
        plt.tight_layout()
        output_file = self.output_dir / 'energy_consumption.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_feature_distribution(self, feature, feature_name):
        """Distribuição de uma variável específica usando histogramas"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Calcular bins comuns para todos os clientes
        all_data = []
        for client_id, data in self.clients_data.items():
            if feature in data.columns:
                q1 = data[feature].quantile(0.01)
                q99 = data[feature].quantile(0.99)
                filtered = data[feature][(data[feature] >= q1) & (data[feature] <= q99)]
                all_data.extend(filtered.values)
        
        # Definir bins baseados em todos os dados
        bins = np.histogram_bin_edges(all_data, bins=30)
        
        for client_id, data in self.clients_data.items():
            if feature in data.columns:
                # Remover outliers extremos
                q1 = data[feature].quantile(0.01)
                q99 = data[feature].quantile(0.99)
                filtered_data = data[feature][(data[feature] >= q1) & (data[feature] <= q99)]
                
                # Histograma
                ax.hist(filtered_data, bins=bins, 
                       label=f'Cliente {client_id}',
                       color=COLORS[f'client_{client_id}'],
                       alpha=0.5, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(feature_name, fontsize=16)
        ax.set_ylabel('Frequência', fontsize=16)
        ax.set_title(f'Distribuição: {feature_name}', 
                    fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'dist_{feature}.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_correlation_matrix(self, client_id):
        """Matriz de correlação para um cliente específico"""
        fig, ax = plt.subplots(figsize=(9, 8))
        
        features = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y', 'P_kW']
        feature_labels = ['Velocidade', 'RPM Motor', 'Acel. X', 'Acel. Y', 'Potência']
        
        if client_id in self.clients_data:
            data = self.clients_data[client_id]
            
            # Calcular matriz de correlação
            corr_data = data[features].dropna()
            corr_matrix = corr_data.corr()
            
            # Criar heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            
            # Configurar ticks
            ax.set_xticks(np.arange(len(features)))
            ax.set_yticks(np.arange(len(features)))
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
            ax.set_yticklabels(feature_labels)
            
            # Adicionar valores
            for i in range(len(features)):
                for j in range(len(features)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                                 fontsize=12, fontweight='bold')
            
            ax.set_title(f'Matriz de Correlação - Cliente {client_id}', 
                        fontsize=20, fontweight='bold', color=INMETRO_BLUE)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Coeficiente de Correlação', fontsize=14)
        
        plt.tight_layout()
        output_file = self.output_dir / f'correlation_client_{client_id}.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_temporal_pattern(self, variable, variable_name, unit):
        """Padrão temporal de uma variável usando time_sec"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for client_id, data in self.clients_data.items():
            # Pegar o primeiro trajeto com time_sec
            trajectory_id = data['trajectory_id'].unique()[0]
            traj_data = data[data['trajectory_id'] == trajectory_id].copy()
            
            # Verificar se tem coluna time_sec
            if 'time_sec' in traj_data.columns:
                # Usar time_sec diretamente
                traj_data = traj_data.sort_values('time_sec')
                # Pegar os primeiros 60 segundos de dados
                max_time = 60  # segundos
                traj_sample = traj_data[traj_data['time_sec'] <= max_time]
                
                if len(traj_sample) > 0:
                    ax.plot(traj_sample['time_sec'], traj_sample[variable], 
                           label=f'Cliente {client_id}',
                           color=COLORS[f'client_{client_id}'],
                           linewidth=2, alpha=0.8)
            else:
                # Fallback: criar índice temporal baseado no número de amostras
                traj_sample = traj_data.head(600)  # ~60 segundos a 10Hz
                time_index = np.arange(len(traj_sample)) * 0.1
                
                ax.plot(time_index, traj_sample[variable], 
                       label=f'Cliente {client_id}',
                       color=COLORS[f'client_{client_id}'],
                       linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Tempo (segundos)', fontsize=16)
        ax.set_ylabel(f'{variable_name} ({unit})', fontsize=16)
        ax.set_title(f'Padrão Temporal - {variable_name} (Primeiros 60s)', 
                    fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 60])
        
        plt.tight_layout()
        output_file = self.output_dir / f'temporal_{variable}.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
    def plot_summary_table(self):
        """Tabela resumo das estatísticas"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparar dados da tabela
        table_data = []
        headers = ['Cliente', 'Veículo', 'Trajetos', 'Amostras', 'Vel. Média\n(km/h)', 
                  'RPM Médio', 'Potência\nMédia (kW)']
        
        vehicle_names = {1: 'Spin', 2: 'Van', 3: 'Jetta'}
        
        for client_id, stats in self.statistics.items():
            row = [
                f'Cliente {client_id}',
                vehicle_names.get(client_id, '-'),
                str(stats['trajectories']),
                f"{stats['total_samples']:,}",
            ]
            
            # Adicionar médias das features
            for feature in ['vehicle_speed', 'engine_rpm', 'P_kW']:
                if feature in stats['features']:
                    row.append(f"{stats['features'][feature]['mean']:.1f}")
                else:
                    row.append('-')
            
            table_data.append(row)
        
        # Criar tabela
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.12, 0.12, 0.12, 0.15, 0.13, 0.13, 0.13])
        
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.3, 2.5)
        
        # Estilizar cabeçalho
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(INMETRO_BLUE)
            table[(0, i)].set_text_props(color='white', weight='bold')
        
        # Estilizar linhas alternadas
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                
                # Destacar coluna de cliente
                if j == 0:
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.title('Resumo Estatístico dos Dados Coletados', fontsize=22, fontweight='bold', 
                 color=INMETRO_BLUE, pad=20)
        
        plt.tight_layout()
        output_file = self.output_dir / 'summary_table.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
    
# CÓDIGO CORRIGIDO PARA SUBSTITUIR

    def plot_comparative_histogram(self, feature, feature_name):
        """Histograma comparativo lado a lado para melhor visualização"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Distribuição de {feature_name} por Cliente', 
                    fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        
        for idx, client_id in enumerate([1, 2, 3]):
            ax = axes[idx]
            if client_id in self.clients_data:
                data = self.clients_data[client_id]
                if feature in data.columns:
                    # Remover outliers
                    q1 = data[feature].quantile(0.01)
                    q99 = data[feature].quantile(0.99)
                    filtered = data[feature][(data[feature] >= q1) & (data[feature] <= q99)]
                    
                    # Histograma
                    n, bins, patches = ax.hist(filtered, bins=30, 
                                              color=COLORS[f'client_{client_id}'],
                                              alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Adicionar linha de média
                    mean_val = filtered.mean()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                             label=f'Média: {mean_val:.1f}')
                    
                    ax.set_xlabel(feature_name, fontsize=14)
                    ax.set_ylabel('Frequência', fontsize=14)
                    ax.set_title(f'Cliente {client_id}', fontsize=16)
                    ax.legend(fontsize=12)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'hist_{feature}.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
        
    # <<< AQUI ESTÁ A CORREÇÃO: ADICIONANDO A DEFINIÇÃO DA FUNÇÃO >>>
    def plot_aggregated_statistics(self):
        """Gráfico de barras com estatísticas agregadas"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo']
        x = np.arange(len(metrics))
        width = 0.25
        
        for idx, client_id in enumerate([1, 2, 3]):
            if client_id in self.statistics:
                stats = self.statistics[client_id]['features']['P_kW']
                values = [stats['mean'], stats['std'], stats['min'], stats['max']]
                
                bars = ax.bar(x + idx * width, values, width, 
                             label=f'Cliente {client_id}',
                             color=COLORS[f'client_{client_id}'], alpha=0.8)
                
                # Adicionar valores nas barras
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Métrica Estatística', fontsize=16)
        ax.set_ylabel('Potência (kW)', fontsize=16)
        ax.set_title('Estatísticas de Consumo Energético', 
                    fontsize=20, fontweight='bold', color=INMETRO_BLUE)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.output_dir / 'aggregated_statistics.pdf'
        plt.savefig(output_file, format='pdf')
        plt.close()
        print(f"  ✓ Salvo: {output_file}")
            
    def generate_all_plots(self):
        """Gera todos os gráficos individuais"""
        print("\n=== Gerando Análises para Apresentação ===\n")
        
        self.load_all_data()
        
        if not self.clients_data:
            print("❌ Nenhum dado foi carregado. Verifique o diretório 'data/'")
            return
        
        self.calculate_statistics()
        
        print(f"\nGerando visualizações na pasta '{self.output_dir}/'...")
        
        # Gráficos principais
        self.plot_data_volume()
        self.plot_trajectory_durations()
        self.plot_speed_distribution()
        self.plot_energy_consumption()
        self.plot_aggregated_statistics()
        
        # Distribuições individuais
        features_info = [
            ('vehicle_speed', 'Velocidade (km/h)'),
            ('engine_rpm', 'RPM do Motor'),
            ('accel_x', 'Aceleração X (m/s²)'),
            ('accel_y', 'Aceleração Y (m/s²)'),
            ('P_kW', 'Potência (kW)')
        ]
        
        for feature, name in features_info:
            self.plot_feature_distribution(feature, name)
        
        # Histogramas comparativos
        print("\nGerando histogramas comparativos...")
        for feature, name in features_info:
            self.plot_comparative_histogram(feature, name)
        
        # Matrizes de correlação por cliente
        for client_id in [1, 2, 3]:
            self.plot_correlation_matrix(client_id)
        
        # Padrões temporais
        temporal_vars = [
            ('vehicle_speed', 'Velocidade', 'km/h'),
            ('engine_rpm', 'RPM do Motor', 'rpm'),
            ('P_kW', 'Potência', 'kW')
        ]
        
        for var, name, unit in temporal_vars:
            self.plot_temporal_pattern(var, name, unit)
        
        # Tabela resumo
        self.plot_summary_table()
        
        print(f"\n✅ Todas as visualizações foram geradas com sucesso!")
        print(f"\n📁 Arquivos salvos em: {self.output_dir}/")
        print("\nLista de arquivos gerados:")
        for pdf_file in sorted(self.output_dir.glob('*.pdf')):
            print(f"  • {pdf_file.name}")

def main():
    analyzer = TrajectoryAnalyzer()
    analyzer.generate_all_plots()

if __name__ == "__main__":
    main()