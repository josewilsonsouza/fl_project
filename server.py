import argparse
import flwr as fl
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, ndarrays_to_parameters 
from pathlib import Path
import json
from datetime import datetime

from utils import Net

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Dicionário de estratégias disponíveis
STRATEGIES = {
    "fedavg": fl.server.strategy.FedAvg,
    "fedadam": fl.server.strategy.FedAdam,
    "fedyogi": fl.server.strategy.FedYogi,
    "fedadagrad": fl.server.strategy.FedAdagrad,
}

class MetricsCollector:
    """Coleta e organiza métricas de treinamento e validação."""
    
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.train_metrics = {
            "rounds": [],
            "global_train_loss": [],
            "client_1_train_loss": [],
            "client_2_train_loss": [],
            "client_3_train_loss": []
        }
        self.eval_metrics = {
            "rounds": [],
            "global_eval_loss": [],
            "client_1_eval_loss": [],
            "client_2_eval_loss": [],
            "client_3_eval_loss": []
        }
        self.convergence_metrics = {
            "rounds": [],
            "loss_variance": [],
            "loss_std": [],
            "max_min_diff": []
        }
        
    def add_train_round(self, round_num, metrics):
        """Adiciona métricas de uma rodada de treinamento."""
        self.train_metrics["rounds"].append(round_num)
        for key, value in metrics.items():
            if key in self.train_metrics:
                self.train_metrics[key].append(value)
    
    def add_eval_round(self, round_num, metrics):
        """Adiciona métricas de uma rodada de avaliação."""
        self.eval_metrics["rounds"].append(round_num)
        for key, value in metrics.items():
            if key in self.eval_metrics:
                self.eval_metrics[key].append(value)
    
    def calculate_convergence_metrics(self, client_losses):
        """Calcula métricas de convergência entre clientes."""
        if len(client_losses) > 0:
            variance = np.var(client_losses)
            std_dev = np.std(client_losses)
            max_min_diff = max(client_losses) - min(client_losses)
            return variance, std_dev, max_min_diff
        return 0, 0, 0

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, any]:
    """Função de agregação para métricas de avaliação."""
    individual_losses = {}
    
    for num_examples, m in metrics:
        client_id = m.get("client_id", 0)
        if client_id > 0:
            individual_losses[f"client_{client_id}_eval_loss"] = m["loss"]
    
    # Calcula média ponderada
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_loss = sum(losses) / sum(examples) if sum(examples) > 0 else 0
    
    aggregated_metrics = {"global_eval_loss": avg_loss}
    aggregated_metrics.update(individual_losses)
    
    return aggregated_metrics

def fit_metrics_aggregator(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
    """Agrega métricas de treinamento."""
    individual_losses = {}
    
    for num_examples, m in metrics:
        client_id = m.get("client_id", 0)
        if client_id > 0:
            individual_losses[f"client_{client_id}_train_loss"] = m["train_loss"]
    
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    avg_loss = sum(losses) / sum(examples) if sum(examples) > 0 else 0
    
    aggregated_metrics = {"global_train_loss": avg_loss}
    aggregated_metrics.update(individual_losses)
    
    return aggregated_metrics

def create_visualizations(collector: MetricsCollector, output_dir: Path):
    """Cria todas as visualizações de desempenho."""
    
    # 1. Gráfico de Comparação Geral (Treino vs Validação - Global e Clientes)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Desempenho Global
    ax1 = axes[0, 0]
    rounds = collector.train_metrics["rounds"]
    ax1.plot(rounds, collector.train_metrics["global_train_loss"], 
             'b-', marker='s', label='Treino Global', linewidth=2)
    ax1.plot(collector.eval_metrics["rounds"], collector.eval_metrics["global_eval_loss"], 
             'r-', marker='o', label='Validação Global', linewidth=2)
    ax1.set_title('Desempenho do Modelo Global', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rodada')
    ax1.set_ylabel('Perda (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Comparação entre Clientes (Treino)
    ax2 = axes[0, 1]
    colors = ['#2E7D32', '#1565C0', '#E65100']
    for i in range(1, 4):
        key = f"client_{i}_train_loss"
        if key in collector.train_metrics and collector.train_metrics[key]:
            ax2.plot(rounds, collector.train_metrics[key], 
                    marker='o', label=f'Cliente {i}', color=colors[i-1], linewidth=1.5)
    ax2.plot(rounds, collector.train_metrics["global_train_loss"], 
             'k--', label='Média Global', linewidth=2, alpha=0.7)
    ax2.set_title('Perda de Treinamento por Cliente', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rodada')
    ax2.set_ylabel('Perda de Treino (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Comparação entre Clientes (Validação)
    ax3 = axes[1, 0]
    for i in range(1, 4):
        key = f"client_{i}_eval_loss"
        if key in collector.eval_metrics and collector.eval_metrics[key]:
            ax3.plot(collector.eval_metrics["rounds"], collector.eval_metrics[key], 
                    marker='s', label=f'Cliente {i}', color=colors[i-1], linewidth=1.5)
    ax3.plot(collector.eval_metrics["rounds"], collector.eval_metrics["global_eval_loss"], 
             'k--', label='Média Global', linewidth=2, alpha=0.7)
    ax3.set_title('Perda de Validação por Cliente', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Rodada')
    ax3.set_ylabel('Perda de Validação (MSE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Taxa de Melhoria
    ax4 = axes[1, 1]
    if len(rounds) > 1:
        train_improvement = np.diff(collector.train_metrics["global_train_loss"])
        eval_improvement = np.diff(collector.eval_metrics["global_eval_loss"])
        ax4.plot(rounds[1:], train_improvement, 'g-', marker='v', label='Δ Treino', linewidth=1.5)
        ax4.plot(collector.eval_metrics["rounds"][1:], eval_improvement, 
                'm-', marker='^', label='Δ Validação', linewidth=1.5)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Taxa de Melhoria (Δ Perda)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Rodada')
        ax4.set_ylabel('Mudança na Perda')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Análise de Desempenho - Estratégia: {collector.strategy_name.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'performance_analysis_{collector.strategy_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de Convergência e Heterogeneidade
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calcular métricas de convergência
    for round_idx, round_num in enumerate(rounds):
        client_losses = []
        for i in range(1, 4):
            key = f"client_{i}_eval_loss"
            if key in collector.eval_metrics and round_idx < len(collector.eval_metrics[key]):
                client_losses.append(collector.eval_metrics[key][round_idx])
        
        if client_losses:
            var, std, diff = collector.calculate_convergence_metrics(client_losses)
            collector.convergence_metrics["rounds"].append(round_num)
            collector.convergence_metrics["loss_variance"].append(var)
            collector.convergence_metrics["loss_std"].append(std)
            collector.convergence_metrics["max_min_diff"].append(diff)
    
    # Subplot 1: Variância entre clientes
    ax1 = axes[0]
    ax1.plot(collector.convergence_metrics["rounds"], 
             collector.convergence_metrics["loss_variance"], 
             'b-', marker='o', linewidth=2)
    ax1.fill_between(collector.convergence_metrics["rounds"], 
                     collector.convergence_metrics["loss_variance"], 
                     alpha=0.3)
    ax1.set_title('Variância da Perda entre Clientes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rodada')
    ax1.set_ylabel('Variância')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Desvio padrão
    ax2 = axes[1]
    ax2.plot(collector.convergence_metrics["rounds"], 
             collector.convergence_metrics["loss_std"], 
             'g-', marker='s', linewidth=2)
    ax2.fill_between(collector.convergence_metrics["rounds"], 
                     collector.convergence_metrics["loss_std"], 
                     alpha=0.3, color='green')
    ax2.set_title('Desvio Padrão da Perda entre Clientes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rodada')
    ax2.set_ylabel('Desvio Padrão')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Diferença máx-mín
    ax3 = axes[2]
    ax3.plot(collector.convergence_metrics["rounds"], 
             collector.convergence_metrics["max_min_diff"], 
             'r-', marker='^', linewidth=2)
    ax3.fill_between(collector.convergence_metrics["rounds"], 
                     collector.convergence_metrics["max_min_diff"], 
                     alpha=0.3, color='red')
    ax3.set_title('Diferença Máx-Mín entre Clientes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Rodada')
    ax3.set_ylabel('Diferença')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Análise de Convergência e Heterogeneidade - {collector.strategy_name.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'convergence_analysis_{collector.strategy_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap de desempenho por cliente ao longo do tempo
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Preparar dados para heatmap
    heatmap_data = []
    for i in range(1, 4):
        key = f"client_{i}_eval_loss"
        if key in collector.eval_metrics:
            heatmap_data.append(collector.eval_metrics[key])
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, aspect='auto', cmap='RdYlGn_r')
        
        ax.set_xticks(range(len(collector.eval_metrics["rounds"])))
        ax.set_xticklabels(collector.eval_metrics["rounds"])
        ax.set_yticks(range(3))
        ax.set_yticklabels([f'Cliente {i}' for i in range(1, 4)])
        
        ax.set_xlabel('Rodada', fontsize=12)
        ax.set_title(f'Mapa de Calor - Perda de Validação por Cliente - {collector.strategy_name.upper()}', 
                    fontsize=14, fontweight='bold')
        
        # Adicionar valores nas células
        for i in range(3):
            for j in range(len(collector.eval_metrics["rounds"])):
                if j < len(heatmap_data[i]):
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.4f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Perda (MSE)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_performance_{collector.strategy_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizações salvas em {output_dir}")

def save_detailed_metrics(collector: MetricsCollector, output_dir: Path):
    """Salva métricas detalhadas em diferentes formatos."""
    
    # 1. CSV com todas as métricas
    all_metrics = pd.DataFrame()
    
    # Adicionar métricas de treino
    if collector.train_metrics["rounds"]:
        train_df = pd.DataFrame(collector.train_metrics)
        train_df['phase'] = 'train'
        all_metrics = pd.concat([all_metrics, train_df], ignore_index=True)
    
    # Adicionar métricas de validação
    if collector.eval_metrics["rounds"]:
        eval_df = pd.DataFrame(collector.eval_metrics)
        eval_df['phase'] = 'eval'
        all_metrics = pd.concat([all_metrics, eval_df], ignore_index=True)
    
    # Salvar CSV detalhado
    csv_file = output_dir / f'detailed_metrics_{collector.strategy_name}.csv'
    all_metrics.to_csv(csv_file, index=False)
    print(f"Métricas detalhadas salvas em {csv_file}")
    
    # 2. JSON com análise estatística
    stats = {
        "strategy": collector.strategy_name,
        "total_rounds": len(collector.train_metrics["rounds"]),
        "final_global_train_loss": collector.train_metrics["global_train_loss"][-1] if collector.train_metrics["global_train_loss"] else None,
        "final_global_eval_loss": collector.eval_metrics["global_eval_loss"][-1] if collector.eval_metrics["global_eval_loss"] else None,
        "train_improvement": (collector.train_metrics["global_train_loss"][0] - collector.train_metrics["global_train_loss"][-1]) if len(collector.train_metrics["global_train_loss"]) > 1 else 0,
        "eval_improvement": (collector.eval_metrics["global_eval_loss"][0] - collector.eval_metrics["global_eval_loss"][-1]) if len(collector.eval_metrics["global_eval_loss"]) > 1 else 0,
        "convergence_metrics": collector.convergence_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    json_file = output_dir / f'analysis_{collector.strategy_name}.json'
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Análise estatística salva em {json_file}")

def main():
    parser = argparse.ArgumentParser(description="Flower Server Enhanced")
    parser.add_argument(
        "--strategy", type=str, choices=list(STRATEGIES.keys()), default="fedavg",
        help="Estratégia de agregação a ser usada."
    )
    parser.add_argument(
        "--rounds", type=int, default=15, help="Número de rodadas de FL."
    )
    parser.add_argument(
        "--min-clients", type=int, default=3, help="Número mínimo de clientes."
    )
    parser.add_argument(
    "--prediction-length", type=int, default=10, help="Tamanho da previsão."
    )
    args = parser.parse_args()

    print(f"Tamanho da Previsão: {args.prediction_length}")
    # Cria uma instância do modelo no servidor para obter os parâmetros iniciais
# Os valores de input/hidden devem ser os mesmos do cliente
    net = Net(input_size=5, hidden_size=50, output_size=args.prediction_length)
    initial_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]


    print(f"\n{'='*60}")
    print(f"SERVIDOR DE APRENDIZADO FEDERADO")
    print(f"{'='*60}")
    print(f"Estratégia: {args.strategy.upper()}")
    print(f"Rodadas: {args.rounds}")
    print(f"Clientes mínimos: {args.min_clients}")
    print(f"{'='*60}\n")

    # Cria diretórios de saída
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Inicializa coletor de métricas
    collector = MetricsCollector(args.strategy)
    
    # Função customizada de agregação que também coleta métricas
    def enhanced_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, any]:
        result = weighted_average(metrics)
        # Coletar métricas para visualização
        round_num = max([m.get("round", 0) for _, m in metrics])
        if round_num > 0:
            collector.add_eval_round(round_num, result)
        return result
    
    def enhanced_fit_aggregator(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        result = fit_metrics_aggregator(metrics)
        # Coletar métricas para visualização
        round_num = max([m.get("round", 0) for _, m in metrics])
        if round_num > 0:
            collector.add_train_round(round_num, result)
        return result

    # Cria a estratégia com as funções de agregação aprimoradas
    strategy_params = {
    "min_fit_clients": args.min_clients,
    "min_available_clients": args.min_clients,
    "evaluate_metrics_aggregation_fn": enhanced_weighted_average,
    "fit_metrics_aggregation_fn": enhanced_fit_aggregator,
    }

    if args.strategy != "fedavg":
        strategy_params["initial_parameters"] = ndarrays_to_parameters(initial_parameters)

    strategy = STRATEGIES[args.strategy](**strategy_params)


    # Inicia o servidor
    print("Iniciando servidor FL...")
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO - GERANDO ANÁLISES")
    print("="*60)
    
    # Criar visualizações
    create_visualizations(collector, output_dir)
    
    # Salvar métricas detalhadas
    save_detailed_metrics(collector, output_dir)
    
    # Análise final
    print("\n" + "="*60)
    print("RESUMO DO TREINAMENTO")
    print("="*60)
    
    if collector.train_metrics["global_train_loss"]:
        initial_loss = collector.train_metrics["global_train_loss"][0]
        final_loss = collector.train_metrics["global_train_loss"][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"Perda inicial de treino: {initial_loss:.6f}")
        print(f"Perda final de treino: {final_loss:.6f}")
        print(f"Melhoria no treino: {improvement:.2f}%")
    
    if collector.eval_metrics["global_eval_loss"]:
        initial_eval = collector.eval_metrics["global_eval_loss"][0]
        final_eval = collector.eval_metrics["global_eval_loss"][-1]
        eval_improvement = ((initial_eval - final_eval) / initial_eval) * 100
        
        print(f"\nPerda inicial de validação: {initial_eval:.6f}")
        print(f"Perda final de validação: {final_eval:.6f}")
        print(f"Melhoria na validação: {eval_improvement:.2f}%")
    
    # Análise de convergência
    if collector.convergence_metrics["loss_std"]:
        final_std = collector.convergence_metrics["loss_std"][-1]
        print(f"\nDesvio padrão final entre clientes: {final_std:.6f}")
        print(f"Convergência: {'Boa' if final_std < 0.01 else 'Moderada' if final_std < 0.05 else 'Baixa'}")
    
    print("\n" + "="*60)
    print(f"Resultados salvos em: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()