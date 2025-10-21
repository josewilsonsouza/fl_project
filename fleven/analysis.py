"""Funções para análise e visualização de resultados."""
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from fleven.collector import MetricsCollector

import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_visualizations(collector: MetricsCollector, output_dir: Path):
    """Cria todas as visualizações de desempenho."""

    client_ids = collector.active_client_ids
    print(f"Analisando clientes com IDs: {client_ids}")

    # Para cores dinâmicas
    colors = sns.color_palette("husl", n_colors=len(client_ids))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Desempenho Global
    ax1 = axes[0, 0]
    rounds = collector.train_metrics["rounds"]
    if len(rounds) > 0 and len(collector.train_metrics["global_train_loss"]) > 0:
        ax1.plot(rounds, collector.train_metrics["global_train_loss"], 
                 'b-', marker='s', label='Treino Global', linewidth=2)
    if len(collector.eval_metrics["rounds"]) > 0 and len(collector.eval_metrics["global_eval_loss"]) > 0:
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
    
    for idx, client_id in enumerate(client_ids):
        key = f"client_{client_id}_train_loss"
        if key in collector.train_metrics and collector.train_metrics[key]:
            client_losses = collector.train_metrics[key]
            if len(client_losses) > 0:
                client_rounds = [r for r_idx, r in enumerate(rounds) if r_idx < len(client_losses)]
                if len(client_rounds) == len(client_losses):
                    ax2.plot(client_rounds, client_losses, 
                             marker='o', label=f'Cliente {client_id}', color=colors[idx], linewidth=1.5)
    
    if len(rounds) > 0 and len(collector.train_metrics["global_train_loss"]) > 0:
        ax2.plot(rounds, collector.train_metrics["global_train_loss"], 
                 'k--', label='Média Global', linewidth=2, alpha=0.7)
    ax2.set_title('Perda de Treinamento por Cliente', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rodada')
    ax2.set_ylabel('Perda de Treino (MSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Comparação entre Clientes (Validação)
    ax3 = axes[1, 0]
    eval_rounds = collector.eval_metrics["rounds"]

    for idx, client_id in enumerate(client_ids):
        key = f"client_{client_id}_eval_loss"
        if key in collector.eval_metrics and collector.eval_metrics[key]:
            client_losses = collector.eval_metrics[key]
            if len(client_losses) > 0:
                client_rounds = [r for r_idx, r in enumerate(eval_rounds) if r_idx < len(client_losses)]
                if len(client_rounds) == len(client_losses):
                    ax3.plot(client_rounds, client_losses, 
                             marker='s', label=f'Cliente {client_id}', color=colors[idx], linewidth=1.5)
    
    if len(eval_rounds) > 0 and len(collector.eval_metrics["global_eval_loss"]) > 0:
        ax3.plot(eval_rounds, collector.eval_metrics["global_eval_loss"], 
                 'k--', label='Média Global', linewidth=2, alpha=0.7)
    ax3.set_title('Perda de Validação por Cliente', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Rodada')
    ax3.set_ylabel('Perda de Validação (MSE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Taxa de Melhoria
    ax4 = axes[1, 1]
    if len(rounds) > 1 and len(collector.train_metrics["global_train_loss"]) > 1:
        train_improvement = np.diff(collector.train_metrics["global_train_loss"])
        ax4.plot(rounds[1:], train_improvement, 'g-', marker='v', label='Δ Treino', linewidth=1.5)
    
    if len(eval_rounds) > 1 and len(collector.eval_metrics["global_eval_loss"]) > 1:
        eval_improvement = np.diff(collector.eval_metrics["global_eval_loss"])
        ax4.plot(eval_rounds[1:], eval_improvement, 
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
    
    # Gráfico de Convergência
    _create_convergence_plot(collector, output_dir, eval_rounds)
    
    # Heatmap
    _create_heatmap(collector, output_dir, eval_rounds)
    
    print(f"Visualizações salvas em {output_dir}")


def _create_convergence_plot(collector: MetricsCollector, output_dir: Path, eval_rounds):
    """Cria gráfico de convergência."""
    client_ids = collector.active_client_ids
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for round_idx, round_num in enumerate(eval_rounds):
        client_losses = []
        for client_id in client_ids:
            key = f"client_{client_id}_eval_loss"
            if key in collector.eval_metrics and round_idx < len(collector.eval_metrics[key]):
                client_losses.append(collector.eval_metrics[key][round_idx])
        
        if len(client_losses) > 1:
            var, std, diff = collector.calculate_convergence_metrics(client_losses)
            collector.convergence_metrics["rounds"].append(round_num)
            collector.convergence_metrics["loss_variance"].append(var)
            collector.convergence_metrics["loss_std"].append(std)
            collector.convergence_metrics["max_min_diff"].append(diff)
    
    if len(collector.convergence_metrics["rounds"]) > 0:
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
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'Dados insuficientes', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(f'Análise de Convergência e Heterogeneidade - {collector.strategy_name.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'convergence_analysis_{collector.strategy_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def _create_heatmap(collector: MetricsCollector, output_dir: Path, eval_rounds):
    """Cria heatmap de performance."""
    client_ids = collector.active_client_ids
    fig, ax = plt.subplots(figsize=(12, 6))
    
    heatmap_data = []
    max_rounds = len(eval_rounds) if eval_rounds else 0
    
    has_data = False
    for client_id in client_ids:
        key = f"client_{client_id}_eval_loss"
        if key in collector.eval_metrics and collector.eval_metrics[key]:
            has_data = True
            break
    
    if has_data and max_rounds > 0:
        for client_id in client_ids:
            key = f"client_{client_id}_eval_loss"
            if key in collector.eval_metrics and collector.eval_metrics[key]:
                client_data = list(collector.eval_metrics[key])
                while len(client_data) < max_rounds:
                    client_data.append(np.nan)
                heatmap_data.append(client_data[:max_rounds])
            else:
                heatmap_data.append([np.nan] * max_rounds)
        
        heatmap_array = np.array(heatmap_data, dtype=float)
        masked_array = np.ma.masked_invalid(heatmap_array)
        
        im = ax.imshow(masked_array, aspect='auto', cmap='RdYlGn_r')
        
        ax.set_xticks(range(max_rounds))
        ax.set_xticklabels(eval_rounds[:max_rounds])
        ax.set_yticks(range(len(client_ids)))
        ax.set_yticklabels([f'Cliente {cid}' for cid in client_ids])
        
        ax.set_xlabel('Rodada', fontsize=12)
        ax.set_title(f'Mapa de Calor - Perda de Validação por Cliente - {collector.strategy_name.upper()}', 
                    fontsize=14, fontweight='bold')
        
        for i in range(len(heatmap_data)):
            for j in range(min(len(heatmap_data[i]), max_rounds)):
                if not np.isnan(heatmap_data[i][j]):
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.4f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Perda (MSE)')
    else:
        ax.text(0.5, 0.5, 'Dados insuficientes para gerar heatmap', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Mapa de Calor - Perda de Validação por Cliente - {collector.strategy_name.upper()}', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmap_performance_{collector.strategy_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_metrics(collector: MetricsCollector, output_dir: Path):
    """Salva métricas detalhadas em diferentes formatos."""
    
    # Salvar CSVs de treino e avaliação
    if collector.train_metrics_by_round:
        train_df = pd.DataFrame(collector.train_metrics_by_round)
        train_df['phase'] = 'train'
        train_csv = output_dir / f'train_metrics_{collector.strategy_name}.csv'
        train_df.to_csv(train_csv, index=False)
        print(f"Métricas de treino salvas em {train_csv}")
    
    if collector.eval_metrics_by_round:
        eval_df = pd.DataFrame(collector.eval_metrics_by_round)
        eval_df['phase'] = 'eval'
        eval_csv = output_dir / f'eval_metrics_{collector.strategy_name}.csv'
        eval_df.to_csv(eval_csv, index=False)
        print(f"Métricas de avaliação salvas em {eval_csv}")

    # Calcular estatísticas
    stats = {
        "strategy": collector.strategy_name,
        "total_rounds": len(collector.train_metrics["rounds"]),
        "final_global_train_loss": float(collector.train_metrics["global_train_loss"][-1]) if collector.train_metrics["global_train_loss"] else None,
        "final_global_eval_loss": float(collector.eval_metrics["global_eval_loss"][-1]) if collector.eval_metrics["global_eval_loss"] else None,
        "train_improvement": float((collector.train_metrics["global_train_loss"][0] - collector.train_metrics["global_train_loss"][-1])) if len(collector.train_metrics["global_train_loss"]) > 1 else 0,
        "eval_improvement": float((collector.eval_metrics["global_eval_loss"][0] - collector.eval_metrics["global_eval_loss"][-1])) if len(collector.eval_metrics["global_eval_loss"]) > 1 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    if collector.convergence_metrics["rounds"]:
        stats["convergence_metrics"] = {
            "rounds": collector.convergence_metrics["rounds"],
            "final_variance": float(collector.convergence_metrics["loss_variance"][-1]) if collector.convergence_metrics["loss_variance"] else None,
            "final_std": float(collector.convergence_metrics["loss_std"][-1]) if collector.convergence_metrics["loss_std"] else None,
            "final_max_min_diff": float(collector.convergence_metrics["max_min_diff"][-1]) if collector.convergence_metrics["max_min_diff"] else None,
        }
    
    # Salvar JSON
    json_file = output_dir / f'analysis_{collector.strategy_name}.json'
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Análise estatística salva em {json_file}")
    
    # Salvar sumário em texto
    _save_summary_text(collector, stats, output_dir)


def _save_summary_text(collector: MetricsCollector, stats: dict, output_dir: Path):
    """Salva sumário em formato texto."""
    summary_file = output_dir / f'summary_{collector.strategy_name}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"RELATÓRIO DE TREINAMENTO - {collector.strategy_name.upper()}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de rodadas: {stats['total_rounds']}\n\n")
        
        if stats['final_global_train_loss']:
            f.write("--- MÉTRICAS DE TREINO ---\n")
            f.write(f"Loss inicial: {collector.train_metrics['global_train_loss'][0]:.6f}\n")
            f.write(f"Loss final: {stats['final_global_train_loss']:.6f}\n")
            f.write(f"Melhoria: {stats['train_improvement']:.6f} ({(stats['train_improvement']/collector.train_metrics['global_train_loss'][0]*100):.2f}%)\n\n")
        
        if stats['final_global_eval_loss']:
            f.write("--- MÉTRICAS DE AVALIAÇÃO ---\n")
            f.write(f"Loss inicial: {collector.eval_metrics['global_eval_loss'][0]:.6f}\n")
            f.write(f"Loss final: {stats['final_global_eval_loss']:.6f}\n")
            f.write(f"Melhoria: {stats['eval_improvement']:.6f} ({(stats['eval_improvement']/collector.eval_metrics['global_eval_loss'][0]*100):.2f}%)\n\n")
        
        if 'convergence_metrics' in stats:
            f.write("--- MÉTRICAS DE CONVERGÊNCIA ---\n")
            f.write(f"Desvio padrão final: {stats['convergence_metrics']['final_std']:.6f}\n")
            f.write(f"Variância final: {stats['convergence_metrics']['final_variance']:.6f}\n")
            f.write(f"Diferença máx-mín final: {stats['convergence_metrics']['final_max_min_diff']:.6f}\n")
    
    print(f"Sumário salvo em {summary_file}")


def print_final_summary(collector: MetricsCollector):
    """Imprime sumário final no console."""
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