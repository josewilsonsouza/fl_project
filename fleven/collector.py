"""Coletor de métricas para análise de treinamento federado."""
import numpy as np

import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Coleta e organiza métricas de treinamento e validação."""
    
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.train_metrics_by_round = []
        self.eval_metrics_by_round = []
        
        self.convergence_metrics = {
            "rounds": [],
            "loss_variance": [],
            "loss_std": [],
            "max_min_diff": []
        }
    
    @property
    def active_client_ids(self):
        """
        Inspeciona os dados coletados e retorna uma lista ordenada de 
        IDs de clientes únicos que enviaram métricas.
        """
        ids = set()
        all_metrics_by_round = self.train_metrics_by_round + self.eval_metrics_by_round
        
        for round_data in all_metrics_by_round:
            for key in round_data.keys():
                if key.startswith("client_"):
                    # Extrai o número de 'client_X_...'
                    try:
                        client_id = int(key.split("_")[1])
                        ids.add(client_id)
                    except (ValueError, IndexError):
                        # Ignora chaves que não seguem o padrão esperado
                        continue
                        
        return sorted(list(ids))
    
    def add_train_round(self, round_num, metrics):
        """Adiciona métricas de uma rodada de treinamento."""
        round_data = {
            "round": round_num,
            "global_train_loss": metrics.get("global_train_loss", None),
            "client_0_train_loss": metrics.get("client_0_train_loss", None),
            "client_1_train_loss": metrics.get("client_1_train_loss", None),
            "client_2_train_loss": metrics.get("client_2_train_loss", None),
        }
        self.train_metrics_by_round.append(round_data)
    
    def add_eval_round(self, round_num, metrics):
        """Adiciona métricas de uma rodada de avaliação."""
        round_data = {
            "round": round_num,
            "global_eval_loss": metrics.get("global_eval_loss", None),
            "client_0_eval_loss": metrics.get("client_0_eval_loss", None),
            "client_1_eval_loss": metrics.get("client_1_eval_loss", None),
            "client_2_eval_loss": metrics.get("client_2_eval_loss", None),
        }
        self.eval_metrics_by_round.append(round_data)
    
    def calculate_convergence_metrics(self, client_losses):
        """Calcula métricas de convergência entre clientes."""
        if len(client_losses) > 0:
            variance = np.var(client_losses)
            std_dev = np.std(client_losses)
            max_min_diff = max(client_losses) - min(client_losses)
            return variance, std_dev, max_min_diff
        return 0, 0, 0
    
    @property
    def train_metrics(self):
        """Retorna métricas de treino no formato para os gráficos."""
        result = {
            "rounds": [r["round"] for r in self.train_metrics_by_round],
            "global_train_loss": [r["global_train_loss"] for r in self.train_metrics_by_round if r["global_train_loss"] is not None],
            "client_0_train_loss": [r["client_0_train_loss"] for r in self.train_metrics_by_round if r["client_0_train_loss"] is not None],
            "client_1_train_loss": [r["client_1_train_loss"] for r in self.train_metrics_by_round if r["client_1_train_loss"] is not None],
            "client_2_train_loss": [r["client_2_train_loss"] for r in self.train_metrics_by_round if r["client_2_train_loss"] is not None],
        }
        return result
    
    @property
    def eval_metrics(self):
        """Retorna métricas de avaliação no formato para os gráficos."""
        result = {
            "rounds": [r["round"] for r in self.eval_metrics_by_round],
            "global_eval_loss": [r["global_eval_loss"] for r in self.eval_metrics_by_round if r["global_eval_loss"] is not None],
            "client_0_eval_loss": [r["client_0_eval_loss"] for r in self.eval_metrics_by_round if r["client_0_eval_loss"] is not None],
            "client_1_eval_loss": [r["client_1_eval_loss"] for r in self.eval_metrics_by_round if r["client_1_eval_loss"] is not None],
            "client_2_eval_loss": [r["client_2_eval_loss"] for r in self.eval_metrics_by_round if r["client_2_eval_loss"] is not None],
        }
        return result