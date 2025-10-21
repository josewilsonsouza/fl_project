import mlflow
from pathlib import Path
from typing import Dict, Optional
import torch

import logging

logger = logging.getLogger(__name__)

"""Utilitários para integração com MLflow."""

class MLflowTracker:
    """Gerencia logging de experimentos com MLflow."""
    
    def __init__(self, tracking_uri: str, experiment_name: str, enabled: bool = True):
        """
        Inicializa o tracker do MLflow.
        
        Args:
            tracking_uri: URI do servidor MLflow
            experiment_name: Nome do experimento
            enabled: Se True, habilita logging no MLflow
        """
        self.enabled = enabled
        
        if not self.enabled:
            print("[MLflow] Tracking desabilitado")
            return
        
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            print(f"[MLflow] Conectado ao experimento '{experiment_name}' em {tracking_uri}")
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao conectar: {e}")
            self.enabled = False
    
    def start_run(self, run_name: str, tags: Optional[Dict] = None) -> Optional[mlflow.ActiveRun]:
        """Inicia um novo run no MLflow."""
        if not self.enabled:
            return None
        
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            print(f"[MLflow] Run iniciado: {run_name} (ID: {run.info.run_id})")
            return run
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao iniciar run: {e}")
            return None
    
    def end_run(self):
        """Finaliza o run atual."""
        if not self.enabled:
            return
        
        try:
            mlflow.end_run()
            print("[MLflow] Run finalizado")
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao finalizar run: {e}")
    
    def log_params(self, params: Dict):
        """Loga parâmetros do experimento."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_params(params)
            print(f"[MLflow] {len(params)} parâmetros logados")
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao logar parâmetros: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Loga uma métrica."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao logar métrica {key}: {e}")
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Loga múltiplas métricas."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao logar métricas: {e}")
    
    def log_artifact(self, local_path: str):
        """Loga um arquivo como artifact."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifact(local_path)
            print(f"[MLflow] Artifact logado: {local_path}")
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao logar artifact: {e}")
    
    def log_artifacts(self, local_dir: str):
        """Loga um diretório inteiro como artifacts."""
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifacts(local_dir)
            print(f"[MLflow] Artifacts logados do diretório: {local_dir}")
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao logar artifacts: {e}")
    
    def log_model(self, model: torch.nn.Module, artifact_path: str = "model"):
        """Loga o modelo PyTorch."""
        if not self.enabled:
            return
        
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            print(f"[MLflow] Modelo PyTorch logado em '{artifact_path}'")
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao logar modelo: {e}")
    
    def set_tag(self, key: str, value: str):
        """Define uma tag para o run."""
        if not self.enabled:
            return
        
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao definir tag: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """Define múltiplas tags."""
        if not self.enabled:
            return
        
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            print(f"[MLflow] AVISO: Erro ao definir tags: {e}")


def get_mlflow_tracker(context) -> MLflowTracker:
    """
    Cria um MLflowTracker a partir do contexto do Flower.
    
    Args:
        context: Context do Flower contendo configurações
        
    Returns:
        Instância de MLflowTracker
    """
    mlflow_enabled = context.run_config.get("mlflow-enable", True)
    tracking_uri = context.run_config.get("mlflow-tracking-uri", "http://127.0.0.1:5000")
    experiment_name = context.run_config.get("mlflow-experiment-name", "FLEVEn-Experiments")
    
    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        enabled=mlflow_enabled
    )