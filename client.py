import argparse
import flwr as fl
import torch
import json
import numpy as np
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

from utils import Net, load_data, train, test

# Argumentos da linha de comando para identificar o cliente
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="ID do Cliente (1, 2, ou 3)")
parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Endereço do servidor")
parser.add_argument(
    "--prediction-length",
    type=int,
    default=10,
    help="Define quantos passos no futuro o modelo deve prever."
)

# Checa se a GPU está disponível
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetricsTracker:
    """Classe para rastrear e salvar métricas locais do cliente."""
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.metrics = {
            "client_id": client_id,
            "rounds": [],
            "train_losses": [],
            "eval_losses": [],
            "timestamps": [],
            "model_updates": []
        }
        self.output_dir = Path(f"metrics/client_{client_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_train_metrics(self, round_num, loss):
        """Adiciona métricas de treinamento."""
        self.metrics["rounds"].append(round_num)
        self.metrics["train_losses"].append(float(loss))
        self.metrics["timestamps"].append(datetime.now().isoformat())
    
    def add_eval_metrics(self, loss):
        """Adiciona métricas de avaliação."""
        self.metrics["eval_losses"].append(float(loss))
    
    def add_model_update(self, params_diff):
        """Rastreia mudanças nos parâmetros do modelo."""
        self.metrics["model_updates"].append(float(params_diff))
    
    def save_metrics(self):
        """Salva as métricas em arquivo JSON."""
        output_file = self.output_dir / f"metrics_history.json"
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[Cliente {self.client_id}] Métricas salvas em {output_file}")
    
    def save_checkpoint(self, model, round_num):
        """Salva checkpoint do modelo."""
        checkpoint_file = self.output_dir / f"model_round_{round_num}.pt"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"[Cliente {self.client_id}] Checkpoint salvo: {checkpoint_file}")

def calculate_params_diff(old_params, new_params):
    """Calcula a diferença entre parâmetros antigos e novos."""
    if old_params is None:
        return 0.0
    
    total_diff = 0.0
    for old, new in zip(old_params, new_params):
        diff = np.mean(np.abs(old - new))
        total_diff += diff
    
    return total_diff / len(old_params)

def main():
    args = parser.parse_args()
    
    # Carrega dados específicos deste cliente
    print(f"Carregando dados para o cliente {args.client_id}...")
    print(f"Dispositivo: {DEVICE}")
    
    trainloader, testloader, num_features = load_data(
        client_id=args.client_id,
        prediction_length=args.prediction_length
    )
    
    print(f"Dados carregados: {len(trainloader)} batches de treino, {len(testloader)} batches de teste")
    
    # Instancia o modelo
    print(f"Criando modelo LSTM para prever {args.prediction_length} passos.")
    net = Net(
        input_size=num_features,
        hidden_size=50,
        output_size=args.prediction_length
    ).to(DEVICE)
    
    # Inicializa o rastreador de métricas
    metrics_tracker = MetricsTracker(args.client_id)
    
    # Implementação do cliente Flower
    class FlClient(fl.client.NumPyClient):
        def __init__(self):
            self.round_num = 0
            self.previous_params = None
        
        def get_parameters(self, config):
            """Retorna os pesos do modelo local."""
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            """Atualiza os pesos do modelo local com os do servidor."""
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            
            # Calcula a diferença entre parâmetros
            if self.previous_params is not None:
                params_diff = calculate_params_diff(self.previous_params, parameters)
                metrics_tracker.add_model_update(params_diff)
            
            self.previous_params = [p.copy() for p in parameters]

        def fit(self, parameters, config):
            """Treina o modelo localmente e retorna a perda de treinamento."""
            self.round_num += 1
            print(f"\n[Cliente {args.client_id}] === Rodada {self.round_num} ===")
            
            # Define os parâmetros recebidos do servidor
            self.set_parameters(parameters)
            
            # Treina o modelo
            print(f"[Cliente {args.client_id}] Iniciando treinamento...")
            avg_train_loss = train(net, trainloader, epochs=1, device=DEVICE)
            
            # Registra métricas
            metrics_tracker.add_train_metrics(self.round_num, avg_train_loss)
            print(f"[Cliente {args.client_id}] Perda de treinamento local: {avg_train_loss:.6f}")
            
            # Salva checkpoint a cada 5 rodadas
            if self.round_num % 5 == 0:
                metrics_tracker.save_checkpoint(net, self.round_num)
            
            # Salva métricas
            metrics_tracker.save_metrics()
            
            return self.get_parameters(config={}), len(trainloader.dataset), {
                "train_loss": avg_train_loss,
                "client_id": args.client_id,
                "round": self.round_num
            }

        def evaluate(self, parameters, config):
            """Avalia o modelo localmente."""
            self.set_parameters(parameters)
            print(f"[Cliente {args.client_id}] Avaliando modelo...")
            
            loss, num_examples = test(net, testloader, device=DEVICE)
            
            # Registra métricas de avaliação
            metrics_tracker.add_eval_metrics(loss)
            metrics_tracker.save_metrics()
            
            print(f"[Cliente {args.client_id}] Perda de validação: {loss:.6f}")
            
            return loss, num_examples, {
                "loss": loss, 
                "client_id": args.client_id,
                "round": self.round_num
            }

    # Inicia o cliente
    print(f"\n{'='*50}")
    print(f"Iniciando cliente {args.client_id}")
    print(f"Conectando ao servidor em {args.server_address}")
    print(f"Modelo: LSTM com {sum(p.numel() for p in net.parameters())} parâmetros")
    print(f"{'='*50}\n")
    
    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=FlClient().to_client(),
        )

    except Exception as e:
        print(f"[Cliente {args.client_id}] Erro: {e}")
        metrics_tracker.save_metrics()
    finally:
        print(f"\n[Cliente {args.client_id}] Finalizado. Métricas salvas em {metrics_tracker.output_dir}")

if __name__ == "__main__":
    main()