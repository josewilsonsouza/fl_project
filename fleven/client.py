import torch
from flwr.app import Context, Message, ArrayRecord, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from fleven.utils import get_model, load_data, train, test, set_seed
from fleven.client_mapping import get_valid_client_id
from pathlib import Path
import json
from datetime import datetime

import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetricsTracker:
    """Classe para rastrear e salvar métricas locais do cliente."""
    
    def __init__(self, client_id, metrics_base_path=None):
        self.client_id = client_id
        
        # diretório de métricas
        if metrics_base_path:
            # Usa o caminho configurado
            self.metrics_dir = Path(metrics_base_path) / f"client_{self.client_id}"
        else:
            # Tenta usar caminho relativo ao arquivo atual
            base_dir = Path(__file__).parent.parent
            self.metrics_dir = base_dir / "metrics" / f"client_{self.client_id}"
        
        # Cria o diretório se não existir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.metrics_dir / "metrics_history.json"
        self.history = self.load_history()
        
        print(f"[Cliente {self.client_id}] Métricas serão salvas em: {self.metrics_dir.absolute()}")

    def load_history(self):
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"train": [], "eval": []}
    
    def get_next_round_number(self):
        """Retorna o próximo número de rodada baseado no histórico."""
        if not self.history["train"]:
            return 1
        return self.history["train"][-1]["round"] + 1

    def add_train_metrics(self, round_num, loss):
        existing_rounds = [entry["round"] for entry in self.history["train"]]
        if round_num not in existing_rounds:
            self.history["train"].append({
                "round": round_num, 
                "loss": loss, 
                "timestamp": datetime.now().isoformat()
            })
        else:
            for entry in self.history["train"]:
                if entry["round"] == round_num:
                    entry["loss"] = loss
                    entry["timestamp"] = datetime.now().isoformat()
                    break

    def add_eval_metrics(self, round_num, loss):
        existing_rounds = [entry["round"] for entry in self.history["eval"]]
        if round_num not in existing_rounds:
            self.history["eval"].append({
                "round": round_num, 
                "loss": loss, 
                "timestamp": datetime.now().isoformat()
            })
        else:
            for entry in self.history["eval"]:
                if entry["round"] == round_num:
                    entry["loss"] = loss
                    entry["timestamp"] = datetime.now().isoformat()
                    break

    def save_metrics(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=4)
            print(f"[Cliente {self.client_id}] Métricas salvas com sucesso")
        except Exception as e:
            print(f"[Cliente {self.client_id}] AVISO: Não foi possível salvar métricas: {e}")

    def save_checkpoint(self, net, round_num):
        try:
            model_path = self.metrics_dir / f"model_round_{round_num}.pt"
            torch.save(net.state_dict(), model_path)
            print(f"[Cliente {self.client_id}] Checkpoint salvo: {model_path}")
        except Exception as e:
            print(f"[Cliente {self.client_id}] AVISO: Não foi possível salvar checkpoint: {e}")

# Estado do ator para evitar recarregar dados a cada rodada
model_state = {
    "net": None,
    "trainloader": None,
    "testloader": None,
    "metrics_tracker": None,
    "client_id": None,
}

def initialize_client_state(client_id: int, context: Context):
    """Inicializa o estado do cliente lendo configurações do Context."""
    if model_state["client_id"] == client_id:
        return

    global_seed = int(context.run_config.get("seed", 42))
    # seed única para cada cliente
    client_seed = global_seed + client_id
    set_seed(client_seed)

    # lendo algumas configs do context
    sequence_length = int(context.run_config.get("sequence-length", 60))
    prediction_length = int(context.run_config.get("prediction-length", 10))
    batch_size = int(context.run_config.get("batch-size", 32))
    train_test_split = float(context.run_config.get("train-test-split", 0.8))

    target_column = str(context.run_config.get("target-column", "P_kW"))

    model_type = context.run_config.get("model-type", "lstm")
    num_layers = int(context.run_config.get("num-layers", 1))

    # Parâmetros para "lstm" e "mlp"
    hidden_size = int(context.run_config.get("hidden-size", 32))
    
    # Parâmetros para "lstm_dense" (o novo modelo adaptado)
    lstm_hidden_size = int(context.run_config.get("lstm-hidden-size", 32))
    dense_hidden_size = int(context.run_config.get("dense-hidden-size", 16))
    
    # Parâmetro de Dropout para "lstm" e "lstm_dense"
    dropout = float(context.run_config.get("dropout", 0.0))

    # Parâmetro para "fedper"
    personal_hidden_size = int(context.run_config.get("personal-hidden-size", 32))

    # 🔧 Lê os caminhos configurados
    data_base_path = context.run_config.get("data-base-path", None)
    metrics_base_path = context.run_config.get("metrics-base-path", None)
    
    # Carrega dados com as configurações
    trainloader, testloader, _ = load_data(
        client_id,
        sequence_length,
        prediction_length,
        batch_size,
        train_test_split,
        data_base_path=data_base_path,
        target_column=target_column,
    )

    # 🔧 Usa input-size da configuração (não dos dados) para garantir consistência com o servidor
    configured_input_size = int(context.run_config.get("input-size", 1))

    # 🔧 Dicionário de configuração do modelo
    model_config = {
        "name": model_type,
        "input_size": configured_input_size,  # Usa o configurado, não o retornado por load_data
        "output_size": prediction_length,
        "num_layers": num_layers,
        "sequence_length": sequence_length,

        # Parâmetros para "lstm" e "mlp"
        "hidden_size": hidden_size,

        # Parâmetros para "lstm_dense"
        "lstm_hidden_size": lstm_hidden_size,
        "dense_hidden_size": dense_hidden_size,

        # Parâmetro para "fedper"
        "personal_hidden_size": personal_hidden_size,

        # Parâmetro de Dropout
        "dropout": dropout
    }
        
    # Cria rede com as configurações
    net = get_model(model_config).to(DEVICE)
    
    model_state.update({
        "net": net,
        "trainloader": trainloader,
        "testloader": testloader,
        "metrics_tracker": MetricsTracker(client_id, metrics_base_path),
        "client_id": client_id,
    })
    print(f"[Cliente {client_id}] OK - Inicializado ({len(trainloader.dataset)} amostras)")

# Cria a aplicação cliente
app = ClientApp()

@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    """Treina o modelo localmente."""
    # 🔧 Mapeia partition-id para client_id válido (pula clientes removidos em ruins/)
    partition_id = int(context.node_config["partition-id"])
    client_id = get_valid_client_id(partition_id)

    initialize_client_state(client_id, context)

    net = model_state["net"]
    trainloader = model_state["trainloader"]
    metrics_tracker = model_state["metrics_tracker"]

    # Obtém o próximo número de rodada do histórico
    round_num = int(msg.content["config"].get("server-round", 0))
    
    arrays = msg.content["arrays"]

    # Para FedPer: carrega apenas parâmetros globais (shared_*)
    model_type = context.run_config.get("model-type", "lstm")
    if model_type == "fedper" and hasattr(net, 'set_global_params'):
        # FedPer: atualiza apenas cabeça global, mantém cauda local
        global_params = arrays.to_torch_state_dict()
        net.set_global_params(global_params)
    else:
        # Modelo padrão: carrega tudo
        net.load_state_dict(arrays.to_torch_state_dict())

    # Lê configurações de treino do context
    local_epochs = int(context.run_config.get("local-epochs", 1))
    learning_rate = float(context.run_config.get("learning-rate", 1e-5))
    max_grad_norm = float(context.run_config.get("max-grad-norm", 1.0))
    save_checkpoint_every = int(context.run_config.get("save-checkpoint-every", 5))

    avg_train_loss = train(
        net,
        trainloader,
        epochs=local_epochs,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        device=DEVICE
    )
    print(f"[Cliente {client_id}] Round {round_num} - Loss: {avg_train_loss:.4f}")

    metrics_tracker.add_train_metrics(round_num, avg_train_loss)
    if round_num % save_checkpoint_every == 0:
        metrics_tracker.save_checkpoint(net, round_num)
    metrics_tracker.save_metrics()

    # Para FedPer: envia apenas parâmetros globais (shared_*)
    if model_type == "fedper" and hasattr(net, 'get_global_params'):
        # Envia apenas cabeça global
        global_params_dict = net.get_global_params()
        model_record = ArrayRecord(global_params_dict)
    else:
        # Modelo padrão: envia tudo
        model_record = ArrayRecord(net.state_dict())

    metrics = MetricRecord({
        "train_loss": avg_train_loss,
        "num-examples": len(trainloader.dataset),
        "client_id": client_id,
    })
    
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    """Avalia o modelo localmente."""
    
    # Mapeia partition-id para client_id válido (pula clientes removidos em ruins/)
    partition_id = int(context.node_config["partition-id"])
    client_id = get_valid_client_id(partition_id)

    initialize_client_state(client_id, context)

    net = model_state["net"]
    testloader = model_state["testloader"]
    metrics_tracker = model_state["metrics_tracker"]

    # Usa o último número de rodada do histórico de treino
    if metrics_tracker.history["train"]:
        round_num = int(msg.content["config"].get("server-round", 0))
    else:
        round_num = 1

    arrays = msg.content["arrays"]

    # Para FedPer: carrega apenas parâmetros globais
    model_type = context.run_config.get("model-type", "lstm")
    if model_type == "fedper" and hasattr(net, 'set_global_params'):
        global_params = arrays.to_torch_state_dict()
        net.set_global_params(global_params)
    else:
        net.load_state_dict(arrays.to_torch_state_dict())

    loss, num_examples = test(net, testloader, device=DEVICE)
    print(f"[Cliente {client_id}] Round {round_num} - Eval Loss: {loss:.4f}")
    
    metrics_tracker.add_eval_metrics(round_num, loss)
    metrics_tracker.save_metrics()
    
    metrics = MetricRecord({
        "eval_loss": loss,
        "num-examples": num_examples,
        "client_id": client_id,
    })
    
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)

if __name__ == "__main__":
    print("Cliente pronto para ser executado com Flower 1.22.0")
    print("Use: flwr run . ou flower-supernode para deployment")