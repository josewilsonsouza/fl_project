import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def set_seed(seed: int):
    """Seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# LSTM
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            # Dropout entre camadas LSTM empilhadas (se num_layers > 1)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.linear(last_time_step_out)
        return out
    
# LSTM -> Dropout -> Dense(ReLU) -> Dense(Output)
class LSTMDenseNet(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, dense_hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LSTMDenseNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            lstm_hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Dropout aplicado à saída da camada LSTM
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, dense_hidden_size)
        self.fc2 = nn.Linear(dense_hidden_size, output_size)

    def forward(self, x):
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # saída do último passo de tempo
        # (batch_size, lstm_hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        
        # Aplica dropout
        out = self.dropout(last_time_step_out)
        
        # Passa pelas camadas densas
        out = self.fc1(out)
        out = F.relu(out) # Aplicando ReLU como no notebook
        out = self.fc2(out)
        return out

# MLP
class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPNet, self).__init__()
        # O input será a sequência achatada
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # deforma/achata o input de (batch, sequence_length, features) para (batch, sequence_length * features)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        out = self.fc1(x_flat)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def create_sliding_windows(data, sequence_length, prediction_length):
    """Cria janelas deslizantes para problemas de séries temporais."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + prediction_length), -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_data(client_id: int, sequence_length: int, prediction_length: int, 
              batch_size: int, train_test_split: float, data_base_path: str = None,
              target_column: str = "P_kW"):
    """
    Carrega os dados para um cliente específico, processa e retorna DataLoaders.
    
    Args:
        client_id: ID do cliente
        sequence_length: Tamanho da janela de entrada
        prediction_length: Número de passos à frente para prever
        batch_size: Tamanho do batch
        train_test_split: Proporção de dados para treino (ex: 0.8 = 80%)
        data_base_path: Caminho base para os dados (opcional),
        target_column: O nome da coluna a ser usada como alvo da previsão
    """
    # 🔧 Define o diretório de dados de forma robusta
    if data_base_path:
        # Usa o caminho configurado
        data_dir = Path(data_base_path) / f"client_{client_id}"
        print(f"[Cliente {client_id}] Usando data_base_path configurado: {data_dir}")
    else:
        # Usa caminho relativo ao arquivo atual
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data" / f"client_{client_id}"
        print(f"[Cliente {client_id}] Usando caminho relativo: {data_dir}")
    
    print(f"[Cliente {client_id}] Procurando dados em: {data_dir.absolute()}")
    
    # Verifica se o diretório existe
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Diretório não encontrado para o cliente {client_id}: {data_dir.absolute()}"
        )
    
    # Carrega todos os arquivos CSV do diretório
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado para o cliente {client_id} no diretório {data_dir.absolute()}"
        )
    
    print(f"[Cliente {client_id}] Encontrados {len(csv_files)} arquivos CSV")
    all_routes_df = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(all_routes_df, ignore_index=True)

    all_columns = ['vehicle_speed', 'engine_rpm', 'P_kW']
    
    # se a coluna alvo existe
    if target_column not in all_columns:
        raise ValueError(
            f"A coluna alvo '{target_column}' não é uma das colunas válidas: {all_columns}"
        )
    
    # Reordena as colunas para garantir que a coluna alvo seja a ÚLTIMA
    feature_columns = [col for col in all_columns if col != target_column] + [target_column]
    processed_df = combined_df[feature_columns].dropna()

    split_index = int(len(processed_df) * train_test_split)
    train_df = processed_df.iloc[:split_index]
    test_df = processed_df.iloc[split_index:]

    scaler = MinMaxScaler()
    scaler.fit(train_df)

    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = create_sliding_windows(train_scaled, sequence_length, prediction_length)
    X_test, y_test = create_sliding_windows(test_scaled, sequence_length, prediction_length)
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"A divisão de dados para o cliente {client_id} resultou em um conjunto vazio."
        )

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    num_features = X_train_tensor.shape[2]
    
    print(f"[Cliente {client_id}] Dados carregados: {len(train_dataset)} treino, {len(test_dataset)} teste")

    return trainloader, testloader, num_features

def train(net, trainloader, epochs: int, learning_rate: float, 
          max_grad_norm: float, device):
    """Treina e retorna a perda média por amostra."""
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.to(device)
    net.train()

    total_loss_sum = 0.0
    total_samples = 0

    for _ in range(epochs):
        for sequences, labels in trainloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            batch_size = sequences.size(0)
            total_loss_sum += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss_sum / total_samples


def test(net, testloader, device):
    """Avalia e retorna (avg_loss_per_sample, num_examples)."""
    criterion = torch.nn.MSELoss(reduction="mean")
    net.to(device)
    net.eval()
    total_loss_sum = 0.0
    total_samples = 0
    with torch.no_grad():
        for sequences, labels in testloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = net(sequences)
            loss = criterion(outputs, labels)
            batch_size = sequences.size(0)
            total_loss_sum += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0
    avg_loss = total_loss_sum / total_samples
    return avg_loss, total_samples

def get_model(model_config: dict):
    """
    Fábrica de modelos que retorna uma instância de modelo com base na configuração.
    """
    model_type = model_config.get("name", "lstm").lower()
    
    if model_type == "lstm":
        print(f"Criando modelo LSTMNet (Simples: LSTM -> Linear)...")
        # Modelo original do projeto, agora com dropout
        return LSTMNet(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"], # Usa 'hidden_size'
            output_size=model_config["output_size"],
            num_layers=model_config.get("num_layers", 1),
            dropout=model_config.get("dropout", 0.0)
        )
    
    elif model_type == "lstm_dense":
        print(f"Criando modelo LSTMDenseNet (Adaptado: LSTM -> Dense -> Linear)...")
        # modelo adaptado de um dos notebook do DACAI
        return LSTMDenseNet(
            input_size=model_config["input_size"],
            lstm_hidden_size=model_config["lstm_hidden_size"],   # <-- Novo parâmetro pro pyproject tbm
            dense_hidden_size=model_config["dense_hidden_size"], # <-- Novo parâmetro pro pyproject tbm
            output_size=model_config["output_size"],
            num_layers=model_config.get("num_layers", 1),
            dropout=model_config.get("dropout", 0.0)
        )

    elif model_type == "mlp":
        print(f"Criando modelo MLPNet...")
        # Para o MLP, o tamanho da entrada é a sequência inteira achatada
        mlp_input_size = model_config["sequence_length"] * model_config["input_size"]
        return MLPNet(
            input_size=mlp_input_size,
            hidden_size=model_config["hidden_size"], # Usa 'hidden_size'
            output_size=model_config["output_size"]
        )
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")