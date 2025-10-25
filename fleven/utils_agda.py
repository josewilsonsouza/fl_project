import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from fleven.models import LSTMNet, LSTMDenseNet, MLPNet

import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_sliding_windows(data, sequence_length, prediction_length):
    """Cria janelas deslizantes para problemas de sÃ©ries temporais."""
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
    Carrega os dados para um cliente especÃ­fico, processa e retorna DataLoaders.
    
    Args:
        client_id: ID do cliente
        sequence_length: Tamanho da janela de entrada
        prediction_length: NÃºmero de passos Ã  frente para prever
        batch_size: Tamanho do batch
        train_test_split: ProporÃ§Ã£o de dados para treino (ex: 0.8 = 80%)
        data_base_path: Caminho base para os dados (opcional),
        target_column: O nome da coluna a ser usada como alvo da previsÃ£o
    """

    if data_base_path:
        # Usa o caminho configurado
        data_dir = Path(data_base_path) / f"client_{client_id}"
        print(f"[Cliente {client_id}] Usando data_base_path configurado: {data_dir}")
    else:
        # Usa caminho relativo ao arquivo atual
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data" / "AGDA_Clients" / f"client_{client_id}"
        print(f"[Cliente {client_id}] Usando caminho relativo: {data_dir}")
    
    print(f"[Cliente {client_id}] Procurando dados em: {data_dir.absolute()}")
    
    # Verifica se o diretório existe
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DiretÃ³rio nÃ£o encontrado para o cliente {client_id}: {data_dir.absolute()}"
        )
    
    # Carrega todos os arquivos CSV
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado para o cliente {client_id} no diretÃ³rio {data_dir.absolute()}"
        )
    
    print(f"[Cliente {client_id}] Encontrados {len(csv_files)} arquivos CSV")
    all_routes_df = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(all_routes_df, ignore_index=True)

    all_columns = ['vehicle_speed', 'engine_rpm', 'P_kW']
    
    # se a coluna alvo existe
    if target_column not in all_columns:
        raise ValueError(
            f"A coluna alvo '{target_column}' nÃ£o Ã© uma das colunas vÃ¡lidas: {all_columns}"
        )
    
    # Reordena as colunas para garantir que a coluna alvo seja a ÃšLTIMA
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
            f"A divisÃ£o de dados para o cliente {client_id} resultou em um conjunto vazio."
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
    """Treina e retorna a perda mÃ©dia por amostra."""
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
    FÃ¡brica de modelos que retorna uma instÃ¢ncia de modelo com base na configuraÃ§Ã£o.
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
            lstm_hidden_size=model_config["lstm_hidden_size"],   # <-- pro pyproject tbm
            dense_hidden_size=model_config["dense_hidden_size"], # <-- pro pyproject tbm
            output_size=model_config["output_size"],
            num_layers=model_config.get("num_layers", 1),
            dropout=model_config.get("dropout", 0.0)
        )

    elif model_type == "mlp":
        print(f"Criando modelo MLPNet...")
        # Para o MLP, o tamanho da entrada inteira achatada
        mlp_input_size = model_config["sequence_length"] * model_config["input_size"]
        return MLPNet(
            input_size=mlp_input_size,
            hidden_size=model_config["hidden_size"], # Usa 'hidden_size'
            output_size=model_config["output_size"]
        )
    
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")