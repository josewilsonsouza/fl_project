import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from fleven.models import LSTMNet, LSTMDenseNet, MLPNet, FedPerLSTM

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
    """
    Cria janelas deslizantes para problemas de séries temporais.

    Args:
        data: Array com features + target (target é a ÚLTIMA coluna)
        sequence_length: Tamanho da janela de entrada
        prediction_length: Quantos timesteps à frente prever

    Returns:
        xs: Sequências de features (SEM o target passado)
        ys: Sequências do target futuro
    """
    xs, ys = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        # X: Apenas features exógenas (todas as colunas EXCETO a última)
        # Não inclui energy_consumption passado - queremos prever apenas com speed, elevation, etc.
        x = data[i:(i + sequence_length), :-1]

        # Y: Apenas o target (última coluna) nos próximos prediction_length timesteps
        y = data[(i + sequence_length):(i + sequence_length + prediction_length), -1]

        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_data(client_id: int, sequence_length: int, prediction_length: int,
              batch_size: int, train_test_split: float = 0.8, data_base_path: str = None,
              target_column: str = "Energy_Consumption"):
    """
    Carrega os dados para um cliente específico do dataset eVED.

    Args:
        client_id: ID do cliente (índice da pasta client_N)
        sequence_length: Tamanho da janela de entrada
        prediction_length: Número de passos à frente para prever
        batch_size: Tamanho do batch
        train_test_split: Proporção de trips para treino (padrão: 0.8 = 80% treino, 20% teste)
        data_base_path: Caminho base para EVED_Clients (ex: "EVED_Clients")
        target_column: Coluna alvo (padrão: "Energy_Consumption")

    Returns:
        trainloader, testloader, num_features
    """

    # Define o diretório base
    if data_base_path:
        base_dir = Path(data_base_path)
    else:
        base_dir = Path(__file__).parent.parent / "data" / "EVED_Clients"

    # Agora usamos APENAS a pasta train/
    client_dir = base_dir / "train" / f"client_{client_id}"

    # Log inicial conciso
    print(f"[Cliente {client_id}] Carregando dados...")

    # Verifica se o diretório existe
    if not client_dir.exists():
        print(f"[Cliente {client_id}] ERRO: Diretorio nao encontrado")
        raise FileNotFoundError(f"Cliente {client_id} nao encontrado")

    # Carrega todos os arquivos parquet (trips)
    trip_files = list(client_dir.glob("trip_*.parquet"))

    # Ignora metadata.json se existir
    trip_files = [f for f in trip_files if 'metadata' not in f.name.lower()]

    if not trip_files:
        print(f"[Cliente {client_id}] ERRO: Nenhum arquivo de trip encontrado")
        raise FileNotFoundError(f"Cliente {client_id} sem trips")

    # Ordena os arquivos por nome para garantir ordem temporal consistente
    trip_files = sorted(trip_files, key=lambda x: x.name)

    # Split temporal: primeiras X trips para treino, últimas Y trips para teste
    split_idx = int(len(trip_files) * train_test_split)
    if split_idx == 0:
        split_idx = 1  # Garante pelo menos 1 trip para treino
    if split_idx >= len(trip_files):
        split_idx = len(trip_files) - 1  # Garante pelo menos 1 trip para teste

    train_files = trip_files[:split_idx]
    test_files = trip_files[split_idx:]

    # Carrega e concatena os dados
    train_dfs = [pd.read_parquet(f) for f in train_files]
    test_dfs = [pd.read_parquet(f) for f in test_files]

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # ========================================
    # FEATURES GLOBAIS (válidas para todos os tipos de veículos)
    # ========================================
    feature_columns = [
        'Vehicle Speed[km/h]',          # Velocidade
        'Gradient',                      # Inclinação da estrada
#        'OAT[DegC]',                    # Temperatura externa
#        'Air Conditioning Power[Watts]', # Ar condicionado
#        'Heater Power[Watts]',          # Aquecedor
        # Você pode adicionar mais se necessário:
        'Elevation Smoothed[m]'      # Elevação (pode ser útil)
    ]
    
    # Verifica se a coluna alvo existe
    if target_column not in train_df.columns:
        print(f"[Cliente {client_id}] ERRO: Coluna '{target_column}' nao encontrada")
        raise ValueError(f"Cliente {client_id} sem coluna target")

    # Verifica quais features estão disponíveis
    available_features = [col for col in feature_columns if col in train_df.columns]

    if not available_features:
        print(f"[Cliente {client_id}] ERRO: Nenhuma feature valida encontrada")
        raise ValueError(f"Cliente {client_id} sem features validas")
    
    # Ordena: features + target (target deve ser a ÚLTIMA coluna)
    all_columns = available_features + [target_column]
    
    # Seleciona e limpa os dados
    train_processed = train_df[all_columns].copy()
    test_processed = test_df[all_columns].copy()

    # Remove linhas com valores nulos ou infinitos
    train_processed = train_processed.replace([np.inf, -np.inf], np.nan).dropna()
    test_processed = test_processed.replace([np.inf, -np.inf], np.nan).dropna()

    # Verifica se há dados suficientes
    min_required_points = sequence_length + prediction_length
    if len(train_processed) < min_required_points or len(test_processed) < min_required_points:
        print(f"[Cliente {client_id}] ERRO: Dados insuficientes apos limpeza "
              f"(treino: {len(train_processed)}, teste: {len(test_processed)}, minimo: {min_required_points})")
        raise ValueError(f"Cliente {client_id} com dados insuficientes")

    # Normalização (MinMaxScaler)
    scaler = MinMaxScaler()
    scaler.fit(train_processed)

    train_scaled = scaler.transform(train_processed)
    test_scaled = scaler.transform(test_processed)

    # Criar janelas deslizantes
    X_train, y_train = create_sliding_windows(train_scaled, sequence_length, prediction_length)
    X_test, y_test = create_sliding_windows(test_scaled, sequence_length, prediction_length)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[Cliente {client_id}] ERRO: Nenhuma janela criada (pontos insuficientes)")
        raise ValueError(f"Cliente {client_id} sem janelas validas")

    # Converter para tensores PyTorch
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Criar DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_features = X_train_tensor.shape[2]

    # Log de sucesso conciso
    print(f"[Cliente {client_id}] OK - {len(trip_files)} trips, "
          f"{len(X_train)} janelas treino, {len(X_test)} janelas teste")

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

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for sequences, labels in trainloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            batch_size = sequences.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            total_loss_sum += loss.item() * batch_size
            total_samples += batch_size
        
        # Log do progresso (opcional, pode comentar se quiser menos output)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            logger.info(f"  Época {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.6f}")

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
        logger.info(f"Criando modelo LSTMNet (Simples: LSTM -> Linear)...")
        return LSTMNet(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            output_size=model_config["output_size"],
            num_layers=model_config.get("num_layers", 1),
            dropout=model_config.get("dropout", 0.0)
        )
    
    elif model_type == "lstm_dense":
        logger.info(f"Criando modelo LSTMDenseNet (Adaptado: LSTM -> Dense -> Linear)...")
        return LSTMDenseNet(
            input_size=model_config["input_size"],
            lstm_hidden_size=model_config["lstm_hidden_size"],
            dense_hidden_size=model_config["dense_hidden_size"],
            output_size=model_config["output_size"],
            num_layers=model_config.get("num_layers", 1),
            dropout=model_config.get("dropout", 0.0)
        )

    elif model_type == "mlp":
        logger.info(f"Criando modelo MLPNet...")
        mlp_input_size = model_config["sequence_length"] * model_config["input_size"]
        return MLPNet(
            input_size=mlp_input_size,
            hidden_size=model_config["hidden_size"],
            output_size=model_config["output_size"]
            )

    elif model_type == "fedper":
        logger.info(f"Criando modelo FedPerLSTM (Cabeça Global + Cauda Local)...")
        return FedPerLSTM(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            output_size=model_config["output_size"],
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.2),
            personal_hidden_size=model_config.get("personal_hidden_size", 32)
        )

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")