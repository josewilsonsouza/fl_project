import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os

# Definição do Modelo LSTM com PyTorch
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Usamos apenas a saída do último passo de tempo para a previsão
        last_time_step_out = lstm_out[:, -1, :]
        out = self.linear(last_time_step_out)
        return out

def create_sliding_windows(data, sequence_length, prediction_length):
    """Cria janelas deslizantes para problemas de séries temporais."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + prediction_length), -1] # A última coluna é 'P_kW'
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_data(client_id: int, sequence_length=60, prediction_length=30, batch_size=32):
    """
    Carrega os dados para um cliente específico, processa e retorna DataLoaders.
    NOVA VERSÃO: Concatena todos os percursos e divide em 80% para treino e 20% para teste.
    """
    data_dir = Path(f"data/client_{client_id}")
    
    # 1. Carrega todos os percursos em uma lista de DataFrames
    all_routes_df = [pd.read_csv(f) for f in data_dir.glob("*.csv")]

    if not all_routes_df:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado para o cliente {client_id} no diretório {data_dir}")

    # 2. Concatena TODOS os percursos em um único DataFrame
    combined_df = pd.concat(all_routes_df, ignore_index=True)

    feature_columns = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y', 'P_kW', 'dt']
    processed_df = combined_df[feature_columns].dropna()

    # 3. Divide o DataFrame combinado em 80% para treino e 20% para teste
    split_index = int(len(processed_df) * 0.8)
    train_df = processed_df.iloc[:split_index]
    test_df = processed_df.iloc[split_index:]

    # 4. AJUSTA O SCALER APENAS NOS DADOS DE TREINO (ESSENCIAL!)
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # 5. TRANSFORMA AMBOS OS CONJUNTOS COM O SCALER AJUSTADO
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)

    # 6. CRIA JANELAS SEPARADAMENTE PARA TREINO E TESTE
    X_train, y_train = create_sliding_windows(train_scaled, sequence_length, prediction_length)
    X_test, y_test = create_sliding_windows(test_scaled, sequence_length, prediction_length)
    
    # Garante que os conjuntos não sejam vazios
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"A divisão de dados para o cliente {client_id} resultou em um conjunto de treino ou teste vazio. Verifique a quantidade de dados.")

    # Conversão para tensores PyTorch
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Criação de DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    num_features = X_train_tensor.shape[2]

    return trainloader, testloader, num_features


def train(net, trainloader, epochs, device):
    """Treina e retorna a perda média por amostra (float)."""
    criterion = torch.nn.MSELoss(reduction="mean")  # perda média por saída
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    net.to(device)
    net.train()

    total_loss_sum = 0.0
    total_samples = 0

    for _ in range(epochs):
        for sequences, labels in trainloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(sequences)
            loss = criterion(outputs, labels)  # média por amostra no batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = sequences.size(0)
            total_loss_sum += loss.item() * batch_size  # soma por amostra
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss_sum / total_samples  # média por amostra

# --- substitua a função test ---
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
