import torch
import torch.nn as nn
import torch.nn.functional as F

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


# FedPer: Federated Personalization
# Modelo híbrido com cabeça global (LSTM) e cauda local (Dense)
class FedPerLSTM(nn.Module):
    """
    Arquitetura FedPer (Federated Personalization):

    - GLOBAL HEAD (shared_lstm): Compartilhado entre todos os clientes, agregado pelo servidor
    - LOCAL TAIL (personal_*): Mantido localmente em cada cliente, NUNCA agregado

    Vantagens:
    - Compartilha padrões temporais gerais via LSTM global
    - Personaliza predição final via camadas locais
    - Cada veículo tem seu próprio modelo adaptado

    Uso:
        model_type = "fedper"
        input-size = N  # Features universais
        hidden-size = 64  # LSTM global
        personal-hidden-size = 32  # Dense local (opcional)
    """
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, dropout=0.2, personal_hidden_size=32):
        super(FedPerLSTM, self).__init__()

        # ========== CABEÇA GLOBAL (Agregada via FedAvg) ==========
        self.shared_lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ========== CAUDA LOCAL (Nunca agregada) ==========
        self.personal_fc1 = nn.Linear(hidden_size, personal_hidden_size)
        self.personal_dropout = nn.Dropout(dropout)
        self.personal_fc2 = nn.Linear(personal_hidden_size, output_size)

    def forward(self, x):
        # Cabeça global: extração de features temporais
        lstm_out, _ = self.shared_lstm(x)
        shared_features = lstm_out[:, -1, :]  # Último timestep

        # Cauda local: personalização
        out = self.personal_fc1(shared_features)
        out = F.relu(out)
        out = self.personal_dropout(out)
        out = self.personal_fc2(out)
        return out

    def get_global_params(self):
        """Retorna apenas os parâmetros da cabeça global (para agregação)"""
        return {
            name: param
            for name, param in self.named_parameters()
            if name.startswith('shared_')
        }

    def get_local_params(self):
        """Retorna apenas os parâmetros da cauda local (mantidos localmente)"""
        return {
            name: param
            for name, param in self.named_parameters()
            if name.startswith('personal_')
        }

    def set_global_params(self, params_dict):
        """Atualiza apenas os parâmetros globais"""
        state_dict = self.state_dict()
        state_dict.update(params_dict)
        self.load_state_dict(state_dict, strict=False)