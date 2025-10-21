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
