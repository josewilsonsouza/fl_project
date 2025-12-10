# Guia: FedPer (Federated Personalization) no FLEVEn

## 🎯 O que é FedPer?

**FedPer** é uma estratégia de aprendizado federado personalizado que divide o modelo em duas partes:

```
┌─────────────────────────────────────────┐
│   CABEÇA GLOBAL (Shared Head)          │
│   - LSTM compartilhado entre todos      │
│   - Agregado pelo servidor via FedAvg   │
│   - Aprende padrões temporais gerais    │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│   CAUDA LOCAL (Personal Tail)           │
│   - Dense layers específicas do veículo │
│   - NUNCA enviadas ao servidor          │
│   - Aprende padrões individuais         │
└─────────────────────────────────────────┘
```

---

## ✅ Vantagens para o Caso eVED

### **1. Compartilhamento de Conhecimento Global**
- Todos os veículos (ICE, HEV, PHEV, EV) contribuem para melhorar a **extração de features temporais** (LSTM)
- Padrões gerais de velocidade → consumo são compartilhados

### **2. Personalização Local**
- Cada veículo tem sua **própria cabeça de predição** (Dense layers)
- ICE aprende seu próprio mapeamento velocidade → consumo
- EV aprende seu próprio (diferente do ICE, mais eficiente)
- HEV/PHEV aprendem padrões híbridos únicos

### **3. Simplicidade**
- **Mesmas features** para todos os veículos (universais)
- Não precisa de clustering por tipo
- Não precisa de feature masking
- Implementação mais fácil que Clustered FL

### **4. Eficiência de Comunicação**
- Apenas **~60-70% dos parâmetros** são enviados ao servidor
- Cauda local fica sempre no dispositivo

---

## 🏗️ Arquitetura do Modelo FedPerLSTM

```python
class FedPerLSTM(nn.Module):
    # ========== CABEÇA GLOBAL (Agregada) ==========
    shared_lstm: LSTM(input_size → hidden_size)
    # Parâmetros: ~200-500 KB
    # Enviados ao servidor: SIM ✓

    # ========== CAUDA LOCAL (Nunca agregada) ==========
    personal_fc1: Linear(hidden_size → personal_hidden_size)
    personal_fc2: Linear(personal_hidden_size → output_size)
    # Parâmetros: ~50-100 KB
    # Enviados ao servidor: NÃO ✗
```

### **Fluxo de Dados:**
```
Input: [batch, sequence_length, input_size]
   ↓
shared_lstm (GLOBAL)
   ↓
Features: [batch, hidden_size]
   ↓
personal_fc1 (LOCAL)
   ↓
ReLU + Dropout
   ↓
personal_fc2 (LOCAL)
   ↓
Output: [batch, prediction_length]
```

---

## 📝 Configuração

### **1. Configurar `pyproject.toml`**

```toml
[tool.flwr.app.config]

# ========== MODELO ==========
model-type = "fedper"  # ← Usar FedPer

# Features de entrada (UNIVERSAIS - funcionam para todos os tipos)
input-size = 4
# Features:
# - Vehicle Speed[km/h]
# - Speed Limit with Direction[km/h]
# - Elevation Smoothed[m]
# - Gradient

# ========== CABEÇA GLOBAL (LSTM) ==========
hidden-size = 64       # Tamanho da camada LSTM compartilhada
num-layers = 2         # Número de camadas LSTM
dropout = 0.2          # Dropout

# ========== CAUDA LOCAL (Dense) ==========
personal-hidden-size = 32  # Tamanho da camada densa local

# ========== SÉRIES TEMPORAIS ==========
sequence-length = 50
prediction-length = 10
target-column = "Energy_Consumption"

# ========== TREINAMENTO ==========
batch-size = 64
learning-rate = 1e-4
local-epochs = 3       # Mais epochs para personalização local
max-grad-norm = 1.0

# ========== FEDERAÇÃO ==========
strategy = "fedavg"    # FedAvg é suficiente (agrega apenas shared_lstm)
rounds = 10
min-nodes = 3

[tool.flwr.federations.local-simulation]
options.num-supernodes = 20
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.0
```

### **2. Atualizar Features em `fleven/utils.py`**

Adicione mais features universais para melhorar performance:

```python
# fleven/utils.py, linha ~117
feature_columns = [
    'Vehicle Speed[km/h]',              # Essencial
    'Speed Limit with Direction[km/h]', # Contexto
    'Elevation Smoothed[m]',            # Topografia
    'Gradient',                         # Inclinação
]
```

**Opcional**: Se quiser usar mais features (requer verificar disponibilidade):
```python
feature_columns = [
    'Vehicle Speed[km/h]',
    'Speed Limit with Direction[km/h]',
    'Elevation Smoothed[m]',
    'Gradient',
    'Engine RPM[RPM]',      # Precisa tratar missing para EVs
    'MAF[g/sec]',           # Precisa tratar missing para EVs
]
```

---

## 🚀 Como Executar

### **Teste Rápido (3 clientes)**

```bash
flwr run . --run-config "model-type=fedper rounds=3 min-nodes=3" local-simulation
```

### **Teste Médio (20 clientes, configuração padrão)**

```bash
flwr run . local-simulation
```

### **Escala Completa (227 clientes)**

Primeiro, atualize o `pyproject.toml`:
```toml
options.num-supernodes = 227
```

Depois execute:
```bash
flwr run . --run-config "model-type=fedper rounds=10" local-simulation
```

---

## 📊 Comparação de Estratégias

Para avaliar se FedPer é melhor que FedAvg padrão:

### **1. Baseline FedAvg (apenas Vehicle Speed)**
```bash
flwr run . --run-config "model-type=lstm input-size=1 rounds=5" local-simulation
```
- **Features**: `Vehicle Speed[km/h]`
- **Modelo**: LSTM global único
- **Personalização**: Nenhuma

### **2. FedPer (features universais)**
```bash
flwr run . --run-config "model-type=fedper input-size=4 rounds=5" local-simulation
```
- **Features**: `Vehicle Speed, Speed Limit, Elevation, Gradient`
- **Modelo**: LSTM global + Dense local
- **Personalização**: Alta (cada veículo tem sua cauda)

### **3. Comparar Resultados**

Verifique em `results/`:
- `train_metrics_{strategy}.csv`: Compare train loss
- `eval_metrics_{strategy}.csv`: Compare eval loss
- `performance_analysis_{strategy}.pdf`: Visualize convergência

**Espera-se que FedPer tenha:**
- ✅ **Menor eval loss** (personalização melhora predição)
- ✅ **Convergência mais rápida** (LSTM global aprende padrões gerais rapidamente)
- ✅ **Menor variância entre clientes** (cada um se adapta localmente)

---

## 🔍 Monitoramento e Debug

### **Logs Importantes**

Durante o treinamento, você verá:

```
[Cliente 0] FedPer: Cabeça global atualizada, cauda local mantida
[Cliente 0] Perda de treino: 0.023456
[Cliente 0] FedPer: Enviando apenas 8 parâmetros globais
```

**Verificações:**
- ✓ "Cabeça global atualizada" → Cliente recebeu LSTM do servidor
- ✓ "cauda local mantida" → Dense layers não foram sobrescritas
- ✓ "Enviando apenas N parâmetros globais" → Apenas LSTM é enviado

### **Métricas Locais**

Cada cliente salva métricas em `metrics/client_N/metrics_history.json`:

```json
{
  "train": [
    {"round": 1, "loss": 0.045, "timestamp": "..."},
    {"round": 2, "loss": 0.032, "timestamp": "..."},
    {"round": 3, "loss": 0.025, "timestamp": "..."}
  ],
  "eval": [
    {"round": 1, "loss": 0.048, "timestamp": "..."},
    {"round": 2, "loss": 0.035, "timestamp": "..."}
  ]
}
```

**Padrão Esperado (FedPer):**
- Train loss deve **convergir rapidamente** (shared LSTM aprende padrões gerais)
- Eval loss deve ser **menor que FedAvg padrão** (personalização local)
- Variância entre clientes deve **diminuir ao longo das rodadas**

---

## ⚙️ Hiperparâmetros

### **Cabeça Global (Shared LSTM)**

| Parâmetro | Valor Padrão | Ajustar quando... |
|-----------|--------------|-------------------|
| `hidden-size` | 64 | ↑ 128 se dataset grande, ↓ 32 se poucos dados |
| `num-layers` | 2 | ↑ 3 para padrões mais complexos |
| `dropout` | 0.2 | ↑ 0.3 se overfitting, ↓ 0.1 se underfitting |

### **Cauda Local (Personal Dense)**

| Parâmetro | Valor Padrão | Ajustar quando... |
|-----------|--------------|-------------------|
| `personal-hidden-size` | 32 | ↑ 64 para mais personalização, ↓ 16 para simplificar |

### **Treinamento**

| Parâmetro | Valor Padrão | Ajustar quando... |
|-----------|--------------|-------------------|
| `local-epochs` | 3 | ↑ 5 para mais personalização local |
| `learning-rate` | 1e-4 | ↓ 1e-5 se instável, ↑ 1e-3 para convergência rápida |

---

## 🎓 Quando Usar FedPer?

### ✅ **Use FedPer quando:**

1. **Heterogeneidade de clientes** (diferentes tipos de veículos)
2. **Features universais disponíveis** (velocidade, elevação, etc.)
3. **Quer simplicidade** (sem clustering, sem feature masking)
4. **Precisa de personalização** (cada veículo tem padrão único)

### ❌ **NÃO use FedPer quando:**

1. **Features são muito específicas** (ex: só tem Engine RPM, sem features universais)
2. **Poucos clientes** (< 10) - não compensa a complexidade
3. **Todos os clientes são idênticos** - FedAvg padrão é suficiente

---

## 📈 Próximos Passos

### **Fase 1**: Validar FedPer funciona
```bash
flwr run . --run-config "model-type=fedper rounds=3 min-nodes=3"
```

### **Fase 2**: Comparar com FedAvg
```bash
# FedAvg
flwr run . --run-config "model-type=lstm rounds=5"
# FedPer
flwr run . --run-config "model-type=fedper rounds=5"
```

### **Fase 3**: Adicionar mais features
Edite `fleven/utils.py` para incluir mais features universais (Speed Limit, Elevation, etc.)

### **Fase 4**: Escalar
```bash
flwr run . --run-config "model-type=fedper rounds=10" local-simulation
# Com num-supernodes=227 em pyproject.toml
```

---

## 🔬 Alternativas Futuras

Se FedPer não for suficiente, considere:

1. **Clustered FL**: Agrupe por tipo de veículo (ICE/HEV/PHEV/EV), use features específicas
2. **Per-FedAvg**: Meta-learning para adaptação rápida
3. **Multi-Task Learning**: Uma tarefa por tipo de veículo

---

**Resumo:** FedPer é **ideal para o caso eVED** porque permite compartilhar conhecimento global (padrões de velocidade → consumo) enquanto cada veículo se especializa localmente (combustão vs elétrico vs híbrido). É mais simples que Clustered FL e mais eficaz que FedAvg padrão.

Quer testar agora? Execute: `flwr run . --run-config "model-type=fedper rounds=3"`
