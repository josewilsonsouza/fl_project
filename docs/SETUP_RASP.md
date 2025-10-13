# 🔧 Troubleshooting - Federated Learning LSTM

## 🚨 Problemas Comuns e Soluções

### 1. Erro de Dimensão (RuntimeError)

#### Erro:
```
RuntimeError: The size of tensor a (10) must match the size of tensor b (200) at non-singleton dimension 1
```

#### Causa:
O `prediction-length` no modelo não corresponde ao tamanho dos labels gerados pelos dados.

#### Solução:
```toml
# Em pyproject.toml, certifique-se de que:
[tool.flwr.app.config]
prediction-length = 10  # Deve ser consistente em todo o projeto
```

#### Como verificar:
```bash
python test_config.py
```

---

### 2. Dados Insuficientes

#### Erro:
```
ValueError: A divisão de dados para o cliente X resultou em um conjunto vazio.
```

#### Causa:
O arquivo CSV não tem linhas suficientes para criar janelas deslizantes.

#### Fórmula:
```
Mínimo de linhas = sequence_length + prediction_length
```

Exemplo com configuração padrão:
```
60 (sequence) + 10 (prediction) = 70 linhas mínimas
```

#### Solução:
1. Adicione mais dados aos CSVs
2. OU reduza `sequence-length`:
```toml
sequence-length = 30  # Reduzido de 60
```

---

### 3. Nós Insuficientes

#### Erro:
```
INFO: Waiting for at least 3 nodes to connect...
```

#### Causa:
Menos SuperNodes conectados que o mínimo especificado.

#### Solução:
1. Inicie mais SuperNodes:
```bash
# Terminal adicional
flower-supernode --insecure --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9097 \
  --node-config "partition-id=3"
```

2. OU reduza `min-nodes`:
```toml
min-nodes = 2  # Reduzido de 3
```

---

### 4. Perda Muito Alta / Não Converge

#### Sintoma:
```
Loss inicial: 15.234567
Loss final: 15.198765
Melhoria: 0.23%
```

#### Possíveis Causas:

**A. Learning Rate muito baixo**
```toml
learning-rate = 1e-4  # Aumentar de 1e-5
```

**B. Poucos rounds**
```toml
rounds = 20  # Aumentar de 5
```

**C. Modelo muito simples**
```toml
hidden-size = 100  # Aumentar de 50
num-layers = 2     # Aumentar de 1
```

**D. Batch size inadequado**
```toml
batch-size = 64  # Ajustar de 32
```

---

### 5. Overfitting

#### Sintoma:
```
Loss de treino: 0.05 (muito baixo)
Loss de validação: 2.35 (muito alto)
```

#### Soluções:

**A. Reduzir epochs locais**
```toml
local-epochs = 1  # Reduzir de 2 ou 3
```

**B. Aumentar regularização (gradient clipping)**
```toml
max-grad-norm = 0.5  # Reduzir de 1.0
```

**C. Aumentar dados de treino**
```toml
train-test-split = 0.9  # Aumentar de 0.8
```

---

### 6. Out of Memory (OOM)

#### Erro:
```
RuntimeError: CUDA out of memory
```

#### Soluções em ordem de preferência:

**A. Reduzir batch size**
```toml
batch-size = 16  # Reduzir de 32
```

**B. Reduzir tamanho do modelo**
```toml
hidden-size = 32  # Reduzir de 50
num-layers = 1    # Manter mínimo
```

**C. Reduzir sequence length**
```toml
sequence-length = 30  # Reduzir de 60
```

**D. Usar CPU**
```python
# O código já detecta automaticamente, mas força CPU:
DEVICE = torch.device("cpu")
```

---

### 7. Estratégia Não Encontrada

#### Erro:
```
KeyError: 'fedadam'
```

#### Causa:
Nome da estratégia incorreto ou não suportado.

#### Estratégias disponíveis:
```toml
strategy = "fedavg"      # ✅ FedAvg (padrão)
strategy = "fedadam"     # ✅ FedAdam
strategy = "fedyogi"     # ✅ FedYogi
strategy = "fedadagrad"  # ✅ FedAdagrad
```

---

### 8. Arquivo CSV com Colunas Faltando

#### Erro:
```
KeyError: 'vehicle_speed'
```

#### Causa:
CSV não contém todas as colunas necessárias.

#### Colunas obrigatórias:
- `vehicle_speed`
- `engine_rpm`
- `accel_x`
- `accel_y`
- `P_kW`
- `dt`

#### Verificar:
```python
import pandas as pd
df = pd.read_csv("data/client_0/route1.csv")
print(df.columns.tolist())
```

---

### 9. Problemas com Certificados TLS

#### Erro:
```
SSL: CERTIFICATE_VERIFY_FAILED
```

#### Causa:
Certificados inválidos ou path incorreto.

#### Solução temporária (apenas desenvolvimento):
```toml
[tool.flwr.federations.raspberry-deployment]
address = "127.0.0.1:9093"
insecure = true  # ⚠️ Apenas para desenvolvimento!
```

#### Solução para produção:
Veja o guia oficial: [Enable TLS connections](https://flower.ai/docs/framework/how-to-enable-tls-connections.html)

---

### 10. SuperNode Desconecta Durante Treinamento

#### Sintoma:
```
INFO: Received 2 results and 1 failures
```

#### Causas Possíveis:

**A. Timeout muito curto**

Adicione no `server.py`:
```python
result = strategy.start(
    grid=grid,
    initial_arrays=initial_arrays,
    num_rounds=num_rounds,
    timeout=7200,  # Aumentar para 2 horas
)
```

**B. Dados corrompidos em um cliente**

Execute o teste:
```bash
python test_config.py
```

**C. Memória insuficiente**

Reduza `batch-size` ou `hidden-size`.

---

### 11. Gráficos Vazios / Sem Dados

#### Sintoma:
PDFs gerados mas com mensagem "Dados insuficientes"

#### Causa:
Todos os clientes falharam ou estratégia não agregou resultados.

#### Verificar logs:
```bash
# Procure por:
INFO: aggregate_train: Received 0 results and 3 failures
```

#### Solução:
1. Verifique se todos os clientes têm dados válidos
2. Execute `python test_config.py` para cada cliente

---

### 12. Configuração Não Aplicada

#### Sintoma:
Alterou `pyproject.toml` mas valores antigos ainda são usados.

#### Causa:
Configurações sendo sobrescritas via linha de comando.

#### Verificar:
```bash
# NÃO use --run-config se quer usar pyproject.toml
flwr run .  # ✅ Correto

# Isso sobrescreve pyproject.toml:
flwr run . --run-config "rounds=10"  # ⚠️ Sobrescreve
```

---

### 13. Erro ao Importar Módulos

#### Erro:
```
ModuleNotFoundError: No module named 'utils'
```

#### Solução:
```bash
# Certifique-se de estar no diretório correto
cd /caminho/para/seu/projeto

# Reinstale o projeto
pip install -e .
```

---

### 14. Divergência entre Clientes

#### Sintoma:
```
Desvio padrão final entre clientes: 5.432100
Convergência: Baixa
```

#### Causas e Soluções:

**A. Dados muito heterogêneos**
- Isso é esperado em FL! 
- Considere usar mais rounds:
```toml
rounds = 20  # Aumentar
```

**B. Learning rate muito alto**
```toml
learning-rate = 5e-6  # Reduzir de 1e-5
```

**C. Clientes com quantidade muito diferente de dados**
- Verifique a distribuição:
```bash
python test_config.py
```

---

## 🔍 Diagnóstico Sistemático

### Passo 1: Validar Configuração
```bash
python test_config.py
```

### Passo 2: Verificar Estrutura de Dados
```bash
ls -la data/client_*/
```

### Passo 3: Testar Modelo Isoladamente
```python
from utils import Net, load_data
import torch

# Teste básico
trainloader, testloader, num_features = load_data(0, 60, 10, 32, 0.8)
net = Net(num_features, 50, 10)

for sequences, labels in trainloader:
    outputs = net(sequences)
    print(f"Input: {sequences.shape}, Output: {outputs.shape}, Labels: {labels.shape}")
    break
```

### Passo 4: Verificar Logs
```bash
# Procure por padrões:
grep "ERROR" logfile.txt
grep "RuntimeError" logfile.txt
grep "failure" logfile.txt
```

---

## 📊 Métricas de Referência

### Boas Métricas de Treinamento

```
✅ Loss de treino diminuindo: 2.5 → 0.8 → 0.3 → 0.15
✅ Loss de validação diminuindo: 2.8 → 1.2 → 0.5 → 0.25
✅ Convergência boa: Desvio padrão < 0.05
✅ Melhoria > 60% após 10 rounds
```

### Métricas Problemáticas

```
❌ Loss estagnado: 2.5 → 2.48 → 2.47 → 2.46
❌ Overfitting: Train=0.05, Val=2.5
❌ Convergência baixa: Desvio padrão > 1.0
❌ Melhoria < 10% após 10 rounds
```

---

## 🛠️ Ferramentas de Debug

### 1. Modo Verbose
```python
# Em utils.py, adicione prints:
def train(net, trainloader, epochs, learning_rate, max_grad_norm, device):
    print(f"[DEBUG] Training with LR={learning_rate}, Epochs={epochs}")
    # ...
```

### 2. Checkpoint de Debug
```python
# Salve modelo a cada round para inspecionar:
save-checkpoint-every = 1  # Salvar sempre
```

### 3. Visualizar Gradientes
```python
# Adicione ao loop de treino:
for name, param in net.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

---

## 📞 Obtendo Ajuda

### 1. Issues do GitHub
Se o problema persistir, abra uma issue em:
- [Flower GitHub Issues](https://github.com/adap/flower/issues)

### 2. Documentação Oficial
- [Flower Docs](https://flower.ai/docs/)
- [PyTorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

### 3. Fórum Flower
- [Flower Discuss](https://discuss.flower.ai/)

---

## ✅ Checklist Pré-Execução

Antes de rodar `flwr run .`, verifique:

- [ ] `python test_config.py` passou todos os testes
- [ ] Dados existem em `data/client_0/`, `data/client_1/`, etc.
- [ ] CSVs têm todas as colunas necessárias
- [ ] `prediction-length` está correto no `pyproject.toml`
- [ ] Número de SuperNodes >= `min-nodes`
- [ ] Espaço em disco suficiente para checkpoints
- [ ] Memória RAM/GPU suficiente para o batch-size escolhido

---

## 🎯 Configurações Recomendadas por Cenário

### Teste Rápido (2-3 minutos)
```toml
rounds = 3
sequence-length = 30
prediction-length = 5
batch-size = 64
local-epochs = 1
```

### Desenvolvimento (10-15 minutos)
```toml
rounds = 10
sequence-length = 60
prediction-length = 10
batch-size = 32
local-epochs = 1
```

### Produção (1-2 horas)
```toml
rounds = 50
sequence-length = 120
prediction-length = 20
batch-size = 32
local-epochs = 2
hidden-size = 100
num-layers = 2
```

### GPU Limitada
```toml
batch-size = 16
hidden-size = 32
sequence-length = 30
num-layers = 1
```

### CPU Only (mais lento)
```toml
batch-size = 8
hidden-size = 32
sequence-length = 30
local-epochs = 1
```

---

## 📝 Log de Mudanças nas Configurações

Mantenha um histórico das suas configurações:

```toml
# Em pyproject.toml, adicione comentários:
[tool.flwr.app.config]
# Experimento 1 (2025-01-08): Baseline
# rounds = 5, hidden-size = 50
# Resultado: Loss=0.45

# Experimento 2 (2025-01-09): Aumentar capacidade
# rounds = 10, hidden-size = 100
# Resultado: Loss=0.32 (melhoria de 28%)

# Configuração atual:
rounds = 10
hidden-size = 100
```

---

## 🔄 Workflow de Debugging Recomendado

1. **Execute teste de configuração**
   ```bash
   python test_config.py
   ```

2. **Rode com poucos rounds primeiro**
   ```toml
   rounds = 2
   ```

3. **Verifique logs cuidadosamente**
   - Procure por erros
   - Verifique se todos os clientes participaram
   - Confirme que agregação funcionou

4. **Se funcionou, aumente gradualmente**
   ```toml
   rounds = 5  # Depois 10, 20, etc.
   ```

5. **Monitore recursos do sistema**
   ```bash
   # Linux/Mac
   htop
   
   # Windows
   Gerenciador de Tarefas
   ```

6. **Salve configurações bem-sucedidas**
   - Faça commit do `pyproject.toml`
   - Documente no README ou comments

---

**Lembre-se**: A maioria dos problemas vem de incompatibilidade entre `prediction-length` e os dados. Execute `python test_config.py` sempre que mudar configurações!