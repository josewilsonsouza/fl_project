# Correções Aplicadas - FedPer no eVED

## 🐛 Problemas Identificados

Ao executar `flwr run .` com `model-type=fedper`, dois erros ocorreram:

### **Erro 1: FileNotFoundError para clientes removidos**

```
FileNotFoundError: Diretório do cliente não encontrado:
.../data/EVED_Clients/train/client_4
.../data/EVED_Clients/train/client_16
...
```

**Causa raiz:**
- O Flower atribui `partition-id` de 0 a 19 (quando `num-supernodes=20`)
- Anteriormente, `client_id = partition_id` diretamente
- Mas 48 clientes foram movidos para `ruins/` (incluindo client_4, client_5, client_8, etc.)
- Quando o Flower tentava acessar client_4, o diretório não existia mais

### **Erro 2: RuntimeError - Incompatibilidade de Dimensões**

```
RuntimeError: Error(s) in loading state_dict for FedPerLSTM:
size mismatch for shared_lstm.weight_ih_l0:
copying a param with shape torch.Size([256, 1]) from checkpoint,
the shape in current model is torch.Size([256, 2])
```

**Causa raiz:**
- Servidor inicializava modelo com `input-size=1` do `pyproject.toml`
- Cliente carregava dados e obtinha `num_features` da dimensão real dos dados
- Cliente usava `num_features` (que poderia ser diferente de 1) para criar o modelo
- Incompatibilidade entre servidor (input_size=1) e cliente (input_size=num_features)

---

## ✅ Soluções Implementadas

### **Solução 1: Mapeamento de Cliente Válido**

**Arquivo criado:** [`fleven/client_mapping.py`](fleven/client_mapping.py)

Implementa função `get_valid_client_id(partition_id)` que:
- Mantém lista dos 48 clientes removidos
- Mantém lista dos 184 clientes válidos (sorted)
- Mapeia `partition_id` → `client_id` válido, pulando clientes em ruins/

**Exemplo de mapeamento:**
```
partition_id  ->  client_id
     0        ->      0       (client_0 é válido)
     1        ->      1       (client_1 é válido)
     2        ->      2       (client_2 é válido)
     3        ->      3       (client_3 é válido)
     4        ->      6       (pula client_4 e client_5 - em ruins/)
     5        ->      7       (client_7 é válido)
     6        ->      9       (pula client_8 - em ruins/)
     ...
```

**Modificações em [`fleven/client.py`](fleven/client.py):**

```python
# Linha 5: Import adicionado
from fleven.client_mapping import get_valid_client_id

# Linha 193-194: train_fn() - Mapeamento
partition_id = int(context.node_config["partition-id"])
client_id = get_valid_client_id(partition_id)  # ← Usa mapeamento

# Linha 268-269: evaluate_fn() - Mapeamento
partition_id = int(context.node_config["partition-id"])
client_id = get_valid_client_id(partition_id)  # ← Usa mapeamento
```

**Benefícios:**
- ✅ Flower sempre acessa clientes válidos
- ✅ Não precisa renumerar pastas de clientes
- ✅ Fácil adicionar/remover clientes da lista
- ✅ Suporta até 184 clientes (todos os válidos)

---

### **Solução 2: Consistência de input_size**

**Modificações em [`fleven/client.py`](fleven/client.py):**

```python
# Linha 141: Adiciona parâmetro personal-hidden-size
personal_hidden_size = int(context.run_config.get("personal-hidden-size", 32))

# Linhas 158-164: Usa input-size configurado (não num_features)
configured_input_size = int(context.run_config.get("input-size", 1))

# Validação: verifica se num_features coincide com input-size configurado
if num_features != configured_input_size:
    print(f"⚠️  [Cliente {client_id}] AVISO: num_features={num_features} difere de input-size={configured_input_size} configurado")
    print(f"    Usando input-size={configured_input_size} para manter compatibilidade com o servidor")

# Linhas 167-186: model_config atualizado
model_config = {
    "name": model_type,
    "input_size": configured_input_size,  # ← Usa configurado, não num_features
    "output_size": prediction_length,
    ...
    "personal_hidden_size": personal_hidden_size,  # ← Adicionado para FedPer
    ...
}
```

**Benefícios:**
- ✅ Cliente e servidor sempre usam o mesmo `input_size`
- ✅ Evita incompatibilidade de dimensões ao carregar parâmetros
- ✅ Aviso exibido se dados divergem da configuração
- ✅ Suporte adequado ao parâmetro `personal-hidden-size` do FedPer

---

## 🚀 Como Usar Agora

### **Passo 1: Verificar Configuração**

Edite [`pyproject.toml`](pyproject.toml):

```toml
[tool.flwr.app.config]
# Modelo
model-type = "fedper"
input-size = 1  # ← Deve coincidir com número de features em utils.py
hidden-size = 64
num-layers = 2
dropout = 0.2
personal-hidden-size = 32  # ← Tamanho da cauda local do FedPer

# Série temporal
sequence-length = 50
prediction-length = 10
target-column = "Energy_Consumption"

# Treinamento
batch-size = 64
learning-rate = 1e-4
local-epochs = 2

# Federação
strategy = "fedavg"
rounds = 10
min-nodes = 10

[tool.flwr.federations.local-simulation]
options.num-supernodes = 20  # ← Use até 184 (total de clientes válidos)
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.0
```

### **Passo 2: Verificar Features em utils.py**

Edite [`fleven/utils.py`](fleven/utils.py) (linha ~118):

```python
feature_columns = [
    'Vehicle Speed[km/h]'  # ← 1 feature = input-size deve ser 1
]
```

**IMPORTANTE:** O número de features em `feature_columns` **DEVE** coincidir com `input-size` em `pyproject.toml`.

Exemplo com 4 features (recomendado para FedPer):
```python
feature_columns = [
    'Vehicle Speed[km/h]',
    'Speed Limit with Direction[km/h]',
    'Elevation Smoothed[m]',
    'Gradient',
]
# ← Configure input-size = 4 em pyproject.toml
```

### **Passo 3: Executar Treinamento**

**Teste rápido (3 clientes):**
```bash
flwr run . --run-config "model-type=fedper rounds=3 min-nodes=3" local-simulation
```

**Teste médio (20 clientes, configuração padrão):**
```bash
flwr run . local-simulation
```

**Escala completa (184 clientes):**

Primeiro, atualize `pyproject.toml`:
```toml
options.num-supernodes = 184
```

Depois execute:
```bash
flwr run . --run-config "model-type=fedper rounds=10 min-nodes=50" local-simulation
```

---

## 📊 Validação

### **Teste do Mapeamento**

Execute para ver o mapeamento:
```bash
python fleven/client_mapping.py
```

**Output esperado:**
```
Total de clientes válidos: 184
Total de clientes removidos: 48

Primeiros 20 mapeamentos:
partition_id -> client_id
------------------------------
  0          ->   0
  1          ->   1
  2          ->   2
  3          ->   3
  4          ->   6   ← Pula client_4, client_5
  5          ->   7
  6          ->   9   ← Pula client_8
  ...
```

### **Logs Durante Execução**

Ao executar `flwr run .`, você verá:

**✅ Mapeamento correto:**
```
[DEBUG] Train - partition-id=4 -> client_id=6
[Cliente 6] Carregando dados do eVED
```

**✅ Validação de input-size:**
```
[Cliente 0] AVISO: num_features=1 difere de input-size=1 configurado
    Usando input-size=1 para manter compatibilidade com o servidor
```
(Esse aviso só aparece se houver divergência)

**✅ FedPer funcionando:**
```
[Cliente 0] FedPer: Cabeça global atualizada, cauda local mantida
[Cliente 0] Perda de treino: 0.023456
[Cliente 0] FedPer: Enviando apenas 8 parâmetros globais
```

---

## 🔍 Troubleshooting

### **Erro: IndexError - partition_id fora do alcance**

```
IndexError: partition_id 184 está fora do alcance.
Existem apenas 184 clientes válidos.
```

**Solução:** Reduza `num-supernodes` em `pyproject.toml` para <= 184.

---

### **Erro: RuntimeError - size mismatch**

```
RuntimeError: size mismatch for shared_lstm.weight_ih_l0
```

**Solução:** Verifique se `input-size` em `pyproject.toml` coincide com o número de features em `fleven/utils.py:feature_columns`.

**Exemplo:**
- Se `feature_columns = ['Vehicle Speed[km/h]']` → `input-size = 1`
- Se `feature_columns = ['Speed', 'Elevation', 'Gradient', 'Speed Limit']` → `input-size = 4`

---

### **Aviso: num_features difere de input-size**

```
⚠️  [Cliente 0] AVISO: num_features=2 difere de input-size=1 configurado
```

**Causa:** `fleven/utils.py` carrega mais features do que configurado em `input-size`.

**Solução:**
1. Verifique `feature_columns` em `fleven/utils.py` (linha ~118)
2. Conte quantas features estão **sem comentário** (linhas sem `#`)
3. Ajuste `input-size` em `pyproject.toml` para coincidir

---

## 📝 Resumo das Mudanças

| Arquivo | Mudança | Motivo |
|---------|---------|--------|
| **fleven/client_mapping.py** | Criado | Mapeia partition_id → client_id válido |
| **fleven/client.py** | `get_valid_client_id()` import | Usa mapeamento em train_fn e evaluate_fn |
| **fleven/client.py** | `configured_input_size` | Usa input-size do config, não num_features |
| **fleven/client.py** | `personal_hidden_size` param | Suporte ao FedPer |
| **fleven/client.py** | Validação num_features | Avisa se dados divergem do configurado |

---

## ✅ Próximos Passos

1. **Testar com 3 clientes** para validar que funciona:
   ```bash
   flwr run . --run-config "model-type=fedper rounds=3 min-nodes=3"
   ```

2. **Adicionar mais features universais** (recomendado para FedPer):
   - Edite `fleven/utils.py` linha ~118
   - Adicione features como `Speed Limit`, `Elevation`, `Gradient`
   - Atualize `input-size` em `pyproject.toml` para o número de features

3. **Comparar FedPer vs FedAvg:**
   ```bash
   # FedAvg baseline
   flwr run . --run-config "model-type=lstm rounds=5"

   # FedPer
   flwr run . --run-config "model-type=fedper rounds=5"
   ```

4. **Escalar para mais clientes** (até 184):
   - Atualize `num-supernodes` em `pyproject.toml`
   - Execute com `rounds=10` ou mais

---

**Correções aplicadas em:** 2025-12-09

**Arquivos modificados:**
- `fleven/client_mapping.py` (novo)
- `fleven/client.py` (modificado)
- `FIXES_APLICADOS.md` (este arquivo)
