# FLEVEn — Federated Learning for Vehicular Environment

## 📋 Visão Geral

Este projeto implementa Aprendizado Federado para previsão de dados OBD, como energia. 
Utiliza o framework Flower para orquestrar o treinamento colaborativo de modelos LSTM 
em múltiplos nós, sem centralizar os dados. O objetivo é prever variáveis veiculares (ex: potência, velocidade) 
a partir de séries temporais coletadas de diferentes clientes, promovendo privacidade e escalabilidade.

## 🎯 Vantagens da Configuração Centralizada

✅ **Fácil de manter**: Todas as configurações em um único lugar  
✅ **Flexível**: Altere parâmetros sem modificar código  
✅ **Reproduzível**: Documente configurações de experimentos  
✅ **Versionável**: Track de mudanças no Git  

## ⚙️ Configurações Disponíveis

### 1. Configurações de Federação

```toml
[tool.flwr.app.config]
strategy = "fedavg"          # Estratégia: fedavg, fedadam, fedyogi, fedadagrad
rounds = 5                   # Número de rodadas de treinamento
min-nodes = 3                # Número mínimo de nós necessários
```

### 2. Configurações do Modelo LSTM

```toml
input-size = 6               # Número de features de entrada
hidden-size = 50             # Tamanho da camada oculta LSTM
num-layers = 1               # Número de camadas LSTM empilhadas
```

### 3. Configurações de Séries Temporais

```toml
sequence-length = 60         # Tamanho da janela de entrada (timesteps)
prediction-length = 10       # Quantos timesteps prever no futuro
```

⚠️ **IMPORTANTE**: `prediction-length` deve corresponder ao `output_size` da rede!

### 4. Configurações de Treinamento

```toml
batch-size = 32              # Tamanho do batch
learning-rate = 1e-5         # Taxa de aprendizado
local-epochs = 1             # Épocas de treino local por rodada
max-grad-norm = 1.0          # Clip de gradiente
```

### 5. Configurações de Dados

```toml
train-test-split = 0.8       # Proporção treino/teste (80%/20%)
```

### 6. Configurações de Checkpoint

```toml
save-checkpoint-every = 5    # Salvar modelo a cada N rodadas
```

## 🚀 Como Usar

### Método 1: Simulação Local (Recomendado para Testes)

```bash
# Executa com configurações do pyproject.toml
flwr run .

# Ou especificamente a federação de simulação
flwr run . local-simulation
```

### Método 2: Deployment com SuperLink/SuperNodes

#### Terminal 1 - SuperLink
```bash
flower-superlink --insecure
```

#### Terminal 2, 3, 4 - SuperNodes (3 nós)
```bash
# Node 1
flower-supernode --insecure --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0"

# Node 2
flower-supernode --insecure --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config "partition-id=1"

# Node 3
flower-supernode --insecure --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9096 \
  --node-config "partition-id=2"
```

#### Terminal 5 - Executar Treinamento
```bash
flwr run . fleven-deployment --stream
```
C:\Users\abece\OneDrive\Documentos\fleven\data\client_1\trajeto_1.csv
## 🔧 Alterando Configurações

### Opção 1: Editar pyproject.toml

Edite o arquivo `pyproject.toml` e rode novamente:

```toml
[tool.flwr.app.config]
# Experimento: Aumentar prediction length
prediction-length = 20
hidden-size = 100
learning-rate = 5e-6
```

### Opção 2: Sobrescrever via Linha de Comando (CLI)

```bash
# Sobrescrever configurações específicas
flwr run . --run-config "rounds=10 learning-rate=1e-4 prediction-length=15"
```

## 📊 Estrutura de Dados Esperada

```
data/
├── client_1/
│   ├── route1.csv
│   └── route2.csv
├── client_2/
│   ├── route1.csv
│   └── route2.csv
└── client_3/
    ├── route1.csv
    └── route2.csv
```

Cada CSV deve conter as colunas:
- `vehicle_speed`
- `engine_rpm`
- `accel_x`
- `accel_y`
- `P_kW`
- `dt`

## 📈 Resultados

Após a execução, os resultados serão salvos em:

```
results/
├── performance_analysis_fedavg.pdf     # Gráficos de desempenho
├── convergence_analysis_fedavg.pdf     # Análise de convergência
├── heatmap_performance_fedavg.pdf      # Mapa de calor
├── train_metrics_fedavg.csv            # Métricas de treino
├── eval_metrics_fedavg.csv             # Métricas de avaliação
├── analysis_fedavg.json                # Análise estatística
└── summary_fedavg.txt                  # Resumo textual
```

Métricas locais de cada cliente:
```
metrics/
├── client_1/
│   ├── metrics_history.json
│   └── model_round_5.pt
├── client_2/
│   └── ...
└── client_3/
    └── ...
```

## 🧪 Exemplos de Experimentos

### Experimento 1: Previsão de Curto Prazo
```toml
prediction-length = 5
sequence-length = 30
learning-rate = 1e-4
```

## 🐛 Troubleshooting

### Erro: Dimensões não correspondem
```
RuntimeError: The size of tensor a (10) must match the size of tensor b (200)
```

**Solução**: Verifique que `prediction-length` no `pyproject.toml` corresponde ao tamanho esperado dos labels.

### Erro: Dados insuficientes
```
ValueError: A divisão de dados resultou em um conjunto vazio
```

**Solução**: Certifique-se de que cada cliente tem pelo menos `sequence-length + prediction-length` linhas de dados.

### Erro: Poucos nós conectados
```
INFO: Waiting for at least 3 nodes to connect...
```

**Solução**: Verifique que você iniciou pelo menos `min-nodes` SuperNodes.

## 📝 Notas Adicionais

### 🔐 Segurança (Produção)

Para ambientes de produção, **NUNCA use `--insecure`**. Configure TLS:

```toml
[tool.flwr.federations.production]
address = "your-server.com:9093"
root-certificates = "certificates/ca.crt"
```

### Habilitar TLS (Recomendado para Produção)
1. Gere certificados TLS (veja documentação Flower)
2. Atualize `pyproject.toml`:
```toml
[tool.flwr.federations.raspberry-deployment]
address = "<IP-SERVIDOR>:9093"
root-certificates = "./certificates/ca.crt"
insecure = false
```
3. Inicie SuperLink com certificados
4. Inicie SuperNodes com certificados

### Monitoramento
Use ferramentas como:
- `htop` para monitorar recursos nos Raspberry Pis
- Logs do Flower para debug
- Métricas salvas em `results/` e `metrics/`

## Referências
- [Flower Documentation](https://flower.ai/docs/framework/v1.22.0/)
- [Run Flower with Deployment Engine](https://flower.ai/docs/framework/v1.22.0/en/how-to-run-flower-with-deployment-engine.html)
- [Enable TLS](https://flower.ai/docs/framework/v1.22.0/en/how-to-enable-tls-connections.html)


## 📚 Recursos Adicionais

- [Flower Documentation](https://flower.ai/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

## 🤝 Citação

--------

## 📄 Licença

--------