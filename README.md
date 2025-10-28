# FLEVEn — Federated Learning for Vehicular Environment

Este projeto implementa Aprendizado Federado para previsão de dados OBD, como energia, velocidade, rpm, etc. 
Utiliza o framework [Flower](https://flower.ai) para orquestrar o treinamento colaborativo de modelos LSTM 
em múltiplos nós, sem centralizar os dados. O objetivo é prever variáveis veiculares (ex: potência, velocidade) 
a partir de séries temporais coletadas de diferentes clientes, promovendo privacidade e escalabilidade. Essa aplicação também permite 
visualização do desempenho dos clientes e permite testar diversas estratégias, tudo isso por contar com a integração com o [MLflow](https://mlflow.org/) 
facilidado o FLOps (uma adaptação do MLOps).

Resumo das pastas

- `/analysis`: contém um arquivo `.py` que verifica a estrutura e treina uma pequena rede para ver se as configurações do `pyproject.toml` estão ok.
- `/data`: dados utilizados nos testes
- `/images`: imagens
- `/fleven`: scripts `.py` do projeto FLEVEn

## 🚀 Como usar o FLEVEn (com MLflow)

Primeiro clone este repositório

```bash
git clone https://github.com/josewilsonsouza/fleven.git
cd fleven
```

Crie um ambiente virtual

```bash
python -m venv venv
```
Ative-o

```bash
venv/Scripts/activate # no windowns

source venv/bin/activate # no linux
```

Com seu ambiente virtual `venv` ativado, instale as dependencias do projeto nele

```bash
pip install -e .
```

Em outro terminal, inicie o servidor *MLflow*
```bash
./start_mlflow
```
ou

```bash
mlflow ui
```

Agora está quase pronto para iniciar o FLEVEn. Existem dois métodos de reproduzi-lo, seguindo o padrão de apps do [Flower](https://flower.ai).

### Método 1: Simulação Local (Recomendado para Testes)

```bash
# Executa com configurações do pyproject.toml (onde o default é local-simulation)
flwr run .
# ou
flwr run . local-simulation
```
### Método 2: Deployment com SuperLink/SuperNodes

Se escolher esse método, primeiro ajuste os caminhos especificados no `pyproject.toml` para conicidir com seu pc:

```bash
data-base-path = "C:/user/fleven/data"
metrics-base-path = "C:/user/fleven/metrics"
results-base-path = "C:/user/fleven/results"
```
Agora rode

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

## Visualizando os resultados

Você pode acompanhar o desempenho do servidor e dos clientes de duas formas. A mais direta é ir até a pasta `results` criada na raiz do projeto.
A forma mais interessante e que será usada para você ver a evolução do modelo com mudanças de configurações é atraves do MLflow, através da UI.
Acesse

```bash
http://127.0.0.1:500
```
Veja a documentação oficial do [MLflow](https://mlflow.org/) para mais detalhes da interface.

![Print da UI do MLflow para o FLEVEn](/images/mlflow_print.png)


## 🔧 Alterando Configurações

### Opção 1: Editar pyproject.toml

Edite o arquivo `pyproject.toml` e rode novamente (e acompanhe as mudanças no mlflow):

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
├── client_0/
│   ├── route1.csv
│   └── route2.csv
├── client_1/
│   ├── route1.csv
│   └── route2.csv
└── client_2/
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

Após a execução, os resultados serão salvos na pasta raiz, mas varias opções ficarão disponiveis no mlflow, de todos os testes.

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
├── client_0/
│   ├── metrics_history.json
│   └── model_round_5.pt
├── client_1/
│   └── ...
└── client_2/
    └── ...
```

## 📝 Notas Adicionais

### 🔐 Segurança (Produção)

Para ambientes de produção, **NUNCA usar `--insecure`**. Configure TLS:

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