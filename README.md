# Federated Learning para Previsão de Consumo de Combustível

Sistema de Aprendizado Federado (FL) para prever consumo de combustível usando dados de sensores OBD de diferentes veículos, mantendo a privacidade dos dados em cada cliente.

## 📋 Visão Geral

Este projeto implementa um sistema de Aprendizado Federado usando o framework Flower, onde:
- **3 clientes** (Ubuntu) representam diferentes veículos com seus dados locais
- **1 servidor** (Windows) coordena o treinamento sem acessar os dados brutos
- Modelo LSTM para previsão de séries temporais de consumo (P_kW)
- Múltiplas estratégias de agregação: FedAvg, FedAdam, FedYogi, FedAdagrad

## 🏗️ Arquitetura do Sistema

```
┌─────────────────┐
│  Servidor (Win) │
│   16GB RAM      │
│   Porta: 8080   │
└────────┬────────┘
         │
    ┌────┴────┬──────────┐
    │         │          │
┌───▼───┐ ┌──▼───┐ ┌────▼───┐
│Cliente│ │Cliente│ │Cliente │
│   1   │ │   2  │ │   3    │
│Ubuntu │ │Ubuntu│ │Ubuntu  │
│ 8GB   │ │ 8GB  │ │  8GB   │
└───────┘ └──────┘ └────────┘
```

## 📁 Estrutura do Projeto

```
fl_project/
├── data/                    # Dados dos veículos (não versionado)
│   ├── client_1/           # Percursos do veículo 1
│   │   ├── percurso_1.csv
│   │   ├── percurso_2.csv
│   │   └── ...
│   ├── client_2/           # Percursos do veículo 2
│   └── client_3/           # Percursos do veículo 3
├── server.py               # Código do servidor FL
├── client.py               # Código dos clientes FL
├── utils.py                # Modelo LSTM e funções auxiliares
├── analysis_tool.py        # Ferramenta de análise pós-treinamento
├── run.sh                  # Script para execução local
├── run_all_strategies.sh   # Script para testar todas as estratégias
├── requirements.txt        # Dependências Python
└── README.md              # Este arquivo
```

## 🔧 Requisitos do Sistema

### Hardware Mínimo
- **Servidor**: 8GB RAM (recomendado 16GB)
- **Clientes**: 4GB RAM cada (recomendado 8GB)
- **Rede**: Conexão estável entre servidor e clientes

### Software
- **Python**: 3.10 - 3.11
- **Sistema Operacional**:
  - Servidor: Windows 10/11 ou Linux
  - Clientes: Ubuntu 20.04/22.04

## 📦 Instalação

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/fl_project.git
cd fl_project
```

### 2. Crie um Ambiente Virtual

**No Ubuntu (Clientes):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**No Windows (Servidor):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Prepare os Dados

Organize os dados de cada veículo na estrutura:
```
data/
├── client_1/  # Dados do veículo 1
├── client_2/  # Dados do veículo 2
└── client_3/  # Dados do veículo 3
```

**Formato esperado dos CSVs:**
- Colunas principais: `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y`, `P_kW`, `dt`
- Cada arquivo representa um percurso diferente
- Mínimo de 2 percursos por cliente recomendado

## 🚀 Execução em Ambiente Distribuído

### Configuração de Rede

1. **Identifique o IP do servidor Windows:**
   ```powershell
   ipconfig
   ```
   Procure pelo IPv4 Address (ex: 192.168.1.100)

2. **Teste a conectividade dos clientes Ubuntu:**
   ```bash
   ping 192.168.1.100
   ```

### Passo 1: Iniciar o Servidor (Windows)

```powershell
# Ative o ambiente virtual
.\venv\Scripts\activate

# Execute o servidor
python server.py --strategy fedavg --rounds 15

# Ou com parâmetros customizados
python server.py --strategy fedadam --rounds 20 --min-clients 3
```

O servidor iniciará na porta 8080 e aguardará a conexão dos clientes.

### Passo 2: Iniciar os Clientes (Ubuntu)

**Em cada máquina Ubuntu, execute em terminais separados:**

**Cliente 1:**
```bash
# Ative o ambiente virtual
source venv/bin/activate

# Execute o cliente 1
python client.py --client-id 1 --server-address 192.168.1.100:8080 --prediction-length 10
```

**Cliente 2:**
```bash
source venv/bin/activate
python client.py --client-id 2 --server-address 192.168.1.100:8080 --prediction-length 10
```

**Cliente 3:**
```bash
source venv/bin/activate
python client.py --client-id 3 --server-address 192.168.1.100:8080 --prediction-length 10
```

### Monitoramento

O progresso será exibido em tempo real:
- **Servidor**: Mostra rodadas completas e métricas globais
- **Clientes**: Exibem perdas locais de treino/validação

## 📊 Análise dos Resultados

### Após o Treinamento

1. **Executar análise automática:**
   ```bash
   python analysis_tool.py --results-dir results
   ```

2. **Visualizações geradas (PDFs):**
   - `performance_analysis_*.pdf`: Análise de desempenho completa
   - `convergence_analysis_*.pdf`: Métricas de convergência
   - `heatmap_performance_*.pdf`: Mapa de calor temporal
   - `comparative_analysis.pdf`: Comparação entre estratégias
   - `client_evolution_analysis.pdf`: Evolução individual

3. **Métricas salvas:**
   - `results/detailed_metrics_*.csv`: Dados completos
   - `results/summary_report.json`: Relatório consolidado
   - `metrics/client_*/metrics_history.json`: Histórico por cliente

## 🔬 Estratégias de Agregação

| Estratégia | Descrição | Quando Usar |
|------------|-----------|-------------|
| **FedAvg** | Média ponderada simples | Dados homogêneos |
| **FedAdam** | Otimização adaptativa | Convergência mais rápida |
| **FedYogi** | Adam com controle de variância | Dados heterogêneos |
| **FedAdagrad** | Taxa de aprendizado adaptativa | Dados esparsos |

### Comparar Todas as Estratégias

```bash
# Linux/Ubuntu
chmod +x run_all_strategies.sh
./run_all_strategies.sh 15 10

# Windows (usando Git Bash ou WSL)
bash run_all_strategies.sh 15 10
```

## 🛠️ Troubleshooting

### Erro de Conexão

**Problema**: Clientes não conseguem conectar ao servidor

**Soluções**:
1. Verifique o firewall do Windows:
   ```powershell
   # Permitir porta 8080
   netsh advfirewall firewall add rule name="FL Server" dir=in action=allow protocol=TCP localport=8080
   ```

2. Confirme que o servidor está rodando:
   ```powershell
   netstat -an | findstr :8080
   ```

### Erro de Memória

**Problema**: Out of Memory durante treinamento

**Soluções**:
1. Reduza o batch_size em `utils.py`
2. Diminua sequence_length ou prediction_length
3. Use menos épocas por rodada

### Dados Insuficientes

**Problema**: "conjunto de treino ou teste vazio"

**Soluções**:
1. Verifique se há dados suficientes em `data/client_X/`
2. Ajuste sequence_length e prediction_length
3. Confirme que os CSVs têm as colunas esperadas

## 📈 Parâmetros Importantes

### Server.py
- `--strategy`: Estratégia de agregação (fedavg, fedadam, etc.)
- `--rounds`: Número de rodadas de FL (default: 10)
- `--min-clients`: Clientes mínimos para iniciar (default: 3)

### Client.py
- `--client-id`: ID do cliente (1, 2 ou 3)
- `--server-address`: Endereço IP:porta do servidor
- `--prediction-length`: Passos futuros a prever (default: 10)

### Utils.py (configurações internas)
- `sequence_length`: Janela de entrada (default: 60)
- `batch_size`: Tamanho do batch (default: 32)
- `learning_rate`: Taxa de aprendizado (default: 1e-5)

## 📝 Notas de Desenvolvimento

### Modelo LSTM
- Entrada: 6 features (velocidade, RPM, acelerações, consumo, tempo)
- Hidden size: 50 neurônios
- Saída: Previsão de N passos futuros de consumo (P_kW)

### Divisão dos Dados
- 80% para treinamento
- 20% para validação
- Normalização MinMaxScaler por cliente

### Métricas
- Loss: MSE (Mean Squared Error)
- Avaliação: Por cliente e global
- Convergência: Variância entre clientes

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 👥 Autores

- José Wilson C. Souza
- Erick Andrade Borba
- João Alfredo Cal Braz

## 🙏 Agradecimentos

- [Flower Framework](https://flower.dev/) - Framework de Aprendizado Federado
- [PyTorch](https://pytorch.org/) - Framework de Deep Learning
- Dados coletados via OBD Link

---
