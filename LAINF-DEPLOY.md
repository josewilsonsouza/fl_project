# FLEVEn
## Deploy em trêes Raspbarry Pi
Vamos simular o deploy do Fleven em 3 Raspbarry Pi conecatado em uma rede Taiscale e um PC como servidor (na mesma rede).

1. Em cada Pi clonamos o repositório:
```bash
git clone https://github.com/josewilsonsouza/fleven.git
cd fleven
```
⚠️ **Obs.** No nosso caso todos os Pi tem usuário `lainf`, isso vai ser importante mais na frente. Com isso o Fleven clonado está no seguinte _path_ em cada Pi:
```bash
/home/lainf/fleven
```
2. Agora, no SERVIDOR, altere essea parte do arquivo `pyproject.toml`, substituíndo `<IP-SERVIDOR>` pelo ip da rede Taiscale do servidor (seu PC). Não se preocupe com o `pyproject.toml` do cliente, durante a execução ele é ignorado (para os clientes ele serve apenas para fazer o `pip install -e .`).
```toml
[tool.flwr.federations.fleven-deployment]
address = "<IP-SERVIDOR>:9093"
insecure = true
```

3. Como nosso diretório `fleven`, em todos os Pi, tem o mesmo caminho `/home/lainf/fleven`, no SERVIDOR
altere o `pyproject.toml` para ficar assim
```toml
data-base-path = "/home/lainf/fleven/data"
metrics-base-path = "/seu-usuario/fleven/metrics" # na prática você pode colocar qualquer caminho onde deseja ver os resultados, salvos. Poderia inclusive ser /home/lainf/fleven/metrics
results-base-path = "/seu-usuario/fleven/results" # o mesmo comentário anterior para metrics vale pros results.
```
Note que por termos todos os Pi da forma `/home/lainf`, não precisamos editar o arquivo `utils.py` em cada Pi para especificar o caminho para `/data`.

4.  **Instalação das Dependências:**
    * **Em cada Raspberry Pi (Cliente):** Certifique-se de ter um ambiente Python (recomendado usar `venv`) e instale o projeto e suas dependências.
        ```bash
        # Dentro de /home/lainf/fleven em cada Pi
        python -m venv venv
        source venv/bin/activate
        pip install -e .
        ```
    * **No PC (Servidor):** Faça o mesmo, garantindo que o ambiente virtual esteja configurado.
        ```powershell
        # No diretório do projeto no seu PC
        python -m venv venv
        .\venv\Scripts\Activate # No Windows PowerShell
        # source venv/bin/activate # No Linux/macOS
        pip install -e .
        ```

5.  **Iniciar o SuperLink (no Servidor):**
    Abra um terminal no seu PC, ative o ambiente virtual e inicie o SuperLink em modo inseguro (já que estamos na VPN Tailscale). Ele vai escutar no endereço definido no passo 2 (`<IP-SERVIDOR>:9093`) para o `flwr run` e na porta 9092 para os SuperNodes.
    ```bash
    # No PC (Servidor)
    flower-superlink --insecure
    ```

6.  **Iniciar os SuperNodes (em cada Pi):**
    Conecte-se via SSH a cada Raspberry Pi, ative o ambiente virtual e inicie o `flower-supernode`. Cada Pi precisa de um `partition-id` diferente e deve apontar para o IP e porta (9092) do SuperLink.

    * **No Pi 1 (partition-id=0):**
        ```bash
        # Dentro de /home/lainf/fleven no Pi 1
        source venv/bin/activate
        flower-supernode --insecure --superlink <IP-SERVIDOR>:9092 --clientappio-api-address 0.0.0.0:9094 --node-config "partition-id=0 num-partitions=3"
        ```
    * **No Pi 2 (partition-id=1):**
        ```bash
        # Dentro de /home/lainf/fleven no Pi 2
        source venv/bin/activate
        flower-supernode --insecure --superlink <IP-SERVIDOR>:9092 --clientappio-api-address 0.0.0.0:9095 --node-config "partition-id=1 num-partitions=3"
        ```
    * **No Pi 3 (partition-id=2):**
        ```bash
        # Dentro de /home/lainf/fleven no Pi 3
        source venv/bin/activate
        flower-supernode --insecure --superlink <IP-SERVIDOR>:9092 --clientappio-api-address 0.0.0.0:9096 --node-config "partition-id=2 num-partitions=3"
        ```
7.  **Executar a Federação (no Servidor):**
    Abra *outro* terminal no seu PC, ative o ambiente virtual e execute o `flwr run`. Ele vai ler o `pyproject.toml`, encontrar a federação `fleven-deployment`, conectar-se ao SuperLink (no endereço `<IP-SERVIDOR>:9093`) e iniciar o processo.
    ```bash
    # No PC (Servidor), em um NOVO terminal
    flwr run . fleven-deployment --stream
    ```

8. **Resultados**
    - Voce pode acessar os reasultados no link que foi definido no `pyprojct.toml` do servidor: [https://jwsouza13-fleven.hf.space](https://jwsouza13-fleven.hf.space).
    - No `pyproject.toml` você pode mudar o nome dos experimetos que deseja realizar. Atualmente está `Fleven-Deploy`.
    - No `pyproject.toml` atual temos estamos prevendo `vehicle_speed` a partir de outras 3 variáveis, isso pode ser alterado também.

## Como o Flower distribui as configurações

**Componentes Principais (Modo Fleet API)**: SuperLink (Servidor - PC). Ele gerencia a conexão dos SuperNodes, recebe "Runs" (trabalhos de FL) submetidos pelo flwr run, e encaminha as tarefas (instruções + código + configuração) para os SuperNodes selecionados. Ele escuta na porta 9092 para os SuperNodes e na **9093** para o `flwr run`.

**SuperNode (Cliente - Raspberry Pi)**: É o agente de longa duração que roda em cada dispositivo cliente. Ele se conecta ao SuperLink, anuncia sua disponibilidade, recebe tarefas, executa o ClientApp correspondente à tarefa recebida, e envia os resultados de volta.

**flwr run (Servidor - PC)**: É um comando CLI de curta duração usado para iniciar um "Run". Ele lê o `pyproject.toml` do servidor , empacota o código do projeto (`ServerApp`, `ClientApp`, `utils.py`, etc.) e a configuração ([tool.flwr.app.config]) em um arquivo chamado *FAB (Flower Application Bundle)* , e o submete ao SuperLink pela porta 9093.

**ServerApp (Servidor - PC)**: Código Python de curta duração (definido em `server.py`) que contém a lógica específica do servidor para um determinado Run (ex: a estratégia FedAvg, inicialização do modelo global). É iniciado pelo `SuperLink/SuperExec` quando um Run começa.

**ClientApp (Cliente - Raspberry Pi)**: Código Python de curta duração (definido em `client.py` e usando `utils.py`) que contém a lógica específica do cliente para um determinado Run (ex: carregar dados locais, treinar/avaliar o modelo). É iniciado pelo `SuperNode/SuperExec` quando recebe uma tarefa.

**Distribuição das Configurações:**

O comando `flwr run` no PC lê o `pyproject.toml` do PC, incluindo a seção `[tool.flwr.app.config]` (onde foi definido `data-base-path="/home/lainf/fleven/data"`). Essa configuração é empacotada junto com o código (`client.py`, `utils.py`, etc.) no FAB. O `flwr run` envia o FAB para o `SuperLink`.

Quando o `ServerApp` (rodando no servidor) decide iniciar um round e seleciona os Pis, o `SuperLink` envia as instruções e o FAB para os `SuperNodes` selecionados. Cada `SuperNode` recebe o FAB, descompacta o código e a configuração em um diretório temporário (como `/home/lainf/.flwr/apps/...`).

O `SuperNode` inicia o `ClientApp` (o `client.py`). O código dentro do `ClientApp` (por exemplo, a função `load_data` em `utils.py`) recebe a configuração que veio do FAB, não do `pyproject.toml` local do Pi.
O pyproject.toml presente nos Raspberry Pis não é lido durante a execução do `ClientApp` neste modo; ele serve apenas para o `pip install -e .` saber quais dependências instalar.

### Portas

┌─────────────────────────────────────────────────────┐
│                    SERVIDOR                         │
│                                                     │
│  ┌───────────────────────────────────────────┐     │
│  │         flower-superlink                  │     │
│  │                                           │     │
│  │  ┌─────────────────────────────────────┐ │     │
│  │  │ Control API                         │ │     │
│  │  │ Porta: 9093                         │ │     │
│  │  │ Recebe: flwr run/ls/stop           │ │     │
│  │  └─────────────────────────────────────┘ │     │
│  │                                           │     │
│  │  ┌─────────────────────────────────────┐ │     │
│  │  │ Fleet API (gRPC-rere)               │ │     │
│  │  │ Porta: 9092                         │ │     │
│  │  │ Recebe: SuperNodes (clients)        │ │     │
│  │  └─────────────────────────────────────┘ │     │
│  │                                           │     │
│  │  ┌─────────────────────────────────────┐ │     │
│  │  │ ServerAppIo API                     │ │     │
│  │  │ Porta: 9091                         │ │     │
│  │  │ Comunicação interna ServerApp       │ │     │
│  │  └─────────────────────────────────────┘ │     │
│  └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
           ▲                           ▲
           │                           │
    Porta 9093                   Porta 9092
           │                           │
           │                           │
    ┌──────┴──────┐           ┌────────┴─────────┐
    │             │           │                  │
    │  flwr run   │           │  SuperNodes      │
    │  (PC/User)  │           │  (Raspberry Pi)  │
    │             │           │                  │
    └─────────────┘           └──────────────────┘

## Tabela de Portas

| Porta | Componente | Protocolo | Conecta de | Propósito |
|-------|-----------|-----------|------------|-----------|
| **9093** | Control API | HTTP/REST | `flwr` CLI | Comandos admin (run/ls/stop) |
| **9092** | Fleet API | gRPC-rere | SuperNodes | Coordenação clientes ↔ servidor |
| **9091** | ServerAppIo | gRPC | Interno | SuperLink ↔ ServerApp |
| **9094-9096** | ClientAppIo | gRPC | Local | SuperNode ↔ ClientApp |


### 1. **Execução de `flwr run`**
```
[Seu PC] --→ [9093 Control API] no SuperLink
         "Inicie um novo run com este FAB"
```

### 2. **SuperLink recebe o FAB**
```
[SuperLink] ← FAB recebido
            ↓
      [Armazena FAB internamente]
            ↓
      [Aguarda SuperNodes]
```

### 3. **SuperNodes conectam**
```
[Rasp1] --→ [9092 Fleet API] "Estou disponível!"
[Rasp2] --→ [9092 Fleet API] "Estou disponível!"
[Rasp3] --→ [9092 Fleet API] "Estou disponível!"
```

### 4. **SuperLink distribui tarefas**
```
[SuperLink via 9092] --→ [Rasp1] "Execute treino com estes params"
[SuperLink via 9092] --→ [Rasp2] "Execute treino com estes params"
[SuperLink via 9092] --→ [Rasp3] "Execute treino com estes params"
```

### 5. **SuperNodes executam ClientApp**
```
[Rasp1] 
  SuperNode (porta 9092 ↔ SuperLink)
      ↓↑
  ClientApp (porta 9094 local) ← Executa seu código
      ↓
  Resultado enviado de volta via 9092
```

### 6. **Resultados voltam**
```
[Rasp1] --→ [9092 Fleet API] "Treino concluído, aqui o modelo"
[Rasp2] --→ [9092 Fleet API] "Treino concluído, aqui o modelo"
[Rasp3] --→ [9092 Fleet API] "Treino concluído, aqui o modelo"
            ↓
      [SuperLink agrega]
            ↓
      [Próxima rodada...]