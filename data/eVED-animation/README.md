# Visualização de Trajetos eVED

Este pacote fornece ferramentas para criar animações interativas dos trajetos de veículos do dataset eVED usado no FLEVEn.

## 📦 Instalação

```bash
# Navegar para o diretório
cd data/eVED-animation

# Instalar dependências
pip install -r requirements.txt
```

## 🚀 Quick Start

### 🌟 Aplicação Web Interativa (RECOMENDADO)

Execute a aplicação Streamlit para uma experiência visual completa:

```bash
# Windows
run_app.bat

# Linux/Mac
bash run_app.sh

# Ou diretamente
streamlit run app_streamlit.py
```

A aplicação oferece:
- 🗺️ Visualização interativa de trajetos animados
- 📊 **Gráficos sincronizados em tempo real** (Velocidade, Energia/Combustível, Elevação)
- 🎯 Sincronização perfeita entre animação e gráficos
- 📈 Estatísticas detalhadas
- 🎬 Geração de mapas animados HTML com painéis de dados
- 🔍 Filtros por tipo de veículo
- 📥 Download de dados e visualizações

### Outras Opções

- **Tutorial Jupyter**: Abra o [tutorial_visualizacao.ipynb](tutorial_visualizacao.ipynb) para exemplos interativos
- **Scripts Python**: Execute [exemplo.py](exemplo.py) para criar visualizações programaticamente
- **Uso Direto**: Execute `python eved_vizu.py` para uma visualização rápida

## Características

- 🎬 Mapas interativos com animação JavaScript suave
- 🎮 Controles de play/pause, velocidade e timeline
- 📊 **Gráficos sincronizados em tempo real:**
  - ⚡ Velocidade do veículo
  - 🔋 Consumo de energia (EV/PHEV) ou ⛽ Taxa de combustível (ICE/HEV)
  - ⛰️ Elevação do trajeto
- 🔢 Valores numéricos atualizados em tempo real
- 📈 Visualização rolante dos últimos 50 pontos
- 🚗 Suporte para diferentes tipos de veículos (EV, ICE, HEV, PHEV)
- 🎨 Cores diferenciadas por tipo de veículo
- 📍 Marcadores de início e fim do trajeto
- 🗺️ Trajeto completo em segundo plano

## Uso

### Importando o pacote

```python
from data.eVED_animation import visualize_trip, EVEDVisualizer
```

### Uso Rápido

```python
# Visualizar uma trip específica
visualize_trip(client_id=0, trip_id=706.0)

# Visualizar a primeira trip disponível de um cliente
visualize_trip(client_id=0, split='train')
```

### Uso Avançado

```python
# Criar instância do visualizador
viz = EVEDVisualizer()

# Listar clientes disponíveis
clientes_train = viz.get_available_clients('train')
clientes_test = viz.get_available_clients('test')

# Listar trips de um cliente
trips = viz.get_available_trips(client_id=0, split='train')

# Criar animação customizada
viz.create_animated_map(
    client_id=0,
    split='train',
    trip_id=706.0,
    output_file='minha_animacao.html'
)
```

### Executar diretamente

```bash
# A partir do diretório raiz do projeto
cd data/eVED-animation
python eved_vizu.py
```

Ou:

```bash
# A partir do diretório raiz do projeto
python -m data.eVED_animation.eved_vizu
```

## Estrutura de Dados Esperada

O visualizador espera que os dados estejam organizados assim:

```
data/
└── EVED_Clients/
    ├── train/
    │   ├── client_0/
    │   │   ├── trip_706.0.parquet
    │   │   ├── trip_707.0.parquet
    │   │   └── metadata.json (opcional)
    │   └── client_1/
    │       └── ...
    └── test/
        ├── client_0/
        └── ...
```

## Colunas Necessárias nos Arquivos Parquet

- `Latitude[deg]`: Latitude GPS
- `Longitude[deg]`: Longitude GPS
- `Timestamp(ms)`: Timestamp em milissegundos
- `EngineType` ou `Vehicle Type`: Tipo do veículo (opcional, para cores)

## Saída

O visualizador gera arquivos HTML interativos que podem ser abertos diretamente no navegador. Cada arquivo contém:

- Mapa base OpenStreetMap
- Trajeto completo em cinza claro
- Marcadores de início (verde) e fim (vermelho)
- Animação do veículo com controles:
  - Botão Play/Pause
  - Botão Reset
  - Slider de timeline
  - Slider de velocidade

## Tipos de Veículos Suportados

- **EV** (Electric Vehicle): Verde - `#00ff00`
- **ICE** (Internal Combustion Engine): Vermelho - `#ff0000`
- **HEV** (Hybrid Electric Vehicle): Laranja - `#ffaa00`
- **PHEV** (Plug-in Hybrid Electric Vehicle): Azul - `#0088ff`

## Otimizações

O sistema automaticamente reduz o número de pontos para até 300 para melhor performance da animação, mantendo a qualidade visual do trajeto.
