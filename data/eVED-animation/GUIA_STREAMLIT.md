# 🌟 Guia da Aplicação Streamlit - eVED Trajectory Viewer

## 📋 Visão Geral

A aplicação Streamlit oferece uma interface web intuitiva e interativa para visualizar e analisar trajetos de veículos do dataset eVED.

## 🚀 Como Iniciar

### Opção 1: Scripts Automáticos

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
bash run_app.sh
```

### Opção 2: Comando Direto

```bash
streamlit run app_streamlit.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

## 🎯 Funcionalidades Principais

### 1. 🗺️ Tab: Visualização

**O que você pode fazer:**
- Visualizar trajeto interativo no mapa Plotly
- Zoom e pan no mapa
- Ver marcadores de início (verde) e fim (vermelho)
- Gerar mapa animado HTML com Folium
- Download do mapa HTML gerado

**Como usar:**
1. Selecione cliente e trip na sidebar
2. O mapa aparecerá automaticamente
3. Clique em "Gerar Visualização Interativa" para criar o mapa HTML animado
4. Use o botão de download para salvar o mapa

### 2. 📊 Tab: Análise de Dados

**O que você pode fazer:**
- Ver gráficos de velocidade ao longo do trajeto
- Analisar histograma de distribuição de velocidade
- Visualizar box plot de velocidade
- Ver consumo de energia ao longo do trajeto
- Analisar relação velocidade × consumo (scatter plot com trendline)
- Visualizar tabela de dados brutos
- Download dos dados em CSV

**Gráficos disponíveis:**
- 📈 Linha: Velocidade ao longo do trajeto
- 📊 Histograma: Distribuição de velocidade
- 📦 Box Plot: Dispersão de velocidade
- 🔋 Linha: Consumo de energia
- ⚡🔋 Scatter: Velocidade vs Energia (com linha de tendência)

### 3. 📈 Tab: Estatísticas

**O que você pode fazer:**
- Ver estatísticas detalhadas de GPS
- Ver estatísticas de velocidade (média, mediana, desvio padrão, etc.)
- Ver estatísticas de energia (total, média, máximo, mínimo)
- Ver informações gerais da trip

**Métricas disponíveis:**
- 🌍 GPS: lat/lon min/max, total de pontos
- ⚡ Velocidade: média, mediana, std, min, max, pontos > 80 km/h
- 🔋 Energia: total, média, máximo, mínimo
- ℹ️ Info: cliente, trip, tipo de veículo, memória

### 4. ℹ️ Tab: Informações

**O que você pode fazer:**
- Ver documentação do sistema
- Ver estatísticas gerais do dataset
- Ver distribuição de tipos de veículos
- Entender as funcionalidades disponíveis

**Visualizações:**
- Métricas totais de clientes (train/test)
- Gráfico de pizza: distribuição de tipos de veículos

## 🎛️ Controles da Sidebar

### Configurações Principais

1. **Conjunto de Dados** (Train/Test)
   - Escolha entre dados de treino ou teste

2. **Filtrar por Tipo de Veículo**
   - Todos (padrão)
   - EV (Elétrico) - Verde
   - ICE (Combustão) - Vermelho
   - HEV (Híbrido) - Laranja
   - PHEV (Plug-in Híbrido) - Azul

3. **Selecionar Cliente**
   - Lista de todos os clientes disponíveis
   - Atualiza automaticamente baseado no filtro

4. **Selecionar Trip**
   - Lista de todas as trips do cliente selecionado
   - Ordenadas numericamente

### Opções Avançadas (Expandível)

- ☑️ **Mostrar Estatísticas Detalhadas**: Ativa/desativa tab de estatísticas
- ☑️ **Mostrar Gráficos Analíticos**: Ativa/desativa gráficos na tab de análise
- ☑️ **Mostrar Tabela de Dados**: Ativa/desativa tabela de dados brutos

## 💡 Dicas de Uso

### Navegação Rápida

1. **Para análise rápida:**
   - Deixe filtro em "Todos"
   - Selecione primeiro cliente
   - O mapa carrega automaticamente

2. **Para comparar tipos de veículos:**
   - Mude o filtro de tipo de veículo
   - Compare gráficos de diferentes tipos
   - Note as diferenças de consumo de energia

3. **Para análise detalhada:**
   - Ative todas as opções avançadas
   - Explore todas as tabs
   - Use os gráficos interativos do Plotly (zoom, pan, hover)

### Performance

- A primeira carga pode demorar alguns segundos
- Mapas com muitos pontos (>500) podem ser mais lentos
- Use o filtro de tipo para reduzir o número de clientes
- Desative gráficos se estiver rodando em máquina lenta

### Exportação de Dados

1. **Mapa HTML:**
   - Clique em "Gerar Visualização Interativa"
   - Aguarde a geração
   - Clique em "Download Mapa HTML"
   - Abra o arquivo em qualquer navegador

2. **Dados CSV:**
   - Vá para tab "Análise de Dados"
   - Ative "Mostrar Tabela de Dados"
   - Clique em "Download CSV"

## 🎨 Cores e Legendas

### Tipos de Veículos

| Tipo | Nome | Cor | Hex |
|------|------|-----|-----|
| EV | Elétrico | 🟢 Verde | #00ff00 |
| ICE | Combustão | 🔴 Vermelho | #ff0000 |
| HEV | Híbrido | 🟠 Laranja | #ffaa00 |
| PHEV | Plug-in Híbrido | 🔵 Azul | #0088ff |

### Marcadores no Mapa

- 🟢 **Verde**: Início do trajeto
- 🔴 **Vermelho**: Fim do trajeto
- **Linha colorida**: Trajeto completo (cor varia por tipo de veículo)

## 🔧 Solução de Problemas

### Aplicação não abre

```bash
# Verificar se Streamlit está instalado
pip install streamlit

# Verificar versão
streamlit --version

# Reinstalar se necessário
pip install --upgrade streamlit
```

### Erro ao carregar dados

1. Verifique se a pasta `../EVED_Clients` existe
2. Verifique se há arquivos `.parquet` nos diretórios dos clientes
3. Tente outro cliente/trip

### Gráficos não aparecem

1. Ative "Mostrar Gráficos Analíticos" nas opções avançadas
2. Verifique se os dados têm as colunas necessárias
3. Recarregue a página (F5)

### Performance lenta

1. Desative gráficos não utilizados
2. Use filtros para reduzir dados
3. Feche outras tabs do navegador
4. Reinicie a aplicação

## 📱 Responsividade

A aplicação é responsiva e funciona em:
- 💻 Desktop (recomendado)
- 📱 Tablet (funcional)
- 📱 Mobile (limitado, melhor usar landscape)

## 🔗 Atalhos de Teclado

| Tecla | Ação |
|-------|------|
| `R` | Recarregar aplicação |
| `C` | Limpar cache |
| `Ctrl/Cmd + K` | Menu de comandos |
| `?` | Ajuda rápida |

## 📊 Exemplos de Uso

### Caso 1: Comparar Consumo de EVs

1. Filtro: `EV`
2. Selecione vários clientes diferentes
3. Compare gráficos de energia na tab "Análise de Dados"
4. Note padrões de consumo

### Caso 2: Analisar Velocidade em Trajetos Urbanos

1. Selecione cliente com trajeto urbano
2. Tab "Análise de Dados"
3. Veja histograma de velocidade
4. Identifique padrões de stop-and-go

### Caso 3: Exportar Visualização para Apresentação

1. Selecione trajeto interessante
2. Gere mapa HTML animado
3. Download do arquivo
4. Incorpore em apresentação PowerPoint/Google Slides

## 🆘 Suporte

- 📖 Documentação: [README.md](README.md)
- 💻 Código: [app_streamlit.py](app_streamlit.py)
- 🐛 Issues: Repositório do FLEVEn

---

**Desenvolvido com ❤️ para o projeto FLEVEn**
