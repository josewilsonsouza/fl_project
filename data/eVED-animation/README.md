# 🌟 eVED Trajectory Viewer

## 📦 Instalação

```bash
# Navegar para o diretório
cd data/eVED-animation

# Instalar dependências
pip install -r requirements.txt
```


## 🚀 Como Iniciar
Você pode iniciar o app eVED de duas formas.

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

## 🆘 Suporte

- 💻 Código: [app_streamlit.py](app_streamlit.py)
- 🐛 Issues: Repositório do [FLEVEn](https://github.com/josewilsonsouza/fleven)