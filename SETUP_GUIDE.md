# Guia Rápido - Configuração Distribuída FL

## 🎯 Objetivo
Executar treinamento federado com 1 servidor Windows e 3 clientes Ubuntu em rede local.

## ⚡ Setup Rápido

### No Servidor Windows (192.168.1.100)

1. **Abra o PowerShell como Administrador**

2. **Configure o Firewall:**
```powershell
# Liberar porta do FL
New-NetFirewallRule -DisplayName "FL Server 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

3. **Prepare o ambiente:**
```powershell
cd C:\fleven_v0
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

4. **Inicie o servidor:**
```powershell
python server.py --strategy fedavg --rounds 15
```

### Nos Clientes Ubuntu

**Para cada cliente (em máquinas diferentes):**

1. **Clone e prepare:**
```bash
git clone -b fleven_v0 --single-branch https://github.com/josewilsonsouza/fleven.git
cd fleven_v0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure os dados:**
```bash
# Cliente 1: Copie apenas data/client_1
# Cliente 2: Copie apenas data/client_2  
# Cliente 3: Copie apenas data/client_3
```

3. **Execute o cliente correspondente:**

**Máquina 1:**
```bash
python client.py --client-id 1 --server-address 192.168.1.100:8080
```

**Máquina 2:**
```bash
python client.py --client-id 2 --server-address 192.168.1.100:8080
```

**Máquina 3:**
```bash
python client.py --client-id 3 --server-address 192.168.1.100:8080
```

## 📊 Monitoramento

### No Servidor (Windows)
```
================================================
SERVIDOR DE APRENDIZADO FEDERADO
================================================
Estratégia: FEDAVG
Rodadas: 15
Clientes mínimos: 3
================================================

INFO flower 2024-12-XX Starting Flower server...
INFO flower Flower ECE: gRPC server running (15 rounds)
INFO flower Starting round 1...
```

### Nos Clientes (Ubuntu)
```
[Cliente 1] === Rodada 1 ===
[Cliente 1] Iniciando treinamento...
[Cliente 1] Perda de treinamento local: 0.004523
[Cliente 1] Perda de validação: 0.003892
```

## 🔍 Verificação de Conectividade

**Nos clientes Ubuntu:**
```bash
# Testar conexão com servidor
nc -zv 192.168.1.100 8080
```

**No servidor Windows:**
```powershell
# Ver conexões ativas
netstat -an | findstr :8080
```

## 📈 Resultados

Após conclusão, no servidor Windows:

1. **Visualizar PDFs gerados:**
   - Navegue até `C:\fleven_v0\results`
   - Abra os arquivos `.pdf` gerados

2. **Análise detalhada:**
```powershell
python analysis_tool.py
```

## ⚠️ Dicas Importantes

1. **Sincronização**: Inicie todos os clientes em até 30 segundos após o servidor
2. **Memória**: Monitore o uso de RAM, especialmente nos clientes (8GB)
3. **Rede**: Use cabo Ethernet para melhor estabilidade
4. **Dados**: Cada cliente precisa apenas dos seus próprios dados (`client_X`)

## 🆘 Problemas Comuns

| Problema | Solução |
|----------|---------|
| Conexão recusada | Verificar IP do servidor e firewall |
| Out of memory | Reduzir batch_size para 16 |
| Cliente travado | Reiniciar com `--prediction-length 5` |
| Servidor não inicia | Porta 8080 em uso, matar processo |

## 📞 Comandos Úteis

**Windows (PowerShell):**
```powershell
# Ver IP
ipconfig | findstr IPv4

# Matar processo na porta 8080
netstat -ano | findstr :8080
taskkill /PID [PID_NUMBER] /F
```

**Ubuntu:**
```bash
# Ver uso de memória
free -h

# Monitorar processo
htop

# Matar processo Python
pkill -f "python client.py"
```