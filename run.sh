#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
STRATEGY=${1:-fedavg}  # Estratégia padrão: fedavg
ROUNDS=${2:-15}         # Número de rodadas padrão: 15
PREDICTION_LENGTH=${3:-10}  # Tamanho da previsão padrão: 10

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    SIMULAÇÃO DE APRENDIZADO FEDERADO${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Estratégia: ${YELLOW}$STRATEGY${NC}"
echo -e "${GREEN}Rodadas: ${YELLOW}$ROUNDS${NC}"
echo -e "${GREEN}Tamanho da Previsão: ${YELLOW}$PREDICTION_LENGTH passos${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Limpar logs anteriores
echo -e "${YELLOW}Limpando logs anteriores...${NC}"
rm -f *.log
rm -rf metrics/
mkdir -p metrics
mkdir -p results

# Verificar se os dados existem
if [ ! -d "data" ]; then
    echo -e "${RED}Erro: Diretório 'data' não encontrado!${NC}"
    echo -e "${YELLOW}Certifique-se de que os dados estão organizados em:${NC}"
    echo "  data/client_1/*.csv"
    echo "  data/client_2/*.csv"
    echo "  data/client_3/*.csv"
    exit 1
fi

# Função para verificar se o servidor está pronto
wait_for_server() {
    echo -e "${YELLOW}Aguardando servidor iniciar...${NC}"
    for i in {1..30}; do
        if netstat -an | grep ":8080" | grep -q "LISTENING"; then
            echo -e "${GREEN}Servidor pronto!${NC}"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    echo -e "\n${RED}Timeout esperando servidor!${NC}"
    return 1
}

# Iniciar servidor
echo -e "${BLUE}Iniciando servidor FL...${NC}"
#python server.py --strategy $STRATEGY --rounds $ROUNDS > server_$STRATEGY.log 2>&1 &
python server.py --strategy $STRATEGY --rounds $ROUNDS --prediction-length $PREDICTION_LENGTH > server_$STRATEGY.log 2>&1 &
SERVER_PID=$!

# Aguardar servidor
if ! wait_for_server; then
    echo -e "${RED}Falha ao iniciar servidor. Verifique server_$STRATEGY.log${NC}"
    exit 1
fi

# Iniciar clientes
echo -e "\n${BLUE}Iniciando 3 clientes FL...${NC}"
CLIENT_PIDS=()

for i in {1..3}; do
    echo -e "${GREEN}  → Iniciando cliente $i${NC}"
    python client.py \
        --client-id $i \
        --prediction-length $PREDICTION_LENGTH \
        > client_${i}_$STRATEGY.log 2>&1 &
    CLIENT_PIDS+=($!)
    sleep 2  # Pequeno delay entre clientes
done

# Monitorar progresso
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}    TREINAMENTO EM PROGRESSO${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "${YELLOW}Monitorando treinamento...${NC}"
echo -e "${YELLOW}Logs disponíveis em:${NC}"
echo -e "  • Servidor: server_$STRATEGY.log"
echo -e "  • Clientes: client_*_$STRATEGY.log"
echo -e "  • Métricas: metrics/client_*/metrics_history.json"

# Função para mostrar progresso
show_progress() {
    while kill -0 $SERVER_PID 2>/dev/null; do
        if [ -f "server_$STRATEGY.log" ]; then
            CURRENT_ROUND=$(grep -o "round [0-9]*:" server_$STRATEGY.log | tail -1 | grep -o "[0-9]*" || echo "0")
            if [ ! -z "$CURRENT_ROUND" ] && [ "$CURRENT_ROUND" != "0" ]; then
                echo -ne "\r${GREEN}Progresso: Rodada $CURRENT_ROUND de $ROUNDS${NC}    "
            fi
        fi
        sleep 2
    done
}

# Mostrar progresso
show_progress

# Aguardar conclusão
echo -e "\n\n${YELLOW}Aguardando conclusão do treinamento...${NC}"
wait $SERVER_PID
SERVER_EXIT_CODE=$?

# Verificar se o servidor terminou com sucesso
if [ $SERVER_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Servidor concluído com sucesso!${NC}"
else
    echo -e "${RED}✗ Servidor terminou com erro (código: $SERVER_EXIT_CODE)${NC}"
fi

# Aguardar clientes (com timeout)
echo -e "${YELLOW}Aguardando clientes finalizarem...${NC}"
TIMEOUT=30
for pid in ${CLIENT_PIDS[@]}; do
    COUNT=0
    while kill -0 $pid 2>/dev/null && [ $COUNT -lt $TIMEOUT ]; do
        sleep 1
        COUNT=$((COUNT + 1))
    done
    if [ $COUNT -ge $TIMEOUT ]; then
        echo -e "${YELLOW}  Cliente PID $pid demorou muito. Finalizando...${NC}"
        kill $pid 2>/dev/null
    fi
done

# Limpar processos órfãos
echo -e "${YELLOW}Limpando processos...${NC}"
pkill -f "python client.py" 2>/dev/null
pkill -f "python server.py" 2>/dev/null

# Executar análise
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}    EXECUTANDO ANÁLISE DOS RESULTADOS${NC}"
echo -e "${BLUE}================================================${NC}"

if [ -f "analysis_tool.py" ]; then
    echo -e "${GREEN}Gerando visualizações e relatórios...${NC}"
    python analysis_tool.py --results-dir results
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Análise concluída com sucesso!${NC}"
    else
        echo -e "${RED}✗ Erro na análise dos resultados${NC}"
    fi
else
    echo -e "${YELLOW}Arquivo analysis_tool.py não encontrado. Pulando análise.${NC}"
fi

# Resumo final
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}    SIMULAÇÃO CONCLUÍDA${NC}"
echo -e "${BLUE}================================================${NC}"

# Verificar arquivos gerados
echo -e "${GREEN}Arquivos gerados:${NC}"

# Resultados
if [ -d "results" ]; then
    echo -e "\n${YELLOW}Resultados em 'results/':${NC}"
    ls -la results/*.pdf 2>/dev/null | awk '{print "  • " $9}'
    ls -la results/*.csv 2>/dev/null | awk '{print "  • " $9}'
    ls -la results/*.json 2>/dev/null | awk '{print "  • " $9}'
fi

# Métricas dos clientes
if [ -d "metrics" ]; then
    echo -e "\n${YELLOW}Métricas dos clientes em 'metrics/':${NC}"
    for i in {1..3}; do
        if [ -f "metrics/client_$i/metrics_history.json" ]; then
            echo -e "  • Cliente $i: metrics/client_$i/metrics_history.json"
        fi
    done
fi

# Logs
echo -e "\n${YELLOW}Logs de execução:${NC}"
ls -la *.log | awk '{print "  • " $9 " (" $5 " bytes)"}'

# Estatísticas finais dos logs
echo -e "\n${BLUE}Estatísticas Finais:${NC}"

# Extrair perda final do servidor
if [ -f "server_$STRATEGY.log" ]; then
    FINAL_LOSS=$(grep "Perda final de treino:" server_$STRATEGY.log | tail -1 | grep -o "[0-9.]*" | head -1)
    if [ ! -z "$FINAL_LOSS" ]; then
        echo -e "  ${GREEN}• Perda final de treino (global): $FINAL_LOSS${NC}"
    fi
    
    FINAL_EVAL=$(grep "Perda final de validação:" server_$STRATEGY.log | tail -1 | grep -o "[0-9.]*" | head -1)
    if [ ! -z "$FINAL_EVAL" ]; then
        echo -e "  ${GREEN}• Perda final de validação (global): $FINAL_EVAL${NC}"
    fi
fi

# Extrair perdas finais dos clientes
for i in {1..3}; do
    if [ -f "client_${i}_$STRATEGY.log" ]; then
        CLIENT_LOSS=$(grep "Perda de validação:" client_${i}_$STRATEGY.log | tail -1 | grep -o "[0-9.]*" | head -1)
        if [ ! -z "$CLIENT_LOSS" ]; then
            echo -e "  ${GREEN}• Cliente $i - Perda final: $CLIENT_LOSS${NC}"
        fi
    fi
done

echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}✓ Simulação completa!${NC}"
echo -e "${BLUE}================================================${NC}"

# Sugestões
echo -e "\n${YELLOW}Próximos passos sugeridos:${NC}"
echo -e "  1. Visualizar os gráficos em ${GREEN}results/*.pdf${NC}"
echo -e "  2. Analisar métricas detalhadas em ${GREEN}results/detailed_metrics_$STRATEGY.csv${NC}"
echo -e "  3. Revisar o relatório resumido em ${GREEN}results/summary_report.json${NC}"
echo -e "  4. Comparar com outras estratégias executando:"
echo -e "     ${BLUE}./run_enhanced.sh fedadam $ROUNDS${NC}"
echo -e "     ${BLUE}./run_enhanced.sh fedyogi $ROUNDS${NC}"
echo -e "     ${BLUE}./run_enhanced.sh fedadagrad $ROUNDS${NC}"

exit $SERVER_EXIT_CODE