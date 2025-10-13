#!/bin/bash

# Script para executar todas as estratégias e gerar comparação final

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configurações
ROUNDS=${1:-15}
PREDICTION_LENGTH=${2:-10}
STRATEGIES=("fedavg" "fedadam" "fedyogi" "fedadagrad")

echo -e "${PURPLE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║     EXECUÇÃO COMPLETA - TODAS AS ESTRATÉGIAS FL     ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}Rodadas: ${YELLOW}$ROUNDS${NC}"
echo -e "${GREEN}Tamanho da Previsão: ${YELLOW}$PREDICTION_LENGTH passos${NC}"
echo -e "${GREEN}Estratégias: ${YELLOW}${STRATEGIES[@]}${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}\n"

# Criar diretório para resultados consolidados
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_comparison_$TIMESTAMP"
mkdir -p $RESULTS_DIR

# Log geral
MAIN_LOG="$RESULTS_DIR/execution_log.txt"
echo "Execução iniciada em: $(date)" > $MAIN_LOG

# Executar cada estratégia
for STRATEGY in "${STRATEGIES[@]}"; do
    echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Executando estratégia: ${YELLOW}$STRATEGY${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    
    # Registrar no log
    echo -e "\n\n=== ESTRATÉGIA: $STRATEGY ===" >> $MAIN_LOG
    echo "Início: $(date)" >> $MAIN_LOG
    
    # Executar estratégia
    ./run.sh $STRATEGY $ROUNDS $PREDICTION_LENGTH
    
    # Verificar sucesso
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $STRATEGY concluída com sucesso!${NC}"
        echo "Status: SUCESSO" >> $MAIN_LOG
    else
        echo -e "${RED}✗ Erro ao executar $STRATEGY${NC}"
        echo "Status: ERRO" >> $MAIN_LOG
    fi
    
    echo "Fim: $(date)" >> $MAIN_LOG
    
    # Copiar resultados para diretório consolidado
    echo -e "${YELLOW}Copiando resultados...${NC}"
    
    # Criar subdiretório para a estratégia
    STRATEGY_DIR="$RESULTS_DIR/$STRATEGY"
    mkdir -p $STRATEGY_DIR
    
    # Copiar arquivos de resultados
    if [ -d "results" ]; then
        cp -r results/* $STRATEGY_DIR/ 2>/dev/null
    fi
    
    # Copiar métricas dos clientes
    if [ -d "metrics" ]; then
        cp -r metrics $STRATEGY_DIR/ 2>/dev/null
    fi
    
    # Copiar logs
    cp *_$STRATEGY.log $STRATEGY_DIR/ 2>/dev/null
    
    # Renomear arquivos para incluir estratégia
    if [ -f "results/detailed_metrics_$STRATEGY.csv" ]; then
        cp "results/detailed_metrics_$STRATEGY.csv" "$RESULTS_DIR/metrics_$STRATEGY.csv"
    fi
    
    # Aguardar antes da próxima estratégia
    echo -e "${YELLOW}Aguardando 5 segundos antes da próxima estratégia...${NC}"
    sleep 5
done

echo -e "\n${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}     GERANDO ANÁLISE COMPARATIVA FINAL${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"

# Mover todos os CSVs de métricas para results/
echo -e "${YELLOW}Consolidando métricas...${NC}"
cp $RESULTS_DIR/metrics_*.csv results/ 2>/dev/null

# Executar análise comparativa final
echo -e "${GREEN}Executando análise comparativa...${NC}"
python analysis_tool.py --results-dir results

# Copiar análise comparativa para diretório consolidado
if [ -f "results/comparative_analysis.pdf" ]; then
    cp results/comparative_analysis.pdf $RESULTS_DIR/
    echo -e "${GREEN}✓ Análise comparativa gerada!${NC}"
fi

if [ -f "results/summary_report.json" ]; then
    cp results/summary_report.json $RESULTS_DIR/
fi

# Criar relatório resumido em texto
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "════════════════════════════════════════════════════════" > $SUMMARY_FILE
echo "     RESUMO DA COMPARAÇÃO DE ESTRATÉGIAS FL" >> $SUMMARY_FILE
echo "════════════════════════════════════════════════════════" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "Data: $(date)" >> $SUMMARY_FILE
echo "Rodadas: $ROUNDS" >> $SUMMARY_FILE
echo "Tamanho da Previsão: $PREDICTION_LENGTH passos" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "RESULTADOS POR ESTRATÉGIA:" >> $SUMMARY_FILE
echo "----------------------------" >> $SUMMARY_FILE

# Extrair métricas finais de cada estratégia
for STRATEGY in "${STRATEGIES[@]}"; do
    echo "" >> $SUMMARY_FILE
    echo "[$STRATEGY]" >> $SUMMARY_FILE
    
    LOG_FILE="$STRATEGY_DIR/$STRATEGY/server_$STRATEGY.log"
    if [ -f "$LOG_FILE" ]; then
        FINAL_TRAIN=$(grep "Perda final de treino:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        FINAL_VAL=$(grep "Perda final de validação:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        IMPROVEMENT=$(grep "Melhoria no treino:" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+%" | head -1)
        
        echo "  Perda Final (Treino): ${FINAL_TRAIN:-N/A}" >> $SUMMARY_FILE
        echo "  Perda Final (Validação): ${FINAL_VAL:-N/A}" >> $SUMMARY_FILE
        echo "  Melhoria: ${IMPROVEMENT:-N/A}" >> $SUMMARY_FILE
    else
        echo "  Status: Dados não disponíveis" >> $SUMMARY_FILE
    fi
done

echo "" >> $SUMMARY_FILE
echo "════════════════════════════════════════════════════════" >> $SUMMARY_FILE

# Exibir resumo no terminal
cat $SUMMARY_FILE

# Estatísticas finais
echo -e "\n${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}     EXECUÇÃO COMPLETA${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"

echo -e "\n${GREEN}Resultados salvos em: ${YELLOW}$RESULTS_DIR/${NC}"
echo -e "${GREEN}Estrutura de arquivos:${NC}"
echo -e "  ${YELLOW}$RESULTS_DIR/${NC}"
echo -e "    ├── ${BLUE}summary.txt${NC} - Resumo geral"
echo -e "    ├── ${BLUE}execution_log.txt${NC} - Log de execução"
echo -e "    ├── ${BLUE}comparative_analysis.pdf${NC} - Análise comparativa"
echo -e "    ├── ${BLUE}summary_report.json${NC} - Relatório detalhado"

for STRATEGY in "${STRATEGIES[@]}"; do
    echo -e "    ├── ${GREEN}$STRATEGY/${NC}"
    echo -e "    │   ├── Resultados específicos"
    echo -e "    │   ├── Métricas dos clientes"
    echo -e "    │   └── Logs de execução"
done

echo -e "\n${YELLOW}Visualizações disponíveis:${NC}"
echo -e "  • ${GREEN}comparative_analysis.pdf${NC} - Comparação entre estratégias"
echo -e "  • ${GREEN}client_evolution_analysis.pdf${NC} - Evolução dos clientes"
echo -e "  • ${GREEN}performance_analysis_*.pdf${NC} - Análise por estratégia"
echo -e "  • ${GREEN}convergence_analysis_*.pdf${NC} - Análise de convergência"
echo -e "  • ${GREEN}heatmap_performance_*.pdf${NC} - Mapas de calor"

# Determinar melhor estratégia
echo -e "\n${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}     MELHOR ESTRATÉGIA${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"

BEST_STRATEGY=""
BEST_LOSS=999999

for STRATEGY in "${STRATEGIES[@]}"; do
    CSV_FILE="results/detailed_metrics_$STRATEGY.csv"
    if [ -f "$CSV_FILE" ]; then
        # Pegar última linha com fase 'eval' e coluna 'global_eval_loss'
        FINAL_LOSS=$(tail -n 20 "$CSV_FILE" | grep "eval" | tail -1 | cut -d',' -f3)
        
        if [ ! -z "$FINAL_LOSS" ]; then
            # Comparar usando bc para números decimais
            if (( $(echo "$FINAL_LOSS < $BEST_LOSS" | bc -l) )); then
                BEST_LOSS=$FINAL_LOSS
                BEST_STRATEGY=$STRATEGY
            fi
        fi
    fi
done

if [ ! -z "$BEST_STRATEGY" ]; then
    echo -e "${GREEN}Melhor estratégia: ${YELLOW}$BEST_STRATEGY${NC}"
    echo -e "${GREEN}Perda final de validação: ${YELLOW}$BEST_LOSS${NC}"
else
    echo -e "${YELLOW}Não foi possível determinar a melhor estratégia${NC}"
fi

echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Comparação completa de todas as estratégias!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

# Fim
echo "Execução finalizada em: $(date)" >> $MAIN_LOG
exit 0