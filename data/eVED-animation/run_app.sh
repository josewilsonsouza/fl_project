#!/bin/bash
# Script para iniciar a aplicação Streamlit no Linux/Mac

echo "========================================"
echo " eVED Trajectory Viewer - Streamlit"
echo "========================================"
echo ""

# Verificar se streamlit está instalado
if ! python -c "import streamlit" 2>/dev/null; then
    echo "[ERRO] Streamlit não encontrado!"
    echo ""
    echo "Instalando dependências..."
    pip install -r requirements.txt
    echo ""
fi

echo "Iniciando aplicação Streamlit..."
echo "A aplicação abrirá automaticamente no navegador."
echo ""
echo "Para parar, pressione Ctrl+C"
echo "========================================"
echo ""

streamlit run app_streamlit.py
