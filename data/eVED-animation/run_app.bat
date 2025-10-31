@echo off
REM Script para iniciar a aplicação Streamlit no Windows

echo ========================================
echo  eVED Trajectory Viewer - Streamlit
echo ========================================
echo.

REM Verificar se streamlit está instalado
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [ERRO] Streamlit nao encontrado!
    echo.
    echo Instalando dependencias...
    pip install -r requirements.txt
    echo.
)

echo Iniciando aplicacao Streamlit...
echo A aplicacao abrira automaticamente no navegador.
echo.
echo Para parar, pressione Ctrl+C
echo ========================================
echo.

streamlit run app_streamlit.py

pause
