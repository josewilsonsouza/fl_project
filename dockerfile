# Usa uma imagem base do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos da pasta local para dentro do container
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

ENV CLIENT_ID=""
ENV SERVER_ADDRESS=""

# Comando para rodar a aplicação
CMD ["python", "client.py", "--client-id", "$CLIENT_ID", "--server-address", "$SERVER_ADDRESS:8080"]

