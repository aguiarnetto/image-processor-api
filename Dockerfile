# Etapa 1: Usar uma imagem base Python oficial e leve
FROM python:3.11-slim-bullseye

# Etapa 2: Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Etapa 3: Instalar as dependências do sistema operacional para OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Etapa 4: Copiar o arquivo de requisitos e instalar as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Etapa 5: Copiar todo o código da sua aplicação para o contêiner
COPY . .

# Etapa 6: Expor a porta que a aplicação usará (informativo)
EXPOSE 10000

# Etapa 7: Comando para iniciar a aplicação
# ALTERAÇÃO AQUI: de 'app:app' para 'src.app:app'
CMD gunicorn src.app:app --bind "0.0.0.0:$PORT" --timeout 120
