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

# --- COMANDOS DE DEBUG AQUI ---
RUN ls -la /app
RUN echo $PYTHONPATH
RUN python3.11 -c "import sys; print(sys.path)"
# --- FIM DOS COMANDOS DE DEBUG ---

# Etapa 6: Expor a porta que a aplicação usará (informativo)
EXPOSE 10000

# Forcar rebuild - Remova esta linha após a depuração
# Etapa 7: Comando para iniciar a aplicação
CMD gunicorn --pythonpath . app:app --bind "0.0.0.0:$PORT" --timeout 120
