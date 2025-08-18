import sys
import os
import logging
from flask import Flask
from flask_cors import CORS

# === Ajuste de path ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# === Importa blueprint ===
try:
    from routes.image_processor import image_bp
except ImportError as e:
    raise ImportError(f"Erro ao importar 'image_bp' de routes.image_processor: {e}")

# === Configuração da aplicação ===
app = Flask(__name__)

# Habilita CORS apenas para /api/*
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Registro do blueprint com prefixo /api
app.register_blueprint(image_bp, url_prefix="/api")

# Rota raiz opcional para verificação rápida
@app.route("/")
def index():
    return {"status": "ok", "message": "API está rodando com sucesso!"}

# Configuração de logs (útil no Render)
logging.basicConfig(level=logging.INFO)

# Porta dinâmica para compatibilidade com Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
