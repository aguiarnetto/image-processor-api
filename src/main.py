import sys
import os

# Garante que o diretório atual está no sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS
from routes.image_processor import image_bp  # Importa seu blueprint de rotas

app = Flask(__name__)
CORS(app)

# Registro do blueprint com prefixo /api
app.register_blueprint(image_bp, url_prefix="/api")

# Rota raiz opcional para verificação rápida
@app.route("/")
def index():
    return "API está rodando com sucesso!"

# Porta dinâmica para compatibilidade com Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
