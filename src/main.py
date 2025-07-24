import sys
import os

# Adiciona o diretório onde este arquivo está ao sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS
from routes.image_processor import image_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(image_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
