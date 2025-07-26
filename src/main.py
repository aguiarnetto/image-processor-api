from flask import Flask
from flask_cors import CORS  # ✅ Importa o CORS
from src.routes.image_processor import image_bp

app = Flask(__name__)

# ✅ Ativa CORS globalmente
CORS(app)

# ✅ Registra o blueprint
app.register_blueprint(image_bp)

# (opcional) executa localmente
if __name__ == '__main__':
    app.run(debug=True)
