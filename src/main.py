from flask import Flask
from flask_cors import CORS
from routes.image_processor import image_processor

app = Flask(__name__)
CORS(app)
app.register_blueprint(image_processor)

@app.route("/")
def home():
    return "API de Processamento de Imagem está rodando!"

if __name__ == "__main__":
    app.run(debug=True)