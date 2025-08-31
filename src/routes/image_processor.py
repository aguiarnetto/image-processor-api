import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configurações
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Criar pastas se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, output_path):
    # Carregar imagem em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Suavizar para reduzir ruído
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Técnica de "dodge" (clareamento base)
    img_dodge = cv2.divide(img, img_blur, scale=256)

    # Destacar linhas com detecção de bordas
    edges = cv2.Canny(img, 50, 150)

    # Inverter para linhas pretas no fundo branco
    edges_inv = cv2.bitwise_not(edges)

    # Combinar resultado (dodge + linhas)
    final = cv2.bitwise_and(img_dodge, edges_inv)

    # Binarizar para estilo decalque forte
    final = cv2.adaptiveThreshold(
        final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5
    )

    cv2.imwrite(output_path, final)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], filename)
        file.save(input_path)

        # Processar imagem
        process_image(input_path, output_path)

        return send_file(output_path, mimetype="image/png")

    return jsonify({"error": "Tipo de arquivo não permitido"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
