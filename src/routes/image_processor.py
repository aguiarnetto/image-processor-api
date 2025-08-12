import os
import io
import psutil
import logging
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Configuração básica do Flask
app = Flask(__name__)
CORS(app)

# Configuração de logs
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log_event(message):
    """Grava evento no log com uso de memória."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / (1024 * 1024)
    logging.info(f"{message} | Memória: {mem_info:.2f} MB")

def process_image(image_bytes):
    """Processa imagem para arte em linha otimizada."""
    log_event("Iniciando processamento de imagem")

    # Ler imagem
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionar se necessário
    max_size = 1024
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Converter para cinza e suavizar
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bordas fixas
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Inverter para estilo "arte em linha"
    inverted = cv2.bitwise_not(edges)

    log_event("Processamento concluído")

    # Retornar imagem como PNG
    pil_img = Image.fromarray(inverted)
    output = io.BytesIO()
    pil_img.save(output, format="PNG")
    output.seek(0)
    return output

@app.route("/process", methods=["POST"])
def process_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        processed_image = process_image(image_bytes)
        return send_file(processed_image, mimetype="image/png")
    except Exception as e:
        log_event(f"Erro ao processar imagem: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / (1024 * 1024)
    return jsonify({
        "status": "OK",
        "memory_usage_MB": round(mem_info, 2),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
