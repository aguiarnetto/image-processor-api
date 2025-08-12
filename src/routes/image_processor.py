# routes/image_processor.py
import io
import logging
import psutil
from flask import Blueprint, request, send_file, jsonify
from PIL import Image, ImageOps
import numpy as np
import cv2
import time

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cria o blueprint
image_bp = Blueprint("image_processor", __name__)

@image_bp.route("/process-image", methods=["POST"])
def process_image():
    start_time = time.time()
    logger.info("Início do processamento de imagem")

    try:
        # Verifica se o arquivo foi enviado
        if "image" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files["image"]

        # Abre a imagem
        image = Image.open(file.stream).convert("RGB")
        logger.info(f"Tamanho original: {image.size}, formato: {image.format}")

        # Converte para array OpenCV
        np_image = np.array(image)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # --- PROCESSAMENTO FIXO PARA RESULTADO CONSISTENTE ---
        # Ajusta contraste e brilho fixos
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # Aplica detecção de bordas Canny com parâmetros fixos
        edges = cv2.Canny(gray, threshold1=80, threshold2=150)

        # Inverte as cores (para ficar linhas pretas em fundo branco)
        edges = cv2.bitwise_not(edges)

        # Converte de volta para PIL
        result_img = Image.fromarray(edges)

        # Salva em buffer
        img_io = io.BytesIO()
        result_img.save(img_io, "PNG", quality=100)
        img_io.seek(0)

        end_time = time.time()
        process_time = round(end_time - start_time, 2)

        # Log de uso de memória
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Processamento concluído em {process_time}s | Memória: {mem:.2f} MB")

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.exception("Erro ao processar imagem")
        return jsonify({"error": str(e)}), 500
