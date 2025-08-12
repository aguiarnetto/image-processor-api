# routes/image_processor.py
import io
import logging
import psutil
from flask import Blueprint, request, send_file, jsonify
from PIL import Image
import numpy as np
import cv2
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_bp = Blueprint("image_processor", __name__)

def apply_levels(img, in_black, in_gamma, in_white, out_black=0, out_white=255):
    """Replica a função Levels do Photoshop."""
    img = img.astype(np.float32)
    img = np.clip((img - in_black) / (in_white - in_black), 0, 1)
    img = np.power(img, 1.0 / in_gamma)
    img = img * (out_white - out_black) + out_black
    return np.clip(img, 0, 255).astype(np.uint8)

def unsharp_mask(image, radius, amount, threshold):
    """Replica Máscara de Nitidez do Photoshop."""
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return np.clip(sharpened, 0, 255)

@image_bp.route("/process-image", methods=["POST"])
def process_image():
    start_time = time.time()
    logger.info("Início do processamento de imagem")

    try:
        if "image" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        logger.info(f"Tamanho original: {image.size}, formato: {image.format}")

        np_image = np.array(image)

        # Passo 1 - Converter para tons de cinza
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # Passo 2 - Níveis de entrada: 0 ; 2,25 ; 255
        gray = apply_levels(gray, 0, 2.25, 255)

        # 🔹 Intensificar contraste em 30% antes dos Unsharp Masks
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)

        # Passo 3 - Máscara de nitidez: intensidade 500%, raio 3.4, limiar 0
        gray = unsharp_mask(gray, radius=3.4, amount=5.0, threshold=0)

        # Passo 4 - Reaplicar máscara de nitidez: intensidade 500%, raio 2.8, limiar 0
        gray = unsharp_mask(gray, radius=2.8, amount=5.0, threshold=0)

        # Passo 5 - Níveis de entrada: 38 ; 3,48 ; 174
        gray = apply_levels(gray, 38, 3.48, 174)

        # Passo 6 - Níveis de saída: 33 ; 255
        gray = apply_levels(gray, 0, 1.0, 255, out_black=33, out_white=255)

        # 🔹 Ajuste extra final para preto mais intenso e menos cinza
        gray = apply_levels(gray, in_black=40, in_gamma=1.0, in_white=200, out_black=0, out_white=255)
        gray = unsharp_mask(gray, radius=1.5, amount=3.0, threshold=0)

        # Passo 7 - Salvar
        result_img = Image.fromarray(gray)
        img_io = io.BytesIO()
        result_img.save(img_io, "PNG", quality=100)
        img_io.seek(0)

        process_time = round(time.time() - start_time, 2)
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Processamento concluído em {process_time}s | Memória: {mem:.2f} MB")

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        logger.exception("Erro ao processar imagem")
        return jsonify({"error": str(e)}), 500
