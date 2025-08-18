# routes/image_processor.py
import io
import logging
import psutil
from flask import Blueprint, request, send_file, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_bp = Blueprint("image_processor", __name__)

def apply_levels(img, in_black, in_gamma, in_white, out_black=0, out_white=255):
    """Replica a função Levels do Photoshop."""
    img = img.astype(np.float32)
    img = np.clip((img - in_black) / (in_white - in_black + 1e-8), 0, 1)  # evita divisão por zero
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
        # Verifica envio
        if "image" not in request.files:
            logger.error("Nenhum arquivo enviado")
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files["image"]

        # Abrir imagem
        try:
            image = Image.open(file.stream).convert("RGB")
        except UnidentifiedImageError:
            logger.error("Arquivo enviado não é uma imagem válida")
            return jsonify({"error": "Arquivo inválido. Envie um PNG ou JPG"}), 400

        logger.info(f"Tamanho original: {image.size}, formato: {image.format}")
        np_image = np.array(image)

        # === PASSO 1: Converter para tons de cinza ===
        logger.info("Convertendo para tons de cinza")
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        logger.info(f"Gray min/max: {gray.min()}/{gray.max()}")

        # === PASSO 2: Níveis de entrada 0 ; 2,25 ; 255 ===
        logger.info("Aplicando levels inicial")
        gray = apply_levels(gray, 0, 2.25, 255)
        logger.info(f"Min/max após levels inicial: {gray.min()}/{gray.max()}")

        # Ajuste suave para intensificar preto e reduzir cinza sem perder detalhes
        logger.info("Aplicando ajuste suave preto/cinza")
        gray = apply_levels(gray, 25, 1.0, 230)
        logger.info(f"Min/max após ajuste suave: {gray.min()}/{gray.max()}")

        # === PASSO 3: Máscara de nitidez 1 ===
        logger.info("Aplicando unsharp mask 1")
        gray = unsharp_mask(gray, radius=3.4, amount=5.0, threshold=0)
        logger.info(f"Min/max após unsharp 1: {gray.min()}/{gray.max()}")

        # === PASSO 4: Máscara de nitidez 2 ===
        logger.info("Aplicando unsharp mask 2")
        gray = unsharp_mask(gray, radius=2.8, amount=5.0, threshold=0)
        logger.info(f"Min/max após unsharp 2: {gray.min()}/{gray.max()}")

        # === PASSO 5: Levels finos 38 ; 3,48 ; 174 ===
        logger.info("Aplicando levels finos")
        gray = apply_levels(gray, 38, 3.48, 174)
        logger.info(f"Min/max após levels finos: {gray.min()}/{gray.max()}")

        # === PASSO 6: Níveis de saída 33 ; 255 ===
        logger.info("Aplicando níveis de saída")
        gray = apply_levels(gray, 0, 1.0, 255, out_black=33, out_white=255)
        logger.info(f"Min/max após níveis de saída: {gray.min()}/{gray.max()}")

        # Ajuste extra final
        logger.info("Aplicando ajuste final preto intenso e nitidez")
        gray = apply_levels(gray, 40, 1.0, 200, out_black=0, out_white=255)
        gray = unsharp_mask(gray, radius=1.5, amount=3.0, threshold=0)
logger.info(f"Min/max final: {gray.min()}/{gray.max()}")
