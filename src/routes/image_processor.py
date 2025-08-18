# routes/image_processor.py
import io
import logging
import psutil
from flask import Blueprint, request, send_file, jsonify
from PIL import Image
import numpy as np
import cv2
import time
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_bp = Blueprint("image_processor", __name__)

def apply_levels(img, in_black, in_gamma, in_white, out_black=0, out_white=255):
    """Replica a função Levels do Photoshop com proteção contra erros."""
    try:
        img = img.astype(np.float32)
        img = np.clip((img - in_black) / (in_white - in_black + 1e-6), 0, 1)
        img = np.power(img, 1.0 / (in_gamma + 1e-6))
        img = img * (out_white - out_black) + out_black
        return np.clip(img, 0, 255).astype(np.uint8)
    except Exception:
        logger.error(f"Erro em apply_levels: in_black={in_black}, in_gamma={in_gamma}, in_white={in_white}, out_black={out_black}, out_white={out_white}")
        raise

def unsharp_mask(image, radius, amount, threshold):
    """Replica Máscara de Nitidez do Photoshop com conversão segura."""
    try:
        img = image.astype(np.float32)
        blurred = cv2.GaussianBlur(img, (0, 0), radius)
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        if threshold > 0:
            low_contrast_mask = np.abs(img - blurred) < threshold
            np.copyto(sharpened, img, where=low_contrast_mask)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception:
        logger.error(f"Erro em unsharp_mask: radius={radius}, amount={amount}, threshold={threshold}")
        raise

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

        # --- Passo 1: Converter para tons de cinza ---
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        logger.info(f"Passo 1: tons de cinza OK, shape={gray.shape}")

        # --- Passo 2: Níveis de entrada 0;2,25;255 ---
        gray = apply_levels(gray, 0, 2.25, 255)
        logger.info(f"Passo 2: apply_levels 0;2.25;255 OK")

        # 🔹 Ajuste suave para intensificar preto e reduzir cinza
        gray = apply_levels(gray, in_black=25, in_gamma=1.0, in_white=230)
        logger.info("Ajuste extra: preto mais intenso sem perder detalhes OK")

        # --- Passo 3: Máscara de nitidez 500%, raio 3.4, limiar 0 ---
        gray = unsharp_mask(gray, radius=3.4, amount=5.0, threshold=0)
        logger.info("Passo 3: unsharp_mask raio 3.4 OK")

        # --- Passo 4: Reaplicar máscara de nitidez 500%, raio 2.8 ---
        gray = unsharp_mask(gray, radius=2.8, amount=5.0, threshold=0)
        logger.info("Passo 4: unsharp_mask raio 2.8 OK")

        # --- Passo 5: Níveis de entrada 38;3.48;174 ---
        gray = apply_levels(gray, 38, 3.48, 174)
        logger.info("Passo 5: apply_levels 38;3.48;174 OK")

        # --- Passo 6: Níveis de saída 33;255 ---
        gray = apply_levels(gray, 0, 1.0, 255, out_black=33, out_white=255)
        logger.info("Passo 6: níveis de saída 33;255 OK")

        # 🔹 Ajuste extra final para preto mais intenso e menos cinza
        gray = apply_levels(gray, in_black=40, in_gamma=1.0, in_white=200, out_black=0, out_white=255)
        gray = unsharp_mask(gray, radius=1.5, amount=3.0, threshold=0)
        logger.info("Ajuste final: preto intenso e sharpness extra OK")

        result_img = Image.fromarray(gray)
        img_io = io.BytesIO()
        result_img.save(img_io, "PNG", quality=100)
        img_io.seek(0)

        process_time = round(time.time() - start_time, 2)
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Processamento concluído em {process_time}s | Memória: {mem:.2f} MB")
        logger.info(f"Min/max final: {gray.min()}/{gray.max()}")

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Erro completo no processamento:\n{tb_str}")
        return jsonify({"error": str(e), "traceback": tb_str}), 500
