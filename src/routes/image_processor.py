import os
import uuid
import base64
import cv2
import numpy as np
import logging
import psutil
from io import BytesIO
from PIL import Image
from flask import Blueprint, request, jsonify, send_file
from flask import current_app as app

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

image_bp = Blueprint("image_bp", __name__)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Uso atual de memória: {mem:.2f} MB")

def color_dodge(base, blend):
    result = cv2.divide(base, 255 - blend, scale=256)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_levels(img, black, gamma, white):
    img = img.astype(np.float32) / 255.0
    img = np.clip((img - black / 255) * (1 / gamma) * (255 / (white - black)), 0, 1)
    return (img * 255).astype(np.uint8)

def apply_unsharp_mask(image, amount=2.0, radius=1.4, threshold=6):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    mask = cv2.subtract(image, blurred)
    sharpened = np.where(np.abs(mask) > threshold, image + amount * mask, image)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def photoshop_style_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=2.0, sigmaY=2.0)
    dodged = color_dodge(gray, blurred)
    leveled = apply_levels(dodged, black=223, gamma=1.32, white=255)
    sharpened = apply_unsharp_mask(leveled, amount=2.0, radius=1.4, threshold=6)
    final = apply_levels(sharpened, black=0, gamma=0.70, white=164)

    rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    return rgb

@image_bp.route("/process-image", methods=["POST"])
def process_image():
    logger.info("Recebida requisição para processar imagem")
    log_memory_usage()

    img = None

    if "image" in request.files:
        file = request.files["image"]
        logger.info(f"Arquivo recebido: {file.filename}, tamanho: {request.content_length} bytes")
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    else:
        data = request.get_json(silent=True)
        if data and "image_base64" in data:
            logger.info("Imagem recebida em base64 no JSON")
            image_b64 = data["image_base64"].split(",")[-1]
            try:
                img_data = base64.b64decode(image_b64)
                pil_img = Image.open(BytesIO(img_data)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Erro ao decodificar imagem base64: {e}")
                return jsonify({"error": f"Erro ao decodificar imagem base64: {str(e)}"}), 400

    if img is None:
        logger.warning("Nenhuma imagem válida enviada na requisição")
        return jsonify({"error": "Nenhuma imagem válida enviada"}), 400

    logger.info("Imagem recebida e decodificada com sucesso. Iniciando processamento.")
    log_memory_usage()

    processed_img = photoshop_style_sketch(img)

    filename = f"{uuid.uuid4().hex}.jpg"
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(BASE_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    logger.info(f"Imagem salva em: {output_path}")
    log_memory_usage()

    return send_file(
        output_path,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="processed.jpg"
    )
