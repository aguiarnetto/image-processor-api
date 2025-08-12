import os
import uuid
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Blueprint, request, jsonify, send_file
from flask import current_app as app

image_bp = Blueprint("image_bp", __name__)

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
    img = None

    # Tenta obter imagem via form-data (upload de arquivo)
    if "image" in request.files:
        file = request.files["image"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    else:
        # Tenta obter imagem em base64 via JSON
        data = request.get_json(silent=True)
        if data and "image_base64" in data:
            image_b64 = data["image_base64"].split(",")[-1]  # remove header data:image/...
            try:
                img_data = base64.b64decode(image_b64)
                pil_img = Image.open(BytesIO(img_data)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({"error": f"Erro ao decodificar imagem base64: {str(e)}"}), 400

    if img is None:
        return jsonify({"error": "Nenhuma imagem válida enviada"}), 400

    processed_img = photoshop_style_sketch(img)

    filename = f"{uuid.uuid4().hex}.jpg"
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(BASE_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"Imagem salva em: {output_path}")

    return send_file(
        output_path,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="processed.jpg"
    )
