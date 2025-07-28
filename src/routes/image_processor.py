import os
import uuid
import cv2
import numpy as np
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
    duplicated = gray.copy()
    inverted = cv2.bitwise_not(duplicated)
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=2.0, sigmaY=2.0)
    dodged = color_dodge(gray, blurred)
    leveled = apply_levels(dodged, black=223, gamma=1.32, white=255)
    sharpened = apply_unsharp_mask(leveled, amount=2.0, radius=1.4, threshold=6)
    final = apply_levels(sharpened, black=0, gamma=0.70, white=164)
    return final

@image_bp.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Falha ao processar a imagem"}), 400

    processed_img = photoshop_style_sketch(img)

    filename = f"{uuid.uuid4().hex}.png"

    # Gera caminho absoluto para pasta 'outputs'
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(BASE_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, processed_img)

    print(f"Imagem salva em: {output_path}")  # Log opcional para debug

    return send_file(output_path, mimetype="image/png")
