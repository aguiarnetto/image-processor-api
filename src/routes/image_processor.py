import cv2
import numpy as np
from flask import Blueprint, request, send_file
import os
import uuid

image_bp = Blueprint('image_bp', __name__)

@image_bp.route('/process-image', methods=['POST'])
def process_image():
    file = request.files.get('image')
    if not file:
        return 'No image uploaded', 400

    # Ler a imagem
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    # Etapa 1: Tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_step(gray, "1_gray")

    # Etapa 2 e 3: duplicar e inverter
    inverted = cv2.bitwise_not(gray)
    save_step(inverted, "2_inverted")

    # Etapa 4: Subexposição (dodge)
    inverted_blur = cv2.GaussianBlur(inverted, (0, 0), 2.0)
    dodge = cv2.divide(gray, 255 - inverted_blur, scale=256.0)
    save_step(dodge, "3_dodge")

    # Etapa 5: Níveis de entrada: 223;1.32;255
    levels_adjusted = apply_levels(dodge, 223, 1.32, 255)
    save_step(levels_adjusted, "4_levels1")

    # Etapa 6: Máscara de nitidez (200%, raio 1.4, limiar 6)
    sharpened = apply_unsharp_mask(levels_adjusted, 1.4, 6, amount=2.0)
    save_step(sharpened, "5_sharpened")

    # Etapa 7: Níveis finais: 0; 0.70; 164
    final = apply_levels(sharpened, 0, 0.70, 164)
    save_step(final, "6_levels2")

    # Salvar imagem final para envio
    temp_filename = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_filename, final)

    return send_file(temp_filename, mimetype='image/jpeg')


def apply_levels(img, in_black, gamma, in_white):
    # Normalizar imagem
    img = img.astype(np.float32) / 255.0
    img = np.clip((img - in_black / 255.0) / ((in_white - in_black) / 255.0), 0, 1)
    img = np.power(img, 1.0 / gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def apply_unsharp_mask(image, radius, threshold, amount=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    mask = cv2.absdiff(image, blurred)
    low_contrast_mask = mask < threshold
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    sharpened[low_contrast_mask] = image[low_contrast_mask]
    return sharpened

def save_step(img, name):
    path = f"/tmp/{name}.jpg"
    cv2.imwrite(path, img)
