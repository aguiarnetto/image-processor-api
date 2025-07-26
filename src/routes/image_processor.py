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

    # Etapa 2: Níveis de entrada: 0; 2.25; 255
    step2 = apply_levels(gray, 0, 2.25, 255)
    save_step(step2, "2_levels_0_2.25_255")

    # Etapa 3: Máscara de nitidez: intensidade 500%, raio 3.4, limiar 0
    step3 = apply_unsharp_mask(step2, radius=3.4, threshold=0, amount=5.0)
    save_step(step3, "3_sharpen_3.4")

    # Etapa 4: Reaplicar máscara de nitidez: intensidade 500%, raio 2.8, limiar 0
    step4 = apply_unsharp_mask(step3, radius=2.8, threshold=0, amount=5.0)
    save_step(step4, "4_sharpen_2.8")

    # Etapa 5: Níveis de entrada: 38; 3.48; 174
    step5 = apply_levels(step4, 38, 3.48, 174)
    save_step(step5, "5_levels_38_3.48_174")

    # Etapa 6: Níveis de saída: 33; 255
    step6 = apply_output_levels(step5, 33, 255)
    save_step(step6, "6_output_levels")

    # Etapa 7: Salvar imagem final
    temp_filename = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_filename, step6, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return send_file(temp_filename, mimetype='image/jpeg')

def apply_levels(img, in_black, gamma, in_white):
    img = img.astype(np.float32) / 255.0
    img = np.clip((img - in_black / 255.0) / ((in_white - in_black) / 255.0), 0, 1)
    img = np.power(img, 1.0 / gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def apply_output_levels(img, out_black, out_white):
    img = img.astype(np.float32) / 255.0
    img = img * ((out_white - out_black) / 255.0) + (out_black / 255.0)
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
