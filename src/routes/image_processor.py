
from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image
import tempfile
import os

image_processor_bp = Blueprint('image_processor', __name__)

def process_image_cv2(image_data):
    """
    Processa a imagem para criar um efeito de arte de linha com mais contraste e segurança.
    """
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Não foi possível decodificar a imagem")

    # Inverter imagem
    inverted_img = cv2.bitwise_not(img)

    # Desfoque leve para suavizar
    blurred = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)

    # Dodge blend
    dodge = cv2.divide(img, 255 - blurred, scale=256)

    # Normalização para melhorar o contraste final
    normalized = cv2.normalize(dodge, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

@image_processor_bp.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    img_bytes = file.read()

    try:
        sketch = process_image_cv2(img_bytes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Salvar em memória com PIL
    pil_img = Image.fromarray(sketch)
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')
