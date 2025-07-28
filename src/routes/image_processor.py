from flask import Blueprint, request, jsonify, send_file
from PIL import Image, ImageFilter, ImageOps
import io
import base64

image_processor = Blueprint('image_processor', __name__)

@image_processor.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert("L")  # Escala de cinza
        image = ImageOps.invert(image)  # Inverter cor
        image = image.filter(ImageFilter.CONTOUR)  # Filtro traçado

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

        return jsonify({'image_base64': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 500