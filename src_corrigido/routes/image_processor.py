from flask import Blueprint, request, jsonify, send_file
from PIL import Image, ImageDraw
import numpy as np
import io

image_bp = Blueprint("image_processor", __name__)

@image_bp.route("/process", methods=["POST"])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    try:
        img = Image.open(file.stream).convert("RGB")
        img_array = np.array(img)

        # Exemplo simples de detecção de borda em tons de cinza
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        edges = np.abs(np.diff(gray, axis=0)).astype(np.uint8)
        edges = np.pad(edges, ((0,1),(0,0)), mode='constant')

        traced = Image.fromarray(np.stack([edges]*3, axis=2))
        traced = traced.resize(img.size)

        output = io.BytesIO()
        traced.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png")
    except Exception as e:
        return jsonify({'error': str(e)}), 500