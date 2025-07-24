from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import io

bp = Blueprint('image_processor', __name__)

@bp.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    result = Image.fromarray(edges)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")