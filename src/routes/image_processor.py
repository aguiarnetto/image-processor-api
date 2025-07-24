import cv2
import numpy as np
from flask import Blueprint, request, send_file, jsonify
from io import BytesIO
from PIL import Image

image_bp = Blueprint("image_processor", __name__)

@image_bp.route("/process-image", methods=["POST"])
def process_image():
    try:
        file = request.files["image"]
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img_color = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # 1. Tons de cinza
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 2. Duplicar camada
        duplicated = gray.copy()

        # 3. Inverter camada superior
        inverted = cv2.bitwise_not(duplicated)

        # 4. Subexposição (blending com divisão)
        blend = cv2.divide(gray, 255 - inverted, scale=256)

        # 5. Desfoque gaussiano (raio 2.0)
        blurred = cv2.GaussianBlur(blend, (0, 0), sigmaX=2)

        # 6. Ajuste de níveis: 223;1.32;255
        norm1 = np.clip(((blurred.astype(np.float32) - 223) / (255 - 223)) ** (1/1.32), 0, 1)
        leveled1 = (norm1 * 255).astype(np.uint8)

        # 7. Máscara de nitidez: 200% 1.4px limiar 6
        blur_nitidez = cv2.GaussianBlur(leveled1, (0, 0), sigmaX=1.4)
        mask = leveled1 - blur_nitidez
        mask = np.where(mask > 6, mask, 0).astype(np.uint8)
        sharpened = cv2.addWeighted(leveled1, 1.0, mask, 2.0, 0)

        # 8. Níveis: 0;0.70;164
        norm2 = np.clip((sharpened.astype(np.float32) / 255.0) / 0.70, 0, 164/255.0)
        final = (norm2 * 255).astype(np.uint8)

        # 9. Salvar como PNG
        pil_img = Image.fromarray(final)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
