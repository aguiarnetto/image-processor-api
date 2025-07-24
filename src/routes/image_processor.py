import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file
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

        # 1. Converter em tons de cinza
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 2. Duplicar a camada
        gray_copy = gray.copy()

        # 3. Inverter tons da camada superior
        inverted = cv2.bitwise_not(gray_copy)

        # 4. Converter camada superior em subexposição de cores (blend)
        blend = cv2.divide(gray, 255 - inverted, scale=256)

        # 5. Aplicar desfoque gaussiano (2.0 pixels)
        blurred = cv2.GaussianBlur(blend, (0, 0), sigmaX=2)

        # 6. Aplicar níveis de entrada: 223; 1.32; 255
        img_float = blurred.astype(np.float32) / 255.0
        img_gamma = np.clip((img_float - 223/255.0) * (1 / 1.32), 0, 1)
        leveled1 = (img_gamma * 255).astype(np.uint8)

        # 7. Aplicar máscara de nitidez: 200%, 1.4px, limiar 6
        blurred_sharp = cv2.GaussianBlur(leveled1, (0, 0), sigmaX=1.4)
        mask = cv2.subtract(leveled1, blurred_sharp)
        mask = np.where(mask > 6, mask, 0)
        sharpened = cv2.addWeighted(leveled1, 1.0, mask.astype(np.uint8), 2.0, 0)

        # 8. Aplicar níveis de entrada: 0; 0.70; 164
        img_float2 = sharpened.astype(np.float32) / 255.0
        img_gamma2 = np.clip((img_float2 - 0) * (1 / 0.70), 0, 164/255.0)
        final = (img_gamma2 * 255).astype(np.uint8)

        # 9. Salvar
        pil_img = Image.fromarray(final)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
