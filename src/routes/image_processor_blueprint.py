import logging
import io
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS

# ========================
# PARÂMETROS EDITÁVEIS
# ========================
GAMMA = 1.9        # Clareamento dos médios
BLOCKSIZE = 15     # Threshold adaptativo (quanto menor, mais contraste)
C = 7              # Constante do adaptativo (maior = mais branco)
SHARPEN = 0.15     # Nitidez (0.1 suave, 0.3 forte)
UPSCALE = 2.0      # Aumentar resolução
WHITE_CUTOFF = 200 # <<< ponto onde o branco "clipa" (antes 255)

# Configuração do Blueprint
image_bp = Blueprint("image_processor", __name__)
_logger = logging.getLogger(__name__)

@image_bp.route("/process", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo 'file' na requisição"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Falha ao decodificar a imagem"}), 400

        # Aumentar resolução
        if UPSCALE > 1.0:
            image = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

        # Cinza + blur
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Bordas
        edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=5)
        edges_inv = cv2.bitwise_not(edges)

        processed_image = cv2.addWeighted(gray_image, 0.7, edges_inv, 0.3, 0)

        # Gamma correction
        lookup_table = np.array([
            ((i / 255.0) ** (1.0 / GAMMA)) * 255
            for i in np.arange(256)
        ]).astype("uint8")
        processed_image = cv2.LUT(processed_image, lookup_table)

        # Threshold adaptativo
        adaptive = cv2.adaptiveThreshold(
            processed_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            BLOCKSIZE,
            C
        )
        processed_image = cv2.addWeighted(processed_image, 0.5, adaptive, 0.5, 0)

        # 🔥 Nivelamento igual ao Photoshop (255 → 200)
        _, processed_image = cv2.threshold(processed_image, WHITE_CUTOFF, 255, cv2.THRESH_BINARY)

        # Nitidez (Unsharp Mask)
        if SHARPEN > 0:
            blur = cv2.GaussianBlur(processed_image, (0, 0), 3)
            processed_image = cv2.addWeighted(processed_image, 1 + SHARPEN, blur, -SHARPEN, 0)

        # Salvar em JPG
        _, img_encoded = cv2.imencode(".jpg", processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        response = img_encoded.tobytes()

        return send_file(
            io.BytesIO(response),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="processed_image.jpg"
        )

    except Exception as e:
        return jsonify({"error": f"Erro ao processar a imagem: {str(e)}"}), 500


# App Flask
app = Flask(__name__)
CORS(app)
app.register_blueprint(image_bp)

if __name__ == '__main__':
    app.run(debug=True)
