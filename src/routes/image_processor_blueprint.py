import logging
import io
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS  # Import para habilitar CORS

# Configuração do Blueprint
image_bp = Blueprint("image_processor", __name__)
_logger = logging.getLogger(__name__)

@image_bp.route("/process", methods=["POST"])
def process_image():
    _logger.info("Requisição de processamento de imagem recebida.")
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo 'file' na requisição"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    if file:
        try:
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({"error": "Falha ao decodificar a imagem"}), 400

            # -------------------------------
            # Processamento da imagem
            # -------------------------------
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Suavizar ruídos
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Detecção de bordas (Laplacian dá traço mais "desenhado")
            edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=5)

            # Inverter para deixar linhas escuras em fundo claro
            edges_inv = cv2.bitwise_not(edges)

            # Combinar bordas com a imagem original em tons de cinza
            processed_image = cv2.addWeighted(gray_image, 0.7, edges_inv, 0.3, 0)

            # Clarear em 80% (reduzir cinza)
            processed_image = cv2.convertScaleAbs(processed_image, alpha=0.2, beta=0)

            # -------------------------------
            # Exportar como JPG
            # -------------------------------
            _, img_encoded = cv2.imencode(".jpg", processed_image)
            response = img_encoded.tobytes()

            return send_file(
                io.BytesIO(response),
                mimetype="image/jpeg",
                as_attachment=False,
                download_name="processed_image.jpg"
            )
        except Exception as e:
            _logger.exception(f"Erro durante o processamento da imagem: {e}")
            return jsonify({"error": f"Erro ao processar a imagem: {str(e)}"}), 500

    return jsonify({"error": "Erro desconhecido no upload do arquivo"}), 500


# Criação da aplicação Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Registro do Blueprint
app.register_blueprint(image_bp)

if __name__ == '__main__':
    app.run(debug=True)
