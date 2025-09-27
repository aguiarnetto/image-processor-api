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
        _logger.error("Nenhum arquivo 'file' na requisição.")
        return jsonify({"error": "Nenhum arquivo 'file' na requisição"}), 400

    file = request.files["file"]
    if file.filename == "":
        _logger.error("Nome de arquivo vazio.")
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    if file:
        try:
            _logger.info(f"Arquivo recebido: {file.filename}")
            # Ler a imagem
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            _logger.info("Imagem decodificada com sucesso.")

            if image is None:
                _logger.error("Falha ao decodificar a imagem. Verifique o formato.")
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

            # ============================
            # Clarear médios (Gamma 1.5 ≈ reduz ~50%)
            # ============================
            gamma = 1.5
            lookup_table = np.array([
                ((i / 255.0) ** (1.0 / gamma)) * 255 
                for i in np.arange(256)
            ]).astype("uint8")
            processed_image = cv2.LUT(processed_image, lookup_table)

            # ============================
            # Aumentar nitidez (+20%)
            # ============================
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            sharpened = cv2.filter2D(processed_image, -1, kernel)
            processed_image = cv2.addWeighted(processed_image, 0.8, sharpened, 0.2, 0)

            _logger.info("Imagem processada com sucesso (tons médios reduzidos e nitidez aplicada).")

            # Codificar a imagem processada para JPG
            _, img_encoded = cv2.imencode(".jpg", processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            response = img_encoded.tobytes()
            _logger.info("Imagem codificada para JPG.")

            return send_file(
                io.BytesIO(response),
                mimetype="image/jpeg",
                as_attachment=False,
                download_name="processed_image.jpg"
            )
        except Exception as e:
            _logger.exception(f"Erro durante o processamento da imagem: {e}")
            return jsonify({"error": f"Erro ao processar a imagem: {str(e)}"}), 500

    _logger.error("Condição de arquivo não atendida.")
    return jsonify({"error": "Erro desconhecido no upload do arquivo"}), 500


# Criação da aplicação Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Registro do Blueprint
app.register_blueprint(image_bp)

if __name__ == '__main__':
    app.run(debug=True)
