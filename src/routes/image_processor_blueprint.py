import logging
import io
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS  # Import para habilitar CORS

# ========================
# PARÂMETROS EDITÁVEIS
# ========================
THRESH_MAX = 200   # Reduz o branco máximo (Photoshop: níveis 255 → 200)
SHARPEN_FACTOR = 0.4  # 0.4 deixa 60% menos nítido que antes
BILATERAL_D = 9       # Suavização sem borrar bordas
BILATERAL_SIGMA = 75  # Força da suavização

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

            # Converter para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Suavização preservando bordas (remove sujeira)
            gray = cv2.bilateralFilter(gray, d=BILATERAL_D,
                                       sigmaColor=BILATERAL_SIGMA,
                                       sigmaSpace=BILATERAL_SIGMA)

            # Sharpen mais suave (60% menos que antes)
            sharpened = cv2.addWeighted(
                gray, 1 + SHARPEN_FACTOR,
                cv2.GaussianBlur(gray, (0, 0), 5),
                -SHARPEN_FACTOR,
                0
            )

            # Remover cinzas → limitar nível de branco a 200
            _, processed_image = cv2.threshold(
                sharpened, THRESH_MAX, 255, cv2.THRESH_BINARY
            )

            _logger.info("Imagem processada com sucesso.")

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
