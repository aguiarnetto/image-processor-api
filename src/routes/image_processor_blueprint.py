import logging
import io
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS

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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Clareamento via correção gama (remove tons médios)
            gamma = 3.0
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
            gamma_corrected = cv2.LUT(gray, table)

            # Filtro adaptativo para reforçar linhas
            edges = cv2.adaptiveThreshold(
                gamma_corrected, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=9,  # menor valor = contraste mais forte
                C=15          # maior valor = mais branco
            )

            # Binarização final (remove quase todos os tons intermediários)
            _, binary = cv2.threshold(edges, 180, 255, cv2.THRESH_BINARY)

            # Morfologia leve para reforçar traços
            kernel = np.ones((2, 2), np.uint8)
            processed_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            _logger.info("Imagem processada com sucesso (tons médios removidos).")

            # Codificar em JPG
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
CORS(app)
app.register_blueprint(image_bp)

if __name__ == '__main__':
    app.run(debug=True)
