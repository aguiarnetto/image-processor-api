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
            npimg = np.frombuffer(filestr, np.uint8)  # Uso correto do np.frombuffer
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            _logger.info("Imagem decodificada com sucesso.")

            if image is None:
                _logger.error("Falha ao decodificar a imagem. Verifique o formato.")
                return jsonify({"error": "Falha ao decodificar a imagem"}), 400

            # -------------------------------
            # Processamento estilo decalque
            # -------------------------------
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Reduz ruído preservando bordas
            gray = cv2.medianBlur(gray, 5)

            # Detecta bordas → linhas pretas no fundo branco
            processed_image = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=9,
                C=5
            )

            _logger.info("Imagem processada com sucesso (modo decalque).")

            # Codificar a imagem processada para JPG
            _, img_encoded = cv2.imencode(".jpg", processed_image)
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
# Se quiser restringir apenas ao Hoppscotch:
# CORS(app, resources={r"/*": {"origins": "https://hoppscotch.io"}})

# Registro do Blueprint
app.register_blueprint(image_bp)

if __name__ == '__main__':
    app.run(debug=True)
