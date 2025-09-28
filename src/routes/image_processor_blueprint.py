import logging
import io
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS  # Import para habilitar CORS

# ========================
# PARÂMETROS EDITÁVEIS
# ========================
GAMMA = 2.2        # Clareamento mais forte → imagem entra mais branca
BLOCKSIZE = 21     # Blocos maiores suavizam a variação de tons
C = 10             # Constante maior → fundo mais branco
SHARPEN = 0.01     # Nitidez bem mais suave
UPSCALE = 2.0      # Fator de aumento da resolução (1.0 = sem alteração)
EDGE_WEIGHT = 0.2  # Peso das bordas no mix (reduzido)
BASE_WEIGHT = 0.8  # Peso da imagem original no mix

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
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            _logger.info("Imagem decodificada com sucesso.")

            if image is None:
                _logger.error("Falha ao decodificar a imagem. Verifique o formato.")
                return jsonify({"error": "Falha ao decodificar a imagem"}), 400

            # ============================
            # Aumentar resolução
            # ============================
            if UPSCALE > 1.0:
                image = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

            # ============================
            # Processamento principal
            # ============================
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Pequeno blur inicial já ajuda a "limpar" o ruído
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Detecção de bordas menos agressiva
            edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=3)
            edges_inv = cv2.bitwise_not(edges)

            # Mistura mais leve das bordas
            processed_image = cv2.addWeighted(gray_image, BASE_WEIGHT, edges_inv, EDGE_WEIGHT, 0)

            # Gamma correction (clarear a imagem já no início)
            lookup_table = np.array([
                ((i / 255.0) ** (1.0 / GAMMA)) * 255
                for i in np.arange(256)
            ]).astype("uint8")
            processed_image = cv2.LUT(processed_image, lookup_table)

            # Threshold adaptativo mais suave
            adaptive = cv2.adaptiveThreshold(
                processed_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                BLOCKSIZE,
                C
            )
            processed_image = cv2.addWeighted(processed_image, 0.6, adaptive, 0.4, 0)

            # ============================
            # Ajustar níveis (255 → 200)
            # ============================
            processed_image = np.clip(processed_image, 0, 200)
            processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)

            # ============================
            # Nitidez mais suave
            # ============================
            if SHARPEN > 0:
                blur = cv2.GaussianBlur(processed_image, (0, 0), 3)
                sharpened = cv2.addWeighted(processed_image, 1 + SHARPEN, blur, -SHARPEN, 0)
                processed_image = sharpened

            _logger.info("Imagem processada com sucesso.")

            # ============================
            # Salvar em JPG alta qualidade
            # ============================
            _, img_encoded = cv2.imencode(".jpg", processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
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
CORS(app)  # Habilita CORS

# Registro do Blueprint
app.register_blueprint(image_bp)

if __name__ == '__main__':
    app.run(debug=True)
