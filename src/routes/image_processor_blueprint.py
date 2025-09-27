import logging
import io
import numpy as np
import cv2
from flask import Flask, Blueprint, request, jsonify, send_file
from flask_cors import CORS

# ========================
# PARÂMETROS EDITÁVEIS (teste e ajuste)
# ========================
# redução de tamanho (se quiser) - 1.0 = sem redimensão, 0.5 = 50% (use para uploads grandes)
RESIZE_FACTOR = 1.0

# remoção de ruído preservando bordas
BILATERAL_DIAM = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# mediana para remover speckles
MEDIAN_KSIZE = 3  # 1,3,5,...

# bordas
CANNY_T1 = 50
CANNY_T2 = 150
LAPLACIAN_KSIZE = 3

# threshold adaptativo
ADAPT_BLOCK = 15   # ímpar, 11..31
ADAPT_C = 6

# dilatação/erosão para reforço de traços e limpeza
DILATE_ITER = 1
ERODE_ITER = 0
MORPH_KSIZE = 2    # kernel para limpeza

# binarização final
FINAL_THRESH = 200  # 180-220 testar
MASK_OFFSET = 20    # FINAL_THRESH - MASK_OFFSET

# nitidez final leve
SHARPEN = 0.12  # 0 = off, 0.1 suave, 0.2 forte

# saída: 'png' (recomendado) ou 'jpg'
DEFAULT_OUT_FORMAT = 'png'
JPEG_QUALITY = 95

# ------------------------
image_bp = Blueprint("image_processor", __name__)
_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def to_three_channel(img_gray):
    """Converte imagem gray (H,W) para BGR (H,W,3)"""
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

@image_bp.route("/process", methods=["POST"])
def process_image():
    _logger.info("Requisição de processamento recebida.")
    # permite definir formato por query param ?format=jpg
    fmt = request.args.get('format', DEFAULT_OUT_FORMAT).lower()
    if "file" not in request.files:
        _logger.error("Nenhum arquivo 'file' na requisição.")
        return jsonify({"error": "Nenhum arquivo 'file' na requisição"}), 400

    file = request.files["file"]
    if file.filename == "":
        _logger.error("Nome de arquivo vazio.")
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            _logger.error("Falha ao decodificar a imagem.")
            return jsonify({"error": "Falha ao decodificar a imagem"}), 400

        # opcional: reduzir para acelerar / evitar arquivos enormes
        if RESIZE_FACTOR != 1.0 and RESIZE_FACTOR > 0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)), interpolation=cv2.INTER_AREA)

        # Converter para cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) Denoise: bilateral preserva bordas
        den = cv2.bilateralFilter(gray, d=BILATERAL_DIAM,
                                  sigmaColor=BILATERAL_SIGMA_COLOR,
                                  sigmaSpace=BILATERAL_SIGMA_SPACE)

        # 2) Suavização de pequenos pontos (speckle)
        if MEDIAN_KSIZE > 1:
            den = cv2.medianBlur(den, MEDIAN_KSIZE)

        # 3) Detectar bordas - combinação Canny + Laplacian para captar diferentes traços
        edges_canny = cv2.Canny(den, CANNY_T1, CANNY_T2)
        lap = cv2.Laplacian(den, cv2.CV_8U, ksize=LAPLACIAN_KSIZE)
        lap_inv = cv2.bitwise_not(lap)

        # Normalizar e combinar
        # Convert edges_canny to same scale as den (0/255)
        edges_comb = cv2.addWeighted(edges_canny, 0.6, lap, 0.4, 0)

        # 4) Combinar bordas com imagem suavizada para manter textura fina
        combined = cv2.addWeighted(den, 0.65, edges_comb, 0.35, 0)

        # 5) Threshold adaptativo para reduzir médios
        adapt = cv2.adaptiveThreshold(combined,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      ADAPT_BLOCK,
                                      ADAPT_C)

        # suaviza excesso: misturar adapt e combined para manter traços
        mix = cv2.addWeighted(combined, 0.55, adapt, 0.45, 0)

        # 6) reforçar linhas: dilate (fazer traços mais contínuos), opcional erosion
        kernel = np.ones((MORPH_KSIZE, MORPH_KSIZE), np.uint8)
        if DILATE_ITER > 0:
            mix = cv2.dilate(mix, kernel, iterations=DILATE_ITER)
        if ERODE_ITER > 0:
            mix = cv2.erode(mix, kernel, iterations=ERODE_ITER)

        # 7) nitidez suave
        if SHARPEN and SHARPEN > 0:
            sharpen_kernel = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]])
            sharp = cv2.filter2D(mix, -1, sharpen_kernel)
            mix = cv2.addWeighted(mix, 1 - SHARPEN, sharp, SHARPEN, 0)

        # 8) BINARIZAÇÃO FINAL: força branco/ preto
        # threshold simples
        _, bw_simple = cv2.threshold(mix, FINAL_THRESH, 255, cv2.THRESH_BINARY)

        # máscara extra: tudo abaixo de (FINAL_THRESH - MASK_OFFSET) vira PRETO, acima vira BRANCO
        mask = mix < (FINAL_THRESH - MASK_OFFSET)
        forced = np.where(mask, 0, 255).astype('uint8')

        # combinar com binário simples para reforçar linhas
        final = cv2.bitwise_and(forced, bw_simple)

        # limpeza final: abrir para remover pontos soltos
        small_kernel = np.ones((2, 2), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_OPEN, small_kernel, iterations=1)

        # Opcional: preencher pequenos buracos (closing) se necessário
        # final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, small_kernel, iterations=1)

        # Garantir tipo uint8
        final = final.astype('uint8')

        # Para JPEG: converta para 3 canais BGR (evita problemas com visualizadores)
        if fmt == 'jpg' or fmt == 'jpeg':
            out_img = to_three_channel(final)
            success, enc = cv2.imencode('.jpg', out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            mime = 'image/jpeg'
            filename = 'processed_image.jpg'
        else:
            # PNG aceita grayscale sem problemas (recomendado para linhas)
            success, enc = cv2.imencode('.png', final)
            mime = 'image/png'
            filename = 'processed_image.png'

        if not success:
            _logger.exception("Falha ao codificar a imagem.")
            return jsonify({"error": "Falha ao codificar a imagem"}), 500

        return send_file(io.BytesIO(enc.tobytes()), mimetype=mime, as_attachment=False, download_name=filename)

    except Exception as e:
        _logger.exception(f"Erro no processamento: {e}")
        return jsonify({"error": str(e)}), 500


# app principal
app = Flask(__name__)
CORS(app)
app.register_blueprint(image_bp)

if __name__ == '__main__':
    port = int(__import__('os').environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
