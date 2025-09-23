import io
import uuid
import cv2
import numpy as np
from flask import Blueprint, request, send_file, jsonify

# =====================
# ADICIONE APENAS ESTO
# =====================
from flask_cors import CORS

# Mantém o mesmo nome do blueprint para o main.py
image_bp = Blueprint("image_bp", __name__)

# Habilita CORS **para todas as rotas deste blueprint**
CORS(image_bp, resources={r"/*": {"origins": "*"}})
# =====================

# ------------------------
# Funções auxiliares
# ------------------------
def color_dodge(base_gray: np.ndarray, blend_gray: np.ndarray) -> np.ndarray:
    base = base_gray.astype(np.float32)
    blend = blend_gray.astype(np.float32)
    denom = 255.0 - blend
    denom[denom < 1.0] = 1.0
    out = np.minimum(255.0, (base * 255.0) / denom)
    return out.clip(0, 255).astype(np.uint8)

def apply_levels(img: np.ndarray, in_black: float, gamma: float, in_white: float) -> np.ndarray:
    x = img.astype(np.float32)
    x = (x - in_black) / (in_white - in_black + 1e-6)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 1.0 / max(gamma, 1e-6))
    return (x * 255.0).clip(0, 255).astype(np.uint8)

def unsharp_mask(img: np.ndarray, radius: float, amount: float, threshold: float = 0.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    mask = cv2.subtract(img, blurred).astype(np.int16)

    if threshold > 0:
        low = (np.abs(mask) < threshold)
        mask[low] = 0

    sharp = img.astype(np.int16) + (amount * mask)
    return np.clip(sharp, 0, 255).astype(np.uint8)

# ------------------------
# Rota principal
# ------------------------
@image_bp.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado no campo 'file'."}), 400

    file = request.files["file"]
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Falha ao ler imagem."}), 400

    # 1) tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) inverter
    inverted = cv2.bitwise_not(gray)

    # 3) color dodge com blur
    inv_blur = cv2.GaussianBlur(inverted, (0, 0), 7.0)
    dodge = color_dodge(gray, inv_blur)

    # 4) máscara de nitidez
    sharpened = unsharp_mask(dodge, radius=4.5, amount=3.42, threshold=0.0)

    # 5) níveis
    leveled = apply_levels(sharpened, in_black=97.0, gamma=0.54, in_white=220.0)

    # ------------------------
    # Garantir fundo branco (3 canais BGR)
    # ------------------------
    h, w = leveled.shape
    white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
    leveled_rgb = cv2.cvtColor(leveled, cv2.COLOR_GRAY2BGR)
    final_img = cv2.addWeighted(leveled_rgb, 1.0, white_bg, 0.0, 0)

    # ------------------------
    # Salvar como PNG
    # ------------------------
    ok, enc = cv2.imencode(".png", final_img)
    if not ok:
        return jsonify({"error": "Falha ao codificar PNG."}), 500

    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/png")
