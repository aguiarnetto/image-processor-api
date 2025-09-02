import io
import uuid
import cv2
import numpy as np
from flask import Blueprint, request, send_file, jsonify

# Mantém o mesmo nome do blueprint para o main.py
image_bp = Blueprint("image_bp", __name__)

# ------------------------
# Funções auxiliares
# ------------------------

def color_dodge(base_gray: np.ndarray, blend_gray: np.ndarray) -> np.ndarray:
    """
    Subexposição de cores (Color Dodge):
    result = min(255, base * 255 / (255 - blend))
    """
    base = base_gray.astype(np.float32)
    blend = blend_gray.astype(np.float32)
    denom = 255.0 - blend
    denom[denom < 1.0] = 1.0  # evita divisão por zero
    out = np.minimum(255.0, (base * 255.0) / denom)
    return out.clip(0, 255).astype(np.uint8)

def apply_levels(img: np.ndarray, in_black: float, gamma: float, in_white: float) -> np.ndarray:
    """
    Níveis de entrada + gamma:
    1) normaliza para [0..1] com in_black/in_white
    2) aplica gamma (Photoshop usa expoente 1/gamma)
    """
    x = img.astype(np.float32)
    x = (x - in_black) / (in_white - in_black + 1e-6)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 1.0 / max(gamma, 1e-6))
    return (x * 255.0).clip(0, 255).astype(np.uint8)

def unsharp_mask(img: np.ndarray, radius: float, amount: float, threshold: float = 0.0) -> np.ndarray:
    """
    Unsharp mask clássico:
    sharp = img + amount * (img - blur(img, radius))
    amount: 3.42 == 342%
    """
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    mask = cv2.subtract(img, blurred).astype(np.int16)

    if threshold > 0:
        low = (np.abs(mask) < threshold)
        mask[low] = 0

    # img + amount * mask
    sharp = img.astype(np.int16) + (amount * mask)
    return np.clip(sharp, 0, 255).astype(np.uint8)

# ------------------------
# Rota principal
# ------------------------

@image_bp.route("/process", methods=["POST"])
def process():
    # Espera campo 'file' (igual ao Hoppscotch)
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado no campo 'file'."}), 400

    file = request.files["file"]
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Falha ao ler imagem."}), 400

    # 1) tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) duplicar (conceitual) 3) inverter a superior
    inverted = cv2.bitwise_not(gray)

    # 4) color dodge usando 5) blur (raio 7 px) aplicado na camada invertida
    inv_blur = cv2.GaussianBlur(inverted, (0, 0), 7.0)
    dodge = color_dodge(gray, inv_blur)

    # 6) achatar (já está achatado porque estamos em 1 canal)

    # 7) máscara de nitidez: 342% (3.42), raio 4.5, limiar 0
    sharpened = unsharp_mask(dodge, radius=4.5, amount=3.42, threshold=0.0)

    # 8) níveis: 97 ; 0.54 ; 220
    leveled = apply_levels(sharpened, in_black=97.0, gamma=0.54, in_white=220.0)

    # 9) salvar JPG
    ok, enc = cv2.imencode(
        ".jpg",
        leveled,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1],
    )
    if not ok:
        return jsonify({"error": "Falha ao codificar JPG."}), 500

    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")
