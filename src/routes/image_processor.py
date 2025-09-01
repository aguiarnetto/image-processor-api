import uuid
import cv2
import numpy as np
from flask import Blueprint, request, send_file

image_bp = Blueprint("image_bp", __name__)

# ------------------------
# Funções auxiliares
# ------------------------

def apply_levels(img, in_black, gamma, in_white):
    """Ajuste de níveis (similar ao Photoshop)"""
    img = np.clip(img, 0, 255).astype(np.float32)
    img = (img - in_black) / (in_white - in_black)
    img = np.clip(img, 0, 1)
    img = np.power(img, 1.0 / gamma)
    return (img * 255).astype(np.uint8)

def apply_unsharp_mask(img, radius, threshold, amount=1.0):
    """Máscara de nitidez"""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    mask = cv2.subtract(img, blurred)
    sharp = cv2.add(img, cv2.multiply(mask, amount))
    # aplica limiar para suavizar áreas lisas
    low_contrast_mask = np.absolute(mask) < threshold
    np.copyto(sharp, img, where=low_contrast_mask)
    return np.clip(sharp, 0, 255).astype(np.uint8)

# ------------------------
# Rota principal
# ------------------------

@image_bp.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return {"error": "Nenhum arquivo enviado"}, 400

    file = request.files["file"]

    # Ler a imagem em memória
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    # Etapa 1: Tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Etapa 2 e 3: duplicar e inverter
    inverted = cv2.bitwise_not(gray)

    # Etapa 4: Subexposição (dodge)
    inverted_blur = cv2.GaussianBlur(inverted, (0, 0), 2.0)
    dodge = cv2.divide(gray, 255 - inverted_blur, scale=256.0)

    # Etapa 5: Níveis de entrada: 223; 1.32; 255
    levels_adjusted = apply_levels(dodge, 223, 1.32, 255)

    # Etapa 6: Máscara de nitidez (200%, raio 1.4, limiar 6)
    sharpened = apply_unsharp_mask(levels_adjusted, 1.4, 6, amount=2.0)

    # Etapa 7: Níveis finais: 0; 0.70; 164
    final = apply_levels(sharpened, 0, 0.70, 164)

    # Salvar imagem final como JPG achatado
    temp_filename = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_filename, final)

    return send_file(temp_filename, mimetype="image/jpeg")
