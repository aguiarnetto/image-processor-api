import cv2
import numpy as np
import uuid
import io
from PIL import Image

# Função auxiliar para salvar etapas (debug opcional)
def save_step(img, step_name):
    debug = False  # ✅ coloque True se quiser salvar steps
    if debug:
        cv2.imwrite(f"/tmp/{step_name}.jpg", img)

# Função para aplicar ajuste de níveis
def apply_levels(img, in_black, gamma, in_white):
    # Normalizar para float
    img_float = img.astype(np.float32) / 255.0
    # Aplicar níveis
    img_adj = np.clip((img_float - (in_black / 255.0)) / ((in_white - in_black) / 255.0), 0, 1)
    img_gamma = np.power(img_adj, 1.0 / gamma)
    return np.uint8(img_gamma * 255)

# Função para aplicar unsharp mask (máscara de nitidez)
def apply_unsharp_mask(image, radius=1.0, threshold=0, amount=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    mask = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    # Limiar para preservar áreas lisas
    low_contrast_mask = np.abs(image - blurred) < threshold
    np.copyto(mask, image, where=low_contrast_mask)
    return mask

# Função principal de processamento
def process_image(file):
    # Ler imagem do upload
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    # Etapa 1: Tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_step(gray, "1_gray")

    # Etapa 2: Inverter
    inverted = cv2.bitwise_not(gray)
    save_step(inverted, "2_inverted")

    # Etapa 3: Dodge (subexposição)
    inverted_blur = cv2.GaussianBlur(inverted, (0, 0), 2.0)
    dodge = cv2.divide(gray, 255 - inverted_blur, scale=256.0)
    save_step(dodge, "3_dodge")

    # Etapa 4: Ajuste de níveis (223;1.32;255)
    levels_adjusted = apply_levels(dodge, 223, 1.32, 255)
    save_step(levels_adjusted, "4_levels1")

    # Etapa 5: Máscara de nitidez (200%, raio 1.4, limiar 6)
    sharpened = apply_unsharp_mask(levels_adjusted, radius=1.4, threshold=6, amount=2.0)
    save_step(sharpened, "5_sharpened")

    # Etapa 6: Níveis finais (0;0.70;164)
    final = apply_levels(sharpened, 0, 0.70, 164)
    save_step(final, "6_levels2")

    # 🔥 Garantir que a imagem seja achatada (1 canal → 3 canais)
    final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    # Salvar em JPG
    temp_filename = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_filename, final_bgr)

    # Retornar como buffer
    img_io = io.BytesIO()
    pil_img = Image.open(temp_filename)
    pil_img.save(img_io, "JPEG", quality=95)
    img_io.seek(0)

    return img_io
