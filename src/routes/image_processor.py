import cv2
import numpy as np
from PIL import Image
import io

def process_image(file):
    image = Image.open(file.stream).convert("RGB")
    img_array = np.array(image)

    # Converter para escala de cinza
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Inverter a imagem
    inverted = 255 - gray

    # Aplicar blur para simular lápis
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Criar efeito de sketch
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # 🔥 Aumentar contraste para reforçar os traços
    sketch = cv2.equalizeHist(sketch)

    # Converter de volta para imagem PIL
    result = Image.fromarray(sketch)

    # Salvar em buffer
    img_io = io.BytesIO()
    result.save(img_io, "PNG")
    img_io.seek(0)

    return img_io
