import cv2 
import numpy as np
from PIL import Image
import io

def process_image(file):
    # Abrir imagem recebida
    image = Image.open(file.stream).convert("RGB")
    img_array = np.array(image)

    # Converter para escala de cinza
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Inverter a imagem
    inverted = 255 - gray

    # Aplicar blur (reduzido de 21 → 17 para menos lavagem e mais detalhe)
    blur = cv2.GaussianBlur(inverted, (17, 17), 0)

    # Criar efeito de sketch (divisão)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # Equalizar histograma para reforçar contraste
    sketch = cv2.equalizeHist(sketch)

    # Ajustar contraste extra (puxa mais os pretos e brancos)
    sketch = cv2.convertScaleAbs(sketch, alpha=1.2, beta=0)

    # Converter de volta para imagem PIL
    result = Image.fromarray(sketch)

    # Salvar em buffer
    img_io = io.BytesIO()
    result.save(img_io, "PNG")
    img_io.seek(0)

    return img_io
