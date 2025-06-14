from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image
import tempfile
import os

image_processor_bp = Blueprint('image_processor', __name__)

def process_image_cv2(image_data):
    """
    Processa a imagem para criar um efeito de arte de linha
    """
    # Converter bytes para array numpy
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem")

    # Inverter a imagem para que os traços sejam pretos em fundo branco
    inverted_img = cv2.bitwise_not(img)

    # Aplicar um desfoque gaussiano para suavizar a imagem e remover ruídos
    blurred_img = cv2.GaussianBlur(inverted_img, (25, 25), 0)

    # Inverter a imagem desfocada
    inverted_blurred_img = cv2.bitwise_not(blurred_img)

    # Criar o efeito de "dodge" de cor, que realça as bordas
    sketch = cv2.divide(img, inverted_blurred_img, scale=256.0)

    # Binarizar a imagem para obter traços mais definidos
    _, binarized_sketch = cv2.threshold(sketch, 150, 255, cv2.THRESH_BINARY)

    return binarized_sketch

@image_processor_bp.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Verificar se foi enviado um arquivo
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem foi enviada'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nenhuma imagem foi selecionada'}), 400
        
        # Ler os dados da imagem
        image_data = file.read()
        
        # Processar a imagem
        processed_image = process_image_cv2(image_data)
        
        # Converter para PIL Image para salvar como PNG
        pil_image = Image.fromarray(processed_image)
        
        # Criar um buffer em memória para a imagem processada
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='imagem_processada.png'
        )
        
    except Exception as e:
        return jsonify({'error': f'Erro ao processar a imagem: {str(e)}'}), 500

@image_processor_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API de processamento de imagem funcionando'})

