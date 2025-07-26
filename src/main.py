from flask import Flask
from flask_cors import CORS
from routes.image_processor import image_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(image_bp)
