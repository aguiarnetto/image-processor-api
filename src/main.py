
from flask import Flask
from routes.image_processor import image_bp

app = Flask(__name__)
app.register_blueprint(image_bp)
