from flask import Flask, request, jsonify
from sys import exit
import os
from utils.face_mask import FaceMaskDetectionPipeline

UPLOAD_FOLDER = 'images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

detection = FaceMaskDetectionPipeline("./Face Mask Dataset")
detection.load_dataset()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/image', methods = ['GET'])
def alalyzeImage():
    if 'file' not in request.files:
        return "Not a File", 400
    file = request.files['file']
    if file.filename == '':
        return 'Название файла не валидно', 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify(detection.analyze_image(filepath))
