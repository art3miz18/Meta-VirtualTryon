import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

logging.basicConfig(level=logging.INFO)

HUGGING_FACE_URL = "https://huggingface.co/spaces/InvincibleMeta/Meta-Tryon"
# HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

@app.route('/')
def index():
    return "API works!"

@app.route('/tryon', methods=['POST'])
def tryon():
    try:
        human_img_file = request.files['human_image']
        garm_img_file = request.files['garm_image']
        garment_des = request.form.get('garment_des', '')
        is_checked = request.form.get('is_checked', 'false').lower() == 'true'
        is_checked_crop = request.form.get('is_checked_crop', 'false').lower() == 'true'
        denoise_steps = int(request.form.get('denoise_steps', 30))
        seed = int(request.form.get('seed', 42))

        human_img = Image.open(human_img_file)
        garm_img = Image.open(garm_img_file)

        human_img.save('/tmp/human_image.png')
        garm_img.save('/tmp/garm_image.png')

        files = {
            "dict.background": open('/tmp/human_image.png', 'rb'),
            "garm_img": open('/tmp/garm_image.png', 'rb'),
            "garment_des": (None, garment_des),
            "is_checked": (None, str(is_checked).lower()),
            "is_checked_crop": (None, str(is_checked_crop).lower()),
            "denoise_steps": (None, str(denoise_steps)),
            "seed": (None, str(seed)),
        }
        
        # headers = {
        #     "Authorization": f"Bearer {HUGGING_FACE_TOKEN}"
        # }
        
        response = requests.post(HUGGING_FACE_URL, files=files)
        # response = requests.post(HUGGING_FACE_URL, files=files, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        return jsonify(result)
    
    except Exception as e:
        logging.error("Error processing the request", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
