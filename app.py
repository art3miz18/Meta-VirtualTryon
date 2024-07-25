import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
from PIL import Image
import io
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

logging.basicConfig(level=logging.INFO)

HUGGING_FACE_URL = "https://huggingface.co/spaces/InvincibleMeta/Meta-Tryon/tryon"
# HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

@app.route('/')
def index():
    return "API works!"

@app.route('/tryon', methods=['POST'])
def tryon():
    data = request.json
    human_img_data = base64.b64decode(data['human_image'])
    garm_img_data = base64.b64decode(data['garm_image'])
    garment_des = data['garment_des']
    is_checked = data['is_checked']
    is_checked_crop = data['is_checked_crop']
    denoise_steps = data['denoise_steps']
    seed = data['seed']
    
    human_img = Image.open(io.BytesIO(human_img_data))
    garm_img = Image.open(io.BytesIO(garm_img_data))

    human_img_bytes = io.BytesIO()
    human_img.save(human_img_bytes, format='PNG')
    human_img_str = base64.b64encode(human_img_bytes.getvalue()).decode('utf-8')
    
    garm_img_bytes = io.BytesIO()
    garm_img.save(garm_img_bytes, format='PNG')
    garm_img_str = base64.b64encode(garm_img_bytes.getvalue()).decode('utf-8')
    
    payload = {
        "human_image": human_img_str,
        "garm_image": garm_img_str,
        "garment_des": garment_des,
        "is_checked": is_checked,
        "is_checked_crop": is_checked_crop,
        "denoise_steps": denoise_steps,
        "seed": seed
    }
    
    # headers = {
    #     "Authorization": f"Bearer {HUGGING_FACE_TOKEN}"
    # }
    
    # response = requests.post(HUGGING_FACE_URL, json=payload, headers=headers) # with Access key header token
    response = requests.post(HUGGING_FACE_URL, json=payload, )
    result = response.json()
    
    result_img = Image.open(io.BytesIO(base64.b64decode(result['result_image'])))
    mask_img = Image.open(io.BytesIO(base64.b64decode(result['mask_image'])))
    
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    result_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    mask_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({
        "result_image": result_img_str,
        "mask_image": mask_img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
