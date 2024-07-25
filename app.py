from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from gradio_client import Client, file

import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

logging.basicConfig(level=logging.INFO)

HUGGING_FACE_SPACE = "InvincibleMeta/Meta-Tryon"
# HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
client = Client(HUGGING_FACE_SPACE)

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

        result = client.predict(
            dict={
                "background": file('/tmp/human_image.png'),
                "layers": [],
                "composite": None
            },
            garm_img=file('/tmp/garm_image.png'),
            garment_des=garment_des,
            is_checked=is_checked,
            is_checked_crop=is_checked_crop,
            denoise_steps=denoise_steps,
            seed=seed,
            api_name="/tryon"
        )
        
        return jsonify({
            "result_image": result[0],
            "mask_image": result[1]
        })
        
    except Exception as e:
        logging.error("Error processing the request", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        logging.error("Error processing the request", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
