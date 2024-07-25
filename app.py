import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client, handle_file
from PIL import Image
import io
import logging
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

logging.basicConfig(level=logging.INFO)

# Replace this with your actual Hugging Face Gradio app name
HUGGING_FACE_SPACE = "InvincibleMeta/Meta-Tryon"

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

        with tempfile.NamedTemporaryFile(delete=False) as human_temp, tempfile.NamedTemporaryFile(delete=False) as garm_temp:
            human_img = Image.open(human_img_file)
            garm_img = Image.open(garm_img_file)

            human_img.save(human_temp.name)
            garm_img.save(garm_temp.name)

            result = client.predict(
                dict={
                    "background": handle_file(human_temp.name),
                    "layers": [],
                    "composite": None
                },
                garm_img=handle_file(garm_temp.name),
                garment_des=garment_des,
                is_checked=is_checked,
                is_checked_crop=is_checked_crop,
                denoise_steps=denoise_steps,
                seed=seed,
                api_name="/tryon"
            )

            result_image_url = result["result_image"]
            mask_image_url = result["mask_image"]

            return jsonify({
                "result_image": result_image_url,
                "mask_image": mask_image_url
            })
    except Exception as e:
        logging.error("Error processing the request", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
