import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client, handle_file
from PIL import Image
import io
import logging
import tempfile
from functools import wraps


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

logging.basicConfig(level=logging.INFO)

# Replace this with your actual Hugging Face Gradio app name
HUGGING_FACE_SPACE = "InvincibleMeta/Meta-Tryon"

client = Client(HUGGING_FACE_SPACE, download_files = False)

# Get the token from the environment variable
VALID_TOKEN = "sk-89QmxJSClcA4EceONWSfF-qw1qwur6cNyE6FeW4sGgs"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        logging.info(f"Received token: {token}")
        if not token or not token.startswith('Bearer ') or token.split(' ')[1] != VALID_TOKEN:
            return jsonify({'message': 'Token is missing or invalid!'}), 401
        return f(*args, **kwargs)
    return decorated
@app.route('/')
def index():
    return "API works!"

@app.route('/tryon', methods=['POST'])
@token_required
def tryon():
    try:
        human_img_file = request.files['human_image']
        garm_img_file = request.files['garm_image']
        garment_des = request.form.get('garment_des', '')
        is_checked = request.form.get('is_checked', 'false').lower() == 'true'
        is_checked_crop = request.form.get('is_checked_crop', 'false').lower() == 'true'
        denoise_steps = int(request.form.get('denoise_steps', 30))
        seed = int(request.form.get('seed', 42))

        
        human_temp = tempfile.NamedTemporaryFile(delete=False)
        garm_temp = tempfile.NamedTemporaryFile(delete=False)
        try:            
            human_img = Image.open(human_img_file)
            garm_img = Image.open(garm_img_file)

            # Determine the format of the image and save accordingly
            human_img_format = human_img.format if human_img.format else "PNG"
            garm_img_format = garm_img.format if garm_img.format else "PNG"

            human_img.save(human_temp.name, format=human_img_format)
            garm_img.save(garm_temp.name, format=garm_img_format)

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

            logging.info(f"Tryon module returned: {result}")

            # Extract URLs from the result assuming it's a tuple
            if isinstance(result, tuple) and len(result) > 0:
                result_image_dict, mask_image_dict = result
                result_image_url = result_image_dict['url']
                mask_image_url = mask_image_dict['url']

                return jsonify({
                    "result_image": result_image_url,
                    "mask_image": mask_image_url
                })
            else:
                logging.error("Unexpected result structure.")
                return jsonify({"error": "Unexpected result structure."}), 500
        finally:
            # Ensure temporary files are closed before removing
            human_temp.close()
            garm_temp.close()
            os.remove(human_temp.name)
            os.remove(garm_temp.name)
            
    except Exception as e:
        logging.error("Error processing the request", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
