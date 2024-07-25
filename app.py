import os
import base64
from flask import Flask, request, render_template, send_from_directory
from gradio_client import Client, handle_file

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = Client("http://192.168.0.165:7860/") # local Server


def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tryon', methods=['POST'])
def tryon():
    human_image = request.files['human_image']
    clothe_image = request.files['clothe_image']
    garment_des = request.form.get('garment_des', 'Default description')
    
    human_image_path = os.path.join(UPLOAD_FOLDER, human_image.filename)
    clothe_image_path = os.path.join(UPLOAD_FOLDER, clothe_image.filename)
    
    human_image.save(human_image_path)
    clothe_image.save(clothe_image_path)
    
    result = client.predict(
        dict={"background": handle_file(human_image_path), "layers": [], "composite": None},
        garm_img=handle_file(clothe_image_path),
        garment_des=garment_des,
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )
    
    synthesized_image_base64 = encode_image_to_base64(result[0])
    mask_image_base64 = encode_image_to_base64(result[1])
    
    return render_template('result.html', synthesized_image=synthesized_image_base64, mask_image=mask_image_base64)

if __name__ == '__main__':
    app.run(debug=True)
