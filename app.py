import gradio as gr
import requests
from PIL import Image
import io

def tryon_interface(human_image, garm_image, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    human_img_bytes = io.BytesIO()
    human_image.save(human_img_bytes, format='PNG')
    human_img_bytes.seek(0)
    
    garm_img_bytes = io.BytesIO()
    garm_image.save(garm_img_bytes, format='PNG')
    garm_img_bytes.seek(0)
    
    files = {
        'human_image': ('human_image.png', human_img_bytes, 'image/png'),
        'garm_image': ('garm_image.png', garm_img_bytes, 'image/png')
    }
    data = {
        'garment_des': garment_des,
        'is_checked': is_checked,
        'is_checked_crop': is_checked_crop,
        'denoise_steps': denoise_steps,
        'seed': seed
    }
    
    response = requests.post("https://meta-virtualtryon.onrender.com/tryon", files=files, data=data)
    result = response.json()
    
    result_image_url = result["result_image"]
    mask_image_url = result["mask_image"]
    
    result_image = Image.open(requests.get(result_image_url, stream=True).raw)
    mask_image = Image.open(requests.get(mask_image_url, stream=True).raw)
    
    return result_image, mask_image

iface = gr.Interface(
    fn=tryon_interface,
    inputs=[
        gr.Image(type="pil", label="Human Image"),
        gr.Image(type="pil", label="Garment Image"),
        gr.Textbox(placeholder="Description of garment", label="Garment Description"),
        gr.Checkbox(label="Use auto-generated mask"),
        gr.Checkbox(label="Use auto-crop & resizing"),
        gr.Number(label="Denoising Steps", default=30),
        gr.Number(label="Seed", default=42)
    ],
    outputs=[
        gr.Image(label="Synthesized Image"),
        gr.Image(label="Mask Image")
    ],
    title="Virtual Try-On"
)

iface.launch()
