import gradio as gr
import requests

def tryon_interface(human_image, garm_image, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    # Convert images to files
    human_img_bytes = human_image.tobytes()
    garm_img_bytes = garm_image.tobytes()

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
    
    result_image = gr.Image.update(result_image_url)
    mask_image = gr.Image.update(mask_image_url)
    
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

iface.launch(server_name="0.0.0.0", server_port=7860)
