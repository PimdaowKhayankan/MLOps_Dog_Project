import gradio as gr
import requests
import os
from io import BytesIO
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Backend URL (adjusted for Docker linking)
BACKEND_URL = "http://34.126.135.147:8087/predict"
# BACKEND_URL = "http://localhost:8087/predict"

def skin_detection(image):  # รับภาพจากผู้ใช้
    # แปลง numpy array ที่ได้จาก Gradio เป็น Image
    pil_image = Image.fromarray(image.astype(np.uint8))

    # แปลงภาพเป็น bytes เพื่อนำไปส่งผ่าน POST request
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # ส่งภาพไปยัง FastAPI โดยใช้ POST request
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post(BACKEND_URL, files=files)

    # ตรวจสอบสถานะการตอบกลับจาก FastAPI และดึงข้อมูลจาก API
    if response.status_code == 200:
        result = response.json()
        return f"{result['Prediction']} ({result['Confidence']})"
    else:
        return "Error: Unable to detect disease."

examples = [('images/BacterialDermatosis.jpg', 'Bacterial Dermatosis'), ('images/FungalInfection.jpg', 'Fungal Infection'),\
             ('images/HypersensitivityDermatitis.jpg', 'Hypersensitivity Dermatitis'),('images/Healthy.jpg', 'Healthy')]

# Gradio interface
theme = gr.themes.Soft(
    secondary_hue="rose",
    neutral_hue="violet",
    radius_size="lg",
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    body_background_fill='*primary_50',
    body_background_fill_dark='*primary_950',
    body_text_color='*neutral_900',
    background_fill_primary='*neutral_100',
    background_fill_primary_dark='*primary_800',
    background_fill_secondary='*neutral_100',
    border_color_accent='*primary_950',
    border_color_accent_subdued='*primary_300',
    shadow_drop='*button_primary_shadow_active',
    shadow_drop_lg='0 5px 8px 0 rgb(0 0 0 / 0.1)',
    shadow_inset='*shadow_drop_lg',
    shadow_spread='20px',
    block_background_fill='*primary_100',
    block_background_fill_dark='*neutral_300',
    block_border_color='*border_color_accent',
    block_info_text_color_dark='*primary_800',
    block_info_text_weight='500',
    block_label_background_fill='*primary_50',
    block_label_background_fill_dark='*primary_500',
    block_shadow='*shadow_spread',
    checkbox_shadow='*shadow_drop_lg',
    button_secondary_background_fill_hover='*primary_300'
)

with gr.Blocks(theme=theme) as demo:
    ...

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 🐶 Dog Skin Disease Detector")
            gr.Markdown("Upload an image of suspicious areas of your pet's skin for primary skin disease detection.")
            gr.Image("images/Dog paw-amico.png", width=415, height=415, show_label=False)
        with gr.Column(scale=2):
            image_input = gr.Image(type="numpy", label="Upload your pet's image")
            gr.Gallery(examples, height=80, label='Skin Disease Example')
            output_label = gr.Label(label="Your pet's skin disease Detection:")
            button = gr.Button("Detect")
            button.click(skin_detection, inputs=image_input, outputs=output_label)

if __name__ == "__main__":
   demo.launch(server_name="0.0.0.0", server_port=8085)




