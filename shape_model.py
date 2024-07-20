import torch
import gradio as gr
import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

def gif_widget(images):
    pil_images = [PILImage.fromarray((img * 255).astype('uint8')) if isinstance(img, np.ndarray) else img for img in images]
    buf = BytesIO()
    pil_images[0].save(buf, format='GIF', save_all=True, append_images=pil_images[1:], loop=0, duration=100)
    data = buf.getvalue()
    return f'<img src="data:image/gif;base64,{base64.b64encode(data).decode()}" />'

def generate_image(prompt):
    try:
        batch_size = 1
        guidance_scale = 50.0

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        render_mode = 'nerf'
        size = 64
        cameras = create_pan_cameras(size, device)

        images = []
        for latent in latents:
            images.extend(decode_latent_images(xm, latent, cameras, rendering_mode=render_mode))

        gif_html = gif_widget(images)
        return gif_html

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"<p style='color: red;'>An error occurred: {e}</p>"

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Introduce tu prompt aquí..."),
    outputs=gr.HTML(),
    title="Generación de Imágenes 3D con Shap-E",
    description="Introduce un prompt para generar una imagen en 3D."
)

# Launch the interface
interface.launch(share=True, debug=True)