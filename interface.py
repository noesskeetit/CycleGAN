import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2

from generator_model import Generator
import config

# load state dict
gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

print("=> Loading checkpoint gen_H")
checkpoint_H = torch.load(config.CHECKPOINT_GEN_H, map_location=config.DEVICE)
gen_H.load_state_dict(checkpoint_H["state_dict"])

print("=> Loading checkpoint gen_Z")
checkpoint_Z = torch.load(config.CHECKPOINT_GEN_Z, map_location=config.DEVICE)
gen_Z.load_state_dict(checkpoint_Z["state_dict"])

gen_H.eval()
gen_Z.eval()

# Inference
@torch.no_grad()
def translate_image(image: Image.Image, direction: str, upscale: str):
    image_np = np.array(image)
    transformed = config.infer_transforms(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(config.DEVICE)

    # Generator
    if direction == "photo → vangogh":
        fake = gen_Z(image_tensor)
    else:
        fake = gen_H(image_tensor)

    # Denorm + PIL
    fake = (fake.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu()
    fake_pil = transforms.ToPILImage()(fake)

    # Upscaling
    scale = int(upscale.replace("×", ""))
    img_np = np.array(fake_pil)
    h, w = img_np.shape[:2]
    upscaled_np = cv2.resize(img_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    upscaled_pil = Image.fromarray(upscaled_np)

    return upscaled_pil

# Interface
demo = gr.Interface(
    fn=translate_image,
    inputs=[
        gr.Image(type="pil", label="Загрузите изображение"),
        gr.Radio(["photo → vangogh", "vangogh → photo"], label="Направление"),
        gr.Radio(["×1", "×2", "×4"], label="Масштаб апскейла", value="×4"),
    ],
    outputs=gr.Image(type="pil", label="Результат"),
    title="🎨 CycleGAN + Апскейл через OpenCV",
    description="Переведите изображение в стиль Ван Гога (или обратно) и увеличьте разрешение",
)

if __name__ == "__main__":
    demo.launch()
