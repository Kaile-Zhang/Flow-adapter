import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import json, random

from ip_adapter import IFAdapter

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "runwayml/stable-diffusion-v1-5"
ip_ckpt = "output-flow0/checkpoint-1400/model.safetensors"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae").to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
ip_model = IFAdapter(pipe, ip_ckpt, device)

json_path = 'data_full.json'
# json_path = 'annotations.json'
data = json.load(open(json_path, 'r', encoding='utf-8'))
image_data = random.choice(data)
print(image_data['edit_prompt'])
print(image_data["ori_img_path"])
print(image_data["gen_img_path"])
# read image prompt
image_o = Image.open(image_data["ori_img_path"])
image_g = Image.open(image_data["gen_img_path"])
grid = image_grid([image_o.resize((256, 256)), image_g.resize((256, 256))], 1, 2)
images = ip_model.generate(image_data["edit_prompt"], pil_image=image_o, num_samples=4, num_inference_steps=50, seed=42)
grid = image_grid(images, 1, 4)
