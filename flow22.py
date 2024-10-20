import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IFAdapter2

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "runwayml/stable-diffusion-v1-5"
image_encoder_path = "models/image_encoder"
ip_ckpt = "output/checkpoint-60000/model.safetensors"
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
ip_model = IFAdapter2(pipe, image_encoder_path, ip_ckpt, device)

import json, random
json_path = 'data_full.json'
# json_path = 'annotations.json'
data = json.load(open(json_path, 'r', encoding='utf-8'))
data_str = """Shrink the animal uniformly markedly
/data/Hszhu/dataset/Gedi_full/source_img/31.png
/data/Hszhu/dataset/Gedi_full/gen_img/31/0/11.png
a dog that is sitting on top of a metal object""".split('\n')
image_data = dict()
image_data['edit_prompt'] = data_str[0]
image_data["ori_img_path"] = data_str[1]
image_data["gen_img_path"] = data_str[2]
image_data["tag_caption"] = data_str[3]
# read image prompt
image_o = Image.open(image_data["ori_img_path"])
image_g = Image.open(image_data["gen_img_path"])
grid = image_grid([image_o.resize((256, 256)), image_g.resize((256, 256))], 1, 2)
grid
images = ip_model.generate(image_data["edit_prompt"], pil_image=image_o, num_samples=1, prompt=image_data["tag_caption"], num_inference_steps=50, seed=42)
grid = image_grid(images, 1, 1)
grid
