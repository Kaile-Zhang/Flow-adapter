from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt


def load_image(pil_image, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = pil_image.convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def ddim_inversion(pil_image, prompt="", negative_prompt="", num_steps: int = 50, verify: Optional[bool] = False) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae

    input_img = load_image(pil_image).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt=prompt, negative_prompt=prompt, guidance_scale=1.,
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)

    # verify
    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()
    return inv_latents


if __name__ == '__main__':
    image = Image.open("test/rec/0.png")
    print(image.size)
    inv_latents = ddim_inversion(image, num_steps=100, verify=True)
    print(inv_latents.shape)
