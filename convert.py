import torch
from safetensors.torch import load_file

# ckpt = "output/checkpoint-10000/optimizer.bin"
# sd = torch.load(ckpt, map_location="cpu")
for step_num in [100, 200, 300, 400, 500, 700, 1000]:
    ckpt = f"output-bear/checkpoint-{step_num}/model.safetensors"
    sd = load_file(ckpt)

    adapter_modules_path = f"output-bear/adapter_modules-{step_num}.pth"
    adapter_modules = torch.load(adapter_modules_path).state_dict()

    image_proj_sd = {}
    ip_sd = {}

    for i in adapter_modules:
        ip_sd[i] = adapter_modules[i]

    for k in sd:
        if k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            # print(k)

    # print(ip_sd)
    torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, f"output-bear/ip_adapter-{step_num}.bin")
