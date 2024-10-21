
## Installation

环境配置参照：[tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter?tab=readme-ov-file)

## Download Models

相关模型下载：
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)

## How to Use

- [**flow_original**](flow22.ipynb): 初始flow应用模块

## How to Train
For training, you should install [accelerate](https://github.com/huggingface/accelerate) and make your own dataset into a json file.

```
accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
  tutorial_train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5/" \
  --image_encoder_path="{image_encoder_path}" \
  --data_json_file="{data.json}" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="{output_dir}" \
  --save_steps=10000
```

