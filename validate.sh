#!/usr/bin/env bash
export PYTHONPATH="$PWD:$PWD/diffusers/src:$PYTHONPATH"

# 路径都可用环境变量覆盖，默认值对应 README 里的 ./models 目录结构
MODEL_DIR="${REVEALLAYER_MODEL_DIR:-./models/FLUX.1-Kontext-dev}"
TRANSP_VAE="${TRANSP_VAE_CKPT:-./models/RevealLayer/xvae/transparent_decoder_ckpt.pth}"

# 同步 custom_pipeline.py 后跑：无 shift + 50 步
python validate.py \
  --input_json data/subset_1000/val_100.json \
  --root_dir data/subset_1000 \
  --ckpt_dir output/stage2_full/checkpoint-30000 \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --cfg_path configs/kontext_train_1024.py \
  --transp_vae_ckpt "$TRANSP_VAE" \
  --max_samples 50 \
  --steps 30 \
  --output_dir validate_out/checkpoint-30000