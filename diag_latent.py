"""
diag_latent.py — 诊断推理噪声来源
==================================
对单张图跑完整推理, 然后分别检查:
  1) 模型输出的各层 latent 统计量 (mean/std, 与纯噪声/干净 latent 对比)
  2) 主 VAE 解码 latent 得到的 RGB (绕开 TranspVAE, 看 latent 本身有无结构)
  3) TranspVAE 解码的 RGBA 的 alpha 通道统计 (是否全 0 → 透明马赛克)

用法:
  python diag_latent.py \
      --image <full_image.png> \
      --boxes '[[x1,y1,x2,y2], ...]' \
      --ckpt_dir output/stage1_test/final \
      --pretrained_model_name_or_path ./models/FLUX.1-Kontext-dev \
      --transp_vae_ckpt ./models/RevealLayer/xvae/transparent_decoder_ckpt.pth \
      --out diag_out

判定:
  - layer0 (full 条件层) 的 VAE-RGB 应清晰; 若连 layer0 都噪声 → latent 输入就有问题
  - 各层 latent std ≈ 1.0 接近纯噪声 → 模型输出没学到结构 (训练/输入问题)
  - VAE-RGB 有结构但 TranspVAE RGBA 噪声/alpha≈0 → TranspVAE 解码问题
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image
from torchvision import transforms

from infer import (
    seed_everything,
    parse_config,
    initialize_pipeline,
    resize_with_bbox_edge,
    filter_and_align_bboxes,
    _transform_image_consistent_crop,
)
from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE


def main():
    p = argparse.ArgumentParser(description="diagnose noise source in RevealLayer inference")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--boxes", type=str, required=True, help='JSON: [[x1,y1,x2,y2],...]')
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--transp_vae_ckpt", type=str, default="")
    p.add_argument("--cfg_path", type=str, default="./configs/ld_resolution1024_test.py")
    p.add_argument("--out", type=str, default="./diag_out")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_layer", type=int, default=12)
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    seed_everything(args.seed)
    config = parse_config(args.cfg_path)
    config.pretrained_model_name_or_path = args.pretrained_model_name_or_path

    class _A:
        pass
    ia = _A()
    ia.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    ia.ckpt_dir = args.ckpt_dir
    ia.gpu_id = args.gpu_id
    ia.max_layer = args.max_layer
    ia.extra_lora_dir = None

    print("[diag] initializing pipeline...")
    pipeline = initialize_pipeline(config, ia)
    vae = pipeline.vae
    device = torch.device("cuda", index=args.gpu_id)

    transp_vae = None
    if args.transp_vae_ckpt and os.path.exists(args.transp_vae_ckpt):
        transp_vae = CustomVAE()
        transp_vae.load_state_dict(torch.load(args.transp_vae_ckpt, map_location="cpu"), strict=False)
        transp_vae.to(device).eval()
        print("[diag] transp_vae loaded.")

    # ---- 图像预处理 (与 infer.py 完全一致) ----
    img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img.size
    # infer.resize_with_bbox_edge 签名: (image, bbox, resolution, k), 返回 (content, target, offset, new_bbox)
    (cont_h, cont_w), (tgt_h, tgt_w), (off_y, off_x), _ = resize_with_bbox_edge(img, [], resolution=1024)
    boxes = json.loads(args.boxes)
    resized_boxes = filter_and_align_bboxes(
        boxes, orig_w, orig_h, cont_w, cont_h, off_x, off_y, tgt_w, tgt_h
    )
    resized = _transform_image_consistent_crop(img, cont_w, cont_h, tgt_w, tgt_h, off_x, off_y)
    tf = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    full_tensor = tf(resized).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    vboxes = [[0, 0, tgt_w, tgt_h], [0, 0, tgt_w, tgt_h]] + resized_boxes
    if len(vboxes) > args.max_layer:
        vboxes = vboxes[:args.max_layer]
    print(f"[diag] image={os.path.basename(args.image)} tgt={tgt_w}x{tgt_h} layers={len(vboxes)} boxes={vboxes}")

    # ---- 自检 0: 编码->解码自循环 (无模型参与), 验证 VAE 链路本身是否正常 ----
    print("\n===== 0) VAE 自检: encode(full_image) -> decode 应还原原图 =====")
    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        z_raw = vae.encode(full_tensor).latent_dist.sample()          # raw latent
        z_scaled = (z_raw - vae.config.shift_factor) * vae.config.scaling_factor
        z_back = (z_scaled / vae.config.scaling_factor) + vae.config.shift_factor
        rgb_self = vae.decode(z_back, return_dict=False)[0].float()
        rgb_self = (rgb_self.clamp(-1, 1) + 1) / 2
        arr0 = (rgb_self[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(arr0).save(os.path.join(args.out, "selfcheck_encdec.png"))
        print(f"selfcheck enc->dec: raw_std={z_raw.float().std():.4f} scaled_std={z_scaled.float().std():.4f} "
              f"rgb mean={arr0.mean():.1f} std={arr0.std():.1f} -> selfcheck_encdec.png")
        # 对照: 原图直接显示
        arr_orig = (full_tensor[0].float().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        Image.fromarray((arr_orig * 255).astype("uint8")).save(os.path.join(args.out, "selfcheck_orig.png"))
        print("selfcheck_orig.png (原图) saved for comparison")

    # ---- 推理 (拿第 4 个返回值 = 处理后的 latents) ----
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out, rgba_output, _, latents = pipeline(
        prompt="Decompose the image into foreground and background.",
        full_image=full_tensor,
        validation_box=vboxes,
        generator=gen,
        height=tgt_h, width=tgt_w,
        num_layers=len(vboxes),
        guidance_scale=1.0,
        num_inference_steps=args.steps,
        transparent_decoder=transp_vae,
    )

    print("\n===== 1) latent 统计 (unshift 后, bbox 外为 grey) =====")
    print("latents shape:", tuple(latents.shape))
    for i in range(latents.shape[0]):
        z = latents[i].float()
        print(f"layer{i} box={vboxes[i]} mean={z.mean():.4f} std={z.std():.4f} "
              f"min={z.min():.3f} max={z.max():.3f}")

    print("\n===== 2) 主 VAE 解码每层 RGB (绕开 TranspVAE) =====")
    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        rgb = vae.decode(latents, return_dict=False)[0]  # [n,3,H,W] -1~1
        rgb = (rgb.clamp(-1, 1) + 1) / 2
    for i in range(rgb.shape[0]):
        arr = (rgb[i].float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(os.path.join(args.out, f"vae_rgb_{i}.png"))
        print(f"layer{i}: VAE-RGB mean={arr.mean():.1f} std={arr.std():.1f} -> vae_rgb_{i}.png")

    print("\n===== 3) TranspVAE RGBA 的 alpha 统计 =====")
    if rgba_output is not None:
        for i, arr in enumerate(rgba_output):
            alpha = arr[:, :, 3]
            rgbp = arr[:, :, :3]
            print(f"transp layer{i}: alpha mean={alpha.mean():.3f} max={int(alpha.max())} "
                  f"rgb mean={rgbp.mean():.1f} -> transp_rgba_{i}.png")
            Image.fromarray(arr, "RGBA").save(os.path.join(args.out, f"transp_rgba_{i}.png"))
    else:
        print("(transp_vae 未加载, 无 RGBA 输出)")

    print(f"\n[diag] DONE -> {args.out}")
    print("判定: layer0 的 VAE-RGB 应清晰; 若所有层 VAE-RGB 都噪声 -> 模型输出 latent 本身是噪声;")


if __name__ == "__main__":
    main()
