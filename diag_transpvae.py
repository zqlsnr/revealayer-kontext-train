#!/usr/bin/env python
"""诊断 TranspVAE 对不同颜色 latent 解码的 alpha 行为。

目标: 定位"bbox 内物体外黑色不透明"的根因:
- 如果 TranspVAE 对【黑色 latent】解码出 alpha≈255 → 推理侧 pixel_grey=zeros(黑色) 用错, 应改灰色
- 如果 TranspVAE 对【黑色 latent】解码 alpha≈0 → 问题在模型生成的 latent (训练侧)

测试矩阵: 黑色(0) / 灰色(0.5) / 白色(1) / 随机噪声 / 真实图像
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "diffusers" / "src"))

from diffusers import AutoencoderKL


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--transp_vae_ckpt", required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--resolution", type=int, default=512, help="色块分辨率 (越小越快)")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda", index=args.gpu_id)
    weight_dtype = torch.bfloat16
    R = args.resolution

    # 1) 加载 VAE
    print("[diag] loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)
    vae.eval()

    # 2) 加载 TranspVAE (与 train.py 一致)
    print("[diag] loading TranspVAE...")
    from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE
    transp_vae = CustomVAE().to(device, dtype=weight_dtype)
    ckpt = torch.load(args.transp_vae_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = transp_vae.load_state_dict(ckpt, strict=False)
    print(f"[diag] TranspVAE missing={len(missing)} unexpected={len(unexpected)}")
    transp_vae.eval()

    # 3) 构造测试色块 (RGB -1~1 输入 VAE)
    def make_img(value):
        """value: 标量 0~1 → 全图该色 [1,3,H,W] -1~1"""
        v = value * 2.0 - 1.0
        return torch.full((1, 3, R, R), v, device=device, dtype=weight_dtype)

    tests = {
        "黑色(0.0)": make_img(0.0),
        "灰底(0.5)": make_img(0.5),
        "白色(1.0)": make_img(1.0),
        "噪声": torch.randn(1, 3, R, R, device=device, dtype=weight_dtype),
    }
    # 真实图像: 用随机色块拼一个"灰底+物体"模拟 fg 层
    real = torch.zeros(1, 3, R, R, device=device, dtype=weight_dtype)
    real[:, :, R//4:3*R//4, R//4:3*R//4] = 1.0  # 中心白色块 (模拟物体)
    tests["中心白块(灰底模拟)"] = real

    # 4) 解码并统计 alpha
    # TranspVAE 的 box 参数结构: [batch][layer_idx] = (x1,y1,x2,y2)
    # z_in 是 [T=1, C, H, W] 单层, 所以 box = [[(0,0,R,R)]]
    full_box = [[(0, 0, R, R)]]
    print("\n===== TranspVAE 解码统计 (alpha -1~1 → 0~1) =====")
    print(f"{'输入':<22}{'alpha mean':>10}{'alpha max':>10}{'alpha std':>10}{'RGB mean':>10}")
    with torch.no_grad():
        for name, img in tests.items():
            lat = vae.encode(img).latent_dist.sample()   # VAE 原始空间 latent [1,16,h,w]
            lat = lat.to(weight_dtype)
            # TranspVAE 输入: [T, C, H, W] (T=1)
            z_in = lat[0].unsqueeze(0)
            try:
                fg, alpha = transp_vae(z_in, full_box)   # [1,3,H*8,W*8], [1,1,H*8,W*8]
                alpha01 = ((alpha + 1.0) / 2.0).clamp(0, 1)
                rgb01 = ((fg + 1.0) / 2.0).clamp(0, 1)
                print(f"{name:<22}{alpha01.mean().item():>10.3f}{alpha01.max().item():>10.3f}"
                      f"{alpha01.std().item():>10.3f}{rgb01.mean().item():>10.3f}")
            except Exception as e:
                print(f"{name:<22} ERROR: {e}")

    # 5) 关键判别
    print("\n===== 判别 =====")
    print("若 黑色 latent → alpha 高: 推理侧 pixel_grey=zeros(黑) 是元凶, 应改灰色(0.5)")
    print("若 黑色 latent → alpha≈0: TranspVAE 正常, 问题在模型生成的 latent (训练侧)")


if __name__ == "__main__":
    main()