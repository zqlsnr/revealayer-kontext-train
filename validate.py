"""
RevealLayer 验证推理脚本
========================
加载训练得到的 checkpoint (LoRA + layer_pe + Refiner), 对指定 JSON 推理,
计算 PSNR / SSIM 指标, 保存可视化对比图, 输出是否通过验证的判定。

典型用途 (两阶段流程):
  # stage1: 1000 样本测试训练后, 验证同一批样本
  python validate.py \
      --input_json ./data/reveallayer_100k_subset_1000.json \
      --ckpt_dir ./output/kontext_train_1024/final \
      --pretrained_model_name_or_path ./models/FLUX.1-Kontext-dev \
      --cfg_path ./configs/kontext_train_1024.py \
      --max_samples 50 \
      --output_dir ./output/stage1_validate

判定标准 (默认, 可在命令行覆盖):
  - 背景 PSNR >= 20 dB 且 SSIM >= 0.7  → PASS
  - 否则 → FAIL (建议排查训练/数据问题后再跑全量)

依赖: 复用 infer.py 的 pipeline 初始化与几何工具。
"""

import os
import sys
import json
import math
import argparse
import shutil

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用 infer.py 的组件
from infer import (
    parse_config, seed_everything, initialize_pipeline,
    filter_and_align_bboxes, adjust_coordinate,
    _transform_image_consistent_crop,
)
from dataset import resize_with_bbox_edge
from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE


# ============================================================================ #
#  指标
# ============================================================================ #
def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: [H,W,3] uint8 或 float [0,1]。返回 PSNR (dB)。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    max_val = 255.0 if a.max() > 1.5 else 1.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """简化 SSIM (单尺度, 11x11 高斯窗)。a,b: [H,W,3] uint8。返回 [0,1]。"""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    except Exception:
        # fallback: 用 OpenCV
        a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu_a = cv2.GaussianBlur(a_g, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b_g, (11, 11), 1.5)
        mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
        sig_a2 = cv2.GaussianBlur(a_g ** 2, (11, 11), 1.5) - mu_a2
        sig_b2 = cv2.GaussianBlur(b_g ** 2, (11, 11), 1.5) - mu_b2
        sig_ab = cv2.GaussianBlur(a_g * b_g, (11, 11), 1.5) - mu_ab
        num = (2 * mu_ab + c1) * (2 * sig_ab + c2)
        den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
        return float(np.mean(num / den))


def composite_rgba_on_bg(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """把 RGBA 合成到背景上, 返回 RGB uint8。"""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    bg = bg.astype(np.float32)
    out = rgb * alpha + bg * (1 - alpha)
    return out.clip(0, 255).astype(np.uint8)


# ============================================================================ #
#  单样本推理 + 指标
# ============================================================================ #
def _resolve_path(p: str, root_dir: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p) or not root_dir:
        return p
    return os.path.join(root_dir, p)


def validate_one_sample(args, entry, pipeline, transp_vae, device, idx):
    """推理一个样本并计算与 GT 的指标。返回 dict。"""
    full_image_path = _resolve_path(entry.get("full_image", ""), args.root_dir)
    if not full_image_path or not os.path.exists(full_image_path):
        print(f"[Validate] missing image: {full_image_path}")
        return None

    imgid = entry.get("imgid", f"{idx:06d}")
    full_image = Image.open(full_image_path).convert("RGB")
    orig_w, orig_h = full_image.size

    (cont_h, cont_w), (tgt_h, tgt_w), (off_y, off_x) = resize_with_bbox_edge(
        full_image, resolution=1024,
    )
    layout = [d["bbox"] for d in entry.get("detections", [])]
    resized_boxes = filter_and_align_bboxes(
        layout, orig_w, orig_h, cont_w, cont_h, off_x, off_y, tgt_w, tgt_h,
    )
    resized_image = _transform_image_consistent_crop(
        full_image, cont_w, cont_h, tgt_w, tgt_h, off_x, off_y,
    )
    validation_boxes = [[0, 0, tgt_w, tgt_h], [0, 0, tgt_w, tgt_h]] + resized_boxes
    if len(validation_boxes) > 12:
        validation_boxes = validation_boxes[:12]

    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    full_tensor = image_transform(resized_image).to(device=device, dtype=torch.bfloat16).unsqueeze(0)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    output, rgba_output, _, _ = pipeline(
        prompt="Decompose the image into foreground and background.",
        full_image=full_tensor,
        validation_box=validation_boxes,
        generator=generator,
        height=tgt_h, width=tgt_w,
        num_layers=len(validation_boxes),
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        transparent_decoder=transp_vae,
    )

    # rgba_output: List[np.ndarray HxWx4 uint8]
    metrics = {"imgid": imgid, "n_layers": len(rgba_output)}
    sample_dir = os.path.join(args.output_dir, imgid)
    os.makedirs(sample_dir, exist_ok=True)

    # 保存推理结果
    rgba_images = [Image.fromarray(arr, "RGBA") for arr in rgba_output]
    for i, img in enumerate(rgba_images):
        img.save(os.path.join(sample_dir, f"pred_{i}.png"))
    # 合成图
    merged = rgba_images[0].copy()
    for i in range(1, len(rgba_images)):
        merged = Image.alpha_composite(merged, rgba_images[i])
    merged.save(os.path.join(sample_dir, "merged.png"))

    # ---- 与 GT 对比 ----
    # 背景 GT (layer 1)
    bg_path = _resolve_path(entry.get("background", ""), args.root_dir)
    if bg_path and os.path.exists(bg_path):
        try:
            bg_gt = Image.open(bg_path).convert("RGB").resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)
            bg_gt_arr = np.array(bg_gt)
            if len(rgba_output) >= 2:
                pred_bg = rgba_output[1][:, :, :3]  # 背景层 RGB
                metrics["bg_psnr"] = psnr_rgb(pred_bg, bg_gt_arr)
                metrics["bg_ssim"] = ssim_rgb(pred_bg, bg_gt_arr)
                # 可视化对比
                cmp = np.concatenate([bg_gt_arr, pred_bg], axis=1)
                Image.fromarray(cmp).save(os.path.join(sample_dir, "cmp_bg.png"))
        except Exception as e:
            metrics["bg_error"] = str(e)

    # 前景层 GT
    layer_paths = [_resolve_path(lp, args.root_dir) for lp in (entry.get("LayerInfoRaw", []) or [])]
    fg_psnr_list, fg_ssim_list = [], []
    for i, lp in enumerate(layer_paths):
        if i + 2 >= len(rgba_output):
            break
        if not lp or not os.path.exists(lp):
            continue
        try:
            gt_layer = Image.open(lp).convert("RGBA").resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)
            gt_arr = np.array(gt_layer)
            pred_layer = rgba_output[i + 2]
            # 对齐尺寸
            if pred_layer.shape[:2] != gt_arr.shape[:2]:
                pred_layer = np.array(
                    Image.fromarray(pred_layer, "RGBA").resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)
                )
            # RGB 指标 — 只在物体区域 (GT alpha>0) 计算, 避免透明区灰底差异拉低分数 (8/24 修复)
            # 原逻辑: 全图 RGB 对比 (含 alpha=0 透明区, GT 灰底 vs pred 灰底差异大 → fg_psnr 假低 ~10.9)
            mask = gt_arr[:, :, 3] > 0
            if mask.sum() == 0:
                continue
            fg_psnr_list.append(psnr_rgb(pred_layer[:, :, :3][mask], gt_arr[:, :, :3][mask]))
            # SSIM 需要 [H,W,3] 形状: mask 外填 GT (不贡献差异)
            gt_fill = gt_arr[:, :, :3].copy()
            pred_fill = pred_layer[:, :, :3].copy()
            pred_fill[~mask] = gt_fill[~mask]
            fg_ssim_list.append(ssim_rgb(pred_fill, gt_fill))
            # alpha 指标
            if pred_layer.shape[2] == 4 and gt_arr.shape[2] == 4:
                a_psnr = psnr_rgb(pred_layer[:, :, 3:], gt_arr[:, :, 3:])
                metrics[f"fg{i}_alpha_psnr"] = a_psnr
            cmp = np.concatenate([gt_arr[:, :, :3], pred_layer[:, :, :3]], axis=1)
            Image.fromarray(cmp).save(os.path.join(sample_dir, f"cmp_fg{i}.png"))
        except Exception as e:
            metrics[f"fg{i}_error"] = str(e)

    if fg_psnr_list:
        metrics["fg_psnr_mean"] = float(np.mean(fg_psnr_list))
        metrics["fg_ssim_mean"] = float(np.mean(fg_ssim_list))

    return metrics


# ============================================================================ #
#  main
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="RevealLayer 验证推理 + 指标")
    parser.add_argument("--input_json", type=str, required=True,
                        help="验证集 JSON (可用 train.py --subset_json 生成的 1000 子集)")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="训练 checkpoint 目录 (含 pytorch_lora_weights.safetensors, layer_pe.pt, Refiner.pt)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="FLUX.1-Kontext-dev 模型目录")
    parser.add_argument("--cfg_path", type=str, default="./configs/kontext_train_1024.py")
    parser.add_argument("--transp_vae_ckpt", type=str, default="",
                        help="透明解码器权重 (可选, 不传则不输出 alpha 指标)")
    parser.add_argument("--output_dir", type=str, default="./output/validate")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="验证样本数 (-1 = 全部)")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cfg", type=float, default=1.0, help="guidance scale")
    parser.add_argument("--steps", type=int, default=30, help="inference steps")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_layer", type=int, default=12)
    # 验收阈值
    parser.add_argument("--bg_psnr_threshold", type=float, default=20.0)
    parser.add_argument("--bg_ssim_threshold", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    # ---- 加载数据 ----
    with open(args.input_json, "r") as f:
        entries = json.load(f)
    if args.max_samples > 0 and len(entries) > args.max_samples:
        import random
        rng = random.Random(args.seed)
        entries = rng.sample(entries, args.max_samples)
    print(f"[Validate] {len(entries)} samples from {args.input_json}")

    # ---- 初始化 pipeline (复用 infer.py) ---- #
    # 用一个简易 args 对象喂给 infer.initialize_pipeline
    class InferArgs:
        pass
    infer_args = InferArgs()
    infer_args.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    infer_args.ckpt_dir = args.ckpt_dir
    infer_args.gpu_id = args.gpu_id
    infer_args.max_layer = args.max_layer
    infer_args.extra_lora_dir = None

    config = parse_config(args.cfg_path)
    # 覆盖配置中的模型路径 (用命令行传入的)
    config.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    device = torch.device("cuda", index=args.gpu_id)

    # ---- checkpoint 完整性检查 ----
    required_files = {
        "LoRA": os.path.join(args.ckpt_dir, "pytorch_lora_weights.safetensors"),
        "layer_pe": os.path.join(args.ckpt_dir, "layer_pe.pt"),
        "Refiner": os.path.join(args.ckpt_dir, "Refiner.pt"),
    }
    missing = []
    for name, path in required_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[Validate] {name} found: {path} ({size_mb:.2f} MB)")
        else:
            print(f"[Validate] [ERROR] {name} NOT found: {path}")
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"Checkpoint 不完整, 缺少: {missing}. "
            f"请确认训练已正常完成并保存到 {args.ckpt_dir}"
        )

    print("[Validate] initializing pipeline...")
    pipeline = initialize_pipeline(config, infer_args)

    # 确认 LoRA adapter 已激活
    # 注意: 训练/推理主路径用 adapter_name="default" (见 infer._load_loras -> lora_utils.load_lora_weights_into_transformer),
    #       旧版 fallback 路径用 "layer", 这里两种都接受。
    active_adapters = []
    if hasattr(pipeline, "get_active_adapters"):
        try:
            active_adapters = pipeline.get_active_adapters()
        except Exception:
            pass
    print(f"[Validate] active adapters: {active_adapters}")
    if not any(a in active_adapters for a in ("layer", "default")):
        raise RuntimeError(
            "LoRA adapter ('layer'/'default') 未成功加载到 pipeline. "
            "可能原因: LoRA state dict key 格式不对 / checkpoint 损坏 / 训练未收敛保存. "
            "请检查训练日志和 checkpoint 文件."
        )

    transp_vae = None
    if args.transp_vae_ckpt and os.path.exists(args.transp_vae_ckpt):
        transp_vae = CustomVAE()
        transp_vae.load_state_dict(torch.load(args.transp_vae_ckpt, map_location="cpu"), strict=False)
        transp_vae.to(device).eval()
    else:
        print("[Validate] transp_vae_ckpt not provided, alpha metrics disabled.")

    # ---- 推理 + 指标 ---- #
    all_metrics = []
    for idx, entry in enumerate(tqdm(entries, desc="Validating")):
        try:
            m = validate_one_sample(args, entry, pipeline, transp_vae, device, idx)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            print(f"[Validate] sample {entry.get('imgid','?')} failed: {e}")
            import traceback; traceback.print_exc()

    # ---- 汇总 ---- #
    def agg(key):
        vals = [m[key] for m in all_metrics if key in m and m[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_samples": len(all_metrics),
        "bg_psnr_mean": agg("bg_psnr"),
        "bg_ssim_mean": agg("bg_ssim"),
        "fg_psnr_mean": agg("fg_psnr_mean"),
        "fg_ssim_mean": agg("fg_ssim_mean"),
    }

    # 判定
    bg_psnr = summary["bg_psnr_mean"] or 0.0
    bg_ssim = summary["bg_ssim_mean"] or 0.0
    passed = (bg_psnr >= args.bg_psnr_threshold) and (bg_ssim >= args.bg_ssim_threshold)
    summary["passed"] = bool(passed)
    summary["thresholds"] = {
        "bg_psnr": args.bg_psnr_threshold,
        "bg_ssim": args.bg_ssim_threshold,
    }

    summary_path = os.path.join(args.output_dir, "validate_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "per_sample": all_metrics}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(" 验证结果汇总")
    print("=" * 60)
    print(f"  样本数        : {summary['n_samples']}")
    print(f"  背景 PSNR     : {summary['bg_psnr_mean']:.4f} dB  (阈值 {args.bg_psnr_threshold})")
    print(f"  背景 SSIM     : {summary['bg_ssim_mean']:.4f}     (阈值 {args.bg_ssim_threshold})")
    print(f"  前景 PSNR     : {summary['fg_psnr_mean']:.4f} dB" if summary['fg_psnr_mean'] else "  前景 PSNR     : N/A")
    print(f"  前景 SSIM     : {summary['fg_ssim_mean']:.4f}" if summary['fg_ssim_mean'] else "  前景 SSIM     : N/A")
    print(f"  判定          : {'PASS ✅' if passed else 'FAIL ❌  (建议排查后再跑全量)'}")
    print(f"  明细          : {summary_path}")
    print(f"  可视化        : {args.output_dir}/<imgid>/cmp_*.png")
    print("=" * 60)

    # 写判定标志文件, 供 train.sh 判断是否继续 stage2
    flag_path = os.path.join(args.output_dir, "VALIDATE_RESULT")
    with open(flag_path, "w") as f:
        f.write("PASS\n" if passed else "FAIL\n")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
