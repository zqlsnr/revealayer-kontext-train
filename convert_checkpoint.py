"""
转换/修复 RevealLayer 训练 checkpoint, 使其能被 infer.py/validate.py 正确加载。

用法:
    python convert_checkpoint.py \
        --src ./output/stage1_test/final \
        --dst ./output/stage1_test/final_fixed

主要修复:
1. LoRA key 格式统一为 diffusers pipeline.load_lora_weights 期望的
   transformer.<block_path>.lora_A/B.weight
2. 如果源文件是 pytorch_lora_weights.safetensors.pt, 转成 pytorch_lora_weights.safetensors
3. 如果源文件是 extra_modules.pt, 拆分为 layer_pe.pt 和 Refiner.pt
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def correct_lora_keys(lora_sd):
    """统一 LoRA key 格式为 diffusers 期望的 transformer.xxx。"""
    new_sd = {}
    for k, v in lora_sd.items():
        k = k.replace("module.", "")
        k = k.replace("base_model.model.", "")
        if k.startswith("transformer.module.transformer."):
            k = k.replace("transformer.module.transformer.", "transformer.", 1)
        if not k.startswith("transformer."):
            k = "transformer." + k
        new_sd[k] = v
    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="源 checkpoint 目录")
    parser.add_argument("--dst", type=str, required=True, help="目标 checkpoint 目录")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    src = Path(args.src)
    dst = Path(args.dst)

    # ---------- LoRA ----------
    lora_candidates = [
        src / "pytorch_lora_weights.safetensors.pt",
        src / "pytorch_lora_weights.pt",
        src / "pytorch_lora_weights.safetensors",
    ]
    lora_src = None
    for c in lora_candidates:
        if c.exists():
            lora_src = c
            break

    if lora_src is None:
        print(f"[WARN] 未找到 LoRA 文件, 跳过. 搜索路径: {lora_candidates}")
    else:
        print(f"[LoRA] 加载 {lora_src}")
        if str(lora_src).endswith(".safetensors"):
            lora_sd = load_file(str(lora_src))
        else:
            lora_sd = torch.load(str(lora_src), map_location="cpu", weights_only=True)

        lora_sd = correct_lora_keys(lora_sd)
        print(f"[LoRA] key 数量: {len(lora_sd)}, 前5个: {list(lora_sd.keys())[:5]}")
        if not lora_sd:
            raise ValueError("LoRA state dict 为空!")

        save_file(lora_sd, str(dst / "pytorch_lora_weights.safetensors"))
        print(f"[LoRA] 已保存到 {dst / 'pytorch_lora_weights.safetensors'}")

    # ---------- layer_pe / Refiner ----------
    # layer_pe.pt 可能是 tensor 或 dict; extra_modules.pt 是 dict
    layer_pe_src = src / "layer_pe.pt"
    extra_modules_src = src / "extra_modules.pt"

    # layer_pe
    if layer_pe_src.exists():
        print(f"[Extra] 加载 {layer_pe_src}")
        layer_pe_data = torch.load(str(layer_pe_src), map_location="cpu", weights_only=True)
        torch.save(layer_pe_data, str(dst / "layer_pe.pt"))
        print(f"[Extra] layer_pe 已保存")
    elif extra_modules_src.exists():
        print(f"[Extra] 加载 {extra_modules_src}")
        extra_sd = torch.load(str(extra_modules_src), map_location="cpu", weights_only=True)
        if isinstance(extra_sd, dict):
            if "layer_pe" in extra_sd:
                torch.save(extra_sd["layer_pe"], str(dst / "layer_pe.pt"))
                print(f"[Extra] layer_pe 已保存 (from extra_modules.pt)")
            else:
                print("[WARN] extra_modules.pt 中没有 layer_pe")
        else:
            torch.save(extra_sd, str(dst / "layer_pe.pt"))
            print(f"[Extra] extra_modules.pt 内容是 tensor, 直接保存为 layer_pe.pt")
    else:
        print(f"[WARN] 未找到 layer_pe.pt 或 extra_modules.pt")

    # Refiner
    if extra_modules_src.exists():
        extra_sd = torch.load(str(extra_modules_src), map_location="cpu", weights_only=True)
        if isinstance(extra_sd, dict):
            refiner_sd = {
                k.replace("refiner.", ""): v
                for k, v in extra_sd.items() if k.startswith("refiner.")
            }
            if refiner_sd:
                torch.save(refiner_sd, str(dst / "Refiner.pt"))
                print(f"[Extra] Refiner 已保存, keys={len(refiner_sd)}")
            else:
                print("[WARN] extra_modules.pt 中没有 refiner")
        else:
            print("[WARN] extra_modules.pt 是 tensor, 不含 refiner")

    print(f"\n转换完成: {args.dst}")
    print("请用 --ckpt_dir 指向新目录重新跑 validate/infer。")


if __name__ == "__main__":
    main()
