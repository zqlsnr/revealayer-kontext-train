"""
诊断脚本: 检查训练好的 checkpoint 里 LoRA 实际覆盖了哪些模块,
以及这些模块的权重是否有效(非零/非 NaN)。

用法:
    python check_lora.py --ckpt_dir output/stage1_test/final
"""
import os
import sys
import argparse
from collections import defaultdict

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    args = parser.parse_args()

    lora_path = os.path.join(args.ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        print(f"[ERROR] 找不到 {lora_path}")
        return

    print(f"[INFO] 加载 {lora_path}")
    sd = load_file(lora_path)

    print(f"\n总 LoRA key 数: {len(sd)}")

    # 按模块分组
    modules = defaultdict(list)
    for k in sd.keys():
        # 去掉 lora_A/lora_B 后缀
        module_name = k.rsplit(".lora_", 1)[0]
        modules[module_name].append(k)

    print(f"\n实际覆盖的模块数: {len(modules)}")
    print("\n模块分布:")
    for module_name in sorted(modules.keys()):
        keys = modules[module_name]
        print(f"  {module_name}: {len(keys)} keys")

    # 检查是否包含核心 attention 层
    core_blocks = [m for m in modules if "transformer_blocks" in m and "attn" in m]
    single_blocks = [m for m in modules if "single_transformer_blocks" in m and "attn" in m]
    print(f"\n核心 joint attention 层 (transformer_blocks.*.attn): {len(core_blocks)}")
    print(f"single attention 层 (single_transformer_blocks.*.attn): {len(single_blocks)}")

    if not core_blocks:
        print("\n[ERROR] 没有 LoRA 覆盖 transformer_blocks 的 attention 层!")
        print("这意味着 FLUX 主 transformer 没有被微调, 推理必然是噪声。")
        print("\n可能原因:")
        print("  1. apply_lora 的 target_modules 没有匹配到 attention 层")
        print("  2. PEFT 版本与 diffusers 不兼容")
        print("  3. 训练代码版本过旧, 未正确注入 LoRA")
        return

    # 统计权重异常
    nan_keys = []
    zero_keys = []
    for k, v in sd.items():
        if not torch.isfinite(v).all():
            nan_keys.append(k)
        if v.abs().max().item() < 1e-12:
            zero_keys.append(k)

    print(f"\n含 NaN/Inf 的 key 数: {len(nan_keys)}")
    print(f"全零 key 数: {len(zero_keys)}")
    if nan_keys:
        print("NaN keys (前5):", nan_keys[:5])
    if zero_keys:
        print("Zero keys (前5):", zero_keys[:5])

    # 权重大小统计
    lora_a_norms = []
    lora_b_norms = []
    for k, v in sd.items():
        if ".lora_A." in k:
            lora_a_norms.append(v.norm().item())
        elif ".lora_B." in k:
            lora_b_norms.append(v.norm().item())

    if lora_a_norms:
        print(f"\nlora_A weight norm: mean={sum(lora_a_norms)/len(lora_a_norms):.4e}, max={max(lora_a_norms):.4e}")
    if lora_b_norms:
        print(f"lora_B weight norm: mean={sum(lora_b_norms)/len(lora_b_norms):.4e}, max={max(lora_b_norms):.4e}")

    print("\n[INFO] 诊断完成")


if __name__ == "__main__":
    main()
