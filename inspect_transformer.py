"""
查看 CustomFluxTransformer2DModel 中所有 nn.Linear 模块的完整路径,
用于确认 LoRA target_modules 应该写什么。

用法:
    python inspect_transformer.py \
        --pretrained_model_name_or_path /path/to/FLUX.1-Kontext-dev \
        --max_layer 12
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict
from models.custom_model_mmdit import CustomFluxTransformer2DModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--max_layer", type=int, default=12)
    args = parser.parse_args()

    print("Loading base transformer...")
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    config = dict(transformer_orig.config)
    config["_class_name"] = "CustomSD3Transformer2DModel"
    config["max_layer_num"] = args.max_layer
    config = FrozenDict(config)

    print("Building custom transformer...")
    transformer = CustomFluxTransformer2DModel.from_config(config)
    transformer.load_state_dict(transformer_orig.state_dict(), strict=False)

    print("\nAll nn.Linear modules (full paths):")
    linear_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(name)
            print(f"  {name}")

    print(f"\nTotal nn.Linear modules: {len(linear_modules)}")

    print("\nAttention-related modules (contain 'attn'):")
    for name in linear_modules:
        if "attn" in name:
            print(f"  {name}")

    print("\nSuggested target_modules for LoRA:")
    suffixes = set()
    for name in linear_modules:
        if "attn" in name:
            # take last 2 or 3 components
            parts = name.split(".")
            if len(parts) >= 2:
                suffixes.add(".".join(parts[-2:]))
            if len(parts) >= 3:
                suffixes.add(".".join(parts[-3:]))
    for s in sorted(suffixes):
        print(f"  {s}")


if __name__ == "__main__":
    main()
