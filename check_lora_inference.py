"""
诊断脚本: 验证 infer.py 加载 LoRA 后, transformer 的实际权重是否变化。
如果 load_lora_weights 前后某个 attention 层的 effective weight 没变,
说明 LoRA 没有真正生效, 推理必然等价于基座模型(噪声)。

用法:
    python check_lora_inference.py \
        --ckpt_dir output/stage1_test/final \
        --cfg_path configs/kontext_train_1024.py \
        --pretrained_model_name_or_path /path/to/FLUX.1-Kontext-dev
"""
import os
import sys
import argparse
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict
from safetensors.torch import load_file

from models.custom_model_mmdit import CustomFluxTransformer2DModel
from models.lora_utils import apply_lora, load_lora_weights_into_transformer


def parse_config(path):
    cfg = Config.fromfile(path)
    cfg.config_dir = path
    return cfg


def build_transformer(base_path, max_layer=12):
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        base_path, subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = max_layer
    mmdit_config = FrozenDict(mmdit_config)
    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    del transformer_orig
    return transformer


def get_effective_weight(module):
    """获取 PEFT LoRA 层的 effective weight (base + lora_B @ lora_A * scale)。"""
    base = module.base_layer.weight
    if hasattr(module, "lora_A") and "default" in module.lora_A:
        lora_A = module.lora_A["default"].weight
        lora_B = module.lora_B["default"].weight
        scale = module.scaling["default"]
        return base + lora_B @ lora_A * scale
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, default="./configs/kontext_train_1024.py")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--max_layer", type=int, default=12)
    parser.add_argument("--target_layer", type=str, default="transformer_blocks.0.attn.to_q")
    args = parser.parse_args()

    cfg = parse_config(args.cfg_path)
    base_path = args.pretrained_model_name_or_path or cfg.pretrained_model_name_or_path

    # 1) 加载基座 transformer (无 LoRA)
    transformer = build_transformer(base_path, args.max_layer)
    base_weight = dict(transformer.named_parameters())[f"{args.target_layer}.weight"].clone().float()
    print(f"[INFO] Base {args.target_layer}.weight norm: {base_weight.norm().item():.4f}")

    # 2) 先转成 PEFT 模型 (add_adapter) —— 这是 infer.py 新逻辑
    print("[INFO] Applying LoRA adapter structure...")
    transformer = apply_lora(transformer, cfg)

    # 3) 用新的 load_lora_weights_into_transformer 加载训练权重
    lora_path = os.path.join(args.ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        print(f"[ERROR] 找不到 {lora_path}")
        return
    try:
        load_lora_weights_into_transformer(transformer, lora_path, adapter_name="default")
    except Exception as e:
        print(f"[ERROR] load_lora_weights_into_transformer failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 4) 加载后检查 effective weight
    target_module = dict(transformer.named_modules()).get(args.target_layer)
    if target_module is None:
        print(f"[ERROR] 找不到模块 {args.target_layer}")
        return

    if hasattr(target_module, "base_layer"):
        eff_weight = get_effective_weight(target_module).float()
        print(f"[INFO] After LoRA effective {args.target_layer}.weight norm: {eff_weight.norm().item():.4f}")
        diff_norm = (eff_weight - base_weight).norm().item()
        print(f"[INFO] diff norm (effective - base): {diff_norm:.4f}")
        if diff_norm < 1e-3:
            print("\n[ERROR] LoRA 没有生效! effective weight 与基座几乎相同。")
        else:
            print("\n[OK] LoRA 已成功合并到 transformer 权重, 差异显著。")
    else:
        print(f"[ERROR] {args.target_layer} 不是 PEFT LoRA 层, 加载逻辑异常。")


if __name__ == "__main__":
    main()
