"""
LoRA 工具函数: 训练与推理共用, 避免 train.py / infer.py 重复实现。
"""
import os
from typing import Dict

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None


def apply_lora(transformer: nn.Module, cfg) -> nn.Module:
    """对 transformer 的 attention 线性层挂 LoRA, 并确保核心 attention 层被注入。"""
    if LoraConfig is None:
        raise ImportError("peft is required for LoRA training. pip install peft")
    rank = cfg.get("rank", 64) if hasattr(cfg, "get") else getattr(cfg, "rank", 64)

    target_modules_candidates = [
        ["to_q", "to_k", "to_v", "to_out.0",
         "to_qkv", "proj_out", "context_embedder", "x_embedder"],
        ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
         "to_qkv", "proj_out", "context_embedder", "x_embedder"],
    ]

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        BaseTunerLayer = None

    def count_core_attn():
        if BaseTunerLayer is None:
            return -1
        return sum(
            1 for name, module in transformer.named_modules()
            if isinstance(module, BaseTunerLayer)
            and "transformer_blocks" in name and "attn" in name
        )

    last_exception = None
    for target_modules in target_modules_candidates:
        try:
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
                init_lora_weights="gaussian",
            )
            transformer.add_adapter(lora_config)
            core_attn = count_core_attn()
            print(f"[LoRA] 尝试 target_modules={target_modules}, 核心 attention 注入数={core_attn}")
            if core_attn > 0:
                break
            try:
                transformer.delete_adapter("default")
            except Exception:
                pass
        except Exception as e:
            last_exception = e
            print(f"[LoRA] target_modules={target_modules} 失败: {e}")
    else:
        raise RuntimeError("LoRA 未注入核心 attention 层, 无法训练/推理。") from last_exception

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    print(f"[LoRA] rank={rank} | trainable={trainable/1e6:.2f}M / total={total/1e9:.2f}B")

    if BaseTunerLayer is not None:
        lora_modules = [name for name, module in transformer.named_modules()
                        if isinstance(module, BaseTunerLayer)]
        print(f"[LoRA] 实际注入模块数: {len(lora_modules)}")
        core_attn = [n for n in lora_modules if "transformer_blocks" in n and "attn" in n]
        if not core_attn:
            raise RuntimeError("LoRA 未注入核心 attention 层。")
    return transformer


def correct_lora_keys_for_inference(lora_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    把训练时 PEFT/DDP 保存的 LoRA key 转成 diffusers / PEFT 推理可识别的格式。
    最终 key 形如: transformer.transformer_blocks.0.attn.to_q.lora_A.weight
    """
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


def load_lora_weights_into_transformer(
    transformer: nn.Module,
    lora_path: str,
    adapter_name: str = "default",
) -> None:
    """
    直接对 PEFT transformer 加载 LoRA 权重 (用于推理)。
    支持 safetensors / .pt, 自动做 key 修正。
    """
    if set_peft_model_state_dict is None:
        raise ImportError("peft is required. pip install peft")

    if lora_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        lora_sd = load_file(lora_path)
    elif lora_path.endswith(".pt") or lora_path.endswith(".pth"):
        lora_sd = torch.load(lora_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported lora format: {lora_path}")

    lora_sd = correct_lora_keys_for_inference(lora_sd)
    # PEFT set_peft_model_state_dict 期望去掉 transformer. 前缀
    peft_sd = {}
    prefix = "transformer."
    for k, v in lora_sd.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        peft_sd[k] = v

    set_peft_model_state_dict(transformer, peft_sd, adapter_name=adapter_name)
    print(f"[LoRA] Loaded into transformer from {lora_path}, {len(peft_sd)} keys.")
