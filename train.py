"""
RevealLayer 多 GPU 训练脚本 (基于 FLUX.1-Kontext-dev)
=====================================================

按论文 "RevealLayer: Disentangling Hidden and Visible Layers via Occlusion-Aware
Image Decomposition" (arXiv:2605.11818) 第 3 节训练协议实现：

  * 基座: FLUX.1-Kontext-dev  (本脚本相对原 infer.py 的关键改动)
  * 微调: LoRA (rank=64) + 新初始化的 layer_pe / HybridRefiner (OGA)
  * 训练目标: Rectified Flow Matching Loss (论文式 5-7, 对齐 FLUX 官方 / 推理 scheduler)
        z_t = t·noise + (1-t)·sample        (t=噪声权重: t=1 纯噪声, t=0 clean, 与推理 sigmas 一致)
        v_target = noise - sample
        L_FM = Σ_i || v_θ^i(z_t, t, c_text, z_c) - v_target^i ||²
  * 时间采样: logit-normal  t = sigmoid(N(μ=0, σ=1))
  * 文本条件: 固定 prompt "Decompose the image into foreground and background"
  * 多 GPU: accelerate DDP, bf16 混合精度, gradient_checkpointing
  * 优化器: Prodigy (lr=1.0) 或 AdamW fallback
  * 损失: L_FM (+ 可选 alpha_loss / orthogonality_loss)
  * 配置: configs/kontext_train_1024.py

启动方式见 train.sh:
    torchrun --nproc_per_node=8 train.py --cfg_path configs/kontext_train_1024.py
"""

import os
import sys
import math
import json
import random
import argparse
import functools
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import logging as _logging
from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm

# ---------------------- accelerate ---------------------- #
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# ---------------------- diffusers ---------------------- #
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.utils import check_min_version
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

# ---------------------- peft / optim ---------------------- #
try:
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
except Exception:  # pragma: no cover
    LoraConfig = None
try:
    from prodigyopt import Prodigy
except Exception:
    Prodigy = None

# ---------------------- project ---------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config

from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE
from models.custom_model_mmdit import CustomFluxTransformer2DModel
from models.lora_utils import apply_lora, correct_lora_keys_for_inference
from dataset import RevealLayerDataset, collate_fn

check_min_version("0.31.0.dev0")

logger = get_logger(__name__)
# accelerate 的 logger 默认级别是 WARNING (ACCELERATE_LOG_LEVEL 默认不输出 INFO),
# 会导致 step/loss 日志看不到, 这里强制 INFO 级别。
logger.setLevel(_logging.INFO)


# ============================================================================ #
#  Utilities
# ============================================================================ #
def parse_config(path: str) -> Config:
    cfg = Config.fromfile(path)
    cfg.config_dir = path
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def logit_normal_sample_t(batch_size: int, mu: float, sigma: float, device, dtype) -> torch.Tensor:
    """论文式 (7): 时间采样 p(t; μ, σ=1.0) 为 logit-normal 分布。
    返回 t ∈ (0,1), shape [B]。"""
    u = torch.randn(batch_size, device=device, dtype=dtype) * sigma + mu
    return torch.sigmoid(u)


# ============================================================================ #
#  模型加载 (关键: 从 FLUX.1-Kontext-dev 加载基座)
# ============================================================================ #
def build_transformer(cfg, args) -> CustomFluxTransformer2DModel:
    """
    与 infer.py._prepare_custom_transformer 等价，但基座路径指向 FLUX.1-Kontext-dev。
    FLUX.1-Kontext-dev 的 transformer 架构 (in_channels=64, num_layers=19,
    num_single_layers=38, attention_head_dim=128) 与 FLUX.1-dev 完全一致，
    CustomFluxTransformer2DModel 可以无损承接权重，新模块 (layer_pe, refiner) 随机初始化。
    """
    base_path = cfg.pretrained_model_name_or_path
    logger.info(f"[Transformer] Loading base FLUX weights from: {base_path}")

    transformer_orig = FluxTransformer2DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        revision=cfg.get("revision", None),
        variant=cfg.get("variant", None),
        torch_dtype=torch.bfloat16,
        cache_dir=cfg.get("cache_dir", None),
    )

    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = cfg.get("max_layer_num", args.max_layer)
    mmdit_config = FrozenDict(mmdit_config)

    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    missing, unexpected = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    logger.info(
        f"[Transformer] Loaded base FLUX weights. "
        f"missing={len(missing)} (expected: layer_pe, refiner.*), unexpected={len(unexpected)}"
    )

    del transformer_orig
    return transformer


def build_vae(cfg) -> AutoencoderKL:
    """加载 FLUX VAE (冻结)。"""
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        revision=cfg.get("revision", None),
        variant=cfg.get("variant", None),
        torch_dtype=torch.bfloat16,
    )
    vae.requires_grad_(False).eval()
    return vae


def build_text_encoders(cfg, device):
    """加载 CLIP + T5 文本编码器 (冻结)。"""
    from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_2 = T5Tokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_1 = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).requires_grad_(False).eval()
    text_encoder_2 = T5EncoderModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    ).requires_grad_(False).eval()

    text_encoder_1.to(device)
    text_encoder_2.to(device)
    return (tokenizer_1, tokenizer_2), (text_encoder_1, text_encoder_2)


def build_transparent_decoder(cfg, device):
    """加载透明解码器 (xvae, 用于 alpha loss)。冻结。"""
    if not cfg.get("transp_vae_ckpt", None):
        return None
    if not os.path.exists(cfg.transp_vae_ckpt):
        logger.warning(f"transp_vae_ckpt not found: {cfg.transp_vae_ckpt}, alpha loss disabled.")
        return None
    transp_vae = CustomVAE()
    transp_vae.load_state_dict(
        torch.load(cfg.transp_vae_ckpt, map_location="cpu"), strict=False
    )
    transp_vae.to(device).eval()
    transp_vae.requires_grad_(False)
    return transp_vae


# ============================================================================ #
#  Text encoding (离线, no_grad)
# ============================================================================ #
@torch.no_grad()
def encode_prompt(tokenizers, text_encoders, prompt: str, max_seq_len: int = 512, device=None):
    """复刻 custom_pipeline.encode_prompt 但只编码单条 prompt。"""
    tok1, tok2 = tokenizers
    te1, te2 = text_encoders
    device = device or te1.device

    # CLIP pooled
    inputs1 = tok1([prompt], padding="max_length",
                   max_length=te1.config.max_position_embeddings,
                   truncation=True, return_tensors="pt")
    pooled = te1(inputs1.input_ids.to(device), output_hidden_states=False).pooler_output
    pooled = pooled.to(dtype=torch.bfloat16)

    # T5
    inputs2 = tok2([prompt], padding="max_length", max_length=max_seq_len,
                   truncation=True, return_tensors="pt")
    embeds = te2(inputs2.input_ids.to(device), output_hidden_states=False)[0]
    embeds = embeds.to(dtype=torch.bfloat16)

    text_ids = torch.zeros(embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
    return embeds, pooled, text_ids


# ============================================================================ #
#  VAE encoding (no_grad)
# ============================================================================ #
@torch.no_grad()
def vae_encode(vae: AutoencoderKL, image: torch.Tensor) -> torch.Tensor:
    """image: [B, 3, H, W] -1~1 → latent [B, 16, H/8, W/8] (已 shift+scale)。"""
    latent = vae.encode(image).latent_dist.sample()
    latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    return latent


@torch.no_grad()
def vae_encode_rgba(vae: AutoencoderKL, rgba: torch.Tensor) -> torch.Tensor:
    """rgba: [B, 4, h, w] 0~1 → 灰底合成后编码。

    8/13 修复: 原论文乘性灰底 Î_fg = (0.5·α + 0.5) × I_RGB 对【黑色背景】失效
    (物体外 I_RGB=0 → 0.5×0=0 仍是黑), 而 TranspVAE 预训练行为是:
       灰色(0.5) latent → alpha≈0(透明), 黑色(0) latent → alpha≈1(不透明)
    实测: 黑色 latent 解码 alpha=0.992, 灰色 latent 解码 alpha=0.015。
    因此改为【加性灰底】: Î_fg = α·I_RGB + (1-α)·0.5
    → 物体外(α=0) 恒为 0.5 灰, 与 TranspVAE 的"灰=透明"行为对齐, 消除黑框。
    alpha 单独保留作为 GT (下采样到 latent 分辨率)。
    返回 (latent [B,16,h/8,w/8], alpha [B,1,h/8,w/8] 0~1)。"""
    alpha = rgba[:, 3:4, :, :]                        # 0~1
    rgb_in = rgba[:, :3, :, :]                        # 0~1
    grey_based_rgb = alpha * rgb_in + (1.0 - alpha) * 0.5  # 加性灰底 0~1 (物体外=0.5灰)
    rgb = grey_based_rgb * 2.0 - 1.0                  # -1~1
    latent = vae_encode(vae, rgb)
    # downsample alpha to latent resolution
    alpha_lat = F.avg_pool2d(alpha, kernel_size=8, stride=8)
    return latent, alpha_lat


# ============================================================================ #
#  Latent 序列构造 (论文式 2-4)
# ============================================================================ #
def build_layer_latents(
    full_image: torch.Tensor,           # [3, H, W] -1~1
    background: torch.Tensor,           # [3, H, W] -1~1
    layer_rgba_list: List[torch.Tensor],# List[[4, h_i, w_i] 0~1]
    list_layer_box: List[List[float]],  # 像素坐标, 对齐 16
    vae: AutoencoderKL,
    device,
    dtype,
) -> Dict[str, torch.Tensor]:
    """
    返回:
        z_1         : [1, N, 16, H_lat, W_lat]   clean latents (layer0=full, 1=bg, 2..=fg)
        alpha_gt    : [1, N, 1, H_lat, W_lat]    各层 alpha (layer0/1 = 1.0)
        layer_valid : [N] bool                   该层是否参与 loss (layer0=条件→False)
        split_sizes: List[int]                   每层 patch token 数 (H_lat/2 * W_lat/2)
    """
    H, W = full_image.shape[-2:]
    H_lat, W_lat = H // 8, W // 8

    # ---- layer 0: full image (condition) ----
    full_lat = vae_encode(vae, full_image.unsqueeze(0).to(device, dtype))  # [1,16,h,w]
    # ---- layer 1: background ----
    bg_lat = vae_encode(vae, background.unsqueeze(0).to(device, dtype))    # [1,16,h,w]

    # ---- layer 2..N: foreground RGBA, 裁剪到 bbox ----
    fg_lats, fg_alphas = [], []
    fg_boxes = []
    for i, rgba in enumerate(layer_rgba_list):
        if i + 2 >= len(list_layer_box):
            break
        box = list_layer_box[i + 2]
        x1, y1, x2, y2 = [int(v) for v in box]
        rgba = rgba.to(device, dtype)
        # rgba 是裁剪后的 [4, h, w]
        lat, alpha = vae_encode_rgba(vae, rgba.unsqueeze(0))  # [1,16,h/8,w/8], [1,1,h/8,w/8]
        fg_lats.append(lat)
        fg_alphas.append(alpha)
        fg_boxes.append(box)

    # ---- 拼成 [1, N, 16, H_lat, W_lat], 前景层在 bbox 外填 0 (训练时被 mask) ----
    n_total = 2 + len(fg_lats)
    z_1 = torch.zeros(1, n_total, 16, H_lat, W_lat, device=device, dtype=dtype)
    alpha_gt = torch.ones(1, n_total, 1, H_lat, W_lat, device=device, dtype=dtype)

    z_1[:, 0] = full_lat
    z_1[:, 1] = bg_lat
    alpha_gt[:, 0] = 1.0
    alpha_gt[:, 1] = 1.0  # 背景层 alpha=1 (整层不透明)

    for i, (lat, alpha, box) in enumerate(zip(fg_lats, fg_alphas, fg_boxes)):
        x1, y1, x2, y2 = [int(v) // 8 for v in box]
        h_lat = y2 - y1
        w_lat = x2 - x1
        if h_lat <= 0 or w_lat <= 0:
            continue
        # 裁剪 latent 到 bbox 区域 (与推理一致: bbox 外保持 0)
        z_1[:, i + 2, :, y1:y2, x1:x2] = lat[:, :, :h_lat, :w_lat]
        alpha_gt[:, i + 2, :, y1:y2, x1:x2] = alpha[:, :, :h_lat, :w_lat]

    # ---- valid mask: layer0=条件不参与 loss, layer1=bg参与, layer2..=fg参与 ----
    layer_valid = torch.ones(n_total, dtype=torch.bool, device=device)
    layer_valid[0] = False  # full image 作为条件

    # ---- split_sizes: 必须与推理 custom_pipeline 约定一致 = [full_size, *fg_sizes] ----
    # 关键: 不含 bg 的 size。因为 bg box 是整图 [0,0,W,H], bg_size == full_size,
    # refiner 内部 hidden_states[:, l0_len:] 去掉 layer0 后,
    # split_sizes[0] (full_size) 复用为 bg 层的 size (见 custom_model_mmdit.py:403-414)。
    # 若误把 bg_size 也加进去, sum(split_sizes) 会比 hidden_states[:, l0_len:] 长, 报 split_with_sizes 错误。
    full_box = list_layer_box[0]
    fx1, fy1, fx2, fy2 = [int(v) for v in full_box]
    split_sizes = [((fy2 - fy1) // 16) * ((fx2 - fx1) // 16)]  # full_size (也作 l0_len)
    for box in list_layer_box[2:]:  # 跳过 full(0) 和 bg(1), 只取前景层
        x1, y1, x2, y2 = [int(v) for v in box]
        h_p = max(1, (y2 - y1) // 16)
        w_p = max(1, (x2 - x1) // 16)
        split_sizes.append(h_p * w_p)

    return {
        "z_1": z_1,
        "alpha_gt": alpha_gt,
        "layer_valid": layer_valid,
        "split_sizes": split_sizes,
    }


def build_adapter_data(
    full_image: torch.Tensor,
    list_layer_box: List[List[float]],
    vae: AutoencoderKL,
    device,
    dtype,
) -> torch.Tensor:
    """
    构造 Occlusion-Guided Adapter 的输入 (masked_image_latents)。
    复刻 custom_pipeline.CustomFluxPipelineCfg.__call__ 中的 adapter_data 构造逻辑:
      - 对每个 bg/fg 层, 把原图在 overlap 区域 mask 掉, 编码 VAE
      - 拼接 masked_latent + mask_latent → [B, num_bg_fgs, h_patch, w_patch, 64]
    """
    b = 1
    h, w = full_image.shape[-2:]
    n_bg_fgs = len(list_layer_box) - 1  # 去掉 layer0 (full image 条件)
    if n_bg_fgs <= 0:
        return None

    init_full = full_image.unsqueeze(0).unsqueeze(0).expand(-1, n_bg_fgs, -1, -1, -1)
    all_images = init_full.reshape(b * n_bg_fgs, 3, h, w).to(device, dtype)

    # overlap 计数
    overlap_counter = torch.zeros((b, h, w), device=device, dtype=dtype)
    for fg_idx in range(2, len(list_layer_box)):
        x1, y1, x2, y2 = [int(v) for v in list_layer_box[fg_idx]]
        overlap_counter[:, y1:y2, x1:x2] += 1
    is_fg_union = (overlap_counter > 0)
    is_overlap = (overlap_counter > 1)

    mask_pixel = torch.ones(b, n_bg_fgs, 1, h, w, device=device, dtype=dtype)
    mask_pixel[:, 0] = is_fg_union  # layer1 (bg): overlap = fg union
    for i, fg_idx in enumerate(range(2, len(list_layer_box)), start=1):
        x1, y1, x2, y2 = [int(v) for v in list_layer_box[fg_idx]]
        mask_pixel[:, i, :, y1:y2, x1:x2] = is_overlap[:, y1:y2, x1:x2]

    mask_flat = mask_pixel.view(b * n_bg_fgs, 1, h, w)
    masked_image = all_images * (1 - mask_flat)

    masked_latents = vae_encode(vae, masked_image)  # [B*n, 16, h/8, w/8]
    b_m, c_m, h_m, w_m = masked_latents.shape
    # patchify latent (2x2) → [B*n, h/16, w/16, 64]
    masked_latents = masked_latents.view(b_m, c_m, h_m // 2, 2, w_m // 2, 2)
    masked_latents = masked_latents.permute(0, 2, 4, 1, 3, 5)
    masked_latents = masked_latents.reshape(b_m, h_m // 2, w_m // 2, c_m * 4)

    # mask downsample to latent patch res
    h_lat = h // 8
    w_lat = w // 8
    mask_latents = mask_flat.view(b_m, 1, h_lat, 8, w_lat, 8).permute(0, 2, 4, 1, 3, 5)
    mask_latents = mask_latents.reshape(b_m, h_lat, w_lat, 64)
    mask_latents = mask_latents.permute(0, 3, 1, 2)
    mask_latents = F.max_pool2d(mask_latents, kernel_size=2, stride=2)
    mask_latents = mask_latents.permute(0, 2, 3, 1)

    masked_image_latents = torch.cat([masked_latents, mask_latents], dim=-1)  # [B*n, h_p, w_p, 64+64]
    _, h_p, w_p, c_tot = masked_image_latents.shape
    return masked_image_latents.view(b, n_bg_fgs, h_p, w_p, c_tot)


# ============================================================================ #
#  Latent image ids (3D-RoPE, 第一维=layer_id)
# ============================================================================ #
def prepare_latent_image_ids(height_lat: int, width_lat: int, list_layer_box, device, dtype):
    """与 custom_pipeline._prepare_latent_image_ids 一致: 第一维放 layer_idx。"""
    ids_list = []
    for layer_idx, box in enumerate(list_layer_box):
        if box is None:
            continue
        ids = torch.zeros(height_lat // 2, width_lat // 2, 3)
        ids[..., 0] = layer_idx
        ids[..., 1] = torch.arange(height_lat // 2)[:, None]
        ids[..., 2] = torch.arange(width_lat // 2)[None, :]
        x1, y1, x2, y2 = [int(v) // 16 for v in box]
        ids = ids[y1:y2, x1:x2, :]
        h_, w_, _ = ids.shape
        ids_list.append(ids.reshape(h_ * w_, 3))
    return torch.cat(ids_list, dim=0).to(device=device, dtype=dtype)


# ============================================================================ #
#  训练 step
# ============================================================================ #
def compute_orthogonality_loss(
    vae, z_pred_clean: torch.Tensor, list_layer_box, layer_valid,
    background: torch.Tensor, layer_rgba_list, latent_space: bool = False,
) -> torch.Tensor:
    """
    论文: Soft-Constraint Orthogonality Loss (式 15):
        L_orth = Σⱼ |⟨Î_RGB^bg, Î_RGB^fgj⟩_Rj - ⟨I_RGB^bg, I_RGB^fgj⟩_Rj|
    在像素空间计算背景层与前景层在 bbox 区域 Rj 内的余弦相似度,
    预测侧与 GT 侧的差值取绝对值后求和。

    说明:
      - z_pred_clean 为预测的 clean latent (z_t - t·v_θ, t=噪声权重), 保留梯度;
      - 预测 RGB 用主 VAE 一次 batch 解码所有层 (autocast 内);
      - GT 侧直接用训练数据的像素 RGB (背景整图 + 前景 RGBA 的 RGB 通道);
      - latent_space=True 时退化为旧实现 (latent 空间前景层两两正交, 仅调试/省显存)。
    """
    n_layers = z_pred_clean.shape[1]

    if latent_space:
        # ---- latent 近似实现: 前景层 latent 两两余弦相似度平方和 ----
        # 注意: 不同层的 bbox 大小不同, flatten 后长度不同无法 stack,
        #       因此取所有前景层 bbox 的"交集"作为公共区域再比较。
        #       交集为空 (层间不重叠) 时返回 0, 符合"不重叠无需正交惩罚"的语义。
        boxes = []
        for i in range(2, n_layers):
            if not layer_valid[i]:
                continue
            box = list_layer_box[i]
            x1, y1, x2, y2 = [int(v) // 8 for v in box]
            boxes.append((x1, y1, x2, y2))
        if len(boxes) < 2:
            return torch.tensor(0.0, device=z_pred_clean.device)
        # 公共交集 (latent 分辨率)
        ix1 = max(b[0] for b in boxes)
        iy1 = max(b[1] for b in boxes)
        ix2 = min(b[2] for b in boxes)
        iy2 = min(b[3] for b in boxes)
        if ix2 - ix1 < 1 or iy2 - iy1 < 1:
            return torch.tensor(0.0, device=z_pred_clean.device)
        feats = []
        for i in range(2, n_layers):
            if not layer_valid[i]:
                continue
            feat = z_pred_clean[0, i, :, iy1:iy2, ix1:ix2].flatten()
            feats.append(feat)
        if len(feats) < 2:
            return torch.tensor(0.0, device=z_pred_clean.device)
        F_stack = torch.stack(feats, dim=0).float()
        F_norm = F.normalize(F_stack, dim=-1)
        sim = F_norm @ F_norm.t()
        L = sim.shape[0]
        eye = torch.eye(L, device=sim.device, dtype=sim.dtype)
        off = sim - eye
        return (off ** 2).sum() / (L * (L - 1))

    # ---- 像素空间版 (论文式 15) ----
    if n_layers < 3:
        return torch.tensor(0.0, device=z_pred_clean.device)

    # 预测 latent → 像素 RGB [-1,1]。
    # 显存策略: 一次 batch decode M 层 1024² 会让 M 层中间激活同时驻留 → 80G 也 OOM
    # (此前被 except 吞 → orth 恒 0, 后端到 backward 直接崩)。这里:
    #   1) 逐层 decode, 峰值显存 ≈ 单层;
    #   2) 用 torch.utils.checkpoint 包裹, forward 不保存中间激活, backward 重算;
    #   3) 开启 VAE slicing (逐条 decode), 进一步压低单层峰值。
    z_un = (z_pred_clean[0] / vae.config.scaling_factor) + vae.config.shift_factor  # [N,16,h,w]

    # 开启 VAE slicing (对 decode 无副作用; 失败则忽略, 不影响正确性)
    if not getattr(vae, "_orth_slicing_enabled", False):
        try:
            vae.enable_slicing()
        except Exception:
            pass
        vae._orth_slicing_enabled = True

    def _decode_layer(z_chunk: torch.Tensor) -> torch.Tensor:
        # checkpoint 要求可调用对象; autocast 放内部, 反向重算同样走 bf16
        with autocast(enabled=True):
            return vae.decode(z_chunk, return_dict=False)[0]

    def _decode_one(lid: int):
        """解码第 lid 层 → 0~1 RGB [3,H,W]; 失败返回 None 并仅告警一次。"""
        try:
            d = torch.utils.checkpoint.checkpoint(
                _decode_layer, z_un[lid:lid + 1],
                use_reentrant=False, preserve_rng_state=False,
            )
            return ((d.float()[0] + 1.0) / 2.0).clamp(0.0, 1.0)
        except Exception as e:
            if not getattr(compute_orthogonality_loss, "_warned", False):
                compute_orthogonality_loss._warned = True
                logger.warning(
                    f"[orth_loss] vae.decode 失败, orth loss 恒为 0: {type(e).__name__}: {e}"
                )
            return None

    bg_pred = _decode_one(1)                       # layer 1 = 背景层
    if bg_pred is None:
        return torch.tensor(0.0, device=z_pred_clean.device)
    bg_gt = (background.float() + 1.0) / 2.0       # [3,H,W] 0~1

    def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        na = a.norm() + 1e-8
        nb = b.norm() + 1e-8
        return (a @ b) / (na * nb)

    loss = torch.tensor(0.0, device=z_pred_clean.device)
    cnt = 0
    for j, fg_idx in enumerate(range(2, n_layers)):
        if not layer_valid[fg_idx]:
            continue
        x1, y1, x2, y2 = [int(v) for v in list_layer_box[fg_idx]]
        if (x2 - x1) < 8 or (y2 - y1) < 8 or j >= len(layer_rgba_list):
            continue
        p_fg_layer = _decode_one(fg_idx)           # 逐层 decode, 用完即弃
        if p_fg_layer is None:
            continue
        p_bg = bg_pred[:, y1:y2, x1:x2]
        p_fg = p_fg_layer[:, y1:y2, x1:x2]
        g_bg = bg_gt[:, y1:y2, x1:x2]
        g_fg = layer_rgba_list[j][:3].float()      # 与 fg 层同序 (dataset 已同步 shuffle)
        if p_bg.numel() == 0 or p_fg.numel() == 0 or g_fg.numel() == 0:
            continue
        loss = loss + torch.abs(_cos_sim(p_bg, p_fg) - _cos_sim(g_bg, g_fg))
        cnt += 1
        del p_fg_layer
    return loss / max(cnt, 1)


def compute_alpha_loss(
    transp_vae, vae, z_pred: torch.Tensor, alpha_gt: torch.Tensor,
    list_layer_box, layer_valid,
) -> torch.Tensor:
    """
    论文: Hard-Constraint Alpha Loss (式 13-14, focal 形式):
        δᵢ = τ·|Î_α^i - I_α,gt^i|,   τ = 0.95
        L_α = -Σᵢ (δᵢ^γ · log(1 - δᵢ + ε)),   γ = 1.5
    用冻结的 TranspVAE 从预测 clean latent 解码 alpha (像素分辨率),
    GT alpha 由 latent 分辨率上采样回像素分辨率以对齐 (修复分辨率不匹配 bug)。
    """
    if transp_vae is None:
        return torch.tensor(0.0, device=z_pred.device)

    tau = 0.95
    gamma = 1.5
    eps = 1e-6

    n_layers = z_pred.shape[1]
    # 透明解码器输入: [N, 16, H_lat, W_lat] (unshift/scale 后)
    z_in = (z_pred[0] / vae.config.scaling_factor) + vae.config.shift_factor
    try:
        decoded_fg, decoded_alpha = transp_vae(z_in, [list_layer_box])
    except Exception:
        return torch.tensor(0.0, device=z_pred.device)
    decoded_alpha = ((decoded_alpha + 1.0) / 2.0).float().clamp(0.0, 1.0)  # -1~1 → 0~1, clamp 防解码越界

    # GT alpha 上采样回像素分辨率 (decoded_alpha 为像素分辨率, 二者对齐)
    # alpha_gt 是 5D [B, N, 1, H_lat, W_lat], bilinear 只支持 4D 输入, 先压平再还原
    b_n, n_n, c_n, h_n, w_n = alpha_gt.shape
    alpha_gt_px = F.interpolate(
        alpha_gt.float().reshape(b_n * n_n, c_n, h_n, w_n),
        scale_factor=8, mode="bilinear", align_corners=False,
    ).reshape(b_n, n_n, c_n, h_n * 8, w_n * 8)

    loss = torch.tensor(0.0, device=z_pred.device)
    cnt = 0
    for i in range(2, n_layers):
        if not layer_valid[i]:
            continue
        box = list_layer_box[i]
        x1, y1, x2, y2 = [int(v) for v in box]
        pred_a = decoded_alpha[i, :, y1:y2, x1:x2]          # 像素分辨率
        gt_a = alpha_gt_px[0, i, :, y1:y2, x1:x2]           # 像素分辨率
        if pred_a.numel() == 0 or pred_a.shape != gt_a.shape:
            continue
        delta = tau * torch.abs(pred_a - gt_a)
        # 论文式 14 的 log(1-δ+ε) 要求 δ < 1; 解码输出即使 clamp 后仍可能接近 1,
        # 这里再 clamp δ 保证 log 真数为正, 防止 nan
        delta = torch.clamp(delta, max=1.0 - eps)
        focal_term = -(delta ** gamma) * torch.log1p(-delta + eps)
        # 8/13 回退: 原物体外加权 (w_bg=3.0) 已移除。
        # 根因(黑框)已由"加性灰底合成"修复: 物体外目标=灰色(0.5) latent,
        # TranspVAE 天生对灰色解码透明(实测 alpha≈0.015), 无需额外惩罚。
        # 加权会导致 alpha 边缘过硬/物体边缘误杀, 且偏离论文 (λα=1.0 均匀监督)。
        loss = loss + focal_term.mean()
        cnt += 1
    return loss / max(cnt, 1)


# ============================================================================ #
#  主训练循环
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="RevealLayer multi-GPU training (FLUX.1-Kontext-dev)")
    parser.add_argument("--cfg_path", type=str, default="./configs/kontext_train_1024.py")
    parser.add_argument("--data_json", type=str, default=None,
                        help="覆盖 cfg.data_json")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="数据集根目录, 拼接相对路径")
    parser.add_argument("--max_layer", type=int, default=12)
    parser.add_argument("--transp_vae_ckpt", type=str, default=None,
                        help="透明解码器权重, 覆盖 cfg")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="checkpoint 目录或 'latest'")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="覆盖 cfg.max_train_steps (命令行优先级最高)")
    # ---- 小样本测试流程支持 ----
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="小样本测试: 从全量数据随机采样 N 条 (如 1000)。"
                             "-1 表示用全量。与 --subset_seed 配合可复现。")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="子集采样种子, 保证训练/验证用同一批样本")
    parser.add_argument("--subset_json", type=str, default=None,
                        help="若指定, 把采样的子集写入该 JSON (供 validate.py 复用)")
    parser.add_argument("--run_validate_after", action="store_true",
                        help="训练结束后自动调用 validate.py 验证 (仅 main process)")
    parser.add_argument("--validate_samples", type=int, default=50,
                        help="自动验证时推理的样本数 (从训练子集中取)")
    args = parser.parse_args()

    cfg = parse_config(args.cfg_path)
    # 命令行覆盖 (优先级最高)
    if args.data_json: cfg.data_json = args.data_json
    if args.root_dir: cfg.root_dir = args.root_dir
    if args.transp_vae_ckpt: cfg.transp_vae_ckpt = args.transp_vae_ckpt
    if args.max_layer: cfg.max_layer = args.max_layer
    if args.max_train_steps is not None:
        cfg.max_train_steps = args.max_train_steps

    # ---------------------- accelerate init ---------------------- #
    output_dir = cfg.get("output_dir", "./output/kontext_train")
    logging_dir = os.path.join(output_dir, cfg.get("logging_dir", "logs"))
    accel_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    # 默认关闭自动 tracker; 需要 wandb 时显式在 config 中设置 report_to="wandb"
    report_to = cfg.get("report_to", None)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        log_with=report_to,
        project_config=accel_config,
        step_scheduler_with_optimizer=False,
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    seed_everything(cfg.get("seed", 42) + accelerator.process_index)
    weight_dtype = torch.bfloat16

    # ---------------------- 数据 ---------------------- #
    # ---- 小样本测试: 生成可复用的子集 JSON ---- #
    actual_max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else cfg.get("max_samples", -1)
    subset_seed = args.subset_seed
    if actual_max_samples > 0 and args.subset_json:
        # 把采样子集落盘, validate.py 复用同一份
        from dataset import make_subset_json
        if accelerator.is_main_process:
            make_subset_json(cfg.data_json, args.subset_json,
                             n_samples=actual_max_samples, seed=subset_seed)
        accelerator.wait_for_everyone()
        # 用落盘的子集 JSON 构造 dataset, 保证多卡一致
        dataset = RevealLayerDataset(
            data_json=args.subset_json,
            root_dir=cfg.get("root_dir", ""),
            resolution=cfg.get("resolution", 1024),
            max_layers=cfg.get("max_layer_num", args.max_layer),
            shuffle_layers=cfg.get("shuffle_layers", True),
            max_samples=-1,  # 已是子集, 不再二次采样
        )
    else:
        dataset = RevealLayerDataset(
            data_json=cfg.data_json,
            root_dir=cfg.get("root_dir", ""),
            resolution=cfg.get("resolution", 1024),
            max_layers=cfg.get("max_layer_num", args.max_layer),
            shuffle_layers=cfg.get("shuffle_layers", True),
            max_samples=actual_max_samples,
            subset_seed=subset_seed,
        )
    if accelerator.is_main_process:
        logger.info(f"[Dataset] total samples = {len(dataset)}  "
                    f"(max_samples={actual_max_samples})")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("train_batch_size", 1),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---------------------- 模型 ---------------------- #
    device = accelerator.device
    transformer = build_transformer(cfg, args)
    vae = build_vae(cfg)
    tokenizers, text_encoders = build_text_encoders(cfg, device)
    transp_vae = build_transparent_decoder(cfg, device)

    # LoRA
    transformer = apply_lora(transformer, cfg)
    # 开启 gradient_checkpointing
    if cfg.get("gradient_checkpointing", True):
        transformer.gradient_checkpointing = True
        transformer.enable_gradient_checkpointing()
    # 允许 layer_pe / refiner 训练 (它们不是 LoRA, 是全参数)
    for name, p in transformer.named_parameters():
        if "layer_pe" in name or "refiner" in name:
            p.requires_grad = True
    # VAE / TE 冻结
    vae.to(device)
    transformer.to(device)

    # 在 accelerator.prepare 之前取出配置标志 (prepare 后被 DDP 包裹)
    guidance_embeds = bool(transformer.config.guidance_embeds)

    # ---------------------- 优化器 ---------------------- #
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    opt_name = cfg.get("optimizer", "prodigy").lower()
    raw_lr = cfg.get("learning_rate", 1.0)
    if opt_name == "prodigy":
        if Prodigy is None:
            logger.warning("prodigyopt not installed, fallback to AdamW.")
            opt_name = "adamw"
        else:
            optimizer = Prodigy(
                trainable_params,
                lr=raw_lr,
                betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
                weight_decay=cfg.get("adam_weight_decay", 1e-3),
                decouple=cfg.get("prodigy_decouple", True),
                use_bias_correction=cfg.get("prodigy_use_bias_correction", True),
                safeguard_warmup=cfg.get("prodigy_safeguard_warmup", True),
            )
    if opt_name in ("adamw", "adamw_8bit"):
        # AdamW 的学习率通常比 Prodigy 低 1e4 量级; 若用户仍配了 1.0 则自动降级防止发散
        lr = cfg.get("adamw_learning_rate", None) or raw_lr
        if lr >= 1.0:
            logger.warning(
                f"[Optimizer] AdamW lr={lr} 过大, 自动降级到 1e-4。"
                f"如需使用 Prodigy 的 lr=1.0, 请安装 prodigyopt (pip install prodigyopt)。"
            )
            lr = 1e-4
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
            weight_decay=cfg.get("adam_weight_decay", 1e-3),
            eps=cfg.get("adam_epsilon", 1e-8),
        )
    elif opt_name == "adam":
        lr = cfg.get("adamw_learning_rate", None) or raw_lr
        if lr >= 1.0:
            logger.warning(f"[Optimizer] Adam lr={lr} 过大, 自动降级到 1e-4")
            lr = 1e-4
        optimizer = torch.optim.Adam(
            trainable_params, lr=lr,
            weight_decay=cfg.get("adam_weight_decay", 1e-3),
        )
    if accelerator.is_main_process:
        logger.info(f"[Optimizer] {opt_name}, lr={optimizer.param_groups[0]['lr']}, "
                    f"trainable_params={sum(p.numel() for p in trainable_params):,}")

    # ---------------------- scheduler ---------------------- #
    from transformers import get_constant_schedule_with_warmup
    n_warmup = cfg.get("lr_warmup_steps", 0)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, n_warmup)

    # ---------------------- accelerator prepare ---------------------- #
    transformer, optimizer, loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, loader, lr_scheduler
    )
    # VAE / TE 保持裸 module (不参与 DDP)
    vae.to(device, dtype=weight_dtype)
    text_encoders[0].to(device, dtype=weight_dtype)
    text_encoders[1].to(device, dtype=weight_dtype)
    if transp_vae is not None:
        transp_vae.to(device, dtype=weight_dtype)

    # 预编码固定 prompt
    prompt = cfg.get("prompt", RevealLayerDataset.PROMPT)
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        tokenizers, text_encoders, prompt,
        max_seq_len=cfg.get("max_sequence_length", 512),
        device=device,
    )

    # ---------------------- 训练状态 ---------------------- #
    max_train_steps = cfg.get("max_train_steps", None)
    num_epochs = cfg.get("num_train_epochs", 1)
    checkpointing_steps = cfg.get("checkpointing_steps", 2000)
    checkpoints_total_limit = cfg.get("checkpoints_total_limit", None)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    layer_weighting = cfg.get("layer_weighting", 5.0)
    alpha_loss_weight = cfg.get("alpha_loss_weight", 0.0)
    orth_loss_weight = cfg.get("orth_loss_weight", 0.0)
    logit_mean = cfg.get("logit_mean", 0.0)
    logit_std = cfg.get("logit_std", 1.0)

    # ---------------------- loss CSV 日志 (main process, 不受终端/tqdm 影响) ---------------------- #
    _loss_csv = None
    if accelerator.is_main_process:
        try:
            loss_csv_path = os.path.join(output_dir, "loss_log.csv")
            _loss_csv = open(loss_csv_path, "w", newline="")
            _loss_csv.write("step,loss,loss_fm,loss_alpha,loss_orth,lr,t\n")
            _loss_csv.flush()
        except Exception as e:
            logger.warning(f"loss csv init failed: {e}")
            _loss_csv = None

    # VAE 编码结果缓存: 同一张图 (按 imgid+层顺序) 每步重复编码是最大性能瓶颈,
    # 编码结果只依赖数据不依赖 t, 按 key 缓存后每张图只编码一次。
    latent_cache: dict = {}
    latent_cache_max = cfg.get("latent_cache_max", 2048)

    global_step = 0
    start_epoch = 0
    resume_path = args.resume_from_checkpoint or cfg.get("resume_from_checkpoint", None)

    # ---------------------- resume ---------------------- #
    if resume_path:
        if resume_path == "latest":
            dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                          key=lambda x: int(x.split("-")[-1]))
            if dirs:
                resume_path = os.path.join(output_dir, dirs[-1])
            else:
                resume_path = None
                logger.info("No checkpoint found, training from scratch.")
        if resume_path and os.path.isdir(resume_path):
            logger.info(f"Resuming from {resume_path}")
            accelerator.load_state(resume_path)
            global_step = int(resume_path.split("-")[-1])
            start_epoch = global_step // max(len(loader), 1)

    # ---------------------- tracker (wandb/tensorboard) ---------------------- #
    # 注意: Accelerator 没有 is_initialized() 方法; 用 len(trackers)==0 防重复初始化
    report_to = cfg.get("report_to", None)
    if accelerator.is_main_process and len(accelerator.trackers) == 0 and report_to:
        try:
            if report_to == "wandb":
                accelerator.init_trackers(
                    cfg.get("tracker_project_name", "reveallayer"),
                    config=dict(cfg),
                    init_kwargs={"wandb": {"name": cfg.get("wandb_job_name", "kontext_train")}},
                )
            elif report_to == "tensorboard":
                accelerator.init_trackers(
                    cfg.get("tracker_project_name", "reveallayer"),
                    config=dict(cfg),
                )
        except Exception as e:
            logger.warning(f"{report_to} init failed: {e} (训练继续, 仅无 tracker 日志)")

    # ---------------------- 训练循环 ---------------------- #
    total_steps = max_train_steps or (num_epochs * len(loader))
    if accelerator.is_main_process:
        logger.info(f"[Train] max_train_steps={max_train_steps}, "
                    f"num_epochs={num_epochs}, dataset_len={len(dataset)}, "
                    f"loader_len={len(loader)}, total_steps={total_steps}, "
                    f"checkpointing_steps={checkpointing_steps}")
    progress = tqdm(range(total_steps), disable=not accelerator.is_main_process,
                    desc="Training")
    transformer.train()

    epoch = start_epoch
    while global_step < total_steps:
        # 分布式 sampler 需要 set_epoch 以保证每个 epoch shuffle 不同
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            with accelerator.accumulate(transformer):
                # ---------- 数据准备 ----------
                full_image = batch["full_image"].to(device, dtype=weight_dtype)
                background = batch["background"].to(device, dtype=weight_dtype)
                layer_rgba_list = [
                    t.to(device, dtype=weight_dtype) for t in batch["layer_rgba_list"]
                ]
                list_layer_box = batch["list_layer_box"]
                n_layers = batch["n_layers"]
                H, W = full_image.shape[-2:]
                H_lat, W_lat = H // 8, W // 8

                # ---------- VAE 编码 (no_grad) + 缓存 (同一图+层顺序只编码一次) ----------
                # 注意: dataset 的 shuffle_layers 会打乱前景层顺序, 缓存 key 必须含层顺序
                imgid = batch["imgid"]
                cache_key = (imgid, tuple(tuple(b) for b in list_layer_box))
                cached = latent_cache.get(cache_key)
                if cached is None:
                    with torch.no_grad():
                        data = build_layer_latents(
                            full_image, background, layer_rgba_list,
                            list_layer_box, vae, device, weight_dtype,
                        )
                        adapter_data = build_adapter_data(
                            full_image, list_layer_box, vae, device, weight_dtype,
                        )
                        latent_image_ids = prepare_latent_image_ids(
                            H_lat, W_lat, list_layer_box, device, weight_dtype,
                        )
                    if len(latent_cache) >= latent_cache_max:
                        latent_cache.pop(next(iter(latent_cache)))
                    latent_cache[cache_key] = (data, adapter_data, latent_image_ids)
                else:
                    data, adapter_data, latent_image_ids = cached

                z_1 = data["z_1"]                    # [1, N, 16, h, w] clean
                alpha_gt = data["alpha_gt"]           # [1, N, 1, h, w]
                layer_valid = data["layer_valid"]     # [N] bool
                split_sizes = data["split_sizes"]

                # ---------- Rectified Flow 采样 ----------
                bsz = 1
                t = logit_normal_sample_t(bsz, logit_mean, logit_std, device, weight_dtype)
                # timestep 传入 [0,1] 范围: transformer 内部会 *1000 (custom_model_mmdit.py:357)
                # 与推理 timestep=timestep/1000 一致。若这里再乘 1000 会导致内部变成 [0,1e6] 数值错误。
                timestep = t
                noise = torch.randn_like(z_1)
                # FLUX 约定: t 是"噪声权重" (t=1 → 纯噪声, t=0 → clean), 与推理 scheduler 的 sigmas 一致。
                # 之前误把 t 当"clean 权重", 导致 v_target = z_1 - noise 与推理期望的
                # v = noise - z_1 符号相反 → 推理时 scheduler 每步往噪声方向积分 → 全雪花。
                z_t = t.view(1, 1, 1, 1, 1) * noise + (1.0 - t.view(1, 1, 1, 1, 1)) * z_1
                z_t[:, 0] = z_1[:, 0]      # layer0 条件层始终干净 (与推理 latents[:, :1]=full_image_latent 一致)
                v_target = noise - z_1      # = d z_t/dt (t: 噪声权重)
                v_target[:, 0] = 0.0        # 条件层不计算 loss

                # ---------- transformer forward ----------
                # guidance (FLUX.1-Kontext-dev 通常 guidance_embeds=True)
                guidance = None
                if guidance_embeds:
                    guidance = torch.full([bsz], cfg.get("guidance_scale", 1.0),
                                          device=device, dtype=weight_dtype)

                model_out = transformer(
                    hidden_states=z_t,
                    adapter_data=adapter_data,
                    split_sizes=split_sizes,
                    list_layer_box=list_layer_box,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    guidance=guidance,
                    return_dict=False,
                )[0]  # [1, N, 16, h, w]

                # ---------- Loss 计算 (fp32, 在 autocast 外) ----------
                # 某些 PyTorch 版本 MSE/L1 的 backward 不支持 bf16 输入,
                # 因此将参与 loss 计算的所有张量显式转 fp32。
                v_pred_f = model_out.float()
                v_target_f = v_target.float()

                loss_fm = torch.tensor(0.0, device=device)          # fp32
                loss_cnt = 0
                for i in range(n_layers):
                    if not layer_valid[i]:
                        continue
                    box = list_layer_box[i]
                    x1, y1, x2, y2 = [int(v) // 8 for v in box]
                    pred = v_pred_f[0, i, :, y1:y2, x1:x2]
                    tgt = v_target_f[0, i, :, y1:y2, x1:x2]
                    if pred.numel() == 0:
                        continue
                    w_i = layer_weighting if i >= 2 else 1.0
                    loss_fm = loss_fm + w_i * F.mse_loss(pred, tgt)
                    loss_cnt += 1
                loss_fm = loss_fm / max(loss_cnt, 1)

                loss_aux = torch.tensor(0.0, device=device)         # fp32
                z_pred_clean = None
                if orth_loss_weight > 0 or (alpha_loss_weight > 0 and transp_vae is not None):
                    # 预测 clean latent: z_t = t·noise + (1-t)·z_1, v = noise - z_1
                    # → z_1 = z_t - t·v_θ (注意与旧公式 z_t + (1-t)·v 符号相反)
                    # 必须保留 grad, 使辅助损失的梯度流回 transformer。
                    z_pred_clean = z_t - t.view(1, 1, 1, 1, 1) * model_out
                if orth_loss_weight > 0:
                    # 论文式 15: 像素空间余弦相似度 (或 latent 近似, 见配置 orth_loss_latent_space)
                    loss_orth = compute_orthogonality_loss(
                        vae, z_pred_clean.float(), list_layer_box, layer_valid,
                        background, layer_rgba_list,
                        latent_space=cfg.get("orth_loss_latent_space", False),
                    )
                    loss_aux = loss_aux + orth_loss_weight * loss_orth
                if alpha_loss_weight > 0 and transp_vae is not None:
                    # 论文式 13-14: TranspVAE 解码 alpha 的 focal loss (autocast 内, 需梯度)
                    with autocast(enabled=True):
                        loss_alpha = compute_alpha_loss(
                            transp_vae, vae, z_pred_clean, alpha_gt,
                            list_layer_box, layer_valid,
                        )
                    loss_aux = loss_aux + alpha_loss_weight * loss_alpha

                loss = (loss_fm + loss_aux).float()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0:
                        accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # ---------- logging ----------
                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(1)
                    if accelerator.is_main_process and global_step % 20 == 0:
                        loss_val = loss.detach().item()
                        if not math.isfinite(loss_val):
                            _parts = f"fm={loss_fm.detach().item():.4e}"
                            if orth_loss_weight > 0:
                                _parts += f", orth={loss_orth.detach().item():.4e}"
                            if alpha_loss_weight > 0 and transp_vae is not None:
                                _parts += f", alpha={loss_alpha.detach().item():.4e}"
                            logger.error(
                                f"[Train] step={global_step} loss={loss_val} 非有限值, 训练发散。"
                                f"分项: {_parts}。请检查学习率/优化器/数据。"
                            )
                            raise RuntimeError(f"Loss diverged at step {global_step}: {loss_val}")
                        log_dict = {
                            "loss": loss_val,
                            "loss_fm": loss_fm.detach().item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "t": t.detach().item(),
                            "step": global_step,
                        }
                        if alpha_loss_weight > 0 and transp_vae is not None:
                            log_dict["loss_alpha"] = loss_alpha.detach().item()
                        if orth_loss_weight > 0:
                            log_dict["loss_orth"] = loss_orth.detach().item()
                        try:
                            accelerator.log(log_dict, step=global_step)
                        except Exception:
                            pass
                        logger.info(
                            f"step={global_step} loss={log_dict['loss']:.4f} "
                            f"fm={log_dict['loss_fm']:.4f} lr={log_dict['lr']:.4e}"
                        )
                        # 写入 loss CSV (main process), 方便 tail 查看趋势
                        if _loss_csv is not None:
                            try:
                                _row = [str(global_step), f"{loss_val:.6f}", f"{log_dict['loss_fm']:.6f}"]
                                _row.append(f"{loss_alpha.detach().item():.6f}" if "loss_alpha" in log_dict else "")
                                _row.append(f"{loss_orth.detach().item():.6f}" if "loss_orth" in log_dict else "")
                                _row.append(f"{log_dict['lr']:.6e}")
                                _row.append(f"{log_dict['t']:.6f}")
                                _loss_csv.write(",".join(_row) + "\n")
                                _loss_csv.flush()
                            except Exception as _e:
                                logger.warning(f"[loss_csv] write failed: {_e}")

                    # ---------- checkpoint ----------
                    if checkpointing_steps and global_step % checkpointing_steps == 0:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # 额外保存 LoRA + layer_pe + refiner (便于直接推理加载)
                        if accelerator.is_main_process:
                            unwrapped = accelerator.unwrap_model(transformer)
                            lora_sd = get_peft_model_state_dict(unwrapped)
                            lora_sd_corr = correct_lora_keys_for_inference(lora_sd)
                            try:
                                from safetensors.torch import save_file as safesave
                                safesave(
                                    lora_sd_corr,
                                    os.path.join(save_path, "pytorch_lora_weights.safetensors"),
                                )
                            except Exception as e:
                                logger.warning(f"[Checkpoint] safetensors save failed: {e}, fallback to pt")
                                torch.save(
                                    lora_sd_corr,
                                    os.path.join(save_path, "pytorch_lora_weights.pt"),
                                )
                            # 保存 layer_pe / refiner (与 infer.py 期望的文件名一致)
                            extra_sd = {}
                            for k, v in unwrapped.state_dict().items():
                                if "layer_pe" in k or "refiner" in k:
                                    extra_sd[k] = v.cpu()
                            torch.save(
                                extra_sd["layer_pe"] if "layer_pe" in extra_sd else extra_sd,
                                os.path.join(save_path, "layer_pe.pt"),
                            )
                            refiner_sd = {
                                k.replace("refiner.", ""): v
                                for k, v in extra_sd.items() if k.startswith("refiner.")
                            }
                            if refiner_sd:
                                torch.save(refiner_sd, os.path.join(save_path, "Refiner.pt"))
                            # 清理旧 checkpoint
                            if checkpoints_total_limit:
                                dirs = sorted(
                                    [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                                    key=lambda x: int(x.split("-")[-1]),
                                )
                                while len(dirs) > checkpoints_total_limit:
                                    import shutil
                                    shutil.rmtree(os.path.join(output_dir, dirs[0]))
                                    dirs = dirs[1:]

                    if max_train_steps and global_step >= max_train_steps:
                        break

            if max_train_steps and global_step >= max_train_steps:
                break

        epoch += 1

    # ---------------------- 最终保存 ---------------------- #
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        # LoRA
        lora_sd = get_peft_model_state_dict(unwrapped)
        lora_sd_corrected = correct_lora_keys_for_inference(lora_sd)
        try:
            from safetensors.torch import save_file as safesave
            safesave(lora_sd_corrected, os.path.join(final_dir, "pytorch_lora_weights.safetensors"))
        except Exception as e:
            logger.warning(f"safetensors save failed: {e}, fallback to torch.save")
            torch.save(lora_sd_corrected, os.path.join(final_dir, "pytorch_lora_weights.pt"))
        # layer_pe / refiner
        extra_sd = {}
        for k, v in unwrapped.state_dict().items():
            if "layer_pe" in k or "refiner" in k:
                extra_sd[k] = v.cpu()
        torch.save(extra_sd["layer_pe"] if "layer_pe" in extra_sd else extra_sd,
                   os.path.join(final_dir, "layer_pe.pt"))
        refiner_sd = {k.replace("refiner.", ""): v for k, v in extra_sd.items() if k.startswith("refiner.")}
        if refiner_sd:
            torch.save(refiner_sd, os.path.join(final_dir, "Refiner.pt"))
        logger.info(f"Final checkpoint saved to {final_dir}")

    try:
        accelerator.end_training()
    except Exception as e:
        logger.warning(f"accelerator.end_training() warning: {e} (通常与 wandb tracker 有关, 可忽略)")

    # 关闭 loss CSV 句柄
    if _loss_csv is not None:
        try:
            _loss_csv.close()
        except Exception:
            pass

    # ---------------------- 训练后自动验证 (可选) ---------------------- #
    if args.run_validate_after and accelerator.is_main_process:
        import subprocess
        val_json = args.subset_json or cfg.data_json
        val_ckpt = os.path.join(output_dir, "final")
        val_cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "validate.py"),
            "--input_json", val_json,
            "--root_dir", cfg.get("root_dir", ""),
            "--ckpt_dir", val_ckpt,
            "--pretrained_model_name_or_path", cfg.pretrained_model_name_or_path,
            "--cfg_path", args.cfg_path,
            "--max_samples", str(args.validate_samples),
            "--output_dir", os.path.join(output_dir, "validate_after_train"),
        ]
        if cfg.get("transp_vae_ckpt", None):
            val_cmd += ["--transp_vae_ckpt", cfg.transp_vae_ckpt]
        logger.info(f"[Auto-validate] running: {' '.join(val_cmd)}")
        try:
            subprocess.run(val_cmd, check=False, cwd=PROJECT_ROOT)
        except Exception as e:
            logger.warning(f"[Auto-validate] failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 异常退出时也清理 NCCL process group, 避免 PyTorch 2.4+ 的警告
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
