"""
RevealLayer-100K 数据集加载器
============================
按论文 "RevealLayer: Disentangling Hidden and Visible Layers via Occlusion-Aware
Image Decomposition" (arXiv:2605.11818) 第 3 节实现：

  - 输入图像 I、背景 I_bg、前景层 {I_fg^i} 都过 VAE 编码器 E_VAE 提取 latent
  - 按 bbox 裁剪并展平为变长 token 序列
  - 拼接成统一序列  z_0 = [z_0^c ; z_0^0 ; z_0^1 ; ... ; z_0^N]   (论文式 2-4)
  - 3D-RoPE 用第一维承载 layer_id (论文式 3D-RoPE 段落)

数据格式 (与 README 中 JSON 示例一致)：
    [{
        "imgid": "xxx",
        "full_image": "path/to/full.png",
        "background": "path/to/bg.png",          # 可选
        "LayerInfoRaw": ["path/to/L0.png", ...], # RGBA 前景层
        "detections": [{"bbox": [x1,y1,x2,y2]}, ...]
    }, ...]

返回的字段会交给训练循环做 VAE 编码 (VAE 在主进程共享、no_grad)。
"""

import os
import json
import math
import random
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ----------------------------------------------------------------------------- #
#  几何工具 (与 infer.py 保持一致，保证训练/推理对齐)
# ----------------------------------------------------------------------------- #
def adjust_coordinate(value, floor_or_ceil, k=16, min_val=0, max_val=1024):
    if floor_or_ceil == "floor":
        rounded = math.floor(value / k) * k
    else:
        rounded = math.ceil(value / k) * k
    return max(min_val, min(rounded, max_val))


def resize_with_bbox_edge(image: Image.Image, resolution=1024, k=16):
    """按长边缩放到 resolution，再 crop 到 k 的倍数。返回 (content_hw, target_hw, offset, None)。"""
    orig_w, orig_h = image.size
    long_side = max(orig_w, orig_h)
    ideal_scale = resolution / long_side

    new_content_w = int(round(orig_w * ideal_scale))
    new_content_h = int(round(orig_h * ideal_scale))

    target_w = (new_content_w // k) * k
    target_h = (new_content_h // k) * k
    target_w = max(k, target_w)
    target_h = max(k, target_h)

    offset_x = (new_content_w - target_w) // 2
    offset_y = (new_content_h - target_h) // 2
    return (new_content_h, new_content_w), (target_h, target_w), (offset_y, offset_x)


def consistent_crop(image: Image.Image, content_w, content_h, target_w, target_h, off_x, off_y):
    resized = image.resize((content_w, content_h), Image.Resampling.LANCZOS)
    return resized.crop((off_x, off_y, off_x + target_w, off_y + target_h))


def scale_bbox(bbox, orig_w, orig_h, cont_w, cont_h, off_x, off_y, tgt_w, tgt_h):
    """把原图坐标 bbox 转到裁剪后图像坐标。"""
    sx1 = bbox[0] * cont_w / orig_w
    sy1 = bbox[1] * cont_h / orig_h
    sx2 = bbox[2] * cont_w / orig_w
    sy2 = bbox[3] * cont_h / orig_h
    cx1 = max(0, min(sx1 - off_x, tgt_w))
    cy1 = max(0, min(sy1 - off_y, tgt_h))
    cx2 = max(0, min(sx2 - off_x, tgt_w))
    cy2 = max(0, min(sy2 - off_y, tgt_h))
    return [cx1, cy1, cx2, cy2]


def align_bbox_to_k(bbox, tgt_w, tgt_h, k=16):
    """把 bbox 对齐到 k 的倍数 (与推理 filter_and_align_bboxes 一致)。"""
    x1 = adjust_coordinate(bbox[0], "floor", k=k, max_val=tgt_w)
    y1 = adjust_coordinate(bbox[1], "floor", k=k, max_val=tgt_h)
    x2 = adjust_coordinate(bbox[2], "ceil", k=k, max_val=tgt_w)
    y2 = adjust_coordinate(bbox[3], "ceil", k=k, max_val=tgt_h)
    if x2 - x1 < k:
        x2 = min(tgt_w, x1 + k)
    if y2 - y1 < k:
        y2 = min(tgt_h, y1 + k)
    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------------- #
#  子集采样工具 (保证训练集与验证集用同一批样本)
# ----------------------------------------------------------------------------- #
def make_subset_json(
    src_json: str,
    dst_json: str,
    n_samples: int = 1000,
    seed: int = 42,
):
    """从全量标注 src_json 随机采样 n_samples 条, 写入 dst_json。
    用固定 seed 保证可复现: 训练 / 验证 / 调试共用同一份子集。"""
    with open(src_json, "r") as f:
        entries = json.load(f)
    rng = random.Random(seed)
    if n_samples >= len(entries):
        subset = list(entries)
        rng.shuffle(subset)
    else:
        subset = rng.sample(entries, n_samples)
    os.makedirs(os.path.dirname(os.path.abspath(dst_json)), exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    print(f"[subset] {len(subset)} samples (seed={seed}) -> {dst_json}")
    return dst_json


# ----------------------------------------------------------------------------- #
#  Dataset
# ----------------------------------------------------------------------------- #
class RevealLayerDataset(Dataset):
    """
    训练数据集。每个样本返回：

        full_image      : [3, H, W]  float32 (-1~1)   整图 (作为条件)
        background      : [3, H, W]  float32 (-1~1)   背景图 (layer 1 目标)
        layer_rgba_list: List[[4, h_i, w_i]]           各前景层 RGBA (裁剪到 bbox)
        list_layer_box : [[0,0,W,H], [0,0,W,H], bbox1, bbox2, ...]  像素坐标, 已对齐 16
        n_layers       : int                            = 2 + len(fgs)
        imgid          : str
        prompt         : str                            固定提示词

    注意：layer 0 是 full image (条件), layer 1 是 background, layer 2..N 是前景层。
    这与 infer.py 的 validation_boxes_processed 约定一致。

    小样本测试: 传 max_samples + subset_seed 即可从全量随机采样 N 条,
    与 make_subset_json 共用同一 seed 可保证训练/验证集一致。
    """

    PROMPT = "Decompose the image into foreground and background."

    def __init__(
        self,
        data_json: str,
        root_dir: str = "",
        resolution: int = 1024,
        max_layers: int = 12,
        shuffle_layers: bool = True,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        max_samples: int = -1,
        subset_seed: int = 42,
    ):
        super().__init__()
        if not os.path.exists(data_json):
            raise FileNotFoundError(
                f"数据集 JSON 不存在: {data_json}\n"
                f"请先下载 RevealLayer-100K:\n"
                f"  bash download_data.sh\n"
                f"或从 https://huggingface.co/datasets/qihoo360/RevealLayer-100K 获取。"
            )
        with open(data_json, "r") as f:
            entries = json.load(f)

        # 小样本测试: 随机采样
        if max_samples is not None and max_samples > 0 and max_samples < len(entries):
            rng = random.Random(subset_seed)
            entries = rng.sample(entries, max_samples)
            print(f"[Dataset] 使用子集: {max_samples}/{len(entries) if False else 'full'} 样本 "
                  f"(seed={subset_seed})")

        self.entries = entries
        self.root_dir = root_dir
        self.resolution = resolution
        self.max_layers = max_layers
        self.shuffle_layers = shuffle_layers

        self.rgb_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ])
        self.rgba_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGBA")),
            transforms.ToTensor(),  # [0,1], 4 channels
        ])

    def __len__(self):
        return len(self.entries)

    def _resolve(self, p: str) -> str:
        if not p:
            return ""
        if os.path.isabs(p) or not self.root_dir:
            return p
        return os.path.join(self.root_dir, p)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        imgid = entry.get("imgid", f"{idx:06d}")

        full_image_path = self._resolve(entry.get("full_image", ""))
        if not full_image_path or not os.path.exists(full_image_path):
            # 跳过缺失样本：返回下一个 (循环)
            return self.__getitem__((idx + 1) % len(self.entries))

        try:
            full_image = Image.open(full_image_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.entries))

        orig_w, orig_h = full_image.size
        (cont_h, cont_w), (tgt_h, tgt_w), (off_y, off_x) = resize_with_bbox_edge(
            full_image, resolution=self.resolution
        )

        # 缩放后的整图
        full_resized = consistent_crop(full_image, cont_w, cont_h, tgt_w, tgt_h, off_x, off_y)
        full_tensor = self.rgb_transform(full_resized)  # [3, H, W] -1~1

        # 背景
        bg_path = self._resolve(entry.get("background", ""))
        if bg_path and os.path.exists(bg_path):
            try:
                bg_image = Image.open(bg_path).convert("RGB").resize(
                    full_resized.size, Image.Resampling.LANCZOS
                )
                bg_tensor = self.rgb_transform(bg_image)
            except Exception:
                bg_tensor = full_tensor.clone()
        else:
            # 没有背景 GT：用整图占位 (训练时该层 loss 会被 mask 掉, 见 train.py)
            bg_tensor = full_tensor.clone()

        # 前景层 + bbox
        layer_paths = entry.get("LayerInfoRaw", []) or []
        detections = entry.get("detections", []) or []
        n_det = min(len(detections), len(layer_paths))

        # 把 bbox 缩放到裁剪坐标系
        scaled_boxes = []
        for d in detections:
            bb = d.get("bbox", None)
            if bb is None or len(bb) != 4:
                continue
            sb = scale_bbox(bb, orig_w, orig_h, cont_w, cont_h, off_x, off_y, tgt_w, tgt_h)
            # 过滤过小的框 (与 infer.filter_and_align_bboxes 一致: 可见区域 >= 50%)
            x1, y1, x2, y2 = sb
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            scaled_boxes.append(align_bbox_to_k(sb, tgt_w, tgt_h, k=16))

        # 读取前景 RGBA 并裁剪到对应 bbox
        layer_rgba_list = []
        used_boxes = []
        for i in range(min(n_det, self.max_layers - 2)):
            if i >= len(scaled_boxes):
                break
            p = self._resolve(layer_paths[i]) if i < len(layer_paths) else ""
            if not p or not os.path.exists(p):
                continue
            try:
                layer_img = Image.open(p).convert("RGBA")
            except Exception:
                continue
            x1, y1, x2, y2 = scaled_boxes[i]
            # 先把 layer 缩到整图尺寸 (LayerInfoRaw 通常已经是整图大小, 含 alpha)
            if layer_img.size != (tgt_w, tgt_h):
                layer_img = layer_img.resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)
            # 裁剪到 bbox
            layer_crop = layer_img.crop((int(x1), int(y1), int(x2), int(y2)))
            layer_tensor = self.rgba_transform(layer_crop)  # [4, h, w] 0~1
            layer_rgba_list.append(layer_tensor)
            used_boxes.append([x1, y1, x2, y2])

        # shuffle 前景层顺序 (论文 Region-Aware Attention 不依赖顺序)
        if self.shuffle_layers and len(layer_rgba_list) > 1:
            order = list(range(len(layer_rgba_list)))
            random.shuffle(order)
            layer_rgba_list = [layer_rgba_list[i] for i in order]
            used_boxes = [used_boxes[i] for i in order]

        # list_layer_box: [full_box, bg_box, *fg_boxes]
        full_box = [0, 0, tgt_w, tgt_h]
        list_layer_box = [full_box, full_box] + used_boxes

        return {
            "imgid": imgid,
            "full_image": full_tensor,                    # [3, H, W] -1~1
            "background": bg_tensor,                       # [3, H, W] -1~1
            "layer_rgba_list": layer_rgba_list,            # List[[4, h, w] 0~1]
            "list_layer_box": list_layer_box,              # List[[x1,y1,x2,y2]] 像素, 已对齐16
            "n_layers": len(list_layer_box),
            "prompt": self.PROMPT,
            "tgt_h": tgt_h,
            "tgt_w": tgt_w,
        }


def collate_fn(batch):
    """
    简单 collate: 训练用 batch_size=1 (per GPU) + gradient accumulation,
    与论文配置 train_batch_size=1 一致。这里直接取 batch[0]。
    """
    return batch[0]
