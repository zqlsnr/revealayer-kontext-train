"""
RevealLayer 训练/验证子集准备脚本
==================================

从全量数据集中随机采样 N 个训练样本, 把涉及的所有图片文件拷贝到指定目录,
并生成指向新位置的训练 JSON 和验证 JSON (验证集从训练集中再随机采样 M 个)。

用途 (两阶段流程 stage1):
    python prepare_subset.py \
        --src_json ./data/reveallayer_100k.json \
        --output_dir ./data/subset_1000 \
        --n_train 1000 \
        --n_val 100 \
        --seed 42

输出:
    ./data/subset_1000/
        train_1000.json      # 1000 条, 图片路径指向 ./images/<imgid>/
        val_100.json         # 100 条, 从 train_1000 中采样
        images/
            <imgid>/
                full_image.png
                background.png
                layer_0.png
                layer_1.png
                ...

后续训练直接用这个 JSON:
    python train.py --cfg_path configs/kontext_train_1024.py \
        --data_json ./data/subset_1000/train_1000.json
"""

import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any


def collect_image_paths(entry: Dict[str, Any]) -> List[Tuple[tuple, Any, str]]:
    """
    收集 entry 中所有图片路径。
    返回 List[(字段路径(相对容器), 字段值容器, 图片路径)]。
    字段路径用于 rewrite_paths 时回填新路径。
    """
    results = []

    # full_image / background: 单路径
    for key in ("full_image", "background"):
        if key in entry and isinstance(entry[key], str) and entry[key].strip():
            results.append(((key,), entry, entry[key].strip()))

    # LayerInfoRaw: list of paths
    if "LayerInfoRaw" in entry and isinstance(entry["LayerInfoRaw"], list):
        for i, p in enumerate(entry["LayerInfoRaw"]):
            if isinstance(p, str) and p.strip():
                results.append(((i,), entry["LayerInfoRaw"], p.strip()))

    # layers: list of dict with image-like fields
    if "layers" in entry and isinstance(entry["layers"], list):
        for i, layer in enumerate(entry["layers"]):
            if not isinstance(layer, dict):
                continue
            for field in ("rgba", "image", "path", "img"):
                if field in layer and isinstance(layer[field], str) and layer[field].strip():
                    results.append(((field,), layer, layer[field].strip()))

    return results


def rewrite_paths(entry: Dict[str, Any], path_map: Dict[str, str]) -> Dict[str, Any]:
    """把 entry 中所有图片路径替换为 path_map 中的新路径。"""
    entry = json.loads(json.dumps(entry))  # 深拷贝
    for field_path, container, old_path in collect_image_paths(entry):
        if old_path not in path_map:
            continue
        new_path = path_map[old_path]
        # field_path 只含最后一级 key/index
        if len(field_path) == 1:
            container[field_path[0]] = new_path
        else:
            raise ValueError(f"unexpected field_path length: {field_path}")
    return entry


def safe_copy(src: str, dst: str):
    """拷贝文件, 失败时打印警告。"""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"[WARN] copy failed: {src} -> {dst}: {e}")


def prepare_subset(
    src_json: str,
    output_dir: str,
    n_train: int = 1000,
    n_val: int = 100,
    seed: int = 42,
    root_dir: str = "",
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    with open(src_json, "r") as f:
        entries = json.load(f)
    print(f"[Prepare] loaded {len(entries)} entries from {src_json}")

    rng = random.Random(seed)
    if n_train >= len(entries):
        train_entries = list(entries)
        rng.shuffle(train_entries)
    else:
        train_entries = rng.sample(entries, n_train)

    # 拷贝训练集图片并重写路径
    new_train_entries = []
    for entry in train_entries:
        imgid = entry.get("imgid") or f"sample_{len(new_train_entries):06d}"
        sample_img_dir = os.path.join(images_dir, imgid)
        os.makedirs(sample_img_dir, exist_ok=True)

        path_map = {}
        for field_path, container, old_path in collect_image_paths(entry):
            # root_dir 拼接
            src_path = old_path
            if not os.path.isabs(src_path) and root_dir:
                src_path = os.path.join(root_dir, src_path)
            src_path = os.path.normpath(src_path)

            if not os.path.exists(src_path):
                print(f"[WARN] missing image: {src_path} (imgid={imgid})")
                continue

            ext = os.path.splitext(src_path)[1] or ".png"
            # 根据字段命名目标文件
            if field_path[0] == "full_image":
                dst_name = f"full_image{ext}"
            elif field_path[0] == "background":
                dst_name = f"background{ext}"
            elif field_path[0] == "LayerInfoRaw":
                dst_name = f"layer_{field_path[1]}{ext}"
            elif field_path[0] == "layers":
                dst_name = f"layer_{field_path[1]}{ext}"
            else:
                dst_name = f"image_{len(path_map)}{ext}"

            dst_path = os.path.join(sample_img_dir, dst_name)
            rel_dst = os.path.relpath(dst_path, output_dir)
            safe_copy(src_path, dst_path)
            path_map[old_path] = rel_dst

        new_entry = rewrite_paths(entry, path_map)
        new_entry["imgid"] = imgid
        new_train_entries.append(new_entry)

    # 从训练集中采样验证集
    if n_val >= len(new_train_entries):
        val_entries = list(new_train_entries)
        rng.shuffle(val_entries)
    else:
        val_entries = rng.sample(new_train_entries, n_val)

    train_json = os.path.join(output_dir, f"train_{len(new_train_entries)}.json")
    val_json = os.path.join(output_dir, f"val_{len(val_entries)}.json")

    with open(train_json, "w") as f:
        json.dump(new_train_entries, f, ensure_ascii=False, indent=2)
    with open(val_json, "w") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)

    print(f"[Prepare] train: {len(new_train_entries)} -> {train_json}")
    print(f"[Prepare] val  : {len(val_entries)} -> {val_json}")
    print(f"[Prepare] images copied to: {images_dir}")
    return train_json, val_json


def main():
    parser = argparse.ArgumentParser(description="Prepare RevealLayer train/val subset with copied images")
    parser.add_argument("--src_json", type=str, required=True, help="全量数据集 JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="子集输出目录")
    parser.add_argument("--n_train", type=int, default=1000, help="训练样本数")
    parser.add_argument("--n_val", type=int, default=100, help="验证样本数 (从训练集中采样)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--root_dir", type=str, default="", help="JSON 中相对路径的根目录")
    args = parser.parse_args()

    prepare_subset(
        src_json=args.src_json,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()
