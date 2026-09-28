"""
RevealLayer 单图推理脚本 (基于 FLUX.1-Kontext-dev)
=====================================================

输入: 一张图片 + 一组前景 bbox
输出: 分层 RGBA 图 + 合成图 + 背景图

用法示例:
    python infer_single.py \
        --image /path/to/RevealLayer/11.png \
        --boxes '[[639, 246, 1089, 1328], [194, 404, 682, 1328], [161, 49, 383, 498], [1020, 139, 1229, 457]]' \
        --output_dir ./results_single/11 \
        --ckpt_dir ./ckpts/RevealLayer \
        --pretrained_model_name_or_path /path/to/FLUX.1-Kontext-dev \
        --transp_vae_ckpt ./ckpts/xvae/transparent_decoder_ckpt.pth

Boxes 也可以从 JSON 文件读取:
    python infer_single.py \
        --image 11.png \
        --boxes_file boxes.json \
        --output_dir ./results_single/11
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch

from infer import (
    seed_everything,
    parse_config,
    initialize_pipeline,
    test_one_sample,
)
from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE


def parse_boxes(s: str):
    """解析命令行传入的 boxes JSON 字符串。"""
    try:
        boxes = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--boxes 必须是 JSON 数组字符串, 错误: {e}")

    if not isinstance(boxes, list):
        raise ValueError("--boxes 必须是一个 list")

    cleaned = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError(f"每个 box 必须是 [x1, y1, x2, y2], 得到: {box}")
        cleaned.append([int(v) for v in box])
    return cleaned


def build_args_from_cmdline():
    parser = argparse.ArgumentParser(
        description="Single-image inference for RevealLayer"
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "--boxes",
        type=str,
        default=None,
        help='前景 bbox JSON 字符串, 例如 "[[639,246,1089,1328],...]"',
    )
    parser.add_argument(
        "--boxes_file",
        type=str,
        default=None,
        help="前景 bbox JSON 文件路径 (与 --boxes 二选一)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录, 默认 ./results_single/<图片名>",
    )

    # 模型路径
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=os.environ.get("REVEALLAYER_MODEL_DIR", "./models/FLUX.1-Kontext-dev"),
        help="FLUX.1-Kontext-dev 基座目录 (默认 ./models/FLUX.1-Kontext-dev, "
        "可用环境变量 REVEALLAYER_MODEL_DIR 覆盖)",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./configs/ld_resolution1024_test.py",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpts/RevealLayer",
        help="包含 pytorch_lora_weights.safetensors / layer_pe*.pt / Refiner.pt 的目录",
    )
    parser.add_argument(
        "--transp_vae_ckpt",
        type=str,
        default="./ckpts/xvae/transparent_decoder_ckpt.pth",
    )

    # 生成参数
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--steps", type=int, default=30, help="扩散步数")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_layer", type=int, default=12)

    args = parser.parse_args()

    # 解析 boxes
    if args.boxes is not None and args.boxes_file is not None:
        raise ValueError("--boxes 和 --boxes_file 不能同时指定")
    if args.boxes is None and args.boxes_file is None:
        raise ValueError("必须指定 --boxes 或 --boxes_file 之一")

    if args.boxes_file is not None:
        with open(args.boxes_file, "r") as f:
            boxes = json.load(f)
        args.boxes_list = boxes
    else:
        args.boxes_list = parse_boxes(args.boxes)

    # 默认输出目录
    if args.output_dir is None:
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output_dir = os.path.join("./results_single", image_name)

    return args


def main():
    args = build_args_from_cmdline()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        seed_everything(args.seed)

    config = parse_config(args.cfg_path)
    device = torch.device("cuda", index=args.gpu_id)

    print(f"--- RevealLayer single-image inference ---")
    print(f"Image : {args.image}")
    print(f"Boxes : {args.boxes_list}")
    print(f"Output: {args.output_dir}")

    pipeline = initialize_pipeline(config, args)

    transp_vae = CustomVAE()
    transp_vae.load_state_dict(
        torch.load(args.transp_vae_ckpt, map_location="cpu"),
        strict=False,
    )
    transp_vae.to(device).eval()

    sample = {
        "index": os.path.splitext(os.path.basename(args.image))[0],
        "description": "A detailed image.",
        "layout": args.boxes_list,
        "full_image_path": args.image,
    }

    test_one_sample(args, sample, pipeline, transp_vae, device, shuffle=False)

    print(f"\nDone. Results saved to: {args.output_dir}")

    del pipeline, transp_vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
