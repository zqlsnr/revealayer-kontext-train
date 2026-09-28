#!/usr/bin/env python
"""决定性诊断: 模型在训练分布内能否正常预测 bg/fg?

原理:
- 从 val JSON 取一条样本 (imgid, boxes)
- 复用 infer.initialize_pipeline 加载 LoRA + layer_pe + Refiner + Kontext 基座
- 按 train.py 完全一致的方式构造输入:
    z_t = (1-t)*noise + t*z_1, layer0 始终干净, t ~ logit_normal(1.0, 1.0)
    adapter_data / latent_image_ids / split_sizes / text_ids 与训练一致
- 跑 transformer 一次, 对比 v_pred vs v_target = z_1 - noise (layer0 除外)

输出判定:
    layer1 (bg) loss_fm < 1.0  -> 模型训练分布内正常, 问题在推理侧输入分布
    layer1 (bg) loss_fm > 5.0  -> 模型根本没学会 (训练 bug / 权重未正确加载)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "diffusers" / "src"))

from infer import initialize_pipeline, parse_config, filter_and_align_bboxes, _transform_image_consistent_crop
from dataset import resize_with_bbox_edge
from train import (
    build_layer_latents, build_adapter_data, prepare_latent_image_ids,
    logit_normal_sample_t,
)


def get_args():
    p = argparse.ArgumentParser()
    # 二选一: --input_json (自动取第一条样本) 或 --image + --boxes (手动指定)
    p.add_argument("--input_json", default=None,
                   help="val JSON, 自动取第一条样本的 full_image 和 detections bbox")
    p.add_argument("--image", default=None)
    p.add_argument("--boxes", default=None, help="JSON list of [x1,y1,x2,y2]")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--transp_vae_ckpt", required=True)
    p.add_argument("--out", default="diag_traindist")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--max_layer", type=int, default=12)
    p.add_argument("--cfg_path", default="./configs/kontext_train_1024.py")
    p.add_argument("--gpu_id", type=int, default=0)
    return p.parse_args()


def resolve_sample(args):
    """从 input_json 自动取第一条样本, 或直接用 --image/--boxes。"""
    if args.input_json:
        with open(args.input_json, "r") as f:
            entries = json.load(f)
        entry = entries[0]
        image = entry["full_image"]
        dets = entry.get("detections") or []
        # detections 里每项含 bbox [x1,y1,x2,y2]; 与 validate.py 读取方式一致
        boxes = []
        for d in dets:
            if isinstance(d, dict) and "bbox" in d:
                boxes.append([int(v) for v in d["bbox"]])
            elif isinstance(d, (list, tuple)) and len(d) >= 4:
                boxes.append([int(v) for v in d[:4]])
        print(f"[diag] 从 {os.path.basename(args.input_json)} 取第一条: {entry.get('imgid', '?')}  {len(boxes)} fg boxes")
        return image, boxes
    if args.image is None or args.boxes is None:
        raise SystemExit("必须提供 --input_json 或 --image+--boxes")
    return args.image, json.loads(args.boxes)


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda", index=args.gpu_id)
    weight_dtype = torch.bfloat16

    image_path, boxes = resolve_sample(args)
    # 补齐 layer0/layer1 的 bbox (全图), boxes 参数只含 fg 层
    print(f"[diag] image: {image_path}")
    print(f"[diag] fg boxes: {boxes}")

    # 1) 复用 infer.py 的 pipeline 初始化 (基座 + custom transformer + LoRA + layer_pe + Refiner)
    config = parse_config(args.cfg_path)
    config.pretrained_model_name_or_path = args.pretrained_model_name_or_path

    class InferArgs:
        pass
    infer_args = InferArgs()
    infer_args.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    infer_args.ckpt_dir = args.ckpt_dir
    infer_args.gpu_id = args.gpu_id
    infer_args.max_layer = args.max_layer
    infer_args.extra_lora_dir = None
    infer_args.transp_vae_ckpt = args.transp_vae_ckpt

    pipeline = initialize_pipeline(config, infer_args)
    vae = pipeline.vae
    transformer = pipeline.transformer
    transformer.eval()

    # 2) 编码 prompt (与推理一致; 多层分解主要靠层结构, prompt 用通用描述)
    from models.custom_pipeline import encode_prompt
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        (pipeline.tokenizer, pipeline.tokenizer_2),
        (pipeline.text_encoder, pipeline.text_encoder_2),
        "a high quality photograph",
    )

    # 2) 准备图像 (与 validate.py 完全一致: resize_with_bbox_edge + bbox 对齐 + crop)
    from PIL import Image
    from torchvision import transforms
    full_image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = full_image_pil.size
    (cont_h, cont_w), (tgt_h, tgt_w), (off_y, off_x) = resize_with_bbox_edge(
        full_image_pil, resolution=args.resolution,
    )
    resized_boxes = filter_and_align_bboxes(
        boxes, orig_w, orig_h, cont_w, cont_h, off_x, off_y, tgt_w, tgt_h,
    )
    resized_image = _transform_image_consistent_crop(
        full_image_pil, cont_w, cont_h, tgt_w, tgt_h, off_x, off_y,
    )
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    full_image = image_transform(resized_image).to(device=device, dtype=weight_dtype).unsqueeze(0)
    background = full_image.clone()  # 近似: bg GT = full image (仅用于层结构构造)

    # 3) 构造 layer 结构 (layer0=full, layer1=bg, 2..=fg), 与训练一致
    n_total = 2 + len(resized_boxes)  # layer0 + bg + fg 层
    list_layer_box = [None] * n_total
    list_layer_box[0] = (0, 0, tgt_w, tgt_h)
    list_layer_box[1] = (0, 0, tgt_w, tgt_h)  # bg = 全图
    for i, box in enumerate(resized_boxes):
        list_layer_box[i + 2] = tuple(box)

    # 合成空 fg (仅用于 build_layer_latents 的层结构; bbox 大小正确即可)
    layer_rgba_list = []
    for box in resized_boxes:
        h = int(box[3]) - int(box[1])
        w = int(box[2]) - int(box[0])
        rgba = torch.zeros(4, h, w, device=device, dtype=weight_dtype)
        rgba[3] = 1.0
        layer_rgba_list.append(rgba)

    # 4) 训练分布输入 (与 train.py 主循环完全一致)
    #    注意: build_layer_latents / build_adapter_data 期望 full_image 为 3D [3,H,W] (内部自行 unsqueeze)
    H_lat = tgt_h // 8
    W_lat = tgt_w // 8
    data = build_layer_latents(full_image[0], background[0], layer_rgba_list,
                               list_layer_box, vae, device, weight_dtype)
    z_1 = data["z_1"]
    layer_valid = data["layer_valid"]
    adapter_data = build_adapter_data(full_image[0], list_layer_box, vae, device, weight_dtype)
    latent_image_ids = prepare_latent_image_ids(H_lat, W_lat, list_layer_box, device, weight_dtype)

    # 5) 多次 t 采样 (覆盖训练分布), 统计每层平均 loss
    print(f"\n[diag] 训练分布: logit_normal(mu=1.0, std=1.0), {len(boxes)} 层")
    logit_mean, logit_std = 1.0, 1.0
    print("        layer0=条件(clean), layer1=bg, layer2..=fg")
    print("        v_pred 由 transformer 直接输出, 与训练主循环一致")

    n_trials = 8
    per_layer_loss = {i: [] for i in range(z_1.shape[1])}
    per_layer_cos = {i: [] for i in range(z_1.shape[1])}
    sampled_ts = []

    with torch.no_grad():
        for _ in range(n_trials):
            t = logit_normal_sample_t(1, logit_mean, logit_std, device, weight_dtype)
            sampled_ts.append(t.item())
            noise = torch.randn_like(z_1)
            # 与修复后的 train.py 一致: t=噪声权重, z_t = t·noise + (1-t)·z_1, v = noise - z_1
            z_t = t.view(1,1,1,1,1) * noise + (1.0 - t.view(1,1,1,1,1)) * z_1
            z_t[:, 0] = z_1[:, 0]          # layer0 始终干净
            v_target = noise - z_1
            v_target[:, 0] = 0.0

            model_out = transformer(
                hidden_states=z_t,
                adapter_data=adapter_data,
                split_sizes=data["split_sizes"],
                list_layer_box=list_layer_box,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=torch.full([1], 1.0, device=device, dtype=weight_dtype),
                return_dict=False,
            )[0]

            for i in range(z_1.shape[1]):
                if not layer_valid[i]:
                    continue
                box = list_layer_box[i]
                x1, y1, x2, y2 = [int(v) // 8 for v in box]
                vp = model_out[0, i, :, y1:y2, x1:x2].float()
                vt = v_target[0, i, :, y1:y2, x1:x2].float()
                if vp.numel() == 0:
                    continue
                loss = ((vp - vt) ** 2).mean().item()
                cos = torch.nn.functional.cosine_similarity(
                    vp.flatten().unsqueeze(0), vt.flatten().unsqueeze(0)).item()
                per_layer_loss[i].append(loss)
                per_layer_cos[i].append(cos)

    print(f"\n===== 训练分布下 {n_trials} 次 t 采样 (t={[f'{v:.2f}' for v in sorted(sampled_ts)]}) =====")
    for i in range(z_1.shape[1]):
        if not layer_valid[i]:
            print(f"layer{i} (条件, 不参与 loss)")
            continue
        losses = per_layer_loss[i]
        coss = per_layer_cos[i]
        if not losses:
            print(f"layer{i}: 无有效 bbox")
            continue
        avg = sum(losses) / len(losses)
        cos_avg = sum(coss) / len(coss)
        name = "bg" if i == 1 else f"fg{i-1}"
        print(f"layer{i} ({name}): loss_fm(mean)={avg:.4f}  cos_sim(mean)={cos_avg:.4f}")

    bg_loss = sum(per_layer_loss[1]) / max(len(per_layer_loss[1]), 1)
    print(f"\n[KEY] layer1 (bg) 训练分布内 loss_fm = {bg_loss:.4f}")
    if bg_loss < 1.0:
        print("[OK] 模型在训练分布内能预测 bg -> 问题在推理侧输入分布 (timestep/噪声/分布外)")
    elif bg_loss < 5.0:
        print("[WARN] 部分学会但不太对 -> 训练量不足或训练侧小问题")
    else:
        print("[FAIL] 模型在训练分布内也预测不了 bg -> 训练 bug 或权重未真正加载")


if __name__ == "__main__":
    main()