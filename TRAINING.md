# RevealLayer 训练说明 (基于 FLUX.1-Kontext-dev)

本目录新增的多 GPU 训练代码严格按论文
**"RevealLayer: Disentangling Hidden and Visible Layers via Occlusion-Aware Image Decomposition"**
(arXiv:2605.11818) 第 3 节的训练协议实现，并把基座模型从 `FLUX.1-dev` 改为
`FLUX.1-Kontext-dev`。

## 一、新增/修改文件

| 文件 | 说明 |
| --- | --- |
| `dataset.py` | RevealLayer-100K 数据集加载，VAE 编码 + 多层 token 序列拼接 (论文式 2-4)，含子集采样 `make_subset_json` |
| `prepare_subset.py` | 随机采样训练/验证子集，并**拷贝图片到独立目录**，生成指向新位置的 train/val JSON |
| `train.py` | 多 GPU 训练主程序 (accelerate DDP + Rectified Flow + LoRA)，支持 `--max_samples` 小样本测试 + 训练后自动验证 |
| `validate.py` | 验证推理脚本：加载训练 checkpoint 推理，计算 PSNR/SSIM 指标，输出 PASS/FAIL 判定 |
| `train.sh` | 两阶段流程脚本：`stage1`(拷贝1000张图+训练) → `validate`(100张推理) → `stage2`(全量) |
| `download_data.sh` | RevealLayer-100K 数据集 + FLUX.1-Kontext-dev 基座下载脚本 |
| `configs/kontext_train_1024.py` | 基于 FLUX.1-Kontext-dev 的训练配置 |
| `configs/base.py` | 基座路径由 `FLUX.1-dev` 改为 `FLUX.1-Kontext-dev` |
| `infer.py` | 推理默认基座改为 `FLUX.1-Kontext-dev` |
| `infer_single.py` | 单图推理脚本：输入图片 + boxes，输出分层 RGBA 和合成图 |
| `convert_checkpoint.py` | 修复旧 checkpoint 的 LoRA key 格式 / 拆分 extra_modules.pt |
| `check_lora.py` | 诊断 checkpoint 中 LoRA 覆盖的模块和权重状态 |
| `inspect_transformer.py` | 查看模型所有 nn.Linear 模块路径, 辅助确定 LoRA target_modules |
| `TRAINING.md` | 本文档 |

## 二、为什么从 FLUX.1-dev 改为 FLUX.1-Kontext-dev

论文原文 §3.2 选择 `FLUX.1 [dev]` 作为骨干，采用 **多层 latent token 拼接 + 3D-RoPE**
把不同图层 join 成一个变长序列 (论文式 4)。这与 `FLUX.1-Kontext-dev` 的设计目标
高度契合 —— Kontext 本身就是用 **序列拼接** 把上下文图像 token 附加到目标 token 后、
用 **3D-RoPE** 的第一维给上下文 token 加常数偏移 (Kontext 论文 §3 的 *virtual time step*)。

因此把基座换成 `FLUX.1-Kontext-dev` 后：

1. 模型从一开始就具备 **图像上下文编辑能力** 的先验，RevealLayer 的多图层分解任务
   可以视为 Kontext 单图编辑的多图扩展；
2. 架构无需任何改动 —— `FLUX.1-Kontext-dev` 的 `FluxTransformer2DModel` 配置
   (`in_channels=64, num_layers=19, num_single_layers=38, attention_head_dim=128`)
   与 `FLUX.1-dev` 完全一致，`CustomFluxTransformer2DModel` 可无损承接权重；
3. 新增的 `layer_pe` 与 `HybridRefiner` (OGA) 仍然随机初始化后参与训练。

## 三、训练协议 ↔ 代码对应

| 论文章节 | 论文方法 | 代码位置 |
| --- | --- | --- |
| §3.2 式 (2)(3)(4) | I / I_bg / {I_fg^i} 经 VAE 编码后拼接为统一序列 z_0 | `dataset.build_layer_latents` |
| §3.2 3D-RoPE | 第一维承载 layer_id | `train.prepare_latent_image_ids` |
| §3.2 式 (5)(6) | Rectified Flow: z_t = t·z_0 + (1-t)·z_1, v_t = z_0 − z_1 | `train.py` 训练循环 (注: 采用 FLUX 约定 t=1 clean, t=0 noise, 与论文等价) |
| §3.2 式 (7) | L_FM = Σ_i ‖v_θ^i − v_t^i‖² | `compute loss_fm` 循环 |
| §3.2 时间采样 | logit-normal p(t; μ, σ=1.0) | `logit_normal_sample_t` |
| §3.2 文本条件 | 固定 prompt "Decompose the image into foreground and background" | `configs/kontext_train_1024.py: prompt` |
| §3.3 RAA | Region-Aware Attention (attention mask) | `CustomFluxTransformer2DModel.build_additive_spatial_roi_mask_v2` (已有) |
| §3.4 OGA | Occlusion-Guided Adapter (HybridRefiner) | `models/adapter.py` + `build_adapter_data` |
| §3.5 Hard-Constraint Alpha Loss | 透明边界锐化 | `compute_alpha_loss` (默认关闭, 设 `alpha_loss_weight>0` 开启) |
| §3.5 Soft-Constraint Orthogonality Loss | 层间特征正交 | `compute_orthogonality_loss` |
| §3.5 layer_weighting | 前景层 loss 加权 | `configs: layer_weighting=5.0` |
| 训练超参 | LoRA rank=64, Prodigy lr=1.0, bf16, grad_checkpoint | `configs/kontext_train_1024.py` |
| 多 GPU | FSDP/DDP, bf16 all-gather + fp32 reduce-scatter, 激活检查点 | `accelerate` DDP + `gradient_checkpointing` |

> 注：论文里 FLUX.1-Kontext 原文训练用 FSDP2 全参数训练；RevealLayer 用 LoRA
> (rank=64) 微调 + Prodigy 优化器，这是 RevealLayer 自己的训练设置 (见原仓库
> `configs/ld_resolution1024_test.py`)。本训练代码遵循 **RevealLayer 论文的设置**。

## 四、快速开始

### 1. 准备环境

```bash
conda create -n reveallayer python=3.10
conda activate reveallayer
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
cd diffusers && pip install . && cd ..
# 训练额外依赖
pip install prodigyopt accelerate peft wandb
```

### 2. 下载模型

```bash
git lfs install
# 基座 (改为 Kontext)
git clone https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev models/FLUX.1-Kontext-dev
# RevealLayer 已发布权重 (含 transparent_decoder_ckpt.pth, layer_pe.pt, Refiner.pt)
git clone https://huggingface.co/qihoo360/RevealLayer models/RevealLayer
```

### 3. 准备数据

`data/reveallayer_100k.json` 格式 (与 README 中推理 JSON 一致):

```json
[
  {
    "imgid": "000001",
    "full_image": "images/000001_full.png",
    "background": "images/000001_bg.png",
    "LayerInfoRaw": ["images/000001_layer_0.png", "images/000001_layer_1.png"],
    "detections": [{"bbox": [x1, y1, x2, y2]}, ...]
  }, ...
]
```

### 4. 修改配置

编辑 `configs/kontext_train_1024.py`，把 `pretrained_model_name_or_path` 改为本地
`FLUX.1-Kontext-dev` 路径，并确认 `data_json` / `transp_vae_ckpt` 路径正确。

### 5. 两阶段训练流程（推荐）

为避免直接在全量 100K 数据上跑出问题，采用 **stage1 测试 → validate → stage2 全量** 三步流程。
stage1 会先随机采样 1000 张图片并**拷贝到独立目录**，直接用这个 1000 张图训练；
训练完成后从这 1000 张中再随机采样 100 张做推理验证。

```bash
# 一键跑完整流程 (推荐): stage1(1000样本拷贝+训练) → validate(100张推理验证) → stage2(全量训练)
bash train.sh all

# 或分步执行
bash train.sh stage1     # 1000 样本拷贝到 data/subset_1000, 然后训练 (默认 2000 steps)
bash train.sh validate   # 100 样本推理验证, 计算 PSNR/SSIM, 输出 PASS/FAIL
bash train.sh stage2     # 验证通过后跑全量训练 (默认 100000 steps)
```

可调环境变量：

```bash
TEST_SAMPLES=1000        # stage1 训练样本数
TEST_STEPS=2000          # stage1 训练步数
FULL_STEPS=100000        # stage2 全量训练步数
VALIDATE_SAMPLES=100     # 验证推理样本数 (从这 1000 张中采样)
DATA_JSON=data/reveallayer_100k.json   # 全量数据 JSON
SUBSET_DIR=data/subset_1000            # stage1 子集输出目录
ROOT_DIR=data/                         # JSON 中相对路径的根目录
NPROC=8                  # GPU 数
bash train.sh all 8
```

stage1 完成后目录结构：

```
data/subset_1000/
  train_1000.json       # 1000 条训练样本, 图片路径指向 images/<imgid>/
  val_100.json          # 100 条验证样本
  images/
    <imgid>/
      full_image.png
      background.png
      layer_0.png
      ...
```

**验证判定标准**（可在 `validate.py` 命令行覆盖）：
- 背景 PSNR ≥ 20 dB 且 SSIM ≥ 0.7 → PASS，自动继续 stage2
- 否则 → FAIL，需排查训练/数据问题后再跑全量

验证产物：
- `output/stage1_test/validate/validate_summary.json` — 汇总 + 逐样本指标
- `output/stage1_test/validate/<imgid>/cmp_*.png` — GT vs 预测可视化对比
- `output/stage1_test/validate/VALIDATE_RESULT` — `PASS`/`FAIL` 标志（train.sh 据此决定是否继续）

### 6. 单独训练/验证（不用流程脚本）

```bash
# 直接全量训练 (跳过两阶段)
bash train.sh stage2 8

# 或手动调用 train.py (torchrun)
torchrun --nproc_per_node=8 train.py --cfg_path configs/kontext_train_1024.py

# 手动验证某个 checkpoint
python validate.py \
    --input_json data/reveallayer_100k_subset_1000.json \
    --ckpt_dir output/stage1_test/final \
    --pretrained_model_name_or_path models/FLUX.1-Kontext-dev \
    --cfg_path configs/kontext_train_1024.py \
    --max_samples 50

# 训练后自动验证
python train.py --cfg_path configs/kontext_train_1024.py \
    --run_validate_after

## 四、推理出现噪声 / checkpoint 加载失败

如果 `validate.py` 或 `infer_single.py` 输出纯噪声, 最常见原因是 **checkpoint 没有正确加载**,
模型退化成未微调的基座 FLUX。请按下面顺序排查:

### 4.1 检查 checkpoint 文件是否存在

stage1 完成后 `output/stage1_test/final/` 应包含:

```
pytorch_lora_weights.safetensors   # LoRA 权重 (必须有)
layer_pe.pt                         # layer_pe 全参数权重
Refiner.pt                          # HybridRefiner 权重
```

如果缺少其中任何一个, 说明训练保存有问题, 需要重新训练或从中间 checkpoint 恢复。

### 4.2 修复旧 checkpoint 的 LoRA key 格式

早期版本保存的 LoRA key 可能与 `infer.py` 期望的 `transformer.xxx` 前缀不一致, 导致加载时被跳过。
运行转换脚本修复:

```bash
python convert_checkpoint.py \
    --src output/stage1_test/final \
    --dst output/stage1_test/final_fixed

# 然后用修复后的目录验证
python validate.py \
    --input_json data/subset_1000/val_100.json \
    --root_dir data/subset_1000 \
    --ckpt_dir output/stage1_test/final_fixed \
    --pretrained_model_name_or_path models/FLUX.1-Kontext-dev \
    --cfg_path configs/kontext_train_1024.py \
    --max_samples 10
```

### 4.3 确认训练步数足够

stage1 默认 `TEST_STEPS=2000`、1000 张图, 仅用于**验证流程不发散**。
想看到可接受的分层效果, 建议:

```bash
# 小样本测试: 100 张图 + 1000~2000 steps
TEST_SAMPLES=100 TEST_STEPS=2000 bash train.sh stage1

# 想看效果: 1000 张图 + 10000 steps
TEST_STEPS=10000 bash train.sh stage1

# 正式训练: 全量 100K + 100000 steps
bash train.sh stage2
```

### 4.4 推荐安装 prodigyopt

论文使用 Prodigy 优化器 (lr=1.0)。若未安装会 fallback 到 AdamW (lr=1e-4), 也能跑但收敛较慢:

```bash
pip install prodigyopt
```
    --max_samples 1000 --subset_json data/subset_1000.json \
    --run_validate_after --validate_samples 50
```

### 7. 训练产物

checkpoint 保存在 `output/stage1_test/checkpoint-{step}/` 或 `output/stage2_full/checkpoint-{step}/`:
- accelerator state (optimizer / scheduler / model)
- `pytorch_lora_weights.safetensors.pt` — LoRA 权重
- `extra_modules.pt` — `layer_pe` + `HybridRefiner` 权重

训练结束 `{stage}/final/` 包含可直接被 `infer.py` 加载的:
- `pytorch_lora_weights.safetensors`
- `layer_pe.pt`
- `Refiner.pt`

## 五、关键超参 (对应论文)

| 参数 | 值 | 来源 |
| --- | --- | --- |
| `rank` (LoRA) | 64 | 原仓库配置 |
| `optimizer` | prodigy | 原仓库配置 |
| `learning_rate` | 1.0 | 原仓库配置 |
| `guidance_scale` | 1.0 | 论文 IMPORTANT 标注 |
| `layer_weighting` | 5.0 | 原仓库配置 |
| `logit_mean / logit_std` | 0.0 / 1.0 | 论文式 (7) |
| `train_batch_size` | 1 (per GPU) | 原仓库配置 |
| `mixed_precision` | bf16 | 论文实现细节 |
| `gradient_checkpointing` | True | 论文实现细节 |
| `max_grad_norm` | 1.0 | 原仓库配置 |
| `resolution` | 1024 | 论文 |
| `max_layer_num` | 12 | 原仓库配置 |
| `prompt` | "Decompose the image into foreground and background." | 论文式 (7) |

## 六、注意事项

1. **Rectified Flow 时间约定**: 论文式 (5) 写 `z_t = t·z_0 + (1-t)·z_1` (z_0 噪声,
   z_1 clean)，FLUX/diffusers 标准约定是 `t=1` clean、`t=0` 噪声。两者等价 (t 翻转)，
   代码采用 FLUX 约定以与推理 `FlowMatchEulerDiscreteScheduler` 对齐。
2. **条件层不加噪**: layer 0 (full image) 作为条件输入，训练时 `z_t[:,0] = z_1[:,0]`
   恒定，与推理 `latents[:,0] = full_image_latent` 一致。
3. **alpha / orthogonality loss 默认关闭**: 论文未公开这两个 loss 的精确公式，
   代码给出合理实现但默认权重 0；如需复现可设 `alpha_loss_weight` / `orth_loss_weight > 0`。
4. **显存**: bs=1 + LoRA + gradient_checkpointing 下，单卡 80G (A100/H100) 可跑 1024 分辨率。
   多卡 DDP 线性扩展 batch。
