"""
RevealLayer 训练配置 — 基于 FLUX.1-Kontext-dev
=============================================
对应论文 (arXiv:2605.11818) 第 3 节训练协议:
  - 基座: black-forest-labs/FLUX.1-Kontext-dev  (原 infer.py 用 FLUX.1-dev)
  - 微调: LoRA rank=64
  - Rectified Flow + logit-normal 时间采样
  - 固定 prompt: "Decompose the image into foreground and background"
  - Prodigy 优化器 lr=1.0, bf16, gradient_checkpointing
  - guidance_scale=1.0 (论文 IMPORTANT 标注)
  - layer_weighting=5.0 (前景层 loss 加权)
"""

_base_ = "./base.py"

### path & device settings
output_path_base = "./output/"
cache_dir = None

### tracker settings
report_to = "tensorboard"    # 默认启用 tensorboard; 不需要时设为 None
wandb_job_name = "kontext_" + '{{fileBasenameNoExtension}}'

### 数据集
data_json = "./data/reveallayer_100k.json"   # RevealLayer-100K 标注文件
root_dir = ""                                 # 数据集根目录 (相对路径前缀)
resolution = 1024

### 模型设置 — 关键: 指向 FLUX.1-Kontext-dev
# 覆盖 base.py 中的 pretrained_model_name_or_path
# 默认相对路径与 download_data.sh 的下载位置 (./models/FLUX.1-Kontext-dev) 一致;
# 若模型在其他位置, 请改为本地绝对路径。
pretrained_model_name_or_path = "./models/FLUX.1-Kontext-dev"

rank = 64
text_encoder_rank = 64
train_text_encoder = False
max_layer_num = 12
learnable_proj = True

### 透明解码器 (论文 Hard-Constraint Alpha Loss 必需)
transp_vae_ckpt = "./models/RevealLayer/xvae/transparent_decoder_ckpt.pth"
# 论文 composite loss (式 16): L = L_FM + λα·L_α + λo·L_orth
# 最终确认 (8/14 起): λα=1.0, λo=1.0 (论文原值, 无物体外加权 w_bg)。
# 黑框根因已由"加性灰底合成"修复, 无需额外惩罚 → 全部回归论文配置。
# 注意: orth 像素空间版每步需主 VAE 解码参与层 (跳过 layer0), 训练变慢/显存增加;
#       若显存紧张, 可临时设 orth_loss_latent_space=True 用 latent 近似 (非论文公式, 仅调试)。
alpha_loss_weight = 1.0    # 论文 Hard-Constraint Alpha Loss (focal 形式, 式 13-14)
orth_loss_weight = 1.0     # 论文 Soft-Constraint Orthogonality Loss (像素空间, 式 15)
orth_loss_latent_space = False  # False=像素空间(论文公式); True=latent 近似(调试/省显存)

### 训练设置
weighting_scheme = "none"
# 实验: 论文式 7 写 logit-normal μ=0 σ=1, 但 t>0.95 样本仅 0.16% → 模型从未见过纯噪声输入
# → 推理从 t=1 起步崩溃 (出雪花). FLUX 官方实际用 logit_normal_mean=0.5 (偏向 t=1)
# 改 μ=1.0 是最贴近"覆盖 t≈1 训练"且不偏离论文谱系的选择, 重训小样本验证
logit_mean = 1.0
logit_std = 1.0
mode_scale = 1.29
guidance_scale = 1.0       ### IMPORTANT (论文与原配置一致)
layer_weighting = 5.0

# 文本条件
prompt = "Decompose the image into foreground and background."
max_sequence_length = 512

# steps
train_batch_size = 1
num_train_epochs = 1
max_train_steps = 50000   # 论文: 50,000 iterations (全局 batch = 8, 8卡 × bs=1)
checkpointing_steps = 2000
checkpoints_total_limit = 4
resume_from_checkpoint = None
gradient_accumulation_steps = 1

# lr
optimizer = "prodigy"
learning_rate = 1.0
scale_lr = False
lr_scheduler = "constant"
lr_warmup_steps = 0
lr_num_cycles = 1
lr_power = 1.0

# optim
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-3
adam_epsilon = 1e-8
prodigy_beta3 = None
prodigy_decouple = True
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True
max_grad_norm = 1.0

# logging
tracker_project_name = "reveallayer"
tracker_task_name = '{{fileBasenameNoExtension}}'
output_dir = output_path_base + "{{fileBasenameNoExtension}}"

### 验证设置
num_validation_images = 1
validation_steps = 2000
