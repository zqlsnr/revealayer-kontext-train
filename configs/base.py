### Model Settings
# 基座从 FLUX.1-dev 改为 FLUX.1-Kontext-dev (与 train.py 保持一致)
# RevealLayer 的多层 token 拼接 + 3D-RoPE 方案与 Kontext 的序列拼接方案兼容,
# 用 FLUX.1-Kontext-dev 作为基座可继承其图像上下文编辑先验。
# 默认相对路径与 download_data.sh 的下载位置 (./models/FLUX.1-Kontext-dev) 一致;
# 若模型在其他位置, 请改为本地绝对路径。
pretrained_model_name_or_path = "./models/FLUX.1-Kontext-dev"
revision = None
variant = None
cache_dir = None

### Training Settings
seed = 42
report_to = "wandb"          # tracker backend; configs/kontext_train_1024.py switches this to "tensorboard"
tracker_project_name = "multilayer"
# mmengine placeholder: the run name defaults to the config file name. Override
# it per experiment (the Kontext recipe uses "kontext_" + the same placeholder),
# or leave it and train.py falls back to "kontext_train".
wandb_job_name = "{{fileBasenameNoExtension}}"
logging_dir = "logs"
max_train_steps = None
checkpoints_total_limit = None

# gpu
allow_tf32 = True
gradient_checkpointing = True
mixed_precision = "bf16"
