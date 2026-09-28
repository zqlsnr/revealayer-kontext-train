#!/usr/bin/env bash
# ============================================================================
# RevealLayer-100K 数据集 + FLUX.1-Kontext-dev 基座模型 下载脚本
# ----------------------------------------------------------------------------
# 用法:
#   bash download_data.sh         # 下载数据集 + 基座
#   bash download_data.sh data    # 仅下载数据集
#   bash download_data.sh model   # 仅下载基座模型
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
MODEL_DIR="${PROJECT_ROOT}/models"

TARGET="${1:-all}"

download_data() {
    echo "============================================================"
    echo " 下载 RevealLayer-100K 数据集"
    echo "============================================================"
    mkdir -p "${DATA_DIR}"
    # 数据集托管在 HuggingFace: qihoo360/RevealLayer-100K
    if command -v git-lfs >/dev/null 2>&1; then
        git lfs install
    else
        echo "[警告] 未安装 git-lfs, 大文件可能无法下载. 请先: apt install git-lfs / brew install git-lfs"
    fi
    if [ ! -d "${DATA_DIR}/RevealLayer-100K" ]; then
        git clone https://huggingface.co/datasets/qihoo360/RevealLayer-100K "${DATA_DIR}/RevealLayer-100K"
    fi
    # 数据集内通常含标注 JSON + 图片。建立软链接到 data/reveallayer_100k.json
    if [ -f "${DATA_DIR}/RevealLayer-100K/reveallayer_100k.json" ]; then
        ln -sf "${DATA_DIR}/RevealLayer-100K/reveallayer_100k.json" "${DATA_DIR}/reveallayer_100k.json"
    fi
    echo "数据集下载完成: ${DATA_DIR}/RevealLayer-100K"
    echo "请确认 ${DATA_DIR}/reveallayer_100k.json 指向正确的标注文件"
}

download_model() {
    echo "============================================================"
    echo " 下载 FLUX.1-Kontext-dev 基座模型"
    echo "============================================================"
    mkdir -p "${MODEL_DIR}"
    if command -v git-lfs >/dev/null 2>&1; then
        git lfs install
    fi
    if [ ! -d "${MODEL_DIR}/FLUX.1-Kontext-dev" ]; then
        git clone https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev "${MODEL_DIR}/FLUX.1-Kontext-dev"
    fi
    # RevealLayer 已发布权重 (含 transparent_decoder_ckpt.pth, layer_pe.pt, Refiner.pt)
    if [ ! -d "${MODEL_DIR}/RevealLayer" ]; then
        git clone https://huggingface.co/qihoo360/RevealLayer "${MODEL_DIR}/RevealLayer"
    fi
    echo "模型下载完成:"
    echo "  基座: ${MODEL_DIR}/FLUX.1-Kontext-dev"
    echo "  RevealLayer: ${MODEL_DIR}/RevealLayer"
    echo ""
    echo "请把 configs/kontext_train_1024.py 中的 pretrained_model_name_or_path"
    echo "改为本地绝对路径, 例如: ${MODEL_DIR}/FLUX.1-Kontext-dev"
}

case "${TARGET}" in
    data)  download_data ;;
    model) download_model ;;
    all)   download_data; download_model ;;
    *) echo "用法: bash download_data.sh [all|data|model]"; exit 1 ;;
esac

echo ""
echo "下载完成后, 修改 configs/kontext_train_1024.py 中的路径, 然后运行:"
echo "  bash train.sh all"
