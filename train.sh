#!/usr/bin/env bash
# ============================================================================
# RevealLayer 两阶段训练流程 (基于 FLUX.1-Kontext-dev)
# ----------------------------------------------------------------------------
# 流程:
#   stage1  从 RevealLayer-100K 随机采样 1000 样本, 拷贝到指定目录,
#           用这 1000 张图直接训练 (少量 steps)
#   validate 从这 1000 张中随机采样 100 张做推理验证, 输出 PASS/FAIL
#   stage2  验证通过后才跑全量 100K 训练
#
# 用法:
#   bash train.sh all                 # 完整三步流程 (推荐)
#   bash train.sh stage1              # 只跑 stage1 测试训练
#   bash train.sh validate            # 只跑验证 (需先 stage1)
#   bash train.sh stage2              # 只跑全量训练 (需先 validate PASS)
#   bash train.sh stage1 8            # 指定 8 卡
#
# 环境变量:
#   NPROC            GPU 数 (默认自动检测)
#   TEST_SAMPLES     stage1 训练样本数 (默认 1000)
#   TEST_STEPS       stage1 训练步数 (默认 2000)
#   FULL_STEPS       stage2 全量训练步数 (默认 50000, 论文 50k iterations)
#   VALIDATE_SAMPLES 验证推理样本数 (默认 100, 从 TEST_SAMPLES 中采样)
#   DATA_JSON        全量数据 JSON (默认 data/reveallayer_100k.json)
#   SUBSET_DIR       stage1 子集输出目录 (默认 data/subset_1000)
#   ROOT_DIR         JSON 中相对路径的根目录 (默认 data/)
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/diffusers/src:${PYTHONPATH:-}"

STAGE="${1:-all}"
NPROC="${2:-${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 8)}}"

# ---- 可配置参数 ----
CFG_NAME="${CFG_NAME:-kontext_train_1024}"
CFG_PATH="${PROJECT_ROOT}/configs/${CFG_NAME}.py"
DATA_JSON="${DATA_JSON:-${PROJECT_ROOT}/data/reveallayer_100k.json}"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data}"
SUBSET_DIR="${SUBSET_DIR:-${PROJECT_ROOT}/data/subset_1000}"
TEST_SAMPLES="${TEST_SAMPLES:-1000}"
TEST_STEPS="${TEST_STEPS:-2000}"
FULL_STEPS="${FULL_STEPS:-50000}"   # 论文: 50,000 iterations
VALIDATE_SAMPLES="${VALIDATE_SAMPLES:-100}"
TRANSP_VAE_CKPT="${TRANSP_VAE_CKPT:-${PROJECT_ROOT}/models/RevealLayer/xvae/transparent_decoder_ckpt.pth}"

# 子集 JSON (prepare_subset.py 生成)
TRAIN_JSON="${SUBSET_DIR}/train_${TEST_SAMPLES}.json"
VAL_JSON="${SUBSET_DIR}/val_${VALIDATE_SAMPLES}.json"

# 输出目录
STAGE1_OUT="${PROJECT_ROOT}/output/stage1_test"
STAGE2_OUT="${PROJECT_ROOT}/output/stage2_full"
VALIDATE_OUT="${STAGE1_OUT}/validate"
LOGDIR="${PROJECT_ROOT}/logs/train"
mkdir -p "${LOGDIR}"

PYTHON="${PYTHON:-python}"

# ---- 工具函数 ----
log()  { echo -e "\033[1;34m[train.sh]\033[0m $*"; }
err()  { echo -e "\033[1;31m[train.sh ERROR]\033[0m $*" >&2; }

check_data() {
    if [ ! -f "${DATA_JSON}" ]; then
        err "全量数据 JSON 不存在: ${DATA_JSON}"
        err "请先下载数据集:  bash download_data.sh"
        exit 1
    fi
}

check_model() {
    local model_dir
    model_dir=$(python -c "
import sys; sys.path.insert(0,'${PROJECT_ROOT}')
from mmengine.config import Config
print(Config.fromfile('${CFG_PATH}').pretrained_model_name_or_path)
" 2>/dev/null || echo "")
    if [ -n "${model_dir}" ] && [ ! -d "${model_dir}" ]; then
        err "基座模型目录不存在: ${model_dir}"
        err "请下载 FLUX.1-Kontext-dev:  git clone https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev \"${model_dir}\""
        exit 1
    fi
}

# ============================================================================ #
#  准备 stage1 子集 (拷贝图片 + 生成 train/val JSON)
# ============================================================================ #
prepare_subset_data() {
    if [ -f "${TRAIN_JSON}" ] && [ -f "${VAL_JSON}" ]; then
        log "子集已存在, 跳过准备: ${TRAIN_JSON} / ${VAL_JSON}"
        log "如需重新准备, 请删除 ${SUBSET_DIR} 后重试"
        return 0
    fi

    log "===== PREPARE: 采样 ${TEST_SAMPLES} 训练样本 + ${VALIDATE_SAMPLES} 验证样本 ====="
    log "输出目录: ${SUBSET_DIR}"
    check_data

    ${PYTHON} "${PROJECT_ROOT}/prepare_subset.py" \
        --src_json "${DATA_JSON}" \
        --output_dir "${SUBSET_DIR}" \
        --n_train "${TEST_SAMPLES}" \
        --n_val "${VALIDATE_SAMPLES}" \
        --seed 42 \
        --root_dir "${ROOT_DIR}" \
        2>&1 | tee "${LOGDIR}/prepare_subset_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "${TRAIN_JSON}" ] || [ ! -f "${VAL_JSON}" ]; then
        err "prepare_subset.py 未生成预期 JSON"
        exit 1
    fi
}

run_torchrun() {
    # $1=output_dir  $2=max_steps  $3=data_json  $4=root_dir  $5=resume_ckpt(可选)
    local out_dir="$1"; local max_steps="$2"; local data_json="$3"; local root_dir="$4"; local resume_ckpt="${5:-}"
    # 临时配置只用于覆盖 output_dir (避免污染原配置)
    local tmp_cfg="${PROJECT_ROOT}/configs/_tmp_${STAGE}.py"
    cat > "${tmp_cfg}" <<EOF
_base_ = "./${CFG_NAME}.py"
output_dir = "${out_dir}"
EOF
    log "[run_torchrun] tmp_cfg=${tmp_cfg} max_train_steps=${max_steps} output_dir=${out_dir} resume=${resume_ckpt:-none}"
    torchrun --nproc_per_node="${NPROC}" \
        "${PROJECT_ROOT}/train.py" \
        --cfg_path "${tmp_cfg}" \
        --data_json "${data_json}" \
        --root_dir "${root_dir}" \
        --transp_vae_ckpt "${TRANSP_VAE_CKPT}" \
        --max_train_steps "${max_steps}" \
        --max_layer 12 \
        --num_workers 2 \
        ${resume_ckpt:+--resume_from_checkpoint "${resume_ckpt}"} \
        2>&1 | tee "${LOGDIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
}

# ============================================================================ #
#  stage1: 1000 样本测试训练
# ============================================================================ #
do_stage1() {
    log "===== STAGE 1: 测试训练 (${TEST_SAMPLES} 样本, ${TEST_STEPS} steps, ${NPROC} GPU) ====="
    check_model
    prepare_subset_data
    run_torchrun "${STAGE1_OUT}" "${TEST_STEPS}" "${TRAIN_JSON}" "${SUBSET_DIR}"
    log "stage1 完成. checkpoint: ${STAGE1_OUT}/final"
}

# ============================================================================ #
#  validate: 推理验证 (从 1000 中采样的 100 张)
# ============================================================================ #
do_validate() {
    log "===== VALIDATE: 推理验证 (${VALIDATE_SAMPLES} 样本) ====="
    if [ ! -d "${STAGE1_OUT}/final" ]; then
        err "stage1 checkpoint 不存在: ${STAGE1_OUT}/final, 请先运行 bash train.sh stage1"
        exit 1
    fi
    if [ ! -f "${VAL_JSON}" ]; then
        err "验证集 JSON 不存在: ${VAL_JSON}, 请先运行 stage1"
        exit 1
    fi
    local model_dir
    model_dir=$(python -c "
import sys; sys.path.insert(0,'${PROJECT_ROOT}')
from mmengine.config import Config
print(Config.fromfile('${CFG_PATH}').pretrained_model_name_or_path)
" 2>/dev/null)
    CUDA_VISIBLE_DEVICES=0 ${PYTHON} "${PROJECT_ROOT}/validate.py" \
        --input_json "${VAL_JSON}" \
        --root_dir "${SUBSET_DIR}" \
        --ckpt_dir "${STAGE1_OUT}/final" \
        --pretrained_model_name_or_path "${model_dir}" \
        --cfg_path "${CFG_PATH}" \
        --transp_vae_ckpt "${TRANSP_VAE_CKPT}" \
        --output_dir "${VALIDATE_OUT}" \
        2>&1 | tee "${LOGDIR}/validate_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "${VALIDATE_OUT}/VALIDATE_RESULT" ]; then
        err "验证未生成结果文件, 视为 FAIL"
        exit 1
    fi
    local result
    result=$(cat "${VALIDATE_OUT}/VALIDATE_RESULT")
    if [ "${result}" = "PASS" ]; then
        log "验证 PASS, 可继续 stage2 全量训练"
    else
        err "验证 FAIL, 请检查 ${VALIDATE_OUT}/validate_summary.json 与可视化结果"
        err "排查后再运行: bash train.sh stage2"
        exit 1
    fi
}

# ============================================================================ #
#  stage2: 全量训练
# ============================================================================ #
do_stage2() {
    log "===== STAGE 2: 全量训练 (100K 样本, ${FULL_STEPS} steps, ${NPROC} GPU) ====="
    check_data; check_model
    # 可选: 要求 stage1 验证通过
    if [ -f "${VALIDATE_OUT}/VALIDATE_RESULT" ]; then
        local result
        result=$(cat "${VALIDATE_OUT}/VALIDATE_RESULT")
        if [ "${result}" != "PASS" ]; then
            err "stage1 验证未通过 (${result}), 强制继续请删除 ${VALIDATE_OUT}/VALIDATE_RESULT"
            exit 1
        fi
    else
        log "[警告] 未找到 stage1 验证结果, 直接启动 stage2 (建议先 all 或 stage1+validate)"
    fi
    # resume 优先级: ① stage2 自己的最新 checkpoint (中断续训) ② stage1 最新 checkpoint (warm start)
    local resume_ckpt=""
    if [ -d "${STAGE2_OUT}" ]; then
        resume_ckpt=$(ls -d "${STAGE2_OUT}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "${resume_ckpt}" ]; then
            log "stage2 续训从自身 checkpoint 恢复: ${resume_ckpt}"
        fi
    fi
    if [ -z "${resume_ckpt}" ] && [ -d "${STAGE1_OUT}" ]; then
        resume_ckpt=$(ls -d "${STAGE1_OUT}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "${resume_ckpt}" ]; then
            log "stage2 warm start 从 stage1 checkpoint 恢复: ${resume_ckpt}"
        else
            log "[警告] 未找到 stage1 的 checkpoint-* 目录, stage2 从零开始训练"
        fi
    fi
    run_torchrun "${STAGE2_OUT}" "${FULL_STEPS}" "${DATA_JSON}" "${ROOT_DIR}" "${resume_ckpt}"
    log "stage2 完成. checkpoint: ${STAGE2_OUT}/final"
}

# ============================================================================ #
#  all: 完整三步流程
# ============================================================================ #
do_all() {
    do_stage1
    do_validate
    do_stage2
    log "===== 全流程完成 ====="
    log "stage1 (测试) : ${STAGE1_OUT}/final"
    log "验证结果      : ${VALIDATE_OUT}/validate_summary.json"
    log "stage2 (全量) : ${STAGE2_OUT}/final"
}

case "${STAGE}" in
    stage1)  do_stage1 ;;
    validate) do_validate ;;
    stage2)  do_stage2 ;;
    all)     do_all ;;
    *)
        echo "用法: bash train.sh [all|stage1|validate|stage2] [GPU数]"
        echo ""
        echo "  all       完整三步流程 (默认): stage1 → validate → stage2"
        echo "  stage1    1000 样本测试训练 (拷贝图片到 ${SUBSET_DIR})"
        echo "  validate  100 样本推理验证 (需先 stage1)"
        echo "  stage2    全量训练 (需先 validate PASS)"
        echo ""
        echo "环境变量: TEST_SAMPLES, TEST_STEPS, FULL_STEPS, VALIDATE_SAMPLES, DATA_JSON, SUBSET_DIR, ROOT_DIR, NPROC"
        exit 1
        ;;
esac
