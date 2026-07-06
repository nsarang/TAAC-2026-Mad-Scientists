#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH}"
pip install -q -r "${SCRIPT_DIR}/extra.txt"

# GPU detection
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES// /}" ]]; then
    NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
elif command -v nvidia-smi &> /dev/null; then
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    NGPUS=1
fi
echo "Using ${NGPUS} GPU(s) (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')"

shopt -s nullglob
CONFIGS=(${CONFIGS:-${SCRIPT_DIR}/config*.yaml})

for CONFIG in "${CONFIGS[@]}"; do
    LAUNCH_ARGS=(
        "${SCRIPT_DIR}/execute.py" train
        --config "${CONFIG}"
        "data.dataset_path=${TRAIN_DATA_PATH}"
        "data.schema_path=${TRAIN_DATA_PATH}/schema.json"
        "train.checkpoint.dir=${TRAIN_CKPT_PATH}"
        "train.output_dir=${TRAIN_LOG_PATH}"
        "diagnostics.log_dir=${TRAIN_TF_EVENTS_PATH}"
        "$@"
    )
    if [[ "${NGPUS}" -gt 1 ]]; then
        echo "Launching DDP with torchrun --nproc_per_node=${NGPUS}"
        torchrun --standalone --nproc_per_node="${NGPUS}" "${LAUNCH_ARGS[@]}"
    else
        python3 -u "${LAUNCH_ARGS[@]}"
    fi
done
