#!/usr/bin/env bash
set -euo pipefail

# Evaluate gFID for a non-EMA RAEJiT-LM/16 checkpoint with CFG=1.0.
CKPT_DIR=${CKPT_DIR:-/work_dirs/raejit_lm_16_joint}
OUTPUT_DIR=${OUTPUT_DIR:-${CKPT_DIR}/eval_noema_cfg1_gfid}
DATA_PATH=${DATA_PATH:-./imagenet}
LATENT_MODEL=${LATENT_MODEL:-offline_models/dinov3_vit_large_patch16}
GEN_BSZ=${GEN_BSZ:-256}
NUM_IMAGES=${NUM_IMAGES:-50000}

if [[ -n "${ARNOLD_WORKER_0_PORT:-}" ]]; then
  IFS="," read -r -a ARNOLD_PORTS <<< "${ARNOLD_WORKER_0_PORT}"
  DEFAULT_MASTER_PORT="${ARNOLD_PORTS[0]}"
else
  DEFAULT_MASTER_PORT=29500
fi

MASTER_ADDR=${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${DEFAULT_MASTER_PORT}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}
NNODES=${NNODES:-${ARNOLD_WORKER_NUM:-1}}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_NUM_THREADS
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${CKPT_DIR}/checkpoint-last.pth" ]]; then
  echo "Missing checkpoint: ${CKPT_DIR}/checkpoint-last.pth" >&2
  exit 1
fi

if [[ ! -d "${DATA_PATH}/train" || ! -d "${DATA_PATH}/val" ]]; then
  echo "Missing ImageNet folders: expected ${DATA_PATH}/train and ${DATA_PATH}/val" >&2
  exit 1
fi

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  main_jit.py \
  --model RAEJiT-LM/16 \
  --latent_model "${LATENT_MODEL}" \
  --D_mean -1.2 --D_std 1.0 \
  --P_mean -0.4 --P_std 0.8 \
  --batch_size 64 --blr 5e-5 \
  --gen_bsz "${GEN_BSZ}" --num_images "${NUM_IMAGES}" \
  --cfg 1.0 --cfg_dino 1.0 \
  --interval_min 0.0 --interval_max 1.0 \
  --interval_min_dino 0.0 --interval_max_dino 1.0 \
  --dino_weight 0.333 --choose_dino_p 0.4 \
  --sample_mode dino_first_cascaded_noised \
  --dh_depth 2 --dh_hidden_size 1024 \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${CKPT_DIR}" \
  --data_path "${DATA_PATH}" \
  --evaluate_gen --num_sampling_steps 50 \
  --sampling_method heun \
  --guidance_method cfg \
  --generation_ema none
