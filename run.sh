OUTPUT_DIR=work_dirs/jitcot_lm_16
DATA_PATH=./imagenet

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
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
# main_jit.py \
# --model RAEJiT-LM/16 \
# --latent_model offline_models/dinov3_vit_large_patch16 \
# --D_mean -1.2 --D_std 1.0 \
# --P_mean -0.4 --P_std 0.8 \
# --batch_size 64 --blr 5e-5 \
# --epochs 200 --warmup_epochs 5 \
# --gen_bsz 256 --rec_bsz 256 --num_images 50000 \
# --cfg 1.0 --cfg_dino 1.0 \
# --interval_min 0.0 --interval_max 1.0 \
# --dino_weight 0.333 --choose_dino_p 0.4 \
# --sample_mode dino_first_cascaded_noised \
# --dh_depth 2 --dh_hidden_size 1024 \
# --output_dir ${OUTPUT_DIR} \
# --resume ${OUTPUT_DIR} \
# --data_path ${DATA_PATH} \
# --online_eval

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
main_jit.py \
--model JiTCoT-LM/16 \
--D_mean -1.2 --D_std 1.0 \
--P_mean -0.4 --P_std 0.8 \
--batch_size 128 --blr 5e-5 \
--epochs 200 --warmup_epochs 5 \
--gen_bsz 256 --num_images 50000 \
--cfg 1.0 --cfg_dino 1.0 \
--interval_min 0.0 --interval_max 1.0 \
--dino_weight 0.333 --choose_dino_p 0.4 \
--sample_mode dino_first_cascaded_noised \
--dh_depth 2 --dh_hidden_size 1024 \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR} \
--data_path ${DATA_PATH} \
--online_eval

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
main_jit.py \
--model JiTRepa-L/16 \
--P_mean -0.4 --P_std 0.8 \
--batch_size 64 --blr 5e-5 \
--epochs 200 --warmup_epochs 5 \
--gen_bsz 256 --num_images 50000 \
--cfg 1.0 \
--interval_min 0.0 --interval_max 1.0 \
--num_workers 4 --prefetch_factor 2 \
--dino_weight 1.0 \
--output_dir "$OUTPUT_DIR" \
--resume "$OUTPUT_DIR" \
--data_path "$DATA_PATH" \
--online_eval