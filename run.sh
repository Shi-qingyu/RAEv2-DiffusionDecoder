OUTPUT_DIR=work_dirs/jitrepa_b_16
DATA_PATH=./imagenet

torchrun --nproc_per_node=8 --standalone \
main_jit.py \
--model RAEJiT-BM/16 \
--latent_model offline_models/dinov3_vit_large_patch16 \
--D_mean -1.2 --D_std 1.0 \
--P_mean -0.4 --P_std 0.8 \
--batch_size 128 --blr 5e-5 \
--epochs 200 --warmup_epochs 5 \
--gen_bsz 256 --num_images 50000 \
--cfg 1.0 --cfg_dino 1.0 \
--interval_min 0.0 --interval_max 1.0 \
--dino_weight 0.333 --choose_dino_p 0.4 \
--sample_mode pixel_only \
--dh_depth 2 --dh_hidden_size 768 \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR} \
--data_path ${DATA_PATH} \
--online_eval

# torchrun --nproc_per_node=8 --standalone \
# main_jit.py \
# --model JiTCoT-BM/16 \
# --D_mean -1.2 --D_std 1.0 \
# --P_mean -0.4 --P_std 0.8 \
# --batch_size 128 --blr 5e-5 \
# --epochs 200 --warmup_epochs 5 \
# --gen_bsz 256 --num_images 50000 \
# --cfg 1.0 --cfg_dino 1.0 \
# --interval_min 0.0 --interval_max 1.0 \
# --dino_weight 0.333 --choose_dino_p 0.4 \
# --sample_mode dino_first_cascaded_noised \
# --dh_depth 2 --dh_hidden_size 768 \
# --output_dir ${OUTPUT_DIR} \
# --resume ${OUTPUT_DIR} \
# --data_path ${DATA_PATH} \
# --online_eval

# torchrun --nproc_per_node=8 --standalone \
# main_jit.py \
# --model JiTRepa-B/16 \
# --P_mean -0.4 --P_std 0.8 \
# --batch_size 128 --blr 5e-5 \
# --epochs 200 --warmup_epochs 5 \
# --gen_bsz 256 --num_images 50000 \
# --cfg 1.0 \
# --interval_min 0.0 --interval_max 1.0 \
# --num_workers 4 --prefetch_factor 2 \
# --dino_weight 0.5 \
# --output_dir "$OUTPUT_DIR" \
# --resume "$OUTPUT_DIR" \
# --data_path "$DATA_PATH" \
# --online_eval