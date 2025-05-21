export CUDA_VISIBLE_DEVICES=0

python main.py --model_name ts_adapter/unica \
 --base_model chronos_bolt_base \
 --with_gate --datasets epf_sub \
 --num_workers 4 --num_batches_per_epoch 50 \
 --max_epochs 100 --indexed_sample --split_val \
 --gradient_clip 0.1 --batch_size 128

CUDA_VISIBLE_DEVICES=1 python main.py --model_name ts_adapter/unica \
 --base_model timesfm_2_500m \
 --with_gate --datasets epf_sub --num_workers 4 \
 --num_batches_per_epoch 50 --max_epochs 100 \
 --indexed_sample --split_val --gradient_clip 0.1 --batch_size 64 \
 --only_quantile_loss

