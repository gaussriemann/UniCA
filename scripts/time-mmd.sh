export CUDA_VISIBLE_DEVICES=0
# Climate
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/Climate --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 100 --d_multi_modal 32 --dropout 0.5 --weight_decay 0.1 --with_future --future_with_gate --only_quantile_loss --normalized_loss --sample_output
# Energy
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/Energy --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 100 --d_multi_modal 2 --dropout 0.4 --weight_decay 1e-05 --with_gate --with_future --future_with_gate --only_quantile_loss --normalized_loss --sample_output
# Environment
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/Environment --num_batches_per_epoch 50 --max_epochs 100 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --with_gate --with_future --future_with_gate
# Public_Health
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/Public_Health --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 100 --d_multi_modal 2 --dropout 0.2 --weight_decay 0.1 --with_gate --future_with_gate --only_quantile_loss --normalized_loss --sample_output
# Security
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/Security --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 100 --d_multi_modal 2 --dropout 0.2 --weight_decay 0.1 --with_gate --future_with_gate --only_quantile_loss --normalized_loss --sample_output
# SocialGood
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/SocialGood --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 100 --d_multi_modal 4 --dropout 0.0 --weight_decay 0.001 --with_gate --only_quantile_loss --normalized_loss --sample_output
# Traffic
python main.py --model_name ts_adapter/unica --base_model chronos_bolt_base --datasets time-mmd/Traffic --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 100 --d_multi_modal 2 --dropout 0.1 --weight_decay 0.001 --with_future --only_quantile_loss --normalized_loss --sample_output
# Climate
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_gate --with_future --future_with_gate --datasets time-mmd/Climate --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 50 --d_multi_modal 16 
# Energy
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_gate --with_future  --future_with_gate --datasets time-mmd/Energy --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 50 --d_multi_modal 4 
# Environment
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_gate --with_future --future_with_gate --datasets time-mmd/Environment --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --lr 1e-5 --max_epochs 50 --d_multi_modal 16 
# Public_Health
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_g ate --with_future --future_with_gate --datasets time-mmd/Public_Health --num_batches_per_epoch 50 --max_epochs 100 --indexed_sample --split_val --gradient_clip 0.1 --batch_size 64 --d_multi_modal 1 --lr 1e-5
# Security
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_gate --with_future --future_with_gate --datasets time-mmd/Security --num_batches_per_epoch 50 --max_epochs 100 --indexed_sample --split_val --gradient_clip 0.1 --batch_size 64 --d_multi_modal 1 --lr 1e-5
# SocialGood
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_gate --with_future --future_with_gate --datasets time-mmd/SocialGood --num_batches_per_epoch 50 --max_epochs 100 --indexed_sample --split_val --gradient_clip 0.1 --batch_size 64 --d_multi_modal 1 --lr 1e-5
# Traffic
python main.py --model_name ts_adapter/unica --base_model timesfm_2_500m --with_gate --with_future --future_with_gate --datasets time-mmd/Traffic --num_batches_per_epoch 50 --max_epochs 100 --indexed_sample --split_val --gradient_clip 0.1 --batch_size 64 --d_multi_modal 1 --lr 1e-5

# Climate
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/Climate --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
# Energy
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/Energy --num_workers 0 --num_batches_per_epoch 50 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
# Environment
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/Environment --num_workers 0 --num_batches_per_epoch 50 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
# Public_Health
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/Public_Health --num_workers 0 --num_batches_per_epoch 50 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
# Security
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/Security --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
# SocialGood
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/SocialGood --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
# Traffic
python main.py --model_name chattime --with_gate --with_future --future_with_gate --datasets time-mmd/Traffic --num_workers 0 --num_batches_per_epoch 20 --indexed_sample --split_val --gradient_clip 1.0 --batch_size 32 --encoder_path chattime
