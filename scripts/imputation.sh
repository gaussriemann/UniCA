python models.train_unica_imputation.py \
  --model-path /path/to/MOMENT-1-large \
  --dataset data/datasets/imputation/ETTh1.csv \
  --output-checkpoint checkpoints/unica_etth1.pt \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --stride 8
