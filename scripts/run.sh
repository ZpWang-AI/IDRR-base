python main.py \
    --train_or_test train+test \
    --label_level level1 \
    --model_name_or_path roberta-base \
    --data_name pdtb2 \
    --data_path /content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv \
    --log_path ./custom_log.log \
    --output_dir ./ckpt/ \
    --ckpt_fold ./ckpt/ckpt-best_acc \
    --epochs 4 \
    --batch_size 8 \
    --eval_steps 5 \
    --log_steps 5 \
    --seed 2023 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --learning_rate 5e-06