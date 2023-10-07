python main.py \
    --version colab_data_aug \
    --do_train True \
    --do_eval False \
    --label_level level1 \
    --model_name_or_path roberta-base \
    --data_name pdtb2 \
    --data_path /content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv \
    --output_dir ./output_space/ \
    --log_path log.out \
    --load_ckpt_dir ./ckpt_fold \
    --label_expansion_positive 0 \
    --label_expansion_negative 0 \
    --data_augmentation True \
    --epochs 5 \
    --max_steps -1 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --eval_steps 100 \
    --log_steps 10 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --learning_rate 5e-06