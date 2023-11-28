python main.py \
    --version colab_baseline \
    --do_train True \
    --do_eval True \
    --training_iteration 3 \
    --save_ckpt True \
    --label_level level1 \
    --model_name_or_path roberta-base \
    --data_name pdtb2 \
    --data_path /content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv \
    --cache_dir /content/drive/MyDrive/IDRR/plm_cache \
    --output_dir ./output_space/ \
    --log_dir /content/drive/MyDrive/IDRR/log_space \
    --load_ckpt_dir ./ckpt_fold \
    --data_augmentation_connective_arg2 False \
    --epochs 5 \
    --max_steps -1 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --eval_steps 100 \
    --log_steps 10 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --learning_rate 5e-06