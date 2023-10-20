python main.py \
    --version colab \
    --mini_dataset False \
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
    --loss_type CELoss \
    --rank_loss_type ListMLELoss \
    --data_augmentation False \
    --rank_order_file ./rank_order/rank_order1.json \
    --epochs 5 \
    --max_steps -1 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --eval_steps 100 \
    --log_steps 10 \
    --gradient_accumulation_steps 1 \
    --rank_epochs 2 \
    --rank_eval_steps 400 \
    --rank_log_steps 40 \
    --rank_gradient_accumulation_steps 2 \
    --seed 2023 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --learning_rate 5e-06