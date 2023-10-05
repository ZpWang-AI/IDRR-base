python main.py \
    -- train_or_test train+test \
    -- label_level level1 \
    -- model_name_or_path roberta-base \
    -- data_name pdtb2 \
    -- data_path ./CorpusData/PDTB-2.0/pdtb2.csv \
    -- log_path ./custom_log.log \
    -- output_dir ./ckpt \
    -- ckpt_fold ./ckpt/ckpt-best_acc \
    -- batch_size 8 \
    -- eval_steps 5 \
    -- epochs 2 \
    -- seed 2023 \
    -- warmup_ratio 0.05 \
    -- weight_decay 0.01 \
    -- learning_rate 5e-06