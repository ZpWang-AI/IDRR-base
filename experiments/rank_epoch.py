import os
import sys
from pathlib import Path as path

sys.path.append(path(__file__).parent)

from arguments import CustomArgs, StageArgs
from run_server import server_base_args, CODE_SPACE


def server_experiment_args(args=None):
    if args is None:
        args = server_base_args(test_setting=False)
    
    args.cuda_cnt = 2
    args.secondary_label_weight = 0.5
    args.data_augmentation_secondary_label = False
    args.data_augmentation_connective_arg2 = False
    args.rank_data_sampler = 'random'
    args.rank_dataset_size_multiplier = 1
    
    args.training_stages = [
        StageArgs(
            stage_name='rank',
            epochs=2,
            train_batch_size=1,
            eval_batch_size=1,
            eval_steps=800,
            log_steps=40,
            gradient_accumulation_steps=2,
            eval_per_epoch=-15,
            warmup_ratio=0.05,
            weight_decay=0.01,
            learning_rate=5e-6,
        ),
        StageArgs(
            stage_name='ft',
            epochs=5,
            train_batch_size=8,
            eval_batch_size=32,
            eval_steps=100,
            log_steps=10,
            gradient_accumulation_steps=1,
            eval_per_epoch=-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            learning_rate=5e-6,
        ),
    ]
    
    args.version = 'v1-arg'
    return args
    
    
if __name__ == '__main__':
    os.chdir(CODE_SPACE)
    
    def experiment_multi_times():
        cuda_cnt = 2  # === TODO: prepare gpu ===
        cuda_id = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=cuda_cnt) 
        from main import main
        
        batch_size = 4
        for epoch in [1,2,3,4,5]:
            todo_args = server_experiment_args()

            # === TODO: prepare args ===
            todo_args.version = f'repoch{epoch}_bs{batch_size}^{cuda_cnt}'
            todo_args.training_stages[0].epochs = epoch
            todo_args.training_stages[0].learning_rate = 3e-5
            todo_args.training_stages[0].train_batch_size = batch_size             
            # === TODO: prepare args ===
            
            todo_args.cuda_id = cuda_id
            todo_args.cuda_cnt = cuda_cnt
            todo_args.complete_path(
                show_cur_time=True,
                show_server_name=False,
                show_data_name=False,
                show_label_level=False,
            )
            
            main(todo_args)
                    
    experiment_multi_times()