import os
import sys
from pathlib import Path as path

sys.path.append(path(__file__).parent)

from arguments import CustomArgs
from run_server import server_base_args, CODE_SPACE


def server_experiment_args(args=None):
    if args is None:
        args = server_base_args(test_setting=False)
    
    args.secondary_label_weight = 0.5
    args.data_augmentation_secondary_label = False
    args.data_augmentation_connective_arg2 = False
    
    args.cuda_cnt = 2
    args.warmup_ratio = 0.05
    args.epochs = 10
    args.train_batch_size = 32
    args.eval_batch_size = 32
    args.eval_steps = 100
    args.log_steps = 10
    args.gradient_accumulation_steps = 1
    args.eval_per_epoch = 4
    
    args.weight_decay = 0.01
    args.learning_rate = 3e-5
    
    args.version = 'base'
    return args
    
    
if __name__ == '__main__':
    os.chdir(CODE_SPACE)
    
    def experiment_multi_times():
        cuda_cnt = 2  # === TODO: prepare gpu ===
        cuda_id = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=cuda_cnt) 
        from main import main
         
        for epoch in [5,10,20,30]:
            for mu in [5,15,30,45]:
                for batch_size in [32]:
                    todo_args = server_experiment_args()

                    # === TODO: prepare args ===
                    todo_args.version = f'epoch{epoch}_lr{mu}mu_bs{batch_size}^2'
                    todo_args.epochs = epoch
                    todo_args.learning_rate = float(f'{mu}e-6') 
                    todo_args.train_batch_size = batch_size             
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