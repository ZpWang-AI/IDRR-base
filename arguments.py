import os
import json

from pathlib import Path as path
from datetime import datetime
from typing import Any


def fill_with_delimiter(s):
    return f'{"="*10} {s} {"="*(30-len(s))}'


class StageArgs(dict):
    def __init__(self,
                 stage_name,
                 epochs, 
                 train_batch_size, eval_batch_size,
                 eval_steps, log_steps,
                 gradient_accumulation_steps,
                 eval_per_epoch,
                 warmup_ratio,
                 weight_decay,
                 learning_rate,
                 ) -> None:
        self.stage_name = stage_name
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps
        self.log_steps = log_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_per_epoch = eval_per_epoch
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.real_batch_size = -1
        self.sample_per_eval = -1
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value
        super().__init__(self.__dict__)
    
    def recalculate_eval_log_steps(self, cuda_cnt, trainset_size):
        self.real_batch_size = self.train_batch_size*self.gradient_accumulation_steps*cuda_cnt
        if self.eval_per_epoch > 0:
            self.eval_steps = max(1, int(
                trainset_size / self.eval_per_epoch / self.real_batch_size
            ))
            self.log_steps = max(1, int(
                trainset_size / self.eval_per_epoch / self.real_batch_size / 10
            ))
        self.sample_per_eval = self.real_batch_size*self.eval_steps
    

class CustomArgs:
    def __init__(self, test_setting=False) -> None:
        self.version = 'base'
        
        # ========== base setting ==================
        self.part1 = 'base setting'
        # self.do_train = True
        # self.do_eval = True
        self.save_ckpt = False
        self.seed = 2023
        
        self.training_iteration = 5
        self.cuda_cnt = 1
        
        # ========== data ==========================
        self.part2 = 'data'
        self.mini_dataset = False
        self.label_level = 'level1'
        self.data_name = 'pdtb2'
        self.secondary_label_weight = 0.
        self.data_augmentation_secondary_label = False
        self.data_augmentation_connective_arg2 = False
        
        self.rank_data_sampler = 'shuffle'
        self.rank_balance_class = False
        self.rank_balance_batch = False
        self.rank_fixed_sampling = False
        self.rank_dataset_size_multiplier = 1

        self.trainset_size = -1
        self.devset_size = -1
        self.testset_size = -1
        self.rank_trainset_size = -1
        
        # ========== file path =====================
        self.part3 = 'file path'
        self.model_name_or_path = 'roberta-base'
        self.data_path = '/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv'
        self.cache_dir = '/content/drive/MyDrive/IDRR/plm_cache'
        self.output_dir = './output_space/'
        self.log_dir = '/content/drive/MyDrive/IDRR/log_space'
        # self.load_ckpt_dir = './ckpt_fold'
        
        self.rank_order_file = './rank_order/rank_order1.json'

        # ========== loss ==========================
        self.part4 = 'loss'
        self.loss_type = 'CELoss'
        
        self.rank_loss_type = 'ListMLELoss'
        
        # ========== stage =========================
        self.part5 = 'stage'
        self.training_stages = [
            StageArgs(
                stage_name='rank',
                epochs=2,
                train_batch_size=1,
                eval_batch_size=1,
                eval_steps=800,
                log_steps=40,
                gradient_accumulation_steps=2,
                eval_per_epoch=15,
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
                eval_per_epoch=4,
                warmup_ratio=0.05,
                weight_decay=0.01,
                learning_rate=5e-6,
            ),
        ]
        
        # ========== additional details ============
        self.part7 = 'additional details'
        self.cuda_id = ''
        self.cur_time = ''
        self.server_name = ''
        
        for p in range(10):
            attr_name = f'part{p}'
            if hasattr(self, attr_name):
                init_attr = self.__getattribute__(attr_name)
                self.__setattr__(attr_name, fill_with_delimiter(init_attr))
        
        if test_setting:
            self.version = 'test'
            self.training_iteration = 2
            
            self.mini_dataset = True
            self.data_augmentation_secondary_label = False
            self.data_augmentation_connective_arg2 = False
            self.rank_balance_class = False
            self.rank_fixed_sampling = False
            self.rank_dataset_size_multiplier = 1
            
            self.training_stages = [
                StageArgs(
                    stage_name='rank',
                    epochs=1,
                    train_batch_size=2,
                    eval_batch_size=2,
                    eval_steps=4,
                    log_steps=4,
                    gradient_accumulation_steps=1,
                    eval_per_epoch=-15,
                    warmup_ratio=0.05,
                    weight_decay=0.01,
                    learning_rate=5e-6,
                ),
                StageArgs(
                    stage_name='ft',
                    epochs=2,
                    train_batch_size=8,
                    eval_batch_size=8,
                    eval_steps=4,
                    log_steps=4,
                    gradient_accumulation_steps=1,
                    eval_per_epoch=-4,
                    warmup_ratio=0.05,
                    weight_decay=0.01,
                    learning_rate=5e-6,
                ),
            ]
    
    def complete_path(self,
                      show_cur_time=True,
                      show_server_name = True,
                      show_data_name=True,
                      show_label_level=True,
                      ):
        if not self.cur_time:
            self.cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            specific_fold_name = []
            if show_cur_time:
                specific_fold_name.append(self.cur_time)
            if show_server_name:
                specific_fold_name.append(self.server_name)
            specific_fold_name.append(self.version)
            if show_data_name:
                specific_fold_name.append(self.data_name)
            if show_label_level:
                specific_fold_name.append(self.label_level)
            specific_fold_name = '.'.join(map(str, specific_fold_name))
            self.output_dir = os.path.join(self.output_dir, specific_fold_name)
            self.log_dir = os.path.join(self.log_dir, specific_fold_name) 
    
    def estimate_cuda_memory(self):
        max_batch_size = 1
        for stage in self.training_stages:
            if 'rank' in stage.stage_name:
                if self.label_level == 'level1':
                    max_batch_size = max(max_batch_size, stage.train_batch_size*4)
                else:
                    raise ValueError('wrong label_level when estimating cuda memory')
            elif 'ft' in stage.stage_name:
                max_batch_size = max(max_batch_size, stage.train_batch_size)
        if max_batch_size > 16:
            return 10500
        elif max_batch_size > 8:
            return 7000
        else:
            return 5000
    
    def prepare_gpu(self, target_mem_mb=10000, gpu_cnt=None):
        if not self.cuda_id:
            if target_mem_mb < 0:
                target_mem_mb = self.estimate_cuda_memory()
            if gpu_cnt is None:
                gpu_cnt = self.cuda_cnt

            from gpuManager import GPUManager
            free_gpu_ids = GPUManager.get_some_free_gpus(
                gpu_cnt=gpu_cnt, 
                target_mem_mb=target_mem_mb,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_ids
            self.cuda_id = free_gpu_ids
            print(f'=== CUDA {free_gpu_ids} ===')
        return self.cuda_id
    
    def recalculate_eval_log_steps(self):
        for stage in self.training_stages:
            stage.recalculate_eval_log_steps(self.cuda_cnt, self.trainset_size)
        
    def check_path(self):
        # self.data_path
        # self.cache_dir
        # self.output_dir
        # self.log_dir
        # self.load_ckpt_dir
        
        assert path(self.data_path).exists(), 'wrong data path'
        
        if str(self.cache_dir) in ['None', '']:
            self.cache_dir = None
        else:
            path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
        path(self.output_dir).mkdir(parents=True, exist_ok=True)
        path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # if not self.do_train and self.do_eval and not path(self.load_ckpt_dir).exists():
        #     raise Exception('no do_train and load_ckpt_dir does not exist')  

        if not path(self.rank_order_file).exists():
            raise Exception('rank_order_file not exists')  
    
    def __iter__(self):
        return iter(self.__dict__.items())
    
    def __str__(self) -> str:
        return json.dumps(dict(self), ensure_ascii=False, indent=2)
        

if __name__ == '__main__':
    sample_args = CustomArgs(test_setting=False)
    sample_args.cuda_cnt = 1
    sample_args.trainset_size = 1234
    sample_args.recalculate_eval_log_steps()
    print(sample_args)