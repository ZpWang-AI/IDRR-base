import os

from pathlib import Path as path
from datetime import datetime


def fill_with_delimiter(s):
    return f'{"="*10} {s} {"="*(30-len(s))}'


class CustomArgs:
    def __init__(self, test_setting=False) -> None:
        self.version = 'colab'
        
        # base setting
        self.part1 = 'base setting'
        self.do_train = True
        self.do_eval = True
        self.save_ckpt = False
        self.training_iteration = 5
        self.cuda_cnt = 1
        
        # data
        self.part2 = 'data'
        self.mini_dataset = False
        self.label_level = 'level1'
        self.data_name = 'pdtb2'
        self.secondary_label_weight = 0.5  
        self.data_augmentation_secondary_label = False
        self.data_augmentation_connective_arg2 = False
        
        self.rank_data_sampler = 'shuffle'
        self.rank_balance_class = False
        self.rank_fixed_sampling = False
        self.rank_dataset_size_multiplier = 1

        self.trainset_size = -1
        self.devset_size = -1
        self.testset_size = -1
        self.rank_trainset_size = -1
        
        # file path
        self.part3 = 'file path'
        self.model_name_or_path = 'roberta-base'
        self.data_path = '/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv'
        self.cache_dir = '/content/drive/MyDrive/IDRR/plm_cache'
        self.output_dir = './output_space/'
        self.log_dir = '/content/drive/MyDrive/IDRR/log_space'
        self.load_ckpt_dir = './ckpt_fold'
        
        self.rank_order_file = './rank_order/rank_order1.json'

        # loss
        self.part4 = 'loss'
        self.loss_type = 'CELoss'
        
        self.rank_loss_type = 'ListMLELoss'
        
        # epoch, batch, step
        self.part5 = 'epoch, batch, step'
        self.max_steps = -1
        self.epochs = 5
        self.train_batch_size = 8
        self.eval_batch_size = 32
        self.eval_steps = 100
        self.log_steps = 10
        self.gradient_accumulation_steps = 1
        self.eval_per_epoch = 4

        self.rank_epochs = 2
        self.rank_train_batch_size = 8
        self.rank_eval_batch_size = 8
        self.rank_eval_steps = 800
        self.rank_log_steps = 40
        self.rank_gradient_accumulation_steps = 1
        self.rank_eval_per_epoch = 4

        self.real_batch_size = -1
        self.sample_per_eval = -1
        
        self.rank_real_batch_size = -1
        self.rank_sample_per_eval = -1
        
        # seed, lr
        self.part6 = 'seed, lr'
        self.seed = 2023
        self.warmup_ratio = 0.05
        self.weight_decay = 0.01
        self.learning_rate = 5e-6
        
        self.rank_learning_rate = 3e-5
        
        # additional details
        self.part7 = 'additional details'
        self.cuda_id = ''
        self.cur_time = ''
        self.server_name = ''
        
        for p in range(1, 8):
            attr_name = f'part{p}'
            init_attr = self.__getattribute__(attr_name)
            self.__setattr__(attr_name, fill_with_delimiter(init_attr))
        
        if test_setting:
            self.version = 'colab_test'
            self.training_iteration = 2
            
            self.mini_dataset = True
            self.data_augmentation_secondary_label = False
            self.data_augmentation_connective_arg2 = False
            self.rank_balance_class = False
            self.rank_fixed_sampling = False
            self.rank_dataset_size_multiplier = 1
            
            self.epochs = 2
            self.train_batch_size = 8
            self.eval_batch_size = 8
            self.eval_steps = 4
            self.log_steps = 4
            self.eval_per_epoch = -1
            self.gradient_accumulation_steps = 1

            self.rank_epochs = 1
            self.rank_train_batch_size = 2
            self.rank_eval_batch_size = 2
            self.rank_eval_steps = 4
            self.rank_log_steps = 4
            self.rank_eval_per_epoch = -1
            self.rank_gradient_accumulation_steps = 1
    
    def complete_path(self,
                      show_cur_time=True,
                      show_data_name=True,
                      show_label_level=True,
                      ):
        if not self.cur_time:
            self.cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # train_eval_string = '_train'*self.do_train + '_eval'*self.do_eval
            specific_fold_name = ''
            if show_cur_time:
                specific_fold_name += self.cur_time+'_'
            specific_fold_name += self.version+'_'
            if show_data_name:
                specific_fold_name += '_'+self.data_name
            if show_label_level:
                specific_fold_name += '_'+self.label_level
            self.output_dir = os.path.join(self.output_dir, specific_fold_name)
            self.log_dir = os.path.join(self.log_dir, specific_fold_name) 
    
    def prepare_gpu(self, target_mem_mb=10000):
        if not self.cuda_id:
            from gpuManager import GPUManager
            free_gpu_ids = GPUManager.get_some_free_gpus(
                gpu_cnt=self.cuda_cnt, 
                target_mem_mb=target_mem_mb,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_ids
            self.cuda_id = free_gpu_ids
            print(f'=== CUDA {free_gpu_ids} ===')
    
    def recalculate_eval_log_steps(self):
        self.real_batch_size = self.train_batch_size*self.gradient_accumulation_steps*self.cuda_cnt
        if self.eval_per_epoch > 0:
            self.eval_steps = max(1, int(
                self.trainset_size / self.eval_per_epoch / self.real_batch_size
            ))
            self.log_steps = max(1, int(
                self.trainset_size / self.eval_per_epoch / self.real_batch_size / 10
            ))
        self.sample_per_eval = self.real_batch_size*self.eval_steps
            
        self.rank_real_batch_size = self.rank_train_batch_size*self.rank_gradient_accumulation_steps*self.cuda_cnt
        if self.rank_eval_per_epoch > 0:
            self.rank_eval_steps = max(1, int(
                self.trainset_size / self.rank_eval_per_epoch / self.rank_real_batch_size
            ))
            self.rank_log_steps = max(1, int(
                self.trainset_size / self.rank_eval_per_epoch / self.rank_real_batch_size / 10
            ))
        self.rank_sample_per_eval = self.rank_real_batch_size*self.rank_eval_steps
        
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
        
        if not self.do_train and self.do_eval and not path(self.load_ckpt_dir).exists():
            raise Exception('no do_train and load_ckpt_dir does not exist')  

        if not path(self.rank_order_file).exists():
            raise Exception('rank_order_file not exists')  

    def __iter__(self):
        return iter(self.__dict__.items())
        

if __name__ == '__main__':
    import json
    
    sample_args = CustomArgs(test_setting=False)
    # print(list(sample_args))
    print(json.dumps(dict(sample_args), ensure_ascii=False, indent=2))
