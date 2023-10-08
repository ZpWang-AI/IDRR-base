import json
import logging

from pathlib import Path as path


# class CustomLogger(logging.Logger):
class CustomLogger:
    def __init__(self, log_dir='./log_space', logger_name='custom_logger', print_output=False) -> None:
        self.log_dir = path(log_dir)
        
        # 创建一个logger
        self.logger = logging.getLogger(logger_name)

        # 设置全局级别为DEBUG
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(self.log_dir/'log.out')
        ch = logging.StreamHandler()

        # 定义handler的输出格式
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        if print_output:
            self.logger.addHandler(ch)
    
    def info(self, *args):
        self.logger.info(' '.join(map(str, args)))
        
    def log_json(self, content, log_file, log_info=False):
        content_string = json.dumps(content, ensure_ascii=False, indent=2)
        if log_info:
            self.logger.info('\n'+content_string)
        
        log_file = self.log_dir/log_file
        with open(log_file, 'w', encoding='utf8')as f:
            f.write(content_string)
            
    def log_jsonl(self, content, log_file, log_info=False):
        if log_info:
            self.logger.info('\n'+json.dumps(content, ensure_ascii=False, indent=2))
        
        log_file = self.log_dir/log_file
        with open(log_file, 'a', encoding='utf8')as f:
            json.dump(content, f, ensure_ascii=False)
            f.write('\n')
            

if __name__ == '__main__':
    sample_logger = CustomLogger(print_output=True)
    sample_logger.info('123', {'1231':1231})