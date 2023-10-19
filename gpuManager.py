import time
import pynvml

from typing import *


class GPUManager:
    @staticmethod
    def query_gpu_memory(cuda_id:Optional[int]=None, show=True, to_mb=True):
        if cuda_id is None:
            for p in GPUManager.get_all_cuda_id():
                GPUManager.query_gpu_memory(cuda_id=p, show=show, to_mb=to_mb)
            return
        
        def norm_mem(mem):
            if to_mb:
                return f'{mem/(1024**2):.0f}MB'
            unit_lst = ['B', 'KB', 'MB', 'GB', 'TB']
            for unit in unit_lst:
                if mem < 1024:
                    return f'{mem:.2f}{unit}'
                mem /= 1024
            return f'{mem:.2f}TB'
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        if show:
            print(
                f'cuda: {cuda_id}, '
                f'free: {norm_mem(info.free)}, '
                f'used: {norm_mem(info.used)}, '
                f'total: {norm_mem(info.total)}'
            )
        return info.free, info.used, info.total

    @staticmethod
    def get_all_cuda_id():
        pynvml.nvmlInit()
        cuda_cnt = list(range(pynvml.nvmlDeviceGetCount()))
        pynvml.nvmlShutdown()
        return cuda_cnt
        
    @staticmethod
    def _get_most_free_gpu(device_range=None):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()
        max_free = -1
        free_id = -1
        for cuda_id in device_range:
            cur_free = GPUManager.query_gpu_memory(cuda_id, show=False)[0]
            if cur_free > max_free:
                max_free = cur_free
                free_id = cuda_id
        return max_free, free_id
    
    @staticmethod
    def get_free_gpu(
        target_mem_mb=8000, 
        force=False, 
        wait=True, 
        wait_gap=5, 
        show_waiting=False,
        device_range=None, 
    ):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()

        if force:
            return GPUManager._get_most_free_gpu(device_range=device_range)[1]
            
        if not wait:
            target_mem_mb *= 1024**2
            for cuda_id in device_range:
                if GPUManager.query_gpu_memory(cuda_id=cuda_id, show=False)[0] > target_mem_mb:
                    return cuda_id
            return -1
        else:
            while 1:
                device_id = GPUManager.get_free_gpu(
                    target_mem_mb=target_mem_mb,
                    force=False,
                    wait=False,
                    device_range=device_range,
                )
                if device_id != -1:
                    return device_id
                if show_waiting:
                    print('waiting cuda ...')
                time.sleep(wait_gap)
        
        
    @staticmethod
    def _occupy_one_gpu(cuda_id, target_mem_mb=8000):
        import torch
        '''
        < release by following >
        gpustat -cpu
        kill -9 <num>
        '''
        device = torch.device(f'cuda:{cuda_id}')
        used_mem = GPUManager.query_gpu_memory(cuda_id=cuda_id, show=False)[1]
        used_mem_mb = used_mem/(1024**2)
        one_gb = torch.zeros(224*1024**2)  # about 951mb
        gb_cnt = int((target_mem_mb-used_mem_mb)/1000)
        if gb_cnt < 0:
            return
        lst = [one_gb.detach().to(device) for _ in range(gb_cnt+1)]
        while 1:
            time.sleep(2**31)
            
    @staticmethod
    def wait_and_occupy_free_gpu(
        target_mem_mb=8000,
        wait_gap=5,
        show_waiting=False,
        device_range=None, 
    ):
        if not device_range:
            device_range = GPUManager.get_all_cuda_id()
        cuda_id = GPUManager.get_free_gpu(
            target_mem_mb=target_mem_mb,
            force=False,
            wait=True,
            wait_gap=wait_gap,
            show_waiting=show_waiting,
            device_range=device_range,
        )
        GPUManager._occupy_one_gpu(
            cuda_id=cuda_id,
            target_mem_mb=target_mem_mb,
        )
        

if __name__ == '__main__':
    # GPUManager._occupy_one_gpu(6)
    print(GPUManager.query_gpu_memory())
    # print(GPUManager.get_all_cuda_id())
    # free_cuda_id = GPUManager.get_free_gpu(wait=True, force=False)
    # print(free_cuda_id)
    # GPUManager.query_gpu_memory(free_cuda_id)