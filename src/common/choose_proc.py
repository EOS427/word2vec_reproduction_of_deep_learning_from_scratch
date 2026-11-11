from .load_config import load_tool

is_use_GPU=load_tool("processor")

GPU=is_use_GPU["GPU"]

# def set_GPU(bool):
#     global GPU
#     if bool:
#         GPU=True

if GPU:
    import cupy as proc
    proc.cuda.set_allocator(proc.cuda.MemoryPool().malloc)

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')#提示框打印
    print('\033[92m' + '-' * 60 + '\033[0m\n')

else:
    print("CPU mode(numpy)")
    import numpy as proc


