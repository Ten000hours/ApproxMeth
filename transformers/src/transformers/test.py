import torch
import sys
module_directory = '/cis/home/zwang/yuanzhong/MPCFormer/transformers/src/transformers'
sys.path.append(module_directory)
import fastrsqrt_cpp


a = torch.tensor([[ 0.0892899,  4.62455, 5.44279],
        [0.1678,  1.6609,  0.1513]])
print(a)
print("correct: ", 1/torch.sqrt(a))
print(fastrsqrt_cpp.fastrsqrt2PC(a))
