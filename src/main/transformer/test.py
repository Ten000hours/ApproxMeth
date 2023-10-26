import torch
import sys
module_directory = '/cis/home/zwang/yuanzhong/MPCFormer/src/main/transformer'
sys.path.append(module_directory)
import fastrsqrt_cpp


a = torch.randn(4,3)
print(a)
print(fastrsqrt_cpp.fastrsqrt(a))
