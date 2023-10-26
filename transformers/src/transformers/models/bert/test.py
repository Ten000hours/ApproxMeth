import torch
import sys
module_directory = '/cis/home/zwang/yuanzhong/MPCFormer/transformers/src/transformers/models/bert'
sys.path.append(module_directory)
import fastrsqrtcpp


a = torch.tensor([[ 0.3782,  11.126356, 7.44279],
        [0.1678,  1.6609,  0.1513]])
print(a)
print("correct: ", torch.exp(a))
print(fastrsqrtcpp.fastrexp2PC(a))
