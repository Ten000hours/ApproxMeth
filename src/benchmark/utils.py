import math
import torch
from torch import Tensor, nn

import crypten
import crypten.communicator as comm
import crypten.nn as cnn

def encrypt_tensor(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    assert world_size  == 2
    
    # assumes party 1 is the actual data provider
    src_id = 1

    if rank == src_id:
        input_upd = input.cuda()
    else:
        input_upd = torch.empty(input.size()).cuda()
    private_input = crypten.cryptensor(input_upd, src=src_id)
#    print(private_input)
    return private_input

def encrypt_model(model, modelFunc, config, dummy_input):
    rank = comm.get().get_rank()
    
    # assumes party 0 is the actual model provider
    if rank == 0:
        model_upd = model.cuda()
    else:
        if isinstance(config, tuple):
            model_upd = modelFunc(config[0], config[1]).cuda()
        else:
            model_upd = modelFunc(config).cuda()

    private_model = model_upd.encrypt(src=0)
    return private_model


class softmax_2RELU(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = cnn.ReLU()
        self.dim = dim

    def forward(self, x):
        func_x = self.func(x)
        return func_x / func_x.sum(keepdim=True, dim=self.dim)

class softmax_2QUAD(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def forward(self, x):
        a, b, c, d = x.size()
        #quad = x#self.norm(x)
        quad = (x+5) * (x+5)
        return quad / quad.sum(dim=self.dim, keepdims=True)

class activation_newGeLU(cnn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.three = torch.tensor([3.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.pi_const = torch.tensor([math.sqrt(2/math.pi)]).item()
        self.pow = cnn.Pow()
        self.tanh = cnn.Hardtanh()

    def forward(self, x):
        return self.half * x * (self.one + self.tanh(self.pi_const * (x + self.constant * self.pow((x, self.three)))))

class TrainLearnableAlphaGeLU1(cnn.Module):
    def __init__(self, word_length):
        super(TrainLearnableAlphaGeLU1, self).__init__()
        self.alphas = nn.Parameter(torch.full((1, word_length), 0.000001), requires_grad=False)
        self.mu = nn.Parameter(torch.full((1, word_length), 0.363), requires_grad=False)

    def forward(self, x):
        # print("alphas: ",1-self.alphas.expand_as(x))
        # out = F.relu(x) * self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * x 
        # out = ((1+self.alphas.expand_as(x)) * x +  torch.sqrt(self.mu.expand_as(x) * (1-self.alphas.expand_as(x) * x)))/2
        return ((1+self.alphas)*x+(torch.square(x-self.alphas*x)+torch.square(self.mu))*(1/torch.sqrt(torch.square(x-self.alphas*x)+torch.square(self.mu))))/2
        # return out
        # return ((1)*x+torch.square(x)+torch.square(0.4)*ctypes_isqrt(torch.square(x)+torch.square(0.4)))/2
class InferLearnableAlphaGeLU1(cnn.Module):
    def __init__(self, word_length):
        super(InferLearnableAlphaGeLU1, self).__init__()
        self.alphas = torch.tensor(0.000001).item()
        self.mu =torch.tensor(0.363).item()
        self.mut =torch.tensor(0.363)
        self.two = torch.tensor([2.0]).item()
        self.pow = cnn.Pow()
        self.inv_sqrt = cnn.inv_Sqrt()

    def forward(self, x):
        # print("alphas: ",1-self.alphas.expand_as(x))
        # out = F.relu(x) * self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * x 
        # out = ((1+self.alphas.expand_as(x)) * x +  torch.sqrt(self.mu.expand_as(x) * (1-self.alphas.expand_as(x) * x)))/2
        # x = x.to("cpu")
        # return (((1 + self.alphas) * x) + fastrsqrt_cpp.fastrsqrt(torch.square(x - self.alphas * x) + torch.square(self.mu))) / 2
        out = ((1-self.alphas) * self.pow((x,self.two)))+ torch.square(self.mut)
        return (((1 + self.alphas)* x) + (out)* self.inv_sqrt(out)) / 2
        # return out
        # return ((1)*x+torch.square(x)+torch.square(0.4)*ctypes_isqrt(torch.square(x)+torch.square(0.4)))/2


class activation_quad(cnn.Module):
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef
        #return x*x

