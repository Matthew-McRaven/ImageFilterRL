import functools

import torch
from torch import nn
class FlattenCatKernel(nn.Module):
    def __init__(self, lhs, rhs):
        super(FlattenCatKernel, self).__init__()
        self.lhs, self.rhs = lhs, rhs
        self.lhs_size = functools.reduce(lambda x,y: x*y, lhs.output_dimension, 1)
        self.rhs_size = functools.reduce(lambda x,y: x*y, rhs.output_dimension, 1)
        self.input_dimensions = None
        self.output_dimensions = (self.lhs_size+self.rhs_size,)
    def forward(self, inputs):
        lhs, rhs = inputs
        lhs = self.lhs(lhs).view(-1, self.lhs_size)
        rhs = self.rhs(rhs).view(-1, self.rhs_size)
        stacked = torch.cat([lhs, rhs], dim=1)
        #print(lhs.shape, rhs.shape, stacked.shape)
        return stacked