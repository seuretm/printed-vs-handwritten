import torch
from torch.nn.modules import Module

class DontCareLoss(Module):
    def __init__(self, dimensionality):
        super(DontCareLoss, self).__init__()
        
    def forward(self, input, target, dont_care):
        loss = 0
        for n in range(input.size(0)):
            t  = target[n]
            dc = dont_care[n]
            for i in range(input[n].size(0)):
                if i==t:
                    loss += (1 - input[n][i])**2
                elif not i in dc:
                    loss += input[n][i]**2
        return loss
