# coding=utf-8
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import Conv2d

# This class use some of the code from
# https://github.com/pytorch/examples/blob/master/vae/main.py

class VarConv2d(Module):
    """ Variational convolution 2D module for PyTorch.
    
    This class is based on the Variational Auto-Encoder; more details are
    given in the code comments.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, is_variational=True):
        """ Constructor of the class, following the same syntax as the
            PyTorch Conv2D class, with an optional is_variational
            parameter.
        
            Note that by default it is variational. You can switch on
            and off this behavior by modifying the value of the attribute
            is_variational.
        """
        
        super(VarConv2d, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.is_variational = is_variational
        
        self.mu_layer      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.logvar_layer  = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        

    def forward(self, x):
        """ Forward step of the variational convolution
            
            Parameters
            ----------
                x: PyTorch tensor / batch
                    Data to process
            
            Returns
            -------
                (output, variational loss)
                    A pair composed of the output tensor, and the variational
                    loss. If the layer is not being trained or if its
                    is_variational attribute is false, then the variational
                    loss is 0.
        """
        mu = self.mu_layer(x)
        if self.training and self.is_variational:
            logvar = self.logvar_layer(x)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            mu = eps.mul(std).add_(mu)
            varloss = self.kld(mu, logvar)
        else:
            varloss = 0
        return (mu, varloss)
    
    def kld(self, mu, logvar):
        """ Computes the Kullback-Leibler Divergence, which is used as
        an extra-loss when training the variational layer """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def extra_repr(self):
        """ Returns a string description of the instance """
        return 'out_channels={out_channels}, kernel_size={kernel_size}, ' \
            'padding={padding}, stride={stride}, bias={bias}, is_variational={is_variational}'.format(
                **self.__dict__)
