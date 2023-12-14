import torch
from torch import nn
import os

import pickle
import re
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import partial

import matplotlib.pyplot as plt

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}



  

#############

from torch_geometric.data import Batch, Data

import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList, Linear

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx,glob=False):
        
        super(ConcatSquashLinear, self).__init__()
        self.glob=glob
        
        self._layer = Linear(dim_in+dim_ctx, dim_out)
        if self.glob:
            self._layer_f = Linear(dim_out*2, dim_out)
        
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

#     def forward(self, ctx, x):
    def forward(self, ctx, x, m):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(torch.cat([x,ctx.expand(x.shape[0],x.shape[1],ctx.shape[-1])],-1))     
        
        
        #ret = ret * gate + bias
        
        if self.glob:
#             ret = torch.cat([ret, ret.mean(-2,keepdim=True).expand(*ret.shape)],-1)
            ret = torch.cat([ret, (ret*m).sum(-2,keepdim=True).expand(*ret.shape)],-1)# mascherare per maschera nodi
            ret = self._layer_f(ret)
    
        return ret*m
    

class PointwiseNet(Module):

    def __init__(self, marginal_prob_std, point_dim=7):
        print("V1")
        super().__init__()
        self.act = F.leaky_relu
        self.residual = False
        self.marginal_prob_std = marginal_prob_std
    
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, 3),
            ConcatSquashLinear(128, 256, 3, True),
            ConcatSquashLinear(256, 512, 3),
            ConcatSquashLinear(512, 256, 3, True),
            ConcatSquashLinear(256, 128, 3),
            ConcatSquashLinear(128, point_dim, 3)
        ])
        

    def forward(self, x,m, beta):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            labels:   point labels (B, N, 1).
        """
        batch_size = x.size(0)
        
#         print('X shape {}'.format(x.shape))
#         print('beta {}'.format(beta))
#         print('beta size{}'.format(beta.shape))
#         print('mask {}'.format(m))
#         print('beta view {}'.format(beta.view(batch_size, 1, 1)))
        
        
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        
#         x = x.transpose(1,2).contiguous()# when input is dxn uncomment
        
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
#         print('time size {}'.format(time_emb.shape))
#         print('time {}'.format(time_emb))
        
        out = x
        
        for i, layer in enumerate(self.layers):
            
            out = layer(x=out, m=m, ctx=time_emb)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return (x + out).contiguous() / self.marginal_prob_std(beta)
        else:
            return out.contiguous() / self.marginal_prob_std(beta)
    
ScoreNet = PointwiseNet


#@title Define the loss function (double click to expand or collapse)

#def loss_fn(model, x,  marginal_prob_std, eps=1e-5):
def loss_fn(model, x, m,  marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
#   print('LOSS')
  
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  #z = torch.randn_like(x)
  z = torch.randn_like(x)*m
#   print('Z shape <<<{}'.format(z.shape))
#   print('Z   <<<{}'.format(z))
    
#   print('x shape <<<{}'.format(x.shape))
#   print('x   <<<{}'.format(x))
    
  std = marginal_prob_std(random_t)
#   print('std shape <<<{}'.format(std.shape))
#   print('std   <<<{}'.format(std))

  perturbed_x = x + z * std[:, None, None]
  perturbed_x = perturbed_x.float()
#   print('perturbed_x shape <<<{}'.format(perturbed_x.shape))
#   print('perturbed_x   <<<{}'.format(perturbed_x))
 
  #score = model(perturbed_x, random_t)
  score = model(perturbed_x, m, random_t)
 
  #loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1)))
  loss = torch.mean(torch.sum( ((score * std[:, None, None] + z)*m)**2, dim=(1)))
  return loss



class PointwiseNetEIGVA(Module):

    def __init__(self, marginal_prob_std, point_dim=7):
        print("EIGVA")
        super().__init__()
        self.act = F.leaky_relu
        self.residual = False
        self.marginal_prob_std = marginal_prob_std
               
        self.layers = ModuleList([
                nn.Linear(point_dim+3, 32),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                            nn.Linear(128, 128),
                            nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
                nn.Linear(32, point_dim)
                ])
    
    
    def forward(self, x,m, beta):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            labels:   point labels (B, N, 1).
        """
        batch_size = x.size(0) 
       
        
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
#         x = x.transpose(1,2).contiguous()# when input is dxn uncomment
        
        
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)

#         print('X: ',x.shape)
#         print('T: ',time_emb.shape)
        out = x
        out = torch.cat([out, time_emb.expand(batch_size,out.shape[1],3)],-1)
        
        for i, layer in enumerate(self.layers):

            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return (x + out).contiguous() / self.marginal_prob_std(beta)
        else:
            return out.contiguous() / self.marginal_prob_std(beta)
        
ScoreNetEIGVA = PointwiseNetEIGVA