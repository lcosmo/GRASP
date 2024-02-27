import torch
import torch.distributions as td
from torch_geometric.data import Data

def zero_diag(matrix: torch.Tensor):
    matrix = matrix * (1 - torch.eye(matrix.size(-2), matrix.size(-1), device=matrix.device))
    return matrix

def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
#     mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(mask[:,None,:,:] * x * mask[:,:,None,:], dim=[1,2]) / (torch.sum(mask, dim=[1])**2))   # (N,C)
    
    var_term = (mask[:,None,:,:] * (x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) *  mask[:,:,None,:])**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / (torch.sum(mask, dim=[1])**2))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    
    instance_norm = mask[:,None,:,:] *instance_norm * mask[:,:,None,:]
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    raise NotImplementedError("Not implemented")
    
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm
