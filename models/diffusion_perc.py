import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .common import *


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
            betas = torch.sigmoid(torch.linspace(-6, 2.5, steps=num_steps))

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)#torch.sigmoid(betas*8-4)
        sigmas_flex = torch.sigmoid(betas*8-4)
        
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


############################################################################## ORI
    
class PointwiseNet(Module):

    def __init__(self, point_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, 3),
            ConcatSquashLinear(128, 256, 3),
            ConcatSquashLinear(256, 512, 3),
            ConcatSquashLinear(512, 256, 3),
            ConcatSquashLinear(256, 128, 3),
            ConcatSquashLinear(128, point_dim, 3)
        ])
        
        
#         x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
#         x = self.conv1(x,time_emb)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv2(x,time_emb)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        

    def forward(self, x, beta):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            labels:   point labels (B, N, 1).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        
        
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        #ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(x=out, ctx=time_emb)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


############################################################################## ORI

# import torch.nn as nn
# import torch.nn.functional as F
# from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation



# class Conv2d(Module):
#     def __init__(self, dim_in, dim_out, bias, kernel_size=1, dim_ctx=3):
#         super(Conv2d, self).__init__()
#         self._layer = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        
#         self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
#         self._hyper_gate = Linear(dim_ctx, dim_out)
        
# #         self.bn = nn.BatchNorm2d(dim_out)
#         self.att = nn.LeakyReLU(negative_slope=0.2)

#     def forward(self, x, ctx):
#         gate = torch.sigmoid(self._hyper_gate(ctx))[...,None,None]
#         bias = self._hyper_bias(ctx)[...,None,None]
#         ret = self._layer(x)

#         ret = ret * gate + bias
# #         self.bn(ret)
#         ret = self.att(ret)
#         return ret
    
# class Conv1d(Module):
#     def __init__(self, dim_in, dim_out, bias, kernel_size=1, dim_ctx=3):
#         super(Conv1d, self).__init__()
#         self._layer = nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=bias)
        
#         self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
#         self._hyper_gate = Linear(dim_ctx, dim_out)

#         #         self.bn = nn.BatchNorm1d(dim_out)
#         self.att = nn.LeakyReLU(negative_slope=0.2)
    
#     def forward(self, x, ctx):
#         gate = torch.sigmoid(self._hyper_gate(ctx))[...,None]
#         bias = self._hyper_bias(ctx)[...,None]
#         ret = self._layer(x)      
#         ret = ret * gate + bias
# #         self.bn(ret)
#         ret = self.att(ret)        
#         return ret

    
# class PointwiseNet(Module):

#     def __init__(self, point_dim, residual):
#         super().__init__()
        
#         self.k = 11
        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn5 = nn.BatchNorm2d(64)
#         self.bn6 = nn.BatchNorm1d(256)
#         self.bn7 = nn.BatchNorm1d(512)
#         self.bn8 = nn.BatchNorm1d(256)

#         self.conv1 = Conv2d(2*(point_dim), 64, kernel_size=1, bias=False)
#         self.conv2 = Conv2d(64, 64, kernel_size=1, bias=False)
#         self.conv3 = Conv2d(64*2, 64, kernel_size=1, bias=False)
#         self.conv4 = Conv2d(64, 64, kernel_size=1, bias=False)
#         self.conv5 = Conv2d(64*2, 64, kernel_size=1, bias=False)
#         self.conv6 = Conv1d(192, 256, kernel_size=1, bias=False)
#         self.conv7 = Conv1d(448 , 512, kernel_size=1, bias=False)
#         self.conv8 = Conv1d(512, 256, kernel_size=1, bias=False)
# #         self.dp1 = nn.Dropout(p=args.dropout)
#         self.conv9 = Conv1d(256, point_dim, kernel_size=1, bias=False)
        

#     def forward(self, x, beta):
#         inx = x
        
#         x = x.transpose(1,2)
#         batch_size = x.size(0)
#         num_points = x.size(2)

#         beta = beta.view(batch_size, 1)          # (B, 1, 1)
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 2, 1)
# #         assert(len(time_emb.shape)==3)
# #         x = torch.cat([x,time_emb.expand(batch_size,3,num_points)],1)
        
#         x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
#         x = self.conv1(x,time_emb)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv2(x,time_emb)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#         x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv3(x,time_emb)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv4(x,time_emb)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#         x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv5(x,time_emb)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#         x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

#         x = self.conv6(x,time_emb)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
#         x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

#         x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
#         x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

#         x = self.conv7(x,time_emb)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
#         x = self.conv8(x,time_emb)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
# #         x = self.dp1(x)
#         x = self.conv9(x,time_emb)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
#         x = inx+x.transpose(1,2)
#         return x.contiguous()
# import torch.nn.functional as F


# def knn(x, k):
#     inner = -2*torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
#     return idx


# def get_graph_feature(x, k=20, idx=None, dim9=False):
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         if dim9 == False:
#             idx = knn(x, k=k)   # (batch_size, num_points, k)
#         else:
#             idx = knn(x[:, 6:], k=k)
#     device = torch.device('cuda')

#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

#     idx = idx + idx_base

#     idx = idx.view(-1)
 
#     _, num_dims, _ = x.size()

#     x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size*num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims) 
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
#     feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
#     return feature      # (batch_size, 2*num_dims, num_points, k)

# ################################################################################################################
# from torch_points3d.applications.pointnet2 import PointNet2
# from torch_geometric.data import Batch, Data

# class PointwiseNet(Module):

#     def __init__(self, point_dim, residual):
#         super().__init__()
        
#         self.model = PointNet2(architecture="unet", input_nc=point_dim+3, num_layers=3, output_nc=64)
#         self.last = ConcatSquashLinear(64, point_dim, 3)

        
#     def forward(self, x, beta):
#         """
#         Args:
#             x:  Point clouds at some timestep t, (B, N, d).
#             beta:     Time. (B, ).
#             labels:   point labels (B, N, 1).
#         """
#         batch_size = x.size(0)
#         num_points = x.size(1)
#         beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
                
#         pos = x[:,:,:3].contiguous()
#         T = torch.cat([x[:,:,:],time_emb.expand(batch_size,num_points,3)],-1)
        
# #         data = lambda x:x
# #         x.pos = pos
# #         x.x=T
#         data = Data(pos=pos, x=T)
        
#         res = self.model(data).x.transpose(1,2).contiguous()
# #         print(res.shape)
#         res = self.last(time_emb,res)
# #         print(data.x.shape)
#         res = res + x
#         return res
    





#######################################################################################################################

# device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
# def marginal_prob_std(t, sigma=25.0):
#   t = torch.tensor(t, device=device)
#   return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

# def diffusion_coeff(t, sigma=25):
#   return torch.tensor(sigma**t, device=device)
  
# class DiffusionPoint(Module):

#     def __init__(self, net, var_sched:VarianceSchedule):
#         super().__init__()
#         self.net = net
#         self.var_sched = var_sched

#     def get_loss(self, x_0, t=None, eps=1e-5):
#         """
#         Args:
#             x_0:  Input point cloud, (B, N, d).
#             l_0:  Point labels, (B, N, 1).
#         """
#         batch_size, _, point_dim = x_0.size()
#         self.point_dim = point_dim
        
#         t = torch.rand(batch_size).to(x_0.device)[:,None,None]  * (1. - eps) + eps  

#         z = torch.randn_like(x_0)
#         std = marginal_prob_std(t)
#         perturbed_x = x_0 + z * std
#         score = self.net(perturbed_x, t)/std
#         loss = torch.mean(torch.sum((score * std + z)**2,dim=(2)))
#         return loss
    

#     def sample(self, num_points, batch_size, point_dim=7, flexibility=0.0, ret_traj=False, num_steps=1000, eps=1e-5):
        
#         t = torch.ones(batch_size, device=device)
#         init_x = torch.randn(batch_size, num_points, point_dim, device=device) * marginal_prob_std(t)[:, None, None]
#         time_steps = torch.linspace(1., eps, num_steps, device='cuda')
#         step_size = time_steps[0] - time_steps[1]
#         x = init_x
#         with torch.no_grad():
#             for time_step in time_steps:      
#               batch_time_step = torch.ones((batch_size,1), device=device) * time_step
#               g = diffusion_coeff(batch_time_step)
#               mean_x = x + (g**2)[:, None] * self.net(x, batch_time_step)/marginal_prob_std(time_step) * step_size
#               x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)     
              
#             # Do not include any noise in the last sampling step.
#         return mean_x


    
class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, m=1, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            l_0:  Point labels, (B, N, 1).
        """
        batch_size, _, point_dim = x_0.size()
        self.point_dim = point_dim
        
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        m_loss = m
        if isinstance(m, torch.Tensor) and len(m.shape)>1:
            m_loss = torch.nn.functional.pad(m,(0,1),value=1)[:,:,None]
            
        e_rand = torch.randn_like(x_0)*m_loss  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, m=m, beta=beta)*m_loss

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, batch_size, point_dim=16, flexibility=0.0, ret_traj=False, device='cuda'):
#         batch_size = context.size(0)
        
        x_T = torch.randn([batch_size, num_points, point_dim]).to(device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size].to(x_t.device)
            
            e_theta = self.net(x_t, beta=beta)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
#             print(x_next)
        
        if ret_traj:
            return traj
        else:
            return traj[0]


###############################################################################################################
from models.attention import MultiHeadAttention
from torch import nn

class SelfCrossAttentionDown(Module):
    
    def __init__(self, dim_in, dim_out, n_out):
        super(SelfCrossAttentionDown, self).__init__()
        
        self.selfaL = MultiHeadAttention(4, dim_out, dim_out)
        self.selfaP = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaL = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaP = MultiHeadAttention(4, dim_out, dim_out)
                
        self.anchors = torch.nn.Parameter(torch.randn(1,n_out,dim_out))
        self.downP = MultiHeadAttention(4, dim_out, dim_out)
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(dim_in,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(dim_in,dim_out,3,padding=1),nn.LeakyReLU(),nn.Conv1d(dim_out,dim_out,3,padding=1))
        
    def forward(self, x, y, m=None):
        #x = BxNxD
        #y = BxDxK
        
        #self
        y_ = self.init_layer_LAM(y).transpose(1,2)
        x_ = self.init_layer_PHI(x)

        x_ = x_+self.selfaP(x_,x_,x_, m)
        y_ = y_+self.selfaL(y_,y_,y_)
        
        #cross        
        x = x+0.5*(x_+self.crossaP(x_,y_,y_))
        y = y+0.5*(y_+self.crossaL(y_,x_,x_,m)).transpose(1,2)
        
#         #downsample x
#         _,n,d = self.anchors.shape
#         x = self.crossaP(self.anchors.expand(x.shape[0],n,d),x,x, m)        
        
        return x,y

class SelfCrossAttentionUp(Module):
    
    def __init__(self, dim_in, dim_out,):
        super(SelfCrossAttentionUp, self).__init__()
        
        self.selfaL = MultiHeadAttention(4, dim_out, dim_out)
        self.selfaP = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaL = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaP = MultiHeadAttention(4, dim_out, dim_out)
                
        self.upP = MultiHeadAttention(4, dim_out, dim_out)
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(dim_in,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(dim_in,dim_out,3,padding=1),nn.LeakyReLU(),nn.Conv1d(dim_out,dim_out,3,padding=1))
        
    def forward(self, x, x_prev, y, m=None):
        #x = BxNxD
        #y = BxDxK
    
#         #upsample x
#         x = self.upP(x_prev,x,x, m)    
        x_prev = x
    
        #self
        y_ = self.init_layer_LAM(y).transpose(1,2)
        x_ = self.init_layer_PHI(x)
#         
        x_ = x_+self.selfaP(x_,x_,x_, m)
        y_ = y_+self.selfaL(y_,y_,y_)
               
        #cross        
        x = x_prev+0.5*(x_+self.crossaP(x_,y_,y_))
        y = y+0.5*(y_+self.crossaL(y_,x_,x_,m)).transpose(1,2)
        

        return x,y 
    
class PointwiseNet(Module):
    
      def __init__(self, point_dim, residual=False, args=None):
          
        super().__init__()
        embdim = args.latent_dim
        
        self.act = F.leaky_relu
        
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(point_dim+3,32),nn.LeakyReLU(),nn.Linear(32,embdim))
#         self.init_layer_LAM = nn.Sequential(nn.Linear(point_dim+3,32),nn.LeakyReLU(),nn.Linear(32,64))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(4,32,3,padding=1),nn.LeakyReLU(),nn.Conv1d(32,embdim,3,padding=1))
        
#         self.layers = ModuleList([ SelfCrossAttention(embdim, embdim) for i in range(args.layers)])
        self.layers_down = ModuleList([ SelfCrossAttentionDown(embdim, embdim,2**i) for i in range(args.layers,0,-1)])
        self.layers_up   = ModuleList([ SelfCrossAttentionUp(embdim, embdim) for i in range(args.layers-1,-1,-1)])

        
        self.final_layer_PHI = nn.Sequential(nn.Linear(embdim,32),nn.LeakyReLU(),nn.Linear(32,point_dim))
        self.final_layer_LAM = nn.Sequential(nn.Conv1d(embdim,16,3,padding=1),nn.LeakyReLU(),nn.Conv1d(16,1,3,padding=1))
        
      
      def forward(self,xy,beta,m=None):
            
            x = xy[:,:-1,:]# EIGENVECTORS            
            y = xy[:,-1:,:]#EIGENVALUES
            
#             if m is None:
#                 m = torch.ones_like(xy[:,:,:1])

            batch_size = x.shape[0] 
            beta = beta.view(x.size(0) , 1, 1)          # (B, 1, 1) #TIME      
            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-2)  # (B, 3, 1)   
            
            outx = torch.cat([x,time_emb.transpose(1,2).expand(x.shape[0], x.shape[1], time_emb.shape[-2])],-1)
            outy = torch.cat([y,time_emb.expand(y.shape[0], time_emb.shape[1], y.shape[-1])],-2)
            
            outx = self.init_layer_PHI(outx)
            
            outy = self.init_layer_LAM(outy) #b x d x k
            
#             for jj, layer in enumerate(self.layers):  
#                 outx,outy = layer(outx,outy,m)
            
            if m==1:
                m=None
                
            outputs = []
            for jj, layer in enumerate(self.layers_down):  
                outx_new, out_y = layer(outx,outy,m) # 
                outputs.append([outx,outx_new])
                outx = outx_new
                
            for jj, layer in enumerate(self.layers_up):  
                outx_prev, outx = outputs[-(jj+1)]
                outx, out_y = layer(outx,outx_prev,outy,m) # 
                
#             outx = torch.cat([outx,time_emb.expand(x.shape[0], x.shape[1], time_emb.shape[-1])],-1)
#             outy = torch.cat([outy,time_emb.expand(y.shape[0], y.shape[1], time_emb.shape[-1])],-1)
            outx = self.final_layer_PHI(outx)
            outy = self.final_layer_LAM(outy)
            
            outfinal = xy+torch.cat([outx,outy],-2) 
            outfinal = outfinal# / self.marginal_prob_std(beta)      

            return outfinal
        
       
        
        
        


###############################################################################################################
from models.attention import MultiHeadAttention
from torch import nn

class SelfCrossAttention(Module):
    
    def __init__(self, dim_in, dim_out):
        super(SelfCrossAttention, self).__init__()
        
        self.selfaL = MultiHeadAttention(4, dim_out, dim_out)
        self.selfaP = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaL = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaP = MultiHeadAttention(4, dim_out, dim_out)
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(dim_in,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
#         self.init_layer_LAM = nn.Sequential(nn.Linear(dim_in,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(dim_in,dim_out,3,padding=1),nn.LeakyReLU(),nn.Conv1d(dim_out,dim_out,3,padding=1))
        
        
    def forward(self, x, y, m=None):
        y_ = self.init_layer_LAM(y).transpose(1,2)
        x_ = self.init_layer_PHI(x)
#         
        x_ = x_+self.selfaP(x_,x_,x_, m)
        y_ = y_+self.selfaL(y_,y_,y_)
        
        x = x+0.5*(x_+self.crossaP(x_,y_,y_))
        y = y+0.5*(y_+self.crossaL(y_,x_,x_,m)).transpose(1,2)
        
        return x,y


class Down(Module):
    def __init__(self, dim_in, dim_out, n_out):
        super(Down, self).__init__()
        
        self.sca = SelfCrossAttention(dim_in,dim_out)
        
        self.anchors = torch.nn.Parameter(torch.randn(1,n_out,dim_out))
        self.downP = MultiHeadAttention(4, dim_out, dim_out)
        
    def forward(self, x, y, m=None):
        #downsample x
        x, y = self.sca(x,y,m)
        
#         _,n,d = self.anchors.shape
#         x = self.downP(self.anchors.expand(x.shape[0],n,d),x,x, m)    
        return x, y
    

class Up(Module):
    def __init__(self, dim_in, dim_out):
        super(Up, self).__init__()
        
        self.sca = SelfCrossAttention(dim_in,dim_out)
        
        self.anchors = torch.nn.Parameter(torch.randn(1,n_out,dim_out))
        self.upP = MultiHeadAttention(4, dim_out, dim_out)
        
    def forward(self, x, xpre, y, m=None):
        #upsample x
#         x = self.upP(xpre,x,x)
        x, y = self.sca(x,y)

        
        return x, y
    

    
    
class PointwiseNet(Module):
    
      def __init__(self, point_dim, residual=False, args=None):
          
        super().__init__()
        embdim = args.latent_dim
        
        self.act = F.leaky_relu
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(point_dim+3,32),nn.LeakyReLU(),nn.Linear(32,embdim))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(4,32,3,padding=1),nn.LeakyReLU(),nn.Conv1d(32,embdim,3,padding=1))
            
        self.layers = ModuleList([ SelfCrossAttention(embdim, embdim) for i in range(args.layers)])
       
#         self.layers_down = ModuleList([ Down(embdim, embdim,2**i) for i in range(args.layers,0,-1)])
#         self.layers_up = ModuleList([ Up(embdim, embdim) for i in range(args.layers)])
       
        self.final_layer_PHI = nn.Sequential(nn.Linear(embdim,32),nn.LeakyReLU(),nn.Linear(32,point_dim))
        self.final_layer_LAM = nn.Sequential(nn.Conv1d(embdim,16,3,padding=1),nn.LeakyReLU(),nn.Conv1d(16,1,3,padding=1))
        
      
      def forward(self,xy,beta,m=None):
            
            x = xy[:,:-1,:]# EIGENVECTORS            
            y = xy[:,-1:,:]#EIGENVALUES
            
#             if m is None:
#                 m = torch.ones_like(xy[:,:,:1])
            if m==1:
                m=None

            batch_size = x.shape[0] 
            beta = beta.view(x.size(0) , 1, 1)          # (B, 1, 1) #TIME      
            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-2)  # (B, 3, 1)   
            
            outx = torch.cat([x,time_emb.transpose(1,2).expand(x.shape[0], x.shape[1], time_emb.shape[-2])],-1)
            outy = torch.cat([y,time_emb.expand(y.shape[0], time_emb.shape[1], y.shape[-1])],-2)
            
            outx = self.init_layer_PHI(outx)
            
            outy = self.init_layer_LAM(outy) #b x d x k
            
            for jj, layer in enumerate(self.layers):  
                outx,outy = layer(outx,outy,m)
                
#             outputs = []
#             for jj, layer in enumerate(self.layers_down):  
#                 outputs.append(outx)
#                 outx,outy = layer(outx,outy,m)
            
#             for jj, layer in enumerate(self.layers_up):  
#                 outx,outy = layer(outx,outputs[-(jj+1)],outy,m)
                
            outx = self.final_layer_PHI(outx)
            outy = self.final_layer_LAM(outy)
            
            outfinal = xy+torch.cat([outx,outy],-2) 
            outfinal = outfinal# / self.marginal_prob_std(beta)      

            return outfinal
        
        