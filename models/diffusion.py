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

        

# ###############################################################################################################
class SuperSquIsh(Module):
    
    def __init__(self, dim_in, dim_out, dim_ctx,glob=False):
        
        super(SuperSquIsh, self).__init__()
        
        self.glob=glob
        self.act = F.leaky_relu
        self.dim_in = dim_in
        

        self._layerA = Linear(dim_in+dim_ctx, dim_out)
        
        self._layerA_bis = Linear(dim_out, dim_out)
        
        self._layerC = Linear(dim_in+dim_ctx , dim_out)
        self._layerC_bis = Linear(dim_out , dim_out)

        self._layerD = Linear(dim_out , dim_out)
        self._layerDboth = Linear(2*dim_in , dim_out)
        
    def forward(self, ctx, x, y,m):
        
        time = ctx

        # concatenate phi and time 
        retPHI = torch.cat([x,time.expand(x.shape[0],x.shape[1],time.shape[-1])],-1)     #64X28X10    
        retPHI = self._layerA(retPHI)# LAYER A 64X28X64 
#         retPHI = self.act(retPHI)    
        retPHIout  = retPHI
  
        # concatenate lambda and time        
        retLAM = torch.cat([y,time.expand(y.shape[0],y.shape[1],time.shape[-1])],-1)#  64X1X10
        retLAM  = self._layerC(retLAM)#  64X1X64
#         retLAM  = self.act(retLAM)
        retLAMout = retLAM
 
               
        # STACK all   
        retPHIoutretLAMout = torch.cat([retPHIout*m[...,None],retLAMout],-2)   #64x29x64
        retPHIoutretLAMout = self.act(retPHIoutretLAMout)
        
#         print('Super SquIsh OUTPUT >>{}\n\n\n'.format(retPHIoutretLAMout.shape))
       
        return retPHIoutretLAMout
    
class Supersquash(Module):
    
    def __init__(self, dim_in, dim_out, dim_ctx,glob=False):
        
        super(Supersquash, self).__init__()

        self.glob=glob
        
        self.act = F.leaky_relu



        
        self._layer = Linear(dim_in+dim_ctx, dim_out) 
       
        self._layer_neutral = Linear( dim_out, dim_out) 
        
        self._layer2 = Linear(dim_in+dim_ctx, dim_out)

        self._layer2bis = Linear(4*dim_out, dim_out)

        self._layer3 = Linear(4*dim_out, dim_out,bias=True)

        self._layer4 = Linear(2*dim_out, dim_out)
 
        
        
        
    def forward(self, ctx, x, y,m):
        
        
        
       # CONCAT EIGENVECTORS AND TIME 
        ret = torch.cat([x,ctx.expand(x.shape[0],x.shape[1],ctx.shape[-1])],-1)# 64, 28, 35     #5, 28, 67  
        ret = self._layer(ret)      # 64, 28, 32 
  

        # GET MEAN  
        retout1  = ret.mean(-2,keepdim=True)#.expand(*ret.shape)# 64,1,32
        retout1b = ret.std(-2,keepdim=True)#.expand(*ret.shape)# 64,1,32
        retout1c =  ret.sum(-2,keepdim=True)#[0].expand(*ret.shape)# 64,1,32
           
        # CONCAT EIGENVALUES AND TIME
        ret2 = torch.cat([y,ctx.expand(y.shape[0],y.shape[1],ctx.shape[-1])],-1)#64, 1, 35     
        ret2 = self._layer2(ret2)
 
        
        # CONCAT OVER ROW EIGENVALUES and MEAN EIGEVECTOR
        ret2in = torch.cat([retout1,retout1b,retout1c,ret2],-1)#64, 1, 64
        ret2 = self._layer3(ret2in)# 64 1 32
        ret2out = ret2
         
        # CONCAT EIGENVECTORS with time and EIGENVALUES with mean
#         ret = self._layer_neutral(ret) 
        ret = ret*m[...,None]
        ret3 =  torch.cat([ret,ret2out.expand(ret.shape[0],ret.shape[1],ret2out.shape[-1])],-1) 
        ret3 = self._layer4(ret3) # 64 28 32
 
        # EIGENVALUES and MEAN EIGEVECTOR
        ret4 = self._layer2bis(ret2in) # 64 1 32
      

        return ret3,ret4
    

class PointwiseNet(Module):
      def __init__(self, point_dim, residual=False, args=None):  
#       def __init__(self,marginal_prob_std,  point_dim=7):
        super().__init__()
        
        self.act = F.leaky_relu
#         self.marginal_prob_std = marginal_prob_std
        
        self.layerINITs = ModuleList([ 
            SuperSquIsh(point_dim, 64, 3)
        ])
            
        self.layers = ModuleList([
            Supersquash(64, 64, 3),
            Supersquash(64, 64, 3),
            Supersquash(64, 64, 3),
            Supersquash(64, point_dim, 3),
        ]
        
        )
      
      def forward(self,xy,beta,m):
            
            x = xy[:,:-1,:]# EIGENVECTORS
            
            y = xy[:,-1:,:] #EIGENVALUES
     
            batch_size = x.shape[0] 
            beta = beta.view(batch_size, 1, 1)          # (B, 1, 1) #TIME      
#             beta = beta.view(x.size(0) , 1, 1)          # (B, 1, 1) #TIME      
            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)   
            
            outx = x
            outy = y
            
            for jj, layerinit in enumerate(self.layerINITs):  
            
                outxouty = layerinit(x=outx, y=outy, m=m, ctx=time_emb)    
                
                outx = outxouty[:,:-1,:]# EIGENVECTORS            
                outy = outxouty[:,-1:,:]#EIGENVALUES
                
            for i, layer in enumerate(self.layers): 

                outx, outy = layer(x=outx, y=outy, m=m, ctx=time_emb)

                if i < len(self.layers) - 1:
                    outx = self.act(outx)
                    outy = self.act(outy)
 
            outfinal = xy + torch.cat([outx,outy],-2) 
#             outfinal = outfinal / self.marginal_prob_std(beta)      

            return outfinal
    
    
        
###############################################################################################################
from models.attention import MultiHeadAttention
from torch import nn

class SelfCrossAttention(Module):
    
    def __init__(self, dim_in, dim_out,point_dim):
        super(SelfCrossAttention, self).__init__()
        
        self.selfaL = MultiHeadAttention(4, dim_out, dim_out)
        self.selfaP = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaL = MultiHeadAttention(4, dim_out, dim_out)
        self.crossaP = MultiHeadAttention(4, dim_out, dim_out)
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(dim_in,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
#         self.init_layer_LAM = nn.Sequential(nn.Linear(dim_in,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(dim_in,dim_out,3,padding=1),nn.LeakyReLU(),nn.Conv1d(dim_out,dim_out,3,padding=1))
        
        self.embed_time_scale_x =  nn.Sequential(nn.Linear(3,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        self.embed_time_bias_x =  nn.Sequential(nn.Linear(3,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        
        self.embed_time_scale_y =  nn.Sequential(nn.Linear(3,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,point_dim))
        self.embed_time_bias_y =  nn.Sequential(nn.Linear(3,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,point_dim))
        
    def forward(self, x, y, m, time):
        #x = BxNxD
        #y = BxDxK
        
        
        y_ = self.init_layer_LAM(y).transpose(1,2)
#         x_ = torch.cat([x,y_.expand(x.shape[0],x.shape[1],y_.shape[-1])],-1)
        x_ = self.init_layer_PHI(x)
#         
        x_ = x_+self.selfaP(x_,x_,x_, m)
        y_ = y_+self.selfaL(y_,y_,y_)
        
        x_ = 0.5*(x_+self.crossaP(x_,y_,y_))
        y_ = 0.5*(y_+self.crossaL(y_,x_,x_,m)).transpose(1,2)

        x = x + x_*self.embed_time_scale_x(time) + self.embed_time_bias_x(time)
        y = y + y_*self.embed_time_scale_y(time) + self.embed_time_bias_y(time) #.transpose(1,2)

#         x = x+0.5*(x_+self.crossaP(x_,y_,y_))
#         y = y+0.5*(y_+self.crossaL(y_,x_,x_,m)).transpose(1,2)

#         x = x*self.embed_time_scale_x(time)
#         x = x+self.embed_time_bias_x(time)
        
#         y = y*self.embed_time_scale_y(time)#.transpose(1,2)
#         y = y+self.embed_time_bias_y(time)#.transpose(1,2)
        
        return x,y

    
    
class PointwiseNet(Module):
    
      def __init__(self, point_dim, residual=False, args=None):
          
        super().__init__()
        embdim = args.latent_dim
        
        self.act = F.leaky_relu
        
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(point_dim,32),nn.LeakyReLU(),nn.Linear(32,embdim))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.LeakyReLU(),nn.Conv1d(32,embdim,3,padding=1))
        
        
            
        self.layers = ModuleList([ SelfCrossAttention(embdim, embdim,point_dim) for i in range(args.layers)])
       
        self.final_layer_PHI = nn.Sequential(nn.Linear(embdim,32),nn.LeakyReLU(),nn.Linear(32,point_dim))
        self.final_layer_LAM = nn.Sequential(nn.Conv1d(embdim,16,3,padding=1),nn.LeakyReLU(),nn.Conv1d(16,1,3,padding=1))
        
      
      def forward(self,xy,beta,m=None):
            
            x = xy[:,:-1,:]# EIGENVECTORS            
            y = xy[:,-1:,:]#EIGENVALUES
            
#             if m is None:
#                 m = torch.ones_like(xy[:,:,:1])

            batch_size = x.shape[0] 
            beta = beta.view(x.size(0) , 1, 1)          # (B, 1, 1) #TIME      
            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-2).transpose(1,2)  # (B, 3, 1)   
            
#             outx = torch.cat([x,time_emb.transpose(1,2).expand(x.shape[0], x.shape[1], time_emb.shape[-2])],-1)
#             outy = torch.cat([y,time_emb.expand(y.shape[0], time_emb.shape[1], y.shape[-1])],-2)
            outx,outy = x,y
    
            outx = self.init_layer_PHI(outx)
            
            outy = self.init_layer_LAM(outy) #b x d x k
            
            for jj, layer in enumerate(self.layers):  
                outx,outy = layer(outx,outy,m,time_emb)
            
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
from torch.nn import TransformerDecoderLayer


class SelfCrossAttention(Module):
    
    def __init__(self, dim_out, point_dim, cond_size=3):
        super(SelfCrossAttention, self).__init__()
        
        self.cond_size=cond_size
        
        self.vec_vs_val = TransformerDecoderLayer(dim_out, 4, batch_first=True,dropout=0.02)
        self.val_vs_vec = TransformerDecoderLayer(dim_out, 4, batch_first=True,dropout=0.02)
        
        
        self.embed_time_scale_x =  nn.Sequential(nn.Linear(self.cond_size,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        self.embed_time_bias_x =  nn.Sequential(nn.Linear(self.cond_size,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,dim_out))
        
        self.embed_time_scale_y =  nn.Sequential(nn.Linear(self.cond_size,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,point_dim))
        self.embed_time_bias_y =  nn.Sequential(nn.Linear(self.cond_size,dim_out),nn.LeakyReLU(),nn.Linear(dim_out,point_dim))
        
    def forward(self, x, y, m, time):
        #x = BxNxD
        #y = BxDxK
        
        x_ = self.vec_vs_val(x,y, tgt_key_padding_mask=~m.bool())
        y_ = self.val_vs_vec(y,x, memory_key_padding_mask=~m.bool())

        x = x + x_*self.embed_time_scale_x(time) + self.embed_time_bias_x(time)
        y = y + y_*self.embed_time_scale_y(time).transpose(-1,-2) + self.embed_time_bias_y(time).transpose(-1,-2) #.transpose(1,2)

        return x,y

    
    
class PointwiseNet(Module):
    
      def __init__(self, point_dim, residual=False, args=None):
          
        super().__init__()
        embdim = args.latent_dim
        
        self.act = F.leaky_relu
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(point_dim,32),nn.LeakyReLU(),nn.Linear(32,embdim))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.LeakyReLU(),nn.Conv1d(32,embdim,3,padding=1))
        
        
        self.layers = ModuleList([ SelfCrossAttention(embdim,point_dim,4) for i in range(args.layers)])
       
        self.final_layer_PHI = nn.Sequential(nn.Linear(embdim,embdim),nn.LeakyReLU(),nn.Linear(embdim,point_dim))
        self.final_layer_LAM = nn.Sequential(nn.Conv1d(embdim,embdim,3,padding=1),nn.LeakyReLU(),nn.Conv1d(embdim,1,3,padding=1))
        
        # number of nodes scaling/bias factor
#         self.LAM_scale = nn.Sequential(nn.Linear(1,4),nn.LeakyReLU(),nn.Linear(4,1))
        
      
      def forward(self,xy,beta,m=None):
            
            x = xy[:,:-1,:]#EIGENVECTORS            
            y = xy[:,-1:,:]#EIGENVALUES
            
#             if m is None:
#                 m = torch.ones_like(xy[:,:,:1])
            if m is not None:
                nnodes =  m.sum(-1)[:,None,None]*1e-2
            

            batch_size = x.shape[0] 
            beta = beta.view(x.size(0) , 1, 1)          # (B, 1, 1) #TIME      
            time_emb = torch.cat([nnodes, beta, torch.sin(beta), torch.cos(beta)], dim=-2).transpose(1,2)  # (B, 3, 1)   
            
            outx,outy = x,y
            outx = self.init_layer_PHI(outx)
            outy = self.init_layer_LAM(outy).transpose(-1,-2) #b x d x k
            
            for jj, layer in enumerate(self.layers):  
                outx,outy = layer(outx,outy,m,time_emb)
            
#             loffset = 0
#             if m is not None:
#                 loffset =  self.LAM_scale(m.sum(-1,keepdim=True))[...,None]
                
                
            outx = self.final_layer_PHI(outx)
            outy = self.final_layer_LAM(outy.transpose(-1,-2))
            
            
            outfinal = xy+torch.cat([outx,outy],-2) 
            outfinal = outfinal# / self.marginal_prob_std(beta)      

            return outfinal
        
        

        
    
class PointwiseNet2(Module):
    
      def __init__(self, point_dim, out_emb_size, residual=False, args=None):
          
        super().__init__()
        embdim = args.latent_dim
        self.out_emb_size = out_emb_size
        self.latent_dim = embdim
        self.act = F.leaky_relu
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(point_dim,point_dim),nn.LeakyReLU(),nn.Linear(point_dim,embdim))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(1,point_dim,3,padding=1),nn.LeakyReLU(),nn.Conv1d(point_dim,embdim,3,padding=1))
        
        
        self.layers = ModuleList([ SelfCrossAttention(embdim,out_emb_size,4) for i in range(args.layers)])
       
        self.final_layer_PHI = nn.Sequential(nn.Linear(embdim,out_emb_size),nn.LeakyReLU(),nn.Linear(out_emb_size,out_emb_size))
        self.final_layer_LAM = nn.Sequential(nn.Conv1d(embdim,embdim,3,padding=1),nn.LeakyReLU(),nn.Conv1d(embdim,1,3,padding=1))
        
        # number of nodes scaling/bias factor
#         self.LAM_scale = nn.Sequential(nn.Linear(1,4),nn.LeakyReLU(),nn.Linear(4,1))
        
      
      def forward(self,x,y,m,beta=0,return_evecs = False):
            
#             x = xy[:,:-1,:]#EIGENVECTORS            
#             y = xy[:,-1:,:]#EIGENVALUES
            y = torch.nn.functional.pad(y,(0,self.out_emb_size-y.shape[-1]))[:,None]
    
#             if m is None:
#                 m = torch.ones_like(xy[:,:,:1])
            if m is not None:
                nnodes =  m.sum(-1)[:,None,None]*1e-2
            

            batch_size = x.shape[0] 
#             beta = beta.view(x.size(0) , 1, 1)          # (B, 1, 1) #TIME      
#             time_emb = torch.cat([nnodes, beta, torch.sin(beta), torch.cos(beta)], dim=-2).transpose(1,2)*0+1  # (B, 3, 1)   
            time_emb = torch.ones(batch_size,1,4, device=x.device)
            outx,outy = x,y
            outx = self.init_layer_PHI(outx)
            
            outy = self.init_layer_LAM(outy).transpose(-1,-2) #b x d x k
            
            for jj, layer in enumerate(self.layers):  
                outx,outy = layer(outx,outy,m,time_emb)

#             loffset = 0
#             if m is not None:
#                 loffset =  self.LAM_scale(m.sum(-1,keepdim=True))[...,None]
                
                
            outx = self.final_layer_PHI(outx)
            outy = self.final_layer_LAM(outy.transpose(-1,-2))
            
            
#             outfinal = xy+torch.cat([outx,outy],-2) 
#             outfinal = outfinal# / self.marginal_prob_std(beta)      
            outx[:,:,:x.shape[-1]] += x
            outy[:,:,:y.shape[-1]] += y

            outx = outx/outx.norm(dim=-2)[:,None]
        
            L = (outx*outy)@outx.transpose(-1,-2)
            D = 1/(torch.diagonal(L.abs(),dim1=-2, dim2=-1).sqrt()+1e-6)
            A = 1-D[:,:,None]*L*D[:,None,:]

            if return_evecs:
                return A, outx, outy[:,0]
                
            return A, None, None
        
        
        
        
        