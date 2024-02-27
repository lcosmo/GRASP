import torch
import wandb
import numpy as np
from torch import nn
from torch.nn import Module, Parameter, ModuleList
from torch.nn import TransformerDecoderLayer
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import pytorch_lightning as L


from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy
import networkx as nx
from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, eval_fraction_unique, eval_fraction_unique_non_isomorphic_valid, spectral_stats
from utils import eval_helper_torch

import matplotlib.pyplot as plt


from diffusers import DDIMPipeline, DDPMScheduler, DDIMScheduler
from utils.misc import create_vis

class SpectralDiffusion(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        self.args = self.hparams
        
        #retrocompatibility check
        if not hasattr(self.args, 'feature_size'):
            self.args.feature_size = 0
        
        self.diffusion = Transformer(self.args.k+self.args.feature_size, args = self.args)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False,beta_schedule='linear')
        
        self.automatic_optimization = False
        
    def get_loss(self, x, y, m, em, writer=None):
          
        bs = x.shape[0]

        x = torch.cat([x,y[:,None,:]],-2).float()#.to(args.device)

        #loss computation
        noise = torch.randn_like(x)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=x.device
        ).long()

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.diffusion(noisy_x*em[:,None,:], timesteps/1000.0, m)

        assert(not torch.isnan(noise_pred).any())
        m = torch.nn.functional.pad(m,(0,1),value=1)[...,None]

        loss = (F.mse_loss(noise_pred, noise,reduction='none')*m*em[:,None,:]).sum()/(m.sum()*em.sum(-1).mean())

        return loss
    

    def sample(self, num_points, batch_size, flexibility, point_dim, scale_xy, unscale_xy, sampling_steps=100, device='cuda',reproject=False):
#         train_set = self.trainer.train_dataloader.dataset.datasets
#         test_set = self.trainer.val_dataloaders[0].dataset
        
        scheduler =  self.noise_scheduler
        scheduler = DDIMScheduler(num_train_timesteps=1000,clip_sample=False,beta_schedule='linear')
        
        if isinstance(num_points,list):
            max_points = max(num_points)+1
            batch_size = len(num_points)
        else:
            max_points = num_points+1
        
        noise = torch.randn(batch_size,max_points,point_dim).to(device) *  self.noise_scheduler.init_noise_sigma

        em = noise[:,0,:]*0+1
        if (max_points-1)<self.hparams.k:
            em[:,:self.hparams.k-max_points+1]=0
            
        m_all = noise*0+1        
        if isinstance(num_points,list):        
            for i,n in enumerate(num_points):
                m_all[i,n:-1,:] = 0  
                if n<self.hparams.k:
                    em[i,:self.hparams.k-n]=0
     
        m = m_all[:,:-1,0]
        
        noise = noise * em[:,None,:]
        
        scheduler.set_timesteps(sampling_steps)
        for t in scheduler.timesteps:
            with torch.no_grad():
                t_ = t[None].repeat(noise.shape[0]).to(noise.device)
                noisy_residual = self.diffusion(noise, t_/1000, m)
                previous_noisy_sample = scheduler.step(noisy_residual, t, noise).prev_sample

        #     reproject to orthonormal bases
            if reproject and t<100:
                M = scheduler.step(noisy_residual, t, noise).pred_original_sample*m_all
                M,yy = unscale_xy(M[:,:-1,],M[:,-1,:])
                
                padsize = M.shape[-1]-self.hparams.k
                M = M[:,:,:self.hparams.k]*em[:,None,:self.hparams.k]
                yy = yy[:,:self.hparams.k]*em[:,:self.hparams.k]
                
                svd = torch.svd(M[:,:])
                orth = svd.U@svd.V.transpose(-1,-2)
                
                grad = scale_xy(torch.nn.functional.pad(orth-M,(0,padsize)),torch.nn.functional.pad(yy,(0,padsize)))[0]
                previous_noisy_sample[:,:-1,:self.hparams.k] += 1e-1*grad.to(previous_noisy_sample.device)[:,:,:self.hparams.k]
                

            noise = previous_noisy_sample*m_all*em[:,None,:]
    
        ################################
        return noise,  m_all, em

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.args.lr
        )

        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps = 500,
        #     num_training_steps = self.hparams.train_loop_batches*self.hparams.max_epochs
        # )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.1, self.hparams.max_epochs//2)
        
        return {"optimizer": optimizer,  "lr_scheduler":scheduler}

    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        x = batch[0].float()
        y = batch[1].float()
        m = batch[2].float()
        em = batch[3].float()
        
        # Forward
        loss = self.get_loss(x, y, m=m, em=em)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Reset grad and model state
        optimizer.zero_grad()
        
        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        scheduler = self.lr_schedulers()
        scheduler.step()
        
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self, outputs):
        if self.trainer.train_dataloader is None:
            return
        
        train_set = self.trainer.train_dataloader.dataset.datasets
        test_set = self.trainer.val_dataloaders[0].dataset

        degree, cluster, orbit, unique, novel, spectral = self.evaluate(train_set, test_set, device=self.diffusion.init_layer_PHI[0].bias.device)
        
        self.log('degree', torch.tensor(degree).float(), on_step=False, on_epoch=True)
        self.log('cluster',  torch.tensor(cluster).float(), on_step=False, on_epoch=True)
        self.log('orbit',  torch.tensor(orbit).float(), on_step=False, on_epoch=True)
        self.log('unique',  torch.tensor(unique).float(), on_step=False, on_epoch=True)
        self.log('novel',  torch.tensor(novel).float(), on_step=False, on_epoch=True)
        self.log('spectral',  torch.tensor(spectral).float(), on_step=False, on_epoch=True)

        
    def sample_eigs(self, max_nodes, num_eigs, scale_xy, unscale_xy, num_graphs=256, oversample_mult=4, device='cuda', sampling_steps=100, reproject=False):
        # Sample eigenvectors and eigenvalues
        assert(type(max_nodes) is list)
        
        gen_pcs = []
        with torch.no_grad():
            x,m,em = self.sample(max_nodes*oversample_mult, num_graphs*oversample_mult, 1, num_eigs, scale_xy, unscale_xy, device=device, sampling_steps=sampling_steps, reproject=reproject)
            
            samples_EIGVEC = x.detach()#.cpu()        
            xx = samples_EIGVEC[:,:-1,:self.hparams.k]
            yy = samples_EIGVEC[:,-1:,:self.hparams.k]
            
            xx,yy = unscale_xy(xx,yy)
            
            xx = xx*m[:,:-1,:]*em[:,None,:]
            yy = yy.float()*em[:,None,:]
            
            #keep best generations
            score = (xx.transpose(-1,-2)@xx - em[:,None,:]*torch.eye(xx.shape[-1],device=xx.device)[None].repeat(xx.shape[0],1,1)*em[:,:,None] ).pow(2).sum((-1,-2))/(em.sum(-1)**2)
            
            score_idx = score.argsort()
            xx = xx[score_idx[:len(score_idx)//oversample_mult]]
            yy = yy[score_idx[:len(score_idx)//oversample_mult]]

            shuffle = torch.randperm(xx.shape[0])
            xx = xx[shuffle]
            yy = yy[shuffle]

        return xx,yy
    
    
    def sample_graphs(self, max_nodes, num_eigs, scale_xy, unscale_xy, num_graphs=256, oversample_mult=4, device='cuda'):
        # Sample eigenvectors and eigenvalues
        gen_pcs = []
        with torch.no_grad():
            x,m,em = self.sample(max_nodes, num_graphs*oversample_mult, 1, num_eigs, scale_xy, unscale_xy, device=device)
            samples_EIGVEC = x.detach().cpu()

            
        # reconstruct laplacian matrix
        recon_list = []
        for i,(X,gm,gem) in enumerate(zip(samples_EIGVEC.cpu(),m.cpu(),em.cpu())):
            n_nodes = int(np.round(gm[:,0].sum())-1)
            n_pad = int(np.round((1-gem).sum()))               
                  
            xx = X[:n_nodes,:]
            yy = X[-1:,:]
            
            xx,yy = unscale_xy(xx,yy)
            xx = xx[:,n_pad:self.hparams.k]
            yy = yy[:,n_pad:self.hparams.k]
            
            yy=yy.float()

            L = (xx*yy)@xx.t()
            
            err = ((xx.t())@xx-torch.eye(xx.shape[-1])).norm() 
            recon_list.append((err,L.numpy(),xx.t()@xx))

        #keep only best generations in therms of quasi-orthonormal basis
        LLLall  = [l[1] for l in sorted(recon_list, key=lambda e:e[0])][:len(recon_list)//oversample_mult]
        LLLorth = [l[2] for l in sorted(recon_list, key=lambda e:e[0])][:len(recon_list)//oversample_mult]

        #reconstruct graph
        graph_pred_list = []
        for i,pp in enumerate(range(len(LLLall))):
            mynew = np.around(LLLall[pp],2)
            np.fill_diagonal(mynew,np.around(np.diag(mynew),0))
            mask = np.diag(mynew)>0

            siad = len(np.diag(mynew))
            adjb = np.zeros((siad,siad))

            for jj in range(len(np.diag(mynew))):
                if np.diag(mynew)[jj]!=np.diag(mynew)[jj]:
                    break
                diagval = int(np.diag(mynew)[jj])

                row = mynew[jj,:]
                row[jj]=100
                idx =row.argsort()[:diagval]
                adjb[jj,idx] =1

            Arec = nx.Graph(adjb[mask,:][:,mask])
            graph_pred_list.append(Arec) 

        return graph_pred_list, LLLorth      


    def evaluate(self, train_set, test_set, device='cuda'):
        train_set.get_extra_data(True)
        test_set.get_extra_data(True)
        
        
        
        batch = next(iter(DataLoader(test_set, batch_size=min(2048,len(test_set)), shuffle=False, num_workers=0)))
        b,n,d = batch[0].shape
        
        graph_pred_list, orth_pred_list = self.sample_graphs(max_nodes=n, num_eigs=d, scale_xy = train_set.scale_xy, unscale_xy=train_set.unscale_xy, num_graphs=max(256, b), device=device)
        graph_pred_list = graph_pred_list

        graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0][:b]

        if self.trainer.logger is not None:
            self.trainer.logger.experiment.log({
                "G_1": wandb.Plotly(create_vis(graph_pred_list[0]))
                })
            self.trainer.logger.experiment.log({
                "G_2": wandb.Plotly(create_vis(graph_pred_list[1]))
                })
            self.trainer.logger.experiment.log({
                "G_3": wandb.Plotly(create_vis(graph_pred_list[2]))
                })
    
        
            self.trainer.logger.experiment.log({
                "orth": [wandb.Image(img, caption="") 
                for img in orth_pred_list[:3]]
                })

        
        MAX_GRAPHS = 2048
        train_subset = [train_set[i] for i in np.random.permutation(len(train_set))[:MAX_GRAPHS]]
        test_subset = [test_set[i] for i in np.random.permutation(len(test_set))[:MAX_GRAPHS]]
        
        adj_list_train = [g[-1][0][:g[-2]][:,:g[-2]].cpu() for g in train_subset]
        adj_list_test = [g[-1][0][:g[-2]][:,:g[-2]].cpu() for g in test_subset]

        adj_list_train = [a[m,:][:,m] for a,m in zip(adj_list_train,[z.sum(-1)>0 for z in adj_list_train])] #remove isolated
        adj_list_test = [a[m,:][:,m] for a,m in zip(adj_list_test,[z.sum(-1)>0 for z in adj_list_test])] #remove isolated
        adj_list_pred = [torch.tensor(nx.to_numpy_array(g)).float() for g in graph_pred_list_remove_empty]

        degree, cluster, orbit, unique, novel, spectral = 0,0,0,0,0,0
        if len(graph_pred_list_remove_empty)>0:
            degree = eval_helper_torch.degree_stats( adj_list_test,adj_list_pred, compute_emd=False)
            cluster = eval_helper_torch.clustering_stats( adj_list_test,adj_list_pred, compute_emd=False)
#             spectral = eval_helper_torch.spectral_stats(adj_list_test, adj_list_pred)
        
        
#         #compute metrics
#         graph_test_list = []
#         for jj in range(len(test_set)):
#             laplacian_matrix = np.array(test_set[jj][3].cpu())[:test_set[jj][4],:test_set[jj][4]]
#             Aori = np.copy(laplacian_matrix)
#             np.fill_diagonal(Aori,0)
#             Aori= Aori*(-1)
#             graph_test_list.append(nx.from_numpy_array(Aori)) 

#         graph_train_list = []
#         for jj in range(len(train_set)):
#             laplacian_matrix = np.array(train_set[jj][3].cpu())[:train_set[jj][4],:train_set[jj][4]]
#             Aori = np.copy(laplacian_matrix)
#             np.fill_diagonal(Aori,0)
#             Aori= Aori*(-1)
#             graph_train_list.append(nx.from_numpy_array(Aori)) 


#         degree, cluster, orbit, unique, novel, spectral = 0,0,0,0,0,0
#         if len(graph_pred_list_remove_empty)>0:
#             degree = degree_stats( graph_test_list,graph_pred_list_remove_empty, compute_emd=False)
#             cluster = clustering_stats( graph_test_list,graph_pred_list_remove_empty, compute_emd=False)
# #             orbit = orbit_stats_all(graph_test_list, graph_pred_list_remove_empty, compute_emd=False)
# #             unique,novel,_ = eval_fraction_unique_non_isomorphic_valid(graph_pred_list_remove_empty,graph_train_list)
#             spectral = spectral_stats(graph_test_list, graph_pred_list_remove_empty)

        train_set.get_extra_data(False)
        test_set.get_extra_data(False)
        
        return degree, cluster, orbit, unique, novel, spectral
    
        
    
######################################################################################### Score Model


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

    
    
class Transformer(Module):
      def __init__(self, point_dim, residual=False, args=None):
          
        super().__init__()
        embdim = args.latent_dim
        
        self.act = F.leaky_relu
        
        self.init_layer_PHI = nn.Sequential(nn.Linear(point_dim,32),nn.LeakyReLU(),nn.Linear(32,embdim))
        self.init_layer_LAM = nn.Sequential(nn.Conv1d(1,32,3,padding=1),nn.LeakyReLU(),nn.Conv1d(32,embdim,3,padding=1))
        
        
        self.layers = ModuleList([ SelfCrossAttention(embdim,point_dim,4) for i in range(args.layers)])
       
        self.final_layer_PHI = nn.Sequential(nn.Linear(embdim,embdim),nn.LeakyReLU(),nn.Linear(embdim,point_dim))
        self.final_layer_LAM = nn.Sequential(nn.Conv1d(embdim,embdim,3,padding=1),nn.LeakyReLU(),nn.Conv1d(embdim,1,3,padding=1))

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
            
            outx = self.final_layer_PHI(outx)
            outy = self.final_layer_LAM(outy.transpose(-1,-2))
            
        
            outfinal = xy+torch.cat([outx,outy],-2) 
            outfinal = outfinal# / self.marginal_prob_std(beta)      

            return outfinal    
    