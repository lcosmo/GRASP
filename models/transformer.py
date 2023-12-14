import torch
import wandb
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from .common import *
from .encoders import *
from .diffusion import *

import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy
import networkx as nx
from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, eval_fraction_unique, eval_fraction_unique_non_isomorphic_valid, spectral_stats

import matplotlib.pyplot as plt
import plotly.graph_objects as go


from diffusers import UNet2DModel, UNet3DConditionModel
from .diffusion import PointwiseNet
from diffusers import DDIMPipeline, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
  
class Transformer(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        self.args = self.hparams
        
        self.diffusion = PointwiseNet(self.args.k, args = self.args)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False,beta_schedule='linear')
        
        self.automatic_optimization = False
        
    def get_loss(self, x, y, m, writer=None):
          
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
        noise_pred = self.diffusion(noisy_x, timesteps/1000.0, m)

        assert(not torch.isnan(noise_pred).any())
        m = torch.nn.functional.pad(m,(0,1),value=1)[...,None]

        loss = (F.mse_loss(noise_pred, noise,reduction='none')*m).sum()/(m.sum()*noise.shape[-1])

        return loss
    

    def sample(self, num_points, batch_size, flexibility, point_dim, scale_xy, unscale_xy, sampling_steps=100, device='cuda',reproject=True):
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

        m_all = noise*0+1        
        if isinstance(num_points,list):        
            for i,n in enumerate(num_points):
                m_all[i,n:-1,:] = 0                
        m = m_all[:,:-1,0]
        
        scheduler.set_timesteps(sampling_steps)
        for t in scheduler.timesteps:
            with torch.no_grad():
                t_ = t[None].repeat(noise.shape[0]).to(noise.device)
                noisy_residual = self.diffusion(noise, t_/1000, m)
            

        #     reproject to orthonormal bases
            if reproject and t<sampling_steps*0.1:
                M = scheduler.step(noisy_residual, t, noise).pred_original_sample*m_all
                M,yy = unscale_xy(M[:,:-1,:],M[:,-1,:])
                
                svd = torch.svd(M)
                orth = svd.U@svd.V.transpose(-1,-2)
                grad = scale_xy(orth-M,yy)[0]
                previous_noisy_sample[:,:-1,:] += 1e-2*grad.to(previous_noisy_sample.device)
            else:
                previous_noisy_sample = scheduler.step(noisy_residual, t, noise).prev_sample

            noise = previous_noisy_sample*m_all
    
        ################################
        return noise

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.args.lr
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(self.hparams.train_loop_batches*200),
            num_training_steps = self.hparams.train_loop_batches*self.hparams.max_epochs
        )

        return {"optimizer": optimizer,  "lr_scheduler":scheduler}

    
    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        x = batch[0].float()
        y = batch[1].float()
        m = batch[2].float()
        
        

        # Forward
#         if not self.hparams.use_mask:
#             m=1
        loss = self.get_loss(x, y, m=m)

        # Backward and optimize
        loss.backward()

#         orig_grad_norm = clip_grad_norm_(self.parameters(), self.hparams.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Reset grad and model state
        optimizer.zero_grad()
        
        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True)
        
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

        
    def sample_eigs(self, max_nodes, num_eigs, scale_xy, unscale_xy, num_graphs=1024, device='cuda', reproject=True):
        # Sample eigenvectors and eigenvalues
        gen_pcs = []
        with torch.no_grad():
            x = self.sample(max_nodes, num_graphs, 1, num_eigs, scale_xy, unscale_xy, device=device, reproject=reproject)
            samples_EIGVEC = x.detach().cpu()        
            xx = samples_EIGVEC[:,:-1,:]
            yy = samples_EIGVEC[:,-1:,:]
            
            xx,yy = unscale_xy(xx,yy)
            yy=yy.float()
        return xx,yy
    
    
    def sample_graphs(self, max_nodes, num_eigs, scale_xy, unscale_xy, num_graphs=1024, device='cuda'):
        # Sample eigenvectors and eigenvalues
        gen_pcs = []
        with torch.no_grad():
            x = self.sample(max_nodes, num_graphs, 1, num_eigs, scale_xy, unscale_xy, device=device)
            samples_EIGVEC = x.detach().cpu()

        # reconstruct laplacian matrix
        recon_list = []
        for i,X in enumerate(samples_EIGVEC.cpu()):
            xx = X[:-1,:]
            yy = X[-1:,:]

            xx,yy = unscale_xy(xx,yy)
            yy=yy.float()

            xx_o=xx

            #orthogonal projection
            if False:
                U,s,V = torch.svd(xx)
                xx_o = (U@V).float()

            #discard samples that did not led to a quasi-orthonormal basis
            L = (xx_o*yy)@xx_o.t()  
            err = ((xx_o.t())@xx_o-torch.eye(xx.shape[-1])).norm()

            recon_list.append((err,L.numpy(),xx_o.t()@xx_o))


        LLLall  = [l[1] for l in sorted(recon_list, key=lambda e:e[0])][:]
        LLLorth = [l[2] for l in sorted(recon_list, key=lambda e:e[0])][:]

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
        
        
        
        batch = next(iter(DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0)))
        b,n,d = batch[0].shape

        graph_pred_list, orth_pred_list = self.sample_graphs(max_nodes=n, num_eigs=d, scale_xy = train_set.scale_xy, unscale_xy=train_set.unscale_xy, num_graphs=128, device=device)
        graph_pred_list = graph_pred_list[:b]

        graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

#         #log to wandb
#         plt.figure()
#         nx.draw_kamada_kawai(graph_pred_list[0])
#         self.trainer.logger.experiment.log({"generated_graph_1": plt})
        
#         plt.figure()
#         nx.draw_kamada_kawai(graph_pred_list[1])
#         self.trainer.logger.experiments.log({"generated_graph_2": plt})
        
#         plt.figure()
#         plt.imshow(orth_pred_list[0])
#         plt.colorbar()
#         self.trainer.logger.experiments.log({"generated_orth_1": plt})
        
#         plt.figure()
#         plt.imshow(orth_pred_list[1])
#         plt.colorbar()
#         self.trainer.logger.experiments.log({"generated_orth_2": plt})
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

        
        #compute metrics
        graph_test_list = []
        for jj in range(len(test_set)):
            laplacian_matrix = np.array(test_set[jj][3].cpu())[:test_set[jj][4],:test_set[jj][4]]
            Aori = np.copy(laplacian_matrix)
            np.fill_diagonal(Aori,0)
            Aori= Aori*(-1)
            graph_test_list.append(nx.from_numpy_array(Aori)) 

        graph_train_list = []
        for jj in range(len(train_set)):
            laplacian_matrix = np.array(train_set[jj][3].cpu())[:train_set[jj][4],:train_set[jj][4]]
            Aori = np.copy(laplacian_matrix)
            np.fill_diagonal(Aori,0)
            Aori= Aori*(-1)
            graph_train_list.append(nx.from_numpy_array(Aori)) 


        degree, cluster, orbit, unique, novel, spectral = 0,0,0,0,0,0
        if len(graph_pred_list_remove_empty)>0:
            degree = degree_stats( graph_test_list,graph_pred_list_remove_empty, compute_emd=False)
            cluster = clustering_stats( graph_test_list,graph_pred_list_remove_empty, compute_emd=False)
            orbit = orbit_stats_all(graph_test_list, graph_pred_list_remove_empty, compute_emd=False)
            unique,novel,_ = eval_fraction_unique_non_isomorphic_valid(graph_pred_list_remove_empty,graph_train_list)
            spectral = spectral_stats(graph_test_list, graph_pred_list_remove_empty)

        
        train_set.get_extra_data(False)
        test_set.get_extra_data(False)
        
        return degree, cluster, orbit, unique, novel, spectral
    
    
    
    
    
    
    
#########################################################################################
import networkx as nx
def create_vis(G):
#     G = to_networkx(graph)
    pos = nx.kamada_kawai_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='none',
        line_width=2
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout())
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
    
    
    
    