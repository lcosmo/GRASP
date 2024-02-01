import torch
import wandb
from torch.nn import Module

from .common import *
from .diffusion import *

import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy
import networkx as nx

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from models.ppgn_gan import PPGNGenerator, PPGNDiscriminator

from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, eval_fraction_unique, eval_fraction_unique_non_isomorphic_valid, spectral_stats

import copy

class Refiner2(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        args = self.hparams

        if not hasattr(args,'generator_noise_latent_dim') or args.generator_noise_latent_dim is None:
            args.generator_noise_latent_dim = args.k
            
        #########################
        self.generator = PPGNGenerator(alpha=0.2, n_max=args.n_max, noise_latent_dim=args.generator_noise_latent_dim, n_layers=args.generator_layers, data_channels=args.generator_data_channels, 
                                  gelu=True, k_eigval=args.k, use_fixed_emb=False, normalization='instance',
                                    dropout=0,
                                    skip_connection=True,
                                    cat_eigvals=False, cat_mult_eigvals=False, no_extra_n=True,
                                    no_cond=False, init_emb_channels=args.generator_init_emb_channels, qm9=args.qm9,
                                    data_channels_mult=1)


        self.criterion = nn.BCEWithLogitsLoss()        
        self.automatic_optimization = False

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.lr)
   
        return optimizer#{"optimizer": optimizer,  "lr_scheduler":scheduler}

    
    def training_step(self, batch, batch_idx):
        generator = self.generator
        criterion = self.criterion
        optimizer = self.optimizers()
#         scheduler = self.lr_schedulers()
       
                
        noisy_real_eigval = batch[1][:,:self.hparams.k]#.to(device) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_real_eigvec = batch[0][:,:,:self.hparams.k]#.to(device) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_real_node_features = batch[0][:,:,self.hparams.k:] #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_adj = batch[2]#.to(device)
        
        real_edge_features = batch[3]
        
        #masks computation
        emask_real = (noisy_real_eigval!=0).float() #<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
            
        mask_real = (noisy_real_eigvec.abs().sum(-1) > 1e-5)#.to(device)
       
        noisy_real_eigval = noisy_real_eigval + emask_real*torch.randn_like(noisy_real_eigval)*1e-3 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_real_eigvec = noisy_real_eigvec + emask_real[:,None,:]*torch.randn_like(noisy_real_eigvec)*1e-3 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<        
        noisy_real_eigvec[torch.logical_not(mask_real)] = 0
        noisy_real_node_features = noisy_real_node_features + torch.randn_like(noisy_real_node_features)*1e-4 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<     
        noisy_real_eigvec = noisy_real_eigvec/noisy_real_eigvec.norm(dim=1)[:,None,:]
        device = mask_real.device
          
        
        optimizer.zero_grad()
        
        #generate fake_data
        noise =  torch.randn(list(mask_real.shape[:2])+[generator.latent_dim-self.hparams.feature_size], device=device)*0
        noise = torch.cat([noise,noisy_real_node_features],-1)
        fake_adj, fake_node_features, fake_edge_features = generator(noise, noisy_real_eigval, noisy_real_eigvec, mask_real)
                
        rec_loss = torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_adj,noisy_adj)
        if fake_node_features is not None:
            rec_loss = rec_loss + torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_node_features,  batch[0][:,:,self.hparams.k:])
            rec_loss = rec_loss + (torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_edge_features, real_edge_features,reduction="none")*noisy_adj[...,None]).sum()/noisy_adj[...,None].sum()
        
            
        rec_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
        optimizer.step()
        
        self.log('rec_loss', rec_loss, on_step=False, on_epoch=True)
        
        return {'rec_loss': rec_loss}

        
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self, outputs):
        if self.trainer.train_dataloader is None:
            return
                    
        ori_train_set = gen_test_set = self.trainer.val_dataloaders[0].dataset.datasets[0]
        gen_test_set = self.trainer.val_dataloaders[0].dataset.datasets[1]

        degree, cluster,  unique, novel, spectral, rec_loss = self.evaluate(ori_train_set, gen_test_set, device=self.generator.powerful.in_lin[0].bias.device)
        
        self.log('degree', torch.tensor(degree).float(), on_step=False, on_epoch=True)
        self.log('cluster',  torch.tensor(cluster).float(), on_step=False, on_epoch=True)
        # self.log('orbit',  torch.tensor(orbit).float(), on_step=False, on_epoch=True)
        self.log('unique',  torch.tensor(unique).float(), on_step=False, on_epoch=True)
        self.log('novel',  torch.tensor(novel).float(), on_step=False, on_epoch=True)
        self.log('spectral',  torch.tensor(spectral).float(), on_step=False, on_epoch=True)
        self.log('val_rec_loss',  torch.tensor(rec_loss).float(), on_step=False, on_epoch=True)

#         self.log('degree_degrad', torch.tensor(degree_degrad).float(), on_step=False, on_epoch=True)
#         self.log('cluster_degrad',  torch.tensor(cluster_degrad).float(), on_step=False, on_epoch=True)
#         self.log('spectral_degrad',  torch.tensor(spectral_degrad).float(), on_step=False, on_epoch=True)
#         self.log('avg_degrad',  torch.tensor(avg_degrad).float(), on_step=False, on_epoch=True)
        
    def evaluate(self, train_set, test_set, device='cuda'):
        train_set.get_extra_data(True)
#         test_set.get_extra_data(True)
                       
        dl = DataLoader(test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0)
        
        total_rec_loss=0
        graph_pred_list = []            
        for batch in dl:
            b,n,d = batch[0].shape

            
            noisy_gen_eigvec =  batch[0][:,:,:self.hparams.k].to(device)
            noisy_gen_eigval =  batch[1][:,0,:self.hparams.k].to(device)
            noisy_gen_node_features = batch[0][:,:,self.hparams.k:].to(device)
            noisy_adj = batch[2].to(device)
            
            mask = noisy_gen_eigvec.abs().sum(-1)>1e-5


            with torch.no_grad():
                noise =  torch.randn(list(mask.shape[:2])+[self.generator.latent_dim-self.hparams.feature_size], device=device)*1e-2
                noise = torch.cat([noise,noisy_gen_node_features],-1)
                fake_adj, fake_node_features, fake_edge_features = self.generator(noise, noisy_gen_eigval, noisy_gen_eigvec, mask)

    #             score = -self.discriminator( noisy_gen_eigval, noisy_gen_eigvec, mask, fake_adj,node_features=fake_node_features, edge_features=fake_edge_features).cpu()
#                 fake_adj = fake_adj.cpu()

                rec_loss = torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_adj,noisy_adj)
                if fake_node_features is not None:
                    rec_loss = rec_loss + torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_node_features,  batch[0][:,:,self.hparams.k:])
                    rec_loss = rec_loss + (torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_edge_features, real_edge_features,reduction="none")*noisy_adj[...,None]).sum()/noisy_adj[...,None].sum()
                total_rec_loss+=rec_loss.item()*b
            
            fake_adj = fake_adj.cpu()
            for i, A in enumerate(fake_adj.cpu()):
                A = (A>0.5).float()
                mask = A.sum(-1)>0
                G = nx.Graph(A[mask,:][:,mask].numpy())
                if fake_node_features is not None:
                    nx.set_node_attributes(G,{i:j.argmax(-1).item() for i,j in enumerate(fake_node_features[i])},'x')
                graph_pred_list.append(G)

        graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0][:]


        try:
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
                for img in fake_adj[:3]]
                })
        except:
            pass

        
        #compute metrics
        graph_test_list = [] #should be on test set graphs
        for jj in range(len(train_set)):
            laplacian_matrix = np.array(train_set[jj][-3].cpu())[:train_set[jj][-2],:train_set[jj][-2]]
            Aori = np.copy(laplacian_matrix)
            np.fill_diagonal(Aori,0)
            Aori= Aori*(-1)
            graph_test_list.append(nx.from_numpy_array(Aori)) 

        graph_train_list = graph_test_list
        # for jj in range(len(train_set)):
        #     laplacian_matrix = np.array(train_set[jj][3].cpu())[:train_set[jj][4],:train_set[jj][4]]
        #     Aori = np.copy(laplacian_matrix)
        #     np.fill_diagonal(Aori,0)
        #     Aori= Aori*(-1)
        #     graph_train_list.append(nx.from_numpy_array(Aori)) 


        degree, cluster, orbit, unique, novel, spectral = 0,0,0,0,0,0
        if len(graph_pred_list_remove_empty)>0:
            degree = degree_stats( graph_test_list,graph_pred_list_remove_empty, compute_emd=False)
            cluster = clustering_stats( graph_test_list,graph_pred_list_remove_empty, compute_emd=False)
            # orbit = orbit_stats_all(graph_test_list, graph_pred_list_remove_empty, compute_emd=False)
            unique,novel,_ = eval_fraction_unique_non_isomorphic_valid(graph_pred_list_remove_empty,graph_train_list)
            spectral = spectral_stats(graph_test_list, graph_pred_list_remove_empty)

#         degree_degrad = degree/train_set.degree
#         cluster_degrad = cluster/train_set.cluster
#         spectral_degrad = spectral/train_set.spectral

#         avg_degrad = (degree_degrad + cluster_degrad + spectral_degrad)/3
#         train_set.get_extra_data(False)
#         test_set.get_extra_data(False)
        
        return degree, cluster, unique, novel, spectral,total_rec_loss/len(test_set)
    
    
    
    
    
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
    
    