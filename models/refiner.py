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

class Refiner(L.LightningModule):

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

        self.discriminator = PPGNDiscriminator(alpha=0.2, n_max=args.n_max, n_layers=args.discriminator_layers, data_channels=args.discriminator_data_channels,
                                        use_spectral_norm=True, normalization='instance', gelu=True,
                                        k_eigval=args.k, dropout=0, cat_eigvals=False, cat_mult_eigvals=False,
                                        partial_laplacian=False, no_cond=False,
                                        qm9=args.qm9, data_channels_mult=1)


        self.criterion = nn.BCEWithLogitsLoss()
        
        self.automatic_optimization = False
        
        #start with reconstruction loss warmup
        self.train_dicriminator = False 
        self.train_generator = False 
        
        self.last_tot_dis_loss = 0
        
    
    def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), 
#             lr=self.hparams.lr
#         )

        optimizerD = torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr*1e-1)
        optimizerG = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.lr)
        optimizer = [optimizerD,optimizerG]

#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps = int(self.hparams.train_loop_batches*200),
#             num_training_steps = self.hparams.train_loop_batches*self.hparams.max_epochs
#         )

        return optimizer#{"optimizer": optimizer,  "lr_scheduler":scheduler}

    
    def training_step(self, batch, batch_idx):
        generator, discriminator = self.generator, self.discriminator
        criterion = self.criterion
        optimizerD, optimizerG = self.optimizers()
#         scheduler = self.lr_schedulers()
        
        batch, gen = batch
                
        noisy_real_eigval = batch[1][:,:self.hparams.k]#.to(device) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_real_eigvec = batch[0][:,:,:self.hparams.k]#.to(device) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_real_node_features = batch[0][:,:,self.hparams.k:] #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_adj = batch[2]#.to(device)
        
        real_edge_features = batch[3]
        
        #masks computation
        emask_real = (noisy_real_eigval!=0).float() #<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
            
        mask_real = (noisy_real_eigvec.abs().sum(-1) > 1e-5)#.to(device)
        mask_half = mask_real[:,:].float()
        
        noisy_real_eigval = noisy_real_eigval + emask_real*torch.randn_like(noisy_real_eigval)*1e-1 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_real_eigvec = noisy_real_eigvec + emask_real[:,None,:]*torch.randn_like(noisy_real_eigvec)*3e-2 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<        
        noisy_real_eigvec[torch.logical_not(mask_real)] = 0
        noisy_real_node_features = noisy_real_node_features + torch.randn_like(noisy_real_node_features)*1e-4 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<        
        
        num_gt = noisy_real_eigval.shape[0]
        noisy_gen_eigval = torch.cat([noisy_real_eigval, gen[1][:,0,:self.hparams.k]],0)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_gen_eigvec = torch.cat([noisy_real_eigvec, gen[0][:,:,:self.hparams.k]],0)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        noisy_gen_node_features = torch.cat([noisy_real_node_features, gen[0][:,:,self.hparams.k:]],0)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        #masks computation
        mask_gen = (noisy_gen_eigvec.abs().sum(-1) > 1e-5)#.to(device)
        noisy_gen_eigvec[torch.logical_not(mask_gen)] = 0
        mask = mask_gen[:,:].float()
    
        #normalize 
        noisy_gen_eigvec = noisy_gen_eigvec/noisy_gen_eigvec.norm(dim=1)[:,None,:]
        noisy_real_eigvec = noisy_real_eigvec/noisy_real_eigvec.norm(dim=1)[:,None,:]
        
        device = mask.device
    
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        tot_dis_loss = float('nan')
        if self.train_dicriminator:
            discriminator.train()
            generator.eval()
            optimizerD.zero_grad()

            #generate fake_data
            with torch.no_grad(): #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< new gen 
                noise =  torch.randn(list(mask.shape[:2])+[generator.latent_dim-self.hparams.feature_size], device=device)*1e-2
                noise = torch.cat([noise,noisy_gen_node_features],-1)
                fake_adj, fake_node_features, fake_edge_features = generator(noise, noisy_gen_eigval, noisy_gen_eigvec, mask)
            
            fake_label = torch.zeros((mask[num_gt:].shape[0],), device=mask.device)
            if fake_node_features is not None:
                fake_pred = discriminator(noisy_gen_eigval[num_gt:], noisy_gen_eigvec[num_gt:], mask[num_gt:], fake_adj[num_gt:], node_features=fake_node_features[num_gt:], edge_features=fake_edge_features[num_gt:])
            else:
                fake_pred = discriminator(noisy_gen_eigval[num_gt:], noisy_gen_eigvec[num_gt:], mask[num_gt:], fake_adj[num_gt:])
            fake_loss = criterion(fake_pred[:,0],fake_label)
            fake_loss.backward()                                   
            fake_loss = fake_loss.item()

            true_label = torch.ones((mask_half.shape[0],), device=mask.device)

            if self.hparams.disc_ori:
                true_adj = noisy_adj
            else:
                true_adj = fake_adj[:num_gt]
            
            if fake_node_features is not None:
                true_pred = discriminator(noisy_gen_eigval[:num_gt], noisy_gen_eigvec[:num_gt], mask[:num_gt], true_adj, node_features=fake_node_features[:num_gt], edge_features=fake_edge_features[:num_gt])
            else:
                true_pred = discriminator(noisy_gen_eigval[:num_gt], noisy_gen_eigvec[:num_gt], mask[:num_gt], true_adj)
            true_loss = criterion(true_pred[:,0],true_label)
            true_loss.backward()            
            
#             true_label = torch.ones((mask_half.shape[0],), device=mask.device)
#             true_pred = discriminator(noisy_gen_eigval[:num_gt], noisy_gen_eigvec[:num_gt], mask[:num_gt], fake_adj[:num_gt], node_features=noisy_real_node_features*0, edge_features=real_edge_features*0)
#             true_loss = criterion(true_pred[:,0],true_label)
#             true_loss.backward()            
            true_loss = true_loss.item()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            optimizerD.step()
            
            tot_dis_loss=(fake_loss+true_loss)/2
            self.log('dis_loss', tot_dis_loss, on_step=False, on_epoch=True)
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################  
        discriminator.eval()
        generator.train()
        optimizerG.zero_grad()
        
        #generate fake_data
        noise =  torch.randn(list(mask.shape[:2])+[generator.latent_dim-self.hparams.feature_size], device=device)*1e-2
        noise = torch.cat([noise,noisy_gen_node_features],-1)
        fake_adj, fake_node_features, fake_edge_features = generator(noise, noisy_gen_eigval, noisy_gen_eigvec, mask)

        true_label = torch.ones((noisy_gen_eigval[num_gt:].shape[0],), device=mask.device)
        if fake_node_features is not None:
            fake_pred = discriminator(noisy_gen_eigval[num_gt:], noisy_gen_eigvec[num_gt:], mask[num_gt:], fake_adj[num_gt:], node_features=fake_node_features[num_gt:], edge_features=fake_edge_features[num_gt:])
        else:
            fake_pred = discriminator(noisy_gen_eigval[num_gt:], noisy_gen_eigvec[num_gt:], mask[num_gt:], fake_adj[num_gt:])
            
        gen_loss = criterion(fake_pred[:,0],true_label)
                
        rec_loss = torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_adj[:num_gt],noisy_adj)
        if fake_node_features is not None:
            rec_loss = rec_loss + torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_node_features[:num_gt],  batch[0][:,:,self.hparams.k:])
            rec_loss = rec_loss + (torch.nn.functional.torch.nn.functional.binary_cross_entropy(fake_edge_features[:num_gt], real_edge_features,reduction="none")*noisy_adj[...,None]).sum()/noisy_adj[...,None].sum()
        
        if self.train_generator:
            genrec_loss = gen_loss + 1e0*rec_loss
        else:
            gen_loss = torch.tensor(0)
            genrec_loss = rec_loss
            
        genrec_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
        optimizerG.step()
        
        tot_gen_loss=gen_loss.item()
        tot_rec_loss=rec_loss.item()
        
        self.log('gen_loss', tot_gen_loss, on_step=False, on_epoch=True)
        self.log('rec_loss', tot_rec_loss, on_step=False, on_epoch=True)
        
#         self.log('lr', self.optimizers()[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        return {'tot_gen_loss':tot_gen_loss, 'tot_rec_loss':tot_rec_loss, 'tot_dis_loss':tot_dis_loss}

        
    def training_epoch_end(self, outputs):
#         print(outputs)
        tot_gen_loss = sum([o['tot_gen_loss'] for o in outputs])
        tot_dis_loss = sum([o['tot_dis_loss'] for o in outputs])
        tot_rec_loss = sum([o['tot_rec_loss'] for o in outputs])
        
        if tot_dis_loss!=tot_dis_loss:
            tot_dis_loss = self.last_tot_dis_loss

        self.train_generator = self.current_epoch>=50
        self.train_dicriminator =  self.current_epoch>=40  and \
                                   (tot_gen_loss<=1.5*tot_dis_loss or self.current_epoch%100==0)
        self.last_tot_dis_loss = tot_dis_loss
        
#         self.log('TOT_gen_loss', tot_gen_loss, on_step=False, on_epoch=True)
#         self.log('TOT_dis_loss', tot_dis_loss, on_step=False, on_epoch=True)
#         self.log('TOT_rec_loss', tot_rec_loss, on_step=False, on_epoch=True)
        self.log('discriminating', self.train_dicriminator, on_step=False, on_epoch=True)
        
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self, outputs):
        if self.trainer.train_dataloader is None:
            return
        
        ori_train_set = self.trainer.val_dataloaders[0].dataset.datasets[0]
        gen_test_set = self.trainer.val_dataloaders[0].dataset.datasets[1]

        degree, cluster,  unique, novel, spectral, degree_degrad, cluster_degrad, spectral_degrad, avg_degrad = self.evaluate(ori_train_set, gen_test_set, device=self.generator.powerful.in_lin[0].bias.device)
        
        self.log('degree', torch.tensor(degree).float(), on_step=False, on_epoch=True)
        self.log('cluster',  torch.tensor(cluster).float(), on_step=False, on_epoch=True)
        # self.log('orbit',  torch.tensor(orbit).float(), on_step=False, on_epoch=True)
        self.log('unique',  torch.tensor(unique).float(), on_step=False, on_epoch=True)
        self.log('novel',  torch.tensor(novel).float(), on_step=False, on_epoch=True)
        self.log('spectral',  torch.tensor(spectral).float(), on_step=False, on_epoch=True)

        self.log('degree_degrad', torch.tensor(degree_degrad).float(), on_step=False, on_epoch=True)
        self.log('cluster_degrad',  torch.tensor(cluster_degrad).float(), on_step=False, on_epoch=True)
        self.log('spectral_degrad',  torch.tensor(spectral_degrad).float(), on_step=False, on_epoch=True)
        self.log('avg_degrad',  torch.tensor(avg_degrad).float(), on_step=False, on_epoch=True)
        
    def evaluate(self, train_set, test_set, device='cuda'):
        train_set.get_extra_data(True)
#         test_set.get_extra_data(True)
                       
        batch = next(iter(DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0)))
        b,n,d = batch[0].shape
        
        noisy_gen_eigvec =  batch[0][:,:,:self.hparams.k].to(device)
        noisy_gen_eigval =  batch[1][:,0,:self.hparams.k].to(device)
        noisy_gen_node_features = batch[0][:,:,self.hparams.k:].to(device)

        mask = noisy_gen_eigvec.abs().sum(-1)>1e-5


        with torch.no_grad():
            noise =  torch.randn(list(mask.shape[:2])+[self.generator.latent_dim-self.hparams.feature_size], device=device)*1e-2
            noise = torch.cat([noise,noisy_gen_node_features],-1)
            fake_adj, fake_node_features, fake_edge_features = self.generator(noise, noisy_gen_eigval, noisy_gen_eigvec, mask)

#             score = -self.discriminator( noisy_gen_eigval, noisy_gen_eigvec, mask, fake_adj,node_features=fake_node_features, edge_features=fake_edge_features).cpu()
            fake_adj = fake_adj.cpu()

        graph_pred_list = []
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

        degree_degrad = degree/train_set.degree
        cluster_degrad = cluster/train_set.cluster
        spectral_degrad = spectral/train_set.spectral

        avg_degrad = (degree_degrad + cluster_degrad + spectral_degrad)/3
#         train_set.get_extra_data(False)
#         test_set.get_extra_data(False)
        
        return degree, cluster, unique, novel, spectral, degree_degrad, cluster_degrad, spectral_degrad, avg_degrad
    
    
    
    
    
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
    
    