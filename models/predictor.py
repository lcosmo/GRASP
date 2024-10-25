import torch
import wandb
from torch.nn import Module

from .diffusion import SpectralDiffusion

import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from models.ema import ExponentialMovingAverage
from models.ppgn_gan import PPGNGenerator, PPGNDiscriminator
from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, eval_fraction_unique, eval_fraction_unique_non_isomorphic_valid, spectral_stats

from utils .molecular_eval import BasicMolecularMetrics
from rdkit import Chem

from utils.misc import create_vis
import copy

def get_masks(self,x,y):
    mask  = x.abs().sum(-1)[...,None]>1e-8
    emask = y.abs() > 1e-8
    return mask, emask

def gen_noise(x,s=2e-2):
    if x.numel()==0:
        return 0
    return x.abs().max()*torch.randn_like(x)*s
    
class Predictor(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        args = self.hparams

        if not hasattr(args,'generator_noise_latent_dim') or args.generator_noise_latent_dim is None:
            args.generator_noise_latent_dim = args.k

        if not hasattr(args,'normalized'):
            args.normalized = True
            
        #########################
        self.generator_train = PPGNGenerator(alpha=0.2, n_max=args.n_max, noise_latent_dim=args.generator_noise_latent_dim, 
                                    n_layers=args.generator_layers, data_channels=args.generator_data_channels, 
                                    gelu=True, k_eigval=args.k, use_fixed_emb=False, normalization='instance',
                                    dropout=0,
                                    skip_connection=True,
                                    cat_eigvals=False, cat_mult_eigvals=False, no_extra_n=True,
                                    no_cond=False, init_emb_channels=args.generator_init_emb_channels, qm9=args.qm9,
                                    data_channels_mult=1, normalized=args.normalized)

        self.discriminator = PPGNDiscriminator(alpha=0.2, n_max=args.n_max, n_layers=args.discriminator_layers, 
                                        data_channels=args.discriminator_data_channels,
                                        use_spectral_norm=True, normalization='instance', gelu=True,
                                        k_eigval=args.k, dropout=0, cat_eigvals=False, cat_mult_eigvals=False,
                                        partial_laplacian=False, no_cond=False,
                                        qm9=args.qm9, data_channels_mult=1)

        self.generator = self.generator_train
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.automatic_optimization = False
        
        #start with reconstruction loss warmup
        self.train_dicriminator = False 
        self.train_generator = False 
        
        self.last_tot_dis_loss = 0
        self.training_step_outputs = []
        
        self.ema = ExponentialMovingAverage(self.generator.parameters(), decay=0.99)
        
        if self.hparams.dataset=='qm9':
            self.molecular_metrics = None
        
    
    def configure_optimizers(self):
        optimizerD = torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr*1e-1)
        optimizerG = torch.optim.AdamW(self.generator_train.parameters(), lr=self.hparams.lr)
        optimizer = [optimizerD,optimizerG]

        return optimizer
    
    def training_step(self, batch, batch_idx):
        generator, discriminator = self.generator_train, self.discriminator
        criterion = self.criterion
        optimizerD, optimizerG = self.optimizers()
        
        batch, gen = batch
                
        noisy_real_eigval = batch[1][:,:self.hparams.k]
        noisy_real_eigvec = batch[0][:,:,:self.hparams.k].clone()
        noisy_real_node_features = batch[0][:,:,self.hparams.k:].clone()
        noisy_adj = batch[2]
        
        real_edge_features = batch[3]
        
        #masks computation
        emask_real = (noisy_real_eigval.abs() > 1e-8).float()  
            
        mask_real = noisy_real_eigvec.abs().sum(-1) > 1e-5
        mask_half = mask_real[:,:].float()
        
        noisy_real_eigval = noisy_real_eigval + emask_real*gen_noise(noisy_real_eigval)
        noisy_real_eigvec = noisy_real_eigvec + emask_real[:,None,:]*gen_noise(noisy_real_eigvec)       
        noisy_real_eigvec[torch.logical_not(mask_real)] = 0   
        
        num_gt = noisy_real_eigval.shape[0]
        noisy_gen_eigval = torch.cat([noisy_real_eigval, gen[1][:,0,:self.hparams.k]],0)
        noisy_gen_eigvec = torch.cat([noisy_real_eigvec, gen[0][:,:,:self.hparams.k]],0)
        noisy_gen_node_features = torch.cat([noisy_real_node_features, gen[0][:,:,self.hparams.k:]],0)

        #masks computation
        mask_gen = (noisy_gen_eigvec.abs().sum(-1) > 1e-5)#.to(device)
        noisy_gen_eigvec[torch.logical_not(mask_gen)] = 0
        mask = mask_gen[:,:].float()
    
        #normalize 
        noisy_gen_eigvec = noisy_gen_eigvec/(noisy_gen_eigvec.norm(dim=1)[:,None,:]+1e-12)
        noisy_real_eigvec = noisy_real_eigvec/(noisy_real_eigvec.norm(dim=1)[:,None,:]+1e-12)
        
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
                noise =  torch.zeros(list(mask.shape[:2])+[generator.latent_dim-self.hparams.feature_size], device=device)
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
                # true_pred = discriminator(noisy_gen_eigval[:num_gt], noisy_gen_eigvec[:num_gt], mask[:num_gt], true_adj, node_features=noisy_real_node_features[:num_gt], edge_features=real_edge_features[:num_gt])
            else:
                true_pred = discriminator(noisy_gen_eigval[:num_gt], noisy_gen_eigvec[:num_gt], mask[:num_gt], true_adj)
            true_loss = criterion(true_pred[:,0],true_label)
            true_loss.backward()                       
            true_loss = true_loss.item()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5)
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
        noise =  torch.zeros(list(mask.shape[:2])+[generator.latent_dim-self.hparams.feature_size], device=device)
        noise = torch.cat([noise,noisy_gen_node_features],-1)
        fake_adj, fake_node_features, fake_edge_features = generator(noise, noisy_gen_eigval, noisy_gen_eigvec, mask)

        true_label = torch.ones((noisy_gen_eigval[num_gt:].shape[0],), device=mask.device)
        if fake_node_features is not None:
            fake_pred = discriminator(noisy_gen_eigval[num_gt:], noisy_gen_eigvec[num_gt:], mask[num_gt:], fake_adj[num_gt:], node_features=fake_node_features[num_gt:], edge_features=fake_edge_features[num_gt:])
        else:
            fake_pred = discriminator(noisy_gen_eigval[num_gt:], noisy_gen_eigvec[num_gt:], mask[num_gt:], fake_adj[num_gt:])
            
        gen_loss = criterion(fake_pred[:,0],true_label)

        avg_denisty = noisy_adj.mean([-1,-2],keepdims=True)
        weight = noisy_adj*(1-avg_denisty) + avg_denisty*(1-noisy_adj)
        
        
        if fake_node_features is not None:
            rec_loss = (torch.nn.functional.cross_entropy(fake_node_features[:num_gt].permute([0,2,1]), noisy_real_node_features.argmax(-1),reduction="none")*mask_real).sum()/mask_real.sum()
            rec_loss = rec_loss + torch.nn.functional.cross_entropy(\
                torch.cat([(1-fake_adj[...,None]),fake_edge_features],-1)[:num_gt].permute([0,3,1,2]),\
                torch.cat([(1-noisy_adj[...,None]),real_edge_features],-1).argmax(-1))
        else:
            rec_loss = torch.nn.functional.binary_cross_entropy(fake_adj[:num_gt],noisy_adj)#,weight=weight)            
        
        if self.train_generator:
            genrec_loss = gen_loss + self.hparams.rec_weight*rec_loss
        else:
            gen_loss = torch.tensor(0)
            genrec_loss = rec_loss
            
        genrec_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
        optimizerG.step()
        
        tot_gen_loss=gen_loss.item()
        tot_rec_loss=rec_loss.item()
        
        self.log('gen_loss', tot_gen_loss, on_step=False, on_epoch=True)
        self.log('rec_loss', tot_rec_loss, on_step=False, on_epoch=True)

        self.training_step_outputs.append({'tot_gen_loss':tot_gen_loss, 'tot_rec_loss':tot_rec_loss, 'tot_dis_loss':tot_dis_loss})
#         self.log('lr', self.optimizers()[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        return self.training_step_outputs[-1]


    
    def on_train_epoch_end(self):
        #ema
        self.ema.update(self.generator.parameters())

        outputs = self.training_step_outputs
        tot_gen_loss = sum([o['tot_gen_loss'] for o in outputs])
        tot_dis_loss = sum([o['tot_dis_loss'] for o in outputs])
        tot_rec_loss = sum([o['tot_rec_loss'] for o in outputs])
        
        if tot_dis_loss!=tot_dis_loss:
            tot_dis_loss = self.last_tot_dis_loss

        self.train_generator = self.current_epoch>=50
        self.train_dicriminator =  self.current_epoch>=45  and \
                                   (tot_gen_loss<=1.5*tot_dis_loss or self.current_epoch%100==0)
        self.last_tot_dis_loss = tot_dis_loss
        self.training_step_outputs.clear()
        self.log('discriminating', self.train_dicriminator, on_step=False, on_epoch=True)
        
    def validation_step(self, batch, batch_idx):
        pass
    
    def on_validation_epoch_end(self):
        if self.trainer.train_dataloader is None:
            return

        ###ema###
        self.ema.store(self.generator.parameters())
        self.ema.copy_to(self.generator.parameters())
                    
        ori_train_set = self.trainer.val_dataloaders.dataset.datasets[0]
        ori_val_set = self.trainer.val_dataloaders.dataset.datasets[1]
        gen_test_set = self.trainer.val_dataloaders.dataset.datasets[2]

        if self.hparams.dataset=='qm9':
            valid,unique,novel = self.evaluate(ori_train_set, ori_val_set, gen_test_set, device=self.generator.powerful.in_lin[0].bias.device)
            self.log('unique',  torch.tensor(unique).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('novel',  torch.tensor(novel).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('valid',  torch.tensor(valid).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('avg_degrad',  torch.tensor(1-valid*unique*novel).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
        else:
            degree, cluster,  unique, novel, spectral, degree_degrad, cluster_degrad, spectral_degrad, avg_degrad = self.evaluate(ori_train_set, ori_val_set, gen_test_set, device=self.generator.powerful.in_lin[0].bias.device)
            
            self.log('degree', torch.tensor(degree).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('cluster',  torch.tensor(cluster).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            # self.log('orbit',  torch.tensor(orbit).float(), on_step=False, on_epoch=True)
            self.log('unique',  torch.tensor(unique).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('novel',  torch.tensor(novel).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('spectral',  torch.tensor(spectral).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
    
            self.log('degree_degrad', torch.tensor(degree_degrad).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('cluster_degrad',  torch.tensor(cluster_degrad).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('spectral_degrad',  torch.tensor(spectral_degrad).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
            self.log('avg_degrad',  torch.tensor(avg_degrad).float().cuda(), on_step=False, on_epoch=True, sync_dist=True)
        
        ###ema###
        self.ema.restore(self.generator.parameters())
        
    def evaluate(self, train_set, val_set, test_set, device='cuda'):
        train_set.get_extra_data(True)
        val_set.get_extra_data(True)

        ##############################
        batch = next(iter(DataLoader(self.trainer.train_dataloader.dataset, batch_size=2, shuffle=False, num_workers=0)))[0]
        
        noisy_real_eigvec =  batch[0][:,:,:self.hparams.k].to(device)
        noisy_real_eigval =  batch[1][:,:self.hparams.k].to(device)
        noisy_real_node_features = batch[0][:,:,self.hparams.k:].to(device)
        _mask = noisy_real_eigvec.abs()>1e-8
        _emask = noisy_real_eigval.abs()>1e-8

        
        noisy_real_eigvec = torch.cat([noisy_real_eigvec,noisy_real_eigvec+_mask*gen_noise(noisy_real_eigvec)],0)
        noisy_real_eigval = torch.cat([noisy_real_eigval,noisy_real_eigval+_emask*gen_noise(noisy_real_eigval)],0)
        noisy_real_node_features = torch.cat([noisy_real_node_features,noisy_real_node_features],0)
        mask = (noisy_real_eigvec.abs().sum(-1) > 1e-5)#.to(device)
        
        with torch.no_grad():
            noise = torch.zeros(list(mask.shape[:2])+[self.generator.latent_dim-self.hparams.feature_size], device=device)
            noise = torch.cat([noise,noisy_real_node_features],-1)
            fake_adj, fake_node_features, fake_edge_features = self.generator(noise, noisy_real_eigval, noisy_real_eigvec, mask)

        lap = (noisy_real_eigvec*noisy_real_eigval[:,None])@noisy_real_eigvec.transpose(-2,-1)
        for i in range(2):
            L1 = -lap[i]
            L1.fill_diagonal_(0)
            L2 = -lap[i+2]
            L2.fill_diagonal_(0)
            images = [wandb.Image(batch[2][i], caption="Ori"),
                      wandb.Image(L1, caption="Lap"),
                      wandb.Image(fake_adj[i], caption="Rec"),
                      wandb.Image(L2, caption="LapN"),
                      wandb.Image(fake_adj[i+2], caption="RecN")]                      
            self.trainer.logger.experiment.log({f"recon {i}": images})

        ############################        
        graph_pred_list = []
        all_adj = []
        all_node_features = []
        all_edge_features = []
        for batch in DataLoader(test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0):            
            b,n,d = batch[0].shape
            
            noisy_gen_eigvec =  batch[0][:,:,:self.hparams.k].to(device)
            noisy_gen_eigval =  batch[1][:,0,:self.hparams.k].to(device)
            noisy_gen_node_features = batch[0][:,:,self.hparams.k:].to(device)
    
            mask = noisy_gen_eigvec.abs().sum(-1)>1e-5
            
            with torch.no_grad():
                noise =  torch.randn(list(mask.shape[:2])+[self.generator.latent_dim-self.hparams.feature_size], device=device)*1e-2*0
                noise = torch.cat([noise,noisy_gen_node_features],-1)
                fake_adj, fake_node_features, fake_edge_features = self.generator(noise, noisy_gen_eigval, noisy_gen_eigvec, mask)
                fake_adj = fake_adj.cpu()
     
                all_adj.append((fake_adj>0.5).float().cpu()) #Bxnxn
                all_node_features.append(fake_node_features.argmax(-1).cpu()) #Bxn
                all_edge_features.append(fake_edge_features.argmax(-1).cpu()) #Bxnxn

                for i, A in enumerate(fake_adj.cpu()):
                    A = (A>0.5).float()
                    mask = A.sum(-1)>0
                    G = nx.Graph(A[mask,:][:,mask].numpy())
                    if fake_node_features is not None:
                        nx.set_node_attributes(G,{i:j.argmax(-1).item() for i,j in enumerate(fake_node_features[i])},'x')
                    graph_pred_list.append(G)

                del fake_edge_features
                del fake_node_features                

        all_adj = torch.cat(all_adj,0)
        all_node_features = torch.cat(all_node_features,0)
        all_edge_features = torch.cat(all_edge_features,0)
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
        graph_test_list = [] #should be on val set graphs
        for jj in range(min(2048,len(val_set))):
            laplacian_matrix = np.array(train_set[jj][-3].cpu())[:train_set[jj][-2],:train_set[jj][-2]]
            Aori = np.copy(laplacian_matrix)
            np.fill_diagonal(Aori,0)
            Aori= Aori*(-1)
            graph_test_list.append(nx.from_numpy_array(Aori)) 

        graph_train_list = []
        for jj in range(min(2048,len(train_set))):
            laplacian_matrix = np.array(train_set[jj][-3].cpu())[:train_set[jj][-2],:train_set[jj][-2]]
            Aori = np.copy(laplacian_matrix)
            np.fill_diagonal(Aori,0)
            Aori= Aori*(-1)
            graph_train_list.append(nx.from_numpy_array(Aori))

        if self.hparams.dataset=='qm9':
            def get_cc(A):
                G = nx.Graph(A[1].numpy())
                c = list(list(sorted(nx.connected_components(G), key=len, reverse=True))[0])
                
                return A[0][c], A[1][c,:][:,c], A[2][c,:][:,c]

            if self.molecular_metrics is None:
                atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}       #  Warning: hydrogens have been removed
                bond_dict = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
                self.molecular_metrics = BasicMolecularMetrics(atom_dict, bond_dict, train_set, strict=False)
                
            # gen_good = [get_cc(A) for A in list(zip(all_node_features, all_adj, all_edge_features)) if nx.is_connected(nx.Graph(A[1].numpy()))]
            gen_good = list(zip(all_node_features, all_adj, all_edge_features))
            
            (valid,unique,novel),_ = self.molecular_metrics.evaluate(gen_good)
            return valid,unique,novel
        else:    
            #remove highly connected graphs that would slow down the metrics computation (this is done only during training)
            graph_pred_list_remove_empty = [g for g in graph_pred_list_remove_empty if np.mean(nx.adjacency_matrix(g))<0.33]
    
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
    
            if len(graph_pred_list_remove_empty)<15:
                avg_degrad =  1e3
            
            return degree, cluster, unique, novel, spectral, degree_degrad, cluster_degrad, spectral_degrad, avg_degrad