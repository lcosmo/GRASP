import torch
import wandb
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from .common import *
from .diffusion import *

import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy
import networkx as nx
from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, eval_fraction_unique, eval_fraction_unique_non_isomorphic_valid, spectral_stats

import matplotlib.pyplot as plt
import plotly.graph_objects as go
    
class GaussianVAE(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        self.args = self.hparams
        
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=self.args.point_dim, residual=self.args.residual, args=self.args),
            var_sched = VarianceSchedule(
                num_steps=self.args.num_steps,
                beta_1=self.args.beta_1,
                beta_T=self.args.beta_T,
                mode=self.args.sched_mode
            )
        )
        
        self.automatic_optimization = False
        
    def get_loss(self, x, m, writer=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        loss_recons = self.diffusion.get_loss(x, m)
        loss = loss_recons

#         if writer is not None:
# #             writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
# #             writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
#             writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, num_points, batch_size, flexibility, point_dim=10, truncate_std=None, device='cuda'):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
#         if truncate_std is not None:
#             z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        
        samples = self.diffusion.sample(num_points, batch_size, flexibility=flexibility, point_dim=point_dim, device=device)
        return samples

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.args.lr
        )

        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=self.args.sched_start_epoch,
            end_epoch=self.args.sched_end_epoch,
            start_lr=self.args.lr,
            end_lr=self.args.end_lr
        )

        return {"optimizer": optimizer,  "lr_scheduler":scheduler}

    
    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        x = batch[0].float()
        y = batch[1].float()
        m = batch[2].float()

        x = torch.cat([x,y[:,None,:]],-2).float()#.to(args.device)

        # Reset grad and model state
        optimizer.zero_grad()

        # Forward
        if not self.hparams.use_mask:
            m=1
        
        
        loss = self.get_loss(x, m=m)

        # Backward and optimize
        loss.backward()

        orig_grad_norm = clip_grad_norm_(self.parameters(), self.hparams.max_grad_norm)
        optimizer.step()
        scheduler.step()

        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True)
        
    def validation_step(self, batch, batch_idx):
        pass
    
    def validation_epoch_end(self, outputs):
        if self.trainer.train_dataloader is None:
            return
        
        train_set = self.trainer.train_dataloader.dataset.datasets
        test_set = self.trainer.val_dataloaders[0].dataset

        degree, cluster, orbit, unique, novel, spectral = self.evaluate(train_set, test_set, device=self.diffusion.net.init_layer_PHI[0].bias.device)
        
        self.log('degree', torch.tensor(degree).float(), on_step=False, on_epoch=True)
        self.log('cluster',  torch.tensor(cluster).float(), on_step=False, on_epoch=True)
        self.log('orbit',  torch.tensor(orbit).float(), on_step=False, on_epoch=True)
        self.log('unique',  torch.tensor(unique).float(), on_step=False, on_epoch=True)
        self.log('novel',  torch.tensor(novel).float(), on_step=False, on_epoch=True)
        self.log('spectral',  torch.tensor(spectral).float(), on_step=False, on_epoch=True)

    def sample_graphs(self, max_nodes, num_eigs, unscale_xy, num_graphs=1024, device='cuda'):
        # Sample eigenvectors and eigenvalues
        gen_pcs = []
        with torch.no_grad():
            x = self.sample(max_nodes, num_graphs, 1, point_dim=num_eigs, device=device)
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

        graph_pred_list, orth_pred_list = self.sample_graphs(max_nodes=n, num_eigs=d, unscale_xy=train_set.unscale_xy, num_graphs=1024, device=device)
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
    
    
    
    
    
    
    
    
