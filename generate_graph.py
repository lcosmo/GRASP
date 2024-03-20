import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time 
import math
import torch
import scipy
import networkx as nx
import gdown
import tarfile 

from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from models.predictor import Predictor
from models.diffusion import SpectralDiffusion
from dataset.load_data_generated import LaplacianDatasetNX

from utils.visualization import NonMolecularVisualization

if not os.path.isdir("./data/model_weights/"):
    gdown.download(id="19mC9gQCpoecBGWaL3__v69Ixu0Rny22r", output="data/model_weights.tar.gz")
    tarfile.open('./data/model_weights.tar.gz') .extractall('./data/') 
    
diffusion_model_checkpoint = 'data/model_weights/diffusion_sbm_200.ckpt'
predictor_model_checkpoint = 'data/model_weights/predictor_sbm_200.ckpt'

diffusion_model_checkpoint = 'data/model_weights/diffusion_planar_64_200.ckpt'
predictor_model_checkpoint = 'data/model_weights/predictor_planar_64_200.ckpt'

# diffusion_model_checkpoint = 'data/model_weights/diffusion_proteins.ckpt'
# predictor_model_checkpoint = 'data/model_weights/predictor_proteins.ckpt'

device = 'cuda'
n_graphs = 10
sampling_steps = 200

model_predictor = Predictor.load_from_checkpoint(predictor_model_checkpoint, strict=False).generator
model_diffusion = SpectralDiffusion.load_from_checkpoint(diffusion_model_checkpoint, strict=False)
model_predictor.eval()
model_diffusion.eval()
args = model_diffusion.hparams

model_predictor.to(device)
model_diffusion.to(device)

#load training set and graph's size distribution
datasetname = model_diffusion.hparams.dataset
train_set = LaplacianDatasetNX(datasetname,'data/'+datasetname,point_dim=args.k, smallest=args.smallest, split='train')
n_nodes = list(train_set.sample_n_nodes(n_graphs-1)) + [train_set.n_max]

start = time.time()
with torch.no_grad():
    #generate 
    xx,yy = model_diffusion.sample_eigs(n_nodes, args.k, scale_xy=train_set.scale_xy, unscale_xy=train_set.unscale_xy, 
                              reproject=False,sampling_steps=sampling_steps)
    
    #predict
    mask = xx.abs().sum(-1)>1e-6
    xx = xx/(xx.norm(dim=1)[:,None,:]+1e-12)

    inno =  torch.randn(list(mask.shape[:2])+[model_predictor.latent_dim], device=args.device)*0
    fake_adj,_,_ = model_predictor(inno, yy[:,0], xx, mask)

    #extract graphs
    LLLall_ =[]
    di=0
    graph_pred_list = []
    for i, A in enumerate(fake_adj.cpu()):
        A = (A>0.5).float()
        mask = A.sum(-1)>0
        if sum(mask)==0:
            continue;
        G = nx.Graph(A[mask,:][:,mask].numpy())
        graph_pred_list.append(G)
        
end = time.time()
print(f"Generated {n_graphs} graphs in {time.time()-start} seconds")

#saving graphs here
NonMolecularVisualization().visualize(f'results/{datasetname}/',graph_pred_list,10)