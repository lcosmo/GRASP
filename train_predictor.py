import os
import glob
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sklearn
from sklearn import preprocessing
import torch
import random
torch.set_float32_matmul_precision('medium')

from dataset.load_data_generated import LaplacianDatasetNX
from torch.utils.data import DataLoader, TensorDataset

# import sklearn.preprocessing 
from models.diffusion import SpectralDiffusion
from models.predictor import Predictor
from utils.misc import seed_all

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, shuffle=-1):
        self.datasets = datasets
        self.shuffle = [False]*len(datasets)
        if shuffle is not None:
            self.shuffle[shuffle] = True

    def __getitem__(self, i):
        return tuple(d[random.randint(1,len(d))-1] if s else d[i]  for d,s in zip(self.datasets,self.shuffle))

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
def get_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--diffusion_model', type=str, default="", required=True)
    parser.add_argument('--diffusion_path', type=str, default="graph_diffusion_perceptron_2", required=False)
    parser.add_argument('--resume', type=str, default=None, required=False)

    # Predictor arguments
    parser.add_argument('--generator_layers', type=int, default=8)
    parser.add_argument('--generator_data_channels', type=int, default=32)
    parser.add_argument('--generator_init_emb_channels', type=int, default=64)
    parser.add_argument('--generator_noise_latent_dim', type=int, default=2)
    parser.add_argument('--discriminator_layers', type=int, default=4)
    parser.add_argument('--discriminator_data_channels', type=int, default=32)
    parser.add_argument('--rec_weight', type=float, default=1e-1)

    
    # Diffusion generation
    parser.add_argument('--n_graphs_train', type=int, default=512)
    parser.add_argument('--n_graphs_test', type=int, default=256)
    parser.add_argument('--reproject', type=eval, default=True, choices=[True, False])
    parser.add_argument('--disc_ori', type=eval, default=False, choices=[True, False])
    parser.add_argument('--normalized', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sampling_steps', type=int, default=100)#500
    
    #optimization
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_validation', type=eval, default=True, choices=[True, False])
    
    parser.add_argument('--max_epochs', type=int, default=500000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--val_check_interval', type=int, default=1000)#500
    parser.add_argument('--wandb', type=eval, default=True, choices=[True, False])    
    
    return parser

if __name__ == "__main__":   
    args = get_arg_parser().parse_args()
    seed_all(args.seed)
  
    #################### load diffusion_model ##############################
    diffusion_model_path = glob.glob(f'{args.diffusion_path}/{args.diffusion_model}/checkpoints/epoch*.ckpt')[-1]
    model = SpectralDiffusion.load_from_checkpoint(diffusion_model_path)

    model.hparams.update(args.__dict__)
    args = model.hparams
    args.qm9 = args.dataset[:3] in ["qm9"]

    ################### load real graphs training set ######################    
    if args.use_validation:
        graphs_train_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train_train', nodefeatures=args.dataset[:3] in ["qm9"])
        graphs_val_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train_val', nodefeatures=args.dataset[:3] in ["qm9"])
    else:
        graphs_train_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train', nodefeatures=args.dataset[:3] in ["qm9"])
        graphs_val_set = graphs_train_set
    
    graphs_test_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='test', nodefeatures=args.dataset[:3] in ["qm9"])
    
    graphs_train_set.get_extra_data()
    real_eval = torch.stack([t[1] for t in graphs_train_set],0)
    real_evec = torch.stack([t[0] for t in graphs_train_set],0)
    real_adj = torch.stack([t[-1][0] for t in graphs_train_set],0)
    
    real_emask = torch.stack([t[3] for t in graphs_train_set],0)
    real_edge_features = torch.stack([t[4] for t in graphs_train_set],0)

    real_evec,real_eval = graphs_train_set.unscale_xy(real_evec,real_eval)
    real_evec *= real_emask[:,None,:] 
    real_eval *= real_emask   

    real_evec[:,:,args.k:] = torch.min(torch.ones_like(real_evec[:,:,args.k:]), 
                                       torch.max(torch.zeros_like(real_evec[:,:,args.k:]),real_evec[:,:,args.k:]))
    


    train_set = torch.utils.data.TensorDataset(real_evec,real_eval,real_adj,real_edge_features)
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0,pin_memory=True)
        
    ############### generate graphs with diffusion #######################
    model.to(args.device)

    n_graphs_tot = args.n_graphs_train + args.n_graphs_test

    max_gen=2048
    if args.dataset=='proteins':
        max_gen = 256

    generations_x_=[]
    generations_y_=[]
    for i in range(n_graphs_tot//max_gen+1):
        n_graphs = min(max_gen,n_graphs_tot)
        n_nodes = list(graphs_train_set.sample_n_nodes(n_graphs-1)) + [graphs_train_set.n_max]
        
        generations_x,generations_y = model.sample_eigs(max_nodes=n_nodes, num_eigs=args.k+args.feature_size, scale_xy=graphs_train_set.scale_xy, unscale_xy=graphs_train_set.unscale_xy, device=args.device, num_graphs=n_graphs, reproject=args.reproject, sampling_steps=args.sampling_steps)
        generations_x_.append(generations_x.cpu())
        generations_y_.append(generations_y.cpu())
        del generations_x
        del generations_y
        
    generations_x=torch.cat(generations_x_,0)
    generations_y=torch.cat(generations_y_,0)
    
    
    generations_dataset = torch.utils.data.TensorDataset(generations_x[:args.n_graphs_train],generations_y[:args.n_graphs_train])
    generations_dataset_val = torch.utils.data.TensorDataset(generations_x[args.n_graphs_train:],generations_y[args.n_graphs_train:])

    del model
    # torch.save([generations_dataset,generations_dataset_val],"tmp.data")

    dataloader = DataLoader(ConcatDataset(train_set,generations_dataset), batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)
    val_dataloader = DataLoader(ConcatDataset(graphs_train_set,graphs_val_set,generations_dataset_val,shuffle=None), batch_size=args.batch_size, shuffle=False, num_workers=0,pin_memory=True)

    ###################################
    args.n_max = graphs_train_set.n_max
    if args.resume is not None:
        ref = Predictor.load_from_checkpoint(args.resume, strict=True)
    else:
        ref = Predictor(args)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='avg_degrad',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='avg_degrad',
        min_delta=0,
        patience=2000,
        verbose=False,
        mode='min')

    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.model_tag}_k-{args.k}_sm-{args.smallest}_dm-{args.diffusion_model}",
            project="graph_diffusion_predictor",
            offline=False
        )
    else:
        wandb_logger = None
    
    args.check_val_every_n_epoch = None                   
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=len(dataloader),
        check_val_every_n_epoch = None,
        val_check_interval = args.val_check_interval, 
        max_epochs = args.max_epochs
    )

    if args.resume is not None:
        trainer.fit(ref, dataloader, val_dataloader, ckpt_path = args.resume)
    else:
        trainer.fit(ref, dataloader, val_dataloader)

# aaaa

