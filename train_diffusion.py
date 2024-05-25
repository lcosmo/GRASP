import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import argparse
import torch
import torch.utils.tensorboard 
from torch.utils.data import DataLoader, TensorDataset

from utils.misc import seed_all
from dataset.load_data_generated import LaplacianDatasetNX

from models.diffusion import SpectralDiffusion
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def get_arg_parser():
    THOUSAND = 1000
    parser = argparse.ArgumentParser()
    
    # Diffusion arguments
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])#not_used

    # Optimizer and scheduler
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=500*THOUSAND)

    # Training
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--val_check_interval', type=int, default=5000)
    parser.add_argument('--log_every_n_steps', type=int, default=500)
    parser.add_argument('--wandb', type=eval, default=True, choices=[True, False])

    #Dataset
    parser.add_argument('--model_tag', type=str, default='self-cross-hugg')
    parser.add_argument('--dataset', type=str, default='community_12_21_100')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--smallest', type=eval, default=False, choices=[True, False])
    parser.add_argument('--scaler', type=str, default="standard") #stadard if not specified (standard or minmax)

    #Score Model
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--use_mask', type=eval, default=True, choices=[True, False])

    return parser

if __name__ == "__main__":   
    args = get_arg_parser().parse_args()
    args.point_dim=args.k
    
    seed_all(args.seed)    

    #load training and validation data
    train_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train_train', scaler=args.scaler, nodefeatures=args.dataset in ["qm9","zinc"], device="cpu")
    valid_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train_val', scaler=args.scaler, nodefeatures=args.dataset in ["qm9","zinc"])
    
    train_set.get_extra_data(False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=False)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, num_workers=0,pin_memory=False)

    run_params=args
    args.train_loop_batches = len(train_loader)
    args.max_epochs = args.max_epochs//args.train_loop_batches + 1
    args.feature_size = train_set[0][0].shape[-1]-args.k
    
    model = SpectralDiffusion(args)

    # trainer = pl.Trainer(limit_train_batches=100, max_epochs=10,accelerator="auto")
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='loss',
        min_delta=-1e-1,
        patience=10,
        verbose=False,
        mode='min')

    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.model_tag}_k-{args.k}_sm-{args.smallest}_l-{args.layers}_d-{args.latent_dim}",
            project="graph_diffusion_perceptron_2",
            entity="l_cosmo"
        )
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=None,
        val_check_interval = args.val_check_interval,
        max_epochs = args.max_epochs
    )

    trainer.fit(model, train_loader, valid_loader)


