import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.misc import *
from utils.utils_score import *
from dataset.load_data_generated import *

import sklearn
from sklearn import preprocessing

from models.transformer import Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def get_arg_parser():
    THOUSAND = 1000
    parser = argparse.ArgumentParser()
    
    # Diffusion arguments
#     parser.add_argument('--model', type=str, default='gaussian', choices=['flow', 'gaussian'])
    parser.add_argument('--num_steps', type=int, default=1000)
#     parser.add_argument('--beta_1', type=float, default=1e-4)
#     parser.add_argument('--beta_T', type=float, default=1-1e-4)
#     parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])#not_used

    # Optimizer and scheduler
    parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
#     parser.add_argument('--end_lr', type=float, default=1e-5)
#     parser.add_argument('--sched_start_epoch', type=int, default=100*THOUSAND)
#     parser.add_argument('--sched_end_epoch', type=int, default=200*THOUSAND)
    parser.add_argument('--max_epochs', type=int, default=100*THOUSAND)

    # Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1000)

    #Dataset & Score Model
    parser.add_argument('--model_tag', type=str, default='self-cross-hugg')
    parser.add_argument('--dataset', type=str, default='sbm_200')
    parser.add_argument('--k', type=int, default=33)
    parser.add_argument('--smallest', type=eval, default=False, choices=[True, False])
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=9)
    parser.add_argument('--use_mask', type=eval, default=True, choices=[True, False])

    return parser

if __name__ == "__main__":   
    args = get_arg_parser().parse_args()
    args.point_dim=args.k
    seed_all(args.seed)
    
    
    train_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train')
    test_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='test')
    
    train_set.get_extra_data(False)
    
#     test_len = int(len(dataset)*0.2)
#     train_len = len(dataset) - test_len
#     train_set, test_set = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(1234))

#     #rescale data
#     data = next(iter(DataLoader(train_set, batch_size=len(train_set), shuffle=False, num_workers=0,pin_memory=False)))

#     Lscaler = sklearn.preprocessing.StandardScaler()
#     Lscaler.fit(data[1])
#     Lscaler.transform(data[1])

#     Wscaler = sklearn.preprocessing.StandardScaler()
#     Wscaler.fit(data[0].reshape(-1,data[0].shape[-1]))
#     Wscaler.transform(data[0].reshape(-1,data[0].shape[-1]))

#     wm = torch.tensor(Wscaler.mean_)[None,:].float()
#     ws = torch.tensor(Wscaler.var_)[:].float()**0.5

#     lm = torch.tensor(Lscaler.mean_)[:].float()
#     ls = torch.tensor(Lscaler.var_)[:].float()**0.5

#     train_set.dataset.set_scale(wm,ws,lm,ls)
#     test_set.dataset.set_scale(wm,ws,lm,ls)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)
    valid_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0,pin_memory=False)

    run_params=args
    args.train_loop_batches = len(train_loader)
    model = Transformer(args)

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
    
    wandb_logger = WandbLogger(
        name=f"{args.model_tag}_k-{args.k}_sm-{args.smallest}_l-{args.layers}_d-{args.latent_dim}",
        project="graph_diffusion_perceptron",
        entity="l_cosmo"
    )

    trainer = pl.Trainer.from_argparse_args(
        run_params,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=250
    )

    trainer.fit(model, train_loader, valid_loader)


