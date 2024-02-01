import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import sklearn
from sklearn import preprocessing

from utils.misc import *
from utils.utils_score import *
from dataset.load_data_generated import *

from dataset.load_data_generated import *
import sklearn.preprocessing 

from models.transformer import Transformer

from models.reconstruction import Refiner2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
def get_arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--diffusion_model', type=str, default="", required=True)

    # GAN arguments
    parser.add_argument('--generator_layers', type=int, default=8)
    parser.add_argument('--generator_data_channels', type=int, default=32)
    parser.add_argument('--generator_init_emb_channels', type=int, default=64)
    parser.add_argument('--generator_noise_latent_dim', type=int, default=128)
    parser.add_argument('--val_check_interval', type=int, default=100)
    
    # Diffusion generation
    parser.add_argument('--n_graphs_train', type=int, default=256)
    parser.add_argument('--n_graphs_test', type=int, default=64)
    
    #optimizaation
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2024)
    
    parser.add_argument('--model_tag', type=str, default='self-cross-hugg')
    parser.add_argument('--dataset', type=str, default='planar')
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--smallest', type=eval, default=False, choices=[True, False])
    
    return parser

if __name__ == "__main__":   
    args = get_arg_parser().parse_args()
#     seed_all(args.seed)
    
    
    
#     #################### load diffusion_model ##############################
#     diffusion_model_path = glob.glob(f'graph_diffusion_perceptron/{args.diffusion_model}/checkpoints/epoch*.ckpt')[0]
#     model = Transformer.load_from_checkpoint(diffusion_model_path)

#     model.hparams.update(args.__dict__)
#     args = model.hparams
    args.qm9 = args.dataset[:3] in ["qm9"]
    

    
    ################### load real graphs training set ######################
    graphs_train_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='train', nodefeatures=args.dataset[:3] in ["qm9"])
    graphs_test_set = LaplacianDatasetNX(args.dataset,'data/'+args.dataset,point_dim=args.k, smallest=args.smallest, split='test', nodefeatures=args.dataset[:3] in ["qm9"])

    graphs_train_set.get_extra_data(False)

    graphs_train_set.get_extra_data()
    real_eval = torch.stack([t[1] for t in graphs_train_set],0)
    real_evec = torch.stack([t[0] for t in graphs_train_set],0)
    real_adj = torch.stack([t[-1][0] for t in graphs_train_set],0)

    real_emask = torch.stack([t[3] for t in graphs_train_set],0)
    real_edge_features = torch.stack([t[4] for t in graphs_train_set],0)

    mask = (real_evec.abs().sum(-1)>1e-5).float()
    real_evec,real_eval = graphs_train_set.unscale_xy(real_evec,real_eval)
    real_evec *= mask[:,:,None] 
    real_evec *= real_emask[:,None,:] 
    real_eval *= real_emask           


    train_set = torch.utils.data.TensorDataset(real_evec,real_eval,real_adj,real_edge_features)
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0,pin_memory=True)

    args.feature_size = graphs_train_set[0][0].shape[-1]-args.k

    
    ################### load real graphs training set ######################
    graphs_test_set.get_extra_data()
    real_eval = torch.stack([t[1] for t in graphs_test_set],0)
    real_evec = torch.stack([t[0] for t in graphs_test_set],0)
    real_adj = torch.stack([t[-1][0] for t in graphs_test_set],0)

    real_emask = torch.stack([t[3] for t in graphs_test_set],0)
    real_edge_features = torch.stack([t[4] for t in graphs_test_set],0)

    mask = (real_evec.abs().sum(-1)>1e-5).float()
    real_evec,real_eval = graphs_train_set.unscale_xy(real_evec,real_eval)
    real_evec *= mask[:,:,None] 
    real_evec *= real_emask[:,None,:] 
    real_eval *= real_emask           
    real_eval = real_eval[:,None]

    val_set = torch.utils.data.TensorDataset(real_evec,real_eval,real_adj,real_edge_features)
    val_dataloader = DataLoader(ConcatDataset(graphs_test_set,val_set), batch_size=args.batch_size//2, shuffle=True, num_workers=0,pin_memory=True)


    
    ###################################
    args.n_max = graphs_train_set.n_max
    ref = Refiner2(args)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_rec_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_rec_loss',
        min_delta=0,
        patience=10,
        verbose=False,
        mode='min')

    wandb_logger = WandbLogger(
        name=f"{args.model_tag}_k-{args.k}_sm-{args.smallest}_dm-{args.diffusion_model}",
        project="graph_diffusion_recon",
        entity="l_cosmo",
        offline=False
    )
    # wandb_logger=None

    args.check_val_every_n_epoch = None
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=len(train_dataloader),
        check_val_every_n_epoch = None,
        val_check_interval = args.val_check_interval, 
        max_epochs = args.max_epochs,
        auto_scale_batch_size="binsearch",
        auto_lr_find=True
    )

    trainer.fit(ref, train_dataloader, val_dataloader)

# aaaa

