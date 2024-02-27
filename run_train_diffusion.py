import os
from itertools import product
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["WANDB_MODE"]="online"
# datasetname='community_12_21_100'
datasetname='planar_64_200'
# datasetname='sbm_200'
# datasetname='proteins'
# 
ks = [8,16,63]
ks = [32]
layers = [9]
latent_dims = [256]



params = list(product(ks,layers,latent_dims)) 
random.shuffle(params)
params


for k, s_layer, s_dim in params:

    if datasetname=='proteins':
        b_size=32
        if s_dim>256:
            b_size=64

    if datasetname=='planar_64_200':
        b_size=256
        if s_dim>256:
            b_size=128
        
    print(f'python train_diffusion.py --dataset={datasetname} --layers={s_layer} --latent_dim={s_dim} --k={k} --batch_size={b_size} --scaler=standard')
    os.system(f'python train_diffusion.py --dataset={datasetname} --layers={s_layer} --latent_dim={s_dim} --k={k} --batch_size={b_size} --scaler=standard')