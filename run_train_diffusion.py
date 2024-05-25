import os
from itertools import product
import random

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["WANDB_MODE"]="online"
# datasetname='community_12_21_100'
datasetname='planar_64_200'
# datasetname='sbm_200'
datasetname='proteins'
# datasetname='zinc'
# 
# ks = [8,16,63]
ks = [16]
# ks = [4,16, 32]
layers = [8]
latent_dims = [512]



params = list(product(ks,layers,latent_dims)) 
random.shuffle(params) 
params

b_size = 128
for k, s_layer, s_dim in params:

    if datasetname=='proteins':
        b_size=32
        if s_dim>256:
            b_size=64

    if datasetname=='planar_64_200':
        b_size=128
        if s_dim>256:
            b_size=128
        
    print(f'python train_diffusion.py --dataset={datasetname} --layers={s_layer} --latent_dim={s_dim} --k={k} --batch_size={b_size} --scaler=standard')
    os.system(f'python train_diffusion.py --dataset={datasetname} --layers={s_layer} --latent_dim={s_dim} --k={k} --batch_size={b_size} --scaler=standard --smallest=True')