import torch
from torch import nn
import os

import tqdm
from tqdm import tqdm

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import pickle
import re
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from torch_geometric.data import Batch, Data

import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList, Linear

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

from scipy.sparse.csgraph import laplacian
from numpy import linalg as LA

from collections import defaultdict
import math 
import sklearn

from utils.eval_helper import degree_stats, clustering_stats, orbit_stats_all, eval_fraction_unique, eval_fraction_unique_non_isomorphic_valid, spectral_stats
 
 
# def minmax_norm(value,maxval, minval):
#     a = -1
#     b = 1
#     if (maxval-minval)==0:
#         print('AHIII')
#     normalized_val = ((b-a)*((value-minval)/(maxval-minval)))+a
#     return normalized_val

# def deminmax_norm(N,maxval, minval):
#     a = -1
#     b = 1
#     V = ((maxval-minval)*((N-a)/(b-a)))+minval
#     return V

# def standa(value, mu, std):
#     stad_v = (value-mu)/std
#     return stad_v

# def destanda(val, mu, std):
#     n = (val*std)+mu
#     return n
    

class LaplacianDatasetNX(Dataset):
    
    def __init__(self,_folder,filename,point_dim=7, smallest=True, split='all'):
        
        self.point_dim = point_dim
        self.samples = []
        print('Point dim {}'.format(self.point_dim))
        self.label = []
#         folder = '/home/lcosmo/GIORGIA/'
                
        with open( filename+'.pkl', "rb") as f:
#         with open(folder+'/'+filename+'.pkl', "rb") as f:
             graph_list = pickle.load(f)
        
        print('Comp dimensions...')
        
        indices = []     
        
        maxdimensions = 1
        mindimensions = point_dim
        for ids in range(len(graph_list)):     
            
            nodelist = graph_list[ids].number_of_nodes()   
            
            if  nodelist<point_dim:
                   continue   
            else:
                              
                indices.append(ids)                
                maxdimensions = max(maxdimensions,(nodelist))

        print('Comp stats...')
        ori_eigenvalues = []       
        eigen_dic = defaultdict(dict)
        
        
        try:
            eigen_dic = torch.load(filename+'.eigen')
            print("Loaded precomuted eigenquantities")
        
        except:

            for ids,indiceori in enumerate(indices):

                H = graph_list[indiceori].copy()

                dims = (nx.adjacency_matrix(H)).shape    
                adj = nx.adjacency_matrix(H).todense()
                lap = laplacian(adj)                
                w, v = LA.eig(lap)

                if any(np.iscomplex(w)):
                    w = np.array( [bb.real for bb in w])

                eigen_dic[ids]['w']=w
                eigen_dic[ids]['v']=v
                eigen_dic[ids]['dims']=dims
                eigen_dic[ids]['lap']=lap
                eigen_dic[ids]['A']=adj


#                 eigva_ids_sort = w.argsort()[::-1]# descending order  
#                 eigva = w[eigva_ids_sort]        
#                 eigva = eigva[:point_dim] 
#                 ori_eigenvalues.append(eigva)
            
            torch.save(eigen_dic,filename+'.eigen')
        
        self.mu = 0#mu
        self.std = 1#std
        
        print('Comp Samples...')
        
        ids_list = eigen_dic.keys()
        
        
         
        for ids,indiceori in enumerate(indices): 
            label = []
 
            if ids not in ids_list:
                          print('ERRRRRROR')
                          quit()
            else:

                w = eigen_dic[ids]['w']
                v = eigen_dic[ids]['v']
                dims = eigen_dic[ids]['dims']
                lap = eigen_dic[ids]['lap'] 
                A = eigen_dic[ids]['A']
                
                

                eigva_ids_sort = w.argsort()[::-1]# descending order  
                eigva = w[eigva_ids_sort]
                
                if not smallest:
                    eigva = eigva[:point_dim]
                else:
                    eigva = eigva[-(1+point_dim):-1]
                
                eigva_norm = eigva#np.zeros(eigva.shape)

#                 for jj,value in enumerate(eigva):                      
#                       eigva_norm[jj] = minmax_norm(value, maxval[jj], minval[jj] ) 

                if not smallest:
                    eigvec = v[:,eigva_ids_sort][:,:point_dim] # v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
                else:
                    eigvec = v[:,eigva_ids_sort][:,-point_dim:] # v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
                    
                num_zeros = maxdimensions-dims[0]   
                n_nodes = dims[0]

                arr_pad = np.pad(lap, [(0, num_zeros), (0, num_zeros)], mode='constant')  
                
                eigvec = np.pad(eigvec, [(0, num_zeros), (0, 0)], mode='constant')    
                A = np.pad(A, [(0, num_zeros), (0, num_zeros)], mode='constant')  
                
                self.samples.append((torch.tensor(label).float(),torch.tensor(eigvec).float(),
                                     torch.tensor(eigva_norm).float(),torch.tensor(arr_pad).float(),
                                     torch.tensor(n_nodes),torch.tensor(A).float()))
                
        print('Tot #{}'.format(len(ids_list))  )
        
        #train test
        test_len = int(len(self.samples)*0.2)
        train_len = len(self.samples) - test_len
        train_set, test_set = random_split(torch.arange(len(self.samples)), [train_len, test_len], generator=torch.Generator().manual_seed(1234))

        #rescale data
        train_evecs = torch.stack([self.samples[i][1] for i in train_set],0)
        train_evals = torch.stack([self.samples[i][2] for i in train_set],0)
        
        Lscaler = sklearn.preprocessing.StandardScaler()
        Lscaler.fit(train_evals)
        
        Wscaler = sklearn.preprocessing.StandardScaler()
        Wscaler.fit(train_evecs.reshape(-1,train_evecs.shape[-1]))
                
        self.wm = torch.tensor(Wscaler.mean_)[None,:].float()
        self.ws = torch.tensor(Wscaler.var_)[:].float()**0.5

        self.lm = torch.tensor(Lscaler.mean_)[:].float()
        self.ls = torch.tensor(Lscaler.var_)[:].float()**0.5
        
        self.n_max = self.samples[0][1].shape[0]
        self.n_dist = np.histogram([int(self.samples[i][4]) for i in train_set],self.n_max+1,range=(0,self.n_max+1),density=True)[0]

        if split == 'train':
            self.compute_mmd_statistics([self.samples[i] for i in train_set], [self.samples[i] for i in test_set])
            self.samples = [self.samples[i] for i in train_set]
        else:
            if split == 'test':
                self.samples = [self.samples[i] for i in test_set]
            else:
                assert split=='all'
                
            
        self.extra_data = False
        
    def compute_mmd_statistics(self,train_set,test_set):
        #compute metrics
        graph_test_list = [] #should be on test set graphs
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

        self.degree = degree_stats( graph_test_list,graph_train_list, compute_emd=False)
        self.cluster = clustering_stats( graph_test_list,graph_train_list, compute_emd=False)
        self.spectral = spectral_stats(graph_test_list, graph_train_list)

        
    def __len__(self):
        return len(self.samples)

    
    def scale_xy(self,x,y):
        wm_,ws_,lm_,ls_ = [t.to(x.device) for t in [self.wm,self.ws,self.lm,self.ls]]
        x = (x-wm_)/ws_
        y = (y-lm_)/ls_
        return x,y

    def unscale_xy(self,x,y):
        wm_,ws_,lm_,ls_ = [t.to(x.device) for t in [self.wm,self.ws,self.lm,self.ls]]
        x = x*ws_ + wm_
        y = y*ls_ + lm_
        return x,y
    
    def sample_n_nodes(self, n):
        return np.random.choice(self.n_max+1, n, p=self.n_dist)
        
    def get_extra_data(self, flag=True):
        self.extra_data = flag
        
    def __getitem__(self, idx):
        
        class_id, eigevc_tensor, eigva_tensor, lap_tensor,n_nodes,A = self.samples[idx] 
        
        eigevc_tensor,eigva_tensor = self.scale_xy(eigevc_tensor,eigva_tensor)

        if self.extra_data:
            return eigevc_tensor, eigva_tensor, class_id[None,:],lap_tensor,n_nodes,A[None,:]
        
        m = torch.zeros(eigevc_tensor.shape[0])
        m[:n_nodes] = 1
        
        return eigevc_tensor, eigva_tensor, m#,A[None,:]
    
# import sys, inspect
# def print_classes():
#     for name, obj in inspect.getmembers(sys.modules[__name__]):
#         if inspect.isclass(obj):
#             print(obj)
# print_classes()
def n_community(num_communities, max_nodes, p_inter=0.05):
    assert num_communities > 1
    
    one_community_size = max_nodes // num_communities
 
    c_sizes = [one_community_size] * num_communities
    total_nodes = one_community_size * num_communities
    
    """ 
    here we calculate `p_make_a_bridge` so that `p_inter = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes `
    
    To make it more clear: 
    let `M = num_communities` and `N = one_community_size`, then
    
    ```
    p_inter
    = \mathbb{E}(Number_of_bridge_edges) / Total_number_of_nodes
    = (p_make_a_bridge * C_M^2 * N^2) / (MN)  # see the code below for this derivation
    = p_make_a_bridge * (M-1) * N / 2
    ```
    
    so we have:
    """
    p_make_a_bridge = p_inter * 2 / ((num_communities - 1) * one_community_size)
    
#     print(num_communities, total_nodes, end=' ')
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]

    G = nx.disjoint_union_all(graphs)
#     communities = list(nx.connected_component_subgraphs(G))
    communities = list(G.subgraph(c) for c in nx.connected_components(G))
    add_edge = 0
    for i in range(len(communities)):
        
        subG1 = communities[i]
         
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):  # loop for C_M^2 times
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:  # loop for N times
                for n2 in nodes2:  # loop for N times
                    if np.random.rand() < p_make_a_bridge:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1
#     print('connected comp: ', len(list(nx.connected_component_subgraphs(G))),
#           'add edges: ', add_edge)
#     print(G.number_of_edges())
    return G