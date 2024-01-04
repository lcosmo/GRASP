import pyemd
import numpy as np
import torch
import networkx as nx
import scipy
import concurrent.futures
from functools import partial
from scipy.linalg import toeplitz
from sklearn.metrics import pairwise_distances




def degree_stats(adj_ref_list, adj_pred_list, compute_emd=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of adjacecy matrices to be evaluated
        '''
    degs_ref = [g.sum(-1) for g in adj_ref_list]
    degs_pred  = [g.sum(-1) for g in adj_pred_list]
    
    max_deg = max([max(d) for d in degs_ref]+[max(d) for d in degs_pred])
    
#     train_hist = torch.tensor(np.stack([np.histogram(d,np.arange(+0.5,max_val+1.5),density=True)[0] for d in degs_ref],0)).float()
#     test_hist = torch.tensor(np.stack([np.histogram(d,np.arange(+0.5,max_val+1.5),density=True)[0] for d in degs_pred],0)).float()

    ref_hist = torch.stack([torch.histc(d, bins=int(max_deg)+1,min=-0.5,max=max_deg+0.5) for d in degs_ref],0)#.float()
    pred_hist = torch.stack([torch.histc(d, bins=int(max_deg)+1,min=-0.5,max=max_deg+0.5) for d in degs_pred],0)#.float()
    
#     print("ref_hist")
#     print(ref_hist)
#     print("pred_hist")
#     print(pred_hist)
    
    ref_hist/=ref_hist.sum(-1,keepdims=True)
    pred_hist/=pred_hist.sum(-1,keepdims=True)

    mmd_dist = mmd_tv(ref_hist, pred_hist)

    return mmd_dist

def clustering_stats(adj_ref_list, adj_pred_list, bins=100, compute_emd=False):

    graph_train_list = [nx.from_numpy_array(a.numpy()) for a in adj_ref_list]
    graph_test_list = [nx.from_numpy_array(a.numpy()) for a in adj_pred_list]

    train_hist = np.stack([np.histogram(list(nx.clustering(g).values()), bins=bins, range=(0,1.),density=True)[0]/100 for g in graph_train_list])
    test_hist = np.stack([np.histogram(list(nx.clustering(g).values()), bins=bins, range=(0,1.),density=True)[0]/100 for g in graph_test_list])

    return mmd_tv(torch.tensor(train_hist).float(),torch.tensor(test_hist).float(),sigma=1.0 / 10)

                     

def spectral_stats(adj_ref_list, adj_pred_list, bins=200, compute_emd=False):

    L_train = [ (1/d**0.5)[:,None]*(torch.diag(d)-a)*(1/d**0.5)[None,:] for a,d in zip(adj_ref_list,[a.sum(-1) for a in adj_ref_list])]
    L_test = [ (1/d**0.5)[:,None]*(torch.diag(d)-a)*(1/d**0.5)[None,:] for a,d in zip(adj_pred_list,[a.sum(-1) for a in adj_pred_list])]

    train_hist = torch.stack([torch.histc(torch.tensor(scipy.linalg.eigvalsh(L)), bins=200, min=-1e-5,max=2) for L in L_train])
    test_hist =  torch.stack([torch.histc(torch.tensor(scipy.linalg.eigvalsh(L)), bins=200, min=-1e-5,max=2) for L in L_test])
    train_hist/=train_hist.sum(-1,keepdims=True)
    test_hist/=test_hist.sum(-1,keepdims=True)
    
    return mmd_tv(train_hist,test_hist,sigma=1.0)

                     

    
def mmd_tv(X, Y, sigma=1.0):
    def gaussian_tv(X,Y,sigma=1):
#         dist = pairwise_distances(X,Y,'minkowski', p=1)/2
        dist = torch.cdist(X[None],Y[None], p=1)[0]/2
        return torch.exp(-dist*dist / (2 * sigma * sigma))
        
        dist = np.abs(x - y).sum() / 2.0
    
    XX = gaussian_tv(X, X, sigma)
    YY = gaussian_tv(Y, Y, sigma)
    XY = gaussian_tv(X, Y, sigma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

