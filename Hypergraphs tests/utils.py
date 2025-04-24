import gudhi as gd
import numpy as np


def simp_comp_adj(SpT,n_nodes,dim):
    adj_mats = {}
    for d in range(dim+1):
        adj_mats[d] = np.zeros([n_nodes]*(d+1))
    
    for splx in SpT.get_simplices():
        idx = splx[0]
        adj_mats[len(idx)-1][tuple(idx)] = 1
        
    return adj_mats

def simp_comp_abs_dist(spt1,spt2,n_nodes,dim):
    adj_mats1 = simp_comp_adj(spt1,n_nodes,dim)
    adj_mats2 = simp_comp_adj(spt2,n_nodes,dim)
    
    #dim = min(spt1.dimension(),spt2.dimension())
    #dim = 1
    S = 0
    
    for d in range(dim+1):
        S+= np.sum(np.abs(adj_mats1[d]-adj_mats2[d]))
    
    return S