import numpy as np
import gudhi
from sklearn.manifold import (
  MDS, 
  Isomap, 
  LocallyLinearEmbedding, 
  SpectralEmbedding, 
  TSNE
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def corr_metric(X):
    '''
    

    Parameters
    ----------
    X : numpy array of dim [channels,times]
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    D = np.abs(np.corrcoef(X))
    
    return 1-D

def compute_complex(dist_mat,max_dim = 3):
    rips_cplx = gudhi.RipsComplex(distance_matrix = dist_mat)
    cplx = rips_cplx.create_simplex_tree(max_dimension = max_dim)
    return cplx

def plot_embeddings(
    embeds,
    col,
    ax,
    dim = 0, 
    name = "???",
    phase = "???",
    distance = "Wasserstein",
    projection = "???",
    title = True
    ):
    
    title = f"Subject {name} during {phase}, persistence diagram (dim {dim})\nfor each of the {len(col)} colours described by {distance} distance, {projection} projection"
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    for i in range(len(col)):
        ax.scatter(embeds[:,0],embeds[:,1], color = col)
    #ax.legend()
    if title:
        ax.set_title(projection+' with '+distance)
    #plt.show()


def plot_pca(
    embeds,
    B,
    col,
    ax,
    dim = 0, 
    name = "???",
    phase = "???",
    distance = "Wasserstein",
    title = True
    ):
    
    title = f"Subject {name} during {phase}, persistence diagram (dim {dim})\nfor each of the {len(col)} colours described by {distance} distance, PCA projection"
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    
    for i in range(len(col)):
      ax.scatter(np.dot(embeds[0], B[i]), np.dot(embeds[1], B[i]), color = col[i])
    
    if title:
        ax.set_title("PCA"+' with '+distance)
    #plt.show()


def compute_pers_diag_manifold(B,met,n_comp,col, distance, axs = False):
        
    if axs.any() == False:
        fig, axs = plt.subplots(nrows = 1, ncols = len(B), sharex = True, sharey = True, figsize = (15,5))
    # MDS
    if met == 0:
        for i in range(len(B)):
            mds = MDS(
                n_components = 2, 
                max_iter = 10000, 
                eps = 1e-9, 
                dissimilarity = "precomputed", 
                n_jobs = 1
            )
            mds_fit = mds.fit(B[i])
            mds_embeds = mds_fit.embedding_
            if i>0:
                title = False
            else:
                title = True
            
            
            plot_embeddings(mds_embeds,col = col, dim = i, projection = "MDS", distance = distance, ax = axs[i], title = title)
    
    
    # PCA
    if met == 1:
        for i in range(len(B)):
            pca_fit = PCA(n_components=2).fit(B[i])
            pca_embeds = pca_fit.components_
            if i>0:
                title = False
            else:
                title = True
            plot_pca(pca_embeds, B[i], col, distance = distance, ax = axs[i],title = title)
    
    
    # Isomap
    if met == 2:
        for i in range(len(B)):
            iso_fit = Isomap(n_components = 2, p = 1).fit(B[i])
            iso_embeds = iso_fit.embedding_
            if i>0:
                title = False
            else:
                title = True
            plot_embeddings(iso_embeds,col = col, dim = i, projection = "Isomap", distance = distance, ax = axs[i], title = title)
            
    
    # TSNE
    if met == 3:
        for i in range(len(B)):
            tsne_fit = TSNE(n_components = 2).fit(B[i])
            tsne_embeds = tsne_fit.embedding_
            if i>0:
                title = False
            else:
                title = True
                
            plot_embeddings(tsne_embeds,col = col, dim = i, projection = "tSNE", distance = distance, ax = axs[i],title = title)
    
    
    # LLE
    # lle_fit = LocallyLinearEmbedding(method = "standard", n_components = 2).fit(B0)
    # lle_fit = LocallyLinearEmbedding(method = "ltsa", n_components = 2).fit(B0)
    # lle_fit = LocallyLinearEmbedding(method = "hessian", n_components = 2).fit(B0)
    if met == 4:
        for i in range(len(B)):
            lle_fit = LocallyLinearEmbedding(method = "modified", n_components = 2).fit(B[i])
            lle_embeds = lle_fit.embedding_
            if i>0:
                title = False
            else:
                title = True
                
            plot_embeddings(lle_embeds,col = col, dim = i, projection = "LLE", distance = distance, ax = axs[i],title = title)
    
    
    # Spectral Embedding
    if met == 5:
        for i in range(len(B)):
            se_fit = SpectralEmbedding(n_components = 2).fit(B[i])
            se_embeds = se_fit.embedding_
            if i>0:
                title = False
            else:
                title = True
                
            plot_embeddings(se_embeds,col = col, dim = i, projection = "SpecEmb", distance = distance, ax = axs[i],title = title)