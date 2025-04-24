import numpy as np
import signatory
import torch
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
#from knockpy.knockoff_filter import KnockoffFilter
#from knockpy.knockoffs import GaussianSampler
import matplotlib.pyplot as plt
import gudhi
from gudhi import SimplexTree
from sortedl1 import Slope
from tqdm import tqdm
#from draw_gif import draw_2d_simplicial_complex
from dataset import MA, MA_betti
import sys
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::ConvergenceWarning')


class SigComplex:
    def __init__(self, MultiTS : torch.tensor, win: int, depth: int = 3, logsig = False, alpha_1d: int = 1, alpha_2d: int = 1, lasso: bool = True, max_iter: int = 2000):
        '''
        

        Parameters
        ----------
        MultiTS : torch.tensor
            Time series. Accepted shape : (1,time,channels)
            
        win : int
            Windows. Length of complex expression : if win = W, the complex
            express the interval [t0-W,t0] at time t0
            
        depth : int, optional
            Depth of signature. The default is 3.The higher you set depth,
            the most expressive variable selection regression you get.
            Minimal value is 2.
            Warning: a high value of depth implies a decrease of alpha_2d
            
        alpha_1d : int, optional
            Lasso parameters for 1D creation. The default is 1. A high value
            bring sparsification (for 1D).
            
        alpha_2d : int, optional
            Lasso parameters for 2D creation. The default is 1. A high value
            bring sparsification (for 2D).
            
        lasso : bool, optional
            Decide between Lasso and Slope. The default is True (Lasso).
            
        max_iter : int, optional
            Maximum number of iteration for variable selection. The default is 2000.

        Returns
        -------
        hyper_coherence: list of int
            Quantify how much triangle are created with
            all their edges. The higher the better.
        
        life_durationN: list of int
            quantify the number of creation for each simplex 
            of dimension N
        
        bettiN: list of int
            All betti numbers (of dimension N) through time.
            Warning: the first betti number is for t_0 = win
        
        persNalongT: list of float 
            The persistance entropy through time.
            Same warning that for betti numbers.
        
        nND: list of int
            Number of simplex of dimension N for each time.

        '''
        self.MultiTS = MultiTS 
        self.win = win
        self.T = MultiTS.shape[-2]
        self.D = MultiTS.shape[-1]
        if MultiTS.shape[0]==1:
            self.batch = self.D
        else:
            self.batch = self.MultiTS.shape[0]
        self.alpha_1d = alpha_1d
        self.alpha_2d = alpha_2d
        self.knockoffs = False
        self.complexT = [None] * MultiTS.shape[-2] # list to stock simplicial complex
        self.hyper_coherenceT = [None] * MultiTS.shape[-2] # list to stock hyper coherence
        self.life_duration1 = {} # life duration of 1-simplices
        self.life_duration2 = {} # life duration of 2-simplices
        self.lasso = lasso
        self.only_fixed_deg = False
        self.max_iter = max_iter
        self.betti1 = []
        self.betti2 = []
        self.betti3 = []
        self.depth = depth
        self.pers0alongT = []
        self.pers1alongT = []
        self.n1D = []
        self.n2D = []
        self.coherence1 = []
        self.coherence2 = []
        self.adj_mat_1d = [None] * MultiTS.shape[-2]
        self.logsig = logsig
        
    
    
    def simplex_1d(self,t,verbose = True):     
        '''
        Compute augmented signatures of TS and perform lasso regression to create a adjacent matrix

        Parameters
        ----------
        t : int

        Returns
        -------
        adj_mat_1d : numpy.array
            adjacent matrix

        '''
        if self.batch == 1:
            k = int(1)
        else:
            k = self.D
        depth = self.depth
        k_uplets = list(combinations(np.arange(self.batch),1))
        adj_mat_1d = np.zeros([len(k_uplets),len(k_uplets)])
        if self.logsig :
            chansig = signatory.logsignature_channels(k+1, depth = self.depth)
        else:
            chansig = signatory.signature_channels(k+1, depth = self.depth)
        # compute (augmented) signature features for all k-uplets

        df_sig = np.zeros([chansig,len(k_uplets)])

        for i in range(len(k_uplets)):
            if self.MultiTS.shape[0] == 1:
                TS_group = torch.cat( 
                                (self.MultiTS[:,t:t+self.win,(i,)],torch.tensor(
                                    np.linspace(start = 0, stop = 1, num = self.win).reshape(1,self.win,1))) , 
                                 dim=2)
            else :
                TS_group = torch.cat( 
                                (self.MultiTS[(i,),t:t+self.win],torch.tensor(
                                    np.linspace(start = 0, stop = 1, num = self.win).reshape(1,self.win,1))) , 
                                 dim=2)
            
            if self.logsig:
                df_sig[:,i] = signatory.logsignature(TS_group, depth=depth, basepoint=True)
            else :
                df_sig[:,i] = signatory.signature(TS_group, depth = depth)

        self.coherence1.append(coherence_prop(df_sig))
        # for each k-uplet, select his neighbors by lasso 
        for i in range(len(k_uplets)): 
            if self.lasso:
                clf = Lasso(alpha=self.alpha_1d, fit_intercept= False) # choice of alpha depends of dataset, but generaly large for our configuration
                #clf = LassoCV(fit_intercept= False)
                #clf = LassoLarsCV(fit_intercept= False)
            else:
                clf = Slope(alpha=self.alpha_1d, fit_intercept= False,max_iter=self.max_iter)
            # get k-uplets which have distinct elements from the elements of target k-uplet
            indices = np.array([j for j, tup in enumerate(k_uplets) 
                                if tup[0] != k_uplets[i] ])                 
            # regression by lasso
            clf.fit(df_sig[:,indices],df_sig[:,i])
            # only regresion with R2 > 0.67 are kept
            w = clf.score(df_sig[:,indices],df_sig[:,i])
            if verbose:
                print(w)
            if w > 0.67 :
                if not self.knockoffs:
                    if self.lasso:
                        adj_mat_1d[indices[np.nonzero(clf.coef_)], i] = clf.coef_[clf.coef_ != 0]
                    else:
                        adj_mat_1d[indices[np.nonzero(clf.coef_)[0]], i] = clf.coef_[clf.coef_ != 0]
                if self.knockoffs:
                    kfilter = KnockoffFilter( 
                                                fstat='mlr', 
                                                ksampler='gaussian', 
                                                knockoff_kwargs={"method":"mmi"}, 
                                            )
                    rejections = kfilter.forward(X=df_sig[:,indices],y=df_sig[:,i])
                    adj_mat_1d[indices[np.nonzero(1-rejections)], i] = clf.coef_[rejections == 0]
        return adj_mat_1d
    
    def simplex_2d(self,t,verbose = True):
        '''
        Lasso regressions are applied between signatures (depth 2) of couples of TS (predictors) and lead-lag transformation of one TS (target)

        Parameters
        ----------
        t : t

        Returns
        -------
        adj_mat_2d : numpy.array
            adjacent matrix
        k_uplets : list
            all possible combinations of couples
        '''
        k = int(2)
        depth = self.depth
        chans = self.MultiTS.shape[-1]
        if self.logsig :
            chansig = signatory.logsignature_channels(k+1, depth = self.depth)
        else:
            chansig = signatory.signature_channels(k+1, depth = self.depth)        
        k_uplets = list(combinations(np.arange(self.batch),k))
        adj_mat_2d = np.zeros([self.batch,len(k_uplets)])

        # compute signature features for all k-uplets
        df_sig_predictors = np.zeros([chansig,len(k_uplets)]) 
        for i in range(len(k_uplets)):
            if self.MultiTS.shape[0] == 1:
                if self.logsig:
                    df_sig_predictors[:,i] = signatory.logsignature(self.MultiTS[:,t:t+self.win,k_uplets[i]], depth=depth)   
                else: 
                    df_sig_predictors[:,i] = signatory.signature(self.MultiTS[:,t:t+self.win,k_uplets[i]], depth=depth)
            else:
                series_to_sig = torch.cat([self.MultiTS[k_uplets[i][0],t:t+self.win],self.MultiTS[k_uplets[i][0],t:t+self.win]],dim = -1)[None]
                if self.logsig:
                    df_sig_predictors[:,i] = signatory.logsignature(series_to_sig, depth=depth)
                else:
                    df_sig_predictors[:,i] = signatory.signature(series_to_sig, depth=depth)

        df_sig_predictors -= df_sig_predictors.mean(axis = 0)
        self.coherence2.append(coherence_prop(df_sig_predictors))
        # compute signature features of lead-lag transformation of target
        df_sig_target = np.zeros([chansig,self.batch])
        for d in range(self.D):
            TS_ll = leadlag(self.MultiTS[(d,),t:t+self.win])
            df_sig_target[:,d] = signatory.signature(TS_ll, depth = depth)
        
        # for each node, select his edge neighbors by lasso 
        #Knockoffs = GaussianSampler(df_sig_predictors).sample_knockoffs()
        for i in range(self.D): 
            if self.lasso:
                clf = Lasso(alpha=self.alpha_2d, max_iter=self.max_iter, fit_intercept= False) # choice of alpha depends of dataset, but generaly large for our configuration
            else:
                clf = Slope(alpha=self.alpha_2d, fit_intercept=False,max_iter=self.max_iter)
            # get k-uplets which have distinct elements from the elements of target k-uplet
            indices = np.array([j for j, tup in enumerate(k_uplets) if tup[0] != i and tup[1] != i ])                 
            # regression by lasso
            clf.fit(df_sig_predictors[:,indices],df_sig_target[:,i])
            # only regresion with R2 > 0.67 are kept
            w = clf.score(df_sig_predictors[:,indices],df_sig_target[:,i])
            if verbose:
                print("Score: "+str(w))
            if w > 0.67 :
                if not self.knockoffs:
                    if self.lasso:
                        adj_mat_2d[i, indices[np.nonzero(clf.coef_)]] = clf.coef_[clf.coef_ != 0]
                    else:
                        adj_mat_2d[i, indices[np.nonzero(clf.coef_[:,0])]] = clf.coef_[clf.coef_ != 0]

                if self.knockoffs:
                    kfilter = KnockoffFilter( 
                                                fstat='mlr',
                                                ksampler='fx', 
                                                knockoff_kwargs={"method":"mvr"}, 
                                            )
                    rejections = kfilter.forward(X=df_sig_predictors[:,indices],y=df_sig_target[:,i],fdr=0.9)
                    adj_mat_2d[indices[np.nonzero(1-rejections)], i] = clf.coef_[rejections == 0]
        return adj_mat_2d, k_uplets
           
    def complex_creation(self,t, along_t = False, verbose = True, with_wnorm = False):
        '''
        Core method to creat complex, the simplices are added by order (0 -> 1 ->2)
        The simplices are defined by the prediction capability of their signatures feartures 
        2-simplex that violate order rules are kept by adding low-order simplices to the complex
        Parameters
        ----------
        t : int
        
        Returns
        -------
        c : SimplicialComplex

        '''
        # initialize complex
        c = SimplexTree()
        # add 0-simplex
        for d in range(self.D):
            _ = c.insert([d])
        # adjance matrix between 0-simplices
        adj_mat_1d = self.simplex_1d(t,verbose = verbose)
        # Add 1-simplex with weight
        num1 = 0
        for i in range(adj_mat_1d.shape[0]):
            for j in range(i + 1, adj_mat_1d.shape[1]):  # Only add edges once (above the main diagonal)
                if adj_mat_1d[i,j] != 0 or adj_mat_1d[j,i] != 0:
                    # when two 0-simplices are predictor of each other, the weight become the sum of their coefficients
                    id1 = str(i) + '-' + str(j)
                    _ = c.insert([i,j],filtration=1/(np.abs(adj_mat_1d[i,j])+np.abs(adj_mat_1d[j,i])))
                    num1 += 1
                    if along_t:
                        self.life_duration1[id1] = self.life_duration1[id1] + 1 if id1 in self.life_duration1 else 1
                    if verbose:
                        print("add ",str(i) + '-' + str(j))
        if verbose:
            print("Epoch ",t," with "+str(num1)+" 1-simplex")
        self.n1D.append(num1)
        # adjance matrix between 0-simplice and 1-simplices
        
        # We go through the filtration
        c_gen = c.get_filtration() 
        count_with = 0
        count_without = 0
        for splx in c_gen :
            for splx_b in c.get_boundaries(splx[0]):
                n = 0
                if c.filtration(splx_b[0])>c.filtration(splx[0]):
                    c.assign_filtration(splx_b[0],c.filtration(splx[0]))
                    n += 1
                
                if n == 0:
                    count_with+=1
                else:
                    count_without+=1
                
        hyper_coherence = count_without/count_with if count_with != 0 else "no 2-simplex has all needed 1-simplex"
        return c, hyper_coherence, adj_mat_1d
    def complex_along_T(self, a, b, verbose = False, with_wnorm = False):
        '''
        This method generate complex for a given 
        
        a : int 
        a is just t_0
        
        b : int
        b is just t_final-win
        

        '''
        for t in range(a,b): 
            self.complexT[t], self.hyper_coherenceT[t],  = self.complex_creation(t, verbose = verbose, along_t=True, with_wnorm = False)
            # The following line is a necessity to compute betti3
            self.complexT[t].set_dimension(3)
            # The following line is a necessity to compute persistance entropy
            self.complexT[t].compute_persistence()
            bn = self.complexT[t].betti_numbers()
            self.pers0alongT.append(self.complexT[t].persistence_intervals_in_dimension(0))
            self.pers1alongT.append(self.complexT[t].persistence_intervals_in_dimension(1))
            self.betti1.append(bn[0])
            self.betti2.append(bn[1])
            self.betti3.append(bn[2])
            
    def plot_betti(self, win_MA: int, quantiles: tuple = (0.05,0.95), fig_size: tuple = (15,10)):
        '''
        

        Parameters
        ----------
        win_MA : int
            Window size for moving average of betti numbers
        quantiles : tuple
            Quantiles for confidence. The default is (0.05,0.95)
        absc : np.array(), optional
            Array for times. The default is None.
        fig_size : tuple, optional
            Figure size. The default is (15,10).

        Returns
        -------
        None.

        '''
        b1_to_plot = MA_betti(self.betti1,win_MA,median = False)
        b2_to_plot = MA_betti(np.array(self.betti2),win_MA,median = False)
        b3_to_plot = MA_betti(np.array(self.betti3),win_MA,median = False)
        
        av_b1 = np.array(self.betti1).mean()
        av_b2 = np.array(self.betti2).mean()
        av_b3 = np.array(self.betti3).mean()
        
        quant_b1 = np.repeat(np.quantile(np.array(self.betti1),quantiles)[None],len(self.betti1))
        quant_b2 = np.repeat(np.quantile(np.array(self.betti2),quantiles)[None],len(self.betti1))
        quant_b3 = np.repeat(np.quantile(np.array(self.betti3),quantiles)[None],len(self.betti1))
        
        fig, axs = plt.subplots(nrows=3, ncols = 1, sharex = True,figsize = fig_size)
        
        absc = np.arange(self.win,len(b1_to_plot)+self.win)
        
        axs[0].plot(absc, b1_to_plot)
        axs[0].axhline(y=av_b1, color='g', linestyle='-')
        axs[0].fill_between(absc, quant_b1[0], quant_b1[1])
        axs[0].legend(["b1","Average b1","90% quantile band"])
        axs[0].set_title('b1 curve along time')
        
        axs[1].plot(absc, b2_to_plot)
        axs[1].axhline(y=av_b2, color='g', linestyle='-')
        axs[1].fill_between(absc, quant_b2[0], quant_b2[1])
        axs[1].legend(["b2","Average b2","90% quantile band"])
        axs[1].set_title("b2 curve along time")
        
        axs[2].plot(absc, b3_to_plot)
        axs[2].axhline(y=av_b3, color='g', linestyle='-')
        axs[2].fill_between(absc, quant_b3[0], quant_b3[1])
        axs[2].legend(["b3","Average b3","90% quantile band"])
        axs[2].set_title("b3 curve along time")
        
        plt.show()
        
    def plot_PE(self, win_MA:int, quantiles: tuple = (0.05,0.95), fig_size: tuple = (15,10)):
        '''
        

        Parameters
        ----------
        win_MA : int
            Window size for moving average of betti numbers
        quantiles : tuple
            Quantiles for confidence. The default is (0.05,0.95)
        absc : np.array(), optional
            Array for times. The default is None.
        fig_size : tuple, optional
            Figure size. The default is (15,10).

        Returns
        -------
        None.
        '''
        
        remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1]!=np.inf])
        
        
        list_pers = list(map(remove_infinity,self.pers0alongT))
        PE = gudhi.representations.Entropy()
        pe_alongT = PE.fit_transform(list_pers)
        
        pe_to_plot = MA_betti(pe_alongT,win_MA,median = False)
        
        av_pe = pe_alongT.mean()
        
        quant_pe = np.repeat(np.quantile(pe_alongT,quantiles)[None],len(pe_to_plot))


        absc = np.arange(self.win,len(pe_alongT)-win_MA+self.win)
        plt.plot(absc,pe_to_plot)
        plt.title("PE curve along time")
        plt.legend(["PE","Average PE","90% quantile band"])
        plt.axhline(y=av_pe, color='g', linestyle='-')
        plt.fill_between(absc, quant_pe[0], quant_pe[1])
        plt.show()
        
    def plot_all(self, win_MA: int, quantiles: tuple = (0.05,0.95), fig_size: tuple = (25,10)):
        '''
        

        Parameters
        ----------
        win_MA : int
            Window size for moving average of betti numbers
        quantiles : tuple
            Quantiles for confidence. The default is (0.05,0.95)
        absc : np.array(), optional
            Array for times. The default is None.
        fig_size : tuple, optional
            Figure size. The default is (15,10).

        Returns
        -------
        None.

        '''
        b1_to_plot = MA_betti(self.betti1,win_MA,median = False)
        b2_to_plot = MA_betti(np.array(self.betti2),win_MA,median = False)
        b3_to_plot = MA_betti(np.array(self.betti3),win_MA,median = False)
        
        av_b1 = np.array(self.betti1).mean()
        av_b2 = np.array(self.betti2).mean()
        av_b3 = np.array(self.betti3).mean()
        
        quant_b1 = np.repeat(np.quantile(np.array(self.betti1),quantiles)[None],len(self.betti1))
        quant_b2 = np.repeat(np.quantile(np.array(self.betti2),quantiles)[None],len(self.betti1))
        quant_b3 = np.repeat(np.quantile(np.array(self.betti3),quantiles)[None],len(self.betti1))
        
        fig, axs = plt.subplots(nrows=5, ncols = 1, sharex = True,figsize = fig_size)
        
        absc = np.arange(self.win,len(b1_to_plot)+self.win)
        
        axs[0].plot(self.MultiTS[0].numpy())
        axs[0].set_title('Time Series')
        
        axs[1].plot(absc, b1_to_plot)
        axs[1].axhline(y=av_b1, color='g', linestyle='-')
        axs[1].fill_between(absc, quant_b1[0], quant_b1[1])
        axs[1].legend(["b1","Average b1","90% quantile band"])
        axs[1].set_title('b1 curve along time')
        
        axs[2].plot(absc, b2_to_plot)
        axs[2].axhline(y=av_b2, color='g', linestyle='-')
        axs[2].fill_between(absc, quant_b2[0], quant_b2[1])
        axs[2].legend(["b2","Average b2","90% quantile band"])
        axs[2].set_title("b2 curve along time")
        
        axs[3].plot(absc, b3_to_plot)
        axs[3].axhline(y=av_b3, color='g', linestyle='-')
        axs[3].fill_between(absc, quant_b3[0], quant_b3[1])
        axs[3].legend(["b3","Average b3","90% quantile band"])
        axs[3].set_title("b3 curve along time")
        
        remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1]!=np.inf])
        
        
        list_pers = list(map(remove_infinity,self.pers0alongT))
        PE = gudhi.representations.Entropy()
        pe_alongT = PE.fit_transform(list_pers)
        
        
        
        
        print(max(pe_alongT))
        
        pe_to_plot = MA_betti(pe_alongT,win_MA,median = False)
        
        av_pe = pe_alongT.mean()
        
        quant_pe = np.repeat(np.quantile(pe_alongT,quantiles)[None],len(pe_to_plot))


        absc = np.arange(self.win,len(pe_alongT)-win_MA+self.win)
        axs[4].plot(absc,pe_to_plot)
        axs[4].set_title("PE curve along time")
        axs[4].legend(["PE","Average PE","90% quantile band"])
        axs[4].axhline(y=av_pe, color='g', linestyle='-')
        axs[4].fill_between(absc, quant_pe[0], quant_pe[1])
        
        plt.show()


            
    def life_duration_analyse(self):
        plt.hist(self.life_duration1.values())
        plt.title("1-simplices duration distribution")
        plt.show()
        print("The 10 most persistent 1-simplices")
        print(sorted(self.life_duration1.items(), key=lambda x:x[1], reverse=True)[0:10])
        plt.hist(self.life_duration2.values())
        plt.title("2-simplices duration distribution")
        plt.show()
        print("The 10 most persistent 2-simplices")
        print(sorted(self.life_duration2.items(), key=lambda x:x[1], reverse=True)[0:10])
    
    def hyper_coherence_analyse(self, a, b):
        print(self.hyper_coherenceT[a:b])
        
def coherence_prop(A):
    return np.max(np.abs(A@A.T))


def leadlag(X):
    '''
    lead-lag transformation of one dimensional TS [T,1] -> [2T-1, 2] 

    Parameters
    ----------
    X : tensor
        Time serie tensor of size [1,T,1]

    Returns
    -------
    TS_ll : tensor
        Time serie tensor of size [1,2T-1,2]

        '''
    T = X.shape[-2]
    d = X.shape[-1]
    
    # Initiate path
    lead_lag = np.empty([1,(T-1)*2+1,d*2])
    lead_lag_pair = np.tile(X,2)
    zeros_beg = np.insert(X,0,0,axis=1)
    lead_lag_impair = np.concatenate((np.insert(X,T,0,axis=1).reshape([1,T+1,d]),zeros_beg),axis=-1)
    
    for t in range(T-1):
        #When t' = 2t, LL = [X_1(t),X_2(t),...,X_n(t),X_1(t),X_2(t),...,X_n(t)]
        lead_lag[:,2*t] = lead_lag_pair[:,t,:]
        #When t' = 2t+1, LL = [X_1(t+1),X_2(t+1),...,X_n(t+1),X_1(t),X_2(t),...,X_n(t)]
        lead_lag[:,2*t+1] = lead_lag_impair[:,t+1,:]
    lead_lag[:,-1] = lead_lag_pair[:,-1]
    return torch.tensor(lead_lag)
