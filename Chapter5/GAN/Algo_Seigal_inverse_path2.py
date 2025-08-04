#import pywavelets

import torch
from torch import optim
import torch.nn as nn
from signatory import Signature, signature_channels, signature_combine, all_words
import numpy as np
#from torch.func import grad
from numpy import polynomial
from skfda.representation import basis
import skfda



class SeigalAlgo:
    
    def __init__(self, size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan = True, sig_TS = None, A_init = None, decal_length = 1):
        """initialize the inversion algorithm

        Args:
            len_base (int): the number of base functions
            chan (int): number of channels of the time series
            real_chan (int): deprecated
            depth (int): depth of the signature
            n_recons (int): deprecated
            size_base (int): size of the base
            time_chan (bool, optional): indicates if we have time augmentation. Defaults to True.
            decal_length (int, optional): _description_. Defaults to 1.
            sig_TS (tensor): signature from which retrieve the TS.
        """
        self.len_base = len_base
        self.chan = chan
        self.T = np.linspace(start = 0, stop = 1, num = size_ts)
        self.T_base = np.linspace(start = 0, stop = 1, num = size_base)
        self.depth = depth
        self.real_chan = real_chan
        self.insertion = []
        self.n_recons = n_recons
        self.time_chan = time_chan
        self.decal_length = decal_length
        self.sig_TS = sig_TS
        self.A_init = A_init

        
    def define_base(self, base_name, T_original = False, order = 3):
        """_summary_

        Args:
            base_name (str): type of the base
            T_original (bool, optional): If true, we keep the original time grid. Defaults to False.
            order (int, optional): _description_. Defaults to 3.

        Returns:
            ndarray: base created
        """
        if T_original:
            T= self.T
        else:
            T = np.linspace(self.T.min(),self.T.max(),self.T.shape[0])
        if base_name == 'polynomials':
            base_obj = basis.MonomialBasis(domain_range=(np.min(self.T),np.max(self.T)),n_basis = self.len_base)
            base = base_obj(self.T_base).T
            X = skfda.FDataGrid(self.T)
            self.T_basis = X.to_basis(base_obj).coefficients
        if base_name == 'poly-che':
            base = np.empty([self.len_base, len(self.T)])
            for k in range(self.len_base):
                base[k] = polynomial.Chebyshev.basis(k)(self.T_base)
            base = base.T[None]
        if base_name == 'poly-her':
            base = np.empty([self.len_base, len(self.T)])
            for k in range(self.len_base):
                base[k] = polynomial.Hermite.basis(k)(self.T_base)
            base = base.T[None]
        if base_name == 'poly-lag':
            base = np.empty([self.len_base, len(self.T)])
            for k in range(self.len_base):
                base[k] = polynomial.Laguerre.basis(k)(self.T_base)
            base = base.T[None]
        if base_name == 'poly-leg':
            base = np.empty([self.len_base, len(self.T)])
            for k in range(self.len_base):
                base[k] = polynomial.Legendre.basis(k)(self.T_base)
            base = base.T[None]
        if base_name == 'BSpline':
            base_obj = basis.BSplineBasis(domain_range=(np.min(self.T),np.max(self.T)),n_basis = self.len_base, order = order)
            base = base_obj(self.T_base).T
            X = skfda.FDataGrid(self.T)
            self.T_basis = X.to_basis(base_obj).coefficients
        if base_name == 'Fourier':
            base_obj = basis.FourierBasis(domain_range=(np.min(self.T),np.max(self.T)),n_basis = self.len_base)
            base = base_obj(self.T_base).T
            X = skfda.FDataGrid(self.T)
            self.T_basis = X.to_basis(base_obj).coefficients
        if base_name == 'PwLinear':
            x = self.T
            base = np.zeros(shape = (len(self.T_base),self.len_base))
            for i in range(1,self.len_base+1):
                base[:,i-1] = linear_base(self.T_base, i, self.len_base,self.T_base)
            base = base[None]
            self.T_basis = np.ones(shape = (1,self.len_base))*1/self.len_base
        self.len_base = base.shape[-1]
        return(torch.from_numpy(base))


    def retrieve_coeff_base(self, base, par, lrs = 1e-3, limits = 1e4,opt = "AdamW",eps=1e-10, params = [1,5,0,0]):
        """ Retrieve A from depth 3 signature.

        Args:
            base (_type_): _description_
            par (int): deprecated
            lrs (float, optional): Learning rate. Defaults to 1e-3.
            limits (int, optional): Number of iterations for optimisation. Defaults to 1e4.
            opt (str, optional): "Adam", "AdamW" or "LBFGS", selected optimizer. Defaults to "AdamW".
            eps (_type_, optional): _description_. Defaults to 1e-10.
            params (list, optional): params are [lambda_length, lambda_frontier, lambda_levy, lambda_ridge].
            [1,5,0,1] , [5,1,0,1] or the same with lambda_ridge = 0 worked well. Defaults to [1,5,0,0].

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
    
        L_control_coeff, bord_control_coeff, LA_control_coeff, ridge_coeff = params
       
       
        if torch.cuda.is_available():
           device = 'cuda'
           dtype = torch.cuda.FloatTensor
        else:
           device = 'cpu'
           dtype = torch.FloatTensor
    
        print("Using "+device)
        
        loss1 = nn.MSELoss().to(device)
       
       
        ########## On calcule les différentes signatures
        signature_base = Signature(depth = self.depth).to(device)

        ### Sig_TS est une collection de tenseurs
        sig_TS = self.sig_TS
        sig_base = signature_base(base) 

        def unflatten_sig2(sig):
            sig_unflat = [np.empty(1) for i in range(self.depth)]
            start = 0
            stop = self.len_base
            for k in range(self.depth):
                sig_unflat[k] = sig[0,start:stop].unflatten(0,[self.len_base]*(k+1))
                start = stop
                stop = start + self.len_base**(k+2)
            return sig_unflat



        sig_base_unf = unflatten_sig2(sig_base.type(dtype))

        d=self.chan
        sig_TS_unf = [sig_TS[:,1:d+1],sig_TS[:,d+1:d+d**2+1].unflatten(dim=-1,sizes = (d,d)),sig_TS[:,d+d**2+1:d+d**2+d**3+1].unflatten(dim=-1,sizes = (d,d,d))]    
        #LA3_TS = sig_TS[2].permute(0,2,1,3)+sig_TS[2].permute(0,2,3,1)-sig_TS[2].permute(0,3,1,2)-sig_TS[2].permute(0,3,2,1)

        #LA4_TS = torch.kron(sig_TS[0],LA3_TS)  
        def mode_dot(x, m, mode):
           #x = np.asarray(x)
           #m = np.asarray(m)
           if mode % 1 != 0:
               raise ValueError('`mode` must be an integer')
           if x.ndim < mode:
               raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
           if m.ndim != 2:
               raise ValueError('Invalid shape of M: {}'.format(m.shape))
           return torch.swapaxes(torch.tensordot(torch.swapaxes(x, mode, -1),m.T,dims = 1), mode, -1)

        def multi_mode_dot(x,m,n):
           MMD = x.T
           for i in range(n):
               MMD = mode_dot(MMD,m,mode = i)
           return MMD.T

        def lengthA(A):
            return torch.sum(torch.norm(A[:,self.decal_length:],dim = 1))   

        def final_points(A,B):  
             RS = torch.matmul(B.float(),A.T)[0]
             checkpoints = RS[[0,-1]]
             return checkpoints.to(device)

        def flatten_MMD(MMD):
            chan_sig = signature_channels(self.chan, self.depth,scalar_term=True)
            sig_flat = torch.ones(chan_sig)
            words = all_words(self.chan, self.depth)
            for k in range(chan_sig-1):
                deg = len(words[k])
                sig_flat[k+1] = MMD[deg-1][words[k]]
            return sig_flat


        def error_func(A):

            #B = torch.cat((torch.tensor(self.T_basis).float().to(device),A),dim = 0).flip([-2]).float()
            #### On calcule [C,A,A,...,A] pour k<= 3
            MMD = [multi_mode_dot(sig_base_unf[i].float(), A, i+1) for i in range(3)]
            MMD_flat = flatten_MMD(MMD)[None].float().to(device)#.type(dtype)
             

            ### Objectif 0
            sig_control = torch.norm(signature_combine(sig_TS.float(),MMD_flat.float(),self.chan,self.depth,scalar_term = True).T[-d**3:])**2
            error = sig_control
            #print(error)

            ### Contrainte de longueur
            if self.time_chan:
                i=1
            else:
                i=0
            L_control = L_control_coeff* loss1(lengthA(A).float(),sig_TS[0,i].float()).float()
            error = error+L_control
            #print(L_control)

            ### controle de bords:
            points = final_points(A,base.flip([-2,-1]).to(device))
            val_to_reach = torch.tensor([sig_TS_unf[2][0,i,i,i] for i in range(self.chan)])
            val_to_move = ((points[-1]-points[0])**3)/6
            bord_control = bord_control_coeff*torch.norm(val_to_reach.float().to(device)-val_to_move.float().to(device))
            error = error+bord_control
            #print(bord_control)
            
            ### Controle Levy Area
            LA_MMD = 0.5*(MMD[1]-MMD[1].T)
            LA = 0.5*(sig_TS_unf[1]-sig_TS_unf[1].T)
            LA_control = LA_control_coeff*torch.norm(LA-LA_MMD)**2
            error = error + LA_control+ridge_coeff*torch.norm(A)#sig_control.float()+L_control.float()+bord_control.float()
            #print(ridge_coeff*torch.norm(A))
            return error
        
        if self.A_init is not None:
            A = self.A_init
        else:
            A = torch.randn([self.chan,self.len_base], requires_grad=True) 

        if opt == "AdamW":
                optimizer = optim.AdamW([A], lr = lrs)
        if opt == "Adam":
                optimizer = optim.Adam([A], lr = lrs)
        if opt == "LBFGS":
                optimizer = optim.LBFGS([A], lr=lrs)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 50, min_lr = 1e-7)
        
        i = 0
        loss_ref = np.inf
        A_temp = np.zeros(shape = (1))

        par = 0.01

        reset_lr = 1e-2

        while i<limits and loss_ref > eps:
            print(f'Step {i}')
            if opt != "LBFGS":
                optimizer.zero_grad()
                #MMD = model.forward(A)   
                loss_val = error_func(A.type(dtype)).to(device)
                loss_val.backward()
                optimizer.step()
                loss_value = loss_val.clone().detach()
                sched.step(loss_value)
                current_lr = optimizer.param_groups[0]['lr']
                print(current_lr)
                if current_lr <= 1.5e-6 and loss_value > 100:
                    print(f"🔁 Resetting LR to {reset_lr}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = reset_lr
                    sched._num_bad_epochs = 0
            else:
                 def closure():
                     optimizer.zero_grad()
                     #MMD = model.forward(A)
                     loss_val = error_func(A.type(dtype)).to(device)
                     loss_val.backward()
                     return loss_val
                 optimizer.step(closure)
                 loss_value = closure().clone().detach()
                 sched.step(loss_value)
                 print(optimizer.param_groups[0]['lr'])

            if loss_value < loss_ref:
                  A_temp = A.clone()
                  loss_ref = loss_value
            i +=1
            print(loss_value)

        if A_temp.max() == 0:
          A_temp = A
        if i%10==0:
         par*=2

        print("gradient descent stopped at step: "+str(i))
        print("Loss min at the end: "+str(loss_ref))

        return A_temp
    
    def calculate_diff(self, A, base, par, lrs = 1e-3, limits = 1e4,opt = "AdamW",eps=1e-10, params = [1,5,0,0]):
        """ Compare sig_TS with the wanted A.

        Args:
            base (_type_): _description_
            par (int): deprecated
            lrs (float, optional): Learning rate. Defaults to 1e-3.
            limits (int, optional): Number of iterations for optimisation. Defaults to 1e4.
            opt (str, optional): "Adam", "AdamW" or "LBFGS", selected optimizer. Defaults to "AdamW".
            eps (_type_, optional): _description_. Defaults to 1e-10.
            params (list, optional): params are [lambda_length, lambda_frontier, lambda_levy, lambda_ridge].
            [1,5,0,1] , [5,1,0,1] or the same with lambda_ridge = 0 worked well. Defaults to [1,5,0,0].

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
    
        L_control_coeff, bord_control_coeff, LA_control_coeff, ridge_coeff = params
       
       
        if torch.cuda.is_available():
           device = 'cuda'
           dtype = torch.cuda.FloatTensor
        else:
           device = 'cpu'
           dtype = torch.FloatTensor
        
        loss1 = nn.MSELoss().to(device)
       
       
        ########## On calcule les différentes signatures
        signature_base = Signature(depth = self.depth).to(device)

        ### Sig_TS est une collection de tenseurs
        sig_TS = self.sig_TS
        sig_base = signature_base(base) 

        def unflatten_sig2(sig):
            sig_unflat = [np.empty(1) for i in range(self.depth)]
            start = 0
            stop = self.len_base
            for k in range(self.depth):
                sig_unflat[k] = sig[0,start:stop].unflatten(0,[self.len_base]*(k+1))
                start = stop
                stop = start + self.len_base**(k+2)
            return sig_unflat

        sig_base_unf = unflatten_sig2(sig_base.type(dtype))

        d=self.chan
        sig_TS_unf = [sig_TS[:,1:d+1],sig_TS[:,d+1:d+d**2+1].unflatten(dim=-1,sizes = (d,d)),sig_TS[:,d+d**2+1:d+d**2+d**3+1].unflatten(dim=-1,sizes = (d,d,d))]    

        def mode_dot(x, m, mode):
           if mode % 1 != 0:
               raise ValueError('`mode` must be an integer')
           if x.ndim < mode:
               raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
           if m.ndim != 2:
               raise ValueError('Invalid shape of M: {}'.format(m.shape))
           return torch.swapaxes(torch.tensordot(torch.swapaxes(x, mode, -1),m.T,dims = 1), mode, -1)

        def multi_mode_dot(x,m,n):
           MMD = x.T
           for i in range(n):
               MMD = mode_dot(MMD,m,mode = i)
           return MMD.T

        def lengthA(A):
            return torch.sum(torch.norm(A[:,self.decal_length:],dim = 1))   

        def final_points(A,B):  
             RS = torch.matmul(B.float(),A.T)[0]
             checkpoints = RS[[0,-1]]
             return checkpoints.to(device)

        def flatten_MMD(MMD):
            chan_sig = signature_channels(self.chan, self.depth,scalar_term=True)
            sig_flat = torch.ones(chan_sig)
            words = all_words(self.chan, self.depth)
            for k in range(chan_sig-1):
                deg = len(words[k])
                sig_flat[k+1] = MMD[deg-1][words[k]]
            return sig_flat


        def error_func(A):

            #### On calcule [C,A,A,...,A] pour k<= 3
            MMD = [multi_mode_dot(sig_base_unf[i].float(), A, i+1) for i in range(3)]
            MMD_flat = flatten_MMD(MMD)[None].float().to(device)#.type(dtype)
             

            ### Objectif 0
            sig_control = torch.norm(signature_combine(sig_TS.float(),MMD_flat.float(),self.chan,self.depth,scalar_term = True).T[-d**3:])**2
            error = sig_control
            #print(error)

            ### Contrainte de longueur
            if self.time_chan:
                i=1
            else:
                i=0
            L_control = L_control_coeff* loss1(lengthA(A).float(),sig_TS[0,i].float()).float()
            error = error+L_control
            #print(L_control)

            ### controle de bords:
            points = final_points(A,base.flip([-2,-1]).to(device))
            val_to_reach = torch.tensor([sig_TS_unf[2][0,i,i,i] for i in range(self.chan)])
            val_to_move = ((points[-1]-points[0])**3)/6
            bord_control = bord_control_coeff*torch.norm(val_to_reach.float().to(device)-val_to_move.float().to(device))
            error = error+bord_control
            #print(bord_control)
            
            ### Controle Levy Area
            LA_MMD = 0.5*(MMD[1]-MMD[1].T)
            LA = 0.5*(sig_TS_unf[1]-sig_TS_unf[1].T)
            LA_control = LA_control_coeff*torch.norm(LA-LA_MMD)**2
            error = error + LA_control+ridge_coeff*torch.norm(A)
            #print(LA_control+ridge_coeff*torch.norm(A))
            return error

        loss_val = error_func(A.type(dtype)).to(device)
             
        return loss_val
       
def leadlag(X):
    '''
    lead-lag transformation of one dimensional TS [T,1] -> [2T-1, 2] 

    Parameters
    ----------
    X : tensor
        Time serie tensor of size [1,T,d]

    Returns
    -------
    TS_ll : tensor
        Time serie tensor of size [1,2T-1,2d]

        '''
    T = X.shape[-2]
    d = X.shape[-1]
    
    # Initiate path
    lead_lag = np.empty([1,(T-1)*2+1,d*2])
    lead_lag_pair = np.tile(X,2)
    zeros_beg = np.insert(X,0,0,-2)
    #lead_lag_impair = np.concatenate((np.insert(X,T,0).reshape([1,T+1,d]),zeros_beg),axis=-1)
    lead_lag_impair = np.concatenate((np.insert(X,T,0,-2),zeros_beg),axis=-1)
    for t in range(T-1):
        #When t' = 2t, LL = [X_1(t),X_2(t),...,X_n(t),X_1(t),X_2(t),...,X_n(t)]
        lead_lag[:,2*t] = lead_lag_pair[:,t,:]
        #When t' = 2t+1, LL = [X_1(t+1),X_2(t+1),...,X_n(t+1),X_1(t),X_2(t),...,X_n(t)]
        lead_lag[:,2*t+1] = lead_lag_impair[:,t+1,:]
    lead_lag[:,-1] = lead_lag_pair[:,-1]
    return torch.tensor(lead_lag)
       

    
def linear_base(x,i,M,T):
    step1 = int((i-1)*len(T)/M)
    step2 = int(i*len(T)/M)
    path = np.zeros(x.shape)
    path[:step1] = 0
    path[step1:step2] = (M*x[step1:step2]-(i-1))
    path[step2:] = 1
    return path  

def brown(size,sig=1):
    jump = np.random.normal(loc = 0, scale = sig, size = size)
    if type(size) is not int:
        jump = jump.reshape(size)
    return jump.cumsum(axis=0)