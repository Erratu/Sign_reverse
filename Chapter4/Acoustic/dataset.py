import pandas as pd
import torch
from scipy import io
from os import listdir
import numpy as np
import mne


def load_MultiTS(dataset,num = 3):
    if dataset == "finance":
        # load data
        df = pd.read_csv("All_data_MTS/Financial_data/NYSE_119stocks_2000Jan_2021June_withdates.csv")
        # preprocessing for signature computation
        array_data = df.iloc[:,1:].to_numpy()
        reshaped_array = array_data.reshape((1, *array_data.shape))
        MultiTS = torch.tensor(reshaped_array)
        return MultiTS
    
    if dataset == "epilepsy":
        if num <10:
            file = "./All_data_MTS/EEG_epil/chb01_0"+str(num)+".edf"
        else:
            file = "./All_data_MTS/EEG_epil/chb01_"+str(num)+".edf"
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()
        names = data.ch_names
        
        MultiTS = torch.tensor(raw_data.T)
        
        return MultiTS, names
    
    if dataset == "fMRI":
        l_MultiTS = []
        for individu in listdir("All_data_MTS/HCP_data/HCP_TS"):
            dic = io.loadmat("All_data_MTS/HCP_data/HCP_TS/"+individu+"/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat")
            array_data = dic["TS"].T
            MultiTS = array_data.reshape((1, *array_data.shape))
            l_MultiTS.append(torch.tensor(MultiTS))
        return l_MultiTS
    
    if dataset == "SOS":
        df = pd.read_csv("./All_data_MTS/SOS/data.csv")
        # add conlumn date
        date_format = '%d/%m/%Y %H:%M:%S'
        df['date'] =  pd.to_datetime(df['DATE_ENTREE_VISITE'], format=date_format).dt.date
        call_volume_by_date = df.groupby('date').count()['id']
        # counts of top 15 motifs
        top_motifs = ["VOMIT","FIEVRE","TOUX","DIARRHEE","RHINO","DL GORGE","MIGRAINE","NAUSEE","DL ABDO","DL ESTOMAC","VERTIGES","CEPHALEE"]
        melted = pd.melt(df, id_vars=['date'], value_vars=['MOTIF1', 'MOTIF2', 'MOTIF3'], var_name='Motif')
        motif_counts_date = melted.groupby(['date','value'])['value'].count().reset_index(name='count')
        lst_m = []
        for motif in top_motifs:
            serie = motif_counts_date.loc[(motif_counts_date['value']==motif)]['count'].rename(motif)
            serie.index = call_volume_by_date.index
            lst_m.append(serie)
        # dataset by date
        df_date = pd.concat(lst_m, axis=1)
        array_data = df_date.values
        reshaped_array = array_data.reshape((1, *array_data.shape))
        MultiTS = torch.tensor(reshaped_array).float()
        return MultiTS
    if dataset == "sleep":
        if num <10:
            file = "./All_data_MTS/EEG_sleep/proj2/Normal_Subject_0"+str(num)+".edf"
        else:
            file = "./All_data_MTS/EEG_sleep/proj2/Normal_Subject_"+str(num)+".edf"
        
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()
        names = data.ch_names
        
        MultiTS = torch.tensor(raw_data.T)
        
        return MultiTS, names
    
    if dataset == "cenosia":
        i=1
        name = "./2024-08-16_6.csv"
        data_imp = pd.read_csv(name,encoding = "ISO-8859-1",
                       sep = ";")
        data_imp.fillna(0, inplace = True)
        data_hz = torch.tensor(MA_cen(data_imp.drop(columns = ['timestamp','LAeq','LAeq(15mn)']).to_numpy()))
        data_laeq = torch.tensor(MA_cen(data_imp.iloc[:,[1,-2]].to_numpy()))

        return data_hz, data_laeq

    raise ValueError("Unsupported dataset: " + dataset)
   

'''    
def preprocess(MultiTS, dim = 0, time = False, scale = True):
   if dim == 0:
       preproc = torch.cat((torch.zeros(size = (1,MultiTS.shape[-1])),(MultiTS - MultiTS.mean(dim=0))),dim=0)
   else:        
       preproc = torch.cat((torch.zeros(size = (1,MultiTS.shape[-1])),(MultiTS.T - MultiTS.mean(dim=1)).T),dim=0)
   if scale:
       preproc = preproc/torch.max(preproc)
   return preproc
'''

    
def preprocess(MultiTS, scale = True):
   preproc = torch.cat((torch.zeros(size = (1,1,MultiTS.shape[-1])),MultiTS - MultiTS.mean(dim=1)),dim=1)
   if scale:
       preproc = preproc/torch.max(preproc)
   return preproc

def MA(TS,win_MA, median = False):
    T_new = int(TS.shape[0]/win_MA)
    MA = torch.empty(size = [T_new,TS.shape[1]])
    for t in range(T_new):
        start = int(t*win_MA)
        stop = int((t+1)*win_MA)
        if not median:
            MA[t] = TS[start:stop].mean(axis=0)
        if median:
            MA[t] = torch.median(TS[start:stop],axis=0)[0]
    return MA

def MA_betti(betti,win,median = False):
    ma_betti = np.zeros(len(betti)-win)

    l = int(win/2)
    for i in range(len(betti)-win):
        if i < l:
            if not median:
                ma_betti[i] = np.mean(betti[:i])
            if median:
                ma_betti[i] = np.median(betti[:i])
        else:
            if not median:
                ma_betti[i] = np.mean(betti[i-l:i+l])
            if median:
                ma_betti[i] = np.median(betti[i-l:i+l])
    return ma_betti

def MA_cen(TS):
    L = TS.shape[0]
    win = 15*60
    mi_win = 15*30
    TS_MA = np.empty(TS.shape)
    TS_MA[0] = TS[0]
    for i in range(1,L):
        if i<win:
            TS_MA[i] = TS[:i].mean(axis=0)
        elif i>L-win-1:
            TS_MA[i] = TS[L-i-1:].mean(axis=0)
        else:
            TS_MA[i] = TS[i-mi_win:i+mi_win].mean(axis=0)
    return TS_MA
    
    