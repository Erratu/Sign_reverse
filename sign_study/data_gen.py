import numpy as np
import matplotlib.pyplot as plt
from signatory import Signature
import torch
from scipy.stats import ks_2samp, wasserstein_distance, pearsonr, skew, kurtosis, kruskal
import scikit_posthocs as sp
import pandas as pd


def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def generate_independent_series(T):
    return normalize(np.cumsum(np.random.randn(T)))

def generate_correlated_series(X, corr_strength=0.5):
    # Le bruit doit être indépendant de X
    bruit = np.random.normal(0, 1, X.shape)

    X_norm = (X - np.mean(X)) / np.std(X)
    bruit_norm = (bruit - np.mean(bruit)) / np.std(bruit)

    Y_norm = corr_strength * X_norm + np.sqrt(1 - corr_strength**2) * bruit_norm

    Y = Y_norm

    #correlation_obtenue = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
    #print(f"Coefficient de corrélation visé : {corr_strength}")
    #print(f"pearsonr : {pearsonr(X.flatten(), Y.flatten())[0]}")
    #print(f"Coefficient de corrélation obtenu : {correlation_obtenue:.4f}")

    return normalize(Y)

def simulate_cde_trajectory(X, V, y0=None):
    """
    Simule la trajectoire complète d'un CDE linéaire :
        dY_t = sum_i V_i Y_t dX_t^i
    par intégration d'Euler.
    
    X : (T, d) chemin de contrôle
    V_list : liste de matrices (m, m)
    y0 : vecteur initial (m,)
    
    Retour : Y (T, m)
    """
    T = X.shape[0]
    if y0 is None:
        y0 = 0
    
    Y = np.zeros(T)
    Y[0] = y0
    
    dX = np.diff(X, prepend=X[0])
    
    for t in range(1, T):
        Y[t] = Y[t-1] + V * Y[t-1] * dX[t]

    return normalize(Y)
# ===============================
# Exemple d’utilisation
# ===============================

def sign_calcul(X, Y):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    XY = torch.cat([X, Y], dim=1).unsqueeze(0)  # batch dim
    sig = signature_TS(XY)

    S_12 = sig[0, -3]
    S_21 = sig[0, -2]

    return S_12, S_21

def compare_marginals(A, B):
    ks_x = ks_2samp(A[:,0], B[:,0])
    ks_y = ks_2samp(A[:,1], B[:,1])
    wass_x = wasserstein_distance(A[:,0], B[:,0])
    wass_y = wasserstein_distance(A[:,1], B[:,1])
    return {"KS_12": ks_x.pvalue, "KS_21": ks_y.pvalue, "Wass_12": wass_x, "Wass_21": wass_y}

def dataset_correlation(X):
    # X shape : (N=200, 2, T=100)
    corrs = []
    for i in range(X.shape[0]):
        x_series = X[i,0]
        y_series = X[i,1]
        r, _ = pearsonr(x_series, y_series)
        corrs.append(r)
    return np.mean(corrs)

def dataset_creation(type, corr_strength=0.8, V=0.1):
    sigs = np.empty((N,2))
    data = np.empty((N,2,T)) 
    for i in range(N):
        X1 = generate_independent_series(T)
        if type == 'corr':
            Y1 = generate_correlated_series(X1, corr_strength)
        elif type == 'cde':
            Y1 = simulate_cde_trajectory(X1, V, y0=1.0)
        elif type == 'indep':
            Y1 = generate_independent_series(T)
        S1, S2 = sign_calcul(X1, Y1)
        sigs[i,0] = S1
        sigs[i,1] = S2
        data[i,0] = X1
        data[i,1] = Y1
    return sigs, data

def datasets_creation_3():
    datasets_sig = {}
    datasets = {}
    
    # --- couples corrélés ---
    sigs, data = dataset_creation('corr', corr_strength=0.4)

    datasets["corr_1"] = data
    datasets_sig["corr_1"] = sigs

    sigs, data = dataset_creation('corr', corr_strength=0.9)

    datasets["corr_2"] = data
    datasets_sig["corr_2"] = sigs

    # --- couples avec CDE ---
    V = np.random.randn()*0.1
    sigs, data = dataset_creation('cde', V=V)

    datasets["cde_1"] = data
    datasets_sig["cde_1"] = sigs

    V = np.random.randn()*0.1
    sigs, data = dataset_creation('cde', V=V)

    datasets["cde_2"] = data
    datasets_sig["cde_2"] = sigs
    
    sigs, data = dataset_creation('indep')

    datasets["indep"] = data
    datasets_sig["indep"] = sigs

    return datasets, datasets_sig

def comparaison_3():
    datasets, datasets_sig = datasets_creation_3()

    print("Indep vs Corr:", compare_marginals(datasets_sig["indep"], datasets_sig["corr_2"]))
    print("Indep vs CDE:", compare_marginals(datasets_sig["indep"], datasets_sig["cde_1"]))
    print("Corr vs CDE:", compare_marginals(datasets_sig["corr_2"], datasets_sig["cde_1"]))
    print("CDE coeffs:", compare_marginals(datasets_sig["cde_1"], datasets_sig["cde_2"]))

    # --- Corrélation dans chaque dataset ---
    #for name, data in datasets.items():
    #    r = dataset_correlation(data)
    #    print(f"Corrélation globale {name} : {r:.2f}")

    # --- Visualisation ---
    plt.figure(figsize=(15,10))
    for i, (name, data) in enumerate(datasets_sig.items(), 1):
        plt.subplot(2,3,i)
        plt.scatter(data[:,0], data[:,1])
        plt.title(name)
        plt.xlabel("S_12")
        plt.ylabel("S_21")
    plt.tight_layout()
    plt.show()

def datasets_creation_corr():

    sigs = []
    sigs.append(dataset_creation('corr', corr_strength=0.1)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.2)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.3)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.4)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.5)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.6)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.7)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.8)[0])
    sigs.append(dataset_creation('corr', corr_strength=0.9)[0])
    sigs.append(dataset_creation('corr', corr_strength=1)[0])

    plt.figure(figsize=(20,15))
    for i, data in enumerate(sigs, 1):
        plt.subplot(3,4,i)
        plt.scatter(data[:,0], data[:,1])
        plt.title(0.1*i)
        plt.xlabel("S_12")
        plt.ylabel("S_21")
    plt.tight_layout()
    plt.show()

def datasets_creation_cde():

    sigs = []
    for i, V in enumerate(np.linspace(0.1, 2.0, 12)):
        sigs.append(dataset_creation('cde', V=V)[0])

    plt.figure(figsize=(20,15))
    for i, data in enumerate(sigs, 1):
        plt.subplot(3,4,i)
        plt.scatter(data[:,0], data[:,1])
        plt.title(V)
        plt.xlabel("S_12")
        plt.ylabel("S_21")
    plt.tight_layout()
    plt.show()

def extract_features(F):
    feats = {}
    feats["mean"] = np.mean(F)
    feats["std"] = np.std(F)
    feats["skew"] = skew(F)
    feats["kurtosis"] = kurtosis(F)
    q = np.quantile(F, [0.1,0.25,0.5,0.75,0.9])
    for i, qi in enumerate(q):
        feats[f"q{i}"] = qi
    return feats

def find_indics(sign=0):
    datasets, datasets_sig = datasets_creation_3()
    methods = ["indep","corr_1","corr_2","cde_1","cde_2"]
    #all_features = []
    #for name, data in datasets_sig.items():
    #    feats = extract_features(data[:,0])
    #    feats["method"] = name
    #    all_features.append(feats)
#
    #df = pd.DataFrame(all_features)
    #print(df["method"].value_counts())
    #indicators = [c for c in df.columns if c != "method"]
    #for ind in indicators:
    #    groups = [df[df.method==m][ind].values for m in methods]
    #    print(groups)
    #    H, p = kruskal(*groups)
    #    print(f"{ind:10s} -> H={H:.3f}, p={p:.4f}")

    groups = [datasets_sig[m][:,0] for m in methods]
    H, p = kruskal(*groups)
    print(f"Kruskal-Wallis: H={H:.3f}, p={p:.4f}")  

    # Comparaisons multiples après Kruskal
    data = np.concatenate(groups)
    labels = np.concatenate([[m]*len(g) for m, g in zip(methods, groups)])
    df_long = pd.DataFrame({"value": data, "group": labels})

    # Post-hoc Dunn
    for m, g in zip(methods, groups):
        print(m, np.median(g))
    posthoc = sp.posthoc_dunn(df_long, val_col="value", group_col="group", p_adjust="bonferroni")
    print(posthoc)

def TS_graph():
    colors = {"indep":"blue", "corr_1":"lightgreen", "corr_2":"green", "cde_2":"red", "cde_1":"orange"}
    datasets, _ = datasets_creation_3()
    plt.figure(figsize=(10,5))

    for m in colors:
        series = datasets[m][0]
        plt.plot(series[1], label=m, color=colors[m])
        plt.plot(series[0], label=m, color=colors[m])

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Comparaison des séries par méthode")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------
# Main : génération
# -----------------------

if __name__ == "__main__":
    T = 100
    N = 200

    signature_TS = Signature(depth = 2,scalar_term= True).to("cpu")

    TS_graph()