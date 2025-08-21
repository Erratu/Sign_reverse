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

def sign_calcul(X, Y):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (T,)->(T,1) ou (B, T)->(B, T, 1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
    XY = torch.cat([X, Y], dim=-1)  # batch dim : (T, 2) ou (B, T, 2)
    if len(XY.shape) < 3:
        XY = XY.unsqueeze(0)
    
    sig = signature_TS(XY)
    S_12 = sig[:, -3]
    S_21 = sig[:, -2]

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
    """generate X and Y with the type of generation

    Args:
        type (str): if the data will be correlated, generated with a cde or independant
        corr_strength (float, optional): stregth of the correlation it the correlation type is choose. Defaults to 0.8.
        V (float, optional): coefficient of the cde it the cde type is choose. Defaults to 0.1.

    Returns:
        ndarray, ndarray: 2 ndarrays, one with the sigs of shape (N,2) and the other with the data of shape (N,2,T)
    """
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
        sigs[i,0] = S1[0]
        sigs[i,1] = S2[0]
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

def central_quantile(arr):
    intervals = [50, 90]

    # centré autour de 0
    for alpha in intervals:
        # calcul du quantile à inclure moitié de chaque côté
        q = np.percentile(np.abs(arr), alpha/2)
        plt.axvline(-q, color="orange", linestyle="--", linewidth=1.2)
        plt.axvline(q, color="orange", linestyle="--", linewidth=1.2)
        plt.text(q, plt.ylim()[1]*0.85, f"{alpha}%", rotation=90,
                 color="orange", va="top", ha="center", fontsize=8)
    
    # Ligne 0
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.text(0, plt.ylim()[1]*0.9, "0", rotation=90, color="red", va="top", ha="center", fontsize=8)


    #centré autour de la médiane (aussi possible : centré autour du pic avec kde)
    #for alpha in intervals:
    #    lower = (100 - alpha)/2
    #    upper = 100 - lower
    #    q_low, q_high = np.percentile(arr, [lower, upper])
    #    plt.axvline(q_low, color="orange", linestyle="--", linewidth=1.2)
    #    plt.axvline(q_high, color="orange", linestyle="--", linewidth=1.2)
    #    plt.text(q_high, plt.ylim()[1]*0.85, f"{alpha}%", rotation=90,
    #             color="orange", va="top", ha="center", fontsize=8)

    ## Ligne pour la médiane
    #median = np.median(arr)
    #plt.axvline(median, color="red", linestyle="--", linewidth=1.5)
    #plt.text(median, plt.ylim()[1]*0.9, "median", rotation=90,
    #         color="red", va="top", ha="center", fontsize=8)

def hd_interval(arr,alpha=0.9):
    """Retourne le plus petit intervalle contenant alpha% des données."""
    sorted_arr = np.sort(arr)
    n = len(arr)
    k = int(np.floor(alpha * n))
    intervals = sorted_arr[k:] - sorted_arr[:n-k]   # largeur de tous les intervalles possibles
    min_idx = np.argmin(intervals)                  # index de l'intervalle le plus petit
    return sorted_arr[min_idx], sorted_arr[min_idx + k]

def graph_hdi(arr):
    # Intervalle HDI à 90%
    lower, upper = hd_interval(arr, alpha=0.9)
    plt.axvline(lower, color="orange", linestyle="--", linewidth=1.5)
    plt.axvline(upper, color="orange", linestyle="--", linewidth=1.5)
    plt.text(upper, plt.ylim()[1]*0.85, "90% HDI", rotation=90, color="orange", va="top", ha="center", fontsize=8)
    
    # Intervalle HDI à 50%
    lower, upper = hd_interval(arr, alpha=0.5)
    plt.axvline(lower, color="yellow", linestyle="--", linewidth=1.5)
    plt.axvline(upper, color="yellow", linestyle="--", linewidth=1.5)
    plt.text(upper, plt.ylim()[1]*0.85, "50% HDI", rotation=90, color="yellow", va="top", ha="center", fontsize=8)
    # quantile à 5-95% pour capter l'asymétrie
    qs = [5, 95]
    quantiles = np.percentile(arr, qs)

    # Ajout des lignes verticales
    for q, val in zip(qs, quantiles):
        plt.axvline(val, color="red" if q==50 else "green", linestyle="--", linewidth=1.5)
        plt.text(val, plt.ylim()[1]*0.85, f"{q}%", rotation=90,
                color="red" if q==50 else "green", va="top", ha="center", fontsize=8)
        
    return upper-lower
        
def peak_measure(arr):
    # proportion dans un petit intervalle autour du pic
    peak = np.median(arr)  # ou mode via KDE
    delta = 0.1 * (arr.max()-arr.min())  # intervalle autour du pic
    prop = np.mean((arr > peak-delta) & (arr < peak+delta))

    from scipy.stats import kurtosis
    k = kurtosis(arr, fisher=True)  # kurtosis de Fisher, 0 pour Gauss

    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    kde = gaussian_kde(arr)
    xs = np.linspace(arr.min(), arr.max(), 1000)
    ys = kde(xs)
    peaks, _ = find_peaks(ys, height=0.05)  # seuil à ajuster
    n_peaks = len(peaks)

    hist, _ = np.histogram(arr, bins=30, density=True)
    hist = hist[hist>0]
    entropy = -np.sum(hist * np.log(hist))

    #print(prop, k, n_peaks, entropy)

    return prop, k

def comparaison_3():
    _, datasets_sig = datasets_creation_3()

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

    plt.figure(figsize=(15,10))
    for i, (name, data) in enumerate(datasets_sig.items(), 1):
        plt.subplot(2,3,i)
        plt.hist(data[:,0], bins=20, color="tab:blue")
        plt.title(name)
        plt.ylabel("S_12")
        graph_hdi(data[:,0])
        peak_measure(data[:,0])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, (name, data) in enumerate(datasets_sig.items(), 1):
        plt.subplot(2,3,i)
        plt.hist(data[:,1], bins=20, color="tab:blue")
        plt.title(name)
        plt.ylabel("S_21")
        graph_hdi(data[:,1])
        peak_measure(data[:,1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    print("Levy")
    for i, (name, data) in enumerate(datasets_sig.items(), 1):
        plt.subplot(2,3,i)
        plt.hist(0.5*(data[:,0]-data[:,1]), bins=20, color="tab:blue")
        plt.title(name)
        plt.ylabel("Levy area")
        hdi = graph_hdi(0.5*(data[:,0]-data[:,1]))
        print(name, hdi)
        peak_measure(0.5*(data[:,0]-data[:,1]))
    plt.tight_layout()
    plt.show()

def test_hdi_indic():
    N = 5
    for i in range(N):
        _, datasets_sig = datasets_creation_3()

        plt.figure(figsize=(15,10))
        print("Levy")
        for i, (name, data) in enumerate(datasets_sig.items(), 1):
            plt.subplot(2,3,i)
            plt.hist(0.5*(data[:,0]-data[:,1]), bins=20, color="tab:blue")
            plt.title(name)
            plt.ylabel("Levy area")
            hdi = graph_hdi(0.5*(data[:,0]-data[:,1]))
            print(name, hdi)
            peak_measure(0.5*(data[:,0]-data[:,1]))
        plt.tight_layout()
        plt.show()

def test_levi_separation():
    measures = {"indep":[],"corr_1":[],"corr_2":[],"cde_1":[],"cde_2":[]}
    for _ in range(20):
        _, datasets_sig = datasets_creation_3()
        for _, (name, data) in enumerate(datasets_sig.items(), 1):
            measures[name].append(peak_measure(0.5*(data[:,0]-data[:,1])))

    # Pourcentiles à calculer
    quantiles = [0, 25, 50, 75, 100]

    summary = {}

    for var, samples in measures.items():
        arr = np.array(samples)  # shape (n_samples, 2)
        stats = {}
        for i, name in enumerate(["prop", "k"]):
            vals = arr[:, i]
            stats[name] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "quantiles": {f"{q}%": np.percentile(vals, q) for q in quantiles}
            }
        summary[var] = stats

    # Affichage
    import pprint
    pprint.pprint(summary)

def search_thre():
    N = 500
    measures = {"indep":[],"corr":[],"cde":[]}
    for _ in range(N):
        _, datasets_sig = datasets_creation_3()
        for _, (name, data) in enumerate(datasets_sig.items(), 1):
            measures[name.split("_")[0]].append(peak_measure(0.5*(data[:,0]-data[:,1])))

    k_values = np.linspace(0, 5, 50)      # à ajuster selon tes données
    prop_values = np.linspace(0, 1, 50)

    best_score = N*5
    best_thresh = (0,0)

    for k_thresh in k_values:
        for prop_thresh in prop_values:
            # Compter les variables qui respectent le seuil
            valid = [v for v in measures["cde"] if v[1] >= k_thresh and v[0] >= prop_thresh]
            invalid = [v for v in measures["indep"]+measures["corr"] if v[1] >= k_thresh and v[0] >= prop_thresh]
            score = len(measures["cde"]) - len(valid) + len(invalid)  # ici score = nb de variables correctement classées
            if score < best_score:
                best_score = score
                best_thresh = (k_thresh, prop_thresh)

    print(f"Seuil optimal : k >= {best_thresh[0]:.3f}, prop >= {best_thresh[1]:.3f}")
    invalid_percent = (best_score / N) * 20
    print(f"Pourcentage de variables qui ne respectent pas le seuil : {invalid_percent:.2f}%")

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

    #plt.figure(figsize=(15,10))
    #for i, data in enumerate(sigs, 1):
    #    plt.subplot(3,4,i)
    #    plt.scatter(data[:,0], data[:,1])
    #    plt.title(i)
    #    plt.xlabel("S_12")
    #    plt.ylabel("S_21")
    #plt.tight_layout()
    #plt.show()
#
    #plt.figure(figsize=(15,10))
    #for i, data in enumerate(sigs, 1):
    #    plt.subplot(3,4,i)
    #    plt.hist(data[:,0], bins=20, color="tab:blue")
    #    plt.title(i)
    #    plt.ylabel("S_12")
    #    graph_hdi(data[:,0])
    #    peak_measure(data[:,0])
    #plt.tight_layout()
    #plt.show()
#
    #plt.figure(figsize=(15,10))
    #for i, data in enumerate(sigs, 1):
    #    plt.subplot(3,4,i)
    #    plt.hist(data[:,1], bins=20, color="tab:blue")
    #    plt.title(i)
    #    plt.ylabel("S_21")
    #    graph_hdi(data[:,1])
    #    peak_measure(data[:,1])
    #plt.tight_layout()
    #plt.show()

    plt.figure(figsize=(15,10))
    print("Levi")
    for i, data in enumerate(sigs, 1):
        plt.subplot(3,4,i)
        plt.hist(0.5*(data[:,0]-data[:,1]), bins=20, color="tab:blue")
        plt.title(i)
        plt.ylabel("Levy area")
        hdi = graph_hdi(0.5*(data[:,0]-data[:,1]))
        print(i, hdi)
    plt.tight_layout()
    plt.show()

    sigs = []
    for i, corr in enumerate(np.linspace(0.1, 1.0, 12)):
        sigs.append(dataset_creation('corr', corr_strength=corr)[0])

    plt.figure(figsize=(15,10))
    print("Levi")
    for i, data in enumerate(sigs, 1):
        plt.subplot(3,4,i)
        plt.hist(0.5*(data[:,0]-data[:,1]), bins=20, color="tab:blue")
        plt.title(i)
        plt.ylabel("Levy area")
        hdi = graph_hdi(0.5*(data[:,0]-data[:,1]))
        print(i, hdi)
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

def offset_test():
    V = np.random.randn()*0.1
    _, data = dataset_creation('cde', V=V)

    def generate_shifted_XY(data, max_lag=8):
        """
        Crée une liste de versions de data où Y est décalé par rapport à X.

        data : array (N, 2, T)
        max_lag : entier, nombre de décalages à générer

        Retourne :
            shifted_datasets : liste de longueur max_lag
                               chaque élément est un array (N, 2, T-lag)
        """

        X = data[:, 0]   # (N, T)
        N,T = X.shape
        Y = data[:, 1, :T-max_lag]   # (N, T-max_lag)

        shifted_datasets = []
        for lag in range(0, max_lag+1):
            X_aligned = X[:, lag:T-max_lag+lag]     # (N, T-max_lag)
            S1, S2 = sign_calcul(X_aligned, Y) # (N,)
            shifted_datasets.append(np.array([S1, S2]))
        return shifted_datasets

    dataset = generate_shifted_XY(data)

    # --- Visualisation ---
    plt.figure(figsize=(15,10))
    for i, data in enumerate(dataset, 1):
        plt.subplot(3,3,i)
        plt.scatter(data[0], data[1])
        plt.title(i-1)
        plt.xlabel("S_12")
        plt.ylabel("S_21")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, data in enumerate(dataset, 1):
        plt.subplot(3,3,i)
        plt.hist(data[0], bins=20, color="tab:blue")
        plt.title(i-1)
        plt.ylabel("S_12")
        graph_hdi(data[0])
        peak_measure(data[0])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, data in enumerate(dataset, 1):
        plt.subplot(3,3,i)
        plt.hist(data[1], bins=20, color="tab:blue")
        plt.title(i-1)
        plt.ylabel("S_21")
        graph_hdi(data[1])
        peak_measure(data[1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, data in enumerate(dataset, 1):
        plt.subplot(3,3,i)
        plt.hist(0.5*(data[0]-data[1]), bins=20, color="tab:blue")
        plt.title(i-1)
        plt.ylabel("Levy area")
        graph_hdi(0.5*(data[0]-data[1]))
        peak_measure(0.5*(data[0]-data[1]))
    plt.tight_layout()
    plt.show()

def test_with_data():
    N=100

    df = pd.read_csv("sign_study/data_test_GECCO.csv").dropna()
    data = df.iloc[:,-3].values.astype(float)

    df = pd.read_csv("sign_study/data_test_TADA.csv").dropna()
    data = df.iloc[:,1].values.astype(float)

    mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
    std = np.where(std == 0, 1e-8, std)
    data_norm = (data - mean) / std
    T = data_norm.shape[0]
    y_ind = np.empty((N, 2))
    y_cde = np.empty((N, 2))
    y_corr = np.empty((N, 2))
    for i in range(N):
        gen = generate_independent_series(T)
        S12, S21 = sign_calcul(data_norm, gen)
        y_ind[i] = S12[0], S21[0]
        gen = simulate_cde_trajectory(data_norm, V=np.random.randn()*0.1, y0=1.0)
        S12, S21 = sign_calcul(data_norm, gen)
        y_cde[i] = S12[0], S21[0]
        gen = generate_correlated_series(data_norm, corr_strength=np.random.random())
        S12, S21 = sign_calcul(data_norm, gen)
        y_corr[i] = S12[0], S21[0]

    # --- Visualisation ---
    plt.figure(figsize=(15,10))
    for i, data in enumerate([y_ind,y_cde,y_corr], 1):
        plt.subplot(1,3,i)
        plt.scatter(data[:,0], data[:,1])
        plt.title(i-1)
        plt.xlabel("S_12")
        plt.ylabel("S_21")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, data in enumerate([y_ind,y_cde,y_corr], 1):
        plt.subplot(1,3,i)
        plt.hist(data[:,0], bins=20, color="tab:blue")
        plt.title(i-1)
        plt.ylabel("S_12")
        graph_hdi(data[:,0])
        peak_measure(data[:,0])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, data in enumerate([y_ind,y_cde,y_corr], 1):
        plt.subplot(1,3,i)
        plt.hist(data[:,1], bins=20, color="tab:blue")
        plt.title(i-1)
        plt.ylabel("S_21")
        graph_hdi(data[:,1])
        peak_measure(data[:,1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,10))
    for i, data in enumerate([y_ind,y_cde,y_corr], 1):
        plt.subplot(1,3,i)
        plt.hist(0.5*(data[:,0]-data[:,1]), bins=20, color="tab:blue")
        plt.title(i-1)
        plt.ylabel("Levy area")
        graph_hdi(0.5*(data[:,0]-data[:,1]))
        peak_measure(0.5*(data[:,0]-data[:,1]))
    plt.tight_layout()
    plt.show()

# -----------------------
# Main : génération
# -----------------------

if __name__ == "__main__":
    T = 100
    N = 200

    signature_TS = Signature(depth = 2,scalar_term= True).to("cpu")

    datasets_creation_cde()