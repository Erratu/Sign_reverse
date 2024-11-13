from pga import pga
from pga import tangent_pga

from pga import recenter_group_elems
from pga import d2h
from utils import depth_inds

from scipy.optimize import fsolve

import numpy as np
import signatory
import matplotlib.pyplot as plt

def compute_pga(path, sig_level, tangent = True, n_components = 3):
  print('Computing signatures')
  sig = signatory.Signature(depth  = sig_level)
  signatures = sig(path,basepoint = True)
  channels = path.shape[-1]

  if tangent:
      print('Computing tangent pga')
      t_principal_directions = tangent_pga(signatures.numpy(), channels, sig_level, n_components=n_components)
  else:
      print('Computing pga')
      t_principal_directions = pga(signatures.numpy(), channels, sig_level, n_components=n_components)
  return t_principal_directions, signatures


def compute_projection(t_principal_directions,path,channels,depth):
  print('Computing tangent pga projections')
  channels = path.shape[-1]
  inds = depth_inds(channels, depth)
  sig = signatory.Signature(depth  = depth)
  signatures = sig(path,basepoint = True)
  projections_t = {}
  n_components = len(t_principal_directions)
  for K in range(n_components):
      optimized_v_t = t_principal_directions[K]
      SXk = recenter_group_elems(signatures.numpy(), channels, depth, inds)
      tis_t = []
      for i in range(len(signatures.numpy())):
          d2h_t = lambda t: d2h(t, optimized_v_t, SXk[i], depth, channels)
          initial_guess_t = 0.
          optimized_ti_t = fsolve(d2h_t, initial_guess_t)[0]
          tis_t.append(optimized_ti_t)
          print(f'Data #{i} with component #{K}')
      tis_t = np.array(tis_t)
      projections_t[K] = tis_t
  var_t = [np.var(projections_t[k]) for k in range(n_components)]
  print('Variance expliquée par les 3 premières géodésiques principales'+str(var_t))

  return projections_t

def visualisation_normales(projections,colour):


  for pg in range(1,len(projections)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(projections[0])):
        ax.scatter(projections[0][i],projections[pg][i], color=colour[i])
    plt.title('Data vizualisation on the two first principal geodesics')
    plt.xlabel("PG1")
    plt.ylabel(f"PG{pg+1}")
    plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(111,projection = '3d')

  for i in range(len(projections[0])):
      ax.scatter(projections[0][i],projections[1][i],projections[2][i] , color=colour[i])
  plt.title('Visualisation des données sur les 3 premières géodésiques')
  plt.show()

def visualisation_all(projections_norm,projections_anorm):

  for pg in range(1,len(projections_norm)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(projections_norm[0])):
        ax.scatter(projections_norm[0][i],projections_norm[pg][i], color=['blue'])
    for i in range(len(projections_anorm[0])):
        ax.scatter(projections_anorm[0][i],projections_anorm[pg][i], color=['red'])

    plt.title('Data vizualisation on the two first principal geodesics')
    plt.xlabel("PG1")
    plt.ylabel(f"PG{pg+1}")
    plt.legend(['Benign','Malignant'])
    plt.show()
