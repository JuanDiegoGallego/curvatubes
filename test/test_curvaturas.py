import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import string
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
def formatted(f): return format(f, '.3f').rstrip('0').rstrip('.')
from utils import *

import torch
dtype = torch.cuda.FloatTensor # we work with GPUs # if torch.cuda.is_available() else torch.FloatTensor

# curvatubes imports
from cvtub.utils import slices, single, load_nii, save_nii, random_init, init_balls
from cvtub.energy import discrepancy, ratio_discr

print(os.path.abspath(niifolder))

d=500
shape = [d,d,d]
center = [d/2,d/2,d/2]
radius = 200
vx_sz =  1
vol = np.zeros(shape)

n=len(shape)
semisizes = (radius,)*len(shape) #[30,30,30]

x,y,z = np.indices(vol.shape)
distances = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
mask = (distances)**2<radius**2
vol[mask] = 1

write_mrc(vol.astype(np.float32), niifolder+"esfera", v_size=vx_sz, dtype=None, no_saxes=True)

kap1_eps, kap2_eps =  kap_eps(vol, eps = 0.02, delta_x=vx_sz, mode = 'periodic', xi = 1e-6)
print("kap1_eps: ", kap1_eps[np.abs(kap1_eps) < 100].max())
print("kap2_eps: ", kap1_eps[np.abs(kap2_eps) < 100].max())

kap1_vals, kap2_vals, areas = curvhist(vol, kap1_eps, kap2_eps, lev = 0, delta_x = vx_sz,show_figs = False)
print("kap1_vals: ", kap1_vals[np.abs(kap1_vals) < 100].max())
print("kap2_vals: ", kap1_vals[np.abs(kap2_vals) < 100].max())

curvaturas, superficie = calc_curvatura_gauss(vol.astype(np.float32))
#write_mrc(curvaturas.astype(np.float32), niifolder+"curvaturas", v_size=vx_sz, dtype=None, no_saxes=True)
#write_mrc(curvaturas.astype(np.float32), niifolder+"superficie", v_size=vx_sz, dtype=None, no_saxes=True)

curvatura_media = np.sum(curvaturas)/np.sum(superficie)
print("La curvatura media es de: ", curvatura_media)
