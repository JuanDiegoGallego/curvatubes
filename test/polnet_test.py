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

import time
start_time = time.time()

# Number of iterations?
MAXEVAL = 5000
# 4000 are generally sufficient, 8000 are over-sufficient

# Sizes
Z,X,Y = 300, 300, 250
delta_x = 1/300

# Initialize A = A0: a random vector field with values in R^3
A0 = 40 * delta_x * np.random.rand(3,Z,X,Y)
A0 = torch.Tensor(A0).type(dtype)
# slices(A0[0]) # show three 2D slices of the 3D scalar field A0[0]

# Parameters
eps = 0.02
# a20, a11, a02, b10, b01, c = 1, 0.85, 6, -80, -7.5, 0
a20, a11, a02, b10, b01, c = 1, 1, 1, 0, 0, 0 # Layers
M0 = -0.3
params = eps, a20, a11, a02, b10, b01, c

# Generate the shape - you will see slices of u along the iterations, and the loss curves are saved in their folder
u = generate(A0, params, M0, delta_x, maxeval = MAXEVAL, snapshot_folder = niifolder,display_all=False)

# Save u as a .nii.gz file
str_tup = tuple([ formatted(x) for x in [eps, a20, a11, a02, b10, b01, c, M0]])
name = 'eps {} coeffs [{}, {}, {}, {}, {}, {}] m {}'.format(*str_tup)
print(name)
# save_nii(np.floor(100 * u) / 100, niifolder + name) # or u if you have more space

# Save also as .mrc
write_mrc(u, fname=niifolder + "_raw.mrc")

ct_den = np.copy(u)
ct_den[ct_den>0] = 1
ct_den[ct_den<=0] = 0
write_mrc(ct_den, fname=niifolder + "_den.mrc")


# Plot and save its curvature diagram
# plot_curvature_diagram(u, save = True, save_name = diagsfolder + name + ' kap1_kap2.png',delta_x=delta_x)
print("Ellapsed time = ", time.time() - start_time)