import os
import sys
import mrcfile
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import string
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
def formatted(f): return format(f, '.3f').rstrip('0').rstrip('.')

import torch
dtype = torch.cuda.FloatTensor # we work with GPUs # if torch.cuda.is_available() else torch.FloatTensor

# curvatubes imports
from cvtub.utils import slices, single, load_nii, save_nii, random_init, init_balls
from cvtub.energy import discrepancy, ratio_discr

# Main function that generates 3D shapes with curvatubes

# optimizer: Adam
# flow type: conservative H^{-1}
# periodic boundary conditions

from cvtub.generator import _generate_shape

def generate(A0, params, M0, delta_x, maxeval=10000, snapshot_folder='', exp_title='',
             display_all=True, check_viable=False, cond_take_snapshot=None):
    '''Optimizes the phase-field Feps(u) ( see paper / see comments in cvtub/energy.py ) '''

    xi = 1e-6
    flow_type = 'cons'
    mode = 'periodic'
    optim_method = 'adam'
    sigma_blur = 2
    Z, X, Y = A0[0].shape

    optim_props = {'maxeval': maxeval, 'sigma_blur': sigma_blur, 'lr': .001, 'eps_adam': 1e-2,
                   'betas': (0.9, 0.999), 'weight_decay': 0, 'amsgrad': False,
                   'display_it_nb': 1000, 'fill_curve_nb': 50}

    u = _generate_shape(A0, params, delta_x, xi, optim_method, optim_props, flow_type, mode,
                        M0=M0, snapshot_folder=snapshot_folder, exp_title=exp_title,
                        cond_take_snapshot=cond_take_snapshot, display_all=display_all,
                        check_viable=check_viable)

    if check_viable == True:
        u, viable_bool = u
        return u.detach().cpu().numpy(), viable_bool

    return u.detach().cpu().numpy()

# Function which plots the curvature diagram of a shape
# defined as the zero level set of a phase-field u

from cvtub.curvdiags import kap_eps, curvhist, density_scatter

def plot_curvature_diagram(u, save = True, save_name = 'curvature_diagram.png', delta_x=0.01):

    kap1_eps, kap2_eps = kap_eps(u, delta_x=delta_x) # MODIFICADO PARA INCLUIR EL DELTA_X

    kap1_vals, kap2_vals, areas = \
    curvhist(u, kap1_eps, kap2_eps, delta_x=delta_x, show_figs = False)

    print("Curvaturas_maxima", max(abs(kap1_vals)), " ", max(abs(kap2_vals))) # MODIFICADO

    """
    x,y = np.clip(kap1_vals, -100,100), np.clip(kap2_vals, -100, 100)
    density_scatter(x,y, areas, showid = True, equalaxis = True,
                    bins = 100, xlabel = 'kap1',
                    ylabel = 'kap2', save = save, save_name = save_name)
    """

niifolder = '../results/Experiment_2/'
curvesfolder = niifolder + 'Curves/'
diagsfolder = niifolder + 'Diagrams/'

def write_mrc(tomo, fname, v_size=1, dtype=None, no_saxes=True):
    """
    Saves a tomo (3D dataset) as MRC file

    :param tomo: tomo to save as ndarray
    :param fname: output file path
    :param v_size: voxel size (default 1)
    :param dtype: data type (default None, then the dtype of tomo is considered)
    :param no_saxes: if True (default) then X and Y axes are swaped to cancel the swaping made by mrcfile package
    :return:
    """
    with mrcfile.new(fname, overwrite=True) as mrc:
        if dtype is None:
            if no_saxes:
                mrc.set_data(np.swapaxes(tomo, 0, 2))
            else:
                mrc.set_data(tomo)
        else:
            if no_saxes:
                mrc.set_data(np.swapaxes(tomo, 0, 2).astype(dtype))
            else:
                mrc.set_data(tomo.astype(dtype))
        mrc.voxel_size.flags.writeable = True
        mrc.voxel_size = (v_size, v_size, v_size)
        mrc.set_volume()
        # mrc.header.ispg = 401
