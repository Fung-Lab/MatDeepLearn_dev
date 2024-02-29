from math import prod

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def cutoff_function(r, rc, ro):
    """
    
    Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """

    return torch.where(
        r < ro,
        1.0,
        torch.where(r < rc, (rc - r) ** 2 * (rc + 2 *
                r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )

def load_params(checkpoint_path):
    state_dict = torch.load(checkpoint_path)['state_dict']
    sigmas, epsilons = [], []
    for i in range(100):
        sigmas.append(state_dict['sigmas.' + str(i)].detach().cpu().numpy()[0][0])
        epsilons.append(state_dict['epsilons.' + str(i)].detach().cpu().numpy()[0][0])
    return sigmas, epsilons
    

if __name__ == '__main__':
    path = 'results/2024-02-23-07-27-46-689-lj_pt_forces/checkpoint_0/checkpoint.pt'
    sigmas, epsilons = load_params(path)
    sigmas, epsilons = sigmas[77], epsilons[77]
    #(sigmas[7] + sigmas[13]) / 2, (epsilons[7] + epsilons[13]) / 2
    
    combination_method = 'average'
    with_coefs, with_exp_coefs = False, False
    rc = 8.
    ro = 0.66 * rc
    X = torch.linspace(0, 8., 500, device='cuda:0')
    r2 = X ** 2
    cutoff_fn = cutoff_function(r2, rc**2, ro**2)
    
    c6 = (sigmas ** 2 / r2) ** 3
    c6[r2 > rc ** 2] = 0.0
    c12 = c6 ** 2

    y = 4 * epsilons * (c12 - c6)
    y = y.detach().cpu().numpy()
    min_y = np.min(y[1:])
    min_x = X[np.where(y == min_y)]
    
    plt.plot(X.detach().cpu().numpy()[1:], y[1:])
    # plt.ylim(top=5, bottom=min_y-0.5)
    plt.ylim(top=1, bottom=-0.25)
    plt.title(f'Pt: original LJ, min at ({min_x[0]:.3f}, {min_y:.3f})')
    plt.savefig('lj.png')

