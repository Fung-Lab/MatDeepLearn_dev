from math import prod

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def cutoff_function(r, rc, ro):
    s = 1.0 - (r - rc) / (ro - rc)
    return (s >= 1.0) + (((0.0 < s) & (s < 1.0)) *
                        (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3))

def load_params(checkpoint_path):
    state_dict = torch.load(checkpoint_path)['state_dict']
    print(state_dict)
    rm, D, sigmas = [], [], []
    for i in range(100):
        rm.append((state_dict['rm_sig_d_o_o.' + str(i)].detach().cpu().numpy()[0][0],
                   state_dict['rm_sig_d_si_si.' + str(i)].detach().cpu().numpy()[0][0],
                   state_dict['rm_sig_d_si_o.' + str(i)].detach().cpu().numpy()[0][0]))
        sigmas.append((state_dict['rm_sig_d_o_o.' + str(i)].detach().cpu().numpy()[1][0],
                   state_dict['rm_sig_d_si_si.' + str(i)].detach().cpu().numpy()[1][0],
                   state_dict['rm_sig_d_si_o.' + str(i)].detach().cpu().numpy()[1][0]))
        D.append((state_dict['rm_sig_d_o_o.' + str(i)].detach().cpu().numpy()[2][0],
                   state_dict['rm_sig_d_si_si.' + str(i)].detach().cpu().numpy()[2][0],
                   state_dict['rm_sig_d_si_o.' + str(i)].detach().cpu().numpy()[2][0]))
    return np.array(rm), np.array(D), np.array(sigmas)
    

if __name__ == '__main__':
    path = 'results/2024-04-19-20-54-14-374-pairwise_morse_sio2/checkpoint_0/best_checkpoint.pt'
    rm, D, sigmas = load_params(path)
    print(rm, D, sigmas)
    # rm, D, sigmas = rm[77], D[77], sigmas[77]
    # print(rm, D, sigmas)
    # #(rm[7] + rm[13]) / 2, (D[7] + D[13]) / 2, (sigmas[7] + sigmas[13]) / 2
    
    # rc = 8.
    # ro = 0.66 * rc
    # X = torch.linspace(0, 8., 500, device='cuda:0')
    # fc = cutoff_function(X, ro, rc)
    
    # y = D * (1 - torch.exp(-sigmas * (X - rm))) ** 2 - D
    
    # y = y.detach().cpu().numpy()
    # min_y = np.min(y)
    # min_x = X[np.where(y == min_y)]
    
    # plt.plot(X.detach().cpu().numpy(), y)
    # # plt.ylim(top=5, bottom=min_y-0.5)
    # plt.ylim(top=0.01, bottom=-0.05)
    # plt.title(f'Si: original Morse, min at ({min_x[0]:.3f}, {min_y:.3f})')
    # plt.savefig('morse.png')

