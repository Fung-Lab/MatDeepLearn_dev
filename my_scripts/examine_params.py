import torch
import numpy as np


def show_all_lj_params(state_dict):
    sigmas, epsilons, base_atomic_energy,\
        coef_12, coef_6, coef_exp_12, coef_exp_6 = [], [], [], [1.] * 100, [1.] * 100, [12.] * 100, [6.] * 100
    for i in range(100):
        sigmas.append(state_dict['sigmas.' + str(i)].detach().cpu().numpy()[0][0])
        epsilons.append(state_dict['epsilons.' + str(i)].detach().cpu().numpy()[0][0])
        base_atomic_energy.append(state_dict['base_atomic_energy.' + str(i)].detach().cpu().numpy()[0][0])
      
    if 'coef_12.0' in state_dict.keys():
        for i in range(100):
            coef_12[i] = state_dict['coef_12.' + str(i)].detach().cpu().numpy()[0][0]
            coef_6[i] = state_dict['coef_6.' + str(i)].detach().cpu().numpy()[0][0]
    
    if 'coef_exp_12.0' in state_dict.keys():
        for i in range(100):
            coef_exp_12[i] = state_dict['coef_exp_12.' + str(i)].detach().cpu().numpy()[0][0]
            coef_exp_6[i] = state_dict['coef_exp_6.' + str(i)].detach().cpu().numpy()[0][0]
    
    print('sigmas:', sigmas[7], sigmas[13], (sigmas[7] + sigmas[13]) / 2)
    print('epsilons:', epsilons[7], epsilons[13], (epsilons[7] + epsilons[13]) / 2)
    print('base_atomic_energy:', base_atomic_energy[7], base_atomic_energy[13], (base_atomic_energy[7] + base_atomic_energy[13]) / 2)
    print('coef_12:', coef_12[7], coef_12[13], (coef_12[7] + coef_12[13]) / 2)
    print('coef_6:', coef_6[7], coef_6[13], (coef_6[7] + coef_6[13]) / 2)
    print('coef_exp_12:', coef_exp_12[7], coef_exp_12[13], (coef_exp_12[7] + coef_exp_12[13]) / 2)
    print('coef_exp_6:', coef_exp_6[7], coef_exp_6[13], (coef_exp_6[7] + coef_exp_6[13]) / 2)
    
def show_all_morse_params(state_dict):
    rm, D, sigmas, base_atomic_energy, coef_e, coef_2e = [], [], [], [], [], []
    for i in range(100):
        rm.append(state_dict['rm.' + str(i)].detach().cpu().numpy()[0][0])
        sigmas.append(state_dict['sigmas.' + str(i)].detach().cpu().numpy()[0][0])
        D.append(state_dict['D.' + str(i)].detach().cpu().numpy()[0][0])
        base_atomic_energy.append(state_dict['base_atomic_energy.' + str(i)].detach().cpu().numpy()[0][0])
        # coef_e.append(state_dict['coef_e.' + str(i)].detach().cpu().numpy()[0][0])
        # coef_2e.append(state_dict['coef_2e.' + str(i)].detach().cpu().numpy()[0][0])
    
    print('rm:', rm[7], rm[13], (rm[7] + rm[13]) / 2)
    print('sigmas:', sigmas[7], sigmas[13], (sigmas[7] + sigmas[13]) / 2)
    print('D:', D[7], D[13], (D[7] + D[13]) / 2)
    print('base_atomic_energy:', base_atomic_energy[7], base_atomic_energy[13], (base_atomic_energy[7] + base_atomic_energy[13]) / 2)
    # print('coef_e:', coef_e)
    # print('coef_2e:', coef_2e)


if __name__ == '__main__':
    # checkpoint_path = 'results/silica_tests/2024-01-23-17-34-49-899-torchmd_morse_sio2/checkpoint_0/best_checkpoint.pt'
    checkpoint_path = 'results/2024-02-02-10-02-00-998-CGCNN_morse_sio2/checkpoint_0/checkpoint.pt'
    state_dict = torch.load(checkpoint_path)['state_dict']
    print(torch.load(checkpoint_path)['epoch'])
    show_all_lj_params(state_dict)