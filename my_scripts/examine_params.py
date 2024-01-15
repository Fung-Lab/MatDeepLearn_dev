import torch
import numpy as np


def show_all_lj_params(state_dict):
    sigmas, epsilons, base_atomic_energy, coef_12, coef_6 = [], [], [], [1.] * 100, [1.] * 100
    for i in range(100):
        sigmas.append(state_dict['sigmas.' + str(i)].detach().cpu().numpy()[0][0])
        epsilons.append(state_dict['epsilons.' + str(i)].detach().cpu().numpy()[0][0])
        base_atomic_energy.append(state_dict['base_atomic_energy.' + str(i)].detach().cpu().numpy()[0][0])
      
    if 'coef_12.0' in state_dict.keys():
        for i in range(100):
            coef_12[i] = state_dict['coef_12.' + str(i)].detach().cpu().numpy()[0][0]
            coef_6[i] = state_dict['coef_6.' + str(i)].detach().cpu().numpy()[0][0]
    
    print('sigmas:', sigmas)
    print('epsilons:', epsilons)
    print('base_atomic_energy:', base_atomic_energy)
    print('coef_12:', coef_12)
    print('coef_6:', coef_6)
    
def show_all_morse_params(state_dict):
    rm, D, sigmas, base_atomic_energy, coef_e, coef_2e = [], [], [], [], [], []
    for i in range(100):
        rm.append(state_dict['rm.' + str(i)].detach().cpu().numpy()[0][0])
        sigmas.append(state_dict['sigmas.' + str(i)].detach().cpu().numpy()[0][0])
        D.append(state_dict['D.' + str(i)].detach().cpu().numpy()[0][0])
        base_atomic_energy.append(state_dict['base_atomic_energy.' + str(i)].detach().cpu().numpy()[0][0])
        coef_e.append(state_dict['coef_e.' + str(i)].detach().cpu().numpy()[0][0])
        coef_2e.append(state_dict['coef_2e.' + str(i)].detach().cpu().numpy()[0][0])
    
    
    for i in range(100):
        for j in range(100):
            res1 = 2 * D[i] * D[j] / (D[i] + D[j])
            res2 = sigmas[i] * sigmas[j] * (sigmas[i] + sigmas[j]) / (sigmas[i] ** 2 + sigmas[j] ** 2)
            if abs(res1 - res2) < 1e-9:
                print(True)
    
    # print('a:', rm)
    # print('sigmas:', sigmas)
    # print('D:', D)
    # print('base_atomic_energy:', base_atomic_energy)
    # print('coef_e:', coef_e)
    # print('coef_2e:', coef_2e)
    
    

if __name__ == '__main__':
    checkpoint_path = 'results/2024-01-12-07-27-21-120-cgcnn_morse_bcomb_1_1/checkpoint_0/best_checkpoint.pt'
    state_dict = torch.load(checkpoint_path)['state_dict']
    show_all_morse_params(state_dict)