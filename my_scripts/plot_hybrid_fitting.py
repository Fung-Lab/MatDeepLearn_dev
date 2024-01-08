import torch
import numpy as np
import matplotlib.pyplot as plt


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

    X = [np.arange(0.4, 2., 0.001) for _ in range(100)]
    Y = []
    for i in range(100):
        r = X[i]
        Y.append(4 * epsilons[i] * (coef_12[i] * (sigmas[i] / r) ** 12 - coef_6[i] * (sigmas[i] / r) ** 6) + base_atomic_energy[i])
    
    fig, axs = plt.subplots(10, 10, figsize=(20,20))

    for i in range(100):
        x = X[i] 
        y = Y[i]
        
        row = i // 10 
        col = i % 10
        
        axs[row, col].plot(x, y)
        if np.min(y) < -10:
            axs[row, col].set_ylim(top=5, bottom=-15)
        else:
            axs[row, col].set_ylim(top=10, bottom=-5)
        axs[row, col].set_title(f"Atomic number = {i+1}")

    plt.tight_layout()
    fig.savefig('./plots.png')
    plt.close(fig)
    
def show_all_morse_params(state_dict):
    a, re, epsilons, base_atomic_energy, coef_const, coef_exp = [], [], [], [], [1.] * 100, [1.] * 100
    for i in range(100):
        a.append(state_dict['atomic_a.' + str(i)].detach().cpu().numpy()[0][0])
        epsilons.append(state_dict['atomic_epsilons.' + str(i)].detach().cpu().numpy()[0][0])
        re.append(state_dict['atomic_re.' + str(i)].detach().cpu().numpy()[0][0])
        base_atomic_energy.append(state_dict['base_atomic_energy.' + str(i)].detach().cpu().numpy()[0][0])
        
    if 'coef_12.0' in state_dict.keys():
        for i in range(100):
            coef_const.append(state_dict['coef_const.' + str(i)].detach().cpu().numpy()[0][0])
            coef_exp.append(state_dict['coef_exp.' + str(i)].detach().cpu().numpy()[0][0])
    
    return a, re, epsilons, base_atomic_energy, coef_const, coef_exp


if __name__ == '__main__':
    checkpoint_path = 'results/late_cgcnn_lj/checkpoint/best_checkpoint.pt'
    state_dict = torch.load(checkpoint_path)['state_dict']
    show_all_lj_params(state_dict)