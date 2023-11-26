import torch


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
    a, re, epsilons, base_atomic_energy = [], [], [], []
    for i in range(100):
        a.append(state_dict['atomic_a.' + str(i)].detach().cpu().numpy()[0][0])
        epsilons.append(state_dict['atomic_epsilons.' + str(i)].detach().cpu().numpy()[0][0])
        re.append(state_dict['atomic_re.' + str(i)].detach().cpu().numpy()[0][0])
        base_atomic_energy.append(state_dict['base_atomic_energy.' + str(i)].detach().cpu().numpy()[0][0])
    
    print('a:', a)
    print('epsilons:', epsilons)
    print('re:', re)
    print('base_atomic_energy:', base_atomic_energy)
    
    

if __name__ == '__main__':
    checkpoint_path = 'results/morse_mpforce_subset/checkpoint/best_checkpoint.pt'
    state_dict = torch.load(checkpoint_path)['state_dict']
    show_all_morse_params(state_dict)