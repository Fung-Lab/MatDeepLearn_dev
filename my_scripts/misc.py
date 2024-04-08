import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file_path = 'results/2024-04-03-12-37-31-452-morse_sio2/checkpoint_0/best_checkpoint.pt'

    checkpoint = torch.load(file_path)
    state_dict = checkpoint['state_dict']
    rm = state_dict['rm.weight'] 
    sigma = state_dict['sigmas.weight'] 
    D = state_dict['D.weight'] 
    
    rm = (rm[7] + rm[13]) / 2
    sigma = (sigma[7] + sigma[13]) / 2
    D = (D[7] + D[13]) / 2
    
    # rm = torch.tensor(0.5)
    # sigma = torch.tensor(0.5)
    # D = torch.tensor(1)

    # Define Morse potential function
    def morse_potential(r, D, a, re):
        return D * (1 - torch.exp(-a * (r - re))) ** 2 - D

    # Generate r values
    r = torch.linspace(0, 5, 100).to('cuda:0')

    # Calculate potential values
    V = morse_potential(r, D, sigma, rm).cpu().numpy()
    
    min_y, min_x = np.min(V), r[np.argmin(V)].item()

    # Create the plot
    plt.figure()
    plt.plot(r.cpu().numpy(), V)
    plt.xlim(left=0, right=4)
    plt.ylim(top=5, bottom=-5)
    plt.title(f'Morse Potential Curve, min at ({min_x:.2f}, {min_y:.2f})')
    plt.savefig('morse_potential_curve.png')
