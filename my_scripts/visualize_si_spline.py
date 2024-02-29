from math import prod

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def vectorized_deboor(t, x, k, interval, result):
    
    hh = result[:, k+1:]
    h = result
    h[:, 0] = 1.
        
    for j in range(1, k + 1):
        hh[:, :j] = h[:, :j]
        h[:, 0] = 0
        for n in range(1, j + 1):
            ind = interval + n
            xb = t[ind]
            xa = t[ind - j]
            zero_mask = xa == xb
            h[zero_mask, n] = 1.
            w = hh[:, n - 1] / (xb - xa)
            h[:, n - 1] += w * (xb - x)
            h[:, n] = w * (x - xa)
            h[zero_mask, n] = 0.
        
    return result

def vec_find_intervals(knots: torch.tensor, x: torch.tensor, k: int):
    n = knots.shape[0] - k - 1

    pos_idxs = torch.empty(len(x), device='cuda:0')
    invalid_mask = (x <= knots[k]) | (x > knots[n])
    pos_idxs[invalid_mask] = -1

    x_masked = x[~invalid_mask]
    knots = knots[k:n+1]

    t = knots - x_masked.view(-1, 1)
    pos_idxs[~invalid_mask] = torch.argmax((t >= 0).int(), dim=1).to(torch.float) + k - 1
    pos_idxs[x == knots[0]] = k
    return pos_idxs.to(torch.long)

def evaluate_spline(t, c, k, xp, out):
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[1]:
        raise ValueError("out and c have incompatible shapes")

    work = torch.empty(xp.shape[0], 2 * k + 2, dtype=torch.float, device='cuda:0')

    intervals = vec_find_intervals(t, xp, k)
    invalid_mask = intervals < 0
    
    out[invalid_mask, :] = np.nan
    out[~invalid_mask, :] = 0.
    
    if invalid_mask.all():
        return out
    
    work[~invalid_mask] = vectorized_deboor(t, xp[~invalid_mask], k, intervals[~invalid_mask], work[~invalid_mask])
    
    c = torch.stack([c[i-k:i+1] for i in intervals[~invalid_mask]]).squeeze(dim=2)

    out[~invalid_mask, :] = torch.sum(work[~invalid_mask, :k+1] * c, dim=1).unsqueeze(dim=-1)

    return out

    
def compute_b_spline(x, subintervals, coef):
    out_stack = None
    for i in range(len(subintervals)):
        t = subintervals[i]
        k = len(t) - 2
        t = torch.cat([torch.tensor((t[0]-1,) * k), t, torch.tensor((t[-1]+1,) * k)]).to('cuda:0')
        c = torch.zeros_like(t, device='cuda:0')
        c[k] = 1.
        
        out = torch.empty((len(x), prod(c.shape[1:])), dtype=c.dtype, device='cuda:0')
        res = torch.nan_to_num(evaluate_spline(t, c.reshape(c.shape[0], -1), k, x, out), nan=0)
        
        out_stack = res.view(1, -1) if out_stack is None else torch.concatenate([out_stack, res.view(1, -1)])
    res = coef @ out_stack
    return res

def load_params(checkpoint_path, coef_name):
    state_dict = torch.load(checkpoint_path)['state_dict']
    # print(state_dict.keys())
    coefs_atomic = []
    for param in state_dict.keys():
        if param.split('.')[0] == coef_name:
            coefs_atomic.append(state_dict[param])
    return coefs_atomic
    

if __name__ == '__main__':
    path = 'results/2024-02-26-13-04-49-028-spline_sio2_cutoff_2.0/checkpoint_0/best_checkpoint.pt'
    coefs = load_params(path, 'coefs_si_o')
    # print(coefs)
    coef = coefs[0]# + coefs[13]) / 2
    # coef = coefs[77]
    print(coef)
    
    # coef = torch.tensor([0.97, -0.25, 0.42,  0], device='cuda:0')
    
    subinterval_size = 5
    cutoff_radius = 2.
    n_intervals = 25
    n_lead = 1
    n_trail = 1
    title = 'Si: Si-O'
    
    init_knots = torch.linspace(0, cutoff_radius, n_intervals + 1)
    init_knots = torch.cat([torch.tensor([init_knots[0] for _ in range(3, 0, -1)]),
                            init_knots,
                            torch.tensor([init_knots[-1] for _ in range(1, 4, 1)])])
    subintervals = torch.stack([init_knots[i:i+subinterval_size] for i in range(len(init_knots)-subinterval_size+1)])
    
    x = torch.arange(0, cutoff_radius, 0.01, device='cuda')
    spline_res = compute_b_spline(x, subintervals[n_lead:-n_trail], coef)
    
    X = x.detach().cpu().numpy()
    y = spline_res.detach().cpu().numpy()
    min_y = np.min(y)
    max_y = np.max(y)
    
    print(y.shape)
    plt.ylim(top=max_y+0.5, bottom=min_y-0.5)
    plt.plot(X, y)
    plt.title(title)
    plt.savefig('pairwise_plot.png')
    
    # print("Finished my plot")

    splines = []
    for interval in subintervals[n_lead:-n_trail]:
        splines.append(interpolate.BSpline.basis_element(interval, extrapolate=False))

    res_scipy = []
    for sp in splines:
        # print(sp(xp))
        res_scipy.append(np.nan_to_num(sp(X), nan=0))
        # res_scipy.append(sp(xp))

        
    res_scipy = np.vstack(res_scipy)
    res_scipy = (coef.cpu().numpy() @ res_scipy).flatten()
    
    print(np.allclose(res_scipy, spline_res.cpu().numpy()))
    
    plt.cla()
    plt.ylim(top=max_y+0.5, bottom=min_y-0.5)
    plt.plot(X, res_scipy)
    plt.title(title)
    plt.savefig('scipy_pairwise_plot_plot.png')
    