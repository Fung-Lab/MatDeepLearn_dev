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

def vec_find_intervals(t: torch.tensor, x: torch.tensor, k: int):
    n = t.shape[0] - k - 1
    tb = t[k]
    te = t[n]

    result = torch.zeros_like(x, dtype=int, device='cuda:0')

    nan_indices = torch.isnan(x)
    out_of_bounds_indices = torch.logical_or(x < tb, x > te)
    result[out_of_bounds_indices | nan_indices] = -1

    l = torch.full_like(x, k, dtype=int)
    while torch.any(torch.logical_and(x < t[l], l != k)):
        l = torch.where(torch.logical_and(x < t[l], l != k), l - 1, l)

    l = torch.where(l != n, l + 1, l)

    while torch.any(torch.logical_and(x >= t[l], l != n)):
        l = torch.where(torch.logical_and(x >= t[l], l != n), l + 1, l)

    result = torch.where(result != -1, l - 1, result)
    return result

def evaluate_spline(t, c, k, xp, out):
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    # if out.shape[1] != c.shape[1]:
    #     raise ValueError("out and c have incompatible shapes")

    # work: (N, 2k + 2)
    work = torch.empty(xp.shape[0], 2 * k + 2, dtype=torch.float, device='cuda:0')

    # intervals: (N, )
    intervals = vec_find_intervals(t, xp, k)
    invalid_mask = intervals < 0
    out[invalid_mask, :] = np.nan
    out[~invalid_mask, :] = 0.
    
    if invalid_mask.all():
        return out
    
    work[~invalid_mask] = vectorized_deboor(t, xp[~invalid_mask], k, intervals[~invalid_mask], work[~invalid_mask])
    # print(work[~invalid_mask].shape, c[intervals[~invalid_mask][:, None] + torch.arange(-k, 1, device='cuda:0')].shape)
    
    print(c.shape)
    # c = c[:, intervals[~invalid_mask][:, None] + torch.arange(-k, 1, device='cuda:0')].squeeze(dim=2)
    indices = intervals[:, None] + torch.arange(-k, 1, device='cuda:0')

    # Index into c using advanced indexing
    c = c[torch.arange(c.size(0))[:, None], indices]
    print(work[~invalid_mask, :k+1].shape, c.shape)
    out[~invalid_mask, :] = torch.sum(work[~invalid_mask, :k+1] * c[~invalid_mask, :], dim=1).unsqueeze(dim=-1)
    return out

def b_spline(k, t, c, x):
    out = torch.empty((len(x), 1), dtype=c.dtype, device='cuda:0')
    res = torch.nan_to_num(evaluate_spline(t, c.reshape(c.shape[0], -1), k, x, out), nan=0)
    return res

def load_params(checkpoint_path):
    state_dict = torch.load(checkpoint_path)['state_dict']
    coefs_atomic = []
    for param in state_dict.keys():
        if param.split('.')[0] == 'coefs':
            coefs_atomic.append(state_dict[param])
    return coefs_atomic
    

if __name__ == '__main__':
    path = 'results/2024-02-28-09-15-16-588-spline_new_sio2/checkpoint_0/best_checkpoint.pt'
    coefs = load_params(path)
    coef = (coefs[7] + coefs[13]) / 2
    # coef = coefs[7]
    print(coef)
    
    # coef = torch.tensor([0.97, -0.25, 0.42,  0], device='cuda:0')
    
    subinterval_size = 5
    cutoff_radius = 8.
    n_intervals = 25
    degree = 3
    n_lead = 1
    n_trail = 1
    
    init_knots = torch.linspace(0, cutoff_radius, n_intervals + 1)
    init_knots = torch.cat([torch.tensor([init_knots[0] - 1 for _ in range(3, 0, -1)]),
                            init_knots,
                            torch.tensor([init_knots[-1] + 1 for _ in range(1, 4, 1)])])
    
    t = init_knots[n_lead:-n_trail].to('cuda:0')
    print(t)
    x = torch.arange(0, cutoff_radius, 0.01, device='cuda')
    coef_stacked = torch.stack([coef for _ in range(len(x))])
    spline_res = b_spline(degree, t, coef_stacked, x)
    
    X = x.detach().cpu().numpy()
    y = spline_res.detach().cpu().numpy()
    min_y = np.min(y)
    max_y = np.max(y)
    
    title = f'Si-O: min at ({x[np.where(y==min_y)[0]].item():.3f}, {min_y:.3f})'
    print(title)
    plt.ylim(top=max_y+0.5, bottom=min_y-0.5)
    plt.plot(X, y)
    plt.title(title)
    plt.savefig('my_spline_plot.png')
    
    print("Finished my plot")

    spl = interpolate.BSpline(t.cpu().numpy(), coef.cpu().numpy(), degree)
    
    res_scipy = spl(X)
    
    print(np.allclose(res_scipy, spline_res.cpu().numpy()))
    
    plt.cla()
    plt.ylim(top=max_y+0.5, bottom=min_y-0.5)
    plt.plot(X, res_scipy)
    plt.title(title)
    plt.savefig('spline_plot.png')
    