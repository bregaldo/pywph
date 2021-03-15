# -*- coding: utf-8 -*-

import numpy as np
import astropy.io.fits as fits
import time
import torch
import scipy.optimize as opt
import pywph as pw
import sys
import multiprocessing as mp
from functools import partial

M, N = 256, 256
J = 5
L = 8
dn = 0
norm = "auto"

optim_params = {"maxiter": 10, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

devices = [0]

SNR = 1000.0
Mn = 10 # Number of noises per iteration


def closure_per_gpu(x_curr, noises, coeffs, wph_op, work_list, device_id):
    device = devices[device_id]
    wph_op.to(device)
    coeffs_target = coeffs.to(device_id)
    
    u = x_curr.clone().to(device).requires_grad_(True)
    n = noises[work_list[device_id]].to(device)
    
    loss_tot = torch.zeros(1)
    for i in range(n.shape[0]):
        un, nb_chunks = wph_op.preconfigure(u + n[i])
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(un, j, norm=norm, ret_indices=True)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_target[indices]) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    grad_tot = u.grad.cpu().numpy()
    
    del n, u # To free GPU memory
    
    return loss_tot.item(), grad_tot


def closure(x=None):
    global eval_cnt
    print(f"Evaluation : {eval_cnt}")
    start = time.time()
    x_reshaped = x.reshape((M, N)).astype(np.float32)
    x_curr = torch.from_numpy(x_reshaped)
    noises = torch.randn((Mn, M, N)) / SNR
    
    # Multi-gpu configuration
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(closure_per_gpu, x_curr, noises, coeffs, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape((M, N)))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print(f"Loss: {loss_tot} (computed in {time.time() - start}s)")
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()


if __name__ == "__main__":
    s = fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32)
    s = s[:M, :N]
    s /= s.std() # For normalization
    n = np.random.normal(0.0, 1 / SNR, s.shape)
    d = s + n
    
    d_torch = torch.from_numpy(d)
    
    print("Building operators...")
    start = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn)
    wph_op.to(0)
    print(f"Done! (in {time.time() - start}s)")
    
    print("Computing stats of target image...")
    start = time.time()
    coeffs = wph_op.apply(d, norm=norm).to("cpu")
    wph_op.to("cpu")
    print(f"Done! (in {time.time() - start}s)")
    
    eval_cnt = 0

    total_start = time.time()
    
    result = opt.minimize(closure, d.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
    final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    
    print(f"Denoising ended in {niter} iterations with optimizer message: {msg}")
    print(f"Denoising time: {time.time() - total_start}s")
    
    np.save("denoising_output.npy", [d, s, n, s_tilde.reshape((M, N))])
    