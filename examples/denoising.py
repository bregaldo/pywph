# -*- coding: utf-8 -*-

import numpy as np
import astropy.io.fits as fits
import time
import sys
import torch
import scipy.optimize as opt
import pywph as pw
import multiprocessing as mp
from functools import partial


"""
    Denoising example.
    
    We build a simulated noisy map d = s + n and attempt to retrieve s.
    n is here a realization of Gaussian white noise.
    The signal-to-noise ratio SNR = s.std()/n.std() is a parameter.
    
    The denoising algorithm is a simplified version of that of Regaldo-Saint Blancard+2021.
    It supports multi-GPU calculation.
"""

#######
# INPUT PARAMETERS
#######

M, N = 256, 256
J = 7
L = 8
dn = 2

norm = "auto"   # Normalization

devices = [0, 1] # List of GPUs to use

optim_params = {"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

SNR = 1.0
Mn = 100 # Number of noises per iteration

# Truth map s
s = fits.open('data/Q_1.fits')[0].data + 1j*fits.open('data/U_1.fits')[0].data
s = s[::2, ::2]
s_std = s.std()
s /= s_std # Normalization
cplx = np.iscomplexobj(s) # Are we dealing with complex maps?

output_filename = "denoising_output.npy"

#######
# DENOISING
#######


def objective_per_gpu(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    coeffs_target = coeffs_target.to(device_id)
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Build len(n_id) noises
    n_id = work_list[device_id]
    n = (torch.randn((len(n_id), M, N), dtype=torch.complex64 if cplx else torch.float32) / SNR).to(device)
    
    # Compute the loss
    loss_tot = torch.zeros(1)
    for i in range(n.shape[0]):
        u_noisy, nb_chunks = wph_op.preconfigure(u + n[i])
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_target[indices]) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    # Extract the corresponding gradient
    if cplx:
        x_grad = np.zeros((M, N, 2))
        x_grad[:, :, 0] = u.grad.real.cpu().numpy()
        x_grad[:, :, 1] = u.grad.imag.cpu().numpy()
    else:
        x_grad = u.grad.cpu().numpy()
    
    del n, u # To free GPU memory
    
    return loss_tot.item(), x_grad


def objective(x):
    """
        Main objective function.
        Dispatch computations on the devices and gathers the results.
    """
    global eval_cnt
    print(f"Evaluation : {eval_cnt}")
    start_time = time.time()
    
    # Reshape u
    if cplx:
        u = x.reshape((M, N, 2))
        u = u[..., 0] + 1j*u[..., 1]
    else:
        u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu, u, coeffs, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape((M, N) + (2,)*cplx))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print(f"Loss: {loss_tot} (computed in {time.time() - start_time}s)")
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()


if __name__ == "__main__":
    ## Build simulated noisy map
    # Noise map
    if cplx:
        n = np.random.normal(0.0, 1 / SNR, s.shape) + 1j*np.random.normal(0.0, 1 / SNR, s.shape)
    else:
        n = np.random.normal(0.0, 1 / SNR, s.shape)
    # Noisy map
    d = s + n
    
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=devices[0])
    print(f"Done! (in {time.time() - start_time}s)")
    
    print("Computing stats of target image...")
    start_time = time.time()
    coeffs = wph_op.apply(d, norm=norm).to("cpu")
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    print(f"Done! (in {time.time() - start_time}s)")
    
    ## Minimization
    
    eval_cnt = 0

    total_start_time = time.time()
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    result = opt.minimize(objective, d.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
    final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    
    ## Output
    
    # Reshaping
    if cplx:
        s_tilde = s_tilde.reshape((M, N, 2)).astype(np.float32)
        s_tilde = s_tilde[..., 0] + 1j*s_tilde[..., 1]
    else:
        s_tilde = s_tilde.reshape((M, N)).astype(np.float32)
    s_tilde = s_tilde * s_std
    
    print(f"Denoising ended in {niter} iterations with optimizer message: {msg}")
    print(f"Denoising total time: {time.time() - total_start_time}s")
    
    if output_filename is not None:
        np.save(output_filename, [d, s * s_std, n * s_std, s_tilde])
