# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
from torch import optim
import pywph as pw

"""
    Does not seem to converge.... TO DEBUG.
"""

#######
# INPUT PARAMETERS
#######

M, N = 128, 128
J = 5
L = 8
dn = 2

norm = "auto"   # Normalization
pbc = True      # Periodic boundary conditions

device = 0

optim_maxiter = 50
optim_lr = 0.04

data = np.load('data/Q_1.npy') + 1j*np.load('data/U_1.npy')
data = data[::4, ::4]

output_filename = "synthesis.npy"

#######
# PREPARATION AND INITIAL GUESS
#######

# Normalize input data
data_std = data.std()
data /= data_std

cplx = np.iscomplexobj(data)

# Initial guess
if cplx:
    x0 = np.random.normal(data.real.mean(), data.real.std(), data.shape) \
        + 1j*np.random.normal(data.imag.mean(), data.imag.std(), data.shape)
    x0 = x0.astype(np.complex64)
else:
    x0 = np.random.normal(data.mean(), data.std(), data.shape).astype(np.float32)
x0 = torch.from_numpy(x0).to(device)
x0.requires_grad_(True)

print("Building operator...")
start_time = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device, cplx=True)
print(f"Done! (in {time.time() - start_time}s)")

print("Computing stats of target image...")
start_time = time.time()
coeffs = wph_op.apply(data, norm=norm, padding=not pbc)
print(f"Done! (in {time.time() - start_time}s)")

#######
# SYNTHESIS
#######

# Optimizer
optimizer = optim.Adam([x0], lr=optim_lr)

total_start_time = time.time()

# Optimization
best_img = None
best_loss = float("inf")
for it in range(optim_maxiter):
    print(f"Iteration: {it}")
    start_time = time.time()
    optimizer.zero_grad()
    
    # Compute the loss (squared 2-norm)
    loss_tot = torch.zeros(1)
    x0, nb_chunks = wph_op.preconfigure(x0)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x0, i, norm=norm, padding=not pbc, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    optimizer.step()
    loss_tot = loss_tot.detach().cpu().item()
    print(f"Loss: {loss_tot} (computed in {time.time() - start_time}s)")
    if loss_tot < best_loss:
        best_loss = loss_tot
        best_img = x0.detach().cpu().numpy()

print(f"Synthesis ended in {optim_maxiter} iterations.")
print(f"Synthesis time: {time.time() - total_start_time}s")

x_final = best_img * data_std

if output_filename is not None:
    np.save(output_filename, x_final)
