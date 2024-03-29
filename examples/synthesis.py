# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import scipy.optimize as opt
import pywph as pw

#######
# INPUT PARAMETERS
#######

M, N = 512, 512
J = 8
L = 4
dn = 2

norm = "auto"   # Normalization
pbc = True      # Periodic boundary conditions

device = 0

optim_params = {"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

data = np.load('data/Q_1.npy') + 1j*np.load('data/U_1.npy')

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
    x0 = np.zeros((M, N, 2), dtype=np.float64)
    x0[:, :, 0] = np.random.normal(data.real.mean(), data.real.std(), data.shape)
    x0[:, :, 1] = np.random.normal(data.imag.mean(), data.imag.std(), data.shape)
else:
    x0 = np.random.normal(data.mean(), data.std(), data.shape)

print("Building operator...")
start_time = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device, cplx=cplx)
print(f"Done! (in {time.time() - start_time}s)")

print("Computing stats of target image...")
start_time = time.time()
coeffs = wph_op.apply(data, norm=norm, pbc=pbc)
print(f"Done! (in {time.time() - start_time}s)")


#######
# SYNTHESIS
#######

eval_cnt = 0


def objective(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    if cplx:
        x_curr = x.reshape((M, N, 2))
        x_curr = x_curr[..., 0] + 1j*x_curr[..., 1]
    else:
        x_curr = x.reshape((M, N))
    
    # Compute the loss (squared 2-norm)
    loss_tot = torch.zeros(1)
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    
    # Reshape the gradient
    if cplx:
        x_grad = np.zeros_like(x).reshape((M, N, 2))
        x_grad[:, :, 0] = x_curr.grad.real.cpu().numpy()
        x_grad[:, :, 1] = x_curr.grad.imag.cpu().numpy()
    else:
        x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()


total_start_time = time.time()

result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
final_loss, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']

print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
print(f"Synthesis time: {time.time() - total_start_time}s")

#######
# OUTPUT
#######

if cplx:
    x_final = x_final.reshape((M, N, 2)).astype(np.float32)
    x_final = x_final[..., 0] + 1j*x_final[..., 1]
else:
    x_final = x_final.reshape((M, N)).astype(np.float32)
x_final = x_final * data_std

if output_filename is not None:
    np.save(output_filename, x_final)
