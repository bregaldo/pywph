# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import scipy.optimize as opt
import pywph as pw
from wph_quijote.wph_syntheses.wph_operator_wrapper import WPHOp_quijote
from wph_quijote_legacy_model import wph_quijote_legacy_model

#######
# INPUT PARAMETERS
#######

M, N = 128, 128
J = 5
L = 8
dn = 2

norm = "auto"   # Normalization

device = 0

optim_params = {"maxiter": 10, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

data = np.load('data/Q_1.npy') + 1j*np.load('data/U_1.npy')
data = data[::4, ::4]

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

#######
# SYNTHESIS WITH PYWPH
#######

print("======== With PyWPH ========")

print("Building operator...")
start_time = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=0, cplx=True)
wph_moments_indices, scaling_moments_indices = wph_quijote_legacy_model(J, L, dn=dn)
wph_op.load_model([], extra_wph_moments=wph_moments_indices, extra_scaling_moments=scaling_moments_indices)
print(f"Done! (in {time.time() - start_time}s)")

print("Computing stats of target image...")
start_time = time.time()
coeffs = wph_op.apply(data, norm=norm)
print(f"Done! (in {time.time() - start_time}s)")

eval_cnt = 0


def objective(x):
    global eval_cnt
    print(f"Evaluation : {eval_cnt}")
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

if cplx:
    x_final = x_final.reshape((M, N, 2)).astype(np.float32)
    x_final = x_final[..., 0] + 1j*x_final[..., 1]
else:
    x_final = x_final.reshape((M, N)).astype(np.float32)
x_final_pywph = x_final * data_std
wph_op.to("cpu") # To free GPU memory

#######
# SYNTHESIS WITH WPH_QUIJOTE
#######

print("======== With Tanguy's code ========")

print("Building operator...")
start_time = time.time()
stat_params = {"J": J, "L": L, "delta_j": J-1, "delta_l": 4, "delta_n": dn,
               "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 30}
wph_quijote_op = WPHOp_quijote(M, N, stat_params)
print(f"Done! (in {time.time() - start_time}s)")

# Data to torch tensor
if cplx:
    data_torch = torch.from_numpy(np.stack((data.real, data.imag), axis=-1))
else:
    data_torch = torch.from_numpy(data)
data_torch = data_torch.unsqueeze(0).to(device)

print("Computing stats of target image...")
start_time = time.time()
wph_chunks = []
for chunk_id in range(wph_quijote_op.nb_chunks + 1):
    wph_chunk = wph_quijote_op.stat_op(data_torch, chunk_id, norm=norm)
    wph_chunks.append(wph_chunk)
print(f"Done! (in {time.time() - start_time}s)")

eval_cnt = 0


def objective_function(x):
    global eval_cnt
    print(f"Evaluation : {eval_cnt}")
    start_time = time.time()
    
    # Reshape x and conversion to torch.tensor
    if cplx:
        x_curr = x.reshape((M, N, 2))
    else:
        x_curr = x.reshape((M, N))
    x_curr = torch.from_numpy(x_curr.astype(np.float32)).unsqueeze(0).to(device).requires_grad_(True)
    
    # Compute the loss (squared 2-norm) and the gradient
    loss_tot = torch.zeros(1)
    for chunk_id in range(wph_quijote_op.nb_chunks + 1):
        wph_chunk = wph_quijote_op.stat_op(x_curr, chunk_id, norm=norm)
        loss = torch.sum(torch.abs(wph_chunk - wph_chunks[chunk_id]) ** 2)
        loss.backward()
        loss_tot += loss.detach().cpu()
        del wph_chunk, loss
    x_grad = x_curr.grad.cpu().numpy().astype(np.float64)
    
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()


total_start_time = time.time()

result = opt.minimize(objective_function, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
final_loss, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']

print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
print(f"Synthesis time: {time.time() - total_start_time}s")

if cplx:
    x_final = x_final.reshape((M, N, 2)).astype(np.float32)
    x_final = x_final[..., 0] + 1j*x_final[..., 1]
else:
    x_final = x_final.reshape((M, N)).astype(np.float32)
x_final_quijote = x_final * data_std

#######
# TEST
#######

assert np.allclose(x_final_pywph, x_final_quijote, rtol=1e-4, atol=1e-7)
