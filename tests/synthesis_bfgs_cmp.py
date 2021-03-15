# -*- coding: utf-8 -*-

import numpy as np
import astropy.io.fits as fits
import time
import torch
import scipy.optimize as opt
import os
import pywph as pw

M, N = 512, 512
J = 8
L = 8
dn = 2
norm = "auto"

optim_params = {"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

data = fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32)
data = data[:M, :N] + 1j*data[:M, :N]/2
data /= data.std()

# Initial guess
x0 = np.zeros((M, N, 2), dtype=np.float64)
x0[:, :, 0] = np.random.normal(data.real.mean(), data.real.std(), data.shape)
x0[:, :, 1] = np.random.normal(data.imag.mean(), data.imag.std(), data.shape)

print("======== With PyWPH ========")

print("Building operator...")
start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=0)
print(f"Done! (in {time.time() - start}s)")

print("Computing stats of target image...")
start = time.time()
coeffs = wph_op.apply(data)
print(f"Done! (in {time.time() - start}s)")

it = 0


def closure(x=None):
    global it
    print(f"Iteration : {it}")
    start = time.time()
    x_reshaped = x.reshape((M, N, 2)).astype(np.float32)
    x_curr = torch.from_numpy(x_reshaped[:, :, 0] + 1j*x_reshaped[:, :, 1])
    loss_tot = torch.zeros(1)
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    x_grad = np.zeros_like(x.reshape((M, N, 2)))
    x_grad[:, :, 0] = x_curr.grad.real.cpu().numpy()
    x_grad[:, :, 1] = x_curr.grad.imag.cpu().numpy()
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start}s)")
    it += 1
    return loss_tot.item(), x_grad.ravel()

total_start = time.time()

result = opt.minimize(closure, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
final_loss, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']

print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
print(f"Synthesis time: {time.time() - total_start}s")

np.save("synthesis_pywph.npy", x_final)
wph_op.to("cpu") # To free GPU memory

print("======== With Tanguy's code ========")

print("Building operator...")
os.chdir('../pywph/')
start = time.time()
stat_params = {"J": J, "L": L, "delta_j": J-1, "delta_l": 4, "delta_n": dn,
               "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 30}
wph_op_old = pw.WPHOp_old(M, N, stat_params)
os.chdir('../tests/')
print(f"Done! (in {time.time() - start}s)")

data_torch = torch.from_numpy(np.stack((data.real, data.imag), axis=-1)).unsqueeze(0)

print("Computing stats of target image...")
start = time.time()
wph_chunks = []
for chunk_id in range(wph_op_old.nb_chunks + 1):
    wph_chunk = wph_op_old.stat_op(data_torch, chunk_id, norm=norm)
    wph_chunks.append(wph_chunk)
    loss = (torch.abs(wph_chunk) ** 2).mean()
    loss.backward()
print(f"Done! (in {time.time() - start}s)")
wph_op_old.stat_op.cpu()


def closure(x=None):
    global it
    print(f"Iteration : {it}")
    start = time.time()
    x_reshaped = x.reshape((M, N, 2)).astype(np.float32)
    x_curr = torch.from_numpy(x_reshaped).unsqueeze(0).requires_grad_(True)
    loss_tot = torch.zeros(1)
    for chunk_id in range(wph_op_old.nb_chunks + 1):
        wph_chunk = wph_op_old.stat_op(x_curr, chunk_id, norm=norm)
        loss = torch.sum(torch.abs(wph_chunk - wph_chunks[chunk_id]) ** 2)
        loss.backward()
        loss_tot += loss.detach().cpu()
        del wph_chunk, loss
    x_grad = np.zeros_like(x.reshape((M, N, 2)))
    x_grad[:, :, 0] = x_curr.grad.real.cpu().numpy()
    x_grad[:, :, 1] = x_curr.grad.imag.cpu().numpy()
    print(f"Loss: {loss_tot.item()} (computed in {time.time() - start}s)")
    it += 1
    return loss_tot.item(), x_grad.ravel()


total_start = time.time()

result = opt.minimize(closure, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
final_loss, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']

print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
print(f"Synthesis time: {time.time() - total_start}s")

np.save("synthesis_tanguyscode.npy", x_final)
