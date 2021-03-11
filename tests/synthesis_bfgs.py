# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import time
import torch
from torch import optim
import scipy.optimize as opt
import torch.nn.functional as F
import os
import pywph as pw
import sys

sys.path.append("/home/bruno/Bureau/These ENS/Outils/misc/")

import misc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

M, N = 512, 512
J = 8
L = 8
dn = 0
cplx = True
norm = "auto"

optim_params = {"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

data = fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32)
data = data[:M, :N] + 1j*data[:M, :N]
data /= data.std()

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
wph_op.to(0)
print(time.time() - start)

# Method 1 bis
start = time.time()
data_torch, nb_chunks = wph_op.preconfigure(data)
coeffs = []
for i in range(nb_chunks):
    coeffs.append(wph_op.apply(data_torch, i, norm=norm))
coeffs = torch.cat(coeffs, -1)
ell_time = time.time() - start
print(ell_time)

# Initial guess
x0 = np.zeros((M, N, 2), dtype=np.float64)
x0[:, :, 0] = np.random.normal(data.real.mean(), data.real.std(), data.shape)
x0[:, :, 1] = np.random.normal(data.imag.mean(), data.imag.std(), data.shape)
# x0 = torch.from_numpy(x0.astype(np.complex64)).to(0)
# x0.requires_grad_(True)

time0 = time.time()

def closure(x=None):
    x_reshaped = x.reshape((M, N, 2)).astype(np.float32)
    x_curr = torch.from_numpy(x_reshaped[:, :, 0] + 1j*x_reshaped[:, :, 1])
    global time0
    loss_tot = torch.zeros(1)
    x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print(x.dtype, x_curr.grad, x_curr.grad.shape)
    x_grad = np.zeros_like(x.reshape((M, N, 2)))
    x_grad[:, :, 0] = x_curr.grad.real.cpu().numpy()
    x_grad[:, :, 1] = x_curr.grad.imag.cpu().numpy()
    print(loss_tot.item())
    return loss_tot.item(), x_grad.ravel()


result = opt.minimize(closure, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
final_loss, x0, niter, msg = result['fun'], result['x'], result['nit'], result['message']

best_img = x0.reshape((M, N, 2))[:, :, 0] + 1j*x0.reshape((M, N, 2))[:, :, 1]

fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, np.sqrt(2) * np.pi, 100)
bins, ps_iso, ps_iso_std = misc.power_spectrum_iso(data, bins)
misc.plot_ps_iso(ax, bins, ps_iso, ps_iso_std, ignore_lastbin=True)
bins, ps_iso, ps_iso_std = misc.power_spectrum_iso(best_img, bins)
misc.plot_ps_iso(ax, bins, ps_iso, ps_iso_std, ignore_lastbin=True)
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
misc.plot_img(fig, axs[0, 0], data.real, 'Original Q')
misc.plot_img(fig, axs[0, 1], data.imag, 'Original U')
misc.plot_img(fig, axs[1, 0], best_img.real, 'Synth Q')
misc.plot_img(fig, axs[1, 1], best_img.imag, 'Synth U')
plt.show()