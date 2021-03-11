# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import time
import torch
from torch import optim
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
x0 = np.random.normal(data.real.mean(), data.real.std(), data.shape) + 1j*np.random.normal(data.imag.mean(), data.imag.std(), data.shape)
x0 = torch.from_numpy(x0.astype(np.complex64)).to(0)
x0.requires_grad_(True)

# Optimizer
optimizer = optim.Adam([x0], lr=0.1)

start_time = time.time()

# Training
best_img = None
best_loss = float("inf")
losses_save = []
img_save = []
for epoch in range(1, 10):
    print(f"Epoch: {epoch}")
    optimizer.zero_grad()
    loss_tot = torch.zeros(1)
    x0, nb_chunks = wph_op.preconfigure(x0)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x0, i, norm=norm, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    print(f"Loss: {loss_tot.numpy()}")
    optimizer.step()
    losses_save.append(loss_tot.detach().cpu().item())
    if losses_save[-1] < best_loss:
        best_loss = loss_tot.item()
        best_img = x0.detach().cpu().numpy()

# fig, ax = plt.subplots(1, 1)
# bins = np.linspace(0, np.sqrt(2) * np.pi, 100)
# bins, ps_iso, ps_iso_std = misc.power_spectrum_iso(data, bins)
# misc.plot_ps_iso(ax, bins, ps_iso, ps_iso_std, ignore_lastbin=True)
# bins, ps_iso, ps_iso_std = misc.power_spectrum_iso(best_img, bins)
# misc.plot_ps_iso(ax, bins, ps_iso, ps_iso_std, ignore_lastbin=True)
# plt.show()

# fig, axs = plt.subplots(2, 2, figsize=(10, 7))
# misc.plot_img(fig, axs[0, 0], data.real, 'Original Q')
# misc.plot_img(fig, axs[0, 1], data.imag, 'Original U')
# misc.plot_img(fig, axs[1, 0], best_img.real, 'Synth Q')
# misc.plot_img(fig, axs[1, 1], best_img.imag, 'Synth U')
# plt.show()