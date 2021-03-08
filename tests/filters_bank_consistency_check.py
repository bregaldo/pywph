# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pywph as pw
import time

sys.path.append('/home/bruno/Bureau/These ENS/Outils/misc/')

import misc

M, N = 512, 512
J = 8
L = 8
dn = 2
cplx = True

input_dir = "/home/bruno/Bureau/These ENS/Projets/Planck_denoising/Scripts/CompSepAlgo/stats/bump_steerable_wavelet/filters/"
bp_filters = np.load(os.path.join(input_dir, f'bump_steerable_wavelet_N_{N}_J_{J}_L{L}_dn{dn}.npy'), allow_pickle=True).item()['filt_fftpsi'].astype(np.complex_)
lp_filters = np.load(os.path.join(input_dir, f'bump_scaling_functions_N_{N}_J_{J}.npy'), allow_pickle=True).astype(np.complex_)

# Create a similar dictionary for psi filters to WPHop.psi
psi = {}
for j in range(J):
    for theta in range(L * (cplx + 1)):
        for n in range(dn + 1):
            if n == 0:
                psi[j, theta, n, 0] = bp_filters[theta, j, 0]
            else:
                psi[j, theta, n, 0] = bp_filters[theta, j, 1 + (n - 1)*4 + 1]
                psi[j, theta, n, 1] = bp_filters[theta, j, 1 + (n - 1)*4 + 0]
                psi[j, theta, n, 2] = bp_filters[theta, j, 1 + (n - 1)*4 + 2]
                psi[j, theta, n, 3] = bp_filters[theta, j, 1 + (n - 1)*4 + 3]

# Same for phi:
phi = {}
for j in range(2, J - 1):
    phi[j] = lp_filters[j - 2]

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
print(time.time() - start)

# Psi
max_diff = 0.0
for key in psi.keys():
    diff = wph_op.psi[key] - psi[key]
    max_val = np.absolute(diff).max()
    if max_val > max_diff:
        print(key)
        max_diff = max_val
print(max_diff)

# Phi
max_diff = 0.0
for key in phi.keys():
    diff = wph_op.phi[key] - phi[key]
    max_val = np.absolute(diff).max()
    if max_val > max_diff:
        print(key)
        max_diff = max_val
print(max_diff)

# key=(0, 8, 0, 0)

# diff = wph_op.psi[key] - psi[key]

# def plot(filt, title=""):
#     vmin = min(filt.real.min(), filt.imag.min())
#     vmax = max(filt.real.max(), filt.imag.max())
    
#     fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
#     misc.plot_img(fig, axs[0], np.fft.fftshift(filt.real), vmin=vmin, vmax=vmax, colorbar=False)
#     im = misc.plot_img(fig, axs[1], np.fft.fftshift(filt.imag), vmin=vmin, vmax=vmax, colorbar=False)
#     fig.colorbar(im)
#     plt.title(title)
#     fig.show()

# plot(wph_op.psi[key])
# plot(psi[key])
# plot(diff)
