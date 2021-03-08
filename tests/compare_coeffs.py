# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pywph as pw
import astropy.io.fits as fits
import torch

sys.path.append('/home/bruno/Bureau/These ENS/Outils/misc/')

import misc


M, N = 512, 512
J = 8
L = 8
dn = 0
cplx = True

data = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
# data.requires_grad = True

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
print(time.time() - start)

wph_op.to(0)
start = time.time()
data_torch, nb_chunks = wph_op.preconfigure(data, mem_chunk_factor=20)
coeffs = []
for i in range(nb_chunks):
    coeffs.append(wph_op.apply(data_torch, i).detach())
coeffs = torch.cat(coeffs, -1)
b = coeffs
print(time.time() - start)
wph_op.to("cpu")

os.chdir('/home/bruno/Bureau/These ENS/Projets/Planck_denoising/Scripts/CompSepAlgo/')
start = time.time()
stat_params = {"J": 8, "L": 8, "delta_j": 7, "delta_l": 4, "delta_n": 0,
               "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 28}
wph_op_old = pw.WPHOp_old(N, N, stat_params)
print(time.time() - start)
os.chdir('/home/bruno/Bureau/These ENS/Outils/pywph/pywph/tests/')

start = time.time()
a = wph_op_old.apply(data)
print(time.time() - start)
wph_op_old.stat_op.cpu()

nb_scaling_moments = 2 * (J - 3) * 4

# Renaming
a_indices = a.wph_indices
a_coeffs = a.wph
b_indices = wph_op.wph_cov_indices.cpu().numpy()
b_coeffs = b.cpu().numpy()[:-nb_scaling_moments, 0] + 1j*b.cpu().numpy()[:-nb_scaling_moments, 1]
assert(a_coeffs.shape == b_coeffs.shape)

# Reordering
a_new_indices = np.lexsort(a_indices[:, ::-1].T)
a_coeffs_new = np.zeros_like(a_coeffs)
for i in range(a_indices.shape[0]):
    a_coeffs_new[i] = a_coeffs[a_new_indices[i]]
b_new_indices = np.lexsort(b_indices[:, ::-1].T)
b_indices_new = np.zeros_like(b_indices)
b_coeffs_new = np.zeros_like(b_coeffs)
for i in range(a_indices.shape[0]):
    b_coeffs_new[i] = b_coeffs[b_new_indices[i]]
    b_indices_new[i] = b_indices[b_new_indices[i]]
print("WPH", np.absolute(a_coeffs_new - b_coeffs_new).max())

# Renaming
a_indices = a.wph_lp_indices
a_coeffs = a.wph_lp.ravel()
b_coeffs = b.cpu().numpy()[-nb_scaling_moments:, 0] + 1j*b.cpu().numpy()[-nb_scaling_moments:, 1]
assert(a_coeffs.shape == b_coeffs.shape)
print("SM", np.absolute(a_coeffs - b_coeffs).max())
