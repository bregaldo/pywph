# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pywph as pw
import astropy.io.fits as fits
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

M, N = 512, 512
J = 8
L = 8
dn = 0
cplx = True

data = fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32)
data = torch.from_numpy(data + 1j*data/2)
data=data[:M, :N]
# data.requires_grad = True

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
print(time.time() - start)

wph_op.to(0)
start = time.time()
data_torch, nb_chunks = wph_op.preconfigure(data)
coeffs = []
for i in range(nb_chunks):
    coeffs.append(wph_op.apply(data_torch, i).detach())
coeffs = torch.cat(coeffs, -1)
b = coeffs
print(time.time() - start)
wph_op.to("cpu")

os.chdir('/home/bruno/Bureau/These ENS/Outils/pywph/pywph/pywph/')
start = time.time()
stat_params = {"J": J, "L": L, "delta_j": J-1, "delta_l": 4, "delta_n": dn,
               "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 28}
wph_op_old = pw.WPHOp_old(N, N, stat_params)
print(time.time() - start)
os.chdir('/home/bruno/Bureau/These ENS/Outils/pywph/pywph/tests/')

start = time.time()
a = wph_op_old.apply(data)
print(time.time() - start)
wph_op_old.stat_op.cpu()

nb_scaling_moments = (1 + int(torch.is_complex(data_torch))) * (J - 3) * 4

# Renaming
a_indices = a.wph_indices
a_coeffs = a.wph
b_indices = wph_op.wph_moments_indices.cpu().numpy()
b_coeffs = b.cpu().numpy()[:-nb_scaling_moments]
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
b_coeffs = b.cpu().numpy()[-nb_scaling_moments:]
if a_coeffs.shape != b_coeffs.shape:
    a_coeffs = a_coeffs[:a_coeffs.shape[0] // 2]
assert(a_coeffs.shape == b_coeffs.shape)
print("SM", np.absolute(a_coeffs - b_coeffs).max())
