# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pywph as pw
import astropy.io.fits as fits
import torch
import cProfile
import torch.autograd.profiler as profiler

sys.path.append('/home/bruno/Bureau/These ENS/Outils/misc/')

import misc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

M, N = 512, 512
J = 8
L = 8
dn = 0
cplx = True

data = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
#data = data+1j*data
data.requires_grad = True

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
wph_op.to(0)
print(time.time() - start)

# # Method 1
# start = time.time()
# a = wph_op.apply(data)
# print(time.time() - start)

# # Method 1 bis
# start = time.time()
# data, nb_chunks = wph_op.preconfigure(data, mem_chunk_factor=20)
# coeffs = []
# for i in range(nb_chunks):
#     coeffs.append(wph_op.apply(data, i).detach())
# coeffs = torch.cat(coeffs, -1)
# b = coeffs
# ell_time = time.time() - start
# print(ell_time)

# # Method 2 with grad but no backward
# start = time.time()
# data, nb_chunks = wph_op.preconfigure(data, requires_grad=True)
# coeffs = []
# for i in range(nb_chunks):
#     coeffs.append(wph_op.apply(data, i).detach())
# coeffs = torch.cat(coeffs, -2)
# b = coeffs
# print(time.time() - start)

# Method 3 with grad but and backward
# p = cProfile.Profile(time.process_time)
# p.enable()
start = time.time()
data, nb_chunks = wph_op.preconfigure(data)
for i in range(nb_chunks):
    print(i)
    coeffs = wph_op.apply(data, i)
    loss = (torch.abs(coeffs) ** 2).mean()
    loss.backward(retain_graph=True)
# p.disable()
# p.dump_stats("nom_de_fichier.prof")

print(time.time() - start)
