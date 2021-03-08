# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import astropy.io.fits as fits
import torch

sys.path.append("/home/bruno/Bureau/These ENS/Outils/pywph/pywph/")
sys.path.append('/home/bruno/Bureau/These ENS/Outils/misc/')
import pywph as pw

import misc


M, N = 512, 512
J = 8
L = 8
dn = 0
cplx = True

list_indices = pw.build_cov_indices(J, L, dn=dn, len_alpha_list=4) # Order : (j1, t1, k1, j2, t2, k2, n)

counts = np.zeros((J, 2*L, J, 2*L))
for elt in list_indices:
    counts[elt[0], elt[1], elt[3], elt[4]] += 1
vals, vals_counts = np.unique(counts, return_counts=True)

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
print(time.time() - start)

data = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
# data.requires_grad = True

# start = time.time()
# a = wph_op.apply(data)
# print(time.time() - start)
wph_op.to(0)
start = time.time()
b = wph_op.apply(data)
print(time.time() - start)
