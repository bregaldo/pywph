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

M, N = 256, 256
J = 6
L = 4
dn = 0
cplx = True

data = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
data = data[:M, :N]
data.requires_grad = True

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
wph_op.to(0)
print(time.time() - start)

# Method with grad but no backward
res_record = []
for mem_chunk_factor in range(20, 100, 5):
    start = time.time()
    data, nb_chunks = wph_op.preconfigure(data, mem_chunk_factor=mem_chunk_factor)
    coeffs = []
    for i in range(nb_chunks):
        coeffs.append(wph_op.apply(data, i).detach())
    coeffs = torch.cat(coeffs, -1)
    b = coeffs
    ell_time = time.time() - start
    print(ell_time)
    res_record.append([mem_chunk_factor, ell_time])
res_record = np.array(res_record)
