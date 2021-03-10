# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pywph as pw
import astropy.io.fits as fits
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cProfile
import torch.autograd.profiler as profiler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

M, N = 512, 512
J = 8
L = 8
dn = 0
cplx = True

data1 = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
data2 = torch.from_numpy(fits.open('data/I_2.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
data3 = torch.from_numpy(fits.open('data/I_2.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
data4 = torch.from_numpy(fits.open('data/Q_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32) \
                         + 1j*fits.open('data/U_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
data = torch.stack([data1, data2, data3, data4])
#data = data+1j*data
data.requires_grad = False

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
wph_op.to(0)
print(time.time() - start)

# # Method 3 with grad but and backward
# # p = cProfile.Profile(time.process_time)
# # p.enable()
# start = time.time()
# data, nb_chunks = wph_op.preconfigure(data)
# for i in range(nb_chunks):
#     print(i)
#     coeffs = wph_op.apply(data, i)
#     loss = (torch.abs(coeffs) ** 2).mean()
#     # loss.backward(retain_graph=True)
#     coeffs = None
#     loss = None
# # p.disable()
# # p.dump_stats("nom_de_fichier.prof")

# print(time.time() - start)

start = time.time()
wph_op = nn.DataParallel(wph_op).to(0)
data_loader = DataLoader(dataset=data, batch_size=4)
for data in data_loader:
    print("===========")
    coeffs = wph_op(data)
