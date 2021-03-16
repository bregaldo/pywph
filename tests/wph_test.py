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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

M, N = 256, 256
J = 5
L = 8
dn = 0
cplx = True
norm = "auto"

data = fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32)
data = data[:M, :N]

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
wph_op.to(0)
print(time.time() - start)

start = time.time()
wph = wph_op.apply(data, norm=norm, ret_wph_obj=True)
print(time.time() - start)


