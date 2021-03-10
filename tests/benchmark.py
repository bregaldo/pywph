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
norm = "auto"

data = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))

start = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, cplx=cplx)
print(time.time() - start)

wph_op.to(0)
start = time.time()
data, nb_chunks = wph_op.preconfigure(data, requires_grad=True)
for i in range(nb_chunks):
    print(i, f"/{nb_chunks}")
    coeffs = wph_op.apply(data, i, norm=norm)
    loss = (torch.abs(coeffs) ** 2).mean()
    loss.backward(retain_graph=True)
    del coeffs, loss
print(time.time() - start)
wph_op.to("cpu")

os.chdir('/home/bruno/Bureau/These ENS/Outils/pywph/pywph/pywph/')
start = time.time()
stat_params = {"J": 8, "L": 8, "delta_j": 7, "delta_l": 4, "delta_n": 0,
               "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 60}
wph_op_old = pw.WPHOp_old(N, N, stat_params)
print(time.time() - start)
os.chdir('/home/bruno/Bureau/These ENS/Outils/pywph/pywph/tests/')

data = torch.from_numpy(fits.open('data/I_1.fits')[0].data.byteswap().newbyteorder().astype(np.float32))
data = wph_op_old._to_torch(np.expand_dims(data, axis=0))
data.requires_grad = True

start = time.time()
for chunk_id in range(wph_op_old.nb_chunks + 1):
    print(chunk_id, f"/{stat_params['nb_chunks'] + 1}")
    wph_chunk = wph_op_old.stat_op(data, chunk_id, norm=norm)  # (nb,nc,nb_channels,1,1,2)
    loss = (torch.abs(wph_chunk) ** 2).mean()
    loss.backward()
print(time.time() - start)
wph_op_old.stat_op.cpu()
