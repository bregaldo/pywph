# -*- coding: utf-8 -*-

import numpy as np
import pywph as pw
import torch
import time

M, N = 256, 256
J = 7
L = 8
dn = 2
norm = None

device = 0

# Load data
data = np.load('data/I_1.npy')[::2, ::2]

# Build WPH operator
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)

# Analysis without any gradient computation
print("====== Without gradient computation ======\n")
for mem_chunk_factor in range(25, 100, 5):
    start_time = time.time()
    data_torch, nb_chunks = wph_op.preconfigure(data,
                                                mem_chunk_factor=mem_chunk_factor)
    for i in range(nb_chunks):
        coeffs_chunk = wph_op.apply(data_torch, i, norm=norm)
        del coeffs_chunk
    print(f"mem_chunk_factor = {mem_chunk_factor} -> ellapsed_time = {time.time() - start_time}")
    del data_torch


# Analysis with gradient computation
print("\n====== With gradient computation ======\n")
for mem_chunk_factor_grad in range(50, 115, 5):
    start_time = time.time()
    data_torch, nb_chunks = wph_op.preconfigure(data,
                                                mem_chunk_factor_grad=mem_chunk_factor_grad,
                                                requires_grad=True)
    for i in range(nb_chunks):
        coeffs_chunk = wph_op.apply(data_torch, i, norm=norm)
        loss_chunk = (torch.absolute(coeffs_chunk) ** 2).sum() # Some loss
        loss_chunk.backward(retain_graph=True)
        del coeffs_chunk, loss_chunk # To free GPU memory
    print(f"mem_chunk_factor_grad = {mem_chunk_factor_grad} -> ellapsed_time = {time.time() - start_time}")
    del data_torch
