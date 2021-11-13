# -*- coding: utf-8 -*-

import pywph as pw
import torch
import numpy as np

M, N = 256, 256
J = 7
L = 8
dn = 0

device = 0

# Load data
data1 = np.load('data/I_1.npy')[::2, ::2]
data2 = np.load('data/I_2.npy')[::2, ::2]

wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)

# Compute the coefficients and the gradient of a final loss by accumulating the gradients of the chunks of the loss
data_torch, nb_chunks = wph_op.preconfigure([data1, data2], requires_grad=True, cross=True) # Divide the computation into chunks
coeffs = []
for i in range(nb_chunks):
    print(f"{i}/{nb_chunks}")
    coeffs_chunk = wph_op([data_torch[0], data_torch[1]], i, cross=True)
    loss_chunk = (torch.absolute(coeffs_chunk) ** 2).sum()
    loss_chunk.backward(retain_graph=True)
    coeffs.append(coeffs_chunk.detach().cpu())
    del coeffs_chunk, loss_chunk # To free GPU memory
coeffs = torch.cat(coeffs, -1)
grad = [data_torch[0].grad, data_torch[1].grad]

print(grad)
