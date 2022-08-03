# -*- coding: utf-8 -*-

import pywph as pw
import numpy as np

M, N = 256, 256
J = 7
L = 4
dn = 0

data = np.load('data/I_1.npy')[::2, ::2]

wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=0)

# Method 1: asking for the torch.tensor of coefficients
stats = wph_op(data).cpu().numpy()
print(stats.shape, stats)

# Method 2: asking for a WPH object
wph = wph_op(data, ret_wph_obj=True)
s11, s11_indices = wph.get_coeffs("S11") # Select the S11 coefficients
print(s11.shape, s11, s11_indices)

# Change the model to only compute S11 coefficients
wph_op.load_model("S11")
s11_bis = wph_op(data).cpu().numpy()
print(s11_bis.shape, s11_bis)
