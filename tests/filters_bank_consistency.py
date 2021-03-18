# -*- coding: utf-8 -*-

import os
import numpy as np
import pywph as pw

from wph_quijote.wph_syntheses.wph_operator_wrapper import WPHOp_quijote

M, N = 256, 256
J = 6
L = 8
dn = 2

# Make sure that we have built the wph_quijote filters
stat_params = {"J": J, "L": L, "delta_j": J - 1, "delta_l": 4, "delta_n": dn,
               "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 15}
wph_quijote_op = WPHOp_quijote(M, N, stat_params)

# Load wph_quijote filters
input_dir = "wph_quijote/bump_steerable_wavelet/filters/"
bp_filters = np.load(os.path.join(input_dir, f'bump_steerable_wavelet_N_{N}_J_{J}_L{L}_dn{dn}.npy'), allow_pickle=True).item()['filt_fftpsi'].astype(np.complex_)
lp_filters = np.load(os.path.join(input_dir, f'bump_scaling_functions_N_{N}_J_{J}.npy'), allow_pickle=True).astype(np.complex_)

# Reordering of psi_f
psi_f = []
for j in range(J):
    for theta in range(2 * L):
        for n in range(dn + 1):
            if n == 0:
                psi_f.append(bp_filters[theta, j, 0])
            else:
                psi_f.append(bp_filters[theta, j, 1 + (n - 1)*4 + 0])
                psi_f.append(bp_filters[theta, j, 1 + (n - 1)*4 + 1])
                psi_f.append(bp_filters[theta, j, 1 + (n - 1)*4 + 2])
                psi_f.append(bp_filters[theta, j, 1 + (n - 1)*4 + 3])
psi_f = np.array(psi_f)

# Same for phi_f
phi_f = []
for j in range(2, J - 1):
    phi_f.append(lp_filters[j - 2])
phi_f = np.array(phi_f)

# Build pywph filters
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn)

# Tests
assert np.allclose(psi_f, wph_op.psi_f)
assert np.allclose(phi_f, wph_op.phi_f)
