# -*- coding: utf-8 -*-

import numpy as np
import pywph as pw

from wph_quijote.wph_syntheses.wph_operator_wrapper import WPHOp_quijote
from wph_quijote_legacy_model import wph_quijote_legacy_model

M, N = 256, 256
J = 6
L = 8
dn = 2
norm = "auto"

data = np.load('data/Q_1.npy')[::2, ::2] + 1j*np.load('data/U_1.npy')[::2, ::2]

for norm in [None, "auto"]:
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=0, cplx=True)
    wph_moments_indices, scaling_moments_indices = wph_quijote_legacy_model(J, L, dn=dn)
    wph_op.load_model([], extra_wph_moments=wph_moments_indices, extra_scaling_moments=scaling_moments_indices)
    wph_pywph = wph_op(data, norm=norm, ret_wph_obj=True)
    wph_op.to("cpu")
    
    stat_params = {"J": J, "L": L, "delta_j": J - 1, "delta_l": 4, "delta_n": dn,
                   "scaling_function_moments": [0, 1, 2, 3], "nb_chunks": 15}
    wph_quijote_op = WPHOp_quijote(M, N, stat_params)
    wph_quijote = wph_quijote_op.apply(data, norm=norm)
    wph_quijote_op.stat_op.cpu()

    assert np.allclose(wph_pywph.wph_coeffs, wph_quijote.wph)
    assert np.allclose(wph_pywph.sm_coeffs, wph_quijote.wph_lp)
