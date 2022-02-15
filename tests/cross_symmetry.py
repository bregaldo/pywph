# -*- coding: utf-8 -*-

import pywph as pw
import numpy as np

M, N = 64, 64
J = 4
L = 4
dn = 0

data1 = np.load('data/I_1.npy')[::8, ::8]
data2 = np.load('data/I_2.npy')[::8, ::8]

wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=0)
wph_op.load_model(cross_moments=True)

stats1 = wph_op([data1, data2], cross=True).cpu().numpy()
stats2 = wph_op([data2, data1], cross=True).cpu().numpy()
print(stats1.shape, stats2.shape)

def unique(x, rtol=1.0000000000000001e-05, atol=1e-08, print_idx=False):

    def eclose(a,b,rtol=rtol, atol=atol):
        return np.abs(a - b) <= (atol + rtol * np.abs(b))

    y = x.flat.copy()
    y.sort()
    ci = 0

    U = np.empty((0,),dtype=y.dtype)

    while ci < y.size:
        ii = eclose(y[ci],y)
        mi = np.max(ii.nonzero())
        if ci != mi and print_idx:
            print(ci, mi)
        U = np.concatenate((U,[y[mi]]))
        ci = mi + 1

    return U

print(np.unique(np.concatenate([np.abs(stats1), np.abs(stats2)])).shape)
print(unique(np.concatenate([np.abs(stats1), np.abs(stats2)]), atol=1e-12).shape)

wph1 = wph_op([data1, data2], cross=True, ret_wph_obj=True)
wph2 = wph_op([data2, data1], cross=True, ret_wph_obj=True)