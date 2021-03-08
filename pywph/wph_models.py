# -*- coding: utf-8 -*-

import numpy as np


def build_cov_indices(J, L, dn=0, len_alpha_list=1):
    l = []
    
    # S11
    for j1 in range (J):
        for t1 in range(2*L):
            dn_eff = min(J - 1 - j1, dn)
            for n in range(dn_eff * len_alpha_list + 1):
                l.append([j1, t1, 1, j1, t1, 1, n])
    
    # S00
    for j1 in range(J):
        for t1 in range(2*L):
            dn_eff = min(J - 1 - j1, dn)
            for n in range(dn_eff * len_alpha_list + 1):
                l.append([j1, t1, 0, j1, t1, 0, n])
    
    # S01
    for j1 in range(J):
        for t1 in range(2*L):
            l.append([j1, t1, 0, j1, t1, 1, 0])
    
    # C01
    for j1 in range(J):
        for j2 in range(j1 + 1, J):
            for t1 in range(2*L):
                for t2 in range(t1 - L // 2, t1 + L // 2 + 1):
                    if t1 == t2:
                        dn_eff = min(J - 1 - j2, dn)
                        for n in range(dn_eff * len_alpha_list + 1):
                            l.append([j1, t1, 0, j2, t2, 1, n])
                    else:
                        l.append([j1, t1, 0, j2, t2 % (2*L), 1, 0])
    
    # Cphase
    for j1 in range(J):
        for j2 in range(j1 + 1, J):
            for t1 in range(2*L):
                dn_eff = min(J - 1 - j2, dn)
                for n in range(dn_eff * len_alpha_list + 1):
                    l.append([j1, t1, 1, j2, t1, 2 ** (j2 - j1), n])
    
    return np.array(l)
