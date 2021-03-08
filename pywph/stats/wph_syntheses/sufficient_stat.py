import torch

from .utils import periodic_dis

def compute_idx_of_sufficient_stat(L, J, dj, dl, dn):
    L2 = L * 2
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2 = [], [], [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)

    # j1=j2, k1=0,1, k2=0 or 1
    for j1 in range(J):
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 0, 1, 0, 0)
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, 0)
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 0, 0, 0, 0)
        if j1 == J - 1:
            max_dn = 0
        elif j1 == J - 2:
            max_dn = min(1, dn)
        else:
            max_dn = dn
        for n in range(4*max_dn):
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, (n+1))
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 0, 0, 0, (n+1))


    # k1 = 0,1
    # k2 = 0,1 or 2**(j2-j1)
    # k2 > k1
    # j1+1 <= j2 <= min(j1+dj,J-1)
    for j1 in range(J):
        for j2 in range(j1 + 1, min(j1 + dj + 1, J)):
            if j2 == J - 1:
                max_dn = 0
            elif j2 == J - 2:
                max_dn = min(1, dn)
            else:
                max_dn = dn
            for n in range(4 * max_dn):
                add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j2, 1, 2 ** (j2 - j1), 0, (n+1))
                add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j2, 0, 1, 0, (n + 1))
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j2, 1, 2**(j2-j1), 0, 0)
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j2, 0, 1, 0, 0)

    print("Total number of coefficient: " + str(len(idx_k2)))

    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)

def compute_idx_of_sufficient_stat_PS(L, J, dj, dl, dn):
    L2 = L * 2
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2 = [], [], [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)

    # j1=j2, k1=1, k2=1
    for j1 in range(J):
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, 0)
        if j1 == J - 1:
            max_dn = 0
        elif j1 == J - 2:
            max_dn = min(1, dn)
        else:
            max_dn = dn
        for n in range(4*max_dn):
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, (n+1))

    print("Total number of coefficient: " + str(len(idx_k2)))

    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)


def add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j2, k1, k2, dn1, dn2):
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2 = idx_lists
    for ell2 in range(L2):
    #for ell2 in range(0,L2,2):
        if periodic_dis(0, ell2, L2) <= dl:
            idx_j1.append(j1)
            idx_j2.append(j2)
            idx_k1.append(k1)
            idx_k2.append(k2)
            idx_ell2.append(ell2)
            idx_dn1.append(dn1)
            idx_dn2.append(dn2)


def get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2):
    idx_wph = dict()
    idx_wph['j1'] = torch.tensor(idx_j1).type(torch.long)
    idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long)
    idx_wph['ell2'] = torch.tensor(idx_ell2).type(torch.long)
    idx_wph['j2'] = torch.tensor(idx_j2).type(torch.long)
    idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long)
    idx_wph['dn1'] = torch.tensor(idx_dn1).type(torch.long)
    idx_wph['dn2'] = torch.tensor(idx_dn2).type(torch.long)

    return idx_wph
