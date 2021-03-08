# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import os
import torch
import sys

from .stats.wph_syntheses.wph_operator import PhaseHarmonics2d
from .stats.wph_syntheses.utils import periodic_dis


class WPHOp_old:
    
    def __init__(self, M, N, stat_params, device=0):
        self.stat_params = stat_params
        self.nb_chunks = self.stat_params['nb_chunks']
        self.M, self.N = M, N
        self.device = device
        self.init = False
        
        self.stat_op = PhaseHarmonics2d(M=self.M, N=self.N, **self.stat_params, devid=device)
        self.stat_op.cuda() # CUDA or CPU actually (dependy on device)
        pass
    
    def _to_torch(self, a):
        if np.iscomplexobj(a):
            ret = torch.zeros(a.shape + (2,), dtype=torch.float)
            ret[..., 0] = torch.tensor(a.real, dtype=torch.float)
            ret[..., 1] = torch.tensor(a.imag, dtype=torch.float)
        else:
            ret = torch.tensor(a, dtype=torch.float)
        ret = ret.to(self.device)
        return ret
    
    def apply(self, data, norm=None):
        if data.shape != (self.M, self.N):
            raise Exception("Expecting 2D data of shape (M, N)!")
            
        if np.iscomplexobj(data):
            print("Applying WPH operator to complex data...")
        else:
            print("Applying WPH operator to real data...")
        data_torch = self._to_torch(np.expand_dims(data, axis=0))
        
        wph = []
        
        if not self.init:
            if self.device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            self.nb_cov = 0
            for chunk_id in range(self.nb_chunks + 1):
                wph_chunk = self.stat_op(data_torch, chunk_id, norm=norm)  # (nb,nc,nb_channels,1,1,2)
                self.nb_cov += wph_chunk.shape[2]
                wph.append(wph_chunk)
            print(f"Number of WPH coefficients: {self.nb_cov}")
            if self.device != "cpu":
                print(f"Max memory usage: {torch.cuda.max_memory_allocated() / 1e9} Go")
        else:
            for chunk_id in range(self.nb_chunks + 1):
                wph_chunk = self.stat_op(data_torch, chunk_id, norm=norm)  # (nb,nc,nb_channels,1,1,2)
                wph.append(wph_chunk)
               
        return WPH(wph, self.stat_params, self.stat_op.wph_by_chunk)


class WPH:
    
    def __init__(self, wph, stat_params, wph_by_chunk):
        self.nb_chunks = len(wph) - 1
        self.L = stat_params['L']
        self.J = stat_params['J']
        self.scaling_function_moments = stat_params['scaling_function_moments']
        
        self.wph = None
        self.wph_indices = None
        self.wph_lp = None
        self.wph_lp_indices = None
        
        # Regular coefficients reformatting into complex numpy arrays
        wph_tmp = []
        for chunk_id in range(self.nb_chunks):
            wph_chunk = wph[chunk_id] # (1, 1, L2 * P_c, 1, 1, 2)
            wph_chunk = wph_chunk.view(-1, 2).cpu().numpy() # (L2 * P_c, 2)
            wph_chunk = wph_chunk[..., 0] + 1j*wph_chunk[..., 1] # (L2 * P_c)
            wph_tmp.append(wph_chunk)
        self.wph = np.concatenate(wph_tmp)
        
        # Regular coefficients indices
        wph_indices_tmp = []
        for chunk_id in range(self.nb_chunks):
            # Indices conversion to numpy arrays into complex numpy arrays
            wph_indices_chunk = {}
            for key in wph_by_chunk[chunk_id].keys():
                wph_indices_chunk[key] = wph_by_chunk[chunk_id][key].cpu().numpy()
                
            # Order : [j1, l1, k1, dn1, j2, l2, k2, dn2]
            len_chunk_l1_fixed = len(wph_indices_chunk['j1'])
            wph_indices_chunk_new = np.zeros((2 * self.L * len_chunk_l1_fixed, 8))
            
            for ell1 in range(2 * self.L):
                ell2 = np.remainder(wph_indices_chunk['ell2'] + ell1, 2 * self.L)
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 0] = wph_indices_chunk['j1']
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 1] = ell1
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 2] = wph_indices_chunk['k1']
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 3] = wph_indices_chunk['dn1']
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 4] = wph_indices_chunk['j2']
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 5] = ell2
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 6] = wph_indices_chunk['k2']
                wph_indices_chunk_new[ell1 * len_chunk_l1_fixed: (ell1 + 1) * len_chunk_l1_fixed, 7] = wph_indices_chunk['dn2']
            wph_indices_tmp.append(wph_indices_chunk_new)
        self.wph_indices = np.concatenate(wph_indices_tmp)
            
        # Low-pass coefficients reformatting
        wph_chunk = wph[-1] # LP coefficients of shape (1, 1, 2 * J*K, 1, 1, 2)
        wph_chunk = wph_chunk.view(2, -1, 2).cpu().numpy() # (2, J*K, 2)
        wph_chunk = wph_chunk[..., 0] + 1j*wph_chunk[..., 1] # (2, J*K)
        self.wph_lp = wph_chunk
        
        # Low-pass coefficients indices
        wph_indices_tmp = []
        for part in [0, 1]:
            wph_indices_tmp_part = []
            for j in range(2, self.J - 1):
                for p in self.scaling_function_moments:
                    wph_indices_tmp_part.append([j, p])
            wph_indices_tmp.append(wph_indices_tmp_part)
        self.wph_lp_indices = np.array(wph_indices_tmp)
        
        self.reorder() # Reorder coefficients (lexicographical order)
    
    def _filter_args(self, j1=None, l1=None, k1=None, dn1=None, j2=None, l2=None, k2=None, dn2=None):
        filtering = np.ones(self.wph_indices.shape[0], np.bool)
        if j1 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 0] == j1)
        if l1 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 1] == l1)
        if k1 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 2] == k1)
        if dn1 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 3] == dn1)
        if j2 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 4] == j2)
        if l2 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 5] == l2)
        if k2 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 6] == k2)
        if dn2 is not None:
            filtering = np.logical_and(filtering, self.wph_indices[:, 7] == dn2)
        return filtering
    
    def reorder(self):
        # Lexicographical reordering on [j1, l1, k1, dn1, j2, l2, k2, dn2]
        indices = np.lexsort(self.wph_indices.T[::-1, :])
        wph_copy = self.wph.copy()
        wph_indices_copy = self.wph_indices.copy()
        for i in range(self.wph.shape[0]):
            self.wph[i] = wph_copy[indices[i]]
            self.wph_indices[i] = wph_indices_copy[indices[i]]
    
    def get_coeffs(self, j1=None, l1=None, k1=None, dn1=None, j2=None, l2=None, k2=None, dn2=None):
        filtering = self._filter_args(j1=j1, l1=l1, k1=k1, dn1=dn1, j2=j2, l2=l2, k2=k2, dn2=dn2)
        return self.wph[filtering, ...], self.wph_indices[filtering, :]
    
    def get_coeffs_class(self, clas="", dn1=None, dn2=None):
        coeffs, indices = self.get_coeffs(dn1=dn1, dn2=dn2) # First dn1 and dn2 filtering
        filtering = np.ones(indices.shape[0], np.bool)
        if clas == "":
            pass
        elif clas == "S11": # Power spectrum coefficients
            filtering = np.logical_and(filtering, indices[:, 0] == indices[:, 4]) # j1 == j2
            filtering = np.logical_and(filtering, indices[:, 2] == 1) # k1 == 1
            filtering = np.logical_and(filtering, indices[:, 6] == 1) # k2 == 1
        elif clas == "S00": # For sparsity
            filtering = np.logical_and(filtering, indices[:, 0] == indices[:, 4]) # j1 == j2
            filtering = np.logical_and(filtering, indices[:, 2] == 0) # k1 == 0
            filtering = np.logical_and(filtering, indices[:, 6] == 0) # k2 == 0
        elif clas == "S01": # First structures
            filtering = np.logical_and(filtering, indices[:, 0] == indices[:, 4]) # j1 == j2
            filtering = np.logical_and(filtering, indices[:, 2] == 0) # k1 == 0
            filtering = np.logical_and(filtering, indices[:, 6] == 1) # k2 == 1
        elif clas == "C00": # Scale coupling moments
            filtering = np.logical_and(filtering, indices[:, 2] == 0) # k1 == 0
            filtering = np.logical_and(filtering, indices[:, 6] == 0) # k2 == 0
        elif clas == "C00ex": # Scale coupling moments excluding j1 = j2 moments
            filtering = np.logical_and(filtering, indices[:, 0] != indices[:, 4]) # j1 != j2
            filtering = np.logical_and(filtering, indices[:, 2] == 0) # k1 == 0
            filtering = np.logical_and(filtering, indices[:, 6] == 0) # k2 == 0
        elif clas == "C01": # Scale coupling moments
            filtering = np.logical_and(filtering, indices[:, 2] == 0) # k1 == 0
            filtering = np.logical_and(filtering, indices[:, 6] == 1) # k2 == 1
        elif clas == "C01ex": # Scale coupling moments excluding j1 = j2 moments
            filtering = np.logical_and(filtering, indices[:, 0] != indices[:, 4]) # j1 != j2
            filtering = np.logical_and(filtering, indices[:, 2] == 0) # k1 == 0
            filtering = np.logical_and(filtering, indices[:, 6] == 1) # k2 == 1
        elif clas == "Cphase": # Phase coupling moments
            filtering = np.logical_and(filtering, indices[:, 2] == 1) # k1 == 1
            # and k2 = xi1 / xi2
        elif clas == "Cphase_ex": # Phase coupling moments excluding S11 moments
            filtering = np.logical_and(filtering, indices[:, 0] != indices[:, 4]) # j1 != j2
            filtering = np.logical_and(filtering, indices[:, 2] == 1) # k1 == 1
            # and k2 = xi1 / xi2
        else:
            raise Exception("Unknown class of coefficients!")
        return coeffs[filtering, ...], indices[filtering, :]
    
    def to_isopar(self):
        """
        Comptute isotropic and parity invariant coefficients.

        Returns
        -------
        None.

        """
        indices_cnt = {}
        wph_isopar = {}
        
        # Filling
        for i in range(self.wph_indices.shape[0]):
            j1, l1, k1, dn1, j2, l2, k2, dn2 = tuple(self.wph_indices[i])
            dl = periodic_dis(l1, l2, 2 * self.L)
            if (j1, 0, k1, dn1, j2, dl, k2, dn2) in indices_cnt.keys():
                indices_cnt[(j1, 0, k1, dn1, j2, dl, k2, dn2)] += 1
                wph_isopar[(j1, 0, k1, dn1, j2, dl, k2, dn2)] += self.wph[i]
            else:
                indices_cnt[(j1, 0, k1, dn1, j2, dl, k2, dn2)] = 1
                wph_isopar[(j1, 0, k1, dn1, j2, dl, k2, dn2)] = self.wph[i]
                
        # Conversion into numpy arrays
        indices = []
        wph_isopar_list = []
        for key in indices_cnt.keys():
            indices.append(key)
            wph_isopar_list.append(wph_isopar[key] / indices_cnt[key])
        indices = np.array(indices)
        wph_isopar = np.array(wph_isopar_list)
        
        # Reordering and save
        self.wph_indices = indices
        self.wph = wph_isopar
        self.reorder()
        return self
        
    def __add__(self, other):
        cpy = copy.deepcopy(self)
        if type(other) == WPH:
            cpy.wph += other.wph
            cpy.wph_lp += other.wph_lp
        else:
            cpy.wph += other
            cpy.wph_lp += other
        return cpy
    
    def __sub__(self, other):
        cpy = copy.deepcopy(self)
        if type(other) == WPH:
            cpy.wph -= other.wph
            cpy.wph_lp -= other.wph_lp
        else:
            cpy.wph -= other
            cpy.wph_lp -= other
        return cpy

    def __pow__(self, power):
        cpy = copy.deepcopy(self)
        cpy.wph = cpy.wph ** power
        cpy.wph_lp = cpy.wph_lp ** power
        return cpy
    
    def __truediv__(self, other):
        cpy = copy.deepcopy(self)
        if type(other) == WPH:
            cpy.wph /= other.wph
            cpy.wph_lp /= other.wph_lp
        else:
            cpy.wph /= other
            cpy.wph_lp /= other
        return cpy
    
    def __abs__(self):
        cpy = copy.deepcopy(self)
        cpy.wph = np.abs(cpy.wph)
        cpy.wph_lp = np.abs(cpy.wph_lp)
        return cpy


def plot_wph(wph_list, labels, var=None, save=False, save_dir="", save_end="",
             logscale=True, vmins=None, vmaxs=None, unique_plot=False):
    if save:
        plt.ioff()
        
        # Make the output directory if it does not exist yet
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if var is None:
        var = [None] * len(wph_list)
    if len(var) != len(wph_list):
        raise Exception("Invalid length for var!")
        
    if vmins is None:
        vmins = [None] * 6
    if vmaxs is None:
        vmaxs = [None] * 6
    
    #
    # S coefficients
    #
    
    if unique_plot:
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs_S00, axs_S11, axs_S01, axs_C01, axs_Cphase, axs_L = axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]
    else:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs_S00, axs_S11, axs_S01 = axs[0], axs[1], axs[2]
    
    # S00
    for index, wph in enumerate(wph_list):
        coeffs = wph.get_coeffs_class("S00", dn1=0, dn2=0)[0].real
        p = axs_S00.plot(coeffs, label=labels[index])
        if var[index] is not None:
            coeffs_std = np.sqrt(var[index].get_coeffs_class("S00", dn1=0, dn2=0)[0].real)
            axs_S00.fill_between(np.arange(len(coeffs_std)),
                                 coeffs - coeffs_std,
                                 coeffs + coeffs_std,
                                 color=p[-1].get_color(),
                                 alpha=0.2)
        
    # S11
    for index, wph in enumerate(wph_list):
        coeffs = wph.get_coeffs_class("S11", dn1=0, dn2=0)[0].real
        p = axs_S11.plot(coeffs, label=labels[index])
        if var[index] is not None:
            coeffs_std = np.sqrt(var[index].get_coeffs_class("S11", dn1=0, dn2=0)[0].real)
            axs_S11.fill_between(np.arange(len(coeffs_std)),
                                 coeffs - coeffs_std,
                                 coeffs + coeffs_std,
                                 color=p[-1].get_color(),
                                 alpha=0.2)
        
    # S01
    for index, wph in enumerate(wph_list):
        coeffs = np.absolute(wph.get_coeffs_class("S01", dn1=0, dn2=0)[0])
        p = axs_S01.plot(coeffs, label=labels[index])
        if var[index] is not None:
            coeffs_std = np.sqrt(np.absolute(var[index].get_coeffs_class("S01", dn1=0, dn2=0)[0]))
            axs_S01.fill_between(np.arange(len(coeffs_std)),
                                 coeffs - coeffs_std,
                                 coeffs + coeffs_std,
                                 color=p[-1].get_color(),
                                 alpha=0.2)
    
    if logscale:
        axs_S00.set_yscale('log')
        axs_S11.set_yscale('log')
        axs_S01.set_yscale('log')
    axs_S00.set_xlabel(r'$j_1 = j_2$')
    axs_S00.set_ylabel(r'$S^{(0,0)}$')
    axs_S00.set_ylim(vmins[0], vmaxs[0])
    axs_S00.legend()
    axs_S11.set_xlabel(r'$j_1 = j_2$')
    axs_S11.set_ylabel(r'$S^{(1,1)}$')
    axs_S11.set_ylim(vmins[1], vmaxs[1])
    axs_S01.set_xlabel(r'$j_1 = j_2$')
    axs_S01.set_ylabel(r'$|S^{(0,1)}|$')
    axs_S01.set_ylim(vmins[2], vmaxs[2])
    
    if not unique_plot:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"wph_s{save_end}.png"))
    
    #
    # C coefficients
    #
    
    if not unique_plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs_C01, axs_Cphase = axs[0], axs[1]
    
    # C01
    for index, wph in enumerate(wph_list):
        coeffs = np.absolute(wph.get_coeffs_class("C01ex", dn1=0, dn2=0)[0])
        p = axs_C01.plot(coeffs, label=labels[index])
        if var[index] is not None:
            coeffs_std = np.sqrt(np.absolute(var[index].get_coeffs_class("C01ex", dn1=0, dn2=0)[0]))
            axs_C01.fill_between(np.arange(len(coeffs_std)),
                                 coeffs - coeffs_std,
                                 coeffs + coeffs_std,
                                 color=p[-1].get_color(),
                                 alpha=0.2)
    indices = wph_list[0].get_coeffs_class("C01ex", dn1=0, dn2=0)[1] # ex to avoid redundancy
    j1_cur = -1
    j2_cur = -1
    for cnt, index in enumerate(indices):
        if index[0] != j1_cur:
            j1_cur = index[0]
            j2_cur = j1_cur + 1
            axs_C01.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
        elif index[4] != j2_cur:
            j2_cur = index[4]
            axs_C01.axvline(cnt, color='black', alpha=0.1, linestyle=':')
    axs_C01.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
    
    # Cphase
    for index, wph in enumerate(wph_list):
        coeffs = np.absolute(wph.get_coeffs_class("Cphase", dn1=0, dn2=0)[0])
        p = axs_Cphase.plot(coeffs, label=labels[index])
        if var[index] is not None:
            coeffs_std = np.sqrt(np.absolute(var[index].get_coeffs_class("Cphase", dn1=0, dn2=0)[0]))
            axs_Cphase.fill_between(np.arange(len(coeffs_std)),
                                    coeffs - coeffs_std,
                                    coeffs + coeffs_std,
                                    color=p[-1].get_color(),
                                    alpha=0.2)
    indices = wph_list[0].get_coeffs_class("Cphase", dn1=0, dn2=0)[1]
    j1_cur = -1
    for cnt, index in enumerate(indices):
        if index[0] != j1_cur:
            j1_cur = index[0]
            axs_Cphase.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
    
    if logscale:
        axs_C01.set_yscale('log')
        axs_Cphase.set_yscale('log')
    axs_C01.set_xticks([])
    axs_C01.set_xlabel(r"$j_1 < j_2$ with $\delta l = 4$")
    axs_C01.set_ylabel(r'$C^{(0,1)}$')
    axs_C01.set_ylim(vmins[3], vmaxs[3])
    if not unique_plot:
        axs_C01.legend()
    axs_Cphase.set_xticks([])
    axs_Cphase.set_xlabel(r"$j_1 < j_2$ with $\delta l = 0$")
    axs_Cphase.set_ylabel(r'$C^{\mathrm{phase}}$')
    axs_Cphase.set_ylim(vmins[4], vmaxs[4])
    
    if not unique_plot:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"wph_c{save_end}.png"))
    
    #
    # L coefficients
    #
    
    if not unique_plot:
        fig, axs = plt.subplots(1, 1)
        axs_L = axs
    
    # L
    for index, wph in enumerate(wph_list):
        coeffs = np.concatenate((wph.wph_lp[0], wph.wph_lp[1])).real
        p = axs_L.plot(coeffs, label=labels[index])
        if var[index] is not None:
            coeffs_var = np.concatenate((var[index].wph_lp[0], var[index].wph_lp[1])).real
            coeffs_std = np.sqrt(coeffs_var)
            axs_L.fill_between(np.arange(len(coeffs_std)),
                               coeffs - coeffs_std,
                               coeffs + coeffs_std,
                               color=p[-1].get_color(),
                               alpha=0.2)
    indices = np.concatenate((wph_list[0].wph_lp_indices[0], wph_list[0].wph_lp_indices[1]))
    j1_cur = -1
    for cnt, index in enumerate(indices):
        if index[0] > j1_cur:
            j1_cur = index[0]
            axs_L.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
        elif index[0] < j1_cur: # We go from real part signal coeffs to imag part signal coeffs
            j1_cur = index[0]
            axs_L.axvline(cnt, color='black')
    axs_L.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
    
    if logscale:
        axs_L.set_yscale('log')
    axs_L.set_xticks([])
    axs_L.set_xlabel(r"$2 \leq j \leq J - 2$ with $p \in \{0, 1, 2, 3\}$")
    axs_L.set_ylabel(r'$L_{j, p}$')
    axs_L.set_ylim(vmins[5], vmaxs[5])
    if not unique_plot:
        axs_L.legend()
    
    plt.tight_layout()
    if not unique_plot:
        plt.savefig(os.path.join(save_dir, f"wph_l{save_end}.png"))
    else:
        plt.savefig(os.path.join(save_dir, f"wph{save_end}.png"))
