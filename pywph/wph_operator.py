# -*- coding: utf-8 -*-

import os
from functools import partial
import numpy as np
import torch
import multiprocessing as mp
import time

from .filters import GaussianFilter, BumpSteerableWavelet
from .utils import to_torch, get_precision_type, get_memory_available, fft, ifft, mul, mean, modulus, PhaseHarmonics, PowerHarmonics
from .wph_models import build_cov_indices
torch.backends.cudnn.enabled = (
    True  # make sure to use cudnn for computational performance
)


# Internal function for the parallel pre-building of the bandpass filters (see WPHOp.load_filters)
def _build_bp_para(theta_list, bp_filter_cls, M, N, j, L, k0, dn, alpha_list):
    ret = []
    for theta in theta_list:
        for n in range(dn + 1):
            if n == 0:
                ret.append(bp_filter_cls(M, N, j, theta*np.pi/L, k0=k0, L=L, fourier=True).data)
            else:
                for alpha in alpha_list:
                    # Here Tanguy uses 3*n/2 instead of n, why ?
                    ret.append(bp_filter_cls(M, N, j, theta*np.pi/L, k0=k0, L=L, n=3*n, alpha=alpha, fourier=True).data)
    return ret


class WPHOp():
    """
    Wavelet Phase Harmonic (WPH) operator.
    """
    
    def __init__(self, M, N, J, L=8, cplx=False,
                 lp_filter_cls=GaussianFilter, bp_filter_cls=BumpSteerableWavelet,
                 j_min=0, dn=0, alpha_list=[0.0, -np.pi/4, np.pi/4, np.pi/2],
                 precision="single", device="cpu", wph_model="default", sm_model="default"):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height of the input images.
        N : int
            Width of the input images.
        J : int
            Number of dyadic scales.
        L : int, optional
            Number of angles between 0 and pi. The default is 8.
        cplx : bool, optional
            Set it to true if the WPHOp instance will ever apply to complex data. This would load in memory the whole set of bandpass filters. The default is False.
        lp_filter_cls : type, optional
            Class corresponding to the low-pass filter. The default is GaussianFilter.
        bp_filter_cls : type, optional
            Class corresponding to the bandpass filter. The default is BumpSteerableWavelet.
        j_min : type, optional
            Minimum dyadic scale. The default is 0.

        Returns
        -------
        None.

        """
        self.M, self.N, self.J, self.L, self.cplx, self.j_min = M, N, J, L, cplx, j_min
        self.alpha_list, self.dn = alpha_list, dn
        self.precision = precision
        self.device = device
        
        self.load_wph_model(wph_model) # WPH moments model
        self.load_sm_model(sm_model) # Scaling moments model
        
        self.load_filters(lp_filter_cls, bp_filter_cls)
        
    def to(self, device):
        if device != "cpu" and not torch.cuda.is_available():
            print("Warning! CUDA is not available, moving to cpu.")
            self.to("cpu")
        else:
            self.psi_f = self.psi_f.to(device)
            self.phi_f = self.phi_f.to(device)
            
            self.wph_cov_indices = self.wph_cov_indices.to(device)
            self._psi_1_indices = self._psi_1_indices.to(device)
            self._psi_2_indices = self._psi_2_indices.to(device)
    
            self.device = device
        
    def load_filters(self, lp_filter_cls, bp_filter_cls):
        """
        Build the set of low pass and bandpass filters that are used for the transform.

        Parameters
        ----------
        lp_filter_cls : type, optional
            Class corresponding to the low-pass filter.
        bp_filter_cls : type, optional
            Class corresponding to the bandpass filter.
            
        Returns
        -------
        None.

        """
        self.psi_f = [] # Bandpass filters (in Fourier space)
        self.psi_indices = [] # Bandpass filters indices
        self.phi_f = [] # Low-pass filters (in Fourier space)
        self.phi_indices = [] # Low-pass filters indices
        
        # Filter parameters
        k0 = 0.85 * np.pi                              # Central wavenumber of the mother wavelet
        sigma0 = 1 / (0.496 * np.power(2, -0.55) * k0) # Std of the mother scaling function
        
        # Build psi filters
        for j in range(self.j_min, self.J):
            # Parallel pre-build
            build_bp_para_loc = partial(_build_bp_para, bp_filter_cls=bp_filter_cls,
                                        M=self.M, N=self.N, j=j, L=self.L,
                                        k0=k0, dn=self.dn, alpha_list=self.alpha_list)
            nb_processes = os.cpu_count()
            work_list = np.array_split(np.arange(self.L * (1 + self.cplx)), nb_processes)
            pool = mp.Pool(processes=nb_processes) # "Spawn" context for macOS compatibility
            results = pool.map(build_bp_para_loc, work_list)
            bp_filters = []
            for i in range(len(results)):
                bp_filters += results[i]
            pool.close()
            
            # We save the filters in a list
            nb_filters_per_theta = len(self.alpha_list) * self.dn + 1
            for theta in range(self.L * (1 + self.cplx)):
                bp_list = bp_filters[theta * nb_filters_per_theta: (theta + 1) * nb_filters_per_theta]
                index = 0
                for n in range(self.dn + 1):
                    if n == 0:
                        self.psi_f.append(bp_list[index])
                        self.psi_indices.append([j, theta, n, 0])
                        index += 1
                    else:
                        for alpha_id in range(len(self.alpha_list)):
                            self.psi_f.append(bp_list[index])
                            self.psi_indices.append([j, theta, n, alpha_id])
                            index += 1
        
        # Build phi filters
        if self.sm_model is not None:
            for j in range(self.sm_model_params['j_min'], self.sm_model_params['j_max']):
                g = lp_filter_cls(self.M, self.N, j, sigma0=sigma0, fourier=True).data
                self.phi_f.append(g)
                self.phi_indices.append(j)
        
        # Convert indices to numpy arrays
        self.psi_indices = np.array(self.psi_indices)
        self.phi_indices = np.array(self.phi_indices)
        
        # Convert filters to torch tensors
        self.psi_f = to_torch(self.psi_f, device=self.device, precision=self.precision)
        self.phi_f = to_torch(self.phi_f, device=self.device, precision=self.precision)
    
    def load_wph_model(self, wph_model):
        self.wph_model = wph_model
        if wph_model == "default":
            # Default model
            self.wph_cov_indices = build_cov_indices(self.J, self.L, dn=self.dn, len_alpha_list=len(self.alpha_list))
            self._psi_1_indices = []
            self._psi_2_indices = []
            for i in range(self.wph_cov_indices.shape[0]):
                elt = self.wph_cov_indices[i] # j1, t1, k1, j2, t2, k2, n
                n_tau = self.dn * len(self.alpha_list) + 1 # Number of translations per oriented scale
                self._psi_1_indices.append((elt[0] - self.j_min)*(2 * self.L * n_tau) + elt[1]*n_tau)
                self._psi_2_indices.append((elt[3] - self.j_min)*(2 * self.L * n_tau) + elt[4]*n_tau + elt[6])
            self.wph_cov_indices = torch.from_numpy(self.wph_cov_indices).to(self.device)
            self._psi_1_indices = torch.from_numpy(np.array(self._psi_1_indices)).to(self.device)
            self._psi_2_indices = torch.from_numpy(np.array(self._psi_2_indices)).to(self.device)
        else:
            print("Warning! Unknown WPH moments model. Using default model.")
            self.load_wph_model("default")
    
    def load_sm_model(self, sm_model):
        self.sm_model = sm_model
        if sm_model == "default":
            self.sm_model_params = {"j_min": max(self.j_min, 2), "j_max": self.J - 1, "p": [0, 1, 2, 3]}
        elif sm_model is None:
            self.sm_model_params = None
        else:
            print("Warning! Unknown scaling moments model. Using default model.")
            self.load_sm_model("default")
            
    def _prepare_computation(self, data_size, requires_grad=False, mem_chunk_factor=15, mem_chunk_factor_grad=30):
        # Compute the number of chunks needed
        mem_avail = get_memory_available(self.device)
        if requires_grad:
            self.nb_wph_cov_per_chunk = mem_avail // (mem_chunk_factor_grad * data_size)
        else:
            self.nb_wph_cov_per_chunk = mem_avail // (mem_chunk_factor * data_size)
        if self.nb_wph_cov_per_chunk == 0:
            raise Exception("Error! Not enough memory on device.")
        print(get_memory_available(self.device), self.nb_wph_cov_per_chunk)
        
        self._indices_k1_0 = []
        self._indices_k1_1 = []
        self._indices_other_k1 = []
        self._indices_k2_0 = []
        self._indices_k2_1 = []
        self._indices_other_k2 = []
        
        cov_index = 0
        while cov_index < len(self.wph_cov_indices):
            curr_cov_indices = self.wph_cov_indices[cov_index: cov_index + self.nb_wph_cov_per_chunk]
            self._indices_k1_0.append(torch.where(curr_cov_indices[:, 2] == 0)[0])
            self._indices_k1_1.append(torch.where(curr_cov_indices[:, 2] == 1)[0])
            self._indices_other_k1.append(torch.where(curr_cov_indices[:, 2] >= 2)[0])
            self._indices_k2_0.append(torch.where(curr_cov_indices[:, 5] == 0)[0])
            self._indices_k2_1.append(torch.where(curr_cov_indices[:, 5] == 1)[0])
            self._indices_other_k2.append(torch.where(curr_cov_indices[:, 5] >= 2)[0])
            cov_index += self.nb_wph_cov_per_chunk
            
        self.nb_chunks_wph = len(self._indices_k1_0)
        self.nb_chunks_sm = int(self.sm_model is not None) # 1 more chunk if scaling moments needed
        self.nb_chunks = self.nb_chunks_wph + self.nb_chunks_sm
        
        if self.nb_chunks_sm != 0:
            self._indices_p = torch.tensor(self.sm_model_params['p']).to(self.device) # (P)
            
    def preconfigure(self, data, requires_grad=False, mem_chunk_factor=15, mem_chunk_factor_grad=30):
        data = to_torch(data, device=self.device, precision=self.precision)
        data_size = 1
        for elt in data.shape:
            data_size *= elt
        data_size *= 8 if self.precision == "double" else 4
        
        if requires_grad:
            data.requires_grad = True
        
        data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
        data_wt_f = data_f * self.psi_f
        del data_f
        data_wt = ifft(data_wt_f)
        del data_wt_f
        self._tmp_data_wt = data_wt # We keep this variable in memory
        
        self._prepare_computation(data_size, requires_grad=data.requires_grad,
                                  mem_chunk_factor=mem_chunk_factor, mem_chunk_factor_grad=mem_chunk_factor_grad)
        
        return data, self.nb_chunks
    
    def free(self):
        del self._tmp_data_wt
    
    def apply_chunk(self, data, data_wt, chunk_id):
        if chunk_id < self.nb_chunks_wph: # Estimation of the WPH moments
            phase_harmonics = PhaseHarmonics.apply
        
            cov_index = chunk_id * self.nb_wph_cov_per_chunk

            curr_cov_indices = self.wph_cov_indices[cov_index: cov_index + self.nb_wph_cov_per_chunk]
            curr_psi_1_indices = self._psi_1_indices[cov_index: cov_index + self.nb_wph_cov_per_chunk]
            curr_psi_2_indices = self._psi_2_indices[cov_index: cov_index + self.nb_wph_cov_per_chunk]
            curr_indices_k1_0 = self._indices_k1_0[chunk_id]
            curr_indices_k1_1 = self._indices_k1_1[chunk_id]
            curr_indices_other_k1 = self._indices_other_k1[chunk_id]
            curr_indices_k2_0 = self._indices_k2_0[chunk_id]
            curr_indices_k2_1 = self._indices_k2_1[chunk_id]
            curr_indices_other_k2 = self._indices_other_k2[chunk_id]
            
            # data_f = fft(data).unsqueeze(-4) # (..., 1, M, N, 2)
            # data_wt_f_1 = mul(data_f, self.psi_f[curr_psi_1_indices])
            # data_wt_f_2 = mul(data_f, self.psi_f[curr_psi_2_indices])
            # del data_f
            # xpsi1 = ifft(data_wt_f_1)
            # xpsi2 = ifft(data_wt_f_2)

            xpsi1 = data_wt[..., curr_psi_1_indices, :, :]
            xpsi1_k1 = phase_harmonics(xpsi1, curr_cov_indices[:, 2],
                                       curr_indices_k1_0, curr_indices_k1_1, curr_indices_other_k1)
            del xpsi1
            
            xpsi2 = data_wt[..., curr_psi_2_indices, :, :]
            xpsi2_k2 = phase_harmonics(xpsi2, curr_cov_indices[:, 5],
                                       curr_indices_k2_0, curr_indices_k2_1, curr_indices_other_k2)
            del xpsi2
            
            # Take the complex conjugate of xpsi2_k2
            xpsi2_k2 = torch.conj(xpsi2_k2)
            
            # Substract the mean
            xpsi1_k1 -= mean(xpsi1_k1, (-1, -2), keepdim=True)
            xpsi2_k2 -= mean(xpsi2_k2, (-1, -2), keepdim=True)
            
            # Compute correlation and covariance
            corr = xpsi1_k1 * xpsi2_k2
            del xpsi1_k1, xpsi2_k2
            cov = mean(corr, (-1, -2))
            del corr
            return cov
        elif chunk_id == self.nb_chunks_wph and self.nb_chunks_sm != 0: # Estimation of the scaling moments (if needed)
            power_harmonics = PowerHarmonics.apply
            
            # Separate real and imaginary parts of input data if complex data
            if torch.is_complex(data):
                data_new = torch.zeros(data.shape[:-2] + (2,) + data.shape[-2:],
                                       dtype=data.dtype, device=data.device) # (...,2, M, N)
                data_new[..., 0, :, :] = data.real
                data_new[..., 1, :, :] = data.imag
                data_new = data_new.unsqueeze(-3).unsqueeze(-3) # (..., 2, 1, 1, M, N)
            else:
                data_new = data.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3) # (..., 1, 1, 1, M, N)
                
            # Subtract the mean to avoid a bias due to a non-zero mean of the signal
            data_new -= mean(data_new, (-1, -2), keepdim=True)
            
            # Convolution with scaling functions
            data_f = fft(data_new)
            del data
            data_st_f = data_f * self.phi_f.unsqueeze(-3)  # (..., ?, J-3, 1, M, N)
            del data_f
            data_st = ifft(data_st_f)
            del data_st_f
            
            # Compute moments
            data_st_p = power_harmonics(data_st, self._indices_p)
            del data_st
            
            # Substract mean
            data_st_p -= mean(data_st_p, (-1, -2), keepdim=True) # (..., 2, J-3, P, M, N)
            
            # Compute covariance
            data_st_p = torch.abs(data_st_p)
            cov = mean(data_st_p * data_st_p, (-1, -2))
            cov = cov.view(data_st_p.shape[:-5] + (data_st_p.shape[-5]*data_st_p.shape[-4]*data_st_p.shape[-3],)) # (..., ?*(J-3)*P)
            del data_st_p
            return cov
        else:
            raise Exception("Invalid chunk_id!")
        
    def apply(self, data, chunk_id=None, requires_grad=False):
        if chunk_id is None:
            data, nb_chunks = self.preconfigure(data, requires_grad=requires_grad)
        else:
            if not hasattr(self, '_tmp_data_wt'):
                raise Exception("Call preconfigure method first!")
        data_wt = self._tmp_data_wt
        
        if chunk_id is None: # Compute all chunks in one time
            coeffs = []
            for i in range(self.nb_chunks):
                cov = self.apply_chunk(data, data_wt, i)
                coeffs.append(cov)
            coeffs = torch.cat(coeffs, -1)
        else: # Compute selected chunk
            coeffs = self.apply_chunk(data, data_wt, chunk_id)
        
        # We free memory when needed
        if chunk_id is None or chunk_id == self.nb_chunks - 1:
            self.free()
        
        return coeffs
