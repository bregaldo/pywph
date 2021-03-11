# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import multiprocessing as mp
from functools import partial

from .filters import GaussianFilter, BumpSteerableWavelet
from .utils import to_torch, get_memory_available, fft, ifft, phase_harmonics, power_harmonics


# Internal function for the parallel pre-building of the bandpass filters (see WPHOp.load_filters)
def _build_bp_para(theta_list, bp_filter_cls, M, N, j, L, k0, dn, alpha_list):
    ret = []
    for theta in theta_list:
        for n in range(dn + 1):
            if n == 0:
                ret.append(bp_filter_cls(M, N, j, theta*np.pi/L, k0=k0, L=L, fourier=True).data)
            else:
                for alpha in alpha_list:
                    # Consistent with Tanguys'code (3*n instead of n)
                    ret.append(bp_filter_cls(M, N, j, theta*np.pi/L, k0=k0, L=L, n=3*n, alpha=alpha, fourier=True).data)
    return ret


class WPHOp(torch.nn.Module):
    """
    Wavelet Phase Harmonic (WPH) operator.
    """
    
    def __init__(self, M, N, J, L=8, cplx=True,
                 lp_filter_cls=GaussianFilter, bp_filter_cls=BumpSteerableWavelet,
                 j_min=0, dn=0, alpha_list=[-np.pi/4, 0.0, np.pi/4, np.pi/2],
                 precision="single", device="cpu"):
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
            Set it to true if the WPHOp instance will ever apply to complex data.
            This would load in memory the whole set of bandpass filters (not really implemented yet).
            The default is False.
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
        super().__init__()
        self.M, self.N, self.J, self.L, self.cplx, self.j_min = M, N, J, L, cplx, j_min
        self.alpha_list, self.dn = alpha_list, dn
        self.precision = precision
        self.device = device
        
        self.load_model()
        
        self.load_filters(lp_filter_cls, bp_filter_cls)
        
        self.preconfigured = False # Precomputation configuration
        
        # Normalization variables
        self.norm_wph_means = None
        self.norm_wph_stds = None
        self.norm_sm_means_1 = None
        self.norm_sm_means_2 = None
        self.norm_sm_stds = None
        
    def to(self, device):
        """
        Move the data (filters, etc) to the specified device (GPU or CPU).

        Parameters
        ----------
        device : int or str
            "cpu" to move to standard RAM, or an integer to specify a GPU.

        Returns
        -------
        None.

        """
        if device != "cpu" and not torch.cuda.is_available():
            print("Warning! CUDA is not available, moving to cpu.")
            self.to("cpu")
        else:
            self.psi_f = self.psi_f.to(device)
            self.phi_f = self.phi_f.to(device)
            
            self.wph_moments_indices = self.wph_moments_indices.to(device)
            self.scaling_moments_indices = self.scaling_moments_indices.to(device)
            self._psi_1_indices = self._psi_1_indices.to(device)
            self._psi_2_indices = self._psi_2_indices.to(device)
            
            if self.norm_wph_means is not None:
                self.norm_wph_means = self.norm_wph_means.to(device)
            if self.norm_wph_stds is not None:
                self.norm_wph_stds = self.norm_wph_stds.to(device)
            if self.norm_sm_means_1 is not None:
                self.norm_sm_means_1 = self.norm_sm_means_1.to(device)
            if self.norm_sm_means_2 is not None:
                self.norm_sm_means_2 = self.norm_sm_means_2.to(device)
            if self.norm_sm_stds is not None:
                self.norm_sm_stds = self.norm_sm_stds.to(device)
            
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
            work = np.arange(self.L * (1 + self.cplx))
            nb_processes = min(os.cpu_count(), len(work))
            work_list = np.array_split(work, nb_processes)
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
        
        # Build phi filters if needed (automatic check of the model)
        if len(self.scaling_moments_indices) != 0:
            j_min = min(self.scaling_moments_indices[:, 0])
            j_max = max(self.scaling_moments_indices[:, 0])
            for j in range(j_min, j_max + 1):
                g = lp_filter_cls(self.M, self.N, j, sigma0=sigma0, fourier=True).data
                self.phi_f.append(g)
                self.phi_indices.append(j)
        
        # Convert indices to numpy arrays
        self.psi_indices = np.array(self.psi_indices)
        self.phi_indices = np.array(self.phi_indices)
        
        # Convert filters to torch tensors
        self.psi_f = to_torch(self.psi_f, device=self.device, precision=self.precision)
        self.phi_f = to_torch(self.phi_f, device=self.device, precision=self.precision)
        
    def load_model(self, classes=["S11", "S00", "S01", "C01", "Cphase", "L"],
                   extra_wph_moments=[], extra_scaling_moments=[], cross_moments=False):
        """
        Load the specified WPH model. A model is made of WPH moments, and scaling moments.
        The default model includes the following class of moments:
            - S11, S00, S01, C01, Cphase (all WPH moments)
            - L (scaling moments)
        These classes are defined in Allys+2020 and Regaldo-Saint Blancard+2021.
        One can add custom WPH and scaling moments using extra_wph_moments and extra_scaling_moments parameters.
        The expected formats are:
            - for extra_wph_moments: list of lists of 7 elements corresponding to [j1, theta1, p1, j2, theta2, p2, n]
            - for extra_scaling_moments: list of lists of 2 elements correponding to [j, p]

        Parameters
        ----------
        classes : list of str, optional
            Classes of WPH/scaling moments constituting the model. The default is ["S11", "S00", "S01", "C01", "Cphase", "L"].
        extra_wph_moments : list of lists of length 7, optional
            Format corresponds to [j1, theta1, p1, j2, theta2, p2, n]. The default is [].
        extra_scaling_moments : list of lists of length 2, optional
            Format corresponds to [j, p]. The default is [].
        cross_moments : bool, optional
            For cross moments (to be implemented). The default is False.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if cross_moments:
            print("Warning! cross_moments not implemented yet.")
        
        wph_indices = []
        sm_indices = []
        
        # Moments end indices
        self._moments_indices = [0, 0, 0, 0, 0] # End indices delimiting the classes of moments: S11, S00, S01/C01, Cphase/extra, L
    
        cnt = 0
        for clas in classes:
            if clas == "S11":
                for j1 in range(self.j_min, self.J):
                    for t1 in range(2 * self.L):
                        dn_eff = min(self.J - 1 - j1, self.dn)
                        for n in range(dn_eff * len(self.alpha_list) + 1):
                            wph_indices.append([j1, t1, 1, j1, t1, 1, n])
                            cnt += 1
                self._moments_indices[0] = cnt
            elif clas == "S00":
                for j1 in range(self.j_min, self.J):
                    for t1 in range(2 * self.L):
                        dn_eff = min(self.J - 1 - j1, self.dn)
                        for n in range(dn_eff * len(self.alpha_list) + 1):
                            wph_indices.append([j1, t1, 0, j1, t1, 0, n])
                            cnt += 1
                self._moments_indices[1] = cnt
            elif clas == "S01":
                for j1 in range(self.j_min, self.J):
                    for t1 in range(2 * self.L):
                        wph_indices.append([j1, t1, 0, j1, t1, 1, 0])
                        cnt += 1
            elif clas == "C01":
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, self.J):
                        for t1 in range(2 * self.L):
                            for t2 in range(t1 - self.L // 2, t1 + self.L // 2 + 1):
                                if t1 == t2:
                                    dn_eff = min(self.J - 1 - j2, self.dn)
                                    for n in range(dn_eff * len(self.alpha_list) + 1):
                                        wph_indices.append([j1, t1, 0, j2, t2, 1, n])
                                        cnt += 1
                                else:
                                    wph_indices.append([j1, t1, 0, j2, t2 % (2 * self.L), 1, 0])
                                    cnt += 1
                self._moments_indices[2] = cnt
            elif clas == "Cphase":
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, self.J):
                        for t1 in range(2 * self.L):
                            dn_eff = min(self.J - 1 - j2, self.dn)
                            for n in range(dn_eff * len(self.alpha_list) + 1):
                                wph_indices.append([j1, t1, 1, j2, t1, 2 ** (j2 - j1), n])
                                cnt += 1
                self._moments_indices[3] = cnt
            elif clas == "L":
                # Scaling moments
                for j in range(max(self.j_min, 2), self.J - 1):
                    for p in range(4):
                        sm_indices.append([j, p])
                        cnt += 1
                self._moments_indices[4] = cnt
            else:
                raise Exception(f"Unknown class of moments: {clas}")
        
        # Extra moments if provided
        wph_indices += extra_wph_moments
        sm_indices += extra_scaling_moments
        
        # Conversion to numpy arrays
        self.wph_moments_indices = np.array(wph_indices)
        self.scaling_moments_indices = np.array(sm_indices)
        
        # WPH moments preparation
        self._psi_1_indices = []
        self._psi_2_indices = []
        for i in range(self.wph_moments_indices.shape[0]):
            elt = self.wph_moments_indices[i] # [j1, t1, p1, j2, t2, p2, n]
            n_tau = self.dn * len(self.alpha_list) + 1 # Number of translations per oriented scale
            self._psi_1_indices.append((elt[0] - self.j_min)*(2 * self.L * n_tau) + elt[1]*n_tau)
            self._psi_2_indices.append((elt[3] - self.j_min)*(2 * self.L * n_tau) + elt[4]*n_tau + elt[6])
        self.wph_moments_indices = torch.from_numpy(self.wph_moments_indices).to(self.device)
        self._psi_1_indices = torch.from_numpy(np.array(self._psi_1_indices)).to(self.device)
        self._psi_2_indices = torch.from_numpy(np.array(self._psi_2_indices)).to(self.device)
        
        # Scaling moments preparation
        self._phi_indices = []
        for i in range(self.scaling_moments_indices.shape[0]):
            elt = self.scaling_moments_indices[i] # [j, p]
            self._phi_indices.append(elt[0] - max(self.j_min, 2))
        self.scaling_moments_indices = torch.from_numpy(self.scaling_moments_indices).to(self.device)
        self._phi_indices = torch.from_numpy(np.array(self._phi_indices)).to(self.device)
        
        # Useful variables
        self.nb_wph_moments = self.wph_moments_indices.shape[0]
        self.nb_scaling_moments = self.scaling_moments_indices.shape[0]
            
    def _prepare_computation(self, data_size, mem_avail, mem_chunk_factor):
        """
        Internal function.
        """
        # Compute the number of chunks needed
        self.nb_wph_cov_per_chunk = mem_avail // (mem_chunk_factor * data_size)
        if self.nb_wph_cov_per_chunk == 0:
            raise Exception("Error! Not enough memory on device.")
        
        #
        # Divide the WPH moments into chunks
        #
        
        # We save the list of moments per chunk, and the final chunk id per class of moments
        cov_index = 0
        self.wph_moments_chunk_list = []
        self.scaling_moments_chunk_list = []
        self.final_chunk_id_per_class = [] # Final chunk id per class of moments : S11, S00, S01/C01, Cphase/extra
        
        # Build the lists for WPH moments
        i = 0
        chunks_cnt = 0
        while cov_index <= self._moments_indices[i]:
            if cov_index == self._moments_indices[i]:
                self.final_chunk_id_per_class.append(chunks_cnt)
                i += 1 # Next class of moments
                if i == len(self._moments_indices) - 1: # -1 to exclude scaling moments
                    break
            else:
                if cov_index + self.nb_wph_cov_per_chunk <= self._moments_indices[i]:
                    self.wph_moments_chunk_list.append(torch.arange(cov_index, cov_index + self.nb_wph_cov_per_chunk).to(self.device))
                    cov_index += self.nb_wph_cov_per_chunk
                else:
                    self.wph_moments_chunk_list.append(torch.arange(cov_index, self._moments_indices[i]).to(self.device))
                    cov_index += self._moments_indices[i] - cov_index
                chunks_cnt += 1
        
        # Same for scaling moments
        cov_index = 0
        while cov_index <= self._moments_indices[i] - self._moments_indices[3]:
            if cov_index == self._moments_indices[i] - self._moments_indices[3]:
                self.final_chunk_id_per_class.append(chunks_cnt)
                i += 1 # Next class of moments
                if i == len(self._moments_indices):
                    break
            else:
                if cov_index + self.nb_wph_cov_per_chunk <= self._moments_indices[i] - self._moments_indices[3]:
                    self.scaling_moments_chunk_list.append(torch.arange(cov_index, cov_index + self.nb_wph_cov_per_chunk).to(self.device))
                    cov_index += self.nb_wph_cov_per_chunk
                else:
                    self.scaling_moments_chunk_list.append(torch.arange(cov_index, self._moments_indices[i] - self._moments_indices[3]).to(self.device))
                    cov_index += self._moments_indices[i] - self._moments_indices[3] - cov_index
                chunks_cnt += 1
        
        self.nb_chunks_wph = self.final_chunk_id_per_class[-2]
        self.nb_chunks_sm = self.final_chunk_id_per_class[-1] - self.final_chunk_id_per_class[-2]
        self.nb_chunks = self.nb_chunks_wph + self.nb_chunks_sm
        
        print(f"Nb of chunks: {self.nb_chunks}")
        
    def preconfigure(self, data, requires_grad=False,
                     mem_chunk_factor=25, mem_chunk_factor_grad=40):
        """
        Preconfiguration before the WPH computation:
            - cast input data to the relevant torch tensor type
            - enforce requires_grad=True if needed
            - precompute the wavelet transform of input data and its corresponding modulus
              if there is enough available memory
            - divide the computation of the coefficients into chunks of coefficients

        Parameters
        ----------
        data : array or tensor or list of arrays
            Input data.
        requires_grad : bool, optional
            If data.requires_grad is False, turn it to True. The default is False.
        mem_chunk_factor : int, optional
            DESCRIPTION. The default is 20.
        mem_chunk_factor_grad : int, optional
            DESCRIPTION. The default is 35.

        Returns
        -------
        data : tensor
            Torch.tensor version on input data.
        int
            Total number of chunks.

        """
        data = to_torch(data, device=self.device, precision=self.precision)
        data_size = 1
        for elt in data.shape:
            data_size *= elt
        data_size *= 8 if self.precision == "double" else 4
        if requires_grad:
            data.requires_grad = True
        mem_avail = get_memory_available(self.device)
        
        # Precompute the wavelet transform if we have enough memory
        if data_size * self.psi_f.shape[0] < 1/4 * mem_avail:
            print("Enough memory to store the wavelet transform of input data.")
            data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
            data_wt_f = data_f * self.psi_f
            del data_f
            data_wt = ifft(data_wt_f)
            del data_wt_f
            self._tmp_data_wt = data_wt # We keep this variable in memory
            mem_avail -= data_size * self.psi_f.shape[0]
        
        # Precompute the modulus of the wavelet transform if we have enough memory
        if data_size * self.psi_f.shape[0] < 1/4 * mem_avail:
            print("Enough memory to store the modulus of the wavelet transform of input data.")
            self._tmp_data_wt_mod = torch.abs(self._tmp_data_wt) # We keep this variable in memory
            mem_avail -= data_size * self.psi_f.shape[0]
        
        self._prepare_computation(data_size, mem_avail, mem_chunk_factor_grad if data.requires_grad else mem_chunk_factor)
        
        self.preconfigured = True
        
        return data, self.nb_chunks
    
    def free(self):
        """
        Free precomputed data memory.

        Returns
        -------
        None.

        """
        if hasattr(self, '_tmp_data_wt'):
            del self._tmp_data_wt
        if hasattr(self, '_tmp_data_wt_mod'):
            del self._tmp_data_wt_mod
        
        del self.wph_moments_chunk_list, self.scaling_moments_chunk_list
        
        self.preconfigured = False
    
    def _wph_normalization(self, xpsi1_k1, xpsi2_k2, norm, cov_chunk):
        """
        Internal function for the normalization of WPH moments.
        """
        if norm == "auto": # Automatic normalization
            # Compute or retrieve means
            if self.norm_wph_means is None or self.norm_wph_means.shape[-4] != self.nb_wph_moments: # If self.norm_wph_means is not complete
                mean1 = torch.mean(xpsi1_k1.detach(), (-1, -2), keepdim=True)
                mean2 = torch.mean(xpsi2_k2.detach(), (-1, -2), keepdim=True)
                means = torch.stack((mean1, mean2), dim=-1)
                if self.norm_wph_means is None:
                    self.norm_wph_means = means
                else:
                    self.norm_wph_means = torch.cat((self.norm_wph_means, means), dim=-4)
            else:
                mean1 = self.norm_wph_means[..., cov_chunk, :, :, 0] # (..., P, 1, 1)
                mean2 = self.norm_wph_means[..., cov_chunk, :, :, 1] # (..., P, 1, 1)
            
            # Substract the means
            xpsi1_k1 -= mean1.to(xpsi1_k1.dtype) # Consistent dtype needed
            xpsi2_k2 -= mean2.to(xpsi2_k2.dtype) # Consistent dtype needed
            
            # Compute or retrieve (approximate) stds
            if self.norm_wph_stds is None or self.norm_wph_stds.shape[-4] != self.nb_wph_moments:  # If self.norm_wph_stds is not complete
                std1 = torch.sqrt(torch.mean(torch.abs(xpsi1_k1.detach()) ** 2, (-1, -2), keepdim=True))
                std2 = torch.sqrt(torch.mean(torch.abs(xpsi2_k2.detach()) ** 2, (-1, -2), keepdim=True))
                stds = torch.stack((std1, std2), dim=-1)
                if self.norm_wph_stds is None:
                    self.norm_wph_stds = stds
                else:
                    self.norm_wph_stds = torch.cat((self.norm_wph_stds, stds), dim=-4)
            else:
                std1 = self.norm_wph_stds[..., cov_chunk, :, :, 0]
                std2 = self.norm_wph_stds[..., cov_chunk, :, :, 1]
            
            # Divide by the (approximate) std
            xpsi1_k1 /= std1
            xpsi2_k2 /= std2
        else: # No normalization
            # Substract the mean
            xpsi1_k1 -= torch.mean(xpsi1_k1, (-1, -2), keepdim=True)
            xpsi2_k2 -= torch.mean(xpsi2_k2, (-1, -2), keepdim=True)
        return xpsi1_k1, xpsi2_k2
    
    def _sm_normalization_1(self, data_new, norm):
        """
        Internal function for the normalization of the scaling moments.
        """
        if norm == "auto": # Automatic normalization
            # Compute or retrieve means
            if self.norm_sm_means_1 is None:
                mean = torch.mean(data_new.detach(), (-1, -2), keepdim=True)
                self.norm_sm_means_1 = mean
            else:
                mean = self.norm_sm_means_1
            # Substract the mean
            data_new -= mean
        else: # No normalization
            # Substract the mean
            data_new -= torch.mean(data_new, (-1, -2), keepdim=True)
            
        return data_new
    
    def _sm_normalization_2(self, data_st_p, norm, cov_chunk):
        """
        Internal function for the normalization of the scaling moments.
        """
        if norm == "auto": # Automatic normalization
            # Compute or retrieve means
            if self.norm_sm_means_2 is None or self.norm_sm_means_2.shape[-3] != self.nb_scaling_moments: # If self.norm_sm_means_2 is not complete
                mean = torch.mean(data_st_p.detach(), (-1, -2), keepdim=True)
                if self.norm_sm_means_2 is None:
                    self.norm_sm_means_2 = mean
                else:
                    self.norm_sm_means_2 = torch.cat((self.norm_sm_means_2, mean), dim=-3)
            else:
                mean = self.norm_sm_means_2[..., cov_chunk, :, :]
                
            # Substract the mean
            data_st_p -= mean
            
            # Compute or retrieve (approximate) stds
            if self.norm_sm_stds is None or self.norm_sm_stds.shape[-3] != self.nb_scaling_moments: # If self.norm_sm_stds is not complete
                std = torch.sqrt(torch.mean(torch.abs(data_st_p.detach()) ** 2, (-1, -2), keepdim=True))
                if self.norm_sm_stds is None:
                    self.norm_sm_stds = std
                else:
                    self.norm_sm_stds = torch.cat((self.norm_sm_stds, std), dim=-3)
            else:
                std = self.norm_sm_stds[..., cov_chunk, :, :]
            
            # Divide by the (approximate) std
            data_st_p /= std
        else: # No normalization
            # Substract the mean
            data_st_p -= torch.mean(data_st_p, (-1, -2), keepdim=True) # (..., ?, P, M, N)
            
        return data_st_p
    
    def clear_normalization(self):
        """
        Clear saved data for normalization of the coefficients.

        Returns
        -------
        None.

        """
        self.norm_wph_means = None
        self.norm_wph_stds = None
        self.norm_sm_means = None
        self.norm_sm_stds = None
    
    def _apply_chunk(self, data, chunk_id, norm, ret_indices):
        """
        Internal function. Use apply instead.
        """
        if chunk_id < self.nb_chunks_wph: # WPH moments:
            cov_chunk = self.wph_moments_chunk_list[chunk_id]
            cov_indices = self.wph_moments_indices[cov_chunk]
            
            curr_psi_1_indices = self._psi_1_indices[cov_chunk]
            curr_psi_2_indices = self._psi_2_indices[cov_chunk]
            
            def get_precomputed_data(psi_indices, p):
                """
                Internal function that load in memory relevant part of precomputed data (wavelet transform, or its modulus).
                If the data was not precomputed, we first compute it.
                """
                if p == 0: # We need the modulus
                    if hasattr(self, '_tmp_data_wt_mod'):
                        data_wt_mod = self._tmp_data_wt_mod
                        xpsi = torch.index_select(data_wt_mod, -3, psi_indices)
                    else:
                        if hasattr(self, '_tmp_data_wt'):
                            data_wt = self._tmp_data_wt
                            xpsi = torch.index_select(data_wt, -3, psi_indices)
                        else:
                            data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
                            data_wt_f = data_f * self.psi_f[psi_indices]
                            xpsi = ifft(data_wt_f)
                            del data_f, data_wt_f
                    return torch.abs(xpsi)
                elif p == 1: # We just need the wavelet transform
                    if hasattr(self, '_tmp_data_wt'):
                        data_wt = self._tmp_data_wt
                        xpsi = torch.index_select(data_wt, -3, psi_indices)
                    else:
                        data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
                        data_wt_f = data_f * self.psi_f[psi_indices]
                        xpsi = ifft(data_wt_f)
                        del data_f, data_wt_f
                    return xpsi
                else:
                    raise Exception(f"Invalid p value: {p}!")
            
            if chunk_id < self.final_chunk_id_per_class[0]: # S11 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 1)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 1)
            elif chunk_id < self.final_chunk_id_per_class[1]: # S00 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 0)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 0)
            elif chunk_id < self.final_chunk_id_per_class[2]: # S01/C01 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 0)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 1)
            elif chunk_id < self.final_chunk_id_per_class[3]: # Other moments
                # Get the relevant part of the wavelet transform and compute the corresponding phase harmonics
                xpsi1 = get_precomputed_data(curr_psi_1_indices, 1)
                xpsi1_k1 = phase_harmonics(xpsi1, cov_indices[:, 2])
                del xpsi1
                xpsi2 = get_precomputed_data(curr_psi_2_indices, 1)
                xpsi2_k2 = phase_harmonics(xpsi2, cov_indices[:, 5])
                del xpsi2
                
            # Take the complex conjugate of xpsi2_k2
            xpsi2_k2 = torch.conj(xpsi2_k2)
            
            # Normalization
            xpsi1_k1, xpsi2_k2 = self._wph_normalization(xpsi1_k1, xpsi2_k2, norm, cov_chunk)
            
            # Compute covariance
            cov = torch.mean(xpsi1_k1 * xpsi2_k2, (-1, -2))
            del xpsi1_k1, xpsi2_k2
            
            if ret_indices:
                return cov, cov_chunk
            else:
                return cov
        elif chunk_id < self.final_chunk_id_per_class[4]: # Scaling moments
            cov_chunk = self.scaling_moments_chunk_list[chunk_id - self.final_chunk_id_per_class[3]]
            cov_indices = self.scaling_moments_indices[cov_chunk]
            
            curr_phi_indices = self._phi_indices[cov_chunk]
        
            # Separate real and imaginary parts of input data if complex data
            if torch.is_complex(data):
                cplx = True
                data_new = torch.zeros(data.shape[:-2] + (2,) + data.shape[-2:],
                                       dtype=data.dtype, device=data.device) # (...,2, M, N)
                data_new[..., 0, :, :] = data.real
                data_new[..., 1, :, :] = data.imag
                data_new = data_new.unsqueeze(-3) # (..., 2, 1, M, N)
            else:
                cplx = False
                data_new = data.clone().unsqueeze(-3).unsqueeze(-3) # (..., 1, 1, M, N)
                
            # Subtract the mean to avoid a bias due to a non-zero mean of the signal
            data_new = self._sm_normalization_1(data_new, norm)
            
            # Convolution with scaling functions
            data_f = fft(data_new)
            data_st_f = data_f * self.phi_f[curr_phi_indices]  # (..., ?, P, M, N)
            del data_new, data_f
            data_st = ifft(data_st_f)
            del data_st_f
            
            # Compute moments
            data_st_p = power_harmonics(data_st, cov_indices[:, 1])
            del data_st
            
            # Normalization
            data_st_p = self._sm_normalization_2(data_st_p, norm, cov_chunk)
            
            # Compute covariance
            data_st_p = torch.abs(data_st_p)
            cov = torch.mean(data_st_p * data_st_p, (-1, -2))
            cov = cov.view(cov.shape[:-2] + (cov.shape[-2]*cov.shape[-1],)) # (..., ?*P)
            del data_st_p
            
            if ret_indices:
                # For scaling moments, shift cov_chunk by the number of WPH moments
                # There are subtleties related to the doubling of coefficients when applying to complex data
                cov_chunk_shifted_1 = cov_chunk[0]*(1 + cplx) + self.nb_wph_moments + cov_chunk
                cov_chunk_shifted_2 = cov_chunk_shifted_1 + len(cov_chunk)
                cov_chunk_shifted = torch.cat([cov_chunk_shifted_1, cov_chunk_shifted_2])
                return cov, cov_chunk_shifted
            else:
                return cov
        else: # Invalid
            raise Exception("Invalid chunk_id!")
        
    def apply(self, data, chunk_id=None, requires_grad=False, norm=None, ret_indices=False):
        """
        Compute the WPH statistics of input data.
        There are two modes of use:
            1) with chunk_id=None: Compute and return the whole set of coefficients.
               If this result in an "out of memory" error, use mode 2.
            2) with chunk_id=i, where i is the required chunk id. This computes and returns the selected
               set of coefficients.

        Parameters
        ----------
        data : array or tensor or list of arrays
            Input data.
        chunk_id : int, optional
            Id of the chunk of WPH coefficients. The default is None.
        requires_grad : bool, optional
            If data.requires_grad is False, turn it to True. The default is False.
        ret_indices : bool, optional
            When computing a specific chunk of coefficients, return the corresponding array 
            of indices of the coefficients. The default is False.

        Returns
        -------
        coeffs : tensor
            WPH coefficients.

        """
        if chunk_id is None: # Compute all chunks at once
            data, nb_chunks = self.preconfigure(data, requires_grad=requires_grad)
            coeffs = []
            for i in range(self.nb_chunks):
                cov = self._apply_chunk(data, i, norm)
                coeffs.append(cov)
            coeffs = torch.cat(coeffs, -1)
            if ret_indices:
                indices = torch.arange(coeffs.shape[-1])
        else: # Compute selected chunk
            if not self.preconfigured:
                raise Exception("First preconfigure data!")
            ret = self._apply_chunk(data, chunk_id, norm, ret_indices)
            if ret_indices:
                coeffs, indices = ret
            else:
                coeffs = ret
        
        # We free memory when needed
        if chunk_id is None or chunk_id == self.nb_chunks - 1:
            self.free()
        
        if ret_indices:
            return coeffs, indices
        else:
            return coeffs
    
    def forward(self, data, chunk_id=None, requires_grad=False, norm=None, ret_indices=False):
        """
        Alias of apply.
        """
        self.apply(data, chunk_id=chunk_id, requires_grad=requires_grad, norm=norm, ret_indices=False)
