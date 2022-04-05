# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import multiprocessing as mp
from itertools import chain
from itertools import product
from functools import partial

from .filters import BumpIsotropicWavelet, GaussianFilter, BumpSteerableWavelet
from .utils import to_torch, get_memory_available, fft, ifft, phase_harmonics, power_harmonics
from .wph import WPH


# List of filters that should be contained in complex-valued arrays (in Fourier space)
complex_filters = []

# Internal function for the parallel pre-building of the bandpass filters (see WPHOp.load_filters)
def _build_bp_para(work_list, bp_filter_cls, M, N, L, k0):
    ret = np.zeros((work_list.shape[0], M, N), dtype=complex if bp_filter_cls in complex_filters else float)
    for i in range(work_list.shape[0]):
            ret[i] = bp_filter_cls(M, N, float(work_list[i, 0]), work_list[i, 1]*np.pi/L, k0=k0, L=L, fourier=True).data
    return ret


class WPHOp(torch.nn.Module):
    """
    Wavelet Phase Harmonic (WPH) operator.
    """
    
    def __init__(self, M, N, J, L=8, cplx=False,
                 lp_filter_cls=GaussianFilter, bp_filter_cls=BumpSteerableWavelet,
                 j_min=0, dn=0, A=4,
                 cut_x_param=1/2, cut_y_param=1/2,
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
            This would load in memory the whole set of bandpass filters.
            The default is False.
        lp_filter_cls : class, optional
            Class corresponding to the low-pass filter. The default is GaussianFilter.
        bp_filter_cls : class, optional
            Class corresponding to the bandpass filter. The default is BumpSteerableWavelet.
        j_min : int, optional
            Minimum dyadic scale. The default is 0.
        dn : int, optional
            \Delta_n parameter. Number of radial steps for the discretization of the \tau variable. The default is 0.
        A : int, optional
            Number of azimuthal steps on [0, \pi[ for the discretization of the \alpha variable.  The default is 4.
            Default value leads to the following set of values for \alpha: {0, \pi/4, \pi/2, 3\pi/4, \pi, -3\pi/4, -\pi/2, -\pi/4}.
        cut_x_param, cut_y_param : float, optional
            Cutting parameters when dealing with data with non periodic boundary conditions. Unit is a factor of 2^J pixels.
            The default is 1/2, conrresponding to a padding of 2^(J-1).
        precision : str, optional
            Float precision of torch tensors. Can be either "single" or "double". The default is "single".
        device : str or int, optional
            Torch default device (cpu or gpu). The default is "cpu".

        Returns
        -------
        None.

        """
        super().__init__()
        self.M, self.N, self.J, self.L, self.cplx, self.j_min = M, N, J, L, cplx, j_min
        self.dn, self.A = dn, A
        self.precision = precision
        self.device = device

        # Cutting parameters (in 2^J unit)
        self.cut_x_param = cut_x_param
        self.cut_y_param = cut_y_param

        # Build a weight map to weight the coefficients in order to take into account the non-periodic boundary conditions
        cut_x = int(self.cut_x_param * 2 ** self.J)
        cut_y = int(self.cut_y_param * 2 ** self.J)
        M_padded = 2*(self.M - 2*cut_y)
        N_padded = 2*(self.N - 2*cut_x)
        pad_data = np.zeros((M_padded, N_padded))
        pad_data[:M_padded // 2,:N_padded // 2] = 1
        self._translation_weight_map = (np.fft.ifft2(np.fft.fft2(pad_data) * np.conjugate(np.fft.fft2(pad_data))) / (self.M * self.N)).real
        
        # Load default model and build filters
        self.load_model()
        self.lp_filter_cls = lp_filter_cls
        self.bp_filter_cls = bp_filter_cls
        self.load_filters()
        
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
        self

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
            self._harmo_indices = self._harmo_indices.to(device)
            self._pseudo_indices = self._pseudo_indices.to(device)
            self._id_cov_indices = self._id_cov_indices.to(device)
            self._translation_pos = self._translation_pos.to(device)
            self._translation_weight = self._translation_weight.to(device)
            self._phi_indices = self._phi_indices.to(device)
            
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
        return self
        
    def load_filters(self):
        """
        Build the set of low pass and bandpass filters that are used for the transform.

        Parameters
        ----------
        None.
            
        Returns
        -------
        None.

        """
        ### BUILD PSI FILTERS
        nb_psi = (self.J - self.j_min) * (self.L * (1 + self.cplx))
        self.psi_f = np.zeros((nb_psi, self.M, self.N), dtype=complex if self.bp_filter_cls in complex_filters else float) # Bandpass filters (in Fourier space)
        self.psi_indices = np.array(list(product(range(self.j_min, self.J),
                                    range(self.L * (1 + self.cplx)))), dtype=int) # Bandpass filters indices
        
        # Filter parameters
        k0 = 0.85 * np.pi                              # Central wavenumber of the mother wavelet
        sigma0 = 1 / (0.496 * np.power(2, -0.55) * k0) # Std of the mother scaling function

        # Parallel computation of psi filters
        build_bp_para_loc = partial(_build_bp_para, bp_filter_cls=self.bp_filter_cls,
                                    M=self.M, N=self.N, L=self.L, k0=k0)
        nb_processes = min(os.cpu_count(), self.psi_indices.shape[0])
        work_list = np.array_split(self.psi_indices, nb_processes)
        pool = mp.Pool(processes=nb_processes)
        results = pool.map(build_bp_para_loc, work_list)
        cnt = 0
        for i in range(len(results)):
            self.psi_f[cnt: cnt + results[i].shape[0]] = results[i]
            cnt += results[i].shape[0]
        pool.close()
        
        ### BUILD PHI FILTERS

        if len(self.scaling_moments_indices) != 0:
            j_min = min(self.scaling_moments_indices[:, 0])
            j_max = max(self.scaling_moments_indices[:, 0])
            nb_phi = j_max - j_min + 1
        else:
            nb_phi = 0 # No phi filter needed

        self.phi_f = np.zeros((nb_phi, self.M, self.N), dtype=complex if self.lp_filter_cls in complex_filters else float) # Low-pass filters (in Fourier space)
        self.phi_indices = np.zeros(nb_phi, dtype=int) # Low-pass filters indices

        # Build phi filters if needed (automatic check of the model)
        if len(self.scaling_moments_indices) != 0:
            for ind, j in enumerate(range(j_min, j_max + 1)):
                g = self.lp_filter_cls(self.M, self.N, j, sigma0=sigma0, fourier=True).data
                self.phi_f[ind] = g
                self.phi_indices[ind] = j

        
        # Convert filters to torch tensors
        self.psi_f = to_torch(self.psi_f, device=self.device, precision=self.precision)
        self.phi_f = to_torch(self.phi_f, device=self.device, precision=self.precision)

    def _get_translations_params(self, j, t, n, a):
        """
        Internal function used for the discretization of the tau variable in the definition of the WPH moments.

        tau variable is discretized thanks to the polar coordinates (n, a) such that
            tau_x = 3n*2^j*cos(theta*pi/L - alpha*pi/A)
            tau_y = 3n*2^j*sin(theta*pi/L - alpha*pi/A)

        Parameters
        ----------
        j : int
            j index.
        t : int
            t index.
        n : int
            n index.
        a : int
            a index.

        Returns
        -------
        list of two ints
            [tau_x, tau_y] coordinates
        float
            Weighting factor used when pbc=False for the correct normalization of the WPH coefficients.
        """
        r_factor = 3 # Radial step size factor for translations
        theta = t * np.pi/self.L
        alpha = a * np.pi/self.A
        tx = np.rint(r_factor * n * 2**j * np.cos(theta - alpha)).astype(int) # Round to the nearest integer
        ty = np.rint(r_factor * n * 2**j * np.sin(theta - alpha)).astype(int) # Round to the nearest integer
        return [tx, ty], 1 / self._translation_weight_map[ty, tx]
        
    def load_model(self, classes=["S11", "S00", "S01", "C01", "Cphase", "L"],
                   extra_wph_moments=[], extra_scaling_moments=[], cross_moments=False,
                   p_list=[0, 1, 2, 3], dj=None, dl=None, dn=None, A=None):
        """
        Load the specified WPH model. A model is made of WPH moments, and scaling moments.
        The default model includes the following class of moments:
            - S11, S00, S01, C01, Cphase (all WPH moments)
            - L (scaling moments)
        These classes are defined in Allys+2020 and Regaldo-Saint Blancard+2021.
        One can add custom WPH and scaling moments using extra_wph_moments and extra_scaling_moments parameters.
        The expected formats are:
            - for extra_wph_moments: list of lists of 9 elements corresponding to [j1, theta1, p1, j2, theta2, p2, n, alpha, pseudo]
            - for extra_scaling_moments: list of lists of 2 elements correponding to [j, p]

        Parameters
        ----------
        classes : str or list of str, optional
            Classes of WPH/scaling moments constituting the model. Possibilities are: "S11", "S00", "C00", "S01", "S10", "C01", "C10", "Cphase", "Cphase_inv", "L".
            The default is ["S11", "S00", "S01", "C01", "Cphase", "L"].
        extra_wph_moments : list of lists of length 9, optional
            Format corresponds to [j1, theta1, p1, j2, theta2, p2, n, alpha, pseudo]. The default is [].
        extra_scaling_moments : list of lists of length 2, optional
            Format corresponds to [j, p]. The default is [].
        cross_moments : bool, optional
            Adapt the model for the computation of cross statistics. This ensures that cross statistics of (x, y) remain the same as those of (y, x), as well as preventing redundancy in the coefficients. The default is False.
        p_list : list of int, optional
            For scaling moments ("L"), list of moments to compute for each low-pass filter.
        dj : int, optional
            Maximum delta j for coefficients quantifying interactions between scales. The default is None, corresponding to J - j_min - 1.
        dl : int, optional
            Maximum delta theta for coefficients quantifying interactions between scales. The default is None, corresponding to L/2.
        dn : int, optional
            \Delta_n parameter. Number of radial steps for the discretization of the \tau variable. The default is None, leaving previous dn parameter unchanged.
        A : int, optional
            Number of azimuthal steps on [0, \pi[ for the discretization of the \alpha variable. The default is None, leaving previous A parameter unchanged.
            
        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Reordering of elements of classes
        if isinstance(classes, str): # Convert to list of str
            classes = [classes]
        classes_new = []
        for clas in ["S11", "S00", "C00", "S01", "C01", "S10", "C10", "Cphase", "Cphase_inv", "L"]:
            if clas in classes:
                classes_new.append(clas)
        classes = classes_new
        
        # Symmetrize if cross_moments is True
        if cross_moments:
            cont = True
            if "S01" in classes and "S10" not in classes:
                classes += ["S10"]
                cont = False
            if "S10" in classes and "S01" not in classes:
                classes += ["S01"]
                cont = False
            if "C01" in classes and "C10" not in classes:
                classes += ["C10"]
                cont = False
            if "C10" in classes and "C01" not in classes:
                classes += ["C01"]
                cont = False
            if "Cphase" in classes and "Cphase_inv" not in classes:
                classes += ["Cphase_inv"]
                cont = False
            if "Cphase_inv" in classes and "Cphase" not in classes:
                classes += ["Cphase"]
                cont = False
            if not cont: # Reload symmetrized model
                self.load_model(classes, extra_wph_moments=extra_wph_moments,
                                extra_scaling_moments=extra_scaling_moments, cross_moments=cross_moments,
                                p_list=p_list, dj=dj, dl=dl, dn=dn, A=A)
                return
        
        wph_indices = [] # [j1, theta1, p1, j2, theta2, p2, n, alpha, pseudo]
        sm_indices = [] # [j, p]
        
        # Default values for dj, dl, dn, alpha_list
        if dj is None:
            dj = self.J - self.j_min - 1 # We consider all possible pair of scales j1 < j2
        if dl is None:
            dl = self.L // 2 # For C01 moments, we consider |t1 - t2| <= pi / 2
        if dn is None:
            dn = self.dn
        else:
            self.dn = dn
        if A is None:
            A = self.A
        else:
            self.A = A
        
        # Moments and indices
        self._moments_indices = np.array([0, 0, 0, 0, 0, 0]) # End indices delimiting the classes of covariances: S11, S00/C00, S01/C01, S10/C10, Cphase/extra, L
    
        for clas in classes:
            cnt = 0
            if clas == "S11":
                for j1 in range(self.j_min, self.J):
                    for t1 in range((1 + self.cplx) * self.L):
                        dn_eff = min(self.J - 1 - j1, dn)
                        for n in range(dn_eff + 1):
                            if n == 0:
                                wph_indices.append([j1, t1, 1, j1, t1, 1, n, 0, 0])
                            else:
                                if not cross_moments:
                                    a_range = range(self.A) # Half of alpha angles is enough even for complex data
                                else:
                                    a_range = range(2 * self.A) # Factor 2 for cross-moments symmetry
                                for a in a_range:
                                    wph_indices.append([j1, t1, 1, j1, t1, 1, n, a, 0])
                        cnt += 1
                    if self.cplx: # Pseudo S11 moments
                        if not cross_moments:
                            t1_range = range(self.L) # Only L because of the pi-periodicity of these moments
                        else:
                            t1_range = range(2 * self.L) # Factor 2 for cross-moments symmetry
                        for t1 in t1_range:
                            dn_eff = min(self.J - 1 - j1, dn)
                            for n in range(dn_eff + 1):
                                if n == 0:
                                    wph_indices.append([j1, t1, 1, j1, (t1 + self.L)  % (2*self.L), 1, n, 0, 1])
                                else:
                                    for a in range(2*self.A): # Here we need the full set of alpha angles
                                        wph_indices.append([j1, t1, 1, j1, (t1 + self.L)  % (2*self.L), 1, n, (a + self.A) % (2 * self.A), 1])
                            cnt += 1
                self._moments_indices[0:] += cnt
            elif clas == "S00":
                for j1 in range(self.j_min, self.J):
                    for t1 in range((1 + self.cplx) * self.L):
                        dn_eff = min(self.J - 1 - j1, dn)
                        for n in range(dn_eff + 1):
                            if n == 0:
                                wph_indices.append([j1, t1, 0, j1, t1, 0, n, 0, 0])
                            else:
                                if not cross_moments:
                                    a_range = range(self.A) # Half of alpha angles is enough even for complex data
                                else:
                                    a_range = range(2 * self.A) # Factor 2 for cross-moments symmetry
                                for a in a_range:
                                    wph_indices.append([j1, t1, 0, j1, t1, 0, n, a, 0])
                        cnt += 1
                self._moments_indices[1:] += cnt
            elif clas == "C00": # Non default moments
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, self.J)):
                        for t1 in range((1 + self.cplx) * self.L):
                            if self.cplx:
                                t2_range = chain(range(t1 - dl, t1 + dl), range(t1 + self.L - dl, t1 + self.L + dl))
                            else:
                                t2_range = range(t1 - dl, t1 + dl)
                            for t2 in t2_range:
                                # No translation here by default
                                wph_indices.append([j1, t1, 0, j2, t2 % ((1 + self.cplx) * self.L), 0, 0, 0, 0])
                                cnt += 1
                self._moments_indices[1:] += cnt
            elif clas == "S01":
                for j1 in range(self.j_min, self.J):
                    for t1 in range((1 + self.cplx) * self.L):
                        wph_indices.append([j1, t1, 0, j1, t1, 1, 0, 0, 0])
                        cnt += 1
                self._moments_indices[2:] += cnt
            elif clas == "C01":
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, self.J)):
                        for t1 in range((1 + self.cplx) * self.L):
                            if self.cplx:
                                t2_range = chain(range(t1 - dl, t1 + dl), range(t1 + self.L - dl, t1 + self.L + dl))
                            else:
                                t2_range = range(t1 - dl, t1 + dl)
                            for t2 in t2_range:
                                if t1 == t2:
                                    dn_eff = min(self.J - 1 - j2, dn)
                                    for n in range(dn_eff + 1):
                                        if n == 0:
                                            wph_indices.append([j1, t1, 0, j2, t2, 1, n, 0, 0])
                                        else:
                                            for a in range(2 * self.A): # Factor 2 needed even for real data
                                                wph_indices.append([j1, t1, 0, j2, t2, 1, n, a, 0])
                                else:
                                    wph_indices.append([j1, t1, 0, j2, t2 % ((1 + self.cplx) * self.L), 1, 0, 0, 0])
                                cnt += 1
                self._moments_indices[2:] += cnt
            elif clas == "S10":
                for j1 in range(self.j_min, self.J):
                    for t1 in range((1 + self.cplx) * self.L):
                        wph_indices.append([j1, t1, 1, j1, t1, 0, 0, 0, 0])
                        cnt += 1
                self._moments_indices[3:] += cnt
            elif clas == "C10":
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, self.J)):
                        for t1 in range((1 + self.cplx) * self.L):
                            if self.cplx:
                                t2_range = chain(range(t1 - dl, t1 + dl), range(t1 + self.L - dl, t1 + self.L + dl))
                            else:
                                t2_range = range(t1 - dl, t1 + dl)
                            for t2 in t2_range:
                                if t1 == t2:
                                    dn_eff = min(self.J - 1 - j2, dn)
                                    for n in range(dn_eff + 1):
                                        if n == 0:
                                            wph_indices.append([j2, t2, 1, j1, t1, 0, 2 ** (j2 - j1) * n, 0, 0])
                                        else:
                                            for a in range(2 * self.A): # Factor 2 needed even for real data
                                                wph_indices.append([j2, t2, 1, j1, t1, 0, 2 ** (j2 - j1) * n, a, 0])
                                else:
                                    wph_indices.append([j2, t2 % ((1 + self.cplx) * self.L), 1, j1, t1, 0, 0, 0, 0])
                                cnt += 1
                self._moments_indices[3:] += cnt
            elif clas == "Cphase":
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, self.J)):
                        for t1 in range((1 + self.cplx) * self.L):
                            dn_eff = min(self.J - 1 - j2, dn)
                            for n in range(dn_eff + 1):
                                if n == 0:
                                    wph_indices.append([j1, t1, 1, j2, t1, 2 ** (j2 - j1), n, 0, 0])
                                else:
                                    for a in range(2 * self.A): # Factor 2 needed even for real data
                                        wph_indices.append([j1, t1, 1, j2, t1, 2 ** (j2 - j1), n, a, 0])
                            cnt += 1
                self._moments_indices[4:] += cnt
            elif clas == "Cphase_inv":
                for j1 in range(self.j_min, self.J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, self.J)):
                        for t1 in range((1 + self.cplx) * self.L):
                            dn_eff = min(self.J - 1 - j2, dn)
                            for n in range(dn_eff + 1):
                                if n == 0:
                                    # Should be 2**(j2 - j1)*n instead of n, but not possible yet..
                                    wph_indices.append([j2, t1, 2 ** (j2 - j1), j1, t1, 1, 2 ** (j2 - j1) * n, 0, 0])
                                else:
                                    for a in range(2 * self.A): # Factor 2 needed even for real data
                                        wph_indices.append([j2, t1, 2 ** (j2 - j1), j1, t1, 1, 2 ** (j2 - j1) * n, a, 0])
                            cnt += 1
                self._moments_indices[4:] += cnt
            elif clas == "L":
                # Scaling moments
                for j in range(max(self.j_min, 2), self.J - 1):
                    for p in p_list:
                        sm_indices.append([j, p])
                        cnt += 1
                self._moments_indices[5:] += cnt
            else:
                raise Exception(f"Unknown class of moments: {clas}")
        
        # Extra moments if provided
        wph_indices += extra_wph_moments
        sm_indices += extra_scaling_moments
        #self._moments_indices[4:] += len(extra_wph_moments) -> needs to be taken care below
        self._moments_indices[5:] += len(extra_scaling_moments)

        # Conversion to numpy arrays
        self.wph_moments_indices = np.array(wph_indices)
        self.scaling_moments_indices = np.array(sm_indices)
        
        # WPH moments preparation
        # nb_cov: the number of covariances, that is the number of distinct values (j1, t1, p1, j2, t2, p2)
        # nb_wph: the number of WPH coefficients, that is the number of distinct values (j1, t1, p1, j2, t2, p2, n, a, pseudo)
        self._psi_1_indices = [] # (nb_cov) : id associated to (j1, t1)
        self._psi_2_indices = [] # (nb_cov) : id associated to (j2, t2)
        self._harmo_indices = [] # (nb_cov, 2) : (p1, p2)
        self._pseudo_indices = [] # (nb_cov)
        self._id_cov_indices = [] # (nb_wph) : mapping of nb_wph in nb_cov
        self._translation_pos = [] # (nb_wph, 2) : (tx, ty) position of translations
        self._translation_weight = [] # (nb_wph) : weight for each translation (for padding purposes)
        self.max_tx = 0 # For padding
        self.max_ty = 0 # For padding
        id_cov = -1
        curr_cov = []
        for i in range(self.wph_moments_indices.shape[0]):
            elt = list(self.wph_moments_indices[i]) # [j1, theta1, p1, j2, theta2, p2, n, alpha, pseudo]
            if elt[:6] + elt[8:] != curr_cov:
                curr_cov = elt[:6] + elt[8:]
                id_cov += 1
                self._psi_1_indices.append((elt[0] - self.j_min)*((1 + self.cplx) * self.L) + elt[1])
                self._psi_2_indices.append((elt[3] - self.j_min)*((1 + self.cplx) * self.L) + elt[4])
                self._harmo_indices.append([elt[2], elt[5]])
                self._pseudo_indices.append(elt[8])
                if i >= self.wph_moments_indices.shape[0] - len(extra_wph_moments): # _moments_indices for extra_wph_moments is incremented for each new value of (j1, t1, p1, j2, t2, p2)
                    self._moments_indices[4:] += 1
            self._id_cov_indices.append(id_cov)
            trans_pos, trans_weight = self._get_translations_params(elt[3], elt[4], elt[6], elt[7])
            self.max_tx = max(abs(trans_pos[0]), self.max_tx)
            self.max_ty = max(abs(trans_pos[1]), self.max_ty)
            self._translation_pos.append(trans_pos)
            self._translation_weight.append(trans_weight)
        
        self.wph_moments_indices = torch.from_numpy(self.wph_moments_indices).to(self.device)
        self._psi_1_indices = torch.from_numpy(np.array(self._psi_1_indices)).to(self.device, torch.long)
        self._psi_2_indices = torch.from_numpy(np.array(self._psi_2_indices)).to(self.device, torch.long)
        self._harmo_indices = torch.from_numpy(np.array(self._harmo_indices)).to(self.device, torch.long)
        self._pseudo_indices = torch.from_numpy(np.array(self._pseudo_indices)).to(self.device, torch.long)
        self._id_cov_indices = torch.from_numpy(np.array(self._id_cov_indices)).to(self.device, torch.long)
        self._translation_pos = torch.from_numpy(np.array(self._translation_pos)).to(self.device, torch.long)
        self._translation_weight = torch.from_numpy(np.array(self._translation_weight)).to(self.device, torch.float)
        
        # Scaling moments preparation
        self._phi_indices = []
        for i in range(self.scaling_moments_indices.shape[0]):
            elt = self.scaling_moments_indices[i] # [j, p]
            self._phi_indices.append(elt[0] - max(self.j_min, 2))
        self.scaling_moments_indices = torch.from_numpy(self.scaling_moments_indices).to(self.device)
        self._phi_indices = torch.from_numpy(np.array(self._phi_indices)).to(self.device, torch.long)
        
        # Useful variables
        self.nb_wph_cov = self._psi_1_indices.shape[0]
        self.nb_wph_moments = self.wph_moments_indices.shape[0]
        self.nb_scaling_moments = self.scaling_moments_indices.shape[0]
            
    def _prepare_computation(self, data_size, mem_avail, mem_chunk_factor, nb_wph_cov_per_chunk=None):
        """
        Internal function.
        """
        if nb_wph_cov_per_chunk is None:
            # Compute the number of chunks needed
            self.nb_wph_cov_per_chunk = mem_avail // (mem_chunk_factor * data_size)
        else:
            self.nb_wph_cov_per_chunk = nb_wph_cov_per_chunk
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
        while cov_index <= self._moments_indices[i] - self._moments_indices[-2]:
            if cov_index == self._moments_indices[i] - self._moments_indices[-2]:
                self.final_chunk_id_per_class.append(chunks_cnt)
                i += 1 # Next class of moments
                if i == len(self._moments_indices):
                    break
            else:
                if cov_index + self.nb_wph_cov_per_chunk <= self._moments_indices[i] - self._moments_indices[-2]:
                    self.scaling_moments_chunk_list.append(torch.arange(cov_index, cov_index + self.nb_wph_cov_per_chunk).to(self.device))
                    cov_index += self.nb_wph_cov_per_chunk
                else:
                    self.scaling_moments_chunk_list.append(torch.arange(cov_index, self._moments_indices[i] - self._moments_indices[-2]).to(self.device))
                    cov_index += self._moments_indices[i] - self._moments_indices[-2] - cov_index
                chunks_cnt += 1
        
        self.nb_chunks_wph = self.final_chunk_id_per_class[-2]
        self.nb_chunks_sm = self.final_chunk_id_per_class[-1] - self.final_chunk_id_per_class[-2]
        self.nb_chunks = self.nb_chunks_wph + self.nb_chunks_sm
        
        #print(f"Nb of chunks: {self.nb_chunks}")
        
    def preconfigure(self, data, requires_grad=False,
                     mem_chunk_factor=25, mem_chunk_factor_grad=40,
                     precompute_wt=True, precompute_modwt=True, cross=False,
                     nb_wph_cov_per_chunk=None, pbc=True):
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
            DESCRIPTION. The default is 40.
        precompute_wt : bool, optional
            Do we precompute the wavelet transform of input data ? (if enough memory is available)
            The default is True.
        precompute_modwt : bool, optional
            Do we precompute the modulust of the wavelet transform of input data ? (if enough memory is available)
            The default is True.
        cross : bool, optional
            The default is False.
        nb_wph_cov_per_chunk : int, optional
            The default is None.

        Returns
        -------
        data : tensor
            Torch.tensor version on input data.
        int
            Total number of chunks.

        """            
        mem_avail = get_memory_available(self.device) # in bytes
        
        def _preconfigure_data(data):
            nonlocal mem_avail
            
            data = to_torch(data, device=self.device, precision=self.precision)
            data_size = data.nelement() * data.element_size() * (1 + (not torch.is_complex(data))) # in bytes (assuming complex data)
            if requires_grad and not data.requires_grad:
                data.requires_grad = True
            
            _tmp_data_wt, _tmp_data_wt_mod = None, None
            
            # Precompute the wavelet transform if we have enough memory
            if data_size * self.psi_f.shape[0] < 1/4 * mem_avail and precompute_wt:
                #print("Enough memory to store the wavelet transform of input data.")
                data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
                data_wt_f = data_f * self.psi_f
                del data_f
                data_wt = ifft(data_wt_f)
                if not pbc: data_wt = self._cutting(data_wt) # Cutting if no periodic bounday conditions
                del data_wt_f
                _tmp_data_wt = data_wt
                mem_avail -= data_size * self.psi_f.shape[0]
            
            # Precompute the modulus of the wavelet transform if we have enough memory
            if data_size * self.psi_f.shape[0] < 1/4 * mem_avail and precompute_modwt and precompute_wt:
                #print("Enough memory to store the modulus of the wavelet transform of input data.")
                _tmp_data_wt_mod = torch.abs(_tmp_data_wt)
                mem_avail -= data_size * self.psi_f.shape[0]
                
            return data, _tmp_data_wt, _tmp_data_wt_mod, data_size
        
        if not cross:
            data, _tmp_data_wt, _tmp_data_wt_mod, data_size = _preconfigure_data(data)
            if _tmp_data_wt is not None:
                self._tmp_data_wt = _tmp_data_wt  # We keep this variable in memory
            if _tmp_data_wt_mod is not None:
                self._tmp_data_wt_mod = _tmp_data_wt_mod  # We keep this variable in memory
                
            req_grad = data.requires_grad
        else:
            if not isinstance(data, list):
                raise Exception("When cross is True, data must be a list!")
            data1, _tmp_data1_wt, _tmp_data1_wt_mod, _ = _preconfigure_data(data[0])
            data2, _tmp_data2_wt, _tmp_data2_wt_mod, data_size = _preconfigure_data(data[1])
            
            if data1.shape != data2.shape:
                raise Exception("data1 and data2 must be of same shape!")
                
            data = [data1, data2]
            
            if _tmp_data1_wt is not None or _tmp_data2_wt is not None:
                self._tmp_data_wt = [_tmp_data1_wt, _tmp_data2_wt]
            if _tmp_data1_wt_mod is not None or _tmp_data2_wt_mod is not None:
                self._tmp_data_wt_mod = [_tmp_data1_wt_mod, _tmp_data2_wt_mod]
                
            req_grad = data1.requires_grad or data2.requires_grad
        
        self._prepare_computation(data_size, mem_avail, mem_chunk_factor_grad if req_grad else mem_chunk_factor, nb_wph_cov_per_chunk=nb_wph_cov_per_chunk)
        
        self.preconfigured = True
        
        return data, self.nb_chunks
    
    def _free(self):
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
            if self.norm_wph_means is None or self.norm_wph_means.shape[-4] != self.nb_wph_cov: # If self.norm_wph_means is not complete
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
            if self.norm_wph_stds is None or self.norm_wph_stds.shape[-4] != self.nb_wph_cov:  # If self.norm_wph_stds is not complete
                std1 = torch.sqrt(torch.mean(torch.abs(xpsi1_k1.detach()) ** 2, (-1, -2), keepdim=True))
                std2 = torch.sqrt(torch.mean(torch.abs(xpsi2_k2.detach()) ** 2, (-1, -2), keepdim=True))
                
                std1[std1 == 0] = 1.0 # To avoid division by zero
                std2[std2 == 0] = 1.0 # To avoid division by zero
                
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
                std[std == 0] = 1.0 # To avoid division by zero
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
        self.norm_sm_means_1 = None
        self.norm_sm_means_2 = None
        self.norm_sm_stds = None
        
    def get_normalization(self):
        """
        Get saved data for normalization of the coefficients.

        Returns
        -------
        None.

        """
        return self.norm_wph_means, self.norm_wph_stds, self.norm_sm_means_1, self.norm_sm_means_2, self.norm_sm_stds
    
    def set_normalization(self, norm_wph_means, norm_wph_stds, norm_sm_means_1, norm_sm_means_2, norm_sm_stds):
        """
        Set normalization of the coefficients.

        Returns
        -------
        None.

        """
        self.norm_wph_means = norm_wph_means
        self.norm_wph_stds = norm_wph_stds
        self.norm_sm_means_1 = norm_sm_means_1
        self.norm_sm_means_2 = norm_sm_means_2
        self.norm_sm_stds = norm_sm_stds

    def _cutting(self, data):
        """
            Internal function for cutting.
        """
        cut_x = int(self.cut_x_param * 2 ** self.J)
        cut_y = int(self.cut_y_param * 2 ** self.J)
        if 2 * cut_y >= self.M or 2 * cut_x >= self.N:
            raise Exception("Invalid cutting configuration! Lower cut_x_param/cut_y_param attributes.")
        return data[..., cut_y:-cut_y, cut_x:-cut_x]
    
    def _padding(self, data, mode):
        """
            Internal function for padding.
        """
        if mode == "zeros":
            # For information:
            # M_padded = self.M - 2*cut_y + self.max_ty
            # N_padded = self.N - 2*cut_x + self.max_tx
            pad_data = torch.nn.functional.pad(data, (0, self.max_tx, 0, self.max_ty))
            return pad_data # (..., M_padded, N_padded)
        elif mode is None:
            return data
        else:
            raise Exception(f"Invalid padding mode: {mode}!")
    
    def _apply_chunk(self, data, chunk_id, norm, pbc):
        """
        Internal function. Use apply instead.
        """
        # Computation of the chunk of coefficients
        if chunk_id < self.nb_chunks_wph: # WPH moments:
            cov_chunk = self.wph_moments_chunk_list[chunk_id] # list id (nb_cov_chunk)
            wph_chunk = torch.nonzero(torch.logical_and(self._id_cov_indices >= cov_chunk[0], self._id_cov_indices <= cov_chunk[-1]))[:, 0] # (nb_wph_chunk)

            curr_id_cov_indices =  self._id_cov_indices[wph_chunk] # (nb_wph_chunk)
            curr_psi_1_indices = self._psi_1_indices[cov_chunk] # (nb_cov_chunk) : id associated to (j1, t1)
            curr_psi_2_indices = self._psi_2_indices[cov_chunk] # (nb_cov_chunk) : id associated to (j2, t2)
            curr_harmo_indices = self._harmo_indices[cov_chunk] # (nb_cov_chunk, 2) : (p1, p2)
            curr_pseudo_indices = self._pseudo_indices[cov_chunk] # (nb_cov_chunk) : pseudo
            curr_translation_pos = self._translation_pos[wph_chunk] # (nb_wph_chunk, 2) : (tx, ty) position for translations
            
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
                            if not pbc: xpsi = self._cutting(xpsi) # Cutting if no periodic bounday conditions
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
                        if not pbc: xpsi = self._cutting(xpsi) # Cutting if no periodic bounday conditions
                        del data_f, data_wt_f
                    return xpsi
                else:
                    raise Exception(f"Invalid p value: {p}!")
            
            if chunk_id < self.final_chunk_id_per_class[0]: # S11 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 1) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 1) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[1]: # S00 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 0) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 0) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[2]: # S01/C01 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 0) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 1) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[3]: # S10/C10 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 1) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 0) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[4]: # Other moments
                # Get the relevant part of the wavelet transform and compute the corresponding phase harmonics
                xpsi1 = get_precomputed_data(curr_psi_1_indices, 1) # (..., P, M, N)
                xpsi1_k1 = phase_harmonics(xpsi1, curr_harmo_indices[:, 0]) # (..., P, M, N)
                del xpsi1
                xpsi2 = get_precomputed_data(curr_psi_2_indices, 1) # (..., P, M, N)
                xpsi2_k2 = phase_harmonics(xpsi2, curr_harmo_indices[:, 1]) # (..., P, M, N)
                del xpsi2

            # Take the complex conjugate of xpsi2_k2 depending on the value of pseudo
            indices_pseudo = torch.where(curr_pseudo_indices == 1)[0]
            xpsi2_k2[..., indices_pseudo, :, :] = torch.conj(torch.index_select(xpsi2_k2, -3, indices_pseudo))
            
            # Normalization
            xpsi1_k1, xpsi2_k2 = self._wph_normalization(xpsi1_k1, xpsi2_k2, norm, cov_chunk)

            # Potential padding
            if not pbc:
                xpsi1_k1 = self._padding(xpsi1_k1, "zeros")
                xpsi2_k2 = self._padding(xpsi2_k2, "zeros")

            # Compute covariance with translation with a convolution
            cov = ifft(fft(xpsi1_k1) * torch.conj(fft(xpsi2_k2))) / self.M / self.N #  (..., nb_cov_chunk, M, N)
            del xpsi1_k1, xpsi2_k2

            # Select the different translations
            cov = cov[...,  curr_id_cov_indices - curr_id_cov_indices[0], curr_translation_pos[:, 1], curr_translation_pos[:, 0]] # (..., nb_wph_chunk)
            
            # Normalisation if padding
            if not pbc:
                cov = cov * self._translation_weight[wph_chunk]  # (..., nb_wph_chunk)
            
            return cov, wph_chunk
        elif chunk_id < self.final_chunk_id_per_class[5]: # Scaling moments
            cov_chunk = self.scaling_moments_chunk_list[chunk_id - self.final_chunk_id_per_class[-2]]
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
            if not pbc: data_st = self._cutting(data_st) # Cutting if no periodic bounday conditions
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
            
            # For scaling moments, shift cov_chunk by the number of WPH moments
            # There are subtleties related to the doubling of coefficients when applying to complex data
            cov_chunk_shifted = cov_chunk[0]*(1 + cplx) + self.nb_wph_moments + cov_chunk
            if cplx:
                cov_chunk_shifted = torch.cat([cov_chunk_shifted, cov_chunk_shifted + len(cov_chunk)])
            return cov, cov_chunk_shifted
        else: # Invalid
            raise Exception("Invalid chunk_id!")
            
    def _apply_cross_chunk(self, data1, data2, chunk_id, norm, pbc):
        """
        Internal function. Use apply instead.
        """
        # Computation of the chunk of coefficients
        if chunk_id < self.nb_chunks_wph: # WPH moments:
            cov_chunk = self.wph_moments_chunk_list[chunk_id] # list id (nb_cov_chunk)
            wph_chunk = torch.nonzero(torch.logical_and(self._id_cov_indices >= cov_chunk[0], self._id_cov_indices <= cov_chunk[-1]))[:, 0] # (nb_wph_chunk)

            curr_id_cov_indices =  self._id_cov_indices[wph_chunk] # (nb_wph_chunk)
            curr_psi_1_indices = self._psi_1_indices[cov_chunk] # (nb_cov_chunk) : id associated to (j1, t1)
            curr_psi_2_indices = self._psi_2_indices[cov_chunk] # (nb_cov_chunk) : id associated to (j2, t2)
            curr_harmo_indices = self._harmo_indices[cov_chunk] # (nb_cov_chunk, 2) : (p1, p2)
            curr_pseudo_indices = self._pseudo_indices[cov_chunk] # (nb_cov_chunk) : pseudo
            curr_translation_pos = self._translation_pos[wph_chunk] # (nb_wph_chunk, 2) : (tx, ty) position for translations
            
            def get_precomputed_data(psi_indices, p, i):
                """
                Internal function that load in memory relevant part of precomputed data (wavelet transform, or its modulus).
                If the data was not precomputed, we first compute it.
                """
                data = data1 if i == 0 else data2
                if p == 0: # We need the modulus
                    if hasattr(self, '_tmp_data_wt_mod') and self._tmp_data_wt_mod[i] is not None:
                        data_wt_mod = self._tmp_data_wt_mod[i]
                        xpsi = torch.index_select(data_wt_mod, -3, psi_indices)
                    else:
                        if hasattr(self, '_tmp_data_wt') and self._tmp_data_wt[i] is not None:
                            data_wt = self._tmp_data_wt[i]
                            xpsi = torch.index_select(data_wt, -3, psi_indices)
                        else:
                            data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
                            data_wt_f = data_f * self.psi_f[psi_indices]
                            xpsi = ifft(data_wt_f)
                            if not pbc: xpsi = self._cutting(xpsi) # Cutting if no periodic bounday conditions
                            del data_f, data_wt_f
                    return torch.abs(xpsi)
                elif p == 1: # We just need the wavelet transform
                    if hasattr(self, '_tmp_data_wt') and self._tmp_data_wt[i] is not None:
                        data_wt = self._tmp_data_wt[i]
                        xpsi = torch.index_select(data_wt, -3, psi_indices)
                    else:
                        data_f = fft(data).unsqueeze(-3) # (..., 1, M, N)
                        data_wt_f = data_f * self.psi_f[psi_indices]
                        xpsi = ifft(data_wt_f)
                        if not pbc: xpsi = self._cutting(xpsi) # Cutting if no periodic bounday conditions
                        del data_f, data_wt_f
                    return xpsi
                else:
                    raise Exception(f"Invalid p value: {p}!")
            
            if chunk_id < self.final_chunk_id_per_class[0]: # S11 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 1, 0) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 1, 1) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[1]: # S00 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 0, 0) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 0, 1) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[2]: # S01/C01 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 0, 0) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 1, 1) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[3]: # S10/C10 moments
                xpsi1_k1 = get_precomputed_data(curr_psi_1_indices, 1, 0) # (..., P, M, N)
                xpsi2_k2 = get_precomputed_data(curr_psi_2_indices, 0, 1) # (..., P, M, N)
            elif chunk_id < self.final_chunk_id_per_class[4]: # Other moments
                # Get the relevant part of the wavelet transform and compute the corresponding phase harmonics
                xpsi1 = get_precomputed_data(curr_psi_1_indices, 1, 0) # (..., P, M, N)
                xpsi1_k1 = phase_harmonics(xpsi1, curr_harmo_indices[:, 0]) # (..., P, M, N)
                del xpsi1
                xpsi2 = get_precomputed_data(curr_psi_2_indices, 1, 1) # (..., P, M, N)
                xpsi2_k2 = phase_harmonics(xpsi2, curr_harmo_indices[:, 1]) # (..., P, M, N)
                del xpsi2
                
            # Take the complex conjugate of xpsi2_k2 depending on the value of pseudo
            indices_pseudo = torch.where(curr_pseudo_indices == 1)[0]
            xpsi2_k2[..., indices_pseudo, :, :] = torch.conj(torch.index_select(xpsi2_k2, -3, indices_pseudo))

            # Normalization
            xpsi1_k1, xpsi2_k2 = self._wph_normalization(xpsi1_k1, xpsi2_k2, norm, cov_chunk)

            # Potential padding
            if not pbc:
                xpsi1_k1 = self._padding(xpsi1_k1, "zeros")
                xpsi2_k2 = self._padding(xpsi2_k2, "zeros")
            
            # Compute covariance with translation with a convolution
            cov = ifft(fft(xpsi1_k1) * torch.conj(fft(xpsi2_k2))) / self.M / self.N #  (..., nb_cov_chunk, M, N)
            del xpsi1_k1, xpsi2_k2

            # Select the different translations
            cov = cov[...,  curr_id_cov_indices - curr_id_cov_indices[0], curr_translation_pos[:, 1], curr_translation_pos[:, 0]] # (..., nb_wph_chunk)
            
            # Normalisation if padding
            if not pbc:
                cov = cov * self._translation_weight[wph_chunk]  # (..., nb_wph_chunk)
            
            return cov, wph_chunk
        elif chunk_id < self.final_chunk_id_per_class[5]: # Scaling moments
            cov_chunk = self.scaling_moments_chunk_list[chunk_id - self.final_chunk_id_per_class[-2]]
            cov_indices = self.scaling_moments_indices[cov_chunk]
            
            curr_phi_indices = self._phi_indices[cov_chunk]
            
            # Simple combination of data1 and data2 if scaling moments are demanded
            data = torch.sqrt(torch.absolute(data1 * data2))
        
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
            if not pbc: data_st = self._cutting(data_st) # Cutting if no periodic bounday conditions
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
            
            # For scaling moments, shift cov_chunk by the number of WPH moments
            # There are subtleties related to the doubling of coefficients when applying to complex data
            cov_chunk_shifted = cov_chunk[0]*(1 + cplx) + self.nb_wph_moments + cov_chunk
            if cplx:
                cov_chunk_shifted = torch.cat([cov_chunk_shifted, cov_chunk_shifted + len(cov_chunk)])
            return cov, cov_chunk_shifted
        else: # Invalid
            raise Exception("Invalid chunk_id!")
        
    def apply(self, data, chunk_id=None, requires_grad=False, norm=None, ret_indices=False, pbc=True, ret_wph_obj=False, cross=False):
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
        pbc : bool, optional
            Can we assume periodic bounday conditions for the input data?
            If not, we cut out the relevant pixels of the wavelet transform of the input data that might be corrupted by the circular convolutions. The default is True.
        ret_wph_obj : bool, optional
            Return a WPH object if needed. The default is False.
        cross : bool, optional
            Estimate cross moments?
            If True, data must be a list of two elements corresponding to the two involved maps.
            The default is False.

        Returns
        -------
        coeffs : tensor
            WPH coefficients.

        """
        if cross:
            if not isinstance(data, list):
                raise Exception("Please provide in data a list of two maps to estimate cross moments!")
        
        if chunk_id is None: # Compute all chunks at once
            data, nb_chunks = self.preconfigure(data, requires_grad=requires_grad, cross=cross, pbc=pbc)
            coeffs = []
            for i in range(self.nb_chunks):
                if cross:
                    cov, _ = self._apply_cross_chunk(data[0], data[1], i, norm, pbc)
                else:
                    cov, _ = self._apply_chunk(data, i, norm, pbc)
                coeffs.append(cov)
            coeffs = torch.cat(coeffs, -1)
            indices = torch.arange(coeffs.shape[-1])
        else: # Compute selected chunk
            if not self.preconfigured:
                raise Exception("First preconfigure data!")
            if cross:
                coeffs, indices = self._apply_cross_chunk(data[0], data[1], chunk_id, norm, pbc)
            else:
                coeffs, indices = self._apply_chunk(data, chunk_id, norm, pbc)
        
        # We free memory when needed
        if chunk_id is None or chunk_id == self.nb_chunks - 1:
            self._free()
            
        if ret_wph_obj:
            if chunk_id is None:
                return WPH(coeffs, self.wph_moments_indices, self.scaling_moments_indices, J=self.J, L=self.L, A=self.A)
            else:
                if chunk_id < self.nb_chunks_wph:
                    cov_chunk = self.wph_moments_chunk_list[chunk_id]
                    wph_chunk_indices = torch.nonzero(torch.logical_and(self._id_cov_indices >= cov_chunk[0], self._id_cov_indices <= cov_chunk[-1]))[:, 0] # (nb_wph_chunk)
                    return WPH(coeffs, wph_chunk_indices, J=self.J, L=self.L, A=self.A)
                else:
                    cov_indices = self.scaling_moments_indices[self.scaling_moments_chunk_list[chunk_id - self.final_chunk_id_per_class[3]]]
                return WPH(coeffs, np.array([]), sm_coeffs_indices=cov_indices, J=self.J, L=self.L, A=self.A)
        elif ret_indices:
            return coeffs, indices
        else:
            return coeffs
    
    def forward(self, data, chunk_id=None, requires_grad=False, norm=None, ret_indices=False, pbc=True, ret_wph_obj=False, cross=False):
        """
        Alias of apply.
        """
        return self.apply(data, chunk_id=chunk_id, requires_grad=requires_grad, norm=norm, ret_indices=ret_indices, pbc=pbc, ret_wph_obj=ret_wph_obj, cross=cross)
