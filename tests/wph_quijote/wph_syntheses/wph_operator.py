import os

from ..bump_steerable_wavelet.bump_steerable_wavelet import compute_bank_of_wavelet, create_bank_scaling_functions
from ..wph_syntheses.sufficient_stat import compute_idx_of_sufficient_stat, compute_idx_of_sufficient_stat_PS

__all__ = ['PhaseHarmonics2d']

import torch
import numpy as np
from .backend import SubInitSpatialMeanC, PhaseHarmonics, PowerHarmonics, DivInitStd, modulus_complex, mul
from .utils import fft2_c2c, ifft2_c2c


class PhaseHarmonics2d(object):
    def __init__(self,
                 M,
                 N,
                 J,
                 L,
                 delta_j,
                 delta_l,
                 delta_n,
                 nb_chunks,
                 scaling_function_moments=[0, 1, 2, 3],
                 devid=0,
                 sufficient_stat=None
                 ):
        self.M, self.N, self.J, self.L = M, N, J, L  # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j  # max scale interactions
        self.dl = delta_l  # max angular interactions
        self.dn = delta_n
        self.nb_chunks = nb_chunks  # number of chunks to cut whp cov
        self.devid = devid  # gpu id
        if self.dl > self.L:
            raise (ValueError('delta_l must be <= L'))
        self.should_check_real_symmetry = False
        self.scaling_function_moments = scaling_function_moments
        self.sufficient_stat = sufficient_stat
        self.build()


    def build(self):
        self.phase_harmonics = PhaseHarmonics.apply
        self.power_harmonic = PowerHarmonics.apply

        self.filters_tensor()
        if self.sufficient_stat is None: # Default behavior
            self.idx_wph = compute_idx_of_sufficient_stat(self.L, self.J, self.dj, self.dl, self.dn)
        elif self.sufficient_stat == "PS":
            self.idx_wph = compute_idx_of_sufficient_stat_PS(self.L, self.J, self.dj, self.dl, self.dn)
        else:
            raise Exception("Invalid sufficient_stat param!")
        self.wph_by_chunk = self.get_this_chunk(self.nb_chunks)
        self.subinitmean1 = {}
        self.subinitmean2 = {}
        self.divinitstd1 = {}
        self.divinitstd2 = {}

        for chunk_id in range(self.nb_chunks+1):
            if chunk_id < self.nb_chunks:
                self.subinitmean1[chunk_id] = SubInitSpatialMeanC()
                self.subinitmean2[chunk_id] = SubInitSpatialMeanC()
                self.divinitstd1[chunk_id] = DivInitStd()
                self.divinitstd2[chunk_id] = DivInitStd()
            else:
                self.subinitmeanJ = SubInitSpatialMeanC()
                self.subinitmeanJabs = SubInitSpatialMeanC()
                self.divinitstdmeanJ = DivInitStd()
                self.subinitmeanPixel = SubInitSpatialMeanC()



    def filters_tensor(self):
        assert(self.M == self.N)
        bump_steerable_filter_filename = os.path.join('wph_quijote', 'bump_steerable_wavelet', 'filters', 'bump_steerable_wavelet_N_'+str(self.N)+'_J_'+str(self.J)+'_L'+str(self.L)+'_dn'+str(self.dn)+'.npy')
        if not os.path.exists(bump_steerable_filter_filename):
            print("The wavelet filters do not exist. Starting the creation of the wavelet filters. This can take few minutes.")
            compute_bank_of_wavelet([-np.pi/4, 0., np.pi/4, np.pi/2], self.N, self.L, self.J, self.dn)
            print("Wavelet filters created.")

        matfilters = np.load(os.path.join('wph_quijote', 'bump_steerable_wavelet', 'filters', 'bump_steerable_wavelet_N_'+str(self.N)+'_J_'+str(self.J)+'_L'+str(self.L)+'_dn'+str(self.dn)+'.npy'), allow_pickle=True).item()
        print("Wavelet filters loaded.")


        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        scaling_function_filename = os.path.join('wph_quijote', 'bump_steerable_wavelet', 'filters', 'bump_scaling_functions_N_{N}_J_{J}.npy'.format(N=self.N, J=self.J))
        if not os.path.exists(scaling_function_filename):
            print("The scaling functions do not exist. Starting the creation of the scaling function")
            create_bank_scaling_functions(self.N, self.J, self.L)
            print("Scaling functions created")

        fftphi = np.load(scaling_function_filename).astype(np.complex_)
        print("Scaling functions loaded")
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi)  # (L2, J, 2*delta_n+1, M,N,2)
        self.hatphi = torch.FloatTensor(hatphi)  # (M,N,2) or (J,M,N,2)


    def get_this_chunk(self, nb_chunks):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['j1'])
        print("Total number of cov: "+str(nb_cov)+" times L2="+str(self.L*2))
        min_chunk = nb_cov // nb_chunks
        print("Number of cov per chunk: "+str(min_chunk)+" or "+str(min_chunk+1))
        nb_cov_chunk = np.zeros(nb_chunks, dtype=np.int32)
        for idxc in range(nb_chunks):
            nb_cov_chunk[idxc] = int(min_chunk)
        for idxc in range(nb_cov - min_chunk*nb_chunks):
            nb_cov_chunk[idxc] = nb_cov_chunk[idxc] + 1

        wph_by_chunk = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            wph_by_chunk[idxc] = {}
            wph_by_chunk[idxc]['j1'] = self.idx_wph['j1'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['j2'] = self.idx_wph['j2'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['ell2'] = self.idx_wph['ell2'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['k1'] = self.idx_wph['k1'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['k2'] = self.idx_wph['k2'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['dn1'] = self.idx_wph['dn1'][offset:offset + nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['dn2'] = self.idx_wph['dn2'][offset:offset + nb_cov_chunk[idxc]]
            offset = offset + nb_cov_chunk[idxc]

        return wph_by_chunk

    def _type(self, _type, devid=None):
        self.hatpsi = self.hatpsi.type(_type).to(devid)
        self.hatphi = self.hatphi.type(_type).to(devid)
            
        for chunk_id in range(self.nb_chunks):
            self.wph_by_chunk[chunk_id]['j1'] = self.wph_by_chunk[chunk_id]['j1'].type(torch.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['j2'] = self.wph_by_chunk[chunk_id]['j2'].type(torch.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['dn1'] = self.wph_by_chunk[chunk_id]['dn1'].type(torch.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['dn2'] = self.wph_by_chunk[chunk_id]['dn2'].type(torch.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['ell2'] = self.wph_by_chunk[chunk_id]['ell2'].type(torch.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['k1'] = self.wph_by_chunk[chunk_id]['k1'].type(torch.FloatTensor).to(devid)
            self.wph_by_chunk[chunk_id]['k2'] = self.wph_by_chunk[chunk_id]['k2'].type(torch.FloatTensor).to(devid)

        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        return self._type(torch.FloatTensor, devid)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        print('call cpu')
        return self._type(torch.FloatTensor, devid="cpu")

    def forward(self, input, chunk_id, norm="auto"):
        M = self.M
        N = self.N
        L2 = self.L*2
        Nimg = input.size(0)


        # input: (Nimg,M,N) if the field is real or (Nimg, M, N, 2) if the field is complex
        if input.dim() == 3:
            x_c = input.new_zeros(input.size(0), input.size(1), input.size(2), 2)
            x_c[:, :, :, 0] = input  # (Nimg, M, N, 2)
        elif input.dim() == 4:
            x_c = input.new_zeros(input.size(0), input.size(1), input.size(2), 2)
            x_c = input  # (Nimg, M, N, 2)
        else:
            raise ValueError("The dim of the input should be 3 or 4: (Nimg, M, N) or (Nimg, M, N, 2")


        if chunk_id < self.nb_chunks:
            hatx_c = fft2_c2c(x_c)  # fft2 -> (Nimg, M, N, 2)
            del x_c
            hatpsi_la_chunk, list_places_1, list_places_2 = self.create_hatpsi_la_chunk(chunk_id)
            hatx_bc = hatx_c.unsqueeze(1).unsqueeze(1) # (Nimg, 1, 1, M, N, 2)
            del hatx_c
            hatxpsi_bc = mul(hatpsi_la_chunk, hatx_bc)  # (Nimg, L2, N_in_chunk, M, N, 2)
            del hatpsi_la_chunk, hatx_bc
            xpsi_bc = ifft2_c2c(hatxpsi_bc)  # (Nimg, L2, N_in_chunk, M, N, 2)
            del hatxpsi_bc


            # select la1, et la2, P_c = number of |la1| in this chunk
            nb_channels = self.wph_by_chunk[chunk_id]['j1'].shape[0]
            xpsi_bc_la1 = xpsi_bc.new(Nimg, L2, nb_channels, M, N, 2)  # (Nimg, L2, P_c, M, N, 2)
            xpsi_bc_la2 = xpsi_bc.new(Nimg, L2, nb_channels, M, N, 2)  # (Nimg, L2, P_c, M, N, 2)
            for ell1 in range(L2):
                ell2 = torch.remainder(self.wph_by_chunk[chunk_id]['ell2'] + ell1, L2)
                xpsi_bc_la1[:, ell1, ...] = xpsi_bc[:, ell1, list_places_1, :, :, :]
                xpsi_bc_la2[:, ell1, ...] = xpsi_bc[:, ell2, list_places_2, :, :, :]

            del xpsi_bc
            xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, self.wph_by_chunk[chunk_id]['k1'])  # (Nimg, L2, P_c, M, N, 2)
            xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, self.wph_by_chunk[chunk_id]['k2'])  # (Nimg, L2, P_c, M, N, 2)

            del xpsi_bc_la1, xpsi_bc_la2

            # Taking the complex conjugate of xpsi_bc_la2k2
            xpsi_bc_la2k2[..., 1] = - xpsi_bc_la2k2[..., 1]

            # substract spatial mean along M and N
            xpsi0_bc_la1k1 = self.subinitmean1[chunk_id](xpsi_bc_la1k1, norm=norm)  # (Nimg, L2, P_c, M, N, 2)
            xpsi0_bc_la2k2 = self.subinitmean2[chunk_id](xpsi_bc_la2k2, norm=norm)  # (Nimg, L2, P_c, M, N, 2)
            # del xpsi_bc_la1k1, xpsi_bc_la2k2

            if norm == "auto": # Auto normalization
                xpsi0_bc_la1k1 = self.divinitstd1[chunk_id](xpsi0_bc_la1k1)  # (Nimg, L2, P_c, M, N, 2)
                xpsi0_bc_la2k2 = self.divinitstd2[chunk_id](xpsi0_bc_la2k2)  # (Nimg, L2, P_c, M, N, 2)


            # compute mean spatial
            corr_xpsi_bc = mul(xpsi0_bc_la1k1, xpsi0_bc_la2k2)    # (Nimg, L2, P_c, M, N, 2)
            del xpsi0_bc_la1k1, xpsi0_bc_la2k2
            corr_bc = torch.mean(torch.mean(torch.mean(corr_xpsi_bc, -2, True), -3, True), 0, True)    # (1, L2, P_c, 1, 1, 2)
            del corr_xpsi_bc


            corr_bc = corr_bc.view(1, 1, nb_channels * L2, 1, 1, 2)
            return corr_bc

        else:
            # We divide here the signal into its real and imaginary part
            # After concatention, the process is the same but as with 2*Nimg
            x_c2 = x_c.new_zeros((2 * x_c.size(0), x_c.size(1), x_c.size(2), x_c.size(3))) # (2 * Nimg, M, N, 2)
            x_c2[:x_c.size(0), :, :, 0] = x_c[:, :, :, 0]
            x_c2[x_c.size(0):, :, :, 0] = x_c[:, :, :, 1]
            del x_c
            
            # FFT2 and dimension check of the filters
            x_c2 = self.subinitmeanPixel(x_c2, norm=norm) # To avoid a bias due to a non-zero mean of the signal
            hatx_c = fft2_c2c(x_c2).unsqueeze(1).unsqueeze(1)  # fft2 -> (2*Nimg, 1, 1, M, N, 2)
            del x_c2
            if self.hatphi.dim() == 3:
                hatphi = self.hatphi.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, M, N, 2)
            else:
                hatphi = self.hatphi.unsqueeze(1).unsqueeze(0)    # (1, J, 1, M, N, 2)

            hatxphi_c = mul(hatx_c, hatphi)  # (2*Nimg, J, 1, M, N, 2)
            xpsi_c = ifft2_c2c(hatxphi_c)  # (2*Nimg, J, 1, M, N, 2)
            tensor_k = xpsi_c.new_tensor(self.scaling_function_moments)
            xpsi_c = xpsi_c.expand((xpsi_c.size(0), xpsi_c.size(1), len(self.scaling_function_moments), M, N, 2)) # (2*Nimg, J, K, M, N, 2)
            xpsi_c_k = self.power_harmonic(xpsi_c, tensor_k)

            # submean from spatial M N
            xpsi0_c = self.subinitmeanJ(xpsi_c_k, norm=norm)   # (2*Nimg, J, K, M, N, 2)
            if norm == "auto": # Auto normalization
                xpsi0_c = self.divinitstdmeanJ(xpsi0_c)  # (2*Nimg, J, K, M, N, 2)
            xpsi0_mod = modulus_complex(xpsi0_c)  # (2*Nimg, J, K, M, N, 2)
            xpsi0_mod2 = mul(xpsi0_mod, xpsi0_mod)  # (2*Nimg, J, K, M, N, 2)
            Sout = input.new(1, 1, 2*len(self.scaling_function_moments)*xpsi0_mod2.size(1), 1, 1, 2) # (1, 1, 2*J*K, 1, 1, 2)
            xpsi0_mod2 = xpsi0_mod2.view(xpsi0_mod2.size(0), 1, xpsi0_mod2.size(1)*xpsi0_mod2.size(2), M, N, 2) # (2*Nimg, 1, J*K, M, N, 2)
        
            # We have to trick here to distribute the real and imaginary part on the dimension
            # on which no mean is performed. Order of commands is related to the way view works
            xpsi0_mod2 = xpsi0_mod2.view(2, xpsi0_mod2.size(0)//2, 1, xpsi0_mod2.size(2), M, N, 2) # (2, Nimg, 1, J*K, M, N, 2)
            xpsi0_mod2 = xpsi0_mod2.permute((1,0,2,3,4,5, 6)).contiguous() # (Nimg, 2, 1, J*K, M, N, 2)
            xpsi0_mod2 = xpsi0_mod2.view(xpsi0_mod2.size(0), 1, 2*xpsi0_mod2.size(3), M, N, 2) # (Nimg, 1, 2 * J*K, M, N, 2)
            Sout[:, :, :, :, :, :] = torch.mean(torch.mean(torch.mean(xpsi0_mod2, -2, True), -3, True), 0, True)
            return Sout


    def create_hatpsi_la_chunk(self, chunk_id):
        list_indices, list_places = torch.unique(
            torch.cat(
                (torch.stack((self.wph_by_chunk[chunk_id]['j1'], self.wph_by_chunk[chunk_id]['dn1']), dim=0),
                 torch.stack((self.wph_by_chunk[chunk_id]['j2'], self.wph_by_chunk[chunk_id]['dn2']), dim=0)),
                dim=1),
            dim=1, return_inverse=True
        )
        list_places_1 = list_places[:list_places.shape[0] // 2]
        list_places_2 = list_places[list_places.shape[0] // 2: ]
        hatpsi_la_chunk = self.hatpsi[:, list_indices[0], list_indices[1], :, :, :]
        return hatpsi_la_chunk.unsqueeze(0), list_places_1, list_places_2


    def __call__(self, input, chunk_id, norm="auto"):
        return self.forward(input, chunk_id, norm=norm)
