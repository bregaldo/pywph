# -*- coding: utf-8 -*-

import math
import os
import numpy as np


class Filter:
    """
        Base class for filters.
    """
    
    def __init__(self, M, N, dtype=np.complex):
        self.M = M
        self.N = N
        self.data = np.zeros((M, N), dtype)
        self.type = self.__class__.__name__


class GaussianFilter (Filter):
    """
    Gaussian filter.
    """
    
    def __init__(self, M, N, j, theta=0.0, gamma=1.0, sigma0=1.0, fourier=False):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float, optional
            Rotation angle. The default is 0.0.
        gamma : float, optional
            Aspect ratio of the envelope. The default is 1.0.
        sigma0 : float, optional
            Standard deviation of the envelope before its dilation. The default is 1.0.
        fourier : bool, optional
            Do we want to store the filter in Fourier space or not? The default is False.

        Returns
        -------
        None.

        """
        self.fourier = fourier
        self.real = True # A Gaussian filter is always real (both in Fourier and physical space)
        super().__init__(M, N, dtype=np.float) # No need for a complex data type
        self.j = j
        self.theta = theta
        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma = self.sigma0 * 2 ** j
        self.build()
        
    def build(self):
        """
        Build the filter for a given set of parameters.
        """
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        RInv = np.array([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]])
        D = np.array([[1, 0], [0, self.gamma ** 2]])
        curv = np.dot(R, np.dot(D, RInv)) / (2 * self.sigma ** 2)

        for ex in [-2, -1, 0, 1]:
            for ey in [-2, -1, 0, 1]:
                [xx, yy] = np.mgrid[ex * self.M:self.M + ex * self.M, ey * self.N:self.N + ey * self.N]
                arg = -(curv[0, 0] * xx ** 2 + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy ** 2)
                self.data += np.exp(arg)
                
        normFactor = 2 * np.pi * self.sigma ** 2 / self.gamma
        self.data /= normFactor
        if self.fourier:
            self.data = np.fft.fft2(self.data).real


class BumpSteerableWavelet (Filter):
    """
    Bump-steerable wavelet.
    """
    
    def __init__(self, M, N, j, theta=0.0, k0=2*np.pi, L=8, n=0, alpha=0., fourier=False):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float, optional
            Rotation angle. The default is 0.0.
        k0 : float, optional
            Central wavenumber before the dilation. The default is 2 * np.pi.
        L : int, optional
            Number of angles between 0 and pi. The default is 8.
        n : int, optional
            Radial coordinate for the translation of the filter. The default is 0.
        alpha : float, optional
            Direction for the translation of the filter (relative to theta direction). The default is 0.
        fourier : bool, optional
            Do we want to store the filter in Fourier space or not? The default is False.

        Returns
        -------
        None.

        """
        self.fourier = fourier
        self.real = fourier and (n == 0) # Real filter in Fourier space when no translation
        super().__init__(M, N, dtype=np.float if self.real else np.complex)
        self.theta = theta
        self.sigma = 2 ** j
        self.k0 = k0
        self.L = L
        self.nx = n * np.cos(self.theta - alpha)
        self.ny = n * np.sin(self.theta - alpha)
        self.build()
    
    def _periodization(self, filter_f):
        filter_f_shifted = np.fft.fftshift(filter_f)
        
        M = self.M
        N = self.N
        
        filter_f_shifted[M - M//2:M, N - N//2:N + (N+1)//2] += filter_f_shifted[M + (M+1)//2:, N - N//2:N + (N+1)//2]
        filter_f_shifted[M:M + (M+1)//2, N - N//2:N + (N+1)//2] += filter_f_shifted[:M - M//2, N - N//2:N + (N+1)//2]
    
        filter_f_shifted[M - M//2:M + (M+1)//2, N - N//2:N] += filter_f_shifted[M - M//2:M + (M+1)//2, N + (N+1)//2:]
        filter_f_shifted[M - M//2:M + (M+1)//2, N:N + (N+1)//2] += filter_f_shifted[M - M//2:M + (M+1)//2, :N - N//2]
    
        filter_f_shifted[M:M + (M+1)//2, N:N + (N+1)//2] += filter_f_shifted[:M - M//2, :N - N//2]
        filter_f_shifted[M - M//2:M, N - N//2:N] += filter_f_shifted[M + (M+1)//2:, N + (N+1)//2:]
    
        filter_f_shifted[M:M + (M+1)//2, N - N//2:N] += filter_f_shifted[:M - M//2, N + (N+1)//2:]
        filter_f_shifted[M - M//2:M, N:N + (N+1)//2] += filter_f_shifted[M + (M+1)//2:, :N - N//2]
        
        return np.fft.ifftshift(filter_f_shifted[M - M//2:M + (M+1)//2, N - N//2:N + (N+1)//2])
    
    def build(self):
        # Normalization
        c = 2 ** (self.L - 1) / 1.29 * np.math.factorial(self.L - 1) / np.sqrt(self.L * np.math.factorial(2 * (self.L - 1)))
        
        # Modulus of the central wavenumber
        xi = self.k0 / self.sigma
        
        # Frequencies (two times larger frequency domain for the periodization)
        kx = 2 * 2 * np.pi * np.fft.fftfreq(2 * self.N)
        ky = 2 * 2 * np.pi * np.fft.fftfreq(2 * self.M)
        k2d_x, k2d_y = np.meshgrid(kx, ky)
        k2d = k2d_x + 1j*k2d_y
        k2d_mod = np.absolute(k2d)
        k2d_angle = np.angle(k2d)
        
        # Bump-steerable wavelet
        car_argk_0_pi2 = np.logical_or((k2d_angle - self.theta) % (2*np.pi) <= np.pi/2,
                                       (k2d_angle - self.theta) % (2*np.pi) >= 3*np.pi/2).astype(float)
        car_k_0_2xi = np.logical_and(k2d_mod > 0.0, k2d_mod < 2*xi).astype(float) # To avoid error and/or warning
        exp_var = -(k2d_mod - xi)**2*car_k_0_2xi / (xi**2*car_k_0_2xi - (k2d_mod - xi)**2)
        psi_f = c * np.exp(exp_var) * car_k_0_2xi * np.cos(k2d_angle  - self.theta)**(self.L - 1) * car_argk_0_pi2
        psi_f = psi_f * np.exp(-1j * self.sigma * (k2d_x*self.nx + k2d_y*self.ny)) # Translation of the filter
        psi_f = self._periodization(psi_f)
        if self.fourier:
            self.data = psi_f.real if self.nx == 0 and self.ny == 0 else psi_f
        else:
            self.data = np.fft.ifft2(psi_f)


class BumpIsotropicWavelet (Filter):
    """
    Bump isotropic wavelet.
    """
    
    def __init__(self, M, N, j, theta=0.0, k0=2*np.pi, L=1, n=0, alpha=0, fourier=False):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float, optional
            Rotation angle. The default is 0.0.
            Note that here the only effect of this parameter for this isotropic filter is to change the reference angle for translations.
        k0 : float, optional
            Central wavenumber before the dilation. The default is 2 * np.pi.
        L : int, optional
            Number of angles between 0 and pi. This parameter is useless here, and kept only for compatibility reasons. The default is 1.
        n : int, optional
            Radial coordinate for the translation of the filter. The default is 0.
        alpha : float, optional
            Direction for the translation of the filter (relative to theta direction). The default is 0.
        fourier : bool, optional
            Do we want to store the filter in Fourier space or not? The default is False.

        Returns
        -------
        None.

        """
        self.fourier = fourier
        self.real = True if not self.fourier else n == 0  # Filter is always real in physical space, and this is the case in Fourier space when no translation
        super().__init__(M, N, dtype=np.float) # No need for a complex data type
        self.theta = theta
        self.sigma = 2 ** j
        self.k0 = k0
        self.nx = n * np.cos(self.theta - alpha)
        self.ny = n * np.sin(self.theta - alpha)
        self.build()
    
    def _periodization(self, filter_f):
        filter_f_shifted = np.fft.fftshift(filter_f)
        
        M = self.M
        N = self.N
        
        filter_f_shifted[M - M//2:M, N - N//2:N + (N+1)//2] += filter_f_shifted[M + (M+1)//2:, N - N//2:N + (N+1)//2]
        filter_f_shifted[M:M + (M+1)//2, N - N//2:N + (N+1)//2] += filter_f_shifted[:M - M//2, N - N//2:N + (N+1)//2]
    
        filter_f_shifted[M - M//2:M + (M+1)//2, N - N//2:N] += filter_f_shifted[M - M//2:M + (M+1)//2, N + (N+1)//2:]
        filter_f_shifted[M - M//2:M + (M+1)//2, N:N + (N+1)//2] += filter_f_shifted[M - M//2:M + (M+1)//2, :N - N//2]
    
        filter_f_shifted[M:M + (M+1)//2, N:N + (N+1)//2] += filter_f_shifted[:M - M//2, :N - N//2]
        filter_f_shifted[M - M//2:M, N - N//2:N] += filter_f_shifted[M + (M+1)//2:, N + (N+1)//2:]
    
        filter_f_shifted[M:M + (M+1)//2, N - N//2:N] += filter_f_shifted[:M - M//2, N + (N+1)//2:]
        filter_f_shifted[M - M//2:M, N:N + (N+1)//2] += filter_f_shifted[M + (M+1)//2:, :N - N//2]
        
        return np.fft.ifftshift(filter_f_shifted[M - M//2:M + (M+1)//2, N - N//2:N + (N+1)//2])
    
    def build(self):
        # Normalization
        c = 1.0 # No normalization

        # Modulus of the central wavenumber
        xi = self.k0 / self.sigma
        
        # Frequencies (two times larger frequency domain for the periodization)
        kx = 2 * 2 * np.pi * np.fft.fftfreq(2 * self.N)
        ky = 2 * 2 * np.pi * np.fft.fftfreq(2 * self.M)
        k2d_x, k2d_y = np.meshgrid(kx, ky)
        k2d = k2d_x + 1j*k2d_y
        k2d_mod = np.absolute(k2d)
        
        # Bump-steerable wavelet
        car_k_0_2xi = np.logical_and(k2d_mod > 0.0, k2d_mod < 2*xi).astype(float) # To avoid error and/or warning
        exp_var = -(k2d_mod - xi)**2*car_k_0_2xi / (xi**2*car_k_0_2xi - (k2d_mod - xi)**2)
        psi_f = c * np.exp(exp_var) * car_k_0_2xi
        psi_f = psi_f * np.exp(-1j * self.sigma * (k2d_x*self.nx + k2d_y*self.ny)) # Translation of the filter
        psi_f = self._periodization(psi_f)
        if self.fourier:
            self.data = psi_f.real if self.nx == 0 and self.ny == 0 else psi_f
        else:
            self.data = np.fft.ifft2(psi_f).real


class TopHatWavelet (Filter):
    """
    Top-Hat wavelet.
    """
    
    def __init__(self, M, N, j, theta=0.0, k0=2*np.pi, L=8, n=0, alpha=0., fourier=False):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float, optional
            Rotation angle. The default is 0.0.
        k0 : float, optional
            Central wavenumber before the dilation. The default is 2 * np.pi.
        L : int, optional
            Number of angles between 0 and pi. The default is 8.
        n : int, optional
            Radial coordinate for the translation of the filter. The default is 0.
        alpha : float, optional
            Direction for the translation of the filter (relative to theta direction). The default is 0.
        fourier : bool, optional
            Do we want to store the filter in Fourier space or not? The default is False.

        Returns
        -------
        None.

        """
        self.fourier = fourier
        self.real = fourier and (n == 0) # Real filter in Fourier space when no translation
        super().__init__(M, N, dtype=np.float if self.real else np.complex)
        self.theta = theta
        self.sigma = 2 ** j
        self.k0 = k0
        self.L = L
        self.nx = n * np.cos(self.theta - alpha)
        self.ny = n * np.sin(self.theta - alpha)
        self.build()
    
    def _periodization(self, filter_f):
        filter_f_shifted = np.fft.fftshift(filter_f)
        
        M = self.M
        N = self.N
        
        filter_f_shifted[M - M//2:M, N - N//2:N + (N+1)//2] += filter_f_shifted[M + (M+1)//2:, N - N//2:N + (N+1)//2]
        filter_f_shifted[M:M + (M+1)//2, N - N//2:N + (N+1)//2] += filter_f_shifted[:M - M//2, N - N//2:N + (N+1)//2]
    
        filter_f_shifted[M - M//2:M + (M+1)//2, N - N//2:N] += filter_f_shifted[M - M//2:M + (M+1)//2, N + (N+1)//2:]
        filter_f_shifted[M - M//2:M + (M+1)//2, N:N + (N+1)//2] += filter_f_shifted[M - M//2:M + (M+1)//2, :N - N//2]
    
        filter_f_shifted[M:M + (M+1)//2, N:N + (N+1)//2] += filter_f_shifted[:M - M//2, :N - N//2]
        filter_f_shifted[M - M//2:M, N - N//2:N] += filter_f_shifted[M + (M+1)//2:, N + (N+1)//2:]
    
        filter_f_shifted[M:M + (M+1)//2, N - N//2:N] += filter_f_shifted[:M - M//2, N + (N+1)//2:]
        filter_f_shifted[M - M//2:M, N:N + (N+1)//2] += filter_f_shifted[M + (M+1)//2:, :N - N//2]
        
        return np.fft.ifftshift(filter_f_shifted[M - M//2:M + (M+1)//2, N - N//2:N + (N+1)//2])
    
    def build(self):
        # Normalization
        c = 1.0
        # Modulus of the central wavenumber
        xi = self.k0 / self.sigma
        
        # Frequencies (two times larger frequency domain for the periodization)
        kx = 2 * 2 * np.pi * np.fft.fftfreq(2 * self.N)
        ky = 2 * 2 * np.pi * np.fft.fftfreq(2 * self.M)
        k2d_x, k2d_y = np.meshgrid(kx, ky)
        k2d = k2d_x + 1j*k2d_y
        k2d_mod = np.absolute(k2d)
        k2d_angle = np.angle(k2d)
        
        # Top-Hat wavelet
        car_argk = np.logical_or((k2d_angle - self.theta) % (2*np.pi) < np.pi/(2*self.L),
                                       (k2d_angle - self.theta) % (2*np.pi) >= 2*np.pi - np.pi/(2*self.L)).astype(float)
        car_kmod = np.logical_and(k2d_mod >= xi - xi/3, k2d_mod < xi + xi/3).astype(float) # To avoid error and/or warning
        psi_f = c * car_argk * car_kmod
        psi_f = psi_f * np.exp(-1j * self.sigma * (k2d_x*self.nx + k2d_y*self.ny)) # Translation of the filter
        psi_f = self._periodization(psi_f)
        if self.fourier:
            self.data = psi_f.real if self.nx == 0 and self.ny == 0 else psi_f
        else:
            self.data = np.fft.ifft2(psi_f)
