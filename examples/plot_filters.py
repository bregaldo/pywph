# -*- coding: utf-8 -*-

import pywph as pw
import matplotlib.pyplot as plt
import numpy as np

M, N = 512, 512

j = 6
theta = np.pi / 4
k0 = 0.85 * np.pi

w = pw.BumpSteerableWavelet(M, N, j, theta, k0=k0)
g = pw.GaussianFilter(M, N, j)

# Bump-steerable filter plot (the wavelet is centered)
fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
axs[0].imshow(np.fft.fftshift(w.data.real), origin='lower')
axs[1].imshow(np.fft.fftshift(w.data.imag), origin='lower')
fig.subplots_adjust(top=0.95, bottom=0.091, left=0.056, right=0.97, hspace=0.19, wspace=0.046)
fig.show()

# Gaussian filter plot (the filter is centered)
fig, axs = plt.subplots(1, 1, figsize=(3.5, 3))
axs.imshow(np.fft.fftshift(g.data), origin='lower')
fig.subplots_adjust(top=0.949, bottom=0.129, left=0.089, right=0.957, hspace=0.2, wspace=0.2)
fig.show()
