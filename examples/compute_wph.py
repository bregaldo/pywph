# -*- coding: utf-8 -*-

import pywph as pw
import astropy.io as fits

M, N = 512, 512
J = 8
L = 8
dn = 0
norm = "auto"

data = fits.open('data/I_1.fits')[0].data
