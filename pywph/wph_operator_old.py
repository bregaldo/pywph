# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import os
import torch
import sys

from .wph_old import WPH_old

from .stats.wph_syntheses.wph_operator import PhaseHarmonics2d


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
               
        return WPH_old(wph, self.stat_params, self.stat_op.wph_by_chunk)