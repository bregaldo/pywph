# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import torch
import sys

from .utils import to_numpy


class WPH:
    """
    Wavelet Phase Harmonics coefficients container.
    """
    
    def __init__(self, coeffs, wph_coeffs_indices, sm_coeffs_indices=np.array([]), J=None, L=None, A=None):
        """
        Constructor.
        Converts and store input coefficients and corresponding indices to numpy arrays.

        Parameters
        ----------
        coeffs : list or np.ndarray or torch.tensor
            Input coefficients.
        wph_coeffs_indices : list or np.ndarray or torch.tensor
            Indices for the WPH moments. Order: [j1, theta1, p1, j2, theta2, p2, n, alpha, pseudo].
        sm_coeffs_indices : list or np.ndarray or torch.tensor
            Indices for the scaling moments. Order: [j, p]. The default is np.array([]).

        Returns
        -------
        None.

        """
        # Consistency check
        if coeffs.shape[-1] != wph_coeffs_indices.shape[0] + sm_coeffs_indices.shape[0] \
            and coeffs.shape[-1] != wph_coeffs_indices.shape[0] + 2 * sm_coeffs_indices.shape[0]:
            raise Exception("Shape inconsistency between coeffs and corresponding indices (wph_coeffs_indices, sm_coeffs_indices).")
        
        # Detect if scaling moments were computed on complex maps or not
        if coeffs.shape[-1] != wph_coeffs_indices.shape[0] + sm_coeffs_indices.shape[0]:
            self.cplx = True
        else:
            self.cplx = False
        
        # Convert and store coefficients
        coeffs = to_numpy(coeffs).copy()
        self.wph_coeffs = coeffs[..., :wph_coeffs_indices.shape[0]] # WPH moments estimates
        self.sm_coeffs = coeffs[..., wph_coeffs_indices.shape[0]:] # Scaling moments estimates
        if self.cplx:
            self.sm_coeffs = self.sm_coeffs.reshape(self.sm_coeffs.shape[:-1] + (2, -1))
        
        # Store indices
        self.wph_coeffs_indices = to_numpy(wph_coeffs_indices).copy()
        self.sm_coeffs_indices = to_numpy(sm_coeffs_indices).copy()
        
        # Lexicographical reordering
        self.reorder()
        
        if J is not None:
            self.J = J
        else: # Auto-detection of J
            self.J = max(self.wph_coeffs_indices[:, 0].max(), self.wph_coeffs_indices[:, 3].max()) + 1
        if L is not None:
            self.L = L
        else: # Auto-detection of L
            self.L = (max(self.wph_coeffs_indices[:, 1].max(), self.wph_coeffs_indices[:, 4].max()) + 1) // (1 + self.cplx)
            
        if A is not None:
            self.A = A
        else: # Auto-detection of A
            self.A = (self.wph_coeffs_indices[:, 7].max() + 1) // 2

    def reorder(self):
        """
        Lexicographical reordering of the coefficients:
        - for WPH moments on [j1, t1, p1, j2, t2, p2, n, a, pseudo]
        - for scaling moments on [j, p]
        Returns
        -------
        None.

        """
        # WPH coefficients
        if self.wph_coeffs_indices.shape[0] != 0:
            indices = np.lexsort(self.wph_coeffs_indices.T[::-1, :])
            wph_coeffs_copy = self.wph_coeffs.copy()
            wph_coeffs_indices_copy = self.wph_coeffs_indices.copy()
            for i in range(self.wph_coeffs.shape[-1]):
                self.wph_coeffs[..., i] = wph_coeffs_copy[..., indices[i]]
                self.wph_coeffs_indices[i] = wph_coeffs_indices_copy[indices[i]]
            
        # Scaling moments coefficients
        if self.sm_coeffs_indices.shape[0] != 0:
            indices = np.lexsort(self.sm_coeffs_indices.T[::-1, :])
            sm_coeffs_copy = self.sm_coeffs.copy()
            sm_coeffs_indices_copy = self.sm_coeffs_indices.copy()
            for i in range(self.sm_coeffs.shape[-1]):
                self.sm_coeffs[..., i] = sm_coeffs_copy[..., indices[i]]
                self.sm_coeffs_indices[i] = sm_coeffs_indices_copy[indices[i]]
            
    def _filter_args(self, clas="", j=None, p=None, j1=None, t1=None, p1=None, j2=None, t2=None, p2=None, n=None, a=None, sm=False, pseudo=None):
        """
        Internal function for coefficients selection.
        """
        if sm or clas == "L":
            filtering = np.ones(self.sm_coeffs_indices.shape[0], np.bool)
            if j is not None:
                filtering = np.logical_and(filtering, self.sm_coeffs_indices[:, 0] == j)
            if j1 is not None: # j and j1 are aliases in this case
                filtering = np.logical_and(filtering, self.sm_coeffs_indices[:, 0] == j1)
            if p is not None:
                filtering = np.logical_and(filtering, self.sm_coeffs_indices[:, 1] == p)
            if p1 is not None: # p and p1 are aliases in this case
                filtering = np.logical_and(filtering, self.sm_coeffs_indices[:, 1] == p1)
        else:
            filtering = np.ones(self.wph_coeffs_indices.shape[0], np.bool)
            if j1 is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] == j1)
            if t1 is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 1] == t1)
            if p1 is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == p1)
            if j2 is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 3] == j2)
            if t2 is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 4] == t2)
            if p2 is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == p2)
            if n is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 6] == n)
            if a is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 7] == a)
            if pseudo is not None:
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 8] == int(pseudo))
            
            # Selection per class
            if clas == "S11":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] == self.wph_coeffs_indices[:, 3]) # j1 == j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 1) # p1 == 1
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 1) # p2 == 1
            elif clas == "S00":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] == self.wph_coeffs_indices[:, 3]) # j1 == j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 0) # p1 == 0
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 0) # p2 == 0
            elif clas == "S01":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] == self.wph_coeffs_indices[:, 3]) # j1 == j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 0) # p1 == 0
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 1) # p2 == 1
            elif clas == "S10":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] == self.wph_coeffs_indices[:, 3]) # j1 == j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 1) # p1 == 1
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 0) # p2 == 0
            elif clas == "C00":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] != self.wph_coeffs_indices[:, 3]) # j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 0) # p1 == 0
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 0) # p2 == 0
            elif clas == "C01":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] != self.wph_coeffs_indices[:, 3]) # j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 0) # p1 == 0
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 1) # p2 == 1
            elif clas == "C10":
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] != self.wph_coeffs_indices[:, 3]) # j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 1) # p1 == 1
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 0) # p2 == 0
            elif clas == "Cphase": # Cphase and all extra moments with p1 = 1, p2 != 0 and j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] != self.wph_coeffs_indices[:, 3]) # j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] == 1) # p1 == 1
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] != 0) # p2 != 0
            elif clas == "Cphase_inv": # Cphase and all extra moments with p1 != 0, p2 == 1 and j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 0] != self.wph_coeffs_indices[:, 3]) # j1 != j2
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 2] != 0) # p1 != 0
                filtering = np.logical_and(filtering, self.wph_coeffs_indices[:, 5] == 1) # p2 == 1
            elif clas == "":
                pass
            else:
                raise Exception("Unknown class of coefficients!")
        return filtering
    
    def get_coeffs(self, clas="", j=None, p=None, j1=None, t1=None, p1=None, j2=None, t2=None, p2=None, n=None, a=None, sm=False, pseudo=None):
        """
        Selection of coefficients.
        To select scaling moments coefficients use sm=True, otherwise select WPH coefficients.

        Parameters
        ----------
        clas : str, optional
            Class of coefficients. The default is "".
        j : int, optional
            DESCRIPTION. The default is None.
        p : int, optional
            DESCRIPTION. The default is None.
        j1 : int, optional
            DESCRIPTION. The default is None.
        t1 : int, optional
            DESCRIPTION. The default is None.
        p1 : int, optional
            DESCRIPTION. The default is None.
        j2 : int, optional
            DESCRIPTION. The default is None.
        t2 : int, optional
            DESCRIPTION. The default is None.
        p2 : int, optional
            DESCRIPTION. The default is None.
        n : int, optional
            DESCRIPTION. The default is None.
        a : int, optional
            DESCRIPTION. The default is None.
        sm : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        array
            Selected coefficients.
        array
            Corresponding indices.

        """
        filtering = self._filter_args(clas=clas, j=j, p=p, j1=j1, t1=t1, p1=p1, j2=j2, t2=t2, p2=p2, n=n, a=a, sm=sm, pseudo=pseudo)
        if sm or clas == "L":
            return self.sm_coeffs[..., filtering], self.sm_coeffs_indices[filtering, :]
        else:
            return self.wph_coeffs[..., filtering], self.wph_coeffs_indices[filtering, :]
        
    def to_isopar(self):
        """
        Comptute isotropic and parity invariant WPH coefficients.

        Returns
        -------
        self

        """
        indices_cnt = {}
        wph_isopar = {}
        
        def periodic_distance(t1, t2, per):
            if t2 > t1:
                return min(t2 - t1, t1 - t2 + per)
            else:
                return min(t1 - t2, t2 - t1 + per)
        
        # Filling
        for i in range(self.wph_coeffs_indices.shape[0]):
            j1, t1, p1, j2, t2, p2, n, a, pseudo = tuple(self.wph_coeffs_indices[i])
            if pseudo == 0:
                dt = periodic_distance(t1, t2, 2 * self.L)
                if (j1, 0, p1, j2, dt, p2, n, a, pseudo) in indices_cnt.keys():
                    indices_cnt[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] += 1
                    wph_isopar[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] += self.wph_coeffs[..., i]
                else:
                    indices_cnt[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] = 1
                    wph_isopar[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] = self.wph_coeffs[..., i]
            else:
                dt = periodic_distance(t1, t2 - self.L, 2 * self.L)
                if (j1, 0, p1, j2, dt, p2, n, a, pseudo) in indices_cnt.keys():
                    indices_cnt[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] += 1
                    wph_isopar[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] += self.wph_coeffs[..., i]
                else:
                    indices_cnt[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] = 1
                    wph_isopar[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] = self.wph_coeffs[..., i]
                
        # Conversion into numpy arrays
        indices = []
        wph_isopar_list = []
        for key in indices_cnt.keys():
            indices.append(key)
            wph_isopar_list.append(wph_isopar[key] / indices_cnt[key])
        indices = np.array(indices)
        wph_isopar = np.moveaxis(np.array(wph_isopar_list), 0, -1)
        
        # Reordering and save
        self.wph_coeffs_indices = indices
        self.wph_coeffs = wph_isopar
        self.reorder()
        return self
        
    def _assert_consistent_with(self, other):
        """
        Stops the program if other is a WPH object that is not consistent with self for comparison and arithmetic operations.
        """
        ret = np.array_equal(self.wph_coeffs_indices, other.wph_coeffs_indices)
        ret &= np.array_equal(self.sm_coeffs_indices, other.sm_coeffs_indices)
        ret &= self.cplx == other.cplx
        assert ret, "Inconsistency between self and other!"
    
    def __add__(self, other):
        """
        Addition of two WPH objects.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        cpy : TYPE
            DESCRIPTION.

        """
        cpy = copy.deepcopy(self)
        if isinstance(other, WPH):
            self._assert_consistent_with(other)
            cpy.wph_coeffs += other.wph_coeffs
            cpy.sm_coeffs += other.sm_coeffs
        else:
            cpy.wph_coeffs += other
            cpy.sm_coeffs += other
        return cpy
    
    def __sub__(self, other):
        """
        Substraction of two WPH objects.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        cpy : TYPE
            DESCRIPTION.

        """
        cpy = copy.deepcopy(self)
        if isinstance(other, WPH):
            self._assert_consistent_with(other)
            cpy.wph_coeffs -= other.wph_coeffs
            cpy.sm_coeffs -= other.sm_coeffs
        else:
            cpy.wph_coeffs -= other
            cpy.sm_coeffs -= other
        return cpy

    def __pow__(self, power):
        """
        Exponentiation of self.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        cpy : TYPE
            DESCRIPTION.

        """
        cpy = copy.deepcopy(self)
        cpy.wph_coeffs = cpy.wph_coeffs ** power
        cpy.sm_coeffs = cpy.sm_coeffs ** power
        return cpy
    
    def __truediv__(self, other):
        """
        Division of self by other.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        cpy : TYPE
            DESCRIPTION.

        """
        cpy = copy.deepcopy(self)
        if isinstance(other, WPH):
            self._assert_consistent_with(other)
            cpy.wph_coeffs /= other.wph_coeffs
            cpy.sm_coeffs /= other.sm_coeffs
        else:
            cpy.wph_coeffs /= other
            cpy.sm_coeffs /= other
        return cpy
    
    def __abs__(self):
        """
        Absolute of self coefficients.

        Parameters
        ----------
        other : TYPE
            DESCRIPTION.

        Returns
        -------
        cpy : TYPE
            DESCRIPTION.

        """
        cpy = copy.deepcopy(self)
        cpy.wph_coeffs = np.abs(cpy.wph_coeffs)
        cpy.sm_coeffs = np.abs(cpy.sm_coeffs)
        return cpy
    
    def _plot(self, axis, clas):
        pass
