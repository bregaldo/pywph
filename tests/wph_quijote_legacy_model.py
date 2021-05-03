# -*- coding: utf-8 -*-

import numpy as np


def wph_quijote_legacy_model(J, L=8, dn=0, dj=None, dl=None,
                             alpha_list=[7, 0, 1, 2], # For A = 4
                             p_list=[0, 1, 2, 3],
                             classes=["S11", "S00", "S01", "C01", "Cphase", "L"]):
        """
        Build the specified WPH model (consistent with wph_quijote legacy code).
        A model is made of WPH moments, and scaling moments.
        The default model includes the following class of moments:
            - S11, S00, S01, C01, Cphase (all WPH moments)
            - L (scaling moments)
        These classes are defined in Allys+2020 and Regaldo-Saint Blancard+2021.
        Retun a list of wph_moment and scaling moments indices ordered as follows:
            - for wph_moments: list of lists of 8 elements corresponding to [j1, theta1, p1, j2, theta2, p2, n, a]
            - for scaling_moments: list of lists of 2 elements correponding to [j, p]
        Parameters
        ----------
        J : int
            J.
        L : int
            L. The default is 8.
        dn : int
            dn. The default is 0.
        p_list : list of int, optional
            For scaling moments ("L"), list of moments to compute for each low-pass filter.
        dj : int
            dj. The default is J - 1.
        dl : int
            dl. The default is L / 2.
        alpha_list : list
            alpha_list. The default is [7, 0, 1, 2] (consistent with A=4).
        classes : str or list of str, optional
            Classes of WPH/scaling moments constituting the model. Possibilities are: "S11", "S00", "C00", "S01", "C01", "Cphase", "L".
            The default is ["S11", "S00", "S01", "C01", "Cphase", "L"].
        Raises
        ------
        Exception
            DESCRIPTION.
        Returns
        -------
        wph_indices : array
            Ordering of dim=1 corresponds to [j1, theta1, p1, j2, theta2, p2, n, a].
        sm_indices : array
            Ordering of dim=1 corresponds to [j, p].
        """
        # Reordering of elements of classes
        if isinstance(classes, str): # Convert to list of str
            classes = [classes]
        classes_new = []
        for clas in ["S11", "S00", "C00", "S01", "C01", "Cphase", "L"]:
            if clas in classes:
                classes_new.append(clas)
        classes = classes_new
        
        wph_indices = []
        sm_indices = []
        
        # Default values for dj, dl, dn, alpha_list
        if dj is None:
            dj = J - 1 # We consider all possible pair of scales j1 < j2
        print('dj = ' + str(dj))
        if dl is None:
            dl = L // 2 # For C01 moments, we consider |t1 - t2| <= pi / 2
        print('dl = ' + str(dl))
        print('dn = ' + str(dn))
        
        for clas in classes:
            cnt = 0
            if clas == "S11":
                for j1 in range(J):
                    for t1 in range(2 * L):
                        dn_eff = min(J - 1 - j1, dn)
                        for n in range(dn_eff + 1):
                            if n == 0:
                                wph_indices.append([j1, t1, 1, j1, t1, 1, n, 0])
                                cnt += 1
                            else:
                                for a in alpha_list:
                                    wph_indices.append([j1, t1, 1, j1, t1, 1, n, a])
                                    cnt += 1
            elif clas == "S00":
                for j1 in range(J):
                    for t1 in range(2 * L):
                        dn_eff = min(J - 1 - j1, dn)
                        for n in range(dn_eff + 1):
                            if n == 0:
                                wph_indices.append([j1, t1, 0, j1, t1, 0, n, 0])
                                cnt += 1
                            else:
                                for a in alpha_list:
                                    wph_indices.append([j1, t1, 0, j1, t1, 0, n, a])
                                    cnt += 1
            elif clas == "C00": # Non default moments
                for j1 in range(J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, J)):
                        for t1 in range(2 * L):
                            for t2 in range(t1 - dl, t1 + dl + 1):
                                # No translation here by default
                                wph_indices.append([j1, t1, 0, j2, t2 % (2 * L), 0, 0, 0])
                                cnt += 1
            elif clas == "S01":
                for j1 in range(J):
                    for t1 in range(2 * L):
                        wph_indices.append([j1, t1, 0, j1, t1, 1, 0, 0])
                        cnt += 1
            elif clas == "C01":
                for j1 in range(J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, J)):
                        for t1 in range(2 * L):
                            for t2 in range(t1 - dl, t1 + dl + 1):
                                if t1 == t2:
                                    dn_eff = min(J - 1 - j2, dn)
                                    for n in range(dn_eff + 1):
                                        if n == 0:
                                            wph_indices.append([j1, t1, 0, j2, t2, 1, n, 0])
                                            cnt += 1
                                        else:
                                            for a in alpha_list:
                                                wph_indices.append([j1, t1, 0, j2, t2, 1, n, a])
                                                cnt += 1
                                else:
                                    wph_indices.append([j1, t1, 0, j2, t2 % (2 * L), 1, 0, 0])
                                    cnt += 1
            elif clas == "Cphase":
                for j1 in range(J):
                    for j2 in range(j1 + 1, min(j1 + 1 + dj, J)):
                        for t1 in range(2 * L):
                            dn_eff = min(J - 1 - j2, dn)
                            for n in range(dn_eff + 1):
                                if n == 0:
                                    wph_indices.append([j1, t1, 1, j2, t1, 2 ** (j2 - j1), n, 0])
                                    cnt += 1
                                else:
                                    for a in alpha_list: # Factor 2 needed even for real data
                                        wph_indices.append([j1, t1, 1, j2, t1, 2 ** (j2 - j1), n, a])
                                        cnt += 1
            elif clas == "L":
                # Scaling moments
                for j in range(2, J - 1):
                    for p in p_list:
                        sm_indices.append([j, p])
                        cnt += 1
            else:
                raise Exception(f"Unknown class of moments: {clas}")
        
        return wph_indices, sm_indices
    