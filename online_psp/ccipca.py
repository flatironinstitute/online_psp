# Title: ccipca.py
# Description: A function for PCA using the Candid Covariance-Free Incremental PCA approach
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: J. Weng, Y. Zhang, and W. S. Hwang, "Candid covariance-free incremental principal component analysis",
#           IEEE Trans. Pattern. Anal. Mach. Intell., vol 25, no. 8, pp. 1034-1040, Aug. 2003

##############################
# Imports
import numpy as np
from online_psp.coord_update import coord_update_trans


##############################


class CCIPCA:
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size D-by-K
    sigma2_0      -- Initial guess for the eigenvalues vector sigma2, must be of size K
    ell           -- Amnesiac parameter (see reference)
    cython        -- Whether to use computationally optimized cython functionality (True, False or 'auto')

    Methods:
    ====================
    fit_next()
    """

    def __init__(self, K, D, Uhat0=None, sigma2_0=None, ell=2, cython='auto'):

        if cython == 'auto':
            if D >= 1000:
                cython = False
            else:
                cython = True

        if Uhat0 is not None:
            assert Uhat0.shape == (D, K), "The shape of the initial guess Uhat0 must be (D,K)=(%d,%d)" % (D, K)
            self.Uhat = Uhat0.T.copy()

        else:
            # random initalization if not provided
            self.Uhat = np.random.normal(loc=0, scale=1 / D, size=(D, K)).T

        self.t = 1

        if sigma2_0 is not None:
            assert sigma2_0.shape == (K,), "The shape of the initial guess sigma2_0 must be (K,)=(%d,)" % (K)
            self.sigma2 = sigma2_0.copy()
        else:
            self.sigma2 = np.abs(np.random.normal(0, 1, (K,)) / np.sqrt(K))

        self.K = K
        self.D = D
        self.ell = ell
        self.cython = cython
        self.v = np.zeros(D)

        if cython:
            self.fit_next = self.fit_next_cython
        else:
            self.fit_next = self.fit_next_no_cython

    def fit_next_cython(self, x_):
        x = x_.copy()
        self.Uhat, self.sigma2 = coord_update_trans(x, self.D, np.double(
            self.t), np.double(self.ell), self.sigma2, self.Uhat, self.K, self.v)
        self.t += 1

    def fit_next_no_cython(self, x_):
        x = x_.copy()
        t, ell, sigma2, Uhat, K = self.t, self.ell, self.sigma2, self.Uhat, self.K
        old_wt = max(1, t - ell) / (t + 1)

        for k in range(K):
            v = old_wt * sigma2[k] * Uhat[k, :] + (1 - old_wt) * np.dot(x, Uhat[k, :]) * x

            sigma2[k]  = np.linalg.norm(v)
            Uhat[k, :] = v / sigma2[k]

            x = x - np.dot(x, Uhat[k, :]) * Uhat[k, :]

        self.Uhat = Uhat
        self.sigma2 = sigma2
        self.t += 1

    def get_components(self, orthogonalize=True):
        '''
        Extract components from object

        orthogonalize: bool
            whether to orthogonalize the components before returning

        Returns
        -------
        components: ndarray
        '''

        components = np.asarray(self.Uhat.T)
        if orthogonalize:
            components, _ = np.linalg.qr(components)

        return components
