# Title: ccipca.py
# Description: A function for PCA using the Candid Covariance-Free Incremental PCA approach
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: J. Weng, Y. Zhang, and W. S. Hwang, "Candid covariance-free incremental principal component analysis", IEEE Trans. Pattern. Anal. Mach. Intell., vol 25, no. 8, pp. 1034-1040, Aug. 2003

##############################
# Imports
import sys

import numpy as np
from scipy.linalg import solve as solve
import util
from util import subspace_error
import time
import coord_update

##############################



class CCIPCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q
    ell           -- Amnesiac parameter (see reference)
    cython: True, False or 'auto'
        whether to use computationally optimized cython function

    Methods:
    ====================
    fit()
    fit_next()
    """

    def __init__(self, q, d, Uhat0=None, lambda0=None, ell=2, cython=True, in_place=False,verbose =False):
        #        if d>=2000 and cython:
        #            raise Exception('Cython Code is Limited to a 2000 dimensions array: use cython=False')
        if cython == 'auto':
            if d>1000:
                cython = False
            else:
                cython = True
        if verbose:
            print('Using Cython:' + str(cython))

        if Uhat0 is not None:
            assert Uhat0.shape == (d, q), "The shape of the initial guess Uhat0 must be (d,q)=(%d,%d)" % (d, q)
            self.Uhat = Uhat0.copy()

        else:
            # random initalization if not provided
            self.Uhat = np.random.normal(loc=0, scale=1 / d, size=(d, q))

        self.t = 1

        if lambda0 is not None:
            assert lambda0.shape == (q,), "The shape of the initial guess lambda0 must be (q,)=(%d,)" % (q)
            self.lambda_ = lambda0.copy()
        else:
            self.lambda_ = np.random.normal(0, 1, (q,)) / np.sqrt(q)

        self.q = q
        self.d = d
        self.ell = ell
        self.cython = cython
        self.v = np.zeros(d)
        self.in_place = in_place

    def fit(self, X):
        self.Uhat, self.lambda_ = coord_update.coord_update_total(X, X.shape[-1], self.d, np.double(self.t),
                                                                  np.double(self.ell), self.lambda_, self.Uhat, self.q,
                                                                  self.v)


    def fit_next(self, x_):
        if not self.in_place:
            x = x_.copy()
        else:
            x = x_

        assert x.shape == (self.d,)

        if self.cython:
            self.Uhat, self.lambda_ = coord_update.coord_update(x, self.d, np.double(self.t), np.double(self.ell),
                                                                self.lambda_, self.Uhat, self.q, self.v)
        else:
            t, ell, lambda_, Uhat, q = self.t, self.ell, self.lambda_, self.Uhat, self.q
            old_wt = max(1,t-ell) / (t+1)
            for i in range(q):
              # TODO: is the max okay?
                v = old_wt * lambda_[i] * Uhat[:, i] + (1-old_wt) * np.dot(x, Uhat[:,i]) * x
                lambda_[i] = np.sqrt(v.dot(v))  # np.linalg.norm(v)
                Uhat[:, i] = v / lambda_[i]
                x = x - np.dot(x, Uhat[:, i]) * Uhat[:, i]
            # print(old_wt)
            # if t > 50:
            #     exit(0)
            self.Uhat = Uhat
            self.lambda_ = lambda_
        self.t += 1

    def get_components(self):
        '''
        Extract components from object

        Returns
        -------
        components: ndarray
        '''

        components = np.asarray(self.Uhat)
        return components


#%%
if __name__ == "__main__":
    # %%
    print('Testing CCIPCA')
    from util import generate_samples
    import pylab as pl
    # Parameters
    n_epoch = 1
    d, q, n = 20, 5, 1000
    X, U, sigma2 = generate_samples(d, q, n)
    lambda_1 = np.random.normal(0, 1, (q,)) / np.sqrt(q)
    Uhat0 = X[:, :q] / (X[:, :q] ** 2).sum(0)
    # %%
    errs = []
    ccipca = CCIPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1, cython='auto', in_place=False)
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X.T:
            ccipca.fit_next(x)
            errs.append(subspace_error(ccipca.get_components(), U[:,:q]))
    time_2 = time.time() - time_1
    pl.semilogy(errs)
    pl.xlabel('relative subspace error')
    pl.xlabel('samples')
    print('Elapsed time:' + str(time_2))
    print('Final subspace error:' + str(subspace_error(ccipca.get_components(), U[:, :q])))
    pl.show()
    pl.pause(3)

