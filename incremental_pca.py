# Title: incremental_pca.py
# Description: A function for PCA using the incremental approach in the onlinePCA R package
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)

##############################
# Imports
import sys

import numpy as np
import util
from util import subspace_error
import time
from scipy.linalg import eigh
import pylab as pl

##############################

class IncrementalPCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q
    f             -- Forgetting factor f, a number in (0,1)
    cython: bool
        whether to use computationally optimized cython function

    Methods:
    ====================
    fit_next()
    """

    def __init__(self, q, d, Uhat0=None, lambda0=None, tol=1e-7, f=None):

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
            self.lambda_ = np.abs(np.random.normal(0, 1, (q,))) / np.sqrt(q)

        if f is not None:
            assert (f > 0) and (f < 1), "The parameter f must be between 0 and 1"
        else:
            # Init as 1.0?
            f = 1.0 / self.t

        self.q = q
        self.d = d
        self.f = f
        self.tol = tol


    def fit_next(self, x):
        ''' Fit samples in online fashion

        Parameters
        ----------
        x_ : ndarray
            input sample


        Returns
        -------

        '''

        assert x.shape == (self.d,)
        self.t += 1
        self.f = 1.0 / self.t

        t, f, lambda_, Uhat, q, tol = self.t, self.f, self.lambda_, self.Uhat, self.q, self.tol

        lambda_ = (1 - f) * lambda_
        x = np.sqrt(f) * x
        # Project X into current estimate and check residual error
        Uhatx = Uhat.T.dot(x)
        x = x - Uhat.dot(Uhatx)
        normx = np.sqrt(x.dot(x))  # np.linalg.norm(x)

        # TODO: fix this atleast_2d for efficiency
        if (normx >= tol):
            lambda_ = np.concatenate((lambda_, [0]))
            Uhatx = np.concatenate((Uhatx, [normx]))
            Uhat = np.concatenate((Uhat, x[:, np.newaxis] / normx), 1)

        M = np.diag(lambda_) + np.outer(Uhatx, Uhatx.T)
        d, V = eigh(M, overwrite_a=True)

        idx = np.argsort(d)[::-1]
        lambda_ = d[idx]
        V = V[:, idx]
        lambda_ = lambda_[:q]
        Uhat = Uhat.dot(V[:, :q])

        self.Uhat = Uhat
        self.lambda_ = lambda_

    def get_components(self, orthogonalize=True):
        '''
        Extract components from object

        orthogonalize: bool
            whether to orthogonalize when computing the error

        Returns
        -------
        components: ndarray
        '''

        components = np.asarray(self.Uhat)
        if orthogonalize:
            components, _ = np.linalg.qr(components)

        return components


#%%
if __name__ == "__main__":
    # %%
    print('Testing IPCA')
    from util import generate_samples, get_scale_data_factor
    import pylab as pl
    # Parameters
    n_epoch = 1
    q = 20
    spiked_covariance_test = True
    if spiked_covariance_test:
        d,  n = 300, 5000
        X, U, sigma2 = generate_samples(q, n, d, method='spiked_covariance')

    else:
        X, U, sigma2 = generate_samples(q, n=None, d=None, method='real_data')
        d, n = X.shape

    method_scaling = None
    # method_scaling = 'norm'
    # method_scaling = 'norm_log'

    scale_factor = get_scale_data_factor(q, X, method=method_scaling)
    X, U, sigma2 = X * scale_factor, U, sigma2 * (scale_factor ** 2)
    #adjust eigenvalues magnitude according to how data is scaled
    lambda_1 = np.abs(np.random.normal(0, 1, (q,))) / np.sqrt(q)
    lambda_1 *= scale_factor**2

    Uhat0 = X[:, :q] / (X[:, :q] ** 2).sum(0)

    #%%
    errs = []
    ipca = IncrementalPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1)
    time_1 = time.time()
    for n_e in range(n_epoch):
        for sample in X.T:
            ipca.fit_next(sample)
            errs.append(subspace_error(ipca.get_components(),U[:,:q]))
    time_2 = time.time() - time_1
    pl.semilogy(errs)
    pl.xlabel('relative subspace error')
    pl.xlabel('samples')
    print('Elapsed time:' + str(time_2))
    print('Final subspace error:' + str(subspace_error(np.asarray(ipca.Uhat), U[:, :q])))
    pl.show()
    pl.pause(3)

