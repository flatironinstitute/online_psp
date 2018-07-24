# Title: letipca.py
# TODO: update me
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


##############################


class LETIPCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q
    ell           -- Amnesiac parameter (see reference)

    Methods:
    ====================
    fit_next()
    """

    def __init__(self, q, d, Uhat0=None, lambda0=None, ell=2, in_place=False, verbose=False):
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

        self.q = q
        self.d = d
        self.ell = ell
        self.in_place = in_place
        self.tol = 1e-12
        self.max_its = 30

    def fit_next(self, x_):
        if not self.in_place:
            x = x_.copy()
        else:
            x = x_

        assert x.shape == (self.d,)

        t, ell, lambda_, Uhat, q = self.t, self.ell, self.lambda_, self.Uhat, self.q
        tol, max_its = self.tol, self.max_its
        old_wt = max(1, t - ell) / (t + 1)
        Uhat_new = Uhat.copy()
        lambda_new = lambda_.copy()
        for i in range(q):
            v_old = np.zeros_like(Uhat_new[:, i]) + 100
            for _ in range(max_its):
                Uhat_new[:, i] /= lambda_new[i]
                if np.linalg.norm(v_old - Uhat_new[:, i]) < tol:
                    break
                old_term = lambda_[i] * Uhat[:, i]
                data_term = np.dot(x, Uhat_new[:, i]) * x
                if i > 0:
                    term2 = lambda_new[:i] * \
                            np.dot(Uhat_new[:, :i].T, Uhat_new[:, i])
                    term2 = Uhat_new[:, :i].dot(term2)

                    term1 = lambda_[:i] * np.dot(Uhat[:, :i].T, Uhat_new[:, i])
                    term1 = Uhat[:, :i].dot(term1)
                else:
                    term1 = 0
                    term2 = 0

                v_old = Uhat_new[:, i]
                Uhat_new[:, i] = old_wt * term1 - term2 + \
                                 (1 - old_wt) * data_term + old_wt * old_term
                lambda_new[i] = np.linalg.norm(Uhat_new[:, i])

        self.Uhat = Uhat_new.copy()
        self.lambda_ = lambda_new
        self.t += 1

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


# %%
if __name__ == "__main__":
    # %%
    print('Testing LETIPCA')
    from util import generate_samples
    import pylab as pl

    # Parameters
    n_epoch = 1
    q = 50
    spiked_covariance_test = True
    if spiked_covariance_test:
        d, n = 1000, 1000
        X, U, sigma2 = generate_samples(
            q, n, d, method='spiked_covariance', scale_data=False)

    else:
        X, U, sigma2 = generate_samples(
            q, n=None, d=None, method='real_data', scale_data=True)
        d, n = X.shape

    # adjust eigenvalues magnitude according to how data is scaled
    lambda_1 = np.abs(np.random.normal(0, 1, (q,))) / np.sqrt(q)
    Uhat0 = X[:, :q] / np.sqrt((X[:, :q] ** 2).sum(0))
    # %%
    errs = []
    letipca = LETIPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1, in_place=False)
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X.T:
            letipca.fit_next(x)
            errs.append(subspace_error(
                letipca.get_components(), U[:, :q]))
    time_2 = time.time() - time_1

    pl.semilogy(errs)
    pl.xlabel('relative subspace error')
    pl.xlabel('samples')
    print('Elapsed time:' + str(time_2))
    print('Final subspace error:' +
          str(subspace_error(letipca.get_components(), U[:, :q])))
    pl.show()
    pl.pause(3)
