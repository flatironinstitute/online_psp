# Title: minimax_subspace_projection.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax algorithm of Pehlevan et al.
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Reference: (Pehlevan et al, Neural Computation, 2017) and working notes

##############################
# Imports
import sys

import numpy as np
import util
from util import subspace_error
import time


##############################
##############################


def eta(t):
    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated

    Output:
    ====================
    step -- learning rate at time t
    """

    return 1.0 / (2 * t + 5)


class Minimax_PCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau           -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    M0            -- Initial guess for the lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d

    Methods:
    ========
    fit_next()

    Output:
    ====================
    M    -- Final iterate of the lateral weight matrix, of size q-by-q (sometimes)
    W    -- Final iterate of the forward weight matrix, of size q-by-d (sometimes)
    """

    def __init__(self, q, d, tau=0.5, M0=None, W0=None, learning_rate=None):

        if M0 is not None:
            assert M0.shape == (q, q), "The shape of the initial guess Minv0 must be (q,q)=(%d,%d)" % (q, q)
            M = M0
        else:
            M = np.eye(q)

        if W0 is not None:
            assert W0.shape == (q, d), "The shape of the initial guess W0 must be (q,d)=(%d,%d)" % (q, d)
            W = W0
        else:
            W = np.random.normal(0, 1.0 / np.sqrt(d), size=(q, d))

        if learning_rate is not None:
            self.eta = learning_rate
        else:
            self.eta = eta
        self.t = 0

        self.q = q
        self.d = d
        self.tau = tau
        self.M = M
        self.W = W
        # variable to allocate memory and optimize outer product
        self.outer_W = np.empty_like(W)
        self.outer_M = np.empty_like(M)

    def fit_next(self, x):

        assert x.shape == (self.d,)

        t, tau, W, M, q = self.t, self.tau, self.W, self.M, self.q

        y = np.linalg.solve(M, W.dot(x))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = self.eta(t)

        np.outer(2 * step * y, x, self.outer_W)
        W = (1 - 2 * step) * W + self.outer_W

        # M <- M + eta(self.t)/tau * (y*y' - M)
        step = step / tau
        np.outer(step * y, y, self.outer_M)
        M = (1 - step) * M + self.outer_M

        self.M = M
        self.W = W

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

        components = np.asarray(np.linalg.solve(self.M, self.W).T)
        if orthogonalize:
            components, _ = np.linalg.qr(components)

        return components


# %%
if __name__ == "__main__":
    # %%
    print('Testing MINMAX_PROJECTION')
    from util import generate_samples, get_scale_data_factor
    import pylab as pl

    # Parameters
    n_epoch = 1
    tau = 0.5
    q = 100
    spiked_covariance_test = True
    scale_data = True
    if spiked_covariance_test:
        d, n = 1000, 5000
        X, U, sigma2 = generate_samples(q, n, d, method='spiked_covariance', scale_data=scale_data)

    else:
        X, U, sigma2 = generate_samples(q, n=None, d=None, method='real_data', scale_data=scale_data)
        d, n = X.shape
    # adjust eigenvalues magnitude according to how data is scaled
    lambda_1 = np.abs(np.random.normal(0, 1, (q,))) / np.sqrt(q)
    Uhat0 = X[:, :q] / np.sqrt((X[:, :q] ** 2).sum(0))
    # %%
    errs = []
    mm_pca = Minimax_PCA_CLASS(q, d, W0=Uhat0.T, M0=None, tau=tau)
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X.T:
            mm_pca.fit_next(x)
            errs.append(subspace_error(mm_pca.get_components(), U[:, :q]))
    time_2 = time.time() - time_1
    pl.semilogy(errs)
    pl.xlabel('relative subspace error')
    pl.xlabel('samples')
    print('Elapsed time:' + str(time_2))
    print('Final subspace error:' + str(subspace_error(mm_pca.get_components(), U[:, :q])))
    pl.show()
