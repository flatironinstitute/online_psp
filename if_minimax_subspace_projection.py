# Title: if_minimax_subspace_projection.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax algorithm of Pehlevan et al. with added Sherman-Morrison formula
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


class IF_minimax_PCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau           -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    Minv0         -- Initial guess for the inverse of the lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d

    Methods:
    ========
    fit_next()

    Output:
    ====================
    Minv -- Final iterate of the inverse lateral weight matrix, of size q-by-q (sometimes)
    W    -- Final iterate of the forward weight matrix, of size q-by-d (sometimes)
    """

    def __init__(self, q, d, tau=0.5, Minv0=None, W0=None, learning_rate=None):

        if Minv0 is not None:
            assert Minv0.shape == (q, q), "The shape of the initial guess Minv0 must be (q,q)=(%d,%d)" % (q, q)
            Minv = Minv0
        else:
            Minv = np.eye(q)

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
        self.Minv = Minv
        self.W = W
        # variable to allocate memory and optimize outer product
        self.outer_W = np.empty_like(W)
        self.outer_Minv = np.empty_like(Minv)



    def fit_next(self, x):

        assert x.shape == (self.d,)

        t, tau, W, Minv, q = self.t, self.tau, self.W, self.Minv, self.q

        y = np.dot(Minv, W.dot(x))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = self.eta(t)

        np.outer(2 * step * y, x, self.outer_W)
        W = (1 - 2 * step) * W + self.outer_W

        # M <- M + eta(self.t)/tau * (y*y' - M), using SMW
        step = step / tau

        Minv = Minv / (1 - step)
        z = Minv.dot(y)
        c = step / (1 + step * np.dot(z, y))
        np.outer(c * z, z.T, self.outer_Minv)
        Minv = Minv - self.outer_Minv

        self.Minv = Minv
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

        components = np.asarray(self.Minv.dot(self.W).T)
        if orthogonalize:
            components, _ = np.linalg.qr(components)

        return components


#%%
if __name__ == "__main__":
#%%
    print('Testing IF_MINMAX_PROJECTION')
    from util import generate_samples, get_scale_data_factor
    import pylab as pl

    # Parameters
    n_epoch = 1
    tau = 0.5
    q = 100
    spiked_covariance_test = True
    scale_data = True
    if spiked_covariance_test:
        d,  n = 1000, 5000
        X, U, sigma2 = generate_samples(q, n, d, method='spiked_covariance', scale_data=scale_data)

    else:
        X, U, sigma2 = generate_samples(q, n=None, d=None, method='real_data', scale_data=scale_data)
        d, n = X.shape
    # adjust eigenvalues magnitude according to how data is scaled
    lambda_1 = np.abs(np.random.normal(0, 1, (q,))) / np.sqrt(q)
    Uhat0 = X[:, :q] / np.sqrt((X[:, :q] ** 2).sum(0))
    # %%
    errs = []
    if_mm_pca = IF_minimax_PCA_CLASS(q, d, W0=Uhat0.T, Minv0=None, tau=tau)
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X.T:
            if_mm_pca.fit_next(x)
            errs.append(subspace_error(if_mm_pca.get_components(), U[:, :q]))
    time_2 = time.time() - time_1
    pl.semilogy(errs)
    pl.xlabel('relative subspace error')
    pl.xlabel('samples')
    print('Elapsed time:' + str(time_2))
    print('Final subspace error:' + str(subspace_error(if_mm_pca.get_components(), U[:, :q])))
    pl.show()
    pl.pause(3)
