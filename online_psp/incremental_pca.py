# Title: incremental_pca.py
# Description: A function for PCA using the incremental PCA approach of Arora et al.
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: R. Arora, A. Cotter, K. Livescu, and N. Srebro, “Stochastic optimizationfor  PCA  and  PLS,”
#            in 2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton),
#            Oct 2012, pp. 861–868

##############################
# Imports

import numpy as np
from util import subspace_error
import time
from scipy.linalg import eigh


##############################

class IPCA:
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size D-by-K
    sigma2_0      -- Initial guess for the eigenvalues vector sigma2, must be of size K
    tol           -- A tolerance that allows optionally only updating components when new data is sufficiently
                     outside of current subspace

    Methods:
    ====================
    fit_next()
    """

    def __init__(self, K, D, Uhat0=None, sigma2_0=None, tol=1e-7):

        if Uhat0 is not None:
            assert Uhat0.shape == (D, K), "The shape of the initial guess Uhat0 must be (D,K)=(%d,%d)" % (D, K)
            self.Uhat = Uhat0.copy()

        else:
            # random initalization if not provided
            self.Uhat = np.random.normal(loc=0, scale=1 / D, size=(D, K))

        self.t = 1

        if sigma2_0 is not None:
            assert sigma2_0.shape == (K,), "The shape of the initial guess lambda0 must be (K,)=(%d,)" % (K)
            self.sigma2 = sigma2_0.copy()
        else:
            self.sigma2 = np.abs(np.random.normal(0, 1, (K,))) / np.sqrt(K)

        self.K = K
        self.D = D
        self.f = 1.0 / self.t
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

        assert x.shape == (self.D,)
        self.t += 1
        self.f = 1.0 / self.t

        t, f, sigma2, Uhat, K, tol = self.t, self.f, self.sigma2, self.Uhat, self.K, self.tol

        sigma2 = (1 - f) * sigma2
        x = np.sqrt(f) * x

        # Project X into current estimate and check residual error
        y     = Uhat.T.dot(x)
        x     = x - Uhat.dot(y)
        normx = np.sqrt(x.dot(x))

        if (normx >= tol):
            sigma2 = np.concatenate((sigma2, [0]))
            y = np.concatenate((y, [normx]))
            Uhat = np.concatenate((Uhat, x[:, np.newaxis] / normx), 1)

        M = np.diag(sigma2) + np.outer(y, y.T)
        d, V = eigh(M, overwrite_a=True)

        idx    = np.argsort(d)[::-1]
        sigma2 = d[idx][:K]
        V      = V[:, idx]
        Uhat   = Uhat.dot(V[:, :K])

        self.Uhat   = Uhat
        self.sigma2 = sigma2

    def get_components(self, orthogonalize=True):
        '''
        Extract components from object

        orthogonalize: bool
            whether to orthogonalize the components before returning

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
    print('Testing IPCA...')
    from util import generate_samples
    import pylab as pl

    #----------
    # Parameters
    #----------
    n_epoch = 2
    K       = 50
    D, N    = 500, 1000
    # ----------

    X, U, sigma2 = generate_samples(K, N, D, method='spiked_covariance', scale_data=True)

    # Initial guess
    sigma2_0 = lambda0 = np.zeros(K)
    Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0))

    errs = []
    ipca = IPCA(K, D, Uhat0=Uhat0, sigma2_0=sigma2_0)
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X.T:
            ipca.fit_next(x)
            errs.append(subspace_error(ipca.get_components(), U[:, :K]))
    time_2 = time.time() - time_1

    # Plotting...
    print('Elapsed time: ' + str(time_2))
    print('Final subspace error: ' + str(subspace_error(ipca.get_components(), U[:, :K])))

    pl.semilogy(errs)
    pl.ylabel('Relative subspace error')
    pl.xlabel('Samples (t)')
    pl.show()

    print('Test complete!')
