# Title: if_minimax_subspace_projection.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax algorithm of Pehlevan et al. with added Sherman-Morrison formula
# Author: Victor Minden (vminden@flatironinstitute.org)
# Reference: (Pehlevan et al, Neural Computation, 2017) and working notes

##############################
# Imports
import sys
sys.path.append('/mnt/home/agiovann/SOFTWARE/online_pca')
import numpy as np
import util
from util import subspace_error
import time
try:
    profile
except:
    def profile(a): return a
##############################
##############################


# from numba.decorators import autojit
#
# @autojit
# def outer_numba(a, b):
#     m = a.shape[0]
#     n = b.shape[0]
#     result = np.empty((m, n), dtype=np.float)
#     for i in range(m):
#         for j in range(n):
#             result[i, j] = a[i]*b[j]
#     return result
#
def eta(t):

    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated

    Output:
    ====================
    step -- learning rate at time t
    """

    return 1.0 / (2*t + 5)

def _iterate(X, Minv, W, tau, n_its, n):
    for t in range(n_its):
        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = np.dot(Minv, W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W +  np.outer(2*step *y,X[:,j])

        # M <- M + eta(t)/tau * (y*y' - M), using SMW
        step = step/tau

        Minv = Minv / (1-step)
        z    = Minv.dot(y)
        c    = step /(1 + step*np.dot(z,y))
        Minv = Minv -  np.outer(c*z, z.T)
        # M    = (1-step) * M + step * np.outer(y,y)

    return Minv,W



def _iterate_and_compute_errors(X, Minv, W, tau, n_its, n, error_options):
    errs = util.initialize_errors(error_options, n_its)

    for t in range(n_its):
        # Record error
        Uhat    = Minv.dot(W).T

        util.compute_errors(error_options, Uhat, t, errs)

        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = np.dot(Minv, W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W +  np.outer(2*step *y,X[:,j])

        # M <- M + eta(t)/tau * (y*y' - M), using SMW
        step = step/tau

        Minv = Minv / (1-step)
        z    = Minv.dot(y)
        c    = step /(1 + step*np.dot(z,y))
        Minv = Minv -  np.outer(c*z, z.T)
        # if not t % 1000:
        #     print((np.linalg.norm(Minv,ord=2), np.linalg.norm(np.linalg.inv(Minv),ord=2)))
        # M    = (1-step) * M + step * np.outer(y,y)


    return errs



class IF_minimax_PCA_Class:
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

        def __init__(self, q, d, tau=0.5, Minv0=None, W0=None):


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


            self.t = 0

            self.q = q
            self.d = d
            self.tau = tau
            self.Minv = Minv
            self.W = W
            # variable to allocate memory and optimize outer product
            self.outer_W = np.empty_like(W)
            self.outer_Minv = np.empty_like(Minv)


        def fit(self, X):
            raise Exception("Not Implemented")

        @profile
        def fit_next(self, x_, in_place=False):
            if not in_place:
                x = x_.copy()
            else:
                x = x_

            assert x.shape == (self.d,)

            t, tau, W, Minv, q = self.t, self.tau, self.W, self.Minv, self.q

            y = np.dot(Minv, W.dot(x))

            # Plasticity, using gradient ascent/descent

            # W <- W + 2 eta(t) * (y*x' - W)
            step = eta(t)

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


def if_minimax_PCA(X, q, tau=0.5, n_epoch=1, error_options=None, Minv0=None, W0=None):

    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau           -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    n_epoch       -- Number of epochs for training, i.e., how many times to loop over the columns of X
    error_options -- A struct with options for computing errors
    Minv0         -- Initial guess for the inverse of the lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d

    Output:
    ====================
    Minv -- Final iterate of the inverse lateral weight matrix, of size q-by-q (sometimes)
    W    -- Final iterate of the forward weight matrix, of size q-by-d (sometimes)
    errs -- The requested evaluation of the subspace error at each step (sometimes)
    """

    d,n = X.shape

    if Minv0 is not None:
        assert Minv0.shape == (q,q), "The shape of the initial guess Minv0 must be (q,q)=(%d,%d)" % (q,q)
        Minv = Minv0
    else:
        Minv = np.eye(q)

    if W0 is not None:
        assert W0.shape == (q,d), "The shape of the initial guess W0 must be (q,d)=(%d,%d)" % (q,d)
        W = W0
    else:
        W = np.random.normal(0, 1.0/np.sqrt(d), size=(q,d))


    n_its = n_epoch * n

    if error_options is not None:
        return _iterate_and_compute_errors(X, Minv, W, tau, n_its, n, error_options)
    else:
        return _iterate(X, Minv, W, tau, n_its, n)





#
if __name__ == "__main__":

    # Run a test of if_minimax_PCA
    print("Testing if_minimax_PCA")
    from util import generate_samples

    # Parameters
    n = 5000
    d = 2000
    q = 200  # Value of q is technically hard-coded below, sorry
    n_epoch = 1
    tau = 0.5

    generator_options = {
        'method': 'spiked_covariance',
        'lambda_q': 5e-1,
        'normalize': True,
        'rho': 1e-2 / 5,
        'return_U': True
    }
    X, U, sigma2 = generate_samples(d, q, n, generator_options)
    #     print([X.sum(),U.sum()])
    lambda_1 = np.random.normal(0, 1, (q,)) / np.sqrt(q)

    #     ccipca = CCIPCA_CLASS(q, d)
    errs = []
    #     print([X.sum(),U.sum()])
    #     np.linalg.norm(Xtest, 'fro')

    # %%
    if_mm_pca = IF_minimax_PCA_Class(q, d, W0=X[:, :q].T, Minv0=None, tau=tau)
    X1 = X.copy()
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X1.T:
            if_mm_pca.fit_next(x, in_place=True)
    #             break
    #             errs.append(subspace_error(ccipca.Uhat,U[:,:q]))
    time_2 = time.time() - time_1
    #     pl.plot(errs)
    print(time_2)
    print([subspace_error(np.asarray(if_mm_pca.Minv.dot(if_mm_pca.W).T), U[:, :q])])
    # %%
    X1 = X.copy()
    time_1 = time.time()
    UU, WW = if_minimax_PCA(X1, q, tau, n_epoch, Minv0=None, W0=X[:, :q].T)
    UU = UU.dot(WW).T
    time_2_loop = time.time() - time_1
    print(time_2_loop)
    print([subspace_error(np.asarray(if_mm_pca.Minv.dot(if_mm_pca.W).T),U[:,:q]),subspace_error(UU,U[:,:q])])
