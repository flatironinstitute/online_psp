# Title: online_similarity_matching.py
# Description: A function for PCA using the online similarity matching algorithm of Pehlevan et al.
# Author: Victor Minden (vminden@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovanni
# Reference: (Pehlevan et al, Neural Computation, 2015) and (Pehlevan et al, NIPS, 2015)

##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
import util
##############################


def _iterate(X, M, W, ysq, lambda_, n_its, n, q):
    for t in range(n_its):
        j = t % n

        # Steady-state of neural dynamics
        y   = solve((np.eye(q)+M), W.dot(X[:,j]))

        ysq = ysq + y**2

        # Update weights
        y_tmp    = y / ysq
        y_tmp_sq = y**2 / ysq;

        # Fix this reshape for efficiency
        y_tmp_sq = y_tmp_sq.reshape(q,1)

        W = W + np.outer(y_tmp, X[:,j]) - y_tmp_sq * W
        M = M + np.outer((1+lambda_)*y_tmp, y) - y_tmp_sq * M

        W[np.isnan(W)] = 0;
        M[np.isnan(M)] = 0;

        # Set diagonal to zero
        np.fill_diagonal(M, 0)

    return M,W



def _iterate_and_compute_errors(X, M, W, ysq, lambda_, n_its, n, q, error_options):
    errs = util.initialize_errors(error_options, n_its)
    for t in range(n_its):
        j = t % n
        # Record error
        Uhat = solve(np.eye(q) + M, W).T
        util.compute_errors(error_options, Uhat, t, errs)

        # Steady-state of neural dynamics
        y   = solve((np.eye(q)+M), W.dot(X[:,j]))

        ysq = ysq + y**2

        # Update weights
        y_tmp    = y / ysq
        y_tmp_sq = y**2 / ysq;

        # Fix this reshape for efficiency
        y_tmp_sq = y_tmp_sq.reshape(q,1)

        W = W + np.outer(y_tmp, X[:,j]) - y_tmp_sq * W
        M = M + np.outer((1+lambda_)*y_tmp, y) - y_tmp_sq * M

        W[np.isnan(W)] = 0;
        M[np.isnan(M)] = 0;

        # Set diagonal to zero
        np.fill_diagonal(M, 0)

    return errs


def OSM_PCA(X, q, lambda_=0, n_epoch=1, error_options=None, M0=None, W0=None, ysq0=None):

    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    lambda_       -- Decorrelation parameter (see Pehlevan et al. NIPS)
    n_epoch       -- Number of epochs for training, i.e., how many times to loop over the columns of X
    error_options -- A struct with options for computing errors
    M0            -- Initial guess for the lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d
    ysq0          -- Initial guess for the squared activity level ysk, must be of size q

    Output:
    ====================
    M    -- Final iterate of the lateral weight matrix, of size q-by-q (this is different than M for minimax_PCA) (sometimes)
    W    -- Final iterate of the forward weight matrix, of size q-by-d (sometimes)
    errs -- The requested evaluation of the subspace error at each step (sometimes)
    """

    d,n = X.shape

    if M0 is not None:
        assert M0.shape == (q,q), "The shape of the initial guess M0 must be (q,q)=(%d,%d)" % (q,q)
        M = M0
    else:
        # Check me
        M = np.eye(q) * 0

    if W0 is not None:
        assert W0.shape == (q,d), "The shape of the initial guess W0 must be (q,d)=(%d,%d)" % (q,d)
        W = W0
    else:
        W = np.random.normal(0, 1.0/np.sqrt(d), size=(q,d))

    if ysq0 is not None:
        assert ysq0.shape == (q,), "The shape of the initial guess W0 must be (q,)=(%d,)" % (q)
        ysq = ysq0
    else:
        ysq = np.ones(q) / q


    n_its = n_epoch * n

    if error_options is not None:
        return _iterate_and_compute_errors(X, M, W, ysq, lambda_, n_its, n, q, error_options)
    else:
        return _iterate(X, M, W, ysq, lambda_, n_its, n, q)


#
# if __name__ == "__main__":
#
#     # Run a test of OSM_PCA
#     print("Testing OSM_PCA")
#     # Parameters
#     n       = 2000
#     d       = 10
#     q       = 3     # Value of q is technically hard-coded below, sorry
#     n_epoch = 10
#     lambda_ = 0
#
#     X     = np.random.normal(0,1,(d,n))
#     # Note: Numpy SVD returns V transpose
#     U,s,Vt = np.linalg.svd(X, full_matrices=False)
#
#     s = np.concatenate( ([np.sqrt(3),np.sqrt(2),1], 1e-1*np.random.random(d-3)))
#     D = np.diag(np.sqrt(n) * s )
#
#     X = np.dot(U, np.dot(D, Vt))
#
#     M,W,errs = OSM_PCA(X, q, lambda_, n_epoch, U=U[:,:q])
#     print('The initial error was %f and the final error was %f.' %(errs[0],errs[-1]))
