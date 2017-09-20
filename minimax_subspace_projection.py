# Title: minimax_subspace_projection.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax algorithm of Pehlevan et al.
# Author: Victor Minden (vminden@flatironinstitute.org)
# Reference: (Pehlevan et al, Neural Computation, 2017)

##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
import util
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

    return 1.0 / (t + 1e3)


def _iterate(X, M, W, tau, n_its, n):
    for t in range(n_its):
        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = solve(M, W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W + 2*step * np.outer(y,X[:,j])

        # M <- M + eta(t)/tau * (y*y' - M)
        step = step/tau
        M    = (1-step) * M + step * np.outer(y,y)

    return M,W



def _iterate_and_compute_errors(X, M, W, tau, n_its, n, error_options):
    errs = util.initialize_errors(error_options, n_its)
    for t in range(n_its):
        # Record error
        Uhat = solve(M, W).T
        util.compute_errors(error_options, Uhat, t, errs)

        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = solve(M, W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W + 2*step * np.outer(y,X[:,j])

        # M <- M + eta(t)/tau * (y*y' - M)
        step = step/tau
        M    = (1-step) * M + step * np.outer(y,y)

    return errs



def minimax_PCA(X, q, tau=0.5, n_epoch=1, error_options=None, M0=None, W0=None):

    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau           -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    n_epoch       -- Number of epochs for training, i.e., how many times to loop over the columns of X
    error_options -- A struct with options for computing errors
    M0            -- Initial guess for the lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d

    Output:
    ====================
    M    -- Final iterate of the lateral weight matrix, of size q-by-q (sometimes)
    W    -- Final iterate of the forward weight matrix, of size q-by-d (sometimes)
    errs -- The requested evaluation of the subspace error at each step (sometimes)
    """

    d,n = X.shape

    if M0 is not None:
        assert M0.shape == (q,q), "The shape of the initial guess M0 must be (q,q)=(%d,%d)" % (q,q)
        M = M0
    else:
        M = np.eye(q)

    if W0 is not None:
        assert W0.shape == (q,d), "The shape of the initial guess W0 must be (q,d)=(%d,%d)" % (q,d)
        W = W0
    else:
        W = np.random.normal(0, 1.0/np.sqrt(d), size=(q,d))


    n_its = n_epoch * n

    if error_options is not None:
        return _iterate_and_compute_errors(X, M, W, tau, n_its, n, error_options)
    else:
        return _iterate(X, M, W, tau, n_its, n)




#
# if __name__ == "__main__":
#
#     # Run a test of minimax_PCA
#     print("Testing minimax_PCA")
#     # Parameters
#     n       = 2000
#     d       = 10
#     q       = 3     # Value of q is technically hard-coded below, sorry
#     n_epoch = 10
#     tau     = 0.5
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
#     M,W,errs = minimax_PCA(X, q, tau, n_epoch, U=U[:,:q])
#     print('The initial error was %f and the final error was %f.' %(errs[0],errs[-1]))
