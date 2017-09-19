# Title: if_minimax_subspace_projection.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax algorithm of Pehlevan et al. with added Sherman-Morrison formula
# Author: Victor Minden (vminden@flatironinstitute.org)
# Reference: (Pehlevan et al, Neural Computation, 2017) and working notes

##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
import util
from matplotlib import pyplot as plt
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


def _iterate(X, Minv, W, tau, n_its, n):
    for t in range(n_its):
        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = np.dot(Minv, W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W + 2*step * np.outer(y,X[:,j])

        # M <- M + eta(t)/tau * (y*y' - M), using SMW
        step = step/tau

        Minv = Minv / (1-step)
        z    = Minv.dot(y)
        Minv = Minv - step*np.outer(z, z.T) / (1 + step*np.dot(z,y))
        # M    = (1-step) * M + step * np.outer(y,y)

    return Minv,W



def _iterate_and_compute_errors(X, Minv, W, tau, n_its, n, U, V):
    if U is not None:
        errsU = np.zeros(n_its)
    if V is not None:
        errsV = np.zeros(n_its)

    for t in range(n_its):
        # Record error
        Uhat    = Minv.dot(W).T
        if U is not None:
            errsU[t] = util.subspace_error(Uhat, U)

        if V is not None:
            errsV[t] = util.subspace_error(Uhat, V)

        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = np.dot(Minv, W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W + 2*step * np.outer(y,X[:,j])

        # M <- M + eta(t)/tau * (y*y' - M), using SMW
        step = step/tau

        Minv = Minv / (1-step)
        z    = Minv.dot(y)
        Minv = Minv - step*np.outer(z, z.T) / (1 + step*np.dot(z,y))
        # M    = (1-step) * M + step * np.outer(y,y)


    if U is not None and V is not None:
        return errsU,errs
    elif U is not None:
        return errsU
    elif V is not None:
        return errsV

def if_minimax_PCA(X, q, tau=0.5, n_epoch=1, U=None, V=None, Minv0=None, W0=None):

    """
    Parameters:
    ====================
    X            -- Numpy array of size d-by-n, where each column corresponds to one observation
    q            -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau          -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    n_epoch      -- Number of epochs for training, i.e., how many times to loop over the columns of X
    U            -- The population PCA basis for error checks, or None to avoid calculation altogether
    V            -- The sample PCA basis for error checks, or None to avoid calculation altogether
    Minv0        -- Initial guess for the inverse of the lateral weight matrix M, must be of size q-by-q
    W0           -- Initial guess for the forward weight matrix W, must be of size q-by-d

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

    if U is not None:
        assert U.shape == (d,q), "The shape of the PCA subspace basis matrix U must be (d,q)=(%d,%d)" % (d,q)
    if V is not None:
        assert V.shape == (d,q), "The shape of the PCA subspace basis matrix V must be (d,q)=(%d,%d)" % (d,q)

    if U is not None or V is not None:
        return _iterate_and_compute_errors(X, Minv, W, tau, n_its, n, U, V)
    else:
        return _iterate(X, Minv, W, tau, n_its, n)






if __name__ == "__main__":

    # Run a test of if_minimax_PCA
    print("Testing if_minimax_PCA")
    # Parameters
    n       = 2000
    d       = 10
    q       = 3     # Value of q is technically hard-coded below, sorry
    n_epoch = 10
    tau     = 0.5

    X     = np.random.normal(0,1,(d,n))
    # Note: Numpy SVD returns V transpose
    U,s,Vt = np.linalg.svd(X, full_matrices=False)

    s = np.concatenate( ([np.sqrt(3),np.sqrt(2),1], 1e-1*np.random.random(d-3)))
    D = np.diag(np.sqrt(n) * s )

    X = np.dot(U, np.dot(D, Vt))

    Minv,W,errs = if_minimax_PCA(X, q, tau, n_epoch, U=U[:,:q])
    print('The initial error was %f and the final error was %f.' %(errs[0],errs[-1]))
    # plt.plot(np.log10(errs))
    # plt.show()
