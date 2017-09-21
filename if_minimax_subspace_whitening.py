# Title: if_minimax_subspace_whitening.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax whitening algorithm of Pehlevan et al. wth Sherman-Morrison
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

def _iterate(X, Minv, W, tau, n_its, n, q):
    for t in range(n_its):
        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = Minv.dot(W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W +  np.outer(2*step*y,X[:,j])

        # take a STOCHASTIC step in M
        v    = np.zeros(q)
        v[np.random.randint(q)] = 1
        # M <- M + eta(t)/tau * (y*y' - q *er*er')
        step = step/tau

        Minvy = Minv.dot(y)

        Minv  = Minv - np.outer(step*Minvy, Minvy.T)/(1+step * np.dot(y,Minvy))

        Minvv = Minv.dot(v)
        step  = q*step
        Minv  = Minv + np.outer(step*Minvv, Minvv.T)/(1+step * np.dot(-v,Minvv))
        # M     = M + np.outer( step*y,y) - np.outer(q*step*v,v)
    return Minv,W



def _iterate_and_compute_errors(X, Minv, W, tau, n_its, n, q, error_options):
    errs = util.initialize_errors(error_options, n_its)
    for t in range(n_its):
        # Record error
        Uhat = (Minv.dot(W)).T
        util.compute_errors(error_options, Uhat, t, errs)

        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = Minv.dot(W.dot(X[:,j]))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W + np.outer(2*step*y,X[:,j])

        # take a STOCHASTIC step in M
        v    = np.zeros(q)
        v[np.random.randint(q)] = 1
        # M <- M + eta(t)/tau * (y*y' - q *er*er')
        step = step/tau

        Minvy = Minv.dot(y)

        Minv  = Minv - np.outer(step*Minvy, Minvy.T)/(1+step * np.dot(y,Minvy))

        Minvv = Minv.dot(v)
        step  = q*step
        Minv  = Minv + np.outer(step*Minvv, Minvv.T)/(1+step * np.dot(-v,Minvv))
        # M     = M + np.outer( step*y,y) - np.outer(q*step*v,v)

    return errs



def if_minimax_whitening_PCA(X, q, tau=0.5, n_epoch=1, error_options=None, Minv0=None, W0=None):

    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau           -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    n_epoch       -- Number of epochs for training, i.e., how many times to loop over the columns of X
    error_options -- A struct with options for computing errors
    Minv0         -- Initial guess for the inverse lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d

    Output:
    ====================
    Minv -- Final iterate of the lateral weight matrix, of size q-by-q (sometimes)
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
        return _iterate_and_compute_errors(X, Minv, W, tau, n_its, n, q, error_options)
    else:
        return _iterate(X, Minv, W, tau, n_its, n, q)
