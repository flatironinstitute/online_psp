# Title: minimax_subspace_alignment.py
# Description: A function for PCA using the Hebbian/anti-Hebbian minimax alignment algorithm
# Author: Victor Minden (vminden@flatironinstitute.org)


##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
from scipy.linalg import fractional_matrix_power as fmp
import util
##############################




def eta_fun(t):

    """
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated

    Output:
    ====================
    step -- learning rate at time t
    """

    return 2/(1e3+1e-3*t)/2#2e-3#50/(1e4 + t)/5#50/(1e4)

def _iterate(X, Lambda, M, W, tau, n_its, n, q, eta):
    for t in range(n_its):
        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        y = solve(M, Lambda.dot(W.dot(X[:,j])))

        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        W    = (1-2*step) * W +  np.outer(2*step*Lambda.dot(y),X[:,j])

        # take a STOCHASTIC step in M
        # v    = np.zeros(q)
        # v[np.random.randint(q)] = 1
#        v = np.random.normal(0,1,(q))
        # M <- M + eta(t)/tau * (y*y' - q *er*er')
        step = step/tau
        M    = M + np.outer( step*y,y) - step*np.eye(q)#np.outer(step*v,v)

    return M,W



def _iterate_and_compute_errors(X, Lambda, M, W, tau, n_its, n, q, error_options, eta):
    errs = util.initialize_errors(error_options, n_its)
    beta = 0#0.55
    dW = np.zeros(W.shape)
    dM = np.zeros(M.shape)
    C = X.dot(X.T)/n
    Chalf = fmp(C,0.5)

    for t in range(n_its):
        Md = np.diag(1./np.diag(M))
        Mo = M - np.diag(np.diag(M))
        # Record error
        Uhat = Lambda.dot(W)
        Uhat = Md.dot(Uhat) - Md.dot(Mo.dot(Md.dot(Uhat)))
        #Uhat = Chalf.dot(Uhat.T)
        Uhat = Uhat.T
        util.compute_errors(error_options, Uhat, t, errs,M)

        # Neural dynamics, short-circuited to the steady-state solution
        j = t % n
        #y = solve(M, Lambda.dot(W.dot(X[:,j])))
        y = Lambda.dot(W.dot(X[:,j]))

        y = Md.dot(y) - Md.dot(Mo.dot(Md.dot(y)))
        # Plasticity, using gradient ascent/descent

        # W <- W + 2 eta(t) * (y*x' - W)
        step = eta(t)
        dW   = beta * dW + np.outer(Lambda.dot(y),X[:,j]) -  W
        W    =  W + 2*step*dW

        # take a STOCHASTIC step in M
        # v    = np.zeros(q)
        # v[np.random.randint(q)] = 1
        #v = np.random.normal(0,1,(q))
        # M <- M + eta(t)/tau * (y*y' - q *er*er')
        step =step/tau
        dM   = beta * dM + np.outer(y,y) - np.eye(q)
        M    = M + step*dM #np.outer(step*v,v)

    return errs



def minimax_alignment_PCA(X, q, tau=0.5, n_epoch=1, error_options=None, M0=None, W0=None, Lambda=None, eta=None):

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
    Lambda        -- The diagonal weights for alignment, must be of size q-by-q

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

    if Lambda is not None:
        assert Lambda.shape == (q,d), "The shape of Lambda must be (q,d)=(%d,%d)" % (q,d)
    else:
        # TODO: who knows how to set Lambda
        Lambda = np.diag(range(q+1,1,-1))
        Lambda = Lambda / (q+1)

    n_its = n_epoch * n

    if eta is None:
        eta = lambda t: eta_fun(t)

    if error_options is not None:
        return _iterate_and_compute_errors(X, Lambda, M, W, tau, n_its, n, q, error_options, eta)
    else:
        return _iterate(X, Lambda, M, W, tau, n_its, n, q, eta)
