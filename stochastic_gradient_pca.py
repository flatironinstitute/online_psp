# Title: stochastic_gradient_pca.py
# Description: A function for PCA using stochastic gradient ascent
# Author: Victor Minden (vminden@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovanni and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)

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

    return 1/ (t + 1e3)


def _iterate(X, Uhat, n_its, n, q):
    A = np.triu(np.ones((q,q)), 1)
    for t in range(n_its):
        j    = t % n

        # Uhat = Uhat + eta(t) * np.outer(X[:,j], Uhat.T.dot(X[:,j,np.newaxis]).T)
        # Uhat,r = np.linalg.qr(Uhat)
        #
        phi  = Uhat.T.dot(X[:,j])
        phiU = Uhat * phi
        Z = - phiU - 2 * phiU.dot(A) + X[:,j,np.newaxis]

        Uhat = Uhat + eta(t) * Z*phi


    return Uhat,errs




def _iterate_and_compute_errors(X, Uhat, n_its, n, q, U):
    errs = np.zeros(n_its)

    A = np.triu(np.ones((q,q)), 1)
    for t in range(n_its):
        # Record error
        errs[t] = util.subspace_error(Uhat, U)

        j    = t % n

        # Uhat = Uhat + eta(t) * np.outer(X[:,j], Uhat.T.dot(X[:,j,np.newaxis]).T)
        # Uhat,r = np.linalg.qr(Uhat)
        #
        phi  = Uhat.T.dot(X[:,j])
        phiU = Uhat * phi
        Z = - phiU - 2 * phiU.dot(A) + X[:,j,np.newaxis]

        Uhat = Uhat + eta(t) * Z*phi


    return Uhat,errs




def SGA_PCA(X, q, n_epoch=1, U=None, Uhat0=None):

    """
    Parameters:
    ====================
    X            -- Numpy array of size d-by-n, where each column corresponds to one observation
    q            -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    n_epoch      -- Number of epochs for training, i.e., how many times to loop over the columns of X
    U            -- The true PCA basis for error checks, or None to avoid calculation altogether
    Uhat0        -- Initial guess for the eigenspace matrix of size d-by-q

    Output:
    ====================
    Uhat -- Final iterate of the eigenspace matrix, of size d-by-q
    errs -- The requested evaluation of the subspace error at each step (sometimes)
    """

    d,n = X.shape

    if Uhat0 is not None:
        assert Uhat0.shape == (q,d), "The shape of the initial guess Uhat0 must be (d,q)=(%d,%d)" % (d,q)
        Uhat = Uhat0
    else:
        Uhat = np.random.normal(0, 1.0/np.sqrt(d), size=(d,q))


    n_its = n_epoch * n

    if U is not None:
        assert U.shape == (d,q), "The shape of the PCA subspace basis matrix must be (d,q)=(%d,%d)" % (d,q)
        return _iterate_and_compute_errors(X, Uhat, n_its, n, q, U)
    else:
        return _iterate(X, Uhat, n_its, n, q)







if __name__ == "__main__":

    # Run a test of SGA_PCA
    print("Testing SGA_PCA")
    # Parameters
    n       = 2000
    d       = 10
    q       = 3     # Value of q is technically hard-coded below, sorry
    n_epoch = 10

    X     = np.random.normal(0,1,(d,n))
    # Note: Numpy SVD returns V transpose
    U,s,Vt = np.linalg.svd(X, full_matrices=False)

    s = np.concatenate( ([np.sqrt(3),np.sqrt(2),1], 1e-1*np.random.random(d-3)))
    D = np.diag(np.sqrt(n) * s )

    X = np.dot(U, np.dot(D, Vt))

    Uhat,errs = SGA_PCA(X, q, n_epoch, U=U[:,:q])
    print('The initial error was %f and the final error was %f.' %(errs[0],errs[-1]))
    # plt.plot(np.log10(errs))
    # plt.show()
