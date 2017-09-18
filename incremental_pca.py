# Title: incremental_pca.py
# Description: A function for PCA using the incremental approach in the onlinePCA R package
# Author: Victor Minden (vminden@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovanni and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)

##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
import util
##############################


def _iterate(X, lambda_, Uhat, tol, f, n_its, n, q):
    for t in range(n_its):
        j = t % n

        lambda_ = (1-f) * lambda_
        x       = np.sqrt(f) * X[:,j]

        # Project X into current estimate and check residual error
        Uhatx    = Uhat.T.dot(x)
        x        = x - Uhat.dot(Uhatx);
        normx    = np.linalg.norm(x)

        if (normx >= tol):
            lambda_  = np.concatenate((lambda_, [0]))
            Uhatx    = np.concatenate((Uhatx, [normx]))
            Uhat     = np.concatenate((Uhat, np.atleast_2d(x.T).T / normx),1)

        # Get new eigenvectors, is this possibly fast at all?
        d,V     = np.linalg.eig(np.diag(lambda_) + np.outer(Uhatx,Uhatx.T))
        idx     = np.flip(np.argsort(d),0)
        lambda_ = d[idx]
        V       = V[:,idx]
        lambda_ = lambda_[:q]
        Uhat    = Uhat.dot(V[:,:q])
    return Uhat


def _iterate_and_compute_errors(X, lambda_, Uhat, tol, f, n_its, n, q, U):
    errs = np.zeros(n_its)
    for t in range(n_its):
        j = t % n
        errs[t] = util.subspace_error(Uhat, U)

        lambda_ = (1-f) * lambda_
        x       = np.sqrt(f) * X[:,j]

        # Project X into current estimate and check residual error
        Uhatx    = Uhat.T.dot(x)
        x        = x - Uhat.dot(Uhatx);
        normx    = np.linalg.norm(x)

        if (normx >= tol):
            lambda_  = np.concatenate((lambda_, [0]))
            Uhatx    = np.concatenate((Uhatx, [normx]))
            Uhat     = np.concatenate((Uhat, np.atleast_2d(x.T).T / normx),1)

        # Get new eigenvectors, is this possibly fast at all?
        d,V     = np.linalg.eig(np.diag(lambda_) + np.outer(Uhatx,Uhatx.T))
        idx     = np.flip(np.argsort(d),0)
        lambda_ = d[idx]
        V       = V[:,idx]
        lambda_ = lambda_[:q]
        Uhat    = Uhat.dot(V[:,:q])
    return Uhat, errs


def incremental_PCA(X, q, n_epoch=1, tol=1e-7, f=None, U=None, Uhat0=None, lambda0=None):

    """
    Parameters:
    ====================
    X            -- Numpy array of size d-by-n, where each column corresponds to one observation
    q            -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tol          -- Tolerance for when we add a vector to the space
    n_epoch      -- Number of epochs for training, i.e., how many times to loop over the columns of X
    f            -- Forgetting factor f, a number in (0,1)
    U            -- The true PCA basis for error checks, or None to avoid calculation altogether
    Uhat0           -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0      -- Initial guess for the eigenvalues vector lambda_, must be of size q

    Output:
    ====================
    M    -- Final iterate of the lateral weight matrix, of size q-by-q
    W    -- Final iterate of the forward weight matrix, of size q-by-d
    errs -- The requested evaluation of the subspace error at each step (sometimes)
    """


    d,n = X.shape

    if Uhat0 is not None:
        assert Uhat0.shape == (d,q), "The shape of the initial guess Uhat0 must be (d,q)=(%d,%d)" % (d,q)
        Uhat = Uhat0
    else:
        # Check me
        Uhat = np.eye(d,q)

    if lambda0 is not None:
        assert lambda0.shape == (q,d), "The shape of the initial guess lambda0 must be (q,)=(%d,)" % (q)
        lambda_ = lambda0
    else:
        lambda_= np.ones((q,))

    if f is not None:
        assert (f>0) and (f<1), "The parameter f must be between 0 and 1"
    else:
        f = 1.0/n


    n_its = n_epoch * n

    if U is not None:
        assert U.shape == (d,q), "The shape of the PCA subspace basis matrix must be (d,q)=(%d,%d)" % (d,q)
        return _iterate_and_compute_errors(X, lambda_, Uhat, tol, f, n_its, n, q, U)
    else:
        return _iterate(X, lambda_, Uhat, tol, f, n_its, n, q)


if __name__ == "__main__":

    # Run a test of incremental_PCA

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

    Uhat,errs = incremental_PCA(X, q, n_epoch, U=U[:,:q])
    print('The initial error was %f and the final error was %f.' %(errs[0],errs[-1]))
