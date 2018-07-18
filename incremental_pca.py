# Title: incremental_pca.py
# Description: A function for PCA using the incremental approach in the onlinePCA R package
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)

##############################
# Imports
import sys
sys.path.append('/mnt/home/agiovann/SOFTWARE/online_pca')
import numpy as np
import util
from util import subspace_error
import time
# from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from scipy.linalg import svd
import coord_update
try:
    profile
except:
    def profile(a): return a
##############################

#%%
def _iterate(X, lambda_, Uhat, tol, f, n_its, n, q):
    for t in range(n_its):
        j = t % n

        lambda_ = (1-f) * lambda_
        x       = np.sqrt(f) * X[:,j]

        # Project X into current estimate and check residual error
        Uhatx    = Uhat.T.dot(x)
        x        = x - Uhat.dot(Uhatx);
        normx    = np.sqrt(x.dot(x))#np.linalg.norm(x)

        #TODO: fix this atleast_2d for efficiency
        if (normx >= tol):
            lambda_  = np.concatenate((lambda_, [0]))
            Uhatx    = np.concatenate((Uhatx, [normx]))
            Uhat     = np.concatenate((Uhat, x[:,np.newaxis] / normx),1)

        # Get new eigenvectors, is this possibly fast at all?
        d,V     = np.linalg.eigh(np.diag(lambda_) + np.outer(Uhatx,Uhatx.T))
        idx     = np.flip(np.argsort(d),0)
        lambda_ = d[idx]
        V       = V[:,idx]
        lambda_ = lambda_[:q]
        Uhat    = Uhat.dot(V[:,:q])
    return Uhat

def _iterate_and_compute_errors(X, lambda_, Uhat, tol, f, n_its, n, q, error_options):
    errs = util.initialize_errors(error_options, n_its)
    for t in range(n_its):
        j = t % n
        util.compute_errors(error_options, Uhat, t, errs)

        lambda_ = (1-f) * lambda_
        x       = np.sqrt(f) * X[:,j]

        # Project X into current estimate and check residual error
        Uhatx    = Uhat.T.dot(x)
        x        = x - Uhat.dot(Uhatx);
        normx    = np.sqrt(x.dot(x)) #np.linalg.norm(x)

        if (normx >= tol):
            lambda_  = np.concatenate((lambda_, [0]))
            Uhatx    = np.concatenate((Uhatx, [normx]))
            Uhat     = np.concatenate((Uhat, np.atleast_2d(x.T).T / normx),1)

        # Get new eigenvectors, is this possibly fast at all?
        d,V     = np.linalg.eigh(np.diag(lambda_) + np.outer(Uhatx,Uhatx.T))
        idx     = np.flip(np.argsort(d),0)
        lambda_ = d[idx]
        V       = V[:,idx]
        lambda_ = lambda_[:q]
        Uhat    = Uhat.dot(V[:,:q])
    return errs

def incremental_PCA(X, q, n_epoch=1, tol=1e-7, f=None, error_options=None, Uhat0=None, lambda0=None):

    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tol           -- Tolerance for when we add a vector to the space
    n_epoch       -- Number of epochs for training, i.e., how many times to loop over the columns of X
    f             -- Forgetting factor f, a number in (0,1)
    error_options -- A struct with options for computing errors
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q

    Output:
    ====================
    Uhat -- Final iterate of the eigenspace matrix, of size d-by-q (sometimes)
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
        assert lambda0.shape == (q,), "The shape of the initial guess lambda0 must be (q,)=(%d,)" % (q)
        lambda_ = lambda0
    else:
        lambda_= np.random.normal(0,1,(q,)) / np.sqrt(q)

    if f is not None:
        assert (f>0) and (f<1), "The parameter f must be between 0 and 1"
    else:
        f = 1.0/n


    n_its = n_epoch * n

    if error_options is not None:
        return _iterate_and_compute_errors(X, lambda_, Uhat, tol, f, n_its, n, q, error_options)
    else:
        return _iterate(X, lambda_, Uhat, tol, f, n_its, n, q)


class IncrementalPCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q
    f             -- Forgetting factor f, a number in (0,1)
    cython: bool
        whether to use computationally optimized cython function

    Methods:
    ====================
    fit_next()
    """

    def __init__(self, q, d, Uhat0=None, lambda0=None, tol=1e-7, f=None, use_svd=False):

        if Uhat0 is not None:
            assert Uhat0.shape == (d,q), "The shape of the initial guess Uhat0 must be (d,q)=(%d,%d)" % (d,q)
            self.Uhat = Uhat0.copy()

        else:
            # random initalization if not provided
            self.Uhat = np.random.normal(loc = 0, scale = 1/d, size=(d,q))

        self.t = 1

        if lambda0 is not None:
            assert lambda0.shape == (q,), "The shape of the initial guess lambda0 must be (q,)=(%d,)" % (q)
            self.lambda_ = lambda0.copy()
        else:
            self.lambda_= np.random.normal(0,1,(q,)) / np.sqrt(q)


        if f is not None:
            assert (f > 0) and (f < 1), "The parameter f must be between 0 and 1"
        else:
            f = 1.0 / n


        self.q = q
        self.d = d
        self.f = f
        self.tol = tol
        self.use_svd = use_svd


    def fit(self, X):
        raise Exception("Not Implemented")
        # self.Uhat, self.lambda_ = coord_update.coord_update_total(X, X.shape[-1],self.d, np.double(self.t), np.double(self.ell), self.lambda_, self.Uhat, self.q, self.v)

    @profile
    def fit_next(self,x_, in_place = False):
        if not in_place:
            x = x_.copy()
        else:
            x = x_

        assert x.shape == (self.d,)


        t, f, lambda_, Uhat, q, tol = self.t, self.f, self.lambda_, self.Uhat, self.q, self.tol

        lambda_ = (1 - f) * lambda_
        x = np.sqrt(f) * x

        # Project X into current estimate and check residual error
        Uhatx = Uhat.T.dot(x)
        x = x - Uhat.dot(Uhatx)
        normx = np.sqrt(x.dot(x))  # np.linalg.norm(x)

        # TODO: fix this atleast_2d for efficiency
        if (normx >= tol):
            lambda_ = np.concatenate((lambda_, [0]))
            Uhatx = np.concatenate((Uhatx, [normx]))
            Uhat = np.concatenate((Uhat, x[:, np.newaxis] / normx), 1)

        M = np.diag(lambda_) + np.outer(Uhatx, Uhatx.T)
        if self.use_svd: # this does not seem to give a better time, why?
            U1,S1,V1 = svd(M, overwrite_a=True)
            d = np.diag(V1.T.dot(M.dot(V1)))
            d = np.diag(np.sign(d) * np.diag(S1))
            V = V1.T
        else:
            # TODO are there other faster methods? Get new eigenvectors, is this possibly fast at all? Can we use SVD?
            d, V = eigh(M, overwrite_a=True)


        idx = np.argsort(d)[::-1]
        lambda_ = d[idx]
        V = V[:, idx]
        lambda_ = lambda_[:q]
        Uhat = Uhat.dot(V[:, :q])


        self.Uhat = Uhat
        self.lambda_ = lambda_


#%%
if __name__ == "__main__":
#%%
     # Run a test of CCIPCA
     print('Testing IPCA')
     from util import generate_samples

     # Parameters
     n       = 1000
     d       = 2000
     q       = 200    # Value of q is technically hard-coded below, sorry
     n_epoch = 1

     generator_options = {
        'method'   : 'spiked_covariance',
        'lambda_q' : 5e-1,
        'normalize': True,
        'rho'      : 1e-2/5,
        'return_U' : True
     }
     X, U, sigma2 = generate_samples(d, q, n, generator_options)
#     print([X.sum(),U.sum()])
     lambda_1 = np.random.normal(0,1,(q,)) / np.sqrt(q)

#     ccipca = CCIPCA_CLASS(q, d)
     errs = []
#     print([X.sum(),U.sum()])
#     np.linalg.norm(Xtest, 'fro')

     #%%
     ipca = IncrementalPCA_CLASS(q, d, Uhat0=X[:,:q], lambda0=lambda_1, use_svd=False)
     X1 = X.copy()
     time_1 = time.time()
     for n_e in range(n_epoch):
         for x in X1.T:
             ipca.fit_next(x,in_place = True)
#             break
#             errs.append(subspace_error(ccipca.Uhat,U[:,:q]))
     time_2 = time.time() - time_1
#     pl.plot(errs)
     print(time_2)
     print([subspace_error(np.asarray(ipca.Uhat),U[:,:q])])
     #%%
     # X1 = X.copy()
     # time_1 = time.time()
     # UU = incremental_PCA(X1, q, n_epoch, Uhat0=X[:,:q], lambda0=lambda_1)
     # time_2_loop = time.time() - time_1
     # print(time_2_loop)
     # print([subspace_error(ipca.Uhat,U[:,:q]),subspace_error(UU,U[:,:q])])
    #%%
