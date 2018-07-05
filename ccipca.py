# Title: ccipca.py
# Description: A function for PCA using the Candid Covariance-Free Incrmental PCA approach
# Author: Victor Minden (vminden@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovanni and Cengiz Pehlevan
# Reference: J. Weng, Y. Zhang, and W. S. Hwang, "Candid covariance-free incremental principal component analysis", IEEE Trans. Pattern. Anal. Mach. Intell., vol 25, no. 8, pp. 1034-1040, Aug. 2003

##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
import util
##############################


def _iterate(X, lambda_, Uhat, ell, n_its, n, q):
    for t in range(1,n_its):
        j       = t % n
        x       = X[:,j]
        for i in range(q):
            v          = ((t-ell)/(t+1) * lambda_[i]) * Uhat[:,i] + ((1+ell)/(t+1) * np.dot(x,Uhat[:,i]))* x
            lambda_[i] = np.sqrt(v.dot(v))#np.linalg.norm(v)
            Uhat[:,i]  = v/lambda_[i]
            # Orthogonalize the data against this approximate eigenvector
            x          = x - np.dot(x,Uhat[:,i]) * Uhat[:,i]

    return Uhat

def _iterate_and_compute_errors(X, lambda_, Uhat, ell, n_its, n, q, error_options):

    errs = util.initialize_errors(error_options, n_its)

    for t in range(1,n_its):
        j       = t % n
        util.compute_errors(error_options, Uhat, t, errs)
        x       = X[:,j]
        for i in range(q):
            v          = (t-ell)/(t+1) * lambda_[i] * Uhat[:,i] + (1+ell)/(t+1) * np.dot(x,Uhat[:,i])* x
            lambda_[i] = np.sqrt(v.dot(v))#np.linalg.norm(v)
            Uhat[:,i]  = v/lambda_[i]
            # Orthogonalize the data against this approximate eigenvector
            x          = x - np.dot(x,Uhat[:,i]) * Uhat[:,i]

    # The algorithm dictates an initial guess of the first data point, so the rest of the errors are not defined
    #
    # for i in range(q):
    #     errs[:,i] = errs[:,q]
    return errs

class CCIPCA_CLASS:
    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    error_options -- A struct with options for computing errors
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q
    ell           -- Amnesiac parameter (see reference)

    Methods:
    ====================
    fit_next()
    compute_errors()
    """

    def __init__(self, q, d, Uhat0=None, error_options=None, lambda0=None, ell=2):
        if Uhat0 is not None:
            assert Uhat0.shape == (d,q), "The shape of the initial guess Uhat0 must be (d,q)=(%d,%d)" % (d,q)
            self.Uhat = Uhat0.copy()

        else:
            # random initalization if not provided
            self.Uhat = np.random.normal(loc = 0, scale = 1/d, size=(d,q))

        self.t = 1

        if lambda0 is not None:
            assert lambda0.shape == (q,d), "The shape of the initial guess lambda0 must be (q,)=(%d,)" % (q)
            self.lambda_ = lambda0
        else:
            self.lambda_= np.random.normal(0,1,(q,)) / np.sqrt(q)

        self.q = q
        self.d = d

        self.error_options = error_options
        self.ell = ell



    def fit_next(self,x_):
        x = x_.copy()
        assert x.shape == (d,)

        t, ell, lambda_, Uhat, q = self.t, self.ell, self.lambda_, self.Uhat, self.q
        for i in range(q):
            v          = max(1,t-ell)/(t+1) * lambda_[i] * Uhat[:,i] + (1+ell)/(t+1) * np.dot(x,Uhat[:,i])* x # is that OK?
            lambda_[i] = np.sqrt(v.dot(v))#np.linalg.norm(v)
            Uhat[:,i]  = v/lambda_[i]
            # Orthogonalize the data against this approximate eigenvector
            x          = x - np.dot(x,Uhat[:,i]) * Uhat[:,i]

        self.Uhat = Uhat
        self.lambda_ = lambda_
        self.t += 1


def CCIPCA(X, q, n_epoch=1, error_options=None, Uhat0=None, lambda0=None, ell=2):

    """
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    n_epoch       -- Number of epochs for training, i.e., how many times to loop over the columns of X
    error_options -- A struct with options for computing errors
    Uhat0         -- Initial guess for the eigenspace matrix U, must be of size d-by-q
    lambda0       -- Initial guess for the eigenvalues vector lambda_, must be of size q
    ell           -- Amnesiac parameter (see reference)

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
        # TODO: maybe replace me with better initialization
        Uhat = X[:,:q].copy()

    if lambda0 is not None:
        assert lambda0.shape == (q,d), "The shape of the initial guess lambda0 must be (q,)=(%d,)" % (q)
        lambda_ = lambda0
    else:
        lambda_= np.random.normal(0,1,(q,)) / np.sqrt(q)

    n_its = n_epoch * n

    if error_options is not None:
        return _iterate_and_compute_errors(X, lambda_, Uhat, ell, n_its, n, q, error_options)
    else:
        return _iterate(X, lambda_, Uhat, ell, n_its, n, q)

#%%
 if __name__ == "__main__":
#%%
     # Run a test of CCIPCA
     print('Testing CCIPCA')
     from util import generate_samples


     # Parameters
     n       = 2000
     d       = 50
     q       = 5    # Value of q is technically hard-coded below, sorry
     n_epoch = 10

     generator_options = {
        'method'   : 'spiked_covariance',
        'lambda_q' : 5e-1,
        'normalize': True,
        'rho'      : 1e-2/5,
        'return_U' : True
     }
     synth = generate_samples(d, q, n, generator_options)
     X = synth['X']
     U = synth['U']
     sigma2 = synth['sigma2']

     ccipca = CCIPCA_CLASS(q, d, Uhat0=X[:,:q])
     ccipca = CCIPCA_CLASS(q, d)
     from util import subspace_error
     errs = []
#     np.linalg.norm(Xtest, 'fro')
     for n_e in range(n_epoch):
         for x in X.T:
             ccipca.fit_next(x)
             errs.append(subspace_error(ccipca.Uhat,U[:,:q]))
     pl.plot(errs)
#%%
     error_options = {
        'n_skip':10,
        'error_func_list' : [('batch_err', lambda Uhat: util.subspace_error(Uhat, U))],
         'orthogonalize_iterate' : False,

        }
     errs = CCIPCA(X, q, n_epoch, Uhat0=U[:,:q], error_options = error_options)
     print('The initial error was %f and the final error was %f.' %(errs[0],errs[-1]))
