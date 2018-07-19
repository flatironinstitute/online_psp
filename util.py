# Title: util.py
# Description: Various utilities useful for online PCA tests
# Author: Victor Minden (vminden@flatironinstitute.org)

##############################
# Imports
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
##############################

def compute_errors(error_options, Uhat, t, errs,M=None):
    if t % error_options['n_skip']:
        return
    if error_options['orthogonalize_iterate']:
        U,_ = np.linalg.qr(Uhat)
    else:
        U = Uhat
    for i,(fname, f) in enumerate(error_options['error_func_list']):
        if fname == 'batch_alignment_err':
            errs[fname][t] = f(Uhat)
        elif fname == 'diag_err':
            errs[fname][t] = f(M)
        else:
            errs[fname][t] = f(U)


def initialize_errors(error_options, n_its):
    # Build a dictionary for storing the error information for each specified error function
    return { fun_name : np.zeros(n_its) for (fun_name, _) in error_options['error_func_list'] }

def subspace_error(Uhat, U, relative_error_flag=True):

    """
    Parameters:
    ====================
    Uhat -- The approximation Uhat of an orthonormal basis for the PCA subspace of size d by q
    U    -- An orthonormal basis for the PCA subspace of size d by q

    Output:
    ====================
    err -- the (relative) Frobenius norm error
    """

    q   = U.shape[1]
    A   = Uhat.T.dot(U)
    B   = Uhat.T.dot(Uhat)
    err = np.sqrt(q + np.trace(B.dot(B)) - 2*np.trace(A.dot(A.T)))
    if relative_error_flag:
        err = err / np.sqrt(q)
    return err
    #return np.linalg.norm(np.dot(Uhat, Uhat.T) - np.dot(U, U.T), ord='fro')



def reconstruction_error(Uhat, X, normsX):
    # Compute the mean relative l2 reconstruction error of the vectors in X
	res = X - Uhat.dot(Uhat.T.dot(X))
	res_norms = np.sum(np.abs(res)**2,0)**0.5
	return np.mean(res_norms / normsX)

def strain_error(Y, XX, normXX):
    # Compute the strain cost function error relative to norm of X^TX
    return np.linalg.norm(Y.T.dot(Y) - XX, 'fro') / normXX



def load_dataset(dataset_name ,return_U = True, q = None):
    '''

    Parameters
    ----------
    dataset_name: str
        name of dataset

    return_U: bool
        whether to also compute the eigenvetor matrix

    Returns
    -------
        X: ndarray
            generated samples

        U: ndarray
            ground truth eigenvectors

        lam: ndarray
            ground truth eigenvalues

    '''

    ld = loadmat(dataset_name)
    fea = ld['fea']
    # gnd = ld['gnd']
    # center data
    X = fea.astype(np.float)
    X -= X.mean(0)[None, :]
    if return_U:
        if q is None:
            q = X.shape[-1]
        pca = PCA(n_components=q)
        pca.fit(X)
        U = pca.components_.T
        lam = pca.explained_variance_
        return X.T, U, lam

    else:
        return X



def generate_samples(d, q, n, options=None):
    '''
    
    Parameters
    ----------
    d: int 
        number of features
    
    q: int
        number of components
    
    n: int 
        number of samples
    
    options: dict
        specific of each method (see code)

    Returns
    -------
        X: ndarray
            generated samples

        U: ndarray
            ground truth eigenvectors

        lam: ndarray
            ground truth eigenvalues


    '''
    # Generate synthetic data samples from a specified model
    if options is None:
        options = {
            'method': 'spiked_covariance',
            'lambda_q': 5e-1,
            'normalize': True,
            'rho': 1e-2 / 5,
            'return_U': True
        }

    method = options['method']

    if method == 'spiked_covariance':
        rho       = options['rho']
        normalize = options['normalize']

        if normalize:
            lambda_q = options['lambda_q']
            sigma    = np.sqrt(np.linspace(1, lambda_q, q))
        else:
            slope    = options['slope']
            gap      = options['gap']
            sigma    = np.sqrt(gap + slope * np.arange(q-1,-1,-1))

        U,_ = np.linalg.qr(np.random.normal(0,1,(d,q)))

        w   = np.random.normal(0,1,(q,n))
        X   = np.sqrt(rho) * np.random.normal(0,1,(d,n))

        X  += U.dot( (w.T*sigma).T)

        if options['return_U']:
            lam = (sigma**2)[:,np.newaxis]
            return X, U, lam
        else:
            return X

    else:
        assert 0, 'Specified method for data generation is not yet implemented!'

