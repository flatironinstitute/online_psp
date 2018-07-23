# Title: util.py
# Description: Various utilities useful for online PCA tests
# Author: Victor Minden (vminden@flatironinstitute.org)

##############################
# Imports
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
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



def load_dataset(dataset_name, return_U=True, q=None):
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
        X = X.T
    else:
        U = 0
        lam = 0
        X = X.T


    return X, U, lam



def get_scale_data_factor(X, method='norm'):
    ''' Scaling for convergence reasons

    Parameters
    ----------
    q
    X
    U
    lambdas

    Returns
    -------

    '''
    # center
    # todo figure out why this works
    if method is not None:
        if method == 'norm':
            log_fact = 1
            norm_fact = np.mean(np.sqrt(np.sum(X ** 2, 0)))
        else:
            raise Exception('Scale data modality not known')

        scale_factor = log_fact / norm_fact

    else:
        scale_factor = 1

    return scale_factor


def generate_samples(q, n=None, d=None, method='spiked_covariance', options=None, scale_data=False,
                     sample_with_replacement=False):
    '''
    
    Parameters
    ----------

    d: int or None
        number of features
    
    q: int
        number of components
    
    n: int or 'auto'
        number of samples, if 'auto' it will return all the samples from real data datasets

    method: str
        so far 'spiked_covariance' or 'real_data'
    
    options: dict
        specific of each method (see code)

    scale_data: bool
        scaling data so that average sample norm is one




    Returns
    -------
        X: ndarray
            generated samples

        U: ndarray
            ground truth eigenvectors

        lam: ndarray
            ground truth eigenvalues


    '''
    # Generate synthetic data samples  from a specified model or load real datasets
    # here making sure that we use the right n when including n_test frames


    if method == 'spiked_covariance':
        if n =='auto':
            raise ValueError('n cannot be "auto" for spiked_covariance model')

        if options is None:
            options = {
                'lambda_q': 5e-1,
                'normalize': True,
                'rho': 1e-2 / 5,
                'return_U': True,
            }
        return_U = options['return_U']

        if n is None or d is None:
            raise Exception('Spiked covariance requires parameters n and d')

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
        lam = (sigma ** 2)[:, np.newaxis]

    elif method == 'real_data':
        if options is None:
            options = {
                # 'filename': './datasets/ATT_faces_112_92.mat',
                'filename': './datasets/ORL_32x32.mat',
                'return_U': True
            }
        return_U = options['return_U']
        filename = options['filename']

        X, U, lam = load_dataset(filename, return_U=return_U, q=q)

        if n != 'auto':
            if n > X.shape[-1]:
                if sample_with_replacement:
                    print('** Warning: You are sampling real data with replacement')
                else:
                    raise Exception("You are sampling real data with replacement "
                                    "but sample_with_replacement flag is set to False")

            X = X[:, np.arange(n) % X.shape[-1]]

    else:
        assert 0, 'Specified method for data generation is not yet implemented!'

    # center data
    X -= X.mean(1)[:, None]
    if scale_data:
        scale_factor = get_scale_data_factor(X)
        X, U, lam = X * scale_factor, U, lam * (scale_factor ** 2)

    if return_U:
        return X, U, lam
    else:
        return X

