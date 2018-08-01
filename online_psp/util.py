# Title: util.py
# Description: Various utilities useful for online PSP tests
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)

#TODO: comments and docstrings

##############################
# Imports
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA


##############################

def compute_errors(error_options, Uhat, t, errs, M=None):
    if t % error_options['n_skip']:
        return

    for i, (fname, f) in enumerate(error_options['error_func_list']):
        errs[fname][t] = f(Uhat)

def proj_error(Uhat, U, relative_error_flag=True):
    K_true = U.shape[1]
    A = Uhat.T.dot(U)
    err = np.linalg.norm(U - Uhat.dot(A))
    if relative_error_flag:
        err = err / np.sqrt(K_true)
    return err



def initialize_errors(error_options, n_its):
    # Build a dictionary for storing the error information for each specified error function
    return {fun_name: np.zeros(n_its) for (fun_name, _) in error_options['error_func_list']}


def subspace_error(Uhat, U, relative_error_flag=True):
    """
    Parameters:
    ====================
    Uhat -- The approximation Uhat of an orthonormal basis for the PCA subspace of size D by K
    U    -- An orthonormal basis for the PCA subspace of size D by K

    Output:
    ====================
    err -- the (relative) Frobenius norm error
    """

    K = U.shape[1]
    A = Uhat.T.dot(U)
    B = Uhat.T.dot(Uhat)
    err = np.sqrt(K + np.trace(B.dot(B)) - 2 * np.trace(A.dot(A.T)))
    if relative_error_flag:
        err = err / np.sqrt(K)
    return err


def load_dataset(dataset_name, return_U=True, K=None):
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
    X = fea.astype(np.float)
    X -= X.mean(0)[None, :]

    if return_U:
        if K is None:
            K = X.shape[-1]

        pca = PCA(n_components=K, svd_solver='arpack')
        pca.fit(X)
        U = pca.components_.T
        lam = pca.explained_variance_
        X = X.T
    else:
        U = 0
        lam = 0
        X = X.T

    return X, U, lam


def get_scale_data_factor(X):
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

    norm_fact = np.mean(np.sqrt(np.sum(X ** 2, 0)))
    scale_factor = 1 / norm_fact

    return scale_factor


def generate_samples(K=None, N=None, D=None, method='spiked_covariance', options=None, scale_data=True,
                     sample_with_replacement=False, shuffle=False):
    '''
    
    Parameters
    ----------

    D: int or None
        number of features
    
    K: int
        number of components
    
    N: int or 'auto'
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

        sigma2: ndarray
            ground truth eigenvalues


    '''
    # Generate synthetic data samples  from a specified model or load real datasets
    # here making sure that we use the right n when including n_test frames

    if method == 'spiked_covariance':
        if N == 'auto':
            raise ValueError('N cannot be "auto" for spiked_covariance model')

        if options is None:
            options = {
                'lambda_K': 5e-1,
                'normalize': True,
                'rho': 1e-2 / 5,
                'return_U': True
            }
        return_U = options['return_U']

        if N is None or D is None:
            raise Exception('Spiked covariance requires parameters N and D')

        rho = options['rho']
        normalize = options['normalize']
        if normalize:
            lambda_K = options['lambda_K']
            sigma = np.sqrt(np.linspace(1, lambda_K, K))
        else:
            slope = options['slope']
            gap = options['gap']
            sigma = np.sqrt(gap + slope * np.arange(K - 1, -1, -1))

        U, _ = np.linalg.qr(np.random.normal(0, 1, (D, K)))

        w = np.random.normal(0, 1, (K, N))
        X = np.sqrt(rho) * np.random.normal(0, 1, (D, N))

        X += U.dot((w.T * sigma).T)
        sigma2 = (sigma ** 2)[:, np.newaxis]

    elif method == 'real_data':
        if options is None:
            options = {
                'filename': './datasets/MNIST.mat',
                'return_U': True
            }
        return_U = options['return_U']
        filename = options['filename']

        X, U, sigma2 = load_dataset(filename, return_U=return_U, K=K)

        if N != 'auto':
            if N > X.shape[-1]:
                if sample_with_replacement:
                    print('** Warning: You are sampling real data with replacement')
                else:
                    raise Exception("You are sampling real data with replacement "
                                    "but sample_with_replacement flag is set to False")

            X = X[:, np.arange(N) % X.shape[-1]]

    else:
        assert 0, 'Specified method for data generation is not yet implemented!'

    # center data
    X -= X.mean(1)[:, None]
    if scale_data:
        scale_factor = get_scale_data_factor(X)
        X, U, sigma2 = X * scale_factor, U, sigma2 * (scale_factor ** 2)

    if shuffle:
        print('Shuffling data!')
        X = X[:,np.random.permutation(X.shape[-1])]

    if return_U:
        return X, U, sigma2
    else:
        return X
