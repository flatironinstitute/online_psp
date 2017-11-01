# Title: util.py
# Description: Various utilities useful for online PCA tests
# Author: Victor Minden (vminden@flatironinstitute.org)

##############################
# Imports
import numpy as np
##############################

def compute_errors(error_options, Uhat, t, errs,M=None):
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
    return { fun_name : np.zeros(n_its) for (fun_name, _) in error_options['error_func_list'] }


def diag_error(M, relative_error_flag=True):
    err = np.linalg.norm(M - np.diag(np.diag(M)),ord='fro')
    if relative_error_flag:
        err = err / np.linalg.norm(M,ord='fro')
    return err

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



def whitening_error(Uhat, U, sigma2, relative_error_flag=True):
    """
    Parameters:
    ====================
    Uhat -- The approximation Uhat of an orthonormal basis for the PCA subspace of size d by q
    U    -- An orthonormal basis for the PCA subspace of size d by q

    Output:
    ====================
    err -- the (relative) Frobenius norm error
    """
    B = np.dot(U, sigma2**(-1) * U.T )

    err = np.linalg.norm(np.dot(Uhat, Uhat.T) - B, ord='fro')
    if relative_error_flag:
        err /= np.linalg.norm(B,ord='fro')
    return err


def alignment_error(Uhat, U):
    err = 0
    for i in range(U.shape[1]):
        colhat = Uhat[:,i]
        colhat = colhat / np.linalg.norm(colhat)
        col    = U[:,i]
        angle  = col.dot(colhat)
        err = err + (1-abs(angle))

    return err / U.shape[1]



def reconstruction_error(Uhat, X, normsX):
	res = X - Uhat.dot(Uhat.T.dot(X))
	res_norms = np.sum(np.abs(res)**2,0)**0.5
	return np.mean(res_norms / normsX)
    #return np.linalg.norm(X - Uhat.dot(Uhat.T.dot(X)), 'fro') / normX

def strain_error(Y, XX, normXX):
    return np.linalg.norm(Y.T.dot(Y) - XX, 'fro') / normXX


def generate_samples(d, q, n, options):
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
            return {'X': X, 'U' : U, 'sigma2' : (sigma**2)[:,np.newaxis]}
        else:
            return {'X' : X}

    else:
        assert 0, 'Specified method for data generation is not yet implemented!'
