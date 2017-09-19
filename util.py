# Title: util.py
# Description: Various utilities useful for online PCA tests
# Author: Victor Minden (vminden@flatironinstitute.org)

##############################
# Imports
import numpy as np
##############################


# TODO: Add a whole bunch of other errors matching Online_PCA_simulations.m (defunct)
def subspace_error(Uhat, U, relative_error_flat=True):

	"""
	Parameters:
	====================
	Uhat -- The approximation Uhat of an orthonormal basis for the PCA subspace of size d by q
	U    -- An orthonormal basis for the PCA subspace of size d by q

	Output:
	====================
	err -- the (relative) Frobenius norm error
	"""

	q = U.shape[1]
	A = Uhat.T.dot(U)
	B = Uhat.T.dot(Uhat)

	return np.sqrt(1 + ( np.trace(B.dot(B)) - 2*np.trace(A.dot(A.T))) / q)
	#return np.linalg.norm(np.dot(Uhat, Uhat.T) - np.dot(U, U.T), ord='fro')


def reconstruction_error(X, Uhat):
	pass


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

		return X,U,sigma**2

	else:
		assert 0, 'Specified method for data generation is not yet implemented!'
