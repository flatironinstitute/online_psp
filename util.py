# Title: util.py
# Description: Various utilities useful for online PCA tests
# Author: Victor Minden (vminden@flatironinstitute.org)

##############################
# Imports
import numpy as np
##############################

def subspace_error(Uhat, U):

	"""
	Parameters:
	====================
	Uhat -- The approximation Uhat of an orthonormal basis for the PCA subspace of size d by q
	U    -- An orthonormal basis for the PCA subspace of size d by q

	Output:
	====================
	err -- the Frobenius norm error
	"""

	return np.linalg.norm(np.dot(Uhat, Uhat.T) - np.dot(U, U.T), ord='fro')