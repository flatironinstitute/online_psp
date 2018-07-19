# Title: incremental_pca.py
# Description: A function for PCA using the incremental approach in the onlinePCA R package
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)

##############################
from incremental_pca import IncrementalPCA_CLASS
from ccipca import CCIPCA_CLASS
from if_minimax_subspace_projection import IF_minimax_PCA_CLASS
import numpy as np
import pylab as pl
import time
from util import generate_samples, subspace_error   
#%% GENERATE TEST DATA
# Parameters
n_epoch = 1
d, q, n = 200, 20, 1000
X, U, sigma2 = generate_samples(d, q, n)
lambda_1 = np.random.normal(0, 1, (q,)) / np.sqrt(q)
# Parameters IF_minimax_PCA_CLASS
tau = 0.5
# Simulation parameters
compute_error = True

#%% RUN ALGORITHMS
errs  = []
# Normalize initial guess
Uhat0 = X[:, :q]/(X[:, :q]**2).sum(0)

ipca      = IncrementalPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1)
if_mm_pca = IF_minimax_PCA_CLASS(q, d, W0=Uhat0.T, Minv0=None, tau=tau)
ccipca    = CCIPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1, cython=False, in_place=False)

algorithms = {'ipca':ipca, 'if_mm_pca':if_mm_pca, 'ccipca':ccipca}

times = {}
errs  = {}
for name, algo in algorithms.items():
    err    = []
    time_1 = time.time()
    for n_e in range(n_epoch):
        for x in X.T:
            algo.fit_next(x)
            Uhat = algo.get_components()
            # TODO: decide if we want to orthogonalize the iterate or not
            Uhat,r = np.linalg.qr(Uhat)
            err.append(subspace_error(Uhat, U[:, :q]))
    time_2      = time.time() - time_1
    errs[name]  = err
    times[name] = time_2

#%% DISPLAY RESULTS
for name in algorithms.keys():
    pl.semilogy(errs[name])
    pl.xlabel('relative subspace error')
    pl.xlabel('samples')
    Uhat = algorithms[name].get_components()
    print('Algorithm: %s' % name)
    print('Elapsed time: ' + str(times[name]))
    print('Final subspace error: ' + str(subspace_error(np.asarray(Uhat), U[:, :q])) + '\n')

pl.legend(algorithms.keys())
pl.show()
