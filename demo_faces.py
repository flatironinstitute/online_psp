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
from util import subspace_error, load_dataset
q = 5
for dset in ['ATT_faces_112_92.mat','ORL_32x32.mat','YaleB_32x32.mat']:
    print('** ' + dset)
    pl.figure()
    #%% GENERATE TEST DATA
    X, U, sigma2 = load_dataset(dataset_name='./datasets/'+dset, return_U=True, q=None)

    #%%
    n_epoch = 1
    d, n = X.shape[0], X.shape[1]
    lambda_1 = np.random.normal(0, 1, (q,)) / np.sqrt(q)
    # Parameters IF_minimax_PCA_CLASS
    tau = 0.5
    # Simulation parameters
    compute_error = True

    #%% RUN ALGORITHMS
    errs = []
    Uhat0 = X[:, :q]/np.sqrt(X[:, :q]**2).sum(0)
    ipca = IncrementalPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1)
    if_mm_pca = IF_minimax_PCA_CLASS(q, d, W0=Uhat0.T, Minv0=None, tau=tau, learning_rate=None)
    ccipca = CCIPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda_1, cython='auto', in_place=False)
    algorithms = {'ipca':ipca, 'if_mm_pca':if_mm_pca, 'ccipca':ccipca}

    times = dict()
    errs = dict()
    for name, algo in algorithms.items():
        err = []
        time_1 = time.time()
        for n_e in range(n_epoch):
            for x in X.T:
                algo.fit_next(x)
                err.append(subspace_error(algo.get_components(), U[:, :q]))
        time_2 = time.time() - time_1
        errs[name] = err
        times[name] = time_2
    #%% DISPLAY RESULTS
    pl.close('all')
    keys = list(algorithms.keys())
    keys.sort()
    for name in keys:
        pl.semilogy(errs[name])
        pl.xlabel('relative subspace error')
        pl.xlabel('samples')
        # print('Elapsed time ' + name + ':' + str(times[name]))
        print('Final subspace error ' + name + ':' + str(errs[name][-1]))

    pl.legend(keys)
    # pl.show()
    pl.savefig(dset[:-3]+'png')