# Title: General Demo file
# Description: Demo testing on spiked covariance and face datasets different online PCA algorithms
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)
##############################
from online_psp.incremental_pca import IPCA
from ccipca import CCIPCA
from online_psp.fast_similarity_matching import FSM
import numpy as np
import pylab as pl
import time
from online_psp.util import subspace_error, generate_samples

q = 16
n_epoch = 50
err_its = 64
# Simulation parameters
compute_error = True
scale_data = True
spiked_covariance_test = False
init_ortho = True

if spiked_covariance_test:
    print('** spiked_covariance')
    d, n = 900, 10000
    X, U, sigma2 = generate_samples(q, n, d, method='spiked_covariance', scale_data=scale_data)
    dset = 'spiked_covariance'
else:
    dsets = ['ATT_faces_112_92.mat', 'ORL_32x32.mat', 'YaleB_32x32.mat', 'MNIST.mat']
    dset = dsets[1]
    print('** ' + dset)
    options = {
        'filename': './datasets/' + dset,
        'return_U': True
    }
    X, U, sigma2 = generate_samples(q, N='auto', D=None,
                                    method='real_data', options=options, scale_data=scale_data,
                                    sample_with_replacement=True)
    d, n = X.shape

# %% initialization
# TODO: decide on what initialization we like
lambda_1 = 1e-8 * np.ones(shape=(q,))  # abs(np.random.normal(0, 1, (q,)))# / np.sqrt(q)
Uhat0 = X[:, :q] / np.sqrt((X[:, :q] ** 2).sum(0))

if init_ortho:
    # Optionally orthogonalize the initial guess
    Uhat0, _ = np.linalg.qr(Uhat0)

ccipca = CCIPCA(q, d, Uhat0=Uhat0, sigma2_0=lambda_1,
                cython='auto')
lambda_1 *= 0
ipca = IPCA(q, d, Uhat0=Uhat0, sigma2_0=lambda_1)
scal = 100
lr = lambda t: 1/(t + 5)
# lr = lambda t: 1e-3

if_mm_pca = FSM(q, d, W0=Uhat0.T / scal,
                Minv0=scal * np.eye(q),
                learning_rate=lr)

algorithms = {'ipca': ipca, 'if_mm_pca': if_mm_pca, 'ccipca': ccipca}
# %% RUN ALGORITHMS

times = dict()
errs = dict()
for name, algo in algorithms.items():
    print('Starting algorithm %s' % name)
    err = []
    time_1 = time.time()
    for n_e in range(n_epoch):
        for its, x in enumerate(X.T):
            algo.fit_next(x)
            if its % err_its == 0:
                err.append(subspace_error(algo.get_components(), U[:, :q]))
    time_2 = time.time() - time_1
    errs[name] = err
    times[name] = time_2
# %% DISPLAY RESULTS
keys = list(algorithms.keys())
keys.sort()
for name in keys:
    pl.loglog(errs[name])
    pl.ylabel('relative subspace error (pop.)')
    pl.xlabel('samples')
    # print('Elapsed time ' + name + ':' + str(times[name]))
    print('Final subspace error ' + name + ':' + str(errs[name][-1]))

pl.legend(keys)
pl.show()
pl.savefig(dset[:-3] + 'png')
