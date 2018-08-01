# Title: General Demo file
# Description: Demo testing on spiked covariance different online PSP algorithms
# Author: Victor Minden (vminden@flatironinstitute.org)and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: (Cardot and Degras, 2015)
##############################
import numpy as np
import pylab as pl
import time

from online_psp.incremental_pca import IPCA
from online_psp.ccipca import CCIPCA
from online_psp.fast_similarity_matching import FSM
from online_psp.similarity_matching import SM

from online_psp.util import subspace_error, generate_samples



# Simulation parameters
K = 10
n_epoch = 5
err_its = 64

scale_data = True
init_ortho = True

D, N = 50, 10000
X, U, sigma2 = generate_samples(K, N, D, method='spiked_covariance', scale_data=scale_data)

# %% initialization
lambda_1 = 1e-8 * np.ones(shape=(K,))
Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0))

if init_ortho:
    # Optionally orthogonalize the initial guess
    Uhat0, _ = np.linalg.qr(Uhat0)

# Build fitters of each type
ccipca = CCIPCA(K, D, Uhat0=Uhat0, sigma2_0=lambda_1,
                cython='auto')
lambda_1 *= 0
ipca = IPCA(K, D, Uhat0=Uhat0, sigma2_0=lambda_1)

scal = 100
lr = lambda t: 1/(t + 5)
fsm = FSM(K, D, W0=Uhat0.T / scal,
                Minv0=scal * np.eye(K),
                learning_rate=lr)
sm = SM(K, D, W0=Uhat0.T / scal,
                M0= np.eye(K) / scal,
                learning_rate=lr)

algorithms = {'IPCA': ipca, 'CCIPCA': ccipca, 'SM': sm, 'FSM': fsm}



times = {}
errs  = {}

for name, algo in algorithms.items():
    print('Starting algorithm %s' % name)
    err = []
    time_1 = time.time()
    for _ in range(n_epoch):
        for its, x in enumerate(X.T):
            algo.fit_next(x)
            if its % err_its == 0:
                err.append(subspace_error(algo.get_components(), U[:, :K]))
    time_2 = time.time() - time_1
    errs[name] = err
    times[name] = time_2

# DISPLAY RESULTS
keys = list(algorithms.keys())
keys.sort()

for name in keys:
    pl.loglog(errs[name])
    pl.ylabel('relative subspace error (pop.)')
    pl.xlabel('samples')

    print('Final subspace error ' + name + ':' + str(errs[name][-1]))

pl.legend(keys)
pl.show()