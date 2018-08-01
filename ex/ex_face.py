# Title: ex_face.py
# Description: An example showing reconstruction with different image data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci and Cengiz Pehlevan
# Reference: None
##############################
from online_psp.incremental_pca import IPCA
from online_psp.ccipca import CCIPCA
from online_psp.fast_similarity_matching import FSM
from online_psp.util import subspace_error, generate_samples

import numpy as np
import pylab as pl
import time
import matplotlib

K = 64
n_epoch = 10
err_its = 50
# Simulation parameters
scale_data = True
init_ortho = True


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dsets = ['ATT_faces_112_92.mat', 'YaleB_32x32.mat', 'MNIST.mat']
dset = dsets[1]
print('** ' + dset)
options = {
    'filename': '../datasets/' + dset,
    'return_U': True
}
X, U, sigma2, avg_pix, scale_factor = generate_samples(K, N='auto', D=None,
                                                       method='real_data', options=options, scale_data=scale_data,
                                                       sample_with_replacement=True, shuffle=True,
                                                       return_scaling=True)

X1, _, _ = generate_samples(K, N='auto', D=None,
                            method='real_data', options=options, scale_data=scale_data,
                            sample_with_replacement=True, shuffle=False)

D, N = X.shape

# %% initialization
# TODO: decide on what initialization we like
# lambda_1 = 1e-8 * np.ones(shape=(K,))  # abs(np.random.normal(0, 1, (q,)))# / np.sqrt(q)
Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0))

if init_ortho:
    # Optionally orthogonalize the initial guess
    Uhat0, _ = np.linalg.qr(Uhat0)

scal = 100
lr = lambda t: 1 / (0.6 * t + 5)

fsm = FSM(K, D, W0=Uhat0.T / scal,
          Minv0=scal * np.eye(K),
          learning_rate=lr)

algorithms = {'FSM': fsm}

# %% RUN ALGORITHMS

times = {}
errs  = {}
Us    = {}

for name, algo in algorithms.items():
    print('Starting algorithm %s' % name)
    err = []
    us = []
    time_1 = time.time()
    for _ in range(n_epoch):
        for its, x in enumerate(X.T):
            algo.fit_next(x)
            if its % err_its == 0:
                us.append(algo.get_components())
                err.append(subspace_error(us[-1], U[:, :K]))

    time_2 = time.time() - time_1
    errs[name] = err
    Us[name] = us
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

# %% plot faces
pl.figure(figsize=(12, 5))
P = U.dot(U.T)

rescale = False

if 'MNIST' in dset:
    dims = (28, 28)
    idx = 2400
elif 'YaleB_32x32' in dset:
    dims = (32, 32)
    idx = 2400
else:
    dims = (92, 112)
    idx = 200

row, cols = 1, 7
for id_name, name in enumerate(keys):
    for id_step, i in enumerate(np.round(np.logspace(0, 2.6, 5)).astype(np.int)):
        err = errs[name][i]
        img = X1[:, idx]
        Uhat = Us[name][i]
        Phat = Uhat.dot(Uhat.T)
        if rescale:
            img_hat = Phat.dot(img) / scale_factor + avg_pix
            img_pca = P.dot(img) / scale_factor + avg_pix
            img_orig = img / scale_factor + avg_pix
        else:
            img_hat = Phat.dot(img)
            img_pca = P.dot(img)
            img_orig = img

        pl.subplot(row, cols, id_step + 1)
        pl.imshow(img_hat.reshape(dims).T, cmap='gray')
        pl.title('it:' + str(i * err_its) + ' PE:' + "{:.2f}".format(errs[name][i]))
        pl.axis('off')

    pl.subplot(row, cols, id_step + 2)
    pl.imshow(img_pca.reshape(dims).T, cmap='gray')
    pl.title('PCA')
    pl.axis('off')

    pl.subplot(row, cols, id_step + 3)
    pl.imshow(img.reshape(dims).T, cmap='gray')
    pl.title('Original')
    pl.axis('off')

pl.show()
