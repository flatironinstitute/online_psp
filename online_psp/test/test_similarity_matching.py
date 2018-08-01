import numpy as np
import pylab as pl
import time

from online_psp.util import generate_samples, subspace_error
from online_psp.similarity_matching import SM

print('Testing SM...')


#----------
# Parameters
#----------
# Number of epochs
n_epoch = 2
# Size of PCA subspace to recover
K = 50
D, N = 500, 1000
scal = 100
#----------

X, U, sigma2 = generate_samples(K, N, D, method='spiked_covariance', scale_data=True)

# Initial guess
Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0)) / scal
M0    = np.eye(K) / scal

errs = []
sm = SM(K, D, W0=Uhat0.T, M0=M0)

time_1 = time.time()
for n_e in range(n_epoch):
    for x in X.T:
        sm.fit_next(x)
        errs.append(subspace_error(sm.get_components(), U[:, :K]))
time_2 = time.time() - time_1

# Plotting...
print('Elapsed time: ' + str(time_2))
print('Final subspace error: ' + str(subspace_error(sm.get_components(), U[:, :K])))

pl.semilogy(errs)
pl.ylabel('Relative subspace error')
pl.xlabel('Samples (t)')
pl.show()

print('Test complete!')
