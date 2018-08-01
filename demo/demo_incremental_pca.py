import numpy as np
import pylab as pl
import time

from online_psp.util import generate_samples, subspace_error
from online_psp.incremental_pca import IPCA

print('Testing IPCA...')

#----------
# Parameters
#----------
n_epoch = 2
K       = 50
D, N    = 500, 1000
# ----------

X, U, sigma2 = generate_samples(K, N, D, method='spiked_covariance', scale_data=True)

# Initial guess
sigma2_0 = lambda0 = np.zeros(K)
Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0))

errs = []
ipca = IPCA(K, D, Uhat0=Uhat0, sigma2_0=sigma2_0)
time_1 = time.time()
for n_e in range(n_epoch):
    for x in X.T:
        ipca.fit_next(x)
        errs.append(subspace_error(ipca.get_components(), U[:, :K]))
time_2 = time.time() - time_1

# Plotting...
print('Elapsed time: ' + str(time_2))
print('Final subspace error: ' + str(subspace_error(ipca.get_components(), U[:, :K])))

pl.semilogy(errs)
pl.ylabel('Relative subspace error')
pl.xlabel('Samples (t)')
pl.show()

print('Test complete!')
