# Title: online_pca_simulations.py
# Description: A function for testing an online PCA algorithm
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None

#TODO COMMENTS

##############################
# Imports
import numpy as np

import util
import time

from fast_similarity_matching import FSM
from ccipca import CCIPCA
from incremental_pca import IPCA
from similarity_matching import SM

from sklearn.decomposition import PCA

from collections import defaultdict
from matplotlib import pyplot as plt


##############################


class Timer:
    # A simple timer class for performance profiling
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def run_simulation(simulation_options, generator_options, algorithm_options):
    # Different types of error to compute:
    # - The relative Frobenius norm error ||UU' - UhatUhat'||_F / K, relative to either the population PCA or the sample PCA
    # - The relative strain cost function (on some test data)  ||X'X - Y'Y||_F / ||X'X||_F

    # Unpack some parameters
    # TODO: default values
    D = simulation_options['D']
    K = simulation_options['K']
    N = simulation_options['N']

    n_epoch    = simulation_options['n_epoch']
    pca_init   = simulation_options['pca_init']
    init_ortho = simulation_options['init_ortho']

    pca_algorithm = algorithm_options['pca_algorithm']

    # We wrap things in a default dict so we don't have to check if keys exist
    error_options = defaultdict(int, simulation_options['error_options'])
    compute_error = any(error_options)

    if error_options['compute_proj_error']:
        assert not (error_options['compute_batch_error'] or error_options['compute_population_error']), 'Cannot compute proj_error at the same time as other errors!'


    if not error_options['n_skip']:
        error_options['n_skip'] = 1

    if compute_error:
        # We will make a list of functions that take in the current iterate and return an error measure
        error_options['error_func_list'] = []

    generator_options = generator_options.copy()

    if error_options['compute_population_error']:
        generator_options['return_U'] = True

        X, U_pop, sigma2 = util.generate_samples(K, N, D, method=generator_options['method'],
                                                 scale_data=generator_options['scale_data'],
                                                 options=generator_options, sample_with_replacement=True,
                                                 shuffle=generator_options['shuffle'])
    else:
        generator_options['return_U'] = False
        X = util.generate_samples(K, N, D, method=generator_options['method'],
                                  scale_data=generator_options['scale_data'],
                                  options=generator_options, sample_with_replacement=True,
                                  shuffle=generator_options['shuffle'])
    # If N was auto, we must get D and N from the data
    D, N = X.shape
    print('Running simulation on input of shape:', X.shape)


    # Add all the error computations that we want
    if error_options['compute_population_error']:
        # Compute the subspace error of the approximation versus the population eigenvectors (use pop not sample)
        error_options['error_func_list'].append(('population_err', lambda Uhat: util.subspace_error(Uhat, U_pop)))

    if error_options['compute_batch_error'] or error_options['compute_proj_error']:
        # Compute the subspace error of the approximation versus the offline estimate of the eigenvectors (use sample not pop)
        pca = PCA(n_components=K, svd_solver='arpack')
        pca.fit(X.T)
        U_batch = pca.components_.T

    if error_options['compute_batch_error']:
        error_options['error_func_list'].append(('batch_err', lambda Uhat: util.subspace_error(Uhat, U_batch)))


    if error_options['compute_reconstruction_error']:
        Xtest      = X
        normsXtest = np.sum(np.abs(Xtest) ** 2, 0) ** 0.5

        # TODO: allow in-sample testing? does this even work anymore
        error_options['error_func_list'].append(
            ('recon_err', lambda Uhat: util.reconstruction_error(Uhat, Xtest, normsXtest)))

    if error_options['compute_proj_error']:
        error_options['error_func_list'].append(
            ('proj_err', lambda Uhat: util.proj_error(Uhat, U_batch)))


    if pca_init:
        # Initialize using pca_init number of data points
        N0 = pca_init
        U, s, V = np.linalg.svd(X[:, :pca_init], full_matrices=False)
        Uhat0 = U[:, :K]
    else:
        # Initialize using just the first data point and the rest random
        N0 = 0
        Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0))

    if init_ortho:
        # Optionally orthogonalize the initial guess
        Uhat0, r = np.linalg.qr(Uhat0)



    print('Starting simulation with algorithm: ' + pca_algorithm)

    if pca_algorithm == 'CCIPCA':
        sigma2_0 = 1e-8 * np.ones(K)
        pca_fitter = CCIPCA(K, D, cython='auto', Uhat0=Uhat0, sigma2_0=sigma2_0)

    elif pca_algorithm == 'IPCA':
        sigma2_0 = np.zeros(K)
        pca_fitter = IPCA(K, D, Uhat0=Uhat0, sigma2_0=sigma2_0)

    elif pca_algorithm == 'FSM':
        scal = algorithm_options.get('scal', 100)
        gamma = algorithm_options.get('gamma', 2)

        Minv0 = np.eye(K) * scal
        Uhat0 = Uhat0 / scal

        def learning_rate(t):
            step = 1.0 / (gamma*t + 5)
            return step
        pca_fitter = FSM(K, D, W0=Uhat0.T, Minv0=Minv0, learning_rate=learning_rate)

    elif pca_algorithm == 'SM':
        scal = algorithm_options.get('scal', 100)
        gamma = algorithm_options.get('gamma', 2)

        M0 = np.eye(K) / scal
        Uhat0 = Uhat0 / scal

        def learning_rate(t):
            step = 1.0 / (gamma*t + 5)
            return step

        pca_fitter = SM(K, D, W0=Uhat0.T, M0=M0, learning_rate=learning_rate)

    else:
        assert 0, 'You did not specify a valid algorithm.  Please choose one of:\n \tCCIPCA, IPCA, SM, FSM'

    if compute_error:
        # Compute errors, do not time algorithms
        n_its = X[:, N0:].shape[1] * n_epoch
        errs = util.initialize_errors(error_options, n_its)
        i = 0

        for iter_epoch in range(n_epoch):
            # reshuffle each epoch if required
            if generator_options['shuffle']:
                order = np.random.permutation(np.arange(N0,X.shape[-1]))
            else:
                order = np.arange(N0, X.shape[-1])

            for idx_sample in order:
                x = X.T[idx_sample]
                pca_fitter.fit_next(x)
                Uhat = pca_fitter.get_components()
                util.compute_errors(error_options, Uhat, i, errs)
                i += 1
        return errs

    else:
        # Do timing, do not compute errors
        with Timer() as t:
            for _ in range(n_epoch):
                for x in X[:, N0:].T:
                    pca_fitter.fit_next(x)
        print('%s took %f sec.' % (pca_algorithm, t.interval))
        return t.interval


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    error_options = defaultdict(int, simulation_options['error_options'])
    compute_error = any(error_options)

    errs = run_simulation(simulation_options, generator_options, algorithm_options)

    handles = []
    if compute_error:
        fig = plt.figure(1)
        plt.title(algorithm_options['pca_algorithm'])

        for err_name in errs:
            err = errs[err_name]
            err = err[err > 0]
            print(err_name + ': %e' % (err[-1]))
            plt.loglog(err, label=err_name)

        plt.legend()
        plt.ylabel('Error (log10 scale)')
        plt.xlabel('Iteration (x n_skip)')

        plt.show()



if __name__ == "__main__":

    error_options = {
        'n_skip': 64,
        'compute_batch_error': True,
        'compute_population_error': True,
        'compute_reconstruction_error': False,
    }

    spiked_covariance = True
    scale_data        = True

    if spiked_covariance:
        generator_options = {
            'method': 'spiked_covariance',
            'lambda_K': 5e-1,
            'normalize': True,
            'rho': 1e-2 / 5,
            'scale_data': scale_data,
            'shuffle': False
        }

        simulation_options = {
            'D': 100,
            'K': 20,
            'N': 5000,
            'N0': 0,
            'n_epoch': 1,
            'error_options': error_options,
            'pca_init': False,
            'init_ortho': True
        }
    else:
        # Use real data
        dsets = ['ATT_faces_112_92.mat', 'YaleB_32x32.mat', 'MNIST.mat']
        dset = dsets[1]
        print('** ' + dset)
        generator_options = {
            'method': 'real_data',
            'filename': './datasets/' + dset,
            'scale_data': scale_data,
            'shuffle': False
        }
        simulation_options = {
            'D': None,
            'K': 16,
            'N': 'auto',  # can set a number here, will sample with replacement if needed
            'N0': 0,
            'n_epoch': 50,
            'error_options': error_options,
            'pca_init': False,
            'init_ortho': True,
        }

    algos = ['FSM', 'IPCA', 'CCIPCA', 'SM']
    algo = algos[0]

    algorithm_options = {
        'pca_algorithm': algo
    }
    run_test(generator_options=generator_options, simulation_options=simulation_options,
             algorithm_options=algorithm_options)
