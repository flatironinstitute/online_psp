# Title: online_pca_simulations.py
# Description: A function for testing an online PCA algorithm
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None
#%%
##############################
# Imports
import numpy as np
import os
from online_psp import util
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


def run_simulation(output_folder, simulation_options, generator_options, algorithm_options):
    # Different types of error to compute:
    # - The relative Frobenius norm error ||UU' - UhatUhat'||_F / q, relative to either the population PCA or the sample PCA
    # - The relative strain cost function (on some test data)  ||X'X - Y'Y||_F / ||X'X||_F
    # - The mean relative reconstruction error (of some test data) mean(||X[:,i] - UU'X[:,i]||_2 / ||X[:,i]||_2)
    ######NOT IMPLEMENTED######## - Compare the iterate to the sample PCA of all data up to the current data point (online)
    #######NOT IMPLEMENTED ###### - Compare the sample PCA of the data to the population PCA of the data (discrepancy)

    # Unpack some parameters
    # TODO: default values
    d = simulation_options['d']
    q = simulation_options['q']
    

    n = simulation_options['n']
    n0 = simulation_options['n0']
    n_epoch = simulation_options['n_epoch']
    pca_init = simulation_options['pca_init']
    init_ortho = simulation_options['init_ortho']

    pca_algorithm = algorithm_options['pca_algorithm']

    # We wrap things in a default dict so we don't have to check if keys exist
    error_options = defaultdict(int, simulation_options['error_options'])
    compute_error = any(error_options)
    q_true = q

    if error_options['compute_proj_error']:
        q_true = simulation_options.get('q_true', q)
        assert not (error_options['compute_batch_error'] or error_options['compute_population_error']), 'Cannot compute proj_error at the same time as other errors!'


    if not error_options['n_skip']:
        error_options['n_skip'] = 1

    if compute_error:
        # We will make a list of functions that take in the current iterate and return an error measure
        error_options['error_func_list'] = []

    generator_options = generator_options.copy()
    # for problems with real data (small number of samples)

    if error_options['compute_population_error']:
        generator_options['return_U'] = True

        X, U_pop, sigma2 = util.generate_samples(q_true, n, d, method=generator_options['method'],
                                                 scale_data=generator_options['scale_data'],
                                                 options=generator_options, sample_with_replacement=True, shuffle=generator_options['shuffle'])


    else:
        generator_options['return_U'] = False
        X = util.generate_samples(q_true, n, d, method=generator_options['method'],
                                  scale_data=generator_options['scale_data'],
                                  options=generator_options, sample_with_replacement=True, shuffle=generator_options['shuffle'])

    d, n = X.shape

    print('RUNNING SIMULATION ON INPUT DATA:', X.shape)
    # if error_options['compute_strain_error'] or error_options['compute_reconstruction error']:


    if error_options['compute_population_error']:
        # Compute the subspace error of the approximation versus the population eigenvectors (use pop not sample)
        error_options['error_func_list'].append(('population_err', lambda Uhat: util.subspace_error(Uhat, U_pop)))

    if error_options['compute_batch_error'] or error_options['compute_proj_error']:
        # Compute the subspace error of the approximation versus the offline estimate of the eigenvectors (use sample not pop)
        # U, s, _ = np.linalg.svd(X, full_matrices=0)
        # eig_val = s[:q] ** 2 / X.shape[-1]
        # U_batch = U[:, :q]

        pca = PCA(n_components=q, svd_solver='arpack')
        pca.fit(X.T)
        U_batch = pca.components_.T
        eig_val = pca.explained_variance_

        # eig_val, V = np.linalg.eigh(X.dot(X.T) / n)
        # idx = np.flip(np.argsort(eig_val), 0)
        # eig_val = eig_val[idx]
        # V = V[:, idx]

    if error_options['compute_batch_error']:
        error_options['error_func_list'].append(('batch_err', lambda Uhat: util.subspace_error(Uhat, U_batch)))

    # if error_options['compute_strain_error']:
    #     # TODO: what is this does it even work anymore
    #     error_options['error_func_list'].append(
    #         ('strain_err', lambda Uhat: util.strain_error(Uhat.T.dot(Xtest), XXtest, normsXXtest)))

    if error_options['compute_reconstruction_error']:
        Xtest = X
        # XXtest = Xtest.T.dot(Xtest)
        normsXtest = np.sum(np.abs(Xtest) ** 2, 0) ** 0.5
        # normsXXtest = np.linalg.norm(XXtest, 'fro')
        # TODO: allow in-sample testing? does this even work anymore
        error_options['error_func_list'].append(
            ('recon_err', lambda Uhat: util.reconstruction_error(Uhat, Xtest, normsXtest)))

    if error_options['compute_proj_error']:
        error_options['error_func_list'].append(
            ('proj_err', lambda Uhat: util.proj_error(Uhat, U_batch)))


    if pca_init:
        # Initialize using pca_init number of data points
        n0 = pca_init
        U, s, V = np.linalg.svd(X[:, :pca_init], full_matrices=False)
        Uhat0 = U[:, :q]
    else:
        # Initialize using just the first data point and the rest random
        n0 = 0
        Uhat0 = X[:, :q] / np.sqrt((X[:, :q] ** 2).sum(0))
        # Uhat0 = np.random.normal(0, 1, (d, q)) / np.sqrt(d)
        # Uhat0[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0], 2)



    if init_ortho:
        # Optionally orthogonalize the initial guess
        Uhat0, r = np.linalg.qr(Uhat0)

    print('Starting simulation with algorithm: ' + pca_algorithm)

    if pca_algorithm == 'CCIPCA':
        # TODO: Maybe lambda0 should be sorted in descending order?
        lambda0 = 1e-8 * np.ones(q)  # np.sort(np.abs(np.random.normal(0, 1, (q,))))[::-1]
        pca_fitter = CCIPCA(q, d, cython='auto', Uhat0=Uhat0, sigma2_0=lambda0)
    elif pca_algorithm == 'incremental_PCA':
        tol = algorithm_options['tol']
        lambda0 = np.zeros(q)  # 0*np.sort(np.abs(np.random.normal(0, 1, (q,))))[::-1]
        pca_fitter = IPCA(q, d, Uhat0=Uhat0, sigma2_0=lambda0, tol=tol)
    elif pca_algorithm == 'if_minimax_PCA':
        # What if we scale W0 down and Minv0 up
        tau = algorithm_options['tau']
        scal = 100
        Minv0 = np.eye(q) * scal
        Uhat0 = Uhat0 / scal
        def learning_rate(t):
            step = 1.0 / (algorithm_options['t']*t + 5)
            return step
        pca_fitter = FSM(q, d, W0=Uhat0.T, Minv0=Minv0, tau=tau, learning_rate=learning_rate)
    elif pca_algorithm == 'minimax_PCA':
        tau = algorithm_options['tau']
        scal = 100
        M0 = np.eye(q) / scal
        Uhat0 = Uhat0 / scal

        def learning_rate(t):
            step = 1.0 / (algorithm_options['t']*t + 5)
            return step

        pca_fitter = SM(
            q, d, W0=Uhat0.T, M0=M0, tau=tau, learning_rate=learning_rate)

    else:
        assert 0, 'You did not specify a valid algorithm.  Please choose one of:\n \tCCIPCA, incremental_PCA, if_minimax_PCA'

    if compute_error:
        # Compute errors, do not time algorithms
        n_its = X[:, n0:].shape[1] * n_epoch
        errs = util.initialize_errors(error_options, n_its)
        i = 0
        for iter_epoch in range(n_epoch):

            # reshuffle each epoch if required
            if generator_options['shuffle']:
                order = np.random.permutation(np.arange(n0,X.shape[-1]))
            else:
                order = np.arange(n0, X.shape[-1])

            for idx_sample in order:
                x = X.T[idx_sample]
                pca_fitter.fit_next(x)
                Uhat = pca_fitter.get_components()
                util.compute_errors(error_options, Uhat, i, errs)
                i += 1

    else:
        # Do timing, do not compute errors
        # TODO: save the timing information, don't just print it out
        with Timer() as t:
            for _ in range(n_epoch):
                for x in X[:, n0:].T:
                    pca_fitter.fit_next(x)
        print('%s took %f sec.' % (pca_algorithm, t.interval))

    # Output results to some specified folder with information filename (but long)
    rho = generator_options.get('rho',0)
    filename = output_folder + '/%s_d_%d_q_%d_n_%d_nepoch_%d_rho_%.2f' %(pca_algorithm, d, q, n, n_epoch, rho)
    if compute_error:
        filename += '_error'
    else:
        filename += '_timing'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename += '_%s' % timestr
    filename += '.mat'

    output_dict = {
        'generator_options': generator_options,
        'simulation_options': simulation_options,
        'algorithm_options': algorithm_options,
        'd': d,
        'q': q,
        'n': n,
        'n_epoch': n_epoch,
        'n0': n0,
    }

    if compute_error:
        for err_name in errs:
            errs[err_name] = errs[err_name][np.nonzero(errs[err_name])]
        output_dict['errs'] = errs
    else:
        output_dict['runtime'] = t.interval

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if output_dict['simulation_options']['d'] is None:
        output_dict['simulation_options']['d'] = 'None'
    if output_dict['simulation_options']['n'] is None:
        output_dict['simulation_options']['n'] = 'None'

    # sio.savemat(filename, output_dict)
    # filename  = filename[:-3] + 'npz'
    # np.savez(filename,**output_dict)

    # print('Output written to %s' % filename)

    if compute_error:
        return errs
    else:
        return t.interval


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    output_folder = os.getcwd() + '/test'

    errs = run_simulation(output_folder, simulation_options,
                          generator_options, algorithm_options)

    handles = []
    # TODO: This will break if we don't compute errors
    if errs:
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        plt.title(algorithm_options['pca_algorithm'])

        for err_name in errs:
            print(err_name + ': %f' % (errs[err_name][-1]))
            handle, = ax.plot(errs[err_name], label=err_name)
            handles.append(handle)

        plt.legend(handles=handles)
        plt.ylabel('Error (log10 scale)')
        plt.xlabel('Iteration')
        # plt.ylim(ymax=1, ymin=0)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()


if __name__ == "__main__":

    error_options = {
        'n_skip': 64,
        # TODO remove ortho option, it is duplicated
        'orthogonalize_iterate': False,
        'compute_batch_error': True,
        'compute_population_error': True,
        'compute_strain_error': False,
        'compute_reconstruction_error': False,
    }

    spiked_covariance = False
    scale_data = True

    if spiked_covariance:
        generator_options = {
            'method': 'spiked_covariance',
            'lambda_q': 5e-1,
            'normalize': True,
            'rho': 1e-2 / 5,
            'scale_data': scale_data,
            'shuffle': False
        }

        simulation_options = {
            'd': 100,
            'q': 20,
            'q_true':10,
            'n': 5000,
            'n0': 0,
            'n_epoch': 1,
            'error_options': error_options,
            'pca_init': False,
            'init_ortho': True,
        }
    else:
        dsets = ['ATT_faces_112_92.mat', 'ORL_32x32.mat', 'YaleB_32x32.mat', 'MNIST.mat']
        dset = dsets[1]
        print('** ' + dset)
        generator_options = {
            'method': 'real_data',
            'filename': './datasets/' + dset,
            'scale_data': scale_data,
            'shuffle': False
        }
        simulation_options = {
            'd': None,
            'q': 16,
            'n': 'auto',  # can set a number here, will select frames multiple times
            'n0': 0,
            'n_epoch': 50,
            'error_options': error_options,
            'pca_init': False,
            'init_ortho': True,
        }

    algos = ['if_minimax_PCA', 'incremental_PCA', 'CCIPCA', 'minimax_PCA']
    algo = algos[0]
    algorithm_options = {
        'pca_algorithm': algo,
        'tau': 0.5,
        'tol': 1e-7
    }
    run_test(generator_options=generator_options, simulation_options=simulation_options,
             algorithm_options=algorithm_options)