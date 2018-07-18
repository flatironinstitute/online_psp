# Title: online_pca_simulations.py
# Description: A function for testing an online PCA algorithm
# Author: Victor Minden (vminden@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None

##############################
# Imports
import numpy as np
import scipy.io as sio
from scipy.linalg import solve as solve
import os
import util
import time

from if_minimax_subspace_projection import if_minimax_PCA_CLASS
from ccipca import CCIPCA_CLASS
from incremental_pca import incremental_PCA_CLASS
# from stochastic_gradient_pca import SGA_PCA
# from subspace_network_learning_pca import SNL_PCA
# from online_similarity_matching import OSM_PCA

from collections import defaultdict
from matplotlib import pyplot as plt
##############################




class Timer:
    # A simple timer class for performance profiling
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end      = time.perf_counter()
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
    d       = simulation_options['d']
    q       = simulation_options['q']
    n       = simulation_options['n']
    n0      = simulation_options['n0']
    n_epoch = simulation_options['n_epoch']
    n_test  = simulation_options['n_test']
    pca_init   = simulation_options['pca_init']
    init_ortho = simulation_options['init_ortho']


    pca_algorithm = algorithm_options['pca_algorithm']


    # We wrap things in a default dict so we don't have to check if keys exist
    error_options = defaultdict(int, simulation_options['error_options'])
    compute_error = any(error_options)

    if not error_options['n_skip']:
        error_options['n_skip'] = 1

    if compute_error:
        # We will make a list of functions that take in the current iterate and return an error measure
        error_options['error_func_list'] = []


    generator_options = generator_options.copy()

    if error_options['compute_population_error']:
        generator_options['return_U'] = True
        X, U_pop, sigma2 = util.generate_samples(d, q, n + n_test, generator_options)
    else:
        generator_options['return_U'] = False
        X, U_pop, sigma2 = util.generate_samples(d, q, n + n_test, generator_options)

    X, Xtest = X[:,:n], X[:,n:]
    XXtest   = Xtest.T.dot(Xtest)

    normsXtest   = np.sum(np.abs(Xtest)**2,0)**0.5
    normsXXtest  = np.linalg.norm(XXtest, 'fro')

    if error_options['compute_population_error']:
        # Compute the subspace error of the approximation versus the population eigenvectors (use pop not sample)
        error_options['error_func_list'].append(('population_err', lambda Uhat: util.subspace_error(Uhat, U_pop)))

    if error_options['compute_batch_error']:
        # Compute the subspace error of the approximation versus the offline estimate of the eigenvectors (use sample not pop)
        eig_val,V = np.linalg.eigh(X.dot(X.T) / n)
        idx       = np.flip(np.argsort(eig_val),0)
        eig_val   = eig_val[idx]
        V         = V[:,idx]
        U_batch   = V[:,:q]
        error_options['error_func_list'].append(('batch_err', lambda Uhat: util.subspace_error(Uhat, U_batch)))

    if error_options['compute_strain_error']:
        # TODO: what is this
        error_options['error_func_list'].append(('strain_err', lambda Uhat: util.strain_error(Uhat.T.dot(Xtest), XXtest, normsXXtest)))

    if error_options['compute_reconstruction_error']:
        # TODO: allow in-sample testing?
        error_options['error_func_list'].append(('recon_err', lambda Uhat: util.reconstruction_error(Uhat, Xtest, normsXtest)))

    if pca_init:
        # Initialize using pca_init number of data points
        n0 = pca_init
        U,s,V = np.linalg.svd(X[:,:pca_init], full_matrices=False)
        Uhat0 = U[:,:q]
    else:
        # Initialize using just the first data point and the rest random
        n0 = 1
        Uhat0 = np.random.normal(0,1,(d,q)) / np.sqrt(d)
        Uhat0[:,0] = X[:,0] / np.linalg.norm(X[:,0],2)

    if init_ortho:
        # Optionally orthogonalize the initial guess
        Uhat0,r = np.linalg.qr(Uhat0)


    print('Starting simulation with algorithm: ' + pca_algorithm)

    if pca_algorithm == 'CCIPCA':
        if compute_error:
            errs = CCIPCA(X[:,n0:], q, n_epoch, error_options=error_options, Uhat0=Uhat0,ell=0)
        else:
            with Timer() as t:
                *_, = CCIPCA(X[:,n0:], q, n_epoch, Uhat0=Uhat0,ell=0)
            print('CCIPCA took %f sec.' % t.interval)

    elif pca_algorithm == 'incremental_PCA':
        tol = algorithm_options['tol']
        if compute_error:
            errs = incremental_PCA(X[:,n0:], q, n_epoch, tol, error_options=error_options, Uhat0=Uhat0)
        else:
            with Timer() as t:
                *_, = incremental_PCA(X[:,n0:], q, n_epoch, tol, Uhat0=Uhat0)
            print('incremental_PCA took %f sec.' % t.interval)

    elif pca_algorithm == 'if_minimax_PCA':
        tau = algorithm_options['tau']
        if compute_error:
            errs = if_minimax_PCA(X[:,n0:], q, tau, n_epoch, error_options, W0=Uhat0.T)
        else:
            with Timer() as t:
                *_, = if_minimax_PCA(X[:,n0:], q, tau, n_epoch, W0=Uhat0.T)
            print('if_minimax_PCA took %f sec.' % t.interval)

    else:
        assert 0, 'You did not specify a valid algorithm.'


    # Output results to some specified folder with information filename (but long)
    filename = output_folder + '/%s_d_%d_q_%d_n_%d_nepoch_%d_n0_%d_ntest_%d' %(pca_algorithm, d, q, n, n_epoch, n0, n_test)
    if compute_error:
        filename += '_error'
    else:
        filename += '_timing'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename += '_%s' %timestr
    filename += '.mat'

    output_dict = {
    'generator_options' : generator_options,
    'simulation_options' : simulation_options,
    'algorithm_options' : algorithm_options,
    'd' : d,
    'q' : q,
    'n' : n,
    'n_epoch' : n_epoch,
    'n0' : n0,
    'n_test' : n_test
    }

    if compute_error:
        for err_name in errs:
            errs[err_name] = errs[err_name][np.nonzero(errs[err_name])]
        output_dict['errs'] = errs
    else:
        output_dict['runtime'] = t.interval

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sio.savemat(filename, output_dict)
    print('Output written to %s' % filename)

    if compute_error:
        return errs
    else:
        return t.interval


if __name__ == "__main__":

    # algo_names = ['CCIPCA', 'SNL_PCA', 'SGA_PCA', 'incremental_PCA', 'minimax_whitening_PCA', 'if_minimax_whitening_PCA', 'minimax_PCA', 'if_minimax_PCA', 'OSM_PCA']
    # output_folder = os.getcwd() + '/test'
    #
    # error_options = {
    #     'n_skip' : 1000, ##NOT IMPLEMENTED
    #     'orthogonalize_iterate' : False,
    #     'compute_batch_error' : True,
    #     'compute_population_error' : True,
    #     # 'compute_strain_error' : False,
    #     # 'compute_reconstruction_error' : False,
    #     #'compute_pop_whitening_error' : True,
    #     # 'compute_batch_whitening_error' : True,
    # }
    #
    # simulation_options = {
    #     'd' : 50,
    #     'q' : 32,
    #     'n' : 100000,#4096,
    #     'n0': 0, ##NOT IMPLEMENTED
    #     'n_epoch': 1,
    #     'n_test' : 256,
    #     'error_options' : error_options,
    #     'pca_init': False,
    #     'init_ortho': True
    # }
    #
    # generator_options = {
    #     'method'   : 'spiked_covariance',
    #     'lambda_q' : 5e-1,
    #     'normalize': True,
    #     'rho'      : 1e-2/5
    # }
    #
    # algorithm_options = {
    #     'pca_algorithm' : algo_names[2],
    #     'tau'           : 0.5,
    #     'tol'           : 1e-7
    # }
    #
    # if algorithm_options['pca_algorithm'] == 'minimax_whitening_PCA' or algorithm_options['pca_algorithm'] == 'if_minimax_whitening_PCA':
    #     algorithm_options['tau'] = simulation_options['q'] * algorithm_options['tau']
    #
    # errs = run_simulation(output_folder, simulation_options, generator_options, algorithm_options)
    #
    # if any(error_options):
    #     handles = []
    #
    #     plt.figure(1)
    #     plt.subplot(211)
    #     plt.title(algorithm_options['pca_algorithm'])
    #     for err_name in errs:
    #         print(err_name +': %f' %(errs[err_name][-1]))
    #     for err_name in errs:
    #         if err_name in ['batch_err', 'population_err', 'pop_whitening_err', 'batch_whitening_err']:
    #             handle, = plt.plot(np.log10(errs[err_name]), label=err_name)
    #             handles.append(handle)
    #     plt.legend(handles=handles)
    #     plt.ylabel('Error (log10 scale)')
    #
    #     handles = []
    #     plt.subplot(212)
    #     for err_name in errs:
    #         if err_name in ['strain_err', 'recon_err']:
    #             handle, = plt.plot(errs[err_name], label=err_name)
    #             handles.append(handle)
    #     plt.legend(handles=handles)
    #     plt.ylabel('Error (linear scale)')
    #     plt.xlabel('Iteration')
    #     plt.show()
