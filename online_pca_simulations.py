# Title: online_pca_simulations.py
# Description: A function for testing an online PCA algorithm
# Author: Victor Minden (vminden@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovanni
# Reference: None

##############################
# Imports
import numpy as np
from scipy.linalg import solve as solve
import util
import time

from minimax_subspace_projection import minimax_PCA
from if_minimax_subspace_projection import if_minimax_PCA
from ccipca import CCIPCA
from incremental_pca import incremental_PCA
from stochastic_gradient_pca import SGA_PCA
from online_similarity_matching import OSM_PCA

from collections import defaultdict
from matplotlib import pyplot as plt
##############################




class Timer:
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
    # - The relative reconstruction error (of some test data) ||X - UU'X||_F / ||X||_F
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


    # We wrap things in a default dict so we don't have to check if keys exist
    error_options = defaultdict(int, simulation_options['error_options'])
    compute_error = any(error_options)

    if compute_error:
        # We will make a list of functions that take in the current iterate and return an error measure
        error_options['error_func_list'] = []


    generator_options = generator_options.copy()

    if error_options['compute_population_error']:
        generator_options['return_U'] = True
    else:
        generator_options['return_U'] = False


    data    = util.generate_samples(d, q, n + n_test, generator_options)
    X       = data['X'][:,:n]
    Xtest   = data['X'][:,n:]
    XXtest  = Xtest.T.dot(Xtest)
    Xtest  /= np.linalg.norm(Xtest, 'fro')
    XXtest /= np.linalg.norm(XXtest, 'fro')

    if error_options['compute_population_error']:
        U_pop = data['U']
        error_options['error_func_list'].append(('population_err', lambda Uhat: util.subspace_error(Uhat, U_pop)))

    if error_options['compute_batch_error']:
        eig_val,V = np.linalg.eigh(X.dot(X.T) / n)
        idx       = np.flip(np.argsort(eig_val),0)
        eig_val   = eig_val[idx]
        V         = V[:,idx]
        U_batch   = V[:,:q]
        error_options['error_func_list'].append(('batch_err', lambda Uhat: util.subspace_error(Uhat, U_batch)))

    if error_options['compute_strain_error']:
        error_options['error_func_list'].append(('strain_err', lambda Uhat: util.strain_error(Uhat.T.dot(Xtest), XXtest)))

    if error_options['compute_reconstruction_error']:
        error_options['error_func_list'].append(('recon_err', lambda Uhat: util.reconstruction_error(Uhat, Xtest)))


# Main chunk of the code follows: run the appropriate algorithm

    pca_algorithm = algorithm_options['pca_algorithm']


    if pca_algorithm == 'CCIPCA':
        if compute_error:
            errs = CCIPCA(X[:,n0:], q, n_epoch, error_options=error_options)
        else:
            with Timer() as t:
                *_, = CCIPCA(X[:,n0:], q, n_epoch)
            print('CCIPCA took %f sec.' % t.interval)

    elif pca_algorithm == 'SGA_PCA':
        if compute_error:
            errs = SGA_PCA(X[:,n0:], q, n_epoch, error_options)
        else:
            with Timer() as t:
                *_, = SGA_PCA(X[:,n0:], q, n_epoch)
            print('SGA_PCA took %f sec.' % t.interval)

    elif pca_algorithm == 'incremental_PCA':
        tol = algorithm_options['tol']
        if compute_error:
            errs = incremental_PCA(X[:,n0:], q, n_epoch, tol, error_options=error_options)
        else:
            with Timer() as t:
                *_, = incremental_PCA(X[:,n0:], q, n_epoch, tol)
            print('incremental_PCA took %f sec.' % t.interval)

    elif pca_algorithm == 'minimax_PCA':
        tau = algorithm_options['tau']
        if compute_error:
            errs = minimax_PCA(X[:,n0:], q, tau, n_epoch, error_options)
        else:
            with Timer() as t:
                *_, = minimax_PCA(X[:,n0:], q, tau, n_epoch)
            print('minimax_PCA took %f sec.' % t.interval)

    elif pca_algorithm == 'if_minimax_PCA':
        tau = algorithm_options['tau']
        if compute_error:
            errs = if_minimax_PCA(X[:,n0:], q, tau, n_epoch, error_options)
        else:
            with Timer() as t:
                *_, = if_minimax_PCA(X[:,n0:], q, tau, n_epoch)
            print('if_minimax_PCA took %f sec.' % t.interval)


    elif pca_algorithm == 'OSM_PCA':
        lambda_ = 0
        if compute_error:
            errs = OSM_PCA(X[:,n0:], q, lambda_, n_epoch, error_options)
        else:
            with Timer() as t:
                *_, = OSM_PCA(X[:,n0:], q, lambda_, n_epoch)
            print('OSM_PCA took %f sec.' % t.interval)

    #TODO: REMOVE ME
    if compute_error:
        return errs


if __name__ == "__main__":

    algo_names = ['SGA_PCA', 'incremental_PCA', 'minimax_PCA', 'if_minimax_PCA', 'OSM_PCA']
    output_folder = "/dev/null"

    error_options = {
        #'subspace_error' : True,
        'n_skip' : 128,
        'orthogonalize_iterate' : False,
        'compute_batch_error' : True,
        'compute_population_error' : True,
        'compute_strain_error' : True,
        'compute_reconstruction_error' : True
    }

    simulation_options = {
        'd' : 100,
        'q' : 4,
        'n' : 2000,
        'n0': 0,
        'n_epoch': 10,
        'n_test' : 256,
        'error_options' : error_options
    }

    generator_options = {
        'method'   : 'spiked_covariance',
        'lambda_q' : 1,
        'normalize': True,
        'rho'      : 0.1
    }

    algorithm_options = {
        'pca_algorithm' : algo_names[-1],
        'tau'           : 0.5,
        'tol'           : 1e-7
    }

    errs = run_simulation(output_folder, simulation_options, generator_options, algorithm_options)
    handles = []
    for err_name in errs:
        handle, = plt.plot(np.log10(errs[err_name]), label=err_name)
        handles.append(handle)

    plt.legend(handles=handles)
    plt.title(algorithm_options['pca_algorithm'])
    plt.show()
