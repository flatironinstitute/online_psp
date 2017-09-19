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

from minimax_subspace_projection import minimax_PCA as minimax_PCA
from if_minimax_subspace_projection import if_minimax_PCA as if_minimax_PCA
from ccipca import CCIPCA as CCIPCA
from incremental_pca import incremental_PCA as incremental_PCA
from stochastic_gradient_pca import SGA_PCA as SGA_PCA

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

    # Unpack some parameters
    d       = simulation_options['d']
    q       = simulation_options['q']
    n       = simulation_options['n']
    n0      = simulation_options['n0']
    n_epoch = simulation_options['n_epoch']

    # What if the error options for the simulation are dictionary inside the simulation_options dictionary?
    error_options = simulation_options['error_options']
    #time_it       = simulation_options['time_it']
    compute_error = any(error_options)

    # TODO: What do we do if we don't have a population matrix?
    X,U,lambda_ = util.generate_samples(d, q, n, generator_options)

    if error_options['compute_population_error']:
        error_options['population_PCA_vectors'] = U[:,:q]

    if error_options['compute_batch_error']:
        eig_val,V = np.linalg.eigh(X.dot(X.T) / n)
        idx       = np.flip(np.argsort(eig_val),0)
        eig_val   = eig_val[idx]
        V          = V[:,idx]

        error_options['batch_PCA_vectors'] = V[:,:q]

    # TODO: REMOVE ME
    #U = V[:,:q]

    pca_algorithm = algorithm_options['pca_algorithm']


    if pca_algorithm == 'CCIPCA':
        if compute_error:
            _,errs = CCIPCA(X[:,n0:], q, n_epoch, U)
        else:
            with Timer() as t:
                *_, = CCIPCA(X[:,n0:], q, n_epoch)
            print('CCIPCA took %f sec.' % t.interval)

    elif pca_algorithm == 'SGA_PCA':
        if compute_error:
            pass
        else:
            *_, = SGA_PCA(X[:,n0:], q, n_epoch)

    elif pca_algorithm == 'incremental_PCA':
        if compute_error:
            pass
        else:
            *_, = incremental_PCA(X[:,n0:], q, n_epoch)

    elif pca_algorithm == 'minimax_PCA':
        tau = algorithm_options['tau']
        if compute_error:
            _,_,errs = minimax_PCA(X[:,n0:], q, tau, n_epoch, U)
        else:
            *_, = minimax_PCA(X[:,n0:], q, tau, n_epoch)

    elif pca_algorithm == 'if_minimax_PCA':
        tau = algorithm_options['tau']
        if compute_error:
            _,_,errs = if_minimax_PCA(X[:,n0:], q, tau, n_epoch, U)
        else:
            with Timer() as t:
                *_, = if_minimax_PCA(X[:,n0:], q, tau, n_epoch)
            print('if_minimax_PCA took %f sec.' % t.interval)


    elif pca_algorithm == 'OSM_PCA':
        if compute_error:
            pass
        else:
            lambda_ = 0
            *_, = OSM_PCA(X[:,n0:], q, lambda_, n_epoch)

    #TODO: REMOVE ME
    if compute_error:
        return errs


if __name__ == "__main__":
    output_folder = "/dev/null"

    error_options = defaultdict(int, {
        #'subspace_error' : True,
        #'n_skip' : 128,
        #'ortho' : False,
        #'compute_batch_error' : True,
        #'compute_population_error' : True
    })

    simulation_options = defaultdict(int, {
        'd' : 256,
        'q' : 4,
        'n' : 4096,
        'n0': 0,
        'n_epoch': 10,
        'error_options' : error_options
    })

    generator_options = defaultdict(int, {
        'method'   : 'spiked_covariance',
        'lambda_q' : 1,
        'normalize': True,
        'rho'      : 0.1
    })

    algorithm_options = defaultdict(int, {
        'pca_algorithm' : 'if_minimax_PCA',
        'tau'           : 0.5
    })

    errs = run_simulation(output_folder, simulation_options, generator_options, algorithm_options)

    plt.plot(np.log10(errs))
    plt.show()
