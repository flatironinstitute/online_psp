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

from if_minimax_subspace_projection import IF_minimax_PCA_CLASS
from ccipca import CCIPCA_CLASS
from incremental_pca import IncrementalPCA_CLASS

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
    # for problems with real data (small number of samples)
    if n is not None:
        ntot = n + n_test
    else:
        ntot = None

    if error_options['compute_population_error']:
        generator_options['return_U'] = True

        X, U_pop, sigma2 = util.generate_samples(q, ntot, d, method=generator_options['method'],
                                                scale_data=generator_options['scale_data'],
                                                scale_with_log_q=generator_options['scale_with_log_q'],
                                                options=generator_options)
    else:
        generator_options['return_U'] = False
        X = util.generate_samples(q, ntot, d, method=generator_options['method'],
                                                scale_data=generator_options['scale_data'],
                                                scale_with_log_q=generator_options['scale_with_log_q'],
                                                options=generator_options)

    # here making sure that we use the right n when including n_test frames
    if d is None:
        d, _ = X.shape
        if ntot is None:
            n = X.shape[-1] - n_test
            print('** Warning: using only {0} samples for computing the PCA, You can set the n value to more than sample size '
                'but you will be resampling the same frames multiple times'.format(str(n)))
        else:
            print('** Warning: You are resampling the same frames multiple times')


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
        # TODO: Maybe lambda0 should be sorted in descending order?
        lambda0 = np.abs(np.random.normal(0, 1, (q,)) / np.sqrt(q))
        pca_fitter = CCIPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda0, in_place=True)
    elif pca_algorithm == 'incremental_PCA':
        tol = algorithm_options['tol']
        lambda0 = np.abs(np.random.normal(0, 1, (q,)) / np.sqrt(q))
        pca_fitter = IncrementalPCA_CLASS(q, d, Uhat0=Uhat0, lambda0=lambda0)
    elif pca_algorithm == 'if_minimax_PCA':
        tau = algorithm_options['tau']
        pca_fitter = IF_minimax_PCA_CLASS(q, d, W0=Uhat0.T, Minv0=None, tau=tau)
    # elif pca_algorithm == 'if_minimax_PCA':
    #     tau = algorithm_options['tau']
    #     if compute_error:
    #         errs = if_minimax_PCA(X[:,n0:], q, tau, n_epoch, error_options, W0=Uhat0.T)
    #     else:
    #         with Timer() as t:
    #             *_, = if_minimax_PCA(X[:,n0:], q, tau, n_epoch, W0=Uhat0.T)
    #         print('if_minimax_PCA took %f sec.' % t.interval)

    else:
        assert 0, 'You did not specify a valid algorithm.  Please choose one of:\n \tCCIPCA, incremental_PCA, if_minimax_PCA'






    if compute_error:
        # Compute errors, do not time algorithms
        n_its =  X[:,n0:].shape[1] * n_epoch
        errs = util.initialize_errors(error_options, n_its)
        i = 0
        for n_e in range(n_epoch):
            for x in X[:,n0:].T:
                pca_fitter.fit_next(x)
                Uhat = pca_fitter.get_components()
                util.compute_errors(error_options, Uhat, i, errs)
                i+=1
                # TODO: implement skip
                # if i == n_skip:
                    # # Compute errors
                    # i = 0
                    # # TODO: implement this in each class
                    # Uhat = pca_fitter.get_components()
                    # util.compute_errors(error_options, Uhat, t, errs)
    else:
        # Do timing, do not compute errors
        # TODO: save the timing information, don't just print it out
        with Timer() as t:
            for n_e in range(n_epoch):
                for x in X[:,n0:].T:
                    pca_fitter.fit_next(x)
        print('%s took %f sec.' % (pca_algorithm, t.interval))




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

    if output_dict['simulation_options']['d'] is None:
        output_dict['simulation_options']['d'] = 'None'
    if output_dict['simulation_options']['n'] is None:
        output_dict['simulation_options']['n'] = 'None'

    sio.savemat(filename, output_dict)
    print('Output written to %s' % filename)

    if compute_error:
        return errs
    else:
        return t.interval


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    output_folder = os.getcwd() + '/test'

    errs = run_simulation(output_folder, simulation_options,
                          generator_options, algorithm_options)

    handles = []

    fig = plt.figure(1)
    ax  = fig.add_subplot(1,1,1)
    plt.title(algorithm_options['pca_algorithm'])
    for err_name in errs:
        print(err_name + ': %f' % (errs[err_name][-1]))
        handle, = ax.plot(errs[err_name], label=err_name)
        handles.append(handle)
    plt.legend(handles=handles)
    plt.ylabel('Error (log10 scale)')
    plt.xlabel('Iteration')
    #plt.ylim(ymax=1, ymin=0)
    ax.set_yscale('log')
    plt.show()




if __name__ == "__main__":



    error_options = {
        'n_skip': 1,
        'orthogonalize_iterate': False,
        'compute_batch_error': True,
        'compute_population_error': True,
        'compute_strain_error': True,
        'compute_reconstruction_error': True
    }
    spiked_covariance = False
    scale_data = True
    scale_with_log_q = True
    if spiked_covariance:
        generator_options = {
            'method': 'spiked_covariance',
            'lambda_q': 5e-1,
            'normalize': True,
            'rho': 1e-2 / 5,
            'scale_data': scale_data,
            'scale_with_log_q': scale_with_log_q
        }

        simulation_options = {
            'd': 200,
            'q': 20,
            'n': 5000,
            'n0': 0,
            'n_epoch': 1,
            'n_test': 256,
            'error_options': error_options,
            'pca_init': False,
            'init_ortho': True,
        }
    else:
        dsets = ['ATT_faces_112_92.mat', 'ORL_32x32.mat', 'YaleB_32x32.mat']
        dset = dsets[1]
        print('** ' + dset)
        generator_options = {
            'method': 'real_data',
            'filename': './datasets/' + dset,
            'lambda_q': 5e-1,
            'scale_data': scale_data,
            'scale_with_log_q': scale_with_log_q

        }
        simulation_options = {
            'd': None,
            'q': 100,
            'n': None, # can set a number here, will select frames multiple times
            'n0': 0,
            'n_epoch': 1,
            'n_test': 128,
            'error_options': error_options,
            'pca_init': False,
            'init_ortho': True,
        }

    algorithm_options = {
        'pca_algorithm': 'if_minimax_PCA',
        'tau': 0.5,
        'tol': 1e-7
    }
    run_test(generator_options=generator_options, simulation_options=simulation_options, algorithm_options=algorithm_options)