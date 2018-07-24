# Title: error_performance_spiked_covariance.py
# Description: Testing online PCA algorithm population and batch error on artificially generated data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None
# %%
# %%
# imports
from online_pca_simulations import run_simulation
import os
import pylab as plt
import numpy as np
from glob import glob
import psutil
import multiprocessing

# general parameters

error_options = {
    'n_skip': 20,
    'orthogonalize_iterate': False,
    'compute_batch_error': True,
    'compute_population_error': True,
    'compute_strain_error': False,
    'compute_reconstruction_error': False
}

generator_options = {
    'method': 'spiked_covariance',
    'lambda_q': 5e-1,
    'normalize': True,
    'rho': 1e-2 / 5,
    'scale_data': True
}

simulation_options = {
    'd': 200,
    'q': 50,
    'n': 1000,
    'n0': 0,
    'n_epoch': 1,
    'error_options': error_options,
    'pca_init': False,
    'init_ortho': True,
}

algos = ['if_minimax_PCA', 'incremental_PCA', 'CCIPCA']
algo = algos[0]
algorithm_options = {
    'pca_algorithm': algo,
    'tau': 0.5,
    'tol': 1e-7
}


# %%
def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    '''function running each iteration of a test
    '''
    output_folder = os.getcwd() + '/test'

    errs = run_simulation(output_folder, simulation_options,
                          generator_options, algorithm_options)

    return errs


# %%
def run_test_wrapper(params):
    ''' Function to parallelize on multiple repetitions os the same simulation

    Parameters
    ----------
    params

    Returns
    -------

    '''
    generator_options, simulation_options, algorithm_options, data_fold, n_repetitions = params
    errs_pop = []
    errs_batch = []
    for _ in range(n_repetitions):
        err = run_test(generator_options=generator_options, simulation_options=simulation_options,
                       algorithm_options=algorithm_options)
        errs_pop.append(err['population_err'])
        errs_batch.append(err['batch_err'])

    errs_pop = np.array(errs_pop)
    errs_batch = np.array(errs_batch)

    output_dict = {
        'generator_options': generator_options,
        'simulation_options': simulation_options,
        'algorithm_options': algorithm_options,
        'd': simulation_options['d'],
        'q': simulation_options['q'],
        'n': simulation_options['n'],
        'rho': generator_options['rho'],
        'n_epoch': 1,
        'n0': 0,
        'population_err': errs_batch,
        'batch_err': errs_pop,
    }
    rho = generator_options['rho']
    d = simulation_options['d']
    q = simulation_options['q']
    algo = algorithm_options['pca_algorithm']
    save_name = os.path.join(data_fold,
                             '__'.join(['rho', "{:.6f}".format(rho), 'd', str(d), 'q', str(q), 'algo', algo]) + '.npz')
    print('Saving in:' + save_name)
    np.savez(
        save_name,
        **output_dict)

    return errs_pop, errs_batch


# %% parameters figure generation
test_mode = 'vary_k'  # can be 'illustrative_examples' or 'vary_k'
rhos = np.logspace(-4, -0.5, 10)  # controls SNR
rerun_simulation = False  # whether to rerun from scratch or just show the results
parallelize = np.logical_and(rerun_simulation, True)  # whether to use parallelization or to show results on the go
# %% start cluster
if parallelize:
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
    if len(multiprocessing.active_children()) > 0:
        try:
            dview.terminate()
        except:
            raise Exception('A cluster is already runnning. Terminate with dview.terminate() if you want to restart.')
    else:
        try:
            if 'kernel' in get_ipython().trait_names():  # If you're on OSX and you're running under Jupyter or Spyder,
                # which already run the code in a forkserver-friendly way, this
                # can eliminate some setup and make this a reasonable approach.
                # Otherwise, seting VECLIB_MAXIMUM_THREADS=1 or using a different
                # blas/lapack is the way to avoid the issues.
                # See https://github.com/flatironinstitute/CaImAn/issues/206 for more
                # info on why we're doing this (for now).
                multiprocessing.set_start_method('forkserver', force=True)
        except:  # If we're not running under ipython, don't do anything.
            pass

        dview = multiprocessing.Pool(n_processes)
# %%
if test_mode == 'illustrative_examples':
    # %%
    data_fold = os.path.abspath('./spiked_cov_4_examples')
    d_q_params = [(16, 2), (64, 8), (256, 32), (1024, 64)]
    colors = ['b', 'r', 'g']
    n_repetitions = 15
    simulation_options['n'] = 3000
    plot = not parallelize
    if rerun_simulation:
        os.mkdir(data_fold)
    counter = 0
    all_pars = []
    for d, q in d_q_params:
        simulation_options['d'] = d
        simulation_options['q'] = q
        counter += 1
        if not parallelize:
            ax = plt.subplot(1, 4, counter)
        for algo in range(3):
            algorithm_options['pca_algorithm'] = algos[algo]
            pop_err_avg = []
            batch_err_avg = []
            for rho in rhos:
                print((d, q, rho))
                generator_options['rho'] = rho
                all_pars.append(
                    [generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold,
                     n_repetitions])

                if parallelize:
                    pop_err_avg = None
                    batch_err_avg = None
                else:
                    if rerun_simulation:
                        errs_pop, errs_batch = run_test_wrapper(all_pars[-1])
                        pop_err_avg.append(errs_pop.mean(0)[-1])
                        batch_err_avg.append(errs_batch.mean(0)[-1])
                        errs_pop = np.array(errs_pop)
                        errs_batch = np.array(errs_batch)
                    else:
                        fname = os.path.join(data_fold, '__'.join(
                            ['rho', "{:.6f}".format(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]]) + '.npz')
                        # fname = os.path.join(data_fold, '__'.join(
                        #     ['rho',str(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]]) + '.npz')
                        with np.load(fname) as ld:
                            pop_err_avg.append(np.mean(ld['batch_err'][()], 0)[-1])
                            batch_err_avg.append(np.mean(ld['population_err'][()], 0)[-1])

            if pop_err_avg is not None:
                line_pop, = ax.loglog(rhos, pop_err_avg, '-d' + colors[algo])
                line_bat, = ax.loglog(rhos, batch_err_avg, '-o' + colors[algo])
                line_pop.set_label(algos[algo] + '_pop')
                line_bat.set_label(algos[algo] + '_batch')
                plt.xlabel('rho')
                plt.xlabel('projection error')
                plt.pause(.1)

    if pop_err_avg is not None:
        ax.legend()
        plt.pause(3)

    if parallelize:
        all_res = dview.map(run_test_wrapper, all_pars)

# %%
elif test_mode == 'vary_k':
    # %% vary k
    data_fold = os.path.abspath('./spiked_cov_vary_k')
    n_repetitions = 15
    simulation_options['n'] = 3000
    d_q_params = [(1024, 2), (1024, 8), (1024, 32), (1024, 64), (1024, 128)]

    if rerun_simulation:
        os.mkdir(data_fold)

    all_pars = []
    counter = 0
    for d, q in d_q_params:
        counter += 1
        simulation_options['d'] = d
        simulation_options['q'] = q
        for algo in range(3):
            algorithm_options['pca_algorithm'] = algos[algo]
            pop_err_avg = []
            batch_err_avg = []
            for rho in rhos:
                generator_options['rho'] = rho
                print((d, q, rho))
                all_pars.append(
                    [generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold,
                     n_repetitions])
                if parallelize:
                    pop_err_avg = None
                    batch_err_avg = None
                else:
                    if rerun_simulation:
                        errs_pop, errs_batch = run_test_wrapper(all_pars[-1])
                        pop_err_avg.append(errs_pop.mean(0)[-1])
                        batch_err_avg.append(errs_batch.mean(0)[-1])
                        errs_pop = np.array(errs_pop)
                        errs_batch = np.array(errs_batch)
                    else:
                        fname = os.path.join(data_fold, '__'.join(
                            ['rho', "{:.6f}".format(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]]) + '.npz')

                        with np.load(fname) as ld:
                            print(pop_err_avg)
                            pop_err_avg.append(np.mean(ld['batch_err'][()], 0)[-1])
                            batch_err_avg.append(np.mean(ld['population_err'][()], 0)[-1])

            if pop_err_avg is not None:
                ax = plt.subplot(3, 2, 2 * algo + 1)
                line_pop, = ax.loglog(rhos, pop_err_avg)
                line_pop.set_label('q=' + str(q))
                ax.legend()
                ax = plt.subplot(3, 2, 2 * algo + 2)
                line_bat, = ax.loglog(rhos, batch_err_avg)
                line_bat.set_label('q=' + str(q))
                ax.legend()
                plt.pause(.1)

    if pop_err_avg is not None:
        for algo in range(3):
            ax = plt.subplot(3, 2, 2 * algo + 1)
            plt.ylabel('population error')
            plt.xlabel('rho')
            plt.title(algos[algo])
            ax.legend()
            ax = plt.subplot(3, 2, 2 * algo + 2)
            plt.ylabel('batch error')
            plt.xlabel('rho')
            plt.title(algos[algo])
            ax.legend()
            plt.pause(2)

    if parallelize:
        all_errs = dview.map(run_test_wrapper, all_pars)

# %% stop cluster
if parallelize:
    dview.terminate()
