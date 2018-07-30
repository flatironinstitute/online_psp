# Title: ex_spiked_covariance_errors.py
# Description: Testing online PCA algorithm population and batch error on artificially generated data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None

# %%
# imports
try:
    if __IPYTHON__:
        print("Running under iPython")
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
        get_ipython().magic('matplotlib osx')
except NameError:
    pass
from online_pca_simulations import run_simulation
import os
import pylab as plt
import numpy as np
from glob import glob
import psutil
import multiprocessing
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# TODO: change the places where number of algorithms is 
# implicitly hard-coded to be 3

# general parameters

error_options = {
    'n_skip': 50,
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
    'scale_data': True,
    'shuffle': False
}

simulation_options = {
    'n0': 0,
    'n_epoch': 1,
    'error_options': error_options,
    'pca_init': False,
    'init_ortho': True,
}

algos = ['if_minimax_PCA', 'incremental_PCA', 'CCIPCA', 'minimax_PCA']
# algo = algos[0]
algorithm_options = {
    # 'pca_algorithm': algo,
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
        'population_err': errs_pop,
        'batch_err': errs_batch,
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
rhos = np.logspace(-4, -0.3, 10)  # controls SNR
rerun_simulation = True  # whether to rerun from scratch or just show the results
parallelize = np.logical_and(rerun_simulation, True)  # whether to use parallelization or to show results on the go
# %% start cluster
if parallelize:
    n_processes = np.maximum(np.int(psutil.cpu_count()/2), 1)
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
all_pars = []
for t_ in [0.6, 0.7, 0.8, 1, 1.5, 2]:
    algorithm_options['t'] = t_
    # %% vary k
    d_q_params = [(256, 16), (256, 128), (2048, 16), (2048, 128)]
    data_fold = os.path.abspath('./spiked_cov_vary_k_d_t_' + str(t_))
    if t_ == 0.5:
        algos = ['if_minimax_PCA', 'incremental_PCA', 'CCIPCA']
    else:
        algos = ['if_minimax_PCA']

    n_repetitions = 10
    simulation_options['n'] = 6000


    if rerun_simulation:
        os.mkdir(data_fold)
    else:
        plt.figure()


    counter = 0
    for d, q in d_q_params:

        simulation_options['d'] = d
        simulation_options['q'] = q
        for algo in range(len(algos)):
            algorithm_options['pca_algorithm'] = algos[algo]
            pop_err_avg = []
            batch_err_avg = []
            for rho in rhos:
                generator_options['rho'] = rho
                print((d, q, rho))
                all_pars.append(
                    [generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold[:],
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
                            pop_err_avg.append(np.mean(ld['population_err'][()], 0)[-1])
                            batch_err_avg.append(np.mean(ld['batch_err'][()], 0)[-1])

            if pop_err_avg is not None:
                ax = plt.subplot(len(d_q_params), 2, 2 * counter + 1)
                if algo ==0:
                    line_pop, = ax.loglog(rhos, pop_err_avg, 'o-')
                else:
                    line_pop, = ax.loglog(rhos, pop_err_avg,'.-')

                line_pop.set_label('algo=' + algos[algo])

                ax = plt.subplot(len(d_q_params), 2, 2 * counter + 2)
                if algo == 0:
                    line_bat, = ax.loglog(rhos, batch_err_avg, 'o-')
                else:
                    line_bat, = ax.loglog(rhos, batch_err_avg,'.-')
                line_bat.set_label('algo=' + algos[algo])

                plt.pause(.1)

        counter += 1


    if pop_err_avg is not None:
        for counter in range(len(d_q_params)):
            ax = plt.subplot(len(d_q_params), 2, 2 * counter + 1)
            plt.ylabel('population error')
            plt.xlabel('rho')
            plt.title('q=' + str(d_q_params[counter][-1]))

            ax = plt.subplot(len(d_q_params), 2, 2 * counter + 2)
            plt.ylabel('batch error')
            plt.xlabel('rho')
            plt.title('q=' + str(d_q_params[counter][-1]))

            plt.pause(2)


        ax = plt.subplot(len(d_q_params), 2, 2 * counter + 2)
        ax.legend()

if parallelize:
    all_errs = dview.map(run_test_wrapper, all_pars)

# %% stop cluster
if parallelize:
    dview.terminate()
