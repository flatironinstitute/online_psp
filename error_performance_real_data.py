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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
    'method': 'real_data',
    'scale_data': True,
    'shuffle' : True
}

simulation_options = {
        'd': None,
        'q': 50,
        'n': 'auto',  # can set a number here, will select frames multiple times
        'n0': 0,
        'error_options': error_options,
        'pca_init': False,
        'init_ortho': True,
        'n_epoch' : 10
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
        'filename': generator_options['filename'],
        'n_epoch': 1,
        'n0': 0,
        'population_err': errs_batch,
        'batch_err': errs_pop,
    }
    d = simulation_options['d']
    q = simulation_options['q']
    filename = generator_options['filename']
    algo = algorithm_options['pca_algorithm']
    save_name = os.path.join(data_fold,
                             '__'.join(['fname', filename.split('/')[-1], 'q', str(q), 'algo', algo]) + '.npz')
    print('Saving in:' + save_name)
    np.savez(
        save_name,
        **output_dict)

    return errs_pop, errs_batch


# %% parameters figure generation
test_mode = 'illustrative_examples'  # can be 'illustrative_examples' or 'vary_q', 'vary_q_fix_qdata'
rhos = np.logspace(-4, -0.5, 10)  # controls SNR
rerun_simulation = True  # whether to rerun from scratch or just show the results
parallelize = np.logical_and(rerun_simulation, False)  # whether to use parallelization or to show results on the go
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
    data_fold = os.path.abspath('./real_data_4_examples')
    #redundant but there for flexibility
    names = ['ORL_32x32.mat','YaleB_32x32.mat','ATT_faces_112_92.mat', 'MNIST.mat'][:-3]
    qs = [8, 16, 32, 64]
    colors = ['b', 'r', 'g']
    n_repetitions = 1
    simulation_options['n'] = 'auto'
    plot = not parallelize
    if rerun_simulation:
        os.mkdir(data_fold)
    counter = 0
    all_pars = []
    for name in names:
        generator_options['filename'] = './datasets/' + name

        counter += 1
        if not parallelize:
            ax = plt.subplot(4, 1, counter)
        for algo in range(3):
            algorithm_options['pca_algorithm'] = algos[algo]
            pop_err_avg = []
            batch_err_avg = []
            for q in qs:
                print((name, q))
                simulation_options['q'] = q
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
                            ['fname',name, 'q', str(q), 'algo', algos[algo]]) + '.npz')
                        # fname = os.path.join(data_fold, '__'.join(
                        #     ['rho',str(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]]) + '.npz')
                        with np.load(fname) as ld:
                            pop_err_avg.append(np.mean(ld['batch_err'][()], 0)[-1])
                            batch_err_avg.append(np.mean(ld['population_err'][()], 0)[-1])

            if pop_err_avg is not None:
                plt.title('k=' + str(q))
                line_pop, = ax.loglog(qs, pop_err_avg, '.-' + colors[algo])
                line_bat, = ax.loglog(qs, batch_err_avg, '-.' + colors[algo])
                line_pop.set_label(algos[algo] + '_pop')
                line_bat.set_label(algos[algo] + '_batch')
                plt.ylabel('projection error')
                plt.pause(.1)

    if pop_err_avg is not None:
        ax.legend()
        plt.xlabel('k')
        plt.pause(3)

    if parallelize:
        all_res = dview.map(run_test_wrapper, all_pars)

# %%
elif test_mode == 'vary_q' or test_mode == 'vary_q_fix_qdata':
    # %% vary k
    if test_mode == 'vary_q':
        data_fold = os.path.abspath('./spiked_cov_vary_k')
    if test_mode == 'vary_q_fix_qdata':
        data_fold = os.path.abspath('./spiked_cov_vary_k_fix_qdata')
        generator_options['q_data'] = 128

    n_repetitions = 15
    simulation_options['n'] = 3000
    d_q_params = [(1024, 2), (1024, 8), (1024, 32), (1024, 64), (1024, 128)]

    if rerun_simulation:
        os.mkdir(data_fold)

    all_pars = []
    counter = 0
    for d, q in d_q_params:

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
                ax = plt.subplot(5, 2, 2 * counter + 1)
                if algo ==0:
                    line_pop, = ax.loglog(rhos, pop_err_avg, 'o-')
                else:
                    line_pop, = ax.loglog(rhos, pop_err_avg,'.-')

                line_pop.set_label('algo=' + algos[algo])

                ax = plt.subplot(5, 2, 2 * counter + 2)
                if algo == 0:
                    line_bat, = ax.loglog(rhos, batch_err_avg, 'o-')
                else:
                    line_bat, = ax.loglog(rhos, batch_err_avg,'.-')
                line_bat.set_label('algo=' + algos[algo])

                plt.pause(.1)

        counter += 1


    if pop_err_avg is not None:
        for counter in range(5):
            ax = plt.subplot(5, 2, 2 * counter + 1)
            plt.ylabel('population error')
            plt.xlabel('rho')
            plt.title('q=' + str(d_q_params[counter][-1]))

            ax = plt.subplot(5, 2, 2 * counter + 2)
            plt.ylabel('batch error')
            plt.xlabel('rho')
            plt.title('q=' + str(d_q_params[counter][-1]))

            plt.pause(2)


        ax = plt.subplot(5, 2, 2 * counter + 2)
        ax.legend()

    if parallelize:
        all_errs = dview.map(run_test_wrapper, all_pars)

# %% stop cluster
if parallelize:
    dview.terminate()
