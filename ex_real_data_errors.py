# Title: error_performance_spiked_covariance.py
# Description: Testing online PCA algorithm population and batch error on artificially generated data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None
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
    'n_skip': 50,
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
        'q': 16,
        'n': 'auto',  # can set a number here, will select frames multiple times
        'n0': 0,
        'error_options': error_options,
        'pca_init': False,
        'init_ortho': True,
}


# algos = ['if_minimax_PCA', 'CCIPCA']

# algo = algos[0]
algorithm_options = {
    # 'pca_algorithm': algo,
    'tau': 0.5,
    'tol': 1e-7,
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
    errs_recon = []
    for _ in range(n_repetitions):
        err = run_test(generator_options=generator_options, simulation_options=simulation_options,
                       algorithm_options=algorithm_options)
        errs_pop.append(err['population_err'])
        errs_batch.append(err['batch_err'])
        errs_recon.append(err['recon_err'])

    errs_pop = np.array(errs_pop)
    errs_batch = np.array(errs_batch)
    errs_recon = np.array(errs_recon)

    output_dict = {
        'generator_options': generator_options,
        'simulation_options': simulation_options,
        'algorithm_options': algorithm_options,
        'd': simulation_options['d'],
        'q': simulation_options['q'],
        'n': simulation_options['n'],
        'filename': generator_options['filename'],
        'n_epoch': simulation_options['n_epoch'],
        'n0': 0,
        # TODO these were swapped
        'population_err': errs_pop,
        'batch_err': errs_batch,
        'recon_err': errs_recon
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

    return errs_pop, errs_batch, errs_recon


# %% parameters figure generation
test_mode = 'real_data_learning_curves'
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
if test_mode == 'real_data_learning_curves':
    # %%
    data_fold = os.path.abspath('./real_data_learning_curves')
    #redundant but there for flexibility
    names = ['ORL_32x32.mat','YaleB_32x32.mat','ATT_faces_112_92.mat', 'MNIST.mat'][:]
    n_epochs = [30, 10, 30, 1][:]
    qs = [16, 64, 128, 256][:3]
    algos = ['if_minimax_PCA', 'incremental_PCA', 'CCIPCA'][:]

    colors = ['b', 'r', 'g']
    n_repetitions = 10

    simulation_options['n'] = 'auto'
    plot = not parallelize
    if rerun_simulation:
        os.makedirs(data_fold, exist_ok=True)
    counter_q = 0
    all_pars = []
    for q in qs:

        counter_name = 0
        for n_ep, name in zip(n_epochs, names):
            simulation_options['n_epoch'] = n_ep
            counter_name += 1
            generator_options['filename'] = './datasets/' + name
            if not parallelize:
                ax = plt.subplot(len(qs), len(names), len(names)*counter_q + counter_name)

            for algo, algor in enumerate(algos):
                print(name)

                algorithm_options['pca_algorithm'] = algor
                print((name, q))
                simulation_options['q'] = q
                all_pars.append(
                    [generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold,
                     n_repetitions])

                if parallelize:
                    batch_err_avg = None
                else:
                    if rerun_simulation:
                        errs_pop, errs_batch, errs_recon = run_test_wrapper(all_pars[-1])
                        batch_err_avg = np.median(errs_batch, 0)
                    else:
                        fname = os.path.join(data_fold, '__'.join(
                            ['fname',name, 'q', str(q), 'algo', algor]) + '.npz')
                        # fname = os.path.join(data_fold, '__'.join(
                        #     ['rho',str(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]]) + '.npz')
                        with np.load(fname) as ld:
                            batch_err_avg = np.median(ld['batch_err'][()], 0)

                if batch_err_avg is not None:
                    plt.title('k=' + str(q))
                    line_bat, = ax.loglog(batch_err_avg.T, colors[algo])
                    line_bat.set_label(algos[algo] + '_batch')
                    plt.ylabel('subspace error')
                    plt.pause(.1)


            plt.show()
        counter_q += 1

    if batch_err_avg is not None:
        ax.legend()
        plt.xlabel('sample')
        plt.show()
        plt.pause(3)

    if parallelize:
        all_res = dview.map(run_test_wrapper, all_pars)

    # %% stop cluster
if parallelize:
    dview.terminate()
