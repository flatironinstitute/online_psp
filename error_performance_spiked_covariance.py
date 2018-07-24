# Title: error_performance_spiked_covariance.py
# Description: Testing online PCA algorithm population and batch error on artificially generated data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None
#%%
#%%
# imports
from online_pca_simulations import run_simulation
import os
import pylab as plt
import numpy as np
import caiman as cm
from glob import glob

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


#%%
def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    '''function running each iteration of a test
    '''
    output_folder = os.getcwd() + '/test'

    errs = run_simulation(output_folder, simulation_options,
                          generator_options, algorithm_options)

    return errs
#%%
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
    d =  simulation_options['d']
    q =  simulation_options['q']
    algo = algorithm_options['pca_algorithm']
    save_name =  os.path.join(data_fold, '__'.join(['rho', "{:.6f}".format(rho), 'd', str(d), 'q', str(q), 'algo', algo]) + '.npz')
    print('Saving in:' + save_name)
    np.savez(
        save_name,
        **output_dict)

    return errs_pop, errs_batch

#%% start cluster
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% parameters figure generation
test_mode = 'illustrative_examples' # can be 'illustrative_examples' or 'vary_k'
rhos = np.logspace(-4,-0.5,10) # controls SNR
rerun_simulation = False # whether to rerun from scratch or just show the results
parallelize = False # whether to use parallelization or to show results on the go
#%%
if test_mode == 'illustrative_examples':
    #%%
    data_fold = os.path.abspath('./spiked_cov_4_examples')
    d_q_params = [(16, 2), (64, 8), (256, 32), (1024, 64)]
    colors = ['b', 'r', 'g']
    n_repetitions = 5  # 15
    simulation_options['n'] = 100  # 2000
    plot = not parallelize
    if rerun_simulation:
        counter = 0
        os.mkdir(data_fold)
        all_pars = []
        for d, q in d_q_params:
            simulation_options['d'] = d
            simulation_options['q'] = q
            counter += 1
            if plot:
                ax = plt.subplot(1,4,counter)

            for algo in range(3):
                algorithm_options['pca_algorithm'] = algos[algo]
                pop_err_avg = []
                batch_err_avg = []
                for rho in rhos:
                    errs_pop = []
                    errs_batch = []
                    print((d, q, rho))
                    generator_options['rho'] = rho

                    all_pars.append([generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold, n_repetitions])
                    if not parallelize:
                        errs_pop, errs_batch= run_test_wrapper(all_pars[-1])

                    pop_err_avg.append(errs_pop.mean(0)[-1])
                    batch_err_avg.append(errs_batch.mean(0)[-1])

                if plot:
                    line_pop, = ax.loglog(rhos, pop_err_avg, '-d' + colors[algo])
                    line_bat, = ax.loglog(rhos, batch_err_avg, '-o' + colors[algo])
                    line_pop.set_label(algos[algo] + '_pop')
                    line_bat.set_label(algos[algo] + '_batch')
                    plt.xlabel('rho')
                    plt.xlabel('projection error')
                    plt.pause(.1)
            if plot:
                ax.legend()

        #%%
        if parallelize:
            all_res = dview.map(run_test_wrapper,all_pars)
    else:
        results = dict()
        fls = glob(data_fold+'/*.npz')
        fls.sort()
        counter = 0
        for d, q in d_q_params:
            counter += 1
            ax = plt.subplot(1,4,counter)
            for algo in range(3):
                pop_err_avg = []
                batch_err_avg = []
                for rho in rhos:
                    fname = os.path.join(data_fold,'__'.join(['rho', str(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]])+'.npz')
                    with np.load(fname) as ld:
                        pop_err_avg.append(np.mean(ld['batch_err'][()],0)[-1])
                        batch_err_avg.append(np.mean(ld['population_err'][()],0)[-1])

                line_pop, = ax.loglog(rhos, pop_err_avg, '-d' + colors[algo])
                line_bat, = ax.loglog(rhos, batch_err_avg, '-o' + colors[algo])
                line_pop.set_label(algos[algo] + '_pop')
                line_bat.set_label(algos[algo] + '_batch')
                plt.pause(.1)
        ax.legend()
#%%
elif test_mode == 'vary_k':
    #%% vary k
    data_fold = os.path.abspath('./spiked_cov_vary_k')
    n_repetitions = 15
    simulation_options['n'] = 200
    d_q_params = [(1024,2), (1024,8), (1024,32), (1024,64), (1024, 128)]
    colors = ['b','r','g']
    plot = not parallelize
    if rerun_simulation:
        all_pars = []
        os.mkdir(data_fold)
        counter = 0
        for d, q in d_q_params:
            counter += 1
            simulation_options['d'] = d
            simulation_options['q'] = q
            for algo in range(3):
                algorithm_options['pca_algorithm'] = algos[algo]
                for rho in rhos:
                    generator_options['rho'] = rho
                    print((d, q, rho))
                    all_pars.append([generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold, n_repetitions])
                    if plot:
                        errs_pop, errs_batch= run_test_wrapper(all_pars[-1])
                        pop_err_avg.append(errs_pop.mean(0)[-1])
                        batch_err_avg.append(errs_batch.mean(0)[-1])

                if plot:
                    line_pop, = ax.loglog(rhos, pop_err_avg, '-d' + colors[algo])
                    line_bat, = ax.loglog(rhos, batch_err_avg, '-o' + colors[algo])
                    line_pop.set_label(algos[algo] + '_pop')
                    line_bat.set_label(algos[algo] + '_batch')
                    plt.xlabel('rho')
                    plt.xlabel('projection error')
                    plt.pause(.1)
        if plot:
            ax.legend()

        if parallelize:
            all_errs = dview.map(run_test_wrapper,all_pars)


#%% stop cluster
dview.terminate()



