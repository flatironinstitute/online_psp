# Title: ex_spiked_covariance_errors.py
# Description: Testing online PSP algorithm population and batch error on artificially generated data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None


# imports
from online_psp.online_psp_simulations import run_simulation
import os
import pylab as plt
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    '''function running each iteration of a test
    '''
    errs = run_simulation(simulation_options,
                          generator_options, algorithm_options)
    return errs


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
        'D': simulation_options['D'],
        'K': simulation_options['K'],
        'N': simulation_options['N'],
        'rho': generator_options['rho'],
        'n_epoch': 1,
        'N0': 0,
        'population_err': errs_pop,
        'batch_err': errs_batch,
    }
    rho = generator_options['rho']
    D = simulation_options['D']
    K = simulation_options['K']
    algo = algorithm_options['pca_algorithm']
    save_name = os.path.join(data_fold,
                             '__'.join(
                                 ['rho', "{:.6f}".format(rho), 'd', str(D), 'q', str(K), 'algo', algo]) + '.npz')
    print('Saving in:' + save_name)
    np.savez(
        save_name,
        **output_dict)

    return errs_pop, errs_batch


###############################

# TODO fix
n_repetitions = 1 #10

# general parameters

error_options = {
    'n_skip': 50,
    'compute_batch_error': True,
    'compute_population_error': True,
    'compute_reconstruction_error': False
}

generator_options = {
    'method': 'spiked_covariance',
    'lambda_K': 5e-1,
    'normalize': True,
    'rho': 1e-2 / 5,
    'scale_data': True,
    'shuffle': False
}

simulation_options = {
    'N': 6000,
    'N0': 0,
    'n_epoch': 1,
    'error_options': error_options,
    'pca_init': False,
    'init_ortho': True,
}

algos = ['FSM', 'IPCA', 'CCIPCA', 'SM']

algorithm_options = {
    'pca_algorithm': None,
    'tau': 0.5,
    'tol': 1e-7
}


#########################################
# parameters figure generation
rhos = np.logspace(-4, -0.3, 10)  # controls SNR
rerun_simulation = True  # whether to rerun from scratch or just show the results

all_pars = []
gammas_ = [0.6, 1.5, 2]
for gamma_ in gammas_:
    algorithm_options['gamma'] = gamma_
    # %% vary k
    D_K_params = [(256, 16), (256, 128), (2048, 16), (2048, 128)]
    # TODO THIS FOLDER GENERATING IS A DISASTER
    data_fold = os.path.abspath('./errors_spiked_cov_vary_k_d_gamma_' + str(gamma_))
    #TODO  What is this...
    if gamma_ == gammas_[0]: # run for IPCA AND CCIPCA only onthe first iteration
        algos = ['FSM', 'IPCA', 'CCIPCA']
    else:
        algos = ['FSM']

    if rerun_simulation:
        os.mkdir(data_fold)

    counter = 0
    for D, K in D_K_params:
        simulation_options['D'] = D
        simulation_options['K'] = K

        for algo in range(len(algos)):
            algorithm_options['pca_algorithm'] = algos[algo]
            pop_err_avg = []
            batch_err_avg = []

            for rho in rhos:
                generator_options['rho'] = rho
                print((D, K, rho))
                all_pars.append(
                    [generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold[:],
                     n_repetitions])

                if rerun_simulation:
                    errs_pop, errs_batch = run_test_wrapper(all_pars[-1])
                    pop_err_avg.append(errs_pop.mean(0)[-1])
                    batch_err_avg.append(errs_batch.mean(0)[-1])
                    errs_pop = np.array(errs_pop)
                    errs_batch = np.array(errs_batch)
                else:
                    fname = os.path.join(data_fold, '__'.join(
                        ['rho', "{:.6f}".format(rho), 'D', str(D), 'K', str(K), 'algo', algos[algo]]) + '.npz')

                    with np.load(fname) as ld:
                        print(pop_err_avg)
                        pop_err_avg.append(np.mean(ld['population_err'][()], 0)[-1])
                        batch_err_avg.append(np.mean(ld['batch_err'][()], 0)[-1])

            if pop_err_avg is not None:
                ax = plt.subplot(len(D_K_params), 2, 2 * counter + 1)
                if algo ==0:
                    line_pop, = ax.loglog(rhos, pop_err_avg)
                else:
                    line_pop, = ax.loglog(rhos, pop_err_avg)

                line_pop.set_label('gamma_:' + str(gamma_) + ',' + algos[algo])

                ax = plt.subplot(len(D_K_params), 2, 2 * counter + 2)
                if algo == 0:
                    line_bat, = ax.loglog(rhos, batch_err_avg)
                else:
                    line_bat, = ax.loglog(rhos, batch_err_avg)
                line_bat.set_label('gamma_:' + str(gamma_) + ',' + algos[algo])

        counter += 1

    # What to do about this
    if pop_err_avg is not None:
        for counter in range(len(D_K_params)):
            ax = plt.subplot(len(D_K_params), 2, 2 * counter + 1)
            plt.ylabel('population error')
            plt.xlabel('rho')
            plt.title('K=' + str(D_K_params[counter][-1]))

            ax = plt.subplot(len(D_K_params), 2, 2 * counter + 2)
            plt.ylabel('batch error')
            plt.xlabel('rho')
            plt.title('K=' + str(D_K_params[counter][-1]))


        ax = plt.subplot(len(D_K_params), 2, 2 * counter + 2)
        ax.legend()
    plt.savefig('./d_%d.png' % (D))
    plt.show()

