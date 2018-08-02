# Title: ex_real_data_errors.py
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

# general parameters
error_options = {
    'n_skip': 50,
    'compute_batch_error': True,
    'compute_population_error': True,
    'compute_reconstruction_error': False
}

generator_options = {
    'method': 'real_data',
    'scale_data': True,
    'shuffle' : True
}

simulation_options = {
    'D': None,
    'K': 16,
    'N': 'auto',  # can set a number here, will select frames multiple times
    'N0': 0,
    'error_options': error_options,
    'pca_init': False,
    'init_ortho': True,
}


algorithm_options = {
    'pca_algorithm': None,
}


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    errs = run_simulation(simulation_options, generator_options, algorithm_options)
    return errs



def run_test_wrapper(params):
    generator_options, simulation_options, algorithm_options, data_fold, n_repetitions = params
    errs_batch = []
    for _ in range(n_repetitions):
        err = run_test(simulation_options, algorithm_options, generator_options, )
        errs_batch.append(err['batch_err'])

    errs_batch = np.array(errs_batch)
    output_dict = {
        'generator_options': generator_options,
        'simulation_options': simulation_options,
        'algorithm_options': algorithm_options,
        'D': simulation_options['D'],
        'K': simulation_options['K'],
        'N': simulation_options['N'],
        'filename': generator_options['filename'],
        'n_epoch': simulation_options['n_epoch'],
        'N0': 0,
        'batch_err': errs_batch,

    }
    K = simulation_options['K']
    filename = generator_options['filename']
    algo = algorithm_options['pca_algorithm']
    save_name = os.path.join(data_fold,
                             '__'.join(['fname', filename.split('/')[-1], 'K', str(K), 'algo', algo]) + '.npz')
    print('Saving in:' + save_name)
    np.savez(save_name, **output_dict)

    return errs_batch


#####################
# parameters figure generation
rhos = np.logspace(-4, -0.5, 10)  # controls SNR
rerun_simulation = True  # whether to rerun from scratch or just show the results

all_pars = []
gammas = [0.6, 1.5, 2]

for gamma_ in  gammas:
    algorithm_options['gamma'] = gamma_
    data_fold = os.path.abspath('./real_data_learning_curves_gamma_' + str(gamma_))
    os.makedirs(data_fold, exist_ok=True)
    names = ['YaleB_32x32.mat','ATT_faces_112_92.mat', 'MNIST.mat']
    n_epochs = [10, 30, 1]
    Ks = [16, 64, 128]

    if gamma_ == gammas[0]:
        algos = ['FSM', 'IPCA', 'CCIPCA']
    else:
        algos = ['FSM']

    colors = ['b', 'r', 'g']
    n_repetitions = 10

    simulation_options['N'] = 'auto'

    for counter_K, K in enumerate(Ks):
        counter_name = 0
        for n_ep, name in zip(n_epochs, names):
            simulation_options['n_epoch'] = n_ep
            counter_name += 1
            generator_options['filename'] = '../datasets/' + name

            ax = plt.subplot(len(Ks), len(names), len(names) * counter_K + counter_name)

            for algo, algor in enumerate(algos):
                print(name)

                algorithm_options['pca_algorithm'] = algor
                print((name, K))
                simulation_options['K'] = K
                all_pars.append(
                    [generator_options.copy(), simulation_options.copy(), algorithm_options.copy(), data_fold[:],
                     n_repetitions])

                if rerun_simulation:
                    errs_batch = run_test_wrapper(all_pars[-1])
                    batch_err_avg = np.median(errs_batch, 0)
                    batch_err_avg = batch_err_avg[batch_err_avg > 0]
                else:
                    fname = os.path.join(data_fold, '__'.join(
                        ['fname', name, 'K', str(K), 'algo', algor]) + '.npz')

                    with np.load(fname) as ld:
                        batch_err_avg = np.median(ld['batch_err'][()], 0)
                        batch_err_avg = batch_err_avg[batch_err_avg > 0]

                if batch_err_avg is not None:
                    plt.title('k=' + str(K) + ',' + name)
                    line_bat, = ax.loglog(batch_err_avg.T)
                    line_bat.set_label('gamma_' + str(gamma_) + ',' + algos[algo])
                    plt.ylabel('subspace error')


if batch_err_avg is not None:
    ax.legend()
    plt.xlabel('sample (x n_skip)')
    plt.savefig('./real_data.png')
    plt.show()

