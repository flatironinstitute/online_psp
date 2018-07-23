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




def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    output_folder = os.getcwd() + '/test'

    errs = run_simulation(output_folder, simulation_options,
                          generator_options, algorithm_options)

    return errs


#%%
data_fold = './spiked_cov_4_examples'
os.mkdir(data_fold)
n_repetitions = 3
rhos = np.logspace(-4,-0.5,10)
simulation_options['n'] = 500
d_q_params = [(16,2), (64,8), (256,32), (1024, 64)]
results = dict()
counter = 0
colors = ['b','r','g']
for d, q in d_q_params:
    counter += 1
    ax = plt.subplot(1,4,counter)
    for algo in range(3):
        pop_err_avg = []
        batch_err_avg = []
        for rho in rhos:
            errs_pop = []
            errs_batch = []
            for reps in range(n_repetitions):
                print((d, q, rho))
                generator_options['rho'] = rho
                simulation_options['d'] = d
                simulation_options['q'] = q
                algorithm_options['pca_algorithm'] = algos[algo]
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
                'd': d,
                'q': q,
                'n': simulation_options['n'],
                'n_epoch': 1,
                'n0': 0,
                'population_err': errs_batch,
                'batch_err': errs_pop,
            }
            np.savez(
                os.path.join(data_fold,'__'.join(['rho', str(rho), 'd', str(d), 'q', str(q), 'algo', algos[algo]])+'.npz'),
                     **output_dict)
            results['__'.join([str(rho), str(d), str(q), algos[algo]])] = output_dict


            pop_err_avg.append(errs_pop.mean(0)[-1])
            batch_err_avg.append(errs_batch.mean(0)[-1])

        line_pop, = ax.semilogx(rhos, pop_err_avg, '-d' + colors[algo])
        line_bat, = ax.semilogx(rhos, batch_err_avg, '-o' + colors[algo])
        line_pop.set_label(algos[algo] + '_pop')
        line_bat.set_label(algos[algo] + '_batch')
        plt.xlabel('rho')
        plt.xlabel('projection error')
        plt.pause(.1)

ax.legend()


#%% LOAD RESULTS AND DISPLAY
from glob import glob
results = dict()
fls = glob('test/*.npz')
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

        line_pop, = ax.semilogx(rhos, pop_err_avg, '-d' + colors[algo])
        line_bat, = ax.semilogx(rhos, batch_err_avg, '-o' + colors[algo])
        line_pop.set_label(algos[algo] + '_pop')
        line_bat.set_label(algos[algo] + '_batch')
        ax.legend()
        plt.pause(.1)



