# Title: ex_spiked_covariance_timings.py
# Description: Testing online PSP algorithm timings on artificially generated data
# Author: Victor Minden (vminden@flatironinstitute.org) and Andrea Giovannucci (agiovannucci@flatironinstitute.org)
# Notes: Adapted from code by Andrea Giovannucci
# Reference: None


# imports
from online_psp.online_psp_simulations import run_simulation
import os
import pylab as plt
import numpy as np

# general parameters
error_options = {}

generator_options = {
    'method': 'spiked_covariance',
    'lambda_K': 5e-1,
    'normalize': True,
    'rho': 1e-2 / 5,
    'scale_data': True,
    'shuffle': False
}

n_repetitions = 10
Ks = [64, 128, 256, 512, 1024, 2048, 4096]

# Can toggle larger
D = 8192 #32768

simulation_options = {
    'D': D,
    'K': None,
    'N': 10,
    'N0': 0,
    'n_epoch': 1,
    'error_options': error_options,
    'pca_init': False,
    'init_ortho': True,
}

colors = ['b', 'r', 'g','k']
algos = ['FSM', 'IPCA', 'CCIPCA', 'SM']
algo = algos[0]

algorithm_options = {
    'pca_algorithm': algo,
}


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    timing = run_simulation(simulation_options, generator_options, algorithm_options)
    return timing

results = {}


ax = plt.subplot(1, 1, 1)

# TODO: fix
filename = './timings/blah.blah'
os.makedirs(os.path.dirname(filename), exist_ok=True)

for algo in range(len(algos)):
    algo_timings = []
    for K in Ks:
        timings = []

        # Take the mean over n_repetitions trials
        for _ in range(n_repetitions):
            simulation_options['K'] = K
            algorithm_options['pca_algorithm'] = algos[algo]
            timing = run_test(simulation_options, algorithm_options, generator_options)
            timings.append(timing)

        algo_timings.append(np.mean(timings))
        output_dict = {
            'generator_options': generator_options,
            'simulation_options': simulation_options,
            'algorithm_options': algorithm_options,
            'D': D,
            'K': K,
            'N': simulation_options['N'],
            'timing': np.mean(timings)
        }
        print('Runtime for (algo,K)=(%s,%d): %f'%(algos[algo], K, output_dict['timing']))
        np.savez('./timings/' + 'D_%d_K_%d_algo_%s'
                 % (D, K, algos[algo]),
                 output_dict)

    line_timing, = ax.plot(Ks, np.array(algo_timings) / simulation_options['N'], '-d' + colors[algo])

    line_timing.set_label(algos[algo])

Ks = np.array(Ks)
# TODO These had better be tuned for pretty plots
ax.plot(Ks, Ks / 5e4, '--r')
ax.plot(Ks, Ks ** 2 / 2e5, '--r')
ax.legend()

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of components')
plt.ylabel('Time per iteration (s)')
plt.savefig('./timings/d_%d.png' % (D))
plt.show()