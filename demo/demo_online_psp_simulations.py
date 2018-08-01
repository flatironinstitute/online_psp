from online_psp.online_psp_simulations import run_simulation

from collections import defaultdict
from matplotlib import pyplot as plt


def run_test(simulation_options=None, algorithm_options=None, generator_options=None):
    error_options = defaultdict(int, simulation_options['error_options'])
    compute_error = any(error_options)

    errs = run_simulation(simulation_options, generator_options, algorithm_options)

    handles = []
    if compute_error:
        fig = plt.figure(1)
        plt.title(algorithm_options['pca_algorithm'])

        for err_name in errs:
            err = errs[err_name]
            err = err[err > 0]
            print(err_name + ': %e' % (err[-1]))
            plt.loglog(err, label=err_name)

        plt.legend()
        plt.ylabel('Error (log10 scale)')
        plt.xlabel('Iteration (x n_skip)')

        plt.show()


error_options = {
    'n_skip': 64,
    'compute_batch_error': True,
    'compute_population_error': True,
    'compute_reconstruction_error': False,
}

spiked_covariance = True
scale_data        = True

if spiked_covariance:
    generator_options = {
        'method': 'spiked_covariance',
        'lambda_K': 5e-1,
        'normalize': True,
        'rho': 1e-2 / 5,
        'scale_data': scale_data,
        'shuffle': False
    }

    simulation_options = {
        'D': 100,
        'K': 20,
        'N': 5000,
        'N0': 0,
        'n_epoch': 1,
        'error_options': error_options,
        'pca_init': False,
        'init_ortho': True
    }
else:
    # Use real data
    dsets = ['ATT_faces_112_92.mat', 'YaleB_32x32.mat', 'MNIST.mat']
    dset = dsets[1]
    print('** ' + dset)
    generator_options = {
        'method': 'real_data',
        'filename': './datasets/' + dset,
        'scale_data': scale_data,
        'shuffle': False
    }
    simulation_options = {
        'D': None,
        'K': 16,
        'N': 'auto',  # can set a number here, will sample with replacement if needed
        'N0': 0,
        'n_epoch': 50,
        'error_options': error_options,
        'pca_init': False,
        'init_ortho': True,
    }

algos = ['FSM', 'IPCA', 'CCIPCA', 'SM']
algo = algos[0]

algorithm_options = {
    'pca_algorithm': algo
}
run_test(generator_options=generator_options, simulation_options=simulation_options,
         algorithm_options=algorithm_options)
