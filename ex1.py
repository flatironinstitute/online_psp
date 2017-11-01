from online_pca_simulations import run_simulation
import os
from matplotlib import pyplot as plt
import numpy as np
import copy



def run_test(simulation_options, algorithm_options):
    #algo_names = ['inv_minimax_PCA', 'rd_minimax_alignment_PCA', 'minimax_alignment_PCA', 'CCIPCA', 'SNL_PCA', 'SGA_PCA', 'incremental_PCA', 'minimax_whitening_PCA', 'if_minimax_whitening_PCA', 'minimax_PCA', 'if_minimax_PCA', 'OSM_PCA']
    output_folder = os.getcwd() + '/ex'

    generator_options = {
        'method'   : 'spiked_covariance',
        'lambda_q' : 5e-1,
        'normalize': True,
        'rho'      : 1e-2/5
    }


    errs = run_simulation(output_folder, simulation_options, generator_options, algorithm_options)

    handles = []

    plt.figure(1)
    plt.subplot(211)
    plt.title(algorithm_options['pca_algorithm'])
    for err_name in errs:
        print(err_name +': %f' %(errs[err_name][-1]))
    for err_name in errs:
        if err_name in ['batch_err','batch_alignment_err','batch_whitening_err']:
            handle, = plt.plot(np.log10(errs[err_name]), label=err_name)
            handles.append(handle)
    plt.legend(handles=handles)
    plt.ylabel('Error (log10 scale)')

    handles = []
    plt.subplot(212)
    for err_name in errs:
        if err_name in ['diag_err']:
            handle, = plt.plot(errs[err_name], label=err_name)
            handles.append(handle)
    plt.legend(handles=handles)
    plt.ylabel('Error (linear scale)')
    plt.xlabel('Iteration')
    plt.ylim(ymax=1, ymin=0)
    plt.show()








        ##############
tests = []
error_options = {
    'n_skip' : 128, ##NOT IMPLEMENTED
    'orthogonalize_iterate' : True,
    'compute_batch_error' : True,
    #'compute_population_error' : True,
    # 'compute_strain_error' : False,
    # 'compute_reconstruction_error' : False,
    #'compute_pop_whitening_error' : True,
    # 'compute_batch_whitening_error' : True,
    'compute_batch_alignment_error' : True,
    'compute_diag_error' : True
}
simulation_options ={
        'd' : 16,
        'q' : 4,
        'n' : 4096,
        'n0': 0, ##NOT IMPLEMENTED
        'n_epoch': 1,
        'n_test' : 256,
        'error_options' : error_options,
        'pca_init': False,
        'init_ortho': True
    }

algorithm_options = {
    'pca_algorithm' : 'minimax_alignment_PCA',
    'tau'           : 0.5,
    'tol'           : 1e-7,
    'step_rule'     : lambda t: 50/(1e4 + t)
}


tests.append((copy.deepcopy(simulation_options), copy.deepcopy(algorithm_options)))

simulation_options['d'] = 160
simulation_options['q'] = 4
algorithm_options['step_rule'] = lambda t: 50/(1e4 + t)/3

tests.append((copy.deepcopy(simulation_options), copy.deepcopy(algorithm_options)))

simulation_options['d'] = 160
simulation_options['q'] = 8
simulation_options['n'] = 4096*4
algorithm_options['step_rule'] =  lambda t: 50/(1e4 + t/10)/8


tests.append((copy.deepcopy(simulation_options), copy.deepcopy(algorithm_options)))

simulation_options['d'] = 160
simulation_options['q'] = 32
simulation_options['n'] = 4096*4*4
algorithm_options['step_rule'] = lambda t: 50/(1e4 + t/10)/64


tests.append((copy.deepcopy(simulation_options), copy.deepcopy(algorithm_options)))


for (sim,alg) in tests:
    run_test(sim,alg)
# second()
# Still some difficulty
# third()
# fourth()
