import autofit as af
import os
import time
import matplotlib.pyplot as plt
import numpy as np 
import sys
sys.path.append('/home/sarah/reVAMP/')
from src.model import profile_models
from src.dataset.spectrum import *
from src.phase.evaluate import Process

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

reduced_chi_threshold = 2.
ntries = 2
ncomp_true = '1'
param_setting = ''
combos = ['a', 'b', 'c', 'd', 'e']

spectrum_dir = '/home/sarah/reVAMP/workspace/runners/multinest_params/data/'
non_linear_class=af.MultiNest

i = 0

vamps = []
times = []

for combo in combos:
	filename = spectrum_dir+'combo_' + combo + '_'+ncomp_true+'_component.h5'
	phase_name = "combo_"+combo + "_true_"+ncomp_true+'_'+param_setting
	print('\nCombo '+combo+'\n')
	dataset = read_from_h5py(filename)
	dataset.flux[dataset.flux < 0.] = 0.
	dataset.estimate_ncomp()

	print('Starting with initial estimate of '+str(dataset.ncomp) +' components')

	t_start = time.time()
	vamps.append(Process(dataset, dataset.ncomp, non_linear_class, phase_name, reduced_chi_threshold, ntries))
	times.append(time.time() - t_start)