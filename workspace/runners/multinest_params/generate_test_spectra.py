import autofit as af
import os
import numpy as np
import sys
sys.path.append('/home/sarah/reVAMP/')
import matplotlib.pyplot as plt
from src.model import profile_models
from src.dataset.spectrum import Spectrum
import src.phase.phase as ph
from workspace.make_data import FakeGauss


workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output")

spectrum_dir = '/home/sarah/reVAMP/workspace/runners/multinest_params/data/'

centers_a = [1., 1., 2., -2., 0.]
centers_b = [0., -2., -1., 1., -1.]
centers_c = [-2., 3., -3., 0.5, 2.]
sigmas_a = [1.0, 1.0, 2.0, 0.5, 1.0]
sigmas_b = [0.5, 2.0, 0.5, 0.5, 1.5]
sigmas_c = [2.0, 3.0, 1.0, 0.5, 2.0]
intensity_a = [0.5, 0.3, 0.5, 0.6, 0.4]
intensity_b = [0.8, 0.5, 0.4, 0.7, 1.0]
intensity_c = [0.3, 0.5, 0.7, 0.6, 0.7]

combos = ['a', 'b', 'c', 'd', 'e']

for i, combo in enumerate(combos):

	attrs = {}
	fakeGaussA = FakeGauss(center=centers_a[i], sigma=sigmas_a[i], intensity=intensity_a[i])
	fakeGaussB = FakeGauss(center=centers_b[i], sigma=sigmas_b[i], intensity=intensity_b[i])
	fakeGaussC = FakeGauss(center=centers_c[i], sigma=sigmas_c[i], intensity=intensity_c[i])

	dataset_a = Spectrum(fakeGaussA.x, fakeGaussA.x, np.exp(-1. * fakeGaussA.gauss) + fakeGaussA.noise, fakeGaussA.noise)
	attrs['center_0'] = centers_a[i]; attrs['sigma_0'] = sigmas_a[i]; attrs['intensity_0'] = intensity_a[i]
	dataset_a.save_as_h5py(spectrum_dir+'combo_' + combo + '_1_component.h5', attributes=attrs)
	dataset_a.plot_spectrum(spectrum_dir+'combo_' + combo + '_1_component.png')
	
	dataset_b = Spectrum(fakeGaussB.x, fakeGaussB.x, np.exp(-1. * (fakeGaussA.gauss + fakeGaussB.gauss)) + fakeGaussB.noise, fakeGaussB.noise)
	attrs['center_1'] = centers_b[i]; attrs['sigma_1'] = sigmas_b[i]; attrs['intensity_1'] = intensity_b[i]
	dataset_b.save_as_h5py(spectrum_dir+'combo_' + combo + '_2_component.h5', attributes=attrs)
	dataset_b.plot_spectrum(spectrum_dir+'combo_' + combo + '_2_component.png')

	dataset_c = Spectrum(fakeGaussC.x, fakeGaussC.x, np.exp(-1. * (fakeGaussA.gauss + fakeGaussB.gauss + fakeGaussC.gauss)) + fakeGaussC.noise, fakeGaussC.noise)
	attrs['center_2'] = centers_c[i]; attrs['sigma_2'] = sigmas_c[i]; attrs['intensity_2'] = intensity_c[i]
	dataset_c.save_as_h5py(spectrum_dir+'combo_' + combo + '_3_component.h5', attributes=attrs)
	dataset_c.plot_spectrum(spectrum_dir+'combo_' + combo + '_3_component.png')