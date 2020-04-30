import autofit as af
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/sarah/VAMP/vamp_2.0')
from vamp_src.model import profile_models
from vamp_src.dataset.spectrum import *
from vamp_src.dataset.preprocess import *
import vamp_src.phase.phase as ph

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output_params"
)

ncomp_true = '2'
param_setting = 'default'
combos = ['a', 'b', 'c', 'd', 'e']
ncomps = [1, 2, 3]
spectrum_dir = '/home/sarah/VAMP/vamp_2.0/vamp_workspace/runners/multinest_params/data/'

chi_squared = np.zeros(len(combos)*len(ncomps))
max_log_l = np.zeros(len(combos)*len(ncomps))
evidence = np.zeros(len(combos)*len(ncomps))

i = 0
for combo in combos:
    filename = spectrum_dir+'combo_' + combo + '_'+ncomp_true+'_component.h5'

    print('\nCombo '+combo+'\n')

    for ncomp in ncomps:

        print('\nNo components: '+str(ncomp)+'\n')

        full_dataset = read_from_h5py(filename)

        # Lets actually narrow this down by only fitting the absorption region
        split = SplitRegions(full_dataset)
        dataset = split.new_region_spectra()[0]

        phase_name="combo_"+combo+"_true_"+ncomp_true+"_phase_"+str(ncomp)+"_"+param_setting
        sigma_max = 0.5 * (np.max(dataset.frequency) - np.min(dataset.frequency))

        if ncomp == 1:
            profiles = af.CollectionPriorModel(
                        gaussian_0=profile_models.Gaussian)

        elif ncomp == 2:
            profiles=af.CollectionPriorModel(
                        gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian)
        
        elif ncomp == 3:
            profiles = af.CollectionPriorModel(
                        gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian)

        for component in profiles.dict:
            profiles.dict[component].sigma = af.UniformPrior(lower_limit=0, upper_limit=sigma_max)
            profiles.dict[component].center = af.UniformPrior(lower_limit=np.min(dataset.frequency), upper_limit=np.max(dataset.frequency)) # might need to adjust this as real frequencies scale inversely

        for n in range(len(profiles) -1):
            profiles.add_assertion(profiles[n].center < profiles[n+1].center)

        phase = ph.Phase(phase_name=phase_name,profiles=profiles)

        result = phase.run(dataset=dataset)

        model = result.most_likely_model_spectrum
        chi_squared[i] = result.analysis.get_reduced_chi_squared(model)
        max_log_l[i] = result.output.maximum_log_likelihood
        evidence[i] = result.output.evidence

        i += 1

print('\nAll chi squareds (a1, a2, b1, b2...)')
print(chi_squared)
print('\nAll max likelihood')
print(max_log_l)
print('\nAll evidence')
print(evidence)