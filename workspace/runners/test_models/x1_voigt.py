import autofit as af
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/sarah/reVAMP/')
from src.model import profile_models
from src.dataset.spectrum import Spectrum
import src.phase.phase as ph
from workspace.make_data import FakeVoigt

# Setup the path to the vamp_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the vamp_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

fakeVoigt = FakeVoigt(center=-1.0, intensity=1.0, fwhm_L=1.0, fwhm_G=2.0)

phase = ph.Phase(phase_name="phase_x1_voigt",
                 profiles=af.CollectionPriorModel(voigt_0=profile_models.Voigt))

dataset = Spectrum(
    fakeVoigt.x, fakeVoigt.x, 1.0 - fakeVoigt.noisy_voigt, fakeVoigt.noise
)

result = phase.run(dataset=dataset)

# We also have an 'output' attribute, which in this case is a MultiNestOutput object:
print(result.output)
# This object acts as an interface between the MultiNest output results on your hard-disk and this Python code. For
# example, we can use it to get the evidence estimated by MultiNest.
print(result.output.evidence)
# We can also use it to get a model-instance of the "most probable" model, which is the model where each parameter is
# the value estimated from the probability distribution of parameter space.
mp_instance = result.output.most_probable_instance
print()
print("Most Probable Model:\n")
print("Centre = ", [i.center for i in mp_instance.profiles])
print("Intensity = ", [i.intensity for i in mp_instance.profiles])
print("FHWM Gaussian = ", [i.fwhm_G for i in mp_instance.profiles])
print("FHWM Lorentzian = ", [i.fwhm_L for i in mp_instance.profiles])


