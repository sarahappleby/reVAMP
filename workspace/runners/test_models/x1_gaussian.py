import autofit as af
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('/disk2/sapple/VAMP/vamp_2.0')
from vamp_src.model import profile_models
from vamp_src.dataset.spectrum import Spectrum
import vamp_src.phase.phase as ph

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

from vamp_workspace.make_data import FakeGauss

fakeGauss = FakeGauss(center=-1.0, sigma=2.0, intensity=0.5)

phase = ph.Phase(phase_name="phase_x1_gaussians",
                 profiles=af.CollectionPriorModel(gaussian_0=profile_models.Gaussian))

dataset = Spectrum(
    fakeGauss.x, fakeGauss.x, 1.0 - fakeGauss.noisy_gauss, fakeGauss.noise
)
result = phase.run(dataset=dataset)

plt.plot(dataset.frequency, result.most_likely_model_spectrum)
plt.plot(dataset.frequency, dataset.flux)
plt.show()

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
print("Centre = ", mp_instance.profiles.gaussian_0.center)
print("Intensity = ", mp_instance.profiles.gaussian_0.intensity)
print("Sigma = ", mp_instance.profiles.gaussian_0.sigma)