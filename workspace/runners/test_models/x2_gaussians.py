import autofit as af
import os
import sys
sys.path.append('/home/sarah/reVAMP')
import matplotlib.pyplot as plt
from src.model import profile_models
from src.dataset.spectrum import Spectrum
import src.phase.phase as ph
from workspace.make_data import FakeGauss

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
config_path = workspace_path + "config"
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

fakeGaussA = FakeGauss(center=-1.0, sigma=2.0, intensity=0.5)
fakeGaussB = FakeGauss(center=1.5, sigma=1.0, intensity=1.0)

fakeGauss_2comp = fakeGaussB.gauss + fakeGaussA.gauss + fakeGaussA.noise
dataset = Spectrum(fakeGaussA.x, fakeGaussA.x, 1.0 - fakeGauss_2comp, fakeGaussA.noise)

phase = ph.Phase(
    phase_name="phase_x2_gaussians",
    profiles=af.CollectionPriorModel(
        gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian
    ),
)
result = phase.run(dataset=dataset)

plt.plot(dataset.frequency, result.most_likely_model_spectrum)
plt.plot(dataset.frequency, dataset.flux)
plt.show()

# We also have an 'output' attribute, which in this case is a MultiNestOutput object:
print(result.samples)
# This object acts as an interface between the MultiNest output results on your hard-disk and this Python code. For
# example, we can use it to get the evidence estimated by MultiNest.
print(result.samples.log_evidence)
# We can also use it to get a model-instance of the "most probable" model, which is the model where each parameter is
# the value estimated from the probability distribution of parameter space.
mp_instance = result.samples.most_probable_instance
print()
print("Most Probable Model:\n")
print("Centre = ", [i.center for i in mp_instance.profiles])
print("Intensity = ", [i.intensity for i in mp_instance.profiles])
print("Sigma = ", [i.sigma for i in mp_instance.profiles])