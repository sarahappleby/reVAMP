import autofit as af
import os

import sys
sys.path.append('/home/sarah/reVAMP/')
from src.model import profile_models
from src.dataset.spectrum import Spectrum
import src.phase.phase as ph
from workspace.make_data import FakeGauss

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

fakeGaussA = FakeGauss(center=-1.0, sigma=2.0, intensity=0.5)
fakeGaussB = FakeGauss(center=1.5, sigma=1.0, intensity=1.0)
fakeGaussC = FakeGauss(center=3., sigma=0.5, intensity=0.7)

fakeGauss_3comp = fakeGaussB.gauss + fakeGaussA.gauss + fakeGaussC.gauss + fakeGaussA.noise
dataset = Spectrum(fakeGaussA.x, fakeGaussA.x, 1.0 - fakeGauss_2comp, fakeGaussA.noise)

mode = 'gaussian'
phases, results, evidences = ph.find_good_fit(mode, dataset)