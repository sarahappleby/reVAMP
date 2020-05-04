import autofit as af
from src.dataset.spectrum import Spectrum
from src.phase.makephase import MakePhase
import numpy as np

class Process()

	def __init__(dataset: Spectrum, ncomp_init, mode='Gaussian', non_linear_class):

		self.dataset = dataset
		self.ncomp_init = ncomp_init
		self.mode = mode
		self.non_linear_class = non_linear_class

	def get_phase(self, phase_name):
		make_phase = MakePhase(self.mode, self.ncomp, phase_name, self.non_linear_class, self.sigma_max, self.freq_min, self.freq_max)
		self.phase = make_phase.phase

	def find_good_fit(self):

		

	    phases = []
	    results = []
	    evidences = []

	    good_fit = False
	    evidences.insert(0, -1. * np.inf)

	    i = 1
	    while not good_fit:

	        # try 1 component, if the chi squared is less than threshold, stop
	        # if not, try 1 component again just in case
	        # try 2 component, if the chi squared is less .... stop
	        # etc
	        # is it possible to adjust what the multinest considers a 'good enough' fit, i.e. stop 'early'?

	        phase = make_phase(mode, n_components)
	        result = phase.run(dataset=dataset)
	        evidence = result.output.evidence

	        ### Evaluate the result
	        if evidence > evidences[i - 1]:
	            n_components += 1
	            i += 1
	            phases.append(phase)
	            results.append(result)
	            evidences.append(evidence)
	        else:
	            good_fit == True

	    return phases, results, evidences


