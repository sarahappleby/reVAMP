import autofit as af
from src.dataset.spectrum import Spectrum
from src.phase.makephase import MakePhase
import numpy as np
import time

class Process():

	def __init__(self, dataset: Spectrum, ncomp_init, non_linear_class, phase_prefix, reduced_chi_threshold, ntries_total=2, 
				mode='Gaussian', extra_components=False, verbose=True):

		self.dataset = dataset
		self.ncomp_init = ncomp_init
		self.non_linear_class = non_linear_class
		self.phase_prefix = phase_prefix
		self.reduced_chi_threshold = reduced_chi_threshold
		self.ntries_total = ntries_total
		self.mode = mode
		self.extra_components = extra_components
		self.verbose = verbose

		self.sigma_max = 0.5 * np.abs(np.max(self.dataset.frequency) - np.min(self.dataset.frequency))
		self.freq_min = np.min(self.dataset.frequency)
		self.freq_max = np.max(self.dataset.frequency)

		self.vamp_process()

	def vamp_process(self):
		self.find_good_fit()
		if self.extra_components:
			self.find_extra_components()

	def get_result(self, phase_name):

		make_phase = MakePhase(self.mode, self.ncomp, phase_name, self.non_linear_class, self.sigma_max, self.freq_min, self.freq_max)
		self.result = make_phase.phase.run(dataset=self.dataset)

	def find_good_fit(self):
		
		self.ncomp = self.ncomp_init + 0

		self.times = []
		self.reduced_chi = []
		self.reduced_chi.insert(0, np.inf)
		if self.non_linear_class == af.MultiNest:
			self.evidence = []
			self.evidence.insert(0, -1. * np.inf)
		self.ntries = self.ntries_total - 1

		tries_remaining = self.ntries
		print('Attempting '+str(int(self.ntries_total)) +' tries for each number of components\n')

		while self.reduced_chi[-1] > self.reduced_chi_threshold:

			t_start = time.time()
			phase_name = self.phase_prefix + 'ncomp_'+str(self.ncomp)+'_attempt_'+str(int(self.ntries - tries_remaining))
			self.get_result(phase_name)
			self.times.append(time.time() - t_start)
			model = self.result.most_likely_model_spectrum

			self.reduced_chi.append(self.result.analysis.get_reduced_chi_squared(model))
			if self.non_linear_class == af.MultiNest:
				self.evidence.append(self.result.samples.log_evidence)

			if self.reduced_chi[-1] > self.reduced_chi_threshold:
				if tries_remaining > 0.:
					tries_remaining -=1
				else:
					self.ncomp +=1
					tries_remaining = self.ntries
			
		self.reduced_chi = self.reduced_chi[1:]
		if self.non_linear_class == af.MultiNest:
			self.evidence = self.evidence[1:]

	def find_extra_components(self, evidence_factor=1.1):
		"""
		Try adding more lines as long as each new line both lowers the reduced chi squared
		and also increases the Bayesian evidence by more than evidence_factor
		"""

		if evidence_factor is not None:

			self.old_result = self.result.copy()
			old_model = self.old_result.most_likely_model_spectrum
			
			self.ncomp += 1 
			
			phase_name = self.phase_prefix + '_phase_'+str(self.ncomp)
			self.get_result(phase_name)
			model = self.result.most_likely_model_spectrum
		
			rchi_new = self.result.analysis.get_reduced_chi_squared(model)
			rchi_old = self.old_result.analysis.get_reduced_chi_squared(old_model)

			while (rchi_new < rchi_old) & (self.result.output.evidence > evidence_factor * self.old_result.output.evidence):

				self.reduced_chi.append(self.result.analysis.get_reduced_chi_squared(self.result.most_likely_model_spectrum))
				self.evidence.append(self.result.output.evidence)
				self.old_result = self.result.copy()

				self.ncomp += 1 

				phase_name = self.phase_prefix + '_phase_'+str(self.ncomp)
				self.get_result(phase_name)

				rchi_new = self.result.analysis.get_reduced_chi_squared(self.result.most_likely_model_spectrum)
				rchi_old = self.old_result.analysis.get_reduced_chi_squared(self.old_result.most_likely_model_spectrum)

			self.result = self.old_result.copy()

