import autofit as af
from src.dataset.spectrum import Spectrum
from src.phase.phase import Phase
import numpy as np

class MakePhase():

	def __init__(self, mode, ncomp, phase_name, non_linear_class, sigma_max, freq_min, freq_max):
		self.mode = mode
		self.ncomp = ncomp
		self.phase_name = phase_name
		self.non_linear_class = non_linear_class
		self.sigma_max = sigma_max
		self.freq_min = freq_min
		self.freq_max = freq_max

		self.make_phase()

	def make_gaussian_profiles(self):

		if self.ncomp == 1:
			self.profiles = af.CollectionPriorModel(
							gaussian_0=profile_models.Gaussian)

		elif self.ncomp == 2:
			self.profiles=af.CollectionPriorModel(
							gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian)
			
		elif self.ncomp == 3:
			self.profiles = af.CollectionPriorModel(
							gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian)

		elif self.ncomp == 4:
			self.profiles = af.CollectionPriorModel(
							gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian, 
							gaussian_3=profile_models.Gaussian)

		elif self.ncomp == 5:
			self.profiles = af.CollectionPriorModel(
							gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian, 
							gaussian_3=profile_models.Gaussian, gaussian_4=profile_models.Gaussian)

	def make_voigt_profiles(self):

		if self.ncomp == 1:
			self.profiles = af.CollectionPriorModel(
							voigt_0=profile_models.Voigt)

		elif self.ncomp == 2:
			self.profiles=af.CollectionPriorModel(
							voigt_0=profile_models.Voigt, voigt_1=profile_models.Voigt)
			
		elif self.ncomp == 3:
			self.profiles = af.CollectionPriorModel(
							voigt_0=profile_models.Voigt, voigt_1=profile_models.Voigt, voigt_2=profile_models.Voigt)

		elif self.ncomp == 4:
			self.profiles = af.CollectionPriorModel(
							voigt_0=profile_models.Voigt, voigt_1=profile_models.Voigt, voigt_2=profile_models.Voigt, 
							voigt_3=profile_models.Voigt)

		elif self.ncomp == 5:
			self.profiles = af.CollectionPriorModel(
							voigt_0=profile_models.Voigt, voigt_1=profile_models.Voigt, voigt_2=profile_models.Voigt, 
							voigt_3=profile_models.Voigt, voigt_4=profile_models.Voigt)

	def adapt_priors(self):

		for component in self.profiles.dict:
			self.profiles.dict[component].sigma = af.UniformPrior(lower_limit=0, upper_limit=self.sigma_max)
			self.profiles.dict[component].center = af.UniformPrior(lower_limit=self.freq_min, upper_limit=self.freq_max)

	def assert_centers(self):

		for n in range(len(self.profiles) -1):
			self.profiles.add_assertion(self.profiles[n].center < self.profiles[n+1].center)

	def make_phase(self):

		if self.mode == 'Gaussian':
			self.make_gaussian_profiles()
		elif self.mode == 'Voigt':
			self.make_voigt_profiles()

		self.adapt_priors()
		self.assert_centers()

		self.phase = Phase(phase_name=self.phase_name, 
							profiles=self.profiles, 
							non_linear_class=self.non_linear_class)