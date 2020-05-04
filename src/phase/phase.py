import autofit as af
from src.dataset.spectrum import Spectrum
from src.phase.analysis import Analysis
from src.phase.result import Result
from src.model import profile_models
import numpy as np

class Phase(af.AbstractPhase):

	profiles = af.PhaseProperty("profiles")

	Result = Result

	@af.convert_paths
	def __init__(self, paths, profiles, non_linear_class=af.MultiNest):

		super().__init__(paths=paths, non_linear_class=non_linear_class)
		self.profiles = profiles

	def run(self, dataset: Spectrum):
		"""
		Pass a dataset to the phase, running the phase and non-linear search.

		Parameters
		----------
		dataset: aa.Imaging
			The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.

		Returns
		-------
		result: AbstractPhase.Result
			A result object comprising the best fit model.
		"""

		analysis = self.make_analysis(dataset=dataset)

		result = self.run_analysis(analysis=analysis)

		return self.make_result(result=result, analysis=analysis)

	def make_analysis(self, dataset):
		"""
		Create an Analysis object, which creates the dataset and contains the functions which perform the fit.

		Parameters
		----------
		dataset: aa.Imaging
			The dataset fitted by the phase, which in this case is a PyAutoArray imaging object.

		Returns
		-------
		analysis : Analysis
			An analysis object that the non-linear search calls to determine the fit likelihood for a given model
			instance.
		"""
		return Analysis(dataset=dataset, dimensions=self.model.prior_count, visualize_path=self.optimizer.paths.image_path)

	def make_result(self, result, analysis):
		return self.Result(
			instance=result.instance,
			likelihood=result.likelihood,
			analysis=analysis,
			output=result.output
		)
