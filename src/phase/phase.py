import autofit as af
from vamp_src.dataset.spectrum import Spectrum
from vamp_src.phase.analysis import Analysis
from vamp_src.phase.result import Result
from vamp_src.model import profile_models
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

def make_gaussian_phase(n_components):
    if n_components == 1:
        return Phase(
            phase_name="phase_1_gaussians",
            profiles=af.CollectionPriorModel(gaussian_0=profile_models.Gaussian)
            )

    elif n_components == 2:
        return Phase(
            phase_name="phase_2_gaussians",
            profiles=af.CollectionPriorModel(gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian)
            )
    elif n_components == 3:
        return Phase(
            phase_name="phase_3_gaussians",
            profiles=af.CollectionPriorModel(gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian)
            )
    elif n_components == 4:
        return Phase(
            phase_name="phase_4_gaussians",
            profiles=af.CollectionPriorModel(gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian, 
                                              gaussian_3=profile_models.Gaussian)
            )
    elif n_components == 5:
        return Phase(
            phase_name="phase_5_gaussians",
            profiles=af.CollectionPriorModel(gaussian_0=profile_models.Gaussian, gaussian_1=profile_models.Gaussian, gaussian_2=profile_models.Gaussian, 
                                              gaussian_3=profile_models.Gaussian, gaussian_4=profile_models.Gaussian)
            )

def make_voigt_phase():
    return ph.Phase(
                    phase_name='phase_1_voigt', 
                    profiles=af.CollectionPriorModel(voigt_0=profile_models.Voigt)
                    )

def make_phase(mode, n_components):
    if mode == 'gaussian':
        return make_gaussian_phase(n_components)
    elif mode == 'voigt':
        return make_voigt_phase()

def find_good_fit(mode, dataset: Spectrum):

    n_initial = dataset.estimate_n()
    n_components = n_initial + 0

    phases = []
    results = []
    evidences = []

    good_fit = False
    evidences.insert(0, -1. * np.inf)

    i = 1
    while not good_fit:

        phase = make_phase(mode, n_components)
        result = phase.run(dataset=dataset)
        evidence = result.output.evidence

        # I think we need to just stop after a certain chi squared tbh

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


