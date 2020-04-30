import autofit as af
from vamp_src.dataset.spectrum import Spectrum
from vamp_src.fit.fit import DatasetFit
from vamp_src.phase import visualizer 
import numpy as np


class Analysis(af.Analysis):
    def __init__(self, dataset: Spectrum, dimensions, visualize_path=None):

        self.dataset = dataset
        self.dimensions = dimensions
        self.visualizer = visualizer.Visualizer(
            dataset=self.dataset, visualize_path=visualize_path
        )

        self.num_pixels = len(self.dataset.flux)

    def fit(self, instance):
        model_spectrum = self.model_spectrum_from_instance(instance=instance)
        spec_fit = self.fit_from_model_spectrum(model_spectrum=model_spectrum)
        return spec_fit.likelihood

    def model_spectrum_from_instance(self, instance):
        return sum(
            list(
                map(
                    lambda profile: profile.model_from_frequencies(
                        self.dataset.frequency
                    ),
                    instance.profiles,
                )
            )
        )

    def fit_from_model_spectrum(self, model_spectrum):
        return DatasetFit(
            data=self.dataset.flux,
            noise_map=self.dataset.noise,
            model_data=model_spectrum,
            dimensions=self.dimensions
        )

    def get_reduced_chi_squared(self, model_spectrum):
        fit = self.fit_from_model_spectrum(model_spectrum)
        return fit.reduced_chi_squared


    def visualize(self, instance, during_analysis):

        # Visualization will be covered in tutorial 4.

        n_components = len(instance.profiles)
        model_spectrum = self.model_spectrum_from_instance(instance=instance)
        reduced_chi_squared = self.get_reduced_chi_squared(model_spectrum=model_spectrum)

        self.visualizer.visualize_fit(fit=model_spectrum, n_components=n_components, 
                                      reduced_chi_squared=reduced_chi_squared, 
                                      during_analysis=during_analysis)
