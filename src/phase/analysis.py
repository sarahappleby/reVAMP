import autofit as af
from src.dataset.spectrum import Spectrum
from src.fit.fit import DatasetFit
from src.phase import visualizer 
import numpy as np


class Analysis(af.Analysis):
    def __init__(self, dataset: Spectrum, dimensions, visualize_path=None):

        self.dataset = dataset
        self.dimensions = dimensions
        self.visualizer = visualizer.Visualizer(
            dataset=self.dataset, visualize_path=visualize_path
        )

        self.num_pixels = len(self.dataset.flux)

    def log_likelihood_function(self, instance):
        model_spectrum = self.model_spectrum_from_instance(instance=instance)
        spec_fit = self.fit_from_model_spectrum(model_spectrum=model_spectrum)
        return spec_fit.likelihood

    def model_spectrum_from_instance(self, instance):
        return np.exp(
            -1. * sum(
                    list(
                        map(
                            lambda profile: profile.model_from_frequencies(
                                self.dataset.frequency
                            ),
                            instance.profiles,
                        )
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

        components = [p.absorption_line_from_frequencies(self.dataset.frequency) for p in instance.profiles]

        model_spectrum = self.model_spectrum_from_instance(instance=instance)
        reduced_chi_squared = self.get_reduced_chi_squared(model_spectrum=model_spectrum)

        self.visualizer.visualize_fit(fit=model_spectrum,
                                      components=components,
                                      reduced_chi_squared=reduced_chi_squared, 
                                      during_analysis=during_analysis)
