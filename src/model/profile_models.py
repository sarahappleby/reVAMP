import numpy as np

import autofit as af

from astropy.modeling.models import Voigt1D


class Profile(af.ModelObject):

    def __init__(self, center, intensity):

        super(Profile, self).__init__()

        self.center = center
        self.intensity = intensity


class Gaussian(Profile):

    def __init__(self, center=0.0, intensity=0.1, sigma=0.01):

        super(Gaussian, self).__init__(center=center, intensity=intensity)
        self.sigma = sigma

    def model_from_frequencies(self, frequencies):
        return self.intensity * np.exp(
            -0.5 * ((frequencies - self.center) / self.sigma) ** 2
        )

    def absorption_line_from_frequencies(self, frequencies):
        return np.exp(-1. * self.model_from_frequencies(frequencies))


class Voigt(Profile):
    
    def __init__(self, center=0.0, intensity=0.1, fwhm_l=0.01, fwhm_g=0.01):

        super(Voigt, self).__init__(center=center, intensity=intensity)

        self.fwhm_l = fwhm_l
        self.fwhm_g = fwhm_g

    def model_from_frequencies(self, frequencies):

        v = Voigt1D(
            x_0=self.center,
            amplitude_L=self.intensity,
            fwhm_L=self.fwhm_l,
            fwhm_G=self.fwhm_g,
        )
        return v(frequencies)

    def absorption_line_from_frequencies(self, frequencies):
        return np.exp(-1.* self.model_from_frequencies(frequencies))