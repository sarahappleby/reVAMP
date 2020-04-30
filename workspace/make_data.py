import numpy as np
from astropy.modeling.models import Voigt1D

class FakeData():
    def __init__(self, center, intensity, x_min, x_max, n_points, snr):
        self.center = center
        self.intensity = intensity
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.snr = snr

        # make the x array
        self.dx = np.abs(self.x_max - self.x_min) / self.n_points
        self.x = np.arange(self.x_min, self.x_max, self.dx)

        # generate noise
        self.noise = np.random.normal(0.0, 1.0 / self.snr, self.n_points)

class FakeGauss(FakeData):
    def __init__(self, center=0.0, intensity=1.0, sigma=1.0, x_min=-5.0, x_max=5.0, n_points=100, snr=30,):

        super(FakeGauss, self).__init__(center=center, intensity=intensity, x_min=x_min, x_max=x_max, n_points=n_points, snr=snr)
        self.sigma = sigma

        # make the gaussian
        self.gauss = self.intensity * np.exp(
            -0.5 * ((self.x - self.center) / self.sigma) ** 2
        )

        self.noisy_gauss = self.gauss + self.noise

class FakeVoigt(FakeData):
    def __init__(self, center=0.0, intensity=1.0, fwhm_L=1.0, fwhm_G=1.0, x_min=-5.0, x_max=5.0, n_points = 100, snr=30):

        super(FakeVoigt, self).__init__(center=center, intensity=intensity, x_min=x_min, x_max=x_max, n_points=n_points, snr=snr)
        self.fwhm_L = fwhm_L
        self.fwhm_G = fwhm_G

        v = Voigt1D(
            x_0=self.center,
            amplitude_L=self.intensity,
            fwhm_L=self.fwhm_L,
            fwhm_G=self.fwhm_G,
        )

        self.voigt = v(self.x)
        self.noisy_voigt = self.voigt + self.noise