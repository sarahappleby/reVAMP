import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema
import h5py

class Spectrum:
    def __init__(self, frequency, wavelength, flux, noise):

        self.frequency = np.array(frequency)
        self.wavelength = np.array(wavelength)
        self.flux = np.array(flux)
        self.noise = np.array(noise)

    def spectrum_region_from_boundary(self,i_start, i_end):
      return Spectrum(self.frequency[i_start:i_end], self.wavelength[i_start:i_end], 
                      self.flux[i_start:i_end], self.noise[i_start:i_end])

    def save_as_h5py(self, filename, attributes=None):

        with h5py.File(filename, 'a') as f:
            
            f.create_dataset('wavelength', data=np.array(self.wavelength))
            f.create_dataset('frequency', data=np.array(self.frequency))
            f.create_dataset('flux', data=np.array(self.flux))
            f.create_dataset('noise', data=np.array(self.noise))

            if attributes:
                for k in list(attributes.keys()):
                    f[k] = attributes[k]

    def plot_spectrum(self, filename):
        import matplotlib.pyplot as plt
        plt.plot(self.wavelength, self.flux, c='k')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.ylim(-0.1, 1.1)
        plt.savefig(filename)
        plt.clf()


def read_from_h5py(filename):

    with h5py.File(filename, 'r') as f:
        wavelength = f['wavelength'][:]
        frequency = f['frequency'][:]
        flux = f['flux'][:]
        noise = f['noise'][:]

    return Spectrum(frequency, wavelength, flux, noise)