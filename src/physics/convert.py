def Tau2flux(tau):
    """
    Convert optical depth to normalised flux profile.

    Args:
        tau (numpy array): array of optical depth values
    """
    return np.exp(-tau)

def Flux2tau(flux):
    """
    Convert normalised flux to optical depth.

    Args:
        flux (numpy array): array of fluxes
    """
    return -1*np.log(flux)

def Freq2wave(frequency):
    """
    Convert frequency to wavelength in Angstroms
    """
    return (constants['c']['value'] / frequency) / 1.e-10

def Wave2freq(wavelength):
    """
    Convert wavelength in Angstroms to frequency
    """
    return constants['c']['value'] / (wavelength*1.e-10)

def Wave2red(wave, rest_wave):
    """
    Args:
        wave (numpy array): array of wavelengths
        wave_rest (float): rest wavelength
    """
    return (wave - rest_wave) / rest_wave
