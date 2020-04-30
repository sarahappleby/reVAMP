import numpy as np


class DatasetFit:
    # noinspection PyUnresolvedReferences
    def __init__(self, data, noise_map, model_data, dimensions):
        """Class to fit simulator where the dataset structures are any dimension.
        Parameters
        -----------
        data : ndarray
            The observed simulator that is fitted.
        noise_map : ndarray
            The noise_map-map of the observed simulator.
        model_data : ndarray
            The model simulate the fitting image is fitted with.
        dimensions : int
            The number of parameters in the model
        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (datas - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((datas - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every simulator-point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every simulator-point.
        noise_normalization : float
            The overall normalization term of the noise_map-map, summed over every simulator-point.
        likelihood : float
            The overall likelihood of the model's fit to the dataset, summed over evey simulator-point.
        """
        self.data = data
        self.noise_map = noise_map
        self.model_data = model_data
        self.dimensions = dimensions

    @property
    def residual_map(self):
        return residual_map_from_data_and_model_data(
            data=self.data, model_data=self.model_data
        )

    @property
    def normalized_residual_map(self):
        return normalized_residual_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self):
        return chi_squared_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def signal_to_noise_map(self):
        """The signal-to-noise_map of the dataset and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def chi_squared(self):
        return chi_squared_from_chi_squared_map(chi_squared_map=self.chi_squared_map)

    @property
    def reduced_chi_squared(self):
        return self.chi_squared / int(len(self.data) - self.dimensions)

    @property
    def noise_normalization(self):
        return noise_normalization_from_noise_map(noise_map=self.noise_map)

    @property
    def likelihood(self):
        return likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )


def residual_map_from_data_and_model_data(data, model_data):
    """Compute the residual map between a masked observed simulate and model simulator, where:
    Residuals = (Data - Model_Data).
    Parameters
    -----------
    data : np.ndarray
        The observed simulator that is fitted.
    mask : np.ndarray
        The mask applied to the dataset, where *False* entries are included in the calculation.
    model_data : np.ndarray
        The model simulator used to fit the observed simulator.
    """
    return np.subtract(data, model_data, out=np.zeros_like(data))


def normalized_residual_map_from_residual_map_and_noise_map(residual_map, noise_map):
    """Compute the normalized residual map between a masked observed simulate and model simulator, where:
    Normalized_Residual = (Data - Model_Data) / Noise
    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the observed simulator.
    noise_map : np.ndarray
        The noise-map of the observed simulator.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))


def chi_squared_map_from_residual_map_and_noise_map(residual_map, noise_map):
    """Computes the chi-squared map between a masked residual-map and noise-map, where:
    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
    Although noise-maps should not contain zero values, it is possible that masking leads to zeros which when \
    divided by create NaNs. Thus, nan_to_num is used to replace these entries with zeros.
    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the observed simulator.
    noise_map : np.ndarray
        The noise-map of the observed simulator.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.square(
        np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))
    )


def chi_squared_from_chi_squared_map(chi_squared_map):
    """Compute the chi-squared terms of each model's simulator-set's fit to an observed simulator-set, by summing the masked
    chi-squared map of the fit.
    Parameters
    ----------
    chi_squared_map : np.ndarray
        The chi-squared map of values of the model-simulator fit to the observed simulator.
    mask : np.ndarray
        The mask applied to the chi-squared map, where *False* entries are included in the calculation.
    """
    return np.sum(chi_squared_map)


def noise_normalization_from_noise_map(noise_map):
    """Compute the noise-map normalization terms of a list of masked 1D noise-maps, summing the noise_map vale in every
    pixel as:
    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
    Parameters
    ----------
    noise_map : np.ndarray
        The masked noise-map of the observed simulator.
    mask : np.ndarray
        The mask applied to the noise-map, where *False* entries are included in the calculation.
    """
    return np.sum(np.log(2 * np.pi * noise_map ** 2.0))


def likelihood_from_chi_squared_and_noise_normalization(
    chi_squared, noise_normalization
):
    """Compute the likelihood of each masked 1D model-simulator fit to the dataset, where:
    Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
    Parameters
    ----------
    chi_squared : float
        The chi-squared term for the model-simulator fit to the observed simulator.
    noise_normalization : float
        The normalization noise_map-term for the observed simulator's noise-map.
    """
    return -0.5 * (chi_squared + noise_normalization)
