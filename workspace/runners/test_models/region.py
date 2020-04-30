from vamp_src.dataset.spectrum import Spectrum

class Region():

    def __init__(self, frequency, wavelength, flux, noise, voigt, chi_limit=2.5):
        self.frequency = frequency
        self.wavelength = wavelength
        self.flux = flux
        self.noise = noise
        self.num_pixels = len(frequency)
        self.voigt = voigt

        self.dataset = Spectrum(self.frequency, self.wavelength, self.flux, self.noise)

        """
        Make initial guess for number of local minima in the region.
            
        Smooth the spectra with a gaussian and find the number of local minima.
        as a safety precaucion, set the initial guess for number of profiles to 1 if
        there are less than 4 local minima.
        """
        self.initial_n = argrelextrema(gaussian_filter(self.dataset.flux, 3), np.less)[0].shape[0]
        if self.initial_n < 4:
            self.initial_n = 1.

        self.n = self.initial_n.copy()

    def make_phase(self):
        if voigt:
            return self.make_voigt_phase()
        else:
            return self.make_gaussian_phase()

    def do_region_fit(self):
        phases = []
        results = []
        good_fit = False
        while not good_fit:
            phase = self.make_phase()
            result = phase.run(dataset=self.dataset)
            phases.append(phase)
            results.append(result)
            ### Evaluate the result
            if test:
                good_fit = True
            else:
                self.n += 1



    """
    1) Make a dataset of the region
    2) try an n dimensional gaussian
        generalise the af.CollectionPriorModel for a list of gaussians
    3) make the phase and run
    4) check the chi-squared of the fit - is it lower than the limit?
        if yes, stop
        if no, add on a line
            5) do the phase again

    5) Now the chi squared is lower than the limit. We can still add on a line if:
        the chi squared is lowered
        and the bayesian evidence is increased by some factor
    """