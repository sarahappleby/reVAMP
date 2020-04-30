import numpy as np

from vamp_src.dataset.spectrum import Spectrum
import vamp_src.model.profile_models

class Gaussian():
	def __init__(self, xarray, center=0.0, intensity=0.1, sigma=0.01):
		self.xarray = xarray
		self.center = center
		self.intensity = intensity
		self.sigma = sigma
		self.yarray = self.intensity * np.exp(-0.5 * ((self.xarray - self.center) / self.sigma) ** 2)

class SplitRegions():

	def __init__ (self, dataset : Spectrum, min_region_width=2, N_sigma=4.0, extend=False, merge=False, std_min=2, std_max=11):
		"""
		Finds detection regions above some detection threshold and minimum width.

		Args:
			min_region_width (int): minimum width of a detection region (pixels)
			N_sigma (float): detection threshold (std deviations)
			extend (boolean): default is False. Option to extend detected regions until tau
							returns to continuum.
			merge (boolean): default is False. Option to merge overlapping regions.
			std_min, std_max (int): range of standard deviations for Gaussian convolution, units of pixels
		"""		

		self.dataset = dataset
		self.min_region_width = min_region_width
		self.N_sigma = N_sigma
		self.extend = extend
		self.merge = merge
		self.std_min = std_min
		self.std_max = std_max

		self.compute_detection_regions()

	def estimate_n(self):
		n = int(argrelextrema(gaussian_filter(self.dataset.flux, 3), np.less)[0].shape[0])
		if n < 4:
			n = 1
			return n

	def convolve_varying_gaussians(self):
		# Convolve varying-width Gaussians with equivalent width of flux and noise
		# to get initial estimate for detection ratio
		xarray = np.array([p - (self.num_pixels-1)/2.0 for p in range(self.num_pixels)])

		for std in range(self.std_min, self.std_max):
			gaussian = Gaussian(xarray, center=0.0, intensity=1.0, sigma=std)
			flux_conv = np.convolve(self.flux_ews, gaussian.yarray, 'same')
			noise_conv = np.convolve(np.square(self.noise_ews), np.square(gaussian.yarray), 'same')
			noise_conv = 1./ np.sqrt(noise_conv)

			mask = flux_conv * noise_conv > self.det_ratio
			self.det_ratio[mask] = flux_conv[mask] * noise_conv[mask]

	def get_endpoints(self):
		# Find absorption regions with Nsigma and length > min_region_width
		start = 0
		self.region_endpoints = []
		for i in range(self.num_pixels):
			if (start == 0) and (self.det_ratio[i] > self.N_sigma) and (self.dataset.flux[i] < 1.0):
				start = i
			elif (start != 0) and (self.det_ratio[i] < self.N_sigma or self.dataset.flux[i] > 1.0):
				if (i - start) > self.min_region_width:
					self.region_endpoints.append([start, i])
				start = 0

	def extend_regions(self):
		self.regions_expanded = []
		for reg in self.region_endpoints:
			i = reg[0]
			while (i > 0) and (self.dataset.flux[i] < 1.):
				i -= 1
			j = reg[1] 
			while (j < self.num_pixels-1) and (self.dataset.flux[j] < 1.):
				j += 1
			self.regions_expanded.append([i, j])

	def compute_detection_regions(self):

		print('Computing detection regions...')
		self.num_pixels = len(self.dataset.wavelength)

		self.flux_ews = np.zeros(self.num_pixels)
		self.noise_ews = np.zeros(self.num_pixels)
		self.det_ratio = np.full(self.num_pixels, -1.*np.inf)

		self.flux_dec = 1.0 - self.dataset.flux
		self.flux_dec[self.flux_dec < self.dataset.noise] = 0.

		# Compute flux decrement either side of each pixel
		for i in range(1, self.num_pixels - 1):
			self.flux_ews[i] = 0.5 * abs(self.dataset.wavelength[i - 1] - self.dataset.wavelength[i + 1]) * self.flux_dec[i]
			self.noise_ews[i] = 0.5 * abs(self.dataset.wavelength[i - 1] - self.dataset.wavelength[i + 1]) * self.dataset.noise[i]

		self.convolve_varying_gaussians()

		self.get_endpoints()

		if self.extend:
			self.extend_regions()
		else:
			self.regions_expanded = self.region_endpoints

		self.region_pixels = []
		buff = 3
		for i in range(len(self.regions_expanded)):
			start = self.regions_expanded[i][0]
			end = self.regions_expanded[i][1]
			if self.merge:
				if (i < len(self.regions_expanded) -1) and (end > self.regions_expanded[i+1][0]):
					end = self.regions_expanded[i+1][1]

			for j in range(start, end):
				if self.flux_dec[j] > abs(self.dataset.noise[j]) * self.N_sigma:
					if start >= buff:
						start -= buff
					if end < self.num_pixels - buff:
						end += buff
					self.region_pixels.append([start, end])
					break

		print('Found {} detection regions.'.format(len(self.region_pixels)))

	def new_region_spectra(self):
		new_spectra = []
		for reg in self.region_pixels:
			new_spectra.append(Spectrum(self.dataset.frequency[reg[0]:reg[1]], 
										self.dataset.wavelength[reg[0]:reg[1]],
										self.dataset.flux[reg[0]:reg[1]],
										self.dataset.noise[reg[0]:reg[1]]))
		return new_spectra

	def plot_with_regions():
		# make plot with brackets indicating the selected regions
		pass

	def difficult_regions():
		pass

	def preprocessing():
		#run compute detection_regions
		#run difficult_regions

		#return list of sub regions, each is a dataset
		pass