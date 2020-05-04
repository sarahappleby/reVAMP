#TODO: find some handling for (a) damped absorbers (b) spectra which need flux to be rescaled

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema

from src.dataset.spectrum import Spectrum
import src.model.profile_models

class Gaussian():
	def __init__(self, xarray, center=0.0, intensity=0.1, sigma=0.01):
		self.xarray = xarray
		self.center = center
		self.intensity = intensity
		self.sigma = sigma
		self.yarray = self.intensity * np.exp(-0.5 * ((self.xarray - self.center) / self.sigma) ** 2)

class SplitRegions():

	def __init__ (self, dataset : Spectrum, min_region_width=2, N_sigma=4.0, std_min=2, std_max=11, extend=False, merge=False,
				  max_single_region_components=15, ideal_single_region_components=5, min_region_percentage=2.):
		"""
		Find absorption regions above some detection threshold and minimum width.

		Args:
			min_region_width (int): 		Minimum width of an absorption region (pixels)
			N_sigma (float): 				Detection threshold (std deviations)
			std_min, std_max (int): 		Range of standard deviations for Gaussian convolution, units of pixels
			extend (boolean): 				Default is False. Option to extend detected regions until tau
											returns to continuum.
			merge (boolean): 				Default is False. Option to merge overlapping regions.
			max_single_region_components:	Maximum number of components a single region should have. If a region
											has more than this number of components, it will be split up.
			ideal_single_region_components:	The number of regions to group into a single region if max_single_region_components
											is reached.
			min_region_percentage:			Minimum % of pixels in a spectra a region must take up (when forcing split regions)
		"""		

		self.dataset = dataset
		self.min_region_width = min_region_width
		self.N_sigma = N_sigma
		self.std_min = std_min
		self.std_max = std_max
		self.extend = extend
		self.merge = merge
		self.max_single_region_components = max_single_region_components
		self.ideal_single_region_components = ideal_single_region_components
		self.min_region_percentage = min_region_percentage

		self.compute_detection_regions()
		self.split_difficult_region()

	def estimate_ncomp(self):
		"""
		Estimate the initial number of gaussians to fit to region by smoothing with a gaussian filter
		"""
		self.ncomp = int(argrelextrema(gaussian_filter(self.dataset.flux, 3), np.less)[0].shape[0])
		if self.ncomp < 4:
			self.ncomp = 1
			return self.ncomp

	def convolve_varying_gaussians(self):
		"""
		Convolve varying-width Gaussians with equivalent width of flux and noise
		to get initial estimate for detection ratio
		"""
		xarray = np.array([p - (self.num_pixels-1)/2.0 for p in range(self.num_pixels)])

		for std in range(self.std_min, self.std_max):
			gaussian = Gaussian(xarray, center=0.0, intensity=1.0, sigma=std)
			flux_conv = np.convolve(self.flux_ews, gaussian.yarray, 'same')
			noise_conv = np.convolve(np.square(self.noise_ews), np.square(gaussian.yarray), 'same')
			noise_conv = 1./ np.sqrt(noise_conv)

			mask = flux_conv * noise_conv > self.det_ratio
			self.det_ratio[mask] = flux_conv[mask] * noise_conv[mask]

	def get_endpoints(self):
		"""
		Find absorption regions with significance > Nsigma, and length > min_region_width in pixels
		"""
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
		"""
		Extend identified regions up to the continuum level
		"""
		self.regions_expanded = []
		for reg in self.region_endpoints:
			i = reg[0]
			while (i > 0) and (self.dataset.flux[i] < 1.):
				i -= 1
			j = reg[1] 
			while (j < self.num_pixels-1) and (self.dataset.flux[j] < 1.):
				j += 1
			self.regions_expanded.append([i, j])

	def new_region_spectra(self):
		"""
		Return new spectrum object for each region
		"""
		self.region_spectra = []
		for reg in self.region_pixels:
			self.region_spectra.append(Spectrum(self.dataset.frequency[reg[0]:reg[1]], 
												self.dataset.wavelength[reg[0]:reg[1]],
												self.dataset.flux[reg[0]:reg[1]],
												self.dataset.noise[reg[0]:reg[1]]))

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
		self.new_region_spectra()

	def split_difficult_region(self):
		"""
		Check whether fit is going to be difficult, i.e. 1 region with way too many components
		TODO: find some handling for (a) damped absorbers (b) spectra which need flux to be rescaled
		"""
		self.difficult_fit = False
		
		if len(self.region_pixels) == 1:
			start = self.region_pixels[0][0]
			end = self.region_pixels[0][1]

			self.region_spectra[0].estimate_ncomp()

			if self.region_spectra[0].ncomp > self.max_single_region_components:

				# flag the region as being difficult
				self.difficult_fit = True
				print(str(self.region_spectra[0].ncomp) + " components; should be in more than 1 region!")

				self.forced_number_regions = self.region_spectra[0].ncomp // self.ideal_single_region_components
				print("Trying to force-split into " + str(self.forced_number_regions) + " regions.")
				ind = np.argpartition(self.region_spectra[0].flux, -10*self.forced_number_regions)[-10*self.forced_number_regions:]
				self.ind_sorted = np.flip(ind[np.argsort(self.region_spectra[0].flux[ind])], axis=0)
				print(str(len(self.ind_sorted)) + " possible split points to choose from")

				print("There are: " + str(len(self.region_spectra[0].flux)) + " pixels.")
				self.min_region_size = len(self.region_spectra[0].flux) * (self.min_region_percentage / 100.0)
				print("Minimum region pixels: " + str(self.min_region_size))

				self.splitting_points = [start, end]
				# original "start" is the beginning of the 1st region
				# original "end" is the end of the last region
				# each region will be contiguous, so need (forced_number_regions - 1) indexes
				# these indexes will have to be at least <min_region_size> away from each other.

				for i in range(len(self.ind_sorted)):
					# go through indices of maximum flux, in descending order
					# see if they can be the required distance away from each other.
					# if they can't be made to work, try working down the list of maximum fluxes

					if (len(self.splitting_points) == (self.forced_number_regions+1)):
						print("Found enough splitting regions")
						break #stop once enough points to split have been found

					else:
						dist_is_fine = True
						for j in range(len(self.splitting_points)):
							# check the distance between a possible "splitting point" and the existing splitting points
							dist = abs(self.ind_sorted[i] - self.splitting_points[j])
							if (dist < self.min_region_size):
								dist_is_fine = False

						if dist_is_fine: 
							#if the region would be large enough, then split along this point
							self.splitting_points.append(self.ind_sorted[i])

				print("Have managed to split into " + str(len(self.splitting_points)-1) + " regions!")
				self.splitting_points.sort() #sort the pixels into ascending order
				self.region_pixels = []

				for i in range(len(self.splitting_points)-1):
					start = self.splitting_points[i]
					end = self.splitting_points[i+1]
					self.region_pixels.append([start,end])

		def plot_regions(self):
			"""
			Plot the absorption regions
			"""

			def plot_bracket(x, axis, dir):
				height = .2
				arm_length = 0.1
				axis.plot((x, x), (1-height/2, 1+height/2), color='magenta')

				if dir=='left':
					xarm = x+arm_length
				if dir=='right':
					xarm = x-arm_length

				axis.plot((x, xarm), (1-height/2, 1-height/2), color='magenta')
				axis.plot((x, xarm), (1+height/2, 1+height/2), color='magenta')



