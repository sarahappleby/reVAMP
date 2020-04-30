class Measurement():
	def __init__(self, data, error):
		self.data = data
		self.error = error

class AbsorptionFeature():

	def __init__(self, wavelength : Measurement, frequency : Measurement, equivalent_width : Measurement, 
				 column_density : Measurement, doppler_parameter : Measurement):

		self.wavelength = wavelength
		self.frequency = frequency
		self.equivalent_width = equivalent_width
		self.column_density = column_density
		self.doppler_parameter = doppler_parameter

class TotalAbsorption(): # for whole spectrum

	def init(self)

	get  list of doppler parameters
		e.g.	i.doppler for i in features (w/ error propogation)
	get total column density with error
		e.g. sum (i.density for i in features)
	get total equivalent width with error
		e.g. sum (i.ew for i in features)

