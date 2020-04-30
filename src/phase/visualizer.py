import matplotlib.pyplot as plt

class Visualizer(): # do i need the abstract class?
	
	def __init__(self, dataset, visualize_path):

		self.dataset = dataset
		self.visualize_path = visualize_path

	def plot_fit(self, fit, n_components, reduced_chi_squared):
		plt.plot(self.dataset.frequency, self.dataset.flux, c='k', label='Data')
		plt.plot(self.dataset.frequency, fit, c='b', ls='--', label='fit, %g components' % n_components,)
		plt.legend(loc=3)
		plt.xlabel('Frequency')
		plt.ylabel('Flux')
		plt.ylim(-0.1, 1.1)
		plt.title('reduced Chi squared: %.2f' % round(reduced_chi_squared, 2))
		plt.savefig(self.visualize_path + 'fit.png')
		plt.clf()

	def plot_residuals(self, fit, reduced_chi_squared):
		residual = self.dataset.flux - fit
		plt.plot(self.dataset.frequency, residual, c='b')
		plt.xlabel('Frequency')
		plt.ylabel('Residual')
		plt.title('reduced Chi squared: %.2f' % round(reduced_chi_squared, 2))
		plt.savefig(self.visualize_path + 'residual.png')
		plt.clf()

	def visualize_fit(self, fit, n_components, reduced_chi_squared, during_analysis):

		self.plot_fit(fit, n_components, reduced_chi_squared)
		self.plot_residuals(fit, reduced_chi_squared)
