import autofit as af


class Result(af.Result):

    def __init__(self, samples, previous_model, analysis):
        """
        The result of a non-linear search.
        """
        super().__init__(samples=samples, previous_model=previous_model)
        self.analysis = analysis

    @property
    def most_likely_model_spectrum(self):
        return self.analysis.model_spectrum_from_instance(instance=self.instance)

    @property
    def most_likely_fit(self):
        return self.analysis.fit_from_model_spectrum(
            model_spectrum=self.most_likely_model_spectrum
        )