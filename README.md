# reVAMP
Voigt Automatic Multinest Profiles

reVAMP is the new version of VAMP:
https://github.com/sarahappleby/VAMP/
and continues where 'vamp_2.0' left off.

reVAMP fits absorption spectral features with Voigt and Gaussian profiles and finds the optimal number of profiles to fit.

# Setup

To use reVAMP, you will need autofit and autoarray:
https://pypi.org/project/autofit/
https://pypi.org/project/autoarray/

It is recommended that you set up a conda environment for using autofit. Once you have a conda enviornment for autofit, see workspace/setup_autofit.sh

# To do:

- Determine the Multinest parameters that will fit features correctly every time.
- Optimize this for speed.
- Fix setup.sh script such that it works from the command line.
- Put together notebooks with examples.
- Find proper handling of saturated lines.
- Find some handling for (a) damped absorbers (b) spectra which need flux to be rescaled

