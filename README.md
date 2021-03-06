# reVAMP
Voigt Automatic Multinest Profiles

reVAMP fits absorption spectral features with Voigt and Gaussian profiles and finds the optimal number of profiles to fit.

reVAMP is the new version of VAMP:
https://github.com/sarahappleby/VAMP/
and continues where 'vamp_2.0' left off.

## Setup

To use reVAMP, you will need autofit, autoarray and autoconf:

https://pypi.org/project/autofit/

https://pypi.org/project/autoarray/

https://www.gnu.org/software/autoconf/

It is recommended that you set up a conda environment for using autofit. Once you have a conda environment for autofit, see workspace/setup_autofit.sh to set up your paths.

## To do:

- Determine the Multinest parameters that will fit features correctly every time.
- Optimize this for speed.
- Fix setup.sh script such that it works from the command line.
- Put together notebooks with examples.
- Find proper handling of saturated lines.
- Find some handling for (a) damped absorbers (b) spectra which need flux to be rescaled

