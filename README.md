<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="FastYield logo.png">
  <img alt="FastYield Logo" src="FastYield logo.png">
</picture>

# FastYield

Python package for exoplanet detection and performance estimation using the molecular mapping technique. FastYield is an update of [FastCurves](https://github.com/ABidot/FastCurves). FastCurves can estimate detection limits related to various instrument parameters (PSF profiles, transmission, detector characteristics, etc.) and planet characteristics (magnitude, temperature, gravity, albedo, etc.). FastCurves shows promise as an ETC for predicting IFS performance when using molecular mapping (stellar halo subtraction with spectral high-pass filtering and cross-correlation technique) as a post-processing method for efficient speckle removal. FastYield extends FastCurves by applying it to an archival or synthetic planet table to estimate the yield performance that an instrument can achieve. For more information, see [Bidot et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/02/aa46185-23.pdf) or Martos et al. (in prep.).

<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="FastYield example.png">
  <img alt="FastYield Example" src="FastYield example.png">
</picture>

## Instruments considered:

* ELT/HARMONI (molecular mapping)
* ELT/ANDES (molecular mapping)
* VLT/ERIS (molecular mapping)
* JWST/MIRI/MRS (molecular mapping)
* JWST/NIRSpec/IFU (molecular mapping)
* JWST/NIRCam (ADI+RDI)

## Add an instrument:

If you would like to add your instruments, please send an e-mail to [steven.martos@univ-grenoble-alpes.fr](steven.martos@univ-grenoble-alpes.fr) with the following data:

* spectral range and resolution for each band
* total system transmission for each band
* representative PSF (2D image or 3D cube) for each band 
* expected background flux
* expected read-out noise (in electron/pixel)
* dark current (in electron/s/pixel)
* effective spatial resolution (in arcsec/pixel)

## Prerequisites:

### Download the spectra:

To perform SNR or contrast calculations, you'll need planetary spectra (BT-Settl, Exo-REM and PICASO) and stellar spectra (BT-NextGen) downloadable [here](https://filesender.renater.fr/?s=download&token=04938e89-9c2e-49b4-a1c8-f1a2a717e982). Once downloaded, just put the "Spectra" file in "sim_data".

### Download the packages:

You will also need the following packages:

* [PyAstronomy](https://github.com/sczesla/PyAstronomy)
```
pip install PyAstronomy
```
* [astropy](https://github.com/astropy/astropy)
```
pip install astropy
```
* [pandas](https://github.com/pandas-dev/pandas)
```
pip install pandas
```
* [coronagraph](https://github.com/jlustigy/coronagraph)
```
pip install coronagraph
```
* [statsmodels](https://github.com/statsmodels/statsmodels)
```
pip install statsmodels
```
* [ttkwidgets](https://github.com/TkinterEP/ttkwidgets)
```
pip install ttkwidgets
```
* [pyvo](https://github.com/astropy/pyvo)
```
pip install pyvo
```
* [pyvo](https://github.com/astropy/pyvo)
```
pip install pyvo
```
