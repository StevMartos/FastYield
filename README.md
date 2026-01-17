<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="FastYield logo.png">
  <img alt="FastYield Logo" src="FastYield logo.png">
</picture>

# FastYield

Python package for exoplanet detection and performance estimation using the molecular mapping technique. **FastYield** is an updated version of [FastCurves](https://github.com/ABidot/FastCurves), designed to extend its capabilities. **FastCurves** estimates detection limits based on various instrument parameters (PSF profiles, transmission, detector characteristics, etc.) and planetary properties (magnitude, temperature, gravity, albedo, etc.). It also provides retrieval performance estimates through corner plots. 

FastCurves serves as a promising exposure time calculator (ETC) for predicting the performance of integral field spectrographs (IFS) when molecular mapping is used as a post-processing technique. This method, which involves stellar halo subtraction through spectral high-pass filtering and cross-correlation, efficiently removes speckles. **FastYield** builds upon FastCurves by applying it to archival or synthetic planet catalogs, allowing for yield performance estimations for a given instrument. 

For more details, see [Bidot et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/02/aa46185-23.pdf) and [Martos et al. 2025](https://arxiv.org/pdf/2504.06890).

<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="FastYield example.png">
  <img alt="FastYield Example" src="FastYield example.png">
</picture>

## Instruments considered:

* ELT/HARMONI      (molecular mapping)
* ELT/ANDES        (molecular mapping)
* VLT/ERIS         (molecular mapping)
* VLT/HiRISE       (molecular mapping)
* JWST/MIRI/MRS    (molecular mapping)
* JWST/NIRSpec/IFU (molecular mapping)
* JWST/NIRCam      (ADI+RDI)
* OHP/VIPAPYRUS    (molecular mapping)

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

### Clone the repository:

```
git clone https://github.com/StevMartos/FastYield.git
```

### Download the spectra:

To perform SNR or contrast calculations, you'll need planetary spectra (BT-Settl, Exo-REM and PICASO) and stellar spectra (BT-NextGen) downloadable [here](https://filesender.renater.fr/?s=download&token=d57de379-db5a-4a7c-9faf-9be861689f3d). Once downloaded, just put the "Spectra" file in "sim_data".

### Download the packages:

You will also need the following packages:

* [astropy](https://github.com/astropy/astropy)
```
pip install astropy
```
* [pandas](https://github.com/pandas-dev/pandas)
```
pip install pandas
```
* [ttkwidgets](https://github.com/TkinterEP/ttkwidgets)
```
pip install ttkwidgets
```
