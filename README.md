# FastYield

Python package for exoplanet detection and performance estimation using the molecular mapping technique. FastYield is an update of [FastCurves](https://github.com/ABidot/FastCurves). FastCurves can estimate detection limits related to various instrument parameters (PSF profiles, transmission, detector characteristics, etc.) and planet characteristics (magnitude, temperature, gravity, albedo, etc.). FastCurves shows promise as an ETC for predicting IFS performance when using molecular mapping (stellar halo subtraction with spectral high-pass filtering and cross-correlation technique) as a post-processing method for efficient speckle removal. FastYield extends FastCurves by applying it to an archival or synthetic planet table to estimate the yield performance that an instrument can achieve. For more information, see [Bidot et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/02/aa46185-23.pdf) or Martos et al. (in prep.).

##instruments considered :

###Molecular mapping: 
ELT/HARMONI; ELT/ANDES; VLT/ERIS; JWST/MIRI/MRS; JWST/NIRSpec/IFU

###ADI+RDI:
JWST/NIRCam

## Download the spectra

To run SNR or contrast calculations, you will need planetary spectra (BT-Settl, Exo-REM and PICASO) and stellar spectra (BT-NextGen) [here](https://filesender.renater.fr/?s=download&token=ba931d28-dde5-4eb7-838d-32416934f072).
