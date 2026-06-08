<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="FastYield logo.png">
  <img alt="FastYield Logo" src="FastYield logo.png">
</picture>

# FastYield

**FastYield** is a Python package for exoplanet detection and performance estimation using high-contrast imaging combined with medium- to high-resolution spectroscopy, with a particular focus on the **molecular mapping** technique.

FastYield is an extended version of [FastCurves](https://github.com/ABidot/FastCurves). FastCurves estimates detection limits from instrumental parameters such as PSF profiles, transmission, detector characteristics, background, and spectral resolution, together with planetary properties such as magnitude, temperature, surface gravity, and albedo. It can also provide retrieval-performance estimates through corner plots.

FastYield builds on this framework by applying the FastCurves formalism to archival or synthetic planet catalogs, making it possible to estimate the expected **detection yield** of a given instrument or observing configuration.

The molecular mapping approach relies on stellar-halo subtraction through spectral high-pass filtering, followed by cross-correlation with planetary atmospheric templates. This makes it particularly efficient at reducing low-frequency stellar and speckle residuals while exploiting the high-frequency molecular signatures of exoplanet spectra.

For more details, see [Bidot et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/02/aa46185-23.pdf) and [Martos et al. 2025](https://arxiv.org/pdf/2504.06890).

<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="FastYield example.png">
  <img alt="FastYield Example" src="FastYield example.png">
</picture>

---

## Instruments currently included

FastYield currently includes configurations for the following instruments:

- **ELT/HARMONI**
- **ELT/ANDES**
- **VLT/ERIS**
- **VLT/HiRISE**
- **VLT/CRIRES+**
- **JWST/MIRI-MRS**
- **JWST/NIRSpec IFU**
- **JWST/NIRCam**
- **OHP/VIPAPYRUS**

---

## Installation

### Clone the repository

```bash
git clone https://github.com/StevMartos/FastYield.git
cd FastYield
```

### Install the required Python packages

FastYield requires several standard scientific Python packages:

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Astropy](https://www.astropy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numba](https://numba.pydata.org/)
- [ttkwidgets](https://github.com/TkinterEP/ttkwidgets)

They can be installed with:

```bash
pip install numpy scipy matplotlib astropy pandas numba ttkwidgets
```

---

## Spectral library

FastYield requires a local spectral library to perform S/N, contrast, and yield calculations. This library contains the stellar, planetary, molecular, and albedo spectra used by the package.

The spectral library can be downloaded here:

[Download the Spectra directory](https://filesender.renater.fr/?s=download&token=938ee99e-dbf6-4a05-93b8-136f733c2712)

After downloading, you should have a directory named:

```text
Spectra/
```

This directory should contain subdirectories such as:

```text
Spectra/
├── planet_spectrum/
├── star_spectrum/
└── molecular/
```

---

## Setting the Spectra path

The `Spectra` directory is not stored directly inside the GitHub repository because it can be large. FastYield therefore needs to know where this directory is located on your machine.

There are three possible ways to define the path.

---

### Option 1 — Recommended: use an environment variable

This is the recommended solution if you want FastYield to automatically find your `Spectra` directory every time you use it.

On Linux or macOS, add the following line to your `~/.bashrc`, `~/.zshrc`, or equivalent shell configuration file:

```bash
export FASTYIELD_SPECTRA_PATH="/path/to/Spectra"
```

For example:

```bash
export FASTYIELD_SPECTRA_PATH="/home/user/data/Spectra"
```

Then reload your shell configuration:

```bash
source ~/.bashrc
```

You can check that the path has been correctly defined with:

```bash
echo $FASTYIELD_SPECTRA_PATH
```

---

### Option 2 — Set the path from Python

You can also define the path directly in your Python script:

```python
from src.config import set_spectra_path

# Required: set the path to your local Spectra directory.
set_spectra_path("/path/to/Spectra")
```

For example:

```python
from src.config import set_spectra_path

set_spectra_path("/home/user/data/Spectra")
```

This should be done before loading any stellar, planetary, albedo, or molecular spectrum.

---

### Option 3 — Default local location

If no path is provided, FastYield will look for the `Spectra` directory in the default location:

```text
FastYield/sim_data/Spectra
```

Therefore, you may also place the downloaded `Spectra` directory directly inside:

```text
FastYield/sim_data/
```

so that the final structure is:

```text
FastYield/
├── sim_data/
│   └── Spectra/
│       ├── planet_spectrum/
│       ├── star_spectrum/
│       └── molecular/
├── src/
└── ...
```

---

## Path priority

FastYield searches for the `Spectra` directory in the following order:

1. the path defined with `set_spectra_path("/path/to/Spectra")`;
2. the `FASTYIELD_SPECTRA_PATH` environment variable;
3. the default local path `sim_data/Spectra`.

This means that users can either keep the spectral library inside the repository or store it elsewhere on their machine.

---

## Quick start

Example of a minimal Python setup:

```python
from src.config import set_spectra_path
from src.spectrum import load_planet_spectrum
from src.FastYield_interface import FastYield_interface

# Set this only if you have not defined FASTYIELD_SPECTRA_PATH.
set_spectra_path("/path/to/Spectra")

# To test the spectra path
planet_spectrum = load_planet_spectrum(T_planet=1000, lg_planet=4.0, model="BT-Settl")

# To open the FastYield GUI
FastYield_interface()

```

If you have already defined the environment variable `FASTYIELD_SPECTRA_PATH`, then the call to `set_spectra_path()` is not required.

---

## Adding a new instrument

If you would like to add a new instrument to FastYield, please contact:

[steven.martos@univ-grenoble-alpes.fr](mailto:steven.martos@univ-grenoble-alpes.fr)

Please include, when available:

- spectral range and spectral resolution for each band;
- total system transmission (without tellurics) for each band;
- representative PSF, either as a 2D image or a 3D spectral cube;
- expected background flux;
- expected read-out noise;
- expected dark current;
- effective spatial sampling;
- detector saturation limit;
- coronagraphic and/or apodizer transmission, if relevant.

---

## References

If you use FastYield or FastCurves in your work, please cite:

- Bidot et al. 2024, *FastCurves: a performance estimation tool for molecular mapping*, A&A.
- Martos et al. 2025, *Combining high-contrast imaging and high-resolution spectroscopy: MIRI/MRS on-sky results compared to expectations*.

---

## Contact

For questions, bug reports, or instrument additions, please contact:

[steven.martos@univ-grenoble-alpes.fr](mailto:steven.martos@univ-grenoble-alpes.fr)
