````markdown
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="FastYield%20logo.png">
    <img alt="FastYield Logo" src="FastYield%20logo.png">
  </picture>
</p>

# FastYield

**FastYield** is a Python package for estimating exoplanet detection performances with high-contrast imaging combined with medium- to high-resolution spectroscopy, with a particular focus on the **molecular mapping** technique.

FastYield is an extended version of [FastCurves](https://github.com/ABidot/FastCurves). FastCurves estimates detection limits from instrumental properties such as PSF profiles, transmission, detector characteristics, background, and spectral resolution, together with planetary properties such as magnitude, temperature, surface gravity, and albedo. It can also provide retrieval-performance estimates through corner plots.

FastYield builds on this framework by applying the FastCurves formalism to archival or synthetic planet catalogs. It can therefore be used to estimate the expected **detection yield** of a given instrument, survey, or observing configuration.

The molecular mapping approach relies on stellar-halo subtraction through spectral high-pass filtering, followed by cross-correlation with planetary atmospheric templates. This makes it particularly efficient at reducing low-frequency stellar and speckle residuals while exploiting the high-frequency molecular signatures of exoplanet spectra.

For more details, see:

- [Bidot et al. 2024](https://www.aanda.org/articles/aa/pdf/2024/02/aa46185-23.pdf)
- [Martos et al. 2025](https://arxiv.org/pdf/2504.06890)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="FastYield%20example.png">
    <img alt="FastYield Example" src="FastYield%20example.png">
  </picture>
</p>

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

FastYield can be installed directly from GitHub with `pip`.

### Standard installation

Use this option if you only want to use FastYield:

```bash
python -m pip install git+https://github.com/StevMartos/FastYield.git
````

This installs FastYield and its required dependencies.

To force `pip` to upgrade FastYield and all compatible dependencies to their latest available versions, use:

```bash
python -m pip install --upgrade --upgrade-strategy eager git+https://github.com/StevMartos/FastYield.git
```

You can then test the installation with:

```bash
python -c "import fastyield; print(fastyield.__version__)"
python -c "from fastyield.FastCurves import FastCurves; print(FastCurves)"
```

### Development installation

Use this option if you want to modify the code locally:

```bash
git clone https://github.com/StevMartos/FastYield.git
cd FastYield
python -m pip install -e ".[dev]"
```

In development mode, changes made directly inside the local repository are immediately reflected in the installed package.

For example, if you modify:

```text
src/fastyield/FastCurves.py
```

then the local `fastyield` package will use the modified version.

To force an update of FastYield and all compatible dependencies during a development installation, use:

```bash
python -m pip install --upgrade --upgrade-strategy eager -e ".[dev]"
```

If you modify package metadata or dependencies in `pyproject.toml`, reinstall the package with:

```bash
python -m pip install -e ".[dev]"
```

---

## Dependencies

FastYield dependencies are automatically installed from `pyproject.toml`.

FastYield mainly relies on:

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Astropy](https://www.astropy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/)
* [tqdm](https://tqdm.github.io/)
* [Numba](https://numba.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [statsmodels](https://www.statsmodels.org/)
* [corner](https://corner.readthedocs.io/)
* [ttkwidgets](https://github.com/TkinterEP/ttkwidgets)
* [timezonefinder](https://github.com/jannikmi/timezonefinder)
* [pytz](https://pythonhosted.org/pytz/)
* [dace-query](https://dace-query.readthedocs.io/)

Development tools such as `pytest`, `ruff`, `build`, and `twine` are not required to use FastYield. They are only installed when using the development option:

```bash
python -m pip install -e ".[dev]"
```

---

## External data

FastYield requires external data for most S/N, contrast, and yield calculations.

There are two main types of data:

1. the **instrumental and simulation data** stored in `sim_data/`;
2. the **spectral library** stored in `Spectra/`.

The Python package itself contains the code. The large scientific data files should be kept locally and made available to FastYield through the appropriate paths.

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

The `Spectra` directory is not stored directly inside the Python package because it can be large. FastYield therefore needs to know where this directory is located on your machine.

There are three possible ways to define the path.

---

### Option 1 — Recommended: environment variable

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
from fastyield.config import set_spectra_path

set_spectra_path("/path/to/Spectra")
```

For example:

```python
from fastyield.config import set_spectra_path

set_spectra_path("/home/user/data/Spectra")
```

This should be done before loading any stellar, planetary, albedo, or molecular spectrum.

---

### Option 3 — Default local location

If no path is provided, FastYield will look for the `Spectra` directory in the default local location:

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
│   └── fastyield/
└── ...
```

This default layout is especially convenient when using FastYield in development mode from a local clone.

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
from fastyield.config import set_spectra_path
from fastyield.spectrum import load_planet_spectrum
from fastyield.FastYield_interface import FastYield_interface

# Set this only if you have not defined FASTYIELD_SPECTRA_PATH.
set_spectra_path("/path/to/Spectra")

# Test the spectra path by loading a planetary spectrum.
planet_spectrum = load_planet_spectrum(
    T_planet=1000,
    lg_planet=4.0,
    model="BT-Settl",
)

# Open the FastYield graphical interface.
FastYield_interface()
```

If you have already defined the environment variable `FASTYIELD_SPECTRA_PATH`, then the call to `set_spectra_path()` is not required.

---

## Development workflow

A typical development workflow is:

```bash
git clone https://github.com/StevMartos/FastYield.git
cd FastYield
python -m pip install -e ".[dev]"
```

Then test the package with:

```bash
python -c "import fastyield; print(fastyield.__version__)"
python -c "from fastyield.FastCurves import FastCurves; print(FastCurves)"
```

Run the tests with:

```bash
pytest
```

Build the package locally with:

```bash
python -m build
```

This creates distribution files in:

```text
dist/
```

---

## Adding a new instrument

If you would like to add a new instrument to FastYield, please contact:

[steven.martos@univ-grenoble-alpes.fr](mailto:steven.martos@univ-grenoble-alpes.fr)

Please include, when available:

* spectral range and spectral resolution for each band;
* total system transmission, excluding tellurics, for each band;
* representative PSF, either as a 2D image or a 3D spectral cube;
* expected background flux;
* expected read-out noise;
* expected dark current;
* effective spatial sampling;
* detector saturation limit;
* coronagraphic and/or apodizer transmission, if relevant.

---

## References

If you use FastYield or FastCurves in your work, please cite:

* Bidot et al. 2024, *FastCurves: a performance estimation tool for molecular mapping*, A&A.
* Martos et al. 2025, *Combining high-contrast imaging and high-resolution spectroscopy: MIRI/MRS on-sky results compared to expectations*, A&A.

---

## Contact

For questions, bug reports, or instrument additions, please contact:

[steven.martos@univ-grenoble-alpes.fr](mailto:steven.martos@univ-grenoble-alpes.fr)

```
```

