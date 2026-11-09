# import FastYield modules
from fastyield.config import rad2arcsec, get_sim_data_path
from fastyield.utils import box_convolution, annular_mask, interp_extrap_sep, crop, fill_invalid_by_nearest, fill_invalid_by_symmetry, rebin_flux_conserving, plot_profiles, add_if_necessary

# import astropy modules
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift

# import other modules
from pathlib import Path
from tqdm import tqdm
import re
from collections import Counter
from itertools import product


#
# Parameters
#
instru      = "HWO"
coronagraph = "APLC"
apodizer    = "APLC"
strehl      = "NO_ABERRATION"
model_PSF   = "data"
sigfactor   = None
debug       = False

# Output convention:
#   - the raw APLC cubes are measured in lambda/D units and have no physical
#     wavelength dependence;
#   - the three PSF quantities are first computed on a dimensionless
#     separation_lD axis: (C, IWA_lD, separation_lD);
#   - they are then projected onto the final FastYield-like angular axis
#     separation_mas for each wavelength:
#
#         separation_lD(lambda) = separation_mas / (lambda/D in mas).
#
sep_unit = "mas"

# Approximate FWHM-sized square aperture used for the core fraction, we assume a pxscale such that we have size_core pixel per FWHM for each wavelength
size_core = 3 # [px/FWHM]

# Telescope diameter used only for the lambda/D -> mas conversion.
# Change this value if your HWO configuration uses another aperture diameter.
D = 6.0  # [m]

# Data location.  The script is meant to be launched from FastCurves-main.
instru_dir = get_sim_data_path() / instru
data_dir   = Path("/home/martoss/Documents/PhD/work/MAIN/data/HWO/HWO APLC cubes")
outdir     = instru_dir / "PSF_inputs"
outdir.mkdir(parents=True, exist_ok=True)

# Raw HWO/APLC cube convention from Lucie's email:
#   - FITS cube shape: (NbOff, NbLine, NbColumn)
#   - cropped PSF cube: 102 x 102 pixels
#   - field of view:    50 lambda/D
#   - source shifts:    0.1 lambda/D
FOV_lD        = 50.0   # [lambda/D], full simulated field of view
shift_step_lD = 0.1    # [lambda/D], offset sampling in the raw cubes
FWHM_lD       = 1.029  # [lambda/D], approximate diffraction-limited FWHM

# Physical wavelength and final angular-separation axes.
lmin    = 0.1    # [um]
lmax    = 3.0    # [um]
sep_max = 1_000  # [mas]
N_wave  = 40
N_sep   = 400

wave           = np.linspace(lmin,        lmax,              num=N_wave) # [µm]
separation_mas = np.logspace(np.log10(1), np.log10(sep_max), num=N_sep)  # [mas]
separation_mas = add_if_necessary(array=separation_mas, value=0.0)       # [mas]

# Internal lambda/D separation axis.
# It must cover the largest angular separation at the shortest wavelength.
# The raw APLC cubes do not necessarily reach such large separations, so the
# outer part is extrapolated exactly as in the ANDES/FastYield workflow.
sep_lD_max    = sep_max / (lmin * 1e-6 / D * 1000.0 * rad2arcsec)
separation_lD = np.logspace(np.log10(1e-1), np.log10(sep_lD_max), num=N_sep)
separation_lD = add_if_necessary(array=separation_lD, value=0.0)

# Internal options
order_PSF           = 1      # Best value for sub-pixel shifts
PLOT_EACH_CUBE      = True
PLOT_FINAL_PRODUCTS = True

# Numerical safety for log/logit interpolation
EPS    = 1e-32
safe01 = lambda x: np.clip(x, EPS, 1.0 - EPS)



def lambda_over_D_mas(wavelength_um, diameter_m):
    """
    Convert one lambda/D to mas.

    Parameters
    ----------
    wavelength_um : float or ndarray
        Wavelength [um].
    diameter_m : float
        Telescope diameter [m].

    Returns
    -------
    lD_mas : float or ndarray
        Angular size of lambda/D [mas].
    """
    return np.asarray(wavelength_um, dtype=float) * 1e-6 / diameter_m * 1000.0 * rad2arcsec



# %%
# APLC off-axis apodizer transmission table
#
# The values below are the off-axis, fully-separated apodizer transmissions from
# APLC_survey_transmission.pdf.  They are used only to set the asymptotic radial
# transmission of each raw cube.  The radial shape itself is still measured from
# the spatial sum of each shifted PSF cube, exactly as in the ANDES workflow.
#
# Rows are IWA [lambda/D].  Columns are raw contrast C = 10^{-Cexp}.
#

C_exp_table = np.array([10.75, 10.50, 10.25, 10.00, 9.75,  9.50,  9.25,  9.00, 8.75,  8.50,  8.25,  8.00, 7.75,  7.50,  7.25,  7.00,], dtype=float)
C_table     = 10.0**(-C_exp_table)  # [raw contrast], monotonically increasing

IWA_table_desc = np.array([4.500, 4.375, 4.250, 4.125, 4.000, 3.875, 3.750, 3.625, 3.500, 3.375, 3.250, 3.125, 3.000, 2.875, 2.750, 2.625, 2.500, 2.375, 2.250, 2.125, 2.000,], dtype=float)

# Transmission in percent, in the same row/column order as the PDF table:
# IWA decreasing, C_exp decreasing from 10.75 to 7.00.
T_apod_percent_desc = np.array([
    [74.1, 75.1, 75.7, 75.9, 76.1, 76.3, 76.5, 76.7, 77.0, 77.3, 77.7, 78.2, 78.8, 79.5, 80.4, 81.2],
    [72.2, 73.8, 75.3, 76.4, 76.8, 77.2, 77.5, 77.8, 78.2, 78.5, 78.9, 79.3, 79.7, 80.0, 80.4, 80.8],
    [70.4, 71.5, 72.0, 72.3, 72.8, 73.3, 74.0, 74.8, 75.8, 76.9, 78.2, 78.9, 79.4, 79.7, 80.0, 80.3],
    [66.9, 67.2, 67.5, 67.9, 68.4, 69.0, 69.7, 70.5, 71.5, 72.8, 74.2, 75.8, 77.7, 79.3, 80.1, 80.5],
    [64.3, 64.5, 64.8, 65.2, 65.7, 66.2, 66.9, 67.7, 68.6, 69.7, 71.1, 72.7, 74.7, 76.9, 79.3, 81.0],
    [63.1, 63.3, 63.6, 64.0, 64.4, 64.9, 65.4, 66.0, 66.7, 67.5, 68.6, 70.0, 71.7, 73.7, 76.3, 79.2],
    [63.2, 63.4, 63.6, 63.9, 64.2, 64.5, 64.8, 65.3, 65.8, 66.5, 67.4, 68.4, 69.7, 71.4, 73.5, 76.1],
    [64.0, 64.1, 64.2, 64.4, 64.6, 64.8, 65.1, 65.4, 65.8, 66.3, 66.9, 67.7, 68.7, 70.0, 71.6, 73.8],
    [62.6, 63.8, 64.5, 64.9, 65.2, 65.5, 65.7, 66.1, 66.4, 66.9, 67.4, 68.2, 69.1, 70.2, 71.7, 73.6],
    [51.5, 53.5, 55.6, 58.1, 60.5, 62.6, 64.7, 65.9, 66.8, 67.6, 68.5, 69.4, 70.3, 71.4, 72.7, 74.4],
    [41.2, 42.2, 43.3, 44.8, 46.6, 48.8, 51.4, 54.5, 58.0, 61.7, 64.9, 67.8, 69.7, 71.8, 73.6, 74.9],
    [37.6, 38.2, 39.0, 40.0, 41.3, 42.9, 44.9, 47.5, 50.6, 54.4, 58.8, 64.0, 68.5, 71.9, 74.4, 75.6],
    [36.0, 36.4, 36.9, 37.6, 38.5, 39.6, 41.1, 43.0, 45.4, 48.4, 52.2, 56.9, 62.6, 68.8, 73.3, 75.8],
    [36.2, 36.5, 36.9, 37.5, 38.2, 39.2, 40.4, 42.1, 44.2, 46.9, 50.3, 54.6, 60.0, 66.3, 72.6, 76.5],
    [37.0, 37.3, 37.7, 38.2, 38.8, 39.6, 40.7, 42.1, 44.0, 46.3, 49.3, 53.1, 57.8, 63.7, 70.6, 74.6],
    [38.5, 38.8, 39.2, 39.7, 40.3, 41.1, 42.1, 43.4, 45.1, 47.2, 49.8, 53.1, 57.2, 61.8, 64.9, 67.6],
    [40.5, 40.8, 41.2, 41.7, 42.3, 43.0, 43.9, 44.9, 46.1, 47.7, 49.7, 52.2, 54.3, 56.1, 57.9, 59.8],
    [42.9, 43.3, 43.7, 44.2, 44.6, 45.1, 45.8, 46.6, 47.6, 48.7, 49.6, 50.5, 51.6, 52.6, 53.8, 55.4],
    [35.7, 44.0, 46.1, 46.4, 46.7, 46.9, 47.1, 47.3, 47.6, 48.0, 48.5, 49.1, 49.8, 50.8, 52.0, 53.4],
    [16.4, 20.2, 24.6, 29.4, 32.6, 35.9, 39.9, 44.8, 47.7, 48.1, 48.6, 49.3, 50.0, 51.0, 52.1, 53.3],
    [ 8.1, 10.2, 12.4, 14.8, 17.6, 21.0, 25.0, 29.8, 35.7, 42.8, 50.1, 50.8, 51.6, 52.4, 53.1, 53.8],
], dtype=float)

# RegularGridInterpolator needs monotonic axes.  The C axis is already monotonic
# in raw contrast, while the IWA axis is reversed to become increasing.
IWA_table       = IWA_table_desc[::-1]
T_apod_fraction = T_apod_percent_desc[::-1] / 100.0
T_apod_interp   = RegularGridInterpolator((IWA_table, np.log10(C_table)), T_apod_fraction, bounds_error=False, fill_value=None,)



def parse_aplc_filename(filename):
    """
    Parse the HWO/APLC FITS filename and return its IWA and raw contrast.
    """
    def canonicalize_C_exp(C_exp):
        """
        Map rounded filename values such as 7.2 or 10.8 to the exact survey grid
        values 7.25 and 10.75.
        """
        return float(C_exp_table[np.argmin(np.abs(C_exp_table - float(C_exp)))])
    name  = Path(filename).name
    m_iwa = re.search(r"IWA=([0-9.]+)", name)
    m_c   = re.search(r"MaxTau_C=([0-9.]+)", name)
    if m_iwa is None or m_c is None:
        raise ValueError(f"Could not parse IWA/MaxTau_C from filename: {name}")
    IWA_lD = float(m_iwa.group(1))
    C_exp  = canonicalize_C_exp(float(m_c.group(1)))
    C_raw  = 10.0**(-C_exp)
    return IWA_lD, C_exp, C_raw



def get_aplc_off_axis_transmission(IWA_lD, C_raw):
    """
    Off-axis apodizer transmission for a fully separated source.
    """
    point = np.array([[float(IWA_lD), np.log10(float(C_raw))]])
    return float(np.clip(T_apod_interp(point)[0], EPS, 1.0 - EPS))



def rebin_and_crop_image(image_2D, pxscale_in, pxscale_out):
    """
    Rebin a 2D PSF image while preserving its physical field of view.

    The cropping radius is determined AFTER spatial resampling to retain
    the largest centered, odd-sized square available in the output image.
    """
    image_2D = rebin_flux_conserving(data=image_2D, pxscale_in=pxscale_in, pxscale_out=pxscale_out)
    R_crop   = (min(image_2D.shape) - 1) // 2
    image_2D = crop(image_2D, Y0=image_2D.shape[0]//2, X0=image_2D.shape[1]//2, R_crop=R_crop)
    image_2D = fill_invalid_by_symmetry(image_2D)
    image_2D = fill_invalid_by_nearest(image_2D)
    return image_2D



def box_convolution_2d(image_2D, size_core):
    """
    Apply the same NaN-aware box convolution convention as in FastYield.
    """
    if size_core <= 1:
        return np.copy(image_2D)
    return box_convolution(data=image_2D, size_core=size_core, mode="mean")



def normalize_image(image_2D, context=""):
    """
    Normalize a PSF image to unit integrated flux.
    """
    total = np.nansum(image_2D)
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"Cannot normalize image with non-positive total flux. {context}")
    return image_2D / total



def compute_radial_profile_density_lD(image_2D, pxscale_lD):
    """
    Compute the radial PSF-profile density in lambda/D units.

    Parameters
    ----------
    image_2D : ndarray
        Normalized PSF image. Its sum must be equal to one.
    pxscale_lD : float
        Pixel scale [lambda/D/px].

    Returns
    -------
    separation_profile_lD : ndarray
        Native annular separation axis [lambda/D].
    PSF_profile_density_lD : ndarray
        Radial PSF profile density [flux fraction / (lambda/D)^2].
    """
    NbLine, NbColumn      = image_2D.shape
    N_sep_profile         = int(round(np.sqrt((NbLine / 2)**2 + (NbColumn / 2)**2)))
    separation_profile_lD = np.zeros(N_sep_profile)
    PSF_profile_1D        = np.zeros(N_sep_profile)

    for r in range(N_sep_profile):
        r_int = max(1, r - 1) if r > 1 else r
        r_ext = r

        separation_profile_lD[r] = (r_int + r_ext) / 2.0 * pxscale_lD

        amask             = annular_mask(r_int, r_ext, size=(NbLine, NbColumn)) == 1
        PSF_profile_1D[r] = np.nanmean(image_2D[amask])

    PSF_profile_density_lD = PSF_profile_1D / pxscale_lD**2

    return separation_profile_lD, PSF_profile_density_lD



def compute_hwo_cube_products_lD(filename, target_off_axis_transmission, pxscale_raw_lD, pxscale_lD, size_core):    
    """
    Compute the three FastYield PSF quantities from one HWO/APLC FITS cube.

    The FITS cube is assumed to have shape:

        (NbOff, NbLine, NbColumn)

    Returned quantities are first computed in lambda/D units and then
    interpolated on the common separation_lD axis.

    Returned arrays
    ---------------
    PSF_profile_density_lD : ndarray, shape (N_sep,)
        On-axis coronagraphic stellar PSF-profile density
        [flux fraction / (lambda/D)^2].
    fraction_core_lD : ndarray, shape (N_sep,)
        Off-axis flux fraction enclosed in the FWHM-sized core aperture.
    radial_transmission_lD : ndarray, shape (N_sep,)
        Radial transmission profile normalized to the APLC survey off-axis
        apodizer transmission.
    """
    cube_3D                                = fits.getdata(filename)
    N_offset_raw, NbLine_raw, NbColumn_raw = cube_3D.shape
    offset_raw_lD                          = np.arange(N_offset_raw, dtype=float) * shift_step_lD

    # ------------------------------------------------------------------
    # 1) On-axis coronagraphic stellar PSF profile density
    # ------------------------------------------------------------------
    image_star      = np.copy(cube_3D[0]) # without offset / perfect centering with the coronagraph
    image_star      = rebin_and_crop_image(image_2D=image_star, pxscale_in=pxscale_raw_lD, pxscale_out=pxscale_lD)
    image_star      = normalize_image(image_star, context=f"on-axis PSF in {Path(filename).name}")
    image_star_conv = box_convolution_2d(image_2D=image_star, size_core=size_core)

    separation_profile_native_lD, PSF_profile_density_native_lD = compute_radial_profile_density_lD(image_2D=image_star_conv, pxscale_lD=pxscale_lD)

    PSF_profile_density_lD = interp_extrap_sep(separation_ref=separation_profile_native_lD, separation_new=separation_lD, y_ref=np.clip(PSF_profile_density_native_lD, EPS, None), mode="log", tail_model="powerlaw", slope=-3)

    # ------------------------------------------------------------------
    # 2) Radial transmission from spatial sums
    # ------------------------------------------------------------------
    flux_offset = np.nansum(cube_3D, axis=(1, 2))

    radial_transmission_raw = flux_offset / np.nanmax(flux_offset) * target_off_axis_transmission
    radial_transmission_raw = np.clip(radial_transmission_raw, EPS, 1.0 - EPS)
    radial_transmission_raw = np.maximum.accumulate(radial_transmission_raw)
    radial_transmission_raw = np.minimum(radial_transmission_raw, target_off_axis_transmission)

    radial_transmission_lD = interp_extrap_sep(separation_ref=offset_raw_lD, separation_new=separation_lD, y_ref=safe01(radial_transmission_raw), mode="logit", tail_model="flat")
    radial_transmission_lD = safe01(np.maximum.accumulate(radial_transmission_lD))
    radial_transmission_lD = np.minimum(radial_transmission_lD, target_off_axis_transmission)

    # ------------------------------------------------------------------
    # 3) Core fraction of the off-axis PSF vs source offset
    # ------------------------------------------------------------------
    fraction_core_raw = np.zeros(N_offset_raw)

    for ioff in range(N_offset_raw):

        image_off = np.copy(cube_3D[ioff])

        dy_shift = 0.0
        dx_shift = - offset_raw_lD[ioff] / pxscale_raw_lD

        image_off = shift(image_off, shift=(dy_shift, dx_shift), order=order_PSF, mode="constant", cval=np.nan)
        image_off = fill_invalid_by_symmetry(image_off)
        image_off = fill_invalid_by_nearest(image_off)

        image_off      = rebin_and_crop_image(image_2D=image_off, pxscale_in=pxscale_raw_lD, pxscale_out=pxscale_lD)
        image_off      = normalize_image(image_off, context=f"off-axis PSF offset index {ioff} in {Path(filename).name}")
        image_off_conv = box_convolution_2d(image_2D=image_off, size_core=size_core)

        y0, x0                  = image_off_conv.shape[0] // 2, image_off_conv.shape[1] // 2
        fraction_core_raw[ioff] = size_core**2 * image_off_conv[y0, x0]
        
        if ioff == 0:
            if not np.allclose(image_star_conv[y0, x0], image_off_conv[y0, x0], rtol=1e-6, atol=1e-10):
                raise ValueError("This should not happen!")
    

    fraction_core_raw = safe01(fraction_core_raw)

    fraction_core_lD = interp_extrap_sep(separation_ref=offset_raw_lD, separation_new=separation_lD, y_ref=safe01(fraction_core_raw), mode="logit", tail_model="flat")
    fraction_core_lD = safe01(np.maximum.accumulate(fraction_core_lD))

    return PSF_profile_density_lD, fraction_core_lD, radial_transmission_lD



def project_lD_products_to_mas(PSF_profile_density_lD, fraction_core_lD, radial_transmission_lD, target_off_axis_transmission):
    """
    Project raw lambda/D products onto the output (wave, separation_mas) grid.

    The raw products are defined on the common separation_lD axis. The final
    separation axis is fixed in mas. Therefore, each wavelength samples a
    different dimensionless separation:

        separation_lD(lambda) = separation_mas / (lambda/D)_mas.

    Only PSF_profile_density needs a unit conversion:

        density_mas = density_lD / [(lambda/D in mas)^2].
    """
    PSF_profile_density_2D = np.zeros((len(wave), len(separation_mas)))
    fraction_core_2D = np.zeros((len(wave), len(separation_mas)))
    radial_transmission_2D = np.zeros((len(wave), len(separation_mas)))

    for iw, l0 in enumerate(wave):

        lD_mas              = lambda_over_D_mas(wavelength_um=l0, diameter_m=D)
        separation_query_lD = separation_mas / lD_mas

        PSF_profile_density_query_lD = interp_extrap_sep(separation_ref=separation_lD, separation_new=separation_query_lD, y_ref=np.clip(PSF_profile_density_lD, EPS, None), mode="log",   tail_model="powerlaw", slope=-3)
        fraction_core_query          = interp_extrap_sep(separation_ref=separation_lD, separation_new=separation_query_lD, y_ref=safe01(fraction_core_lD),                   mode="logit", tail_model="flat")
        radial_transmission_query    = interp_extrap_sep(separation_ref=separation_lD, separation_new=separation_query_lD, y_ref=safe01(radial_transmission_lD),             mode="logit", tail_model="flat")

        PSF_profile_density_2D[iw] = PSF_profile_density_query_lD / lD_mas**2
        fraction_core_2D[iw]       = safe01(np.maximum.accumulate(fraction_core_query))

        radial_transmission_query  = safe01(np.maximum.accumulate(radial_transmission_query))
        radial_transmission_query  = np.minimum(radial_transmission_query, target_off_axis_transmission)
        radial_transmission_2D[iw] = safe01(radial_transmission_query)

    return PSF_profile_density_2D, fraction_core_2D, radial_transmission_2D



def write_fits(filename, data, header_items=None):
    """
    Write a FITS file with optional header metadata.
    """
    hdu = fits.PrimaryHDU(data)

    if header_items is not None:
        for key, value in header_items.items():
            hdu.header[key] = value

    hdu.writeto(filename, overwrite=True)



# %%
# Getting all HWO/APLC FITS cubes
#

files = sorted(data_dir.glob("cube_tiptilt_HWO_APLC_*.fits"))
if len(files) == 0:
    raise FileNotFoundError(f"No HWO/APLC FITS cube found in {data_dir}.\n""Check data_dir or launch this script from FastCurves-main.")

# Use only files that are part of the APLC survey table.
metadata = []
for f in files:
    IWA_value_lD, C_exp_value, C_raw_value = parse_aplc_filename(f)
    if np.min(np.abs(IWA_table - IWA_value_lD)) > EPS:
        continue
    if np.min(np.abs(C_exp_table - C_exp_value)) > EPS:
        continue
    metadata.append((f, IWA_value_lD, C_exp_value, C_raw_value))
if len(metadata) == 0:
    raise RuntimeError("FITS files were found, but none matched the APLC survey grid.")
# Check that every expected coronagraphic configuration is present exactly once.
keys       = [(m[1], m[2]) for m in metadata]
counts     = Counter(keys)
duplicates = [key for key, count in counts.items() if count > 1]
expected   = set(product(IWA_table.tolist(), C_exp_table.tolist()))
missing    = expected - set(keys)
if duplicates:
    raise RuntimeError(f"Duplicate HWO/APLC configurations: {duplicates}")
if missing:
    raise RuntimeError(f"Missing {len(missing)} HWO/APLC configurations. First missing pairs: {sorted(missing)[:10]}")


IWA_lD = np.array(sorted({m[1] for m in metadata}), dtype=float)

# Keep the raw-contrast axis in the same order as the PDF table, i.e. from the
# most stringent contrast to the most relaxed one.
C_exp = np.array(C_exp_table, dtype=float)
C     = np.array([10.0**(-ce) for ce in C_exp], dtype=float)

N_C   = len(C)
N_IWA = len(IWA_lD)
N_sep = len(separation_mas)

print("\n\n\n" + "\033[4m" + "HWO/APLC survey grid:" + "\033[0m")
print(f"Number of FITS files       = {len(metadata)}")
print(f"IWA axis                   = {IWA_lD} lambda/D")
print(f"C exponent axis            = {C_exp}")
print(f"Raw contrast axis          = {C}")
print(f"Internal separation_lD     = {separation_lD[0]:.3g}--{separation_lD[-1]:.3g} lambda/D")
print(f"Output separation_mas      = {separation_mas[0]:.1f}--{separation_mas[-1]:.1f} mas")
print(f"Telescope diameter         = {D:.2f} m")

# Determine the native lambda/D pixel scale from the first cube.
first_cube                             = fits.getdata(metadata[0][0])
N_offset_raw, NbLine_raw, NbColumn_raw = first_cube.shape

pxscale_raw_lD = FOV_lD  / NbLine_raw # [lambda/D/px]
pxscale_lD     = FWHM_lD / size_core  # [lambda/D/px]

# Determine the actual PSF dimensions after spatial resampling.
image_rebinned = rebin_flux_conserving(data=first_cube[0], pxscale_in=pxscale_raw_lD, pxscale_out=pxscale_lD)
R_crop         = (min(image_rebinned.shape) - 1) // 2
NbLine         = 2 * R_crop + 1
NbColumn       = 2 * R_crop + 1
del image_rebinned

print(f"Raw cube shape             = {first_cube.shape}")
print(f"Raw offset axis            = 0--{(N_offset_raw - 1) * shift_step_lD:.2f} lambda/D")
print(f"pxscale_raw_lD             = {pxscale_raw_lD:.6f} lambda/D/px")
print(f"cropped PSF size           = {NbLine} x {NbColumn} px")
print(f"size_core                  = {size_core} px")



# %%
# Computing raw 3D products in lambda/D: (C, IWA_lD, separation_lD)
#

PSF_profile_density_3D_lD  = np.zeros((N_C, N_IWA, len(separation_lD))) * np.nan
fraction_core_3D_lD        = np.zeros((N_C, N_IWA, len(separation_lD))) * np.nan
radial_transmission_3D_lD  = np.zeros((N_C, N_IWA, len(separation_lD))) * np.nan
off_axis_transmission_2D   = np.zeros((N_C, N_IWA)) * np.nan

pbar_metadata = tqdm(enumerate(metadata, start=1), total=len(metadata), desc="HWO/APLC cubes", unit="cube", position=0, leave=True, dynamic_ncols=True, smoothing=0.05, bar_format="{l_bar}{bar:30}{r_bar}")
for k, (filename, IWA_value_lD, C_exp_file, C_raw_file) in pbar_metadata:
    
    idx_IWA = np.argmin(np.abs(IWA_lD - IWA_value_lD))
    idx_C   = np.argmin(np.abs(C_exp  - C_exp_file))

    target_off_axis_transmission = get_aplc_off_axis_transmission(IWA_lD=IWA_value_lD, C_raw=C_raw_file)
    
    PSF_profile_density_lD, fraction_core_lD, radial_transmission_lD = compute_hwo_cube_products_lD(filename=filename, target_off_axis_transmission=target_off_axis_transmission, pxscale_raw_lD=pxscale_raw_lD, pxscale_lD=pxscale_lD, size_core=size_core,)
    PSF_profile_density_3D_lD[idx_C, idx_IWA] = PSF_profile_density_lD
    fraction_core_3D_lD[idx_C, idx_IWA]       = fraction_core_lD
    radial_transmission_3D_lD[idx_C, idx_IWA] = radial_transmission_lD
    off_axis_transmission_2D[idx_C, idx_IWA]  = target_off_axis_transmission

    if PLOT_EACH_CUBE:

        # Plot the three HWO/APLC quantities in lambda/D units
        mask_sep         = separation_lD > 0
        sep_plot         = separation_lD[mask_sep]
        psf_plot         = PSF_profile_density_lD[mask_sep]
        fc_plot          = fraction_core_lD[mask_sep]
        rt_plot          = radial_transmission_lD[mask_sep]
        off_axis_percent = 100.0 * target_off_axis_transmission
        fig, axs         = plt.subplots(3, 1, figsize=(6, 8), dpi=300, sharex=True)
        fig.subplots_adjust(hspace=0.35)
        fig.suptitle(f"HWO/APLC PSF quantities\n"f"IWA = {IWA_value_lD:.3f} $\\lambda/D$, "f"C = $10^{{-{C_exp_file:.2f}}}$", fontsize=12, fontweight="bold")
        # Subplot 1: stellar PSF radial profile density
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].plot(sep_plot, psf_plot, c="black", lw=1.5)
        axs[0].set_xlim(sep_plot[0], sep_plot[-1])
        axs[0].set_ylabel(r"Mean flux fraction density" "\n" r"[$(\lambda/D)^{-2}$]", fontsize=10)
        axs[0].set_title("PSF profile density (for the star)", fontsize=10)
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        axs[0].minorticks_on()
        axs[0].tick_params(labelsize=8)
        axs[0].set_ylim(1e-12, 1e-2)
        # Subplot 2: fraction inside the FWHM-sized core
        axs[1].set_yscale("log")
        axs[1].plot(sep_plot, 100.0 * fc_plot, c="tab:blue", lw=1.5)
        axs[1].set_ylabel("Core PSF flux fraction [%]", fontsize=10)
        axs[1].set_title("Flux inside the FWHM-sized core (for the planet)", fontsize=10)
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        axs[1].minorticks_on()
        axs[1].tick_params(labelsize=8)
        axs[1].set_ylim(1e-2, 100.0)
        # Subplot 3: radial transmission
        axs[2].set_yscale("log")
        axs[2].plot(sep_plot, 100.0 * rt_plot, c="tab:orange", lw=1.5, label="Radial transmission")
        axs[2].axhline(off_axis_percent, ls="--", c="gray", lw=1.2, alpha=0.9)    
        axs[2].text(0.98, off_axis_percent, f"off-axis APLC = {off_axis_percent:.1f}%", transform=axs[2].get_yaxis_transform(), ha="right", va="bottom", fontsize=10, color="gray", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),)
        axs[2].set_xlabel(r"Angular offset separation from the coronagraph [$\lambda/D$]", fontsize=10)
        axs[2].set_ylabel("Transmission [%]", fontsize=10)
        axs[2].set_title("Coronagraph radial throughput", fontsize=10)
        axs[2].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        axs[2].minorticks_on()
        axs[2].tick_params(labelsize=8)
        axs[2].set_ylim(1e-1, 100.0)
        plt.tight_layout()
        plt.show()
        
        

# %%
# Projecting raw lambda/D products to FastYield-like 4D products:
# (wave, C, IWA_lD, separation_mas)
#

PSF_profile_density_4D  = np.zeros((len(wave), N_C, N_IWA, len(separation_mas))) * np.nan
fraction_core_4D        = np.zeros((len(wave), N_C, N_IWA, len(separation_mas))) * np.nan
radial_transmission_4D  = np.zeros((len(wave), N_C, N_IWA, len(separation_mas))) * np.nan

for idx_C in range(N_C):
    for idx_IWA in range(N_IWA):

        target_off_axis_transmission = off_axis_transmission_2D[idx_C, idx_IWA]

        PSF_profile_density_2D, fraction_core_2D, radial_transmission_2D = project_lD_products_to_mas(PSF_profile_density_lD=PSF_profile_density_3D_lD[idx_C, idx_IWA], fraction_core_lD=fraction_core_3D_lD[idx_C, idx_IWA], radial_transmission_lD=radial_transmission_3D_lD[idx_C, idx_IWA], target_off_axis_transmission=target_off_axis_transmission,)

        PSF_profile_density_4D[:, idx_C, idx_IWA, :] = PSF_profile_density_2D
        fraction_core_4D[:, idx_C, idx_IWA, :]       = fraction_core_2D
        radial_transmission_4D[:, idx_C, idx_IWA, :] = radial_transmission_2D

fraction_core_4D       = safe01(fraction_core_4D)
radial_transmission_4D = safe01(radial_transmission_4D)



# %%
# Plotting and saving
#

if PLOT_FINAL_PRODUCTS:
    # Representative slices: strict, middle, and relaxed raw contrast.
    idx_C_list   = [0, len(C)      // 2, -1]
    idx_IWA_list = [0, len(IWA_lD) // 2, -1]

    for idx_C in idx_C_list:
        for idx_IWA in idx_IWA_list:

            psf = PSF_profile_density_4D[:, idx_C, idx_IWA]
            fc  = fraction_core_4D[:,       idx_C, idx_IWA]
            rt  = radial_transmission_4D[:, idx_C, idx_IWA]

            plot_profiles(
                instru=instru,
                coronagraph=coronagraph,
                apodizer=apodizer,
                strehl=strehl,
                pxscale=None,
                sep_unit=sep_unit,
                size_core=size_core,
                wave=wave,
                wave_raw=None,
                separation=separation_mas,
                PSF_profile_density_2D=psf,
                fraction_core_2D=fc,
                radial_transmission_2D=rt,
                type_PSF=None,
                title_suffix=(
                    f" \n C = {C_exp[idx_C]:.2f}, "
                    f"IWA = {IWA_lD[idx_IWA]:.3f} lambda/D"
                ),
            )

# --- Saving ---
suffix_PSF = (
    f"{apodizer}_{coronagraph}_D{D:.2f}m_"
    f"px{pxscale_lD:.6f}lD_"
    f"wave{lmin:.2f}_{lmax:.2f}um_"
    f"C{C[0]:.1e}_{C[-1]:.1e}_"
    f"IWA{IWA_lD[0]:.3f}_{IWA_lD[-1]:.3f}lD_"
    f"sep{sep_max:.0f}mas"
)

common_header = {
    "INSTRU":   instru,
    "CORO":     coronagraph,
    "APODIZER": apodizer,
    "STREHL":   strehl,
    "D_M":      float(D),
    "SEPUNIT":  "mas",
    "IWAUNIT":  "lambda/D",
    "PXRAWLD":  float(pxscale_raw_lD),
    "PXOUTLD":  float(pxscale_lD),
    "FOVLD":    float(FOV_lD),
    "SHIFTLD":  float(shift_step_lD),
    "COMMENT":  "Input cubes are achromatic in lambda/D; final separation axis is mas.",
}

write_fits(outdir / f"{instru}_C_{suffix_PSF}.fits",                         C,                         common_header)
write_fits(outdir / f"{instru}_Cexp_{suffix_PSF}.fits",                      C_exp,                     common_header)
write_fits(outdir / f"{instru}_IWA_lD_{suffix_PSF}.fits",                    IWA_lD,                    common_header)
write_fits(outdir / f"{instru}_wave_{suffix_PSF}.fits",                      wave,                      common_header)
write_fits(outdir / f"{instru}_separation_lD_{suffix_PSF}.fits",             separation_lD,             common_header)
write_fits(outdir / f"{instru}_separation_mas_{suffix_PSF}.fits",            separation_mas,            common_header)
write_fits(outdir / f"{instru}_lambda_over_D_mas_{suffix_PSF}.fits",         lambda_over_D_mas(wave, D), common_header)
write_fits(outdir / f"{instru}_off_axis_transmission_2D_{suffix_PSF}.fits",  off_axis_transmission_2D,  common_header)

# Raw dimensionless products
write_fits(outdir / f"{instru}_PSF_profile_density_3D_lD_{suffix_PSF}.fits", PSF_profile_density_3D_lD, common_header)
write_fits(outdir / f"{instru}_fraction_core_3D_lD_{suffix_PSF}.fits",       fraction_core_3D_lD,       common_header)
write_fits(outdir / f"{instru}_radial_transmission_3D_lD_{suffix_PSF}.fits", radial_transmission_3D_lD, common_header)

# Final FastYield-like products
write_fits(outdir / f"{instru}_PSF_profile_density_4D_{suffix_PSF}.fits",    PSF_profile_density_4D,    common_header)
write_fits(outdir / f"{instru}_fraction_core_4D_{suffix_PSF}.fits",          fraction_core_4D,          common_header)
write_fits(outdir / f"{instru}_radial_transmission_4D_{suffix_PSF}.fits",    radial_transmission_4D,    common_header)

print("\nSaved HWO/APLC PSF products in:")
print(outdir)

print("\nIntermediate lambda/D convention:")
print("  PSF_profile_density_3D_lD -> (C, IWA_lD, separation_lD) [flux fraction / (lambda/D)^2]")
print("  fraction_core_3D_lD       -> (C, IWA_lD, separation_lD) [fraction]")
print("  radial_transmission_3D_lD -> (C, IWA_lD, separation_lD) [transmission]")

print("\nFinal FastYield-like output convention:")
print("  PSF_profile_density_4D    -> (wave, C, IWA_lD, separation_mas) [flux fraction / mas^2]")
print("  fraction_core_4D          -> (wave, C, IWA_lD, separation_mas) [fraction]")
print("  radial_transmission_4D    -> (wave, C, IWA_lD, separation_mas) [transmission]")