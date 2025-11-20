# import instru configs
from src.config import *

# import astropy modules
from astropy import constants as const
from astropy import units as u
from astropy.io import fits, ascii
from astropy.coordinates import EarthLocation, AltAz, get_body, SkyCoord
from astropy.convolution import Gaussian1DKernel
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy.table import QTable, Table, Column, MaskedColumn
from astropy.utils.masked.core import Masked
from astropy.units import Quantity

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# import scipy modules
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy.ndimage import shift, zoom, distance_transform_edt, fourier_shift, convolve
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.special import comb
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.fft import fftn, ifftn

# import other modules
import time
import copy
import emcee
import os
import pandas as pd
import pyvo as vo
import statsmodels.api as sm
import warnings
from PyAstronomy import pyasl
from tqdm import tqdm
from itertools import combinations
from numba import njit, prange
from multiprocessing import Pool, cpu_count
import corner
from collections import OrderedDict
from functools import lru_cache

warnings.filterwarnings('ignore', category=UserWarning, append=True)

# Defining constants
h          = const.h.value     # [J.s]
c          = const.c.value     # [m/s]
kB         = const.k_B.value   # [J/K]
rad2arcsec = 180/np.pi*3600    # [arcsec/rad] (to convert [rad] => [arcsec])
sr2arcsec2 = 4.25e10           # [arcsec2/sr]  (to convert [steradians] => [arcsec2])
R_earth    = 6.371e6           # [m]
G          = 6.67430e-11       # [m3/kg/s2]
G_cgs      = const.G.cgs.value # [cm3/g/s2]



# ============ Load & Caches ============

@lru_cache(maxsize=50)
def _load_instru_trans(instru, band):
    """Load *once* the native instrument-throughput curve for (instru, band)."""
    wave, trans = fits.getdata(f"sim_data/Transmission/{instru}/transmission_{band}.fits")
    return wave, trans

@lru_cache(maxsize=50)
def _load_tell_trans(airmass):
    """Load *once* the native sky transmission curve for a given airmass."""
    wave_tell, tell = fits.getdata(f"sim_data/Transmission/sky_transmission_airmass_{airmass:.1f}.fits")
    # format attendu: 2 x N
    return wave_tell, tell

@lru_cache(maxsize=50)
def _load_psf_profile(instru, band, strehl, apodizer, coronagraph):
    """Load *once* the PSF profile arrays + header for (instru, band, strehl, apodizer, coronagraph)."""
    if coronagraph is None:
        psf_file = f"sim_data/PSF/PSF_{instru}/PSF_{band}_{strehl}_{apodizer}.fits"
    else:
        psf_file = f"sim_data/PSF/PSF_{instru}/PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits"
    hdul      = fits.open(psf_file)
    hdr       = hdul[0].header
    sep, prof = hdul[0].data
    hdul.close()
    return hdr, sep, prof, psf_file

@lru_cache(maxsize=50)
def _load_corona_profile(instru, band, strehl, apodizer, coronagraph):
    """Load *once* the coronagraphic profile arrays."""
    sep, fraction_core, radial_transmission = fits.getdata(f"sim_data/PSF/PSF_{instru}/fraction_PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits")
    return sep, fraction_core, radial_transmission

@lru_cache(maxsize=50)
def _load_noiseless_cube(instru, band, use_data, T_star_sim, correction):
    """Load *once* noiseless data cubes."""
    if use_data: 
        S_noiseless = fits.getdata(f"sim_data/Systematics/{instru}/S_data_star_center_s3d_{band}.fits")              # On-sky data cube used to estimate the modulations [e-]
        pxscale     = fits.getheader(f"sim_data/Systematics/{instru}/S_data_star_center_s3d_{band}.fits")['pxscale'] # [arcsec/px]
        wave        = fits.getdata(f"sim_data/Systematics/{instru}/wave_data_star_center_s3d_{band}.fits")           # Wavelength axis of the data
    else:
        S_noiseless = fits.getdata(f"sim_data/Systematics/{instru}/S_noiseless_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits")              # MIRISim noiseless data cube used to estimate the modulations [e-]
        pxscale     = fits.getheader(f"sim_data/Systematics/{instru}/S_noiseless_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits")['pxscale'] # [arcsec/px]
        wave        = fits.getdata(f"sim_data/Systematics/{instru}/wave_noiseless_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits")           # Wavelength axis of the data
    return S_noiseless, wave, pxscale

@lru_cache(maxsize=50)
def _load_corr_factor(instru, band):
    """Load *once* the corrective factor."""
    sep, r_corr = fits.getdata(f"sim_data/R_corr/R_corr_{instru}/R_corr_{band}.fits")
    return sep, r_corr

@lru_cache(maxsize=50)
def _load_bkg_flux(instru, band, background):
    """Load *once* the background flux in [e-/px/s]."""
    raw_wave, raw_bkg = fits.getdata(f"sim_data/Background/{instru}/{background}/background_{band}.fits")
    return raw_wave, raw_bkg



############################# Basic functions #################################

def linear_interpolate(y1, y2, x1, x2, x):
    """
    Linear interpolation between (x1, y1) and (x2, y2) evaluated at x.

    Parameters
    ----------
    y1, y2 
        Function values at x1 and x2.
    x1, x2 
        Abscissae. Must satisfy x2 != x1.
    x  or array_like
        Target abscissa(e).

    Returns
    -------
    float or ndarray
        Interpolated value(s) at x.
    """
    if x2 == x1:
        raise ValueError(f"x1 and x2 must differ (got {x1}).")
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a * x + b



def fill_nan_linear(x, y):
    """
    Fill NaN or infinite values in an array 'y' by linear interpolation.

    Parameters
    ----------
    x : (N,) array_like
        Monotonic coordinate array corresponding to 'y'.
    y : (N,) array_like
        Input values, possibly containing NaN or ±inf.

    Returns
    -------
    y_filled : (N,) ndarray
        Copy of 'y' where invalid entries are replaced by linear interpolation
        between the nearest valid neighbors. If NaNs occur at the edges,
        no extrapolation is made.

    Notes
    -----
    - This function preserves the shape of 'y'.
    - If all values in 'y' are NaN, a ValueError is raised.
    """
    mask_valid = np.isfinite(y)
    
    if not np.any(mask_valid):
        raise ValueError("'y' contains no valid (finite) values to interpolate.")
    
    elif not np.any(~mask_valid):
        return y

    else:
        if x is None:
            x = np.arange(len(y), dtype=float)
        return interp1d(x[mask_valid], y[mask_valid], kind="linear", bounds_error=False, fill_value=np.nan)(x)



def improved_power_law_extrapolation(x, x0, y0, alpha, rc):
    """
    Hybrid PSF-wing extrapolation: power law with exponential roll-off.

    Model
    -----
    y(x) = y0 * (x / x0)^(-|alpha|) * exp(-(x - x0) / rc)

    Parameters
    ----------
    x : array_like
        Radial coordinate(s).
    x0 : float
        Reference radius (> 0).
    y0 : float
        Value at x0.
    alpha : float
        Power-law index (absolute value is used).
    rc : float
        Exponential scale length (> 0).

    Returns
    -------
    ndarray
        Extrapolated profile.
    """
    if x0 <= 0 or rc <= 0:
        raise ValueError("x0 and rc must be positive.")

    x = np.asarray(x, dtype=float)
    a = abs(alpha)

    return y0 * np.power(x / x0, -a) * np.exp(-(x - x0) / rc)



def gaussian(x, mu, sigma):
    """
    Normalized Gaussian PDF.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    mu : float
        Mean.
    sigma : float
        Standard deviation (> 0).

    Returns
    -------
    ndarray
        Gaussian values at x.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    arg  = -0.5 * ((x - mu) / sigma)**2
    return norm * np.exp(arg)



def gaussian0(x, sigma):
    """
    Normalized zero-mean Gaussian PDF.

    Equivalent to gaussian(x, 0, sigma).
    """
    return gaussian(x=x, mu=0., sigma=sigma)



def lorentzian(x, x0, L):
    """
    Normalized Lorentzian (Cauchy) profile with FWHM = L.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    x0 : float
        Center.
    L : float
        Full width at half maximum (> 0).

    Returns
    -------
    ndarray
        Lorentzian values at x.
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)



def smoothstep(x, Rc=None, N=10, x_min=None, x_max=None, filtering=True):
    """
    Generalized smoothstep polynomial (order N), with optional symmetric
    high-pass-like shaping around a cutoff Rc.

    If 'filtering' is True and Rc is provided, the domain is set to [0, 2*Rc].

    Parameters
    ----------
    x : array_like
        Input coordinate(s).
    Rc : float or None
        Cutoff scale (used only if filtering is True).
    N : int
        Polynomial order (>= 0).
    x_min, x_max : float or None
        Domain bounds. If None and filtering is True, set to (0, 2*Rc).
    filtering : bool
        If True, mirror and invert the polynomial to produce a symmetric
        transition suitable for filtering use-cases.

    Returns
    -------
    ndarray
        Smoothstep values in [0, 1].
    """
    if filtering:
        if Rc is None or Rc <= 0:
            raise ValueError("When filtering=True, Rc must be a positive number.")
        x_min = 0.
        x_max = 2*Rc

    t = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)

    # Polynomial smoothstep of order N (see e.g. Ken Perlin generalization).
    # result = sum_{n=0..N} data2_crop(N+n, n) * data2_crop(2N+1, N-n) * (-t)^n * t^{N+1}
    result = np.zeros_like(t)
    for n in range(N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * ((-t) ** n)
    result *= t ** (N + 1)

    if filtering:
        # Symmetric shaping and inversion to get a bell-like rejection near Rc
        sym    = result[::-1]
        result = np.abs(sym + result - 1)

    return result



def get_logL(flux_observed, flux_model, sigma_l, method="classic"): # see https://github.com/exoAtmospheres/ForMoSA/blob/activ_dev/ForMoSA/nested_sampling/nested_logL_functions.py
    """
    Compute log-likelihoods used in HRS/molecular mapping literature.

    Methods
    -------
    - "classic":        Weighted least-squares with optimal scaling R.
    - "classic_bis":    L2-normalized f and t, noise-weighted chi².
    - "extended":       Classic + variance term (s²) marginalization.
    - "extended_bis":   As above but with L2-normalized inputs.
    - "Brogi":          Brogi & Line-style (2019) form on normalized inputs.
    - "Zucker":         Zucker (2003)-style using correlation coefficient C².
    - "custom":         Custom quadratic form with noise harmonic-mean weight.

    Parameters
    ----------
    flux_observed, flux_model, sigma_l : array_like
        Observed flux, model flux, and per-sample uncertainties (must be the same for all flux_model parameters).
        Must be broadcastable to the same shape.
    method : str
        One of the methods listed above.

    Returns
    -------
    float
        log-likelihood value.

    Notes
    -----
    - Non-finite samples are masked out.
    - A small epsilon is added where needed to avoid division by zero.
    """    
    f = np.asarray(flux_observed, dtype=float)
    t = np.asarray(flux_model,    dtype=float)
    s = np.asarray(sigma_l,       dtype=float)

    # Consistent mask for all terms
    m       = np.isfinite(f) & np.isfinite(t) & np.isfinite(s) & (s > 0)
    f, t, s = f[m], t[m], s[m]

    if f.size == 0:
        raise ValueError("No valid samples after masking (check NaNs/Infs and s > 0).")

    s2 = s * s
    w  = 1.0 / s2

    if method == "classic":  # true Gaussian log-likelihood with optimal scale R
        tt   = np.sum(w * t * t)
        ft   = np.sum(w * f * t)
        R    = ft / tt
        res  = f - R * t
        chi2 = np.sum(w * res * res)
        logL = -0.5 * (chi2 + np.sum(np.log(2.0 * np.pi * s2)))

    elif method == "extended":
        R = np.nansum(f * t / s2) / np.nansum(t * t / s2)
        chi2 = np.nansum(((f - R * t) ** 2) / s2)
        s2_hat = chi2 / N
        logL = -(chi2 / (2.0 * s2_hat) + (N / 2.0) * np.log(2.0 * np.pi * s2_hat) + 0.5 * np.log(np.nansum(s2)))

    elif method == "extended_bis":
        f_n = f / np.sqrt(np.nansum((f * f) / s2) + eps)
        g_n = t / np.sqrt(np.nansum((t * t) / s2) + eps)
        chi2 = np.nansum(((f_n - g_n) ** 2) / s2)
        s2_hat = chi2 / N
        logL = -(chi2 / (2.0 * s2_hat) + (N / 2.0) * np.log(2.0 * np.pi * s2_hat) + 0.5 * np.log(np.nansum(s2)))

    elif method == "Brogi":
        f_n = f / np.sqrt(np.nansum(f * f) + eps)
        g_n = t / np.sqrt(np.nansum(t * t) + eps)
        Sf2 = np.nansum(f_n * f_n) / N
        Sg2 = np.nansum(g_n * g_n) / N
        R = np.nansum(f_n * g_n) / N
        logL = - (N / 2.0) * np.log(Sf2 - 2.0 * R + Sg2 + eps)

    elif method == "Zucker":
        f_n = f / np.sqrt(np.nansum(f * f) + eps)
        g_n = t / np.sqrt(np.nansum(t * t) + eps)
        Sf2 = np.nansum(f_n * f_n) / N
        Sg2 = np.nansum(g_n * g_n) / N
        R = np.nansum(f_n * g_n) / N
        C2 = (R * R) / (Sf2 * Sg2 + eps)
        logL = - (N / 2.0) * np.log(1.0 - C2 + eps)

    elif method == "custom":
        f_n = f / np.sqrt(np.nansum(f * f) + eps)
        g_n = t / np.sqrt(np.nansum(t * t) + eps)
        Sf2 = np.nansum(f_n * f_n) / N
        Sg2 = np.nansum(g_n * g_n) / N
        R = np.nansum(f_n * g_n) / N
        # Harmonic-mean-like noise weight
        sigma2_weight = 1.0 / (np.nansum(1.0 / (s2 + eps)) / N + eps)
        logL = - (N / (2.0 * sigma2_weight)) * (Sf2 + Sg2 - 2.0 * R)

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(logL)



def get_bracket_values(x, grid):
    """
    Return (x_lo, x_hi) bracketing 'x' on a sorted 1D grid.

    Parameters
    ----------
    x : float
        Target value to bracket.
    grid : ndarray
        1D monotonic grid.

    Returns
    -------
    x_lo : float
        Lower bracket (or boundary if x <= min(grid)).
    x_hi : float
        Upper bracket (or boundary if x >= max(grid)).
    """
    grid = np.asarray(grid, float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("'grid' must be a non-empty 1D array.")

    if grid[0] > grid[-1]:
        raise ValueError("'grid' should be in increasing order.")

    if x <= grid[0]:
        return grid[0], grid[0]
    if x >= grid[-1]:
        return grid[-1], grid[-1]

    idx = np.searchsorted(grid, x)
    return grid[idx - 1], grid[idx]



#################### Image processing/analysis functions ######################

def box_convolution(data, size_core, mode="sum"):
    """
    Fast box-convolution (sum or mean) with NaN handling.

    For each pixel (and each channel if 3D), computes either the sum or mean of all
    finite values inside a centered size_core×size_core window. Windows are truncated
    at borders (no wrap/reflect). If the window contains no finite values, the result
    is NaN.

    Parameters
    ----------
    data : ndarray
        Input array; either (NbLine, NbColumn) or (NbChannel, NbLine, NbColumn). Can contain NaNs.
    size_core : int, optional (default: 3)
        Odd side length of the square window.
    mode : {"sum", "mean"}, optional (default: "sum")
        Whether to compute summed values ("sum") or averaged values ("mean").

    Returns
    -------
    data_conv : ndarray
        Same shape as 'data'. Box-sum or box-mean per pixel (and per channel if 3D).
    """
    if data.ndim not in (2, 3):
        raise ValueError("Input must be 2D (NbLine, NbColumn) or 3D (NbChannel, NbLine, NbColumn).")
    if size_core < 1 or size_core % 2 != 1:
        raise ValueError("size_core must be a positive odd integer.")
    if mode not in ("sum", "mean"):
        raise ValueError("mode must be 'sum' or 'mean'.")

    k = size_core
    if data.ndim == 3:
        kernel = np.ones((1, k, k), dtype=float)  # no mixing along channel axis
    else:
        kernel = np.ones((k, k), dtype=float)

    # Replace NaNs by 0 for sum pass
    valid_mask = np.isfinite(data)
    data_0     = np.where(valid_mask, data, 0.0)

    # Box-sum of values
    summed = convolve(data_0, kernel, mode="constant", cval=0.0)

    if mode == "sum":
        data_conv = summed
    elif mode == "mean":
        # Count of finite samples in each window
        count = convolve(valid_mask.astype(float), kernel, mode="constant", cval=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            data_conv = summed / count

    # Enforce NaN where no valid entries
    data_conv[data_conv==0] = np.nan

    return data_conv



def annular_mask(r_in, r_ext, size, value=np.nan):
    """
    Creates a 2D annular mask with value inside the annulus [r_in, r_ext],
    and 'value' elsewhere.

    Parameters
    ----------
    r_in 
        Inner radius of the annulus (in pixels).
    r_ext 
        Outer radius of the annulus (in pixels).
    size : tuple of int
        (NbLine, NbColumn) shape of the output mask.
    value  or np.nan, optional
        Value to assign outside the annulus.

    Returns
    -------
    mask : 2D ndarray
        Mask with 1s inside the annulus and 'value' elsewhere.
    """
    if r_ext < r_in:
        raise ValueError("'r_ext' must be greater than 'r_in'.")
    NbLine, NbColumn = size
    y0, x0 = NbLine // 2, NbColumn // 2
    Y, X   = np.ogrid[:NbLine, :NbColumn]
    r2     = (Y - y0)**2 + (X - x0)**2
    mask   = np.full((NbLine, NbColumn), value, dtype=float)
    mask[(r2 >= r_in**2) & (r2 <= r_ext**2)] = 1.0
    return mask



def circular_mask(y0, x0, r, size, value=np.nan):
    """
    Build a 2D circular mask centered at (y0, x0).
    Inside the circle (r inclusive) the mask is 1.0, elsewhere 'value'.

    Parameters
    ----------
    y0, x0 : float
        Center coordinates in pixels.
    r : float
        Radius in pixels (>= 0).
    size : tuple[int, int]
        Output shape (NbLine, NbColumn).
    value : float, optional
        Value outside the circle. Default: np.nan.

    Returns
    -------
    mask : ndarray, shape (NbLine, NbColumn)
    """
    if r < 0:
        raise ValueError("'r' must be >= 0.")
    NbLine, NbColumn = size
    Y, X   = np.ogrid[:NbLine, :NbColumn]
    r2     = (Y - y0)**2 + (X - x0)**2
    mask   = np.full((NbLine, NbColumn), value, dtype=float)
    mask[r2 <= r**2] = 1.0
    return mask



def _compute_crop_slices(Y0, X0, R_crop, NbLine, NbColumn):
    """Internal helper to compute integer crop slices and placement indices."""
    Y0, X0, R_crop = int(Y0), int(X0), int(R_crop)

    y_min = max(0, Y0 - R_crop)
    y_max = min(NbLine, Y0 + R_crop + 1)
    x_min = max(0, X0 - R_crop)
    x_max = min(NbColumn, X0 + R_crop + 1)

    crop_y_min = R_crop - (Y0 - y_min)
    crop_y_max = crop_y_min + (y_max - y_min)
    crop_x_min = R_crop - (X0 - x_min)
    crop_x_max = crop_x_min + (x_max - x_min)

    return (slice(y_min, y_max), slice(x_min, x_max)), (slice(crop_y_min, crop_y_max), slice(crop_x_min, crop_x_max))



def crop(data, Y0=None, X0=None, R_crop=None, return_center=False):
    """
    Crop 2D or 3D data to a square of size (2*R_crop+1) centered on (Y0, X0).
    Missing edges are filled with NaNs.

    If Y0/X0 are not provided, the peak of the median (for 3D) or the image
    (for 2D) is used as center.

    Parameters
    ----------
    data : ndarray
        2D (NbLine, NbColumn) or 3D (N, NbLine, NbColumn) array.
    Y0, X0 : int or None
        Center coordinates in pixels. If None, determined automatically.
    R_crop : int or None
        Crop radius. If None, uses the maximum possible radius to keep the
        center in view.
    return_center : bool
        If True, also return (Y0, X0).

    Returns
    -------
    data_crop : ndarray
        Cropped array with NaNs outside original data footprint.
    (Y0, X0) : tuple[int, int], optional
        Returned if 'return_center' is True.
    """
    if data.ndim not in (2, 3):
        raise ValueError("'data' must be 2D or 3D.")

    if Y0 is None or X0 is None:
        A = np.nanmedian(np.nan_to_num(data), axis=0) if data.ndim == 3 else np.nan_to_num(data)
        Y0, X0 = np.unravel_index(np.nanargmax(A), A.shape)

    NbLine, NbColumn = data.shape[-2], data.shape[-1]

    if R_crop is None:
        R_crop = int(max(NbLine - Y0, NbColumn - X0, Y0, X0))

    data_slice, data_crop_slice = _compute_crop_slices(Y0, X0, R_crop, NbLine, NbColumn)

    if data.ndim == 3:
        data_crop = np.full((data.shape[0], 2 * R_crop + 1, 2 * R_crop + 1), np.nan, dtype=float)
        data_crop[:, data_crop_slice[0], data_crop_slice[1]] = data[:, data_slice[0], data_slice[1]]
    else:
        data_crop = np.full((2 * R_crop + 1, 2 * R_crop + 1), np.nan, dtype=float)
        data_crop[data_crop_slice[0], data_crop_slice[1]] = data[data_slice[0], data_slice[1]]

    return (data_crop, int(Y0), int(X0)) if return_center else data_crop

    

def crop_both(data1, data2, Y0=None, X0=None, R_crop=None, return_center=False):
    """
    Crop two arrays identically around the same center/radius (see 'crop').

    Parameters
    ----------
    data1, data2 : ndarray
        2D or 3D arrays to crop in the same way (same spatial shape).
    Y0, X0, R_crop, return_center :
        See 'crop'.

    Returns
    -------
    data1_crop, data2_crop : ndarray
        Cropped versions of 'data1' and 'data2'.
    (Y0, X0) : tuple[int, int], optional
        Returned if 'return_center' is True.
    """
    if data1.shape[-2:] != data2.shape[-2:]:
        raise ValueError("'data1' and 'data2' must share the same spatial shape.")

    # Determine center from data1 if needed
    if Y0 is None or X0 is None:
        A = np.nanmedian(np.nan_to_num(data1), axis=0) if data1.ndim == 3 else np.nan_to_num(data1)
        Y0, X0 = np.unravel_index(np.nanargmax(A), A.shape)

    NbLine, NbColumn = data1.shape[-2], data1.shape[-1]
    if R_crop is None:
        R_crop = int(max(NbLine - Y0, NbColumn - X0, Y0, X0))

    data_slice, data_crop_slice = _compute_crop_slices(Y0, X0, R_crop, NbLine, NbColumn)

    if data1.ndim == 3:
        data1_crop = np.full((data1.shape[0], 2 * R_crop + 1, 2 * R_crop + 1), np.nan, dtype=float)
        data2_crop = np.full((data2.shape[0], 2 * R_crop + 1, 2 * R_crop + 1), np.nan, dtype=float)
        data1_crop[:, data_crop_slice[0], data_crop_slice[1]] = data1[:, data_slice[0], data_slice[1]]
        data2_crop[:, data_crop_slice[0], data_crop_slice[1]] = data2[:, data_slice[0], data_slice[1]]
    else:
        data1_crop = np.full((2 * R_crop + 1, 2 * R_crop + 1), np.nan, dtype=float)
        data2_crop = np.full((2 * R_crop + 1, 2 * R_crop + 1), np.nan, dtype=float)
        data1_crop[data_crop_slice[0], data_crop_slice[1]] = data1[data_slice[0], data_slice[1]]
        data2_crop[data_crop_slice[0], data_crop_slice[1]] = data2[data_slice[0], data_slice[1]]

    return (data1_crop, data2_crop, int(Y0), int(X0)) if return_center else (data1_crop, data2_crop)



def PSF_profile_ratio(PSF, pxscale, size_core, show=True):
    """
    Compute a radial profile (mean flux per annulus) and fraction in the core.

    The PSF is normalized to unit sum first. If 'size_core != 1', a local
    NaN-aware box-average of width 'size_core' is applied before profiling.

    Parameters
    ----------
    PSF : 2D ndarray
        Point-spread function image.
    pxscale : float
        Pixel scale [arcsec/px] or mas/px (used only for the x-axis labeling).
    size_core : int
        Width of the square "core" (in pixels).
    show : bool
        If True, this function simply returns results (no plotting inside).

    Returns
    -------
    profile : ndarray, shape (2, N)
        Row 0: annulus radius (center of the ring) in same unit as pxscale.
        Row 1: mean surface brightness per annulus (per unit area).
    fraction_core : float
        Approximate flux fraction in the core (size_core×size_core around the peak).
    """
    PSF              = np.copy(PSF)
    NbLine, NbColumn = PSF.shape
    y0, x0           = NbLine//2, NbColumn//2

    # Normalize
    PSF_flux = np.nansum(PSF)
    if not np.isfinite(PSF_flux) or PSF_flux <= 0:
        raise ValueError("PSF sum must be positive and finite.")
    PSF /= PSF_flux

    # Optional local NaN-aware box average (size_core x size_core)
    if size_core != 1:
        PSF = box_convolution(data=PSF, size_core=size_core, mode="mean")

    # Fraction in the "core" (approximate central pixel block)
    fraction_core = size_core**2 * PSF[y0, x0]

    # Radial profile using 1-pixel-wide annuli out to N rings
    N       = int(round(np.sqrt( (NbLine/2)**2 + (NbColumn/2)**2 )))
    profile = np.zeros((2, N))
    for r in range(N):
        r_int         = max(1, r - 1) if r > 1 else r
        r_ext         = r
        profile[0, r] = (r_int + r_ext)/2 * pxscale
        profile[1, r] = np.nanmean(PSF * annular_mask(r_int, r_ext, size=(NbLine, NbColumn)))
    profile[1, :] /= pxscale**2
        
    return profile, fraction_core



def register_PSF_ratio(instru, profile, fraction_core, aper_corr, band, strehl, apodizer, coronagraph=None):
    """
    Save a PSF radial profile and metadata to a FITS file.

    The file is stored under: sim_data/PSF/PSF_<instru>/
    with a name that encodes band, coronagraph (if any), strehl and apodizer.

    Parameters
    ----------
    instru : str
        Instrument name.
    profile : ndarray
        Output from 'PSF_profile_ratio' (shape (2, N)).
    fraction_core : float
        Core flux fraction.
    aper_corr : float
        Aperture correction factor to store in the header.
    band, strehl, apodizer : str
        Tags for the filename.
    coronagraph : str or None
        Optional coronagraph tag for the filename.
    """
    hdr       = fits.Header()
    hdr["FC"] = (fraction_core, "Core flux fraction")
    hdr["AC"] = (aper_corr, "Aperture correction")
    
    if coronagraph is None:
        psf_file = f"sim_data/PSF/PSF_{instru}/PSF_{band}_{strehl}_{apodizer}.fits"
    else:
        psf_file = f"sim_data/PSF/PSF_{instru}/PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits"

    fits.writeto(psf_file, profile, header=hdr, overwrite=True)



def fitting_PSF(instru, data, wave, pxscale, model="gaussian", Y0=None, X0=None, sigfactor=5, debug=False):
    """
    Fit the PSF centroid and FWHM for 2D or 3D data using a specified PSF model.

    Parameters
    ----------
    instru : object
        Instrument configuration or identifier used to retrieve telescope diameter.
    data : 2D or 3D ndarray
        PSF image or spectral cube with shape (NbLine, NbColumn) or (NbChannel, NbLine, NbColumn).
    wave : 1D ndarray or None
        Wavelength array in microns. Required for 3D cubes for plotting.
    pxscale 
        Pixel scale in arcsec or mas/pixel.
    model : str, optional
        PSF model to use: 'gaussian', 'moffat', or 'airy_disk'.
    Y0, X0 : int or None
        Initial guess for the centroid position. If None, peak is automatically located.
    sigfactor , optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian
        noise..
    debug : bool, optional
        If True, debug plots are shown during the fit.

    Returns
    -------
    y_center 
        Estimated y-coordinate of the PSF center.
    x_center 
        Estimated x-coordinate of the PSF center.
    fwhm_y 
        Estimated FWHM along the y-axis.
    fwhm_x 
        Estimated FWHM along the x-axis.
    """
    
    import vip_hci as vip
    config_data = get_config_data(instru)
    sep_unit    = config_data["sep_unit"]
    D           = config_data["telescope"]["diameter"]  # diameter  in m
    if wave is None:
        lmin = config_data["lambda_range"]["lambda_min"]
        lmax = config_data["lambda_range"]["lambda_max"]
    else:
        lmin = wave[0]
        lmax = wave[-1]
    lambda0  = (lmin + lmax) / 2 * 1e-6  # wavelength in m
    FWHM_ang = lambda0 / D * rad2arcsec  # FWHM [arcsec]
    if sep_unit == "mas":
        FWHM_ang *= 1000  # FWHM [mas]
    FWHM_px = FWHM_ang / pxscale  # FWHM [pixels]
    dpx     = int(round(FWHM_px)) + 1
    
    if len(data.shape) == 2:
        NbLine, NbColumn = data.shape
        if Y0 is None or X0 is None: # First guess
            Y0, X0 = np.unravel_index(np.nanargmax(data, axis=None), data.shape)
        img         = np.nan_to_num(data)
        ymin        = max(Y0 - 2*dpx    , 0)
        ymax        = min(Y0 + 2*dpx + 1, NbLine-1)
        xmin        = max(X0 - 2*dpx    , 0)
        xmax        = min(X0 + 2*dpx + 1, NbColumn-1)
        img_cropped = img[ymin:ymax, xmin:xmax]
        if not (img_cropped==0).all() and not (img_cropped==np.nan).all():
            if model == "gaussian":
                results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=FWHM_px, fwhmy=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False if sigfactor is not None else False, full_output=True, debug=debug)
                fwhm_y     = results["fwhm_y"][0]
                fwhm_x     = results["fwhm_x"][0]
                fwhm_y_err = results["fwhm_y_err"][0]
                fwhm_x_err = results["fwhm_x_err"][0]
            elif model == "airy_disk":
                results = vip.var.fit_2d.fit_2dairydisk(img_cropped, fwhm=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False, full_output=True, debug=debug)
                fwhm_x     = fwhm_y     = results["fwhm"][0]
                fwhm_x_err = fwhm_y_err = results["fwhm_err"][0]
            elif model == "moffat":
                results = vip.var.fit_2d.fit_2dmoffat(img_cropped, fwhm=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False, full_output=True, debug=debug)
                fwhm_x     = fwhm_y     = results["fwhm"][0]
                fwhm_x_err = fwhm_y_err = results["fwhm_err"][0]
            y0_err  = results["centroid_y_err"][0]
            x0_err  = results["centroid_x_err"][0]
            centroid_y = results["centroid_y"][0]
            centroid_x = results["centroid_x"][0]
            y0 = centroid_y + ymin
            x0 = centroid_x + xmin
    
    elif len(data.shape) == 3:
        NbChannel, NbLine, NbColumn = data.shape
        if Y0 is None or X0 is None: # First guess
            Y0, X0 = np.unravel_index(np.nanargmax(np.nanmedian(np.nan_to_num(data), axis=0), axis=None), data[0].shape)
        y0         = np.zeros((NbChannel)) + np.nan
        x0         = np.zeros((NbChannel)) + np.nan
        y0_err     = np.zeros((NbChannel)) + np.nan
        x0_err     = np.zeros((NbChannel)) + np.nan
        fwhm_y     = np.zeros((NbChannel)) + np.nan
        fwhm_x     = np.zeros((NbChannel)) + np.nan
        fwhm_y_err = np.zeros((NbChannel)) + np.nan
        fwhm_x_err = np.zeros((NbChannel)) + np.nan
        for i in range(NbChannel):
            img         = np.nan_to_num(data[i])
            ymin        = max(Y0 - 2*dpx    , 0)
            ymax        = min(Y0 + 2*dpx + 1, NbLine-1)
            xmin        = max(X0 - 2*dpx    , 0)
            xmax        = min(X0 + 2*dpx + 1, NbColumn-1)
            img_cropped = img[ymin:ymax, xmin:xmax]
            if not (img_cropped==0).all() and not (img_cropped==np.nan).all():
                if model == "gaussian":
                    results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=FWHM_px, fwhmy=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False, full_output=True, debug=(i==NbChannel//2) and debug)
                    fwhm_y[i]     = results["fwhm_y"][0]
                    fwhm_x[i]     = results["fwhm_x"][0]
                    fwhm_y_err[i] = results["fwhm_y_err"][0]
                    fwhm_x_err[i] = results["fwhm_x_err"][0]
                elif model == "airy_disk":
                    results = vip.var.fit_2d.fit_2dairydisk(img_cropped, fwhm=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False, full_output=True, debug=(i==NbChannel//2) and debug)
                    fwhm_y[i]     = fwhm_x[i]     = results["fwhm"][0]
                    fwhm_y_err[i] = fwhm_x_err[i] = results["fwhm_err"][0]
                elif model == "moffat":
                    results = vip.var.fit_2d.fit_2dmoffat(img_cropped, fwhm=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False, full_output=True, debug=(i==NbChannel//2) and debug)
                    fwhm_y[i]     = fwhm_x[i]     = results["fwhm"][0]
                    fwhm_y_err[i] = fwhm_x_err[i] = results["fwhm_err"][0]
                y0_err[i]  = results["centroid_y_err"][0]
                x0_err[i]  = results["centroid_x_err"][0]
                centroid_y = results["centroid_y"][0]
                centroid_x = results["centroid_x"][0]
                y0[i] = centroid_y + ymin
                x0[i] = centroid_x + xmin
            
        if wave is not None:        
            plt.figure(figsize=(14, 7), dpi=300)
            plt.suptitle(f"PSF fitting with {model.replace('_', ' ')} model", fontsize=22, fontweight="bold", color="#333")                
            plt.subplot(1, 2, 1)
            plt.axhline(0, color="k", linestyle="--")
            plt.fill_between(wave, y0-np.nanmean(y0) - y0_err, y0-np.nanmean(y0) + y0_err, color="crimson", alpha=0.3)
            plt.fill_between(wave, x0-np.nanmean(x0) - x0_err, x0-np.nanmean(x0) + x0_err, color="royalblue", alpha=0.3)
            plt.plot(wave, y0-np.nanmean(y0), label=f"y ("r"$\mu$"f" = {round(np.nanmean(y0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(y0_err), 3)} px)", linewidth=2.5, color="crimson")
            plt.plot(wave, x0-np.nanmean(x0), label=f"x ("r"$\mu$"f" = {round(np.nanmean(x0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(x0_err), 3)} px)", linewidth=2.5, color="royalblue")
            plt.ylabel("PSF Centroid shift [px]", fontsize=16)
            plt.xlabel("Wavelength [µm]", fontsize=16)
            plt.title("Centroid Position as a Function of Wavelength", fontsize=18, pad=10)
            plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xlim(lmin, lmax)
            plt.minorticks_on()
            plt.subplot(1, 2, 2)
            if model == "gaussian":
                plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="crimson", alpha=0.3)
                plt.fill_between(wave, fwhm_x - fwhm_x_err, fwhm_x + fwhm_x_err, color="royalblue", alpha=0.3)
                plt.plot(wave, fwhm_y, label=f"y FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="crimson")
                plt.plot(wave, fwhm_x, label=f"x FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_x), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_x_err), 2)} px)", linewidth=2.5, color="royalblue")
            else:
                plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="black", alpha=0.3)
                plt.plot(wave, fwhm_y, label=f"FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="black")
            plt.ylabel("PSF FWHM [px]", fontsize=16)
            plt.xlabel("Wavelength [µm]", fontsize=16)
            plt.title("FWHM as a Function of Wavelength", fontsize=18, pad=10)
            plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xlim(lmin, lmax)
            plt.minorticks_on()
            if pxscale is not None:
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.set_ylabel(f"PSF FWHM [{sep_unit}]", fontsize=16, labelpad=20, rotation=270)
                ax2.tick_params(axis='y')
                ax2.minorticks_on() 
                ymin, ymax = ax1.get_ylim()
                ax2.set_ylim(pxscale*ymin, pxscale*ymax)    
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
    return y0, x0, fwhm_y, fwhm_x



def align_HC_bench_psf(cube_desat, cube, model="airy_disk", dpx=5, wave=None, pxscale=None, sigfactor=5):
    """
    Align each slice of the HC bench cubes. 
    """
    import vip_hci as vip
    NbChannel, NbLine, NbColumn = cube_desat.shape
    cube_desat_med = np.nanmedian(cube_desat, axis=0)
    Y0, X0         = np.unravel_index(np.argmax(np.nan_to_num(cube_desat_med), axis=None), cube_desat_med.shape)
    aligned_cube       = np.zeros_like(cube)
    aligned_cube_desat = np.zeros_like(cube_desat)
    y0                 = np.zeros((NbChannel))
    x0                 = np.zeros((NbChannel))
    y0_err             = np.zeros((NbChannel))
    x0_err             = np.zeros((NbChannel))
    fwhm_y             = np.zeros((NbChannel))
    fwhm_x             = np.zeros((NbChannel))
    fwhm_y_err         = np.zeros((NbChannel))
    fwhm_x_err         = np.zeros((NbChannel))
    for i in range(NbChannel):
        img         = np.nan_to_num(cube_desat[i])
        ymin        = max(Y0 - 2*dpx    , 0)
        ymax        = min(Y0 + 2*dpx + 1, NbLine-1)
        xmin        = max(X0 - 2*dpx    , 0)
        xmax        = min(X0 + 2*dpx + 1, NbColumn-1)
        img_cropped = img[ymin:ymax, xmin:xmax]
        if model == "gaussian":
            results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=2.77, fwhmy=2.69, threshold=True if sigfactor is not None else False, full_output=True, debug=(i==NbChannel//2))
            fwhm_y[i]     = results["fwhm_y"][0]
            fwhm_x[i]     = results["fwhm_x"][0]
            fwhm_y_err[i] = results["fwhm_y_err"][0]
            fwhm_x_err[i] = results["fwhm_x_err"][0]
        elif model == "airy_disk":
            results = vip.var.fit_2d.fit_2dairydisk(img_cropped, fwhm=2.81, threshold=True if sigfactor is not None else False, full_output=True, debug=(i==NbChannel//2))
            fwhm_y[i]     = results["fwhm"][0]
            fwhm_y_err[i] = results["fwhm_err"][0]
        elif model == "moffat":
            results = vip.var.fit_2d.fit_2dmoffat(img_cropped, fwhm=2.81, threshold=True if sigfactor is not None else False, full_output=True, debug=(i==NbChannel//2))
            fwhm_y[i]     = results["fwhm"][0]
            fwhm_y_err[i] = results["fwhm_err"][0]
        y0_err[i]  = results["centroid_y_err"][0]
        x0_err[i]  = results["centroid_x_err"][0]
        centroid_y = results["centroid_y"][0]
        centroid_x = results["centroid_x"][0]
        y0[i] = centroid_y + ymin
        x0[i] = centroid_x + xmin
        dy = Y0 - y0[i]
        dx = X0 - x0[i]
        aligned_cube[i]       = shift(cube[i], shift=(dy, dx), order=3, cval=np.nan)
        aligned_cube_desat[i] = shift(cube_desat[i], shift=(dy, dx), order=3, cval=np.nan)
    aligned_cube             = fill_invalid_by_nearest(aligned_cube)
    aligned_cube_desat       = fill_invalid_by_nearest(aligned_cube_desat)
    # Plot
    if wave is not None:   
        plt.figure(figsize=(14, 7), dpi=300)
        plt.suptitle(f"PSF fitting with {model.replace('_', ' ')} model", fontsize=22, fontweight="bold", color="#333")                
        plt.subplot(1, 2, 1)
        plt.axhline(0, color="k", linestyle="--")
        plt.fill_between(wave, y0-np.nanmean(y0) - y0_err, y0-np.nanmean(y0) + y0_err, color="crimson", alpha=0.3)
        plt.fill_between(wave, x0-np.nanmean(x0) - x0_err, x0-np.nanmean(x0) + x0_err, color="royalblue", alpha=0.3)
        plt.plot(wave, y0-np.nanmean(y0), label=f"y ("r"$\mu$"f" = {round(np.nanmean(y0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(y0_err), 3)} px)", linewidth=2.5, color="crimson")
        plt.plot(wave, x0-np.nanmean(x0), label=f"x ("r"$\mu$"f" = {round(np.nanmean(x0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(x0_err), 3)} px)", linewidth=2.5, color="royalblue")
        plt.ylabel("PSF Centroid shift [px]", fontsize=16)
        plt.xlabel("Wavelength [µm]", fontsize=16)
        plt.title("Centroid Position as a Function of Wavelength", fontsize=18, pad=10)
        plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim(wave[0], wave[-1])
        plt.minorticks_on()
        plt.subplot(1, 2, 2)
        if model == "gaussian":
            plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="crimson", alpha=0.3)
            plt.fill_between(wave, fwhm_x - fwhm_x_err, fwhm_x + fwhm_x_err, color="royalblue", alpha=0.3)
            plt.plot(wave, fwhm_y, label=f"y FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="crimson")
            plt.plot(wave, fwhm_x, label=f"x FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_x), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_x_err), 2)} px)", linewidth=2.5, color="royalblue")
        else:
            plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="black", alpha=0.3)
            plt.plot(wave, fwhm_y, label=f"FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="black")
        plt.ylabel("PSF FWHM [px]", fontsize=16)
        plt.xlabel("Wavelength [µm]", fontsize=16)
        plt.title("FWHM as a Function of Wavelength", fontsize=18, pad=10)
        plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim(wave[0], wave[-1])
        plt.minorticks_on()
        if pxscale is not None:
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.set_ylabel(f"PSF FWHM [{sep_unit}]", fontsize=16, labelpad=20, rotation=270)
            ax2.tick_params(axis='y')
            ax2.minorticks_on() 
            ymin, ymax = ax1.get_ylim()
            ax2.set_ylim(pxscale*ymin, pxscale*ymax)    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    return aligned_cube_desat, aligned_cube



def fill_invalid_by_nearest(psf):
    """
    Fill NaNs and zeros in a 2D image or 3D cube with the nearest non-zero and non-NaN pixel value.

    Parameters
    ----------
    psf : ndarray
        A 2D image or a 3D cube (shape: N, NbLine, NbColumn) potentially containing NaNs or zeros.

    Returns
    -------
    psf_filled : ndarray
        The input image/cube with invalid values replaced by the nearest valid pixel.
    """
    psf = psf.copy()

    if psf.ndim == 2:
        psf = _fill_nearest_2d(psf)
    elif psf.ndim == 3:
        for i in range(psf.shape[0]):
            psf[i] = _fill_nearest_2d(psf[i])
    else:
        raise ValueError("Input must be a 2D image or 3D cube (shape = (N, NbLine, NbColumn))")

    return psf



def _fill_nearest_2d(image):
    """
    Internal helper function to fill invalid values in a 2D image.

    Parameters
    ----------
    image : 2D ndarray
        A single 2D frame with potential NaNs or zeros.

    Returns
    -------
    filled_image : 2D ndarray
        The same image with invalid values replaced by nearest valid ones.
    """
    # Mask of invalid pixels: either NaN or non-positive
    invalid_mask = np.isnan(image) | (image <= 0)

    # If there are invalid pixels, apply nearest-neighbor filling
    if np.any(invalid_mask):
        # Get the distance transform and indices of nearest valid pixels
        dist, indices = distance_transform_edt(invalid_mask, return_indices=True)

        # Retrieve coordinates of the nearest valid pixel for each invalid pixel
        nearest_y = indices[0][invalid_mask]
        nearest_x = indices[1][invalid_mask]

        # Replace invalid values with the nearest valid values
        image[invalid_mask] = image[nearest_y, nearest_x]

    return image



def fill_invalid_by_symmetry(psf):
    """
    Fill NaN or non-positive values in a centrally symmetric PSF using its mirrored counterpart.
    
    For 2D input: PSF must have odd dimensions (NbLine, NbColumn).
    For 3D input: PSF shape must be (NbChannel, NbLine, NbColumn), and NbLine, NbColumn must be odd.

    Parameters:
        psf : np.ndarray
            2D or 3D PSF array.

    Returns:
        filled_psf : np.ndarray
            PSF with invalid values filled by central symmetry.
    """
    psf = psf.copy()
    
    if psf.ndim == 2:
        psf = psf[np.newaxis, ...]  # Add dummy channel axis

    NbChannel, NbLine, NbColumn = psf.shape

    assert NbLine % 2 == 1 and NbColumn % 2 == 1, "PSF dimensions must be odd in spatial axes"

    y_center = NbLine // 2
    x_center = NbColumn // 2

    for k in range(NbChannel):
        mask = np.isnan(psf[k]) | (psf[k] <= 0)

        for y in range(NbLine):
            for x in range(NbColumn):
                if mask[y, x]:
                    # Mirror coordinates
                    y_mirror = 2 * y_center - y
                    x_mirror = 2 * x_center - x
                    # Check if mirror pixel is within bounds
                    if 0 <= y_mirror < NbLine and 0 <= x_mirror < NbColumn:
                        value = psf[k, y_mirror, x_mirror]
                        if not np.isnan(value) and value > 0:
                            psf[k, y, x] = value

    return psf[0] if psf.shape[0] == 1 else psf



def shift_fft(image, shift_vals, pad=True):
    """
    Perform a subpixel shift of an image using FFT-based phase translation.

    This method is highly accurate and preserves both the total flux and 
    the spatial structure of the image (e.g., PSF). Zero-padding is applied 
    by default to avoid wrap-around artifacts from the FFT.

    Parameters
    ----------
    image : 2D ndarray
        Input image to shift (e.g., a PSF).
    shift_vals : tuple of float (dy, dx)
        Subpixel shift to apply along the (y, x) axes.
    pad : bool, optional (default=True)
        If True, zero-padding is applied before the shift to prevent 
        periodic boundary artifacts. The result is cropped back to 
        the original image size.

    Returns
    -------
    shifted_image : 2D ndarray
        The shifted image with the same shape as the input.
    """
    if pad:
        pad_y = image.shape[0] // 2
        pad_x = image.shape[1] // 2
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
        shifted_fft = fourier_shift(fftn(padded), shift_vals)
        shifted_padded = np.real(ifftn(shifted_fft))
        # Crop back to original size
        shifted_image = shifted_padded[pad_y:pad_y + image.shape[0], pad_x:pad_x + image.shape[1]]
    else:
        shifted_fft = fourier_shift(fftn(image), shift_vals)
        shifted_image = np.real(ifftn(shifted_fft))

    return shifted_image



#################### Spectral processing/analysis functions ###################

def air2vacuum(wavelength):
    """
    Convert wavelength from air to vacuum (wavelength in µm)
    """
    s = 1e4 / (wavelength * 1e4) # wavelength in Angstrom
    n =  1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wavelength * n



def gaussian_lowpass_nanaware(y, sigma, eps: float = 1e-12, mode: str = "reflect", truncate: float = 4.0):
    """
    NaN-aware Gaussian low-pass filtering (1D).

    Computes a locally normalized convolution:
        out = (G_sigma ⋆ (y * M)) / (G_sigma ⋆ M)
    where M = 1 on finite samples of 'y' and 0 on NaNs, and G_sigma is a
    unit-area Gaussian kernel of standard deviation 'sigma' (in samples).
    This avoids bias near gaps and at boundaries while preserving
    DC/flux locally when sufficient support exists.

    Parameters
    ----------
    y : array_like, shape (N,)
        Input 1D signal; NaNs are treated as missing data.
    sigma : float
        Standard deviation of the Gaussian kernel (in samples). Must be > 0.
    eps : float, optional
        Minimum denominator to accept local normalization. Locations where
        (G_sigma ⋆ M) <= eps are set to NaN. Default 1e-6.
    mode : {'reflect','nearest','mirror','constant','wrap'}, optional
        Boundary handling passed to 'gaussian_filter1d'. Default 'reflect'.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default 4.0.

    Returns
    -------
    out : ndarray, shape (N,)
        Low-pass filtered signal with NaNs where local support is insufficient.

    Raises
    ------
    ValueError
        If 'sigma' is non-positive or 'y' is not 1D.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("'y' must be a 1D array.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("'sigma' must be a positive finite float.")
    if eps < 0:
        raise ValueError("'eps' must be non-negative.")

    mask = np.isfinite(y).astype(float)
    y0   = np.nan_to_num(y, nan=0.0)

    num = gaussian_filter1d(y0 * mask, sigma=sigma, mode=mode, truncate=truncate)
    den = gaussian_filter1d(mask,     sigma=sigma, mode=mode, truncate=truncate)

    out = np.where(den > eps, num / den, np.nan)
    return out




def PCA_subtraction(S_res, N_PCA, y0=None, x0=None, size_core=None, PCA_annular=False, PCA_mask=False, scree_plot=False, PCA_plots=False, PCA_plots_PDF=False, path_PDF=None, wave=None, R=None, pxscale=None):
    """
    Perform PCA subtraction on the input data cube.

    This function applies Principal Component Analysis (PCA) to subtract signal
    components from the input data cube (S_res). Optionally, it masks out regions 
    around a specified planet location before performing the PCA. It can also display 
    plots for the first few PCA components interactively, and if a file path is provided, 
    it saves plots of all PCA components to a PDF (one per page).

    Parameters
    ----------
    S_res : numpy.ndarray
        Input data cube with dimensions (NbChannel, NbColumn, NbLine).
    N_PCA : int
        Number of PCA components to subtract. If 0, no PCA subtraction is performed.
    y0 : int, optional
        Y-coordinate of the planet location.
    x0 : int, optional
        X-coordinate of the planet location.
    size_core : int, optional
        Size parameter for masking around the planet (FWHM size in px).
    PCA_annular : bool, optional
        If True, applies PCA on an annular mask around the planet location.
    PCA_mask : bool, optional
        If True, applies a rectangular mask (of size_core x size_core) around the planet location.
    scree_plot : bool, optional
        If True, displays a scree plot of the eigenvalues.
    PCA_plots : bool, optional
        If True, displays plots for up to the first 5 PCA components.
    PCA_plots_PDF : bool, optional
        If True, saves plots of all PCA components into a PDF.
    path_PDF : str, optional
        If provided, saves plots of all PCA components into a PDF at this path.
    wave : numpy.ndarray, optional
        Array of wavelengths used for plotting the PCA component curves.
    R : int or float, optional
        Resolution parameter used for plotting the Power Spectral Density (PSD).
    pxscale , optional
        Pixel scale used for plotting the correlation maps.

    Returns
    -------
    S_res_sub : numpy.ndarray
        Data cube after PCA subtraction.
    pca : PCA object or None
        The fitted PCA object if N_PCA is not 0; otherwise, None.
    """
    
    if N_PCA != 0:
        from sklearn.decomposition import PCA
        NbChannel, NbColumn, NbLine = S_res.shape # Retrieve the shape of the data cube
        pca = PCA(n_components=N_PCA)
        S_res_wo_planet = np.copy(S_res) # Create a copy of the input data for masking purposes
        
        # If planet coordinates are provided, apply the masks if specified
        if y0 is not None and x0 is not None:
            if PCA_annular:
                # Calculate the separation from the center to the planet
                planet_sep = int(round(np.sqrt((y0 - NbLine // 2) ** 2 + (x0 - NbColumn // 2) ** 2)))
                # Apply an annular mask with a given core size
                S_res_wo_planet *= annular_mask(max(1, planet_sep - size_core - 1), planet_sep + size_core, value=np.nan, size=(NbLine, NbColumn))
            if PCA_mask:
                # Apply a circular mask around the planet location
                planet_mask                        = circular_mask(y0, x0, r=size_core, size=(NbLine, NbColumn))
                S_res_wo_planet[:, planet_mask==1] = np.nan
        
        # Reshape the cube to 2D (pixels x channels) and replace NaNs with the mean value of their spectral channel
        S_res_wo_planet = np.reshape(S_res_wo_planet, (NbChannel, NbColumn * NbLine)).T        
        col_mean        = np.nanmean(S_res_wo_planet, axis=0)        
        bad_cols        = ~np.isfinite(col_mean)   # True si NaN/inf
        if np.any(bad_cols):
            col_mean[bad_cols] = 0.0
        inds = np.isnan(S_res_wo_planet)
        if inds.any():
            S_res_wo_planet[inds] = np.take(col_mean, np.where(inds)[1])        
        S_res_wo_planet = np.nan_to_num(S_res_wo_planet)

        # Fit the PCA on the masked data
        pca.fit(S_res_wo_planet)
        
        # Prepare the data for subtraction (reshape and replace NaNs)
        nan_mask            = np.isnan(S_res)
        S_res_sub           = np.reshape(np.copy(S_res), (NbChannel, NbColumn * NbLine)).transpose()
        inds_sub            = np.where(np.isnan(S_res_sub))
        S_res_sub[inds_sub] = np.take(col_mean, inds_sub[1])
        
        # Transform and inverse transform to obtain the PCA model reconstruction
        X = pca.transform(S_res_sub)
        X = pca.inverse_transform(X)
        
        # Subtract the PCA reconstruction from the original data
        S_res_sub = (S_res_sub - X).transpose()
        S_res_sub = np.reshape(S_res_sub, (NbChannel, NbColumn, NbLine))
        S_res_sub[nan_mask] = np.nan
        
        # ---------------------------
        # PCA plots
        # ---------------------------
        if PCA_plots:
            # Display up to the first 5 components interactively
            from src.spectrum import Spectrum, get_psd
            Nk      = min(N_PCA, 5)
            cmap    = plt.get_cmap("Spectral", Nk)
            extent  = [-NbColumn/2*pxscale, NbColumn/2*pxscale, -NbLine/2*pxscale, NbLine/2*pxscale]
            fig, ax = plt.subplots(Nk, 3, figsize=(16, Nk * 3), sharex='col', sharey='col', layout="constrained", gridspec_kw={'wspace': 0.05, 'hspace': 0}, dpi=300)
            for k in range(Nk):
                # Retrieve the k-th PCA component
                pca_comp = pca.components_[k]
                
                # First column: PCA component curve
                ax[k, 0].plot(wave, pca_comp, c=cmap(k), label=f"$n_k$ = {k+1}")
                ax[k, 0].legend(fontsize=14, loc="upper center")
                if k == Nk - 1:
                    ax[k, 0].set_xlim(wave[0], wave[-1])
                    ax[k, 0].set_xlabel("Wavelength [µm]", fontsize=14)
                ax[k, 0].set_ylabel("Modulation (normalized)", fontsize=14)
                ax[k, 0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                ax[k, 0].minorticks_on()
                
                # Second column: Plot the Power Spectral Density (PSD)
                res, psd = get_psd(wave, pca_comp, R=R, smooth=0)
                ax[k, 1].plot(res, psd, c=cmap(k))
                if k == Nk - 1:
                    ax[k, 1].set_xlim(10, R)
                    ax[k, 1].set_xlabel("Resolution", fontsize=14)
                    ax[k, 1].set_xscale('log')
                    ax[k, 1].set_yscale('log')
                ax[k, 1].set_ylabel("PSD", fontsize=14)
                ax[k, 1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                ax[k, 1].minorticks_on()

                # Third column: Vectorized computation and plot of the correlation map
                norm       = np.sqrt(np.nansum(S_res**2, axis=0))
                corr       = np.nansum(S_res * pca_comp[:, None, None], axis=0)
                CCF        = np.zeros((NbLine, NbColumn)) + np.nan
                valid      = norm != 0
                CCF[valid] = corr[valid] / norm[valid]
                cax  = ax[k, 2].imshow(CCF, extent=extent, zorder=3)
                cbar = fig.colorbar(cax, ax=ax[k, 2], orientation='vertical', shrink=0.8)
                cbar.set_label("Correlation", fontsize=14, labelpad=20, rotation=270)
                if k == Nk - 1:
                    ax[k, 2].set_xlabel('x offset (in ")', fontsize=14)
                ax[k, 2].set_ylabel('y offset (in ")', fontsize=14)
                ax[k, 2].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                ax[k, 2].minorticks_on()

            plt.show()
        
        # ---------------------------
        # Save PCA component plots to a PDF
        # ---------------------------
        if PCA_plots_PDF and path_PDF is not None:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf = PdfPages(path_PDF)
            # Create a colormap with N_PCA distinct colors
            N = N_PCA # min(N_PCA, 100)
            cmap_pdf = get_cmap("Spectral", N)
            from src.spectrum import Spectrum
            for k in tqdm(range(N), desc="Saving PCA components in PDF"):
                pca_comp = pca.components_[k]
                
                fig, ax = plt.subplots(1, 3, figsize=(16, 3), dpi=100)
                
                # First column: PCA component curve
                ax[0].plot(wave, pca_comp, c=cmap_pdf(k), label=f"$n_k$ = {k+1}")
                ax[0].legend(fontsize=14, loc="upper center")
                ax[0].set_xlim(wave[0], wave[-1])
                ax[0].set_xlabel("wavelength [µm]", fontsize=14)
                ax[0].set_ylabel("modulation (normalized)", fontsize=14)
                ax[0].grid(True)
                
                # Second column: PSD plot
                res, psd = get_psd(wave, pca_comp, R=R, smooth=0)
                ax[1].plot(res, psd, c=cmap_pdf(k))
                ax[1].set_xlim(10, R)
                ax[1].set_xlabel("resolution", fontsize=14)
                ax[1].set_xscale('log')
                ax[1].set_yscale('log')
                ax[1].set_ylabel("PSD", fontsize=14)
                ax[1].grid(True)
                
                # Third column: correlation map
                numerator = np.nansum(S_res * pca_comp[:, None, None], axis=0)
                denom = np.sqrt(np.nansum(S_res ** 2, axis=0))
                mask = ~np.all(np.isnan(S_res), axis=0)
                CCF = np.full(S_res.shape[1:], np.nan)
                CCF[mask] = np.nan_to_num(numerator[mask]) / denom[mask]
                cax = ax[2].imshow(CCF, extent=[-(CCF.shape[0] + 1) // 2 * pxscale, (CCF.shape[0]) // 2 * pxscale, -(CCF.shape[1] - 2) // 2 * pxscale, (CCF.shape[1]) // 2 * pxscale], zorder=3)
                cbar = fig.colorbar(cax, ax=ax[2], orientation='vertical', shrink=0.8)
                cbar.set_label("correlation", fontsize=14, labelpad=20, rotation=270)
                ax[2].set_xlabel('x offset (in ")', fontsize=14)
                ax[2].set_ylabel('y offset (in ")', fontsize=14)
                ax[2].grid(True)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
            pdf.close()
            print(f"PCA PDF saved in {path_PDF}")

        # ---------------------------
        # Scree plot (Eigenvalues)
        # ---------------------------
        # - pca.explained_variance_ gives the eigenvalues (λ_k),
        #   i.e. the variance explained by each principal component.
        #
        # Interpretation:
        #   - Large eigenvalue = this component captures a strong correlated
        #     structure in the data (e.g. speckles, fringing, flat-field residuals).
        #   - Small eigenvalue = this component only explains weak fluctuations,
        #     often corresponding to noise.
        #
        # The scree plot shows how much variance each component explains.
        # - At the beginning: very large eigenvalues → dominant systematics.
        # - Elbow region: transition from structured variance to weaker effects.
        # - Tail (flat): very small eigenvalues → mostly white noise.
        if scree_plot:
            eigenvalues = pca.explained_variance_
            plt.figure(figsize=(10, 6), dpi=300)
            plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', c="gray")
            plt.xlabel('Principal Components', fontsize=14)
            plt.ylabel('Eigenvalues (variance explained)', fontsize=14)
            plt.title('Scree Plot', fontsize=16, fontweight="bold")
            plt.yscale('log')
            plt.xlim(1, N_PCA)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            xlim = plt.xlim()
            ylim = plt.ylim()
            plt.text((xlim[0] + xlim[1]) / 2, ylim[1]*0.5,"Systematics-dominated", fontsize=12, color="crimson", va="center", ha="center")
            plt.text((xlim[0] + xlim[1]) / 2, ylim[0]*2,"White-noise-dominated", fontsize=12, color="royalblue", va="center", ha="center")
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
            
    elif N_PCA == 0:
        S_res_sub = np.copy(S_res)
        pca       = None
        
    return S_res_sub, pca



def cut_spectral_frequencies(input_flux, R, Rmin, Rmax, filter_type='empirical', show=False, target_name=None, force_new_calc=False):
    """
    Removes spectral fringes from the input flux by setting specific frequency components to zero or applying a smoother filter.

    Parameters:
    input_flux (array-like): The flux values to be filtered, which may contain NaN values.
    R (float): The spectral resolution of the data.
    Rmin (float): The lower bound of the fringe domain in resolution units.
    Rmax (float): The upper bound of the fringe domain in resolution units.
    filter_type (str): The type of smoother filter to apply, either 'gaussian', 'step' or 'empirical'.
    data (bool): In order to estimate the empirical filter response profile from data.

    Returns:
    array-like: The filtered flux with spectral fringes removed or smoothed.
    """
    # Ensure there are no NaN values in the input flux
    valid_data = np.isfinite(input_flux)
    # Perform FFT on the valid part of the input flux
    fft_values = np.fft.fft(input_flux[valid_data])
    frequencies = np.fft.fftfreq(len(input_flux[valid_data]))
    # Convert frequencies to resolution scale
    res_values = frequencies * R * 2
    if filter_type == 'gaussian': # Apply a Gaussian filter to smoothly reduce the amplitudes in the fringe domain
        sigma = (Rmax - Rmin) / (2 * np.sqrt(2 * np.log(2)))  # assuming FWHM = Rmax - Rmin
        n = 1 # Control the sharpness of the filter, n > 1 for super-gaussian
        gaussian_filter = (1 - np.exp(-0.5 * ((res_values - (Rmin + Rmax) / 2) / sigma) ** (2*n))) / 2 + (1 - np.exp(-0.5 * ((res_values + (Rmin + Rmax) / 2) / sigma) ** (2*n))) / 2
        filter_response = gaussian_filter
    elif filter_type == 'step': # Apply a window filter for a sharp cutoff of the fringe domain
        step_filter = np.ones_like(res_values)
        step_filter[(Rmax > np.abs(res_values)) & (np.abs(res_values) > Rmin)] = 0
        filter_response = step_filter
    elif filter_type == 'empirical': 
        try : # Opening existing filter response profile
            if force_new_calc:
                raise ValueError("force_new_calc = True")
            empirical_res_values, empirical_filter_response = fits.getdata(f"utils/empirical_filter_response/{target_name}_{R}_{Rmin}_{Rmax}_empirical_filter_response.fits")
            f = interp1d(empirical_res_values, empirical_filter_response, bounds_error=False, fill_value="extrapolate")
            filter_response = f(res_values)
        except Exception as e: # Calculating empirical filter response profile (needs to be done once on data!!! and not on models)
            print(f"empirical cut_spectral_frequencies(): {e}")
            # Calcul de la PSD des données
            psd_values = np.abs(fft_values)**2
            # Lissage de la PSD des données
            psd_values_LF = gaussian_filter1d(psd_values, sigma=100)
            # Interpolation de la PSD en masquant le pic des franges
            psd_values_LF_interp = np.copy(psd_values_LF)
            psd_values_LF_interp[(Rmax > np.abs(res_values)) & (np.abs(res_values) > Rmin)] = np.nan
            f = interp1d(res_values[np.isfinite(psd_values_LF_interp)], psd_values_LF_interp[np.isfinite(psd_values_LF_interp)], bounds_error=False, fill_value=np.nan)
            psd_values_LF_interp = f(res_values)
            # Calcul de la forme du pic des franges (psd_values_LF = PSD lissée avec le pic et psd_values_LF_interp PSD lissée sans le pic)
            peak = np.abs(psd_values_LF - psd_values_LF_interp)
            # Normalisation du pic (on s'intéresse uniquement à la forme)
            peak /= np.nanmax(peak)
            # Lissage de la forme du pic trouvé (semble mieux marcher), sigma=400 est une valeur qui fonctionne bien pour les données de Beta Pic c, mais il faudra surement modifier cette valeur pour d'autres données
            if "beta_Pic_c" in target_name:
                sigma = 666
            else:
                sigma = 1000
            filter_response = gaussian_filter1d(1 - peak, sigma=sigma)
            # Sauvegarde du profil trouvé
            empirical_filter_response = np.zeros((2, len(filter_response)))
            empirical_filter_response[0] = res_values ; empirical_filter_response[1] = filter_response
            fits.writeto(f"utils/empirical_filter_response/{target_name}_{R}_{Rmin}_{Rmax}_empirical_filter_response.fits", empirical_filter_response, overwrite=True)
            plt.figure(dpi=300)
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(psd_values), label="PSD des données")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(psd_values_LF), label="PSD lissée (avec le pic)")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(psd_values_LF_interp), label="PSD lissée (sans le pic)")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(np.abs(psd_values_LF - psd_values_LF_interp)), label="Estimation du pic")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(filter_response), label="normalisation, lissage et inversion du pic")
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.xlabel("resolution")
            plt.ylabel("PSD")
            plt.show()
    else:
        raise ValueError("Invalid filter_type. Use 'gaussian', 'lorentzian', or 'step'.")
    if show: # Plot the filter and the effect on FFT values
        plt.figure(dpi=300)
        plt.plot(np.fft.fftshift(res_values), abs(np.fft.fftshift(fft_values))**2, label='Original PSD')
        plt.plot(np.fft.fftshift(res_values), abs(np.fft.fftshift(filter_response))**2, label=f'{filter_type.capitalize()} Filter Response')
        plt.plot(np.fft.fftshift(res_values), abs(np.fft.fftshift(fft_values * filter_response))**2, label='Filtered PSD')
        plt.xscale('log') ; plt.yscale('log')
        plt.xlabel("resolution") ; plt.ylabel("PSD")
        plt.legend()
        plt.show()
    fft_values               *= filter_response
    filtered_flux             = np.copy(input_flux)
    filtered_flux[valid_data] = np.real(np.fft.ifft(fft_values))
    return filtered_flux



def keep_true_chunks(mask_bool, N):
    """
    Keep only contiguous True runs of length >= N; set shorter True runs to False.

    Parameters
    ----------
    mask_bool : array_like of bool
        1D boolean mask. True values represent the segments to be (potentially) kept.
    N : int
        Minimum run length (in samples) required to preserve a True chunk.

    Returns
    -------
    out : np.ndarray, dtype=bool
        Boolean array of the same shape as 'mask_bool', where only True runs with
        length >= N are preserved; all other True values are set to False.

    Notes
    -----
    - Time complexity is O(n) to detect runs + O(r_short) to clear short runs,
      where r_short is the number of short True runs (usually small).
    - Does not modify the input; returns a new boolean array.
    - If N <= 1, the input mask is returned as-is (converted to np.bool_).

    Examples
    --------
    >>> keep_true_chunks([0, 1, 1, 0, 1, 0, 1, 1, 1], N=2)
    array([False,  True,  True, False, False, False,  True,  True,  True])
    """
    m = np.asarray(mask_bool, dtype=bool)
    n = m.size
    if n == 0:
        return m.copy()
    if N <= 1:
        return m.copy()

    # Find run starts: index 0 and any position where the value changes.
    # Then derive run ends and values from those starts.
    starts = np.r_[0, 1 + np.flatnonzero(m[1:] != m[:-1])]
    ends   = np.r_[starts[1:], n]
    vals   = m[starts]
    lens   = ends - starts

    # Output initially equals the input
    out = m.copy()

    # Identify True runs that are too short
    short_true = (vals & (lens < N))
    if short_true.any():
        # Clear each short True run
        for s, e in zip(starts[short_true], ends[short_true]):
            out[s:e] = False

    return out



########################### Q-Q plot functions ################################
    
def qqplot_CCF(CCF_map, sep_lim, sep_unit, pxscale, band, target_name):
    """
    Q–Q plot of CCF samples split by separation (inner vs outer annulus).

    The CCF map is split into two groups:
      - inner:    separation <= sep_lim
      - outer:    separation >  sep_lim

    Each group is z-scored (NaN-safe) before building Q–Q plots against a
    standard normal distribution.

    Parameters
    ----------
    CCF_map : 2D ndarray
        Cross-correlation function map (NbLine, NbColumn).
    sep_lim : float
        Separation threshold in 'sep_unit' used to split the map.
    sep_unit : {"arcsec", "mas"}
        Unit of separation; used for labeling only.
    pxscale : float
        Pixel scale in 'sep_unit' per pixel (e.g., mas/px or arcsec/px).
    band : str
        Band name for the plot title.
    target_name : str
        Target name for the plot title.
    """
    NbLine, NbColumn = CCF_map.shape
    map1 = CCF_map * annular_mask(0, int(round(sep_lim/pxscale)), size=(NbLine, NbColumn))
    map1 = map1[np.isfinite(map1)]                     # Filtrage nan
    map1 = (map1 - np.nanmean(map1)) / np.nanstd(map1) # Centered normal law (mu=0, std=1)
    map2 = CCF_map * annular_mask(int(round(sep_lim/pxscale))+1, max(NbLine//2, NbColumn//2), size=(NbLine, NbColumn))
    map2 = map2[np.isfinite(map2)]                     # Filtrage nan
    map2 = (map2 - np.nanmean(map2)) / np.nanstd(map2) # Centered normal law (mu=0, std=1)
    # PLOT
    plt.figure(dpi=300, figsize=(6, 6))
    ax = plt.gca()    
    sm.qqplot(map1, line=None, ax=ax, marker='o', markerfacecolor='royalblue', markeredgecolor='royalblue', alpha=0.6, label=f'sep < {sep_lim} {sep_unit}', lw=1)
    sm.qqplot(map2, line=None, ax=ax, marker='o', markerfacecolor='crimson', markeredgecolor='crimson', alpha=0.6, label=f'sep > {sep_lim} {sep_unit}', lw=1)    
    sm.qqline(ax=ax, line='45', fmt='k--', lw=2)    
    plt.title(f"Q-Q plot of the CCF of {target_name} on {band}", fontsize=16, fontweight="bold")
    plt.xlabel("Theoretical quantiles", fontsize=14)
    plt.ylabel("Sample quantiles", fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

  
  
def qqplot2_hirise(CCF_signal, CCF_bkgd, band, target_name):
    """
    Q–Q plots comparing signal CCF distribution to a set of background CCF maps.

    Each background map is independently standardized (NaN-safe) and plotted
    against the standard normal. The signal CCF is also standardized and
    over-plotted for comparison.

    Parameters
    ----------
    CCF_signal : 2D ndarray
        Signal CCF map (NbLine, NbColumn).
    CCF_bkgd : sequence of 2D ndarrays
        Background/noise CCF maps to be shown as a cloud of Q–Q points.
    band : str
        Band name for title.
    target_name : str
        Target name for title.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()
    CCF_signal = CCF_signal[np.isfinite(CCF_signal)] # filtrage nan
    CCF_signal = (CCF_signal-np.mean(CCF_signal))/np.std(CCF_signal) #loi normale centrée (std=1)
    plt.plot([], [], 'o', c="gray", alpha=0.5, label='noise')
    for i in range(len(CCF_bkgd)):
        map_bkgd = CCF_bkgd[i][np.isfinite(CCF_bkgd[i])] # filtrage nan
        map_bkgd = (map_bkgd-np.mean(map_bkgd))/np.std(map_bkgd) # loi normale centrée (normalisée => std=1)
        sm.qqplot(map_bkgd, ax=ax, markerfacecolor='gray', markeredgecolor='gray', alpha=0.1)
    sm.qqline(ax=ax, line='45', fmt='k')
    sm.qqplot(CCF_signal, ax=ax, alpha=1, label='planet')
    plt.title(f'Q-Q plots of the CCF of {target_name.replace("_", " ")} on {band}', fontsize=16, fontweight="bold")
    plt.xlabel("Theoretical quantiles", fontsize=14)
    plt.ylabel("Sample quantiles", fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()



########################## Extracting data functions ##########################

def extract_jwst_data(instru, target_name, band, crop_band=True, outliers=False, sigma_outliers=5, file=None, crop_cube=True, X0=None, Y0=None, R_crop=None, verbose=True):
    
    # Instrument specs
    config_data = get_config_data(instru)
    area        = config_data['telescope']['area'] # collective area m2

    # Opening file
    if file is None :
        if instru=="MIRIMRS" : 
            if "sim" in target_name.lower():
                file = f"data/MIRIMRS/MIRISim/{target_name}_{band}_s3d.fits"
            else :
                file = f"data/MIRIMRS/MAST/{target_name}_ch{band[0]}-shortmediumlong_s3d.fits"
        elif instru=="NIRSpec":
            file = f"data/NIRSpec/MAST/{target_name}_nirspec_{band}_s3d.fits"
        else:
            raise KeyError(f"Unknown instrument {instru}")
    f = fits.open(file)
    
    # Retrieving header values
    hdr0 = f[0].header
    hdr  = f[1].header
    if instru=="MIRIMRS" :
        if "sim" in target_name.lower() or "shortmediumlong" not in file: # MIRISIM are already per band
            exposure_time = f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
        elif crop_band:
            target_name   = hdr0['TARGNAME']
            exposure_time = f[0].header['EFFEXPTM']/3/60 # in mn / Effective exposure time
        else:
            target_name   = hdr0['TARGNAME']
            exposure_time = f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
    elif instru == "NIRSpec":
        target_name   = hdr0['TARGNAME']
        exposure_time = f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
    DIT         = f[0].header['EFFINTTM']/60 # in mn
    pxsteradian = hdr['PIXAR_SR'] # Nominal pixel area in steradians
    pxscale     = hdr['CDELT1']*3600 # data pixel scale in "/px
    step        = hdr['CDELT3'] # delta_lambda in µm
    wave        = (np.arange(hdr['NAXIS3'])+hdr['CRPIX3']-1)*hdr['CDELT3']+hdr['CRVAL3'] # axe de longueur d'onde des données en µm

    # Retrieving data
    cube = f[1].data # en MJy/Steradian (densité "angulaire" de flux mesurée dans chaque pixel)
    err  = f[2].data # erreur sur le flux en MJy/Sr
    if crop_cube:
        cube, err = crop_both(cube, err, X0=X0, Y0=Y0, R_crop=R_crop)
    NbChannel, NbLine, NbColumn = cube.shape
    
    # Converting data in total e- (or ph if crop_band) / px
    cube *= pxsteradian*1e6              # MJy/Steradian => Jy/px
    err  *= pxsteradian*1e6
    cube *= 1e-26                        # Jy/pixel => J/s/m²/Hz/px
    err  *= 1e-26
    for i in range(NbChannel):
        cube[i] *= c/((wave[i]*1e-6)**2) # J/s/m²/Hz/px => J/s/m²/m/px
        err[i]  *= c/((wave[i]*1e-6)**2)
    cube *= step*1e-6                    # J/s/m²/m/px => J/s/m²/px
    err  *= step*1e-6
    for i in range(NbChannel):
        cube[i] *= wave[i]*1e-6/(h*c)    # J/s/m²/px => ph/s/m²/px
        err[i] *= wave[i]*1e-6/(h*c)
    cube *= area                         # ph/s/m²/px => photons/s/px
    err  *= area
    if crop_band: 
        lmin = config_data['gratings'][band].lmin
        lmax = config_data['gratings'][band].lmax
        band_mask = (wave >= config_data['gratings'][band].lmin) & (wave <= config_data['gratings'][band].lmax)
        cube = cube[band_mask]
        err  = err[band_mask]
        wave = wave[band_mask]
        NbChannel = cube.shape[0]
        from src.signal_noise_estimate import get_transmission
        trans = get_transmission(instru, wave, band, tellurics=False, apodizer="NO_SP", strehl="NO_JQ", coronagraph=None)
        AC    = fits.getheader(f"sim_data/PSF/PSF_{instru}/PSF_{band}_NO_JQ_NO_SP.fits")['AC']
        for i in range(NbChannel):
            cube[i] *= trans[i]/AC # AC is obviously already taken into account
            err[i]  *= trans[i]/AC # e-/s/pixel
    else:
        trans = 1
    cube *= exposure_time*60
    err  *= exposure_time*60 # e-/px or ph/px
    
    # Cropping first and last empty slices
    valid_slices = np.array([np.any(np.isfinite(cube[i]) & (cube[i] != 0)) for i in range(NbChannel)]) # Identify valid slices (not all zeros or NaNs)    
    start = np.argmax(valid_slices)  # first True -> first valid slice
    end   = len(valid_slices) - np.argmax(valid_slices[::-1])  # last True + 1 -> last valid slice
    wave = wave[start:end] # Crop only the empty slices at the beginning and the end
    cube = cube[start:end]
    err  = err[start:end]
    if crop_band:
        trans = trans[start:end]    
    NbChannel = cube.shape[0] # Update the number of channels

    # Flagging bad pixels
    if instru == "MIRIMRS": # flagging edge effects
        bad_pixels  = np.sum(cube, axis=(0))
        bad_pixels *= 0
    else:
        bad_pixels = np.zeros((NbLine, NbColumn))        
    cube += bad_pixels
    err  += bad_pixels
    cube[cube==0] = np.nan
    err[err==0]   = np.nan
    
    # Flagging outliers
    if outliers :
        NbChannel, NbLine, NbColumn = cube.shape
        Y = np.reshape(cube, (NbChannel, NbLine*NbColumn))
        Z = np.reshape(err, (NbChannel, NbLine*NbColumn))
        for k in range(Y.shape[1]):
            if not all(np.isnan(Y[:, k])):
                sg = sigma_clip(Y[:, k], sigma=sigma_outliers)
                Y[:, k] = np.array(np.ma.masked_array(Y[:, k], mask=sg.mask).filled(np.nan))
                Z[:, k] = np.array(np.ma.masked_array(Z[:, k], mask=sg.mask).filled(np.nan))
                sg = sigma_clip(Z[:, k], sigma=sigma_outliers)
                Y[:, k] = np.array(np.ma.masked_array(Y[:, k], mask=sg.mask).filled(np.nan))
                Z[:, k] = np.array(np.ma.masked_array(Z[:, k], mask=sg.mask).filled(np.nan))
        cube = Y.reshape((NbChannel, NbLine, NbColumn))
        err  = Z.reshape((NbChannel, NbLine, NbColumn))

    if verbose :
        dl = np.gradient(wave)       # delta lambda array
        R  = np.nanmean(wave/(2*dl)) # calculating the resolution
        print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave.min(), 2)} to {round(wave.max(), 2)} µm with R={R:.0f}):"+"\033[0m")
        print(f" Pixel scale  : {pxscale:.2f} arcsec/px")
        print(f" Exposure time: {exposure_time:.1f} mn (with DIT of {DIT*60:.1f} s)") 
        fields = [  ("TITLE", "{}"),
                    ("CATEGORY", "{}"),
                    ("SCICAT", "{}"),
                    ("DATE-OBS", "{}"),
                    ("TARGPROP", "{}"),
                    ("TARGNAME", "{}"),
                    ("TARGTYPE", "{}"),
                    ("TARGCAT", "{}"),
                    ("TARGDESC", "{}"),
                    ("TARG_RA", "{:.2f}"),
                    ("TARG_DEC", "{:.2f}"),
                    ("PATTTYPE", "{}")]
        for key, fmt in fields:
            try:
                print(f" {key:<13}: {fmt.format(hdr0[key])}")
            except KeyError:
                pass

    return cube, wave, pxscale, err, trans, exposure_time, DIT



###########################################################################################



def compute_filter_variance_factor(N, R_sampling, Rc, filter_type, Rc_init, filter_init_type, Rc_noise, filter_noise_type, Rc_conv, filter_conv_type):
    """
    Compute the fraction of input noise variance that passes through the
    low-pass (LF) and high-pass (HF) branches of a **given** filter, taking
    into account that the input noise may have already been colored by
    earlier filtering steps.

    The method works in the frequency domain. It first builds the input
    transfer function of the noise,
        TF_noise(f) = ∏ H_prior,LF(f),
    as the product of the LF responses of the optional **prior** filters
    (initial coloring, extra low-pass, convolution broadening). It then
    evaluates the variance fractions of the **considered** filter
    (Rc, filter_type) as
        fn_LF = Σ |TF_noise·H_LF|² / Σ |TF_noise|²
        fn_HF = Σ |TF_noise·H_HF|² / Σ |TF_noise|² ,
    where the sums run over the full FFT frequency grid. For ideal splits,
    fn_LF + fn_HF ≈ 1 up to numerical error.

    Parameters
    ----------
    N : int
        FFT length (assumes uniformly sampled data). Also sets the
        frequency resolution used in the calculation.
    R_sampling : float
        Sampling resolving power of the HR grid (Nyquist convention).
        Defines the resolution mapping 'res = 2 * R_sampling * f' with
        'f = np.fft.fftfreq(N)'.
    Rc : float
        Cutoff (in "resolution" units) of the **considered** filter for which
        you want the variance fractions.
    filter_type : {'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep'}
        Shape of the **considered** filter. If 'gaussian' is given, the
        implementation uses the fast discrete version ('gaussian_fast').
    Rc_init : float or None
        Cutoff of an optional **prior** initial low-pass used to color the
        input noise before this step. If None, this stage is skipped.
    filter_init_type : same set as 'filter_type' or None
        Shape of the optional **prior** initial low-pass at 'Rc_init'.
    Rc_noise : float or None
        Cutoff of an optional **prior** extra low-pass. If None, skipped.
    filter_noise_type : same set as 'filter_type' or None
        Shape of the optional **prior** extra low-pass at 'Rc_noise'.
    Rc_conv : float or None
        Cutoff equivalent to an optional **prior** real-space broadening
        (e.g., degrading to a lower effective resolution). If None, skipped.
    filter_conv_type : same set as 'filter_type' or None
        Shape of the optional **prior** broadening filter at 'Rc_conv'.

    Returns
    -------
    fn_HF : float
        Fraction of input noise variance that ends up in the high-pass branch
        of the **considered** filter, given the already-colored input noise.
    fn_LF : float
        Fraction of input noise variance that ends up in the low-pass branch
        of the **considered** filter, given the already-colored input noise.

    Notes
    -----
    * This function assumes **uniform sampling** and computes on the **full**
      FFT grid (no rFFT half-spectrum tricks).
    * Prior filters are applied only to build the input PSD ('|TF_noise|²').
      The **considered** filter ('Rc', 'filter_type') is applied **once** when
      forming 'H_LF'/'H_HF'. Do **not** include it among the prior filters to
      avoid double-counting.
    * Gaussian shapes passed as 'gaussian' are internally mapped to
      'gaussian_fast' for consistency with the discrete implementation.
    * Boundary conditions are effectively circular (FFT). If your pipeline
      uses different real-space boundaries (e.g., reflect), small discrepancies
      can occur near the edges.
    """
    from src.spectrum import _fft_filter_response

    # Frequency grid and "resolution" axis (your convention)
    f   = np.fft.fftfreq(N)          # cycles / HR sample
    res = 2.0 * R_sampling * f

    # Initially white noise
    TF_noise = np.ones(N, dtype=np.complex128) # initially white noise

    # 1) Optional initial HR coloring at Rc_init
    if Rc_init is not None and filter_init_type is not None:
        ftype_init = "gaussian_fast" if (filter_init_type == "gaussian") else filter_init_type
        TF_noise  *= _fft_filter_response(N=N, R=R_sampling, Rc=Rc_init, filter_type=ftype_init)[1]

    # 2) Optional extra low-pass
    if Rc_noise is not None and filter_noise_type is not None:
        ftype_lp  = "gaussian_fast" if (filter_noise_type == "gaussian") else filter_noise_type
        TF_noise *= _fft_filter_response(N=N, R=R_sampling, Rc=Rc_noise, filter_type=ftype_lp)[1]

    # 3) Optional convolution broadening
    if Rc_conv is not None and filter_conv_type is not None:
        ftype_cv  = "gaussian_fast" if (filter_conv_type == "gaussian") else filter_conv_type
        TF_noise *= _fft_filter_response(N=N, R=R_sampling, Rc=Rc_conv, filter_type=ftype_cv)[1]
    
    # Frequency response of the considered filtering
    ftype      = "gaussian_fast" if (filter_type == "gaussian") else filter_type
    H_HF, H_LF = _fft_filter_response(N=N, R=R_sampling, Rc=Rc, filter_type=ftype)
    
    POWER = np.nansum( np.abs(TF_noise)**2 )
    fn_LF = np.nansum( np.abs(TF_noise*H_LF)**2 ) / POWER
    fn_HF = np.nansum( np.abs(TF_noise*H_HF)**2 ) / POWER
    
    return fn_HF, fn_LF



def compute_rebin_variance_factor(lamHR, maskHR, lamLR, dlam, R_sampling, Rc_init, filter_init_type, Rc_noise, filter_noise_type, Rc_conv, filter_conv_type):
    """
    Compute the per-bin variance inflation factor, fn, for top-hat rebinning when
    the high-resolution (HR) noise is temporally/spectrally correlated by prior
    filtering.

    Use this to correct the naive standard deviation of a mean within each
    low-resolution (LR) bin:
        sigma_LR_correct = sigma_LR_naive * sqrt(fn),
    where sigma_LR_naive = sqrt(sum_j sigma_j^2) / N_i assumes i.i.d. HR samples.

    Parameters
    ----------
    lamHR : (N_HR,) array-like of float
        Monotonic HR wavelength grid.
    maskHR : (N_HR,) array-like of bool
        Valid-sample mask on the HR grid (True = keep).
    lamLR : (N_LR,) array-like of float
        LR bin centers (monotonic).
    dlam : (N_LR,) array-like of float
        LR bin widths (> 0). Bin edges are defined as
        [lamLR - 0.5*dlam, ..., lamLR[-1] + 0.5*dlam[-1]].
    R_sampling : float
        Sampling resolving power of the HR grid (Nyquist convention).
        Internally sets the frequency-to-resolution mapping
        'res = 2 * R_sampling * f' with 'f = np.fft.fftfreq(M)'.
    Rc_init : float or None
        Optional cutoff (in “resolution” units) for an initial low-pass that
        colors the HR noise before any rebinning. If None, skipped.
    filter_init_type : {'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep'} or None
        Shape of the initial low-pass at 'Rc_init'. If "gaussian", the
        implementation uses the fast variant ("gaussian_fast").
    Rc_noise : float or None
        Optional *additional* low-pass cutoff applied to the HR noise. If None, skipped.
    filter_noise_type : same as 'filter_init_type' or None
        Shape of the additional low-pass at 'Rc_noise'.
    Rc_conv : float or None
        Optional cutoff representing real-space broadening (e.g., degrading to a
        lower effective resolution). If None, skipped.
    filter_conv_type : same as 'filter_init_type' or None
        Shape of the broadening filter at 'Rc_conv'.

    Returns
    -------
    fn : (N_LR,) ndarray of float
        Per-bin variance inflation factor (typically ≥ 1 for positively
        correlated noise). If a bin contains ≤ 1 HR sample, fn is set to 1.

    Notes
    -----
    The total frequency response is the product of the selected low-pass filters:
        H(f) = H_init(f) * H_LP(f) * H_conv(f)   (omitting terms set to None).
    The normalized autocovariance on the HR grid is computed as
        C = IFFT(|H|^2), then C /= C[0],
    and the per-bin factor for a bin with 'N_i' HR samples equals
        fn(N_i) = (1 / N_i) * sum_{tau = -(N_i-1)}^{N_i-1} (N_i - |tau|) * C[tau].
    When no filters are applied, C is a delta and fn = 1 for all bins.
    """
    from src.spectrum import _fft_filter_response

    # --- HR samples per LR bin (N_i) ---
    LRedges = np.hstack([lamLR - 0.5*dlam, lamLR[-1] + 0.5*dlam[-1]])
    count   = binned_statistic(lamHR[maskHR], np.ones(np.sum(maskHR)),
                               statistic="sum", bins=LRedges)[0].astype(int)
    if np.all(count <= 1):
        return np.ones_like(lamLR, dtype=float)

    # --- choose FFT size on the HR grid (large enough for lags up to max N_i) ---
    Nmax = int(np.nanmax(count))
    M    = 1 << int(np.ceil(np.log2(max(2048, 8*Nmax + 1))))  # power-of-two, >= ~8*Nmax

    # Frequency grid and "resolution" axis (your convention)
    f   = np.fft.fftfreq(M)          # cycles / HR sample
    res = 2.0 * R_sampling * f

    # --- Total frequency response H(f) ---
    H = np.ones(M, dtype=np.complex128)

    # 1) Optional initial HR coloring at Rc_init
    if Rc_init is not None and filter_init_type is not None:
        ftype_init = "gaussian_fast" if (filter_init_type == "gaussian") else filter_init_type
        H         *= _fft_filter_response(N=M, R=R_sampling, Rc=Rc_init, filter_type=ftype_init)[1]

    # 2) Optional extra low-pass
    if Rc_noise is not None and filter_noise_type is not None:
        ftype_lp = "gaussian_fast" if (filter_noise_type == "gaussian") else filter_noise_type
        H       *= _fft_filter_response(N=M, R=R_sampling, Rc=Rc_noise, filter_type=ftype_lp)[1]

    # 3) Optional convolution broadening
    if Rc_conv is not None and filter_conv_type is not None:
        ftype_cv = "gaussian_fast" if (filter_conv_type == "gaussian") else filter_conv_type
        H       *= _fft_filter_response(N=M, R=R_sampling, Rc=Rc_conv, filter_type=ftype_cv)[1]

    # --- Normalized autocovariance on the HR grid: C = IFFT(|H|^2), with C[0]=1 ---
    C   = np.fft.ifft(np.abs(H)**2).real
    C  /= C[0]
    C   = np.fft.fftshift(C)
    mid = len(C) // 2

    # --- Per-bin fn(N_i) = (1/(N_i*C[0])) * sum_{tau=-(N_i-1)}^{N_i-1} (N_i - |tau|) * C[tau] ---
    fn = np.ones_like(lamLR, dtype=float)
    for i, Ni in enumerate(count):
        if Ni <= 1:
            fn[i] = 1.0
            continue
        taus = np.arange(-(Ni-1), Ni, dtype=int)
        fn[i] = np.sum((Ni - np.abs(taus)) * C[mid + taus]) / (Ni * 1.0)

    return fn



def extract_vipa_data(instru, target_name, gain, label_fiber, degrade_data=True, outliers=False, sigma_outliers=5, use_weight=True, mask_nan_values=False, filter_noise=False, R_target=80_000, Rc=100, filter_type="gaussian", use_trans=True, high_pass_flux=True, verbose=True):
    """
    Load a VIPA 1D spectrum from FITS and optionally:
      (i) filter high-frequency content (assumed noise above the instrumental R),
      (ii) degrade the effective resolving power by Gaussian convolution,
      (iii) rebin to a Nyquist grid at the target resolution,
      while propagating 1σ uncertainties consistently (including correlated-noise effects),
      and producing a “high-pass” version of the flux if a transmission vector is provided.

    Workflow (high level)
    ---------------------
    1) Read arrays from the FITS file:
       (wave0, flux0, sigma0, weight0, trans0, sigma_trans0) plus header keywords.
       Wavelengths are converted from nm to µm.
    2) Optional outlier masking via high-pass residual clipping on flux (and trans if present).
    3) Build a “realistic” noise realization ('noise0') consistent with 'sigma0'
       and optionally color it with the same initial low-pass as the data.
    4) Optional low-pass filtering at Rc = R_instru (e.g. smoothstep) to remove
       content deemed above the instrumental resolution. Uncertainties are scaled by
         sigma *= sqrt(fn_LF),
       where 'fn_LF' is the fraction of input noise variance transmitted through the LP,
       computed with 'compute_filter_variance_factor(…, this_filter=(Rc, type))'.
    5) Optional resolution degradation to R_target < R_instru by Gaussian convolution.
       The cutoff equivalent (Rc_conv) is derived from sigma_kernel; uncertainties are
       scaled by the corresponding variance fraction (as above).
    6) Rebin to a Nyquist grid at R_target:
         λ_Nyquist step = λ_centre / (2*R_target)  (here approximated with the band midpoint).
       Rebinning uses top-hat averaging. Because pre-filtering introduces correlations,
       the naive i.i.d. error propagation is corrected by a per-bin factor:
         sigma_rebinned *= sqrt(F),
       where 'F' is returned by 'compute_rebin_variance_factor(…)' built from the
       total noise coloring prior to the rebinning.
    7) If a transmission vector 'trans' is present, compute the “high-pass” product
       following the multiplicative model:
         flux_HF = trans * HP(flux / trans),
       propagate its uncertainty with
         sigma_HF = sqrt(fn_HF) * sigma,
       where 'fn_HF' is computed for the chosen HF/LF split (Rc, filter_type)
       on the colored noise PSD.
    8) Optional weighting: 'weight' is normalized to max=1 and applied to
       (sigma, noise, sigma_HF, noise_HF) for downstream compatibility.

    Parameters
    ----------
    instru : str
        Instrument name (used to locate the FITS file under 'data/{instru}/…').
    target_name : str
        Target identifier (part of the FITS filename).
    gain : int or str
        Detector gain setting (part of the FITS filename).
    label_fiber : str
        Fiber label (part of the FITS filename).
    degrade_data : bool, default True
        If True, degrade to 'R_target' (Gaussian LSF broadening + Nyquist rebin).
        If False, keep the native sampling (no convolution, no rebin).
    outliers : bool, default False
        If True, sigma-clip high-pass residuals to flag and mask outliers.
    sigma_outliers : float, default 5
        Clipping threshold (in σ) for outlier detection in the HF residuals.
    use_weight : bool, default True
        If True, normalize 'weight' to [0,1] and multiply (sigma, noise, sigma_HF, noise_HF)
        by 'weight' to keep S/N consistent with later weighted fits.
    mask_nan_values : bool, default False
        If True, propagate NaN masks to all returned arrays.
    filter_noise : bool, default False
        If True, apply an additional low-pass at Rc = R_instru (default shape: "smoothstep")
        before any resolution degradation/rebinning, and scale uncertainties by the
        corresponding variance fraction.
    R_target : float, default 80_000
        Desired resolving power for the degraded+rebinned spectrum (must satisfy
        R_target ≤ R_instru if 'degrade_data=True').
    Rc : float, default 100
        Cutoff parameter (in the “resolution” axis, res = 2*R*f) used for the final HF/LF
        split when building 'flux_HF'/'sigma_HF'.
    filter_type : {'gaussian','gaussian_fast','gaussian_true','step','smoothstep'}, default 'gaussian'
        Filter shape used by 'filtered_flux' and the HF/LF split when computing
        'flux_HF'/'sigma_HF'. The internal frequency-domain helpers map 'gaussian'
        to the discrete 'gaussian_fast' variant for consistency.
    verbose : bool, default True
        If True, print a summary and sanity-check metrics (e.g., rms_z).

    Returns
    -------
    wave : (M,) ndarray
        Output wavelength grid (µm). If 'degrade_data=True', this is the Nyquist grid
        at 'R_target'; otherwise it is the native 'wave0'.
    flux : (M,) ndarray
        Processed flux on 'wave'.
    sigma : (M,) ndarray
        1-σ uncertainty on 'flux', including (i) filter-induced variance reduction and
        (ii) rebinning correction for correlated noise.
    noise : (M,) ndarray
        Realistic noise realization consistent with 'sigma' and the applied filters
        (useful for residual checks).
    flux_HF : (M,) ndarray or None
        High-pass product 'trans * HP(flux/trans)' if a transmission vector is present;
        otherwise None.
    sigma_HF : (M,) ndarray or None
        1-σ uncertainty for 'flux_HF' using 'sqrt(fn_HF)' scaling; None if 'trans' is absent.
    noise_HF : (M,) ndarray or None
        High-pass of 'noise/trans' multiplied by 'trans'; None if 'trans' is absent.
    weight : (M,) ndarray or None
        Normalized weight function (max=1) on the output grid if 'use_weight=True';
        otherwise None.
    trans : (M,) ndarray or None
        Transmission vector rebinned to 'wave' if present in the input; otherwise None.
    sigma_trans : (M,) ndarray or None
        1-σ uncertainty on 'trans' after filtering/rebinning if applicable; otherwise None.
    exposure_time : float
        Exposure time in minutes (from the FITS header).

    Notes
    -----
    * Assumes uniformly sampled input for filtering stages; if your native grid is
      not uniform, resample upstream (the function rebins only at the end).
    * Variance fractions for filters are computed in the frequency domain with the
      already-colored input noise PSD; this avoids the “white-noise” assumption bias.
    * Rebin-variance correction uses the frequency-domain window of the top-hat
      average and the colored noise autocovariance (via 'compute_rebin_variance_factor').
    * Boundary conditions are effectively circular in frequency-domain filtering;
      small edge discrepancies may remain if your data have strong discontinuities.
    * If 'R_target == R_instru' and 'degrade_data=True', no convolution is applied;
      only the Nyquist rebinning step occurs.
    * The final “rms_z” prints provide a quick sanity check:
        rms_z ≈ std(noise / sigma)  (and similarly for HF).
    """
    from src.spectrum import filtered_flux, downbin_spec, rebin_spec

    f                                                   = fits.open(f"data/{instru}/VIPA_Final_Spectrum_{target_name}_gain_{gain}_fiber_{label_fiber}.fits")
    wave0, flux0, sigma0, weight0, trans0, sigma_trans0 = f[0].data
    wave0                                              *= 1e-3                  # [nm] => [µm]
    header                                              = f[0].header
    exposure_time                                       = header["INTTIME"]/60 # [mn]
    lmin                                                = header["lmin"]*1e-3  # [nm] => [µm]
    lmax                                                = header["lmax"]*1e-3  # [nm] => [µm]
    R_instru                                            = header["R"]
    R_sampling                                          = header["R_sampling"]
    date_obs                                            = header['ACQTIME1']
    try:
        T_star                                          = header["TEFF"] 
        lg_star                                         = header["LG"] 
        rv_star_bary                                    = header["RV_bary"]
        rv_star_corr                                    = header["bary_corr"]
        rv_star_obs                                     = header["RV_obs"]
        vsini_star                                      = header["Vsin(i)"]
        magH_star                                       = header["mag(H)"]
        is_star = True
    except:
        is_star = False
    f.close()
        
    if degrade_data and R_target > R_instru:
        raise KeyError(f"The target resolution ({R_target}) can not be greater than the instrumental resolution ({R_instru}).")
        
    if verbose:
        print()
        print(" Observation Summary")
        print("=" * 40)
        print(f" Target name      : {target_name}")
        print(f" Observation date : {date_obs}")
        print(f" Exposure time    : {round(exposure_time, 1):>6} mn")
        if is_star:
            print()
            print("Star Properties")
            print("=" * 40)
            print(f" H-band mag : {round(magH_star, 1):>6}")
            print(f" Teff       : {round(T_star, 0):>6} K")
            print(f" logg       : {round(lg_star, 1):>6} dex[cm/s2]")
            print(f" Vsini      : {round(vsini_star, 2):>6} km/s")
            print(f" RV (bary)  : {round(rv_star_bary, 3):>6} km/s")
            print(f" RV (obs)   : {round(rv_star_obs, 3):>6} km/s")
            print(f" Bary corr  : {round(rv_star_corr, 3):>6} km/s")
        print()
        print("Spectral Properties")
        print("=" * 40)
        print(f" R (instrument)        : {R_instru:.0f}")
        print(f" R (sampling)          : {R_sampling:.0f}")
        if degrade_data:
            print(f" R (target)            : {R_target:.0f}")
        if Rc is not None:
            print(f" Rc (cutoff frequency) : {Rc:.0f}")
        print(f" Filter                : {filter_type}")
        print()
    
    # Missing values
    nan_values0 = ~np.isfinite(flux0)
    if use_trans:
        nan_values0 |= ~np.isfinite(trans0)
    
    # (first) OUTLIERS FILTERING (if wanted)
    if outliers:
        NbNaN0 = nan_values0.sum()
        flux0_HF     = filtered_flux(flux0, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values0 |= sigma_clip(flux0_HF, sigma=sigma_outliers).mask
        if use_trans:
            trans0_HF    = filtered_flux(trans0, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
            nan_values0 |= sigma_clip(trans0_HF, sigma=sigma_outliers).mask
        if high_pass_flux:
            flux0_HF     = trans0 * filtered_flux(flux0/trans0, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
            nan_values0 |= sigma_clip(flux0_HF, sigma=sigma_outliers).mask
        print(f"{nan_values0.sum() - NbNaN0} outliers found...")
        weight0[nan_values0] = np.nan
        flux0[nan_values0]   = np.nan
        sigma0[nan_values0]  = np.nan
        if use_trans:
            trans0[nan_values0]       = np.nan
            sigma_trans0[nan_values0] = np.nan
    
    # Realistic noise realisation (for residual comparison purposes)
    noise0  = np.random.normal(0, sigma0, len(wave0)) # white noise at sigma0    
    
    Rc_init          = R_instru
    filter_init_type = "gaussian"
    
    # Rc_init          = None
    # filter_init_type = None
    
    if Rc_init is not None:
        noise0 = filtered_flux(noise0, R=R_sampling, Rc=Rc_init, filter_type=filter_init_type)[1] # the noise is not entirely white.. (low pass filter at Rc_init)
    
    noise0 *= np.sqrt(np.nanmean(sigma0**2)) / np.nanstd(noise0)
    
    # Low-pass (if wanted): every "signal" above R_instru is filtered and is assumed to be noise
    if filter_noise:
        Rc_noise          = R_instru
        filter_noise_type = "gaussian"
        
        flux0  = filtered_flux(flux=flux0,  R=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type)[1]
        noise0 = filtered_flux(flux=noise0, R=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type)[1]
        
        # Assuming constant sigma: estimating the power fraction of noise that would be filtered
        fn_LF   = compute_filter_variance_factor(N=len(wave0), R_sampling=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=None, filter_noise_type=None, Rc_conv=None, filter_conv_type=None)[1]
        sigma0 *= np.sqrt(fn_LF)
        
        if use_trans:
            trans0        = filtered_flux(flux=trans0, R=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type)[1]
            sigma_trans0 *= np.sqrt(fn_LF)
    else:
        Rc_noise          = None
        filter_noise_type = None
    
    # Artificially degrating the data to an arbitrary resolution R (if wanted)
    if degrade_data:
        
        # --- Convoluing the data to R_target
        
        if R_target == R_instru:
            Rc_conv          = None # No convolution needed
            filter_conv_type = None

        else:
            sigma_kernel     = np.sqrt(  (R_sampling / R_target)**2 - (R_sampling / R_instru)**2 ) / np.sqrt(2*np.log(2))
            Rc_conv          = 2*R_sampling / (np.pi * sigma_kernel) * np.sqrt(np.log(2)/2)
            filter_conv_type = "gaussian"
            
            flux0  = filtered_flux(flux=flux0,  R=R_sampling, Rc=Rc_conv, filter_type=filter_conv_type)[1]
            noise0 = filtered_flux(flux=noise0, R=R_sampling, Rc=Rc_conv, filter_type=filter_conv_type)[1]

            # Assuming constant sigma: estimating the power fraction of noise that would be filtered
            fn_LF    = compute_filter_variance_factor(N=len(wave0), R_sampling=R_sampling, Rc=Rc_conv, filter_type=filter_conv_type, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=Rc_noise, filter_noise_type=filter_noise_type, Rc_conv=None, filter_conv_type=None)[1]
            sigma0  *= np.sqrt(fn_LF)
            
            if use_trans:
                trans0        = filtered_flux(flux=trans0,  R=R_sampling, Rc=Rc_conv, filter_type=filter_conv_type)[1]
                sigma_trans0 *= np.sqrt(fn_LF)
        
        # --- Rebinning to Nyquist sampling
        
        # Nyquist sampled wavelength axis
        R     = R_target
        dl    = (lmin + lmax)/2 /(2*R)
        wave  = np.arange(lmin, lmax, dl)
        dwave = np.gradient(wave)
        
        # Interpolating weight
        nan_values = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan)(wave) != 0
        
        # # Rebinning data
        # flux, sigma, weight = downbin_spec(specHR=flux0,  sigmaHR=sigma0, weightHR=weight0, lamHR=wave0, lamLR=wave, dlam=dwave)
        # noise, _, _         = downbin_spec(specHR=noise0, sigmaHR=None,   weightHR=None,    lamHR=wave0, lamLR=wave, dlam=dwave)
        # if use_trans:
        #     trans, sigma_trans, _ = downbin_spec(specHR=trans0,  sigmaHR=sigma_trans0, weightHR=None, lamHR=wave0, lamLR=wave, dlam=dwave)
        # else:
        #     trans       = None
        #     sigma_trans = None
            
        flux, sigma, weight, _ = rebin_spec(specHR=flux0,  sigmaHR=sigma0, weightHR=weight0, lamHR=wave0, lamLR=wave, dlam=dwave)
        noise, _, _, _         = rebin_spec(specHR=noise0, sigmaHR=None,   weightHR=None,    lamHR=wave0, lamLR=wave, dlam=dwave)
        if use_trans:
            trans, sigma_trans, _, _ = rebin_spec(specHR=trans0,  sigmaHR=sigma_trans0, weightHR=None, lamHR=wave0, lamLR=wave, dlam=dwave)
        else:
            trans       = None
            sigma_trans = None
        
        # Rebinning does not propagate correctly the sigmas since it assumses i.d.d noise:
        fn_rebin     = compute_rebin_variance_factor(lamHR=wave0, maskHR=~nan_values0, lamLR=wave, dlam=dwave, R_sampling=R_sampling, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=Rc_noise, filter_noise_type=filter_noise_type, Rc_conv=Rc_conv, filter_conv_type=filter_conv_type)
        sigma       *= np.sqrt(fn_rebin)
        if use_trans:
            sigma_trans *= np.sqrt(fn_rebin)
        
        # New sampling resolution
        R_sampling = R

    # Otherwise, takes the raw data
    else:
        Rc_conv          = None # No convolution needed
        filter_conv_type = None
        R                = R_instru
        wave             = wave0
        weight           = weight0
        nan_values       = nan_values0
        flux             = flux0
        sigma            = sigma0
        noise            = noise0
        trans            = trans0
        sigma_trans      = sigma_trans0
        
    # MM-like post-processing
    if high_pass_flux:
        flux_HF  = trans * filtered_flux(flux/trans,  R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        noise_HF = trans * filtered_flux(noise/trans, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]        
        
        # Assuming constant sigma: estimating the power fraction of noise that would be filtered
        fn_HF    = compute_filter_variance_factor(N=len(wave), R_sampling=R_sampling, Rc=Rc, filter_type=filter_type, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=Rc_noise, filter_noise_type=filter_noise_type, Rc_conv=Rc_conv, filter_conv_type=filter_conv_type)[0]
        sigma_HF = np.sqrt(fn_HF) * sigma
  
    else:
        flux_HF  = None
        sigma_HF = None
        noise_HF = None
    
    # (final) OUTLIERS FILTERING (if wanted)
    if outliers:
        NbNaN = nan_values.sum()
        flux_HF     = filtered_flux(flux, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values |= sigma_clip(flux_HF, sigma=sigma_outliers).mask
        if use_trans:
            trans_HF    = filtered_flux(trans, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
            nan_values |= sigma_clip(trans_HF, sigma=sigma_outliers).mask
        if high_pass_flux:
            flux_HF     = trans * filtered_flux(flux/trans, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
            nan_values |= sigma_clip(flux_HF, sigma=sigma_outliers).mask
        print(f"{nan_values.sum() - NbNaN} outliers found...")

    # Removing the flagged NaN values
    if mask_nan_values:
        weight[nan_values] = np.nan
        flux[nan_values]   = np.nan
        sigma[nan_values]  = np.nan
        noise[nan_values]  = np.nan
        if use_trans:
            trans[nan_values]       = np.nan
            sigma_trans[nan_values] = np.nan
        if high_pass_flux:
            flux_HF[nan_values]     = np.nan
            sigma_HF[nan_values]    = np.nan
            noise_HF[nan_values]    = np.nan

    # Weight function (if wanted)
    if use_weight:
        weight /= np.nanmax(weight)
        sigma  *= weight # since the signals (flux, flux_HF) will be multiplied by the weight, the noise needs also to be multiplied by it
        noise  *= weight
        if high_pass_flux:
            sigma_HF *= weight
            noise_HF *= weight
    else:
        weight = None
    
    # Sigma propagation sanity check
    if verbose:
        print(f"\nrms_z      = {np.sqrt(np.nanmean((noise / sigma)**2)):.3f} ({np.nanstd(noise) / np.sqrt(np.nanmean(sigma**2)):.3f})")
        if high_pass_flux:
            print(f"rms_z (HF) = {np.sqrt(np.nanmean((noise_HF / sigma_HF)**2)):.3f} ({np.nanstd(noise_HF) / np.sqrt(np.nanmean(sigma_HF**2)):.3f})")
    
    # Filters dictionnary
    filters = dict(Rc_init=Rc_init, filter_init_type_init=filter_init_type, Rc_noise=Rc_noise, filter_noise_type_init=filter_noise_type, Rc_conv=Rc_conv, filter_conv_type_init=filter_conv_type)
    
    return wave, flux, sigma, noise, flux_HF, sigma_HF, noise_HF, weight, trans, sigma_trans, exposure_time, filters, R



###########################################################################################



def extract_hirise_data(target_name, interpolate, degrade_resolution, R, Rc, filter_type, order_by_order, outliers, sigma_outliers, only_high_pass=False, cut_fringes=False, Rmin=None, Rmax=None, use_weight=True, mask_nan_values=False, keep_only_good=False, wave_input=None, reference_fibers=True, crop_tell_orders=False, shift_star_corr=False, verbose=True): # OPENING DATA AND DEGRADATING THE RESOLUTION (if wanted)
    from src.spectrum import Spectrum, filtered_flux, interpolate_flux_with_error, estimate_resolution
    
    # hard coded values
    GAIN     = np.nanmean([2.28, 2.19, 2.00]) # in e-/ADU
    noffsets = 1
    nrefs    = 3
    R_instru = 140_000
    if degrade_resolution and R > R_instru:
        raise KeyError(f"The input resolution R ({R}) can not be greater than the instrumental resolution ({R_instru}).")
    
    # OPENING DATA 
    file              = f"data/HiRISE/{target_name}.fits"
    f                 = fits.open(file)
    hdr               = f[0].header
    tn                = target_name.split('_20')[0].replace('_', ' ')
    date_obs_comp     = mjd_to_date(hdr['HIERARCH COMP MJD MEAN'])
    date_obs_star     = mjd_to_date(hdr['HIERARCH STAR MJD MEAN'])
    T_star            = hdr["HIERARCH STAR TEFF"] 
    lg_star           = hdr["HIERARCH STAR LOGG"]
    rv_comp_corr      = hdr["HIERARCH COMP HELCORR MEAN"]
    rv_comp_corr_err  = np.abs(hdr["HIERARCH COMP HELCORR START"] - hdr["HIERARCH COMP HELCORR END"]) / 2
    rv_star_helio     = hdr["HIERARCH STAR RV"]
    rv_star_helio_err = hdr["HIERARCH STAR RV ERR"]
    rv_star_corr      = hdr["HIERARCH STAR HELCORR MEAN"]
    rv_star_corr_err  = np.abs(hdr["HIERARCH STAR HELCORR START"] - hdr["HIERARCH STAR HELCORR END"]) / 2
    rv_star_obs       = rv_star_helio - rv_star_corr
    rv_star_obs_err   = np.sqrt(rv_star_helio_err**2 + rv_star_corr_err**2)
    vsini_star        = hdr["HIERARCH STAR VSINI"]
    vsini_star_err    = hdr["HIERARCH STAR VSINI ERR"]
    t_exp_comp        = hdr["HIERARCH COMP DIT"] * hdr["HIERARCH COMP NEXP"]
    t_exp_star        = hdr["HIERARCH STAR DIT"] * hdr["HIERARCH STAR NEXP"]
    
    if verbose:
        print()
        print(" Observation Summary")
        print("=" * 40)
        print(f" Target name             : {tn}")
        print(f" Observation date (comp) : {date_obs_comp}")
        print(f" Observation date (star) : {date_obs_star}")
        print(f" Exposure time (comp)    : {round(t_exp_comp / 60):>6} min")
        print(f" Exposure time (star)    : {round(t_exp_star / 60):>6} min")
        print()
        print("Star Properties")
        print("=" * 40)
        print(f" Teff       : {round(T_star, 0):>6}          K")
        print(f" logg       : {round(lg_star, 1):>6}          dex[cm/s2]")
        print(f" Vsini      : {round(vsini_star, 3):>6} ± {round(vsini_star_err, 3):>6} km/s")
        print(f" RV (helio) : {round(rv_star_helio, 3):>6} ± {round(rv_star_helio_err, 3):>6} km/s")
        print(f" RV (obs)   : {round(rv_star_obs, 3):>6} ± {round(rv_star_obs_err, 3):>6} km/s")
        print()
        print("Heliocentric Corrections")
        print("=" * 40)
        print(f" Heliocentric correction (comp) : {round(rv_comp_corr, 3):>6} ± {round(rv_comp_corr_err, 3):>6} km/s")
        print(f" Heliocentric correction (star) : {round(rv_star_corr, 3):>6} ± {round(rv_star_corr_err, 3):>6} km/s")
        print()
        print("Spectral Properties")
        print("=" * 40)
        print(f" R (instrument)        : {R_instru:.0f}")
        print(f" R (input)             : {R:.0f}")
        print(f" Rc (cutoff frequency) : {Rc:.0f}")
        print(f" Filter                : {filter_type}")
        print()
    
    bkg_flux0  = []
    wave0 = f[1].data["pipeline"]*1e-3     # in µm, raw CRIRES wavelength solution
    wave0 = f[1].data["recalibrated"]*1e-3 # in µm, reclibrated  (in observer referential)
    for ioff in range(noffsets):
        # Response data
        data_response = f[f.index_of(f'RESPONSE,OFFSET{ioff}')].data
        trans0        = data_response["response"]       # no unit
        trans_model0  = data_response["response_model"] # no unit
        # Star data
        data_star       = f[f.index_of(f'STAR,OFFSET{ioff},SCI')].data
        star_flux0      = data_star["signal"]           # in e-
        star_wave0      = data_star["wave"]*1e-3        # in µm, recalibrated (in heliocentric referential for star)
        star_weight0    = data_star["weight"]           # no unit
        star_sigma_tot0 = data_star["noise"]            # e-
        star_sigma_bkg0 = data_star["noise_background"] # e-
        #star_sigma_bkg0 = np.sqrt(star_sigma_tot0**2 - star_flux0)   # e-
        for iref in range(nrefs):
            data_star_ref = f[f.index_of(f'STAR,OFFSET{ioff},REF{iref}')].data
            bkg_flux0.append(data_star_ref["signal"])             # e-
        # Comp data
        data_planet       = f[f.index_of(f'COMP,OFFSET{ioff},SCI')].data
        planet_flux0      = data_planet["signal"]           # in e-
        planet_wave0      = data_planet["wave"]*1e-3        # in µm, recalibrated (in heliocentric referential for comp)
        planet_weight0    = data_planet["weight"]           # no unit
        planet_sigma_tot0 = data_planet["noise"]            # e-
        planet_sigma_bkg0 = data_planet["noise_background"] # e-
        #planet_sigma_bkg0 = np.sqrt(planet_sigma_tot0**2 - planet_flux0)  # e-
        for iref in range(nrefs):
            data_planet_ref = f[f.index_of(f'COMP,OFFSET{ioff},REF{iref}')].data
            bkg_flux0.append(data_planet_ref["signal"]) # e-
                    
    # sky_transmission_path = os.path.join("sim_data/Transmission/sky_transmission_airmass_1.0.fits")
    # sky_trans             = fits.getdata(sky_transmission_path)
    # trans_tell_band       = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
    # planet_flux0          = trans_tell_band.interpolate_wavelength(wave0, renorm=False).flux # degraded tellurics transmission on the considered band
    # #planet_flux0          = trans_model0

    # calib_name = "HD_26820"
    # wave0      = fits.open("data/HiRISE/"+calib_name+".fits")[1].data["recalibrated"]*1e-3      # in µm
    
    if shift_star_corr:
        if date_obs_comp > date_obs_star:
            delta_corr = rv_star_corr - rv_comp_corr
        else:
            delta_corr = rv_comp_corr - rv_star_corr
        star_flux0 = Spectrum(wave0, star_flux0/trans0).doppler_shift(delta_corr, renorm=False).flux * trans0
        
    # Wavelength axis properties
    if crop_tell_orders:
        lmin = 1.536
        lmax = 1.696 # Cropping two firsts and one last order
    else:
        lmin = wave0[0]
        lmax = wave0[-1]
    R0  = estimate_resolution(wave0) # Raw sampling resolution
    dl0 = (lmin+lmax)/2 / (2*R0)

    # Flagging NaN values
    nan_values0 = np.isnan(star_flux0)|np.isnan(planet_flux0)|np.isnan(trans0) # missing values
    valid0      = ~nan_values0

    # Flagging the orders limits (for "order_by_order" filtering method)      
    if order_by_order:          
        transitions = np.where(np.diff(wave0) > 1000 * dl0)[0] + 1  # Indexes where the order changes
        lmin_orders = wave0[transitions - 1]
        lmax_orders = wave0[transitions]
    
    # (first) OUTLIERS FILTERING (if wanted)
    if outliers:
        trans0_HF       = filtered_flux(trans0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(trans0_HF, sigma=2*sigma_outliers)
        trans0          = np.array(np.ma.masked_array(trans0, mask=sg.mask).filled(np.nan))
        trans_model0_HF = filtered_flux(trans_model0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(trans_model0_HF, sigma=2*sigma_outliers)
        trans_model0    = np.array(np.ma.masked_array(trans_model0, mask=sg.mask).filled(np.nan))
        star_flux0_HF   = filtered_flux(star_flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(star_flux0_HF, sigma=2*sigma_outliers)
        star_flux0      = np.array(np.ma.masked_array(star_flux0, mask=sg.mask).filled(np.nan))
        planet_flux0_HF = filtered_flux(planet_flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(planet_flux0_HF, sigma=2*sigma_outliers)
        planet_flux0    = np.array(np.ma.masked_array(planet_flux0, mask=sg.mask).filled(np.nan))
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0_HF = filtered_flux(bkg_flux0[i], R=R0, Rc=Rc, filter_type=filter_type)[0]
                sg           = sigma_clip(bkg_flux0_HF, sigma=2*sigma_outliers)
                bkg_flux0[i] = np.array(np.ma.masked_array(bkg_flux0[i], mask=sg.mask).filled(np.nan))
                
    # Interpolation of the data (should not really degrade data) (if wanted)
    if interpolate: # in order to have a regular wavelength axis equivalent to a Nyquist sampling of the instrumental resolution R_instru
        # new wavelength axis
        if wave_input is not None and not degrade_resolution:
            wave = wave_input
        else:
            wave = np.arange(lmin, lmax, dl0) # constant and regular wavelength array : 0.01 µm ~ doppler shift at few thousands of km/s
        # Converting to densities (for interpolations to make sense)
        dwave  = np.gradient(wave)  # [µm/px]
        dwave0 = np.gradient(wave0) # [µm/px]
        star_flux0      /= dwave0 # [e-/µm]
        star_sigma_tot0 /= dwave0 # [e-/µm]
        star_sigma_bkg0 /= dwave0 # [e-/µm]
        planet_flux0      /= dwave0 # [e-/µm]
        planet_sigma_tot0 /= dwave0 # [e-/µm]
        planet_sigma_bkg0 /= dwave0 # [e-/µm]
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0[i] /= dwave0 # [e-/µm]
        # Interpolations (and noise propagation)
        star_flux0, star_sigma_tot0, star_weight0 = interpolate_flux_with_error(wave=wave0[valid0], flux=star_flux0[valid0], sigma=star_sigma_tot0[valid0], weight=star_weight0[valid0], wave_new=wave)                
        _, star_sigma_bkg0, _                     = interpolate_flux_with_error(wave=wave0[valid0], flux=None, sigma=star_sigma_bkg0[valid0], weight=None, wave_new=wave)                
        planet_flux0, planet_sigma_tot0, planet_weight0 = interpolate_flux_with_error(wave=wave0[valid0], flux=planet_flux0[valid0], sigma=planet_sigma_tot0[valid0], weight=planet_weight0[valid0], wave_new=wave)                
        _, planet_sigma_bkg0, _                         = interpolate_flux_with_error(wave=wave0[valid0], flux=None, sigma=planet_sigma_bkg0[valid0], weight=None, wave_new=wave)                
        trans0, _, _       = interpolate_flux_with_error(wave=wave0[valid0], flux=trans0[valid0], sigma=None, weight=None, wave_new=wave)                
        trans_model0, _, _ = interpolate_flux_with_error(wave=wave0[valid0], flux=trans_model0[valid0], sigma=None, weight=None, wave_new=wave)                
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0[i], _, _ = interpolate_flux_with_error(wave=wave0[valid0], flux=bkg_flux0[i][valid0], sigma=None, weight=None, wave_new=wave)                
        # Reconverting in flux per bins
        star_flux0      *= dwave # [e-/bin]
        star_sigma_tot0 *= dwave # [e-/bin]
        star_sigma_bkg0 *= dwave # [e-/bin]
        planet_flux0      *= dwave # [e-/bin]
        planet_sigma_tot0 *= dwave # [e-/bin]
        planet_sigma_bkg0 *= dwave # [e-/bin]
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0[i] *= dwave # [e-/bin]
        # NaN values interpolation
        interp_nv   = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan)
        nan_values0 = interp_nv(wave) != 0
        valid0      = ~nan_values0
        # new wavelength axis
        wave0 = wave

    # Artificially degrating the data to an arbitrary resolution R (if wanted)
    if degrade_resolution:
        # new wavelength axis
        if wave_input is not None:
            wave = wave_input
        else:
            dl   = np.nanmedian(wave0/(2*R)) # 2*R => Nyquist sampling (Shannon)
            wave = np.arange(lmin, lmax, dl) # new wavelength array (with degrated resolution)
        # Degrade star data
        star_spectrumLR = Spectrum(wave0[valid0], star_flux0[valid0]).degrade_resolution(wave, renorm=True, R_output=R, sigma=star_sigma_tot0[valid0])
        star_flux       = star_spectrumLR.flux
        star_sigma_tot  = star_spectrumLR.sigma
        star_sigma_bkg  = Spectrum(wave0[valid0], star_flux0[valid0]).degrade_resolution(wave, renorm=True, R_output=R, sigma=star_sigma_bkg0[valid0]).sigma
        star_weight     = Spectrum(wave0[valid0], star_weight0[valid0]).interpolate_wavelength(wave, renorm=False).flux
        # Degrade planet data
        planet_spectrumLR = Spectrum(wave0[valid0], planet_flux0[valid0]).degrade_resolution(wave, renorm=True, R_output=R, sigma=planet_sigma_tot0[valid0])
        planet_flux       = planet_spectrumLR.flux
        planet_sigma_tot  = planet_spectrumLR.sigma
        planet_sigma_bkg  = Spectrum(wave0[valid0], planet_flux0[valid0]).degrade_resolution(wave, renorm=True, R_output=R, sigma=planet_sigma_bkg0[valid0]).sigma
        planet_weight     = Spectrum(wave0[valid0], planet_weight0[valid0]).interpolate_wavelength(wave, renorm=False).flux
        # Degrade trans data
        trans       = Spectrum(wave0[valid0], trans0[valid0]).degrade_resolution(wave, renorm=False, R_output=R).flux
        trans_model = Spectrum(wave0[valid0], trans_model0[valid0]).degrade_resolution(wave, renorm=False, R_output=R).flux
        # NaN values interpolation
        interp_nv  = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan)
        nan_values = interp_nv(wave) != 0
        valid      = ~nan_values
        # Degrade bkg data
        if reference_fibers:
            bkg_flux = [0] * len(bkg_flux0)
            for i in range(len(bkg_flux0)):
                bkg_flux[i] = Spectrum(wave0, bkg_flux0[i]).degrade_resolution(wave, renorm=True, R_output=R).flux
            
    else: # otherwise, takes the raw data
        R = R0 ; wave = wave0 ; dl = dl0 ; star_flux = star_flux0 ; star_weight = star_weight0 ; star_sigma_tot = star_sigma_tot0 ; star_sigma_bkg = star_sigma_bkg0 ; planet_flux = planet_flux0 ; planet_weight = planet_weight0 ; planet_sigma_tot = planet_sigma_tot0 ; planet_sigma_bkg = planet_sigma_bkg0 ; trans = trans0 ; trans_model = trans_model0 ; nan_values = nan_values0 ; valid = valid0 ; bkg_flux = bkg_flux0
        
    # HIGH PASS FILTERING
    if order_by_order:
        star_flux_HF, star_flux_LF     = np.full_like(wave, np.nan), np.full_like(wave, np.nan)
        planet_flux_HF, planet_flux_LF = np.full_like(wave, np.nan), np.full_like(wave, np.nan)
        sf_HF, sf_LF                   = np.full_like(wave, np.nan), np.full_like(wave, np.nan)
        # Process each spectral order separately
        for i in range(len(lmin_orders) + 1):
            if i==0:                  # First order
                mask = wave < lmin_orders[i]
            elif i == len(lmin_orders): # Last order
                mask = wave > lmax_orders[i - 1]
            else:                       # Intermediate orders
                mask = (wave > lmax_orders[i - 1]) & (wave < lmin_orders[i])
            if np.any(mask): # Apply filtering if the mask is not empty
                star_flux_HF[mask], star_flux_LF[mask]     = filtered_flux(star_flux[mask], R=R, Rc=Rc, filter_type=filter_type)
                planet_flux_HF[mask], planet_flux_LF[mask] = filtered_flux(planet_flux[mask], R=R, Rc=Rc, filter_type=filter_type)
                sf_HF[mask], sf_LF[mask]                   = filtered_flux(star_flux[mask]/trans[mask], R=R, Rc=Rc, filter_type=filter_type)

    else:
        # Handling LF filtering edge effects due to the gaps bewteen the orders
        star_flux_HF, star_flux_LF     = filtered_flux(star_flux,   R=R, Rc=Rc, filter_type=filter_type)
        planet_flux_HF, planet_flux_LF = filtered_flux(planet_flux, R=R, Rc=Rc, filter_type=filter_type)
        _, trans_LF                    = filtered_flux(trans, R=R, Rc=Rc, filter_type=filter_type)
        f              = interp1d(wave[valid], star_flux_LF[valid], bounds_error=False, fill_value=np.nan) 
        star_flux_LF   = f(wave)
        f              = interp1d(wave[valid], planet_flux_LF[valid], bounds_error=False, fill_value=np.nan) 
        planet_flux_LF = f(wave)
        f              = interp1d(wave[valid], trans_LF[valid], bounds_error=False, fill_value=np.nan) 
        trans_LF       = f(wave)
        # masking inter order regions
        NV              = keep_true_chunks(nan_values, N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
        star_flux[NV]   = star_flux_LF[NV]
        planet_flux[NV] = planet_flux_LF[NV]
        trans[NV]       = trans_LF[NV]
        # HF / LF calculations
        star_flux_HF, star_flux_LF     = filtered_flux(star_flux, R=R, Rc=Rc, filter_type=filter_type)
        planet_flux_HF, planet_flux_LF = filtered_flux(planet_flux, R=R, Rc=Rc, filter_type=filter_type)
        sf_HF, sf_LF                   = filtered_flux(star_flux/trans, R=R, Rc=Rc, filter_type=filter_type)

    if reference_fibers:
        bkg_flux_HF = [0] * len(bkg_flux)
        bkg_flux_LF = [0] * len(bkg_flux)
        for i in range(len(bkg_flux)):
            bkg_flux_HF[i], bkg_flux_LF[i] = filtered_flux(bkg_flux[i], R=R, Rc=Rc, filter_type=filter_type)
        
    # Only apply a high pass filter to the data (no stellar subtraction) (if wanted)
    if only_high_pass: 
        planet_flux_HF, planet_flux_LF = filtered_flux(planet_flux/trans, R=R, Rc=Rc, filter_type=filter_type)
        d_planet                       = trans*planet_flux_HF
        if reference_fibers:
            bkg_flux_HF = [0] * len(bkg_flux)
            bkg_flux_LF = [0] * len(bkg_flux)
            d_bkg       = [0] * len(bkg_flux)
            for i in range(len(bkg_flux)):
                bkg_flux_HF[i], bkg_flux_LF[i] = filtered_flux(bkg_flux[i]/trans, R=R, Rc=Rc, filter_type=filter_type)
                d_bkg[i]                       = trans*bkg_flux_HF[i]
                
    # Standard molecular mapping post-processing
    else:
        d_planet = planet_flux - star_flux * planet_flux_LF / star_flux_LF # high pass planet spectrum extracted = trans*[Sp]_HF
        if reference_fibers:
            d_bkg = [0] * len(bkg_flux)
            for i in range(len(bkg_flux)):
                d_bkg[i] = bkg_flux[i] - star_flux * bkg_flux_LF[i] / star_flux_LF
                
    # Star high-pass filtered data
    d_star = trans*sf_HF # Considered as noise / background flux
        
    # Removing the flagged NaN values
    if mask_nan_values:
        mask = nan_values
    else: # If not masking raw NaN values, isolated nan values are "interpolated" but it is still needed to mask the gaps between the orders
        mask = keep_true_chunks(nan_values, N=50)
    if keep_only_good: # keeping only very good data (i.e. with weight == 1)
        mask = mask|(planet_weight<1)
    trans[mask]            = np.nan
    star_flux[mask]        = np.nan
    star_flux_LF[mask]     = np.nan
    star_flux_HF[mask]     = np.nan
    star_weight[mask]      = np.nan 
    star_sigma_tot[mask]   = np.nan
    star_sigma_bkg[mask]   = np.nan
    d_star[mask]           = np.nan
    planet_flux[mask]      = np.nan
    planet_flux_LF[mask]   = np.nan
    planet_flux_HF[mask]   = np.nan
    planet_weight[mask]    = np.nan
    planet_sigma_tot[mask] = np.nan
    planet_sigma_bkg[mask] = np.nan
    d_planet[mask]         = np.nan
    if reference_fibers:
        for i in range(len(d_bkg)):
            d_bkg[i][mask] = np.nan
        
    # (second final) OUTLIERS FILTERING (if wanted)
    if outliers: 
        sg             = sigma_clip(d_star, sigma=sigma_outliers)
        d_star         = np.array(np.ma.masked_array(d_star, mask=sg.mask).filled(np.nan))
        sg             = sigma_clip(star_sigma_tot, sigma=sigma_outliers)
        star_sigma_tot = np.array(np.ma.masked_array(star_sigma_tot, mask=sg.mask).filled(np.nan))
        sg             = sigma_clip(star_sigma_bkg, sigma=sigma_outliers)
        star_sigma_bkg = np.array(np.ma.masked_array(star_sigma_bkg, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(d_planet, sigma=sigma_outliers)
        d_planet         = np.array(np.ma.masked_array(d_planet, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(planet_sigma_tot, sigma=sigma_outliers)
        planet_sigma_tot = np.array(np.ma.masked_array(planet_sigma_tot, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(planet_sigma_bkg, sigma=sigma_outliers)
        planet_sigma_bkg = np.array(np.ma.masked_array(planet_sigma_bkg, mask=sg.mask).filled(np.nan))
        if reference_fibers:
            for i in range(len(d_bkg)):
                sg       = sigma_clip(d_bkg[i], sigma=sigma_outliers)
                d_bkg[i] = np.array(np.ma.masked_array(d_bkg[i], mask=sg.mask).filled(np.nan))

    # Plots
    if verbose and "fiber" not in target_name:
        # Plot of the filtered data
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), dpi=300, sharex=True)        
        axes[0].set_title("Companion's Signal", fontsize=14, fontweight="bold")
        axes[0].plot(wave, planet_flux, "r-", linewidth=2, label="LF+HF (Total Flux)")
        axes[0].plot(wave, planet_flux_LF, "g-", linewidth=2, label="LF (Low-Frequency)")
        axes[0].plot(wave, planet_flux_HF, "b-", linewidth=2, label="HF (High-Frequency)")
        axes[0].plot(wave, filtered_flux(planet_flux_HF, R=R, Rc=Rc, filter_type=filter_type)[1], "k:", linewidth=2, label="[HF]_LF (Filtered HF)")
        axes[0].set_ylabel("Flux", fontsize=12)
        axes[0].legend(fontsize=10, loc="best", frameon=True)
        axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axes[1].set_title("Star's Signal", fontsize=14, fontweight="bold")
        axes[1].plot(wave, star_flux, "r-", linewidth=2, label="LF+HF (Total Flux)")
        axes[1].plot(wave, star_flux_LF, "g-", linewidth=2, label="LF (Low-Frequency)")
        axes[1].plot(wave, star_flux_HF, "b-", linewidth=2, label="HF (High-Frequency)")
        axes[1].plot(wave, filtered_flux(star_flux_HF, R=R, Rc=Rc, filter_type=filter_type)[1], "k:", linewidth=2, label="[HF]_LF (Filtered HF)")        
        axes[1].set_xlim(np.nanmin(wave[~np.isnan(d_planet)]), np.nanmax(wave[~np.isnan(d_planet)]))
        axes[1].set_xlabel("Wavelength [µm]", fontsize=12)
        axes[1].set_ylabel("Flux", fontsize=12)
        axes[1].legend(fontsize=10, loc="best", frameon=True)
        axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)        
        fig.suptitle("Comparison of Companion & Star Signals", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.minorticks_on()
        plt.show()
        
        # Plot of the noise budget
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), dpi=300, sharex=True)    
        axes[0].plot(wave, planet_flux, color="gray", linestyle="-", linewidth=2, alpha=0.8, label=r"$S$ (Signal)")
        axes[0].plot(wave, np.sqrt(planet_flux), color="red", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{ph}$ (Photon Noise)")
        axes[0].plot(wave, planet_sigma_bkg, color="blue", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{bkg}$ (Background Noise)")
        axes[0].plot(wave, planet_sigma_tot, color="green", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{total}$ (Total Noise)")
        axes[0].set_yscale('log')
        axes[0].set_ylabel("Signal [e-]", fontsize=14)
        axes[0].set_title("Planet - Spectral Signal and Noise Components", fontsize=16, fontweight="bold")
        axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        axes[0].legend(fontsize=12, loc="best", frameon=True)
        axes[1].plot(wave, star_flux, color="gray", linestyle="-", linewidth=2, alpha=0.8, label=r"$S$ (Signal)")
        axes[1].plot(wave, np.sqrt(star_flux), color="red", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{ph}$ (Photon Noise)")
        axes[1].plot(wave, star_sigma_bkg, color="blue", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{bkg}$ (Background Noise)")
        axes[1].plot(wave, star_sigma_tot, color="green", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{total}$ (Total Noise)")
        axes[1].set_yscale('log')
        axes[1].set_xlim(np.nanmin(wave[~np.isnan(d_planet)]), np.nanmax(wave[~np.isnan(d_planet)]))
        axes[1].set_xlabel("Wavelength [µm]", fontsize=14)
        axes[1].set_ylabel("Signal [e-]", fontsize=14)
        axes[1].set_title("Star - Spectral Signal and Noise Components", fontsize=16, fontweight="bold")
        axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        axes[1].legend(fontsize=12, loc="best", frameon=True)    
        plt.tight_layout()
        plt.minorticks_on()
        plt.show()
        
    # NOISE ESTIMATION
    planet_sigma = np.sqrt( planet_sigma_tot**2 + star_sigma_tot**2 * (planet_flux_LF/star_flux_LF)**2 ) 
    
    # Weight function (if wanted)
    if use_weight:
        planet_weight  = (planet_weight + star_weight) / 2 # mean between the two weight functions
        planet_weight /= np.nanmax(planet_weight)
        planet_sigma  *= planet_weight # since the signals will be multiplied by the weight, the noise needs also to be multiplied by it
    else:
        planet_weight = None
        
    # Filtering fringes frequencies (if wanted)
    if cut_fringes:
        if "fiber" not in target_name:
            d_planet = cut_spectral_frequencies(input_flux=d_planet, R=R, Rmin=Rmin, Rmax=Rmax, show=verbose, target_name=target_name, force_new_calc=True)
        else:
            d_planet = cut_spectral_frequencies(input_flux=d_planet, R=R, Rmin=Rmin, Rmax=Rmax, show=False, target_name=target_name[:-7], force_new_calc=False)
    
    d_bkg.append(d_star)
        
    return wave, star_flux, d_star, planet_flux, d_planet, trans, trans_model, R, planet_sigma, planet_weight, T_star, lg_star, rv_star_obs, vsini_star, hdr, d_bkg



############################## Other functions ################################



def faded(color, alpha=0.1):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)



def mjd_to_date(mjd):
    """
    Convert a Modified Julian Date (MJD) to a human-readable date and time string in UTC.

    Parameters
    ----------
    mjd 
        The Modified Julian Date to convert.

    Returns
    -------
    str
        A string representing the corresponding date and time in the format "DD/MM/YYYY HH:MM".
    """
    # Create an astropy Time object from the MJD input
    t = Time(mjd, format='mjd')

    # Convert to datetime and format as a string with date and time (UTC)
    date_str = t.datetime.strftime("%d/%m/%Y %H:%M")

    return date_str



def are_planets_observable(latitude, longitude, altitude, planet_table, date_obs, min_elevation_deg=30.0, hours_span=6, ntimes=24):
    """
    Boolean mask of planets observable at least once around local midnight.

    A planet is “observable” if, for any time in a grid from -hours_span to
    +hours_span around local midnight, it has:
      - altitude > min_elevation_deg, AND
      - the Sun is below the horizon (civil night simplification).

    Parameters
    ----------
    latitude, longitude : float
        Site geodetic coordinates (deg).
    altitude : float
        Site altitude (meters).
    planet_table : astropy.table.Table/QTable
        Must provide 'RA' and 'Dec' in degrees.
    date_obs : str
        "DD/MM/YYYY" local calendar date.
    min_elevation_deg : float, optional
        Minimum altitude for observability (deg).
    hours_span : float, optional
        Half-width time window around local midnight (hours).
    ntimes : int, optional
        Number of time samples within the window.

    Returns
    -------
    np.ndarray (bool), shape (N_planets,)
        True if observable at least once.
    """
    # Site & local midnight
    site = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
    tzname = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)
    local_tz = pytz.timezone(tzname if tzname else "UTC")
    d0 = datetime.strptime(date_obs, "%d/%m/%Y")
    local_midnight = local_tz.localize(d0.replace(hour=0, minute=0, second=0, microsecond=0))

    # Time grid around local midnight
    delta = np.linspace(-hours_span, +hours_span, ntimes) * u.hour
    times = Time(local_midnight) + delta

    # Night mask (Sun below horizon)
    frame_all = AltAz(obstime=times, location=site)
    sun_alt = get_sun(times).transform_to(frame_all).alt
    is_night = (sun_alt < 0 * u.deg)  # (Nt,)

    # Planet coordinates (vectorized over planets)
    ra_deg  = np.asarray(planet_table["RA"].to_value(u.deg),  float)
    dec_deg = np.asarray(planet_table["Dec"].to_value(u.deg), float)
    coords  = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)

    # Altitude vs time (loop on times, vectorized on planets)
    alts = []
    for t in times:
        frame_t = AltAz(obstime=t, location=site)
        alts.append(coords.transform_to(frame_t).alt)
    alt_stack = u.Quantity(alts)  # (Nt, Np)

    # Observable if any time satisfies both constraints
    ok_elev = alt_stack > (min_elevation_deg * u.deg)
    obs = ok_elev & is_night[:, None]
    return np.any(obs, axis=0)



def propagate_coordinates_at_epoch(targetname, date, verbose=True):
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord, Distance
    from astropy.time import Time
    """Get coordinates at an epoch for some target, taking into account proper motions.
    Retrieves the SIMBAD coordinates, applies proper motion, returns the result as an
    astropy coordinates object
    from : https://github.com/jruffio/breads/blob/main/breads/utils.py
    """
    # Configure Simbad query to retrieve some extra fields
    if 'pmra' not in Simbad._VOTABLE_FIELDS:
        Simbad.add_votable_fields("pmra")  # Retrieve proper motion in RA
    if 'pmdec' not in Simbad._VOTABLE_FIELDS:
        Simbad.add_votable_fields("pmdec")  # Retrieve proper motion in Dec.
    if 'plx' not in Simbad._VOTABLE_FIELDS:
        Simbad.add_votable_fields("plx")  # Retrieve parallax
    if verbose:
        print(f"Retrieving SIMBAD coordinates for {targetname}")
    result_table = Simbad.query_object(targetname)
    # Get the coordinates and proper motion from the result table
    ra = result_table["RA"][0]
    dec = result_table["DEC"][0]
    pm_ra = result_table["PMRA"][0]
    pm_dec = result_table["PMDEC"][0]
    plx = result_table["PLX_VALUE"][0]
    # Create a SkyCoord object with the coordinates and proper motion
    target_coord_j2000 = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), pm_ra_cosdec=pm_ra * u.mas / u.year, pm_dec=pm_dec * u.mas / u.year, distance=Distance(parallax=plx * u.mas), frame='icrs', obstime='J2000.0')
    # Convert the desired date to an astropy Time object
    t = Time(date)
    # Calculate the updated SkyCoord object for the desired date
    host_coord_at_date = target_coord_j2000.apply_space_motion(new_obstime=t)
    if verbose:
        print(f"Coordinates at J2000:  {target_coord_j2000.icrs.to_string('hmsdms')}")
        print(f"Coordinates at {date}:  {host_coord_at_date.icrs.to_string('hmsdms')}")
    return host_coord_at_date



def get_coordinates_arrays(filename) :
    """ Determine the relative coordinates in the focal plane relative to the target.
        Compute the coordinates {wavelen, delta_ra, delta_dec, area} for each pixel in a 2D image
        Parameters
        ----------
        save_utils : bool
            Save the computed coordinates into the utils directory
        Returns
        -------
        wavelen_array: in microns
        dra_as_array: in arcsec
        ddec_as_array: in arcsec
        area2d: in arcsec^2
        from : https://github.com/jruffio/breads/blob/main/breads/instruments/jwstnirspec_cal.py#L338
        """
    try :
        wavelen_array = fits.getdata(filename.replace(".fits", "_wavelen_array.fits")) # µm
        dra_as_array = fits.getdata(filename.replace(".fits", "_dra_as_array.fits")) # arcsec
        ddec_as_array = fits.getdata(filename.replace(".fits", "_ddec_as_array.fits")) # arcsec
        area2d = fits.getdata(filename.replace(".fits", "_area2d.fits")) # arcsec^2
    except :
        import jwst.datamodels, jwst.assign_wcs
        from jwst.photom.photom import DataSet
        hdulist = fits.open(filename) #open file
        hdr0 = hdulist[0].header
        host_coord = propagate_coordinates_at_epoch(hdr0["TARGNAME"], hdr0["DATE-OBS"])
        host_ra_deg = host_coord.ra.deg
        host_dec_deg = host_coord.dec.deg
        shape = hdulist[1].data.shape #obtain generic shape of data
        calfile = jwst.datamodels.open(hdulist) #save time opening by passing the already opened file
        photom_dataset = DataSet(calfile)
        ## Determine pixel areas for each pixel, retrieved from a CRDS reference file
        area_fname = hdr0["R_AREA"].replace("crds://", os.path.join("/home/martoss/crds_cache", "references", "jwst", "nirspec") + os.path.sep)
        # Load the pixel area table for the IFU slices
        area_model = jwst.datamodels.open(area_fname)
        area_data = area_model.area_table
        wave2d, area2d, dqmap = photom_dataset.calc_nrs_ifu_sens2d(area_data)
        area2d[np.where(area2d == 1)] = np.nan
        wcses = jwst.assign_wcs.nrs_ifu_wcs(calfile)  # returns a list of 30 WCSes, one per slice. This is slow.
        #change this hardcoding?
        ra_array = np.zeros((2048, 2048)) + np.nan
        dec_array = np.zeros((2048, 2048)) + np.nan
        wavelen_array = np.zeros((2048, 2048)) + np.nan
        for i in range(len(wcses)):
                    print(f"Computing coords for slice {i}")
                    # Set up 2D X, Y index arrays spanning across the full area of the slice WCS
                    xmin = max(int(np.round(wcses[i].bounding_box.intervals[0][0])), 0)
                    xmax = int(np.round(wcses[i].bounding_box.intervals[0][1]))
                    ymin = max(int(np.round(wcses[i].bounding_box.intervals[1][0])), 0)
                    ymax = int(np.round(wcses[i].bounding_box.intervals[1][1]))
                    # print(xmax, xmin, ymax, ymin, ymax - ymin, xmax - xmin)
                    x = np.arange(xmin, xmax)
                    x = x.reshape(1, x.shape[0]) * np.ones((ymax - ymin, 1))
                    y = np.arange(ymin, ymax)
                    y = y.reshape(y.shape[0], 1) * np.ones((1, xmax - xmin))
                    # Transform all those pixels to RA, Dec, wavelength
                    skycoords, speccoord = wcses[i](x, y, with_units=True)
                    # print(skycoords.ra)
                    ra_array[ymin:ymax, xmin:xmax] = skycoords.ra
                    dec_array[ymin:ymax, xmin:xmax] = skycoords.dec
                    wavelen_array[ymin:ymax, xmin:xmax] = speccoord
        dra_as_array = (ra_array - host_ra_deg) * 3600 * np.cos(np.radians(dec_array)) # in arcsec
        ddec_as_array = (dec_array - host_dec_deg) * 3600 # in arcsec
        fits.writeto(filename.replace(".fits", "_wavelen_array.fits"), wavelen_array, overwrite=True)
        fits.writeto(filename.replace(".fits", "_dra_as_array.fits"), dra_as_array, overwrite=True)
        fits.writeto(filename.replace(".fits", "_ddec_as_array.fits"), ddec_as_array, overwrite=True)
        fits.writeto(filename.replace(".fits", "_area2d.fits"), area2d, overwrite=True)
    return wavelen_array, dra_as_array, ddec_as_array, area2d



def instru_thermal_background(temperature, emissivity, wavelengths_um):
    """
    Calcule le flux thermique en ph/s/µm/arcsec².

    Parameters:
        temperature (float): Température du télescope (K).
        emissivity (float): Émissivité du télescope.
        wavelengths_um (array): Longueurs d'onde (µm).
    
    Returns:
        flux_photon (array): Flux thermique en ph/s/µm/arcsec².
    """
    wavelengths_m = wavelengths_um * 1e-6  # Conversion µm -> m
    B_lambda = (2 * h * c**2) / (wavelengths_m**5) / (np.exp((h * c) / (wavelengths_m * kB * temperature)) - 1)  # W/m²/m/sr
    # Conversion en ph/s/µm/sr
    energy_per_photon = (h * c) / wavelengths_m
    B_lambda_ph       = B_lambda / energy_per_photon * 1e-6  # ph/s/m²/µm/sr

    flux = emissivity * B_lambda_ph / sr2arcsec2  # ph/s/µm/arcsec²
    return flux