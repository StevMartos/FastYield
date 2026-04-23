# import FastYield modules
from src.config import h, c, rad2arcsec, sim_data_path
from src.get_specs import get_config_data

# import astropy modules
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d
from scipy.ndimage import shift, distance_transform_edt, fourier_shift, convolve, gaussian_filter1d
from scipy.special import comb, j1, expit, logit
from scipy.stats import binned_statistic
from scipy.fft import fftn, ifftn
from scipy.optimize import lsq_linear

# import other modules
from tqdm import tqdm
from sklearn.decomposition import PCA
import statsmodels.api as sm
 


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
    
    elif not np.any(~mask_valid): # Only valid (finite) values
        return y

    else:
        if x is None:
            x = np.arange(len(y), dtype=float)
        return interp1d(x[mask_valid], y[mask_valid], kind="linear", bounds_error=False, fill_value=np.nan)(x)



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
        result = result + comb(N + n, n) * comb(2 * N + 1, N - n) * ((-t) ** n)
    result = result * t ** (N + 1)

    if filtering:
        # Symmetric shaping and inversion to get a bell-like rejection near Rc
        sym    = result[::-1]
        result = np.abs(sym + result - 1)

    return result



def power_law_extrapolation(x, x0, y0, slope):
    """
    Pure power-law extrapolation anchored at (x0, y0).

    Parameters
    ----------
    x : array_like
        Target radii.
    x0 : float
        Reference radius (> 0).
    y0 : float
        Value at x0 (> 0).
    slope : float
        Log-log slope. Must be negative for a decreasing profile.

    Returns
    -------
    ndarray
        Extrapolated values.
    """
    if x0 <= 0 or y0 <= 0:
        raise ValueError("x0 and y0 must be positive.")
    if slope >= 0:
        raise ValueError("'slope' must be negative.")

    x = np.asarray(x, dtype=float)
    return y0 * (x / x0)**slope



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

    N   = len(f)
    eps = 1e-32
    
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



def weighted_quantile(x, w, qs=(0.16, 0.5, 0.84)):
    """
    Compute weighted quantiles of a 1D sample.
    """
    x = np.asarray(x)
    w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return [np.nan]*len(qs)
    x   = x[m]
    w   = w[m]
    idx = np.argsort(x)
    x   = x[idx]
    w   = w[idx]
    cw  = np.cumsum(w)
    cw  = cw / cw[-1]
    return np.interp(qs, cw, x) # np.interp(x, xp, yp)



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



def airy_profile(wave, separation, sep_unit, D, eps):
    """
    Diffraction-limited PSF radial profile I(λ, ρ), normalized such that I(λ, 0) = 1.

    Parameters
    ----------
    wave : (Nwave,) array_like
        Wavelengths in microns (µm).
    separation : (Nsep,) array_like
        Angular separations in 'sep_unit'. Should include 0.
    sep_unit : {"rad", "arcsec", "mas"}
        Unit of 'separation'.
    D : float
        Telescope diameter in meters.
    eps : float
        Central obscuration ratio (0 for clear circular pupil). Must satisfy 0 <= eps < 1.

    Returns
    -------
    I : (Nwave, Nsep) ndarray
        Radial intensity profile for each wavelength, with I[:, separation==0] = 1.
    """
    wave       = np.asarray(wave,       dtype=float)
    separation = np.asarray(separation, dtype=float)

    if not (0.0 <= eps < 1.0):
        raise ValueError("eps must satisfy 0 <= eps < 1.")
    if sep_unit == "arcsec":
        separation = separation / rad2arcsec
    elif sep_unit == "mas":
        separation = separation * 1e-3 / rad2arcsec
    elif sep_unit != "rad":
        raise ValueError("sep_unit must be 'rad', 'arcsec', or 'mas'.")

    x = np.pi * D * separation[None, :] / (wave[:, None] * 1e-6)  # (Nwave, Nsep)

    A  = np.zeros_like(x)
    m0 = np.isclose(x, 0.0)  # robust zero mask
    m  = ~m0

    if eps == 0:
        A[m] = 2.0 * j1(x[m]) / x[m]
    else:
        A[m] = (2.0 * j1(x[m]) / x[m] - eps**2 * 2.0 * j1(eps * x[m]) / (eps * x[m])) / (1.0 - eps**2)

    A[m0] = 1.0
    return A**2



def get_r_core(separation, profile, level=0.5):
    """
    Return the radius r such that profile(r)/max(profile)=level (linear interpolation).
    Falls back to max(separation) if the crossing is not found.
    """
    
    I0 = profile / np.nanmax(profile)  # peak = 1

    ok  = np.isfinite(I0)
    idx = np.where(ok & (I0 <= level))[0]
    if len(idx) == 0:
        return separation[-1]  # or np.nan, depending on what you prefer

    i = idx[0]
    if i == 0:
        return separation[0]

    r1, r2 = separation[i-1], separation[i]
    y1, y2 = I0[i-1], I0[i]
    if not np.isfinite(y1) or not np.isfinite(y2) or y2 == y1:
        return separation[i]
    return r1 + (level - y1) * (r2 - r1) / (y2 - y1)



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



def compute_PSF_profile(PSF, pxscale, size_core, aperture_correction):
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
    aperture_correction : float
        Aperture correction factor: fraction of total flux inside the FoV.

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

    # Normalizing with total flux: np.nansum(PSF) = total flux inside FoV, aperture_correction: total flux inside FoV / total flux
    PSF_flux = np.nansum(PSF) / aperture_correction
    if not np.isfinite(PSF_flux) or PSF_flux <= 0:
        raise ValueError("PSF sum must be positive and finite.")
    PSF = PSF / PSF_flux

    # Optional local NaN-aware box average (size_core x size_core)
    if size_core > 1:
        PSF = box_convolution(data=PSF, size_core=size_core, mode="mean")

    # Fraction in the "core" (approximate central pixel block): box_convolution("mean") => PSF[y0, x0] = sum PSF(i, j) / A_FWHM (with (i, j) in FWHM and A_FWHM = size_core**2)
    fraction_core = size_core**2 * PSF[y0, x0]

    # Radial profile using 1-pixel-wide annuli out to N rings
    N       = int(round(np.sqrt((NbLine/2)**2+(NbColumn/2)**2)))
    profile = np.zeros((2, N))
    for r in range(N):
        r_int         = max(1, r - 1) if r > 1 else r
        r_ext         = r
        profile[0, r] = (r_int + r_ext)/2 * pxscale
        amask         = annular_mask(r_int, r_ext, size=(NbLine, NbColumn)) == 1
        profile[1, r] = np.nanmean(PSF[amask])
    profile[1, :] = profile[1, :] / pxscale**2
    
    # Sanity check 
    if np.any(profile[1]==0) or np.any(~np.isfinite(profile[1])):
        raise ValueError("Invalid values inside 'profile[1]'")
        
    return profile, fraction_core



def register_PSF_profile(instru, profile, fraction_core, band, strehl, apodizer, coronagraph=None):
    """
    Save a PSF radial profile and metadata to a FITS file.

    The file is stored under: sim_data/PSF/PSF_<instru>/
    with a name that encodes band, coronagraph (if any), strehl and apodizer.

    Parameters
    ----------
    instru : str
        Instrument name.
    profile : ndarray
        Output from 'compute_PSF_profile' (shape (2, N)).
    fraction_core : float
        Core flux fraction.
    band, strehl, apodizer : str
        Tags for the filename.
    coronagraph : str or None
        Optional coronagraph tag for the filename.
    """
    hdr       = fits.Header()
    hdr["FC"] = (fraction_core, "Core flux fraction")
    
    if coronagraph is None:
        psf_file = f"{sim_data_path}/PSF/PSF_{instru}/PSF_{band}_{strehl}_{apodizer}.fits"
    else:
        psf_file = f"{sim_data_path}/PSF/PSF_{instru}/PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits"

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
        FWHM_ang = FWHM_ang * 1000  # FWHM [mas]
    FWHM_px = FWHM_ang / pxscale  # FWHM [pixels]
    dpx     = int(round(FWHM_px)) + 1
    
    if len(data.shape) == 2:
        NbLine, NbColumn = data.shape
        if Y0 is None or X0 is None: # First guess
            Y0, X0 = np.unravel_index(np.nanargmax(data, axis=None), data.shape)
        img         = np.nan_to_num(data)
        ymin        = max(Y0 - 2*dpx    , 0)
        ymax        = min(Y0 + 2*dpx + 1, NbLine)
        xmin        = max(X0 - 2*dpx    , 0)
        xmax        = min(X0 + 2*dpx + 1, NbColumn)
        img_cropped = img[ymin:ymax, xmin:xmax]
        if np.any(np.isfinite(img_cropped)):
            if model == "gaussian":
                results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=FWHM_px, fwhmy=FWHM_px, sigfactor=sigfactor, threshold=True if sigfactor is not None else False, full_output=True, debug=debug)
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
            y_center, x_center = (NbLine-1)/2, (NbColumn-1)/2
            plt.figure(figsize=(14, 7), dpi=300)
            plt.suptitle(f"PSF fitting with {model.replace('_', ' ')} model", fontsize=22, fontweight="bold", color="#333")                
            plt.subplot(1, 2, 1)
            plt.axhline(0, color="k", linestyle="--")
            plt.fill_between(wave, y0-y_center - y0_err, y0-y_center + y0_err, color="crimson", alpha=0.3)
            plt.fill_between(wave, x0-x_center - x0_err, x0-x_center + x0_err, color="royalblue", alpha=0.3)
            plt.plot(wave, y0-y_center, label=f"y ("r"$\mu$"f" = {round(np.nanmean(y0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(y0_err), 3)} px)", linewidth=2.5, color="crimson")
            plt.plot(wave, x0-x_center, label=f"x ("r"$\mu$"f" = {round(np.nanmean(x0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(x0_err), 3)} px)", linewidth=2.5, color="royalblue")
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
            ax2.set_ylabel("PSF FWHM [mas]", fontsize=16, labelpad=20, rotation=270)
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
    Fill NaN or non-positive values in a centrally symmetric PSF
    using its mirrored counterpart.

    Works for both odd and even spatial dimensions.

    Parameters
    ----------
    psf : np.ndarray
        2D array (Ny, Nx) or 3D array (Nch, Ny, Nx).

    Returns
    -------
    filled_psf : np.ndarray
        PSF with invalid values filled by central symmetry when possible.
    """
    psf = np.array(psf, copy=True)

    if psf.ndim not in (2, 3):
        raise ValueError("psf must be a 2D or 3D array")

    was_2d = (psf.ndim == 2)

    if was_2d:
        psf = psf[np.newaxis, ...]

    NbChannel, NbLine, NbColumn = psf.shape

    for k in range(NbChannel):
        mask = np.isnan(psf[k]) | (psf[k] <= 0)

        for y in range(NbLine):
            for x in range(NbColumn):
                if mask[y, x]:
                    y_mirror = NbLine - 1 - y
                    x_mirror = NbColumn - 1 - x

                    value = psf[k, y_mirror, x_mirror]
                    if not np.isnan(value) and value > 0:
                        psf[k, y, x] = value

    return psf[0] if was_2d else psf



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
        Input data cube with dimensions (NbChannel, NbLine, NbColumn).
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
        NbChannel, NbLine, NbColumn = S_res.shape             # Retrieve the shape of the data cube
        pca                         = PCA(n_components=N_PCA) # Creating PCA object
        S_res_wo_planet             = np.copy(S_res)          # Create a copy of the input data for masking purposes
        
        # If planet coordinates are provided, apply the masks if specified
        if y0 is not None and x0 is not None:
            if PCA_annular:
                # Calculate the separation from the center to the planet
                planet_sep = int(round(np.sqrt((y0 - NbLine // 2) ** 2 + (x0 - NbColumn // 2) ** 2)))
                # Apply an annular mask with a given core size
                S_res_wo_planet = S_res_wo_planet * annular_mask(max(1, planet_sep - size_core - 1), planet_sep + size_core, value=np.nan, size=(NbLine, NbColumn))
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
        S_res_sub = np.reshape(S_res_sub, (NbChannel, NbLine, NbColumn))
        S_res_sub[nan_mask] = np.nan
        
        # ---------------------------
        # PCA plots
        # ---------------------------
        if PCA_plots:
            # Display up to the first 5 components interactively
            from src.spectrum import get_psd
            Nk      = min(N_PCA, 5)
            cmap    = plt.get_cmap("rainbow", Nk)
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
                    ax[k, 1].set_xlim(1, R)
                    ax[k, 1].set_ylim(1e-10, 1)
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
            pdf = PdfPages(path_PDF)
            # Create a colormap with N_PCA distinct colors
            N = N_PCA # min(N_PCA, 100)
            cmap_pdf = get_cmap("Spectral", N)
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
        sigma           = (Rmax - Rmin) / (2 * np.sqrt(2 * np.log(2)))  # assuming FWHM = Rmax - Rmin
        n               = 1 # Control the sharpness of the filter, n > 1 for super-gaussian
        G_filter        = (1 - np.exp(-0.5 * ((res_values - (Rmin + Rmax) / 2) / sigma) ** (2*n))) / 2 + (1 - np.exp(-0.5 * ((res_values + (Rmin + Rmax) / 2) / sigma) ** (2*n))) / 2
        filter_response = G_filter
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
            peak = peak / np.nanmax(peak)
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
    fft_values                = fft_values * filter_response
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
    sm.qqplot(map2, line=None, ax=ax, marker='o', markerfacecolor='crimson',   markeredgecolor='crimson',   alpha=0.6, label=f'sep > {sep_lim} {sep_unit}', lw=1)    
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
    """
    Load and preprocess a JWST IFU spectral cube (MIRI/MRS or NIRSpec/IFU).

    This function opens a JWST stage-3 spectral cube, extracts the science and
    error data, optionally crops the field of view, reconstructs the wavelength
    axis, and converts the calibrated fluxes from MJy/sr into integrated photon
    counts per pixel. If requested, the cube is further restricted to the
    selected instrumental band and converted into detector electrons per pixel
    using the instrument transmission model.

    For MIRI/MRS, the default file selection distinguishes between MIRISim
    products (assumed to be already split by band) and on-sky MAST products
    (assumed to be stored per channel). For NIRSpec/IFU, the default path
    points to the corresponding MAST cube.

    Parameters
    ----------
    instru : str
        Instrument name. Supported values are ''"MIRIMRS"'' and ''"NIRSpec"''.
    target_name : str
        Name of the target. This is used both for file-name construction and,
        when relevant, to determine whether the file corresponds to a simulation
        (e.g. if ''"sim"'' is contained in the name).
    band : str
        Spectral band to be considered. This must correspond to a valid entry in
        ''config_data['gratings']'' for the selected instrument.
    crop_band : bool, optional
        If ''True'', restrict the cube to the wavelength range of the selected
        band and convert the data from photons per pixel to electrons per pixel
        using the instrumental transmission. Default is ''True''.
    outliers : bool, optional
        If ''True'', apply a spectral sigma-clipping independently to each
        spatial pixel of the science and error cubes. Flagged values are set to
        ''NaN''. Default is ''False''.
    sigma_outliers : float, optional
        Sigma threshold used for spectral sigma-clipping when ''outliers=True''.
        Default is ''5''.
    file : str or None, optional
        Path to the FITS cube to open. If ''None'', a default path is built from
        ''instru'', ''target_name'', and ''band''. Default is ''None''.
    crop_cube : bool, optional
        If ''True'', crop the spatial dimensions of the science and error cubes
        using ''crop_both''. Default is ''True''.
    X0 : int or float or None, optional
        X-coordinate of the crop center passed to ''crop_both''. Used only if
        ''crop_cube=True''. Default is ''None''.
    Y0 : int or float or None, optional
        Y-coordinate of the crop center passed to ''crop_both''. Used only if
        ''crop_cube=True''. Default is ''None''.
    R_crop : int or float or None, optional
        Cropping radius passed to ''crop_both''. Used only if
        ''crop_cube=True''. Default is ''None''.
    verbose : bool, optional
        If ''True'', print a short summary of the loaded dataset, including
        wavelength range, approximate spectral resolution, pixel scale,
        exposure time, and selected FITS header keywords. Default is ''True''.

    Returns
    -------
    cube : ndarray of shape (n_wave, ny, nx)
        Science cube after preprocessing. The output unit is:
        
        - photons per pixel if ''crop_band=False'',
        - electrons per pixel if ''crop_band=True''.
        
        Empty slices at the spectral edges are removed, and zero-valued pixels
        are replaced by ''NaN''.
    wave : ndarray of shape (n_wave,)
        Wavelength axis in microns.
    pxscale : float
        Spatial pixel scale in arcsec/pixel.
    err : ndarray of shape (n_wave, ny, nx)
        Error cube after preprocessing, expressed in the same unit as ''cube''.
    trans : float or ndarray
        Instrumental transmission used for the conversion from photons to
        electrons. If ''crop_band=True'', this is a 1D array sampled on
        ''wave''. Otherwise, it is set to ''1''.
    exposure_time : float
        Effective total exposure time in minutes.
    DIT : float
        Effective integration time per exposure in minutes.
    
    Notes
    -----
    - The wavelength axis is reconstructed from the FITS header using::

          wave = (np.arange(NAXIS3) + CRPIX3 - 1) * CDELT3 + CRVAL3

      which is appropriate for the linear wavelength sampling assumed here.
    - The input science and error cubes are assumed to be calibrated in
      ''MJy/sr''.
    - The conversion to photon counts uses the telescope collecting area
      provided by ''get_config_data(instru)''.
    - When ''crop_band=True'', the conversion to electrons per pixel relies on
      ''get_transmission(...)'' and on the aperture-correction factor ''AC''
      stored in the corresponding PSF FITS header.
    - For MIRI/MRS on-sky channel cubes named
      ''*_chX-shortmediumlong_s3d.fits'', the effective exposure time is
      assumed to be one third of ''EFFEXPTM''.
    - Spectral outlier rejection is performed independently for each spatial
      pixel, along the wavelength axis.

    Raises
    ------
    KeyError
        If the provided instrument name is not supported.

    See Also
    --------
    get_config_data : Load instrument configuration parameters.
    crop_both : Crop science and error cubes consistently.
    get_transmission : Compute the instrumental transmission curve.
    sigma_clip : Sigma-clipping routine used for outlier rejection.
    """
    
    # Instrument specs
    config_data = get_config_data(instru)
    area        = config_data['telescope']['area'] # collective area m2

    # Opening file
    if file is None :
        # MIRI/MRS
        if instru=="MIRIMRS" : 
            # Simulations data
            if "sim" in target_name.lower():
                file = f"data/MIRIMRS/MIRISim/{target_name}_{band}_s3d.fits"
            # On-sky data
            else :
                file = f"data/MIRIMRS/MAST/{target_name}_ch{band[0]}-shortmediumlong_s3d.fits"
        # NIRSpec/IFU
        elif instru=="NIRSpec":
            file = f"data/NIRSpec/MAST/{target_name}_nirspec_{band}_s3d.fits"
        # Unknown instrument
        else:
            raise KeyError(f"Unknown instrument {instru}")
    f = fits.open(file)
    
    # Retrieving header values
    hdr0 = f[0].header
    hdr1 = f[1].header
    # MIRI/MRS
    if instru=="MIRIMRS" :
        # Files already per band
        if "sim" in target_name.lower() or "shortmediumlong" not in file: # MIRISIM are already per band
            exposure_time = f[0].header['EFFEXPTM']/60 # [mn]
        # Files per channel
        else:
            target_name   = hdr0['TARGNAME']
            exposure_time = f[0].header['EFFEXPTM']/3/60 # [mn]
    # NIRSpec/IFU
    elif instru == "NIRSpec":
        target_name   = hdr0['TARGNAME']
        exposure_time = f[0].header['EFFEXPTM']/60 # [mn]
    DIT         = f[0].header['EFFINTTM']/60 # [mn]
    pxsteradian = hdr1['PIXAR_SR']           # [Sr/px]
    pxscale     = hdr1['CDELT1']*3600        # [arcsec/px]
    dwave       = hdr1['CDELT3']             # [µm/bin]

    # Wavelength axis
    wave = (np.arange(hdr1['NAXIS3']) + hdr1['CRPIX3'] - 1) * hdr1['CDELT3'] + hdr1['CRVAL3'] # [µm]

    # Retrieving data
    cube = f[1].data # [MJy/Sr]
    err  = f[2].data # [MJy/Sr]
    
    # Centering the max (if needed)
    if crop_cube:
        cube, err = crop_both(cube, err, X0=X0, Y0=Y0, R_crop=R_crop)
    
    # Data shapes
    NbChannel, NbLine, NbColumn = cube.shape
    
    # Converting data in total [ph/px]
    cube *= pxsteradian*1e6                   # [MJy/Sr]       => [Jy/px]
    cube *= 1e-26                             # [Jy/pixel]     => [J/s/m²/Hz/px]
    cube *= c/((wave[:, None, None]*1e-6)**2) # [J/s/m²/Hz/px] => [J/s/m²/m/px]
    cube *= dwave*1e-6                        # [J/s/m²/m/px]  => [J/s/m²/px]
    cube *= wave[:, None, None]*1e-6/(h*c)    # [J/s/m²/px]    => [ph/s/m²/px]
    cube *= area                              # [ph/s/m²/px]   => [ph/s/px]
    cube *= exposure_time*60                  # [ph/s/m²/px]   => [ph/px]
    
    err *= pxsteradian*1e6                   # [MJy/Sr]       => [Jy/px]
    err *= 1e-26                             # [Jy/pixel]     => [J/s/m²/Hz/px]
    err *= c/((wave[:, None, None]*1e-6)**2) # [J/s/m²/Hz/px] => [J/s/m²/m/px]
    err *= dwave*1e-6                        # [J/s/m²/m/px]  => [J/s/m²/px]
    err *= wave[:, None, None]*1e-6/(h*c)    # [J/s/m²/px]    => [ph/s/m²/px]
    err *= area                              # [ph/s/m²/px]   => [ph/s/px]
    err *= exposure_time*60                  # [ph/s/m²/px]   => [ph/px]
    
    # Cropping to the considered band and converting in [e-/px] (if needed)
    if crop_band: 
        lmin      = config_data['gratings'][band].lmin
        lmax      = config_data['gratings'][band].lmax
        band_mask = (wave >= lmin) & (wave <= lmax)
        cube      = cube[band_mask]
        err       = err[band_mask]
        wave      = wave[band_mask]
        NbChannel = cube.shape[0]
        from src.get_specs import get_transmission
        trans = get_transmission(instru, wave, band, tellurics=False, apodizer="NO_SP", strehl="NO_JQ", coronagraph=None)
        AC    = fits.getheader(f"{sim_data_path}/PSF/PSF_{instru}/PSF_{band}_NO_JQ_NO_SP.fits")['AC']
        cube *= trans[:, None, None]/AC # [ph/px] => [e-/px] (AC is obviously already taken into account)
        err  *= trans[:, None, None]/AC # [ph/px] => [e-/px] (AC is obviously already taken into account)
    else:
        trans = 1

    # Cropping first and last empty slices
    valid_slices = np.array([np.any(np.isfinite(cube[i]) & (cube[i] != 0)) for i in range(NbChannel)]) # Identify valid slices (not all zeros or NaNs)    
    start        = np.argmax(valid_slices)  # first True -> first valid slice
    end          = len(valid_slices) - np.argmax(valid_slices[::-1])  # last True + 1 -> last valid slice
    wave         = wave[start:end] # Crop only the empty slices at the beginning and the end
    cube         = cube[start:end]
    err          = err[start:end]
    if crop_band:
        trans = trans[start:end]    
    NbChannel = cube.shape[0] # Update the number of channels
    
    # Flagging outliers (if needed)
    cube[cube==0] = np.nan
    err[err==0]   = np.nan
    if outliers:
        CUBE = cube.reshape(NbChannel, -1).copy()
        ERR  = err.reshape(NbChannel,  -1).copy()
        for k in range(CUBE.shape[1]):
            if np.all(np.isnan(CUBE[:, k])):
                continue
            mask_CUBE = sigma_clip(np.ma.masked_invalid(CUBE[:, k]), sigma=sigma_outliers, masked=True).mask
            mask_ERR  = sigma_clip(np.ma.masked_invalid(ERR[:, k]),  sigma=sigma_outliers, masked=True).mask
            mask          = mask_CUBE | mask_ERR
            CUBE[:, k][mask] = np.nan
            ERR[:, k][mask] = np.nan
        cube = CUBE.reshape(NbChannel, NbLine, NbColumn)
        err  = ERR.reshape(NbChannel,   NbLine, NbColumn)
    
    # Printing (if needed)
    if verbose :
        from src.spectrum import get_resolution
        R = get_resolution(wavelength=wave, func=np.nanmedian)
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
    # f   = np.fft.fftfreq(N)          # cycles / HR sample
    # res = 2.0 * R_sampling * f

    # Initially white noise
    TF_noise = np.ones(N, dtype=np.complex128) # initially white noise

    # 1) Optional initial HR coloring at Rc_init
    if Rc_init is not None and filter_init_type is not None:
        ftype_init = "gaussian_fast" if (filter_init_type == "gaussian") else filter_init_type
        TF_noise = TF_noise * _fft_filter_response(N=N, R=R_sampling, Rc=Rc_init, filter_type=ftype_init)[1]

    # 2) Optional extra low-pass
    if Rc_noise is not None and filter_noise_type is not None:
        ftype_lp = "gaussian_fast" if (filter_noise_type == "gaussian") else filter_noise_type
        TF_noise = TF_noise * _fft_filter_response(N=N, R=R_sampling, Rc=Rc_noise, filter_type=ftype_lp)[1]

    # 3) Optional convolution broadening
    if Rc_conv is not None and filter_conv_type is not None:
        ftype_cv = "gaussian_fast" if (filter_conv_type == "gaussian") else filter_conv_type
        TF_noise = TF_noise * _fft_filter_response(N=N, R=R_sampling, Rc=Rc_conv, filter_type=ftype_cv)[1]
    
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
        C = IFFT(|H|^2), then C = C/C[0],
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
    # f   = np.fft.fftfreq(M)          # cycles / HR sample
    # res = 2.0 * R_sampling * f

    # --- Total frequency response H(f) ---
    H = np.ones(M, dtype=np.complex128)

    # 1) Optional initial HR coloring at Rc_init
    if Rc_init is not None and filter_init_type is not None:
        ftype_init = "gaussian_fast" if (filter_init_type == "gaussian") else filter_init_type
        H          = H * _fft_filter_response(N=M, R=R_sampling, Rc=Rc_init, filter_type=ftype_init)[1]

    # 2) Optional extra low-pass
    if Rc_noise is not None and filter_noise_type is not None:
        ftype_lp = "gaussian_fast" if (filter_noise_type == "gaussian") else filter_noise_type
        H        = H * _fft_filter_response(N=M, R=R_sampling, Rc=Rc_noise, filter_type=ftype_lp)[1]

    # 3) Optional convolution broadening
    if Rc_conv is not None and filter_conv_type is not None:
        ftype_cv = "gaussian_fast" if (filter_conv_type == "gaussian") else filter_conv_type
        H        = H * _fft_filter_response(N=M, R=R_sampling, Rc=Rc_conv, filter_type=ftype_cv)[1]

    # --- Normalized autocovariance on the HR grid: C = IFFT(|H|^2), with C[0]=1 ---
    C   = np.fft.ifft(np.abs(H)**2).real
    C   = C/C[0]
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



def extract_vipa_data(instru, target_name, gain, label_fiber, degrade_data=True, outliers=False, sigma_outliers=5, use_weight=True, mask_nan_values=False, filter_noise=False, R_target=80_000, Rc=100, filter_type="gaussian", verbose=True):
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
         sigma = sigma * sqrt(fn_LF),
       where 'fn_LF' is the fraction of input noise variance transmitted through the LP,
       computed with 'compute_filter_variance_factor(…, this_filter=(Rc, type))'.
    5) Optional resolution degradation to R_target < R_instru by Gaussian convolution.
       The cutoff equivalent (Rc_conv) is derived from sigma_kernel; uncertainties are
       scaled by the corresponding variance fraction (as above).
    6) Rebin to a Nyquist grid at R_target:
         λ_Nyquist step = λ_centre / (2*R_target)  (here approximated with the band midpoint).
       Rebinning uses top-hat averaging. Because pre-filtering introduces correlations,
       the naive i.i.d. error propagation is corrected by a per-bin factor:
         sigma_rebinned = sigma_rebinned * sqrt(F),
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
    from src.spectrum import filtered_flux, rebin_spectrum_mean

    f                                                   = fits.open(f"data/{instru}/VIPA_Final_Spectrum_{target_name}_gain_{gain}_fiber_{label_fiber}.fits")
    wave0, flux0, sigma0, weight0, trans0, sigma_trans0 = f[0].data
    wave0                                               = wave0 * 1e-3         # [nm] => [µm]
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
    nan_values0  = ~np.isfinite(flux0)
    nan_values0 |= ~np.isfinite(trans0)
    
    # (first) OUTLIERS FILTERING (if wanted)
    if outliers:
        NbNaN0 = nan_values0.sum()
        flux0_HF     = filtered_flux(flux0, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values0 |= sigma_clip(np.ma.masked_invalid(flux0_HF), sigma=sigma_outliers).mask
        trans0_HF    = filtered_flux(trans0, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values0 |= sigma_clip(np.ma.masked_invalid(trans0_HF), sigma=sigma_outliers).mask
        flux0_HF     = trans0 * filtered_flux(flux0/trans0, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values0 |= sigma_clip(np.ma.masked_invalid(flux0_HF), sigma=sigma_outliers).mask
        print(f"{nan_values0.sum() - NbNaN0} outliers found...")
        weight0[nan_values0]      = np.nan
        flux0[nan_values0]        = np.nan
        sigma0[nan_values0]       = np.nan
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
    
    noise0 = noise0 * np.sqrt(np.nanmean(sigma0**2)) / np.nanstd(noise0)
    
    # Low-pass (if wanted): every "signal" above R_instru is filtered and is assumed to be noise
    if filter_noise:
        Rc_noise          = R_instru
        filter_noise_type = "gaussian"
        
        flux0  = filtered_flux(flux=flux0,  R=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type)[1]
        noise0 = filtered_flux(flux=noise0, R=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type)[1]
        
        # Assuming constant sigma: estimating the power fraction of noise that would be filtered
        fn_LF  = compute_filter_variance_factor(N=len(wave0), R_sampling=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=None, filter_noise_type=None, Rc_conv=None, filter_conv_type=None)[1]
        sigma0 = sigma0 * np.sqrt(fn_LF)
        
        trans0       = filtered_flux(flux=trans0, R=R_sampling, Rc=Rc_noise, filter_type=filter_noise_type)[1]
        sigma_trans0 = sigma_trans0 * np.sqrt(fn_LF)
        
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
            fn_LF  = compute_filter_variance_factor(N=len(wave0), R_sampling=R_sampling, Rc=Rc_conv, filter_type=filter_conv_type, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=Rc_noise, filter_noise_type=filter_noise_type, Rc_conv=None, filter_conv_type=None)[1]
            sigma0 = sigma0 * np.sqrt(fn_LF)
            
            trans0       = filtered_flux(flux=trans0,  R=R_sampling, Rc=Rc_conv, filter_type=filter_conv_type)[1]
            sigma_trans0 = sigma_trans0 * np.sqrt(fn_LF)
        
        # --- Rebinning to Nyquist sampling
        
        # Nyquist sampled wavelength axis
        R     = R_target
        dl    = (lmin + lmax)/2 /(2*R)
        wave  = np.arange(lmin, lmax, dl)
        dwave = np.gradient(wave)
        
        # Interpolating weight
        nan_values = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan)(wave) != 0
        
        # Rebinning data
        flux, sigma, weight   = rebin_spectrum_mean(specHR=flux0,  sigmaHR=sigma0,       weightHR=weight0, lamHR=wave0, lamLR=wave)
        noise, _, _           = rebin_spectrum_mean(specHR=noise0, sigmaHR=None,         weightHR=None,    lamHR=wave0, lamLR=wave)
        trans, sigma_trans, _ = rebin_spectrum_mean(specHR=trans0, sigmaHR=sigma_trans0, weightHR=None,    lamHR=wave0, lamLR=wave)
        
        # from src.spectrum import rebin_spectrum_overlap
        # flux, sigma, weight, _   = rebin_spectrum_overlap(specHR=flux0,  sigmaHR=sigma0, weightHR=weight0, lamHR=wave0, lamLR=wave, dlam=dwave)
        # noise, _, _, _           = rebin_spectrum_overlap(specHR=noise0, sigmaHR=None,   weightHR=None,    lamHR=wave0, lamLR=wave, dlam=dwave)
        # trans, sigma_trans, _, _ = rebin_spectrum_overlap(specHR=trans0,  sigmaHR=sigma_trans0, weightHR=None, lamHR=wave0, lamLR=wave, dlam=dwave)
        
        # Rebinning does not propagate correctly the sigmas since it assumses i.d.d noise:
        fn_rebin    = compute_rebin_variance_factor(lamHR=wave0, maskHR=~nan_values0, lamLR=wave, dlam=dwave, R_sampling=R_sampling, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=Rc_noise, filter_noise_type=filter_noise_type, Rc_conv=Rc_conv, filter_conv_type=filter_conv_type)
        sigma       = sigma       * np.sqrt(fn_rebin)
        sigma_trans = sigma_trans * np.sqrt(fn_rebin)
        
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
    flux_HF  = trans * filtered_flux(flux/trans,  R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
    noise_HF = trans * filtered_flux(noise/trans, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]        
    
    # Assuming constant sigma: estimating the power fraction of noise that would be filtered
    fn_HF    = compute_filter_variance_factor(N=len(wave), R_sampling=R_sampling, Rc=Rc, filter_type=filter_type, Rc_init=Rc_init, filter_init_type=filter_init_type, Rc_noise=Rc_noise, filter_noise_type=filter_noise_type, Rc_conv=Rc_conv, filter_conv_type=filter_conv_type)[0]
    sigma_HF = np.sqrt(fn_HF) * sigma
    
    # (final) OUTLIERS FILTERING (if wanted)
    if outliers:
        NbNaN       = nan_values.sum()
        flux_HF     = filtered_flux(flux, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values |= sigma_clip(np.ma.masked_invalid(flux_HF), sigma=sigma_outliers).mask
        trans_HF    = filtered_flux(trans, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values |= sigma_clip(np.ma.masked_invalid(trans_HF), sigma=sigma_outliers).mask
        flux_HF     = trans * filtered_flux(flux/trans, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
        nan_values |= sigma_clip(np.ma.masked_invalid(flux_HF), sigma=sigma_outliers).mask
        print(f"{nan_values.sum() - NbNaN} outliers found...")

    # Removing the flagged NaN values
    if mask_nan_values:
        weight[nan_values]      = np.nan
        flux[nan_values]        = np.nan
        sigma[nan_values]       = np.nan
        noise[nan_values]       = np.nan
        trans[nan_values]       = np.nan
        sigma_trans[nan_values] = np.nan
        flux_HF[nan_values]     = np.nan
        sigma_HF[nan_values]    = np.nan
        noise_HF[nan_values]    = np.nan

    # Weight function (if wanted)
    if use_weight:
        weight   = weight / np.nanmax(weight)
        sigma    = sigma     * weight # since the signals (flux, flux_HF) will be multiplied by the weight, the noise needs also to be multiplied by it
        noise    = noise    * weight
        sigma_HF = sigma_HF * weight
        noise_HF = noise_HF * weight
    else:
        weight = None
    
    # Sigma propagation sanity check
    if verbose:
        print(f"\nrms_z      = {np.sqrt(np.nanmean((noise / sigma)**2)):.3f} ({np.nanstd(noise) / np.sqrt(np.nanmean(sigma**2)):.3f})")
        print(f"rms_z (HF) = {np.sqrt(np.nanmean((noise_HF / sigma_HF)**2)):.3f} ({np.nanstd(noise_HF) / np.sqrt(np.nanmean(sigma_HF**2)):.3f})")
    
    # Filters dictionnary
    filters = dict(Rc_init=Rc_init, filter_init_type_init=filter_init_type, Rc_noise=Rc_noise, filter_noise_type_init=filter_noise_type, Rc_conv=Rc_conv, filter_conv_type_init=filter_conv_type)
    
    return wave, flux, sigma, noise, flux_HF, sigma_HF, noise_HF, weight, trans, sigma_trans, exposure_time, filters, R



###########################################################################################



def extract_hirise_data(target_name, interpolate, degrade_resolution, R, Rc, filter_type, order_by_order, outliers, sigma_outliers, only_high_pass=False, cut_fringes=False, Rmin=None, Rmax=None, use_weight=True, mask_nan_values=False, keep_only_good=False, wave_input=None, reference_fibers=True, crop_tell_orders=False, shift_star_corr=False, verbose=True): # OPENING DATA AND DEGRADATING THE RESOLUTION (if wanted)
    from src.spectrum import Spectrum, filtered_flux, interpolate_flux_with_error, estimate_resolution
    
    # hard coded values
    # GAIN     = np.nanmean([2.28, 2.19, 2.00]) # in e-/ADU
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
        # star_wave0      = data_star["wave"]*1e-3        # in µm, recalibrated (in heliocentric referential for star)
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
        # planet_wave0      = data_planet["wave"]*1e-3        # in µm, recalibrated (in heliocentric referential for comp)
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
        sg              = sigma_clip(np.ma.masked_invalid(trans0_HF), sigma=2*sigma_outliers)
        trans0          = np.array(np.ma.masked_array(trans0, mask=sg.mask).filled(np.nan))
        trans_model0_HF = filtered_flux(trans_model0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(np.ma.masked_invalid(trans_model0_HF), sigma=2*sigma_outliers)
        trans_model0    = np.array(np.ma.masked_array(trans_model0, mask=sg.mask).filled(np.nan))
        star_flux0_HF   = filtered_flux(star_flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(np.ma.masked_invalid(star_flux0_HF), sigma=2*sigma_outliers)
        star_flux0      = np.array(np.ma.masked_array(star_flux0, mask=sg.mask).filled(np.nan))
        planet_flux0_HF = filtered_flux(planet_flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(np.ma.masked_invalid(planet_flux0_HF), sigma=2*sigma_outliers)
        planet_flux0    = np.array(np.ma.masked_array(planet_flux0, mask=sg.mask).filled(np.nan))
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0_HF = filtered_flux(bkg_flux0[i], R=R0, Rc=Rc, filter_type=filter_type)[0]
                sg           = sigma_clip(np.ma.masked_invalid(bkg_flux0_HF), sigma=2*sigma_outliers)
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
        star_flux0        = star_flux0        / dwave0 # [e-/µm]
        star_sigma_tot0   = star_sigma_tot0   / dwave0 # [e-/µm]
        star_sigma_bkg0   = star_sigma_bkg0   / dwave0 # [e-/µm]
        planet_flux0      = planet_flux0      / dwave0 # [e-/µm]
        planet_sigma_tot0 = planet_sigma_tot0 / dwave0 # [e-/µm]
        planet_sigma_bkg0 = planet_sigma_bkg0 / dwave0 # [e-/µm]
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0[i] = bkg_flux0[i] / dwave0 # [e-/µm]
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
        star_flux0        = star_flux0        * dwave # [e-/bin]
        star_sigma_tot0   = star_sigma_tot0   * dwave # [e-/bin]
        star_sigma_bkg0   = star_sigma_bkg0   * dwave # [e-/bin]
        planet_flux0      = planet_flux0      * dwave # [e-/bin]
        planet_sigma_tot0 = planet_sigma_tot0 * dwave # [e-/bin]
        planet_sigma_bkg0 = planet_sigma_bkg0 * dwave # [e-/bin]
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0[i] = bkg_flux0[i] * dwave # [e-/bin]
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
        sg             = sigma_clip(np.ma.masked_invalid(d_star), sigma=sigma_outliers)
        d_star         = np.array(np.ma.masked_array(d_star, mask=sg.mask).filled(np.nan))
        sg             = sigma_clip(np.ma.masked_invalid(star_sigma_tot), sigma=sigma_outliers)
        star_sigma_tot = np.array(np.ma.masked_array(star_sigma_tot, mask=sg.mask).filled(np.nan))
        sg             = sigma_clip(np.ma.masked_invalid(star_sigma_bkg), sigma=sigma_outliers)
        star_sigma_bkg = np.array(np.ma.masked_array(star_sigma_bkg, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(np.ma.masked_invalid(d_planet), sigma=sigma_outliers)
        d_planet         = np.array(np.ma.masked_array(d_planet, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(np.ma.masked_invalid(planet_sigma_tot), sigma=sigma_outliers)
        planet_sigma_tot = np.array(np.ma.masked_array(planet_sigma_tot, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(np.ma.masked_invalid(planet_sigma_bkg), sigma=sigma_outliers)
        planet_sigma_bkg = np.array(np.ma.masked_array(planet_sigma_bkg, mask=sg.mask).filled(np.nan))
        if reference_fibers:
            for i in range(len(d_bkg)):
                sg       = sigma_clip(np.ma.masked_invalid(d_bkg[i]), sigma=sigma_outliers)
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
        planet_weight = (planet_weight + star_weight) / 2 # mean between the two weight functions
        planet_weight = planet_weight / np.nanmax(planet_weight)
        planet_sigma  = planet_sigma * planet_weight # since the signals will be multiplied by the weight, the noise needs also to be multiplied by it
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



############################## PSF interpolation+extrapolation over separation, lambda, IWA and WFE ################################

def interp_extrap_sep(separation_ref, separation_new, y_ref, mode="log", tail_model="powerlaw"):
    """
    Interpolate a 1D radial profile onto a new separation grid, with explicit edge handling.

    This helper evaluates a profile 'y(separation)' sampled on a strictly increasing
    reference grid 'separation_ref' at a new grid 'separation_new'. Interpolation is
    performed in a transformed space to enforce basic constraints (positivity or boundedness),
    and extrapolation is handled explicitly on both ends:

    - For 'separation_new < separation_ref[0]' (inner side): values are clamped to 'y_ref[0]'.
    - For 'separation_new > separation_ref[-1]' (outer side):
        * 'tail_model="flat"'    : clamp to 'y_ref[-1]'.
        * 'tail_model="powerlaw"': fit a log-log slope on the
          reference tail and extrapolate using 'power_law_extrapolation'. This option is only
          supported for 'mode="log"' (positive quantities).

    Parameters
    ----------
    separation_ref : (N,) array_like
        Reference separation grid. Must be strictly increasing.
    separation_new : (M,) array_like
        Target separation grid where the profile is evaluated.
    y_ref : (N,) array_like
        Profile values sampled on 'separation_ref'.
    mode : {"log", "logit", "linear"}, optional
        Interpolation transform:
        - "log"   : interpolate 'log(y_ref)' then exponentiate (for positive quantities).
        - "logit" : interpolate 'logit(y_ref)' then apply sigmoid (for quantities in (0, 1)).
        - "linear": interpolate 'y_ref' directly (no constraint).
    tail_model : {"flat", "powerlaw"}, optional
        Extrapolation rule for 'separation_new > separation_ref[-1]'.
        '"powerlaw"' is only allowed when 'mode="log"'.

    Returns
    -------
    y_new : (M,) ndarray
        Interpolated (and extrapolated) profile evaluated on 'separation_new'.

    """
    
    if np.any(np.diff(separation_ref) <= 0):
        raise ValueError("separation_ref must be strictly increasing.")
    if mode not in {"log", "logit", "linear"}:
        raise ValueError("mode must be 'log', 'logit', or 'linear'.")
    if tail_model not in {"flat", "powerlaw"}:
        raise ValueError("tail_model must be 'flat' or 'powerlaw'.")
    if tail_model == "powerlaw" and mode != "log":
        raise ValueError("tail_model='powerlaw' is only supported for mode='log'.")

    # --- Interpolation according to different mode --- 
    if mode == "log": # For PSF profile
        y_new = np.exp(interp1d(separation_ref, np.log(y_ref), bounds_error=False, axis=0, assume_sorted=True, fill_value=np.nan)(separation_new))
    elif mode == "logit": # For fraction core and radial transmission
        y_new = expit(interp1d(separation_ref,  logit(y_ref),  bounds_error=False, axis=0, assume_sorted=True, fill_value=np.nan)(separation_new))
    elif mode == "linear": # Linear interpolation with clamping.
        y_new = interp1d(separation_ref,        y_ref,         bounds_error=False, axis=0, assume_sorted=True, fill_value=np.nan)(separation_new)
    
    # --- Extrapolation according to different mode --- 
    m_lo = separation_new < separation_ref[0]
    m_hi = separation_new > separation_ref[-1]
    
    if np.any(m_lo):
        y_new[m_lo] = y_ref[0]
        
    if np.any(m_hi):
        if tail_model == "powerlaw":
            
            # Fit log(y_ref) = alpha*log(rho) + const on the tail of the reference profile.
            x = separation_ref
            y = y_ref

            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.sum(valid) < 2: # If the fit cannot be performed, fall back to flat clamping.
                y_new[m_hi] = y_ref[-1]
            else:
                logx = np.log(x[valid])
                logy = np.log(y[valid])
    
                slope, _ = np.polyfit(logx, logy, 1)   # slope should be negative
                
                if (not np.isfinite(slope)) or (slope >= 0):
                    y_new[m_hi] = y_ref[-1]
                else:
                    y_new[m_hi] = power_law_extrapolation(x=separation_new[m_hi], x0=separation_ref[-1], y0=y_ref[-1], slope=slope
                                                          )
        elif tail_model == "flat": # Flat extrapolation
            y_new[m_hi] = y_ref[-1]

    return y_new



def warp_sep_wave(wave_ref, wave_new, separation, y_ref, mode="log", amp_power=0.0, tail_model="powerlaw"):
    """
    Warp a radial profile from a reference wavelength to a new wavelength using λ/D scaling.

    In the diffraction-dominated regime, the dominant chromatic effect is a geometric rescaling
    of angular coordinates with wavelength (ρ ∝ λ). A profile measured at 'wave_ref' can be
    mapped to 'wave_new' by evaluating it at a rescaled separation:

        rho' = rho * (wave_ref / wave_new)

    The resampled profile is obtained by calling :func:'interp_extrap_sep' on the rescaled
    separations. Optionally, a wavelength-dependent amplitude scaling can be applied:

        y_out = y_warped * (wave_ref / wave_new)**amp_power

    Parameters
    ----------
    wave_ref, wave_new : float
        Reference and target wavelengths (same units).
    separation : (N,) array_like
        Separation grid (ρ) where the output profile is returned.
    y_ref : (N,) array_like
        Reference profile sampled on 'separation' at wavelength 'wave_ref'.
    mode : {"log", "logit", "linear"}, optional
        Transform used for the separation interpolation (passed to :func:'interp_extrap_sep').
    amp_power : float, optional
        Exponent for the optional amplitude scaling
        '(wave_ref / wave_new)**amp_power' applied after the geometric warp.
        For example, a halo term scaling approximately as λ^-2 corresponds to 'amp_power=2'.
    tail_model : {"flat", "powerlaw"}, optional
        Outer-separation extrapolation rule passed to :func:'interp_extrap_sep'.
        
    Returns
    -------
    y_out : (N,) ndarray
        Warped (and optionally amplitude-scaled) profile evaluated on the original 'separation' grid.

    Notes
    -----
    This function only performs the chromatic warp in separation plus an optional global amplitude
    scaling. Any additional chromatic PSF evolution not captured by λ/D scaling must be modeled
    separately.
    """
    
    # Evaluate the reference profile at x = rho * (wave_ref/wave_new)
    separation_ref = separation
    separation_new = separation * (float(wave_ref) / float(wave_new))
    y_new          = interp_extrap_sep(separation_ref=separation_ref, separation_new=separation_new, y_ref=y_ref, mode=mode, tail_model=tail_model)
    
    # Optional wavelength-dependent amplitude scaling.
    return y_new * (float(wave_ref) / float(wave_new)) ** float(amp_power)



def interp_extrap_wave(wave_ref, wave_new, separation, PSF_profile_density_2D, fraction_core_2D, radial_transmission_2D):
    """
    Interpolate (and extrapolate) wavelength-dependent radial profiles onto a new wavelength grid.

    This routine builds (wave, separation) maps on a target wavelength grid 'wave_new' from
    reference maps sampled on 'wave_ref':

    - Inside the reference band '[min(wave_ref), max(wave_ref)]':
        * 'PSF_profile_density_2D' is interpolated in log-log space:
          interpolate 'log(PSF)' vs 'log(wave)' and exponentiate.
        * 'fraction_core_2D' is interpolated in logit-log space:
          interpolate 'logit(fraction_core)' vs 'log(wave)' and apply sigmoid.
        * 'radial_transmission_2D' (if provided) is treated identically to 'fraction_core_2D'.

      Values outside the band are left as NaN at this stage.

    - Outside the reference band:
        the nearest band-edge profile is used as a reference and extrapolated by λ/D warping
        in separation using :func:'warp_sep_wave'. For the PSF halo profile, an additional
        amplitude scaling '(wave_edge / wave_new)**2' is applied ('amp_power=2') and a
        power-law outer tail extrapolation in separation can be used ('tail_model="powerlaw"').

    Parameters
    ----------
    wave_ref : (Nw,) array_like
        Reference wavelength grid. Must be finite and strictly increasing.
    wave_new : (Nn,) array_like
        Target wavelength grid (may extend beyond the reference band).
    separation : (Nr,) array_like
        Separation grid (ρ). Must be finite and strictly increasing.
    PSF_profile_density_2D : (Nw, Nr) array_like
        Stellar PSF halo profile (positive quantity) as a function of wavelength and separation.
    fraction_core_2D : (Nw, Nr) array_like
        Encircled-energy/core flux fraction within a chosen aperture (typically in (0, 1)),
        as a function of wavelength and separation.
    radial_transmission_2D : (Nw, Nr) array_like or None
        Off-axis (coronagraph) transmission map in (0, 1), as a function of wavelength and separation.
        If None, it is ignored and the function returns 'None' for the extrapolated output.

    Returns
    -------
    PSF_profile_density_2D_new : (Nn, Nr) ndarray
        PSF halo profile evaluated on 'wave_new' and 'separation'.
    fraction_core_2D_new : (Nn, Nr) ndarray
        Core flux fraction evaluated on 'wave_new' and 'separation'.
    radial_transmission_2D_new : (Nn, Nr) ndarray or None
        Radial transmission evaluated on 'wave_new' and 'separation' (or None if input was None).

    Raises
    ------
    ValueError
        If input grids are not strictly increasing / finite, or if the input arrays do not
        match the expected shapes.

    Notes
    -----
    This method is intended as a lightweight chromatic extrapolation scheme when a full optical
    propagation is not available. Out-of-band behavior is modeled through λ/D separation warping
    (plus an optional λ^-2 amplitude scaling for the PSF halo) and does not capture all possible
    chromatic effects (e.g., wavelength-dependent aberrations, apodizer/chromatic coronagraph
    response beyond the provided transmission map, detector/instrument effects).
    """
    
    if np.any(np.diff(wave_ref) <= 0):
        raise ValueError("wave_ref must be strictly increasing.")
    if np.any(~np.isfinite(wave_ref)) or np.any(~np.isfinite(wave_new)):
        raise ValueError("wave_ref and wave_new must be finite.")        
    if np.any(~np.isfinite(separation)) or np.any(np.diff(separation) <= 0):
        raise ValueError("separation must be finite and strictly increasing.")

    wmin, wmax   = wave_ref.min(), wave_ref.max()
    i_min, i_max = 0, -1

    # --- Interpolation within the band; outside becomes NaN ---
    # PSF:                        log-log   in lambda 
    # fraction_core/transmission: log-logit in lambda
    PSF_profile_density_2D_new     = np.exp(interp1d(np.log(wave_ref), np.log(PSF_profile_density_2D), bounds_error=False, axis=0, assume_sorted=True, fill_value=np.nan)(np.log(wave_new)))
    fraction_core_2D_new           = expit(interp1d(np.log(wave_ref),  logit(fraction_core_2D),        bounds_error=False, axis=0, assume_sorted=True, fill_value=np.nan)(np.log(wave_new)))
    if radial_transmission_2D is not None:
        radial_transmission_2D_new = expit(interp1d(np.log(wave_ref),  logit(radial_transmission_2D),  bounds_error=False, axis=0, assume_sorted=True, fill_value=np.nan)(np.log(wave_new)))
    else:
        radial_transmission_2D_new = None

    # --- Extrapolation out-of-band wavelengths using λ/D warping from the nearest edge profile ---
    for i, w in enumerate(wave_new):
        if w < wmin:
            PSF_profile_density_2D_new[i]     = warp_sep_wave(wave_ref=wmin, wave_new=w, separation=separation, y_ref=PSF_profile_density_2D[i_min], mode="log",   amp_power=2.0, tail_model="powerlaw")
            fraction_core_2D_new[i]           = warp_sep_wave(wave_ref=wmin, wave_new=w, separation=separation, y_ref=fraction_core_2D[i_min],       mode="logit", amp_power=0.0, tail_model="flat")
            if radial_transmission_2D is not None:
                radial_transmission_2D_new[i] = warp_sep_wave(wave_ref=wmin, wave_new=w, separation=separation, y_ref=radial_transmission_2D[i_min], mode="logit", amp_power=0.0, tail_model="flat")
        elif w > wmax:
            PSF_profile_density_2D_new[i]     = warp_sep_wave(wave_ref=wmax, wave_new=w, separation=separation, y_ref=PSF_profile_density_2D[i_max], mode="log",   amp_power=2.0, tail_model="powerlaw")
            fraction_core_2D_new[i]           = warp_sep_wave(wave_ref=wmax, wave_new=w, separation=separation, y_ref=fraction_core_2D[i_max],       mode="logit", amp_power=0.0, tail_model="flat")
            if radial_transmission_2D is not None:
                radial_transmission_2D_new[i] = warp_sep_wave(wave_ref=wmax, wave_new=w, separation=separation, y_ref=radial_transmission_2D[i_max], mode="logit", amp_power=0.0, tail_model="flat")

    return PSF_profile_density_2D_new, fraction_core_2D_new, radial_transmission_2D_new



# ==========================================================
# VARYING AO PERFORMANCE (r_WFE) AND CORONAGRAPH IWA (r_IWA)
# ==========================================================

def warp_sep_IWA(separation, y_ref, r_IWA, mode="log", tail_model="powerlaw"):
    """
    Warp a set of radial profiles to emulate a change of coronagraph inner working angle (IWA).

    This function rescales the separation axis according to::

        rho_new = rho / r_IWA

    so that increasing 'r_IWA' effectively shifts the profile features outward in separation
    (i.e., a larger IWA makes a given throughput/response occur at larger angular separations).

    The warping is applied independently for each wavelength slice of 'y_ref' by calling
    :func:'interp_extrap_sep', which handles both interpolation in the requested transform
    space and explicit extrapolation rules at the low/high separation ends.

    Parameters
    ----------
    separation : (N_sep,) array_like
        Reference separation grid (rho). Must be strictly increasing.
    y_ref : (N_wave, N_sep) array_like
        Reference radial profiles for each wavelength. Each row 'y_ref[i]' is sampled on
        'separation'.
    r_IWA : float
        Dimensionless IWA scaling factor. The new evaluation grid is 'separation / r_IWA'.
        Must be finite and > 0.
    mode : {"log", "logit", "linear"}, optional
        Transform space used by :func:'interp_extrap_sep':
        - "log"   : interpolate in log-space (positive quantities, preserves dynamic range).
        - "logit" : interpolate in logit-space (quantities in (0, 1)).
        - "linear": plain linear interpolation.
    tail_model : {"flat", "powerlaw"}, optional
        High-separation extrapolation model passed to :func:'interp_extrap_sep'.
        - "flat"    : clamp to the last value.
        - "powerlaw": log-log power-law fit on the tail (only supported for 'mode="log"').

    Returns
    -------
    y_new : (N_wave, N_sep) ndarray
        Warped profiles evaluated back on the original 'separation' grid.

    Notes
    -----
    - This is a geometric rescaling in separation only; it does not modify profile amplitudes.
    - Low-separation extrapolation is clamped to the first sample of each wavelength slice.
    """
    
    # Evaluate the reference curve at rho/r_IWA
    separation_ref = separation
    separation_new = separation / float(r_IWA)

    N_wave, N_sep = y_ref.shape
    y_new         = np.zeros((N_wave, N_sep), dtype=float)

    for iw in range(N_wave):        
        # Evaluate the reference profile at x = rho * (wave_ref/wave_new)
        y_new[iw] = interp_extrap_sep(separation_ref=separation_ref, separation_new=separation_new, y_ref=y_ref[iw], mode=mode, tail_model=tail_model)

    return y_new 



def interp_radial_per_wave(separation, quantity_2D, r_eval, left, right):
    """
    Interpolate a radial quantity of shape (N_wave, N_sep) at radius r_eval.
    r_eval can be a scalar.
    Returns an array of shape (N_wave,).
    """
    out = np.empty(quantity_2D.shape[0], dtype=float)
    if np.ndim(left) == 0:
        left = np.zeros((quantity_2D.shape[0])) + left
    if np.ndim(right) == 0:
        right = np.zeros((quantity_2D.shape[0])) + right
    for iw in range(quantity_2D.shape[0]):
        out[iw] = np.interp(r_eval, separation, quantity_2D[iw], left=left[iw], right=right[iw])
    return out




def estimate_SR(instru, apodizer, strehl, PSF_3D, PSF_DL_2D_lD, pxscale_data, pxscale_DL_lD, FOV, D, wave, aperture_correction, wave_output):
    """
    Estimate an effective Strehl-ratio spectrum and an equivalent achromatic WFE
    from post-AO PSF cubes by comparison with a diffraction-limited PSF.
    
    The input post-AO PSF cube and the diffraction-limited (DL) reference PSF are
    first brought onto a common spatial sampling and cropped to the same field of
    view (FoV). Both cubes are then normalized using the provided aperture
    correction, which approximates the fraction of total flux enclosed in the
    simulated FoV.
    
    For each wavelength, the function estimates an effective Strehl ratio within
    the PSF core by fitting the post-AO core as a weighted linear combination of:
    
        PSF_postAO ~= alpha * PSF_DL + beta
    
    inside a circular region of radius equal to half the DL FWHM. The fitted
    coefficient 'alpha' is used as the effective Strehl estimate at that wavelength.
    This quantity should be interpreted as the amplitude of the diffraction-limited
    component in the PSF core rather than as a strict peak-to-peak Strehl ratio.
    
    The resulting 'SR_raw(wave)' is then converted into an equivalent WFE using the
    Maréchal-like relation:
    
        WFE_rad = sqrt(-log(SR))
    
    and summarized by a single scalar 'WFE_nm', defined as the median WFE over the
    input wavelength grid.
    
    Depending on the instrument, the final Strehl spectrum on 'wave_output' is
    constructed differently:
    - for "HARMONI", a wavelength-independent WFE in nm is assumed, and the
      Strehl is reconstructed from this constant WFE;
    - for "ANDES", the raw Strehl spectrum is interpolated/extrapolated directly
      onto 'wave_output'.
    
    Parameters
    ----------
    instru : str
        Instrument name. Currently used to choose how the output Strehl spectrum is
        propagated to 'wave_output' (e.g. ""HARMONI"" or ""ANDES"").
    
    apodizer : str
        Name of the apodizer, only used for plot labeling.
    
    strehl : str
        AO regime label, only used for plot labeling.
    
    PSF_3D : ndarray of shape (N_wave, Ny, Nx)
        Post-AO PSF cube sampled on the detector grid. The cube is assumed to be
        centered approximately on the optical axis and defined over a finite FoV.
    
    PSF_DL_2D_lD : ndarray of shape (Ny_DL, Nx_DL)
        Diffraction-limited PSF computed on a finely sampled focal grid expressed
        in units of lambda/D per pixel.
    
    pxscale_data : float
        Pixel scale of 'PSF_3D' in mas/pixel.
    
    pxscale_DL_lD : float
        Pixel scale of 'PSF_DL_2D_lD' in units of lambda/D per pixel.
    
    FOV : float
        Field of view size in mas used for the comparison. The PSFs are cropped to
        this FoV after rebinning.
    
    D : float
        Telescope diameter in meters.
    
    wave : ndarray of shape (N_wave,)
        Wavelength grid in microns associated with 'PSF_3D'.
    
    aperture_correction : ndarray of shape (N_wave,)
        Estimated fraction of total flux enclosed within the simulated FoV.
        This is used to normalize both the DL and post-AO PSFs to an approximate
        total-flux convention.
    
    wave_output : ndarray
        Wavelength grid in microns on which the final Strehl spectrum is returned.
    
    Returns
    -------
    SR : ndarray of shape (len(wave_output),)
        Estimated effective Strehl-ratio spectrum on 'wave_output'.
    
    WFE_nm : float
        Median equivalent wavefront error in nanometers, inferred from the raw
        Strehl spectrum over 'wave'.
    
    Notes
    -----
    - The returned Strehl is an *effective* Strehl-like quantity derived from a
      weighted fit of the DL core, not a strict classical Strehl ratio defined as
      the ratio of PSF peaks.
    - The weighting emphasizes the brightest part of the DL core, making the fit
      more sensitive to the central diffraction-limited component than to the local
      halo pedestal.
    - The aperture correction is assumed to be known externally. When the total
      flux of the post-AO PSF is not directly available because the simulations are
      FoV-limited, this correction typically acts as a proxy for the missing flux.
    - The function produces a diagnostic plot of the raw and output Strehl spectra.
    """
    # Rebinning to the greatest pxscale
    lD         = wave*1e-6 / D * 1000*rad2arcsec # lambda/D [mas/(lambda/D)]
    pxscale_DL = pxscale_DL_lD * lD              # DL pxscale [mas/px]
    pxscale    = max(np.nanmax(pxscale_DL), pxscale_data)
    
    # Scales and mask
    R_max    = int(round(FOV/2/pxscale)) # [px] FoV radius (FoV/2 in pixels)
    NbLine   = 2*R_max+1
    NbColumn = 2*R_max+1
    y0, x0   = NbLine//2, NbColumn//2        # Spatial center
    Y, X     = np.ogrid[:NbLine, :NbColumn]
    r_px     = np.sqrt((Y - y0)**2 + (X - x0)**2)
    r        = r_px * pxscale
    N_wave   = len(wave)
    
    # --- Rebinning and retrieving the PSF for each wavelength ---
    PSF_DL_3D              = np.zeros((N_wave, NbLine, NbColumn))
    for i in tqdm(range(len(wave)), desc="Estimating the Strehl ratio"):
                
        # Cropping to the FoV (in order to make computation faster)
        R_2FOV       = int(round(2*FOV/2/pxscale_DL[i])) # 2*FOV (to be sure)
        PSF_DL_2D = crop(PSF_DL_2D_lD, Y0=PSF_DL_2D_lD.shape[0]//2, X0=PSF_DL_2D_lD.shape[1]//2, R_crop=R_2FOV)
                
        # Rebinning the PSF at the band pxscale
        PSF_DL_2D = rebin_flux_conserving(data=PSF_DL_2D, pxscale_in=pxscale_DL[i], pxscale_out=pxscale)
        
        # Cropping in order to have same PSF sizes
        PSF_DL_2D = crop(PSF_DL_2D, R_crop=R_max)
        
        # Saving the array
        PSF_DL_3D[i] = PSF_DL_2D
    
    # Normalizing
    PSF_DL_3D /= np.nansum(PSF_DL_3D, axis=(1, 2))[:, None, None] / aperture_correction[:, None, None]
    
    # Cropping and normalizing
    PSF_3D  = rebin_flux_conserving(data=PSF_3D, pxscale_in=pxscale_data, pxscale_out=pxscale)
    PSF_3D  = crop(PSF_3D, R_crop=R_max)
    PSF_3D  = fill_invalid_by_symmetry(PSF_3D)
    PSF_3D  = fill_invalid_by_nearest(PSF_3D)
    PSF_3D /= np.nansum(PSF_3D, axis=(1, 2))[:, None, None] / aperture_correction[:, None, None]
    
    
    # # Sanity check plots
    # plot_PSF(instru=instru, coronagraph=None, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit="mas", wave=wave, PSF_3D=PSF_DL_3D, type_PSF="DL",      model_PSF="gaussian", sigfactor=None, debug=True)
    # plot_PSF(instru=instru, coronagraph=None, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit="mas", wave=wave, PSF_3D=PSF_3D,    type_PSF="post-AO", model_PSF="gaussian", sigfactor=None, debug=True)

    
    # Computing SR
    SR_raw = np.zeros((N_wave))
    for i, l0 in enumerate(wave):
        r_core    = 1.029 * l0*1e-6 / D * 1000*rad2arcsec # DL FWHM radius [mas]
        mask_core = (r <= r_core)
        
        # PSF       = PSF_3D[i][mask_core]
        # PSF_DL    = PSF_DL_3D[i][mask_core]
        # SR_raw[i] = np.nansum(PSF*PSF_DL) / np.nansum(PSF_DL**2)

        x = PSF_DL_3D[i][mask_core].ravel()
        y = PSF_3D[i][mask_core].ravel()
        
        A = np.column_stack([x, np.ones_like(x)])
        w = x / np.nanmax(x)
        W = np.sqrt(w)
        
        A_w = A * W[:, None]
        y_w = y * W
        
        res         = lsq_linear(A_w, y_w, bounds=([0.0, 0.0], [1.0, np.inf]))
        alpha, beta = res.x
        alpha       = max(alpha, 1e-8)
        SR_raw[i]   = min(alpha, 1.0)

    SR_raw      = np.clip(SR_raw, 1e-8, 1.0)
    WFE_rad_raw = np.sqrt(-np.log(SR_raw))
    WFE_nm      = np.nanmedian(1e3*wave / (2*np.pi) * WFE_rad_raw)
    if instru=="HARMONI":
        WFE_rad = WFE_nm / (1e3*wave_output / (2*np.pi))
        SR      = np.exp(-WFE_rad**2)
    elif instru == "ANDES":
        SR = interp1d(wave, SR_raw, bounds_error=False, fill_value="extrapolate")(wave_output)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(wave_output, 100*SR,     c="black")
    plt.plot(wave,        100*SR_raw, c="gray", ls="--")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.title(f"Strehl Ratio for {instru} with {apodizer.replace('_', ' ')}-apodizer in {strehl} \n WFE = {WFE_nm:.1f} nm", fontsize=14)
    plt.xlabel("Wavelength [µm]", fontsize=12)
    plt.ylabel("SR [%]",          fontsize=12)
    plt.axvspan(wave[0], wave[-1], color="black", alpha=0.1, lw=0, label="Simulated data range")
    plt.xlim(wave_output[0], wave_output[-1])
    plt.ylim(0, 100)
    plt.show()
            
    return SR, WFE_nm
    


def build_PSF_grid(SR, D, wave, r_WFE, r_IWA, separation, PSF_profile_density_no_coro_2D, fraction_core_no_coro_2D, PSF_profile_density_DL_2D, fraction_core_DL_2D, coronagraph, IWA_ref, PSF_profile_density_coro_2D, fraction_core_coro_2D, radial_transmission_2D, PSF_profile_density_speck_2D=None, eps=1e-8):
    """
    Build a 4D grid of radial PSF quantities as a function of wavelength, WFE scaling,
    and coronagraphic IWA scaling.
    
    This routine constructs radial profiles and core-flux fractions for a reference PSF
    model and propagates them over a grid of wavefront-error scalings ('r_WFE') and,
    when applicable, coronagraph inner-working-angle scalings ('r_IWA').
    
    The non-coronagraphic PSF is modeled as the sum of a diffraction-limited core and a
    speckle halo. The reference halo fraction is estimated from the input post-AO and
    diffraction-limited profiles, then regularized spectrally/radially. For each WFE
    scaling, the diffraction-limited contribution is updated through the Strehl ratio,
    while the halo is rescaled approximately as (1 - SR).
    
    If a coronagraph is provided, the reference coronagraphic stellar halo is assumed to
    be represented by 'PSF_profile_density_coro_2D'. The coronagraphic radial profile is
    then warped with IWA, mixed heuristically with the non-coronagraphic profile using a
    leakage-based coefficient, and paired with an updated radial transmission curve.
    
    Parameters
    ----------
    SR : array-like of shape (N_wave,)
        Reference Strehl ratio as a function of wavelength. Values must lie in [0, 1].
        This defines the reference WFE through the Maréchal-like proxy
        "WFE_rad = sqrt(-log(SR))".
    
    D : float
        Telescope diameter in meters. Used to estimate a characteristic smoothing scale
        for the speckle-halo fraction regularization.
    
    wave : array-like of shape (N_wave,)
        Wavelength axis in microns. Must have the same length as the first dimension of
        the 2D input arrays.
    
    r_WFE : array-like of shape (N_WFE,)
        Multiplicative scaling factors applied to the reference WFE proxy.
        'r_WFE = 1' corresponds to the reference PSF.
        'r_WFE = 0' corresponds to a diffraction-limited case.
    
    r_IWA : array-like of shape (N_IWA,)
        Multiplicative scaling factors applied to the reference coronagraphic IWA.
        Only used when 'coronagraph is not None'.
        'r_IWA = 1' corresponds to the reference IWA.
    
    separation : array-like of shape (N_sep,)
        Monotonically increasing radial separation grid in mas. The first value must be 0.
        This grid is used both:
          1. as the radial coordinate of the PSF profiles, and
          2. as the offset grid with respect to the coronagraphic mask for the
             coronagraphic throughput/core-fraction inputs.
    
    PSF_profile_density_no_coro_2D : ndarray of shape (N_wave, N_sep)
        Reference post-AO non-coronagraphic radial profile:
        mean surface-brightness density in each annular bin divided by the total
        non-coronagraphic flux.
    
    fraction_core_no_coro_2D : ndarray of shape (N_wave, N_sep)
        Reference post-AO non-coronagraphic flux fraction enclosed in the PSF core,
        divided by the total non-coronagraphic flux.
        In practice this quantity is usually constant with separation in the
        non-coronagraphic case, but it is kept as a 2D array for interface uniformity.
    
    PSF_profile_density_DL_2D : ndarray of shape (N_wave, N_sep)
        Diffraction-limited non-coronagraphic radial profile:
        mean surface-brightness density in each annular bin divided by the total
        non-coronagraphic flux.
    
    fraction_core_DL_2D : ndarray of shape (N_wave, N_sep)
        Diffraction-limited flux fraction enclosed in the PSF core, divided by the total
        non-coronagraphic flux.
    
    coronagraph : object or None
        Coronagraph descriptor. Its content is not used directly in this routine, but its
        presence controls whether the coronagraphic branch is computed.
        If 'None', all coronagraphic outputs are returned as 'None'.
    
    IWA_ref : float or None
        Reference inner working angle in mas. Must be strictly positive when a
        coronagraph is used.
    
    PSF_profile_density_coro_2D : ndarray of shape (N_wave, N_sep) or None
        Reference post-AO coronagraphic radial profile:
        mean surface-brightness density in each annular bin divided by the total
        coronagraphic flux.
        This quantity is interpreted as the reference stellar coronagraphic halo profile.
    
    fraction_core_coro_2D : ndarray of shape (N_wave, N_sep) or None
        Reference post-AO coronagraphic flux fraction enclosed in the PSF core,
        divided by the total coronagraphic flux.
        For large separations from the mask, the off-axis value is assumed to be
        represented by the last separation bin ('sep = separation[-1]').
    
    radial_transmission_2D : ndarray of shape (N_wave, N_sep) or None
        Reference coronagraphic throughput:
        total flux with coronagraph divided by total flux without coronagraph,
        as a function of source offset from the coronagraphic mask.
        Required when 'coronagraph is not None'.
    
    eps : float, optional
        Small numerical floor used to stabilize divisions, clipping, and logit transforms.
    
    Returns
    -------
    PSF_profile_density_4D : ndarray of shape (N_wave, N_WFE, N_IWA, N_sep)
        Non-coronagraphic radial profile grid:
        mean surface-brightness density in each annular bin divided by the total
        non-coronagraphic flux.
    
    fraction_core_4D : ndarray of shape (N_wave, N_WFE, N_IWA, N_sep)
        Non-coronagraphic core-flux fraction grid:
        total flux enclosed in the core divided by the total non-coronagraphic flux.
    
    PSF_profile_density_coro_4D : ndarray of shape (N_wave, N_WFE, N_IWA, N_sep) or None
        Coronagraphic radial profile grid:
        mean surface-brightness density in each annular bin divided by the total
        coronagraphic flux.
        Returned only when 'coronagraph is not None', otherwise 'None'.
    
    fraction_core_coro_4D : ndarray of shape (N_wave, N_WFE, N_IWA, N_sep) or None
        Coronagraphic core-flux fraction grid:
        total flux enclosed in the core divided by the total coronagraphic flux.
        Returned only when 'coronagraph is not None', otherwise 'None'.
    
    radial_transmission_4D : ndarray of shape (N_wave, N_WFE, N_IWA, N_sep) or None
        Coronagraphic throughput grid:
        total flux with coronagraph divided by total flux without coronagraph,
        as a function of offset from the mask.
        Returned only when 'coronagraph is not None', otherwise 'None'.
    
    Notes
    -----
    - The PSF core is assumed to be defined elsewhere as a fixed 'size_core x size_core'
      pixel box.
    - The non-coronagraphic PSF is modeled heuristically as
      "I_nc ≈ SR * I_DL + I_speck".
    - The reference non-coronagraphic speckle-halo fraction is estimated as
      "eta = I_speck / I_nc", then interpolated and smoothed before being reused.
    - The coronagraphic profile update is heuristic: it relies on radial warping with IWA,
      a leakage-based mixing between coronagraphic and non-coronagraphic shapes, and an
      affine remapping of the throughput curve in logit space.
    - This routine produces a diagnostic plot of the regularized halo fraction 'eta'
      and therefore has a plotting side effect.
    - The function assumes that all 2D input arrays are sampled on the same
      '(wave, separation)' grid.
    
    """
    # Safety helpers and basic dimensions
    safe01 = lambda x: np.clip(x, eps, 1 - eps)
    
    N_WFE         = len(r_WFE)
    N_IWA         = len(r_IWA)
    N_wave, N_sep = PSF_profile_density_no_coro_2D.shape
    
    if radial_transmission_2D is None and coronagraph is not None:
        raise ValueError("'radial_transmission_2D' should not be None")
    if radial_transmission_2D is not None and coronagraph is None:
        raise ValueError("'coronagraph' should not be None")
    if coronagraph is not None:
        w_coronagraph = True
    else:
        w_coronagraph = False
    if not np.isclose(separation[0], 0.0, atol=eps):
        raise ValueError("separation[0] must be 0.")
    if w_coronagraph and (PSF_profile_density_coro_2D is None or fraction_core_coro_2D is None):
        raise ValueError("'PSF_profile_density_coro_2D' or 'fraction_core_coro_2D' should not be None")
    if w_coronagraph and (IWA_ref is None or IWA_ref <= 0):
        raise ValueError("'IWA_ref' should be > 0")
    if np.any(np.diff(separation) <= 0):
        raise ValueError("'separation' must be strictly increasing")
    if np.any(np.asarray(r_IWA) <= 0):
        raise ValueError("'r_IWA' must contain only positive values")
    if np.any(np.asarray(r_WFE) < 0):
        raise ValueError("'r_WFE' must contain only non-negative values")
    if len(wave) != N_wave:
        raise ValueError("'wave' must have length N_wave")
    
    
    # Strehl => WFE proxy
    SR      = np.clip(SR, eps, 1.0)
    WFE_rad = np.sqrt(-np.log(SR)) # [rad]
    
    
    # Radial annulus areas
    r           = separation
    dr          = np.gradient(r) # [mas/px]
    edges       = np.empty(len(r) + 1, dtype=float)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0]    = 0.0
    edges[-1]   = r[-1] + 0.5 * (r[-1] - r[-2]) if len(r) > 1 else r[0]
    area        = np.pi * (edges[1:]**2 - edges[:-1]**2)

    
    
    if PSF_profile_density_speck_2D is not None:
        if PSF_profile_density_DL_2D is None:
            PSF_profile_density_DL_2D = (PSF_profile_density_no_coro_2D - PSF_profile_density_speck_2D) / SR[:, None]

    else:
        # Estimate the reference stellar halo contribution (without coronagraph)
        # I_nc ~= SR * I_DL,nc + I_speck,nc      
        # eta = I_speck,nc / I_nc = (I_nc - SR * I_DL,nc) / I_nc
        # smoothing/regularizing eta
        # I_speck,nc = eta * I_nc
        FWHM          = 1.029*wave*1e-6/D*rad2arcsec * 1000     # [mas]
        sigma         = FWHM / (2*np.sqrt(2*np.log(2)))         # [mas]
        sigma_bins    = np.nanmedian(sigma[:, None] / dr[None, :], axis=1) # [px]
        eta_2D_raw    = ( PSF_profile_density_no_coro_2D - SR[:, None] * PSF_profile_density_DL_2D ) / PSF_profile_density_no_coro_2D # = H_speck_ref / H_ref (H_ref = SR_ref*H_diff_ref + H_speck_ref)
        eta_2D_raw[(eta_2D_raw < 0) | (eta_2D_raw > 1)] = 1
        eta_2D_interp = np.zeros_like(eta_2D_raw)
        eta_2D        = np.zeros_like(eta_2D_raw)
        for iw in range(N_wave):
            valid                = (eta_2D_raw[iw] >= 0) & (eta_2D_raw[iw] <= 1)
            eta_2D_interp[iw] = interp1d(separation[valid], eta_2D_raw[iw][valid], bounds_error=False, fill_value=np.nan)(separation)
            eta_2D_interp[iw] = np.nan_to_num(eta_2D_interp[iw])
            eta_2D[iw]        = gaussian_filter1d(eta_2D_interp[iw], sigma=sigma_bins[iw]) 
        eta_2D[eta_2D <= 0]          = np.nanmin(eta_2D[eta_2D > 0])
        PSF_profile_density_speck_2D = eta_2D * PSF_profile_density_no_coro_2D
        # Sanity check plot
        plt.figure(dpi=300, figsize=(10, 6))
        plt.title("eta = I_speck / I_PSF", fontsize=14)
        plt.xlabel("Separation [mas]", fontsize=12)
        plt.ylabel("eta = I_speck / I_PSF")
        plt.xlim(separation[0], separation[-1])
        plt.ylim(-1, 1)
        plt.axhline(0, c="k", lw=3)
        plt.plot(separation, eta_2D_raw[0],  c="steelblue", ls="--")
        plt.plot(separation, eta_2D[0],      c="steelblue", ls="-", label=f"wave = {wave[0]:.1f} µm")
        plt.plot(separation, eta_2D_raw[-1], c="crimson",   ls="--")
        plt.plot(separation, eta_2D[-1],     c="crimson",   ls="-", label=f"wave = {wave[-1]:.1f} µm")
        plt.legend()
        plt.grid(True)
        plt.show()
    

    if w_coronagraph:
        
        # Estimate the reference stellar halo contribution in coronagraphic profile: 
        # Assuming perfect coronagraph at r_WFE = 1.0: I_c ~= I_speck,c
        PSF_profile_density_coro_speck_2D = np.copy(PSF_profile_density_coro_2D)
        
        # Reference non-coronagraphic encircled energy
        E_inside_r_2D_ref   = np.cumsum(PSF_profile_density_no_coro_2D * area[None, :], axis=1)                          # (wave, separation), Encircled energy inside each separation = total flux inside 'r' / total flux (since PSF_profile_density_no_coro_2D is normalized as such)
        E_inside_FOV_1D_ref = E_inside_r_2D_ref[:, -1]                                                                   # (wave),             Encircled energy inside the FoV         = total flux inside FoV / total flux (since PSF_profile_density_no_coro_2D is normalized as such)
        E_inside_IWA_1D_ref = interp_radial_per_wave(r, E_inside_r_2D_ref, IWA_ref, left=0.0, right=E_inside_FOV_1D_ref) # (wave),             Encircled energy inside the IWA         = total flux inside IWA / total flux (since PSF_profile_density_no_coro_2D is normalized as such)
        
        # Reference stellar leakage outside the reference IWA FPM
        leak_ref = E_inside_FOV_1D_ref - E_inside_IWA_1D_ref # (wave), = total flux inside FoV - total flux inside IWA / total flux
        
        # Reference on-axis anchor of the transmission curve
        radial_transmission_on_axis_1D_ref = safe01(radial_transmission_2D[:, 0]) # (wave)
    
        
    # Allocate outputs
    PSF_profile_density_4D          = np.zeros((N_WFE, N_IWA, N_wave, N_sep), dtype=np.float32) # = mean surface density flux inside bin (without coronagraph) / total flux (without coronagraph)
    fraction_core_4D                = np.zeros((N_WFE, N_IWA, N_wave, N_sep), dtype=np.float32) # = total flux inside core (without coronagraph) / total flux (without coronagraph)
    if w_coronagraph:
        PSF_profile_density_coro_4D = np.zeros((N_WFE, N_IWA, N_wave, N_sep), dtype=np.float32) # = mean surface density flux inside bin (with coronagraph) / total flux (with coronagraph)
        fraction_core_coro_4D       = np.zeros((N_WFE, N_IWA, N_wave, N_sep), dtype=np.float32) # = total flux inside core (with coronagraph) / total flux (with coronagraph)
        radial_transmission_4D      = np.zeros((N_WFE, N_IWA, N_wave, N_sep), dtype=np.float32) # = total flux (with coronagraph) / total flux (without coronagraph)
    else:
        PSF_profile_density_coro_4D = None
        fraction_core_coro_4D       = None
        radial_transmission_4D      = None
    
    
    # Loop over WFE
    for idx_WFE, r_wfe in enumerate(r_WFE):
        
        # Strehl at the CURRENT WFE
        SR_new   = np.exp(-(r_wfe * WFE_rad)**2) # (wave), at the CURRENT WFE
        scale_fc = SR_new / SR                   # (wave), assuming that I_PSF ~ SR*I_DL at sep ~ 0
        
        # Speckle halo rescaling: halo ~ (1 - SR)
        scale_speck = (1 - SR_new) / np.clip(1 - SR, eps, None)
        
        
        # --- WITHOUT CORONAGRAPH ---
        
        # PSF profile scaling at the CURRENT WFE (w/o coronagraph)
        PSF_profile_density_speck_2D_wfe   = PSF_profile_density_speck_2D * scale_speck[:, None]
        PSF_profile_density_no_coro_2D_wfe = SR_new[:, None] * PSF_profile_density_DL_2D + PSF_profile_density_speck_2D_wfe
        
        # Core fractions scaling at the CURRENT WFE (w/o coronagraph)
        fraction_core_no_coro_2D_wfe = fraction_core_no_coro_2D * scale_fc[:, None]                  # (wave, separation), WFE 'gain' on the fractions core
        fraction_core_no_coro_2D_wfe = np.minimum(fraction_core_no_coro_2D_wfe, fraction_core_DL_2D) # (wave, separation), Clipping with the DL fractions core
        
        
        # --- WITH CORONAGRAPH ---
        if w_coronagraph:
            
            # Speckle halo profile scaling at the CURRENT WFE (w/ coronagraph)
            PSF_profile_density_coro_speck_2D_wfe = PSF_profile_density_coro_speck_2D * scale_speck[:, None]
                        
            # Core fractions scaling at the CURRENT WFE (w/ coronagraph)        
            fraction_core_coro_2D_wfe = fraction_core_coro_2D * scale_fc[:, None]                   # (wave, separation), WFE 'gain' on the fractions core
            fraction_core_coro_2D_wfe = np.minimum(fraction_core_coro_2D_wfe, fraction_core_DL_2D)  # (wave, separation) clipped by the non-coronagraphic DL fc
            
            # Non-coronagraphic encircled energy at the CURRENT WFE
            E_inside_r_2D_wfe   = np.cumsum(PSF_profile_density_no_coro_2D_wfe * area[None, :], axis=1) # (wave, separation), Encircled energy inside each separation = total flux inside 'r' / total flux (since PSF_profile_density_no_coro_2D is normalized as such)
            E_inside_FOV_1D_wfe = E_inside_r_2D_wfe[:, -1]                                              # (wave),             Encircled energy inside the FoV         = total flux inside FoV / total flux (since PSF_profile_density_no_coro_2D is normalized as such)            
        
        
        # Loop over IWA
        for idx_IWA, r_iwa in enumerate(r_IWA):
            
            # --- WITHOUT CORONAGRAPH ---

            # Non-coronagraphic quantities are left unchanged
            
            
            # --- WITH CORONAGRAPH ---
            
            # Stretching to the new IWA (if coronagraph)
            if w_coronagraph:
                
                # New physical mask radius
                IWA_mas_new = r_iwa * IWA_ref
                
                
                # Stretching to the new IWA
                PSF_profile_density_coro_speck_2D_wfe_iwa = warp_sep_IWA(separation, PSF_profile_density_coro_speck_2D_wfe, r_iwa, mode="log",   tail_model="powerlaw")
                fraction_core_coro_2D_wfe_iwa             = warp_sep_IWA(separation, fraction_core_coro_2D_wfe,             r_iwa, mode="logit", tail_model="flat")
                radial_transmission_2D_iwa                = warp_sep_IWA(separation, radial_transmission_2D,                r_iwa, mode="logit", tail_model="flat")
                
                
                # PSF profile scaling at the CURRENT WFE AND IWA (w/ coronagraph)
                
                # Current FoV flux fractions in EACH profile own normalization
                # - flux_nc_rel  : FoV fraction relative to total non-coronagraphic flux
                # - flux_c_rel   : FoV fraction relative to total coronagraphic flux
                flux_nc_rel = np.clip(np.nansum(PSF_profile_density_no_coro_2D_wfe        * area[None, :], axis=1), eps, None)
                flux_c_rel  = np.clip(np.nansum(PSF_profile_density_coro_speck_2D_wfe_iwa * area[None, :], axis=1), eps, None)
                
                # Pure radial shapes
                S_nc = PSF_profile_density_no_coro_2D_wfe        / flux_nc_rel[:, None]
                S_c  = PSF_profile_density_coro_speck_2D_wfe_iwa / flux_c_rel[:, None]
                
                # Stellar leakage outside the CURRENT IWA FPM
                E_inside_IWA_1D_wfe_iwa = interp_radial_per_wave(r, E_inside_r_2D_wfe, IWA_mas_new, left=0.0, right=E_inside_FOV_1D_wfe)
                leak_wfe_iwa            = E_inside_FOV_1D_wfe - E_inside_IWA_1D_wfe_iwa
                
                # Mixing coefficient driven by stellar leakage
                mu = (leak_wfe_iwa - leak_ref) / np.clip(E_inside_FOV_1D_wfe - leak_ref, eps, None)
                mu = np.clip(mu, 0.0, 1.0)
                
                # Build the effective coronagraphic radial profile DIRECTLY in coronagraphic relative normalization
                # This profile describes the redistribution of the coronagraphic flux, while the absolute
                # transmission level is handled separately by radial_transmission_2D_wfe_iwa.
                flux_mix_rel = (1.0 - mu)          * flux_c_rel   + mu          * flux_nc_rel
                S_mix        = (1.0 - mu[:, None]) * S_c          + mu[:, None] * S_nc
                
                PSF_profile_density_coro_2D_wfe_iwa = S_mix * flux_mix_rel[:, None]


                # Post-AO fractions core 'off-axis' should be equal with and without coronagraph
                fraction_core_no_coro_off_axis_1D = fraction_core_no_coro_2D_wfe[:, -1]  # Without coronagraph, this is always 'off-axis' by definition, so this is constant with separation, but for homogenety we also take sep = -1
                fraction_core_coro_off_axis_1D    = fraction_core_coro_2D_wfe_iwa[:, -1] # With coronagraph, the off-axis fraction core is assumed to be given at the largest offset, i.e. sep = -1
                scale_fraction_core_off_axis_1D   = (fraction_core_no_coro_off_axis_1D / np.clip(fraction_core_coro_off_axis_1D, eps, None))
                fraction_core_coro_2D_wfe_iwa    *= scale_fraction_core_off_axis_1D[:, None]
                fraction_core_coro_2D_wfe_iwa     = np.minimum(fraction_core_coro_2D_wfe_iwa, fraction_core_DL_2D)          # (wave) clipped by the non-coronagraphic DL fc
                fraction_core_coro_2D_wfe_iwa     = np.minimum(fraction_core_coro_2D_wfe_iwa, fraction_core_no_coro_2D_wfe) # (wave) clipped by the non-coronagraphic fc

                
                # Anchors 
                # y0     = reference anchor after IWA stretch
                # y1     = off-axis asymptote
                # y0_new = new on-axis transmission including WFE + IWA effect
                y0     = safe01(radial_transmission_2D_iwa[:, 0])   # reference anchor after IWA stretch only
                y1     = safe01(radial_transmission_2D_iwa[:, -1])  # off-axis asymptote
                y0_new = safe01(radial_transmission_on_axis_1D_ref * leak_wfe_iwa / np.clip(leak_ref, eps, None))
                y0_new = np.minimum(y0_new, y1 - eps)
                y0_new = safe01(y0_new)
                
                # Affine transform in logit space:
                #   logit(T_new) = a * logit(T_ref) + b
                # constrained by:
                #   T_new(on-axis) = y0_new
                #   T_new(off-axis) = y1
                L0     = logit(y0)                                  # (wave)
                L1     = logit(y1)                                  # (wave)
                L0new  = logit(y0_new)                              # (wave)
                L1new  = L1                                         # (wave) the radial transmission does not change far from the coronagraph
                den_a  = L1 - L0
                den_a  = np.where(np.abs(den_a) < eps, eps, den_a)
                a      = (L1new - L0new) / den_a
                b      = L0new - a * L0                             # (wave)
                radial_transmission_2D_wfe_iwa = expit(a[:, None] * logit(safe01(radial_transmission_2D_iwa)) + b[:, None])
                radial_transmission_2D_wfe_iwa = np.clip(radial_transmission_2D_wfe_iwa, 0.0, 1.0)
                
                
                # Re-evaluating core fractions and radial tranmsissions at sep=0 to avoid unphysical warped values
                for iw in range(N_wave):        
                    fraction_core_coro_2D_wfe_iwa[iw]  = interp_extrap_sep(separation_ref=separation[separation!=0], separation_new=separation, y_ref=fraction_core_coro_2D_wfe_iwa[iw][separation!=0],  mode="logit", tail_model="flat")
                    radial_transmission_2D_wfe_iwa[iw] = interp_extrap_sep(separation_ref=separation[separation!=0], separation_new=separation, y_ref=radial_transmission_2D_wfe_iwa[iw][separation!=0], mode="logit", tail_model="flat")

            
            # Save current models
            PSF_profile_density_4D[idx_WFE, idx_IWA]          = PSF_profile_density_no_coro_2D_wfe.astype(np.float32)
            fraction_core_4D[idx_WFE, idx_IWA]                = fraction_core_no_coro_2D_wfe.astype(np.float32)
            if w_coronagraph:
                PSF_profile_density_coro_4D[idx_WFE, idx_IWA] = PSF_profile_density_coro_2D_wfe_iwa.astype(np.float32)
                fraction_core_coro_4D[idx_WFE, idx_IWA]       = fraction_core_coro_2D_wfe_iwa.astype(np.float32)
                radial_transmission_4D[idx_WFE, idx_IWA]      = radial_transmission_2D_wfe_iwa.astype(np.float32)
    
    
    # Move wavelength axis first: (N_WFE, N_IWA, N_wave, N_sep) -> (N_wave, N_WFE, N_IWA, N_sep)
    PSF_profile_density_4D          = np.moveaxis(PSF_profile_density_4D,      2, 0)
    fraction_core_4D                = np.moveaxis(fraction_core_4D,            2, 0)
    if w_coronagraph:
        PSF_profile_density_coro_4D = np.moveaxis(PSF_profile_density_coro_4D, 2, 0)
        fraction_core_coro_4D       = np.moveaxis(fraction_core_coro_4D,       2, 0)
        radial_transmission_4D      = np.moveaxis(radial_transmission_4D,      2, 0)
    
    
    return PSF_profile_density_4D, fraction_core_4D, PSF_profile_density_coro_4D, fraction_core_coro_4D, radial_transmission_4D



def plot_PSF(instru, coronagraph, apodizer, strehl, pxscale, sep_unit, wave, PSF_3D, type_PSF, model_PSF, sigfactor, debug):
    NbChannel, NbLine, NbColumn = PSF_3D.shape

    # To verify if the PSF is well centered (if debug)
    if debug:
        y_max, x_max, _, _ = fitting_PSF(instru=instru, data=PSF_3D, wave=wave, pxscale=pxscale, model=model_PSF, sigfactor=sigfactor, debug=debug)
        y_center           = (NbLine   - 1) / 2
        x_center           = (NbColumn - 1) / 2
        dy                 = y_center - y_max
        dx                 = x_center - x_max
        print(f"   The {type_PSF} PSF is uncentered by: dy = {np.nanmean(dy):.3f} px = {np.nanmean(dy)*pxscale:.3f} mas, dx = {np.nanmean(dx):.3f} px = {np.nanmean(dx)*pxscale:.3f} mas")
    
    # PSF post AO rebinned with pxscale plot
    if coronagraph is None:
        if apodizer == "NO_SP":
            title = f"ELT/{instru} {type_PSF} PSF without apodizer in {strehl} strehl \n without coronagraph with {pxscale:.2f} mas/px pxscale"
        else:
            title = f"ELT/{instru} {type_PSF} PSF with {apodizer.replace('_', ' ')} apodizer in {strehl} strehl \n without coronagraph with {pxscale:.2f} mas/px pxscale"
    else:
        title = f"ELT/{instru} {type_PSF} PSF without apodizer in {strehl} strehl \n with Lyot coronagraph with {pxscale:.2f} mas/px pxscale"    
    plt.figure(figsize=(8, 8), dpi=300)
    extent = [-pxscale*NbColumn/2, pxscale*NbColumn/2, -pxscale*NbLine/2,   pxscale*NbLine/2]
    plt.imshow(np.nanmean(PSF_3D, 0) / np.nanmax(np.nanmean(PSF_3D, 0)), cmap="inferno", extent=extent, norm=mcolors.LogNorm(vmin=1e-6, vmax=1))
    plt.plot(0, 0, ms=100, marker="+", c="k")
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.xlabel(f'x offset [{sep_unit}]', fontsize=12)
    plt.ylabel(f'y offset [{sep_unit}]', fontsize=12)
    plt.title(title, fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label('PSF [raw contrast]', fontsize=14, labelpad=20, rotation=270)
    plt.show()
    


def plot_profiles(instru, coronagraph, apodizer, strehl, pxscale, sep_unit, size_core, wave, wave_raw, separation, PSF_profile_density_2D, fraction_core_2D, radial_transmission_2D, type_PSF, title_suffix):
    
    N_wave, N_sep = PSF_profile_density_2D.shape
    
    cmap_wave = plt.get_cmap("Spectral_r", N_wave)
    
    if coronagraph is None:
        nrows  = 2
        sharex = False
        if apodizer == "NO_SP":
            title = f"ELT/{instru} {type_PSF} PSF without apodizer in {strehl} strehl \n without coronagraph"
        else:
            title = f"ELT/{instru} {type_PSF} PSF with {apodizer.replace('_', ' ')} apodizer in {strehl} strehl \n without coronagraph"
    else:
        nrows  = 3
        sharex = True
        title = f"ELT/{instru} {type_PSF} PSF without apodizer in {strehl} strehl \n with Lyot coronagraph"
    pxscale_suffix = "" if pxscale is None else f" with {pxscale} mas/px pxscale"
    title += pxscale_suffix
    title += title_suffix
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 8), dpi=300, sharex=sharex)
    fig.subplots_adjust(hspace=0.4)  # espacement vertical entre subplots
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Subplot 1: PSF Profile
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlim(separation[1], separation[-1])
    axs[0].set_ylim(1e-10, 1e-1)
    axs[0].set_xlabel(f"Separation [{sep_unit}]",       fontsize=10)
    axs[0].set_ylabel("Mean flux fraction density per px", fontsize=10)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].minorticks_on()
    axs[0].tick_params(labelsize=8)
    axs[0].set_title("PSF profile (for the star)", fontsize=10)
    for iw in range(N_wave):
        axs[0].plot(separation, PSF_profile_density_2D[iw], c=cmap_wave(iw))
    
    # Subplot 2: Flux inside the FWHM
    axs[1].set_ylim(1e-1, 100)
    axs[1].set_yscale('log')
    axs[1].set_ylabel("Core PSF flux fraction [%]", fontsize=10)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].minorticks_on()
    axs[1].tick_params(labelsize=8)
    axs[1].set_title("Flux inside the FWHM (for the planet)", fontsize=10)
    if coronagraph is None:
        if wave_raw is not None:
            axs[1].axvspan(wave_raw[0], wave_raw[-1], color="black", alpha=0.1, lw=0, label="Simulated data range")
        axs[1].set_xlim(wave[0], wave[-1])
        axs[1].set_xlabel("Wavelength [µm]", fontsize=10)
        axs[1].plot(wave, 100 * fraction_core_2D[:, -1],                              c="black",   ls="-",  alpha=0.7)
        if pxscale is not None:
            axs[1].plot(wave, 100 * PSF_profile_density_2D[:, 0]*pxscale**2*size_core**2, c="crimson", ls="--", alpha=0.7)
    else:
        # Subplot 2: Flux inside the FWHM
        axs[1].set_xlabel(f"Angular offset separation from the coronagraph [{sep_unit}]", fontsize=10)
        
        # Subplot 3: Transmission
        axs[2].set_ylim(1e-1, 100)
        axs[2].set_yscale('log')
        axs[2].set_ylabel("Transmission [%]", fontsize=10)
        axs[2].set_xlabel(f"Angular offset separation from the coronagraph [{sep_unit}]", fontsize=10)
        axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[2].minorticks_on()
        axs[2].tick_params(labelsize=8)
        axs[2].set_title("Coronagraph throughput", fontsize=10)
        
        for iw in range(N_wave):
            axs[1].plot(separation, 100 * fraction_core_2D[iw],       c=cmap_wave(iw))
            axs[2].plot(separation, 100 * radial_transmission_2D[iw], c=cmap_wave(iw))
            
    # Colorbar commune (λ -> couleur)
    norm = mpl.colors.Normalize(vmin=np.nanmin(wave), vmax=np.nanmax(wave))
    sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_wave)
    sm.set_array([])  # required by some matplotlib versions
    cbar = fig.colorbar(sm, ax=axs, location="right", fraction=0.035, pad=0.05)
    cbar.set_label("Wavelength [µm]", fontsize=10, rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=8)
    plt.show()



############################## Rebinning data: detector-like ################################

def ceil_odd(x):
    """
    Smallest odd integer >= x.
    """
    n = int(np.ceil(x))
    if n % 2 == 0:
        n += 1
    return max(1, n)


def shape_out_same_or_larger_fov(shape_in, pxscale_in, pxscale_out):
    """
    Output odd shape that guarantees FoV_out >= FoV_in.
    """
    ny_in, nx_in = shape_in
    ny_out = ceil_odd(ny_in * pxscale_in / pxscale_out)
    nx_out = ceil_odd(nx_in * pxscale_in / pxscale_out)
    return ny_out, nx_out


def _overlap_fraction_matrix(n_in, dx_in, n_out, dx_out):
    """
    Matrix M such that:
        out_flux_1d = M @ in_flux_1d
    with exact flux-conserving overlaps between centered pixel grids.
    """
    edges_in  = (np.arange(n_in + 1)  - n_in  / 2) * dx_in
    edges_out = (np.arange(n_out + 1) - n_out / 2) * dx_out

    M = np.zeros((n_out, n_in), dtype=float)

    j0 = 0
    for i in range(n_out):
        a, b = edges_out[i], edges_out[i + 1]

        while j0 < n_in and edges_in[j0 + 1] <= a:
            j0 += 1

        j = j0
        while j < n_in and edges_in[j] < b:
            overlap = min(b, edges_in[j + 1]) - max(a, edges_in[j])
            if overlap > 0:
                M[i, j] = overlap / dx_in
            j += 1

    return M


def rebin_flux_conserving(data, pxscale_in, pxscale_out, shape_out=None):
    """
    Flux-conserving 2D (or 3D cube) rebinning by exact pixel-overlap integration.

    Parameters
    ----------
    data : ndarray
        2D array (ny, nx) or 3D cube (nch, ny, nx).
    pxscale_in : float
        Input pixel scale.
    pxscale_out : float
        Output pixel scale.
    shape_out : tuple or None
        Output shape (ny_out, nx_out).
        If None, choose the smallest odd shape such that FoV_out >= FoV_in.

    Returns
    -------
    rebinned : ndarray
        Rebinned array with conserved flux over the overlapping FoV.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim not in (2, 3):
        raise ValueError("data must be 2D or 3D")
    if pxscale_in <= 0 or pxscale_out <= 0:
        raise ValueError("pxscale_in and pxscale_out must be > 0")

    ny_in, nx_in = data.shape[-2], data.shape[-1]

    if shape_out is None:
        ny_out, nx_out = shape_out_same_or_larger_fov(
            shape_in=(ny_in, nx_in),
            pxscale_in=pxscale_in,
            pxscale_out=pxscale_out,
        )
    else:
        ny_out, nx_out = map(int, shape_out)
        if ny_out % 2 == 0 or nx_out % 2 == 0:
            raise ValueError("shape_out should be odd in both dimensions.")

    Fy = _overlap_fraction_matrix(ny_in, pxscale_in, ny_out, pxscale_out)
    Fx = _overlap_fraction_matrix(nx_in, pxscale_in, nx_out, pxscale_out)

    if data.ndim == 2:
        return Fy @ data @ Fx.T
    else:
        return np.einsum("yj,bjk,xk->byx", Fy, data, Fx, optimize=True)



############################## Other functions ################################

def estimate_IWA_from_radial_transmission(separation, radial_transmission_1D):
    """
    Estimate an IWA proxy from a radial transmission curve.

    Definition
    ----------
    IWA is defined here as the smallest separation where the transmission reaches
    50% of its maximum value:
        IWA = min{ rho : radial_transmission_1D(rho) >= 0.5 * max(radial_transmission_1D) }

    Parameters
    ----------
    separation : (Nsep,) array_like
        Separation grid (rho), increasing.
    radial_transmission_1D : (Nsep,) array_like
        Transmission profile at a given wavelength (bounded in (0,1)).

    Returns
    -------
    iwa : float
        IWA proxy in the same separation units as 'separation'.
    """
    separation             = np.asarray(separation,             dtype=float)
    radial_transmission_1D = np.asarray(radial_transmission_1D, dtype=float)

    radial_transmission_max = np.nanmax(radial_transmission_1D)
    if not np.isfinite(radial_transmission_max) or radial_transmission_max <= 0:
        return np.nan

    idx = np.where(radial_transmission_1D >= 0.5 * radial_transmission_max)[0]
    IWA = separation[idx[0]] if len(idx) else np.nan
    return IWA



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



def add_if_necessary(array, value):
    if value not in array:
        array = np.append(array, value)
    array = np.unique(array)
    return array





def plot_bkg_skycalc(filename):
    
    # Lecture du fond de ciel
    data                    = fits.getdata(filename)
    wave                    = data["lam"] * 1e-3 # µm
    scattered_moonlight     = data["flux_sml"]
    scattered_starlight     = data["flux_ssl"]
    zodiacal_light          = data["flux_zl"]
    instru_thermal_emission = data["flux_tie"]
    mol_emi_low_atmo        = data["flux_tme"]
    airglow_emi_lines       = data["flux_ael"] + data["flux_arc"]
    background              = data["flux"]  # ph/s/m2/µm/arcsec²

    # --- PLOT 1 : Composantes du fond ---
    alpha = 0.7
    plt.figure(dpi=300, figsize=(10, 6))
    plt.title("Background emission model for the ELT", fontsize=16, fontweight="bold")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Wavelength [µm]", fontsize=14)
    plt.ylabel("Flux [ph/s/m2/µm/arcsec2]", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    
    plt.plot(wave, scattered_moonlight,     alpha=alpha, label="Scattered Moonlight")
    plt.plot(wave, scattered_starlight,     alpha=alpha, label="Scattered Starlight")
    plt.plot(wave, zodiacal_light,          alpha=alpha, label="Zodiacal Light")
    plt.plot(wave, instru_thermal_emission, alpha=alpha, label="Instrument Thermal Emission")
    plt.plot(wave, mol_emi_low_atmo,        alpha=alpha, label="Molecular Emission (low atmo.)")
    plt.plot(wave, airglow_emi_lines,       alpha=alpha, label="Airglow Emission Lines")
    plt.plot(wave, background,              alpha=alpha, label="Total")
    
    plt.xlim(wave[0], wave[-1])
    plt.ylim(1e-5, 1e10)
    plt.legend(loc='upper left', fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()
    
    
    
    hdr = fits.getheader(filename)
    
    # Extract SkyCalc parameters from COMMENT cards
    params = {}
    for line in hdr["COMMENT"]:
        line = str(line).strip()
        if "=" in line and line.startswith(("SKYMODEL.", "TEL.")):
            key, value = line.split("=", 1)
            params[key.strip()] = value.strip()
    
    print()
    print("-------------------------------------------------------------")
    print("Main parameters used for the background calculation (SkyCalc)")
    print("-------------------------------------------------------------")
    print(f"Airmass                          = {params.get('SKYMODEL.TARGET.AIRMASS', 'N/A')}")
    print(f"Target altitude                  = {params.get('SKYMODEL.TARGET.ALT', 'N/A')} deg")
    print(f"PWV                              = {params.get('SKYMODEL.PWV', 'N/A')} mm")
    print(f"Observatory altitude             = {params.get('TEL.SITE.HEIGHT', 'N/A')} m")
    print(f"Solar flux                       = {params.get('SKYMODEL.MSOLFLUX', 'N/A')} sfu")
    print(f"Wavelength range                 = {float(params['SKYMODEL.WAVELENGTH.MIN'])/1000:.1f} - {float(params['SKYMODEL.WAVELENGTH.MAX'])/1000:.1f} µm")
    print(f"Spectral resolution              = {params.get('SKYMODEL.WAVELENGTH.RESOLUTION', 'N/A')}")
    print(f"Thermal component 1 (Telescope)  = {params.get('SKYMODEL.THERMAL.T1', 'N/A')} K, emissivity = {params.get('SKYMODEL.THERMAL.E1', 'N/A')}")
    print(f"Thermal component 2 (Instrument) = {params.get('SKYMODEL.THERMAL.T2', 'N/A')} K, emissivity = {params.get('SKYMODEL.THERMAL.E2', 'N/A')}")
    print(f"Thermal component 3 (Cryostat)   = {params.get('SKYMODEL.THERMAL.T3', 'N/A')} K, emissivity = {params.get('SKYMODEL.THERMAL.E3', 'N/A')}")
    


def plot_trans_tell_tel(trans_tell, trans_tel):
    alpha = 0.7
    wave  = trans_tell.wavelength
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(wave, trans_tell.flux,                c="crimson",   alpha=alpha, label="Tellurics")
    plt.plot(wave, trans_tel.flux,                 c="black",     alpha=alpha, label="Telescope")
    plt.plot(wave, trans_tell.flux*trans_tel.flux, c="steelblue", alpha=alpha, label="Tellurics x Telescope")
    plt.xlim(wave[0], wave[-1])
    plt.ylim(0, 1)
    plt.title('Tellurics and telescope transmissions', fontsize=16, fontweight="bold")
    plt.xlabel("Wavelength [µm]", fontsize=14)
    plt.ylabel("Transmission", fontsize=14)
    plt.legend(loc='upper left', fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    









