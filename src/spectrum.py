# import FastYield modules
from src.config import h, c, R0_min, R0_max, bands, instrus, spectra_path, vega_path, LMIN, LMAX
from src.get_specs import _load_tell_trans, get_config_data, get_band_lims, get_R_instru
from src.utils import smoothstep, fill_nan_linear, get_bracket_values, linear_interpolate
from src.rotbroad import rotBroad, fastRotBroad
from src.prints_helpers import print_warning

# import astropy modules
from astropy import constants as const
from astropy import units as u
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt
from matplotlib import gridspec

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# import other modules
import os
from tqdm import tqdm
from numba import njit
from collections import OrderedDict
from functools import lru_cache

# For fits warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", message="Header block contains null bytes*")




#######################################################################################################################
###################################################### Utils: #########################################################
#######################################################################################################################

@lru_cache(maxsize=128)
def get_wavelength_axis_constant_R(lmin, lmax, R):
    """
    Build a wavelength grid with *constant resolving power* R, assuming Nyquist sampling.

    Parameters
    ----------
    lmin, lmax: float
        Bandwidth limits.
    R: float
        Spectral resolution

    Returns
    -------
    wavelength: array
        Wavelength axis.
    """
    
    # For a spectrograph with resolving power R, one resolution element is:
    #   Δλ_res = λ / R
    # Nyquist sampling requires 2 samples per resolution element, hence a pixel step:
    #   Δλ_pix ≈ (λ / R) / 2  =>  Δλ_pix / λ ≈ 1 / (2R)
    # This means the natural uniform grid is in ln(λ), with constant step: dln(λ)/dλ = 1/λ => dln(λ) = dλ/λ = 1 / (2R)
    #   Δln(λ) = ln(λ_{i+1}) - ln(λ_i) ≈ Δλ_pix / λ ≈ 1/(2R) = constant
    dln = 1.0 / (2.0 * R)
    
    # Number of samples needed to cover [lmin_model, lmax_model] with that constant Δln(λ)
    # ln(λ_i) = ln(λ_min) + k*Δln(λ) => λ_i = λ_min * exp(k*Δln(λ))
    # If we have n points, it means that we have n - 1 between l_min and l_max
    # because: ln(l_max/l_min) = (n-1) * dln  =>  n ≈ ln(l_max/l_min)/dln + 1
    n = int(np.floor(np.log(lmax / lmin) / dln)) + 1
    
    # Log-uniform wavelength grid: λ_i = λ_min * exp(i * Δln(λ))
    # This keeps λ/Δλ (i.e., R) approximately constant across the whole band.
    wavelength = lmin * np.exp(np.arange(n) * dln) # [µm]
    
    return wavelength


@lru_cache(maxsize=128)
def get_wavelength_axis_constant_dl(lmin, lmax, R):
    """
    Build a wavelength grid with *constant wavelength step* dl, assuming Nyquist sampling at (lmin+lmax)/2.

    Parameters
    ----------
    lmin, lmax: float
        Bandwidth limits.
    R: float
        Spectral resolution

    Returns
    -------
    wavelength: array
        Wavelength axis.
    """
    
    dl         = (lmin + lmax)/2 / (2*R)   # Delta lambda [µm/bin] (factor 2 for Nyquist sampling assumption)
    wavelength = np.arange(lmin, lmax, dl) # Constant and linear wavelength array on the considered band 
    
    return wavelength # [µm]



def get_resolution(wavelength, func):
    """
    Estimate the spectral resolution of a spectrum, assuming Nyquist sampling.

    Parameters
    ----------
    wavelength: array_like
        Array of wavelength values [µm].
    func: function
        Method used to return the resolution R (e.g., np.nanmean, np.nanmax, np.array, etc.)

    Returns
    -------
    R
        Estimated resolving power R (dimensionless).
    """
    dl = np.gradient(wavelength)   # Wavelength spacing Δλ
    R  = func(wavelength / (2*dl)) # Resolving power, factor 2 for Nyquist sampling assumption
    return R



def _centers_to_edges(x):
    """
    Convert a one-dimensional array of bin centers into bin edges.

    For an input array of length N, this function returns an array of length
    N + 1 containing the corresponding bin edges. Interior edges are defined
    as the midpoints between consecutive centers, while the first and last
    edges are obtained by symmetric extrapolation from the first and last
    center spacings.

    Parameters
    ----------
    x : array_like
        One-dimensional array of bin centers. The values are assumed to be
        ordered monotonically, typically in increasing order.

    Returns
    -------
    edges : ndarray
        One-dimensional array of bin edges with length 'len(x) + 1'.

    Notes
    -----
    - For 'len(x) >= 2', the interior edges are computed as:
          edges[i] = 0.5 * (x[i-1] + x[i])
      and the outer edges are extrapolated as:
          edges[0]  = x[0]  - 0.5 * (x[1] - x[0])
          edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    - For 'len(x) == 1', an arbitrary symmetric width of 1.0 is assumed,
      so the returned edges are '[x[0] - 0.5, x[0] + 0.5]'.
    - This function does not check whether 'x' is sorted or uniformly spaced.
    """
    if x.size < 2:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
    edges       = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[1:] + x[:-1])
    edges[0]    = x[0]  - 0.5 * (x[1] - x[0])
    edges[-1]   = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges



# “mean-in-bin” (fast but not flux conservative)
def rebin_spectrum_mean(specHR, sigmaHR, weightHR, lamHR, lamLR):
    """
    Rebin a high-resolution spectrum onto a lower-resolution wavelength grid
    by computing the mean value within each target bin.

    This function assigns each high-resolution sample to a low-resolution bin
    defined by the centers 'lamLR', then computes:
      - the mean rebinned flux,
      - the propagated 1-sigma uncertainty assuming independent errors,
      - the mean rebinned weight/quality array.

    The rebinned uncertainty in each bin is computed as:

        sigmaLR = sqrt(sum(sigma_j^2)) / N

    where the sum runs over the valid high-resolution samples falling into the
    bin, and 'N' is the number of contributing samples.

    Parameters
    ----------
    specHR : array_like
        High-resolution spectrum values sampled at 'lamHR'.
    sigmaHR : array_like or None
        One-sigma uncertainties associated with 'specHR'. If None, no
        uncertainty propagation is performed and 'sigmaLR' is returned as None.
    weightHR : array_like or None
        Optional high-resolution weight or quality array sampled at 'lamHR'.
        If provided, it is rebinned by simple arithmetic mean within each
        low-resolution bin.
    lamHR : array_like
        High-resolution wavelength grid. It must be one-dimensional and
        strictly increasing.
    lamLR : array_like
        Low-resolution wavelength-bin centers. It must be one-dimensional and
        strictly increasing.
        
    Returns
    -------
    specLR : ndarray
        Rebinned low-resolution spectrum, computed as the mean of the valid
        'specHR' samples falling into each low-resolution bin.
    sigmaLR : ndarray or None
        Rebinned 1-sigma uncertainties. Returned as None if 'sigmaHR' is None.
    weightLR : ndarray or None
        Rebinned weight/quality array, computed as the mean of the valid
        'weightHR' samples falling into each low-resolution bin. Returned as
        None if 'weightHR' is None.

    Notes
    -----
    - This function is not flux-conservative: it computes bin-wise means, not
      integrated fluxes.
    - Only finite values are used in the rebinning.
    - The function assumes monotonic wavelength axes, but does not enforce this
      explicitly.
    - Samples falling outside the low-resolution bin range are ignored.
    - In the current implementation, 'weightHR' is averaged only where both
      'specHR' and 'weightHR' are finite.
    - Empty output bins are filled with NaN.
    """
    
    # Compute edges of the low-resolution bins
    edgesLR = _centers_to_edges(x=lamLR)
    
    # --- NaN-aware masks
    mF     = np.isfinite(specHR)
    if sigmaHR is not None:
        mS = mF & np.isfinite(sigmaHR)
    if weightHR is not None:
        mW = mF & np.isfinite(weightHR)
    
    # Size
    nLR = lamLR.size
    
    # ---------- flux
    specLR = np.full(nLR, np.nan, dtype=float)
    if np.any(mF):
        idx  = np.searchsorted(edgesLR, lamHR[mF], side="right") - 1
        keep = (idx >= 0) & (idx < nLR)
        idx  = idx[keep]
        val  = specHR[mF][keep]

        s = np.bincount(idx, weights=val, minlength=nLR).astype(float)
        c = np.bincount(idx, minlength=nLR).astype(float)
        np.divide(s, c, out=specLR, where=(c > 0))

    # ---------- sigma
    if sigmaHR is not None:
        sigmaLR = np.full(nLR, np.nan, dtype=float)
        if np.any(mS):
            idx  = np.searchsorted(edgesLR, lamHR[mS], side="right") - 1
            keep = (idx >= 0) & (idx < nLR)
            idx  = idx[keep]
            var  = sigmaHR[mS][keep] ** 2

            var_sum = np.bincount(idx, weights=var, minlength=nLR).astype(float)
            count   = np.bincount(idx, minlength=nLR).astype(float)
            np.divide(np.sqrt(var_sum), count, out=sigmaLR, where=(count > 0))
    else:
        sigmaLR = None

    # ---------- weight
    if weightHR is not None:
        weightLR = np.full(nLR, np.nan, dtype=float)
        if np.any(mW):
            idx  = np.searchsorted(edgesLR, lamHR[mW], side="right") - 1
            keep = (idx >= 0) & (idx < nLR)
            idx  = idx[keep]
            val  = weightHR[mW][keep]

            s = np.bincount(idx, weights=val, minlength=nLR).astype(float)
            c = np.bincount(idx, minlength=nLR).astype(float)
            np.divide(s, c, out=weightLR, where=(c > 0))
    else:
        weightLR = None

    return specLR, sigmaLR, weightLR



# “overlap α_ij” (slower but more accurate, and more general, flux conservative)
def rebin_spectrum_overlap(specHR, sigmaHR, weightHR, lamHR, lamLR):
    """
    Flux-conserving rebinning from an irregular high-resolution grid to a target grid,
    with *measurement* uncertainty propagation via the α² rule.

    Overview
    --------
    Each low-resolution (LR) bin i gathers flux from high-resolution (HR) bins j
    in proportion to their geometric overlap. Let α_ij be the overlap fraction
    of HR bin j contributing to LR bin i. Then
        F_i     = Σ_j α_ij · F_j                          (flux-conserving)
        Var_i   = Σ_j α_ij² · Var_j                       (linear estimator propagation)
    where (F_j, Var_j) are the HR flux and its measurement variance per bin.
    A length-weighted quality/coverage indicator can also be propagated.

    Parameters
    ----------
    specHR: (N_hr,) array_like of float
        HR flux per native bin [e⁻/bin].
    sigmaHR: (N_hr,) array_like of float or None
        HR 1σ measurement uncertainties [e⁻/bin]. If None, LR uncertainties are not returned.
    weightHR: (N_hr,) array_like of float in [0, 1] or None
        HR per-bin quality/coverage. If provided, LR quality is the length-weighted mean
        of contributing HR values in each LR bin.
    lamHR: (N_hr,) array_like of float
        HR wavelength bin *centers* (monotonic, not necessarily uniform) [nm].
    lamLR: (N_lr,) array_like of float
        LR target wavelength bin centers (monotonic) [nm].

    Returns
    -------
    specLR: (N_lr,) ndarray of float
        Rebinned LR flux per bin [e⁻/bin].
    sigmaLR: (N_lr,) ndarray of float or None
        Rebinned LR 1σ measurement uncertainties [e⁻/bin], or None if 'sigmaHR' is None.
    weightLR: (N_lr,) ndarray of float or None
        LR length-weighted quality/coverage in [0, 1], or None if 'weightHR' is None.
    """

    # -- Shape
    nHR = lamHR.size
    nLR = lamLR.size
    
    # -- Build edges from centers:
    edgesLR = _centers_to_edges(x=lamLR)
    edgesHR = _centers_to_edges(x=lamHR)

    # -- Outputs
    specLR        = np.zeros(nLR, dtype=float)
    sigmaLR       = None if sigmaHR  is None else np.full(nLR, np.nan, dtype=float)
    weightLR      = None if weightHR is None else np.full(nLR, np.nan, dtype=float)
    scale_poisson = np.full(nLR, np.nan, dtype=float)

    # -- Sliding pointer over HR bins
    j = 0
    for i in range(nLR):
        A, B = edgesLR[i], edgesLR[i + 1]
        if not np.isfinite(A) or not np.isfinite(B) or B <= A:
            continue
        # Advance j past HR bins that end before A
        while j < nHR and edgesHR[j + 1] <= A:
            j += 1

        Fi = 0.0      # Σ α F
        Vi = 0.0      # Σ α^2 σ^2
        Wi = 0.0      # length-weighted quality
        A1 = 0.0      # Σ α   (for scale_poisson) -- NaN-aware (only finite specHR)
        A2 = 0.0      # Σ α^2 (for scale_poisson) -- NaN-aware
        
        jj = j
        while jj < nHR and edgesHR[jj] < B:
            left  = max(A, edgesHR[jj])
            right = min(B, edgesHR[jj + 1])
            overlap = right - left
            if overlap > 0.0:
                width_hr = edgesHR[jj + 1] - edgesHR[jj]
                if width_hr > 0 and np.isfinite(width_hr):
                    alpha = overlap / width_hr

                    # Flux (α) -- NaN-aware
                    if np.isfinite(specHR[jj]):
                        Fi += alpha * specHR[jj]
                        # Diagnostics for R and R_flat (valid-only)
                        A1  += alpha
                        A2  += alpha**2

                    # Variance (α^2) -- NaN-aware
                    if sigmaHR is not None and np.isfinite(sigmaHR[jj]) and np.isfinite(specHR[jj]):
                        Vi += alpha**2 * sigmaHR[jj]**2

                    # Quality (length-weighted) -- NaN-aware on weight only
                    if weightHR is not None and np.isfinite(weightHR[jj]):
                        Wi += overlap * weightHR[jj]
            jj += 1

        specLR[i] = Fi
        if sigmaHR is not None:
            sigmaLR[i] = np.sqrt(Vi) if Vi > 0.0 else (0.0 if Fi == 0.0 else np.nan)

        if weightHR is not None:
            width_lr    = B - A
            weightLR[i] = (Wi / width_lr) if width_lr > 0.0 else np.nan

        # --- scale_poisson_i diagnostics (photon noise gain scale factor) -- NaN-aware
        scale_poisson[i] = np.sqrt(A2 / A1) if A1 > 0 else np.nan  # see original comment
    
    return specLR, sigmaLR, weightLR, scale_poisson



def interpolate_flux_with_error(wave, flux, sigma, weight, wave_new):
    """
    Vectorized linear interpolation of a spectrum with proper error propagation.

    Parameters
    ----------
    wave: np.ndarray
        Original wavelength axis (must be strictly increasing).
    flux: np.ndarray
        Flux values at each wavelength (e.g., in electrons).
    sigma: np.ndarray
        1-sigma uncertainty on the flux at each wavelength.
    weight: np.ndarray
        weight function at each wavelength.
    wave_new: np.ndarray
        New wavelength axis to interpolate onto.

    Returns
    -------
    flux_new: np.ndarray
        Interpolated flux at wave_new.
    sigma_new: np.ndarray
        Properly propagated uncertainty at wave_new.
    weight_new: np.ndarray
        Properly propagated weight function at wave_new.
    """
    # Mask for valid interpolation range
    valid = (wave_new >= wave[0]) & (wave_new <= wave[-1])

    # Allocate output arrays
    flux_new          = np.full_like(wave_new, np.nan, dtype=float)
    sigma_new         = np.full_like(wave_new, np.nan, dtype=float)
    weight_new        = np.full_like(wave_new, np.nan, dtype=float)
    scale_poisson_new = np.full_like(wave_new, np.nan, dtype=float)

    # For valid points only
    wave_valid = wave_new[valid]

    # Find indices k such that wave[k] <= wave_new[i] < wave[k+1]
    indices = np.searchsorted(wave, wave_valid) - 1
    indices = np.clip(indices, 0, len(wave) - 2)

    # Compute weights
    denom = wave[indices+1]  - wave[indices]
    w0    = (wave[indices+1] - wave_valid)    / denom
    w1    = (wave_valid      - wave[indices]) / denom

    # Interpolate flux and propagate error
    if flux is not None:
        flux_new[valid] = w0 * flux[indices] + w1 * flux[indices+1]
    if sigma is not None:
        sigma_new[valid] = np.sqrt(w0**2 * sigma[indices]**2 + w1**2 * sigma[indices+1]**2)
    if weight is not None:
        weight_new[valid] = w0 * weight[indices] + w1 * weight[indices+1]
    
    # --- scale_poisson_i diagnostics (photon noise gain scale factor ---
    scale_poisson_new[valid] = np.sqrt(w0**2 + w1**2) # Constant-flux approximation and assuming photon noise: scale_poisson_new = sigma_new / sqrt(flux_new) = sqrt( w_0**2 * sigma_0**2 + w_1**2 * sigma_1**2 /  w_0 * flux_0 + w_1 * flux_1 ) = sqrt( w_0**2 * flux_0 + w_1**2 * flux_1 /  w_0 * flux_0 + w_1 * flux_1 ) = sqrt( w_0**2 + w_1**2 ) (w_0 + w_1 = 1 by definition)
    
    return flux_new, sigma_new, weight_new, scale_poisson_new



@njit
def reflect_index(i, n):
    """Reflect index i for 'reflect' mode (like scipy.ndimage)."""
    if n <= 1:
        return 0
    period = 2 * n
    m = i % period
    if m >= n:
        m = period - m - 1
    return m

@njit
def gaussian_filter1d_variable(y, sigma, truncate=4.0):
    """
    Gaussian smoothing with spatially varying sigma and reflect boundary mode.

    Parameters
    ----------
    y: (N,) array
        1D input array.
    sigma: (N,) array
        Per-sample Gaussian sigma in pixel units.
    truncate: float
        Defines kernel size as truncate * sigma[i].

    Returns
    -------
    y_smooth: (N,) array
        Smoothed array, mimicking scipy.ndimage.gaussian_filter1d (if sigma is constant).
    """
    n        = y.shape[0]
    y_smooth = np.empty_like(y)

    for i in range(n):
        s = sigma[i]
        if s <= 0.0 or not np.isfinite(s):
            y_smooth[i] = y[i]
            continue

        r    = int(np.ceil(truncate * s))
        wsum = 0.0
        acc  = 0.0

        for offset in range(-r, r + 1):
            j     = reflect_index(i + offset, n)
            dx    = offset
            w     = np.exp(-0.5 * (dx / s) ** 2)
            wsum += w
            acc  += w * y[j]

        y_smooth[i] = acc / wsum if wsum > 0.0 else y[i]

    return y_smooth



@lru_cache(maxsize=128)
def _fft_filter_response(N, R, Rc, filter_type):
    """
    Build the Fourier-domain response of the low-pass and high-pass filters.

    This function returns the discrete transfer functions associated with the
    spectral filtering operator used on a uniformly sampled spectrum of length
    'N'. The filters are defined in Fourier space and follow the FFT convention
    used by 'numpy.fft'.

    Parameters
    ----------
    N : int
        Number of spectral samples. Must be greater than or equal to 2.
    R : float
        Spectral resolution of the input spectrum. Must be strictly positive.
    Rc : float or None
        Cut-off resolution of the filter.
        - If 'Rc' is None or 0, no filtering is applied:
          the high-pass response is unity and the low-pass response is zero.
        - If 'Rc > 0', the corresponding filter response is constructed
          according to 'filter_type'.
    filter_type : {'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep'}
        Type of low-pass filter to construct.

        - 'gaussian' or 'gaussian_fast':
          discrete truncated Gaussian kernel in direct space, normalized and
          applied as a circular convolution ('wrap'-like behavior).
        - 'gaussian_true':
          exact Gaussian response directly defined in Fourier space.
        - 'step':
          ideal sharp low-pass filter with a hard cut at '|res| = Rc'.
        - 'smoothstep':
          smooth low-pass transition around the cut-off.

    Returns
    -------
    H_HF : ndarray of complex128, shape (N,)
        Fourier-domain response of the high-pass filter.
    H_LF : ndarray of complex128, shape (N,)
        Fourier-domain response of the low-pass filter.

    Notes
    -----
    The frequency grid is defined using 'np.fft.fftfreq(N)' and converted into
    a spectral-resolution-like axis through:

        res = fftfreq(N) * 2 * R

    The returned filters satisfy, by construction:

        H_HF = 1 - H_LF

    so that the high-pass and low-pass operators are complementary.

    For the 'gaussian' and 'gaussian_fast' modes, the low-pass filter is built
    from a finite-width Gaussian kernel in direct space, then transformed into
    Fourier space. This corresponds to a discrete circular convolution and may
    differ slightly from the continuous Gaussian transfer function used in
    'gaussian_true', especially for small 'N' or narrow kernels.

    Raises
    ------
    ValueError
        If 'N < 2', if 'R <= 0', if 'Rc < 0', or if 'filter_type' is invalid.
    """
    
    if N < 2:
        raise ValueError("N must be >= 2.")
    if R <= 0:
        raise ValueError("R must be > 0.")
    if Rc is None or Rc == 0:
        H_HF = np.ones(N, dtype=np.complex128)
        H_LF = np.zeros(N, dtype=np.complex128)
        return H_HF, H_LF
    if Rc < 0:
        raise ValueError("Rc must be >= 0 when filtering is requested.")

    ffreq = np.fft.fftfreq(N) # cycles/sample
    res   = ffreq * 2*R       # "resolution" axis

    if filter_type == "gaussian" or filter_type == "gaussian_fast":
        # Discrete, truncated Gaussian kernel => circular conv (mode='wrap')
        sigma    = 2*R / (np.pi * Rc) * np.sqrt(np.log(2)/2.)  # samples
        truncate = 4.
        radius   = int(truncate * max(sigma, 1e-12) + 0.5)
        radius   = min(radius, max(0, (N//2) - 1))

        if radius == 0:
            H_LF = np.ones(N, dtype=np.complex128)
        else:
            x       = np.arange(-radius, radius+1, dtype=float)
            kernel  = np.exp(-0.5*(x/sigma)**2)
            kernel /= kernel.sum()

            # Pack kernel for circular conv: lags >= 0 at start, lags < 0 at end
            ker_pad = np.zeros(N, dtype=float)
            ker_pad[:radius+1] = kernel[radius:]   # 0..+radius
            ker_pad[-radius:]  = kernel[:radius]   # -radius..-1

            H_LF = np.fft.fft(ker_pad).astype(np.complex128)

    elif filter_type == "gaussian_true":
        sigma = 2*R / (np.pi * Rc) * np.sqrt(np.log(2)/2)
        H_LF  = np.exp(-2*np.pi**2*sigma**2*(res/(2*R))**2).astype(np.complex128)

    elif filter_type == "step":
        H_LF                   = np.ones_like(res, dtype=float)
        H_LF[np.abs(res) > Rc] = 0.
        H_LF                   = H_LF.astype(np.complex128)

    elif filter_type == "smoothstep":
        H_LF = smoothstep(res, Rc).astype(np.complex128)

    else:
        raise ValueError("Invalid filter_type. Use 'gaussian_fast','gaussian_true','step','smoothstep'.")
    
    H_HF = (1. - H_LF).astype(np.complex128)
    
    return H_HF, H_LF



def filtered_flux(flux, R, Rc, filter_type="gaussian", show=False):
    """
    Split an input 1D flux into high-pass and low-pass components using a cut-off
    spectral resolution 'Rc'.

    The mapping between FFT frequency 'f' and "resolution" follows:
        resolution ~ 2 * R * f
    so that applying a frequency-domain mask at |resolution| > Rc behaves like a
    high/low-pass in "lines-per-resolution-element" space.

    Parameters
    ----------
    flux: ndarray, shape (N,)
        Input flux samples on an evenly spaced wavelength grid (in practice,
        any uniform sampling along the spectral axis).
    R: float
        Effective spectral resolution of the input array (assuming Nyquist sampling).
    Rc: float or None
        Cut-off resolution for the filter. If None, no filtering is applied and
        the low-pass component is identically zero.
    filter_type: {'gaussian', 'gaussian_variable', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep', 'savitzky_golay'}, optional
        Filtering method:
        - 'gaussian'         : real-space Gaussian blur with sigma derived from Rc (Appendix A, Martos+2024).
        - 'gaussian_variable': Like 'gaussian' but with variable sigma (Appendix A, Martos+2024).
        - 'gaussian_fast'    : Like 'gaussian' but without using gaussian_filter1d(): Faster but less accurate.
        - 'gaussian_true'    : Analytic Gaussian convolution, as mathematically defined.
        - 'step'             : ideal top-hat in Fourier space (sharp cutoff at Rc).
        - 'smoothstep'       : smooth window in Fourier space around Rc.
        - 'savitzky_golay'   : polynomial smoothing with window matched to Rc.
    show: bool, optional
        If True, plot original, low-pass, and high-pass components.

    Returns
    -------
    flux_HF: ndarray, shape (N,)
        High-pass component (original - low-pass).
    flux_LF: ndarray, shape (N,)
        Low-pass component.

    Raises
    ------
    ValueError
        If inputs are inconsistent (non-finite flux, non-positive R, or invalid Rc).
    KeyError
        If 'filter_type' is unsupported.
    """
    # No filter applied
    if Rc is None or Rc == 0:
        return np.copy(flux), np.zeros_like(flux)
    
    # Sanity check
    Rc_arr = np.asarray(Rc)
    if filter_type == "gaussian_variable":
        if Rc_arr.ndim != 1 or Rc_arr.shape != flux.shape:
            raise ValueError("For 'gaussian_variable', Rc must be a 1D array with same shape as flux.")
    else:
        if Rc_arr.ndim != 0:
            raise ValueError("For non-'gaussian_variable', Rc must be scalar.")

    valid        = np.isfinite(flux)
    flux_filled  = fill_nan_linear(x=None, y=flux) # NaN gaps are filled with linear interpolation (without extrapolation)
    valid_filled = np.isfinite(flux_filled)        # NaN can remain on edges
    flux_valid   = flux_filled[valid_filled]
    
    # Real-space Gaussian sigma that cut at Rc in resolution space: sigma derived from Martos+2025 Appendix A
    if filter_type == "gaussian":
        sigma         = 2*R / (np.pi * Rc) * np.sqrt(np.log(2) / 2) # see Appendix A of Martos et al. (2025)
        flux_valid_LF = gaussian_filter1d(flux_valid, sigma=sigma)
    
    # Same as 'gaussian' but with varying Rc (sigma)
    elif filter_type == "gaussian_variable":
        sigma         = 2*R / (np.pi * Rc) * np.sqrt(np.log(2) / 2) # see Appendix A of Martos et al. (2025)
        sigma_valid   = np.nan_to_num(sigma[valid_filled])
        flux_valid_LF = gaussian_filter1d_variable(flux_valid, sigma=sigma_valid)
    
    # Savitzky-Golay filter
    elif filter_type == "savitzky_golay":
        # Map Rc to a window length (odd integer >= 5)
        sigma = 2*R / (np.pi * Rc) * np.sqrt(np.log(2) / 2)         # see Appendix A of Martos et al. (2025)
        N     = max(int(round(4*sigma)) | 1, 5)                     #  "| 1" :ensure odd and >=5 (gaussian approximation (sigma = N/4 is an empirical approximation) )
        N     = min(N, len(flux_valid) - (1 - len(flux_valid) % 2)) # keep <= size and odd (set window length based on Rc)
        # polyorder=3 safe default (must be < window_length)
        poly          = min(3, N - 2)
        flux_valid_LF = savgol_filter(flux_valid, window_length=N, polyorder=poly)
    
    # Cached frequency responses
    elif filter_type in {"gaussian_fast", "gaussian_true", "step", "smoothstep"}:
        fft           = np.fft.fft(np.asarray(flux_valid, dtype=np.complex128))
        _, H_LF       = _fft_filter_response(N=len(flux_valid), R=R, Rc=Rc, filter_type=filter_type)
        flux_valid_LF = np.real(np.fft.ifft(fft * H_LF))

    else:
        raise KeyError("Invalid 'filter_type'. Use one of: 'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep', 'savitzky_golay'." )

    # Reinsert into an array with original NaNs preserved
    flux_LF               = np.zeros_like(flux) + np.nan
    flux_LF[valid_filled] = flux_valid_LF
    flux_LF[~valid]       = np.nan                     # Retrieving original NaN 
    flux_HF               = flux - flux_LF

    if show:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(flux,    'crimson',   label="Original")
        plt.plot(flux_LF, 'seagreen',  label="Low-Pass")
        plt.plot(flux_HF, 'royalblue', label="High-Pass")
        plt.xlabel("Wavelength axis", fontsize=14)
        plt.ylabel("Flux axis", fontsize=14)
        plt.xlim(0, len(flux))
        plt.legend(fontsize=14, loc="upper right", frameon=True, edgecolor="gray", facecolor="whitesmoke")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()        
        plt.show()
        
    return flux_HF, flux_LF



def get_counts_from_density(wave, density):
    """
    Integrate a spectral flux density into a band-integrated photon rate.

    This helper converts a spectrum expressed as spectral flux density in
    F_lambda units (energy per unit wavelength) into a photon rate integrated
    over the wavelength grid. It assumes a *top-hat bandpass* over the provided
    'wave' array (i.e., no filter transmission curve is applied).

    The conversion uses:
        N_ph(λ) = F_lambda(λ) * (λ / (h c))
    where λ is in meters. The band-integrated photon rate is approximated with a
    Riemann sum using 'dwave = np.gradient(wave)':

        counts = Σ [ F_lambda(λ_i) * (λ_i / (h c)) * Δλ_i ]

    Parameters
    ----------
    wave : array_like
        Wavelength grid in microns [µm]. Must be 1D, and the ordering should be
        monotonic for 'np.gradient' to represent meaningful bin widths.
    density : array_like
        Spectral flux density sampled on 'wave', in [J/s/m^2/µm] (equivalently
        [W/m^2/µm]).

    Returns
    -------
    counts : float
        Band-integrated photon rate per unit collecting area, in [ph/s/m^2].

    Notes
    -----
    - This integration ignores any instrumental/filter throughput S(λ). If you
      need standards-compliant synthetic photometry, include S(λ) in the integrand.
    - NaNs are ignored via 'np.nansum'. Large NaN regions can bias the result.
    - The accuracy depends on the wavelength sampling and the use of
      'np.gradient(wave)' as an estimate of bin widths.

    Raises
    ------
    ValueError
        If the integrated photon rate is not finite or is <= 0.
    """
    dwave  = np.gradient(wave)
    counts = np.nansum(density * wave*1e-6/(h*c) * dwave) # [J/s/m2/µm]*wave*1e-6/(h*c)  => [ph/s/m2/µm]*dwave => sum([ph/s/m2/bin]) => [ph/s/m2]
    
    if not np.isfinite(counts) or counts <= 0.0:
        raise ValueError("Integrated photon fluxes must be finite and strictly positive.")
    
    return counts



def get_scale_to_mag(wave, density_obs, density_vega, mag, counts_vega=None):
    """
    Compute the multiplicative scale factor required to match a target Vega magnitude.

    This function assumes the input spectra are *spectral flux densities* in
    F_lambda units (energy per unit wavelength), i.e. [J s^-1 m^-2 µm^-1].
    It converts both the observed spectrum and the Vega reference spectrum into
    *band-integrated photon rates* (top-hat bandpass; no transmission curve),
    then returns the scale factor 's' such that:

        s * counts_obs = counts_vega * 10^(-0.4 * mag)

    where 'counts_obs' and 'counts_vega' are photon rates integrated over the
    provided wavelength grid.

    Parameters
    ----------
    wave : array_like
        Wavelength grid in microns [µm]. Must be 1D and monotonic.
    density_obs : array_like
        Observed spectral flux density on 'wave', in [J/s/m^2/µm].
    density_vega : array_like
        Vega spectral flux density on 'wave', in [J/s/m^2/µm].
        It must be sampled on the *same* wavelength grid as 'density_obs'.
    mag : float
        Target Vega magnitude to enforce in the (implicit) top-hat band defined
        by 'wave'.

    Returns
    -------
    scale : float
        Multiplicative factor to apply to 'density_obs' so that its Vega magnitude
        over the band equals 'mag'.

    Raises
    ------
    ValueError
        If the integrated observed or Vega photon rates are not positive or finite.
    """
    counts_obs = get_counts_from_density(wave=wave, density=density_obs)  # [ph/s/m2]
    if counts_vega is None:
        counts_vega = get_counts_from_density(wave=wave, density=density_vega) # [ph/s/m2]    
    return counts_vega*10**(-0.4*mag) / counts_obs # Ratio by which to adjust the spectrum flux in order to retrieve the input magnitude
    


def get_mag(wave, density_obs, density_vega, counts_vega=None):
    """
    Compute a Vega-like apparent magnitude from two spectra expressed as
    spectral *energy density* F_lambda in [J/s/m2/µm], by converting
    them to *photon flux* integrated over the provided wavelength grid.

    The conversion uses:
        dN/dt/dA/dλ = F_lambda(λ) * λ / (h c)
    and the band-integrated photon flux is approximated by a discrete sum:
        N = Σ_i F_lambda(λ_i) * λ_i / (h c) * Δλ_i

    Parameters
    ----------
    wave : array_like
        Wavelength grid in microns [µm]. Must be 1D and (ideally) strictly increasing.
    density_obs : array_like
        Observed spectral energy density F_lambda(λ) on 'wave'
        in [J/s/m2/µm].
    density_vega : array_like
        Reference spectral energy density F_lambda,ref(λ) on 'wave'
        in [J/s/m2/µm] (e.g., Vega spectrum over the same band).
    Returns
    -------
    mag : float
        Apparent magnitude defined as:
            mag = -2.5 * log10(N_obs / N_ref)
        where N_obs and N_ref are the band-integrated photon fluxes
        in [ph/s/m2].
    """
    counts_obs = get_counts_from_density(wave=wave, density=density_obs) # [ph/s/m2]
    if counts_vega is None:
        counts_vega = get_counts_from_density(wave=wave, density=density_vega) # [ph/s/m2]
    return -2.5 * np.log10(counts_obs / counts_vega)



def get_mag_from_flux(flux, units, band0, counts_vega=None):
    """
    Compute a Vega-based magnitude from a *single scalar* flux density in a band,
    using a top-hat bandpass and a photon-counting definition consistent with
    'get_counts_from_density'.

    Because only a scalar is provided, the object SED inside the band must be
    approximated:
      - For wavelength-density units (F_lambda): assumes F_lambda is constant
        across the band.
      - For frequency-density units (F_nu, e.g. Jy): assumes F_nu is constant
        across the band.

    The magnitude is then defined from band-integrated photon rates:
        m = -2.5 log10( N_obj / N_vega )

    where N_vega is computed by integrating the Vega spectrum over the same band.

    Parameters
    ----------
    flux : float
        Scalar flux density value (must be finite and strictly positive).
        Interpreted as:
          - constant F_lambda over the band if units are wavelength-density,
          - constant F_nu over the band if units are frequency-density.
    units : str
        Flux-density units. Supported:
          - Wavelength-density (F_lambda): "J/s/m2/um", "W/m2/um", "J/s/m2/µm", "W/m2/µm",
            "erg/s/cm2/A"
          - Frequency-density (F_nu): "Jy", "mJy", "muJy"
    band0 : str
        Photometric band identifier (e.g., "J", "H", "K").
        The band limits are obtained from 'get_band_lims(band0)'.
    counts_vega : float, optional
        Precomputed Vega photon rate integrated over the same top-hat band
        [ph/s/m^2]. If None, it is computed from 'load_vega_spectrum()'.

    Returns
    -------
    mag : float
        Vega-based magnitude under the top-hat assumption and the chosen flat-SED
        approximation.

    Notes
    -----
    - This is NOT standards-compliant synthetic photometry (no real filter
      transmission curve S(lambda)).
    - With only a scalar, the result depends on the flatness assumption
      (flat in F_lambda vs flat in F_nu). This introduces a color-term error
      if the true SED varies significantly across the band.
    """
    flux = float(flux)
    if not np.isfinite(flux) or flux <= 0.0:
        raise ValueError("'flux' must be finite and strictly positive.")

    # Band limits [µm]
    try:
        lmin_um, lmax_um = get_band_lims(band=band0)
    except KeyError:
        raise KeyError(f"{band0} is not a supported band. Please choose among: {bands}, {instrus}")

    # --- Vega photon counts integrated over the band (top-hat), if needed
    if counts_vega is None:
        vega = load_vega_spectrum()
        vega.crop(lmin_um, lmax_um)
        wave_um       = vega.wavelength
        F_lambda_vega = vega.flux  # [J/s/m2/µm]
        counts_vega   = get_counts_from_density(wave=wave_um, density=F_lambda_vega)

    if not np.isfinite(counts_vega) or counts_vega <= 0.0:
        raise ValueError("'counts_vega' must be finite and strictly positive.")

    # --- Object photon counts from scalar flux density, with flat-SED assumption
    units = units.strip()

    if units in {"J/s/m2/um", "W/m2/um", "J/s/m2/µm", "W/m2/µm"}:
        # flux is F_lambda in [J/s/m^2/µm], assumed constant over band
        # N = ∫ F_lambda * (lambda/(hc)) d(lambda)  (lambda in meters, dλ in µm)
        I = 0.5 * (lmax_um**2 - lmin_um**2)  # ∫ λ_µm dλ_µm over top-hat
        counts_obj = flux * (1e-6 / (h * c)) * I  # [ph/s/m^2]

    elif units == "erg/s/cm2/A":
        # Convert to [J/s/m^2/µm] (inverse of your earlier Vega conversion)
        flux_SI = flux * 10.0
        I = 0.5 * (lmax_um**2 - lmin_um**2)
        counts_obj = flux_SI * (1e-6 / (h * c)) * I

    elif units in {"Jy", "mJy", "muJy"}:
        # flux is F_nu in Jy-ish, assumed constant over band
        if units == "mJy":
            flux_Jy = flux * 1e-3
        elif units == "muJy":
            flux_Jy = flux * 1e-6
        else:
            flux_Jy = flux

        F_nu = flux_Jy * 1e-26  # [J/s/m^2/Hz]
        lmin_m = lmin_um * 1e-6
        lmax_m = lmax_um * 1e-6
        nu_max = c / lmin_m
        nu_min = c / lmax_m
        counts_obj = (F_nu / h) * np.log(nu_max / nu_min)  # ∫ F_nu/(h nu) dnu

    else:
        raise ValueError(f"Unit '{units}' not implemented.")

    if not np.isfinite(counts_obj) or counts_obj <= 0.0:
        raise ValueError("Computed object photon rate is invalid (non-finite or <= 0).")

    return -2.5 * np.log10(counts_obj / counts_vega)



def get_mag_from_mag(T, lg, model, mag_input, band0_input, band0_output):
    
    """
    Convert a magnitude from 'band_in' to 'band_out' for a given object SED by
    *scaling the spectrum* so that its synthetic magnitude matches 'mag_input'
    in 'band_in', then computing the synthetic magnitude in 'band_out'.

    Magnitudes are in the Vega system:
        m_band = -2.5 * log10( ∫ F_obj(λ) λ dλ / ∫ F_vega(λ) λ dλ )

    Parameters
    ----------
    T: float
        Effective temperature for the spectrum loader (K).
    logg: float
        Surface gravity (dex, cgs) for the spectrum loader.
    model: str
        Spectrum model key. If in {"BT-NextGen", "Husser"} the object is treated
        as a star, otherwise as a planet (calls the corresponding loader).
    mag_input: float
        Known magnitude of the object in 'band_in' (Vega system).
    band0_input: str
        Input photometric band name (e.g., "K", "Ks", "W1", "W2").
    band0_output: str
        Output photometric band name.

    Returns
    -------
    float
        The synthetic Vega magnitude in 'band_out'.
    """
    
    lmin_band0_input,  lmax_band0_input  = get_band_lims(band=band0_input)  # [µm]
    lmin_band0_output, lmax_band0_output = get_band_lims(band=band0_output) # [µm]

    lmin = min(lmin_band0_input, lmin_band0_output)
    lmax = max(lmax_band0_input, lmax_band0_output)
    R    = R0_max
    wave = get_wavelength_axis_constant_dl(lmin=lmin, lmax=lmax, R=R)
    
    mask_input  = (wave >= lmin_band0_input)  & (wave <= lmax_band0_input)
    mask_output = (wave >= lmin_band0_output) & (wave <= lmax_band0_output)

    vega = load_vega_spectrum() # [J/s/m2/µm]
    vega = vega.interpolate_wavelength(wave_output=wave, renorm=False).flux
    
    if model in {"BT-NextGen", "Husser"}:
        spectrum = load_star_spectrum(T_star=T, lg_star=lg, model=model) # [J/s/m2/µm]
    else:
        spectrum = load_planet_spectrum(T_planet=T, lg_planet=lg, model=model)  # [J/s/m2/µm]
    flux       = spectrum.interpolate_wavelength(wave_output=wave, renorm=False).flux # [J/s/m2/µm]
    scale      = get_scale_to_mag(wave=wave[mask_input], density_obs=flux[mask_input], density_vega=vega[mask_input], mag=mag_input)
    flux      *= scale
    mag_output = get_mag(wave=wave[mask_output], density_obs=flux[mask_output], density_vega=vega[mask_output])
    
    return mag_output



#######################################################################################################################
############################################# class Spectrum: #########################################################
#######################################################################################################################

class Spectrum:

    def __init__(self, wavelength, flux, R=None, T=None, lg=None, model=None, rv=0, vsini=0, sigma=None):
        """
        Container and utilities for 1D spectra.
    
        Attributes
        ----------
        wavelength: (N,) array_like
            Wavelength samples (usually in µm).
        flux: (N,) array_like
            Flux samples (arbitrary units unless specified).
        R: float or None
            Native resolution/LSF of the model or spectral sampling (dimensionless).
        T: float or None
            Characteristic temperature [K].
        lg: float or None
            Surface gravity [dex(cm/s²)].
        model: str or None
            Model family/name.
        rv: float
            Radial velocity of the spectrum [km/s].
        vsini: float
            Projected rotational velocity [km/s].
        sigma: (N,) array_like
            Error samples (arbitrary units unless specified).
        """
        
        # Computing R if missing
        if R is None:
            R = get_resolution(wavelength=wavelength, func=np.array)
        
        self.wavelength = wavelength # Wavelength axis of the spectrum [µm]
        self.flux       = flux       # Flux of the spectrum
        self.R          = R          # Resolution of the spectrum, if R is not an array, R is asssumed to be constant along the wavelength axis
        self.T          = T          # Temperature of the spectrum [K]
        self.lg         = lg         # Surface gravity of the spectrum [dex(cm/s2)]
        self.model      = model      # Model of the spectrum 
        self.rv         = rv         # Radial velocity [km/s]
        self.vsini      = vsini      # Rotational velocity [km/s]
        self.sigma      = sigma      # Error of the spectrum (same unit as self.flux)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def copy(self):
        """
        Return an independent copy of the spectrum.
        """
        arr = lambda x: None if x is None else np.array(x, copy=True)
        return Spectrum(
                        wavelength = arr(self.wavelength),
                        flux       = arr(self.flux),
                        R          = arr(self.R) if np.ndim(self.R) > 0 else self.R,
                        T          = self.T,
                        lg         = self.lg,
                        model      = self.model,
                        rv         = self.rv,
                        vsini      = self.vsini,
                        sigma      = arr(self.sigma),
                    )
    
    def crop(self, lmin, lmax):
        """
        Crop the spectrum to the interval [lmin, lmax].

        Parameters
        ----------
        lmin, lmax: float
            Wavelength bounds (same unit as 'self.wavelength').
        """
        if lmax <= lmin:
            raise ValueError("'lmax' must be greater than 'lmin'.")
        i0 = np.searchsorted(self.wavelength, lmin, side="left")
        i1 = np.searchsorted(self.wavelength, lmax, side="right")
        self.wavelength = self.wavelength[i0:i1]
        self.flux       = self.flux[i0:i1]
        if np.ndim(self.R) > 0:
            self.R = self.R[i0:i1]
        if self.sigma is not None:
            self.sigma = self.sigma[i0:i1]
        
    def crop_nan(self):
        """
        Remove samples where flux is not finite.
        """
        mask_valid      = np.isfinite(self.flux)
        self.wavelength = self.wavelength[mask_valid]
        self.flux       = self.flux[mask_valid]
        if np.ndim(self.R) > 0:
            self.R = self.R[mask_valid]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def set_flux(self, flux_tot):
        """
        Renormalize flux to a specified total (e.g., photon count, J/s/m2/µm, etc.).

        Notes
        -----
        Assumes 'self.flux' is on the same grid and represents quantities
        that should sum linearly over bins (i.e., not a spectral density
        unless the wavelength spacing is constant).

        Parameters
        ----------
        flux_tot: float
            Target total sum of the flux array.
        """
        denom = np.nansum(self.flux)
        if not np.isfinite(denom) or denom == 0:
            raise ValueError("Cannot renormalize: current flux sum is zero or non-finite.")
        self.flux = (flux_tot * self.flux) / denom
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def density_to_photons(self, config_data):
        """
        Convert a spectral *density* [J/s/m2/µm] into [ph/bin/mn] (total ph over the FoV)
        collected by the telescope.

        Parameters
        ----------
        config_data : dict-like
            Instrument configuration; must contain 'telescope['area']' [m2].

        Returns
        -------
        Spectrum
            New Spectrum instance with flux in [ph/bin/mn] (total ph over the FoV).
        """
        spectrum = self.copy()                      # New Spectrum instance
        area     = config_data["telescope"]["area"] # Effective collecting area [m2], accounting for central hole, secondary mirror, and spider obscuration
        wave     = spectrum.wavelength              # Wavelength axis [µm]
        dl       = np.gradient(wave)                # Wavelength spacing Δλ [µm/bin]
        # [J/s/m2/µm] => [ph/s/m2/µm] using λ[m]/(h*c)
        spectrum.flux *= wave*1e-6 / (h*c)
        # [ph/s/m2/µm] => [ph/bin/mn] using collecting area and bin width and minutes:
        spectrum.flux = spectrum.flux * area * dl * 60
        return spectrum # [ph/bin/mn] (total ph over the FoV)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
    def interpolate_wavelength(self, wave_output, renorm=False, fill_value=np.nan):
        """
        Interpolate the spectrum onto a new wavelength grid without changing its
        intrinsic spectral resolution.
    
        This method resamples the spectrum onto 'wave_output' using linear
        interpolation. The flux is interpolated directly, and the uncertainty array
        ('self.sigma'), when present, is propagated onto the new grid through
        'interpolate_flux_with_error'.
    
        In contrast to a true spectral degradation, this operation does not apply
        any line-spread-function (LSF) convolution. Therefore, the intrinsic
        spectral resolution stored in 'self.R' is preserved and simply propagated
        onto the output grid. Any apparent loss of high-frequency information after
        interpolation is only due to the coarser sampling of the new wavelength
        grid, not to a physical broadening of the spectrum.
    
        Parameters
        ----------
        wave_output : (M,) array_like
            Target wavelength grid onto which the spectrum is interpolated. It must
            be expressed in the same units as 'self.wavelength' and is assumed to
            be sorted in increasing order.
        renorm : bool, optional
            If True, renormalize the interpolated spectrum so that the total sampled
            flux is conserved over the wavelength interval shared by the input and
            output grids. This is only appropriate if the stored flux values are
            interpreted consistently with such a discrete-sum convention.
            Default is False.
        fill_value : float, optional
            Value assigned to the interpolated flux outside the wavelength domain
            covered by the input spectrum. Default is 'np.nan'.
    
        Returns
        -------
        Spectrum
            A new 'Spectrum' object sampled on 'wave_output', containing:
            - the interpolated flux,
            - the propagated uncertainty array (if 'self.sigma' is available),
            - the intrinsic spectral resolution 'R' propagated onto the new grid.
    
        Notes
        -----
        - This method changes the wavelength sampling, but does not physically
          degrade the spectrum.
        - The output attribute 'R' represents the intrinsic spectral resolution of
          the spectrum, not the Nyquist-limited resolution of the output grid.
        - Outside the wavelength range covered by the input spectrum, the output
          flux is set to 'fill_value' and the output resolution is set to 'NaN'.
        - If the input spectrum contains NaNs or gaps, they may propagate locally
          through the interpolation.
        - When 'renorm=True', the renormalization conserves the discrete sum of the
          flux samples over the overlap region, not the exact continuous integral.
        """
        # Getting input spectrum
        wave_input  = np.asarray(self.wavelength, dtype=float)
        flux_input  = np.asarray(self.flux,       dtype=float)
        R_input     = self.R
        sigma_input = None if self.sigma is None else np.asarray(self.sigma, dtype=float)
        
        # Keeping the smallest range containing wave_output
        i0           = np.searchsorted(wave_input, wave_output[0],  side="right") - 1
        i1           = np.searchsorted(wave_input, wave_output[-1], side="left")
        i0           = max(i0, 0)
        i1           = min(i1, len(wave_input) - 1)
        mask_overlap = slice(i0, i1+1)
        flux_input   = flux_input[mask_overlap]
        wave_input   = wave_input[mask_overlap]
        R_input      = R_input[mask_overlap] if np.ndim(R_input) > 0 else R_input # Spectral resolution of the LSF
        sigma_input  = sigma_input[mask_overlap] if sigma_input is not None else None
        
        valid_overlap = (wave_output >= wave_input[0]) & (wave_output <= wave_input[-1])
        
        # Interpolates flux values on the new axis (wave_output) and error values (if needed)
        if type(fill_value) == tuple:
            left  = fill_value[0]
            right = fill_value[1]
        else:
            left  = fill_value
            right = fill_value
        flux_output = np.interp(wave_output, wave_input, flux_input, left=left, right=right)
        if sigma_input is not None:
            sigma_output = interpolate_flux_with_error(wave=wave_input, flux=None, sigma=sigma_input, weight=None, wave_new=wave_output)[1]
        else:
            sigma_output = None
        
        # The intrinsic spectral resolution is not changed by an linear interpolation
        if np.ndim(R_input) > 0:
            R_output = np.interp(wave_output, wave_input, R_input, left=np.nan, right=np.nan)
        else: # if R_input is a float, we assume a constant spectral resolution
            R_output                 = np.full_like(wave_output, R_input, dtype=float)
            R_output[~valid_overlap] = np.nan
        
        # Creating interpolated Spectrum instance
        spectrum_output = Spectrum(wavelength=wave_output, flux=flux_output, R=R_output, T=self.T, lg=self.lg, model=self.model, rv=self.rv, vsini=self.vsini, sigma=sigma_output)
        
        # Conserve the *total* flux in the overlapping domain (if needed)
        if renorm:
            print_warning("WARNING: interpolate_wavelength(renorm=True)")
            ratio                 = np.nansum(self.flux[mask_overlap]) / np.nansum(spectrum_output.flux[valid_overlap])
            spectrum_output.flux *= ratio
            if spectrum_output.sigma is not None:
                spectrum_output.sigma *= ratio
                
        return spectrum_output
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def degrade_resolution(self, wave_output, renorm=False, gaussian_filtering=True, R_output=None, verbose=False, eps=1e-9):
        """
        Degrade the spectrum to a lower spectral resolution and rebin it onto a new
        wavelength grid.
    
        This method models a true spectral-resolution degradation by optionally
        convolving the input spectrum with a Gaussian kernel and then down-binning
        it onto 'wave_output'. The final spectrum is returned on the output grid,
        together with its propagated intrinsic spectral resolution.
    
        The intrinsic spectral resolution of the input spectrum is taken from
        'self.R'. The output sampling resolution is estimated from the Nyquist
        sampling of 'wave_output'. If 'R_output' is not provided, the target output
        resolution is assumed to be equal to the Nyquist-equivalent sampling
        resolution of 'wave_output'.
    
        Only down-binning is supported: the output grid must be fully included in
        the input wavelength range, and its sampling must be coarser than or equal
        to that of the input grid. If the requested output grid is finer than the
        input grid, the function raises an error rather than performing an
        interpolation.
    
        Parameters
        ----------
        wave_output : (M,) array_like
            Output wavelength grid. It must be expressed in the same units as
            'self.wavelength', must be sorted in increasing order, and must be
            fully included within the input wavelength range.
        renorm : bool, optional
            If True, renormalize the degraded spectrum so that the total sampled
            flux is conserved over the rebinned output spectrum. This is only
            appropriate if the stored flux values are interpreted consistently with
            such a discrete-sum convention. Default is False.
        gaussian_filtering : bool, optional
            If True, apply a Gaussian LSF convolution before rebinning in order to
            match the requested target resolution. If False, only the rebinning step
            is applied. Default is True.
        R_output : float or ndarray, optional
            Target intrinsic spectral resolution on 'wave_output'. If a scalar is
            provided, a constant output resolution is assumed. If 'None', the target
            resolution is set to the Nyquist-equivalent sampling resolution of
            'wave_output'. Default is None.
        verbose : bool, optional
            If True, print warning messages when the requested output resolution is
            finer than the output sampling, or when no additional Gaussian
            convolution is required because the requested resolution is already
            comparable to or higher than the input one. Default is False.
    
        Returns
        -------
        Spectrum
            A new 'Spectrum' object sampled on 'wave_output', containing:
            - the degraded and rebinned flux,
            - the rebinned uncertainty array (if 'self.sigma' is available),
            - the propagated intrinsic spectral resolution 'R' after Gaussian
              convolution and binning.
    
        Raises
        ------
        ValueError
            If 'wave_output' is not included in the input wavelength range.
        ValueError
            If the output sampling is finer than the input sampling, i.e. if the
            operation would require interpolation rather than true down-binning.
    
        Notes
        -----
        - This method performs a true spectral degradation, unlike
          'interpolate_wavelength', which only changes the sampling grid.
        - The Gaussian convolution is approximated using a single constant kernel
          width, taken as the median of the locally required kernel widths across
          the output grid. Therefore, when the input or target resolution varies
          strongly with wavelength, the degradation is only approximate.
        - The uncertainty array is rebinned, but the uncertainty propagation
          through the optional Gaussian convolution is not treated rigorously.
        - The returned 'R' corresponds to the intrinsic spectral resolution after
          convolution and binning, not to the Nyquist-limited sampling resolution
          of the output grid.
        - No extra spectral padding is added before convolution, so the first and
          last output points may be slightly affected by edge effects.
        - When 'renorm=True', the renormalization conserves the discrete sum of the
          flux samples, not the exact continuous integral of the spectrum.
        """
        # Getting input spectrum
        wave_input  = np.asarray(self.wavelength, dtype=float)
        flux_input  = np.array(self.flux,         dtype=float, copy=True) # because modified later
        R_input     = self.R
        sigma_input = None if self.sigma is None else np.asarray(self.sigma, dtype=float)
        
        #if wave_output[-1] < wave_input[0] or wave_output[0] > wave_input[-1]:
        if wave_output[0] < wave_input[0] or wave_output[-1] > wave_input[-1]:
            raise ValueError("'wave_output' is not included in 'wave_input'")
            
        # Keeping the smallest range containing wave_output
        i0           = np.searchsorted(wave_input, wave_output[0], side="right") - 1
        i1           = np.searchsorted(wave_input, wave_output[-1], side="left")
        i0           = max(i0, 0)
        i1           = min(i1, len(wave_input) - 1)
        mask_overlap = slice(i0, i1+1)
        flux_input   = flux_input[mask_overlap]
        wave_input   = wave_input[mask_overlap]
        R_input      = R_input[mask_overlap] if np.ndim(R_input) > 0 else R_input
        sigma_input  = sigma_input[mask_overlap] if sigma_input is not None else None
        
        valid_overlap = (wave_output >= wave_input[0]) & (wave_output <= wave_input[-1])
        
        # Sampling (at Nyquist) and spectral resolutions on wave_output
        R_nyquist_input = get_resolution(wavelength=wave_input, func=np.array) # Input sampling resolution (= spectral resolution assuming Nyquist sampling)
        R_nyquist_input = np.interp(wave_output, wave_input, R_nyquist_input, left=np.nan, right=np.nan)
        if np.ndim(R_input) > 0:
            R_input = np.interp(wave_output, wave_input, R_input, left=np.nan, right=np.nan)
        else: # if R_input is a float, we assume a constant spectral resolution
            R_input                 = np.full_like(wave_output, R_input, dtype=float)
            R_input[~valid_overlap] = np.nan
        R_nyquist_output = get_resolution(wavelength=wave_output, func=np.array) # Output sampling resolution (= spectral resolution assuming Nyquist sampling)
        if R_output is None: # If R_output is None, we assumed that the desired output spectral resolution is given by the nyquist sampling resolution of wave_output (i.e. R_nyquist_output)
            R_output = R_nyquist_output
        elif np.ndim(R_output) == 0: # if R_output is a float, we assume a constant spectral resolution
            R_output                 = np.full_like(wave_output, R_output, dtype=float)
            R_output[~valid_overlap] = np.nan

        # sigma_LSF will be the spectral LSF propagated along this function in [input px]
        sigma_LSF = R_nyquist_input / ( np.sqrt(2*np.log(2)) * R_input ) # [input px]
        
        # Finding whether downbinning will be applied
        valid_sampling = np.isfinite(R_nyquist_input) & np.isfinite(R_nyquist_output)
        rel_diff       = (R_nyquist_output[valid_sampling] - R_nyquist_input[valid_sampling]) / R_nyquist_input [valid_sampling]
        if np.any(rel_diff > eps):
            plt.figure(dpi=300, figsize=(10, 6))
            plt.plot(wave_output[valid_sampling], rel_diff, lw=0.8, label="(R_nyquist_output - R_nyquist_input) / R_nyquist_input")
            plt.axhline(eps, c="black")
            plt.axvspan(wave_output[valid_sampling][0], wave_output[valid_sampling][-1], color='black', alpha=0.3, lw=0)
            plt.xlabel("Wavelength [µm]")
            plt.ylabel("relative difference")
            plt.xlim(wave_output[0], wave_output[-1])
            plt.legend()
            plt.show()
            raise ValueError(f"WARNING (self.degrade_resolution): The output sampling resolution ({round(np.nanmedian(R_nyquist_output), -2):.0f}) is greater than the input sampling resolution ({round(np.nanmedian(R_nyquist_input), -2):.0f}). This function only supports down-binning (otherwise NaN would be injected). Provide a coarser 'wave_output' to properly apply self.degrade_resolution.")

        # Warning for non-Nyquist sampled output
        if verbose and np.any(R_output > R_nyquist_output): 
            print()
            print_warning(f"WARNING (self.degrade_resolution): The output spectral resolution ({round(np.nanmedian(R_output), -2):.0f}) is greater than the output sampling resolution ({round(np.nanmedian(R_nyquist_output), -2):.0f}). Provide finer 'wave_output' to satisfy Nyquist (at least).")
                    
        # LSF Gaussian convolution (optional) 
        if gaussian_filtering:
            
            # Gaussian LSF + rebinning: we want the *final* variance to match R_output, so in RMS-variance
            # sigma_target^2 ≈ sigma_in^2 + sigma_k^2 + sigma_bin^2, with sigma_px = (R_nyq/sqrt(2 ln2))*(1/R) and
            # sigma_bin,px ≈ Δλ_bin/(sqrt(12) δλ_in) ≈ R_nyq,in/(sqrt(12) R_nyq,out) (assuming Δλ_bin ≈ δλ_out);
            # hence sigma_k^2 = sigma_target^2 - sigma_in^2 - sigma_bin^2, i.e. the first term minus the binning correction.
            sigma_kernel_2 = (R_nyquist_input / np.sqrt(2*np.log(2)))**2 * ( 1/R_output**2 - 1/R_input**2 ) - (R_nyquist_input/(np.sqrt(12)*R_nyquist_output))**2
            
            # Mean computed sigma kernel in [input px] and sanity check value (forbidding negative values)
            sigma_kernel_2 = np.clip(a=sigma_kernel_2, a_min=0, a_max=None)
            sigma_kernel   = np.nanmedian(np.sqrt(sigma_kernel_2))

            # Applying the convolution
            if sigma_kernel > 0:
                sigma_LSF                = np.sqrt(sigma_LSF**2 + sigma_kernel**2)       # [input px]
                valid_flux               = np.isfinite(flux_input)                       # Fill internal NaNs (linear) and crop to avoid edges inside the convolution window
                flux_filled              = fill_nan_linear(wave_input, flux_input)       # NaN gaps are filled with linear interpolation
                valid_filled             = np.isfinite(flux_filled)                      # NaN edges can remain
                flux_input[valid_filled] = gaussian_filter1d(flux_filled[valid_filled], sigma=sigma_kernel)
                flux_input[~valid_flux]  = np.nan                                        # Retrieving original NaN
            
            # The output spectral resolution is greater than the input spectral resolution 
            elif verbose:
                print()
                print_warning(f"WARNING (self.degrade_resolution): The output spectral resolution ({round(np.nanmedian(R_output), -2):.0f}) is greater (or ~) than the input spectral resolution ({round(np.nanmedian(R_input), -2):.0f}). LSF convolution will not be applied.")

        # Down-binning to the output grid (if it is coarser than the input one)
        flux_output, sigma_output, _ = rebin_spectrum_mean(specHR=flux_input, sigmaHR=sigma_input, weightHR=None, lamHR=wave_input, lamLR=wave_output) # down binned flux
        sigma_bin                    = R_nyquist_input / (np.sqrt(12) * R_nyquist_output) # [input px]
        sigma_LSF                    = np.sqrt(sigma_LSF**2 + sigma_bin**2)               # [input px]
        R_LSF                        = R_nyquist_input / (np.sqrt(2*np.log(2)) * sigma_LSF)
        
        # Creating degraded Spectrum instance
        spectrum_output = Spectrum(wavelength=wave_output, flux=flux_output, R=R_LSF, T=self.T, lg=self.lg, model=self.model, rv=self.rv, vsini=self.vsini, sigma=sigma_output)
        
        # # Resolutions sanity check plot
        # plt.figure(dpi=300, figsize=(10, 6))
        # plt.plot(wave_output, R_nyquist_input,  lw=5, label="R_nyquist_input")
        # plt.plot(wave_output, R_nyquist_output, lw=5, label="R_nyquist_output")
        # plt.plot(wave_output, R_input,  lw=2, label="R_input")
        # plt.plot(wave_output, R_output, lw=2, label="R_output")
        # plt.plot(wave_output, R_LSF, lw=2, c="k", label="R_LSF")
        # plt.xlabel("Wavelength [µm]")
        # plt.ylabel("Resolution")
        # plt.legend()
        # plt.yscale('log')
        # plt.show()
        
        # Conserve the *total* flux in the overlapping domain (if needed)
        if renorm:
            print_warning("WARNING: degrade_resolution(renorm=True)")
            ratio                 = np.nansum(self.flux[mask_overlap]) / np.nansum(spectrum_output.flux[valid_overlap])
            spectrum_output.flux *= ratio
            if spectrum_output.sigma is not None:
                spectrum_output.sigma *= ratio

        return spectrum_output
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def evenly_spaced(self, lmin, lmax, R_interp=None, renorm=False, fill_value=np.nan):
        """
        Resample onto a *linear, constant-step* wavelength grid spanning
        either the spectrum range or the (slightly padded) 'wave_output' range.

        Parameters
        ----------
        renorm: bool, optional
            Conserve *total* flux after resampling. Default False.
        fill_value: float, optional
            Extrapolation fill value. Default np.nan.
        sigma: (N,) array_like or None
            Optional per-pixel uncertainties to be propagated.

        Returns
        -------
        Spectrum
            Resampled spectrum on a fine, linear, evenly spaced grid.
        """

        mask_wavelength = (self.wavelength >= lmin) & (self.wavelength <= lmax)
        
        # Interpolation resolution: use max(R) (Nyquist-based) but cap at R0_max
        if R_interp is None:
            R_interp = get_resolution(self.wavelength[mask_wavelength], func=np.nanmax) # Interpolation Resolution (need to be the max res to avoid spectral information loss)        
        R_interp    = min(R_interp, R0_max)                                             # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
        wave_interp = get_wavelength_axis_constant_dl(lmin=lmin, lmax=lmax, R=R_interp) # Constant and linear input wavelength array
        spectrum    = self.interpolate_wavelength(wave_output=wave_interp, renorm=renorm, fill_value=fill_value) 
        return spectrum
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def doppler_shift(self, rv, renorm=False, fill_value=np.nan):
        """
        Apply a Doppler shift for a given radial velocity.

        Parameters
        ----------
        rv: float
            Radial velocity [km/s]. Positive = redshift, negative = blueshift.
        renorm: bool, optional
            Conserve *total* flux after interpolation. Default False.
        fill_value: float, optional
            Extrapolation fill value. Default np.nan.

        Returns
        -------
        Spectrum
            Doppler-shifted spectrum (same sampling as input).
        """
        if rv==0:
            return self.copy()
        else: # λ' = λ * sqrt( (1 + beta) / (1 - beta ) with v in m/s and c in m/s
            spectrum             = self.copy()
            beta                 = 1000*rv / c
            spectrum.wavelength *= np.sqrt( (1 + beta) / (1 - beta) ) # Offset wavelength axis
            spectrum_rv          = spectrum.interpolate_wavelength(wave_output=self.wavelength, renorm=renorm, fill_value=fill_value)
            spectrum_rv.rv       = spectrum_rv.rv + rv
            return spectrum_rv

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def broad(self, vsini, epsilon=0.8, fastbroad=True):
        """
        Convolve the spectrum with a rotational broadening profile.

        Parameters
        ----------
        vsini: float
            Projected rotational velocity [km/s]. Must be >= 0.
        epsilon: float, optional
            Linear limb-darkening coefficient in [0, 1]. Default 0.8.
        fastbroad: bool, optional
            Use 'fastRotBroad' (faster, approximate) if True,
            else 'rotBroad' (slower, more accurate) (from PyAstronomy).

        Returns
        -------
        Spectrum
            Rotationally broadened spectrum.
        """
        if vsini < 0:
            raise ValueError("'vsini' must be non-negative.")
        elif vsini==0:
            return self.copy()
        else:
            if fastbroad: # fast spectral broadening (but less accurate)
                flux = fastRotBroad(self.wavelength*1e4, self.flux, epsilon=epsilon, vsini=vsini)
            else: # slow spectral broadening (but more accurate)
                flux = rotBroad(self.wavelength*1e4, self.flux, epsilon=epsilon, vsini=vsini)
            spectrum_vsini       = self.copy()
            spectrum_vsini.flux  = flux
            spectrum_vsini.vsini = spectrum_vsini.vsini + vsini
            return spectrum_vsini 

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def get_psd(self, smooth=0, one_sided=True):
        """
        Compute the power spectral density (PSD) of the flux array.
    
        Parameters
        ----------
        smooth: float, optional
            Standard deviation (in bins) of a Gaussian smoothing applied to the PSD.
            Use 0 to disable smoothing. Default is 0.
        one_sided: bool, optional
            If True, return a one-sided PSD (positive frequencies) using an rFFT
            with proper energy folding (doubling positive-frequency bins except DC
            and, if applicable, Nyquist). If False, return a two-sided PSD.
            Default is True.
    
        Returns
        -------
        res: ndarray
            “Resolution-like” frequency axis scaled by ~ 2*R (heuristic).
        PSD: ndarray
            Power spectral density (arbitrary units).
        """
        
        # Copy and handle NaNs robustly
        R_nyquist = get_resolution(wavelength=self.wavelength, func=np.nanmedian)
        signal    = np.asarray(self.flux, dtype=float)
        signal    = signal - np.nanmean(signal) # subtracting DC
        if np.isnan(signal).any():
            signal = signal[np.isfinite(signal)]
            print_warning("WARNING (get_psd): NaN values inside self.flux...")
        
        N = signal.size
        if N < 2:
            raise ValueError("Signal too short for PSD computation.")
            
        # “Resolution-like” axis    
        ffreq = np.fft.rfftfreq(N)  # cycles per sample
        res   = ffreq * 2*R_nyquist # 2*R = sampling resolution (assuming Nyquist)

        if one_sided:
            # One-sided PSD via rFFT with correct energy folding
            TF  = np.fft.rfft(signal)
            PSD = np.abs(TF)**2 / N
            if N % 2 == 0:
                # Even N: double bins 1..-2 (exclude DC and Nyquist)
                if PSD.size > 2:
                    PSD[1:-1] = PSD[1:-1] * 2
            else:
                # Odd N: double bins 1..end (exclude DC only)
                if PSD.size > 1:
                    PSD[1:] = PSD[1:] * 2
        else:
            # Two-sided PSD
            TF  = np.fft.fft(signal)
            PSD = np.abs(TF)**2 / N
    
        # Optional Gaussian smoothing
        if smooth > 0:
            PSD = gaussian_filter1d(PSD, sigma=smooth)
    
        return res, PSD



def get_psd(wave, flux, R=None, smooth=0):
    """
    Calculate the psd of the inpux flux.

    Parameters
    ----------
    wave: array 1d
        wavelength axis.
    flux: array 1d
        Flux axis.
    R: float
        Sampling resolution.
    smooth: float, optional
        Smoothing parameters of the PSF. The default is 0.

    Returns
    -------
    res: array 1d
        resolution axis.
    psd: array 1d
        PSD axis.
    """
    if wave is None and R is None:
        raise KeyError("'wave' and 'R' are None...")
    valid = np.isfinite(flux)
    if wave is not None:
        wave = wave[valid]
        R    = get_resolution(wavelength=wave, func=np.array) # NEEDS TO BE THE SAMPLING RESOLUTION 
    else:
        wave = np.arange(flux[valid].size, dtype=float) # “index” axis        
    flux     = flux[valid]
    res, psd = Spectrum(wavelength=wave, flux=flux, R=R).get_psd(smooth=smooth)
    return res, psd



#######################################################################################################################
##################################################### Loading functions: ##############################################
#######################################################################################################################

def get_model_grid(model, instru=None):
    """
    Return (T_grid, lg_grid) for a given model (and optional instrument).

    Parameters
    ----------
    model: str
        Model family (e.g., 'BT-Settl', 'Exo-REM', 'BT-NextGen', 'mol_CO', ...).
    instru: str, optional
        Instrument name; used to decide which subgrid to return for some models.

    Returns
    -------
    T_grid: ndarray
        Supported temperatures (K).
    lg_grid: ndarray
        For thermal/stellar models: supported log10(cm/s^2).
        For molecular models ('mol_*'): array of molecule names (dtype=object).

    Raises
    ------
    KeyError
        If the model name is unsupported.
    """
    
    
    
    # --------------------
    # Planets / substellar
    # ---------------------
    if model == "BT-Settl" or "PICASO" in model:
        T_grid  = np.concatenate([[200, 220, 240, 250, 260, 280, 300, 320, 340, 360, 380, 400, 450], np.arange(500, 900, 50), np.arange(900, 3100, 100)])
        lg_grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

    elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
        T_grid  = np.arange(1400, 3100, 100)
        lg_grid = np.array([4.5, 5.0])

    elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
        if instru is None: # Default: low resolution
            T_grid = np.arange(400, 2050, 50)
            lg_grid = np.arange(3.0, 5.5, 0.5)
        else:
            lmin, lmax = get_band_lims(band=instru)
            if lmin >= 1.0 and lmax <= 5.3: # Very high resolution
                T_grid  = np.arange(200, 1950, 50)
                lg_grid = np.arange(3.0, 5.5, 0.5)
            elif lmin <= 4.0: # Low resolution
                T_grid  = np.arange(400, 2050, 50)
                lg_grid = np.arange(3.0, 5.5, 0.5)
            else: # High res (starts at 4 µm)
                T_grid  = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
                lg_grid = np.array([3.5, 4.0])

    elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
        T_grid  = np.array([200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        g_ms2   = np.array([100, 300, 1000, 3000])
        lg_grid = np.round(np.log10(g_ms2 * 1e2), 4)

    elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
        T_grid  = np.arange(400, 1250, 50)
        g_ms2   = np.array([10, 30, 100, 300, 1000])
        lg_grid = np.round(np.log10(g_ms2 * 1e2), 4)

    elif model == "SONORA": # https://zenodo.org/records/5063476
        T_grid  = np.concatenate([np.arange(200, 1050, 50), np.arange(1100, 2500, 100)])
        g_ms2   = np.array([10, 31, 100, 316, 1000, 3160])
        lg_grid = np.round(np.log10(g_ms2 * 1e2), 4)

    elif "mol_" in model: # https://hitran.org/lbl/
        T_grid = np.concatenate([np.arange(200, 1000, 50), np.arange(1000, 3100, 100)])
        lg_grid = np.array(["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "NO", "SO2", "NO2", "NH3"], dtype=object)
        
    elif model in ["Jupiter", "Saturn", "Uranus", "Neptune"]:
        if model == "Jupiter":
            T_grid  = np.array([88])
            lg_grid = np.array([3.4]) # np.log10(24.79*100) # https://en.wikipedia.org/wiki/Jupiter
        elif model == "Saturn":
            T_grid  = np.array([81])
            lg_grid = np.array([3.0]) # np.log10(10.44*100) # https://en.wikipedia.org/wiki/Saturn
        elif model == "Uranus":
            T_grid  = np.array([49])
            lg_grid = np.array([2.9]) # np.log10(8.69*100) # https://en.wikipedia.org/wiki/Uranus
        elif model == "Neptune":
            T_grid  = np.array([47])
            lg_grid = np.array([3.0]) # np.log10(11.15*100) # https://en.wikipedia.org/wiki/Neptune
            
    elif "PSG" in model:
        T_grid  = np.arange(200, 3100, 100)
        lg_grid = np.arange(2.5, 4.5, 0.5)
    
    # ----------------
    # Stars
    # ----------------
    elif model == "BT-NextGen":
        T_grid = np.concatenate([np.arange(3000, 10000, 200), np.arange(10000, 41000, 1000)])
        lg_grid = np.array([3.0, 3.5, 4.0, 4.5])
    
    elif model == "Husser":
        T_grid = np.concatenate([np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)])
        lg_grid = np.arange(0.0, 6.5, 0.5)
    
    else:
        raise KeyError(f"{model} is not a valid model. Supported: BT-Settl, PICASO, BT-Dusty, Exo-REM, Morley, Saumon, SONORA, BT-NextGen, Husser, Solar-system planets, PSG_*, or mol_*.")
    
    return T_grid, lg_grid



def get_T_lg_valid(T, lg, model, instru=None, T_grid=None, lg_grid=None):
    """
    Return the nearest valid (T, lg) pair on the requested model grid.

    Parameters
    ----------
    T: float or int
        Requested temperature (K).
    lg: float or int or str
        Requested surface gravity log10(cm/s^2) for thermal/stellar models.
        For molecular models ('mol_*'), this is the molecule name (str).
    model: str
        Model family (e.g., 'BT-Settl', 'Exo-REM', 'mol_CO', ...).
    instru: str, optional
        Instrument name (some grids depend on wavelength coverage).
    T_grid: ndarray, optional
        Pre-fetched temperature grid. If None, retrieved via 'get_model_grid'.
    lg_grid: ndarray or list[str], optional
        Pre-fetched gravity grid (or molecule list for 'mol_*'). If None, retrieved.

    Returns
    -------
    T_valid: float
        Closest temperature available in the model grid.
    lg_valid: float or str
        Closest gravity value (thermal/stellar models) or unchanged molecule (molecular models).
    """
    if T_grid is None or lg_grid is None:
        T_grid, lg_grid = get_model_grid(model, instru=instru)
    T_valid = T_grid[(np.abs(T_grid-T)).argmin()] # Closest available value
    if "mol_" in model:
        if model[4:] not in lg_grid:
            raise KeyError(f"{lg} is not a valid molecule, please choose among: {lg_grid}")
        lg_valid = lg
    else:
        lg_valid = lg_grid[(np.abs(lg_grid-lg)).argmin()] # Closest lg value
    return T_valid, lg_valid



@lru_cache(maxsize=128)
def load_spectrum(T, lg, model, instru=None, spectra_path=spectra_path):
    """
    Load a spectrum from disk for a given model and parameters.

    Parameters
    ----------
    T: float
        Temperature (K).
    lg: float or str
        Surface gravity log10(cm/s^2). For molecular models ('mol_*'), the molecule name.
    model: str
        Model family (e.g., 'BT-Settl', 'Exo-REM', 'PICASO', 'mol_CO', 'BT-NextGen', ...).
    instru: str, optional
        Instrument name; may select a different resolution grid for some models.
    spectra_path: str, optional
        Base directory where spectra are stored.

    Returns
    -------
    spectrum: Spectrum
        Spectrum with wavelength in µm and flux in J/s/m2/µm (unitless for albedos).
        The 'R' field is estimated from the wavelength grid.

    Raises
    ------
    KeyError
        If inputs are inconsistent with available files/grids.
    """
    try:
        if model == "BT-Settl": # https://articles.adsabs.harvard.edu/pdf/2013MSAIS..24..128A              
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
        
        elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
            
        elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
            if instru is None: # Default: low resolution
                spectrum_path = f"{spectra_path}/planet_spectrum/{model}/low_res/spectra_YGP_{T:.0f}K_logg{lg:.1f}_met1.00_CO0.50.fits"
            else:
                lmin, lmax = get_band_lims(band=instru) # [µm]
                if lmin >= 1.0 and lmax <= 5.3: # Very high resolution
                    FeH = 0.0  # Métallicité
                    CO  = 0.65 # Ratio C/O
                    spectrum_path = f"{spectra_path}/planet_spectrum/{model}/very_high_res/spect_Teff={T:04.0f}K_logg={lg:.1f}_FeH={FeH:+.1f}_CO={CO:.2f}.fits"
                elif lmin <= 4.0: # Low resolution
                    spectrum_path = f"{spectra_path}/planet_spectrum/{model}/low_res/spectra_YGP_{T:.0f}K_logg{lg}_met1.00_CO0.50.fits"
                else: # High res (starts at 4 µm)
                    spectrum_path = f"{spectra_path}/planet_spectrum/{model}/high_res/spectra_YGP_{T:.0f}K_logg{lg}_met1.00_CO0.50.fits"
            wave, flux = fits.getdata(spectrum_path)
            
        elif model == "PICASO": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/PICASO/thermal_gas_giant_{T:.0f}K_lg{lg:.1f}.fits")
            
        elif model == "PICASO_albedo": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/albedo/PICASO/albedo_gas_giant_{T:.0f}K_lg{lg:.1f}.fits")
            
        elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
            g_planet = round(10**lg*1e-2) # m/s2
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/Morley/sp_t{T:.0f}g{g_planet}.fits")
            
        elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
            g_planet = round(10**lg*1e-2) # m/s2
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/Saumon/sp_t{T:.0f}g{g_planet}nc.fits")
        
        elif model == "SONORA": # https://zenodo.org/records/5063476
            g_planet = round(10**lg*1e-2) # m/s2
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/SONORA/sp_t{T:.0f}g{g_planet}nc_m0.0.fits")
            
        elif model[:4] == "mol_": # https://hitran.org/lbl/
            molecule = model[4:]
            wave, flux = fits.getdata(spectra_path + f"/planet_spectrum/molecular/{molecule}_T{T:.0f}K.fits")
        
        elif model in ["Jupiter", "Saturn", "Uranus", "Neptune"]:
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/solar system/psg_{model}_rad.fits")
        
        elif "PSG" in model:
            ptype      = model.split("_")[-1]
            wave, flux = fits.getdata(f"{spectra_path}/planet_spectrum/PSG/{ptype}/thermal_{ptype}_{T:.0f}K_lg{lg:.1f}.fits")
        
        elif model == "BT-NextGen":
            wave, flux = fits.getdata(f"{spectra_path}/star_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
        
        elif model == "Husser":
            wave = fits.getdata(f"{spectra_path}/star_spectrum/{model}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            flux = fits.getdata(f"{spectra_path}/star_spectrum/{model}/lte{T:05.0f}-{lg:4.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
        
        else:
            raise KeyError(model+" IS NOT A VALID THERMAL MODEL: BT-NextGen, Husser, BT-Settl, BT-Dusty, Exo-REM, PICASO, Morley, Saumon or SONORA.")
    
        # We assume that all models are initially Nyquist sammpled
        R        = get_resolution(wavelength=wave, func=np.array)
        spectrum = Spectrum(wavelength=wave, flux=flux, R=R, T=T, lg=lg, model=model, rv=0, vsini=0)
        return spectrum # [J/s/m2/µm] or [no unit] for albedos
    
    except Exception as e:
        raise KeyError(f"{T}K or {lg} are not valid parameters of the {model} grid: {e}")



def interpolate_T_lg_spectrum(T_valid, lg_valid, T, lg, model, instru=None, spectra_path=spectra_path, T_grid=None, lg_grid=None):
    """
    Interpolate a spectrum at (T, lg) using surrounding grid spectra.

    Parameters
    ----------
    T_valid: float
        Nearest grid temperature to the request.
    lg_valid: float or str
        Nearest grid gravity (or molecule name for 'mol_*').
    T: float
        Target temperature (K).
    lg: float
        Target gravity log10(cm/s^2).
    model: str
        Model family.
    instru: str, optional
        Instrument name; can affect grid choice.
    spectra_path: str, optional
        Base directory of spectra on disk.
    T_grid: ndarray, optional
        Temperature grid; if None, fetched via 'get_model_grid'.
    lg_grid: ndarray or list[str], optional
        Gravity (or molecule-name) grid; if None, fetched via 'get_model_grid'.

    Returns
    -------
    spectrum: Spectrum
        Interpolated spectrum at (T, lg). Wavelength in µm, flux in J/s/m2/µm (unitless for albedos).
    """

    # Retrieve the model grid
    if T_grid is None or lg_grid is None:
        T_grid, lg_grid = get_model_grid(model, instru=instru)
    
    # Molecular models: only interpolate T; lg is a molecule label
    if "mol_" in model:
        T_lo, T_hi = get_bracket_values(T, T_grid)
        if T_lo == T_hi:
            return load_spectrum(T_lo, lg_valid, model, instru=instru, spectra_path=spectra_path).copy()
        else:
            s_lo = load_spectrum(T_lo, lg_valid, model, instru=instru, spectra_path=spectra_path)
            s_hi = load_spectrum(T_hi, lg_valid, model, instru=instru, spectra_path=spectra_path)
            wave = s_lo.wavelength
            if len(s_hi.wavelength) != len(wave):
                s_hi = s_hi.interpolate_wavelength(wave, renorm=False)
            flux = linear_interpolate(s_lo.flux, s_hi.flux, T_lo, T_hi, T)

    # Regular thermal/stellar models: bilinear in (T, lg)
    else:
        T_lo, T_hi   = get_bracket_values(T,  T_grid)
        lg_lo, lg_hi = get_bracket_values(lg, lg_grid)
    
        # No interpolation along T and lg
        if T_lo == T_hi and lg_lo == lg_hi:
            return load_spectrum(T_lo, lg_lo, model, instru=instru, spectra_path=spectra_path).copy()
    
        # Interpolation along T
        elif lg_lo == lg_hi:
            s_lo = load_spectrum(T_lo, lg_lo, model, instru=instru, spectra_path=spectra_path)
            s_hi = load_spectrum(T_hi, lg_lo, model, instru=instru, spectra_path=spectra_path)
            wave = s_lo.wavelength
            if len(s_hi.wavelength) != len(wave):
                s_hi = s_hi.interpolate_wavelength(wave, renorm=False)
            flux = linear_interpolate(s_lo.flux, s_hi.flux, T_lo, T_hi, T)
        
        # Interpolation along lg
        elif T_lo == T_hi:
            s_lo = load_spectrum(T_lo, lg_lo, model, instru=instru, spectra_path=spectra_path)
            s_hi = load_spectrum(T_lo, lg_hi, model, instru=instru, spectra_path=spectra_path)
            wave = s_lo.wavelength
            if len(s_hi.wavelength) != len(wave):
                s_hi = s_hi.interpolate_wavelength(wave, renorm=False)
            flux = linear_interpolate(s_lo.flux, s_hi.flux, lg_lo, lg_hi, lg)

        # Interpolation along T and lg (Full bilinear interpolation)
        else:
            s_ll = load_spectrum(T_lo, lg_lo, model, instru=instru, spectra_path=spectra_path)
            s_lh = load_spectrum(T_lo, lg_hi, model, instru=instru, spectra_path=spectra_path)
            s_hl = load_spectrum(T_hi, lg_lo, model, instru=instru, spectra_path=spectra_path)
            s_hh = load_spectrum(T_hi, lg_hi, model, instru=instru, spectra_path=spectra_path)
            wave = s_ll.wavelength
            if len(s_lh.wavelength) != len(wave):
                s_lh = s_lh.interpolate_wavelength(wave, renorm=False)
            if len(s_hl.wavelength) != len(wave):
                s_hl = s_hl.interpolate_wavelength(wave, renorm=False)
            if len(s_hh.wavelength) != len(wave):
                s_hh = s_hh.interpolate_wavelength(wave, renorm=False)
            f_Tlo = linear_interpolate(s_ll.flux, s_lh.flux, lg_lo, lg_hi, lg)
            f_Thi = linear_interpolate(s_hl.flux, s_hh.flux, lg_lo, lg_hi, lg)
            flux  = linear_interpolate(f_Tlo,     f_Thi,     T_lo,  T_hi,  T)
    
    # We assume that all models are initially Nyquist sammpled
    R        = get_resolution(wavelength=wave, func=np.array)
    spectrum = Spectrum(wavelength=wave, flux=flux, R=R, T=T, lg=lg, model=model, rv=0, vsini=0, sigma=None)
    return spectrum # [J/s/m2/µm] or [no unit] for albedos



def load_planet_spectrum(T_planet=1000, lg_planet=4.0, model="BT-Settl", interpolated_spectrum=True, instru=None, spectra_path=spectra_path, T_grid=None, lg_grid=None):
    """
    Load a planet spectrum at the closest grid point or bilinearly interpolate to (T_planet, lg_planet).

    Parameters
    ----------
    T_planet: float, optional
        Planet temperature (K).
    lg_planet: float, optional
        Planet surface gravity log10(cm/s^2).
    model: str, optional
        Planet model family.
    interpolated_spectrum: bool, optional
        If True, interpolates to the exact (T, lg). If False, returns the nearest grid spectrum.
    instru: str, optional
        Instrument name for model sub-grids.
    spectra_path: str, optional
        Base directory for spectra.
    T_grid: ndarray, optional
        Temperature grid (bypass fetching).
    lg_grid: ndarray, optional
        Gravity grid (bypass fetching).

    Returns
    -------
    spectrum: Spectrum
        Planet spectrum (wavelength in µm, flux in J/s/m2/µm) or unitless for albedos.
    """
    
    if model == "blackbody":
        R        = R0_min
        wave     = get_wavelength_axis_constant_R(lmin=0.98*LMIN, lmax=1.02*LMAX, R=R)
        flux     = np.pi*get_blackbody(wave=wave, Teff=T_planet)
        spectrum = Spectrum(wavelength=wave, flux=flux, R=R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0, sigma=None)
        return spectrum
    
    else:
        # Closest valid values parameters in the model grid
        T_valid, lg_valid = get_T_lg_valid(T=T_planet, lg=lg_planet, model=model, instru=instru, T_grid=T_grid, lg_grid=lg_grid)
    
        # Interpolates the grid in order to have the precise T_planet and lg_planet values
        if interpolated_spectrum and (T_valid != T_planet or lg_valid != lg_planet):
            return interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_planet, lg=lg_planet, model=model, instru=instru, spectra_path=spectra_path, T_grid=T_grid, lg_grid=lg_grid)
    
        # Load the spectrum with the closest parameters values in the model grid
        else:
            return load_spectrum(T_valid, lg_valid, model, instru=instru, spectra_path=spectra_path).copy()



def load_albedo_spectrum(T_planet, lg_planet, model="PICASO", airmass=2.0, interpolated_spectrum=True, spectra_path=spectra_path, T_grid=None, lg_grid=None):
    """
    Load an albedo spectrum and optionally transform it ('flat' or 'tellurics').
    see Eq.(1) of Lovis et al. (2017): https://arxiv.org/pdf/1609.03082

    Parameters
    ----------
    T_planet: float
        Planet temperature (K).
    lg_planet: float
        Planet surface gravity log10(cm/s^2).
    model: {'PICASO', 'flat', 'tellurics'}, optional
        Albedo model to return. 'PICASO' uses the geometric albedo; 'flat' sets a constant
        equal to the mean geometric albedo; 'tellurics' scales a sky-transmission curve
        to match the mean geometric albedo over its wavelength range.
    airmass: float, optional
        Airmass used for 'tellurics' transmission file name.
    interpolated_spectrum: bool, optional
        If True, interpolate the underlying albedo grid to (T, lg).
    instru: str, optional
        Instrument name (not used by 'PICASO' albedo files; included for symmetry).
    spectra_path: str, optional
        Base directory for spectra.

    Returns
    -------
    spectrum: Spectrum
        Unitless albedo spectrum (wavelength in µm).
    """
    # --- Always load the base PICASO albedo, then transform per 'model' ---
    
    # Closest valid values parameters in the model grid
    T_valid, lg_valid = get_T_lg_valid(T=T_planet, lg=lg_planet, model="PICASO_albedo", instru=None, T_grid=T_grid, lg_grid=lg_grid)

    # Interpolates the grid in order to have the precise T_planet and lg_planet values
    if interpolated_spectrum and (T_valid != T_planet or lg_valid != lg_planet):
        albedo = interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_planet, lg=lg_planet, model="PICASO_albedo", instru=None, spectra_path=spectra_path, T_grid=T_grid, lg_grid=lg_grid)

    # Load the spectrum with the closest parameters values in the model grid
    else:
        albedo = load_spectrum(T=T_valid, lg=lg_valid, model="PICASO_albedo", instru=None, spectra_path=spectra_path).copy()
    albedo.model = "PICASO"
    
    if model == "PICASO":
        return albedo

    elif model == "flat":
        albedo_geo  = np.nanmean(albedo.flux)
        albedo.flux = np.zeros_like(albedo.flux) + albedo_geo
        return albedo

    elif model == "tellurics":
        wave_tell, tell1, R_tell = _load_tell_trans(airmass=1.0, return_R=True)
        tell                     = tell1**airmass
        mask_wavelength          = (albedo.wavelength >= wave_tell[0]) & (albedo.wavelength <= wave_tell[-1])
        albedo_geo               = np.nanmean(albedo.flux[mask_wavelength])
        albedo.flux              = albedo_geo / np.nanmean(tell) * tell
        albedo.wavelength        = wave_tell
        albedo.R                 = R_tell
        albedo.model             = f"tellurics(airmass={airmass:.2f})"
        return albedo
    
    else:
        raise KeyError("Invalid reflected model: use 'tellurics', 'flat', or 'PICASO'.")
    


def load_mol_spectrum(T_mol=1000, lg_mol=4.0, model="mol_CO2", interpolated_spectrum=True, spectra_path=spectra_path, T_grid=None, lg_grid=None):
    """
    Load a molecular spectrum at the closest grid point or bilinearly interpolate to (T_mol, lg_mol).

    Parameters
    ----------
    T_mol: float, optional
        Planet temperature (K).
    lg_mol: float, optional
        Planet surface gravity log10(cm/s^2).
    model: str, optional
        Planet model family.
    interpolated_spectrum: bool, optional
        If True, interpolates to the exact (T, lg). If False, returns the nearest grid spectrum.
    instru: str, optional
        Instrument name for model sub-grids.
    spectra_path: str, optional
        Base directory for spectra.
    T_grid: ndarray, optional
        Temperature grid (bypass fetching).
    lg_grid: ndarray, optional
        Gravity grid (bypass fetching).

    Returns
    -------
    spectrum: Spectrum
        Planet spectrum (wavelength in µm, flux in J/s/m2/µm) or unitless for albedos.
    """
    
    if "mol_" not in model:
        raise ValueError(f"model should be in the 'mol_*' format, not {model}")
    
    # Closest valid values parameters in the model grid
    T_valid, lg_valid = get_T_lg_valid(T=T_mol, lg=lg_mol, model=model, instru=None, T_grid=T_grid, lg_grid=lg_grid)

    # Interpolates the grid in order to have the precise T_mol and lg_mol values
    if interpolated_spectrum and (T_valid != T_mol or lg_valid != lg_mol):
        return interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_mol, lg=lg_mol, model=model, instru=None, spectra_path=spectra_path, T_grid=T_grid, lg_grid=lg_grid)

    # Load the spectrum with the closest parameters values in the model grid
    else:
        return load_spectrum(T_valid, lg_valid, model, instru=None, spectra_path=spectra_path).copy()



def load_star_spectrum(T_star, lg_star, model="BT-NextGen", interpolated_spectrum=True, spectra_path=spectra_path):
    """
    Load a stellar spectrum at the closest grid point or interpolate to (T_star, lg_star).

    Parameters
    ----------
    T_star: float
        Stellar effective temperature (K).
    lg_star: float
        Surface gravity log10(cm/s^2).
    model: {'BT-NextGen', 'Husser'}, optional
        Stellar model family.
    interpolated_spectrum: bool, optional
        If True, interpolates to the exact (T, lg). If False, returns nearest grid spectrum.
    spectra_path: str, optional
        Base directory.

    Returns
    -------
    spectrum: Spectrum
        Stellar spectrum with wavelength in µm and flux in J/s/m2/µm.
    """
    # Closest valid values parameters in the model grid
    T_valid, lg_valid = get_T_lg_valid(T=T_star, lg=lg_star, model=model, instru=None, T_grid=None, lg_grid=None)

    # Interpolates the grid in order to have the precise T_star and lg_star values
    if interpolated_spectrum and (T_valid != T_star or lg_valid != lg_star):
        return interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_star, lg=lg_star, model=model, instru=None, spectra_path=spectra_path, T_grid=None, lg_grid=None)

    # Load the spectrum with the closest parameters values in the model grid
    else:
        return load_spectrum(T_valid, lg_valid, model, instru=None, spectra_path=spectra_path).copy()



@lru_cache(maxsize=1)
def _load_vega_spectrum(vega_path=vega_path):
    """
    Load the Vega reference spectrum (for photometric calibration).

    Parameters
    ----------
    vega_path: str, optional
        Path to a 2-column table containing wavelength [nm] and flux [erg/s/cm^2/A].

    Returns
    -------
    vega_spectrum: Spectrum
        Vega spectrum with wavelength in µm and flux in J/s/m2/µm.
    """
    f             = fits.getdata(os.path.join(vega_path))
    wave          = f[:, 0]*1e-3 # nm => µm
    flux          = f[:, 1]*10   # 10 = 1e4 * 1e4 * 1e-7: erg/s/cm2/A => erg/s/cm2/µm => erg/s/m2/µm => J/s/m2/µm
    R             = get_resolution(wavelength=wave, func=np.array)
    vega_spectrum = Spectrum(wavelength=wave, flux=flux, R=R, T=9550, lg=3.95, model="Vega", rv=0, vsini=21.6, sigma=None)
    return vega_spectrum



def load_vega_spectrum():
    """
    Load the Vega reference spectrum (for photometric calibration).

    Parameters
    ----------
    vega_path: str, optional
        Path to a 2-column table containing wavelength [nm] and flux [erg/s/cm^2/A].

    Returns
    -------
    vega_spectrum: Spectrum
        Vega spectrum with wavelength in µm and flux in J/s/m2/µm.
    """

    return _load_vega_spectrum().copy()
        


#######################################################################################################################
############################################# Spectra on instru and band: #############################################
#######################################################################################################################

def get_wave_K():
    """
    Build K-band grid (for photometric purpose only)
    """
    lmin_K, lmax_K = get_band_lims(band="K") # [µm]
    R_K            = R0_min
    wave_K         = get_wavelength_axis_constant_dl(lmin=lmin_K, lmax=lmax_K, R=R_K)
    return wave_K



def get_wave_band(config_data, band):
    """
    Return the wavelength grid associated with a given spectral band.

    This function extracts the minimum wavelength, maximum wavelength, and
    spectral resolution of the selected band from 'config_data', then builds
    a wavelength axis over that band.

    Parameters
    ----------
    config_data : dict
        Instrument configuration dictionary containing the grating properties
        for each spectral band. Each entry in 'config_data['gratings']' is
        expected to provide the attributes 'lmin', 'lmax', and 'R'.
    band : str
        Name of the spectral band to extract.

    Returns
    -------
    numpy.ndarray
        One-dimensional wavelength array for the selected band, in [µm].
    """
    lmin_band = config_data['gratings'][band].lmin       # Lambda_min of the considered band [µm]
    lmax_band = config_data['gratings'][band].lmax       # Lambda_max of the considered band [µm]
    R_band    = config_data['gratings'][band].R          # Spectral resolution of the band
    # TODO: decide whether we want constant dl or constant R
    wave_band = get_wavelength_axis_constant_dl(lmin=lmin_band, lmax=lmax_band, R=R_band) # Constant dl wavelength array on the considered band
    #wave_band = get_wavelength_axis_constant_R(lmin=lmin_band, lmax=lmax_band, R=R_band) # Constant R  wavelength array on the considered band

    return wave_band # [µm]


def get_spectrum_instru(band0, R, config_data, mag, spectrum, wave_instru=None):
    """
    Restrict a spectrum to the instrument wavelength range and scale it to match
    the given Vega magnitude in 'band0'. Return both the photon-rate spectrum
    (photons/min) over the instrument range and the corresponding energy-density
    spectrum (J/s/m2/µm) sampled on the same grid.

    Notes
    -----
    - The input 'spectrum' **must** be in energy density units (J/s/m2/µm).
    - Magnitude is assumed to be Vega-based; the zero point comes from 'load_vega_spectrum()'.
    - No in-place modification is performed on the input 'spectrum'.

    Parameters
    ----------
    band0: str
        Band in which the input magnitude is defined. Use '"instru"' to
        use the full instrument range; otherwise provide a band key like "J", "H", etc.
    R: float
        Reference sampling resolution used to construct intermediate wavelength grids.
        Should be comfortably higher than the instrument resolving power.
    config_data: dict
        Instrument configuration dict containing:
        - "name"
        - "lambda_range": {"lambda_min", "lambda_max"}
        - "telescope": {"area": ...}
    mag: float
        Vega magnitude to impose in 'band0'.
    spectrum: Spectrum
        Input spectrum in J/s/m2/µm.

    Returns
    -------
    spectrum_instru: Spectrum
        Spectrum restricted to the instrument range and converted to ph/bin/mn.
    spectrum_density: Spectrum
        Same wavelength grid as 'spectrum_instru', but in J/s/m2/µm (energy density).

    Raises
    ------
    KeyError
        If 'band0' cannot be resolved to a wavelength range via globals.
    """
    # Resolve the reference band limits for magnitude scaling
    try:
        lmin_band0, lmax_band0 = get_band_lims(band=config_data["name"] if band0.lower() == "instru" else band0) # [µm]
    except Exception:
        raise KeyError(f"{band0} is not a recognized band. Choose among: {bands} or 'instru' for full instrument range.")        
    
    # Build an intermediate grid on band0 to compute the Vega scaling ratio: F_obj = F_vega * 10^{-0.4*mag}
    wave_band0     = get_wavelength_axis_constant_dl(lmin=lmin_band0, lmax=lmax_band0, R=R) # Wavelength array on band0 [µm]
    spectrum_band0 = spectrum.interpolate_wavelength(wave_band0, renorm=False)              # [J/s/m2/µm] Interpolating the input spectrum on band0
    vega_band0     = load_vega_spectrum().interpolate_wavelength(wave_band0, renorm=False)  # [J/s/m2/µm] Getting the vega spectrum
    scale          = get_scale_to_mag(wave=wave_band0, density_obs=spectrum_band0.flux, density_vega=vega_band0.flux, mag=mag)
    
    # Restriction of spectra to instrumental range + adjustment of spectra to the input magnitude
    if wave_instru is None:
        lmin_instru = 0.98*config_data["lambda_range"]["lambda_min"]                          # Lambda min [µm]
        lmax_instru = 1.02*config_data["lambda_range"]["lambda_max"]                          # Lambda max [µm] (a bit larger to avoid edge effects)
        wave_instru = get_wavelength_axis_constant_R(lmin=lmin_instru, lmax=lmax_instru, R=R) # Wavelength array on the instrumental bandwidth with constant sampling resolution
    spectrum_scaled       = spectrum.copy()                                                   # Making a copy of the input spetrum 
    spectrum_scaled.flux *= scale                                                             # Adjusting the spectrum to the input magnitude
    spectrum_density      = spectrum_scaled.interpolate_wavelength(wave_instru, renorm=False) # In order to have a spectrum in density (i.e. J/s/m2/µm)     

    # Conversion to [ph/bin/mn] (total ph over the FoV)
    spectrum_instru = spectrum_density.density_to_photons(config_data=config_data)
    
    return spectrum_instru, spectrum_density # [ph/bin/mn] (total ph over the FoV) and [J/s/m2/µm]



def get_spectrum_band(spectrum_instru, config_data=None, band=None, wave_band=None, R_output=None, degrade_resolution=True, verbose=False):
    """
    Extract and resample an input instrument spectrum onto the wavelength grid of a given band.

    The input spectrum is assumed to be provided in flux per spectral bin ('ph/bin/mn').
    Internally, it is first converted into a flux density ('ph/µm/mn') by dividing by the
    local wavelength bin width. The spectrum is then either:
      - degraded to the target wavelength grid and spectral resolution, or
      - simply interpolated onto the target wavelength grid.

    Finally, the flux density is converted back to flux per bin ('ph/bin/mn') on the
    output wavelength grid.

    Parameters
    ----------
    spectrum_instru : Spectrum
        Input spectrum sampled on the instrument wavelength grid, with flux expressed
        in 'ph/bin/mn'.

    config_data : dict, optional
        Instrument configuration dictionary. Required if 'wave_band' is not provided,
        since it is used together with 'band' to retrieve the wavelength grid.

    band : str, optional
        Name of the spectral band to extract. Must be a valid key of
        'config_data["gratings"]'. Used only if 'wave_band' is not provided.

    wave_band : ndarray, optional
        Target wavelength grid for the selected band, in µm. If provided, it takes
        precedence over 'band' and 'config_data'.

    R_output : float, optional
        Target spectral resolution used when 'degrade_resolution=True'. Passed to
        'spectrum_instru_density.degrade_resolution()'.

    degrade_resolution : bool, optional
        If True, degrade the input spectrum to the target wavelength grid and resolution.
        If False, only interpolate the spectrum onto 'wave_band'. Default is True.

    verbose : bool, optional
        If True, print additional information during the resolution degradation step.
        Default is False.

    Returns
    -------
    Spectrum
        Spectrum resampled on 'wave_band', with flux expressed in 'ph/bin/mn'.
    """
    if wave_band is None:
        if band not in config_data["gratings"]:
            raise KeyError(f"Band '{band}' is not defined in config_data['gratings']: {[band for band in config_data['gratings']]}.")
        wave_band = get_wave_band(config_data=config_data, band=band)
    
    spectrum_instru_density       = spectrum_instru.copy()                          # [ph/bin/mn]
    spectrum_instru_density.flux /= np.gradient(spectrum_instru_density.wavelength) # [ph/µm/mn]
    if degrade_resolution:
        spectrum_band = spectrum_instru_density.degrade_resolution(wave_band, renorm=False, R_output=R_output, verbose=verbose) # [ph/µm/mn]
    else:
        spectrum_band = spectrum_instru_density.interpolate_wavelength(wave_band, renorm=False)                                 # [ph/µm/mn]
    spectrum_band.flux *= np.gradient(wave_band) # [ph/bin/mn]

    return spectrum_band # [ph/bin/mn]



#######################################################################################################################
############################################# FastYield part: #########################################################
#######################################################################################################################

def get_spectrum_contribution_name_model(thermal_model, reflected_model):
    """
    Decide (string) labels for spectrum bookkeeping, given model choices.

    Returns
    -------
    spectrum_contributions: {"thermal", "reflected", "thermal+reflected"}
    name_model: str
        If both are active, the two are joined with '+'.
    """
    if thermal_model == "None" and reflected_model == "None":
        raise KeyError("PLEASE DEFINE A MODEL FOR THE THERMAL OR THE REFLECTED COMPONENT !")
    else:
        if thermal_model == "None":
            spectrum_contributions = "reflected"
        elif reflected_model == "None":
            spectrum_contributions = "thermal"
        else:
            spectrum_contributions = "thermal+reflected"
        name_model             = thermal_model+"+"+reflected_model
    return spectrum_contributions, name_model


def make_pretty_table(ax, params, bbox):
    tbl = ax.table(cellText=list(params.items()), colLabels=["Parameter", "Value"], cellLoc='left', colLoc='left', bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#bbbbbb')
        cell.set_linewidth(0.5)
    return tbl



def get_blackbody(wave, Teff):
    """
    Planck spectral radiance B_λ(λ, T) evaluated at wavelengths 'wave' (in µm).

    This returns the monochromatic *radiance* per unit wavelength (not multiplied
    by π). If you need the flux from a uniform disk, multiply by π and any
    dilution factor (e.g., (R / d)^2) downstream.

    Parameters
    ----------
    wave: array-like or float
        Wavelength(s) in microns (µm). Must be > 0.
    Teff: float
        Effective temperature in Kelvin. Must be > 0.

    Returns
    -------
    B_lambda: np.ndarray
        Spectral radiance in J / s / m2 / µm / sr (after unit conversion from SI),
        returned as a NumPy array with the same shape as 'wave'.

    """
    w  = wave * u.micron
    BB = (2 * const.h * const.c ** 2 / w ** 5) / np.expm1(const.h * const.c / (w * const.k_B * Teff * u.K))
    return BB.to(u.J / u.s / u.m**2 / u.micron).value



def get_auto_model(planet):
    """
    
    Parameters
    ----------
    planet : dict
        Planet/host properties (FastYield-style). 

    Returns
    -------
    None.

    """
    if "earth" in planet["PlanetType"].lower():
        if planet["HasAtmosphere"]:
            thermal_model   = "PSG_Earth-like"
            reflected_model = "tellurics"
        else:
            thermal_model   = "blackbody"
            reflected_model = "flat"
    else:
        thermal_model   = "BT-Settl"
        reflected_model = "PICASO"
    
    return thermal_model, reflected_model



def get_thermal_reflected_spectrum(planet, thermal_model="auto", reflected_model="auto", instru=None, wave_model=None, wave_K=None, counts_vega_K=None, show=True, in_im_mag=True, interpolated_spectrum=True):
    """
    Build the planet's *thermal* and *reflected* spectra (plus the host-star spectrum)
    over an instrument-like wavelength grid, optionally visualize the components,
    and return:
        (planet_total, planet_thermal, planet_reflected, star)

    Notes
    -----
    - All spectra are returned in *energy density* units (J/s/m2/µm), not photons.
    - Vega-based magnitudes in K band are used for renormalization.
    - If both 'thermal_model' and 'reflected_model' are "None", a clear error is raised.
    - No in-place modification is performed on input arguments.

    Parameters
    ----------
    planet: dict
        Planet/host properties (FastYield-style). Must contain the following keys
        (with '.value' where noted), and quantities must be convertible to floats:
        - "StarTeff".value, "StarLogg".value, "StarVsini".value,
          "StarKmag", "StarRadialVelocity".value,
          "StarRadius", "Distance"
        - "PlanetTeff".value, "PlanetLogg".value, "PlanetVsini".value,
          "PlanetRadialVelocity".value, "PlanetName", "PlanetType",
          "PlanetRadius", "SMA", "g_alpha"
        - Optional: "DiscoveryMethod" (string), "PlanetKmag(thermal+reflected)"
    thermal_model: str, optional
        Name of the thermal-emission model ("BT-Settl", "Exo-REM", ... or "None").
    reflected_model: str, optional
        Name of the reflected-light model ("PICASO", "flat", "tellurics", ... or "None").
    instru: str, optional
        Instrument name. If 'wave_model' is not provided, it will be derived
        from this instrument's 'lambda_range' with a very high sampling ('R0_max').
    wave_model: ndarray, optional
        Custom wavelength grid [µm] for the output spectra. If None, it is built from
        'instru' using high sampling between 0.98×λ_min and 1.02×λ_max.
    wave_K: ndarray, optional
        K-band wavelength grid [µm] for magnitude computations. If None, a dense grid
        is created using 'lmin_K'–'lmax_K' at R=R0_min.
    counts_vega_K: float, optional
        Vega counts on 'wave_K'. If None, it is computed.
    show: bool, optional
        If True, display diagnostic plots (components, blackbodies, contrast).
    in_im_mag: bool, optional
        If True and the discovery method is "Imaging" and a K magnitude for the planet
        is known, rescale the planet components to match that observed K magnitude.

    Returns
    -------
    planet_spectrum: Spectrum
        Planet total spectrum (thermal + reflected) on 'wave_model'.
    planet_thermal: Spectrum
        Thermal component on 'wave_model' (zeros if thermal_model == "None").
    planet_reflected: Spectrum
        Reflected component on 'wave_model' (zeros if reflected_model == "None").
    star_spectrum: Spectrum
        Star spectrum on 'wave_model' (renormalized to the input K magnitude,
        rotationally broadened, and Doppler-shifted).

    Raises
    ------
    KeyError
        If both 'thermal_model' and 'reflected_model' are "None", or if 'instru'
        is missing while 'wave_model' is None.
    ValueError
        If required inputs cannot be converted to finite floats.
    """
    
    # Sanity checks
    if thermal_model == "None" and reflected_model == "None":
        raise KeyError("Define at least one component: thermal_model or reflected_model must not be 'None'.")
    
    # Resolve auto models
    if thermal_model == "auto" or reflected_model == "auto":
        thermal_model_auto, reflected_model_auto = get_auto_model(planet)
        if thermal_model == "auto":
            thermal_model = thermal_model_auto
        if reflected_model == "auto":
            reflected_model = reflected_model_auto
    
    # Build K-band grid if needed (photometric only)
    if wave_K is None:
        wave_K = get_wave_K()
        
    # Build instrument grid if needed
    if wave_model is None:
        if instru is None:
            raise KeyError("Either provide 'wave_model' or an 'instru' name to derive it.")
        config_data = get_config_data(instru)
        # Model bandwidth
        lmin_instru = config_data["lambda_range"]["lambda_min"]   # [µm]
        lmax_instru = config_data["lambda_range"]["lambda_max"]   # [µm]
        lmin_model  = 0.98*lmin_instru                            # [µm]
        lmax_model  = 1.02*lmax_instru                            # [µm] (a bit larger than the instrumental bandwidth to avoid edge effects)
        R_instru    = get_R_instru(config_data=config_data)       # Max instrument resolution (factor 2 to be sure to not loose spectral information)
        R_model     = min(R_instru, R0_max)                       # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
        dl_model    = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
        wave_model  = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
    
    # Vega counts on K band
    if counts_vega_K is None:
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)
        counts_vega_K   = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)

    # Host star spectrum (load => resample => broaden => renormalize to K mag)
    T_star     = planet["StarTeff"].value  # [K]
    lg_star    = planet["StarLogg"].value  # [dex(cm/s2)]
    vrot_star  = planet["StarVrot"].value  # [km/s]
    vsini_star = planet["StarVsini"].value # [km/s]
    mag_star_K = planet["StarKmag"].value  # [no unit]
    star_spectrum_raw = load_star_spectrum(T_star=T_star, lg_star=lg_star)                 # Loading raw spectrum                                     [J/s/m2/µm]
    star_spectrum_K   = star_spectrum_raw.interpolate_wavelength(wave_K,     renorm=False) # Interpolating on K-band                                  [J/s/m2/µm]
    star_spectrum     = star_spectrum_raw.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model                              [J/s/m2/µm]
    star_spectrum_ref = star_spectrum.broad(vrot_star)                                     # Broadening the spectrum as seen from the planet (sini=1) [J/s/m2/µm]
    star_spectrum     = star_spectrum.broad(vsini_star)                                    # Broadening the spectrum as seenf from Earth              [J/s/m2/µm]
    
    # Vega-based scaling in K: (in the rest frame)
    scale_star_K            = get_scale_to_mag(wave=wave_K, density_obs=star_spectrum_K.flux, density_vega=None, counts_vega=counts_vega_K, mag=mag_star_K)
    star_spectrum_ref.flux *= scale_star_K # [J/s/m2/µm]
    star_spectrum.flux     *= scale_star_K # [J/s/m2/µm]
    star_spectrum_K.flux   *= scale_star_K # [J/s/m2/µm]
    
    # Thermal emission (energy density, dilution by (R/d)^2)
    T_planet        = planet["PlanetTeff"].value               # [K]
    lg_planet       = planet["PlanetLogg"].value               # [dex(cm/s2)]
    R_planet        = planet["PlanetRadius"]                   # [R_earth]
    distance        = planet["Distance"]                       # [pc]
    dilution_planet = (R_planet/distance).decompose().value**2 # [no unit]
    if thermal_model != "None":
        planet_thermal_raw     = load_planet_spectrum(T_planet=T_planet, lg_planet=lg_planet, model=thermal_model, interpolated_spectrum=interpolated_spectrum) # [J/s/m2/µm]
        planet_thermal_K       = planet_thermal_raw.interpolate_wavelength(wave_K,     renorm=False) # [J/s/m2/µm]
        planet_thermal         = planet_thermal_raw.interpolate_wavelength(wave_model, renorm=False) # [J/s/m2/µm]
        flux_planet_thermal_K  = planet_thermal_K.flux * dilution_planet # [J/s/m2/µm]
        flux_planet_thermal    = planet_thermal.flux   * dilution_planet # [J/s/m2/µm]
        R_planet_thermal       = planet_thermal.R
    else:
        flux_planet_thermal_K = np.zeros_like(wave_K)     # [J/s/m2/µm]
        flux_planet_thermal   = np.zeros_like(wave_model) # [J/s/m2/µm]
        R_planet_thermal      = np.zeros_like(wave_model) + R0_min
    planet_thermal_K = Spectrum(wavelength=wave_K,     flux=np.nan_to_num(flux_planet_thermal_K), R=R0_min,           T=T_planet, lg=lg_planet, model=thermal_model, rv=0, vsini=0, sigma=None)
    planet_thermal   = Spectrum(wavelength=wave_model, flux=np.nan_to_num(flux_planet_thermal),   R=R_planet_thermal, T=T_planet, lg=lg_planet, model=thermal_model, rv=0, vsini=0, sigma=None)
    
    # Reflected light (energy density): F_planet_reflected(λ) = F_star(λ) * A(λ) * g(α) * (Rp/SMA)^2
    airmass        = planet["TelluricEquivalentAirmass"].value # [no unit] airmass for tellurics model
    g_alpha        = planet["g_alpha"].value                   # [no unit] Lambert phase function
    SMA            = planet["SMA"]                             # [AU]
    scaling_planet = (R_planet/SMA).decompose().value**2       # [no unit]
    if reflected_model != "None":
        albedo_raw              = load_albedo_spectrum(T_planet=T_planet, lg_planet=lg_planet, model=reflected_model, airmass=airmass, interpolated_spectrum=interpolated_spectrum) # [no unit]
        albedo_K                = albedo_raw.interpolate_wavelength(wave_K,     renorm=False) # [no unit]
        albedo                  = albedo_raw.interpolate_wavelength(wave_model, renorm=False) # [no unit]
        flux_planet_reflected_K = star_spectrum_K.flux   * albedo_K.flux * g_alpha * scaling_planet # [J/s/m2/µm]
        flux_planet_reflected   = star_spectrum_ref.flux * albedo.flux   * g_alpha * scaling_planet # [J/s/m2/µm]
        R_planet_reflected      = np.fmax.reduce([star_spectrum_ref.R, albedo.R])
        reflected_model         = albedo_raw.model
    else:
        flux_planet_reflected_K = np.zeros_like(wave_K)
        flux_planet_reflected   = np.zeros_like(wave_model)
        R_planet_reflected      = np.zeros_like(wave_model) + R0_min
    planet_reflected_K = Spectrum(wavelength=wave_K,     flux=np.nan_to_num(flux_planet_reflected_K), R=R0_min,             T=T_planet, lg=lg_planet, model=reflected_model, rv=0, vsini=0, sigma=None)
    planet_reflected   = Spectrum(wavelength=wave_model, flux=np.nan_to_num(flux_planet_reflected),   R=R_planet_reflected, T=T_planet, lg=lg_planet, model=reflected_model, rv=0, vsini=0, sigma=None)
        
    # Rotational broadening of planet spectra as seen from Earth
    vsini_planet = planet["PlanetVsini"].value # [km/s]
    if thermal_model != "None":
        planet_thermal = planet_thermal.broad(vsini_planet)     # [J/s/m2/µm]
    if reflected_model != "None":
        planet_reflected = planet_reflected.broad(vsini_planet) # [J/s/m2/µm]
    
    # Doppler shifts
    rv_star       = planet["StarRadialVelocity"].value   # [km/s]
    rv_planet     = planet["PlanetRadialVelocity"].value # [km/s]
    star_spectrum = star_spectrum.doppler_shift(rv_star) # [J/s/m2/µm]
    if thermal_model != "None":
        planet_thermal = planet_thermal.doppler_shift(rv_planet)     # [J/s/m2/µm]
    if reflected_model != "None":
        planet_reflected = planet_reflected.doppler_shift(rv_planet) # [J/s/m2/µm]
    
    # Total planet spectrum (thermal + reflected)
    planet_spectrum_K = Spectrum(wavelength=wave_K,     flux=planet_thermal_K.flux + planet_reflected_K.flux, R=R0_min,                                                 T=T_planet, lg=lg_planet, model=thermal_model+"+"+reflected_model, rv=rv_planet, vsini=vsini_planet, sigma=None)
    planet_spectrum   = Spectrum(wavelength=wave_model, flux=planet_thermal.flux   + planet_reflected.flux,   R=np.fmax.reduce([planet_thermal.R, planet_reflected.R]), T=T_planet, lg=lg_planet, model=thermal_model+"+"+reflected_model, rv=rv_planet, vsini=vsini_planet, sigma=None)
    
    # Enforce observed K magnitude for directly imaged planets (if known and available)
    if in_im_mag and (planet["DiscoveryMethod"] == "Imaging") and (thermal_model != "None"): # Directly imaged planet magnitudes are measured mostly throught the thermal contribution
        mag_planet_K = planet["PlanetKmag(thermal+reflected)"].value
        if np.isfinite(mag_planet_K):
            scale_planet_K           = get_scale_to_mag(wave=wave_K, density_obs=planet_spectrum_K.flux, density_vega=None, counts_vega=counts_vega_K, mag=mag_planet_K)
            planet_spectrum.flux    *= scale_planet_K
            planet_thermal.flux     *= scale_planet_K
            planet_reflected.flux   *= scale_planet_K
            planet_spectrum_K.flux  *= scale_planet_K
            planet_thermal_K.flux   *= scale_planet_K
            planet_reflected_K.flux *= scale_planet_K
    
    # Visualization (optional)
    if show:
        
        # K-band magnitudes
        lmin_K, lmax_K = get_band_lims(band="K")
        mag_planet_total_K = get_mag(wave=wave_K, density_obs=planet_spectrum_K.flux, density_vega=None, counts_vega=counts_vega_K)
        if thermal_model != "None":
            mag_planet_thermal_K = get_mag(wave=wave_K, density_obs=planet_thermal_K.flux, density_vega=None, counts_vega=counts_vega_K)
        if reflected_model != "None":
            mag_planet_reflected_K = get_mag(wave=wave_K, density_obs=planet_reflected_K.flux, density_vega=None, counts_vega=counts_vega_K)
        
        # Blackbodies (for reference only)
        bb_star  = get_blackbody(wave_model, T_star) # [J/s/m2/µm]
        bb_star *= np.nanmean(star_spectrum.flux) / np.nanmean(bb_star)
        if thermal_model != "None":
            bb_planet_thermal  = get_blackbody(wave_model, T_planet) # [J/s/m2/µm]
            bb_planet_thermal *= np.nanmean(planet_thermal.flux) / np.nanmean(bb_planet_thermal)
        else:
            bb_planet_thermal = np.zeros_like(wave_model)
        if reflected_model != "None":
            bb_planet_reflected = bb_star * np.nanmean(planet_reflected.flux) / np.nanmean(bb_star) # [J/s/m2/µm]
        else:
            bb_planet_reflected = np.zeros_like(wave_model)
        # bb_planet = bb_planet_thermal + bb_planet_reflected
        
        # Plot
        leg_loc   = "lower center"
        lw        = 2
        alpha     = 0.7
        fig       = plt.figure(figsize=(13.5, 9.5), dpi=300, constrained_layout=False)
        gs        = gridspec.GridSpec(2, 2, width_ratios=[3.7, 1.3], height_ratios=[1, 1], wspace=0.1, hspace=0.2)
        # Star
        ax_star   = fig.add_subplot(gs[0, 0])
        ax_planet = fig.add_subplot(gs[1, 0], sharex=ax_star)
        ax_table  = fig.add_subplot(gs[:, 1])
        ax_table.axis("off")
        ax_star.plot(wave_model, star_spectrum.flux, c="crimson", lw=lw,   ls="-",  alpha=alpha, label=f"Star ({star_spectrum.model}, R={round(np.nanmedian(star_spectrum.R), -3):.0f}), K = {mag_star_K:.1f}")
        ax_star.plot(wave_model, bb_star,            c="black",   lw=lw/2, ls="--", alpha=1.0,   label="Blackbody (thermal)")
        if np.nanmax(wave_model) > lmin_K and np.nanmin(wave_model) < lmax_K: # Highlight K-band range if covered
            ax_star.axvspan(lmin_K, lmax_K, color="gray", alpha=0.2, lw=0)
        ax_star.set_yscale("log")
        ax_star.set_xlim(wave_model[0], wave_model[-1])
        ax_star.set_ylabel("Flux [J/s/µm/m2]", fontsize=14)
        ax_star.set_title("Star spectrum", fontsize=16, fontweight='bold', pad=10)
        ax_star.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_star.minorticks_on()
        ax_star.legend(loc=leg_loc, fontsize=12, frameon=True, edgecolor='0.3')
        ymin = np.nanmin(star_spectrum.flux) / 2
        ymax = np.nanmax(star_spectrum.flux) * 2
        ax_star.set_ylim(ymin, ymax)
        # Planet
        if thermal_model != "None":
            ax_planet.plot(wave_model, planet_thermal.flux,   c="crimson",   lw=lw,   ls="-",  alpha=alpha, label=f"Thermal ({planet_thermal.model.replace('_Earth-like', '(Earth-like)')}, R={round(np.nanmedian(planet_thermal.R), -3):.0f}), K = {mag_planet_thermal_K:.1f}")
        if reflected_model != "None":
            ax_planet.plot(wave_model, planet_reflected.flux, c="steelblue", lw=lw,   ls="-",  alpha=alpha, label=f"Reflected ({planet_reflected.model}, R={round(np.nanmedian(planet_reflected.R), -3):.0f}), K = {mag_planet_reflected_K:.1f}")
        if thermal_model != "None":
            ax_planet.plot(wave_model, bb_planet_thermal,     c="black",     lw=lw/2, ls="--", alpha=1.0,   label="Blackbody (thermal)")
        if reflected_model != "None":
            ax_planet.plot(wave_model, bb_planet_reflected,   c="black",     lw=lw/2, ls="-.", alpha=1.0,   label="Blackbody (reflected)")
        ax_planet.axvspan(lmin_K, lmax_K, color="gray", alpha=0.2, lw=0)
        ax_planet.set_yscale("log")
        ax_planet.set_xlabel("Wavelength [µm]", fontsize=14)
        ax_planet.set_ylabel("Flux [J/s/µm/m2]", fontsize=14)
        ax_planet.set_title("Planet spectrum", fontsize=16, fontweight='bold', pad=10)
        ax_planet.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_planet.minorticks_on()
        ax_planet.legend(loc=leg_loc, fontsize=12, frameon=True, edgecolor='0.3')
        ymin = max(np.nanmin(planet_spectrum.flux) / 2, np.nanmin(star_spectrum.flux) * 1e-10)
        ymax = np.nanmax(planet_spectrum.flux) * 2
        ax_planet.set_ylim(ymin, ymax)
        # Tables
        star_params = OrderedDict([
            (r"$R$",        f"{planet['StarRadius'].value:.1f} $R_\\odot$"),
            (r"$M$",        f"{planet['StarMass'].value:.1f} $M_\\odot$"),
            (r"$T_{eff}$",  f"{planet['StarTeff'].value:.0f} K"),
            (r"$\log g$",   f"{planet['StarLogg'].value:.2f} dex"),
            (r"$RV$",       f"{planet['StarRadialVelocity'].value:.1f} km/s"),
            (r"$V\sin i$",  f"{planet['StarVsini'].value:.1f} km/s"),
            ("Distance",    f"{planet['Distance'].value:.1f} pc"),
        ])
        planet_params = OrderedDict([
            (r"$R$",            f"{planet['PlanetRadius'].value:.1f} $R_\\oplus$"),
            (r"$M$",            f"{planet['PlanetMass'].value:.1f} $M_\\oplus$"),
            (r"$T_{eff}$",      f"{planet['PlanetTeff'].value:.0f} K"),
            (r"$\log g$",       f"{planet['PlanetLogg'].value:.2f} dex"),
            (r"$RV$",           f"{planet['PlanetRadialVelocity'].value:.1f} km/s"),
            (r"$V\sin i$",      f"{planet['PlanetVsini'].value:.1f} km/s"),
            ("$SMA$",           f"{planet['SMA'].value:.2f} AU"),
            (r"$sep$",          f"{planet['AngSep'].value:.0f} mas"),
        ])
        make_pretty_table(ax_table, star_params,   bbox=[0.05, 0.55, 0.9, 0.44])
        make_pretty_table(ax_table, planet_params, bbox=[0.05, 0.0, 0.9, 0.44])
        fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.08, hspace=0.28, wspace=0.15)
        fig.suptitle(f"{planet['PlanetName']} ({planet['PlanetType']}) modelisation", fontsize=20, fontweight='bold', y=1.0)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)        
        ax.set_xlabel("Wavelength [µm]", fontsize=16, labelpad=10)
        ax.set_ylabel("Flux (contrast unit)", fontsize=16, labelpad=10)
        ax.set_title(f"Final {planet['PlanetName']} modelized spectrum", fontsize=18, pad=15, fontweight="bold")        
        ax.set_yscale('log')
        ax.set_xlim(wave_model[0], wave_model[-1])
        ax.plot(wave_model, planet_spectrum.flux / star_spectrum.flux, color='seagreen', lw=lw, linestyle='-', alpha=alpha, label=f"Thermal+Reflected ({planet_spectrum.model}, R={round(np.nanmedian(planet_spectrum.R), -3):.0f}), K = {mag_planet_total_K:.1f}")        
        if np.nanmax(wave_model) > lmin_K and np.nanmin(wave_model) < lmax_K: # Highlight K-band range if covered
            ax.axvspan(lmin_K, lmax_K, color='gray', alpha=0.2, lw=0, label="K-band")        
        min_flux = np.nanmin(planet_spectrum.flux / star_spectrum.flux)
        max_flux = np.nanmax(planet_spectrum.flux / star_spectrum.flux)
        ax.set_ylim(max(1e-12, min_flux / 10), max_flux * 10)        
        ax.legend(fontsize=13, loc='upper right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='gray')        
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.minorticks_on()
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        ax2.set_ylabel(r'$\Delta$mag', fontsize=16, labelpad=20, rotation=270)
        ax2.tick_params(axis='y')
        ax2.minorticks_on() 
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))  
        plt.tight_layout()
        plt.show()

    return planet_spectrum, planet_thermal, planet_reflected, star_spectrum



#######################################################################################################################
###################################################### PICASO PART: ###################################################
#######################################################################################################################

def import_picaso():
    """
    Imports the picaso packages
    """
    try:
        os.environ['picaso_refdata'] = '/home/martoss/picaso/reference/'
        os.environ['PYSYN_CDBS']     = '/home/martoss/picaso/grp/redcat/trds/'
        import picaso
        from picaso import justdoit as jdi
        return picaso, jdi
    except ImportError as e:
        raise ImportError("picaso not installed / not importable") from e



def simulate_picaso_spectrum(planet, spectrum_contributions='thermal+reflected', planet_type="gas", clouds=True, stellar_mh=0.0122, opacity=None):
    '''
    TAKEN FROM: https://github.com/planetarysystemsimager/psisim/tree/main
    A function that returns the required inputs for picaso, given a row from a universe planet table. 
    
    Inputs:
    planet - a single row, corresponding to a single planet from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice", or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    stellar_mh - stellar metalicity
    Opacity class from justdoit.opannection
    NOTE: this assumes a planet phase of 0. You can change the phase in the resulting params object afterwards.
    '''
    
    picaso, jdi = import_picaso()
    
    # Retrieve star-planet configuration
    phase          =  0. # float(planet['Phase'].value) # in order to have the geometric Albedo (by definition)
    SMA            = float(planet['SMA'].value)
    R_star         = float(planet['StarRadius'].value)
    host_temp_list = np.hstack([np.arange(3500, 13000, 250), np.arange(13000, 50000, 1000)])
    host_logg_list = [5.00, 4.50, 4.00, 3.50, 3.00, 2.50, 2.00, 1.50, 1.00, 0.50, 0.0] # Define the grids that phoenix / ckmodel models like
    f_teff_grid    = interp1d(host_temp_list, host_temp_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    f_logg_grid    = interp1d(host_logg_list, host_logg_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    T_star         = f_teff_grid(float(planet['StarTeff'].to(u.K).value))
    lg_star        = f_logg_grid(float(planet['StarLogg'].to(u.dex(u.cm/ u.s**2)).value))
    T_planet       = float(planet['PlanetTeff'].value)
    #lg_planet = float(planet['PlanetLogg'].value)
    
    params = jdi.inputs()
    params.approx(raman='none') # see justdoit.py => class inputs():
    params.phase_angle(phase)
    params.gravity(gravity=float(planet['PlanetLogg'].value), gravity_unit=planet['PlanetLogg'].physical.unit) # NOTE: picaso gravity() won't use the "gravity" input if mass and radius are provided
    if lg_star > 5.0: # The current stellar models do not like log g > 5, so we'll force it here for now. 
        lg_star = 5.0
    if T_star < 3000: # The current stellar models do not like Teff < 3000, so we'll force it here for now. 
        T_star = 3000
    params.star(opacity, T_star, stellar_mh, lg_star, radius=R_star, semi_major=SMA, semi_major_unit=planet['SMA'].unit, radius_unit=planet['StarRadius'].unit) 
    if planet_type == 'gas': #-- Define atmosphere PT profile, mixing ratios, and clouds
        params.guillot_pt(T_planet, T_int=150, logg1=-0.5, logKir=-1) # T_int = Internal temperature / logg1, logKir = see parameterization Guillot 2010
        params.channon_grid_high() # get chemistry via chemical equillibrium
        if clouds: # may need to consider tweaking these for reflected light
            params.clouds(g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5]) # g0 = Asymmetry factor / w0 = Single Scattering Albedo / opd = Total Extinction in 'dp' / p = Bottom location of cloud deck (LOG10 bars) / dp = Total thickness cloud deck above p (LOG10 bars)
    elif planet_type == 'terrestrial':
        pass # TODO: add Terrestrial type
    elif planet_type == 'ice':
        pass # TODO: add ice type
    
    if phase == 0: # non-0 phases require special geometry which takes longer to run.
        df = params.spectrum(opacity, full_output=True, calculation=spectrum_contributions, plot_opacity=False) # Perform the simple simulation since 0-phase allows simple geometry
    else:
        df1 = params.spectrum(opacity, full_output=True, calculation='thermal') # Perform the thermal simulation as usual with simple geometry
        params.phase_angle(phase, num_tangle=8, num_gangle=8) # Apply the true phase and change geometry for the reflected simulation
        df2 = params.spectrum(opacity, full_output=True, calculation='reflected')
        df = df1.copy()
        df.update(df2) # Combine the output dfs into one df to be returned
        df['full_output_therm'] = df1.pop('full_output')
        df['full_output_ref']   = df2.pop('full_output')
    
    model_wvs  = 1./df['wavenumber'] * 1e4 *u.micron
    argsort    = np.argsort(model_wvs)
    model_wvs  = model_wvs[argsort]
    
    if spectrum_contributions == "thermal":
        planet_thermal    = np.zeros((2, len(model_wvs)))
        planet_thermal[0] = model_wvs
        thermal_flux      = df["thermal"][argsort] * u.erg/u.s/u.cm**2/u.cm
        thermal_flux      = thermal_flux.to(u.J/u.s/u.m**2/u.micron)
        planet_thermal[1] = np.array(thermal_flux.value)
        fits.writeto(f"{spectra_path}/planet_spectrum/PICASO/thermal_gas_giant_{round(float(planet['PlanetTeff'].value))}K_lg{round(float(planet['PlanetLogg'].value), 1)}.fits", planet_thermal, overwrite=True)
        plt.figure(dpi=300) ; plt.plot(planet_thermal[0], planet_thermal[1]) ; plt.title(f'Thermal: T = {round(float(planet["PlanetTeff"].value))}K and lg = {round(float(planet["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength [µm]') ; plt.ylabel("flux (in J/s/µm/m2)") ; plt.yscale('log') ; plt.show()
    
    elif spectrum_contributions == "reflected":
        albedo    = np.zeros((2, len(model_wvs)))
        albedo[0] = model_wvs
        albedo[1] = df['albedo'][argsort]
        fits.writeto(f"{spectra_path}/planet_spectrum/albedo/PICASO/albedo_gas_giant_{round(float(planet['PlanetTeff'].value))}K_lg{round(float(planet['PlanetLogg'].value), 1)}.fits", albedo, overwrite=True)
        plt.figure(dpi=300) ; plt.plot(albedo[0], albedo[1]) ; plt.title(f'Albedo: T = {round(float(planet["PlanetTeff"].value))}K and lg = {round(float(planet["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength [µm]') ; plt.ylabel("albedo") ; plt.yscale('log') ; plt.show()

    

def get_picasso_thermal():
    from src.FastYield import load_planet_table, get_planet_index
    picaso, jdi = import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname         = 'all_opacities_0.6_6_R60000.db'
    dbname         = os.path.join(opacity_folder, dbname)
    opacity        = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx          = get_planet_index(planet_table, "HR 8799 b") # "HR 8799 b" => does not change anything
    planet       = planet_table[idx]
    T0, lg0 = get_model_grid("PICASO")
    for i in tqdm(range(len(T0))):
        T_planet = T0[i]
        for lg_planet in lg0:
            planet["PlanetTeff"]  = T_planet  * planet["PlanetTeff"].unit  # 
            planet["PlanetLogg"] = lg_planet * planet["PlanetLogg"].unit # 
            simulate_picaso_spectrum(planet, spectrum_contributions="thermal", opacity=opacity)


            
def get_picasso_albedo():
    from src.FastYield import load_planet_table, get_planet_index
    picaso, jdi = import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname         = 'all_opacities_0.6_6_R60000.db'
    dbname         = os.path.join(opacity_folder, dbname)
    opacity        = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx          = get_planet_index(planet_table, "HR 8799 b") # "HR 8799 b" => does not change anything
    planet       = planet_table[idx]
    T0, lg0 = get_model_grid("PICASO")
    for i in tqdm(range(len(T0))):
        T_planet = T0[i]
        for lg_planet in lg0:
            planet["PlanetTeff"]  = T_planet  * planet["PlanetTeff"].unit  # 
            planet["PlanetLogg"] = lg_planet * planet["PlanetLogg"].unit # 
            simulate_picaso_spectrum(planet, spectrum_contributions="reflected", opacity=opacity)
            

