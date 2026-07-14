import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["NUMBA_NUM_THREADS"]    = "1"

# import FastYield modules
from fastyield.config import rad2arcsec, h, c, R0_min, R0_max, lmin_bands, lmax_bands, archive_path, simulated_path, get_sim_data_path
from fastyield.utils import plot_trans_tell_tel, plot_bkg_skycalc
from fastyield.get_specs import load_tell_trans, get_detector_specs
from fastyield.FastYield import load_planet_table, get_filename_table, get_mask_planet_type, yield_population_plot, yield_heatmap_ELT
from fastyield.FastYield_helpers import print_simulation_summary, make_suffix, write_meta, memmap_nbytes, format_nbytes, create_memmap_with_log
from fastyield.spectrum import get_wavelength_axis_constant_R, Spectrum, load_vega_spectrum, get_spectrum_contribution_name_model, get_thermal_reflected_spectrum, filtered_flux, get_wave_K, get_counts_from_density
from fastyield.signal_noise import compress_h_for_sigma_base_2, compute_sigma_base_2_speck_numba, compute_sigma_base_2_al_spat_numba, compute_sigma_base_2_al_spec_fast

# import astropy modules
from astropy.io import fits

# import numpy modules
import numpy as np

# import matplotlib modules
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
import matplotlib as mpl

# import scipy modules
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logit, expit

# import other modules
from tqdm import tqdm
from multiprocessing import cpu_count
import multiprocessing as mp
from pathlib import Path
import sys
import gc

# Parameters for multiprocessing
nproc     = max(cpu_count()-3, 1)
chunksize = 8



# TODO :
# Required if FASTYIELD_SIM_DATA_PATH is not defined:
# set the path to your local FastYield sim_data directory.
# This directory should contain the instrumental/simulation data folders
# and the Spectra/ directory.
# set_sim_data_path("/path/to/sim_data")



# -------------------------------------------------------------
# Functions for IFU and imager for main quantities computations
# -------------------------------------------------------------

# --- General helpers ---
def interp_flat_last_axis(x, y, x0, dtype=None):
    """
    Fast linear interpolation along the last axis, with flat extrapolation.
    This is optimized for quantities sampled as (..., Nx), for example
    PSF_profile_5D[..., separation].
    """
    x0  = np.clip(x0, x[0], x[-1])
    i1  = int(np.searchsorted(x, x0, side="right"))
    i1  = int(np.clip(i1, 1, x.size - 1))
    i0  = i1 - 1
    w   = (x0 - x[i0]) / (x[i1] - x[i0])
    out = (1.0 - w) * y[..., i0] + w * y[..., i1]
    return out.astype(dtype, copy=False) if dtype is not None else out

def interp_mag_l0_flat_4d(mag_grid, y_4d, mag_l0, dtype=None):
    """
    Fast linear interpolation along the stellar-magnitude axis for PCS PSF data.

    Parameters
    ----------
    mag_grid : ndarray, shape (Nmag,)
        Stellar magnitude grid.
    y_4d : ndarray, shape (N_l0, N_WFE, N_IWA, Nmag)
        Quantity sampled on the stellar-magnitude grid.
    mag_l0 : ndarray, shape (N_l0,)
        Stellar magnitude evaluated independently at each l0.
    dtype : dtype or None
        Optional output dtype.

    Returns
    -------
    out : ndarray, shape (N_l0, N_WFE, N_IWA)
        Interpolated quantity.
    """
    finite = np.isfinite(mag_l0)
    x0     = np.clip(np.where(finite, mag_l0, mag_grid[0]), mag_grid[0], mag_grid[-1])
    i1     = np.searchsorted(mag_grid, x0, side="right")
    i1     = np.clip(i1, 1, mag_grid.size - 1)
    i0     = i1 - 1
    w      = (x0 - mag_grid[i0]) / (mag_grid[i1] - mag_grid[i0])
    j      = np.arange(y_4d.shape[0])
    v0     = y_4d[j, :, :, i0]
    v1     = y_4d[j, :, :, i1]
    out    = (1.0 - w[:, None, None]) * v0 + w[:, None, None] * v1
    if np.any(~finite):
        out[~finite] = np.nan
    return out.astype(dtype, copy=False) if dtype is not None else out

def normalize_template_or_none(template):
    """
    Normalize a matched-filter template.

    Returns
    -------
    template_norm : ndarray or None
        Normalized template if its norm is finite and strictly positive.
        None otherwise.
    """
    template = np.asarray(template, dtype=float)
    norm2    = np.nansum(template**2)
    if (not np.isfinite(norm2)) or (norm2 <= 0):
        return None
    return template / np.sqrt(norm2)

def window_sums_1d(values, idx_lo, idx_hi):
    """
    Fast sums of a 1D array over many [idx_lo, idx_hi) windows.

    Parameters
    ----------
    values : ndarray, shape (Nwave,)
        1D array to integrate.
    idx_lo, idx_hi : ndarray
        Integer arrays with the same shape, defining integration windows (l0, Nl) for IFU or (l0, Dl) for imager.

    Returns
    -------
    sums : ndarray
        Sum of values[idx_lo:idx_hi] for each window.
    """
    values = np.asarray(values)
    if np.all(np.isfinite(values)):
        v = values
    else:
        v = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    cs    = np.empty(v.size + 1, dtype=v.dtype if np.issubdtype(v.dtype, np.floating) else float)
    cs[0] = 0.0
    np.cumsum(v, out=cs[1:])
    return cs[idx_hi] - cs[idx_lo]

def window_nanmax_1d(values, idx_lo, idx_hi):
    """
    Fast maximum of a 1D array over many [idx_lo, idx_hi) windows.
    """
    values = np.asarray(values, dtype=float)
    out    = np.zeros(np.shape(idx_lo), dtype=float)
    for ind in np.ndindex(out.shape):
        lo = int(idx_lo[ind])
        hi = int(idx_hi[ind])
        if hi <= lo:
            out[ind] = 0.0
            continue
        v = values[lo:hi]
        m = np.isfinite(v)
        if np.any(m):
            out[ind] = np.nanmax(v[m])
        else:
            out[ind] = 0.0
    return out

def warmup_signal_noise_numba(dtype):
    """
    Compile/load the Numba covariance kernels before multiprocessing.

    On Linux, the compiled functions are inherited by forked workers, which avoids
    paying the compilation/loading overhead independently in every worker.
    """
    wave = np.linspace(1.0, 1.1, 32, dtype=dtype)
    h    = np.ones((32, 1), dtype=dtype)
    sep  = np.array([50.0 / (rad2arcsec * 1000.0)], dtype=dtype)
    _    = compute_sigma_base_2_speck_numba(h=h, wave=wave, separation_rad=sep, D=38.54)
    _    = compute_sigma_base_2_al_spat_numba(h=h, wave=wave, separation_rad=sep, pxscale_rad=1.0 / (rad2arcsec * 1000.0))
    _    = compute_sigma_base_2_al_spec_fast(h=h)



# --- Globals (context) used by multiprocessing workers ---
_IM_CTX  = None
_IFU_CTX = None
_SNR_CTX = None

def init_worker_IM(ctx):
    global _IM_CTX
    _IM_CTX = ctx

    # Temp PSF memmaps opened inside each worker
    _IM_CTX["PSF_profile_5D"]     = np.load(_IM_CTX["PSF_profile_5D_path"],     mmap_mode="r")
    _IM_CTX["fraction_core_5D"]   = np.load(_IM_CTX["fraction_core_5D_path"],   mmap_mode="r")
    _IM_CTX["PSF_profile_max_4D"] = np.load(_IM_CTX["PSF_profile_max_4D_path"], mmap_mode="r")

    # Output memmaps opened inside each worker
    _IM_CTX["signal_out"]       = np.load(_IM_CTX["signal_path"],       mmap_mode="r+")
    _IM_CTX["sigma_halo_2_out"] = np.load(_IM_CTX["sigma_halo_2_path"], mmap_mode="r+")
    _IM_CTX["sigma_bkg_2_out"]  = np.load(_IM_CTX["sigma_bkg_2_path"],  mmap_mode="r+")
    _IM_CTX["DIT_out"]          = np.load(_IM_CTX["DIT_path"],          mmap_mode="r+")

def init_worker_IFU(ctx):
    global _IFU_CTX
    _IFU_CTX = ctx

    # Temp PSF memmaps opened inside each worker
    _IFU_CTX["PSF_profile_5D"]     = np.load(_IFU_CTX["PSF_profile_5D_path"],     mmap_mode="r")
    _IFU_CTX["fraction_core_5D"]   = np.load(_IFU_CTX["fraction_core_5D_path"],   mmap_mode="r")
    _IFU_CTX["PSF_profile_max_4D"] = np.load(_IFU_CTX["PSF_profile_max_4D_path"], mmap_mode="r")

    # Output memmaps opened inside each worker
    _IFU_CTX["signal_out"]            = np.load(_IFU_CTX["signal_path"],            mmap_mode="r+")
    _IFU_CTX["sigma_halo_2_out"]      = np.load(_IFU_CTX["sigma_halo_2_path"],      mmap_mode="r+")
    _IFU_CTX["sigma_bkg_2_out"]       = np.load(_IFU_CTX["sigma_bkg_2_path"],       mmap_mode="r+")
    _IFU_CTX["DIT_out"]               = np.load(_IFU_CTX["DIT_path"],               mmap_mode="r+")
    _IFU_CTX["sigma_syst_base_2_out"] = np.load(_IFU_CTX["sigma_syst_base_2_path"], mmap_mode="r+")

def init_worker_SNR(ctx):
    global _SNR_CTX
    _SNR_CTX = ctx

    # Re-open input memmaps inside each worker (portable: fork/spawn)
    _SNR_CTX["signal_planets"]       = np.load(_SNR_CTX["signal_path"],       mmap_mode="r")
    _SNR_CTX["sigma_halo_2_planets"] = np.load(_SNR_CTX["sigma_halo_2_path"], mmap_mode="r")
    _SNR_CTX["sigma_bkg_2_planets"]  = np.load(_SNR_CTX["sigma_bkg_2_path"],  mmap_mode="r")
    _SNR_CTX["DIT_planets"]          = np.load(_SNR_CTX["DIT_path"],          mmap_mode="r")
    if _SNR_CTX["instru_type"] == "IFU":
        _SNR_CTX["sigma_syst_base_2_planets"] = np.load(_SNR_CTX["sigma_syst_base_2_path"], mmap_mode="r")
    else:
        _SNR_CTX["sigma_syst_base_2_planets"] = None

    # Output memmap, opened in read/write mode
    _SNR_CTX["SNR_planets"] = np.lib.format.open_memmap(_SNR_CTX["SNR_path"], mode="r+", dtype=np.dtype(_SNR_CTX["dtype"]), shape=tuple(_SNR_CTX["shape_SNR"]),)



# --- IFU function ---
def process_IFU(idx):

    # --- Context ---
    planet_table           = _IFU_CTX["planet_table"]
    l0                     = _IFU_CTX["l0"]
    A_FWHM                 = _IFU_CTX["A_FWHM"]
    separation             = _IFU_CTX["separation"]
    thermal_model          = _IFU_CTX["thermal_model"]
    reflected_model        = _IFU_CTX["reflected_model"]
    post_processing        = _IFU_CTX["post_processing"]
    wave_model             = _IFU_CTX["wave_model"]
    wave_instru            = _IFU_CTX["wave_instru"]
    wave_K                 = _IFU_CTX["wave_K"]
    counts_vega_K          = _IFU_CTX["counts_vega_K"]
    vega_flux_1D           = _IFU_CTX["vega_flux_1D"]
    mag_star               = _IFU_CTX["mag_star"]
    scale_spectrum         = _IFU_CTX["scale_spectrum"]
    N_R                    = _IFU_CTX["N_R"]
    N_l0                   = _IFU_CTX["N_l0"]
    N_Nl                   = _IFU_CTX["N_Nl"]
    R                      = _IFU_CTX["R"]
    wave_1D_list           = _IFU_CTX["wave_1D_list"]
    dwave_1D_list          = _IFU_CTX["dwave_1D_list"]
    trans_tell_tel_1D_list = _IFU_CTX["trans_tell_tel_1D_list"]
    trans_instru           = _IFU_CTX["trans_instru"]
    background_1D_list     = _IFU_CTX["background_1D_list"]
    pxscales               = _IFU_CTX["pxscales"]
    Rc                     = _IFU_CTX["Rc"]
    filter_type            = _IFU_CTX["filter_type"]
    range_1D_list          = _IFU_CTX["range_1D_list"]
    saturation_e           = _IFU_CTX["saturation_e"]
    min_DIT                = _IFU_CTX["min_DIT"]
    max_DIT                = _IFU_CTX["max_DIT"]
    PSF_profile_5D         = _IFU_CTX["PSF_profile_5D"]
    fraction_core_5D       = _IFU_CTX["fraction_core_5D"]
    PSF_profile_max_4D     = _IFU_CTX["PSF_profile_max_4D"]
    dtype                  = _IFU_CTX["dtype"]
    D                      = _IFU_CTX["D"]
    pxscales_rad           = _IFU_CTX["pxscales_rad"]

    # --- Planet row ---
    planet                    = planet_table[idx]
    separation_planet         = float(planet["AngSep"].value)           # [mas]
    separation_planet_rad     = separation_planet / (rad2arcsec * 1000) # [rad]
    separation_planet_rad_arr = np.array([separation_planet_rad], dtype=dtype)

    # --- Computing the planet and star models for this planet row in [J/s/m²/µm] ---
    planet_spectrum, _, _, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True, interpolated_spectrum=True, apply_kinematics=True)
    # Interpolating on wave_instru (constant sampling resolution wavelength axis)
    star_spectrum   = star_spectrum.interpolate_wavelength(wave_instru,   renorm=False) # [J/s/m2/µm]
    planet_spectrum = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # [J/s/m2/µm]


    # --- Computing mag_star_1D for each l0 ---
    star_flux_1D = np.interp(l0, star_spectrum.wavelength, star_spectrum.flux, left=np.nan, right=np.nan) # (l0) [J/s/m2/µm]
    mag_star_1D  = -2.5 * np.log10(star_flux_1D / vega_flux_1D)                                           # (l0) [no unit]


    # --- PSF data linearly interpolated at the planet's separation and stellar magnitude ---
    # PSF_profile_5D     : (l0, WFE, IWA, mag_star, sep)
    # fraction_core_5D   : (l0, WFE, IWA, mag_star, sep)
    # PSF_profile_max_4D : (l0, WFE, IWA, mag_star)
    #
    # For speed, we first interpolate along separation, because separation_planet is
    # a scalar for this planet. This avoids creating full (l0, WFE, IWA, sep) arrays.
    PSF_profile_mag_4D   = interp_flat_last_axis(separation, PSF_profile_5D,   separation_planet, dtype=dtype) # (l0, WFE, IWA, mag_star)
    fraction_core_mag_4D = interp_flat_last_axis(separation, fraction_core_5D, separation_planet, dtype=dtype) # (l0, WFE, IWA, mag_star)

    # Then interpolate along stellar magnitude. mag_star_1D depends on l0, so this
    # specialized helper interpolates each l0 slice at its own magnitude.
    PSF_profile_sep_3D   = interp_mag_l0_flat_4d(mag_star, PSF_profile_mag_4D,   mag_star_1D, dtype=dtype) # (l0, WFE, IWA) [star flux fraction/px]
    fraction_core_sep_3D = interp_mag_l0_flat_4d(mag_star, fraction_core_mag_4D, mag_star_1D, dtype=dtype) # (l0, WFE, IWA) [planet flux fraction/FWHM]
    PSF_profile_max_3D   = interp_mag_l0_flat_4d(mag_star, PSF_profile_max_4D,   mag_star_1D, dtype=dtype) # (l0, WFE, IWA) [max star flux fraction/px]


    # --- Converting in [ph/mn/µm] ---
    star_spectrum.flux   *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]
    planet_spectrum.flux *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]


    # --- Computing data depending only on R, l0 and Nl ---
    # These quantities do not yet include the spatial PSF factors or
    # the instrumental throughput axis.  They are pure spectral quantities,
    # computed once for each (R, l0, Nl) combination and later broadcast over
    # (C_raw, IWA_lD, trans_instru).
    signal_3D            = np.zeros((N_R, N_l0, N_Nl), dtype=dtype) # = alpha - beta                         (R, l0, Nl) [ph/mn]
    sigma_halo_2_3D      = np.zeros((N_R, N_l0, N_Nl), dtype=dtype) # = sum (trans * S_star * template**2)   (R, l0, Nl) [ph/mn]
    max_star_flux_3D     = np.zeros((N_R, N_l0, N_Nl), dtype=dtype) # = max (trans * S_star)                 (R, l0, Nl) [ph/mn/bin]
    sigma_bkg_2_3D       = np.zeros((N_R, N_l0, N_Nl), dtype=dtype) # = sum (background * template**2)       (R, l0, Nl) [ph/px/mn]
    sigma_syst_base_2_3D = np.zeros((N_R, N_l0, N_Nl), dtype=dtype) # = sigma_syst,base**2                   (R, l0, Nl) [ph/mn]**2
    for i, res in enumerate(R):
        wave_i           = wave_1D_list[i]
        dwave_i          = dwave_1D_list[i]
        trans_tell_tel_i = trans_tell_tel_1D_list[i]
        star_i           = star_spectrum.degrade_resolution(wave_i,   renorm=False, R_output=res).flux # [ph/mn/µm]
        planet_i         = planet_spectrum.degrade_resolution(wave_i, renorm=False, R_output=res).flux # [ph/mn/µm]
        idx_lo, idx_hi   = range_1D_list[i] # (l0, Nl)

        # Converting in [ph/mn/bin]
        # The degraded spectra are still densities [ph/mn/um]; multiplying by
        # the spectral-bin width gives the photon counts per spectral channel.
        star_i   *= dwave_i # [ph/mn/um] => [ph/mn/bin]
        planet_i *= dwave_i # [ph/mn/um] => [ph/mn/bin]

        # HF/LF split for molecular mapping.
        # For MM, the useful signal is the high-frequency planetary component,
        # while beta accounts for the self-subtraction induced by the LF planet
        # term multiplying the HF stellar spectrum.
        if post_processing == "MM":
            star_HF_i, star_LF_i     = filtered_flux(flux=star_i,   R=res, Rc=Rc, filter_type=filter_type) # [ph/mn/bin]
            planet_HF_i, planet_LF_i = filtered_flux(flux=planet_i, R=res, Rc=Rc, filter_type=filter_type) # [ph/mn/bin]

        # Telescope-transmitted stellar spectrum. This quantity is used for the
        # detector-saturation estimate, stellar-halo photon noise, and systematic
        # covariance weights.
        trans_tell_tel_star_i = trans_tell_tel_i * star_i  # [ph/mn/bin]

        # Brightest stellar spectral-bin flux in each spectral window.
        # This is needed even when the planet template is null, because it sets the
        # detector saturation limit.
        max_star_flux_3D[i] = window_nanmax_1d(values=trans_tell_tel_star_i, idx_lo=idx_lo, idx_hi=idx_hi).astype(dtype)

        if post_processing == "MM":
            # Raw matched-filter template before window-by-window normalization (T = trans*planet_HF):
            #     T_i = trans_tell_tel_i * planet_HF_i
            # The actual normalized template in each spectral window is (t = T/norm):
            #     t_ijk = T_i[sl] / sqrt(sum_sl(T_i**2))
            T_i               = trans_tell_tel_i * planet_HF_i                            # (wave_res)
            T2_i              = T_i**2                                                    # (wave_res)
            norm2_2D          = window_sums_1d(values=T2_i, idx_lo=idx_lo, idx_hi=idx_hi) # (l0, Nl)
            valid_2D          = np.isfinite(norm2_2D) & (norm2_2D > 0)                    # (l0, Nl)
            norm_2D           = np.zeros_like(norm2_2D, dtype=dtype)                      # (l0, Nl)
            norm_2D[valid_2D] = np.sqrt(norm2_2D[valid_2D])                               # (l0, Nl)

            # CCF signal computation in [ph/mn], before multiplying by
            # the spatial PSF throughput and instrumental efficiency.
            #   alpha = retained HF planet signal                          = sum(trans*planet_HF * t)                 = sum(T*t) = sum(T*T/norm) = norm**2/norm = norm
            #   beta = self-subtraction from stellar/planet LF-HF coupling = sum(trans*star_HF*planet_LF/star_LF * t) = sum(trans*star_HF*planet_LF/star_LF * T) / norm
            alpha_2D               = norm_2D
            ratio_planet_star_LF_i = np.divide(planet_LF_i, star_LF_i, out=np.zeros_like(planet_LF_i), where=np.isfinite(star_LF_i) & (star_LF_i != 0))
            beta_num_2D            = window_sums_1d(values=trans_tell_tel_i * star_HF_i * ratio_planet_star_LF_i * T_i, idx_lo=idx_lo, idx_hi=idx_hi)
            signal_2D              = np.zeros_like(norm2_2D, dtype=float)
            signal_2D[valid_2D]    = alpha_2D[valid_2D] - beta_num_2D[valid_2D]/norm_2D[valid_2D]

        elif post_processing == "DI":
            # Raw DI template before window-by-window normalization (T = trans*planet):
            #     T_i = trans_tell_tel_i * planet_i
            # The actual normalized template in each spectral window is (t = T/norm):
            #     t_ijk = T_i[sl] / sqrt(sum_sl(T_i**2))
            T_i               = trans_tell_tel_i * planet_i                               # (wave_res)
            T2_i              = T_i**2                                                    # (wave_res)
            norm2_2D          = window_sums_1d(values=T2_i, idx_lo=idx_lo, idx_hi=idx_hi) # (l0, Nl)
            valid_2D          = np.isfinite(norm2_2D) & (norm2_2D > 0)                    # (l0, Nl)
            norm_2D           = np.zeros_like(norm2_2D, dtype=float)                      # (l0, Nl)
            norm_2D[valid_2D] = np.sqrt(norm2_2D[valid_2D])                               # (l0, Nl)

            # CCF/integrated signal computation in [ph/mn], before PSF and throughput factors.
            # delta = sum(trans*planet*t) = sum(T*t) = sum(T*T/norm) = norm**2/norm = norm
            signal_2D           = np.zeros_like(norm2_2D, dtype=float)
            signal_2D[valid_2D] = norm_2D[valid_2D]

        else:
            raise ValueError("post_processing must be 'MM' or 'DI'.")

        # Save signal in [ph/mn] before spatial PSF and instrumental-throughput factors.
        signal_3D[i] = signal_2D.astype(dtype)

        # Stellar halo photon noise in [ph/mn]**2:
        # sigma_halo^2 = sum(trans*star * t**2)
        #              = sum(trans*star * T**2) / norm**2
        sigma_halo_num_2D       = window_sums_1d(values=trans_tell_tel_star_i * T2_i, idx_lo=idx_lo, idx_hi=idx_hi)
        sigma_halo_2D           = np.zeros_like(norm2_2D, dtype=float)
        sigma_halo_2D[valid_2D] = sigma_halo_num_2D[valid_2D] / norm2_2D[valid_2D]
        sigma_halo_2_3D[i]      = sigma_halo_2D.astype(dtype)

        # Background photon noise in [ph/px/mn]**2
        background_i           = background_1D_list[i]                                                    # (wave_res) [ph/mn/bin/mas2]
        sigma_bkg_num_2D       = window_sums_1d(values=background_i * T2_i, idx_lo=idx_lo, idx_hi=idx_hi) # (l0, Nl)   [ph/mn/mas2]
        sigma_bkg_num_2D      *= pxscales[:, None]**2                                                     # (l0, Nl)   [ph/mn/mas2] => [ph/px/mn]
        sigma_bkg_2D           = np.zeros_like(norm2_2D, dtype=float)
        sigma_bkg_2D[valid_2D] = sigma_bkg_num_2D[valid_2D] / norm2_2D[valid_2D]
        sigma_bkg_2_3D[i]      = sigma_bkg_2D.astype(dtype)

        # Systematic base term in [ph/mn]**2
        # This part is harder to vectorize cleanly because the covariance calculation
        # depends on the compressed wavelength grid, the local pxscale, and the exact
        # normalized h_PP vector in each window. We therefore keep the small (j, k)
        # loop only for sigma_syst_base_2.
        for j in range(N_l0):
            pxscale_rad = pxscales_rad[j]
            for k in range(N_Nl):
                if not valid_2D[j, k]:
                    sigma_syst_base_2_3D[i, j, k] = 0.0
                    continue
                sl       = slice(idx_lo[j, k], idx_hi[j, k])
                wave_ijk = wave_i[sl]

                # Window-normalized template:
                #     template_ijk = template_raw / sqrt(sum(template_raw**2)).

                # Sigma stellar-halo systematic noise base term, assuming
                # sigma_m = 1.  This is the spectral covariance projection before
                # multiplying by the spatial halo intensity.
                # This implements:
                #   sigma_m,base^2 = h^T C_m h
                # with, for MM:
                #   C_m = 0.5 * (C_al_spec + C_al_spat).
                # Therefore:
                #   sigma_m,base^2 = 0.5 * (h^T C_al_spec h + h^T C_al_spat h).
                h_PP_native                       = (trans_tell_tel_star_i[sl] * T_i[sl] / norm_2D[j, k])[:, None] # (wave, sep) weighted halo H*t
                wave_c, h_PP_c, _                 = compress_h_for_sigma_base_2(wave=wave_ijk, h=h_PP_native, post_processing=post_processing, D=D, pxscale=pxscale_rad, separation_ref=separation_planet_rad, sep_unit="rad")
                if post_processing == "DI":
                    sigma_syst_base_2_3D[i, j, k] = compute_sigma_base_2_speck_numba(h=h_PP_c, wave=wave_c, separation_rad=separation_planet_rad_arr, D=D)[0]
                elif post_processing == "MM":
                    sigma_base_2_al_spat          = compute_sigma_base_2_al_spat_numba(h=h_PP_c, wave=wave_c, separation_rad=separation_planet_rad_arr, pxscale_rad=pxscale_rad)[0]
                    sigma_base_2_al_spec          = compute_sigma_base_2_al_spec_fast(h=h_PP_native)[0]
                    sigma_syst_base_2_3D[i, j, k] = 0.5 * (sigma_base_2_al_spec + sigma_base_2_al_spat)


    # --- Computing DIT length in [mn/DIT] (R, l0, Nl, WFE, IWA, TI) ---

    # max_star_flux in [ph/px/mn/bin] in the brightest pixel and bin (through coronagraph, if any):
    # [total star ph/mn/bin] in the brightest bin * [star flux fraction/px] in the brightest pixel => [ph/px/mn/bin] in the brightest pixel and bin
    # (R, l0, Nl, 1, 1)                           * (1, l0, 1, WFE, IWA)                           => (R, l0, Nl, WFE, IWA)
    max_star_flux_5D = max_star_flux_3D[:, :, :, None, None] * PSF_profile_max_3D[None, :, None, :, :]

    # max_star_flux in [e-/px/mn/bin] in the brightest pixel with instrumental transmission (through coronagraph, if any):
    # [ph/px/mn/bin] in the brightest pixel and bin * [e-/ph]             => [e-/px/mn/bin] in the brightest pixel and bin
    # (R, l0, Nl, WFE, IWA, 1)                      * (1, 1, 1, 1, 1, TI) => (R, l0, Nl, WFE, IWA, TI)
    max_star_flux_6D = max_star_flux_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]

    # Computing DIT length in [mn/DIT] (+ clipping between min and max DIT)
    saturation_e_6D = saturation_e[None, :, None, None, None, None]
    min_DIT_6D      = min_DIT[None, :, None, None, None, None]
    max_DIT_6D      = max_DIT[None, :, None, None, None, None]
    DIT_6D          = saturation_e_6D / max_star_flux_6D # (R, l0, Nl, WFE, IWA, TI) [mn/DIT]
    DIT_6D          = np.maximum(DIT_6D, min_DIT_6D)     # (R, l0, Nl, WFE, IWA, TI) [mn/DIT]
    DIT_6D          = np.minimum(DIT_6D, max_DIT_6D)     # (R, l0, Nl, WFE, IWA, TI) [mn/DIT]


    # --- Computing signal in [e-/FWHM/DIT] (R, l0, Nl, WFE, IWA, TI) ---

    # Signal at the planet's separation inside the FWHM in [ph/FWHM/mn] (throught coronagraph, if any)
    # [total planet ph/mn] * [planet flux fraction/FWHM] => [ph/FWHM/mn]
    # (R, l0, Nl, 1, 1)    * (1, l0, 1, WFE, IWA)        => (R, l0, Nl, WFE, IWA)
    signal_5D = signal_3D[:, :, :, None, None] * fraction_core_sep_3D[None, :, None, :, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/FWHM/mn]             * [e-/ph]             => [e-/FWHM/mn]
    # (R, l0, Nl, WFE, IWA, 1) * (1, 1, 1, 1, 1, TI) => (R, l0, Nl, WFE, IWA, TI)
    signal_6D = signal_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT (throught coronagraph, if any)
    # [e-/FWHM/mn]              * [mn/DIT]                  => [e-/FWHM/DIT]
    # (R, l0, Nl, WFE, IWA, TI) * (R, l0, Nl, WFE, IWA, TI) => (R, l0, Nl, WFE, IWA, TI)
    signal_6D = signal_6D * DIT_6D


    # --- Computing stellar halo photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Nl, WFE, IWA, TI) ---

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [ph/px/mn] (throught coronagraph, if any)
    # [total star ph/mn] * [star flux fraction/px] => [ph/px/mn]
    # (R, l0, Nl, 1, 1)  * (1, l0, 1, WFE, IWA)    => (R, l0, Nl, WFE, IWA)
    sigma_halo_2_5D = sigma_halo_2_3D[:, :, :, None, None] * PSF_profile_sep_3D[None, :, None, :, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/px/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]               * [e-/ph]             => [e-/px/mn]
    # (R, l0, Nl, WFE, IWA, 1) * (1, 1, 1, 1, 1, TI) => (R, l0, Nl, WFE, IWA, TI)
    sigma_halo_2_6D = sigma_halo_2_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance) (throught coronagraph, if any)
    # [e-/px/mn]                * [px/FWHM]*[mn/DIT]        => [e-/FWHM/DIT]
    # (R, l0, Nl, WFE, IWA, TI) * (R, l0, Nl, WFE, IWA, TI) => (R, l0, Nl, WFE, IWA, TI)
    sigma_halo_2_6D = sigma_halo_2_6D * A_FWHM*DIT_6D


    # --- Computing stellar halo systematic noise variance (at sigma_m = 1) in [e-/FWHM/DIT]**2 (R, l0, Nl, WFE, IWA, TI) (all quantities are squared since sigma_syst**2 propto stellar_halo**2) ---

    # Stellar halo systematic noise variance at the planet's separation in [ph/px/mn]**2 (throught coronagraph, if any)
    # [total star ph/mn]**2 * [star flux fraction/px]**2 => [ph/px/mn]**2
    # (R, l0, Nl, 1, 1)     * (1, l0, 1, WFE, IWA)       => (R, l0, Nl, WFE, IWA)
    sigma_syst_base_2_5D = sigma_syst_base_2_3D[:, :, :, None, None] * PSF_profile_sep_3D[None, :, None, :, :]**2

    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]**2            * [e-/ph]**2          => [e-/px/mn]**2
    # (R, l0, Nl, WFE, IWA, 1) * (1, 1, 1, 1, 1, TI) => (R, l0, Nl, WFE, IWA, TI)
    sigma_syst_base_2_6D = sigma_syst_base_2_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]**2

    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission integrated over the DIT and the FWHM (throught coronagraph, if any) (we take A_FWHM**2 * DIT**2 since sigma_syst_base_2 propto H_integrated**2 propto A_FWHM**2*DIT**2, where H_integrated = the halo integrated over FWHM boxes and the DIT)
    # [e-/px/mn]**2             * [px/FWHM]**2*[mn/DIT]**2  => [e-/FWHM/DIT]**2
    # (R, l0, Nl, WFE, IWA, TI) * (R, l0, Nl, WFE, IWA, TI) => (R, l0, Nl, WFE, IWA, TI)
    sigma_syst_base_2_6D = sigma_syst_base_2_6D * A_FWHM**2*DIT_6D**2


    # --- Computing background photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Nl, WFE, IWA, TI) ---

    # Background photon noise variance (Poisson) in [e-/px/mn] with instrumental transmission
    # [ph/px/mn]     * [e-/ph]       => [e-/px/mn]
    # (R, l0, Nl, 1) * (1, 1, 1, TI) => (R, l0, Nl, TI)
    sigma_bkg_2_4D = sigma_bkg_2_3D[:, :, :, None] * trans_instru[None, None, None, :]

    # Background photon noise variance (Poisson) in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance)
    # [e-/px/mn]            * [px/FWHM]*[mn/DIT]        => [e-/FWHM/DIT]
    # (R, l0, Nl, 1, 1, TI) * (R, l0, Nl, WFE, IWA, TI) => (R, l0, Nl, WFE, IWA, TI)
    sigma_bkg_2_6D = sigma_bkg_2_4D[:, :, :, None, None, :] * A_FWHM*DIT_6D


    # --- Saving ---
    _IFU_CTX["signal_out"][idx]            = signal_6D.astype(dtype,            copy=False) # [e-/FWHM/DIT]
    _IFU_CTX["sigma_halo_2_out"][idx]      = sigma_halo_2_6D.astype(dtype,      copy=False) # [e-/FWHM/DIT]**2
    _IFU_CTX["sigma_bkg_2_out"][idx]       = sigma_bkg_2_6D.astype(dtype,       copy=False) # [e-/FWHM/DIT]**2
    _IFU_CTX["DIT_out"][idx]               = DIT_6D.astype(dtype,               copy=False) # [mn/DIT]
    _IFU_CTX["sigma_syst_base_2_out"][idx] = sigma_syst_base_2_6D.astype(dtype, copy=False) # [e-/FWHM/DIT]**2 (at sigma_m = 1)
    return idx



# --- Imager function ---
def process_IM(idx):

    # --- Context ---
    planet_table       = _IM_CTX["planet_table"]
    l0                 = _IM_CTX["l0"]
    A_FWHM             = _IM_CTX["A_FWHM"]
    separation         = _IM_CTX["separation"]
    thermal_model      = _IM_CTX["thermal_model"]
    reflected_model    = _IM_CTX["reflected_model"]
    wave_instru        = _IM_CTX["wave_instru"]
    wave_K             = _IM_CTX["wave_K"]
    counts_vega_K      = _IM_CTX["counts_vega_K"]
    vega_flux_1D       = _IM_CTX["vega_flux_1D"]
    mag_star           = _IM_CTX["mag_star"]
    scale_spectrum     = _IM_CTX["scale_spectrum"]
    trans_instru       = _IM_CTX["trans_instru"]
    trans_tell_tel     = _IM_CTX["trans_tell_tel"]
    background_flux_2D = _IM_CTX["background_flux_2D"]
    range_1D           = _IM_CTX["range_1D"]
    saturation_e       = _IM_CTX["saturation_e"]
    min_DIT            = _IM_CTX["min_DIT"]
    max_DIT            = _IM_CTX["max_DIT"]
    PSF_profile_5D     = _IM_CTX["PSF_profile_5D"]
    fraction_core_5D   = _IM_CTX["fraction_core_5D"]
    PSF_profile_max_4D = _IM_CTX["PSF_profile_max_4D"]
    dtype              = _IM_CTX["dtype"]


    # --- Planet row ---
    planet            = planet_table[idx]
    separation_planet = float(planet["AngSep"].value) # [mas]


    # --- Computing the planet and star models for this planet row in [J/s/m²/µm] ---
    planet_spectrum, _, _, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True, interpolated_spectrum=True, apply_kinematics=False)


    # --- Computing mag_star_1D for each l0 ---
    star_flux_1D = np.interp(l0, star_spectrum.wavelength, star_spectrum.flux, left=np.nan, right=np.nan) # (l0) [J/s/m2/µm]
    mag_star_1D  = -2.5 * np.log10(star_flux_1D / vega_flux_1D)                                           # (l0) [no unit]


    # --- PSF data linearly interpolated at the planet's separation and stellar magnitude ---
    # PSF_profile_5D     : (l0, WFE, IWA, mag_star, sep)
    # fraction_core_5D   : (l0, WFE, IWA, mag_star, sep)
    # PSF_profile_max_4D : (l0, WFE, IWA, mag_star)
    #
    # For speed, we first interpolate along separation, because separation_planet is
    # a scalar for this planet. This avoids creating full (l0, WFE, IWA, sep) arrays.
    PSF_profile_mag_4D   = interp_flat_last_axis(separation, PSF_profile_5D,   separation_planet, dtype=dtype) # (l0, WFE, IWA, mag_star)
    fraction_core_mag_4D = interp_flat_last_axis(separation, fraction_core_5D, separation_planet, dtype=dtype) # (l0, WFE, IWA, mag_star)

    # Then interpolate along stellar magnitude. mag_star_1D depends on l0, so this
    # specialized helper interpolates each l0 slice at its own magnitude.
    PSF_profile_sep_3D   = interp_mag_l0_flat_4d(mag_star, PSF_profile_mag_4D,   mag_star_1D, dtype=dtype) # (l0, WFE, IWA) [star flux fraction/px]
    fraction_core_sep_3D = interp_mag_l0_flat_4d(mag_star, fraction_core_mag_4D, mag_star_1D, dtype=dtype) # (l0, WFE, IWA) [planet flux fraction/FWHM]
    PSF_profile_max_3D   = interp_mag_l0_flat_4d(mag_star, PSF_profile_max_4D,   mag_star_1D, dtype=dtype) # (l0, WFE, IWA) [max star flux fraction/px]


    # --- Converting in [ph/mn/bin] ---
    star   = star_spectrum.flux   * scale_spectrum # [J/s/m²/µm] => [ph/mn/bin]
    planet = planet_spectrum.flux * scale_spectrum # [J/s/m²/µm] => [ph/mn/bin]


    # --- Computing data (signal and stellar halo photon noise variance (Poisson)) depending only on l0 and Dl ---
    idx_lo, idx_hi = range_1D
    signal_2D      = window_sums_1d(trans_tell_tel * planet, idx_lo, idx_hi).astype(dtype) # = sum ( trans * S_planet ) (l0, Dl) [ph/mn]
    star_flux_2D   = window_sums_1d(trans_tell_tel * star,   idx_lo, idx_hi).astype(dtype) # = sum ( trans * S_star )   (l0, Dl) [ph/mn]

    # --- Computing DIT length in [mn/DIT] (l0, Dl, WFE, IWA, TI) ---

    # max_star_flux in [ph/px/mn] in the brightest pixel (through coronagraph, if any):
    # [total star ph/mn] * [star flux fraction/px] in the brightest pixel => [ph/px/mn] in the brightest pixel
    # (l0, Dl, 1, 1)     * (l0, 1, WFE, IWA)                              => (l0, Dl, WFE, IWA)
    max_star_flux_4D = star_flux_2D[:, :, None, None] * PSF_profile_max_3D[:, None, :, :]

    # max_star_flux in [e-/px/mn] in the brightest pixel with instrumental transmission (through coronagraph, if any):
    # [ph/px/mn] in the brightest pixel * [e-/ph]          => [e-/px/mn] in the brightest pixel
    # (l0, Dl, WFE, IWA, 1)             * (1, 1, 1, 1, TI) => (l0, Dl, WFE, IWA, TI)
    max_star_flux_5D = max_star_flux_4D[:, :, :, :, None] * trans_instru[None, None, None, None, :]

    # Computing DIT length in [mn/DIT] (+ clipping between min and max DIT)
    saturation_e_5D = saturation_e[:, None, None, None, None]
    min_DIT_5D      = min_DIT[:, None, None, None, None]
    max_DIT_5D      = max_DIT[:, None, None, None, None]
    DIT_5D          = saturation_e_5D / max_star_flux_5D # (l0, Dl, WFE, IWA, TI) [mn/DIT]
    DIT_5D          = np.maximum(DIT_5D, min_DIT_5D)     # (l0, Dl, WFE, IWA, TI) [mn/DIT]
    DIT_5D          = np.minimum(DIT_5D, max_DIT_5D)     # (l0, Dl, WFE, IWA, TI) [mn/DIT]

    # --- Computing signal in [e-/FWHM/DIT] (l0, Dl, WFE, IWA, TI) ---

    # Signal at the planet's separation inside the FWHM in [ph/FWHM/mn] (throught coronagraph, if any)
    # [total planet ph/mn] * [planet flux fraction/FWHM] => [ph/FWHM/mn]
    # (l0, Dl, 1, 1)       * (l0, 1, WFE, IWA)           => (l0, Dl, WFE, IWA)
    signal_4D = signal_2D[:, :, None, None] * fraction_core_sep_3D[:, None, :, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/FWHM/mn]          * [e-/ph]          => [e-/FWHM/mn]
    # (l0, Dl, WFE, IWA, 1) * (1, 1, 1, 1, TI) => (l0, Dl, WFE, IWA, TI)
    signal_5D = signal_4D[:, :, :, :, None] * trans_instru[None, None, None, None, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT(throught coronagraph, if any)
    # [e-/FWHM/mn]           * [mn/DIT]               => [e-/FWHM/DIT]
    # (l0, Dl, WFE, IWA, TI) * (l0, Dl, WFE, IWA, TI) => (l0, Dl, WFE, IWA, TI)
    signal_5D = signal_5D * DIT_5D


    # --- Computing stellar halo photon noise variance (Poisson) in [e-/FWHM/DIT] (l0, Dl, WFE, IWA, TI) ---

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [ph/px/mn] (throught coronagraph, if any)
    # [total star ph/mn] * [star flux fraction/px] => [ph/px/mn]
    # (l0, Dl, 1, 1)     * (l0, 1, WFE, IWA)       => (l0, Dl, WFE, IWA)
    sigma_halo_2_4D = star_flux_2D[:, :, None, None] * PSF_profile_sep_3D[:, None, :, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/px/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]            * [e-/ph]          => [e-/px/mn]
    # (l0, Dl, WFE, IWA, 1) * (1, 1, 1, 1, TI) => (l0, Dl, WFE, IWA, TI)
    sigma_halo_2_5D = sigma_halo_2_4D[:, :, :, :, None] * trans_instru[None, None, None, None, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance) (throught coronagraph, if any)
    # [e-/px/mn]             * [px/FWHM]*[mn/DIT]     => [e-/FWHM/DIT]
    # (l0, Dl, WFE, IWA, TI) * (l0, Dl, WFE, IWA, TI) => (l0, Dl, WFE, IWA, TI)
    sigma_halo_2_5D = sigma_halo_2_5D * A_FWHM*DIT_5D


    # --- Computing background photon noise variance (Poisson) in [e-/FWHM/DIT] (l0, Dl, WFE, IWA, TI) ---

    # Background photon noise variance (Poisson) in [e-/px/mn] with instrumental transmission
    # [ph/px/mn]  * [e-/ph]    => [e-/px/mn]
    # (l0, Dl, 1) * (1, 1, TI) => (l0, Dl, TI)
    sigma_bkg_2_3D = background_flux_2D[:, :, None] * trans_instru[None, None, :]

    # Background photon noise variance (Poisson) in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance)
    # [e-/px/mn]         * [px/FWHM]*[mn/DIT]     => [e-/FWHM/DIT]
    # (l0, Dl, 1, 1, TI) * (l0, Dl, WFE, IWA, TI) => (l0, Dl, WFE, IWA, TI)
    sigma_bkg_2_5D = sigma_bkg_2_3D[:, :, None, None, :] * A_FWHM*DIT_5D


    # --- Saving ---
    _IM_CTX["signal_out"][idx]       = signal_5D.astype(dtype,       copy=False) # [e-/FWHM/DIT]
    _IM_CTX["sigma_halo_2_out"][idx] = sigma_halo_2_5D.astype(dtype, copy=False) # [e-/FWHM/DIT]**2
    _IM_CTX["sigma_bkg_2_out"][idx]  = sigma_bkg_2_5D.astype(dtype,  copy=False) # [e-/FWHM/DIT]**2
    _IM_CTX["DIT_out"][idx]          = DIT_5D.astype(dtype,          copy=False) # [mn/DIT]
    return idx



# --------------------------
# Function for computing S/N
# --------------------------

# Function to be multiprocessed over planets
def process_SNR(idx):

    instru_type     = _SNR_CTX["instru_type"]
    exposure_time   = _SNR_CTX["exposure_time"]
    A_FWHM          = _SNR_CTX["A_FWHM"]
    sigma_m         = _SNR_CTX["sigma_m_broadcast"]
    dtype           = _SNR_CTX["dtype"]

    # --- Read one planet slice ---
    signal       = _SNR_CTX["signal_planets"][idx]       # [e-/FWHM/DIT]
    sigma_halo_2 = _SNR_CTX["sigma_halo_2_planets"][idx] # [e-/FWHM/DIT]**2
    sigma_bkg_2  = _SNR_CTX["sigma_bkg_2_planets"][idx]  # [e-/FWHM/DIT]**2
    DIT          = _SNR_CTX["DIT_planets"][idx]          # [mn/DIT]
    if instru_type == "IFU":
        sigma_syst_base_2 = _SNR_CTX["sigma_syst_base_2_planets"][idx] # [e-/FWHM/DIT]**2 (at sigma_m = 1)

    # Detector parameters are arrays of shape (N_l0).  They are broadcast onto the
    # parameter grid.  The l0 axis is axis 1 for IFU arrays and axis 0 for imager arrays.
    N_l0    = _SNR_CTX["N_l0"]
    l0_axis = _SNR_CTX["l0_axis"]
    shape_l0          = [1] * DIT.ndim
    shape_l0[l0_axis] = N_l0
    shape_l0           = tuple(shape_l0)
    RON0     = np.asarray(_SNR_CTX["RON0"],    dtype=dtype).reshape(shape_l0)
    RON_lim  = np.asarray(_SNR_CTX["RON_lim"], dtype=dtype).reshape(shape_l0)
    DC0      = np.asarray(_SNR_CTX["DC0"],     dtype=dtype).reshape(shape_l0)
    min_DIT  = np.asarray(_SNR_CTX["min_DIT"], dtype=dtype).reshape(shape_l0)

    # --- Number of DIT ---
    N_DIT = np.floor(exposure_time / DIT).astype(dtype)
    N_DIT = np.clip(N_DIT, 1, None)

    # --- Number of reads ---
    N_read = np.floor(DIT / min_DIT).astype(np.int32)

    # --- Effective RON variance in UTR mode in [e-/FWHM/DIT]**2 ---
    RON0_grid    = np.broadcast_to(RON0, DIT.shape)
    RON_lim_grid = np.broadcast_to(RON_lim, DIT.shape)
    sigma_RON_2  = (RON0_grid**2).astype(dtype).copy()
    m            = (N_read >= 2)
    if np.any(m):
        n_read         = N_read[m].astype(dtype)
        sigma_RON_2[m] = (RON0_grid[m]**2 * 12 * (n_read - 1) / (n_read * (n_read + 1)) + RON_lim_grid[m]**2).astype(dtype)
    sigma_RON_2 *= A_FWHM

    # --- Dark current variance (Poisson) in [e-/FWHM/DIT]**2 ---
    sigma_DC_2 = (DC0 * DIT * A_FWHM).astype(dtype) # [e-/FWHM/DIT]**2

    # --- Add sigma_m axis at the end ---
    signal       = signal[..., None]       # [e-/FWHM/DIT]
    sigma_halo_2 = sigma_halo_2[..., None] # [e-/FWHM/DIT]**2
    sigma_bkg_2  = sigma_bkg_2[..., None]  # [e-/FWHM/DIT]**2
    sigma_RON_2  = sigma_RON_2[..., None]  # [e-/FWHM/DIT]**2
    sigma_DC_2   = sigma_DC_2[..., None]   # [e-/FWHM/DIT]**2
    N_DIT        = N_DIT[..., None]        # [no unit]

    # --- Systematic/speckle variance in [e-/FWHM/DIT]**2 ---
    if instru_type == "IFU":
        sigma_syst_base_2 = sigma_syst_base_2[..., None] # [e-/FWHM/DIT]**2 (at sigma_m = 1)
    elif instru_type == "imager":
        halo_counts_planets = sigma_halo_2                      # [e-/FWHM/DIT]
        sigma_syst_base_2   = halo_counts_planets**2 # [e-/FWHM/DIT]**2 (at sigma_m = 1)
    else:
        raise ValueError("instru_type must be 'IFU' or 'imager'.")

    # --- SNR ---
    snr = N_DIT*signal / np.sqrt( N_DIT*(sigma_halo_2 + sigma_bkg_2 + sigma_RON_2 + sigma_DC_2) + N_DIT**2 * sigma_m**2 * sigma_syst_base_2)

    # --- Direct write to output memmap ---
    _SNR_CTX["SNR_planets"][idx] = snr.astype(dtype, copy=False)

    return idx



def get_SNR(instru, instru_type, post_processing, exposure_time, min_DIT, RON0, RON_lim, DC0, A_FWHM, Rc, filter_type, signal_planets, sigma_halo_2_planets, sigma_bkg_2_planets, sigma_syst_base_2_planets, DIT_planets, R, Nl, sigma_m, sim_dir, suffix, dtype):

    print("\nComputing SNR from FastYield data...\n")

    # SNR shape (adding sigma_m axis)
    shape_full = signal_planets.shape      # (planets, ...grid..)
    shape_grid = shape_full[1:]            # (...grid...)
    N_PT       = shape_full[0]             # (planets)
    N_dim_grid = len(shape_grid)           # Number of parameters dimension
    N_sigma_m  = len(sigma_m)              # (sigma_m)
    shape_SNR  = shape_full + (N_sigma_m,) # (planets, ...grid..., sigma_m)

    # Broadcasting sigma_m axis
    sigma_m_broadcast = sigma_m.reshape((1,) * N_dim_grid + (N_sigma_m,)).astype(dtype) # (...grid..., sigma_m)

    # Creating the memmap SNR file
    SNR_path    = sim_dir / f"tmp_SNR_{suffix}.npy"
    SNR_planets = create_memmap_with_log(SNR_path, shape_SNR, dtype=dtype, mode="w+")

    # No planet case
    if N_PT == 0:
        return SNR_planets

    # Paths of existing memmaps
    signal_path       = Path(signal_planets.filename)
    sigma_halo_2_path = Path(sigma_halo_2_planets.filename)
    sigma_bkg_2_path  = Path(sigma_bkg_2_planets.filename)
    DIT_path          = Path(DIT_planets.filename)

    if instru_type == "IFU":
        l0_axis                = 1
        sigma_syst_base_2_path = Path(sigma_syst_base_2_planets.filename)
    else:
        l0_axis                = 0
        sigma_syst_base_2_path = None
    N_l0    = len(RON0)
    snr_ctx = dict(instru_type=instru_type, post_processing=post_processing, exposure_time=dtype(exposure_time), min_DIT=dtype(min_DIT), RON0=dtype(RON0), RON_lim=dtype(RON_lim), DC0=dtype(DC0), A_FWHM=dtype(A_FWHM), sigma_m_broadcast=sigma_m_broadcast, signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), sigma_syst_base_2_path=str(sigma_syst_base_2_path) if sigma_syst_base_2_path is not None else None, SNR_path=str(SNR_path), shape_SNR=shape_SNR, dtype=np.dtype(dtype).str, N_l0=N_l0, l0_axis=l0_axis)
    if sys.platform.startswith(("win", "darwin")):
        ctx_mp = mp.get_context("spawn")
    else:
        ctx_mp = mp.get_context("fork")

    print()
    with ctx_mp.Pool(processes=nproc, initializer=init_worker_SNR, initargs=(snr_ctx,)) as pool:
        for _ in tqdm(pool.imap_unordered(process_SNR, range(N_PT), chunksize=chunksize), total=N_PT, desc=f"Computing ELT/{instru}({instru_type}+{post_processing}) S/N for each planet and parameters set",):
            pass
    SNR_planets.flush()

    return SNR_planets



# -------------------------------------
# Function for computing detection mask
# -------------------------------------

def get_mask_detections(SNR_planets, SNR_thr, sim_dir, suffix, target_chunk_mb=2048):

    shape_full = SNR_planets.shape
    shape_grid = shape_full[1:]
    N_PT       = shape_full[0]

    mask_detections_path = sim_dir / f"tmp_mask_detections_{suffix}.npy"
    mask_detections      = create_memmap_with_log(mask_detections_path, shape_full, dtype=bool, mode="w+")

    if N_PT == 0:
        mask_detections.flush()
        return mask_detections

    # Approximate chunk size from memory budget:
    # one float SNR array -> one boolean mask array
    grid_size        = int(np.prod(shape_grid, dtype=np.int64))
    bytes_per_snr    = np.dtype(SNR_planets.dtype).itemsize
    bytes_per_bool   = np.dtype(bool).itemsize
    bytes_per_planet = grid_size * (bytes_per_snr + bytes_per_bool)
    target_bytes     = int(target_chunk_mb * 1024**2)
    chunk_size       = int(max(1, target_bytes // max(bytes_per_planet, 1)))

    print()
    for i0 in tqdm(range(0, N_PT, chunk_size), desc="Building mask_detections"):
        i1                     = min(i0 + chunk_size, N_PT)
        mask_detections[i0:i1] = SNR_planets[i0:i1] >= SNR_thr

    mask_detections.flush()
    return mask_detections



# --------------------------------------------
# Function for computing detection probability
# --------------------------------------------

def get_Pdet(mask_detections, FoV, p0_FoV, dtype, indices=None, target_chunk_mb=2048, jump_before_tqdm=True):
    # If no subset is provided, all planets are included
    if indices is None:
        indices = np.arange(len(p0_FoV), dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)
    N_PT = len(indices)

    shape_grid = mask_detections.shape[1:]  # (...grid..., sigma_m)
    N_FoV      = len(FoV)

    # buckets[..., p] = sum of detection masks for planets with gating index p0 = p
    buckets = np.zeros(shape_grid + (N_FoV,), dtype=np.uint32)

    if N_PT == 0:
        return buckets.astype(dtype)

    # Precompute FoV gating once
    p0_sel = p0_FoV[indices]
    valid  = (p0_sel < N_FoV)

    if not np.any(valid):
        return buckets.astype(dtype)

    indices_valid = indices[valid]
    p0_valid      = p0_sel[valid]

    # Approximate chunk size from memory budget:
    # only one boolean chunk is materialized now
    grid_size        = int(np.prod(shape_grid, dtype=np.int64))
    bytes_per_bool   = np.dtype(bool).itemsize
    bytes_per_planet = grid_size * bytes_per_bool
    target_bytes     = int(target_chunk_mb * 1024**2)
    chunk_size       = int(max(1, target_bytes // max(bytes_per_planet, 1)))
    
    if jump_before_tqdm:
        print()
    for i0 in tqdm(range(0, len(indices_valid), chunk_size), desc="Computing the detection probability for each parameters set"):
        i1        = min(i0 + chunk_size, len(indices_valid))
        idx_chunk = indices_valid[i0:i1]
        p0_chunk  = p0_valid[i0:i1]

        # Materialize only one boolean chunk in RAM
        mask_chunk = mask_detections[idx_chunk]  # shape = (chunk, ...grid...)

        # Accumulate masks by FoV gating index
        for p in np.unique(p0_chunk):
            m                = (p0_chunk == p)
            buckets[..., p] += mask_chunk[m].sum(axis=0, dtype=np.uint32)

    # A planet with gating index p0 contributes to all FoV bins with index >= p0
    N_detections = np.cumsum(buckets, axis=-1)

    return (N_detections / N_PT).astype(dtype)



# ------------------------------------------------
# Function for marginalizing detection probability
# ------------------------------------------------

def get_axis_weights_in_range(axis, pmin, pmax, prior):
    """
    Return the indices and integration weights of the grid cells
    overlapping [pmin, pmax], using either a uniform or log-uniform prior.

    If pmin == pmax, the parameter is treated as fixed and the nearest grid
    point is returned with weight 1. This is useful for manual weighting outside
    reduce_Pdet().
    """
    axis = np.asarray(axis, dtype=float)
    pmin = float(pmin)
    pmax = float(pmax)

    if axis.ndim != 1:
        raise ValueError("axis must be 1D.")
    if len(axis) == 0:
        raise ValueError("axis must not be empty.")
    if np.any(np.diff(axis) <= 0):
        raise ValueError("axis must be strictly increasing.")
    if pmin > pmax:
        raise ValueError("pmin must be <= pmax.")

    tol = 1e-10 * max(1.0, abs(pmin), abs(pmax), abs(axis[0]), abs(axis[-1]))

    if pmin < axis[0] - tol or pmax > axis[-1] + tol:
        raise ValueError(f"Requested range [{pmin:g}, {pmax:g}] is outside axis range [{axis[0]:g}, {axis[-1]:g}].")

    # Clip tiny floating-point overshoots, e.g. 2000.0000000000002
    pmin = np.clip(pmin, axis[0], axis[-1])
    pmax = np.clip(pmax, axis[0], axis[-1])

    # Fixed parameter case
    if np.isclose(pmin, pmax, rtol=0.0, atol=tol):
        x0  = 0.5 * (pmin + pmax)
        idx = int(np.nanargmin(np.abs(axis - x0)))
        return np.array([idx], dtype=int), np.array([1.0], dtype=float)

    if prior == "uniform":
        x    = axis
        xmin = pmin
        xmax = pmax
    elif prior == "log-uniform":
        if np.any(axis <= 0):
            raise ValueError("log-uniform prior requires strictly positive axis values.")
        if pmin <= 0 or pmax <= 0:
            raise ValueError("log-uniform prior requires pmin > 0 and pmax > 0.")
        x    = np.log(axis)
        xmin = np.log(pmin)
        xmax = np.log(pmax)
    else:
        raise ValueError("prior must be 'uniform' or 'log-uniform'.")

    # Cell edges in transformed coordinate
    if len(x) == 1:
        edges   = np.array([xmin, xmax], dtype=float)
        weights = np.array([xmax - xmin], dtype=float)
        return np.array([0], dtype=int), weights

    edges       = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0]    = x[0]  - 0.5 * (x[1]  - x[0])
    edges[-1]   = x[-1] + 0.5 * (x[-1] - x[-2])

    # Overlap of each cell with [xmin, xmax]
    left    = np.maximum(edges[:-1], xmin)
    right   = np.minimum(edges[1:],  xmax)
    weights = np.maximum(right - left, 0.0)

    idx     = np.where(weights > 0)[0]
    weights = weights[idx]

    if len(idx) == 0:
        raise ValueError(f"No cell overlaps requested range [{pmin:g}, {pmax:g}] for this axis.")

    return idx, weights



def interp_along_axis(values, axis, x0, axis_id):
    """
    Linear interpolation of an N-D array along one axis.
    """
    axis   = np.asarray(axis, dtype=float)
    values = np.asarray(values)
    if x0 < axis[0] or x0 > axis[-1]:
        raise ValueError(f"x0={x0:g} is outside axis range [{axis[0]:g}, {axis[-1]:g}].")

    # Robust boundary handling
    if np.isclose(x0, axis[0], rtol=0, atol=10*np.finfo(float).eps*max(1.0, abs(axis[0]))):
        return np.take(values, 0, axis=axis_id)
    if np.isclose(x0, axis[-1], rtol=0, atol=10*np.finfo(float).eps*max(1.0, abs(axis[-1]))):
        return np.take(values, len(axis) - 1, axis=axis_id)

    # Exact grid point
    idx_exact = np.where(np.isclose(axis, x0, rtol=0, atol=10*np.finfo(float).eps*np.maximum(1.0, np.abs(axis))))[0]
    if len(idx_exact) > 0:
        return np.take(values, idx_exact[0], axis=axis_id)

    # Bracketing indices
    i1 = np.searchsorted(axis, x0, side="right")
    i0 = i1 - 1
    if i0 < 0:
        return np.take(values, 0, axis=axis_id)
    if i1 >= len(axis):
        return np.take(values, len(axis) - 1, axis=axis_id)

    x_left  = axis[i0]
    x_right = axis[i1]
    alpha   = (x0 - x_left) / (x_right - x_left)

    v0 = np.take(values, i0, axis=axis_id)
    v1 = np.take(values, i1, axis=axis_id)

    return (1.0 - alpha) * v0 + alpha * v1



def reduce_hcube(hcube, dims_to_keep, params, params_ranges, params_priors, params_names, verbose=False):
    """
    Reduce a hcube grid by fixing and/or marginalizing parameters.

    Parameters
    ----------
    hcube : ndarray
        Detection probability grid.
        Axis order must match the order of 'params' / 'params_names'.
    dims_to_keep : list of int
        Dimensions of parameters to keep in the final reduced hcube.
    params : list of 1D arrays
        Example:
            [R, l0, Nl, WFE, IWA, trans_instru, sigma_m, FoV]
    params_ranges : list of tuple
        One entry per parameter, same order as 'params'.
        Each entry must be (pmin, pmax).
        - if pmin == pmax : fix parameter by linear interpolation
        - if pmin <  pmax : marginalize parameter over [pmin, pmax]
    params_priors : list of str
        One entry per parameter, same order as 'params'.
        Each entry must be "uniform" or "log-uniform".
        Only used for marginalized parameters.
    params_names : list of str
        Example:
            ["R", "l0", "Nl", "WFE", "IWA", "trans_instru", "sigma_m", "FoV"]
    verbose : bool
        Print a summary.

    Returns
    -------
    hcube_red : ndarray
        Reduced detection probability grid.
    """
    dims_to_keep  = sorted(list(dims_to_keep))
    params        = list(params)
    params_names  = list(params_names)
    params_ranges = list(params_ranges)
    params_priors = list(params_priors)

    nparam   = len(params)
    hcube_red = hcube.copy()

    if len(params_names) != nparam:
        raise ValueError("params and params_names must have the same length.")
    if len(params_ranges) != nparam:
        raise ValueError("params_ranges must have the same length as params.")
    if len(params_priors) != nparam:
        raise ValueError("params_priors must have the same length as params.")
    if hcube.ndim != nparam:
        raise ValueError("hcube.ndim must match len(params).")
    if len(set(params_names)) != len(params_names):
        raise ValueError("params_names must be unique.")
    if any(idim < 0 or idim >= nparam for idim in dims_to_keep):
        raise ValueError("dims_to_keep contains invalid axis indices.")

    dims_to_reduce = [idim for idim in range(nparam) if idim not in dims_to_keep]

    # --- initial print ---
    if verbose:
        shape_ini = tuple(len(p) for p in params)
        print("\nPdet reduction")
        print("--------------")
        print(f"\nInitial axes : ({', '.join(params_names)}) = {shape_ini}\n")
        for idim in range(nparam):
            if idim in dims_to_keep:
                print(f"  - Keeping       {params_names[idim]:<20} axis")

    for idim in sorted(dims_to_reduce, reverse=True):

        axis       = params[idim]
        pmin, pmax = params_ranges[idim]
        prior      = params_priors[idim]
        name       = params_names[idim]

        if pmin > pmax:
            raise ValueError(f"For parameter '{name}', pmin must be <= pmax.")
        if pmin < axis[0] or pmax > axis[-1]:
            raise ValueError(f"For parameter '{name}', requested range [{pmin:g}, {pmax:g}] is outside axis range [{axis[0]:g}, {axis[-1]:g}].")

        # --- fix to one value: linear interpolation ---
        if pmin == pmax:
            hcube_red = interp_along_axis(hcube_red, axis=axis, x0=pmin, axis_id=idim)
            if verbose:
                print(f"  - Fixing        {name:<20} at {pmin:<8g} with linear interpolation")

        # --- marginalize on [pmin, pmax] with proper truncated cell weights ---
        else:
            idx, w        = get_axis_weights_in_range(axis=axis, pmin=pmin, pmax=pmax, prior=prior)
            hcube_red      = np.take(hcube_red, idx, axis=idim)
            shape_w       = [1] * hcube_red.ndim
            shape_w[idim] = len(w)
            w             = w.reshape(shape_w)
            hcube_red      = np.sum(hcube_red * w, axis=idim) / np.sum(w)
            if verbose:
                print(f"  - Marginalizing {name:<20} in [{pmin:<8g}, {pmax:<8g}] ({len(idx)} cells) with {prior} prior")

    # --- final print ---
    if verbose:
        params_names_red = [params_names[idim] for idim in dims_to_keep]
        shape_end        = tuple(len(params[idim]) for idim in dims_to_keep)
        print(f"\nFinal axes   : ({', '.join(params_names_red)}) = {shape_end}\n")

    return hcube_red



def save_marginalized_planet_table(planet_table, signal, sigma_fund, sigma_syst, DIT, SNR, exposure_time, table, instru, apodizer, strehl, coronagraph, name_model, path):
    """Save marginalized FastYield quantities and SNR in a standard planet table."""
    planet_table = planet_table.copy()
    planet_table["signal_INSTRU"]     = np.asarray(signal, dtype=float)     # [e-/FWHM/DIT]
    planet_table["sigma_fund_INSTRU"] = np.asarray(sigma_fund, dtype=float) # [e-/FWHM/DIT]
    planet_table["sigma_syst_INSTRU"] = np.asarray(sigma_syst, dtype=float) # [e-/FWHM/DIT]
    planet_table["DIT_INSTRU"]        = np.asarray(DIT, dtype=float)        # [mn/DIT]

    exposure_label = f"{exposure_time/60:g}".replace(".", "p")
    planet_table[f"SNR_INSTRU_{exposure_label}h"] = np.asarray(SNR, dtype=float)

    filename    = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=False, PCA=False, name_model=name_model)
    output_path = Path(path)/filename
    planet_table.write(output_path, format="ascii.ecsv", overwrite=True)
    print(f"\nMarginalized PCS planet table saved to:\n{output_path}")

    return output_path



# %%
def main():

    # ---------------------------
    # Parameters
    # ---------------------------
    PCS_CODE_VERSION = "PCS_sim_v2"
    dtype            = np.float32

    # --- Instrument specs (fixed) ---
    instru     = "PCS"     # Instrument's name
    D          = 38.54     # [m]
    S          = 980.      # [m2]
    N_mirror   = 5         # Number of mirror at the ELT (in order to compute the telescope transmission)
    trans_dust = 0.90      # effect of dust, from common ICD, section 4.11, p37 (Document Number: ESO-253082)

    # --- Detector mode ---
    # "constant_H4RG"
    #     Simple NIR baseline. Recommended if l0_min >= ~0.8 µm.
    #
    # "PCS_visNIR_conservative"
    #     More realistic PCS-like prescription over 0.6--2.5 µm:
    #         visible/red optical : VIS_CCD or EMCCD-like detector
    #         NIR                 : H4RG
    #
    # "PCS_NIR_APD_optimistic"
    #     Technology-study mode:
    #         visible/red optical : EMCCD-like detector
    #         NIR                 : SAPHIRA / LmAPD-like detector
    #     Not a baseline science-detector prescription because current SAPHIRA-like
    #     arrays are small-format and mainly used for fast NIR applications.
    detector = "PCS_visNIR_conservative"

    # --- Paths ---
    instru_dir = get_sim_data_path() / instru
    sim_dir    = instru_dir / "FastYield_simulations"
    psf_dir    = instru_dir / "PSF_simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # --- General parameters for the simulation ---
    coronagraph        = "LYOT"                                # Coronagraph config
    apodizer           = "NO_SP"                               # Shaped pupil mask (NO_SP => no mask)
    strehl             = "Q2"                                  # Sky atmospheric condition (1st quartile, 2nd, etc.)
    SNR_thr            = 5                                     # Detection threshold
    exposure_time      = 10*60                                 # Total exposure time per planet [mn]
    force_new_calc     = False                                 # Forcing new simulations calculations
    thermal_model      = "auto"                                # Model for the thermal spectrum of the planet ("auto", "None", "BT-Settl", "Exo-REM", "SONORA", "PICASO", "Saumon", etc.)
    reflected_model    = "auto"                                # Model for the albedo of the planet ("auto", "tellurics", "flat", "PICASO")
    instru_type        = "IFU"                                 # Type of instrument ("IFU" or "imager")
    post_processing    = "DI"                                  # Post-processing method ("MM" or "DI")
    size_core          = 2                                     # [px/FWHM] Number of pixel per spatial FWHM along 1 direction (size_core >= 2 => Nyquist spatial sampling)
    A_FWHM             = size_core**2                          # Number of pixel per FWHM box area
    Rc                 = 1_00                                  # MM cut-off resolution (Rc~100 is enough to reach ~1e-8 with speckles only, Rc~1000 would allows to go further (more conservative))
    filter_type        = "gaussian"                            # MM filter type
    table_type         = "Archive"                             # "Archive": for all known exoplanets | "Simulated": TODO

    # --- WFE and IWA ref values ---
    WFE_ref = 50.026   # [nm RMS]
    IWA_ref = 33.4 / 2 # [mas]

    # --- Separation range ---
    sep_min = 0     # [mas]
    sep_max = 1_000 # [mas] max separation of the raw PSF data

    # --- IFU parameters space to explore ---
    if instru_type == "IFU":
        # Size of the parameters space to explore (N**Ndim)
        N                = 10             # [dims]
        # Spectral resolution (for each R, it is assumed to be constant along the whole wavelength range)
        R_min            = Rc          # [dimensionless]
        R_max            = 200_000        # [dimensionless]
        # Bandwidth central wavelength (for each l0, the bandwidth is given by lmin,lmax = l0 +- dl*Nl/2, where dl depends on R and l0)
        l0_min           = 0.6            # [µm]
        l0_max           = 2.5            # [µm]
        # Number of spectral channel (number of effective bins sampling the data along the spectral dimension)
        Nl_min           = 1_00           # [bins]
        Nl_max           = 200_000        # [bins]

        # Post-AO wavefront error
        WFE_min          = 10             # [nm]
        WFE_max          = 200            # [nm]
        # Coronagraph inner working angle radius
        IWA_min          = 1              # [mas]
        IWA_max          = 100            # [mas]

        # # TODO: Fixed post-AO wavefront error and IWA (comment this passage to vary WFE and IWA, but huge files will be created)
        # # Post-AO wavefront error
        # WFE_min          = WFE_ref        # [nm]
        # WFE_max          = WFE_ref        # [nm]
        # # Coronagraph inner working angle
        # IWA_min          = IWA_ref        # [mas]
        # IWA_max          = IWA_ref        # [mas]

        # Instrumental transmission (without telescope transmission)
        trans_instru_min = 0.001           # [e-/ph]
        trans_instru_max = 0.5           # [e-/ph]
        # Effective residual halo modulation level:
        # For IFU: sigma_m is the effective fractional modulation of the final integrated stellar halo per spectral bin.
        # In practice, sigma_syst_base_2 stores the projected halo-squared term for one DIT,
        # already including throughput, PSF, FWHM integration, and DIT scaling.
        # The total systematic variance over the full exposure is then:
        # sigma_syst_tot^2 = sigma_m^2 * sigma_syst_base_2*N_DIT^2
        sigma_m_min      = 1e-7           # [dimensionless]
        sigma_m_max      = 1e-1           # [dimensionless]
        # Field of View
        FoV_min          = 10             # [mas]
        FoV_max          = 2*sep_max      # [mas]
        # Axis
        R                = np.logspace(np.log10(R_min),       np.log10(R_max),       N) # i dim
        l0               = np.linspace(l0_min,                l0_max,                N) # j dim
        Nl               = np.logspace(np.log10(Nl_min),      np.log10(Nl_max),      N) # k dim
        WFE              = np.linspace(WFE_min,               WFE_max,               N) # l dim
        IWA              = np.logspace(np.log10(IWA_min),     np.log10(IWA_max),     N) # m dim
        trans_instru     = np.linspace(trans_instru_min,      trans_instru_max,      N) # n dim
        sigma_m          = np.logspace(np.log10(sigma_m_min), np.log10(sigma_m_max), N) # o dim
        FoV              = np.logspace(np.log10(FoV_min),     np.log10(FoV_max),     N) # p dim
        # Nl must have int values
        Nl               = np.round(Nl).astype(int)
        # Axis must have unique and sorted values
        R                = np.unique(R)
        l0               = np.unique(l0)
        Nl               = np.unique(Nl)
        WFE              = np.unique(WFE)
        IWA              = np.unique(IWA)
        trans_instru     = np.unique(trans_instru)
        sigma_m          = np.unique(sigma_m)
        FoV              = np.unique(FoV)
        # Dimensions
        N_R              = len(R)
        N_l0             = len(l0)
        N_Nl             = len(Nl)
        N_WFE            = len(WFE)
        N_IWA            = len(IWA)
        N_TI             = len(trans_instru)
        N_SM             = len(sigma_m)
        N_FoV            = len(FoV)
        # Resolution for the raw models
        R_model          = R0_max

    # --- Imager parameters space to explore ---
    elif instru_type == "imager":
        # Forcing to Differential Imaging only
        post_processing  = "DI"
        # Size of the parameters space to explore (N**Ndim)
        N                = 10             # [dims]
        # Bandwidth central wavelength (for each l0, the bandwidth is given by lmin,lmax = l0 +- Dl/2)
        l0_min           = 0.6            # [µm]
        l0_max           = 2.5            # [µm]
        # Bandwidth width
        Dl_min           = 0.01           # [µm]
        Dl_max           = 0.2            # [µm]
        # Post-AO wavefront error
        WFE_min          = 10             # [nm]
        WFE_max          = 200            # [nm]
        # Coronagraph inner working angle radius
        IWA_min          = 1              # [mas]
        IWA_max          = 100            # [mas]
        
        # # TODO: Fixed post-AO wavefront error and IWA (comment this passage to vary WFE and IWA, but huge files will be created)
        # # Post-AO wavefront error
        # WFE_min          = WFE_ref        # [nm]
        # WFE_max          = WFE_ref        # [nm]
        # # Coronagraph inner working angle
        # IWA_min          = IWA_ref        # [mas]
        # IWA_max          = IWA_ref        # [mas]
        
        # Instrumental transmission (without telescope transmission)
        trans_instru_min = 0.001           # [e-/ph]
        trans_instru_max = 0.5           # [e-/ph]
        # Effective residual halo modulation level:
        # For Imager: sigma_m is the effective fractional modulation of the final integrated stellar halo integrated over the full exposure time, the FWHM and the considered band.
        # The total systematic variance over the full exposure is then:
        # sigma_syst_tot^2 = sigma_m^2 * halo_2*N_DIT^2
        sigma_m_min      = 1e-7       # [dimensionless]
        sigma_m_max      = 1e-1       # [dimensionless]
        # Field of View
        FoV_min          = 10         # [mas]
        FoV_max          = 2*sep_max  # [mas]
        # Axis
        l0               = np.linspace(l0_min,                l0_max,                N) # j dim
        Dl               = np.linspace(Dl_min,                Dl_max,                N) # k dim
        WFE              = np.linspace(WFE_min,               WFE_max,               N) # l dim
        IWA              = np.logspace(np.log10(IWA_min),     np.log10(IWA_max),     N) # m dim
        trans_instru     = np.linspace(trans_instru_min,      trans_instru_max,      N) # n dim
        sigma_m          = np.logspace(np.log10(sigma_m_min), np.log10(sigma_m_max), N) # o dim
        FoV              = np.logspace(np.log10(FoV_min),     np.log10(FoV_max),     N) # p dim
        # Axis must have unique and sorted values
        l0               = np.unique(l0)
        Dl               = np.unique(Dl)
        WFE              = np.unique(WFE)
        IWA              = np.unique(IWA)
        trans_instru     = np.unique(trans_instru)
        sigma_m          = np.unique(sigma_m)
        FoV              = np.unique(FoV)
        # Dimensions
        N_l0             = len(l0)
        N_Dl             = len(Dl)
        N_WFE            = len(WFE)
        N_IWA            = len(IWA)
        N_TI             = len(trans_instru)
        N_SM             = len(sigma_m)
        N_FoV            = len(FoV)
        # Resolution for the raw models
        R_model          = R0_min

    # --- Detector assignment as a function of wavelength ---
    if detector.startswith("constant_"):
        detector_l0 = np.full(N_l0, detector.replace("constant_", "", 1))
    elif detector == "PCS_visNIR_conservative":
        detector_l0 = np.where(l0 < 0.80, "VIS_CCD", "H4RG")
    elif detector == "PCS_low_noise_visNIR":
        detector_l0 = np.where(l0 < 0.80, "EMCCD", "H4RG")
    elif detector == "PCS_NIR_APD_optimistic":
        detector_l0 = np.where(l0 < 0.80, "EMCCD", "SAPHIRA")
    else:
        raise ValueError("detector must be 'constant_H4RG', 'PCS_visNIR_conservative', 'PCS_low_noise_visNIR', or 'PCS_NIR_APD_optimistic'.")
    spec_dict = {name: get_detector_specs(name) for name in np.unique(detector_l0)}
    RON0         = np.array([spec_dict[d]["RON0"]         for d in detector_l0], dtype=dtype)
    RON_lim      = np.array([spec_dict[d]["RON_lim"]      for d in detector_l0], dtype=dtype)
    DC0          = np.array([spec_dict[d]["DC0"]          for d in detector_l0], dtype=dtype)
    saturation_e = np.array([spec_dict[d]["saturation_e"] for d in detector_l0], dtype=dtype)
    min_DIT      = np.array([spec_dict[d]["min_DIT"]      for d in detector_l0], dtype=dtype)
    max_DIT      = np.array([spec_dict[d]["max_DIT"]      for d in detector_l0], dtype=dtype)

    # --- Pixel scales (assuming Nyquist sampling) ---
    l0D          = l0*1e-6 / D * 1000*rad2arcsec  # lambda/D [mas/(lambda/D)]
    FWHM0        = 1.029 * l0D                    # [mas/FWHM]
    pxscales     = FWHM0 / size_core              # [mas/px]
    pxscales_rad = pxscales / (rad2arcsec * 1000) # [rad/px]

    # Residuals modulations considered for the post_processing
    residuals = "Systematics" if post_processing == "MM" else "Speckles"
    
    # Getting labels
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)



    # ---------------------------
    # Getting data
    # ---------------------------

    # Retrieving the planet table
    planet_table = load_planet_table(f"{table_type}_Pull_For_FastYield.ecsv")
    N_PT_raw     = len(planet_table)

    # Filtering separation range
    planet_table       = planet_table[(planet_table["AngSep"].value >= sep_min) & (planet_table["AngSep"].value <= sep_max)]
    separation_planets = planet_table["AngSep"].value # [mas]
    N_PT               = len(planet_table)
    print(f"\nKeeping {N_PT}/{N_PT_raw} planets between {sep_min:.0f} and {sep_max:.0f} mas in thermal and reflected light")

    # Prints simulation parameters
    print_simulation_summary(instru=instru, instru_type=instru_type, detector=detector, post_processing=post_processing, D=D, S=S, N_mirror=N_mirror, trans_dust=trans_dust, RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, thermal_model=thermal_model, reflected_model=reflected_model, SNR_thr=SNR_thr, exposure_time=exposure_time, size_core=size_core, A_FWHM=A_FWHM, Rc=Rc if instru_type == "IFU" else None, filter_type=filter_type if instru_type == "IFU" else None, table_type=table_type, sep_min=sep_min, sep_max=sep_max, N_PT_raw=N_PT_raw, N_PT=N_PT, force_new_calc=force_new_calc, l0=l0, R=R if instru_type == "IFU" else None, Nl=Nl if instru_type == "IFU" else None, Dl=Dl if instru_type == "imager" else None, WFE=WFE, IWA=IWA, trans_instru=trans_instru, sigma_m=sigma_m, FoV=FoV)

    # Build meta for YOUR case (IFU or imager)
    meta = dict(code_version=PCS_CODE_VERSION, dtype=str(dtype),

                # instrument / telescope
                instru=instru, D=D, S=S, N_mirror=N_mirror, trans_dust=trans_dust, detector=detector, detector_l0=detector_l0,

                # detector numeric specs that affect DIT/noise/products
                RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT,

                # science / pipeline
                coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, thermal_model=thermal_model, reflected_model=reflected_model, instru_type=instru_type, post_processing=post_processing, size_core=size_core, R_model=R_model, table_type=table_type,

                # exploration space
                l0_min=l0_min, l0_max=l0_max, WFE_min=WFE_min, WFE_max=WFE_max, IWA_min=IWA_min, IWA_max=IWA_max, trans_instru_min=trans_instru_min, trans_instru_max=trans_instru_max, sep_min=sep_min, sep_max=sep_max, N_PT=N_PT,

                # axis used
                l0=l0, WFE=WFE, IWA=IWA, trans_instru=trans_instru,
                )

    if instru_type == "IFU":
        meta.update(dict(Rc=Rc, filter_type=filter_type, R=R, Nl=Nl,))
    elif instru_type == "imager":
        meta.update(dict(Dl=Dl,))
    else:
        raise ValueError("instru_type must be 'IFU' or 'imager'.")
    suffix, meta_clean, _payload = make_suffix(meta, n=16)
    write_meta(sim_dir, suffix, meta_clean)
    print("\nFastYield data files suffix:", suffix)

    # File paths derived from suffix
    signal_path                 = sim_dir / f"signal_{suffix}.npy"
    sigma_halo_2_path           = sim_dir / f"sigma_halo_2_{suffix}.npy"
    sigma_bkg_2_path            = sim_dir / f"sigma_bkg_2_{suffix}.npy"
    DIT_path                    = sim_dir / f"DIT_{suffix}.npy"
    if instru_type == "IFU":
        sigma_syst_base_2_path  = sim_dir / f"sigma_syst_base_2_{suffix}.npy"
    else:
        sigma_syst_base_2_path  = None
    PSF_profile_5D_tmp_path     = sim_dir / f"tmp_PSF_profile_5D_{suffix}.npy"
    fraction_core_5D_tmp_path   = sim_dir / f"tmp_fraction_core_5D_{suffix}.npy"
    PSF_profile_max_4D_tmp_path = sim_dir / f"tmp_PSF_profile_max_4D_{suffix}.npy"

    for p in [PSF_profile_5D_tmp_path, fraction_core_5D_tmp_path, PSF_profile_max_4D_tmp_path]:
        if p.exists():
            p.unlink()

    do_new = force_new_calc or (not signal_path.exists()) or (not sigma_halo_2_path.exists()) or (not sigma_bkg_2_path.exists()) or (not DIT_path.exists())
    if instru_type == "IFU":
        do_new = do_new or (not sigma_syst_base_2_path.exists())



    ####################
    # Opening existing #
    ####################
    if not do_new:
        # For IFU (7D):    (Planets, R, l0, Nl, WFE, IWA, trans_instru)
        # For Imager (6D): (Planets,    l0, Dl, WFE, IWA, trans_instru)
        signal_planets       = np.load(signal_path,       mmap_mode="r") # [e-/FWHM/DIT]
        sigma_halo_2_planets = np.load(sigma_halo_2_path, mmap_mode="r") # [e-/FWHM/DIT]**2
        sigma_bkg_2_planets  = np.load(sigma_bkg_2_path,  mmap_mode="r") # [e-/FWHM/DIT]**2
        DIT_planets          = np.load(DIT_path,          mmap_mode="r") # [mn/DIT]
        if instru_type == "IFU":
            sigma_syst_base_2_planets = np.load(sigma_syst_base_2_path, mmap_mode="r") # [e-/FWHM/DIT]**2 (at sigma_m = 1)
        elif instru_type == "imager":
            sigma_syst_base_2_planets = None
        print("\nOpening existing FastYield data...")



    ####################################
    # Otherwise doing the calculations #
    ####################################
    else:
        if force_new_calc:
            print(f"\nNew FastYield ELT/{instru} calculations (forcing_new_calc=True).")
        else:
            print(f"\nNew FastYield ELT/{instru} calculations (missing files).")

        #
        # PSF data (PSF profile, fraction core and coronagraph transmission) preparations (l0, WFE, IWA, mag_star, sep)
        #

        # PSF_profile_density (no coronagraph) = mean surface-brightness density in each annular bin divided by the total flux (as function of angular separation)
        # fraction_core       (no coronagraph) = total flux enclosed in the core divided by the total flux                     (constant with separation)

        # PSF_profile_density (coronagraph) = mean surface-brightness density in each annular bin divided by the total coronagraphic flux (as function of angular separation)
        # fraction_core       (coronagraph) = total flux enclosed in the core divided by the total coronagraphic flux                     (for different offset angular separation from the coronagraph, at 0 => on-axis)
        # radial_transmission (coronagraph) = total coronagraphic flux divided by total flux                                              (for different offset angular separation from the coronagraph, at 0 => on-axis)

        # Retrieving PSF data and axis (HARD CODED RANGES...)
        l0_min_PSF              = 0.6   # [µm]
        l0_max_PSF              = 2.5   # [µm]
        WFE_min_PSF             = 10    # [nm]
        WFE_max_PSF             = 200   # [nm]
        IWA_min_PSF             = 1     # [mas]
        IWA_max_PSF             = 100   # [mas]
        mag_star_min_PSF        = 0     # [dimensionless]
        mag_star_max_PSF        = 10    # [dimensionless]
        sep_max_PSF             = 1_000 # [mas]
        suffix_PSF              = f"{apodizer}_{coronagraph}_{strehl}_{l0_min_PSF}_{l0_max_PSF}_{WFE_min_PSF}_{WFE_max_PSF}_{IWA_min_PSF}_{IWA_max_PSF}_{mag_star_min_PSF}_{mag_star_max_PSF}_{sep_max_PSF}"
        wave0                   = fits.getdata(psf_dir / f"{instru}_wave_{suffix_PSF}.fits")
        WFE0                    = fits.getdata(psf_dir / f"{instru}_WFE_{suffix_PSF}.fits")
        IWA0                    = fits.getdata(psf_dir / f"{instru}_IWA_{suffix_PSF}.fits")
        mag_star0               = fits.getdata(psf_dir / f"{instru}_mag_star_{suffix_PSF}.fits")
        separation0             = fits.getdata(psf_dir / f"{instru}_separation_{suffix_PSF}.fits")
        PSF_profile_density0_5D = fits.getdata(psf_dir / f"{instru}_PSF_profile_density_5D_{suffix_PSF}.fits")
        fraction_core0_5D       = fits.getdata(psf_dir / f"{instru}_fraction_core_5D_{suffix_PSF}.fits")
        if coronagraph is not None:
            radial_transmission0_5D = fits.getdata(psf_dir / f"{instru}_radial_transmission_5D_{suffix_PSF}.fits")
        else:
            radial_transmission0_5D = None

        # Keeping the raw mag_star and separation axis
        mag_star   = mag_star0.astype(dtype,   copy=False)
        separation = separation0.astype(dtype, copy=False)
        N_mag_star = len(mag_star)
        N_sep      = len(separation)

        # Temporary PSF memmaps
        PSF_profile_5D     = create_memmap_with_log(PSF_profile_5D_tmp_path,     shape=(N_l0, N_WFE, N_IWA, N_mag_star, N_sep), dtype=dtype, mode="w+")
        fraction_core_5D   = create_memmap_with_log(fraction_core_5D_tmp_path,   shape=(N_l0, N_WFE, N_IWA, N_mag_star, N_sep), dtype=dtype, mode="w+")
        PSF_profile_max_4D = create_memmap_with_log(PSF_profile_max_4D_tmp_path, shape=(N_l0, N_WFE, N_IWA, N_mag_star),        dtype=dtype, mode="w+",)

        # Build interpolators once
        eps_pos  = np.finfo(dtype).tiny
        eps01    = 1e-32
        safe_pos = lambda x: np.clip(x, eps_pos, None)
        safe01   = lambda x: np.clip(x, eps01, 1 - eps01)
        log_interp_psf      = RegularGridInterpolator((wave0, WFE0, IWA0, mag_star0, separation0), np.log(safe_pos(PSF_profile_density0_5D)), bounds_error=True)
        logit_interp_core   = RegularGridInterpolator((wave0, WFE0, IWA0, mag_star0, separation0), logit(safe01(fraction_core0_5D)),          bounds_error=True)
        if coronagraph is not None:
            logit_interp_rt = RegularGridInterpolator((wave0, WFE0, IWA0, mag_star0, separation0), logit(safe01(radial_transmission0_5D)),    bounds_error=True)
        else:
            logit_interp_rt = None

        # Build the temporary PSF memmaps by chunks along l0
        l0_chunk = max(1, min(chunksize, N_l0))
        for j0 in tqdm(range(0, N_l0, l0_chunk), desc="Building temporary PSF memmaps"):
            j1           = min(j0 + l0_chunk, N_l0)
            l0_chunk_arr = l0[j0:j1]
            pts          = np.stack(np.meshgrid(l0_chunk_arr, WFE, IWA, mag_star, separation, indexing="ij"), axis=-1)

            # Interpolating over current grid
            PSF_profile_density_blk     = np.exp(log_interp_psf(pts))
            fraction_core_blk           = expit(logit_interp_core(pts))
            if coronagraph is not None:
                radial_transmission_blk = expit(logit_interp_rt(pts))
            else:
                radial_transmission_blk = None

            # Sanity check of the values
            if (np.any(~np.isfinite(PSF_profile_density_blk)) or np.any(~np.isfinite(fraction_core_blk)) or (coronagraph is not None and np.any(~np.isfinite(radial_transmission_blk)))):
                raise KeyError("The PSF data should only have finite values!")

            # Converting flux density in fraction of flux inside the FOV per spaxel for a bin at l0, WFE, IWA, mag_star as function of separation
            PSF_profile_blk = PSF_profile_density_blk * pxscales[j0:j1, None, None, None, None]**2 # [star flux fraction/px]

            # Star transmission throught coronagraph, given at sep = 0, on-axis, assuming that the coronagraph is perfectly align with the star
            # star_transmission = total star flux through coronagraph / total star flux
            if coronagraph is not None:
                i_sep0 = 0
                if separation[i_sep0] != 0.0:
                    raise KeyError("The first value of 'separation' should be 0.")
                star_transmission_blk = radial_transmission_blk[..., i_sep0] # (l0c, WFE, IWA, mag_star)
                # Star PSF profile throught the coronagraph
                PSF_profile_blk      *= star_transmission_blk[..., None]     # (l0c, WFE, IWA, mag_star, sep)
                # Fraction core signal (for the planet, inside the FWHM) through the coronagraph
                fraction_core_blk    *= radial_transmission_blk              # (l0c, WFE, IWA, mag_star, sep)

            PSF_profile_5D[j0:j1]   = PSF_profile_blk.astype(dtype,   copy=False)
            fraction_core_5D[j0:j1] = fraction_core_blk.astype(dtype, copy=False)

            # PSF_max (brightest pixels in the PSF) [max star flux fraction/px] in order to compute the DIT length to reach saturation
            PSF_profile_max_4D[j0:j1] = np.max(PSF_profile_blk, axis=-1).astype(dtype, copy=False)

            # Free block temporaries immediately
            del pts
            del PSF_profile_density_blk
            del fraction_core_blk
            del PSF_profile_blk
            if radial_transmission_blk is not None:
                del radial_transmission_blk
                del star_transmission_blk
            gc.collect()

        # PSF_profile_5D(l0, WFE, IWA, mag_star, separation)   = [star flux fraction/px]     = star flux per px divided by the total star flux          (as function of angular separation)
        # fraction_core_5D(l0, WFE, IWA, mag_star, separation) = [planet flux fraction/FWHM] = planet flux inside FWHM divided by the total planet flux (constant with separation if no coronagraph, or for different angular separation from the coronagraph)
        # PSF_profile_max_4D(l0, WFE, IWA, mag_star)           = [max star flux fraction/px] = max star flux per px divided by the total star flux      (given at the brightest position)

        PSF_profile_5D.flush()
        fraction_core_5D.flush()
        PSF_profile_max_4D.flush()

        # Free raw PSF arrays as soon as temp memmaps are built
        del wave0, WFE0, IWA0, mag_star0, separation0
        del PSF_profile_density0_5D, fraction_core0_5D
        if radial_transmission0_5D is not None:
            del radial_transmission0_5D
        del log_interp_psf, logit_interp_core, logit_interp_rt
        del PSF_profile_5D, fraction_core_5D, PSF_profile_max_4D
        gc.collect()



        #
        # Spectra preparations
        #

        # K-band for photometry (for star spectra normalizations using K-band magnitudes)
        wave_K = get_wave_K() # [µm]

        # Vega spectrum on K-band in [J/s/m2/µm]
        vega_spectrum   = load_vega_spectrum()                                       # [J/s/m2/µm]
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False) # [J/s/m2/µm]
        counts_vega_K   = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)

        # Vega zeropoints [J/s/m2/µm] (for star magnitues computations)
        vega_flux_1D = vega_spectrum.interpolate_wavelength(l0, renorm=False).flux # [J/s/m2/µm]

        # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotationnal broadening with Vsini)
        lmin_model = 0.98*l0_min                                 # [µm] a bit larger for doppler shifts and to avoid edge effects
        lmax_model = 1.02*l0_max                                 # [µm] a bit larger for doppler shifts and to avoid edge effects
        dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
        wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)

        # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
        wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
        dl_instru   = np.gradient(wave_instru)                                                    # [µm/bin] Nyquist sampling of a spectrum with resolving power R_model: 2 samples per resolution element across the whole axis

        # Effective model range
        lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin
        lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax

        # Tellurics transmission spectrum (from SkyCalc)
        wave_tell, trans_tell = load_tell_trans(airmass=1.0) # at airmass = 1.0
        airmass               = 1.2                          # 1.2 to be coherent with the used background skytable
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell**airmass).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1]))

        # Telescope transmission spectrum
        data_elt_ref = np.genfromtxt(instru_dir / "ELT_reflectivity.csv", delimiter=",", names=True, dtype=float, encoding=None)
        wave_ref     = data_elt_ref["wave"] # [µm]
        elt_ref      = data_elt_ref["M1M5"] # ELT mirror train reflectivity, from common ICD, section 4.11, p37 (Document Number: ESO-253082)
        trans_tel    = trans_dust * elt_ref
        trans_tel    = Spectrum(wavelength=wave_ref, flux=trans_tel).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tel[0], trans_tel[-1]))

        # Tellurics x telescope transmission spectrum
        trans_tell_tel = Spectrum(wavelength=wave_instru, flux=trans_tell.flux*trans_tel.flux, R=np.fmax(trans_tell.R, trans_tel.R), T=None, lg=None, model=None, rv=None, vsini=None, sigma=None)

        # Plotting transmissions
        plot_trans_tell_tel(trans_tell=trans_tell, trans_tel=trans_tel)

        # Background spectrum [ph/mas2/mn/µm] (from SkyCalc)
        plot_bkg_skycalc(filename=instru_dir / "skytable_background.fits")
        data_bkg         = fits.getdata(instru_dir / "skytable_background.fits")
        wave_bkg         = data_bkg["lam"]  * 1e-3             # [nm] => [µm]
        background       = data_bkg["flux"] * 60 * S / 1000**2 # [ph/s/m2/µm/arcsec2] => [ph/mas2/mn/µm]
        background       = Spectrum(wavelength=wave_bkg, flux=background).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(background[0], background[-1])) # [ph/mas2/mn/µm]
        background.flux *= trans_tel.flux # [ph/mas2/mn/µm] through the telescope



        #
        # IFU path
        #
        if instru_type == "IFU":

            # --- Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/µm] ---
            scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
            scale_spectrum *= S * 60                   # [ph/s/m2/µm] => [ph/mn/µm]

            # --- Preparing for wave axis, range, trans_tell_tel, background for each point ---
            trans_tell_tel_1D_list = [None] * len(R) # (R)
            background_1D_list     = [None] * len(R) # (R)     [ph/px/mn/bin]
            wave_1D_list           = [None] * len(R) # (R)     [µm]
            dwave_1D_list          = [None] * len(R) # (R)     [µm/bin]
            range_1D_list          = [None] * len(R) # (R)
            l0_2D                  = l0[:, None]     # (l0, 1) [µm]
            Nl_2D                  = Nl[None, :]     # (1, Nl) [px]
            for i, res in enumerate(R):

                # --- Build a wavelength grid with *constant resolving power* R = res ---
                # For a spectrograph with resolving power R, one resolution element is:
                #   Δλ_res = λ / R
                # Nyquist sampling requires 2 samples per resolution element, hence a pixel step:
                #   Δλ_pix ≈ (λ / R) / 2  =>  Δλ_pix / λ ≈ 1 / (2R)
                # This means the natural uniform grid is in ln(λ), with constant step: dln(λ)/dλ = 1/λ => dln(λ) = dλ/λ = 1 / (2R)
                #   Δln(λ) = ln(λ_{i+1}) - ln(λ_i) ≈ Δλ_pix / λ ≈ 1/(2R) = constant
                dln = 1.0 / (2.0 * res)

                # Number of samples needed to cover [lmin_model, lmax_model] with that constant Δln(λ)
                # ln(λ_i) = ln(λ_min) + k*Δln(λ) => λ_i = λ_min * exp(k*Δln(λ))
                # If we have n points, it means that we have n - 1 between l0_min and l0_max
                # because: ln(l0_max/l0_min) = (n-1) * dln  =>  n ≈ ln(l0_max/l0_min)/dln + 1
                n = int(np.floor(np.log(lmax_model / lmin_model) / dln)) + 1

                # Log-uniform wavelength grid: λ_i = λ_min * exp(i * Δln(λ))
                # This keeps λ/Δλ (i.e., R) approximately constant across the whole band.
                wave_res         = lmin_model * np.exp(np.arange(n) * dln) # [µm]
                dwave_res        = np.gradient(wave_res)                   # [µm/bin]
                wave_1D_list[i]  = wave_res                                # [µm]
                dwave_1D_list[i] = dwave_res                               # [µm/bin]

                # --- Build, for each (l0, Nl), the index range of a contiguous spectral window ---
                # On a log-λ grid, a window of Nl pixels corresponds to a multiplicative span in λ.
                # Half-window in ln(λ) is: half = (Nl/2) * Δln(λ)
                half = (Nl_2D - 1) * dln / 2

                # Convert the ±half span in ln(λ) into wavelength bounds around the central wavelength λ0:
                l0_2D_min = l0_2D * np.exp(-half) # (l0, Nl) [µm]
                l0_2D_max = l0_2D * np.exp(+half) # (l0, Nl) [µm]

                # Find the corresponding index interval [idx_lo, idx_hi) in the wave_res grid.
                # Using side="left"/"right" ensures we include the full requested wavelength span.
                idx_lo           = np.searchsorted(wave_res, l0_2D_min, side="left").astype(np.int32)
                idx_hi           = np.searchsorted(wave_res, l0_2D_max, side="right").astype(np.int32)
                idx_lo           = np.clip(idx_lo, 0, len(wave_res)-1)
                idx_hi           = np.clip(idx_hi, 1, len(wave_res))
                idx_hi           = np.maximum(idx_hi, idx_lo + 1)
                range_1D_list[i] = (idx_lo, idx_hi)

                # Tellurics x Telescope spectrum on the grid at this resolution
                trans_tell_tel_1D_list[i] = trans_tell_tel.degrade_resolution(wave_res, renorm=False, R_output=res).flux

                # Background spectrum on this grid in [ph/px/mn/bin]
                background_res        = background.degrade_resolution(wave_res, renorm=False, R_output=res).flux # (len(wave_res))     [ph/mn/µm/mas2]
                background_res       *= dwave_res                                                                # (len(wave_res))     [ph/mn/µm/mas2]  => [ph/mn/bin/mas2]
                background_1D_list[i] = background_res.astype(dtype, copy=False)                                 # (len(wave_res))     [ph/mn/bin/mas2]


            # --- Global context worker ---
            _IFU_CTX = dict(planet_table=planet_table, l0=l0, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_model=wave_model, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, vega_flux_1D=vega_flux_1D, mag_star=mag_star, scale_spectrum=scale_spectrum, N_R=N_R, N_l0=N_l0, N_Nl=N_Nl, R=R, wave_1D_list=wave_1D_list, dwave_1D_list=dwave_1D_list, trans_tell_tel_1D_list=trans_tell_tel_1D_list, trans_instru=trans_instru, background_1D_list=background_1D_list, pxscales=pxscales, Rc=Rc, filter_type=filter_type, range_1D_list=range_1D_list, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_5D_path=str(PSF_profile_5D_tmp_path), fraction_core_5D_path=str(fraction_core_5D_tmp_path), PSF_profile_max_4D_path=str(PSF_profile_max_4D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), sigma_syst_base_2_path=str(sigma_syst_base_2_path), dtype=dtype, D=D, pxscales_rad=pxscales_rad)


            # --- Arguments for Pool ---
            init_worker  = init_worker_IFU
            _CTX         = _IFU_CTX
            func_process = process_IFU
            shape_SNR    = (N_PT, N_R, N_l0, N_Nl, N_WFE, N_IWA, N_TI)
            tqdm_desc    = f"ELT/{instru} ({instru_type} and {post_processing}) multiprocessing over planets of signal, stellar halo photon noise, DIT length and base for {residuals} noise"


        #
        # Imager path
        #
        elif instru_type == "imager":

            # --- Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/bin] ---
            scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
            scale_spectrum *= S * 60 * dl_instru       # [ph/s/m2/µm] => [ph/mn/bin]

            # Preparing for bandwidth range for each point
            l0_2D  = l0[:, None]        # [µm] (l0, 1)
            Dl_2D  = Dl[None, :]        # [µm] (1, Dl)
            l0_2D_min = l0_2D - Dl_2D/2 # [µm] (l0, Dl)
            l0_2D_max = l0_2D + Dl_2D/2 # [µm] (l0, Dl)

            # Find the corresponding index interval [idx_lo, idx_hi) in the wave_res grid.
            # Using side="left"/"right" ensures we include the full requested wavelength span.
            idx_lo   = np.searchsorted(wave_instru, l0_2D_min, side="left").astype(np.int32)
            idx_hi   = np.searchsorted(wave_instru, l0_2D_max, side="right").astype(np.int32)
            idx_lo   = np.clip(idx_lo, 0, len(wave_instru)-1)
            idx_hi   = np.clip(idx_hi, 1, len(wave_instru))
            idx_hi   = np.maximum(idx_hi, idx_lo + 1)
            range_1D = (idx_lo, idx_hi)

            # Background total flux preparation (l0, Dl) in [ph/px/mn]
            background          = background.flux * dl_instru                                                   # [ph/mn/µm/mas2] => [ph/mn/bin/mas2]
            background_flux_2D  = window_sums_1d(values=background, idx_lo=idx_lo, idx_hi=idx_hi).astype(dtype) # [ph/mn/mas2]
            background_flux_2D *= pxscales[:, None]**2                                                          # [ph/mn/mas2] => [ph/px/mn]

            # --- Global context worker ---
            _IM_CTX = dict(planet_table=planet_table, l0=l0, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, vega_flux_1D=vega_flux_1D, mag_star=mag_star, scale_spectrum=scale_spectrum, N_l0=N_l0, N_Dl=N_Dl, trans_instru=trans_instru, trans_tell_tel=trans_tell_tel.flux, background_flux_2D=background_flux_2D, range_1D=range_1D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_5D_path=str(PSF_profile_5D_tmp_path), fraction_core_5D_path=str(fraction_core_5D_tmp_path), PSF_profile_max_4D_path=str(PSF_profile_max_4D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), dtype=dtype)

            # --- Arguments for Pool ---
            init_worker  = init_worker_IM
            _CTX         = _IM_CTX
            func_process = process_IM
            shape_SNR    = (N_PT, N_l0, N_Dl, N_WFE, N_IWA, N_TI)
            tqdm_desc    = f"ELT/{instru} ({instru_type} and {post_processing}) multiprocessing over planets of signal, stellar halo photon noise and DIT length"



        #
        # Parallelized calculations for each planet
        #

        # Creating files with size logging
        file_specs = [("signal",       signal_path,       shape_SNR, dtype),
                      ("sigma_halo_2", sigma_halo_2_path, shape_SNR, dtype),
                      ("sigma_bkg_2",  sigma_bkg_2_path,  shape_SNR, dtype),
                      ("DIT",          DIT_path,          shape_SNR, dtype),]
        if instru_type == "IFU":
            file_specs.append(("sigma_syst_base_2", sigma_syst_base_2_path, shape_SNR, dtype))
        total_nbytes = sum(memmap_nbytes(shape, dtype) for _, _, shape, dtype in file_specs)
        print(f"\nCreating memmap files in {sim_dir}")
        print(f"Total expected disk usage: {format_nbytes(total_nbytes)}")
        print("---------------------------------------------------------")
        print()
        signal_planets       = create_memmap_with_log(signal_path,       shape_SNR, dtype=dtype, mode="w+")
        sigma_halo_2_planets = create_memmap_with_log(sigma_halo_2_path, shape_SNR, dtype=dtype, mode="w+")
        sigma_bkg_2_planets  = create_memmap_with_log(sigma_bkg_2_path,  shape_SNR, dtype=dtype, mode="w+")
        DIT_planets          = create_memmap_with_log(DIT_path,          shape_SNR, dtype=dtype, mode="w+")
        if instru_type == "IFU":
            sigma_syst_base_2_planets = create_memmap_with_log(sigma_syst_base_2_path, shape_SNR, dtype=dtype, mode="w+")
        elif instru_type == "imager":
            sigma_syst_base_2_planets = None

        # Creating context
        if sys.platform.startswith(("win", "darwin")): # Windows, MACOS
            ctx_mp = mp.get_context("spawn")
        else:
            ctx_mp = mp.get_context("fork") # Linux

        # Warmup for systematics in IFU
        if instru_type == "IFU":
            warmup_signal_noise_numba(dtype=dtype)

        # Computations
        print()
        with ctx_mp.Pool(processes=nproc, initializer=init_worker, initargs=(_CTX,)) as pool:
            print(f'{tqdm_desc}:')
            for _ in tqdm(pool.imap_unordered(func_process, range(N_PT), chunksize=chunksize), total=N_PT):
                pass
        signal_planets.flush()
        sigma_halo_2_planets.flush()
        sigma_bkg_2_planets.flush()
        DIT_planets.flush()
        if instru_type == "IFU":
            sigma_syst_base_2_planets.flush()

        for p in [PSF_profile_5D_tmp_path, fraction_core_5D_tmp_path, PSF_profile_max_4D_tmp_path]:
            if p.exists():
                p.unlink()



    # %%
    # Plot general parameters
    dpi                = 72                  # Dots Per Inch
    fontsize           = 22                  # Standard fontsize
    ms                 = 15                  # Marker size
    ss                 = 400                 # Scatter size
    lw                 = 2                   # Linewidth
    alpha              = 0.9                 # Opacity
    gain               = True                # True for probability gain (normalized yield), False for absolute yields
    light_regime_plot  = "thermal+reflected" # 'thermal", "reflected" or "thermal+reflected"
    band_regime_plot   =  "H"                # Band where the thermal/reflected domination regime is splitted
    ptypes             = ["Jupiter",                 "Saturn",                "Neptune",                 "Earth"]
    marker_ptypes      = {"Jupiter": "s",            "Saturn": "v",           "Neptune": "P",            "Earth": "o"}
    label_ptypes       = {"Jupiter": "Jupiter-like", "Saturn": "Saturn-like", "Neptune": "Neptune-like", "Earth": "Earth-like"}
    
    # Define parameters and their names
    if instru_type == "IFU":
        params         = [R,                           l0,                                                Nl,                                           WFE,                           IWA,                        trans_instru,                                    100*sigma_m,                                         FoV]                    # params axis
        params_ranges  = [(R.min(), R.max()),          (l0.min(), l0.max()),                              (Nl.min(), Nl.max()),                         (WFE_ref, WFE_ref),            (IWA_ref, IWA_ref),         (trans_instru.min(), trans_instru.max()),        ((100*sigma_m).min(), (100*sigma_m).min()),          (FoV.max(), FoV.max())] # params ranges for marginalization
        params_priors  = ["log-uniform",               "uniform",                                         "log-uniform",                                "uniform",                     "log-uniform",              "uniform",                                       "log-uniform",                                       "log-uniform"]          # params priors for marginalization
        params_islog   = [True,                        False,                                             True,                                         False,                         True,                       False,                                           True,                                                True]                   # params space (linspace or logspace)
        params_isint   = [True,                        False,                                             True,                                         False,                         False,                      False,                                           False,                                               True]                   # params rounded format
        params_names   = ["R",                         "l0 [µm]",                                         "Nl",                                         "WFE [nm]",                    "IWA [mas]",                "trans_instru [e-/ph]",                          "sigma_m [%]",                                       "FoV [mas]"]            # params names
        params_names_l = [r"$R$",                      r"$\lambda_0$  [µm]",                              r"$N_{\lambda}$",                             r"$WFE$ [nm]",                r"IWA [mas]",                r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                  r"$FoV$ [mas]"]          # params labels
        params_names_L = [r"Spectral resolution $R$",  r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Number of spectral channel $N_{\lambda}$",  r"Post-AO ${WFE}$ [nm RMS]",  r"Coronagraph $IWA$ [mas]",  r"Instrumental transmission $\gamma_{instru}$",  rf"{residuals} residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]"]  # params detailed labels

    elif instru_type == "imager":
        params         = [l0,                                                Dl,                                         WFE,                                          IWA,                                                 trans_instru,                                    100*sigma_m,                                        FoV]                    # params axis
        params_ranges  = [(l0.min(), l0.max()),                              (Dl.min(), Dl.max()),                       (WFE_ref, WFE_ref),                           (IWA_ref, IWA_ref),                                  (trans_instru.min(), trans_instru.max()),        ((100*sigma_m).min(), (100*sigma_m).min()),         (FoV.max(), FoV.max())] # params ranges for marginalization
        params_priors  = ["uniform",                                         "uniform",                                  "uniform",                                    "log-uniform",                                      "uniform",                                       "log-uniform",                                       "log-uniform"]          # params priors for marginalization
        params_islog   = [False,                                             False,                                      False,                                        True,                                                False,                                           True,                                                True]                  # params space (linspace or logspace)
        params_isint   = [False,                                             False,                                      False,                                        False,                                               False,                                           False,                                               False]                 # params rounded format
        params_names   = ["l0 [µm]",                                         "Dl [µm]",                                  "WFE [nm]",                                   "IWA [mas]",                                         "trans_instru",                                  "sigma_m [%]",                                       "FoV"]                 # params labels
        params_names_l = [r"$\lambda_0$ [µm]",                               r"$\Delta\lambda$ [µm]",                    r"$WFE$ [nm]",                                r"$IWA$ [mas]",                                      r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                  r"FoV [mas]"]           # params labels
        params_names_L = [r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Spectral coverage $\Delta\lambda$ [µm]",  r"Post-AO wavefront error ${WFE}$ [nm RMS]",  r"Coronagraph focal plane mask radius $IWA$ [mas]",  r"Instrumental transmission $\gamma_{instru}$",  rf"{residuals} residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]"] # params detailed labels
    
    
    
    # %%
    # COMPUTING MAIN QUANTITIES

    # Computing the SNR and detection mask (SNR > SNR_thr) for each planet
    # For IFU (8D):    (planets, R, l0, Nl, WFE, IWA, trans_instru, sigma_m)
    # For Imager (7D): (planets,    l0, Dl, WFE, IWA, trans_instru, sigma_m)
    SNR_planets     = get_SNR(instru=instru, instru_type=instru_type, post_processing=post_processing, exposure_time=exposure_time, min_DIT=min_DIT, RON0=RON0, RON_lim=RON_lim, DC0=DC0, A_FWHM=A_FWHM, Rc=Rc, filter_type=filter_type, signal_planets=signal_planets, sigma_halo_2_planets=sigma_halo_2_planets, sigma_bkg_2_planets=sigma_bkg_2_planets, sigma_syst_base_2_planets=sigma_syst_base_2_planets, DIT_planets=DIT_planets, R=R if instru_type == "IFU" else None, Nl=Nl if instru_type == "IFU" else None, sigma_m=sigma_m, sim_dir=sim_dir, suffix=suffix, dtype=dtype)
    mask_detections = get_mask_detections(SNR_planets=SNR_planets, SNR_thr=SNR_thr, sim_dir=sim_dir, suffix=suffix)

    # Precompute FoV gating once
    p0_FoV = np.searchsorted(FoV / 2, separation_planets, side="left").astype(np.int32)

    # Getting ndim detection probability
    # For IFU (8D):    (R, l0, Nl, WFE, IWA, trans_instru, sigma_m, FoV)
    # For Imager (7D): (   l0, Dl, WFE, IWA, trans_instru, sigma_m, FoV)
    Pdet = get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype)

    # --- Removing useless dimensions ---
    # Identify dimensions to remove (if size == 1)
    orig_names    = params_names.copy()
    orig_sizes    = [len(param) for param in params]
    dim_to_remove = [idim for idim, size in enumerate(orig_sizes) if size == 1]
    dim_to_keep   = [idim for idim, size in enumerate(orig_sizes) if size != 1]
    print("\nPdet dimensions reduction:")
    for name, size in zip(orig_names, orig_sizes):
        flag = "remove" if size == 1 else "keep"
        print(f"  - {name:<20} : {size:>3}   [{flag}]")
    # Actual reduction
    for idim in sorted(dim_to_remove, reverse=True):
        Pdet = np.take(Pdet, 0, axis=idim)  # or np.squeeze(Pdet, axis=idim)
        params.pop(idim)
        params_ranges.pop(idim)
        params_priors.pop(idim)
        params_islog.pop(idim)
        params_isint.pop(idim)
        params_names.pop(idim)
        params_names_l.pop(idim)
        params_names_L.pop(idim)
    Ndim  = len(params)
    ncols = min(2, Ndim)
    nrows = int(np.ceil(Ndim / ncols))
    
    # Plot order: display WFE and IWA in the last panels when present
    dims_last  = [next((idim for idim, name in enumerate(params_names) if key in name), None) for key in ("WFE", "IWA")]
    dims_last  = [idim for idim in dims_last if idim is not None]
    plot_order = [idim for idim in range(Ndim) if idim not in dims_last] + dims_last
    
    
    
    # %%
    # Thermal/Reflected regime split of the planet table
    # Considering both thermal and reflected contribution
    if thermal_model != "None" and reflected_model != "None":
        mag_th   = np.asarray(planet_table[f"Planet{band_regime_plot}mag(thermal)"],   dtype=float)
        mag_re   = np.asarray(planet_table[f"Planet{band_regime_plot}mag(reflected)"], dtype=float)
        valid_th = np.isfinite(mag_th)
        valid_re = np.isfinite(mag_re)
        # Lower magnitude = brighter contribution.
        mask_thermal   = valid_th & valid_re & (mag_th < mag_re)
        mask_reflected = valid_th & valid_re & (mag_re < mag_th)
    # Thermal contribution only
    elif thermal_model != "None" and reflected_model == "None":
        light_regime_plot = "thermal"
        mask_thermal      = np.full(N_PT, True)
        mask_reflected    = ~mask_thermal
    # Reflected contribution only
    elif thermal_model == "None" and reflected_model != "None":
        light_regime_plot = "reflected"
        mask_reflected    = np.full(N_PT, True)
        mask_thermal      = ~mask_reflected
    else:
        raise ValueError("At least one of thermal_model or reflected_model must be different from 'None'.")
    mask_unknown   = ~(mask_thermal | mask_reflected)
    print(f"\nPlanet-light regime split in {band_regime_plot}-band:")
    print(f"                  => Thermal-dominated:   {mask_thermal.sum()}/{len(planet_table)}")
    print(f"                  => Reflected-dominated: {mask_reflected.sum()}/{len(planet_table)}")
    print(f"                  => Unknown regime:      {mask_unknown.sum()}/{len(planet_table)}")
    
    # DETECTION PROBABILITY PER TYPE AND REGIME
    N_PT_ptypes_thermal   = np.zeros((len(ptypes)))
    N_PT_ptypes_reflected = np.zeros((len(ptypes)))
    Pdet_ptypes_thermal   = [None] * len(ptypes)
    Pdet_ptypes_reflected = [None] * len(ptypes)
    mask_ptype_all        = np.full(len(planet_table), False) 
    for ipt, ptype in enumerate(ptypes):
        mask_ptype                 = get_mask_planet_type(planet_table=planet_table, planet_type=ptype)
        mask_ptype_all            |= mask_ptype
        N_PT_ptypes_thermal[ipt]   = (mask_ptype & mask_thermal).sum()
        N_PT_ptypes_reflected[ipt] = (mask_ptype & mask_reflected).sum()
        if "thermal" in light_regime_plot:
            print(f"\n{label_ptypes[ptype]} in thermal light:")
            idx_ptype_thermal  = np.where(mask_ptype & mask_thermal)[0]
            Pdet_ptype_thermal = get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, indices=idx_ptype_thermal, jump_before_tqdm=False)
        else:
            Pdet_ptype_thermal = None
        if "reflected" in light_regime_plot:
            print(f"\n{label_ptypes[ptype]} in reflected light:")
            idx_ptype_reflected  = np.where(mask_ptype & mask_reflected)[0]
            Pdet_ptype_reflected = get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, indices=idx_ptype_reflected, jump_before_tqdm=False)
        else:
            Pdet_ptype_reflected = None
        for idim in sorted(dim_to_remove, reverse=True):
            if "thermal" in light_regime_plot:
                Pdet_ptype_thermal = np.take(Pdet_ptype_thermal, 0, axis=idim)
            if "reflected" in light_regime_plot:
                Pdet_ptype_reflected = np.take(Pdet_ptype_reflected, 0, axis=idim)
        Pdet_ptypes_thermal[ipt]   = Pdet_ptype_thermal
        Pdet_ptypes_reflected[ipt] = Pdet_ptype_reflected

    # Plot quantities for the considered ptypes and light_regime
    if light_regime_plot == "thermal":
        Pdet_ptypes_plot = Pdet_ptypes_thermal
        N_PT_ptypes_plot = N_PT_ptypes_thermal
        regime_label     = "thermal"
    elif light_regime_plot == "reflected":
        Pdet_ptypes_plot = Pdet_ptypes_reflected
        N_PT_ptypes_plot = N_PT_ptypes_reflected
        regime_label     = "reflected"
    elif light_regime_plot == "thermal+reflected":
        Pdet_ptypes_plot = []
        N_PT_ptypes_plot = []
        for ipt in range(len(Pdet_ptypes_thermal)):
            Pdet_th = Pdet_ptypes_thermal[ipt]
            Pdet_re = Pdet_ptypes_reflected[ipt]
            N_th    = N_PT_ptypes_thermal[ipt]
            N_re    = N_PT_ptypes_reflected[ipt]
            N_tot   = N_th + N_re
            if N_tot == 0:
                Pdet_ptypes_plot.append(None)
                N_PT_ptypes_plot.append(0)
                continue
            Pdet_combined = np.zeros_like(Pdet_th, dtype=float)
            if N_th > 0:
                Pdet_combined += N_th * Pdet_th
            if N_re > 0:
                Pdet_combined += N_re * Pdet_re
            Pdet_combined /= N_tot
            Pdet_ptypes_plot.append(Pdet_combined)
            N_PT_ptypes_plot.append(N_tot)
        N_PT_ptypes_plot = np.asarray(N_PT_ptypes_plot)
        regime_label     = "thermal+reflected"
        
    # Combine all planet types into a single Pdet corresponding to light_regime_plot
    N_PT_plot = int(np.sum(N_PT_ptypes_plot))
    if N_PT_plot == 0:
        raise RuntimeError(f"No planets available for light_regime_plot='{light_regime_plot}'.")
    Pdet_plot = None
    for ipt in range(len(Pdet_ptypes_plot)):
        if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
            continue
        if Pdet_plot is None:
            Pdet_plot = np.zeros_like(Pdet_ptypes_plot[ipt], dtype=float)
        Pdet_plot += N_PT_ptypes_plot[ipt] * Pdet_ptypes_plot[ipt]
    Pdet_plot /= N_PT_plot
    
    # Helper: normalize to the panel maximum or convert to yield
    def convert_Pdet_to_plot_quantity(Pdet_curve, N_PT, gain, ymax_panel=None):
        y = np.asarray(Pdet_curve, dtype=float)
        if gain:
            y = 100.0*y/ymax_panel if ymax_panel is not None and np.isfinite(ymax_panel) and ymax_panel > 0 else np.zeros_like(y)
        else:
            y = y*N_PT
        return y
    
    def get_panel_ymax(curves):
        values = np.concatenate([np.asarray(curve, dtype=float).ravel() for curve in curves]) if curves else np.array([])
        return np.nanmax(values) if np.any(np.isfinite(values)) else None
    


    # %%
    # 2D CORNER PLOT
    
    # Param values at max pdet_1D
    params_max  = np.zeros((Ndim))
    params_imax = np.zeros((Ndim), dtype=int)
    for idim in range(Ndim):
        pdet_1D           = reduce_hcube(hcube=Pdet_plot, dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names)
        params_imax[idim] = pdet_1D.argmax()
        params_max[idim]  = params[idim][pdet_1D.argmax()]
    
    xmin       = np.array([np.nanmin(param) for param in params])
    xmax       = np.array([np.nanmax(param) for param in params])
    levels     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap       = plt.get_cmap("plasma_r")
    fig, axes  = plt.subplots(Ndim, Ndim, figsize=(1.6 * Ndim, 1.6*Ndim), dpi=dpi)
    axes       = np.atleast_2d(axes)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for ipanel, idim in enumerate(plot_order):
        for jpanel, jdim in enumerate(plot_order):
            ax = axes[ipanel, jpanel]
            ax.tick_params(axis="both", which="major", labelsize=12)
            if jpanel > ipanel:
                ax.axis("off")
                continue

            elif ipanel == jpanel:
                pdet_1D = reduce_hcube(hcube=Pdet_plot, dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                ax.step(params[idim], pdet_1D, color="k", where="mid")
                ax.axvline(params_max[idim], color="k", linestyle="--")
                ax.set_yticks([])
                ax.set_xlabel(params_names_l[idim], fontsize=14)
                if params_isint[idim]:
                    ax.set_title(f"{params_names_l[idim]} = {round(params_max[idim], -2):.0f}", fontsize=11)
                elif params_islog[idim] and params_names[idim] != "IWA [mas]":
                    ax.set_title(f"{params_names_l[idim]} = {params_max[idim]:.1e}", fontsize=11)
                else:
                    ax.set_title(f"{params_names_l[idim]} = {params_max[idim]:.2f}", fontsize=11)
                ax.set_xlim(xmin[idim], xmax[idim])
                if params_islog[idim]:
                    ax.set_xscale("log")
    
            elif jpanel < ipanel:
                pdet_2D = reduce_hcube(hcube=Pdet_plot, dims_to_keep=[jdim, idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                pdet_2D = pdet_2D.T if jdim < idim else pdet_2D
                pmax    = np.nanmax(pdet_2D)
                if (not np.isfinite(pmax)) or (pmax <= 0):
                    ax.text(0.5, 0.5, "no detections", ha="center", va="center", transform=ax.transAxes)
                else:
                    pdet_2D /= pmax
                    ax.contourf(params[jdim], params[idim], pdet_2D, levels=levels, cmap=cmap, alpha=0.8)
                    cs  = ax.contour(params[jdim], params[idim], pdet_2D, levels=levels, colors="black")
                    fmt = {lvl: f"{lvl:.0%}" for lvl in levels}
                    ax.clabel(cs, inline=True, fontsize=10, fmt=fmt)
                ax.axvline(params_max[jdim], color="k", linestyle="--")
                ax.axhline(params_max[idim], color="k", linestyle="--")
                ax.plot(params_max[jdim], params_max[idim], "X", color="black")
                if jpanel == 0:
                    ax.set_ylabel(params_names_l[idim], fontsize=14)
                if ipanel == Ndim - 1:
                    ax.set_xlabel(params_names_l[jdim], fontsize=14)
                ax.set_xlim(xmin[jdim], xmax[jdim])
                ax.set_ylim(xmin[idim], xmax[idim])
                if params_islog[jdim]:
                    ax.set_xscale("log")
                if params_islog[idim]:
                    ax.set_yscale("log")

            if ipanel < Ndim - 1:
                ax.set_xticklabels([])
            if jpanel > 0:
                ax.set_yticklabels([])

    title  = f"ELT/{instru} detection probability corner plot"
    title += f"\n\n assuming an {instru_type} with a Lyot coronagraph"
    title += f"\n \n and {post_processing.replace('DI', 'differential imaging').replace('MM', 'molecular mapping')} as post-processing method"
    title += f"\n\n for {N_PT_plot} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} planets in {regime_label} light"
    #title += f"\n\n for {N_PT_plot} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} rocky planets"

    fig.suptitle(title, fontsize=18, weight="bold", x=0.63, y=0.89)
    fig.savefig(sim_dir / f"ELT_{instru}_{instru_type}_{post_processing}_corner_plot_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi)
    plt.show()



    # %%
    # PLOT : 1D MARGINALIZED DETECTION YIELD/PROBABILITY GAIN PER PARAM, TYPE AND REGIME

    # Layout
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=dpi, sharey=True)
    axes      = np.atleast_2d(axes)
    for ipanel, idim in enumerate(plot_order):
        r  = ipanel // ncols
        t  = ipanel % ncols
        ax = axes[r, t]
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        
        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)

        # Compute all curves before normalizing them to the panel maximum
        curves = []
        for ipt, ptype in enumerate(ptypes):
            if "thermal" in light_regime_plot and N_PT_ptypes_thermal[ipt] > 0:
                Pdet_thermal = reduce_hcube(hcube=Pdet_ptypes_thermal[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                curves.append((Pdet_thermal, N_PT_ptypes_thermal[ipt], ptype, "C3"))
            if "reflected" in light_regime_plot and N_PT_ptypes_reflected[ipt] > 0:
                Pdet_reflected = reduce_hcube(hcube=Pdet_ptypes_reflected[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                curves.append((Pdet_reflected, N_PT_ptypes_reflected[ipt], ptype, "C0"))
        ymax_panel = get_panel_ymax([curve[0] for curve in curves]) if gain else None
        for Pdet_1D, N_PT_curve, ptype, color in curves:
            y = convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_1D, N_PT=N_PT_curve, gain=gain, ymax_panel=ymax_panel)
            ax.plot(params[idim], y, ls="-", lw=lw, c=color, marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, alpha=alpha, zorder=3)
            
        # Axis formatting
        ax.set_xlim(np.nanmin(params[idim]), np.nanmax(params[idim]))
        if params_islog[idim]:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel(params_names_L[idim], fontsize=fontsize + 4, labelpad=10)
        if t == 0:
            if gain:
                ax.set_ylabel("Detection probability\ngain [%]", fontsize=fontsize + 4, labelpad=10)
            else:
                ax.set_ylabel("Yield (number of planets detected)", fontsize=fontsize + 4, labelpad=10)
        else:
            ax.tick_params(labelleft=False)

        # Planet-type legend
        if ipanel ==2:
            handles_ptype = [Line2D([], [], ls="", marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, color="k", label=f"{label_ptypes[ptype]}") for ipt, ptype in enumerate(ptypes)]
            leg_ptype = ax.legend(handles=handles_ptype, fontsize=fontsize, loc="upper left", frameon=True, edgecolor="gray", facecolor="white")#, title="Planet type", title_fontsize=fontsize + 4)
            ax.add_artist(leg_ptype)

        # Planet-regime legend
        if ipanel == 3 and light_regime_plot == "thermal+reflected":
            handles_regime = [Line2D([], [], ls="-", lw=lw + 2, color="C3", label="Thermal"), Line2D([], [], ls="-", lw=lw + 2, color="C0", label="Reflected")]
            leg_regime = ax.legend(handles=handles_regime, fontsize=fontsize+2, loc="lower right", frameon=True, edgecolor="gray", facecolor="white")#, title="Planet-light regime", title_fontsize=fontsize + 4)
            ax.add_artist(leg_regime)

        # Systematics budget with JWST/MIRI/MRS ~ 1%.
        if params_names[idim] == "sigma_m [%]" and post_processing == "MM":
            x0 = 1.0 # [%]
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("JWST/MIRI/MRS", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        if params_names[idim] == "sigma_m [%]" and post_processing == "DI":
            x0 = 0.1  # [%] = 1e-3 in fractional units with VLT/SPHERE PACO
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        
        # WFE and IWA ref values
        if params_names[idim] == "WFE [nm]":
            x0 = WFE_ref # [nm]
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("Expected WFE at Q2", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        if params_names[idim] == "IWA [mas]":
            x0 = IWA_ref  # [mas]
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("ANDES coronagraph", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        
    # Turn off unused panels
    for k in range(Ndim, nrows * ncols):
        r = k // ncols
        t = k % ncols
        axes[r, t].axis("off")

    # Common y-scale
    axes[0, 0].set_yscale("log")
    if gain:
        axes[0, 0].set_ylim(1, 102)
    else:
        axes[0, 0].set_ylim(0.5, None)

    # Title
    quantity_name = "detection probability gain" if gain else "detection yield"
    title         = f"ELT/{instru} {quantity_name} in {exposure_time/60:.0f}hr for {N_PT} "
    title        += f"{table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} planets"
    title        += f"\n\n assuming a Lyot coronagraphic {instru_type} with "
    title        += f"{post_processing.replace('DI', 'differential imaging').replace('MM', 'molecular mapping')}"
    fig.suptitle(title, fontsize=fontsize + 6, weight="bold", y=1.00)
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    fig.savefig(sim_dir / f"ELT_{instru}_{instru_type}_{post_processing}_detection_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi)
    plt.show()



    # %%
    # Plot : 1D MARGINALIZED DETECTION YIELD/PROBABILITY GAIN PER PARAM, TYPE, REGIME AND BANDS
    
    ptypes_plot = ["Earth"]                      # Choose the planet types to show
    bands_plot  = ["R", "I", "Y", "J", "H", "K"] # Choose the spectral bands to show
    
    # Identify lambda0 axis (required for this plot) and identify a l0 for each considered band
    idx_l0 = [idx for idx, param_name in enumerate(params_names) if "l0" in param_name]
    try:
        idx_l0 = idx_l0[0]
    except:
        raise RuntimeError("Could not identify the lambda0 axis in params_names / params_names_L.")
    l0_axis_values = params[idx_l0]
    band_l0_values = np.array([(lmin_bands[band]+lmax_bands[band])/2 for band in bands_plot], dtype=float)
    NbBand         = len(bands_plot)
    band_labels    = [f"{band}\n$\\lambda_0$={l0_band:.2f} µm" for band, l0_band in zip(bands_plot, band_l0_values)]
    cmap           = plt.get_cmap("rainbow", NbBand)
    if np.any(band_l0_values < l0_axis_values[0]) or np.any(band_l0_values > l0_axis_values[-1]):
        raise ValueError(f"At least one selected band central wavelength is outside the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm.")

    # Layout
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=dpi, sharey=True)
    axes      = np.atleast_2d(axes)
    for ipanel, idim in enumerate(plot_order):
        r  = ipanel // ncols
        t  = ipanel % ncols
        ax = axes[r, t]
        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

        # Compute all curves (before normalizing them to the panel maximum if gain)
        curves = []
        if idim == idx_l0:
            for ptype in ptypes_plot:
                ipt = ptypes.index(ptype)
                if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                    continue
                Pdet_1D = reduce_hcube(hcube=Pdet_ptypes_plot[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                curves.append((Pdet_1D, N_PT_ptypes_plot[ipt], ptype, "k"))
        else:
            for ptype in ptypes_plot:
                ipt = ptypes.index(ptype)
                if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                    continue
                for iband, l0_band in enumerate(band_l0_values):
                    params_ranges_band         = [tuple(rng) for rng in params_ranges]
                    params_ranges_band[idx_l0] = (l0_band, l0_band)
                    Pdet_1D                    = reduce_hcube(hcube=Pdet_ptypes_plot[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges_band, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_1D, N_PT_ptypes_plot[ipt], ptype, cmap(iband)))
        ymax_panel = get_panel_ymax([curve[0] for curve in curves]) if gain else None
        for Pdet_1D, N_PT_curve, ptype, color in curves:
            y = convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_1D, N_PT=N_PT_curve, gain=gain, ymax_panel=ymax_panel)
            ax.plot(params[idim], y, ls="-", lw=lw, c=color, marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, alpha=alpha, zorder=3)
        if idim == idx_l0:
            for iband, l0_band in enumerate(band_l0_values):
                ax.axvline(l0_band, c=cmap(iband), ls="-", lw=3*lw, alpha=0.3, zorder=2)
                
        # Axis formatting
        ax.set_xlim(np.nanmin(params[idim]), np.nanmax(params[idim]))
        if params_islog[idim]:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel(params_names_L[idim], fontsize=fontsize + 2, labelpad=10)
        if t == 0:
            if gain:
                ax.set_ylabel("Detection probability gain [%]", fontsize=fontsize + 2, labelpad=10)
            else:
                ax.set_ylabel("Yield (number of planets detected)", fontsize=fontsize + 2, labelpad=10)
        else:
            ax.tick_params(labelleft=False)

        # Planet-type legend
        if ipanel == 0:
            handles_ptype = [Line2D([], [], ls="", marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, color="k", label=label_ptypes[ptype]) for ptype in ptypes_plot]
            leg_ptype     = ax.legend(handles=handles_ptype, fontsize=fontsize + 2, loc="upper right", frameon=True, edgecolor="gray", facecolor="white", title="Planet type", title_fontsize=fontsize + 4)
            ax.add_artist(leg_ptype)

        # Systematics budget with JWST/MIRI/MRS ~ 1%.
        if params_names[idim] == "sigma_m [%]" and post_processing == "MM":
            x0 = 1.0 # [%]
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("JWST/MIRI/MRS", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        if params_names[idim] == "sigma_m [%]" and post_processing == "DI":
            x0 = 0.1  # [%] = 1e-3 in fractional units with VLT/SPHERE PACO
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        
        # WFE and IWA ref values
        if params_names[idim] == "WFE [nm]":
            x0 = WFE_ref # [nm]
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("Expected WFE at Q2", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        if params_names[idim] == "IWA [mas]":
            x0 = IWA_ref  # [mas]
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("ANDES coronagraph", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        
    # Turn off unused panels
    for k in range(Ndim, nrows * ncols):
        r = k // ncols
        t = k % ncols
        axes[r, t].axis("off")

    # Common y-scale
    axes[0, 0].set_yscale("log")
    if gain:
        axes[0, 0].set_ylim(1, 102)
    else:
        axes[0, 0].set_ylim(0.5, None)

    # Common discrete colorbar for selected bands
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(NbBand + 1) - 0.5, ncolors=NbBand)
    sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", fraction=0.025, pad=0.05, ticks=np.arange(NbBand))
    cbar.ax.set_yticklabels(band_labels)
    cbar.set_label(r"Selected spectral band / fixed central wavelength $\lambda_0$", fontsize=fontsize + 2, rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=fontsize)

    # Title
    if gain:
        quantity_name = "detection probability gain"
    else:
        quantity_name = "detection yield"
    N_PT_ptypes_plot_1D = int(np.sum([N_PT_ptypes_plot[ptypes.index(ptype)] for ptype in ptypes_plot]))
    ptype_plot_label    = "+".join(ptypes_plot)
    title  = f"ELT/{instru} {quantity_name} in {exposure_time/60:.0f}hr for {N_PT_ptypes_plot_1D} "
    title += f"{table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} {ptype_plot_label}-like planets"
    title += f"\n\n assuming a Lyot coronagraphic {instru_type} with "
    title += f"{post_processing.replace('DI', 'differential imaging').replace('MM', 'molecular mapping')}"
    title += f"\n\n for the {light_regime_plot} planet-light regime"
    fig.suptitle(title, fontsize=fontsize + 6, weight="bold", y=1.00)
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.88, wspace=0.15, hspace=0.3)
    fig.savefig(sim_dir / f"ELT_{instru}_{instru_type}_{post_processing}_detection_band_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi)
    plt.show()



    #%%
    # HEATMAP: BAND x RESOLUTION / BANDWIDTH HEATMAPS PER PLANET TYPE AND LIGHT REGIME
    
    ptypes_heatmap = ["Jupiter", "Saturn", "Neptune", "Earth"]
    bands_plot     = ["R", "I", "Y", "J", "H", "K"]
    heatmap_mode   = "marginalized" # "marginalized" or "max"
    regimes_plot   = [regime for regime in ["thermal", "reflected"] if regime in light_regime_plot]
    if not regimes_plot:
        raise RuntimeError("No valid regime to plot. light_regime_plot must contain 'thermal' and/or 'reflected'.")

    # Choose the second heatmap axis depending on the instrument type
    if post_processing == "MM": #instru_type == "IFU":
        spec_plot      = [1_000, 3_000, 10_000, 30_000, 100_000]
        idx_spec       = next((idx for idx, name in enumerate(params_names) if "R" in name), None)
        spec_axis_name = "Resolution"
    else:
        spec_plot      = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
        idx_spec       = next((idx for idx, name in enumerate(params_names) if "sigma_m" in name), None)
        spec_axis_name = r"Speckle residuals $\sigma_m$ [%]"

    idx_l0 = next((idx for idx, name in enumerate(params_names) if "l0" in name), None)
    if idx_l0 is None:
        raise RuntimeError("Could not identify the lambda0 axis in params_names.")
    if idx_spec is None:
        raise RuntimeError("Could not identify the secondary spectral axis in params_names.")
    if heatmap_mode not in {"marginalized", "max"}:
        raise ValueError("heatmap_mode must be 'marginalized' or 'max'.")

    l0_axis_values   = np.asarray(params[idx_l0],   dtype=float)
    spec_axis_values = np.asarray(params[idx_spec], dtype=float)
    band_l0_values   = np.array([(lmin_bands[band] + lmax_bands[band]) / 2 for band in bands_plot], dtype=float)
    spec_values      = np.asarray(spec_plot, dtype=float)
    spec_values      = np.clip(spec_values, spec_axis_values[0], spec_axis_values[-1])
    spec_plot        = spec_values.tolist()
    spec_plot_labels = [f"{spec:g}" for spec in spec_plot] if instru_type == "IFU" else [f"{spec:.3f}" for spec in spec_plot]

    regime_to_Pdet = {"thermal": Pdet_ptypes_thermal, "reflected": Pdet_ptypes_reflected}
    regime_to_NPT  = {"thermal": N_PT_ptypes_thermal, "reflected": N_PT_ptypes_reflected}
    dims_keep_max  = [idim for idim in range(Ndim) if idim not in (idx_spec, idx_l0)]
    heatmaps       = {}

    for ptype in ptypes_heatmap:
        ipt = ptypes.index(ptype)
        for regime in regimes_plot:
            Pdet_cube = regime_to_Pdet[regime][ipt]
            N_PT_reg  = regime_to_NPT[regime][ipt]
            heatmap   = np.full((len(spec_plot), len(bands_plot)), np.nan, dtype=float)
            if Pdet_cube is not None and N_PT_reg > 0:
                for ispec, spec_fixed in enumerate(spec_plot):
                    for iband, l0_band in enumerate(band_l0_values):
                        ranges           = [tuple(rng) for rng in params_ranges]
                        ranges[idx_l0]   = (l0_band, l0_band)
                        ranges[idx_spec] = (spec_fixed, spec_fixed)
                        if heatmap_mode == "marginalized":
                            Pdet_value            = reduce_hcube(hcube=Pdet_cube, dims_to_keep=[], params=params, params_ranges=ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                            heatmap[ispec, iband] = float(np.asarray(Pdet_value)) * N_PT_reg
                        else:
                            Pdet_subcube          = np.asarray(reduce_hcube(hcube=Pdet_cube, dims_to_keep=dims_keep_max, params=params, params_ranges=ranges, params_priors=params_priors, params_names=params_names, verbose=False), dtype=float)
                            heatmap[ispec, iband] = float(np.nanmax(Pdet_subcube)) * N_PT_reg if np.any(np.isfinite(Pdet_subcube)) else np.nan
            heatmaps[(ptype, regime)] = heatmap

    panel_keys   = [(ptype, regime) for ptype in ptypes_heatmap for regime in regimes_plot]
    panel_labels = {(ptype, regime): f"{label_ptypes[ptype]} — {regime.capitalize()}" for ptype, regime in panel_keys}
    title        = f"ELT/{instru} band × {'resolution' if instru_type == 'IFU' else 'speckle residuals'} yield - {instru_type} with {post_processing}"#"\nheatmap mode: {heatmap_mode}"
    
    yield_heatmap_ELT(instru=instru, exposure_time=exposure_time, heatmaps=heatmaps, ptypes_heatmap=panel_keys, bands_plot=bands_plot, config_labels=spec_plot_labels, config_axis_name=spec_axis_name, x_axis_name="Spectral band", panel_labels=panel_labels, title=title, save_dir=sim_dir, filename=f"ELT_{instru}_{instru_type}_{post_processing}_band_spectral_heatmap_grid_{table_type}_{light_regime_plot}_{heatmap_mode}.png")



    #%%
    # POPULATION DIAGNOSTICS
    
    # Band used for the contrast shown on the y-axis
    band_contrast_plot = "H"
    
    # Choose how the detection status is defined for population plots.
    # "max"             : use the SNR at the global maximum of the Pdet hypercube.
    # "marginalized"    : average the SNR over the parameter ranges/priors, with FoV gating.
    snr_population_mode = "marginalized"
    
    idx_FoV      = next(idx for idx, name in enumerate(params_names) if "FoV" in name)
    idx_sigma_m  = next(idx for idx, name in enumerate(params_names) if "sigma_m" in name)
    idx_l0       = next(idx for idx, name in enumerate(params_names) if "l0" in name)
    noise_labels = np.array(["Stellar halo", "Background", "Read noise", "Dark current", str(residuals)], dtype=object)

    # Best-Pdet configuration
    if snr_population_mode == "max":
        params_imax                             = np.array(np.unravel_index(np.nanargmax(Pdet), Pdet.shape), dtype=int)
        params_imax_snr                         = [params_imax[idim] for idim in range(Ndim) if idim != idx_FoV]
        SNR_plot                                = np.asarray(SNR_planets.squeeze()[(slice(None),) + tuple(params_imax_snr)], dtype=float).copy()
        SNR_plot[p0_FoV > params_imax[idx_FoV]] = 0.0

        # Raw arrays do not contain sigma_m or FoV
        params_imax_raw   = [params_imax[idim] for idim in range(Ndim) if idim not in (idx_sigma_m, idx_FoV)]
        raw_selection     = (slice(None),) + tuple(params_imax_raw)
        sigma_halo_2_best = np.asarray(sigma_halo_2_planets.squeeze()[raw_selection], dtype=float)
        sigma_bkg_2_best  = np.asarray(sigma_bkg_2_planets.squeeze()[raw_selection], dtype=float)
        DIT_best          = np.asarray(DIT_planets.squeeze()[raw_selection], dtype=float)

        il0_best     = params_imax[idx_l0]
        RON0_best    = float(np.asarray(RON0)[il0_best])
        RON_lim_best = float(np.asarray(RON_lim)[il0_best])
        DC0_best     = float(np.asarray(DC0)[il0_best])
        min_DIT_best = float(np.asarray(min_DIT)[il0_best])
        N_DIT_best   = np.clip(np.floor(exposure_time / DIT_best), 1, None)
        N_read_best  = np.floor(DIT_best / min_DIT_best).astype(np.int32)

        sigma_RON_2_best = np.full_like(DIT_best, RON0_best**2, dtype=float)
        mask_read        = N_read_best >= 2
        if np.any(mask_read):
            n_read                      = N_read_best[mask_read].astype(float)
            sigma_RON_2_best[mask_read] = RON0_best**2 * 12 * (n_read - 1) / (n_read * (n_read + 1)) + RON_lim_best**2
        sigma_RON_2_best *= A_FWHM
        sigma_DC_2_best   = DC0_best * DIT_best * A_FWHM
        sigma_m_best      = float(params[idx_sigma_m][params_imax[idx_sigma_m]]) / 100.0

        if instru_type == "IFU":
            sigma_syst_base_2_best = np.asarray(sigma_syst_base_2_planets.squeeze()[raw_selection], dtype=float)
        elif instru_type == "imager":
            sigma_syst_base_2_best = sigma_halo_2_best**2
        else:
            raise ValueError("instru_type must be 'IFU' or 'imager'.")

        var_halo = N_DIT_best * sigma_halo_2_best
        var_bkg  = N_DIT_best * sigma_bkg_2_best
        var_RON  = N_DIT_best * sigma_RON_2_best
        var_DC   = N_DIT_best * sigma_DC_2_best
        var_syst = N_DIT_best**2 * sigma_m_best**2 * sigma_syst_base_2_best

    # Marginalized parameter space
    elif snr_population_mode == "marginalized":
        non_FoV_dims      = [idim for idim in range(Ndim) if idim != idx_FoV]
        params_snr        = [params[idim] for idim in non_FoV_dims]
        params_ranges_snr = [params_ranges[idim] for idim in non_FoV_dims]
        params_priors_snr = [params_priors[idim] for idim in non_FoV_dims]
        params_names_snr  = [params_names[idim] for idim in non_FoV_dims]

        SNR_plot = np.zeros(N_PT, dtype=float)
        for ip in tqdm(range(N_PT), desc="Marginalizing SNR over parameter grid"):
            SNR_i        = np.nan_to_num(np.asarray(SNR_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            SNR_plot[ip] = float(reduce_hcube(hcube=SNR_i, dims_to_keep=[], params=params_snr, params_ranges=params_ranges_snr, params_priors=params_priors_snr, params_names=params_names_snr, verbose=False))

        # FoV marginalization
        idx_FoV_range, w_FoV = get_axis_weights_in_range(axis=np.asarray(params[idx_FoV], dtype=float), pmin=params_ranges[idx_FoV][0], pmax=params_ranges[idx_FoV][1], prior=params_priors[idx_FoV])
        w_FoV               /= np.sum(w_FoV)
        FoV_weight_planets   = np.zeros(N_PT, dtype=float)
        for iFoV, weight in zip(idx_FoV_range, w_FoV):
            FoV_weight_planets[p0_FoV <= iFoV] += weight
        SNR_plot *= FoV_weight_planets

        # Raw quantities contain neither sigma_m nor FoV
        raw_dims          = [idim for idim in range(Ndim) if idim not in (idx_sigma_m, idx_FoV)]
        params_raw        = [params[idim] for idim in raw_dims]
        params_ranges_raw = [params_ranges[idim] for idim in raw_dims]
        params_priors_raw = [params_priors[idim] for idim in raw_dims]
        params_names_raw  = [params_names[idim] for idim in raw_dims]
        idx_l0_raw        = raw_dims.index(idx_l0)

        idx_sigma_m_range, w_sigma_m = get_axis_weights_in_range(axis=np.asarray(params[idx_sigma_m], dtype=float), pmin=params_ranges[idx_sigma_m][0], pmax=params_ranges[idx_sigma_m][1], prior=params_priors[idx_sigma_m])
        w_sigma_m                   /= np.sum(w_sigma_m)
        sigma_m_axis                 = np.asarray(params[idx_sigma_m], dtype=float)
        mean_sigma_m_2               = np.sum(w_sigma_m * (sigma_m_axis[idx_sigma_m_range] / 100.0)**2)

        shape_l0             = [1] * len(raw_dims)
        shape_l0[idx_l0_raw] = len(params[idx_l0])
        shape_l0             = tuple(shape_l0)
        RON0_grid            = np.asarray(RON0, dtype=float).reshape(shape_l0)
        RON_lim_grid         = np.asarray(RON_lim, dtype=float).reshape(shape_l0)
        DC0_grid             = np.asarray(DC0, dtype=float).reshape(shape_l0)
        min_DIT_grid         = np.asarray(min_DIT, dtype=float).reshape(shape_l0)
        
        def marginalize_raw(quantity):
            return float(reduce_hcube(hcube=quantity, dims_to_keep=[], params=params_raw, params_ranges=params_ranges_raw, params_priors=params_priors_raw, params_names=params_names_raw, verbose=False))
        
        var_halo = np.zeros(N_PT, dtype=float)
        var_bkg  = np.zeros(N_PT, dtype=float)
        var_RON  = np.zeros(N_PT, dtype=float)
        var_DC   = np.zeros(N_PT, dtype=float)
        var_syst = np.zeros(N_PT, dtype=float)
        N_DIT_marg        = np.zeros(N_PT, dtype=float)
        signal_total_marg = np.zeros(N_PT, dtype=float)
        
        for ip in tqdm(range(N_PT), desc="Marginalizing dominant-noise terms"):
            signal_i        = np.nan_to_num(np.asarray(signal_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            sigma_halo_2_i = np.nan_to_num(np.asarray(sigma_halo_2_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            sigma_bkg_2_i  = np.nan_to_num(np.asarray(sigma_bkg_2_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            DIT_i          = np.asarray(DIT_planets[ip], dtype=float).squeeze()
            valid_DIT      = np.isfinite(DIT_i) & (DIT_i > 0)
            DIT_safe       = np.where(valid_DIT, DIT_i, 1.0)
            RON0_i         = np.broadcast_to(RON0_grid, DIT_safe.shape)
            RON_lim_i      = np.broadcast_to(RON_lim_grid, DIT_safe.shape)
            DC0_i          = np.broadcast_to(DC0_grid, DIT_safe.shape)
            min_DIT_i      = np.broadcast_to(min_DIT_grid, DIT_safe.shape)
            N_DIT_i        = np.where(valid_DIT, np.clip(np.floor(exposure_time / DIT_safe), 1, None), 0.0)
            N_read_i       = np.where(valid_DIT, np.floor(DIT_safe / min_DIT_i), 0).astype(np.int32)
            sigma_RON_2_i  = RON0_i**2
            mask_read      = N_read_i >= 2
        
            if np.any(mask_read):
                n_read                   = N_read_i[mask_read].astype(float)
                sigma_RON_2_i[mask_read] = RON0_i[mask_read]**2 * 12 * (n_read - 1) / (n_read * (n_read + 1)) + RON_lim_i[mask_read]**2
        
            sigma_RON_2_i *= A_FWHM
            sigma_RON_2_i  = np.where(valid_DIT, sigma_RON_2_i, 0.0)
            sigma_DC_2_i   = np.where(valid_DIT, DC0_i * DIT_safe * A_FWHM, 0.0)
        
            if instru_type == "IFU":
                sigma_syst_base_2_i = np.nan_to_num(np.asarray(sigma_syst_base_2_planets[ip], dtype=float).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            elif instru_type == "imager":
                sigma_syst_base_2_i = sigma_halo_2_i**2
            else:
                raise ValueError("instru_type must be 'IFU' or 'imager'.")
        
            var_halo[ip]         = marginalize_raw(N_DIT_i * sigma_halo_2_i)
            var_bkg[ip]          = marginalize_raw(N_DIT_i * sigma_bkg_2_i)
            var_RON[ip]          = marginalize_raw(N_DIT_i * sigma_RON_2_i)
            var_DC[ip]           = marginalize_raw(N_DIT_i * sigma_DC_2_i)
            var_syst[ip]         = marginalize_raw(N_DIT_i**2 * mean_sigma_m_2 * sigma_syst_base_2_i)
            N_DIT_marg[ip]       = marginalize_raw(N_DIT_i)
            signal_total_marg[ip] = marginalize_raw(N_DIT_i * signal_i)
        
        signal_INSTRU     = np.full(N_PT, np.nan, dtype=float)
        sigma_fund_INSTRU = np.full(N_PT, np.nan, dtype=float)
        sigma_syst_INSTRU = np.full(N_PT, np.nan, dtype=float)
        DIT_INSTRU        = np.full(N_PT, np.nan, dtype=float)
        
        var_fund = var_halo + var_bkg + var_RON + var_DC
        valid    = np.isfinite(N_DIT_marg) & np.isfinite(signal_total_marg) & np.isfinite(var_fund) & np.isfinite(var_syst) & (N_DIT_marg > 0) & (signal_total_marg > 0) & (var_fund >= 0) & (var_syst >= 0)
        
        DIT_INSTRU[valid]        = exposure_time / N_DIT_marg[valid]
        signal_INSTRU[valid]     = signal_total_marg[valid] / N_DIT_marg[valid]
        sigma_fund_INSTRU[valid] = np.sqrt(var_fund[valid] / N_DIT_marg[valid])
        sigma_syst_INSTRU[valid] = np.sqrt(var_syst[valid]) / N_DIT_marg[valid]
        
        if instru_type == "IFU" and post_processing == "MM":
            path = archive_path if table_type == "Archive" else simulated_path
            save_marginalized_planet_table(
                planet_table=planet_table,
                signal=signal_INSTRU,
                sigma_fund=sigma_fund_INSTRU,
                sigma_syst=sigma_syst_INSTRU,
                DIT=DIT_INSTRU,
                SNR=SNR_plot,
                exposure_time=exposure_time,
                table=table_type,
                instru=instru,
                apodizer=apodizer,
                strehl=strehl,
                coronagraph=coronagraph,
                name_model=name_model,
                path=path,
            )

    else:
        raise ValueError("snr_population_mode must be 'max' or 'marginalized'.")

    noise_stack                 = np.vstack([var_halo, var_bkg, var_RON, var_DC, var_syst])
    valid_noise                 = np.any(np.isfinite(noise_stack), axis=0)
    dominant_noise              = np.full(N_PT, "Unknown", dtype=object)
    dominant_noise[valid_noise] = noise_labels[np.nanargmax(noise_stack[:, valid_noise], axis=0)]
    
    yield_population_plot(table=table_type, instru=instru, thermal_model=thermal_model, reflected_model=reflected_model, exposure_time=exposure_time, band_contrast_plot=band_contrast_plot, band_regime_plot=band_regime_plot, planet_table=planet_table, SNR_plot=SNR_plot, dominant_noise=dominant_noise, SNR_thr=SNR_thr, save_dir=sim_dir, DL_mas=np.nan)



    # %%
    # --- Clean temporary SNR and mask_detections memmap files ---
    snr_tmp_path  = Path(SNR_planets.filename)
    mask_tmp_path = Path(mask_detections.filename)

    del SNR_planets, mask_detections, signal_planets, sigma_halo_2_planets, sigma_bkg_2_planets, sigma_syst_base_2_planets, DIT_planets
    gc.collect()

    if snr_tmp_path.exists():
        snr_tmp_path.unlink()
    if mask_tmp_path.exists():
        mask_tmp_path.unlink()

    return locals()



# %%

if __name__ == "__main__":
    __spec__ = None

    if sys.platform.startswith("win"):
        mp.freeze_support() # for Windows
    _out = main()
    globals().update(_out)



