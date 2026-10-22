import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["NUMBA_NUM_THREADS"]    = "1"

# import FastYield modules
from fastyield.config import h, c, rad2arcsec, planet_types
from fastyield.get_specs import get_config_data, load_tell_trans, get_transmission, get_PSF_profile
from fastyield.spectrum import get_wavelength_axis_constant_R, Spectrum, load_vega_spectrum, get_thermal_reflected_spectrum, filtered_flux, get_wave_K, get_wave_band, get_counts_from_density
from fastyield.FastYield import load_planet_table, get_mask_planet_type, get_filename_table, get_SNR_from_table, get_planet_index
from fastyield.FastYield_helpers import print_simulation_summary, make_suffix, write_meta
from fastyield.utils import add_if_necessary, plot_bkg_skycalc
from fastyield.prints_helpers import print_subheader, print_metric
from fastyield.signal_noise import compress_h_for_sigma_base_2, compute_sigma_base_2_speck_numba, compute_sigma_base_2_al_spat_numba, compute_sigma_base_2_al_spec_fast

# import astropy modules
from astropy.io import fits

# import numpy modules
import numpy as np
from numpy.lib.format import open_memmap

# import matplotlib modules
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import matplotlib as mpl
from matplotlib.patches import Rectangle

# import scipy modules
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit, logit
from scipy.spatial import cKDTree, Delaunay, QhullError

# import other modules
from pathlib import Path
from tqdm import tqdm
import sys
import multiprocessing as mp
from multiprocessing import cpu_count
import gc

# Numerical parameters for simulations
nproc     = max(cpu_count()//2+3, 1)
chunksize = 8
dtype     = np.float32

# --- General helper ---
def interp_flat_last_axis(x, y, x0, dtype=None):
    """
    Fast linear interpolation along the last axis, with flat extrapolation.
    This is optimized for quantities sampled as (..., Nx), for example
    PSF_profile_6D[..., separation].
    """
    x0  = np.clip(x0, x[0], x[-1])
    i1  = int(np.searchsorted(x, x0, side="right"))
    i1  = int(np.clip(i1, 1, x.size - 1))
    i0  = i1 - 1
    w   = (x0 - x[i0]) / (x[i1] - x[i0])
    out = (1.0 - w) * y[..., i0] + w * y[..., i1]
    return out.astype(dtype, copy=False) if dtype is not None else out

def sigma_RON_2_from_DIT(DIT, min_DIT, RON0, RON_lim, A_FWHM, NbPixel=None, dtype=np.float32):
    """
    Effective readout-noise variance in [e-/FWHM/DIT]**2 for UTR sampling.
    """
    DIT         = np.asarray(DIT, dtype=dtype)
    N_read      = np.floor(DIT / min_DIT).astype(np.int32)
    sigma_RON_2 = np.full(DIT.shape, RON0**2, dtype=dtype)
    m           = N_read >= 2
    if np.any(m):
        n              = N_read[m].astype(dtype)
        sigma_RON_2[m] = (RON0**2 * 12 * (n - 1) / (n * (n + 1)) + RON_lim**2).astype(dtype)
    sigma_RON_2 *= A_FWHM
    if NbPixel is not None:
        sigma_RON_2 *= NbPixel
    return sigma_RON_2

def warmup_signal_noise_numba():
    """
    Compile/load the Numba covariance kernels before multiprocessing.

    On Linux, the compiled functions are inherited by forked workers, which avoids
    paying the compilation/loading overhead independently in every worker.
    """
    wave = np.linspace(1.0, 1.1, 32, dtype=dtype)
    h_PP = np.ones((32, 1), dtype=dtype)
    sep  = np.array([50.0 / (rad2arcsec * 1000.0)], dtype=dtype)
    _    = compute_sigma_base_2_speck_numba(h=h_PP, wave=wave, separation_rad=sep, D=38.54)
    _    = compute_sigma_base_2_al_spat_numba(h=h_PP, wave=wave, separation_rad=sep, pxscale_rad=1.0 / (rad2arcsec * 1000.0))
    _    = compute_sigma_base_2_al_spec_fast(h=h_PP)



# -------------------------------------
# Main functions for IFU 
# -------------------------------------

def metrics_slice(sl, post_processing, wave_i, trans_tell_i, star_i, planet_i, background_i, star_HF_i, star_LF_i, planet_HF_i, planet_LF_i, valid_i, separation_planet_rad, separation_planet_rad_arr, pxscale_rad, D, background_scale, fast_comp=True):
    wave_ijk       = wave_i[sl]                          # [µm]
    trans_tell_ijk = trans_tell_i[sl]                    # [no unit]
    star_ijk       = star_i[sl]                          # [ph/mn/bin]
    planet_ijk     = planet_i[sl]                        # [ph/mn/bin]
    background_ijk = background_i[sl] * background_scale # [ph/mas2/mn/bin] => [ph/px/mn/bin]
    
    # Valid range (if force_inside_bands)
    if valid_i is not None:
        valid_ijk = valid_i[sl]
        if valid_ijk.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        wave_ijk       = wave_ijk[valid_ijk]
        trans_tell_ijk = trans_tell_ijk[valid_ijk]
        star_ijk       = star_ijk[valid_ijk]
        planet_ijk     = planet_ijk[valid_ijk]
        background_ijk = background_ijk[valid_ijk]

    # MM post-processing method
    if post_processing == "MM":
        star_HF_ijk   = star_HF_i[sl]   # [ph/mn/bin]
        star_LF_ijk   = star_LF_i[sl]   # [ph/mn/bin]
        planet_HF_ijk = planet_HF_i[sl] # [ph/mn/bin]
        planet_LF_ijk = planet_LF_i[sl] # [ph/mn/bin]
        # Valid range (if force_inside_bands)
        if valid_i is not None:
            star_HF_ijk   = star_HF_ijk[valid_ijk]
            star_LF_ijk   = star_LF_ijk[valid_ijk]
            planet_HF_ijk = planet_HF_ijk[valid_ijk]
            planet_LF_ijk = planet_LF_ijk[valid_ijk]
        
        # Template computation
        template = trans_tell_ijk*planet_HF_ijk
        norm2    = np.nansum(template**2)        
        if (not np.isfinite(norm2)) or (norm2 <= 0):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        template /= np.sqrt(norm2)
        
        # CCF signal computation in [ph/mn] (not normalized with PSF data and transmission)
        alpha                    = np.nansum(trans_tell_ijk * planet_HF_ijk * template)
        ratio_planet_star_LF_ijk = np.divide(planet_LF_ijk, star_LF_ijk, out=np.zeros_like(planet_LF_ijk), where=np.isfinite(star_LF_ijk) & (star_LF_ijk != 0))
        beta                     = np.nansum(trans_tell_ijk * star_HF_ijk * ratio_planet_star_LF_ijk * template)
        signal                   = alpha - beta
    
    # DI post-processing method
    elif post_processing == "DI":
        # Template computation
        template = trans_tell_ijk*planet_ijk
        norm2    = np.nansum(template**2)
        if (not np.isfinite(norm2)) or (norm2 <= 0):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        template /= np.sqrt(norm2)
        
        # CCF signal computation in [ph/mn] (not normalized with PSF data and transmission)
        delta  = np.nansum(trans_tell_ijk*planet_ijk * template)
        signal = delta
    
    # Sigma stellar halo photon noise computation in [ph/mn] (not normalized with PSF data and transmission)
    sigma_halo_2 = np.nansum(trans_tell_ijk*star_ijk * template**2)
    
    # Max star flux in [ph/mn] (not normalized with PSF data and transmission)
    max_star_flux = np.nanmax(trans_tell_ijk * star_ijk)       
    
    # Sigma background photon noise computation in [ph/px/mn] (without transmission)
    sigma_bkg_2 = np.nansum(background_ijk * template**2)
    
    # Sigma stellar halo systematic noise (assuming sigma_m = 1) computation in [ph/mn]**2 (not normalized with PSF data and transmission)
    # This implements:
    #   sigma_m,base^2 = h^T C_m h
    # with, for MM:
    #   C_m = 0.5 * (C_al_spec + C_al_spat).
    # Therefore:
    #   sigma_m,base^2 = 0.5 * (h^T C_al_spec h + h^T C_al_spat h).
    h_PP = (trans_tell_ijk*star_ijk * template)[:, None] # (wave, sep) Ponderated halo, H*t
    if fast_comp:
        wave_ijk, h_PP_c, n_merge = compress_h_for_sigma_base_2(wave=wave_ijk, h=h_PP, post_processing=post_processing, D=D, pxscale=pxscale_rad, separation_ref=separation_planet_rad, sep_unit="rad")        
    if post_processing == "DI":
        sigma_syst_base_2 = compute_sigma_base_2_speck_numba(     h=h_PP_c if fast_comp else h_PP, wave=wave_ijk, separation_rad=separation_planet_rad_arr, D=D)[0]
    elif post_processing == "MM":
        sigma_base_2_al_spat = compute_sigma_base_2_al_spat_numba(h=h_PP_c if fast_comp else h_PP, wave=wave_ijk, separation_rad=separation_planet_rad_arr, pxscale_rad=pxscale_rad)[0]
        sigma_base_2_al_spec = compute_sigma_base_2_al_spec_fast(   h=h_PP)[0]
        sigma_syst_base_2    = 0.5 * (sigma_base_2_al_spec + sigma_base_2_al_spat)

    return signal, sigma_halo_2, max_star_flux, sigma_bkg_2, sigma_syst_base_2




# --- Globals used by multiprocessing workers ---
_CTX       = None
_OUT_NAMES = [
    "signal_4D_R", "signal_4D_l0", "signal_4D_Nl", "signal_4D_TI", "signal_4D_WFE", "signal_4D_IWA",
    "sigma_halo_2_4D_R", "sigma_halo_2_4D_l0", "sigma_halo_2_4D_Nl", "sigma_halo_2_4D_TI", "sigma_halo_2_4D_WFE", "sigma_halo_2_4D_IWA",
    "sigma_syst_base_2_4D_R", "sigma_syst_base_2_4D_l0", "sigma_syst_base_2_4D_Nl", "sigma_syst_base_2_4D_TI", "sigma_syst_base_2_4D_WFE", "sigma_syst_base_2_4D_IWA",
    "sigma_bkg_2_4D_R", "sigma_bkg_2_4D_l0", "sigma_bkg_2_4D_Nl", "sigma_bkg_2_4D_TI", "sigma_bkg_2_4D_WFE", "sigma_bkg_2_4D_IWA",
    "DIT_4D_R", "DIT_4D_l0", "DIT_4D_Nl", "DIT_4D_TI", "DIT_4D_WFE", "DIT_4D_IWA",
]
def init_worker(ctx):
    global _CTX
    _CTX = dict(ctx)

    # Open PSF memmaps in read-only mode inside each worker
    _CTX["PSF_profile_6D"]   = open_memmap(_CTX["PSF_profile_6D_path"], mode="r")
    _CTX["fraction_core_6D"] = open_memmap(_CTX["fraction_core_6D_path"], mode="r")

    # Open output memmaps in read/write mode inside each worker
    for name in _OUT_NAMES:
        _CTX[f"{name}_mm"] = open_memmap(_CTX[f"{name}_path"], mode="r+")



# --- IFU function ---
def process_ELT(idx):
    
    # --- Context ---
    planet_table       = _CTX["planet_table"]
    separation         = _CTX["separation"]
    pxscales           = _CTX["pxscales"]
    PSF_profile_6D     = _CTX["PSF_profile_6D"]
    fraction_core_6D   = _CTX["fraction_core_6D"]
    idx_R_ref          = _CTX["idx_R_ref"]
    idx_l0_ref         = _CTX["idx_l0_ref"]
    idx_Nl_ref         = _CTX["idx_Nl_ref"]
    idx_WFE_ref        = _CTX["idx_WFE_ref"]
    idx_IWA_ref        = _CTX["idx_IWA_ref"]
    R                  = _CTX["R"]
    trans_instru       = _CTX["trans_instru"]
    wave_1D_list       = _CTX["wave_1D_list"]
    dwave_1D_list      = _CTX["dwave_1D_list"]
    range_1D_list      = _CTX["range_1D_list"]
    valid_2D_list      = _CTX["valid_2D_list"]
    trans_tell_1D_list = _CTX["trans_tell_1D_list"]
    background_1D_list = _CTX["background_1D_list"]
    wave_model         = _CTX["wave_model"]
    wave_instru        = _CTX["wave_instru"]
    wave_K             = _CTX["wave_K"]
    counts_vega_K      = _CTX["counts_vega_K"]
    scale_spectrum     = _CTX["scale_spectrum"]
    thermal_model      = _CTX["thermal_model"]
    reflected_model    = _CTX["reflected_model"]
    post_processing    = _CTX["post_processing"]
    Rc                 = _CTX["Rc"]
    filter_type        = _CTX["filter_type"]
    D                  = _CTX["D"]
    saturation_e       = _CTX["saturation_e"]
    min_DIT            = _CTX["min_DIT"]
    max_DIT            = _CTX["max_DIT"]
    A_FWHM             = _CTX["A_FWHM"]
    trans_instru_ref   = _CTX["trans_instru_ref"]
    N_band             = _CTX["N_band"]
    N_config           = _CTX["N_config"]
    N_R                = _CTX["N_R"]
    N_l0               = _CTX["N_l0"]
    N_Nl               = _CTX["N_Nl"]

    
    # --- Planet row ---
    planet                    = planet_table[idx]
    separation_planet         = float(planet["AngSep"].value)                  # [mas]
    separation_planet_rad     = separation_planet / (rad2arcsec * 1000)        # [rad]
    separation_planet_rad_arr = np.array([separation_planet_rad], dtype=dtype) # [rad]
    
    # --- PSF data linearly interpolated at the planet's separation ---
    # PSF_profile_6D   : (band, config, l0, WFE, IWA, separation)
    # fraction_core_6D : (band, config, l0, WFE, IWA, separation)
    PSF_profile_sep_5D   = interp_flat_last_axis(separation, PSF_profile_6D,   separation_planet, dtype=dtype) # (band, config, l0, WFE, IWA)
    fraction_core_sep_5D = interp_flat_last_axis(separation, fraction_core_6D, separation_planet, dtype=dtype) # (band, config, l0, WFE, IWA)
    
    
    # --- PSF data at ref values (bands, configs, *) ---
    PSF_profile_sep_3D_l0    = PSF_profile_sep_5D[:,   :, :, idx_WFE_ref, idx_IWA_ref]                          # [star flux fraction/px]     (band, config, l0)
    fraction_core_sep_3D_l0  = fraction_core_sep_5D[:, :, :, idx_WFE_ref, idx_IWA_ref]                          # [planet flux fraction/FWHM] (band, config, l0)
    PSF_profile_sep_3D_WFE   = PSF_profile_sep_5D[np.arange(N_band),   :, idx_l0_ref, :, idx_IWA_ref]           # [star flux fraction/px]     (band, config, WFE)
    fraction_core_sep_3D_WFE = fraction_core_sep_5D[np.arange(N_band), :, idx_l0_ref, :, idx_IWA_ref]           # [planet flux fraction/FWHM] (band, config, WFE)
    PSF_profile_sep_3D_IWA   = PSF_profile_sep_5D[np.arange(N_band),   :, idx_l0_ref, idx_WFE_ref, :]           # [star flux fraction/px]     (band, config, IWA)
    fraction_core_sep_3D_IWA = fraction_core_sep_5D[np.arange(N_band), :, idx_l0_ref, idx_WFE_ref, :]           # [planet flux fraction/FWHM] (band, config, IWA)
    PSF_profile_sep_2D_ref   = PSF_profile_sep_5D[np.arange(N_band),   :, idx_l0_ref, idx_WFE_ref, idx_IWA_ref] # [star flux fraction/px]     (band, config)
    fraction_core_sep_2D_ref = fraction_core_sep_5D[np.arange(N_band), :, idx_l0_ref, idx_WFE_ref, idx_IWA_ref] # [planet flux fraction/FWHM] (band, config)
    
    
    # --- Computing the planet and star models for the planet row in [J/s/m²/µm] ---
    planet_spectrum, _, _, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True, in_star_mag=True, interpolated_spectrum=True)
    star_spectrum                        = star_spectrum.interpolate_wavelength(wave_instru,   renorm=False) # Interpolating on wave_instru [J/s/m2/µm]
    planet_spectrum                      = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru [J/s/m2/µm]
        
    # --- Converting in [ph/mn/µm] ---
    star_spectrum.flux   *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]
    planet_spectrum.flux *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]
    
    
    # --- Pre-computing downgraded spectra and HF/LF splits
    star_1D_list      = [None] * N_R # [ph/mn/bin]
    planet_1D_list    = [None] * N_R # [ph/mn/bin]
    star_HF_1D_list   = [None] * N_R # [ph/mn/bin]
    star_LF_1D_list   = [None] * N_R # [ph/mn/bin]
    planet_HF_1D_list = [None] * N_R # [ph/mn/bin]
    planet_LF_1D_list = [None] * N_R # [ph/mn/bin]
    for i, res in enumerate(R):
        wave_i   = wave_1D_list[i]  # [µm]
        dwave_i  = dwave_1D_list[i] # [µm/bin]
        star_i   = star_spectrum.degrade_resolution(wave_i,   renorm=False, R_output=res).flux # [ph/mn/µm]
        planet_i = planet_spectrum.degrade_resolution(wave_i, renorm=False, R_output=res).flux # [ph/mn/µm]
    
        # Converting in [ph/mn/bin]
        star_i   *= dwave_i # [ph/mn/µm] => [ph/mn/bin]
        planet_i *= dwave_i # [ph/mn/µm] => [ph/mn/bin]
        
        # HF/LF split for MM
        if post_processing == "MM": # by definition res == R_nyquist (from get_resolution(wave_i))
            star_HF_i, star_LF_i     = filtered_flux(flux=star_i,   R=res, Rc=Rc, filter_type=filter_type) # [ph/mn/bin]
            planet_HF_i, planet_LF_i = filtered_flux(flux=planet_i, R=res, Rc=Rc, filter_type=filter_type) # [ph/mn/bin]
        
        # Saving in the lists
        star_1D_list[i]   = star_i   # [ph/mn/bin]
        planet_1D_list[i] = planet_i # [ph/mn/bin]
        if post_processing == "MM":
            star_HF_1D_list[i]   = star_HF_i   # [ph/mn/bin]
            star_LF_1D_list[i]   = star_LF_i   # [ph/mn/bin]
            planet_HF_1D_list[i] = planet_HF_i # [ph/mn/bin]
            planet_LF_1D_list[i] = planet_LF_i # [ph/mn/bin]
    
    
    # --- Computing data (signal, stellar halo photon noise variance, max star flux and base systematic noise) depending only on R, l0 and Nl ---
    signal_2D_R            = np.zeros((N_band, N_R), dtype=dtype) # alpha - beta                         [ph/mn]
    sigma_halo_2_2D_R      = np.zeros((N_band, N_R), dtype=dtype) # sum ( trans * S_star * template**2 ) [ph/mn]
    max_star_flux_2D_R     = np.zeros((N_band, N_R), dtype=dtype) # max ( trans * S_star )               [ph/mn] in the brightest spectral bin
    sigma_bkg_2_2D_R       = np.zeros((N_band, N_R), dtype=dtype) # sum ( background * template**2 )     [ph/px/mn]
    sigma_syst_base_2_2D_R = np.zeros((N_band, N_R), dtype=dtype) # sigma_syst_base**2                   [ph/mn]**2 (assuming sigma_m = 1)
    
    signal_2D_l0            = np.zeros((N_band, N_l0), dtype=dtype) # alpha - beta                         [ph/mn]
    sigma_halo_2_2D_l0      = np.zeros((N_band, N_l0), dtype=dtype) # sum ( trans * S_star * template**2 ) [ph/mn]
    max_star_flux_2D_l0     = np.zeros((N_band, N_l0), dtype=dtype) # max ( trans * S_star )               [ph/mn] in the brightest spectral bin
    sigma_bkg_2_2D_l0       = np.zeros((N_band, N_l0), dtype=dtype) # sum ( background * template**2 )     [ph/px/mn]
    sigma_syst_base_2_2D_l0 = np.zeros((N_band, N_l0), dtype=dtype) # sigma_syst_base**2                   [ph/mn]**2 (assuming sigma_m = 1)
    
    signal_2D_Nl            = np.zeros((N_band, N_Nl), dtype=dtype) # alpha - beta                         [ph/mn]
    sigma_halo_2_2D_Nl      = np.zeros((N_band, N_Nl), dtype=dtype) # sum ( trans * S_star * template**2 ) [ph/mn]
    max_star_flux_2D_Nl     = np.zeros((N_band, N_Nl), dtype=dtype) # max ( trans * S_star )               [ph/mn] in the brightest spectral bin
    sigma_bkg_2_2D_Nl       = np.zeros((N_band, N_Nl), dtype=dtype) # sum ( background * template**2 )     [ph/px/mn]
    sigma_syst_base_2_2D_Nl = np.zeros((N_band, N_Nl), dtype=dtype) # sigma_syst_base**2                   [ph/mn]**2 (assuming sigma_m = 1)
    
    # --- Global band loop ---
    for ib in range(N_band):
        
        # --- Reference index for this band ---
        i_ref            = idx_R_ref[ib]
        j_ref            = idx_l0_ref[ib]
        k_ref            = idx_Nl_ref
        pxscale          = pxscales[ib]                  # [mas/px]
        pxscale_rad      = pxscale / (rad2arcsec * 1000) # [rad/px]
        background_scale = pxscales[ib]**2               # [mas/px]**2

        
        # --- Resolution loop for *_R quantities ---
        for i, res in enumerate(R):
            wave_i         = wave_1D_list[i]                       # [µm]
            trans_tell_i   = trans_tell_1D_list[i]                 # [no unit]
            background_i   = background_1D_list[i]                 # [ph/mas2/mn/bin]
            star_i         = star_1D_list[i]                       # [ph/mn/bin]
            planet_i       = planet_1D_list[i]                     # [ph/mn/bin]
            star_HF_i      = star_HF_1D_list[i]                    # [ph/mn/bin]
            star_LF_i      = star_LF_1D_list[i]                    # [ph/mn/bin]
            planet_HF_i    = planet_HF_1D_list[i]                  # [ph/mn/bin]
            planet_LF_i    = planet_LF_1D_list[i]                  # [ph/mn/bin]
            idx_lo, idx_hi = range_1D_list[i]
            sl             = slice(idx_lo[j_ref, k_ref], idx_hi[j_ref, k_ref]) # Cropped range for (l0, Nl)
            valid_i        = valid_2D_list[ib][i]                  #  Valid range (usefull if force_inside_bands)
            signal_2D_R[ib, i], sigma_halo_2_2D_R[ib, i], max_star_flux_2D_R[ib, i], sigma_bkg_2_2D_R[ib, i], sigma_syst_base_2_2D_R[ib, i] = metrics_slice(sl=sl, post_processing=post_processing, wave_i=wave_i, trans_tell_i=trans_tell_i, star_i=star_i, planet_i=planet_i, background_i=background_i, star_HF_i=star_HF_i, star_LF_i=star_LF_i, planet_HF_i=planet_HF_i, planet_LF_i=planet_LF_i, valid_i=valid_i, separation_planet_rad=separation_planet_rad, separation_planet_rad_arr=separation_planet_rad_arr, pxscale_rad=pxscale_rad, D=D, background_scale=background_scale, fast_comp=True)

    
        # --- Data at R_ref (for l0 and Nl loops) ---
        wave_i         = wave_1D_list[i_ref]                       # [µm]
        trans_tell_i   = trans_tell_1D_list[i_ref]                 # [no unit]
        background_i   = background_1D_list[i_ref]                 # [ph/mas2/mn/bin]
        star_i         = star_1D_list[i_ref]                       # [ph/mn/bin]
        planet_i       = planet_1D_list[i_ref]                     # [ph/mn/bin]
        star_HF_i      = star_HF_1D_list[i_ref]                    # [ph/mn/bin]
        star_LF_i      = star_LF_1D_list[i_ref]                    # [ph/mn/bin]
        planet_HF_i    = planet_HF_1D_list[i_ref]                  # [ph/mn/bin]
        planet_LF_i    = planet_LF_1D_list[i_ref]                  # [ph/mn/bin]
        idx_lo, idx_hi = range_1D_list[i_ref]
        valid_i        = valid_2D_list[ib][i_ref]                  #  Valid range (usefull if force_inside_bands)

    
        # --- l0 loop for *_l0 quantities ---
        for j in range(N_l0): # for each l0
            sl = slice(idx_lo[j, k_ref], idx_hi[j, k_ref]) # cropped range for (l0, Nl)
            signal_2D_l0[ib, j], sigma_halo_2_2D_l0[ib, j], max_star_flux_2D_l0[ib, j], sigma_bkg_2_2D_l0[ib, j], sigma_syst_base_2_2D_l0[ib, j] = metrics_slice(sl=sl, post_processing=post_processing, wave_i=wave_i, trans_tell_i=trans_tell_i, star_i=star_i, planet_i=planet_i, background_i=background_i, star_HF_i=star_HF_i, star_LF_i=star_LF_i, planet_HF_i=planet_HF_i, planet_LF_i=planet_LF_i, valid_i=valid_i, separation_planet_rad=separation_planet_rad, separation_planet_rad_arr=separation_planet_rad_arr, pxscale_rad=pxscale_rad, D=D, background_scale=background_scale, fast_comp=True)
    
        
        # --- Nl loop for *_Nl quantities ---
        for k in range(N_Nl): # for each Nl
            sl = slice(idx_lo[j_ref, k], idx_hi[j_ref, k]) # cropped range for (l0, Nl)
            signal_2D_Nl[ib, k], sigma_halo_2_2D_Nl[ib, k], max_star_flux_2D_Nl[ib, k], sigma_bkg_2_2D_Nl[ib, k], sigma_syst_base_2_2D_Nl[ib, k] = metrics_slice(sl=sl, post_processing=post_processing, wave_i=wave_i, trans_tell_i=trans_tell_i, star_i=star_i, planet_i=planet_i, background_i=background_i, star_HF_i=star_HF_i, star_LF_i=star_LF_i, planet_HF_i=planet_HF_i, planet_LF_i=planet_LF_i, valid_i=valid_i, separation_planet_rad=separation_planet_rad, separation_planet_rad_arr=separation_planet_rad_arr, pxscale_rad=pxscale_rad, D=D, background_scale=background_scale, fast_comp=True)
        
    
    # --- Computing DIT length in [mn/DIT] --- 
    
    # max_star_flux at the brightest pixel in [ph/px/mn] in the brightest pixel (through coronagraph, if any):
    # [ph/mn] in the brightest bin over the FoV * [star flux fraction/px] in the brightest pixel => [ph/px/mn] in the brightest pixel
    # (bands, *)                                * (bands, configs, l0)                           => (bands, configs, *)
    max_star_flux_1D_ref = max_star_flux_2D_R[np.arange(N_band), idx_R_ref]                         # (bands)
    max_star_flux_3D_R   = max_star_flux_2D_R[:, None, :]      * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, R)
    max_star_flux_3D_l0  = max_star_flux_2D_l0[:, None, :]     * PSF_profile_sep_3D_l0[:, :, :]     # (bands, configs, l0)
    max_star_flux_3D_Nl  = max_star_flux_2D_Nl[:, None, :]     * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, Nl)
    max_star_flux_3D_WFE = max_star_flux_1D_ref[:, None, None] * PSF_profile_sep_3D_WFE[:, :, :]    # (bands, configs, WFE)
    max_star_flux_3D_IWA = max_star_flux_1D_ref[:, None, None] * PSF_profile_sep_3D_IWA[:, :, :]    # (bands, configs, IWA)
    max_star_flux_2D_ref = max_star_flux_3D_R[np.arange(N_band), :, idx_R_ref]                      # (bands, configs)
    
    # max_star_flux at the brightest pixel in [e-/px/mn] in the brightest pixel with instrumental transmission (through coronagraph, if any):
    # [ph/px/mn] in the brightest pixel * [e-/ph]          => [e-/px/mn] in the brightest pixel
    # (bands, configs, *)               * (bands, configs) => (bands, configs, *)
    max_star_flux_3D_R   = max_star_flux_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    max_star_flux_3D_l0  = max_star_flux_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    max_star_flux_3D_Nl  = max_star_flux_3D_Nl              * trans_instru_ref[:, :, None] # (bands, configs, Nl)
    max_star_flux_3D_TI  = max_star_flux_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    max_star_flux_3D_WFE = max_star_flux_3D_WFE             * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    max_star_flux_3D_IWA = max_star_flux_3D_IWA             * trans_instru_ref[:, :, None] # (bands, configs, IWA)

    # Computing DIT length in [mn/DIT] (+ clipping between min and max DIT)
    DIT_3D_R   = saturation_e / max_star_flux_3D_R     # (bands, configs, R)   [mn/DIT]
    DIT_3D_l0  = saturation_e / max_star_flux_3D_l0    # (bands, configs, l0)  [mn/DIT]
    DIT_3D_Nl  = saturation_e / max_star_flux_3D_Nl    # (bands, configs, Nl)  [mn/DIT]
    DIT_3D_TI  = saturation_e / max_star_flux_3D_TI    # (bands, configs, TI)  [mn/DIT]
    DIT_3D_WFE = saturation_e / max_star_flux_3D_WFE   # (bands, configs, WFE) [mn/DIT]
    DIT_3D_IWA = saturation_e / max_star_flux_3D_IWA   # (bands, configs, IWA) [mn/DIT]
    DIT_3D_R   = np.clip(DIT_3D_R, min_DIT, max_DIT)   # (bands, configs, R)   [mn/DIT]
    DIT_3D_l0  = np.clip(DIT_3D_l0, min_DIT, max_DIT)  # (bands, configs, l0)  [mn/DIT]
    DIT_3D_Nl  = np.clip(DIT_3D_Nl, min_DIT, max_DIT)  # (bands, configs, Nl)  [mn/DIT]
    DIT_3D_TI  = np.clip(DIT_3D_TI, min_DIT, max_DIT)  # (bands, configs, TI)  [mn/DIT]
    DIT_3D_WFE = np.clip(DIT_3D_WFE, min_DIT, max_DIT) # (bands, configs, WFE) [mn/DIT]
    DIT_3D_IWA = np.clip(DIT_3D_IWA, min_DIT, max_DIT) # (bands, configs, IWA) [mn/DIT]

    
    # --- Computing signal in [e-/FWHM/DIT] (R, l0, Nl, WFE, IWA, TI) ---
    
    # Signal at the planet's separation inside the FWHM in [ph/FWHM/mn] (throught coronagraph, if any)
    # [ph/mn] over the FoV * [planet flux fraction/FWHM] => [ph/FWHM/mn]
    # (bands, *)           * (bands, configs, l0)                           => (bands, configs, *)
    signal_1D_ref = signal_2D_R[np.arange(N_band), idx_R_ref]                           # (bands)
    signal_3D_R   = signal_2D_R[:, None, :]      * fraction_core_sep_2D_ref[:, :, None] # (bands, configs, R)
    signal_3D_l0  = signal_2D_l0[:, None, :]     * fraction_core_sep_3D_l0[:, :, :]     # (bands, configs, l0)
    signal_3D_Nl  = signal_2D_Nl[:, None, :]     * fraction_core_sep_2D_ref[:, :, None] # (bands, configs, Nl)
    signal_3D_WFE = signal_1D_ref[:, None, None] * fraction_core_sep_3D_WFE[:, :, :]    # (bands, configs, WFE)
    signal_3D_IWA = signal_1D_ref[:, None, None] * fraction_core_sep_3D_IWA[:, :, :]    # (bands, configs, IWA)
    signal_2D_ref = signal_3D_R[np.arange(N_band), :, idx_R_ref]                        # (bands, configs)
    
    # Signal at the planet's separation inside the FWHM in [e-/FWHM/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/FWHM/mn]        * [e-/ph]          => [e-/FWHM/mn]
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    signal_3D_R   = signal_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    signal_3D_l0  = signal_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    signal_3D_Nl  = signal_3D_Nl              * trans_instru_ref[:, :, None] # (bands, configs, Nl)
    signal_3D_TI  = signal_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    signal_3D_WFE = signal_3D_WFE             * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    signal_3D_IWA = signal_3D_IWA             * trans_instru_ref[:, :, None] # (bands, configs, IWA)


    # Signal at the planet's separation inside the FWHM in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT (throught coronagraph, if any)
    # [e-/FWHM/mn]        * [mn/DIT]            => [e-/FWHM/DIT]
    # (bands, configs, *) * (bands, configs, *) => (bands, configs, *)
    signal_3D_R   = signal_3D_R   * DIT_3D_R   # (bands, configs, R)
    signal_3D_l0  = signal_3D_l0  * DIT_3D_l0  # (bands, configs, l0)
    signal_3D_Nl  = signal_3D_Nl  * DIT_3D_Nl  # (bands, configs, Nl)
    signal_3D_TI  = signal_3D_TI  * DIT_3D_TI  # (bands, configs, TI)
    signal_3D_WFE = signal_3D_WFE * DIT_3D_WFE # (bands, configs, WFE)
    signal_3D_IWA = signal_3D_IWA * DIT_3D_IWA # (bands, configs, IWA)

    
    # --- Computing stellar halo photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Nl, WFE, IWA, TI) ---
    
    # Stellar halo photon noise variance (Poisson) at the planet's separation in [ph/px/mn] (throught coronagraph, if any)
    # [ph/mn] over the FoV * [star flux fraction/px] => [ph/px/mn]
    # (bands, *)           * (bands, configs, l0)    => (bands, configs, *)
    sigma_halo_2_1D_ref = sigma_halo_2_2D_R[np.arange(N_band), idx_R_ref]
    sigma_halo_2_3D_R   = sigma_halo_2_2D_R[:, None, :]      * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, R)
    sigma_halo_2_3D_l0  = sigma_halo_2_2D_l0[:, None, :]     * PSF_profile_sep_3D_l0[:, :, :]     # (bands, configs, l0)
    sigma_halo_2_3D_Nl  = sigma_halo_2_2D_Nl[:, None, :]     * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, Nl)
    sigma_halo_2_3D_WFE = sigma_halo_2_1D_ref[:, None, None] * PSF_profile_sep_3D_WFE[:, :, :]    # (bands, configs, WFE)
    sigma_halo_2_3D_IWA = sigma_halo_2_1D_ref[:, None, None] * PSF_profile_sep_3D_IWA[:, :, :]    # (bands, configs, IWA)
    sigma_halo_2_2D_ref = sigma_halo_2_3D_R[np.arange(N_band), :, idx_R_ref]                      # (bands, configs)
    
    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/px/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]          * [e-/ph]          => [e-/px/mn]
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    sigma_halo_2_3D_R   = sigma_halo_2_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    sigma_halo_2_3D_l0  = sigma_halo_2_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    sigma_halo_2_3D_Nl  = sigma_halo_2_3D_Nl              * trans_instru_ref[:, :, None] # (bands, configs, Nl)
    sigma_halo_2_3D_TI  = sigma_halo_2_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    sigma_halo_2_3D_WFE = sigma_halo_2_3D_WFE             * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    sigma_halo_2_3D_IWA = sigma_halo_2_3D_IWA             * trans_instru_ref[:, :, None] # (bands, configs, IWA)

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance) (throught coronagraph, if any)
    # [e-/px/mn]          * [px/FWHM]*[mn/DIT]  => [e-/FWHM/DIT]
    # (bands, configs, *) * (bands, configs, *) => (bands, configs, *)
    sigma_halo_2_3D_R   = sigma_halo_2_3D_R   * A_FWHM*DIT_3D_R   # (bands, configs, R)
    sigma_halo_2_3D_l0  = sigma_halo_2_3D_l0  * A_FWHM*DIT_3D_l0  # (bands, configs, l0)
    sigma_halo_2_3D_Nl  = sigma_halo_2_3D_Nl  * A_FWHM*DIT_3D_Nl  # (bands, configs, Nl)
    sigma_halo_2_3D_TI  = sigma_halo_2_3D_TI  * A_FWHM*DIT_3D_TI  # (bands, configs, TI)
    sigma_halo_2_3D_WFE = sigma_halo_2_3D_WFE * A_FWHM*DIT_3D_WFE # (bands, configs, WFE)
    sigma_halo_2_3D_IWA = sigma_halo_2_3D_IWA * A_FWHM*DIT_3D_IWA # (bands, configs, IWA)

    
    # --- Computing stellar halo systematic noise variance (assuming sigma_m = 1) in [e-/FWHM/DIT]**2 (R, l0, Nl, WFE, IWA, TI) (all quantities are squared since sigma_syst**2 propto stellar_halo**2) ---
    
    # Stellar halo systematic noise variance at the planet's separation in [ph/px/mn]**2 (throught coronagraph, if any)
    # [ph/mn]**2 over the FoV * [star flux fraction/px]**2 => [ph/px/mn]**2
    # (bands, *)              * (bands, configs, l0)       => (bands, configs, *)
    sigma_syst_base_2_1D_ref = sigma_syst_base_2_2D_R[np.arange(N_band), idx_R_ref]
    sigma_syst_base_2_3D_R   = sigma_syst_base_2_2D_R[:, None, :]      * PSF_profile_sep_2D_ref[:, :, None]**2 # (bands, configs, R)
    sigma_syst_base_2_3D_l0  = sigma_syst_base_2_2D_l0[:, None, :]     * PSF_profile_sep_3D_l0[:, :, :]**2     # (bands, configs, l0)
    sigma_syst_base_2_3D_Nl  = sigma_syst_base_2_2D_Nl[:, None, :]     * PSF_profile_sep_2D_ref[:, :, None]**2 # (bands, configs, Nl)
    sigma_syst_base_2_3D_WFE = sigma_syst_base_2_1D_ref[:, None, None] * PSF_profile_sep_3D_WFE[:, :, :]**2    # (bands, configs, WFE)
    sigma_syst_base_2_3D_IWA = sigma_syst_base_2_1D_ref[:, None, None] * PSF_profile_sep_3D_IWA[:, :, :]**2    # (bands, configs, IWA)
    sigma_syst_base_2_2D_ref = sigma_syst_base_2_3D_R[np.arange(N_band), :, idx_R_ref]                         # (bands, configs)
    
    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]**2       * [e-/ph]**2       => [e-/px/mn]**2
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    sigma_syst_base_2_3D_R   = sigma_syst_base_2_3D_R               * trans_instru_ref[:, :, None]**2 # (bands, configs, R)
    sigma_syst_base_2_3D_l0  = sigma_syst_base_2_3D_l0              * trans_instru_ref[:, :, None]**2 # (bands, configs, l0)
    sigma_syst_base_2_3D_Nl  = sigma_syst_base_2_3D_Nl              * trans_instru_ref[:, :, None]**2 # (bands, configs, Nl)
    sigma_syst_base_2_3D_TI  = sigma_syst_base_2_2D_ref[:, :, None] * trans_instru[None, None, :]**2  # (bands, configs, TI)
    sigma_syst_base_2_3D_WFE = sigma_syst_base_2_3D_WFE             * trans_instru_ref[:, :, None]**2 # (bands, configs, WFE)
    sigma_syst_base_2_3D_IWA = sigma_syst_base_2_3D_IWA             * trans_instru_ref[:, :, None]**2 # (bands, configs, WFE)

    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission integrated over the DIT and the FWHM (throught coronagraph, if any) (we take A_FWHM**2 * DIT**2 since sigma_syst_base_2 propto H_integrated**2 propto A_FWHM**2*DIT**2, where H_integrated = the halo integrated over FWHM boxes and the DIT)
    # [e-/px/mn]**2       * [px/FWHM]**2*[mn/DIT]**2  => [e-/FWHM/DIT]**2
    # (bands, configs, *) * (bands, configs, *)       => (bands, configs, *)
    sigma_syst_base_2_3D_R   = sigma_syst_base_2_3D_R   * A_FWHM**2*DIT_3D_R**2   # (bands, configs, R)
    sigma_syst_base_2_3D_l0  = sigma_syst_base_2_3D_l0  * A_FWHM**2*DIT_3D_l0**2  # (bands, configs, l0)
    sigma_syst_base_2_3D_Nl  = sigma_syst_base_2_3D_Nl  * A_FWHM**2*DIT_3D_Nl**2  # (bands, configs, Nl)
    sigma_syst_base_2_3D_TI  = sigma_syst_base_2_3D_TI  * A_FWHM**2*DIT_3D_TI**2  # (bands, configs, TI)
    sigma_syst_base_2_3D_WFE = sigma_syst_base_2_3D_WFE * A_FWHM**2*DIT_3D_WFE**2 # (bands, configs, WFE)
    sigma_syst_base_2_3D_IWA = sigma_syst_base_2_3D_IWA * A_FWHM**2*DIT_3D_IWA**2 # (bands, configs, WFE)

    
    # --- Computing background photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Nl, WFE, IWA, TI) ---
    
    # (bands, *) => (bands, configs, *)
    sigma_bkg_2_1D_ref = sigma_bkg_2_2D_R[np.arange(N_band), idx_R_ref]
    sigma_bkg_2_3D_R   = sigma_bkg_2_2D_R[:, None, :]      * np.ones((1, N_config, 1)) # (bands, configs, R)
    sigma_bkg_2_3D_l0  = sigma_bkg_2_2D_l0[:, None, :]     * np.ones((1, N_config, 1)) # (bands, configs, l0)
    sigma_bkg_2_3D_Nl  = sigma_bkg_2_2D_Nl[:, None, :]     * np.ones((1, N_config, 1)) # (bands, configs, Nl)
    sigma_bkg_2_3D_WFE = sigma_bkg_2_1D_ref[:, None, None] * np.ones((1, N_config, 1)) # (bands, configs, WFE)
    sigma_bkg_2_3D_IWA = sigma_bkg_2_1D_ref[:, None, None] * np.ones((1, N_config, 1)) # (bands, configs, IWA)
    sigma_bkg_2_2D_ref = sigma_bkg_2_3D_R[np.arange(N_band), :, idx_R_ref]             # (bands, configs)

    # Background photon noise variance (Poisson) in [e-/px/mn] with instrumental transmission
    # [ph/px/mn]          * [e-/ph]          => [e-/px/mn]
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    sigma_bkg_2_3D_R   = sigma_bkg_2_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    sigma_bkg_2_3D_l0  = sigma_bkg_2_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    sigma_bkg_2_3D_Nl  = sigma_bkg_2_3D_Nl              * trans_instru_ref[:, :, None] # (bands, configs, Nl)
    sigma_bkg_2_3D_TI  = sigma_bkg_2_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    sigma_bkg_2_3D_WFE  = sigma_bkg_2_3D_WFE            * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    sigma_bkg_2_3D_IWA  = sigma_bkg_2_3D_IWA            * trans_instru_ref[:, :, None] # (bands, configs, IWA)

    # Background photon noise variance (Poisson) in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance)
    # [e-/px/mn]          * [px/FWHM]*[mn/DIT]  => [e-/FWHM/DIT]
    # (bands, configs, *) * (bands, configs, *) => (bands, configs, *)
    sigma_bkg_2_3D_R   = sigma_bkg_2_3D_R   * A_FWHM*DIT_3D_R   # (bands, configs, R)
    sigma_bkg_2_3D_l0  = sigma_bkg_2_3D_l0  * A_FWHM*DIT_3D_l0  # (bands, configs, l0)
    sigma_bkg_2_3D_Nl  = sigma_bkg_2_3D_Nl  * A_FWHM*DIT_3D_Nl  # (bands, configs, Nl)
    sigma_bkg_2_3D_TI  = sigma_bkg_2_3D_TI  * A_FWHM*DIT_3D_TI  # (bands, configs, TI)
    sigma_bkg_2_3D_WFE = sigma_bkg_2_3D_WFE * A_FWHM*DIT_3D_WFE # (bands, configs, WFE)
    sigma_bkg_2_3D_IWA = sigma_bkg_2_3D_IWA * A_FWHM*DIT_3D_IWA # (bands, configs, IWA)

    
    _CTX["signal_4D_R_mm"][idx]   = signal_3D_R
    _CTX["signal_4D_l0_mm"][idx]  = signal_3D_l0
    _CTX["signal_4D_Nl_mm"][idx]  = signal_3D_Nl
    _CTX["signal_4D_TI_mm"][idx]  = signal_3D_TI
    _CTX["signal_4D_WFE_mm"][idx] = signal_3D_WFE
    _CTX["signal_4D_IWA_mm"][idx] = signal_3D_IWA

    _CTX["sigma_halo_2_4D_R_mm"][idx]   = sigma_halo_2_3D_R
    _CTX["sigma_halo_2_4D_l0_mm"][idx]  = sigma_halo_2_3D_l0
    _CTX["sigma_halo_2_4D_Nl_mm"][idx]  = sigma_halo_2_3D_Nl
    _CTX["sigma_halo_2_4D_TI_mm"][idx]  = sigma_halo_2_3D_TI
    _CTX["sigma_halo_2_4D_WFE_mm"][idx] = sigma_halo_2_3D_WFE
    _CTX["sigma_halo_2_4D_IWA_mm"][idx] = sigma_halo_2_3D_IWA

    _CTX["sigma_syst_base_2_4D_R_mm"][idx]   = sigma_syst_base_2_3D_R
    _CTX["sigma_syst_base_2_4D_l0_mm"][idx]  = sigma_syst_base_2_3D_l0
    _CTX["sigma_syst_base_2_4D_Nl_mm"][idx]  = sigma_syst_base_2_3D_Nl
    _CTX["sigma_syst_base_2_4D_TI_mm"][idx]  = sigma_syst_base_2_3D_TI
    _CTX["sigma_syst_base_2_4D_WFE_mm"][idx] = sigma_syst_base_2_3D_WFE
    _CTX["sigma_syst_base_2_4D_IWA_mm"][idx] = sigma_syst_base_2_3D_IWA

    _CTX["sigma_bkg_2_4D_R_mm"][idx]   = sigma_bkg_2_3D_R
    _CTX["sigma_bkg_2_4D_l0_mm"][idx]  = sigma_bkg_2_3D_l0
    _CTX["sigma_bkg_2_4D_Nl_mm"][idx]  = sigma_bkg_2_3D_Nl
    _CTX["sigma_bkg_2_4D_TI_mm"][idx]  = sigma_bkg_2_3D_TI
    _CTX["sigma_bkg_2_4D_WFE_mm"][idx] = sigma_bkg_2_3D_WFE
    _CTX["sigma_bkg_2_4D_IWA_mm"][idx] = sigma_bkg_2_3D_IWA

    _CTX["DIT_4D_R_mm"][idx]   = DIT_3D_R
    _CTX["DIT_4D_l0_mm"][idx]  = DIT_3D_l0
    _CTX["DIT_4D_Nl_mm"][idx]  = DIT_3D_Nl
    _CTX["DIT_4D_TI_mm"][idx]  = DIT_3D_TI
    _CTX["DIT_4D_WFE_mm"][idx] = DIT_3D_WFE
    _CTX["DIT_4D_IWA_mm"][idx] = DIT_3D_IWA
    
    return idx



def get_SNR(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, NbPixel, SNR_thr, separation_planets,
             idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Nl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
             signal_4D_R,   sigma_halo_2_4D_R,   sigma_syst_base_2_4D_R,   sigma_bkg_2_4D_R,   DIT_4D_R,
             signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_syst_base_2_4D_l0,  sigma_bkg_2_4D_l0,  DIT_4D_l0,
             signal_4D_Nl,  sigma_halo_2_4D_Nl,  sigma_syst_base_2_4D_Nl,  sigma_bkg_2_4D_Nl,  DIT_4D_Nl,
             signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_syst_base_2_4D_TI,  sigma_bkg_2_4D_TI,  DIT_4D_TI,
             signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_syst_base_2_4D_WFE, sigma_bkg_2_4D_WFE, DIT_4D_WFE,
             signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_syst_base_2_4D_IWA, sigma_bkg_2_4D_IWA, DIT_4D_IWA):
    
    # Data shape = (planets, bands, configs, *)
    
    # --- Number of DIT ---
    N_DIT_4D_R = np.floor(exposure_time / DIT_4D_R).astype(np.int32)
    N_DIT_4D_R = np.clip(N_DIT_4D_R, 1, None).astype(dtype)
    
    N_DIT_4D_l0 = np.floor(exposure_time / DIT_4D_l0).astype(np.int32)
    N_DIT_4D_l0 = np.clip(N_DIT_4D_l0, 1, None).astype(dtype)
    
    N_DIT_4D_Nl = np.floor(exposure_time / DIT_4D_Nl).astype(np.int32)
    N_DIT_4D_Nl = np.clip(N_DIT_4D_Nl, 1, None).astype(dtype)
    
    N_DIT_4D_TI = np.floor(exposure_time / DIT_4D_TI).astype(np.int32)
    N_DIT_4D_TI = np.clip(N_DIT_4D_TI, 1, None).astype(dtype)
    
    N_DIT_4D_WFE = np.floor(exposure_time / DIT_4D_WFE).astype(np.int32)
    N_DIT_4D_WFE = np.clip(N_DIT_4D_WFE, 1, None).astype(dtype)
    
    N_DIT_4D_IWA = np.floor(exposure_time / DIT_4D_IWA).astype(np.int32)
    N_DIT_4D_IWA = np.clip(N_DIT_4D_IWA, 1, None).astype(dtype)

    # --- Effective RON0 variance in Up-The-Ramp read-out mode (UTR) in [e-/FWHM/DIT]**2---
    sigma_RON_2_4D_R   = sigma_RON_2_from_DIT(DIT_4D_R,   min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_l0  = sigma_RON_2_from_DIT(DIT_4D_l0,  min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_Nl  = sigma_RON_2_from_DIT(DIT_4D_Nl,  min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_TI  = sigma_RON_2_from_DIT(DIT_4D_TI,  min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_WFE = sigma_RON_2_from_DIT(DIT_4D_WFE, min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_IWA = sigma_RON_2_from_DIT(DIT_4D_IWA, min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)

    # --- Dark current photon noise variance (Poisson) [e-/FWHM/DIT] ---
    sigma_DC_2_4D_R   = (DC0 * DIT_4D_R   * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_l0  = (DC0 * DIT_4D_l0  * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_Nl  = (DC0 * DIT_4D_Nl  * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_TI  = (DC0 * DIT_4D_TI  * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_WFE = (DC0 * DIT_4D_WFE * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_IWA = (DC0 * DIT_4D_IWA * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    
    # --- Pixel detector projection factor, adds quadratically (NbPixel = Number of detector px per cube px) ---
    if NbPixel is not None:
        sigma_RON_2_4D_R   *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_l0  *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_Nl  *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_TI  *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_WFE *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_IWA *= NbPixel # [e-/FWHM/DIT]
        
        sigma_DC_2_4D_R   *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_l0  *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_Nl  *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_TI  *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_WFE *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_IWA *= NbPixel # [e-/FWHM/DIT]
    
    # --- Reference quantities (for sigma_m axis) ---
    idx = idx_R_ref[None, :, None, None]                    # (1, band, 1, 1)
    idx = np.broadcast_to(idx, (N_PT, N_band, N_config, 1)) # (planet, band, config, 1)
    N_DIT_4D_ref             = np.take_along_axis(N_DIT_4D_R,             idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    signal_4D_ref            = np.take_along_axis(signal_4D_R,            idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_halo_2_4D_ref      = np.take_along_axis(sigma_halo_2_4D_R,      idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_bkg_2_4D_ref       = np.take_along_axis(sigma_bkg_2_4D_R,       idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_RON_2_4D_ref       = np.take_along_axis(sigma_RON_2_4D_R,       idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_DC_2_4D_ref        = np.take_along_axis(sigma_DC_2_4D_R,        idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_syst_base_2_4D_ref = np.take_along_axis(sigma_syst_base_2_4D_R, idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
        
    # --- sigma_syst^2 = sigma_m^2 * sigma_syst_base_2, by definition of sigma_syst_base_2 (sigma_m = residuals amplitude over the FWHM and over the entire integration exposure_time) ---
    sigma_syst_2_4D_R   = sigma_m_ref**2                  * sigma_syst_base_2_4D_R    # (planets, bands, configs, R)
    sigma_syst_2_4D_l0  = sigma_m_ref**2                  * sigma_syst_base_2_4D_l0   # (planets, bands, configs, l0)
    sigma_syst_2_4D_Nl  = sigma_m_ref**2                  * sigma_syst_base_2_4D_Nl   # (planets, bands, configs, Nl)
    sigma_syst_2_4D_TI  = sigma_m_ref**2                  * sigma_syst_base_2_4D_TI   # (planets, bands, configs, TI)
    sigma_syst_2_4D_SM  = sigma_m[None, None, None, :]**2 * sigma_syst_base_2_4D_ref  # (planets, bands, configs, SM)
    sigma_syst_2_4D_WFE = sigma_m_ref**2                  * sigma_syst_base_2_4D_WFE  # (planets, bands, configs, WFE)
    sigma_syst_2_4D_IWA = sigma_m_ref**2                  * sigma_syst_base_2_4D_IWA  # (planets, bands, configs, IWA)

    # --- SNR ---
    SNR_4D_R   = N_DIT_4D_R   * signal_4D_R   / np.sqrt( N_DIT_4D_R   * (sigma_halo_2_4D_R   + sigma_bkg_2_4D_R   + sigma_RON_2_4D_R   + sigma_DC_2_4D_R)   + N_DIT_4D_R**2   * sigma_syst_2_4D_R   )
    SNR_4D_l0  = N_DIT_4D_l0  * signal_4D_l0  / np.sqrt( N_DIT_4D_l0  * (sigma_halo_2_4D_l0  + sigma_bkg_2_4D_l0  + sigma_RON_2_4D_l0  + sigma_DC_2_4D_l0)  + N_DIT_4D_l0**2  * sigma_syst_2_4D_l0  )
    SNR_4D_Nl  = N_DIT_4D_Nl  * signal_4D_Nl  / np.sqrt( N_DIT_4D_Nl  * (sigma_halo_2_4D_Nl  + sigma_bkg_2_4D_Nl  + sigma_RON_2_4D_Nl  + sigma_DC_2_4D_Nl)  + N_DIT_4D_Nl**2  * sigma_syst_2_4D_Nl  )
    SNR_4D_TI  = N_DIT_4D_TI  * signal_4D_TI  / np.sqrt( N_DIT_4D_TI  * (sigma_halo_2_4D_TI  + sigma_bkg_2_4D_TI  + sigma_RON_2_4D_TI  + sigma_DC_2_4D_TI)  + N_DIT_4D_TI**2  * sigma_syst_2_4D_TI  )
    SNR_4D_SM  = N_DIT_4D_ref * signal_4D_ref / np.sqrt( N_DIT_4D_ref * (sigma_halo_2_4D_ref + sigma_bkg_2_4D_ref + sigma_RON_2_4D_ref + sigma_DC_2_4D_ref) + N_DIT_4D_ref**2 * sigma_syst_2_4D_SM  )
    SNR_4D_WFE = N_DIT_4D_WFE * signal_4D_WFE / np.sqrt( N_DIT_4D_WFE * (sigma_halo_2_4D_WFE + sigma_bkg_2_4D_WFE + sigma_RON_2_4D_WFE + sigma_DC_2_4D_WFE) + N_DIT_4D_WFE**2 * sigma_syst_2_4D_WFE )
    SNR_4D_IWA = N_DIT_4D_IWA * signal_4D_IWA / np.sqrt( N_DIT_4D_IWA * (sigma_halo_2_4D_IWA + sigma_bkg_2_4D_IWA + sigma_RON_2_4D_IWA + sigma_DC_2_4D_IWA) + N_DIT_4D_IWA**2 * sigma_syst_2_4D_IWA )
    
    return SNR_4D_R, SNR_4D_l0, SNR_4D_Nl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA



def get_Pdet(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, SNR_thr, separation_planets,
             idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Nl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
             SNR_4D_R, SNR_4D_l0, SNR_4D_Nl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA, indices=None):
    
    if indices is None:
        indices = np.arange(N_PT, dtype=int)
    else:
        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.where(indices)[0]
        indices = indices.astype(int)
    N_PT_sel = len(indices)
    if N_PT_sel == 0:
        return (
            np.zeros((N_band, N_config, N_R),   dtype=dtype),
            np.zeros((N_band, N_config, N_l0),  dtype=dtype),
            np.zeros((N_band, N_config, N_Nl),  dtype=dtype),
            np.zeros((N_band, N_config, N_TI),  dtype=dtype),
            np.zeros((N_band, N_config, N_SM),  dtype=dtype),
            np.zeros((N_band, N_config, N_FoV), dtype=dtype),
            np.zeros((N_band, N_config, N_WFE), dtype=dtype),
            np.zeros((N_band, N_config, N_IWA), dtype=dtype),
        )
    
    FoV_half = FoV/2
    
    # Detection counting : nb of detected planet above the threshold for each point
    N_detections_3D_R   = np.zeros((N_band, N_config, N_R),   dtype=np.uint32)
    N_detections_3D_l0  = np.zeros((N_band, N_config, N_l0),  dtype=np.uint32)
    N_detections_3D_Nl  = np.zeros((N_band, N_config, N_Nl),  dtype=np.uint32)
    N_detections_3D_TI  = np.zeros((N_band, N_config, N_TI),  dtype=np.uint32)
    N_detections_3D_SM  = np.zeros((N_band, N_config, N_SM),  dtype=np.uint32)
    N_detections_3D_FoV = np.zeros((N_band, N_config, N_FoV), dtype=np.uint32)
    N_detections_3D_WFE = np.zeros((N_band, N_config, N_WFE), dtype=np.uint32)
    N_detections_3D_IWA = np.zeros((N_band, N_config, N_IWA), dtype=np.uint32)

    for idx in tqdm(indices, total=N_PT_sel, desc="SNR planets"):

        # --- SNR per planet ---
        SNR_3D_R   = SNR_4D_R[idx]
        SNR_3D_l0  = SNR_4D_l0[idx]
        SNR_3D_Nl  = SNR_4D_Nl[idx]
        SNR_3D_TI  = SNR_4D_TI[idx]
        SNR_3D_SM  = SNR_4D_SM[idx]
        SNR_3D_WFE = SNR_4D_WFE[idx]
        SNR_3D_IWA = SNR_4D_IWA[idx]
        
        # # For impossible off-axis      
        # if separation_planets[idx] <= FoV_ref/2:

        # --- Counting detections, whatever the ref FoV value (assuming off-axis is possible) ---     
        N_detections_3D_R   += (np.nan_to_num(SNR_3D_R,   nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_l0  += (np.nan_to_num(SNR_3D_l0,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_Nl  += (np.nan_to_num(SNR_3D_Nl,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_TI  += (np.nan_to_num(SNR_3D_TI,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_SM  += (np.nan_to_num(SNR_3D_SM,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_WFE += (np.nan_to_num(SNR_3D_WFE, nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_IWA += (np.nan_to_num(SNR_3D_IWA, nan=-np.inf) >= SNR_thr).astype(np.uint32)

        # --- FoV axis: update only if the planet can enter the FoV for some values ---
        p0 = np.searchsorted(FoV_half, separation_planets[idx], side="left")  # first FoV that includes the planet
        if p0 < N_FoV:
            # Reference SNR for FoV axis: choose your "reference point"
            # Here: R_ref per band
            SNR_2D_ref = SNR_3D_R[np.arange(N_band), :, idx_R_ref]                             # (bands, configs)
            det_ref    = (np.nan_to_num(SNR_2D_ref, nan=-np.inf) >= SNR_thr).astype(np.uint32) # (bands, configs)
    
            # The planet is inside FoV for all f >= p0
            N_detections_3D_FoV[:, :, p0:] += det_ref[:, :, None]
            
    # --- Computing detection probabilities --- 
    Pdet_3D_R   = N_detections_3D_R.astype(dtype)   / N_PT_sel # [0..1]
    Pdet_3D_l0  = N_detections_3D_l0.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_Nl  = N_detections_3D_Nl.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_TI  = N_detections_3D_TI.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_SM  = N_detections_3D_SM.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_FoV = N_detections_3D_FoV.astype(dtype) / N_PT_sel # [0..1]
    Pdet_3D_WFE = N_detections_3D_WFE.astype(dtype) / N_PT_sel # [0..1]
    Pdet_3D_IWA = N_detections_3D_IWA.astype(dtype) / N_PT_sel # [0..1]
    
    return Pdet_3D_R, Pdet_3D_l0, Pdet_3D_Nl, Pdet_3D_TI, Pdet_3D_SM, Pdet_3D_FoV, Pdet_3D_WFE, Pdet_3D_IWA



def get_exposure_time(min_DIT, DC0, RON0, RON_lim, A_FWHM, NbPixel, SNR_thr, idx_R_ref, sigma_m, sigma_m_ref,
                      signal_4D_R,   sigma_halo_2_4D_R,   sigma_syst_base_2_4D_R,   sigma_bkg_2_4D_R,   DIT_4D_R,
                      signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_syst_base_2_4D_l0,  sigma_bkg_2_4D_l0,  DIT_4D_l0,
                      signal_4D_Nl,  sigma_halo_2_4D_Nl,  sigma_syst_base_2_4D_Nl,  sigma_bkg_2_4D_Nl,  DIT_4D_Nl,
                      signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_syst_base_2_4D_TI,  sigma_bkg_2_4D_TI,  DIT_4D_TI,
                      signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_syst_base_2_4D_WFE, sigma_bkg_2_4D_WFE, DIT_4D_WFE,
                      signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_syst_base_2_4D_IWA, sigma_bkg_2_4D_IWA, DIT_4D_IWA):

    # Data shape = (planets, bands, configs, *)
    N_PT, N_band, N_config, _ = signal_4D_R.shape

    # --- Detector variances per DIT ---
    def detector_variances(DIT):
        sigma_RON_2 = sigma_RON_2_from_DIT(DIT, min_DIT, RON0, RON_lim, A_FWHM, NbPixel=NbPixel, dtype=dtype)
        sigma_DC_2  = (DC0 * DIT * A_FWHM).astype(dtype)
        if NbPixel is not None:
            sigma_DC_2 *= NbPixel
        return sigma_RON_2, sigma_DC_2

    sigma_RON_2_4D_R,   sigma_DC_2_4D_R   = detector_variances(DIT_4D_R)
    sigma_RON_2_4D_l0,  sigma_DC_2_4D_l0  = detector_variances(DIT_4D_l0)
    sigma_RON_2_4D_Nl,  sigma_DC_2_4D_Nl  = detector_variances(DIT_4D_Nl)
    sigma_RON_2_4D_TI,  sigma_DC_2_4D_TI  = detector_variances(DIT_4D_TI)
    sigma_RON_2_4D_WFE, sigma_DC_2_4D_WFE = detector_variances(DIT_4D_WFE)
    sigma_RON_2_4D_IWA, sigma_DC_2_4D_IWA = detector_variances(DIT_4D_IWA)

    # --- Reference quantities (for sigma_m axis) ---
    idx = np.broadcast_to(idx_R_ref[None, :, None, None], (N_PT, N_band, N_config, 1))
    DIT_4D_ref             = np.take_along_axis(DIT_4D_R,             idx, axis=3)[..., 0][..., None]
    signal_4D_ref          = np.take_along_axis(signal_4D_R,          idx, axis=3)[..., 0][..., None]
    sigma_halo_2_4D_ref    = np.take_along_axis(sigma_halo_2_4D_R,    idx, axis=3)[..., 0][..., None]
    sigma_bkg_2_4D_ref     = np.take_along_axis(sigma_bkg_2_4D_R,     idx, axis=3)[..., 0][..., None]
    sigma_RON_2_4D_ref     = np.take_along_axis(sigma_RON_2_4D_R,     idx, axis=3)[..., 0][..., None]
    sigma_DC_2_4D_ref      = np.take_along_axis(sigma_DC_2_4D_R,      idx, axis=3)[..., 0][..., None]
    sigma_syst_base_2_ref  = np.take_along_axis(sigma_syst_base_2_4D_R, idx, axis=3)[..., 0][..., None]

    # --- Systematic-noise variances ---
    sigma_syst_2_4D_R   = sigma_m_ref**2                  * sigma_syst_base_2_4D_R
    sigma_syst_2_4D_l0  = sigma_m_ref**2                  * sigma_syst_base_2_4D_l0
    sigma_syst_2_4D_Nl  = sigma_m_ref**2                  * sigma_syst_base_2_4D_Nl
    sigma_syst_2_4D_TI  = sigma_m_ref**2                  * sigma_syst_base_2_4D_TI
    sigma_syst_2_4D_SM  = sigma_m[None, None, None, :]**2 * sigma_syst_base_2_ref
    sigma_syst_2_4D_WFE = sigma_m_ref**2                  * sigma_syst_base_2_4D_WFE
    sigma_syst_2_4D_IWA = sigma_m_ref**2                  * sigma_syst_base_2_4D_IWA

    # --- Required exposure time ---
    def time_to_SNR(signal, sigma_halo_2, sigma_bkg_2, sigma_RON_2, sigma_DC_2, sigma_syst_2, DIT):
        signal, sigma_halo_2, sigma_bkg_2, sigma_RON_2, sigma_DC_2, sigma_syst_2, DIT = np.broadcast_arrays(signal, sigma_halo_2, sigma_bkg_2, sigma_RON_2, sigma_DC_2, sigma_syst_2, DIT)
        signal       = signal.astype(np.float64, copy=False)
        sigma_fund_2 = (sigma_halo_2 + sigma_bkg_2 + sigma_RON_2 + sigma_DC_2).astype(np.float64, copy=False)
        sigma_syst_2 = sigma_syst_2.astype(np.float64, copy=False)
        DIT          = DIT.astype(np.float64, copy=False)
        Texp         = np.full(signal.shape, np.nan, dtype=np.float64)
        valid        = np.isfinite(signal) & np.isfinite(sigma_fund_2) & np.isfinite(sigma_syst_2) & np.isfinite(DIT) & (signal > 0) & (sigma_fund_2 >= 0) & (sigma_syst_2 >= 0) & (DIT > 0)
        denominator  = signal**2 - SNR_thr**2 * sigma_syst_2
        unreachable  = valid & (denominator <= 0)
        reachable    = valid & (denominator > 0)
        Texp[unreachable] = np.inf
        N_req_cont        = np.zeros(signal.shape, dtype=np.float64)
        N_req_cont[reachable] = SNR_thr**2 * sigma_fund_2[reachable] / denominator[reachable]
        N_req = np.maximum(1.0, np.ceil(N_req_cont))
        # Numerical safety: make sure the rounded integer number of DITs really reaches SNR_thr
        snr_req = np.full(signal.shape, np.nan, dtype=np.float64)
        snr_req[reachable] = N_req[reachable] * signal[reachable] / np.sqrt(N_req[reachable] * sigma_fund_2[reachable] + N_req[reachable]**2 * sigma_syst_2[reachable])
        N_req[reachable & (snr_req < SNR_thr)] += 1
        Texp[reachable] = N_req[reachable] * DIT[reachable]
        return Texp.astype(dtype)

    Texp_4D_R   = time_to_SNR(signal_4D_R,   sigma_halo_2_4D_R,   sigma_bkg_2_4D_R,   sigma_RON_2_4D_R,   sigma_DC_2_4D_R,   sigma_syst_2_4D_R,   DIT_4D_R)
    Texp_4D_l0  = time_to_SNR(signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_bkg_2_4D_l0,  sigma_RON_2_4D_l0,  sigma_DC_2_4D_l0,  sigma_syst_2_4D_l0,  DIT_4D_l0)
    Texp_4D_Nl  = time_to_SNR(signal_4D_Nl,  sigma_halo_2_4D_Nl,  sigma_bkg_2_4D_Nl,  sigma_RON_2_4D_Nl,  sigma_DC_2_4D_Nl,  sigma_syst_2_4D_Nl,  DIT_4D_Nl)
    Texp_4D_TI  = time_to_SNR(signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_bkg_2_4D_TI,  sigma_RON_2_4D_TI,  sigma_DC_2_4D_TI,  sigma_syst_2_4D_TI,  DIT_4D_TI)
    Texp_4D_SM  = time_to_SNR(signal_4D_ref, sigma_halo_2_4D_ref, sigma_bkg_2_4D_ref, sigma_RON_2_4D_ref, sigma_DC_2_4D_ref, sigma_syst_2_4D_SM,  DIT_4D_ref)
    Texp_4D_WFE = time_to_SNR(signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_bkg_2_4D_WFE, sigma_RON_2_4D_WFE, sigma_DC_2_4D_WFE, sigma_syst_2_4D_WFE, DIT_4D_WFE)
    Texp_4D_IWA = time_to_SNR(signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_bkg_2_4D_IWA, sigma_RON_2_4D_IWA, sigma_DC_2_4D_IWA, sigma_syst_2_4D_IWA, DIT_4D_IWA)

    return Texp_4D_R, Texp_4D_l0, Texp_4D_Nl, Texp_4D_TI, Texp_4D_SM, Texp_4D_WFE, Texp_4D_IWA




# %%
def main():
    
    # --- Instrument specs (fixed) --- 
    instru        = "HARMONI"                                    # Instrument's name
    strehl        = "JQ1" if instru == "HARMONI" else "MED"      # AO performance
    config_data   = get_config_data(instru)                      # Getting global instrument specs
    instru_type   = config_data["type"]                          # Type of instrument
    D             = config_data["telescope"]["diameter"]         # [m]
    S             = config_data["telescope"]["area"]             # [m2]
    size_core     = config_data["size_core"]                     # Aperture size on which the signal is integrated [pixels]
    A_FWHM        = size_core**2                                 # Number of pixel per FWHM box area
    saturation_e  = config_data["detector"]["saturation_e"]      # Full well capacity of the detector [e-]
    min_DIT       = config_data["detector"]["minDIT"]            # Minimal integration time [in mn]
    max_DIT       = config_data["detector"]["maxDIT"]            # Maximal integration time [mn]
    RON0          = config_data["detector"]["RON"]               # Read out noise [e-/px/DIT]
    RON_lim       = config_data["detector"]["RON_lim"]           # Read out noise limit [e-/px/DIT]
    DC0           = config_data["detector"]["dark_current"]*60   # Dark current [e-/px/mn]
    coronagraphs  = config_data["coronagraphs"]                  # Available coronagraphs
    apodizers     = config_data["apodizers"]                     # Available apodizers
    w_apodizer    = len(apodizers)    > 1                        # If the instrument has apodizer config
    w_coronagraph = len(coronagraphs) > 1                        # If the instrument has coronagraphic config
    if "fiber" in instru_type: # Detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
        NbPixel   = config_data["pixel_detector_projection"] # Effective number of pixels per spectral channel (here: first guess)
    else:
        NbPixel   = None
        
    # --- Useless specs (already taken into account implicitly, only informative)
    detector   = "H4RG" # Type of detector
    N_px       = 4096   # Detector's size
    N_mirror   = 5      # Number of ELT mirror
    trans_dust = 0.90   # Dust's transmission
    
    # --- General parameters for the simulation ---
    SNR_thr            = 1               # Detection threshold
    exposure_time      = 10*60           # Total exposure time per planet [mn]
    force_new_calc     = False           # Forcing new simulations calculations
    thermal_model      = "None"          # Model for the thermal spectrum of the planet ("BT-Settl", "Exo-REM", "SONORA", "PICASO", "Saumon", etc.)
    reflected_model    = "auto"          # Model for the albedo of the planet ("tellurics", "flat", "PICASO")
    post_processing    = "MM"            # Post-processing method ("MM" or "DI")
    Rc                 = 1_00            # MM cut-off resolution
    filter_type        = "gaussian_fast" # MM filter type
    force_inside_bands = False           # Not allowing wave axis to be outside the original bands ranges
    table_type         = "Archive"       # "Archive": for all known exoplanets | "Simulated": TODO
    
    # --- WFE and IWA ref values ---
    if instru == "HARMONI":
        WFE_ref = (174.7 + 160.5) / 2 # [nm]
    elif instru == "ANDES":
        WFE_ref = 212.2 # [nm]
    else:
        raise KeyError("to define")
    if w_coronagraph:
        if instru == "ANDES":
            IWA_ref = 33.4 / 2 # [mas]
        else:
            raise KeyError("to define")
    else:
        IWA_ref = None  # [mas] (dummy value)
    
    # --- Parameters space to explore ---
    # Size of the parameters space to explore (N**Ndim)
    N                = 20
    #N                = 5
    # Spectral resolution (for each R, it is assumed to be constant along the whole wavelength range)
    R_min            = Rc           # [dimensionless]
    if instru == "HARMONI":  
        R_max        = 100_000      # [dimensionless]
    elif instru == "ANDES":
        R_max        = 200_000      # [dimensionless]
    # Bandwidth central wavelength (for each l0, the wave axis is given by lmin,lmax = l0 +- dl*Nl/2, where Nl depends on R and l0)
    l0_min           = 0.6          # [µm] 
    l0_max           = 3.0          # [µm]
    # Number of spectral channel (number of effective bins sampling the data along the spectral dimension)
    Nl_min           = 1_00         # [bins]
    Nl_max           = 200_000      # [bins]
    # Instrumental transmission range (with telescope transmission)
    trans_instru_min = 0.001        # [e-/ph]
    trans_instru_max = 0.5          # [e-/ph]
    # Effective residual halo modulation level:
    # For IFU: sigma_m is the effective fractional modulation of the final integrated stellar halo per spectral bin.
    # In practice, sigma_syst_base_2 stores the projected halo-squared term for one DIT,
    # already including throughput, PSF, FWHM integration, and DIT scaling.
    # The total systematic variance over the full exposure is then:
    # sigma_syst_tot^2 = sigma_m^2 * N_DIT^2 * sigma_syst_base_2
    sigma_m_min      = 1e-7         # [dimensionless]
    #sigma_m_min      = 1e-4         # [dimensionless]
    sigma_m_max      = 1e-1         # [dimensionless]
    # Separation range considered 
    sep_min          = 0            # [mas]
    sep_max          = 1_000        # [mas] max separation of the raw PSF data
    # Field of View
    FoV_min          = 10           # [mas]
    FoV_max          = 2*sep_max    # [mas]
    # Post-AO wavefront error
    WFE_min          = 10           # [nm]
    WFE_max          = 500          # [nm]
    # Coronagraph inner working angle (size of the FPM)
    if w_coronagraph:
        IWA_min      = 0.1          # [mas]
        IWA_max      = 100          # [mas]
    else:
        IWA_min      = None         # [mas]
        IWA_max      = None         # [mas]
    # Axis
    R            = np.logspace(np.log10(R_min),       np.log10(R_max),         N) # i dim
    l0           = np.linspace(l0_min,                l0_max,                2*N) # j dim
    Nl           = np.logspace(np.log10(Nl_min),      np.log10(Nl_max),      2*N) # k dim
    trans_instru = np.linspace(trans_instru_min,      trans_instru_max,      2*N) # l dim
    sigma_m      = np.logspace(np.log10(sigma_m_min), np.log10(sigma_m_max), 2*N) # m dim
    FoV          = np.logspace(np.log10(FoV_min),     np.log10(FoV_max),     2*N) # n dim
    WFE          = np.linspace(WFE_min,               WFE_max,               2*N) # o dim
    if w_coronagraph:
        IWA      = np.logspace(np.log10(IWA_min),     np.log10(IWA_max),     2*N) # p dim
    else:
        IWA      = np.array([None])                                               # p dim

    # Nl must have int values
    Nl           = np.round(Nl).astype(int)
    # Axis must have unique and sorted values
    R            = np.unique(R)
    l0           = np.unique(l0)
    Nl           = np.unique(Nl)
    trans_instru = np.unique(trans_instru)
    sigma_m      = np.unique(sigma_m)
    FoV          = np.unique(FoV)
    WFE          = np.unique(WFE)
    if w_coronagraph:
        IWA      = np.unique(IWA)
    
    # --- Bands and config --- 
    bands = [band for band in config_data["gratings"]] # Bands
    if instru == "HARMONI":
        configs = ["NO_SP", "SP_Prox"] # Configs
    elif instru == "ANDES":
        configs = [None, "LYOT"]
    N_band   = len(bands)
    N_config = len(configs)
    
    # --- Paths ---
    try:
        ROOT = Path(__file__).resolve().parent
    except NameError:
        ROOT = Path.cwd()
    instru_dir = ROOT / "data" / f"{instru}"
    psf_dir    = instru_dir / "PSF_simulations"
    sim_dir    = instru_dir / "FastYield_simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Cmap and linestyles ---
    if instru == "HARMONI":
        cmap_bands = plt.get_cmap("rainbow", N_band)
    elif instru == "ANDES":
        cmap_bands = plt.get_cmap("viridis", N_band+1)
    ls_configs = ["-", "--", ":"]
    
    # --- Adding reference values in arrays---
    pxscales         = np.zeros((N_band))
    R_ref            = np.zeros((N_band))
    lmin_bands       = np.zeros((N_band))
    lmax_bands       = np.zeros((N_band))
    l0_ref           = np.zeros((N_band))
    Nl_ref           = np.zeros((N_band))
    trans_instru_ref = np.zeros((N_band, N_config))
    for ib, band in enumerate(bands):
        lmin_bands[ib] = config_data['gratings'][band].lmin                 # Lambda min of the considered band [µm]
        lmax_bands[ib] = config_data['gratings'][band].lmax                 # Lambda max of the considered band [µm]
        R_ref[ib]      = config_data['gratings'][band].R                    # Spectral resolution of the band
        wave_band      = get_wave_band(instru=instru, band=band)            # Wavelength axis on this band [µm]
        l0_ref[ib]     = np.exp(np.nanmean(np.log(wave_band)))              # Lambda 0 of the considered band [µm]
        Nl_ref[ib]     = len(wave_band)                                     # Number of spectral channels
        if R_ref[ib] >= R_min and R_ref[ib] <= R_max:
            R = add_if_necessary(array=R,  value=R_ref[ib])
        if l0_ref[ib] >= l0_min and l0_ref[ib] <= l0_max:
            l0 = add_if_necessary(array=l0, value=l0_ref[ib])
        for ic, config in enumerate(configs):
            trans_instru_ref[ib, ic] = np.nanmedian(get_transmission(instru=instru, wave_band=wave_band, band=band, tellurics=False, apodizer=config if w_apodizer else "NO_SP", strehl=strehl, coronagraph=config if w_coronagraph else None, fill_value=np.nan))
            if trans_instru_ref[ib, ic] >= trans_instru_min and trans_instru_ref[ib, ic] <= trans_instru_max:
                trans_instru = add_if_necessary(array=trans_instru, value=trans_instru_ref[ib, ic])
        # Pxscale for this band
        try:
            pxscales[ib] = config_data["pxscale"][band]*1000 # [mas/px] (ANDES: different pxscale)
        except Exception:
            pxscales[ib] = config_data["pxscale"]*1000       # [mas/px] (HARMONI: fixed pxscale)
    # Nl must be common for each band
    Nl_ref = int(np.round(np.nanmean(Nl_ref)))
    if Nl_ref >= Nl_min and Nl_ref <= Nl_max:
        Nl = add_if_necessary(array=Nl, value=Nl_ref)
    # Systematic ref value
    sigma_m_ref = sigma_m_min # TODO: decide which value to choose
    # FoV ref value
    if instru == "HARMONI":
        FoV_ref = config_data["FOV"]*1000 # [mas]
    elif instru == "ANDES":
        FoV_ref = 9*pxscales # [mas] 9 fibers across the FoV
    # WFE ref values
    if WFE_ref >= WFE_min and WFE_ref <= WFE_max:
        WFE = add_if_necessary(array=WFE, value=WFE_ref)
    # IWA ref values
    if w_coronagraph and IWA_ref >= IWA_min and IWA_ref <= IWA_max:
        IWA = add_if_necessary(array=IWA, value=IWA_ref)    
    # --- Reference index for each band ---
    idx_R_ref  = np.zeros((N_band)).astype(int)
    idx_l0_ref = np.zeros((N_band)).astype(int)
    idx_TI_ref = np.zeros((N_band, N_config)).astype(int)
    for ib, band in enumerate(bands):
        idx_R_ref[ib]  = np.abs(R_ref[ib]  - R).argmin()
        idx_l0_ref[ib] = np.abs(l0_ref[ib] - l0).argmin()
        for ic, config in enumerate(configs):
            idx_TI_ref[ib, ic] = np.abs(trans_instru_ref[ib, ic] - trans_instru).argmin()
    # Nl ref idx
    idx_Nl_ref = np.abs(Nl_ref - Nl).argmin()
    # WFE ref idx
    idx_WFE_ref = np.abs(WFE - WFE_ref).argmin()
    # IWA ref idx
    if w_coronagraph:
        idx_IWA_ref = np.abs(IWA - IWA_ref).argmin()
    else:
        idx_IWA_ref = 0

    # --- Real dimension sizes ---
    N_R   = len(R)
    N_l0  = len(l0)
    N_Nl  = len(Nl)
    N_TI  = len(trans_instru)
    N_SM  = len(sigma_m)
    N_FoV = len(FoV)
    N_WFE = len(WFE)
    if w_coronagraph:
        N_IWA = len(IWA)
    else:
        N_IWA = 1
    
    # --- Resolution for the raw models ---
    R_model = R_max
    
    # Residuals modulations considered for the post_processing
    residuals = "Systematics" if post_processing == "MM" else "Speckles"
    
    
    
    # ---------------------------
    # Getting planet table
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
    print_simulation_summary(instru=instru, instru_type=instru_type, detector=detector, post_processing=post_processing, D=D, S=S, N_mirror=N_mirror, trans_dust=trans_dust, RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, N_px=N_px, coronagraph=[str(coronagraph) for coronagraph in coronagraphs], apodizer=[str(apodizer) for apodizer in apodizers], strehl=strehl, thermal_model=thermal_model, reflected_model=reflected_model, SNR_thr=SNR_thr, exposure_time=exposure_time, size_core=size_core, A_FWHM=A_FWHM, Rc=Rc if "IFU" in instru_type else None, filter_type=filter_type if "IFU" in instru_type else None, table_type=table_type, sep_min=sep_min, sep_max=sep_max, N_PT_raw=N_PT_raw, N_PT=N_PT, force_new_calc=force_new_calc, l0=l0, R=R if "IFU" in instru_type else None, Nl=Nl if "IFU" in instru_type else None, Dl=None, WFE=WFE, IWA=IWA, trans_instru=trans_instru, sigma_m=sigma_m, FoV=FoV)
    
    # Build meta for YOUR case (IFU or imager)
    meta = dict(
    
        # instrument / telescope
        instru=instru, instru_type=instru_type, D=D, S=S, strehl=strehl, bands=bands, configs=configs, w_coronagraph=w_coronagraph,
        
        # detector numeric specs that affect DIT/noise/products
        RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, size_core=size_core,
        
        # General parameters for the simulation
        thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, R_model=R_model, Rc=Rc, filter_type=filter_type, force_inside_bands=force_inside_bands, table_type=table_type, N_PT=int(N_PT),
    
        # exploration space (communs)
        l0_min=l0_min, l0_max=l0_max, trans_instru_min=trans_instru_min, trans_instru_max=trans_instru_max, WFE_min=WFE_min, WFE_max=WFE_max, IWA_min=IWA_min, IWA_max=IWA_max, sep_min=sep_min, sep_max=sep_max,
        
        # axis used
        R=R, l0=l0, Nl=Nl, trans_instru=trans_instru, WFE=WFE, IWA=IWA,
    )
    
    suffix, meta_clean, _payload = make_suffix(meta, n=16)    
    write_meta(sim_dir, suffix, meta_clean)
    print("\nFastYield data files suffix:", suffix)
    
    # File paths derived from suffix
    signal_4D_R_path              = sim_dir / f"signal_4D_R_{suffix}.npy"
    signal_4D_l0_path             = sim_dir / f"signal_4D_l0_{suffix}.npy"
    signal_4D_Nl_path             = sim_dir / f"signal_4D_Nl_{suffix}.npy"
    signal_4D_TI_path             = sim_dir / f"signal_4D_TI_{suffix}.npy"
    signal_4D_WFE_path            = sim_dir / f"signal_4D_WFE_{suffix}.npy"
    signal_4D_IWA_path            = sim_dir / f"signal_4D_IWA_{suffix}.npy"
    sigma_halo_2_4D_R_path        = sim_dir / f"sigma_halo_2_4D_R_{suffix}.npy"
    sigma_halo_2_4D_l0_path       = sim_dir / f"sigma_halo_2_4D_l0_{suffix}.npy"
    sigma_halo_2_4D_Nl_path       = sim_dir / f"sigma_halo_2_4D_Nl_{suffix}.npy"
    sigma_halo_2_4D_TI_path       = sim_dir / f"sigma_halo_2_4D_TI_{suffix}.npy"
    sigma_halo_2_4D_WFE_path      = sim_dir / f"sigma_halo_2_4D_WFE_{suffix}.npy"
    sigma_halo_2_4D_IWA_path      = sim_dir / f"sigma_halo_2_4D_IWA_{suffix}.npy"
    sigma_syst_base_2_4D_R_path   = sim_dir / f"sigma_syst_base_2_4D_R_{suffix}.npy"
    sigma_syst_base_2_4D_l0_path  = sim_dir / f"sigma_syst_base_2_4D_l0_{suffix}.npy"
    sigma_syst_base_2_4D_Nl_path  = sim_dir / f"sigma_syst_base_2_4D_Nl_{suffix}.npy"
    sigma_syst_base_2_4D_TI_path  = sim_dir / f"sigma_syst_base_2_4D_TI_{suffix}.npy"
    sigma_syst_base_2_4D_WFE_path = sim_dir / f"sigma_syst_base_2_4D_WFE_{suffix}.npy"
    sigma_syst_base_2_4D_IWA_path = sim_dir / f"sigma_syst_base_2_4D_IWA_{suffix}.npy"
    sigma_bkg_2_4D_R_path         = sim_dir / f"sigma_bkg_2_4D_R_{suffix}.npy"
    sigma_bkg_2_4D_l0_path        = sim_dir / f"sigma_bkg_2_4D_l0_{suffix}.npy"
    sigma_bkg_2_4D_Nl_path        = sim_dir / f"sigma_bkg_2_4D_Nl_{suffix}.npy"
    sigma_bkg_2_4D_TI_path        = sim_dir / f"sigma_bkg_2_4D_TI_{suffix}.npy"
    sigma_bkg_2_4D_WFE_path       = sim_dir / f"sigma_bkg_2_4D_WFE_{suffix}.npy"
    sigma_bkg_2_4D_IWA_path       = sim_dir / f"sigma_bkg_2_4D_IWA_{suffix}.npy"
    DIT_4D_R_path                 = sim_dir / f"DIT_4D_R_{suffix}.npy"
    DIT_4D_l0_path                = sim_dir / f"DIT_4D_l0_{suffix}.npy"
    DIT_4D_WFE_path               = sim_dir / f"DIT_4D_WFE_{suffix}.npy"
    DIT_4D_IWA_path               = sim_dir / f"DIT_4D_IWA_{suffix}.npy"
    DIT_4D_Nl_path                = sim_dir / f"DIT_4D_Nl_{suffix}.npy"
    DIT_4D_TI_path                = sim_dir / f"DIT_4D_TI_{suffix}.npy"
    PSF_profile_6D_path           = sim_dir / f"tmp_PSF_profile_6D_{suffix}.npy"
    fraction_core_6D_path         = sim_dir / f"tmp_fraction_core_6D_{suffix}.npy"
    
    required_paths = [
                        signal_4D_R_path, signal_4D_l0_path, signal_4D_Nl_path, signal_4D_TI_path, signal_4D_WFE_path, signal_4D_IWA_path,
                        sigma_halo_2_4D_R_path, sigma_halo_2_4D_l0_path, sigma_halo_2_4D_Nl_path, sigma_halo_2_4D_TI_path, sigma_halo_2_4D_WFE_path, sigma_halo_2_4D_IWA_path,
                        sigma_syst_base_2_4D_R_path, sigma_syst_base_2_4D_l0_path, sigma_syst_base_2_4D_Nl_path, sigma_syst_base_2_4D_TI_path, sigma_syst_base_2_4D_WFE_path, sigma_syst_base_2_4D_IWA_path,
                        sigma_bkg_2_4D_R_path, sigma_bkg_2_4D_l0_path, sigma_bkg_2_4D_Nl_path, sigma_bkg_2_4D_TI_path, sigma_bkg_2_4D_WFE_path, sigma_bkg_2_4D_IWA_path,
                        DIT_4D_R_path, DIT_4D_l0_path, DIT_4D_Nl_path, DIT_4D_TI_path, DIT_4D_WFE_path, DIT_4D_IWA_path,
                    ]
    do_new = force_new_calc or (not all(p.exists() for p in required_paths))
        
    
    
    ####################
    # Opening existing #
    ####################
    if not do_new:
        signal_4D_R   = np.load(signal_4D_R_path,  mmap_mode="r")
        signal_4D_l0  = np.load(signal_4D_l0_path, mmap_mode="r")
        signal_4D_Nl  = np.load(signal_4D_Nl_path, mmap_mode="r")
        signal_4D_TI  = np.load(signal_4D_TI_path, mmap_mode="r")
        signal_4D_WFE = np.load(signal_4D_WFE_path, mmap_mode="r")
        signal_4D_IWA = np.load(signal_4D_IWA_path, mmap_mode="r")
        
        sigma_halo_2_4D_R   = np.load(sigma_halo_2_4D_R_path,  mmap_mode="r")
        sigma_halo_2_4D_l0  = np.load(sigma_halo_2_4D_l0_path, mmap_mode="r")
        sigma_halo_2_4D_Nl  = np.load(sigma_halo_2_4D_Nl_path, mmap_mode="r")
        sigma_halo_2_4D_TI  = np.load(sigma_halo_2_4D_TI_path, mmap_mode="r")
        sigma_halo_2_4D_WFE = np.load(sigma_halo_2_4D_WFE_path, mmap_mode="r")
        sigma_halo_2_4D_IWA = np.load(sigma_halo_2_4D_IWA_path, mmap_mode="r")

        sigma_syst_base_2_4D_R   = np.load(sigma_syst_base_2_4D_R_path,  mmap_mode="r")
        sigma_syst_base_2_4D_l0  = np.load(sigma_syst_base_2_4D_l0_path, mmap_mode="r")
        sigma_syst_base_2_4D_Nl  = np.load(sigma_syst_base_2_4D_Nl_path, mmap_mode="r")
        sigma_syst_base_2_4D_TI  = np.load(sigma_syst_base_2_4D_TI_path, mmap_mode="r")
        sigma_syst_base_2_4D_WFE = np.load(sigma_syst_base_2_4D_WFE_path, mmap_mode="r")
        sigma_syst_base_2_4D_IWA = np.load(sigma_syst_base_2_4D_IWA_path, mmap_mode="r")

        
        sigma_bkg_2_4D_R   = np.load(sigma_bkg_2_4D_R_path,  mmap_mode="r")
        sigma_bkg_2_4D_l0  = np.load(sigma_bkg_2_4D_l0_path, mmap_mode="r")
        sigma_bkg_2_4D_Nl  = np.load(sigma_bkg_2_4D_Nl_path, mmap_mode="r")
        sigma_bkg_2_4D_TI  = np.load(sigma_bkg_2_4D_TI_path, mmap_mode="r")
        sigma_bkg_2_4D_WFE = np.load(sigma_bkg_2_4D_WFE_path, mmap_mode="r")
        sigma_bkg_2_4D_IWA = np.load(sigma_bkg_2_4D_IWA_path, mmap_mode="r")

        DIT_4D_R   = np.load(DIT_4D_R_path,  mmap_mode="r")
        DIT_4D_l0  = np.load(DIT_4D_l0_path, mmap_mode="r")
        DIT_4D_Nl  = np.load(DIT_4D_Nl_path, mmap_mode="r")
        DIT_4D_TI  = np.load(DIT_4D_TI_path, mmap_mode="r")
        DIT_4D_WFE = np.load(DIT_4D_WFE_path, mmap_mode="r")
        DIT_4D_IWA = np.load(DIT_4D_IWA_path, mmap_mode="r")
        
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
        # PSF data (PSF profile, fraction core and coronagraph transmission) preparations (band, config, l0, WFE, IWA, sep)
        #
        
        separation = None
        
        for ib, band in enumerate(bands):
            
            for ic, config in enumerate(configs):
                
                # Retrieving PSF data and axis
                
                # PSF data ranges (HARD CODED)
                l0_min_PSF        = 0.6   # [µm]
                l0_max_PSF        = 3.0   # [µm]
                sep_max_PSF       = 1_000 # [mas]
                WFE_min_PSF       = 10    # [nm]
                WFE_max_PSF       = 500   # [nm]
                if w_coronagraph:
                    IWA_min_PSF = 0.1     # [mas]
                    IWA_max_PSF = 100     # [mas]
                else:
                    IWA_min_PSF = 1.0     # [mas] (dummy value)
                    IWA_max_PSF = 1.0     # [mas] (dummy value)
                    
                config_str = f"{config}_None" if w_apodizer else f"NO_SP_{config}"
                suffix_PSF = f"{config_str}_{strehl}_{pxscales[ib]}_{l0_min_PSF}_{l0_max_PSF}_{WFE_min_PSF}_{WFE_max_PSF}_{IWA_min_PSF}_{IWA_max_PSF}_{sep_max_PSF}"
                wave0                   = fits.getdata(psf_dir / f"{instru}_wave_{suffix_PSF}.fits")                   # (wave0)       [µm]
                WFE0                    = fits.getdata(psf_dir / f"{instru}_WFE_{suffix_PSF}.fits")                    # (WFE0)        [nm]
                IWA0                    = fits.getdata(psf_dir / f"{instru}_IWA_{suffix_PSF}.fits")                    # (IWA0)        [mas]
                separation0             = fits.getdata(psf_dir / f"{instru}_separation_{suffix_PSF}.fits")             # (separation0) [mas]
                PSF_profile_density0_4D = fits.getdata(psf_dir / f"{instru}_PSF_profile_density_4D_{suffix_PSF}.fits") # (wave0, WFE0, IWA0, separation0) [star flux fraction/px/mas**2]
                fraction_core0_4D       = fits.getdata(psf_dir / f"{instru}_fraction_core_4D_{suffix_PSF}.fits")       # (wave0, WFE0, IWA0, separation0) [planet flux fraction/FWHM]
                if w_coronagraph and config is not None: # with coronagraph
                    radial_transmission0_4D = fits.getdata(psf_dir / f"{instru}_radial_transmission_4D_{suffix_PSF}.fits")   # (wave0, WFE0, IWA0, separation0) coronagraphic transmission
                else: # No coronagraph
                    radial_transmission0_4D = None
                    if not w_coronagraph:
                        IWA_ref = IWA0[0]
                        IWA     = IWA0
                        if IWA.shape != (1,):
                            raise KeyError("This should not happen!")
                    
                # Initializing arrays
                if separation is None:
                    separation       = separation0
                    N_sep            = len(separation)
                    PSF_profile_6D   = open_memmap(PSF_profile_6D_path,   mode="w+", dtype=dtype, shape=(N_band, N_config, N_l0, N_WFE, N_IWA, N_sep))
                    fraction_core_6D = open_memmap(fraction_core_6D_path, mode="w+", dtype=dtype, shape=(N_band, N_config, N_l0, N_WFE, N_IWA, N_sep),)
                elif not np.all(separation == separation0):
                    raise KeyError("This should not happen!")
                                
                # Interpolating over current grid
                pts                    = np.stack((np.meshgrid(l0, WFE, IWA, separation, indexing="ij")), axis=-1)
                PSF_profile_density_4D = np.exp(RegularGridInterpolator((wave0, WFE0, IWA0, separation0), np.log(PSF_profile_density0_4D), bounds_error=True, fill_value=np.nan)(pts)) # (l0, WFE, IWA, sep) [star flux fraction/px/mas**2]
                fraction_core_4D       = expit(RegularGridInterpolator((wave0,  WFE0, IWA0, separation0), logit(fraction_core0_4D),        bounds_error=True, fill_value=np.nan)(pts)) # (l0, WFE, IWA, sep) [planet flux fraction/FWHM]
                if w_coronagraph and config is not None: # with coronagraph
                    radial_transmission_4D = expit(RegularGridInterpolator((wave0, WFE0, IWA0, separation0), logit(radial_transmission0_4D), bounds_error=True, fill_value=np.nan)(pts)) # (l0, WFE, IWA, sep) coronagraphic transmission
                    
                # Converting flux density in fraction of flux inside the FOV per spaxel for a bin at l0, WFE, IWA, separation as function of separation
                PSF_profile_4D = PSF_profile_density_4D * pxscales[ib]**2 # [star flux fraction/px]
                                
                if 1==1:
                    ##############
                    # Sanity check
                    if instru == "ANDES":
                        PSF_profile_EX      = pxscales[ib]**2 * np.nanmean(RegularGridInterpolator((wave0, WFE0, IWA0, separation0), PSF_profile_density0_4D, bounds_error=True, fill_value=np.nan)(np.stack((np.meshgrid(l0[(l0 > 0.95) &  (l0 < 1.8)], WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1))[:, 0, 0, :], axis=0)
                    else:
                        PSF_profile_EX      = pxscales[ib]**2 * RegularGridInterpolator((wave0, WFE0, IWA0, separation0), PSF_profile_density0_4D, bounds_error=True, fill_value=np.nan)(np.stack((np.meshgrid(l0_ref[ib], WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1))[0, 0, 0, :]
                    PSF_profile_OG, _, _, _ = get_PSF_profile(band=band, strehl=strehl, apodizer=config if instru=="HARMONI" else "NO_SP", coronagraph=None if instru=="HARMONI" else config, instru=instru, separation=separation)
                    plt.figure(dpi=300, figsize=(10, 6))
                    plt.title(f"{instru} | {band} | {config}")
                    plt.plot(separation, PSF_profile_OG, c="crimson",   label="Original")
                    plt.plot(separation, PSF_profile_EX, c="steelblue", label="Extrapolated")
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel("Separation [mas]", fontsize=14)
                    plt.ylabel("PSF profile [flux fraction/px]", fontsize=14)
                    plt.xlim(separation[1], separation[-1])
                    plt.show()
                    # Sanity check
                    fraction_core_EX = RegularGridInterpolator((wave0, WFE0, IWA0, separation0), fraction_core0_4D, bounds_error=True, fill_value=np.nan)(np.stack((np.meshgrid(l0_ref[ib], WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1))[0, 0, 0, :]
                    plt.figure(dpi=300, figsize=(10, 6))
                    plt.title(f"{instru} | {band} | {config}")
                    if config == "LYOT":
                        from fastyield.get_specs import _load_corona_profile
                        from scipy.interpolate import interp1d
                        raw_sep, raw_fraction_core, raw_radial_transmission = _load_corona_profile(instru=instru, band=band, strehl=strehl, apodizer="NO_SP", coronagraph="LYOT")
                        logit_fraction_core_interp                          = interp1d(raw_sep, logit(raw_fraction_core), bounds_error=False, fill_value=(logit(raw_fraction_core[0]), logit(raw_fraction_core[-1])))
                        fraction_core_OG                                    = expit(logit_fraction_core_interp(separation))       # Fraction of flux of a PSF inside the FWHM as function of the separation
                        plt.plot(separation, fraction_core_OG, c="crimson",   label="Original")
                    else:
                        _, fraction_core_OG, _, _ = get_PSF_profile(band=band, strehl=strehl, apodizer=config if instru=="HARMONI" else "NO_SP", coronagraph=None if instru=="HARMONI" else config, instru=instru, separation=separation)
                        plt.axhline(fraction_core_OG, c="crimson", label="Original")
                    plt.plot(separation, fraction_core_EX, c="steelblue", label="Extrapolated")
                    plt.legend()
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel("Separation [mas]", fontsize=14)
                    plt.ylabel("Core fraction [flux fraction/FWHM]", fontsize=14)
                    plt.ylim(1e-3, 1)
                    plt.xlim(separation[1], separation[-1])
                    plt.show()
                    if config == "LYOT":
                        radial_transmission_EX = RegularGridInterpolator((wave0, WFE0, IWA0, separation0), radial_transmission0_4D, bounds_error=True, fill_value=np.nan)(np.stack((np.meshgrid(l0_ref[ib], WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1))[0, 0, 0, :]
                        plt.figure(dpi=300, figsize=(10, 6))
                        plt.title(f"{instru} | {band} | {config}")
                        if config == "LYOT":
                            logit_radial_transmission_interp = interp1d(raw_sep, logit(raw_radial_transmission), bounds_error=False, fill_value=(logit(raw_radial_transmission[0]), logit(raw_radial_transmission[-1])))
                            radial_transmission_OG           = expit(logit_radial_transmission_interp(separation))       # Fraction of flux of a PSF inside the FWHM as function of the separation
                            plt.plot(separation, radial_transmission_OG, c="crimson",   label="Original")
                        else:
                            _, radial_transmission_OG, _, _ = get_PSF_profile(band=band, strehl=strehl, apodizer=config if instru=="HARMONI" else "NO_SP", coronagraph=None if instru=="HARMONI" else config, instru=instru, separation=separation)
                            plt.axhline(radial_transmission_OG, c="crimson", label="Original")
                        plt.plot(separation, radial_transmission_EX, c="steelblue", label="Extrapolated")
                        plt.legend()
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.xlabel("Separation [mas]",    fontsize=14)
                        plt.ylabel("Radial transmission", fontsize=14)
                        plt.ylim(1e-3, 1)
                        plt.xlim(separation[1], separation[-1])
                        plt.show()
                    ##############
                
                # Star transmission throught coronagraph (if any), given at sep = 0, assuming that the coronagraph is perfectly align with the star
                if w_coronagraph and config is not None: # with coronagraph
                    i0 = np.argmin(np.abs(separation - 0.0))
                    if i0 != 0:
                        raise KeyError("This should not happen!")
                    star_transmission_3D = radial_transmission_4D[:, :, :, i0] # (l0, WFE, IWA)
                    # PSF profile throught the coronagraph
                    PSF_profile_4D = PSF_profile_4D * star_transmission_3D[:, :, :, None] # (l0, WFE, IWA, sep)
                    
                # Fraction core signal (for the planet, inside the FWHM) through the coronagraph (if any)
                if w_coronagraph and config is not None: # with coronagraph
                    fraction_core_4D = fraction_core_4D * radial_transmission_4D
                
                # Fiber-fed IFUs: mean injection correction (e.g. ANDES): the fact that the position of the planet is unknown
                if "fiber" in instru_type:
                    fraction_core_4D *= config_data["injection"][band]
                
                # Adding to the arrays
                PSF_profile_6D[ib, ic]   = PSF_profile_4D.astype(dtype,   copy=False)
                fraction_core_6D[ib, ic] = fraction_core_4D.astype(dtype, copy=False)
                
                # Deleting useless variables
                del wave0, WFE0, IWA0, separation0
                del PSF_profile_density0_4D, fraction_core0_4D
                del pts, PSF_profile_density_4D, fraction_core_4D, PSF_profile_4D
                if radial_transmission0_4D is not None:
                    del radial_transmission0_4D
                if w_coronagraph and config is not None:
                    del radial_transmission_4D, star_transmission_3D
                gc.collect()
        
        # Deleting useless variables
        PSF_profile_6D.flush()
        fraction_core_6D.flush()
        del PSF_profile_6D, fraction_core_6D
        gc.collect()
        
        
        
        #
        # Spectra preparations
        #
        
        # K-band for photometry (for star spectra normalizations using K-band magnitudes)
        wave_K = get_wave_K()
        
        # Vega counts on K-band [J/s/m2/µm]
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)
        counts_vega_K   = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)

        # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the broadening)
        lmin_model = 0.98*l0_min                                 # [µm] a bit larger for doppler shifts and to avoid edge effects
        lmax_model = 1.02*l0_max                                 # [µm] a bit larger for doppler shifts and to avoid edge effects
        dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
        wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
        
        # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
        wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)

        # Effective model range
        lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin 
        lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax 
        
        # Tellurics transmission spectrum (from SkyCalc)
        wave_tell, trans_tell = load_tell_trans(airmass=1.0) # at airmass = 1.0
        airmass               = 1.2                          # 1.2 to be coherent with the used background skytable
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell**airmass).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1]))

        # Background spectrum [ph/mas2/mn/µm] (from SkyCalc)
        plot_bkg_skycalc(filename=instru_dir / "skytable_medium.fits")
        data_bkg   = fits.getdata(instru_dir / "skytable_medium.fits")
        wave_bkg   = data_bkg["lam"]  * 1e-3             # [nm] => [µm]
        background = data_bkg["flux"] * 60 * S / 1000**2 # [ph/s/m2/µm/arcsec2] => [ph/mas2/mn/µm]
        background = Spectrum(wave_bkg, background).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(background[0], background[-1])) # [ph/mas2/mn/µm]
        
        # Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/µm]
        scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
        scale_spectrum *= S * 60                   # [ph/s/m2/µm] => [ph/mn/µm]
        
        # Preparing for wave axis, range, trans_tell, background for each point
        trans_tell_1D_list = [None]  * N_R # (R)
        background_1D_list = [None]  * N_R # (R) [ph/mas2/mn/bin]
        wave_1D_list       = [None]  * N_R # (R) [µm]
        dwave_1D_list      = [None]  * N_R # (R) [µm/bin]
        range_1D_list      = [None]  * N_R # (R)
        valid_2D_list      = [[None] * N_R for ib in range(N_band)] # (band, R)
        l0_2D              = l0[:, None]  # (l0, 1) [µm]
        Nl_2D              = Nl[None, :]  # (1, Nl) [px]
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
            l0_2D_min = l0_2D * np.exp(-half)
            l0_2D_max = l0_2D * np.exp(+half)
            
            # Find the corresponding index interval [idx_lo, idx_hi) in the wave_res grid.
            # Using side="left"/"right" ensures we include the full requested wavelength span.
            idx_lo           = np.searchsorted(wave_res, l0_2D_min, side="left").astype(np.int32)
            idx_hi           = np.searchsorted(wave_res, l0_2D_max, side="right").astype(np.int32)
            idx_lo           = np.clip(idx_lo, 0, len(wave_res)-1)
            idx_hi           = np.clip(idx_hi, 1, len(wave_res))
            idx_hi           = np.maximum(idx_hi, idx_lo + 1)
            range_1D_list[i] = (idx_lo, idx_hi)
            
            # Valid ranges (if force_inside_bands)
            if force_inside_bands:
                for ib in range(N_band):
                    valid_2D_list[ib][i] = (wave_1D_list[i] >= lmin_bands[ib]) & (wave_1D_list[i] <= lmax_bands[ib])
        
            # Tellurics spectrum on this grid (one call per res)
            trans_tell_1D_list[i] = trans_tell.degrade_resolution(wave_res, renorm=False, R_output=res).flux
            
            # Background spectrum on this grid [ph/mn/bin/px]
            background_res        = background.degrade_resolution(wave_res, renorm=False, R_output=res).flux
            background_res       *= dwave_res      # (len(wave_res)) [ph/mas2/mn/µm]  => [ph/mas2/mn/bin]
            background_1D_list[i] = background_res # (len(wave_res)) [ph/mas2/mn/bin]
        
        # Deleting useless variables
        del vega_spectrum, vega_spectrum_K
        del trans_tell, background
        del data_bkg, wave_bkg
        gc.collect()
        
        
        
        # --- Global context worker ---
        _CTX = dict(planet_table=planet_table, separation=separation, pxscales=pxscales, PSF_profile_6D_path=str(PSF_profile_6D_path), fraction_core_6D_path=str(fraction_core_6D_path), idx_R_ref=idx_R_ref, idx_l0_ref=idx_l0_ref, idx_Nl_ref=idx_Nl_ref, idx_WFE_ref=idx_WFE_ref, idx_IWA_ref=idx_IWA_ref, R=R, trans_instru=trans_instru, wave_1D_list=wave_1D_list, dwave_1D_list=dwave_1D_list, range_1D_list=range_1D_list, valid_2D_list=valid_2D_list, trans_tell_1D_list=trans_tell_1D_list, background_1D_list=background_1D_list, wave_model=wave_model, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, scale_spectrum=scale_spectrum, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, Rc=Rc, filter_type=filter_type, D=D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, A_FWHM=A_FWHM, trans_instru_ref=trans_instru_ref, N_band=N_band, N_config=N_config, N_R=N_R, N_l0=N_l0, N_Nl=N_Nl, signal_4D_R_path=str(signal_4D_R_path), signal_4D_l0_path=str(signal_4D_l0_path), signal_4D_Nl_path=str(signal_4D_Nl_path), signal_4D_TI_path=str(signal_4D_TI_path), signal_4D_WFE_path=str(signal_4D_WFE_path), signal_4D_IWA_path=str(signal_4D_IWA_path), sigma_halo_2_4D_R_path=str(sigma_halo_2_4D_R_path), sigma_halo_2_4D_l0_path=str(sigma_halo_2_4D_l0_path), sigma_halo_2_4D_Nl_path=str(sigma_halo_2_4D_Nl_path), sigma_halo_2_4D_TI_path=str(sigma_halo_2_4D_TI_path), sigma_halo_2_4D_WFE_path=str(sigma_halo_2_4D_WFE_path), sigma_halo_2_4D_IWA_path=str(sigma_halo_2_4D_IWA_path), sigma_syst_base_2_4D_R_path=str(sigma_syst_base_2_4D_R_path), sigma_syst_base_2_4D_l0_path=str(sigma_syst_base_2_4D_l0_path), sigma_syst_base_2_4D_Nl_path=str(sigma_syst_base_2_4D_Nl_path), sigma_syst_base_2_4D_TI_path=str(sigma_syst_base_2_4D_TI_path), sigma_syst_base_2_4D_WFE_path=str(sigma_syst_base_2_4D_WFE_path), sigma_syst_base_2_4D_IWA_path=str(sigma_syst_base_2_4D_IWA_path), sigma_bkg_2_4D_R_path=str(sigma_bkg_2_4D_R_path), sigma_bkg_2_4D_l0_path=str(sigma_bkg_2_4D_l0_path), sigma_bkg_2_4D_Nl_path=str(sigma_bkg_2_4D_Nl_path), sigma_bkg_2_4D_TI_path=str(sigma_bkg_2_4D_TI_path), sigma_bkg_2_4D_WFE_path=str(sigma_bkg_2_4D_WFE_path), sigma_bkg_2_4D_IWA_path=str(sigma_bkg_2_4D_IWA_path), DIT_4D_R_path=str(DIT_4D_R_path), DIT_4D_l0_path=str(DIT_4D_l0_path), DIT_4D_Nl_path=str(DIT_4D_Nl_path), DIT_4D_TI_path=str(DIT_4D_TI_path), DIT_4D_WFE_path=str(DIT_4D_WFE_path), DIT_4D_IWA_path=str(DIT_4D_IWA_path))        
        
        
        
        #
        # Parallelized calculations for each planet in [e-/FWHM/DIT], [e-/FWHM/DIT], [e-/FWHM/DIT]**2, [e-/FWHM/DIT], and [mn/DIT]
        #
        signal_4D_R   = open_memmap(signal_4D_R_path,   mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_R))
        signal_4D_l0  = open_memmap(signal_4D_l0_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_l0))
        signal_4D_Nl  = open_memmap(signal_4D_Nl_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_Nl))
        signal_4D_TI  = open_memmap(signal_4D_TI_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_TI))
        signal_4D_WFE = open_memmap(signal_4D_WFE_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_WFE))
        signal_4D_IWA = open_memmap(signal_4D_IWA_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_IWA))

        sigma_halo_2_4D_R   = open_memmap(sigma_halo_2_4D_R_path,   mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_R))
        sigma_halo_2_4D_l0  = open_memmap(sigma_halo_2_4D_l0_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_l0))
        sigma_halo_2_4D_Nl  = open_memmap(sigma_halo_2_4D_Nl_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_Nl))
        sigma_halo_2_4D_TI  = open_memmap(sigma_halo_2_4D_TI_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_TI))
        sigma_halo_2_4D_WFE = open_memmap(sigma_halo_2_4D_WFE_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_WFE))
        sigma_halo_2_4D_IWA = open_memmap(sigma_halo_2_4D_IWA_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_IWA))

        sigma_syst_base_2_4D_R   = open_memmap(sigma_syst_base_2_4D_R_path,   mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_R))
        sigma_syst_base_2_4D_l0  = open_memmap(sigma_syst_base_2_4D_l0_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_l0))
        sigma_syst_base_2_4D_Nl  = open_memmap(sigma_syst_base_2_4D_Nl_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_Nl))
        sigma_syst_base_2_4D_TI  = open_memmap(sigma_syst_base_2_4D_TI_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_TI))
        sigma_syst_base_2_4D_WFE = open_memmap(sigma_syst_base_2_4D_WFE_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_WFE))
        sigma_syst_base_2_4D_IWA = open_memmap(sigma_syst_base_2_4D_IWA_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_IWA))

        sigma_bkg_2_4D_R   = open_memmap(sigma_bkg_2_4D_R_path,   mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_R))
        sigma_bkg_2_4D_l0  = open_memmap(sigma_bkg_2_4D_l0_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_l0))
        sigma_bkg_2_4D_Nl  = open_memmap(sigma_bkg_2_4D_Nl_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_Nl))
        sigma_bkg_2_4D_TI  = open_memmap(sigma_bkg_2_4D_TI_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_TI))
        sigma_bkg_2_4D_WFE = open_memmap(sigma_bkg_2_4D_WFE_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_WFE))
        sigma_bkg_2_4D_IWA = open_memmap(sigma_bkg_2_4D_IWA_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_IWA))

        DIT_4D_R   = open_memmap(DIT_4D_R_path,   mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_R))
        DIT_4D_l0  = open_memmap(DIT_4D_l0_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_l0))
        DIT_4D_Nl  = open_memmap(DIT_4D_Nl_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_Nl))
        DIT_4D_TI  = open_memmap(DIT_4D_TI_path,  mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_TI))
        DIT_4D_WFE = open_memmap(DIT_4D_WFE_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_WFE))
        DIT_4D_IWA = open_memmap(DIT_4D_IWA_path, mode="w+", dtype=dtype, shape=(N_PT, N_band, N_config, N_IWA))
        
        
        
        # # debug path
        # init_worker(_CTX)
        # for idx in range(N_PT):
        #     process_ELT(idx)
        
        if sys.platform.startswith(("win", "darwin")):
            ctx_mp = mp.get_context("spawn")
        else:
            ctx_mp = mp.get_context("fork")
        print()
        warmup_signal_noise_numba()
        with ctx_mp.Pool(processes=nproc, initializer=init_worker, initargs=(_CTX,)) as pool:
            for _ in tqdm(pool.imap_unordered(process_ELT, range(N_PT), chunksize=chunksize), total=N_PT, desc=f"ELT/{instru} ({strehl}-strehl, {post_processing}-post-processing) multiprocessing over planets of signal, stellar halo photon noise and DIT"):
                pass
            
        signal_4D_R.flush()
        signal_4D_l0.flush()
        signal_4D_Nl.flush()
        signal_4D_TI.flush()
        signal_4D_WFE.flush()
        signal_4D_IWA.flush()

        sigma_halo_2_4D_R.flush()
        sigma_halo_2_4D_l0.flush()
        sigma_halo_2_4D_Nl.flush()
        sigma_halo_2_4D_TI.flush()
        sigma_halo_2_4D_WFE.flush()
        sigma_halo_2_4D_IWA.flush()

        sigma_syst_base_2_4D_R.flush()
        sigma_syst_base_2_4D_l0.flush()
        sigma_syst_base_2_4D_Nl.flush()
        sigma_syst_base_2_4D_TI.flush()
        sigma_syst_base_2_4D_WFE.flush()
        sigma_syst_base_2_4D_IWA.flush()
        
        sigma_bkg_2_4D_R.flush()
        sigma_bkg_2_4D_l0.flush()
        sigma_bkg_2_4D_Nl.flush()
        sigma_bkg_2_4D_TI.flush()
        sigma_bkg_2_4D_WFE.flush()
        sigma_bkg_2_4D_IWA.flush()
        
        DIT_4D_R.flush()
        DIT_4D_l0.flush()
        DIT_4D_Nl.flush()
        DIT_4D_TI.flush()
        DIT_4D_WFE.flush()
        DIT_4D_IWA.flush()
        
        
        # --- Clean temporary memmap files ---
        PSF_profile_6D_path.unlink()
        fraction_core_6D_path.unlink()
        
    
    
    # %%
    # COMPUTING DETECTION PROBABILITY

    # Plot general parameters
    dpi                = 300                 # Dots Per Inch
    fontsize           = 24                  # Standard fontsize
    ms                 = 15                  # Marker size
    lw                 = 3                   # Linewidth
    alpha              = 0.9                 # Opacity
    gain               = True                # True for probability gain (normalized yield), False for absolute yields
    light_regime_plot  = "thermal+reflected" # 'thermal", "reflected" or "thermal+reflected"
    band_regime_plot   =  "H"                # Band where the thermal/reflected domination regime is splitted
    ptypes             = ["Jupiter",                 "Saturn",                "Neptune",                 "Earth"]
    label_ptypes       = {"Jupiter": "Jupiter-like", "Saturn": "Saturn-like", "Neptune": "Neptune-like", "Earth": "Earth-like"}
    
    # Define parameters and their names
    params         = [R,                           l0,                                                Nl,                                           trans_instru,                                    100*sigma_m,                                                                      FoV,                     WFE,                                         IWA]                                                # params axis
    params_islog   = [True,                        False,                                             True,                                         False,                                           True,                                                                             True,                    False,                                       True]                                               # params space (linspace or logspace)
    params_names   = ["R",                         "l0 [µm]",                                         "Nl",                                         "trans_instru [e-/ph]",                          "sigma_m [%]",                                                                    "FoV [mas]",             "WFE [nm RMS]",                              "IWA [mas]"]                                        # params names
    params_names_l = [r"$R$",                      r"$\lambda_0$ [µm]",                               r"$N_{\lambda}$",                             r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                                                r"$FoV$ [mas]",          r"$WFE$ [nm RMS]",                           r"IWA [mas]",]                                      # params labels
    params_names_L = [r"Spectral resolution $R$",  r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Number of spectral channel $N_{\lambda}$",  r"Instrumental transmission $\gamma_{instru}$",  f"{residuals}"r" residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]",  r"Post-AO ${WFE}$ [nm RMS]", r"Coronagraph $IWA$ [mas]"] # params detailed labels
    
    # COMPUTING MAIN QUANTITIES
    
    # Computing SNR (planets, bands, configs, *)
    SNR_4D_R, SNR_4D_l0, SNR_4D_Nl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA = get_SNR(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, NbPixel, SNR_thr, separation_planets,
                                                                                                   idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Nl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
                                                                                                   signal_4D_R,   sigma_halo_2_4D_R,   sigma_syst_base_2_4D_R,   sigma_bkg_2_4D_R,   DIT_4D_R,
                                                                                                   signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_syst_base_2_4D_l0,  sigma_bkg_2_4D_l0,  DIT_4D_l0,
                                                                                                   signal_4D_Nl,  sigma_halo_2_4D_Nl,  sigma_syst_base_2_4D_Nl,  sigma_bkg_2_4D_Nl,  DIT_4D_Nl,
                                                                                                   signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_syst_base_2_4D_TI,  sigma_bkg_2_4D_TI,  DIT_4D_TI,
                                                                                                   signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_syst_base_2_4D_WFE, sigma_bkg_2_4D_WFE, DIT_4D_WFE,
                                                                                                   signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_syst_base_2_4D_IWA, sigma_bkg_2_4D_IWA, DIT_4D_IWA)
    # TODO: Useless now : Computing Pdet (bands, configs, *)
    Pdet_3D_R, Pdet_3D_l0, Pdet_3D_Nl, Pdet_3D_TI, Pdet_3D_SM, Pdet_3D_FoV, Pdet_3D_WFE, Pdet_3D_IWA = get_Pdet(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, SNR_thr, separation_planets,
                                                                                                                idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Nl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
                                                                                                                SNR_4D_R, SNR_4D_l0, SNR_4D_Nl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA)
    Pdet = [Pdet_3D_R, Pdet_3D_l0, Pdet_3D_Nl, Pdet_3D_TI, Pdet_3D_SM, Pdet_3D_FoV, Pdet_3D_WFE, Pdet_3D_IWA]                                        # params detection probabilities
    
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
        Pdet.pop(idim)
        params.pop(idim)
        params_islog.pop(idim)
        params_names.pop(idim)
        params_names_l.pop(idim)
        params_names_L.pop(idim)
    Ndim  = len(params)
    ncols = min(2, Ndim)
    nrows = int(np.ceil(Ndim / ncols))
    
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
    
    # Plot quantities for the considered planet-light regime.
    if light_regime_plot not in {"thermal", "reflected", "thermal+reflected"}:
        raise ValueError("light_regime_plot must be 'thermal', 'reflected', or 'thermal+reflected'.")
    
    if light_regime_plot == "thermal":
        mask_regime_plot = mask_thermal
        regime_label     = "thermal"
    elif light_regime_plot == "reflected":
        mask_regime_plot = mask_reflected
        regime_label     = "reflected"
    elif light_regime_plot == "thermal+reflected":
        mask_regime_plot = mask_thermal | mask_reflected
        regime_label     = "thermal+reflected"
    
    # Detection probability per planet type and light regime.
    N_PT_ptypes_thermal   = np.zeros((len(ptypes)), dtype=int)
    N_PT_ptypes_reflected = np.zeros((len(ptypes)), dtype=int)
    Pdet_ptypes_thermal   = [None] * len(ptypes)
    Pdet_ptypes_reflected = [None] * len(ptypes)
    for ipt, ptype in enumerate(ptypes):
        mask_ptype                 = get_mask_planet_type(planet_table=planet_table, planet_type=ptype)
        N_PT_ptypes_thermal[ipt]   = int(np.sum(mask_ptype & mask_thermal))
        N_PT_ptypes_reflected[ipt] = int(np.sum(mask_ptype & mask_reflected))
        if "thermal" in light_regime_plot:
            print(f"\n{label_ptypes[ptype]} in thermal light:")
            idx_ptype_thermal  = np.where(mask_ptype & mask_thermal)[0]
            Pdet_ptype_thermal = list(get_Pdet(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, SNR_thr, separation_planets,
                                                idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Nl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
                                                SNR_4D_R, SNR_4D_l0, SNR_4D_Nl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA, indices=idx_ptype_thermal))
            for dim in sorted(dim_to_remove, reverse=True):
                Pdet_ptype_thermal.pop(dim)
        else:
            Pdet_ptype_thermal = None
        if "reflected" in light_regime_plot:
            print(f"\n{label_ptypes[ptype]} in reflected light:")
            idx_ptype_reflected  = np.where(mask_ptype & mask_reflected)[0]
            Pdet_ptype_reflected = list(get_Pdet(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, SNR_thr, separation_planets,
                                                  idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Nl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
                                                  SNR_4D_R, SNR_4D_l0, SNR_4D_Nl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA, indices=idx_ptype_reflected))
            for dim in sorted(dim_to_remove, reverse=True):
                Pdet_ptype_reflected.pop(dim)
        else:
            Pdet_ptype_reflected = None
        Pdet_ptypes_thermal[ipt]   = Pdet_ptype_thermal
        Pdet_ptypes_reflected[ipt] = Pdet_ptype_reflected
    
    # Combine thermal/reflected according to light_regime_plot.
    if light_regime_plot == "thermal":
        Pdet_ptypes_plot = Pdet_ptypes_thermal
        N_PT_ptypes_plot = N_PT_ptypes_thermal
    elif light_regime_plot == "reflected":
        Pdet_ptypes_plot = Pdet_ptypes_reflected
        N_PT_ptypes_plot = N_PT_ptypes_reflected
    elif light_regime_plot == "thermal+reflected":
        Pdet_ptypes_plot = [None] * len(ptypes)
        N_PT_ptypes_plot = np.zeros((len(ptypes)), dtype=int)
        for ipt in range(len(ptypes)):
            Pdet_th               = Pdet_ptypes_thermal[ipt]
            Pdet_re               = Pdet_ptypes_reflected[ipt]
            N_th                  = N_PT_ptypes_thermal[ipt]
            N_re                  = N_PT_ptypes_reflected[ipt]
            N_tot                 = N_th + N_re
            N_PT_ptypes_plot[ipt] = N_tot
            if N_tot == 0:
                continue
            Pdet_combined = [None] * Ndim
            for dim in range(Ndim):
                Pdet_combined_dim = np.zeros_like(Pdet_th[dim], dtype=float)
                if N_th > 0:
                    Pdet_combined_dim += N_th * Pdet_th[dim]
                if N_re > 0:
                    Pdet_combined_dim += N_re * Pdet_re[dim]
                Pdet_combined_dim /= N_tot
                Pdet_combined[dim] = Pdet_combined_dim
            Pdet_ptypes_plot[ipt] = Pdet_combined
    
    # Combine all plotted planet types into a single Pdet list.
    N_PT_plot = int(np.sum(N_PT_ptypes_plot))
    if N_PT_plot == 0:
        raise RuntimeError(f"No planets available for light_regime_plot='{light_regime_plot}'.")
    Pdet_plot = [None] * Ndim
    for dim in range(Ndim):
        Pdet_dim = None
        for ipt in range(len(ptypes)):
            if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                continue
            if Pdet_dim is None:
                Pdet_dim = np.zeros_like(Pdet_ptypes_plot[ipt][dim], dtype=float)
            Pdet_dim += N_PT_ptypes_plot[ipt] * Pdet_ptypes_plot[ipt][dim]
        Pdet_dim      /= N_PT_plot
        Pdet_plot[dim] = Pdet_dim

    # Printing number of detections at reference
    planet_table_configs = [None] * N_config
    for ic, config in enumerate(configs):
        planet_table_configs[ic]        = load_planet_table(get_filename_table(table="Archive", instru=instru, apodizer=config if instru=="HARMONI" else "NO_SP", strehl="JQ1" if instru=="HARMONI" else "MED", coronagraph=None if instru=="HARMONI" else config, systematics=False, PCA=False, name_model=thermal_model+"+"+reflected_model))
        planet_table_configs[ic]["SNR"] = get_SNR_from_table(planet_table=planet_table_configs[ic], exposure_time=exposure_time, band="INSTRU")
    
    # Ptypes Ndet    
    N_PT_ptypes        = np.zeros((len(ptypes))).astype(int)
    Ndet_ptypes_ref_2D = np.zeros((len(ptypes), N_config))
    mask_ptypes           = np.full(N_PT, False)
    for ipt, ptype in enumerate(ptypes):        
        mask_ptype       = get_mask_planet_type(planet_table=planet_table, planet_type=ptype) & mask_regime_plot
        mask_ptypes     |= mask_ptype
        N_PT_ptypes[ipt] = int(mask_ptype.sum())        
        
        # ptype arrays at ref values
        SNR_ref_ptype_3D    = np.take_along_axis(SNR_4D_R[mask_ptype],    idx_R_ref[None, :, None, None], axis=3)[:, :, :, 0] # (planet, band, config)
        signal_ref_ptype_3D = np.take_along_axis(signal_4D_R[mask_ptype], idx_R_ref[None, :, None, None], axis=3)[:, :, :, 0] # (planet, band, config)
        DIT_ref_ptype_3D    = np.take_along_axis(DIT_4D_R[mask_ptype],    idx_R_ref[None, :, None, None], axis=3)[:, :, :, 0] # (planet, band, config)

        # ptype arrays at max SNR values
        idx_SNR_max         = np.argmax(SNR_ref_ptype_3D, axis=1)                                               # (planet, config)
        SNR_ref_ptype_2D    = np.take_along_axis(SNR_ref_ptype_3D,    idx_SNR_max[:, None, :], axis=1)[:, 0, :] # (planet, config)
        signal_ref_ptype_2D = np.take_along_axis(signal_ref_ptype_3D, idx_SNR_max[:, None, :], axis=1)[:, 0, :] # (planet, config)
        DIT_ref_ptype_2D    = np.take_along_axis(DIT_ref_ptype_3D,    idx_SNR_max[:, None, :], axis=1)[:, 0, :] # (planet, config)
        
        # Number of ptype detections
        mask_det_ptype_any_band_2D = np.any(SNR_ref_ptype_3D >= SNR_thr, axis=1) # (planet, config)
        Ndet_ptypes_ref_2D[ipt]    = np.sum(mask_det_ptype_any_band_2D, axis=0)  # (config)
        for ic, config in enumerate(configs):
            mask_det_ptype_any_band_1D = mask_det_ptype_any_band_2D[:, ic] # (planet)
            SNR_ref_ptype_1D           = SNR_ref_ptype_2D[:, ic]           # (planet)
            signal_ref_ptype_1D        = signal_ref_ptype_2D[:, ic]        # (planet)
            DIT_ref_ptype_1D           = DIT_ref_ptype_2D[:, ic]           # (planet)
            if mask_det_ptype_any_band_1D.sum() > 0:
                planet_names  = planet_table["PlanetName"][mask_ptype][mask_det_ptype_any_band_1D]
                planet_snr    = SNR_ref_ptype_1D[mask_det_ptype_any_band_1D]
                planet_signal = signal_ref_ptype_1D[mask_det_ptype_any_band_1D]
                planet_dit    = DIT_ref_ptype_1D[mask_det_ptype_any_band_1D]
                print()
                print_subheader(f"{ptype}-like detected (with {str(config).replace('_', ' ')}-config):")
                for ip, pn in enumerate(planet_names):
                    try:
                        idx          = get_planet_index(planet_table=planet_table_configs[ic], planet_name=pn)
                        ratio_snr    = planet_table_configs[ic][idx]["SNR"]           / planet_snr[ip]
                        ratio_dit    = planet_table_configs[ic][idx]["DIT_INSTRU"]    / planet_dit[ip]
                        ratio_signal = planet_table_configs[ic][idx]["signal_INSTRU"] / planet_signal[ip] / ratio_dit
                        print_metric(f"{pn}", "S/N",    f"{planet_snr[ip]:.2f} | ratio_snr = {ratio_snr:.2f} | ratio_signal = {ratio_signal:.2f} | ratio_dit = {ratio_dit:.2f}",    "", label_width=15, key_width=3, value_width=7, units_width=1)
                        #print_metric(f"{pn}", "signal", f"{planet_signal[ip]:.2f} (ratio_signal = {ratio_signal:.2f})", "", label_width=15, key_width=3, value_width=7, units_width=1)
                        #print_metric(f"{pn}", "DIT",    f"{planet_dit[ip]:.2f}    (ratio_dit    = {ratio_dit:.2f})", "", label_width=15, key_width=3, value_width=7, units_width=1)
                    except:
                        print(f"{pn}: WARNING, unrecognized SNR planet_table")
    # Other Ndet
    mask_other                 = ~mask_ptypes & mask_regime_plot
    N_PT_other                 = mask_other.sum()
    SNR_ref_other_3D           = np.take_along_axis(SNR_4D_R[mask_other],    idx_R_ref[None, :, None, None], axis=3)[:, :, :, 0] # (planet, band, config)
    mask_det_other_any_band_2D = np.any(SNR_ref_other_3D >= SNR_thr, axis=1) # (planet, config)
    Ndet_other_ref_1D          = np.sum(mask_det_other_any_band_2D, axis=0)  # (config)

    # All Ndet (config)
    Ndet_1D_all_ref = np.nansum(Ndet_ptypes_ref_2D, axis=0) + Ndet_other_ref_1D
    
    print("\n")
    title       = "Number of detections at reference"
    label_width = max(len("All"), max(len(str(p)) for p in ptypes))
    value_width = 6
    blocks      = []
    box_width   = len(title)
    for ic, config in enumerate(configs):
        header = f"{config}-config"
        lines  = [f"{'All':<{label_width}} : {Ndet_1D_all_ref[ic]:>{value_width}.0f}"]
        for ipt, ptype in enumerate(ptypes):
            lines.append(f"{ptype:<{label_width}} : {Ndet_ptypes_ref_2D[ipt, ic]:>{value_width}.0f}")
        lines.append(f"{'Other':<{label_width}} : {Ndet_other_ref_1D[ic]:>{value_width}.0f}")
        block_inner_width = max(len(header) + 2, max(len(line) for line in lines))
        box_width         = max(box_width, block_inner_width)
        blocks.append((header, lines))
    # Titre
    print(f"╔{'═' * (box_width + 2)}╗")
    print(f"║ {title:^{box_width}} ║")
    print(f"╚{'═' * (box_width + 2)}╝")
    # Blocs
    for header, lines in blocks:
        n_dash = box_width - len(header) - 1
        print(f"\n┌─ {header} " + "─" * n_dash + "┐")
        for line in lines:
            print(f"│ {line:<{box_width}} │")
        print(f"└{'─' * (box_width + 2)}┘")
    
    # Helper: normalize to probability gain or convert to yield
    def convert_Pdet_to_plot_quantity(Pdet_curve, N_PT, gain, ymax_global=None):
        """
        Convert a 1D Pdet curve either to globally normalized gain [%] or to yield.
        """
        y = np.asarray(Pdet_curve, dtype=float)
        if gain:
            if ymax_global is not None and np.isfinite(ymax_global) and ymax_global > 0:
                y = 100.0 * y / ymax_global
            else:
                y = np.zeros_like(y)
        else:
            y = y * N_PT
        return y
    
    
    
    # %%
    # PLOTS
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows), dpi=dpi, sharey=True)    
    axes      = np.atleast_2d(axes)
    for dim in range(Ndim):
        r  = dim // ncols
        t  = dim % ncols
        ax = axes[r, t]
        ax.tick_params(axis='both', which='major', labelsize=fontsize+2)

        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)
    
        # --- Plot ---
        ymax_global = np.nanmax(Pdet_plot[dim]) if gain else None        
        for ib, band in enumerate(bands):
            for ic, config in enumerate(configs):
                y = convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_plot[dim][ib, ic], N_PT=N_PT_plot, gain=gain, ymax_global=ymax_global)
                ax.plot(params[dim], y, ls=ls_configs[ic], lw=lw, c=cmap_bands(ib), alpha=alpha, zorder=3)
                
        # --- Reference values ---
        if len(params[dim]) == N_R and np.all(params[dim] == R):
            if instru == "HARMONI":
                n    = len(bands)
                db   = (n - 1) / 2.0
                dlog = 0.015   # 0.01 decade ≈ facteur 10**0.01 ≈ 1.023 (2.3%) par cran
                for ib, band in enumerate(bands):
                    x0 = R_ref[ib]
                    x  = x0 * 10**(dlog * (ib - db))
                    ax.axvline(x, ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)
            elif instru == "ANDES":
                ax.axvline(np.nanmean(R_ref), ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)

        elif len(params[dim]) == len(l0) and np.all(params[dim] == l0):
            if instru == "HARMONI":
                for ib, band in enumerate(bands):
                    ax.axvline(l0_ref[ib], ls="-", lw=2*lw,    c=cmap_bands(ib),     alpha=alpha, zorder=0)
                    ax.axvspan(lmin_bands[ib], lmax_bands[ib], color=cmap_bands(ib), alpha=0.1, zorder=-10)
            elif instru == "ANDES":
                ax.axvline(np.nanmean(l0_ref), ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
                ax.axvspan(np.nanmean(lmin_bands), np.nanmean(lmax_bands), color="black", alpha=0.1, zorder=-10)
                
        elif len(params[dim]) == len(Nl) and np.all(params[dim] == Nl):
            ax.axvline(Nl_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
            
        elif len(params[dim]) == len(trans_instru) and np.all(params[dim] == trans_instru):
            ax.set_xlim(0, trans_instru_max)
            ic = 0
            if instru=="HARMONI":
                for ib, band in enumerate(bands):
                    ax.axvline(trans_instru_ref[ib, ic], ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)
            elif instru=="ANDES":
                ib = 0
                #ax.axvline(np.nanmean(trans_instru_ref), ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
                ax.axvline(trans_instru_ref[-1, -1], ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)

        elif len(params[dim]) == len(FoV) and np.all(params[dim] == FoV):
            if instru=="HARMONI":
                ax.axvline(FoV_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
            elif instru=="ANDES":
                for ib, band in enumerate(bands):
                    ax.axvline(FoV_ref[ib], ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)

        elif len(params[dim]) == len(WFE) and np.all(params[dim] == WFE):
            ax.axvline(WFE_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
        
        elif len(params[dim]) == len(IWA) and np.all(params[dim] == IWA):
            ax.axvline(IWA_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
        
        # --- X scale + minor ticks ---
        if len(params[dim]) == len(trans_instru) and np.all(params[dim] == trans_instru):
            ax.set_xlim(0, trans_instru_max)
        else:
            ax.set_xlim(np.nanmin(params[dim]), np.nanmax(params[dim]))
        if params_islog[dim]:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        # --- Labels ---
        ax.set_xlabel(params_names_L[dim], fontsize=fontsize+4)
    
        # y label only once per row (left-most), since y is shared per row
        if t == 0:
            if gain:
                ax.set_ylabel("Detection gain [%]", fontsize=fontsize+4)
            else:
                ax.set_ylabel("Yield", fontsize=fontsize+4)
        else:
            ax.tick_params(labelleft=False)  # keep shared-y clean
        
        # --- Legends ---
        if dim == 0:
            # band legend
            if instru == "HARMONI":
                handles_band = [Line2D([], [], ls="-", lw=lw, color=cmap_bands(ib), label=band.replace("_", " ")) for ib, band in enumerate(bands)]
                leg_band     = ax.legend(handles=handles_band, fontsize=fontsize+2, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Bands", title_fontsize=fontsize+2)
                ax.add_artist(leg_band)
            # pxscale legend
            elif instru == "ANDES":
                handles_pxscale = [Line2D([], [], ls="-", lw=lw, color=cmap_bands(ib), label=f"{pxscales[ib]:.0f} mas") for ib, band in enumerate(bands)]
                leg_pxscale     = ax.legend(handles=handles_pxscale, fontsize=fontsize+2, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Fiber scales", title_fontsize=fontsize+2)
                ax.add_artist(leg_pxscale)
        if dim == 2 and len(configs) > 1:
            # config legend
            handles_config = [Line2D([], [], ls=ls_configs[ic], lw=lw, color="k", label=str(config).replace("_", " ").replace("LYOT", "Lyot")) for ic, config in enumerate(configs)]
            leg_config     = ax.legend(handles=handles_config, fontsize=fontsize+2, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Apodizer config" if w_apodizer else "Coronagraph config", title_fontsize=fontsize+2)
            ax.add_artist(leg_config)
        if dim == 3:
            handles_ref = [Line2D([], [], ls="None", marker="|", markersize=30, markeredgewidth=3, color="k", label="Reference values")]
            leg_ref     = ax.legend(handles=handles_ref, fontsize=fontsize+3, loc="lower right", frameon=True, edgecolor="gray", facecolor="white")
            ax.add_artist(leg_ref)
        
        # systematic budget with JWST/MIRI/MRS ~ 1%
        if post_processing == "MM" and params_names[dim] == "sigma_m [%]":
            x0 = 1.0 # [%]
            ax.axvline(x0, c="k", ls=":", lw=lw)
            ax.annotate("JWST/MIRI/MRS", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        elif post_processing == "DI" and params_names[dim] == "sigma_m [%]":
            x0 = 0.1  # [%] = 1e-3 in fractional units with VLT/SPHERE PACO
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
    
    # Turn off unused axes (if Ndim not multiple of ncols)
    for k in range(Ndim, nrows*ncols):
        r = k // ncols
        t = k % ncols
        axes[r, t].axis("off")
        
    # --- Y lims ---
    for ax_i in axes.ravel()[:Ndim]:
        ax_i.set_yscale("log")
        if gain:
            ax_i.set_ylim(1, 102)
        else:
            ax_i.set_ylim(1, None)
    
    # --- Title ---
    quantity_name    = "detection probability gain" if gain else f"detection yield in {exposure_time/60:.0f} hr "
    ptype_plot_label = "+".join([label_ptypes[ptype] for ipt, ptype in enumerate(ptypes) if N_PT_ptypes_plot[ipt] > 0])
    title  = f"{regime_label.upper()} LIGHT\n\nELT/{instru} {quantity_name} with {post_processing.replace('DI', 'differential imaging').replace('MM', 'molecular mapping')}"
    title  = f"ELT/{instru} | {post_processing} | {regime_label.upper()} LIGHT"

    #title  = f"ELT/{instru} {quantity_name} with {post_processing.replace('DI', 'differential imaging').replace('MM', 'molecular mapping')}"
    #title += f"\nfor {N_PT_plot} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} {ptype_plot_label} planets in {regime_label} light"
    #title += f"\nthermal/reflected split estimated in {band_regime_plot}-band"
    fig.suptitle(title, fontsize=fontsize+10, weight="bold", y=0.92)
    fig.tight_layout(h_pad=4.0, w_pad=2.0, rect=[0, 0, 1, 0.91])
    plt.show()
    
    
    
    # %%
    # ============================================================================
    # MASS–SMA LOCAL DETECTION FRACTION
    # P_det(M, SMA) = P_LEVEL contours for different instrumental parameter values
    #
    # IMPORTANT:
    # With table_type="Archive", this is an ARCHIVE-CONDITIONAL local detection
    # fraction, not an occurrence-rate-corrected survey completeness.
    # ============================================================================
    
    # ----------------------------------------------------------------------------
    # User parameters
    # ----------------------------------------------------------------------------
    if instru == "ANDES":
        BAND_FIXED      = "YJH_10mas_100"
        CONFIG_FIXED    = "LYOT"
    elif instru == "HARMONI":
        BAND_FIXED      = "H"
        CONFIG_FIXED    = "SP_Prox"
    
    P_LEVEL         = 0.10     # contour displayed
    MASS_UNIT       = "Mearth" # "Mearth" or "Mjup"
    
    N_CURVES        = 10       # Number of parameter curves
    N_GRID          = 50       # only controls contour sampling / visual resolution

    # Fixed Gaussian kernel in log10(M)-log10(a) space
    BW_A_DEX        = 0.35       # 1 sigma bandwidth in log10(a)
    BW_M_DEX        = 0.35       # 1 sigma bandwidth in log10(M)
    KERNEL_TRUNC    = 10        # ignore objects farther than KERNEL_TRUNC sigma
    MIN_NEFF        = 10        # minimum effective local sample size
    
    GRID_PERCENTILE = (0.5, 99.5)
    
    SHOW_ARCHIVE    = True
    
    # ----------------------------------------------------------------------------
    # Fixed instrumental configuration
    # ----------------------------------------------------------------------------
    if BAND_FIXED not in bands:
        raise ValueError(f"BAND_FIXED={BAND_FIXED!r} not available.\nAvailable bands: {bands}")
    if CONFIG_FIXED not in configs:
        raise ValueError(f"CONFIG_FIXED={CONFIG_FIXED!r} not available.\nAvailable configs: {configs}")
    ib = bands.index(BAND_FIXED)
    ic = configs.index(CONFIG_FIXED)
    
    # ----------------------------------------------------------------------------
    # Physical coordinates of the population
    # ----------------------------------------------------------------------------
    sma_AU      = np.asarray(planet_table["SMA"].value,        dtype=float)
    mass_Mearth = np.asarray(planet_table["PlanetMass"].value, dtype=float)
    
    if MASS_UNIT == "Mearth":
        mass_plot  = mass_Mearth
        mass_label = r"Planet mass [$M_{\oplus}$]"
    elif MASS_UNIT == "Mjup":
        mass_plot  = mass_Mearth / 317.8284065946748
        mass_label = r"Planet mass [$M_{\rm Jup}$]"
    else:
        raise ValueError("MASS_UNIT must be 'Mearth' or 'Mjup'.")
    
    # Use exactly the thermal/reflected population selected in the main script.
    sample_mask  = np.asarray(mask_regime_plot, dtype=bool).copy()
    sample_mask &= np.isfinite(sma_AU) & (sma_AU > 0) & np.isfinite(mass_plot) & (mass_plot > 0)
    
    idx_sample = np.where(sample_mask)[0]
    
    if idx_sample.size < max(30, MIN_NEFF + 1):
        raise RuntimeError(f"Only {idx_sample.size} valid planets in the selected population.")
    
    # ----------------------------------------------------------------------------
    # Work in log10(a) - log10(M) space
    # ----------------------------------------------------------------------------
    log_a = np.log10(sma_AU[idx_sample])
    log_m = np.log10(mass_plot[idx_sample])
    X     = np.column_stack((log_a, log_m))
    
    # ----------------------------------------------------------------------------
    # Common Mass-SMA grid
    #
    # N_GRID only controls the numerical sampling of the displayed probability
    # field. It does NOT control the smoothing scale.
    # ----------------------------------------------------------------------------
    a_lim  = np.nanpercentile(log_a, GRID_PERCENTILE)
    m_lim  = np.nanpercentile(log_m, GRID_PERCENTILE)
    xa     = np.linspace(a_lim[0], a_lim[1], N_GRID)
    ym     = np.linspace(m_lim[0], m_lim[1], N_GRID)
    XA, YM = np.meshgrid(xa, ym)
    G      = np.column_stack((XA.ravel(), YM.ravel()))
    
    # ----------------------------------------------------------------------------
    # Fixed Gaussian kernel
    #
    # Distances are expressed directly in units of the chosen bandwidths:
    #
    #     r^2 = (Delta log10(a) / BW_A_DEX)^2
    #         + (Delta log10(M) / BW_M_DEX)^2
    #
    # and
    #
    #     w = exp(-r^2 / 2).
    #
    # Contrary to the previous adaptive kNN kernel, the physical smoothing scale
    # is therefore identical everywhere in the Mass-SMA plane.
    # ----------------------------------------------------------------------------
    X_kernel = np.column_stack((log_a / BW_A_DEX, log_m / BW_M_DEX))
    G_kernel = np.column_stack((G[:, 0] / BW_A_DEX, G[:, 1] / BW_M_DEX))
    tree     = cKDTree(X_kernel)
    
    # Only objects within KERNEL_TRUNC Gaussian sigmas contribute.
    neighbours = tree.query_ball_point(G_kernel, r=KERNEL_TRUNC)
    
    # Build a sparse kernel-weight matrix:
    #
    #     W[g, i] = weight of planet i at grid point g.
    #
    # Sparse storage avoids allocating a huge N_grid^2 x N_planets dense matrix.
    rows = []
    cols = []
    data = []
    for ig, ids_local in enumerate(neighbours):
        if len(ids_local) == 0:
            continue
        ids_local = np.asarray(ids_local, dtype=int)
        delta     = X_kernel[ids_local] - G_kernel[ig]
        r2        = np.sum(delta**2, axis=1)
        weights   = np.exp(-0.5 * r2)
        rows.extend(np.full(ids_local.size, ig, dtype=int))
        cols.extend(ids_local)
        data.extend(weights)
    W = csr_matrix((np.asarray(data, dtype=np.float32), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=(G.shape[0], idx_sample.size))
    
    # ----------------------------------------------------------------------------
    # Local statistical support
    # ----------------------------------------------------------------------------
    sumw  = np.asarray(W.sum(axis=1)).ravel()
    sumw2 = np.asarray(W.power(2).sum(axis=1)).ravel()
    neff  = np.divide(sumw**2, sumw2, out=np.zeros_like(sumw), where=sumw2 > 0)
    
    # Avoid extrapolation outside the region globally sampled by the archive.
    try:
        inside_hull = Delaunay(X_kernel).find_simplex(G_kernel) >= 0
    except QhullError:
        inside_hull = np.ones(G.shape[0], dtype=bool)
    
    
    # A grid point is considered statistically supported only if:
    #   1) it lies inside the global archive envelope;
    #   2) enough independent/effective planets contribute locally.
    support = inside_hull & (neff >= MIN_NEFF)
    
    
    # ----------------------------------------------------------------------------
    # Local detection-fraction estimator
    #
    #                         sum_i w_i D_i
    # P_det(log a, log M) = -----------------
    #                           sum_i w_i
    #
    # with D_i = 1 for S/N >= threshold and 0 otherwise.
    #
    # This is a fixed-bandwidth Gaussian Nadaraya-Watson estimator.
    #
    # It can also be interpreted as the continuous analogue of:
    #
    #                         N_detected
    #                         ----------
    #                           N_total
    #
    # with soft Gaussian bins instead of arbitrary rectangular bins.
    # ----------------------------------------------------------------------------
    def local_detection_fraction(det_all):
        detected    = np.asarray(det_all, dtype=float)[idx_sample]
        numerator   = np.asarray(W @ detected).ravel()
        p           = np.divide(numerator, sumw, out=np.full_like(sumw, np.nan), where=sumw > 0)
        p[~support] = np.nan
        return p.reshape(N_GRID, N_GRID)

    # Choose N_CURVES representative parameter values, excluding the reference
    def pick_curve_indices(values, n=N_CURVES, ref=None, log=False):
        v        = np.asarray(values, dtype=float)
        good     = np.isfinite(v) & ((v > 0) if log else True)
        ids_good = np.where(good)[0]    
        if ids_good.size == 0:
            return np.array([], dtype=int), None
        # Reference index, treated separately
        j_ref = None
        if ref is not None and np.isfinite(ref) and (ref > 0 or not log):
            j_ref    = ids_good[np.nanargmin(np.abs(v[ids_good] - ref))]
            ids_good = ids_good[ids_good != j_ref]
        if ids_good.size == 0:
            return np.array([], dtype=int), j_ref
        # Sort in linear/log parameter space and select exactly N_CURVES
        z     = np.log10(v[ids_good]) if log else v[ids_good]
        order = ids_good[np.argsort(z)]
        pos   = np.round(np.linspace(0, order.size - 1, min(n, order.size))).astype(int)
        ids   = order[pos]
        return ids, j_ref
    
    # ----------------------------------------------------------------------------
    # Parameter definitions
    # ----------------------------------------------------------------------------
    param_specs = [
        dict(name="R",   values=np.asarray(R,            float), snr=SNR_4D_R,   label=r"Spectral resolution $R$",                         log=True,  ref=float(R_ref[ib])),
        dict(name="l0",  values=np.asarray(l0,           float), snr=SNR_4D_l0,  label=r"Central wavelength $\lambda_0$ [$\mu$m]",         log=False, ref=float(l0_ref[ib])),
        dict(name="Nl",  values=np.asarray(Nl,           float), snr=SNR_4D_Nl,  label=r"Number of spectral channels $N_\lambda$",         log=True,  ref=float(Nl_ref)),
        dict(name="TI",  values=np.asarray(trans_instru, float), snr=SNR_4D_TI,  label=r"Instrumental transmission $\gamma_{\rm instru}$", log=False, ref=float(trans_instru_ref[ib, ic])),
        dict(name="SM",  values=100*np.asarray(sigma_m,  float), snr=SNR_4D_SM,  label=r"Residual amplitude $\sigma_m$ [%]",               log=True,  ref=None),
        dict(name="FoV", values=np.asarray(FoV,          float), snr=None,       label=r"Field of view [mas]",                             log=True,  ref=None),#float(np.atleast_1d(FoV_ref)[ib] if np.ndim(FoV_ref) else FoV_ref)),
        dict(name="WFE", values=np.asarray(WFE,          float), snr=SNR_4D_WFE, label=r"Post-AO WFE [nm RMS]",                            log=False, ref=float(WFE_ref)),
    ]
    
    # Coronagraphic IWA only when relevant.
    if np.asarray(IWA).size > 1 and np.all([x is not None for x in np.asarray(IWA, dtype=object)]):
        param_specs.append(dict(name="IWA", values=np.asarray(IWA, float), snr=SNR_4D_IWA, label=r"Coronagraph IWA [mas]", log=True, ref=float(IWA_ref)))
    
    # ----------------------------------------------------------------------------
    # Reference SNR used for the FoV panel
    # ----------------------------------------------------------------------------
    snr_ref_fov = np.asarray(SNR_4D_R[:, ib, ic, idx_R_ref[ib]], dtype=float)
    
    # ----------------------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------------------
    ncols     = 2
    nrows     = int(np.ceil(len(param_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.8 * nrows), dpi=dpi, sharex=True, sharey=True)
    axes      = np.atleast_1d(axes).ravel()
    
    for iax, spec in enumerate(param_specs):
        ax         = axes[iax]
        values     = spec["values"]
        ids, j_ref = pick_curve_indices(values, ref=spec["ref"], log=spec["log"])
    
        # Colour scale
        if spec["log"]:
            norm = LogNorm(vmin=np.nanmin(values[values > 0]), vmax=np.nanmax(values))
        else:
            norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        cmap = plt.get_cmap("viridis")
    
        # P_det = P_LEVEL contour for each instrumental parameter value
        for j in ids:
            if spec["name"] == "FoV":
                # Keep exactly the current FastYield convention:
                # FoV is applied to the instantaneous/projected AngSep already
                # stored in the archive table.
                det = (np.nan_to_num(snr_ref_fov, nan=-np.inf) >= SNR_thr) & (separation_planets <= values[j] / 2.0)
            else:
                snr = np.asarray(spec["snr"][:, ib, ic, j], dtype=float)
                # Same convention as get_Pdet:
                # NaN -> non-detection.
                det = np.nan_to_num(snr, nan=-np.inf) >= SNR_thr
    
            P      = local_detection_fraction(det)
            finite = np.isfinite(P)
    
            # A P=P_LEVEL contour exists only if the local probability spans P_LEVEL.
            if not np.any(finite) or np.nanmin(P) > P_LEVEL or np.nanmax(P) < P_LEVEL:
                continue
            ax.contour(10**xa, 10**ym, P, levels=[P_LEVEL], colors=[cmap(norm(values[j]))], linewidths=2.5, zorder=4)
        
        # Reference contour, plotted separately in black
        if j_ref is not None:
            if spec["name"] == "FoV":
                det = (np.nan_to_num(snr_ref_fov, nan=-np.inf) >= SNR_thr) & (separation_planets <= values[j_ref] / 2.0)
            else:
                snr = np.asarray(spec["snr"][:, ib, ic, j_ref], dtype=float)
                det = np.nan_to_num(snr, nan=-np.inf) >= SNR_thr
            P_ref      = local_detection_fraction(det)
            finite_ref = np.isfinite(P_ref)
            if np.any(finite_ref) and np.nanmin(P_ref) <= P_LEVEL <= np.nanmax(P_ref):
                ax.contour(10**xa, 10**ym, P_ref, levels=[P_LEVEL], colors="k", linewidths=5.0, zorder=5)
        
        # Archive sampling shown very faintly
        if SHOW_ARCHIVE:
            ax.scatter(sma_AU[idx_sample], mass_plot[idx_sample], s=6, c="0.5", alpha=0.2, edgecolors="none", zorder=1)
        
        # Earth
        earth_mass = 1.0 if MASS_UNIT == "Mearth" else 1.0 / 317.8284065946748
        ax.text(1.0, earth_mass, r"$\oplus$", fontsize=30, ha="center", va="center", color="k", zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        # Axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10**xa[0], 10**xa[-1])
        ax.set_ylim(10**ym[0], 10**ym[-1])
        ax.grid(which="major", ls="--", lw=0.7, alpha=0.35)
        ax.grid(which="minor", ls=":",  lw=0.4, alpha=0.20)
        ax.tick_params(axis="both", which="major", labelsize=fontsize - 3)
    
        # Colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(spec["label"], fontsize=fontsize, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=fontsize - 3)
        
        # Reference instrumental value
        if j_ref is not None:
            cbar.ax.axhline(values[j_ref], color="k", lw=5.0)
    
    # Empty axes
    for ax in axes[len(param_specs):]:
        ax.axis("off")
    
    # Labels
    for r in range(nrows):
        axes[r * ncols].set_ylabel(mass_label, fontsize=fontsize)
    
    for ic in range(ncols):
        ia = (nrows - 1) * ncols + ic
        if ia < len(axes):
            axes[ia].set_xlabel(r"Semi-major axis [AU]", fontsize=fontsize)
    
    # Title
    population_label = "archive-conditional" if table_type == "Archive" else "population-conditional"
    fig.suptitle(rf"ELT/{instru} | {post_processing} | " + BAND_FIXED.replace("_100", "").replace("_", " ") + rf" | {CONFIG_FIXED}" + "\n" + rf"{population_label} $P_{{det}}={100*P_LEVEL:.0f}\%$ contours " + rf"($S/N \geq {SNR_thr:g}$, {exposure_time/60:.0f} h)", fontsize=fontsize + 10, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # %%
    # ============================================================================
    # PHYSICAL SPACE + REFERENCE-WFE / PCS-LIKE-WFE PROBABILITY CONTOURS
    #
    # The "PCS" contour is NOT a separate PCS simulation:
    # it is the same instrument/configuration, with WFE fixed to 50 nm RMS.
    # ============================================================================
    # ----------------------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------------------
    WFE_PCS   = 50.0  # [nm RMS]
    PHYS_XLIM = (10**a_lim[0], 10**a_lim[1])
    PHYS_YLIM = (10**m_lim[0], 10**m_lim[1])
    SS_POP    = 90
    ALPHA_POP = 0.50
    
    # ----------------------------------------------------------------------------
    # WFE indices
    # ----------------------------------------------------------------------------
    WFE_arr        = np.asarray(WFE, dtype=float)
    j_wfe_ref      = int(np.nanargmin(np.abs(WFE_arr - WFE_ref)))
    j_wfe_pcs      = int(np.nanargmin(np.abs(WFE_arr - WFE_PCS)))
    wfe_ref_actual = WFE_arr[j_wfe_ref]
    wfe_pcs_actual = WFE_arr[j_wfe_pcs]
    if not np.isclose(wfe_pcs_actual, WFE_PCS):
        print(f"[WARNING] WFE = {WFE_PCS:g} nm is not present in the grid; using nearest value {wfe_pcs_actual:g} nm.")
    # ----------------------------------------------------------------------------
    # Detection masks and local detection fractions
    # ----------------------------------------------------------------------------
    snr_wfe_ref        = np.asarray(SNR_4D_WFE[:, ib, ic, j_wfe_ref], dtype=float)
    snr_wfe_pcs        = np.asarray(SNR_4D_WFE[:, ib, ic, j_wfe_pcs], dtype=float)
    det_wfe_ref        = np.nan_to_num(snr_wfe_ref, nan=-np.inf) >= SNR_thr
    det_wfe_pcs        = np.nan_to_num(snr_wfe_pcs, nan=-np.inf) >= SNR_thr
    P_wfe_ref          = local_detection_fraction(det_wfe_ref)
    P_wfe_pcs          = local_detection_fraction(det_wfe_pcs)
    P_ratio            = np.full_like(P_wfe_ref, np.nan)
    mask_gain          = np.isfinite(P_wfe_ref) & np.isfinite(P_wfe_pcs) & (P_wfe_ref > 0)
    P_ratio[mask_gain] = P_wfe_pcs[mask_gain] / P_wfe_ref[mask_gain]
    
    # ----------------------------------------------------------------------------
    # Physical-space population data
    # ----------------------------------------------------------------------------
    def _arr(colname):
        col = planet_table[colname]
        return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)
    SMA_phys        = _arr("SMA")
    PlanetMass_phys = _arr("PlanetMass")
    PlanetTeff_phys = _arr("PlanetTeff")
    valid_phys      = (np.isfinite(SMA_phys) & (SMA_phys > 0) & np.isfinite(PlanetMass_phys) & (PlanetMass_phys > 0) & np.isfinite(PlanetTeff_phys) & (PlanetTeff_phys > 0))
    
    # ----------------------------------------------------------------------------
    # Discovery methods: exactly the same logic as yield_population_plot
    # ----------------------------------------------------------------------------
    detection_methods_plot   = ["Transit", "Radial Velocity", "Imaging", "Other"]
    marker_detection_methods = {"Transit": "v", "Radial Velocity": "o", "Imaging": "s", "Other": "P"}
    label_detection_methods  = {"Transit": "Transit", "Radial Velocity": "Radial velocity", "Imaging": "Direct imaging", "Other": "Other"}
    if "DiscoveryMethod" in planet_table.colnames:
        discovery_method       = np.asarray(planet_table["DiscoveryMethod"], dtype=str)
        masks_detection_method = {"Transit": discovery_method == "Transit", "Radial Velocity": discovery_method == "Radial Velocity", "Imaging": discovery_method == "Imaging", "Other": ~np.isin(discovery_method, ["Transit", "Radial Velocity", "Imaging"])}
    else:
        detection_methods_plot   = ["Other"]
        marker_detection_methods = {"Other": "o"}
        label_detection_methods  = {"Other": "All planets"}
        masks_detection_method   = {"Other": np.ones(len(planet_table), dtype=bool)}
    # ----------------------------------------------------------------------------
    # Planet-temperature colour scale
    # ----------------------------------------------------------------------------
    cmap_planet_teff = plt.get_cmap("plasma")
    T_valid          = PlanetTeff_phys[valid_phys]
    norm_planet_teff = mpl.colors.Normalize(vmin=np.nanmin(T_valid), vmax=3000.0, clip=True)
    # ----------------------------------------------------------------------------
    # Planet-type regions
    # ----------------------------------------------------------------------------
    ptypes_plot                 = ["Jupiter", "Saturn", "Neptune", "Earth"]
    ptype_region_colors         = {"Earth": "tab:green", "Neptune": "tab:blue", "Saturn": "tab:orange", "Jupiter": "tab:red"}
    ptype_region_labels         = {"Earth": "Earth-like", "Neptune": "Neptune-like", "Saturn": "Saturn-like", "Jupiter": "Jupiter-like"}
    ptype_region_text_positions = {"Earth": (None, 7e-1), "Neptune": (None, 2e1), "Saturn": (None, None), "Jupiter": (None, None)}
    
    def add_ptype_regions_phys(ax, alpha=1.0):
        for ptype in ptypes_plot:
            mask = np.asarray(get_mask_planet_type(planet_table=planet_table, planet_type=ptype), dtype=bool)
            mask &= (np.isfinite(SMA_phys) & (SMA_phys > 0) & np.isfinite(PlanetMass_phys) & (PlanetMass_phys > 0))
            if np.sum(mask) < 2:
                continue
            xmin, xmax = 10**np.nanpercentile(np.log10(SMA_phys[mask]), [2, 98])
            ymin, ymax = 10**np.nanpercentile(np.log10(PlanetMass_phys[mask]), [2, 98])
            dx         = max(np.log10(xmax / xmin), 0.1)
            dy         = max(np.log10(ymax / ymin), 0.1)
            xmin       = xmin / 10**(0.05 * dx)
            xmax       = xmax * 10**(0.05 * dx)
            ymin       = ymin / 10**(0.08 * dy)
            ymax       = ymax * 10**(0.08 * dy)
            color      = ptype_region_colors[ptype]
            label      = ptype_region_labels[ptype]
            text_pos   = ptype_region_text_positions[ptype]
            xtext      = np.sqrt(xmin * xmax) if text_pos[0] is None else text_pos[0]
            ytext      = np.sqrt(ymin * ymax) if text_pos[1] is None else text_pos[1]
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=mpl.colors.to_rgba(color, 0.25*alpha), edgecolor=mpl.colors.to_rgba(color, 0.50*alpha), linewidth=3.0*alpha, zorder=100))
            ax.text(xtext, ytext, label, color=color, fontsize=fontsize - 1, fontweight="bold", ha="center", va="center", zorder=106, bbox=dict(boxstyle="round,pad=0.25", facecolor=mpl.colors.to_rgba("white", 0.92), edgecolor=mpl.colors.to_rgba(color, 0.95), linewidth=2.0, alpha=alpha), alpha=alpha)
    
    # ----------------------------------------------------------------------------
    # Helper: draw a probability contour with a white halo
    # ----------------------------------------------------------------------------
    def draw_probability_contour(ax, P, color, linestyle, linewidth=4.5):
        finite = np.isfinite(P)
        if not np.any(finite) or np.nanmin(P) > P_LEVEL or np.nanmax(P) < P_LEVEL:
            return None
        ax.contour(10**xa, 10**ym if MASS_UNIT == "Mearth" else 317.8284065946748 * 10**ym, P, levels=[P_LEVEL], colors="white", linewidths=linewidth + 3.0, linestyles=linestyle, zorder=200.0)
        return ax.contour(10**xa, 10**ym if MASS_UNIT == "Mearth" else 317.8284065946748 * 10**ym, P, levels=[P_LEVEL], colors=color, linewidths=linewidth, linestyles=linestyle, zorder=200.1)
    # ----------------------------------------------------------------------------
    # Helper: directly label one contour
    # ----------------------------------------------------------------------------
    def label_probability_contour(ax, cs, label, color, frac=0.55, offset=(12, 12)):
        if cs is None:
            return
        segments = cs.allsegs[0]
        if len(segments) == 0:
            return
        seg = max(segments, key=lambda x: x.shape[0])
        if len(seg) == 0:
            return
        i      = int(np.clip(frac * (len(seg) - 1), 0, len(seg) - 1))
        x0, y0 = seg[i]
        ax.annotate(label, xy=(x0, y0), xytext=offset, textcoords="offset points", fontsize=fontsize, fontweight="bold", color=color, ha="left", va="bottom", zorder=210, bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=color, linewidth=2.0, alpha=0.95))
    
    # ============================================================================
    # Figure
    # ============================================================================
    fig_phys, axes   = plt.subplots(1, 2, figsize=(22, 9), dpi=dpi, sharex=True, sharey=True)
    ax_phys, ax_gain = axes

    # Left panel: physical space + probability contours
    add_ptype_regions_phys(ax_phys)
    for method in detection_methods_plot:
        mask = valid_phys & masks_detection_method[method]
        if not np.any(mask):
            continue
        ax_phys.scatter(SMA_phys[mask], PlanetMass_phys[mask], c=PlanetTeff_phys[mask], cmap=cmap_planet_teff, norm=norm_planet_teff, marker=marker_detection_methods[method], s=SS_POP, edgecolors="k", linewidths=0.35, alpha=ALPHA_POP, zorder=10)
    color_ref = "k"
    color_pcs = "k"
    cs_ref    = draw_probability_contour(ax=ax_phys, P=P_wfe_ref, color=color_ref, linestyle="-")
    cs_pcs    = draw_probability_contour(ax=ax_phys, P=P_wfe_pcs, color=color_pcs, linestyle="--")
    label_probability_contour(ax_phys, cs_ref, label=f"{instru}", color=color_ref, frac=0.40, offset=(14, 14))
    label_probability_contour(ax_phys, cs_pcs, label="PCS", color=color_pcs, frac=0.67, offset=(14, -28))
    ax_phys.set_xscale("log")
    ax_phys.set_yscale("log")
    ax_phys.set_xlim(*PHYS_XLIM)
    ax_phys.set_ylim(*PHYS_YLIM)
    ax_phys.set_xlabel("Semi-major axis [AU]", fontsize=fontsize + 2, labelpad=10)
    ax_phys.set_ylabel(r"Planet mass [$M_\oplus$]", fontsize=fontsize + 2, labelpad=10)
    ax_phys.set_title("Detection probability", fontsize=fontsize + 2, fontweight="bold")
    ax_phys.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-1000)
    ax_phys.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=-1000)
    ax_phys.tick_params(axis="both", which="major", labelsize=fontsize - 2, length=6, width=1.0)
    ax_phys.tick_params(axis="both", which="minor", labelsize=fontsize - 2, length=3, width=0.8)
    handles_method = [Line2D([], [], ls="", marker=marker_detection_methods[method], ms=13, markerfacecolor="white", markeredgecolor="k", markeredgewidth=1.5, color="k", label=label_detection_methods[method]) for method in detection_methods_plot]
    legend_methods = ax_phys.legend(handles=handles_method, fontsize=fontsize - 4, loc="lower right", frameon=True, edgecolor="gray", facecolor="white", title="Discovery method", title_fontsize=fontsize)
    legend_methods.set_zorder(10_000)
    ax_phys.add_artist(legend_methods)
    sm = mpl.cm.ScalarMappable(norm=norm_planet_teff, cmap=cmap_planet_teff)
    sm.set_array([])
    cbar = fig_phys.colorbar(sm, ax=ax_phys, pad=0.03, fraction=0.045)
    cbar.set_label(r"Planet $T_\mathrm{eff}$ [K]", fontsize=fontsize + 2, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    # Right panel: probability-ratio map
    P_ratio_plot = np.ma.masked_invalid(P_ratio)
    finite_gain  = np.isfinite(P_ratio)
    if not np.any(finite_gain):
        raise RuntimeError("No finite probability-ratio value to display.")
    gain_min  = 1.0
    gain_max  = np.nanmax(P_ratio[finite_gain])
    gain_cmap = plt.get_cmap("magma")
    gain_norm = mpl.colors.LogNorm(vmin=gain_min, vmax=gain_max)
    
    # Nice logarithmic levels
    gain_levels = np.geomspace(gain_min, gain_max, 20)
    cf          = ax_gain.contourf(10**xa, 10**ym if MASS_UNIT == "Mearth" else 317.83 * 10**ym, P_ratio_plot, levels=gain_levels, cmap=gain_cmap, norm=gain_norm)
    add_ptype_regions_phys(ax_gain, alpha=0.5)
    ax_gain.set_xscale("log")
    ax_gain.set_yscale("log")
    ax_gain.set_xlim(*PHYS_XLIM)
    ax_gain.set_ylim(*PHYS_YLIM)
    ax_gain.set_xlabel("Semi-major axis [AU]", fontsize=fontsize + 2, labelpad=10)
    ax_gain.set_title("Detection-probability gain", fontsize=fontsize + 2, fontweight="bold")
    ax_gain.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=1000)
    ax_gain.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=1000)
    ax_gain.tick_params(axis="both", which="major", labelsize=fontsize - 2, length=6, width=1.0)
    ax_gain.tick_params(axis="both", which="minor", labelsize=fontsize - 2, length=3, width=0.8)
    
    # Rounded / meaningful colorbar ticks
    candidate_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
    gain_ticks = candidate_ticks[(candidate_ticks >= gain_min) & (candidate_ticks <= gain_max)]
    
    cbar_gain = fig_phys.colorbar(cf, ax=ax_gain, pad=0.03, fraction=0.045, ticks=gain_ticks)
    cbar_gain.ax.set_yticklabels([rf"$\times${t:g}" for t in gain_ticks])
    cbar_gain.set_label(r"Detection-probability ratio $P_{\rm det,PCS}/P_{\rm det,ref}$", fontsize=fontsize + 2, rotation=270, labelpad=30)
    cbar_gain.ax.tick_params(labelsize=fontsize - 2)
    
    # Global title
    fig_phys.suptitle(rf"ELT/{instru} | {post_processing} | " + BAND_FIXED.replace("_100", "").replace("_", " ") + rf" | {CONFIG_FIXED}" + "\n" + rf"{population_label} $P_{{det}}={100*P_LEVEL:.0f}\%$ contours and WFE gain " + rf"($S/N \geq {SNR_thr:g}$, {exposure_time/60:.0f} h)", fontsize=fontsize+3, weight="bold", y=0.99)
    fig_phys.tight_layout()
    plt.show()
    
    
    # %%
    # ============================================================================
    # EXPOSURE-TIME PERFORMANCE
    #
    # Figure 1:
    #   Exposure-time gain at fixed detected fraction:
    #
    #       G_t = T_fraction,ref / T_fraction
    #
    # Figure 2:
    #   Fraction of detected targets as a function of instrumental parameter
    #   and exposure time, for one fixed band/configuration.
    # ============================================================================

    # --- User parameters ---
    fraction = 0.10  # fixed fraction of the population to detect
    t_grid   = np.geomspace(1e-2, 1e4, 250) # [h]

    # --- Exposure times to reach SNR = SNR_thr [min] ---
    Texp_4D_R, Texp_4D_l0, Texp_4D_Nl, Texp_4D_TI, Texp_4D_SM, Texp_4D_WFE, Texp_4D_IWA = get_exposure_time(
        min_DIT, DC0, RON0, RON_lim, A_FWHM, NbPixel, SNR_thr, idx_R_ref, sigma_m, sigma_m_ref,
        signal_4D_R,   sigma_halo_2_4D_R,   sigma_syst_base_2_4D_R,   sigma_bkg_2_4D_R,   DIT_4D_R,
        signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_syst_base_2_4D_l0,  sigma_bkg_2_4D_l0,  DIT_4D_l0,
        signal_4D_Nl,  sigma_halo_2_4D_Nl,  sigma_syst_base_2_4D_Nl,  sigma_bkg_2_4D_Nl,  DIT_4D_Nl,
        signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_syst_base_2_4D_TI,  sigma_bkg_2_4D_TI,  DIT_4D_TI,
        signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_syst_base_2_4D_WFE, sigma_bkg_2_4D_WFE, DIT_4D_WFE,
        signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_syst_base_2_4D_IWA, sigma_bkg_2_4D_IWA, DIT_4D_IWA)

    # --- FoV exposure-time axis ---
    # Same convention as get_Pdet:
    # reference instrumental performance, with a target unreachable if separation > FoV/2.
    idx_ref_R_4D   = np.broadcast_to(idx_R_ref[None, :, None, None], (N_PT, N_band, N_config, 1))
    Texp_3D_ref    = np.take_along_axis(Texp_4D_R, idx_ref_R_4D, axis=3)[..., 0] # (planet, band, config)
    inside_FoV_2D  = separation_planets[:, None] <= FoV[None, :] / 2
    Texp_4D_FoV    = np.where(inside_FoV_2D[:, None, None, :], Texp_3D_ref[..., None], np.inf)

    # --- Exposure-time arrays corresponding exactly to the original params order ---
    Texp_all = [Texp_4D_R, Texp_4D_l0, Texp_4D_Nl, Texp_4D_TI, Texp_4D_SM, Texp_4D_FoV, Texp_4D_WFE, Texp_4D_IWA]
    Texp_plot = [Texp_all[idim] for idim in dim_to_keep]
    if len(Texp_plot) != Ndim:
        raise RuntimeError(f"Texp/params dimension mismatch: {len(Texp_plot)} != {Ndim}")

    # --- Fixed population for every value of every instrumental parameter ---
    mask_targets = np.asarray(mask_regime_plot, dtype=bool)
    N_targets    = int(np.sum(mask_targets))
    if N_targets == 0:
        raise RuntimeError(f"No target available for light_regime_plot='{light_regime_plot}'.")

    # --- Helpers ---
    def time_to_fraction(T, fraction):
        """Exposure time required to detect a fixed fraction of a fixed target population."""
        if not (0 < fraction <= 1):
            raise ValueError("fraction must satisfy 0 < fraction <= 1.")
        T_sorted = np.sort(T, axis=0)
        k        = max(0, int(np.ceil(fraction * T.shape[0])) - 1)
        return T_sorted[k]

    def get_ref_index_Texp(dim, ib, ic):
        """Reference index along the parameter axis for a given band/config."""
        name = params_names[dim]
        if name == "R":
            return int(idx_R_ref[ib])
        if name == "l0 [µm]":
            return int(idx_l0_ref[ib])
        if name == "Nl":
            return int(idx_Nl_ref)
        if name == "trans_instru [e-/ph]":
            return int(idx_TI_ref[ib, ic])
        if name == "sigma_m [%]":
            return int(np.nanargmin(np.abs(sigma_m - sigma_m_ref)))
        if name == "FoV [mas]":
            FoV_ref_ib = np.asarray(FoV_ref).ravel()[ib] if np.asarray(FoV_ref).size > 1 else float(np.asarray(FoV_ref).ravel()[0])
            return int(np.nanargmin(np.abs(FoV - FoV_ref_ib)))
        if name == "WFE [nm RMS]":
            return int(idx_WFE_ref)
        if name == "IWA [mas]":
            return int(idx_IWA_ref)
        raise ValueError(f"Unknown parameter: {name}")

    def get_T_for_plot(dim, ib, ic):
        """Target x parameter exposure-time matrix in hours, with all unreachable targets = inf."""
        T = np.asarray(Texp_plot[dim][:, ib, ic, :], dtype=float)[mask_targets] / 60 # [h]
        return np.where(np.isfinite(T), T, np.inf)

    # ============================================================================
    # COMPUTE FIXED-FRACTION EXPOSURE-TIME GAINS
    # ============================================================================
    gain_T_plot = []
    T_fraction_plot = []
    for dim in range(Ndim):
        N_param        = len(params[dim])
        gain_dim       = np.full((N_band, N_config, N_param), np.nan, dtype=float)
        T_fraction_dim = np.full((N_band, N_config, N_param), np.inf, dtype=float)
        for ib, band in enumerate(bands):
            for ic, config in enumerate(configs):
                T                  = get_T_for_plot(dim, ib, ic)
                T_fraction        = time_to_fraction(T, fraction)
                iref              = get_ref_index_Texp(dim, ib, ic)
                T_fraction_ref    = T_fraction[iref]
                T_fraction_dim[ib, ic] = T_fraction
                if np.isfinite(T_fraction_ref) and T_fraction_ref > 0:
                    gain = np.divide(T_fraction_ref, T_fraction, out=np.full_like(T_fraction, np.nan), where=np.isfinite(T_fraction) & (T_fraction > 0))
                    gain_dim[ib, ic] = gain
        gain_T_plot.append(gain_dim)
        T_fraction_plot.append(T_fraction_dim)

    # ============================================================================
    # FIGURE 1 — EXPOSURE-TIME GAIN AT FIXED DETECTED FRACTION
    # ============================================================================
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows), dpi=dpi, sharey=True)
    axes      = np.atleast_2d(axes)
    for dim in range(Ndim):
        r  = dim // ncols
        t  = dim % ncols
        ax = axes[r, t]
        ax.tick_params(axis="both", which="major", labelsize=fontsize+2)
        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

        # --- Plot ---
        for ib, band in enumerate(bands):
            for ic, config in enumerate(configs):
                ax.plot(params[dim], gain_T_plot[dim][ib, ic], ls=ls_configs[ic], lw=lw, c=cmap_bands(ib), alpha=alpha, zorder=3)
        ax.axhline(1.0, ls="--", lw=lw, c="0.35", alpha=0.8, zorder=1)

        # --- Reference values ---
        if len(params[dim]) == N_R and np.all(params[dim] == R):
            if instru == "HARMONI":
                n    = len(bands)
                db   = (n - 1) / 2.0
                dlog = 0.015
                for ib, band in enumerate(bands):
                    x0 = R_ref[ib]
                    x  = x0 * 10**(dlog * (ib - db))
                    ax.axvline(x, ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)
            elif instru == "ANDES":
                ax.axvline(np.nanmean(R_ref), ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
        elif len(params[dim]) == len(l0) and np.all(params[dim] == l0):
            if instru == "HARMONI":
                for ib, band in enumerate(bands):
                    ax.axvline(l0_ref[ib], ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)
                    ax.axvspan(lmin_bands[ib], lmax_bands[ib], color=cmap_bands(ib), alpha=0.1, zorder=-10)
            elif instru == "ANDES":
                ax.axvline(np.nanmean(l0_ref), ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
                ax.axvspan(np.nanmean(lmin_bands), np.nanmean(lmax_bands), color="black", alpha=0.1, zorder=-10)
        elif len(params[dim]) == len(Nl) and np.all(params[dim] == Nl):
            ax.axvline(Nl_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
        elif len(params[dim]) == len(trans_instru) and np.all(params[dim] == trans_instru):
            ax.set_xlim(0, trans_instru_max)
            if instru == "HARMONI":
                ic = 0
                for ib, band in enumerate(bands):
                    ax.axvline(trans_instru_ref[ib, ic], ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)
            elif instru == "ANDES":
                ax.axvline(trans_instru_ref[-1, -1], ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
        elif len(params[dim]) == len(FoV) and np.all(params[dim] == FoV):
            if instru == "HARMONI":
                ax.axvline(FoV_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
            elif instru == "ANDES":
                for ib, band in enumerate(bands):
                    ax.axvline(FoV_ref[ib], ls="-", lw=2*lw, c=cmap_bands(ib), alpha=alpha, zorder=0)
        elif len(params[dim]) == len(WFE) and np.all(params[dim] == WFE):
            ax.axvline(WFE_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)
        elif len(params[dim]) == len(IWA) and np.all(params[dim] == IWA):
            ax.axvline(IWA_ref, ls="-", lw=2*lw, c="black", alpha=alpha, zorder=0)

        # --- X scale + minor ticks ---
        if len(params[dim]) == len(trans_instru) and np.all(params[dim] == trans_instru):
            ax.set_xlim(0, trans_instru_max)
        else:
            ax.set_xlim(np.nanmin(params[dim]), np.nanmax(params[dim]))
        if params_islog[dim]:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # --- Labels ---
        ax.set_xlabel(params_names_L[dim], fontsize=fontsize+4)
        if t == 0:
            ax.set_ylabel("Exposure-time gain" + "\n" + rf"$T_{{{100*fraction:.0f}\%,\rm ref}}/T_{{{100*fraction:.0f}\%}}$", fontsize=fontsize+4)
        else:
            ax.tick_params(labelleft=False)

        # --- Legends ---
        if dim == 0:
            if instru == "HARMONI":
                handles_band = [Line2D([], [], ls="-", lw=lw, color=cmap_bands(ib), label=band.replace("_", " ")) for ib, band in enumerate(bands)]
                leg_band     = ax.legend(handles=handles_band, fontsize=fontsize+2, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Bands", title_fontsize=fontsize+2)
                ax.add_artist(leg_band)
            elif instru == "ANDES":
                handles_pxscale = [Line2D([], [], ls="-", lw=lw, color=cmap_bands(ib), label=f"{pxscales[ib]:.0f} mas") for ib, band in enumerate(bands)]
                leg_pxscale     = ax.legend(handles=handles_pxscale, fontsize=fontsize+2, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Fiber scales", title_fontsize=fontsize+2)
                ax.add_artist(leg_pxscale)
        if dim == 2 and len(configs) > 1:
            handles_config = [Line2D([], [], ls=ls_configs[ic], lw=lw, color="k", label=str(config).replace("_", " ").replace("LYOT", "Lyot")) for ic, config in enumerate(configs)]
            leg_config     = ax.legend(handles=handles_config, fontsize=fontsize+2, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Apodizer config" if w_apodizer else "Coronagraph config", title_fontsize=fontsize+2)
            ax.add_artist(leg_config)
        if dim == 3:
            handles_ref = [Line2D([], [], ls="None", marker="|", markersize=30, markeredgewidth=3, color="k", label="Reference values")]
            leg_ref     = ax.legend(handles=handles_ref, fontsize=fontsize+3, loc="lower right", frameon=True, edgecolor="gray", facecolor="white")
            ax.add_artist(leg_ref)

        # --- Systematic budget markers ---
        if post_processing == "MM" and params_names[dim] == "sigma_m [%]":
            x0 = 1.0
            ax.axvline(x0, c="k", ls=":", lw=lw)
            ax.annotate("JWST/MIRI/MRS", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)
        elif post_processing == "DI" and params_names[dim] == "sigma_m [%]":
            x0 = 0.1
            ax.axvline(x0, c="k", ls="--", lw=lw)
            ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+4, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=5, clip_on=True)

    # --- Turn off unused axes ---
    for k in range(Ndim, nrows*ncols):
        axes[k // ncols, k % ncols].axis("off")

    # --- Shared Y scale / limits ---
    finite_gains = np.concatenate([g[np.isfinite(g) & (g > 0)] for g in gain_T_plot if np.any(np.isfinite(g) & (g > 0))])
    gain_ymin    = min(1.0, np.nanmin(finite_gains))
    gain_ymax    = max(1.0, np.nanmax(finite_gains))
    for ax_i in axes.ravel()[:Ndim]:
        ax_i.set_yscale("log")
        ax_i.set_ylim(1e-2, 1e2)

    # --- Title ---
    title = f"ELT/{instru} | {post_processing} | {regime_label.upper()} LIGHT" + "\n"
    title += rf"Exposure-time gain at fixed detected fraction = {100*fraction:.0f}%"
    fig.suptitle(title, fontsize=fontsize+10, weight="bold", y=0.94)
    fig.tight_layout(h_pad=4.0, w_pad=2.0, rect=[0, 0, 1, 0.91])
    plt.show()

    # ============================================================================
    # FIGURE 2 — DETECTED FRACTION VS PARAMETER AND EXPOSURE TIME
    # Fixed band and configuration; one independent colorbar per panel.
    # ============================================================================
    if BAND_FIXED not in bands:
        raise ValueError(f"BAND_FIXED={BAND_FIXED!r} not available. Available bands: {bands}")
    if CONFIG_FIXED not in configs:
        raise ValueError(f"CONFIG_FIXED={CONFIG_FIXED!r} not available. Available configs: {configs}")
    ib_fixed = bands.index(BAND_FIXED)
    ic_fixed = configs.index(CONFIG_FIXED)

    fig, axes = plt.subplots(nrows, ncols, figsize=(9*ncols, 6*nrows), dpi=dpi, sharey=True)
    axes      = np.atleast_2d(axes)
    for dim in range(Ndim):
        r  = dim // ncols
        t  = dim % ncols
        ax = axes[r, t]
        ax.tick_params(axis="both", which="major", labelsize=fontsize+2)
        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

        # --- Fixed population / fixed band / fixed configuration ---
        T = get_T_for_plot(dim, ib_fixed, ic_fixed) # (targets, parameter) [h]

        # Fraction of the SAME target population detected within each exposure time
        completeness = np.mean(T[:, :, None] <= t_grid[None, None, :], axis=0) # (parameter, time)
        C            = 100 * completeness.T                                    # (time, parameter)

        # --- Independent color scale for this panel ---
        cmin = float(np.nanmin(C))
        cmax = float(np.nanmax(C))
        if not np.isfinite(cmin) or not np.isfinite(cmax):
            raise RuntimeError(f"No finite completeness for parameter {params_names[dim]}.")
        if np.isclose(cmin, cmax):
            cmin = max(0.0, cmin - 0.5)
            cmax = min(100.0, cmax + 0.5)
            if np.isclose(cmin, cmax):
                cmax = cmin + 1.0
        levels = np.linspace(cmin, cmax, 21)

        # --- 2D map ---
        cf = ax.contourf(params[dim], t_grid, C, levels=levels, cmap="viridis")

        # Contour corresponding exactly to the fraction used in Figure 1
        target_level = 100 * fraction
        if cmin <= target_level <= cmax:
            cs = ax.contour(params[dim], t_grid, C, levels=[target_level], colors="white", linewidths=2.5, zorder=5)
            ax.clabel(cs, fmt={target_level: f"{target_level:.0f}%"}, inline=True, fontsize=fontsize-1)

        # --- Reference value for the fixed band/config ---
        if params_names[dim] != "sigma_m [%]":
            iref = get_ref_index_Texp(dim, ib_fixed, ic_fixed)
            ax.axvline(params[dim][iref], ls="-", lw=2*lw, c="black", alpha=alpha, zorder=6)

        # Band extent for lambda_0
        if len(params[dim]) == len(l0) and np.all(params[dim] == l0):
            ax.axvspan(lmin_bands[ib_fixed], lmax_bands[ib_fixed], color="black", alpha=0.08, zorder=4)

        # --- X scale + minor ticks ---
        if len(params[dim]) == len(trans_instru) and np.all(params[dim] == trans_instru):
            ax.set_xlim(0, trans_instru_max)
        else:
            ax.set_xlim(np.nanmin(params[dim]), np.nanmax(params[dim]))
        if params_islog[dim]:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # --- Y scale ---
        ax.set_yscale("log")
        ax.set_ylim(t_grid[0], t_grid[-1])

        # --- Labels ---
        ax.set_xlabel(params_names_L[dim], fontsize=fontsize+4)
        if t == 0:
            ax.set_ylabel("Exposure time [h]", fontsize=fontsize+4)
        else:
            ax.tick_params(labelleft=False)

        # --- Independent colorbar ---
        step     = 5
        tick_min = step * np.floor(cmin / step)
        tick_max = step * np.ceil(cmax / step)
        ticks    = np.arange(tick_min, tick_max + step, step)
        cbar     = fig.colorbar(cf, ax=ax, pad=0.02, ticks=ticks)
        cbar.set_label("Detected targets [%]", fontsize=fontsize+2, rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=fontsize)

        # --- Systematic budget markers ---
        if post_processing == "MM" and params_names[dim] == "sigma_m [%]":
            x0 = 1.0
            ax.axvline(x0, c="k", ls=":", lw=3*lw, zorder=7)
            ax.annotate("JWST/MIRI/MRS", xy=(1.1*x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+3, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=8, clip_on=True)
        elif post_processing == "DI" and params_names[dim] == "sigma_m [%]":
            x0 = 0.1
            ax.axvline(x0, c="k", ls="--", lw=lw, zorder=7)
            ax.annotate("Optimistic on-sky DI", xy=(x0, 0.5), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points", rotation=270, va="center", ha="left", fontsize=fontsize+3, color="k", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85), zorder=8, clip_on=True)

        # Reference-value legend once
        if dim == 0:
            handles_ref = [Line2D([], [], ls="None", marker="|", markersize=30, markeredgewidth=3, color="k", label="Reference value")]
            leg_ref     = ax.legend(handles=handles_ref, fontsize=fontsize+2, loc="lower left", frameon=True, edgecolor="gray", facecolor="white")
            ax.add_artist(leg_ref)

    # --- Turn off unused axes ---
    for k in range(Ndim, nrows*ncols):
        axes[k // ncols, k % ncols].axis("off")

    # --- Title ---
    band_title   = str(BAND_FIXED).replace("_", " ")
    config_title = str(CONFIG_FIXED).replace("_", " ").replace("LYOT", "Lyot")
    title = f"ELT/{instru} | {post_processing} | {regime_label.upper()} LIGHT"
    title += f"\nDetected fraction vs exposure time | {band_title} | {config_title}"
    fig.suptitle(title, fontsize=fontsize+10, weight="bold", y=0.94)
    fig.tight_layout(h_pad=4.0, w_pad=2.0, rect=[0, 0, 1, 0.91])
    plt.show()
    
    # ============================================================================
    # FIGURE 3 — PHYSICAL SPACE: reference T_exp and relative PCS T_exp
    # ============================================================================
    WFE_PCS            = 50.0   # PCS-like post-AO WFE [nm RMS]
    N_GRID_MS          = 25     # Number of Mass-SMA grid points per axis
    KNN_K_MS           = 120    # Number of nearest planets used for the local estimate
    MIN_NEFF_MS        = 20     # Minimum effective local sample size required
    KERNEL_SCALE_MS    = 0.55   # Gaussian width / distance to the KNN_K_MS-th neighbour
    SUPPORT_Q_MS       = 0.98   # Quantile used to define the characteristic sampling-support radius
    SUPPORT_FACTOR_MS  = 1.35   # Extra tolerance applied to the sampling-support radius
    SS_POP             = 180    # Planet scatter-marker size
    ALPHA_POP          = 0.90   # Planet scatter transparency
    TEXP_HIGHLIGHT     = 50.0   # Exposure time below which targets are considered observationally attractive [h]
    TEXP_HIGHLIGHT_POS = 0.30   # Position of 50 h in the colormap (< 0.5 keeps Texp <= 50 h in the bright part)
    CMAP_TEXP          = plt.get_cmap("viridis_r")
    CMAP_GAIN_TEXP     = plt.get_cmap("magma")
    
    def arr(colname):
        col = planet_table[colname]
        return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)
    
    def make_texp_norm(vmin, vmax, t_break=TEXP_HIGHLIGHT, break_pos=TEXP_HIGHLIGHT_POS):
        """Piecewise-log normalization emphasizing exposure times below t_break."""
        if not (vmin < t_break < vmax):
            return LogNorm(vmin=vmin, vmax=vmax)
        log_min   = np.log10(vmin)
        log_break = np.log10(t_break)
        log_max   = np.log10(vmax)
        def forward(x):
            x    = np.asarray(x, dtype=float)
            logx = np.log10(x)
            return np.where(logx <= log_break, break_pos * (logx - log_min) / (log_break - log_min), break_pos + (1.0 - break_pos) * (logx - log_break) / (log_max - log_break))
        def inverse(y):
            y    = np.asarray(y, dtype=float)
            logx = np.where(y <= break_pos, log_min + (log_break - log_min) * y / break_pos, log_break + (log_max - log_break) * (y - break_pos) / (1.0 - break_pos))
            return 10**logx
        return mpl.colors.FuncNorm((forward, inverse), vmin=vmin, vmax=vmax, clip=True)
    
    SMA_phys        = arr("SMA")
    PlanetMass_phys = arr("PlanetMass")
    PlanetTeff_phys = arr("PlanetTeff")
    
    if "PHYS_XLIM" not in locals():
        PHYS_XLIM = (5e-3, 1e3)
    if "PHYS_YLIM" not in locals():
        PHYS_YLIM = (5e-1, 1e4)
    
    # --- Reference and PCS-like exposure times [h] for the fixed band/config ---
    idx_WFE_pcs = int(np.nanargmin(np.abs(np.asarray(WFE, dtype=float) - WFE_PCS)))
    Texp_ref    = np.asarray(Texp_4D_WFE[:, ib_fixed, ic_fixed, idx_WFE_ref], dtype=float) / 60.0
    Texp_pcs    = np.asarray(Texp_4D_WFE[:, ib_fixed, ic_fixed, idx_WFE_pcs], dtype=float) / 60.0
    
    reachable_ref  = np.isfinite(Texp_ref) & (Texp_ref > 0)
    reachable_pcs  = np.isfinite(Texp_pcs) & (Texp_pcs > 0)
    reachable_both = reachable_ref & reachable_pcs
    
    # --- Per-planet exposure-time gain ---
    # T_ref / T_PCS:
    #   = 1 : no gain
    #   = 2 : PCS requires 2 times less exposure time
    #   = 5 : PCS requires 5 times less exposure time
    #   = 10: PCS requires 10 times less exposure time
    gain_texp_planet = np.full(len(planet_table), np.nan, dtype=float)
    gain_texp_planet[reachable_both] = Texp_ref[reachable_both] / Texp_pcs[reachable_both]
    gain_texp_planet[~(gain_texp_planet > 0)] = np.nan
        
    # --- Valid masks for scatter plot ---
    valid_phys_texp = np.asarray(mask_regime_plot, dtype=bool) & np.isfinite(SMA_phys) & (SMA_phys > 0) & np.isfinite(PlanetMass_phys) & (PlanetMass_phys > 0) & np.isfinite(PlanetTeff_phys) & (PlanetTeff_phys > 0)
    
    # --- Build a local Mass-SMA smoothing grid ---
    sample_mask_ms = np.asarray(mask_regime_plot, dtype=bool) & np.isfinite(SMA_phys) & (SMA_phys > 0) & np.isfinite(PlanetMass_phys) & (PlanetMass_phys > 0)
    idx_sample     = np.where(sample_mask_ms)[0]
    if idx_sample.size < max(30, MIN_NEFF_MS + 1):
        raise RuntimeError(f"Only {idx_sample.size} valid planets in the selected population for the Texp physical-space map.")
    
    log_a  = np.log10(SMA_phys[idx_sample])
    log_m  = np.log10(PlanetMass_phys[idx_sample])
    X      = np.column_stack((log_a, log_m))
    center = np.nanmedian(X, axis=0)
    scale  = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    scale  = np.where(np.isfinite(scale) & (scale > 0), scale, np.nanstd(X, axis=0))
    scale  = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    Xs     = (X - center) / scale
    
    xa     = np.linspace(np.log10(PHYS_XLIM[0]), np.log10(PHYS_XLIM[1]), N_GRID_MS)
    ym     = np.linspace(np.log10(PHYS_YLIM[0]), np.log10(PHYS_YLIM[1]), N_GRID_MS)
    XA, YM = np.meshgrid(xa, ym)
    G      = np.column_stack((XA.ravel(), YM.ravel()))
    Gs     = (G - center) / scale
    
    # --- Adaptive Gaussian kernel ---
    tree = cKDTree(Xs)
    k    = min(int(KNN_K_MS), idx_sample.size - 1)
    dist, neigh = tree.query(Gs, k=k)
    if k == 1:
        dist  = dist[:, None]
        neigh = neigh[:, None]
    
    d    = np.maximum(dist[:, -1], np.finfo(float).eps)
    w    = np.exp(-0.5 * (dist / (KERNEL_SCALE_MS * d[:, None]))**2)
    sumw = np.sum(w, axis=1)
    neff = sumw**2 / np.sum(w**2, axis=1)
    
    try:
        inside_hull = Delaunay(Xs).find_simplex(Gs) >= 0
    except QhullError:
        inside_hull = np.ones(Gs.shape[0], dtype=bool)
    
    d_data, _ = tree.query(Xs, k=min(k + 1, idx_sample.size))
    r_support = np.nanquantile(d_data[:, -1], SUPPORT_Q_MS) * SUPPORT_FACTOR_MS
    support   = inside_hull & (d <= r_support) & (neff >= MIN_NEFF_MS)
    
    # --- Local geometric mean ---
    # Appropriate for a multiplicative quantity such as T_ref/T_PCS.
    def local_geometric_mean_map(values_all, min_neff_local=max(8, MIN_NEFF_MS // 2)):
        y          = np.asarray(values_all, dtype=float)[idx_sample]
        good       = np.isfinite(y) & (y > 0)
        y_neigh    = y[neigh]
        good_neigh = good[neigh]
        ww         = w * good_neigh
        sumww      = np.sum(ww, axis=1)
        sumww2     = np.sum(ww**2, axis=1)
        neff_local = np.divide(sumww**2, sumww2, out=np.zeros_like(sumww), where=sumww2 > 0)
        logy       = np.where(good_neigh, np.log10(y_neigh), 0.0)
        Z          = np.full(G.shape[0], np.nan, dtype=float)
        ok         = support & (sumww > 0) & (neff_local >= min_neff_local)
        Z[ok]      = 10**(np.sum(ww[ok] * logy[ok], axis=1) / sumww[ok])
        return Z.reshape(N_GRID_MS, N_GRID_MS)
    
    # --- Local exposure-time gain map ---
    G_texp = local_geometric_mean_map(gain_texp_planet)

    # ============================================================================
    # Intermediate figure — direct comparison of reference and PCS exposure times
    # ============================================================================
    
    # Common color scale so that identical colors correspond to identical exposure times
    finite_texp_ref_compare = Texp_ref[valid_phys_texp & reachable_ref]
    finite_texp_pcs_compare = Texp_pcs[valid_phys_texp & reachable_pcs]
    if finite_texp_ref_compare.size == 0 or finite_texp_pcs_compare.size == 0:
        raise RuntimeError("No finite exposure time available for the reference/PCS comparison.")
    
    finite_texp_compare = np.concatenate((finite_texp_ref_compare, finite_texp_pcs_compare))
    texp_min_compare    = max(np.nanmin(finite_texp_compare), t_grid[0])
    texp_max_compare    = min(np.nanmax(finite_texp_compare), t_grid[-1])
    if not np.isfinite(texp_min_compare) or not np.isfinite(texp_max_compare) or texp_min_compare <= 0 or texp_max_compare <= 0 or texp_min_compare >= texp_max_compare:
        raise RuntimeError("Invalid common exposure-time range for the reference/PCS comparison.")
    norm_texp_compare = make_texp_norm(texp_min_compare, texp_max_compare)
    
    # --- Helper to draw the exact same physical-space representation for both configurations ---
    def plot_texp_population(ax, Texp, reachable, title):
        add_ptype_regions_phys(ax)
        
        # Non-reachable planets shown in light gray
        for method in detection_methods_plot:
            mask = valid_phys_texp & masks_detection_method[method] & (~reachable)
            if not np.any(mask):
                continue
            ax.scatter(SMA_phys[mask], PlanetMass_phys[mask], c="0.80", marker=marker_detection_methods[method], s=SS_POP, edgecolors="none", alpha=0.20, zorder=2)
        
        # Reachable planets colored by the required exposure time
        for method in detection_methods_plot:
            mask = valid_phys_texp & masks_detection_method[method] & reachable
            if not np.any(mask):
                continue
            ax.scatter(SMA_phys[mask], PlanetMass_phys[mask], c=Texp[mask], cmap=CMAP_TEXP, norm=norm_texp_compare, marker=marker_detection_methods[method], s=SS_POP, edgecolors="k", linewidths=0.35, alpha=ALPHA_POP, zorder=10)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*PHYS_XLIM)
        ax.set_ylim(*PHYS_YLIM)
        ax.set_xlabel("Semi-major axis [AU]", fontsize=fontsize + 2, labelpad=10)
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-1000)
        ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=-1000)
        ax.tick_params(axis="both", which="major", labelsize=fontsize - 2, length=6, width=1.0)
        ax.tick_params(axis="both", which="minor", labelsize=fontsize - 2, length=3, width=0.8)
    
    # --- Figure ---
    fig_texp_compare, axes_compare = plt.subplots(1, 2, figsize=(22, 9), dpi=dpi, sharex=True, sharey=True)
    ax_ref_compare, ax_pcs_compare = axes_compare
    
    # Left panel: reference instrumental configuration
    plot_texp_population(ax=ax_ref_compare, Texp=Texp_ref, reachable=reachable_ref, title=rf"Reference exposure time to reach $S/N={SNR_thr:g}$")
    
    # Right panel: PCS-like WFE configuration
    plot_texp_population(ax=ax_pcs_compare, Texp=Texp_pcs, reachable=reachable_pcs, title=rf"PCS-like exposure time to reach $S/N={SNR_thr:g}$")
    
    ax_ref_compare.set_ylabel(r"Planet mass [$M_\oplus$]", fontsize=fontsize + 2, labelpad=10)
    
    # --- Discovery-method legend, shown only once ---
    handles_method = [Line2D([], [], ls="", marker=marker_detection_methods[method], ms=13, markerfacecolor="white", markeredgecolor="k", markeredgewidth=1.5, color="k", label=label_detection_methods[method]) for method in detection_methods_plot]
    legend_methods = ax_ref_compare.legend(handles=handles_method, fontsize=fontsize - 4, loc="lower right", frameon=True, edgecolor="gray", facecolor="white", title="Discovery method", title_fontsize=fontsize)
    legend_methods.set_zorder(10_000)
    ax_ref_compare.add_artist(legend_methods)
    
    # --- Common exposure-time ticks ---
    candidate_ticks_texp_compare = np.array([1e-2, 1e-1, 1, 10, 100, 1e3, 1e4], dtype=float)
    ticks_texp_compare = candidate_ticks_texp_compare[(candidate_ticks_texp_compare >= texp_min_compare) & (candidate_ticks_texp_compare <= texp_max_compare)]
    if ticks_texp_compare.size == 0:
        ticks_texp_compare = np.array([texp_min_compare, texp_max_compare])
    
    # --- Reference colorbar ---
    sm_texp_ref_compare = mpl.cm.ScalarMappable(norm=norm_texp_compare, cmap=CMAP_TEXP)
    sm_texp_ref_compare.set_array([])
    cbar_texp_ref_compare = fig_texp_compare.colorbar(sm_texp_ref_compare, ax=ax_ref_compare, pad=0.03, fraction=0.045, ticks=ticks_texp_compare)
    cbar_texp_ref_compare.ax.set_yticklabels([f"{tick:g}" for tick in ticks_texp_compare])
    cbar_texp_ref_compare.set_label(rf"Exposure time to reach $S/N={SNR_thr:g}$ [h]", fontsize=fontsize + 2, rotation=270, labelpad=30)
    cbar_texp_ref_compare.ax.tick_params(labelsize=fontsize - 2)
    
    # --- PCS colorbar ---
    sm_texp_pcs_compare = mpl.cm.ScalarMappable(norm=norm_texp_compare, cmap=CMAP_TEXP)
    sm_texp_pcs_compare.set_array([])
    cbar_texp_pcs_compare = fig_texp_compare.colorbar(sm_texp_pcs_compare, ax=ax_pcs_compare, pad=0.03, fraction=0.045, ticks=ticks_texp_compare)
    cbar_texp_pcs_compare.ax.set_yticklabels([f"{tick:g}" for tick in ticks_texp_compare])
    cbar_texp_pcs_compare.set_label(rf"Exposure time to reach $S/N={SNR_thr:g}$ [h]", fontsize=fontsize + 2, rotation=270, labelpad=30)
    cbar_texp_pcs_compare.ax.tick_params(labelsize=fontsize - 2)
    
    # --- Global title ---
    population_label = "archive-conditional" if table_type == "Archive" else "population-conditional"
    band_title       = str(BAND_FIXED).replace("_100", "").replace("_", " ")
    config_title     = str(CONFIG_FIXED).replace("_", " ").replace("LYOT", "Lyot")
    fig_texp_compare.suptitle(rf"ELT/{instru} | {post_processing} | {band_title} | {config_title}" + "\n" + rf"{population_label} exposure-time comparison", fontsize=fontsize + 3, weight="bold", y=0.99)
    fig_texp_compare.tight_layout()
    plt.show()
    
    # ============================================================================
    # Figure
    # ============================================================================
    fig_texp_phys, axes = plt.subplots(1, 2, figsize=(22, 9), dpi=dpi, sharex=True, sharey=True)
    ax_phys, ax_ratio   = axes
    
    # --------------------------------------------------------------------------
    # Left panel: physical space + planet-by-planet reference exposure time
    # --------------------------------------------------------------------------
    add_ptype_regions_phys(ax_phys)
    
    # Non-reachable at reference shown in light gray
    for method in detection_methods_plot:
        mask = valid_phys_texp & masks_detection_method[method] & (~reachable_ref)
        if not np.any(mask):
            continue
        ax_phys.scatter(SMA_phys[mask], PlanetMass_phys[mask], c="0.80", marker=marker_detection_methods[method], s=SS_POP, edgecolors="none", alpha=0.20, zorder=2)
    
    # Reachable at reference colored by Texp_ref
    finite_texp_ref = Texp_ref[valid_phys_texp & reachable_ref]
    if finite_texp_ref.size == 0:
        raise RuntimeError("No finite reference exposure time to display in physical space.")
    texp_min = max(np.nanmin(finite_texp_ref), t_grid[0])
    texp_max = min(np.nanmax(finite_texp_ref), t_grid[-1])
    if not np.isfinite(texp_min) or not np.isfinite(texp_max) or texp_min <= 0 or texp_max <= 0 or texp_min >= texp_max:
        raise RuntimeError("Invalid exposure-time range for the reference panel.")
    norm_texp = make_texp_norm(texp_min, texp_max)
    for method in detection_methods_plot:
        mask = valid_phys_texp & masks_detection_method[method] & reachable_ref
        if not np.any(mask):
            continue
        ax_phys.scatter(SMA_phys[mask], PlanetMass_phys[mask], c=Texp_ref[mask], cmap=CMAP_TEXP, norm=norm_texp, marker=marker_detection_methods[method], s=SS_POP, edgecolors="k", linewidths=0.35, alpha=ALPHA_POP, zorder=10)
    
    ax_phys.set_xscale("log")
    ax_phys.set_yscale("log")
    ax_phys.set_xlim(*PHYS_XLIM)
    ax_phys.set_ylim(*PHYS_YLIM)
    ax_phys.set_xlabel("Semi-major axis [AU]", fontsize=fontsize + 2, labelpad=10)
    ax_phys.set_ylabel(r"Planet mass [$M_\oplus$]", fontsize=fontsize + 2, labelpad=10)
    ax_phys.set_title(rf"Reference exposure time to reach $S/N={SNR_thr:g}$", fontsize=fontsize + 2, fontweight="bold")
    ax_phys.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-1000)
    ax_phys.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=-1000)
    ax_phys.tick_params(axis="both", which="major", labelsize=fontsize - 2, length=6, width=1.0)
    ax_phys.tick_params(axis="both", which="minor", labelsize=fontsize - 2, length=3, width=0.8)
    
    handles_method = [Line2D([], [], ls="", marker=marker_detection_methods[method], ms=13, markerfacecolor="white", markeredgecolor="k", markeredgewidth=1.5, color="k", label=label_detection_methods[method]) for method in detection_methods_plot]
    legend_methods = ax_phys.legend(handles=handles_method, fontsize=fontsize - 4, loc="lower right", frameon=True, edgecolor="gray", facecolor="white", title="Discovery method", title_fontsize=fontsize)
    legend_methods.set_zorder(10_000)
    ax_phys.add_artist(legend_methods)
    
    sm_texp = mpl.cm.ScalarMappable(norm=norm_texp, cmap=CMAP_TEXP)
    sm_texp.set_array([])
    candidate_ticks_texp = np.array([1e-2, 1e-1, 1, 10, 100, 1e3, 1e4], dtype=float)
    ticks_texp = candidate_ticks_texp[(candidate_ticks_texp >= texp_min) & (candidate_ticks_texp <= texp_max)]
    if ticks_texp.size == 0:
        ticks_texp = np.array([texp_min, texp_max])
    
    cbar_texp = fig_texp_phys.colorbar(sm_texp, ax=ax_phys, pad=0.03, fraction=0.045, ticks=ticks_texp)
    cbar_texp.ax.set_yticklabels([f"{tick:g}" for tick in ticks_texp])
    cbar_texp.set_label(rf"Exposure time to reach $S/N={SNR_thr:g}$ [h]", fontsize=fontsize + 2, rotation=270, labelpad=30)
    cbar_texp.ax.tick_params(labelsize=fontsize - 2)
    
    # --------------------------------------------------------------------------
    # Right panel: local exposure-time gain T_ref / T_PCS
    # --------------------------------------------------------------------------
    G_texp_plot = np.ma.masked_invalid(G_texp)
    finite_gain = np.isfinite(G_texp) & (G_texp > 0)
    if not np.any(finite_gain):
        raise RuntimeError("No finite exposure-time gain value to display.")
    gain_min = np.nanmin(G_texp[finite_gain])
    gain_max = np.nanmax(G_texp[finite_gain])
    
    # The reference/no-gain value must always be included
    gain_min    = min(gain_min, 1.0)
    gain_max    = max(gain_max, 1.0)
    gain_norm   = LogNorm(vmin=gain_min, vmax=gain_max)
    gain_levels = np.geomspace(gain_min, gain_max, 20)
    cf_gain     = ax_ratio.contourf(10**xa, 10**ym, G_texp_plot, levels=gain_levels, cmap=CMAP_GAIN_TEXP, norm=gain_norm)
    add_ptype_regions_phys(ax_ratio, alpha=0.5)
    
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlim(*PHYS_XLIM)
    ax_ratio.set_ylim(*PHYS_YLIM)
    ax_ratio.set_xlabel("Semi-major axis [AU]", fontsize=fontsize + 2, labelpad=10)
    ax_ratio.set_title(rf"Exposure-time gain $T_{{\rm {instru}}}/T_{{\rm PCS}}$", fontsize=fontsize + 2, fontweight="bold")
    ax_ratio.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=1000)
    ax_ratio.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=1000)
    ax_ratio.tick_params(axis="both", which="major", labelsize=fontsize - 2, length=6, width=1.0)
    ax_ratio.tick_params(axis="both", which="minor", labelsize=fontsize - 2, length=3, width=0.8)
    
    # Meaningful multiplicative ticks
    candidate_gain_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
    gain_ticks = candidate_gain_ticks[(candidate_gain_ticks >= gain_min) & (candidate_gain_ticks <= gain_max)]
    if gain_ticks.size == 0:
        gain_ticks = np.array([1.0, gain_max])
    cbar_gain = fig_texp_phys.colorbar(cf_gain, ax=ax_ratio, pad=0.03, fraction=0.045, ticks=gain_ticks)
    cbar_gain.ax.set_yticklabels([rf"$\times${tick:g}" for tick in gain_ticks])
    cbar_gain.set_label(rf"Exposure-time gain $T_{{\rm {instru}}}/T_{{\rm PCS}}$", fontsize=fontsize + 2, rotation=270, labelpad=30)    
    cbar_gain.ax.tick_params(labelsize=fontsize - 2)
    
    # --------------------------------------------------------------------------
    # Global title
    # --------------------------------------------------------------------------
    population_label = "archive-conditional" if table_type == "Archive" else "population-conditional"
    band_title       = str(BAND_FIXED).replace("_100", "").replace("_", " ")
    config_title     = str(CONFIG_FIXED).replace("_", " ").replace("LYOT", "Lyot")
    fig_texp_phys.suptitle(rf"ELT/{instru} | {post_processing} | {band_title} | {config_title}" + "\n" + rf"{population_label} exposure time and PCS-like gain ", fontsize=fontsize + 3, weight="bold", y=0.99)
    fig_texp_phys.tight_layout()
    plt.show()
    
    
    
    # %%
    # ============================================================================
    # DETECTION YIELD BY PLANET TYPE — REFERENCE VS PCS-LIKE
    # ============================================================================
    # Compare the number (or fraction) of known planets reaching SNR_thr within a
    # fixed exposure time for the reference instrumental configuration and a
    # PCS-like boosted configuration obtained by changing the post-AO WFE.
    #
    # The plotting style intentionally reproduces yield_hist_instrus_ptypes_ELT:
    #   - detailed Cold / Warm / Hot planet classes
    #   - alternating vertical shading
    #   - temperature-family background bands
    #   - total population shown with faded bars
    #   - logarithmic y-axis for absolute yields
    # ============================================================================
    
    # --- User parameters ---
    EXPOSURE_TIME_YIELD = exposure_time / 60 # [h] exposure time per target
    BAND_YIELD          = BAND_FIXED
    CONFIG_YIELD        = CONFIG_FIXED
    WFE_PCS_YIELD       = 50.0                # PCS-like post-AO WFE [nm RMS]
    FRACTION_YIELD      = False               # False: number of detections; True: detected fraction [%]
    COLOR_REF           = "gray"              # Reference configuration color
    COLOR_PCS           = "steelblue"         # PCS-like configuration color
    
    # --- Check that the exposure-time calculation has already been performed ---
    if "Texp_4D_WFE" not in locals():
        raise RuntimeError("Run the EXPOSURE-TIME PERFORMANCE cell before this one.")
    if BAND_YIELD not in bands:
        raise ValueError(f"BAND_YIELD={BAND_YIELD!r} not available. Available bands: {bands}")
    if CONFIG_YIELD not in configs:
        raise ValueError(f"CONFIG_YIELD={CONFIG_YIELD!r} not available. Available configs: {configs}")
    
    # --- Fixed band/configuration and WFE indices ---
    ib_yield      = bands.index(BAND_YIELD)
    ic_yield      = configs.index(CONFIG_YIELD)
    idx_WFE_pcs_y = int(np.nanargmin(np.abs(np.asarray(WFE, dtype=float) - WFE_PCS_YIELD)))
    
    # --- Exposure times required to reach SNR_thr [h] ---
    Texp_yield_ref = np.asarray(Texp_4D_WFE[:, ib_yield, ic_yield, idx_WFE_ref], dtype=float) / 60
    Texp_yield_pcs = np.asarray(Texp_4D_WFE[:, ib_yield, ic_yield, idx_WFE_pcs_y], dtype=float) / 60
    
    # NaN / inf exposure times naturally correspond to non-detections
    detected_ref = np.isfinite(Texp_yield_ref) & (Texp_yield_ref <= EXPOSURE_TIME_YIELD)
    detected_pcs = np.isfinite(Texp_yield_pcs) & (Texp_yield_pcs <= EXPOSURE_TIME_YIELD)
    
    # --- Detailed planet-type yields ---
    # Keep exactly the order of planet_types so that Cold / Warm / Hot families
    # appear in the same order as in yield_hist_instrus_ptypes_ELT.
    planet_types_array = np.array(list(planet_types.keys()))
    N_ptype             = np.zeros(len(planet_types_array))
    yield_ref           = np.zeros(len(planet_types_array))
    yield_pcs           = np.zeros(len(planet_types_array))
    
    for i in range(len(planet_types_array)):
        ptype      = planet_types_array[i]
        mask_ptype = np.asarray(get_mask_planet_type(planet_table=planet_table, planet_type=ptype), dtype=bool)
        
        N_ptype[i]   = np.sum(mask_ptype)
        Ndet_ref     = np.sum(mask_ptype & detected_ref)
        Ndet_pcs     = np.sum(mask_ptype & detected_pcs)
        
        if FRACTION_YIELD:
            yield_ref[i] = 100 * Ndet_ref / N_ptype[i] if N_ptype[i] > 0 else 0
            yield_pcs[i] = 100 * Ndet_pcs / N_ptype[i] if N_ptype[i] > 0 else 0
        else:
            yield_ref[i] = Ndet_ref
            yield_pcs[i] = Ndet_pcs
    
    if np.sum(N_ptype) == 0:
        raise RuntimeError("No planet available for the selected planet-type population.")
    
    # --- Plot parameters: same structure as yield_hist_instrus_ptypes_ELT ---
    N_table   = 2
    bar_width = 1 / (N_table + 2)
    linewidth = 1
    indices   = np.arange(len(planet_types_array))
    
    # Local helper reproducing the faded population bars of the original figure
    def faded_yield(color, amount=0.82):
        rgb = np.asarray(mpl.colors.to_rgb(color))
        return tuple(rgb + (1 - rgb) * amount)
    
    plt.figure(figsize=(14, 8), dpi=dpi)
    
    # Shade for every type
    for i in range(len(planet_types_array)):
        if i % 2 == 0:
            plt.axvspan(i - 0.5, i + 0.5, facecolor="gray", alpha=0.1, zorder=-10)
    for x in np.arange(0.5, len(planet_types_array), 1.0):
        plt.axvline(x, color="gray", lw=linewidth, ls=":", zorder=-10)
    
    # Shade for every temperature type
    tags     = [s.split(" ", 1)[0] for s in planet_types_array] # "Cold Sub-Earth" -> "Cold"
    segments = []
    start    = 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[i-1]:
            segments.append((tags[start], start, i-1)) # (tag, i0, i1)
            start = i
    
    band = {"Cold": "#E6F0FF", "Warm": "#FFF1D6", "Hot": "#FFE2E2"} # blue / orange / red pale backgrounds
    sep  = {"Cold": "#B7D3FF", "Warm": "#FFD89A", "Hot": "#FFB4B4"} # slightly stronger separation colors
    
    for tag, i0, i1 in segments:
        plt.axvspan(i0 - 0.5, i1 + 0.5, facecolor=band.get(tag, "#EEEEEE"), alpha=0.3, zorder=-20)
        plt.axvline(i1 + 0.5, color=sep.get(tag, "0.8"), lw=linewidth, ls=(0, (4, 3)), alpha=0.9, zorder=-15)
        xmid = 0.5 * (i0 + i1)
        plt.text(xmid, 1.01, tag.upper(), ha="center", va="bottom", transform=plt.gca().get_xaxis_transform(), fontsize=18, color=sep.get(tag, "0.4"), fontweight="bold")
    
    if len(segments) > 0:
        plt.axvline(segments[0][1] - 0.5, color=sep.get(segments[0][0], "0.8"), lw=linewidth, ls=(0, (4, 3)), alpha=0.9, zorder=-15)
    
    # --- Reference configuration ---
    idx = 0 - N_table//2 + 0.5
    plt.bar(indices + idx*bar_width, yield_ref, bar_width, zorder=10, edgecolor="black", color=COLOR_REF, linewidth=linewidth, label=rf"ELT/{instru} reference ($WFE={WFE[idx_WFE_ref]:.0f}$ nm RMS)")
    if not FRACTION_YIELD:
        plt.bar(indices + idx*bar_width, N_ptype, bar_width, edgecolor=COLOR_REF, color=faded_yield(COLOR_REF), linewidth=linewidth, zorder=1)
    
    # --- PCS-like configuration ---
    idx = 1 - N_table//2 + 0.5
    plt.bar(indices + idx*bar_width, yield_pcs, bar_width, zorder=10, edgecolor="black", color=COLOR_PCS, linewidth=linewidth, label=rf"PCS-like ($WFE={WFE[idx_WFE_pcs_y]:.0f}$ nm RMS)")
    if not FRACTION_YIELD:
        plt.bar(indices + idx*bar_width, N_ptype, bar_width, edgecolor=COLOR_PCS, color=faded_yield(COLOR_PCS), linewidth=linewidth, zorder=1)
    
    # --- Axes ---
    plt.xticks(indices, planet_types_array, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=16)
    plt.xlim(-0.5, len(planet_types_array) - 0.5)
    
    if FRACTION_YIELD:
        plt.ylabel("Fraction of planets re-detected [%]", fontsize=18, labelpad=18)
    else:
        plt.ylabel("Number of planets re-detected", fontsize=18, labelpad=18)
        plt.yscale("log")
    
    # --- Title / grid / legend: same layout as yield_hist_instrus_ptypes_ELT ---
    plt.title(f"Known exoplanets detection yield for {EXPOSURE_TIME_YIELD:.0f} h per target", fontsize=20, fontweight="bold", pad=36)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    plt.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25, zorder=-10)
    plt.legend(title="Instrument configurations", title_fontsize=16, fontsize=14, loc="upper center", frameon=True, edgecolor="gray", facecolor="whitesmoke", ncol=2)
    plt.tight_layout()
    plt.show(block=False)
    
    
    

    # %%
    return locals()



# %%

if __name__ == "__main__":
    
    if sys.platform.startswith("win"):
        mp.freeze_support() # for Windows
    _out = main()
    globals().update(_out)





