import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["NUMBA_NUM_THREADS"]    = "1"

# import FastYield modules
# import fastyield.config as config
# config.dpi_fig = 72

from fastyield.config import rad2arcsec
from fastyield.FastYield import get_filename_table
from fastyield.FastYield_helpers import create_memmap_with_log, interp_flat_last_axis, sigma_RON_2_from_DIT, window_nanmax_1d, window_sums_1d, window_sigma_al_spec_1d
from fastyield.spectrum import get_thermal_reflected_spectrum, filtered_flux
from fastyield.signal_noise import compress_h_for_sigma_base_2, compute_sigma_base_2_speck_numba, compute_sigma_base_2_al_spat_numba

# import numpy modules
import numpy as np

# import matplotlib modules
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# import other modules
from tqdm import tqdm
from multiprocessing import cpu_count
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import yaml

# Parameters for multiprocessing
nproc     = max(cpu_count()-3, 1)
chunksize = 8



# -------------------------------------------------------------
# Function to read instrument concepts from YAML files
# -------------------------------------------------------------

def load_instrument_concept(name, concept_dir=None):
    """Load an instrument concept from instrument_concepts/<name>.yaml."""
    concept_dir = Path(__file__).resolve().parent / "instrument_concepts" if concept_dir is None else Path(concept_dir)
    path = concept_dir / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in concept_dir.glob("*.yaml"))
        raise FileNotFoundError(f"Instrument concept '{name}' not found in {concept_dir}. Available: {available}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



# -------------------------------------------------------------
# Functions for IFU and imager for main quantities computations
# -------------------------------------------------------------

# --- General helpers ---

def find_hwo_psf_suffix(psf_dir, instru, apodizer, coronagraph, D, suffix_override=None):
    """Select a complete, diameter-matched HWO PSF library.

    An explicit suffix is recommended when several design grids are present.
    Automatic selection is allowed only if exactly one complete candidate
    matches the requested coronagraph and telescope diameter.
    """
    psf_dir  = Path(psf_dir)
    prefix   = f"{instru}_PSF_profile_density_4D_"
    files    = sorted(psf_dir.glob(f"{prefix}*.fits"))
    suffixes = [p.stem[len(prefix):] for p in files]
    if suffix_override is not None:
        suffixes = [suffix_override]
    else:
        matched = [s for s in suffixes if f"D{D:.2f}m" in s and apodizer in s and coronagraph in s]
        if matched:
            suffixes = matched
        elif len(suffixes) == 1 and apodizer in suffixes[0] and coronagraph in suffixes[0]:
            print("WARNING: HWO PSF filename does not encode matching D; verify its physical telescope aperture before using it.")
        else:
            raise FileNotFoundError(f"No unambiguous HWO PSF library matches {apodizer}/{coronagraph}, D={D:.2f} m in {psf_dir}. Found: {suffixes}")
    required = ("wave", "C", "IWA_lD", "separation_mas", "PSF_profile_density_4D", "fraction_core_4D", "radial_transmission_4D")
    complete = [s for s in suffixes if all((psf_dir / f"{instru}_{key}_{s}.fits").is_file() for key in required)]
    if len(complete) != 1:
        raise FileNotFoundError(f"Expected exactly one complete HWO PSF library in {psf_dir}; available matches: {complete}. Set psf_suffix explicitly to choose one.")
    return complete[0]


# -------------------------------------------------------------

# --- Globals (context) used by multiprocessing workers ---
_IM_CTX  = None
_IFU_CTX = None
_SNR_CTX = None
_MASK_CTX = None
_PDET_CTX = None



def init_worker_IM(ctx):
    global _IM_CTX
    _IM_CTX = ctx

    # Temp PSF memmaps opened inside each worker
    _IM_CTX["PSF_profile_4D"]     = np.load(_IM_CTX["PSF_profile_4D_path"],     mmap_mode="r")
    _IM_CTX["fraction_core_4D"]   = np.load(_IM_CTX["fraction_core_4D_path"],   mmap_mode="r")
    _IM_CTX["PSF_profile_max_3D"] = np.load(_IM_CTX["PSF_profile_max_3D_path"], mmap_mode="r")

    # Output memmaps opened inside each worker
    _IM_CTX["signal_out"]       = np.load(_IM_CTX["signal_path"],       mmap_mode="r+")
    _IM_CTX["sigma_halo_2_out"] = np.load(_IM_CTX["sigma_halo_2_path"], mmap_mode="r+")
    _IM_CTX["sigma_bkg_2_out"]  = np.load(_IM_CTX["sigma_bkg_2_path"],  mmap_mode="r+")
    _IM_CTX["DIT_out"]          = np.load(_IM_CTX["DIT_path"],          mmap_mode="r+")



def init_worker_IFU(ctx):
    global _IFU_CTX
    _IFU_CTX = ctx

    # Temp PSF memmaps opened inside each worker
    _IFU_CTX["PSF_profile_4D"]     = np.load(_IFU_CTX["PSF_profile_4D_path"],     mmap_mode="r")
    _IFU_CTX["fraction_core_4D"]   = np.load(_IFU_CTX["fraction_core_4D_path"],   mmap_mode="r")
    _IFU_CTX["PSF_profile_max_3D"] = np.load(_IFU_CTX["PSF_profile_max_3D_path"], mmap_mode="r")

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
    A_FWHM                 = _IFU_CTX["A_FWHM"]
    separation             = _IFU_CTX["separation"]
    thermal_model          = _IFU_CTX["thermal_model"]
    reflected_model        = _IFU_CTX["reflected_model"]
    post_processing        = _IFU_CTX["post_processing"]
    wave_model             = _IFU_CTX["wave_model"]
    wave_instru            = _IFU_CTX["wave_instru"]
    wave_K                 = _IFU_CTX["wave_K"]
    counts_vega_K          = _IFU_CTX["counts_vega_K"]
    scale_spectrum         = _IFU_CTX["scale_spectrum"]
    N_R                    = _IFU_CTX["N_R"]
    N_l0                   = _IFU_CTX["N_l0"]
    N_Dl                   = _IFU_CTX["N_Dl"]
    R                      = _IFU_CTX["R"]
    wave_1D_list           = _IFU_CTX["wave_1D_list"]
    dwave_1D_list          = _IFU_CTX["dwave_1D_list"]
    trans_tel_1D_list = _IFU_CTX["trans_tel_1D_list"]
    trans_instru           = _IFU_CTX["trans_instru"]
    background_1D_list     = _IFU_CTX["background_1D_list"]
    pxscales               = _IFU_CTX["pxscales"]
    Rc                     = _IFU_CTX["Rc"]
    filter_type            = _IFU_CTX["filter_type"]
    range_1D_list          = _IFU_CTX["range_1D_list"]
    valid_windows_3D       = _IFU_CTX["valid_windows_3D"]
    saturation_e           = _IFU_CTX["saturation_e"]
    min_DIT                = _IFU_CTX["min_DIT"]
    max_DIT                = _IFU_CTX["max_DIT"]
    PSF_profile_4D         = _IFU_CTX["PSF_profile_4D"]
    fraction_core_4D       = _IFU_CTX["fraction_core_4D"]
    PSF_profile_max_3D     = _IFU_CTX["PSF_profile_max_3D"]
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


    # HWO PSFs depend on (l0, C_raw, IWA_lD, separation), not stellar magnitude.
    # Interpolate linearly in the planet's actual angular separation (mas).
    PSF_profile_sep_3D   = interp_flat_last_axis(separation, PSF_profile_4D,   separation_planet, dtype=dtype)
    fraction_core_sep_3D = interp_flat_last_axis(separation, fraction_core_4D, separation_planet, dtype=dtype)


    # --- Converting in [ph/mn/µm] ---
    star_spectrum.flux   *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]
    planet_spectrum.flux *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]


    # --- Computing data depending only on R, l0 and Dl ---
    # These quantities do not yet include the spatial PSF factors or
    # the instrumental throughput axis.  They are pure spectral quantities,
    # computed once for each (R, l0, Dl) combination and later broadcast over
    # (C_raw, IWA_lD, trans_instru).
    signal_3D            = np.zeros((N_R, N_l0, N_Dl), dtype=dtype) # = alpha - beta                         (R, l0, Dl) [ph/mn]
    sigma_halo_2_3D      = np.zeros((N_R, N_l0, N_Dl), dtype=dtype) # = sum (trans * S_star * template**2)   (R, l0, Dl) [ph/mn]
    max_star_flux_3D     = np.zeros((N_R, N_l0, N_Dl), dtype=dtype) # = max (trans * S_star)                 (R, l0, Dl) [ph/mn/bin]
    sigma_bkg_2_3D       = np.zeros((N_R, N_l0, N_Dl), dtype=dtype) # = sum (background * template**2)       (R, l0, Dl) [ph/px/mn]
    sigma_syst_base_2_3D = np.zeros((N_R, N_l0, N_Dl), dtype=dtype) # = sigma_syst,base**2                   (R, l0, Dl) [ph/mn]**2
    for i, res in enumerate(R):

        valid_windows_i = valid_windows_3D[i]
        if not np.any(valid_windows_i):
            signal_3D[i]            = np.nan
            sigma_halo_2_3D[i]      = np.nan
            max_star_flux_3D[i]     = np.nan
            sigma_bkg_2_3D[i]       = np.nan
            sigma_syst_base_2_3D[i] = np.nan
            continue

        wave_i           = wave_1D_list[i]
        dwave_i          = dwave_1D_list[i]
        trans_tel_i = trans_tel_1D_list[i]
        star_i           = star_spectrum.degrade_resolution(wave_i,   renorm=False, R_output=res, filter_type=filter_type).flux # [ph/mn/µm]
        planet_i         = planet_spectrum.degrade_resolution(wave_i, renorm=False, R_output=res, filter_type=filter_type).flux # [ph/mn/µm]
        idx_lo, idx_hi   = range_1D_list[i] # (l0, Dl)

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
        trans_tel_star_i = trans_tel_i * star_i  # [ph/mn/bin]

        # Brightest stellar spectral-bin flux in each spectral window.
        # This is needed even when the planet template is null, because it sets the
        # detector saturation limit.
        max_star_flux_3D[i] = window_nanmax_1d(values=trans_tel_star_i, idx_lo=idx_lo, idx_hi=idx_hi).astype(dtype)

        if post_processing == "MM":
            # Raw matched-filter template before window-by-window normalization (T = trans*planet_HF):
            #     T_i = trans_tel_i * planet_HF_i
            # The actual normalized template in each spectral window is (t = T/norm):
            #     t_ijk = T_i[sl] / sqrt(sum_sl(T_i**2))
            T_i               = trans_tel_i * planet_HF_i                            # (wave_res)
            T2_i              = T_i**2                                                    # (wave_res)
            norm2_2D          = window_sums_1d(values=T2_i, idx_lo=idx_lo, idx_hi=idx_hi) # (l0, Dl)
            valid_2D          = valid_windows_i & np.isfinite(norm2_2D) & (norm2_2D > 0)  # (l0, Dl)
            norm_2D           = np.zeros_like(norm2_2D, dtype=dtype)                      # (l0, Dl)
            norm_2D[valid_2D] = np.sqrt(norm2_2D[valid_2D])                               # (l0, Dl)

            # CCF signal computation in [ph/mn], before multiplying by
            # the spatial PSF throughput and instrumental efficiency.
            #   alpha = retained HF planet signal                          = sum(trans*planet_HF * t)                 = sum(T*t) = sum(T*T/norm) = norm**2/norm = norm
            #   beta = self-subtraction from stellar/planet LF-HF coupling = sum(trans*star_HF*planet_LF/star_LF * t) = sum(trans*star_HF*planet_LF/star_LF * T) / norm
            alpha_2D               = norm_2D
            ratio_planet_star_LF_i = np.divide(planet_LF_i, star_LF_i, out=np.zeros_like(planet_LF_i), where=np.isfinite(star_LF_i) & (star_LF_i != 0))
            beta_num_2D            = window_sums_1d(values=trans_tel_i * star_HF_i * ratio_planet_star_LF_i * T_i, idx_lo=idx_lo, idx_hi=idx_hi)
            signal_2D              = np.zeros_like(norm2_2D, dtype=float)
            signal_2D[valid_2D]    = alpha_2D[valid_2D] - beta_num_2D[valid_2D]/norm_2D[valid_2D]

        elif post_processing == "DI":
            # Raw DI template before window-by-window normalization (T = trans*planet):
            #     T_i = trans_tel_i * planet_i
            # The actual normalized template in each spectral window is (t = T/norm):
            #     t_ijk = T_i[sl] / sqrt(sum_sl(T_i**2))
            T_i               = trans_tel_i * planet_i                               # (wave_res)
            T2_i              = T_i**2                                                    # (wave_res)
            norm2_2D          = window_sums_1d(values=T2_i, idx_lo=idx_lo, idx_hi=idx_hi) # (l0, Dl)
            valid_2D          = valid_windows_i & np.isfinite(norm2_2D) & (norm2_2D > 0)  # (l0, Dl)
            norm_2D           = np.zeros_like(norm2_2D, dtype=float)                      # (l0, Dl)
            norm_2D[valid_2D] = np.sqrt(norm2_2D[valid_2D])                               # (l0, Dl)

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
        sigma_halo_num_2D       = window_sums_1d(values=trans_tel_star_i * T2_i, idx_lo=idx_lo, idx_hi=idx_hi)
        sigma_halo_2D           = np.zeros_like(norm2_2D, dtype=float)
        sigma_halo_2D[valid_2D] = sigma_halo_num_2D[valid_2D] / norm2_2D[valid_2D]
        sigma_halo_2_3D[i]      = sigma_halo_2D.astype(dtype)

        # Background photon noise in [ph/px/mn]**2
        background_i           = background_1D_list[i]                                                    # (wave_res) [ph/mn/bin/mas2]
        sigma_bkg_num_2D       = window_sums_1d(values=background_i * T2_i, idx_lo=idx_lo, idx_hi=idx_hi) # (l0, Dl)   [ph/mn/mas2]
        sigma_bkg_num_2D      *= pxscales[:, None]**2                                                     # (l0, Dl)   [ph/mn/mas2] => [ph/px/mn]
        sigma_bkg_2D           = np.zeros_like(norm2_2D, dtype=float)
        sigma_bkg_2D[valid_2D] = sigma_bkg_num_2D[valid_2D] / norm2_2D[valid_2D]
        sigma_bkg_2_3D[i]      = sigma_bkg_2D.astype(dtype)

        # Systematic base term in [ph/mn]**2
        # This part is harder to vectorize cleanly because the covariance calculation
        # depends on the compressed wavelength grid, the local pxscale, and the exact
        # normalized h_PP vector in each window. We therefore keep the small (j, k)
        # loop only for sigma_syst_base_2.
        if post_processing == "MM":
            g_i              = trans_tel_star_i * T_i
            sigma_al_spec_2D = window_sigma_al_spec_1d(g=g_i, norm2=norm2_2D, idx_lo=idx_lo, idx_hi=idx_hi)
        for j in range(N_l0):
            pxscale_rad = pxscales_rad[j]
            for k in range(N_Dl):
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
                h_PP_native                       = (trans_tel_star_i[sl] * T_i[sl] / norm_2D[j, k])[:, None] # (wave, sep) weighted halo H*t
                wave_c, h_PP_c, _                 = compress_h_for_sigma_base_2(wave=wave_ijk, h=h_PP_native, post_processing=post_processing, D=D, pxscale=pxscale_rad, separation_ref=separation_planet_rad, sep_unit="rad")
                if post_processing == "DI":
                    sigma_syst_base_2_3D[i, j, k] = compute_sigma_base_2_speck_numba(h=h_PP_c, wave=wave_c, separation_rad=separation_planet_rad_arr, D=D)[0]
                elif post_processing == "MM":
                    sigma_base_2_al_spat = compute_sigma_base_2_al_spat_numba(h=h_PP_c, wave=wave_c, separation_rad=separation_planet_rad_arr, pxscale_rad=pxscale_rad)[0]
                    sigma_syst_base_2_3D[i, j, k] = 0.5 * (sigma_al_spec_2D[j, k] + sigma_base_2_al_spat)
        invalid                          = ~valid_windows_i
        signal_3D[i, invalid]            = np.nan
        sigma_halo_2_3D[i, invalid]      = np.nan
        max_star_flux_3D[i, invalid]     = np.nan
        sigma_bkg_2_3D[i, invalid]       = np.nan
        sigma_syst_base_2_3D[i, invalid] = np.nan


    # --- Computing DIT length in [mn/DIT] (R, l0, Dl, C_raw, IWA, TI) ---

    # max_star_flux in [ph/px/mn/bin] in the brightest pixel and bin (through coronagraph, if any):
    # [total star ph/mn/bin] in the brightest bin * [star flux fraction/px] in the brightest pixel => [ph/px/mn/bin] in the brightest pixel and bin
    # (R, l0, Dl, 1, 1)                           * (1, l0, 1, C_raw, IWA)                         => (R, l0, Dl, C_raw, IWA)
    max_star_flux_5D = max_star_flux_3D[:, :, :, None, None] * PSF_profile_max_3D[None, :, None, :, :]

    # max_star_flux in [e-/px/mn/bin] in the brightest pixel with instrumental transmission (through coronagraph, if any):
    # [ph/px/mn/bin] in the brightest pixel and bin * [e-/ph]             => [e-/px/mn/bin] in the brightest pixel and bin
    # (R, l0, Dl, C_raw, IWA, 1)                    * (1, 1, 1, 1, 1, TI) => (R, l0, Dl, C_raw, IWA, TI)
    max_star_flux_6D = max_star_flux_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]

    # Computing DIT length in [mn/DIT] (+ clipping between min and max DIT)
    saturation_e_6D = saturation_e[None, :, None, None, None, None]
    min_DIT_6D      = min_DIT[None, :, None, None, None, None]
    max_DIT_6D      = max_DIT[None, :, None, None, None, None]
    DIT_6D          = saturation_e_6D / max_star_flux_6D # (R, l0, Dl, C_raw, IWA, TI) [mn/DIT]
    DIT_6D          = np.maximum(DIT_6D, min_DIT_6D)     # (R, l0, Dl, C_raw, IWA, TI) [mn/DIT]
    DIT_6D          = np.minimum(DIT_6D, max_DIT_6D)     # (R, l0, Dl, C_raw, IWA, TI) [mn/DIT]


    # --- Computing signal in [e-/FWHM/DIT] (R, l0, Dl, C_raw, IWA, TI) ---

    # Signal at the planet's separation inside the FWHM in [ph/FWHM/mn] (throught coronagraph, if any)
    # [total planet ph/mn] * [planet flux fraction/FWHM] => [ph/FWHM/mn]
    # (R, l0, Dl, 1, 1)    * (1, l0, 1, C_raw, IWA)      => (R, l0, Dl, C_raw, IWA)
    signal_5D = signal_3D[:, :, :, None, None] * fraction_core_sep_3D[None, :, None, :, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/FWHM/mn]               * [e-/ph]             => [e-/FWHM/mn]
    # (R, l0, Dl, C_raw, IWA, 1) * (1, 1, 1, 1, 1, TI) => (R, l0, Dl, C_raw, IWA, TI)
    signal_6D = signal_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT (throught coronagraph, if any)
    # [e-/FWHM/mn]              * [mn/DIT]                  => [e-/FWHM/DIT]
    # (R, l0, Dl, C_raw, IWA, TI) * (R, l0, Dl, C_raw, IWA, TI) => (R, l0, Dl, C_raw, IWA, TI)
    signal_6D = signal_6D * DIT_6D


    # --- Computing stellar halo photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Dl, C_raw, IWA, TI) ---

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [ph/px/mn] (throught coronagraph, if any)
    # [total star ph/mn] * [star flux fraction/px] => [ph/px/mn]
    # (R, l0, Dl, 1, 1)  * (1, l0, 1, C_raw, IWA)  => (R, l0, Dl, C_raw, IWA)
    sigma_halo_2_5D = sigma_halo_2_3D[:, :, :, None, None] * PSF_profile_sep_3D[None, :, None, :, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/px/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]                 * [e-/ph]             => [e-/px/mn]
    # (R, l0, Dl, C_raw, IWA, 1) * (1, 1, 1, 1, 1, TI) => (R, l0, Dl, C_raw, IWA, TI)
    sigma_halo_2_6D = sigma_halo_2_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance) (throught coronagraph, if any)
    # [e-/px/mn]                  * [px/FWHM]*[mn/DIT]          => [e-/FWHM/DIT]
    # (R, l0, Dl, C_raw, IWA, TI) * (R, l0, Dl, C_raw, IWA, TI) => (R, l0, Dl, C_raw, IWA, TI)
    sigma_halo_2_6D = sigma_halo_2_6D * A_FWHM*DIT_6D


    # --- Computing stellar halo systematic noise variance (at sigma_m = 1) in [e-/FWHM/DIT]**2 (R, l0, Dl, C_raw, IWA, TI) (all quantities are squared since sigma_syst**2 propto stellar_halo**2) ---

    # Stellar halo systematic noise variance at the planet's separation in [ph/px/mn]**2 (throught coronagraph, if any)
    # [total star ph/mn]**2 * [star flux fraction/px]**2 => [ph/px/mn]**2
    # (R, l0, Dl, 1, 1)     * (1, l0, 1, C_raw, IWA)     => (R, l0, Dl, C_raw, IWA)
    sigma_syst_base_2_5D = sigma_syst_base_2_3D[:, :, :, None, None] * PSF_profile_sep_3D[None, :, None, :, :]**2

    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]**2              * [e-/ph]**2          => [e-/px/mn]**2
    # (R, l0, Dl, C_raw, IWA, 1) * (1, 1, 1, 1, 1, TI) => (R, l0, Dl, C_raw, IWA, TI)
    sigma_syst_base_2_6D = sigma_syst_base_2_5D[:, :, :, :, :, None] * trans_instru[None, None, None, None, None, :]**2

    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission integrated over the DIT and the FWHM (throught coronagraph, if any) (we take A_FWHM**2 * DIT**2 since sigma_syst_base_2 propto H_integrated**2 propto A_FWHM**2*DIT**2, where H_integrated = the halo integrated over FWHM boxes and the DIT)
    # [e-/px/mn]**2               * [px/FWHM]**2*[mn/DIT]**2  => [e-/FWHM/DIT]**2
    # (R, l0, Dl, C_raw, IWA, TI) * (R, l0, Dl, C_raw, IWA, TI) => (R, l0, Dl, C_raw, IWA, TI)
    sigma_syst_base_2_6D = sigma_syst_base_2_6D * A_FWHM**2*DIT_6D**2


    # --- Computing background photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Dl, C_raw, IWA, TI) ---

    # Background photon noise variance (Poisson) in [e-/px/mn] with instrumental transmission
    # [ph/px/mn]     * [e-/ph]       => [e-/px/mn]
    # (R, l0, Dl, 1) * (1, 1, 1, TI) => (R, l0, Dl, TI)
    sigma_bkg_2_4D = sigma_bkg_2_3D[:, :, :, None] * trans_instru[None, None, None, :]

    # Background photon noise variance (Poisson) in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance)
    # [e-/px/mn]            * [px/FWHM]*[mn/DIT]          => [e-/FWHM/DIT]
    # (R, l0, Dl, 1, 1, TI) * (R, l0, Dl, C_raw, IWA, TI) => (R, l0, Dl, C_raw, IWA, TI)
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
    A_FWHM             = _IM_CTX["A_FWHM"]
    separation         = _IM_CTX["separation"]
    thermal_model      = _IM_CTX["thermal_model"]
    reflected_model    = _IM_CTX["reflected_model"]
    wave_instru        = _IM_CTX["wave_instru"]
    wave_K             = _IM_CTX["wave_K"]
    counts_vega_K      = _IM_CTX["counts_vega_K"]
    scale_spectrum     = _IM_CTX["scale_spectrum"]
    trans_instru       = _IM_CTX["trans_instru"]
    trans_tel     = _IM_CTX["trans_tel"]
    background_flux_2D = _IM_CTX["background_flux_2D"]
    range_1D           = _IM_CTX["range_1D"]
    saturation_e       = _IM_CTX["saturation_e"]
    min_DIT            = _IM_CTX["min_DIT"]
    max_DIT            = _IM_CTX["max_DIT"]
    PSF_profile_4D     = _IM_CTX["PSF_profile_4D"]
    fraction_core_4D   = _IM_CTX["fraction_core_4D"]
    PSF_profile_max_3D = _IM_CTX["PSF_profile_max_3D"]
    dtype              = _IM_CTX["dtype"]


    # --- Planet row ---
    planet            = planet_table[idx]
    separation_planet = float(planet["AngSep"].value) # [mas]


    # --- Computing the planet and star models for this planet row in [J/s/m²/µm] ---
    planet_spectrum, _, _, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True, interpolated_spectrum=True, apply_kinematics=False)


    # HWO PSFs depend on (l0, C_raw, IWA_lD, separation), not stellar magnitude.
    # Interpolate linearly in the planet's actual angular separation (mas).
    PSF_profile_sep_3D   = interp_flat_last_axis(separation, PSF_profile_4D,   separation_planet, dtype=dtype)
    fraction_core_sep_3D = interp_flat_last_axis(separation, fraction_core_4D, separation_planet, dtype=dtype)


    # --- Converting in [ph/mn/bin] ---
    star   = star_spectrum.flux   * scale_spectrum # [J/s/m²/µm] => [ph/mn/bin]
    planet = planet_spectrum.flux * scale_spectrum # [J/s/m²/µm] => [ph/mn/bin]


    # --- Computing data (signal and stellar halo photon noise variance (Poisson)) depending only on l0 and Dl ---
    idx_lo, idx_hi = range_1D
    signal_2D      = window_sums_1d(trans_tel * planet, idx_lo, idx_hi).astype(dtype) # = sum ( trans * S_planet ) (l0, Dl) [ph/mn]
    star_flux_2D   = window_sums_1d(trans_tel * star,   idx_lo, idx_hi).astype(dtype) # = sum ( trans * S_star )   (l0, Dl) [ph/mn]


    # --- Computing DIT length in [mn/DIT] (l0, Dl, C_raw, IWA, TI) ---

    # max_star_flux in [ph/px/mn] in the brightest pixel (through coronagraph, if any):
    # [total star ph/mn] * [star flux fraction/px] in the brightest pixel => [ph/px/mn] in the brightest pixel
    # (l0, Dl, 1, 1)     * (l0, 1, C_raw, IWA)                              => (l0, Dl, C_raw, IWA)
    max_star_flux_4D = star_flux_2D[:, :, None, None] * PSF_profile_max_3D[:, None, :, :]

    # max_star_flux in [e-/px/mn] in the brightest pixel with instrumental transmission (through coronagraph, if any):
    # [ph/px/mn] in the brightest pixel * [e-/ph]          => [e-/px/mn] in the brightest pixel
    # (l0, Dl, C_raw, IWA, 1)           * (1, 1, 1, 1, TI) => (l0, Dl, C_raw, IWA, TI)
    max_star_flux_5D = max_star_flux_4D[:, :, :, :, None] * trans_instru[None, None, None, None, :]

    # Computing DIT length in [mn/DIT] (+ clipping between min and max DIT)
    saturation_e_5D = saturation_e[:, None, None, None, None]
    min_DIT_5D      = min_DIT[:, None, None, None, None]
    max_DIT_5D      = max_DIT[:, None, None, None, None]
    DIT_5D          = saturation_e_5D / max_star_flux_5D # (l0, Dl, C_raw, IWA, TI) [mn/DIT]
    DIT_5D          = np.maximum(DIT_5D, min_DIT_5D)     # (l0, Dl, C_raw, IWA, TI) [mn/DIT]
    DIT_5D          = np.minimum(DIT_5D, max_DIT_5D)     # (l0, Dl, C_raw, IWA, TI) [mn/DIT]


    # --- Computing signal in [e-/FWHM/DIT] (l0, Dl, C_raw, IWA, TI) ---

    # Signal at the planet's separation inside the FWHM in [ph/FWHM/mn] (throught coronagraph, if any)
    # [total planet ph/mn] * [planet flux fraction/FWHM] => [ph/FWHM/mn]
    # (l0, Dl, 1, 1)       * (l0, 1, C_raw, IWA)         => (l0, Dl, C_raw, IWA)
    signal_4D = signal_2D[:, :, None, None] * fraction_core_sep_3D[:, None, :, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/FWHM/mn]            * [e-/ph]          => [e-/FWHM/mn]
    # (l0, Dl, C_raw, IWA, 1) * (1, 1, 1, 1, TI) => (l0, Dl, C_raw, IWA, TI)
    signal_5D = signal_4D[:, :, :, :, None] * trans_instru[None, None, None, None, :]

    # Signal at the planet's separation inside the FWHM in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT(throught coronagraph, if any)
    # [e-/FWHM/mn]             * [mn/DIT]                 => [e-/FWHM/DIT]
    # (l0, Dl, C_raw, IWA, TI) * (l0, Dl, C_raw, IWA, TI) => (l0, Dl, C_raw, IWA, TI)
    signal_5D = signal_5D * DIT_5D


    # --- Computing stellar halo photon noise variance (Poisson) in [e-/FWHM/DIT] (l0, Dl, C_raw, IWA, TI) ---

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [ph/px/mn] (throught coronagraph, if any)
    # [total star ph/mn] * [star flux fraction/px] => [ph/px/mn]
    # (l0, Dl, 1, 1)     * (l0, 1, C_raw, IWA)     => (l0, Dl, C_raw, IWA)
    sigma_halo_2_4D = star_flux_2D[:, :, None, None] * PSF_profile_sep_3D[:, None, :, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/px/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]              * [e-/ph]          => [e-/px/mn]
    # (l0, Dl, C_raw, IWA, 1) * (1, 1, 1, 1, TI) => (l0, Dl, C_raw, IWA, TI)
    sigma_halo_2_5D = sigma_halo_2_4D[:, :, :, :, None] * trans_instru[None, None, None, None, :]

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance) (throught coronagraph, if any)
    # [e-/px/mn]               * [px/FWHM]*[mn/DIT]       => [e-/FWHM/DIT]
    # (l0, Dl, C_raw, IWA, TI) * (l0, Dl, C_raw, IWA, TI) => (l0, Dl, C_raw, IWA, TI)
    sigma_halo_2_5D = sigma_halo_2_5D * A_FWHM*DIT_5D


    # --- Computing background photon noise variance (Poisson) in [e-/FWHM/DIT] (l0, Dl, C_raw, IWA, TI) ---

    # Background photon noise variance (Poisson) in [e-/px/mn] with instrumental transmission
    # [ph/px/mn]  * [e-/ph]    => [e-/px/mn]
    # (l0, Dl, 1) * (1, 1, TI) => (l0, Dl, TI)
    sigma_bkg_2_3D = background_flux_2D[:, :, None] * trans_instru[None, None, :]

    # Background photon noise variance (Poisson) in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance)
    # [e-/px/mn]         * [px/FWHM]*[mn/DIT]       => [e-/FWHM/DIT]
    # (l0, Dl, 1, 1, TI) * (l0, Dl, C_raw, IWA, TI) => (l0, Dl, C_raw, IWA, TI)
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

    instru_type   = _SNR_CTX["instru_type"]
    exposure_time = _SNR_CTX["exposure_time"]
    A_FWHM        = _SNR_CTX["A_FWHM"]
    sigma_m       = _SNR_CTX["sigma_m_broadcast"]
    dtype         = _SNR_CTX["dtype"]

    # --- Read one planet slice ---
    signal       = _SNR_CTX["signal_planets"][idx]       # [e-/FWHM/DIT]
    sigma_halo_2 = _SNR_CTX["sigma_halo_2_planets"][idx] # [e-/FWHM/DIT]**2
    sigma_bkg_2  = _SNR_CTX["sigma_bkg_2_planets"][idx]  # [e-/FWHM/DIT]**2
    DIT          = _SNR_CTX["DIT_planets"][idx]          # [mn/DIT]
    if instru_type == "IFU":
        sigma_syst_base_2 = _SNR_CTX["sigma_syst_base_2_planets"][idx] # [e-/FWHM/DIT]**2 (at sigma_m = 1)

    # Detector parameters are arrays of shape (N_l0).  They are broadcast onto the
    # parameter grid.  The l0 axis is axis 1 for IFU arrays and axis 0 for imager arrays.
    N_l0              = _SNR_CTX["N_l0"]
    l0_axis           = _SNR_CTX["l0_axis"]
    shape_l0          = [1] * DIT.ndim
    shape_l0[l0_axis] = N_l0
    shape_l0          = tuple(shape_l0)
    RON0              = np.asarray(_SNR_CTX["RON0"],    dtype=dtype).reshape(shape_l0)
    RON_lim           = np.asarray(_SNR_CTX["RON_lim"], dtype=dtype).reshape(shape_l0)
    DC0               = np.asarray(_SNR_CTX["DC0"],     dtype=dtype).reshape(shape_l0)
    min_DIT           = np.asarray(_SNR_CTX["min_DIT"], dtype=dtype).reshape(shape_l0)
    is_UTR            = np.asarray(_SNR_CTX["is_UTR"],   dtype=bool).reshape(shape_l0)

    # --- Number of DIT ---
    N_DIT = np.floor(exposure_time / DIT).astype(dtype)
    N_DIT = np.clip(N_DIT, 1, None)

    # --- Effective RON variance according to detector readout mode in [e-/FWHM/DIT]**2 ---
    sigma_RON_2 = sigma_RON_2_from_DIT(DIT=DIT, min_DIT=min_DIT, RON0=RON0, RON_lim=RON_lim, A_FWHM=A_FWHM, dtype=dtype, is_UTR=is_UTR)

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



def get_SNR(instru, instru_type, post_processing, exposure_time, min_DIT, RON0, RON_lim, is_UTR, DC0, A_FWHM, Rc, filter_type, signal_planets, sigma_halo_2_planets, sigma_bkg_2_planets, sigma_syst_base_2_planets, DIT_planets, R, Dl, sigma_m, sim_dir, suffix, dtype):

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
    snr_ctx = dict(instru_type=instru_type, post_processing=post_processing, exposure_time=dtype(exposure_time), min_DIT=dtype(min_DIT), RON0=dtype(RON0), RON_lim=dtype(RON_lim), is_UTR=is_UTR, DC0=dtype(DC0), A_FWHM=dtype(A_FWHM), sigma_m_broadcast=sigma_m_broadcast, signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), sigma_syst_base_2_path=str(sigma_syst_base_2_path) if sigma_syst_base_2_path is not None else None, SNR_path=str(SNR_path), shape_SNR=shape_SNR, dtype=np.dtype(dtype).str, N_l0=N_l0, l0_axis=l0_axis)
    if sys.platform.startswith(("win", "darwin")):
        ctx_mp = mp.get_context("spawn")
    else:
        ctx_mp = mp.get_context("fork")

    print()
    with ctx_mp.Pool(processes=min(nproc, N_PT), initializer=init_worker_SNR, initargs=(snr_ctx,)) as pool:
        for _ in tqdm(pool.imap_unordered(process_SNR, range(N_PT), chunksize=chunksize), total=N_PT, desc=f"Computing {instru}({instru_type}+{post_processing}) S/N for each planet and parameters set",):
            pass
    SNR_planets.flush()

    return SNR_planets



# -------------------------------------
# Function for computing detection mask
# -------------------------------------

# Multiprocessing workers for independent planet slices
def init_worker_mask(ctx):
    global _MASK_CTX
    _MASK_CTX = {"SNR": np.load(ctx["SNR_path"], mmap_mode="r"), "mask": np.load(ctx["mask_path"], mmap_mode="r+"), "SNR_thr": ctx["SNR_thr"]}


def process_mask(idx):
    _MASK_CTX["mask"][idx] = _MASK_CTX["SNR"][idx] >= _MASK_CTX["SNR_thr"]
    return idx


def get_mask_detections(SNR_planets, SNR_thr, sim_dir, suffix, target_chunk_mb=2048, nproc=1, chunksize=8, parallel_min_mb=256):
    """Build the detection mask on disk, optionally parallelized over planets.

    Use the existing vectorized/chunked NumPy path for small volumes, since
    reading and writing the arrays is mostly limited by memory/disk bandwidth.
    """

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

    # Limit concurrent memory-mapped I/O; multiprocessing is not always faster here.
    workers = min(max(1, int(nproc)), N_PT, 4)
    use_parallel = workers > 1 and isinstance(SNR_planets, np.memmap) and bytes_per_planet * N_PT >= parallel_min_mb * 1024**2

    print()
    if use_parallel:
        mask_ctx = {"SNR_path": str(SNR_planets.filename), "mask_path": str(mask_detections.filename), "SNR_thr": float(SNR_thr)}
        ctx_mp = mp.get_context("spawn" if sys.platform.startswith(("win", "darwin")) else "fork")
        with ctx_mp.Pool(processes=workers, initializer=init_worker_mask, initargs=(mask_ctx,)) as pool:
            for _ in tqdm(pool.imap_unordered(process_mask, range(N_PT), chunksize=chunksize), total=N_PT, desc="Building mask_detections"):
                pass
    else:
        for i0 in tqdm(range(0, N_PT, chunk_size), desc="Building mask_detections"):
            i1                     = min(i0 + chunk_size, N_PT)
            mask_detections[i0:i1] = SNR_planets[i0:i1] >= SNR_thr

    mask_detections.flush()
    return mask_detections



# --------------------------------------------
# Function for computing detection probability
# --------------------------------------------

# Each worker accumulates a separate FoV-entry bucket into a shared disk memmap.
def init_worker_Pdet(ctx):
    global _PDET_CTX
    _PDET_CTX = dict(ctx)
    _PDET_CTX["mask"]    = np.load(ctx["mask_path"], mmap_mode="r")
    _PDET_CTX["buckets"] = np.load(ctx["buckets_path"], mmap_mode="r+")


def process_Pdet_bucket(p):
    ctx        = _PDET_CTX
    indices    = ctx["groups"][p]
    chunk_size = ctx["chunk_size"]
    bucket     = np.zeros(ctx["shape_grid"], dtype=np.uint32)

    # Only one FoV bucket is held in memory per worker. No worker writes another's bucket.
    for i0 in range(0, len(indices), chunk_size):
        idx     = indices[i0:i0 + chunk_size]
        bucket += np.sum(ctx["mask"][idx], axis=0, dtype=np.uint32)

    ctx["buckets"][p] = bucket
    return p


def get_Pdet(mask_detections, FoV, p0_FoV, dtype, indices=None, target_chunk_mb=2048, jump_before_tqdm=True, valid_windows_3D=None, nproc=1, parallel_min_mb=64):
    """Compute Pdet with an optional FoV-bucket multiprocessing path.

    Results are normalized by the entire selected population, including planets
    outside the FoV. A sequential fallback preserves the original algorithm.
    """
    # If no subset is provided, all planets are included
    if indices is None:
        indices = np.arange(len(p0_FoV), dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)
    N_PT = len(indices)

    shape_grid = mask_detections.shape[1:]  # (...grid..., sigma_m)
    N_FoV      = len(FoV)

    def apply_validity_mask(Pdet):
        if valid_windows_3D is not None:
            Pdet[~valid_windows_3D] = np.nan
        return Pdet

    # A selected population can be empty; keep the output shape unchanged.
    if N_PT == 0:
        return apply_validity_mask(np.zeros(shape_grid + (N_FoV,), dtype=dtype))

    # Precompute FoV gating once
    p0_sel = p0_FoV[indices]
    valid  = (p0_sel < N_FoV)

    if not np.any(valid):
        return apply_validity_mask(np.zeros(shape_grid + (N_FoV,), dtype=dtype))

    indices_valid = indices[valid]
    p0_valid      = p0_sel[valid]

    # Independent FoV-entry groups can be reduced in parallel with no write conflicts.
    # Keep small subsets and ordinary NumPy arrays on the original sequential path.
    grid_size    = int(np.prod(shape_grid, dtype=np.int64))
    groups       = [indices_valid[p0_valid == p] for p in range(N_FoV)]
    active       = [p for p, group in enumerate(groups) if len(group) > 0]
    workers      = min(max(1, int(nproc)), len(active), 4)
    use_parallel = workers > 1 and isinstance(mask_detections, np.memmap) and np.all(p0_valid >= 0) and len(indices_valid) * grid_size >= parallel_min_mb * 1024**2

    if use_parallel:
        # Each worker holds one uint32 grid plus input chunks capped near 128 MiB
        # (or one full planet slice when a single slice exceeds this limit).
        worker_chunk_mb   = min(target_chunk_mb, 128)
        worker_chunk_size = max(1, int(worker_chunk_mb * 1024**2 // max(grid_size, 1)))

        if jump_before_tqdm:
            print()
        # Keep temporary buckets next to the original memmap, and clean on exit.
        with TemporaryDirectory(prefix="tmp_Pdet_", dir=str(Path(mask_detections.filename).parent)) as tmpdir:
            buckets_path = Path(tmpdir) / "buckets.npy"
            buckets      = np.lib.format.open_memmap(buckets_path, mode="w+", dtype=np.uint32, shape=(N_FoV,) + shape_grid)
            buckets[:]   = 0
            buckets.flush()

            pdet_ctx = {"mask_path": str(mask_detections.filename), "buckets_path": str(buckets_path), "groups": groups, "shape_grid": shape_grid, "chunk_size": worker_chunk_size}
            ctx_mp   = mp.get_context("spawn" if sys.platform.startswith(("win", "darwin")) else "fork")
            with ctx_mp.Pool(processes=workers, initializer=init_worker_Pdet, initargs=(pdet_ctx,)) as pool:
                for _ in tqdm(pool.imap_unordered(process_Pdet_bucket, active, chunksize=1), total=len(active), desc="Computing detection probability by FoV bins"):
                    pass

            # A planet entering at p0 contributes to every FoV bin >= p0.
            # uint64 accumulation is consistent with NumPy's default cumsum on uint32.
            counts = np.cumsum(buckets, axis=0, dtype=np.uint64)
            Pdet   = np.ascontiguousarray(np.moveaxis(counts, 0, -1) / N_PT, dtype=dtype)
            del counts, buckets

        return apply_validity_mask(Pdet)

    # Sequential fallback: buckets[..., p] accumulate planets first visible at p.
    buckets = np.zeros(shape_grid + (N_FoV,), dtype=np.uint32)

    # Approximate chunk size from memory budget:
    # only one boolean chunk is materialized now
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

    return apply_validity_mask((N_detections / N_PT).astype(dtype))



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
    tol = 64*np.finfo(float).eps * max(abs(pmin), abs(pmax), abs(axis[0]), abs(axis[-1]), np.finfo(float).tiny)
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
    if np.isclose(x0, axis[0], rtol=0, atol=32*np.finfo(float).eps*max(abs(axis[0]), np.finfo(float).tiny)):
        return np.take(values, 0, axis=axis_id)
    if np.isclose(x0, axis[-1], rtol=0, atol=32*np.finfo(float).eps*max(abs(axis[-1]), np.finfo(float).tiny)):
        return np.take(values, len(axis) - 1, axis=axis_id)

    # Exact grid point
    idx_exact = np.where(np.isclose(axis, x0, rtol=0, atol=32*np.finfo(float).eps*np.maximum(np.abs(axis), np.finfo(float).tiny)))[0]
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
    v0      = np.take(values, i0, axis=axis_id)
    v1      = np.take(values, i1, axis=axis_id)

    return (1.0 - alpha) * v0 + alpha * v1



def reduce_hcube(hcube, dims_to_keep, params, params_ranges, params_priors, params_names, verbose=False, dtype=None):
    """
    Reduce an N-dimensional grid by fixing and/or marginalizing parameters.

    Invalid configurations (NaN or Inf) contribute neither values nor weights.
    Numerator and denominator are reduced independently and normalized only
    once, ensuring correct joint marginalization over correlated validity domains.

    The input dtype is preserved by default to limit memory consumption.

    Parameters
    ----------
    hcube : ndarray
        N-dimensional grid (e.g. detection probability or SNR).
    dims_to_keep : list of int
        Indices of the dimensions to preserve.
    params : list of 1D arrays
        Parameter axes, in the same order as hcube dimensions.
    params_ranges : list of tuples
        (pmin, pmax) for each parameter:
        - pmin == pmax: fix by linear interpolation.
        - pmin < pmax: marginalize over the specified interval.
    params_priors : list of str
        "uniform" or "log-uniform" for each parameter.
    params_names : list of str
        Names of the parameters.
    verbose : bool
        Print reduction information.
    dtype : numpy dtype, optional
        Output and computation dtype. Defaults to hcube.dtype.

    Returns
    -------
    hcube_red : ndarray
        Reduced grid, with NaN where no valid configurations contribute.
    """

    # --- Parameters ---
    dims_to_keep  = sorted(list(dims_to_keep))
    params        = list(params)
    params_names  = list(params_names)
    params_ranges = list(params_ranges)
    params_priors = list(params_priors)
    nparam        = len(params)

    # --- Input validation ---
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
    if len(set(dims_to_keep)) != len(dims_to_keep):
        raise ValueError("dims_to_keep contains duplicate indices.")

    # --- Preserve FastYield dtype ---
    if dtype is None:
        dtype = np.asarray(hcube).dtype
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("reduce_hcube requires a floating-point dtype.")
    hcube          = np.asarray(hcube, dtype=dtype)
    dims_to_reduce = [idim for idim in range(nparam) if idim not in dims_to_keep]

    # --- Initialize numerator and denominator ---
    # np.where creates an independent array; the input hcube is never modified.
    valid       = np.isfinite(hcube)
    numerator   = np.where(valid, hcube, 0.0)
    denominator = valid.astype(dtype)
    del valid

    # --- Initial print ---
    if verbose:
        shape_ini = tuple(len(p) for p in params)

        print("\nHypercube reduction")
        print("-------------------")
        print(f"\nInitial axes: ({', '.join(params_names)}) = {shape_ini}\n")

        for idim in range(nparam):
            if idim in dims_to_keep:
                print(f"  - Keeping       {params_names[idim]:<20} axis")

    # --- Reduce dimensions in reverse order ---
    for idim in sorted(dims_to_reduce, reverse=True):
        axis       = np.asarray(params[idim], dtype=float)
        pmin, pmax = params_ranges[idim]
        prior      = params_priors[idim]
        name       = params_names[idim]
        if pmin > pmax:
            raise ValueError(f"For parameter '{name}', pmin must be <= pmax.")
        if pmin < axis[0] or pmax > axis[-1]:
            raise ValueError(f"For parameter '{name}', requested range [{pmin:g}, {pmax:g}] is outside axis range [{axis[0]:g}, {axis[-1]:g}].")

        # --- Fix parameter by linear interpolation ---
        if pmin == pmax:
            numerator   = np.asarray(interp_along_axis(numerator,   axis=axis, x0=pmin, axis_id=idim), dtype=dtype)
            denominator = np.asarray(interp_along_axis(denominator, axis=axis, x0=pmin, axis_id=idim), dtype=dtype)
            if verbose:
                print(f"  - Fixing        {name:<20} at {pmin:<8g} with linear interpolation")

        # --- Marginalize with truncated-cell weights ---
        else:
            # The selected cells form a contiguous interval.
            # Slicing avoids the potentially large copies created by np.take.
            idx, w = get_axis_weights_in_range(axis=axis, pmin=pmin, pmax=pmax, prior=prior)
            if len(idx) != idx[-1] - idx[0] + 1:
                raise ValueError(f"Non-contiguous integration cells for '{name}'.")
            sl               = [slice(None)] * numerator.ndim
            sl[idim]         = slice(int(idx[0]), int(idx[-1]) + 1)
            numerator_view   = numerator[tuple(sl)]
            denominator_view = denominator[tuple(sl)]

            # Broadcast integration weights using the FastYield dtype.
            shape_w       = [1] * numerator_view.ndim
            shape_w[idim] = len(w)
            w             = np.asarray(w, dtype=dtype).reshape(shape_w)

            # Multiply in place to avoid allocating large temporary cubes.
            # These are private working arrays, not the original input hcube.
            np.multiply(numerator_view,   w, out=numerator_view)
            np.multiply(denominator_view, w, out=denominator_view)

            # Accumulate and remove the marginalized dimension.
            numerator   = np.sum(numerator_view,   axis=idim, dtype=dtype)
            denominator = np.sum(denominator_view, axis=idim, dtype=dtype)
            if verbose:
                print(f"  - Marginalizing {name:<20} in [{pmin:<8g}, {pmax:<8g}] ({len(idx)} cells) with {prior} prior")

    # --- Final normalization ---
    hcube_red = np.full_like(numerator, np.nan, dtype=dtype)
    np.divide(numerator, denominator, out=hcube_red, where=denominator > 0)

    # --- Final print ---
    if verbose:
        params_names_red = [params_names[idim]     for idim in dims_to_keep]
        shape_end        = tuple(len(params[idim]) for idim in dims_to_keep)
        print(f"\nFinal axes: ({', '.join(params_names_red)}) = {shape_end}\n")

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
    print(f"\nMarginalized HWO planet table saved to:\n{output_path}")

    return output_path



# --------------------------------------------
# Plot/prints helpers
# --------------------------------------------

# Marginalized plots
def convert_Pdet_to_plot_quantity(Pdet_curve, N_PT, gain, ymax_panel=None):
    "Normalize to the panel maximum or convert to yield"
    y = np.asarray(Pdet_curve, dtype=float)
    if gain:
        y = 100.0*y/ymax_panel if ymax_panel is not None and np.isfinite(ymax_panel) and ymax_panel > 0 else np.zeros_like(y)
    else:
        y = y*N_PT
    return y

def get_panel_ymax(curves):
    values = np.concatenate([np.asarray(curve, dtype=float).ravel() for curve in curves]) if curves else np.array([])
    return np.nanmax(values) if np.any(np.isfinite(values)) else None

def set_log_decade_ticks(axis, vmin, vmax, max_labels=4):
    # Major ticks at every decade (the default LogLocator drops most of them on small panels spanning many decades),
    # with labels only every few decades so that they fit in the panel
    exps = np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax)) + 1).astype(int)
    step = max(1, int(np.ceil(len(exps) / max_labels)))
    axis.set_major_locator(FixedLocator(10.0**exps))
    axis.set_major_formatter(FuncFormatter(lambda v, pos: rf"$\mathdefault{{10^{{{round(np.log10(v))}}}}}$" if round(np.log10(v)) % step == 0 else ""))
    axis.set_minor_locator(NullLocator())



# ----------------------------------------------------
# Population plots: precomputed weights and parallelism
# ----------------------------------------------------

def format_range(r):
    return f"{r[0]:g}" if np.isclose(r[0], r[1]) else f"[{r[0]:g}, {r[1]:g}]"



def format_value(value):
    if value is None:
        return "None"
    value = np.asarray(value).ravel()
    return f"{value[0]:g}" if value.size == 1 else f"[{value[0]:g}, {value[1]:g}]"



def print_range(key, value, used=None, warning=""):
    used_str = f"-> {used}" if used is not None else ""
    print(f"  {key+'_plot':<18} = {format_value(value):<14} {used_str:<14} {warning}")



def prepare_population_axis(axis, value, prior, name, verbose=False):
    """
    Select interpolation or marginalization weights for a population parameter.

    Scalar     : fixed value, linearly interpolated between adjacent nodes.
    [min, max] : weighted marginalization over the requested interval.
    None       : marginalize over the full simulated parameter range.

    Requests outside the explored range are cropped with a warning. A singleton
    axis is always fixed, independently of the requested value.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise ValueError(f"Invalid population axis for {name}.")

    # A singleton dimension was fixed by the instrument concept itself.
    if axis.size == 1:
        fixed_value = float(axis[0])
        requested   = np.atleast_1d(value).ravel() if value is not None else np.array([])
        changed =    value is not None and (requested.size != 1 or not np.isclose(requested[0], fixed_value))
        if verbose:
            print_range(name, value, f"{fixed_value:g}" if changed else None, f"(WARNING: parameter fixed at {fixed_value:g})")
        return np.array([0], dtype=int), np.array([1.0]), True, fixed_value

    # None or invalid selections fall back to the full simulated axis.
    original = (axis[0], axis[-1]) if value is None else value
    try:
        requested = np.asarray(original, dtype=float).ravel()
    except (TypeError, ValueError):
        requested = np.array([])
    
    warning = ""
    if requested.size not in (1, 2) or not np.all(np.isfinite(requested)):
        requested = np.array([axis[0], axis[-1]], dtype=float)
        warning   = "(WARNING: expected scalar or [min, max]; full axis used)"
    if requested.size == 1:
        requested = np.repeat(requested, 2)
    if requested[0] > requested[1]:
        requested = requested[::-1]
        warning   = "(WARNING: reversed range reordered)"

    used    = np.clip(requested, axis[0], axis[-1])
    cropped = not np.allclose(requested, used, rtol=0, atol=64*np.finfo(float).eps*max(abs(axis[0]), abs(axis[-1]), np.finfo(float).tiny))
    if cropped:
        warning = f"(WARNING: cropped to [{axis[0]:g}, {axis[-1]:g}])"
    pmin, pmax = map(float, used)
    if verbose:
        changed = value is None or cropped or warning.startswith("(WARNING: expected") or warning.startswith("(WARNING: reversed")
        print_range(name, value, format_range((pmin, pmax)) if changed else None, warning)

    tol = 64*np.finfo(float).eps * max(abs(axis[0]), abs(axis[-1]), np.finfo(float).tiny)
    if np.isclose(pmin, pmax, rtol=0, atol=tol):
        x0    = 0.5 * (pmin + pmax)
        exact = np.flatnonzero(np.isclose(axis, x0, rtol=0, atol=tol))
        if exact.size:
            return np.array([int(exact[0])]), np.array([1.0]), True, x0
        i1    = int(np.searchsorted(axis, x0))
        i0    = i1 - 1
        alpha = (x0 - axis[i0]) / (axis[i1] - axis[i0])
        return np.array([i0, i1]), np.array([1.0 - alpha, alpha]), True, x0

    idx, weights = get_axis_weights_in_range(axis=axis, pmin=pmin, pmax=pmax, prior=prior)
    weights      = np.asarray(weights, dtype=float)
    weights     /= np.sum(weights)
    return np.asarray(idx, dtype=int), weights, False, (pmin, pmax)



_POP_CTX = None

def init_worker_population(ctx):
    """Open the raw per-planet memmaps once in each worker."""
    global _POP_CTX
    _POP_CTX = ctx
    for key in ("signal", "halo", "background", "DIT", "systematics"):
        path = ctx.get(f"{key}_path")
        if path is not None:
            _POP_CTX[key] = np.load(path, mmap_mode="r")



def process_population(idx):
    """Return the mean SNR and five noise variances for one planet."""
    ctx              = _POP_CTX
    indices, weights = ctx["indices"], ctx["weights"]
    rest_names       = ("Dl", "C_raw", "IWA", "trans_instru")
    rest_idx         = [indices[name] for name in rest_names]
    rest_weight      = ctx["rest_weight"]
    sigma_m_percent  = ctx["sigma_m_percent"]
    sigma_m_2        = (sigma_m_percent / 100.0)**2
    vis              = ctx["visibility"][idx]

    if vis <= 0:
        return idx, (0.0, np.nan, np.nan, np.nan, np.nan, np.nan)

    numerator   = np.zeros(6, dtype=float)
    denominator = 0.0

    R_idx     = indices["R"] if ctx["instru_type"] == "IFU" else [None]
    R_weights = weights["R"] if ctx["instru_type"] == "IFU" else [1.0]
    for ir, wr in zip(R_idx, R_weights):
        for il0, wl0 in zip(indices["l0"], weights["l0"]):
            position   = (idx, int(ir), int(il0)) if ir is not None else (idx, int(il0))
            selection  = np.ix_(*rest_idx)
            signal     = np.asarray(ctx["signal"][position][selection], dtype=float)
            halo       = np.asarray(ctx["halo"][position][selection], dtype=float)
            background = np.asarray(ctx["background"][position][selection], dtype=float)
            DIT        = np.asarray(ctx["DIT"][position][selection], dtype=float)
            if ctx["instru_type"] == "IFU":
                syst_base      = np.asarray(ctx["systematics"][position][selection], dtype=float)
                spectral_valid = ctx["valid_windows"][int(ir), int(il0), indices["Dl"]][:, None, None, None, None]
            elif ctx["instru_type"] == "imager":
                syst_base      = halo**2
                spectral_valid = True

            valid_DIT = np.isfinite(DIT) & (DIT > 0)
            DIT_safe  = np.where(valid_DIT, DIT, 1.0)
            N_DIT     = np.clip(np.floor(ctx["exposure_time"] / DIT_safe), 1, None)
            ron       = sigma_RON_2_from_DIT(DIT=DIT_safe, min_DIT=ctx["min_DIT"][il0], RON0=ctx["RON0"][il0], RON_lim=ctx["RON_lim"][il0], A_FWHM=ctx["A_FWHM"], dtype=float, is_UTR=ctx["is_UTR"][il0])
            dark      = ctx["DC0"][il0] * DIT_safe * ctx["A_FWHM"]

            vhalo    = N_DIT * halo
            vbkg     = N_DIT * background
            vron     = N_DIT * ron
            vdark    = N_DIT * dark
            vsyst    = N_DIT[..., None]**2 * syst_base[..., None] * sigma_m_2
            vfund    = vhalo + vbkg + vron + vdark
            variance = vfund[..., None] + vsyst
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                snr = N_DIT[..., None] * signal[..., None] / np.sqrt(variance)

            valid = valid_DIT[..., None] & spectral_valid & np.isfinite(snr) & np.isfinite(variance) & (variance > 0)
            if not np.any(valid):
                continue
            w             = (wr * wl0) * rest_weight
            valid_weights = np.where(valid, w, 0.0)
            denominator  += np.sum(valid_weights, dtype=float)
            numerator[0] += np.sum(np.where(valid, snr, 0.0) * valid_weights, dtype=float)
            for j, v in enumerate((vhalo, vbkg, vron, vdark), start=1):
                numerator[j] += np.sum(np.where(valid, v[..., None], 0.0) * valid_weights, dtype=float)
            numerator[5] += np.sum(np.where(valid, vsyst, 0.0) * valid_weights, dtype=float)

    if denominator <= 0:
        return idx, (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    numerator    /= denominator
    numerator[0] *= vis
    return idx, tuple(numerator)



def get_population_results(instru_type, signal_planets, sigma_halo_2_planets, sigma_bkg_2_planets, sigma_syst_base_2_planets, DIT_planets, separation_planets, axes, prepared, detector_l0, RON0, RON_lim, DC0, min_DIT, is_UTR, exposure_time, A_FWHM, valid_windows_3D=None, nproc=None, chunksize=8):
    """Compute population diagnostics without building the full SNR hypercube."""
    names   = ("R", "l0", "Dl", "C_raw", "IWA", "trans_instru", "sigma_m", "FoV") if instru_type == "IFU" else ("l0", "Dl", "C_raw", "IWA", "trans_instru", "sigma_m", "FoV")
    axes    = dict(zip(names, axes))
    indices = {name: prepared[name][0] for name in names}
    weights = {name: prepared[name][1] for name in names}

    # Keep the legacy interpolation behavior, but identify mixed detector families.
    if prepared["l0"][2] and len(set(np.asarray(detector_l0)[indices["l0"]])) > 1:
        print("WARNING: l0_plot interpolation mixes two detector technologies. The resulting SNR is only an approximation.")

    if instru_type == "IFU":
        if valid_windows_3D is None:
            raise ValueError("valid_windows_3D is required for IFU population diagnostics.")
        r_idx, l_idx, d_idx = (indices[k] for k in ("R", "l0", "Dl"))
        submask             = np.asarray(valid_windows_3D)[np.ix_(r_idx, l_idx, d_idx)]
        positive_weights    = (weights["R"][:, None, None] * weights["l0"][None, :, None] * weights["Dl"][None, None, :]) > 0
        if not np.any(submask & positive_weights):
            print("WARNING: no valid IFU spectral configurations for the requested *_plot values; population SNRs will be undefined.")
            count = len(separation_planets)
            undefined = np.full(count, np.nan, dtype=float)
            return dict(SNR=undefined.copy(), dominant_noise=np.full(count, "Unknown", dtype=object), var_halo=undefined.copy(), var_bkg=undefined.copy(), var_RON=undefined.copy(), var_DC=undefined.copy(), var_syst=undefined.copy())
        if np.any(~submask & positive_weights):
            print("WARNING: some IFU spectral interpolation/marginalization nodes are invalid and will be excluded with weight renormalization.")

    if prepared["FoV"][2]:
        visibility = (2 * np.asarray(separation_planets, dtype=float) <= float(prepared["FoV"][3])).astype(float)
    else:
        fov_values = axes["FoV"][indices["FoV"]]
        visibility = np.sum(weights["FoV"][:, None] * (2 * np.asarray(separation_planets)[None, :] <= fov_values[:, None]), axis=0)

    rest_names  = ("Dl", "C_raw", "IWA", "trans_instru", "sigma_m")
    rest_shape  = tuple(len(indices[name]) for name in rest_names)
    rest_weight = np.ones(rest_shape, dtype=float)
    for dim, name in enumerate(rest_names):
        shape        = [1] * len(rest_names)
        shape[dim]   = len(indices[name])
        rest_weight *= weights[name].reshape(shape)

    ctx       = dict(instru_type=instru_type, exposure_time=float(exposure_time), A_FWHM=float(A_FWHM), indices=indices, weights=weights, rest_weight=rest_weight, sigma_m_percent=np.asarray(axes["sigma_m"])[indices["sigma_m"]], visibility=visibility, valid_windows=valid_windows_3D, RON0=np.asarray(RON0, dtype=float), RON_lim=np.asarray(RON_lim, dtype=float), DC0=np.asarray(DC0, dtype=float), min_DIT=np.asarray(min_DIT, dtype=float), is_UTR=np.asarray(is_UTR, dtype=bool), signal_path=str(signal_planets.filename), halo_path=str(sigma_halo_2_planets.filename), background_path=str(sigma_bkg_2_planets.filename), DIT_path=str(DIT_planets.filename), systematics_path=str(sigma_syst_base_2_planets.filename) if instru_type == "IFU" else None)
    n_planets = len(separation_planets)
    results   = np.full((n_planets, 6), np.nan, dtype=float)
    workers   = max(1, min(n_planets, (cpu_count() - 3 if nproc is None else nproc)))
    if workers == 1:
        init_worker_population(ctx)
        for i in tqdm(range(n_planets), desc="Computing population diagnostics"):
            idx, values  = process_population(i)
            results[idx] = values
    else:
        mp_ctx = mp.get_context("spawn" if sys.platform.startswith(("win", "darwin")) else "fork")
        with mp_ctx.Pool(processes=workers, initializer=init_worker_population, initargs=(ctx,)) as pool:
            for idx, values in tqdm(pool.imap_unordered(process_population, range(n_planets), chunksize=chunksize), total=n_planets, desc="Computing population diagnostics"):
                results[idx] = values

    labels          = np.array(["Stellar halo", "Background", "Read noise", "Dark current", "Systematics"], dtype=object)
    noise           = results[:, 1:]
    known           = np.any(np.isfinite(noise), axis=1) & (visibility > 0)
    dominant        = np.full(n_planets, "Unknown", dtype=object)
    dominant[known] = labels[np.nanargmax(noise[known], axis=1)]
    return dict(SNR=results[:, 0], dominant_noise=dominant, var_halo=results[:, 1], var_bkg=results[:, 2], var_RON=results[:, 3], var_DC=results[:, 4], var_syst=results[:, 5])


