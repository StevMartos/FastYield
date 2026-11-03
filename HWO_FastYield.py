import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["NUMBA_NUM_THREADS"]    = "1"

# import FastYield modules
# import fastyield.config as config
# config.dpi_fig = 72
from fastyield.config import rad2arcsec, h, c, R0_min, R0_max, lmin_bands, lmax_bands, get_sim_data_path, dpi_fig
from fastyield.get_specs import get_detector_specs
from fastyield.FastYield import load_planet_table, get_mask_planet_type, yield_population_plot, yield_heatmap_ELT
from fastyield.FastYield_helpers import print_simulation_summary, make_suffix, write_meta, memmap_nbytes, format_nbytes, create_memmap_with_log, warmup_signal_noise_numba, get_valid_IFU_windows
from fastyield.spectrum import get_wavelength_axis_constant_R, Spectrum, load_vega_spectrum, get_spectrum_contribution_name_model, get_wave_K, get_counts_from_density

# import HWO package helpers module
import HWO_FastYield_helpers as fyh

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

# import other modules
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import cpu_count
from pathlib import Path
import sys
import gc
import inspect

# Parameters for multiprocessing
nproc     = max(cpu_count()-3, 1)
chunksize = 8



# %%
def main():

    # ---------------------------
    # Parameters
    # ---------------------------
    HWO_CODE_VERSION = "HWO_FastYield_v2.0"
    dtype            = np.float32

    # --- HWO telescope (prospective architecture; no fixed flight baseline) ---
    instru     = "HWO"
    D          = 6.0                         # [m] reference off-axis aperture
    eps_D      = 0.0                         # projected central-obscuration diameter / D
    S          = np.pi*(D/2)**2*(1-eps_D**2) # [m2] geometric collecting area
    N_mirror   = 3                           # provisional OTA reflective surfaces
    trans_dust = 1.0                         # no ground atmospheric dust term

    # --- Detector assignment ---
    # HWO_conservative: UV_MCP (<0.30 um), EMCCD (0.30-0.95 um), H4RG (>=0.95 um)
    # HWO_photon_counting: UV_MCP / SKIPPER_CCD / MKID (technology study only)
    # constant_<detector_name>: user-defined single detector from get_detector_specs()
    detector = "HWO_conservative"

    # --- Paths ---
    instru_dir = get_sim_data_path() / instru
    local_dir  = Path(__file__).resolve().parent / "data" / instru
    if not (instru_dir / "PSF_simulations").is_dir() and (local_dir / "PSF_simulations").is_dir():
        instru_dir = local_dir  # compatibility with the previous HWO local data layout
    sim_dir = instru_dir / "FastYield_simulations"
    psf_dir = instru_dir / "PSF_simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # --- Coronagraphic PSF library ---
    coronagraph = "APLC"
    apodizer    = "APLC"
    strehl      = "NO_ABERRATION" # space-telescope label, not an AO Strehl
    psf_suffix  = None            # set to exact FITS suffix if several libraries match

    # --- Instrument / science ---
    instru_type       = "imager"               # "IFU" or "imager"
    post_processing   = "DI"                   # "MM" or "DI" (imager forces DI)
    SNR_thr           = 5
    exposure_time     = 100*60                 # [mn]
    force_new_calc    = False
    thermal_model     = "auto"
    reflected_model   = "auto"
    spatial_sampling  = 3                      # [px/FWHM]; must match the PSF extraction convention
    core_box_width_lD = 1.029                 # [lambda/D]; width of the 1-FWHM square photometric box
    A_FWHM            = spatial_sampling**2    # [px] HWO legacy FWHM BOX; must match fraction_core FITS convention
    Rc                = 1_000
    filter_type       = "gaussian_fast"
    table_type        = "Archive"

    # PSF density/fraction_core must use the same aperture definition as A_FWHM.
    # In particular, DO NOT substitute the 2.4 lambda/D PCS circular aperture
    # unless the entire HWO PSF library is recomputed using that convention.

    # --- Native HWO PSF angular coverage ---
    sep_min = 0       # [mas]
    sep_max = 1_000   # [mas]

    # --- HWO prospective coronagraph axes ---
    C_raw_min, C_raw_max = 10.0**(-10.75), 10.0**(-7.0) # target raw dark-hole contrast
    IWA_min, IWA_max     = 2.0, 4.5                   # [lambda/D]

    # --- Global and population plots ---
    plot_global_results     = True
    plot_populations        = True
    light_regime_plot       = "thermal+reflected"
    band_plot               = "Y"
    planet_population_space = "phys" # "obs", "phys", "both"

    # scalar=fixed/interpolated, [min,max]=marginalized, None=full exploration range.
    R_plot            = None
    l0_plot           = None
    Dl_plot           = None
    C_raw_plot        = None
    IWA_plot          = None
    trans_instru_plot = None
    sigma_m_plot      = None # [%] : plot axes use 100*sigma_m
    FoV_plot          = None

    # --- Spectral / instrumental exploratory ranges ---
    # The HWO concept is NOT fixed; these are design-trade hypotheses.
    N                  = 3
    R_min, R_max       = 1_000, 10_000
    l0_min, l0_max     = 0.2, 2.0              # [um] restricted by the FITS wavelength axis
    Dl_min, Dl_max     = 0.02, 0.3             # [um] used for IFU and imager
    Nl_max             = 10_000               # IFU max spectral channels; None => unrestricted
    trans_instru_min   = 0.01                  # [e-/ph]
    trans_instru_max   = 0.50
    sigma_m_min        = 1e-5 if post_processing == "MM" else 1e-3
    sigma_m_max        = 1e-1
    FoV_min, FoV_max   = 10, 2*sep_max         # [mas], FoV interpreted as full diameter

    if instru_type == "IFU":
        R       = np.unique(np.logspace(np.log10(R_min), np.log10(R_max), N))
        N_R     = len(R)
        R_model = R0_max
    elif instru_type == "imager":
        post_processing = "DI"
        R_model         = R0_min
    else:
        raise ValueError("instru_type must be 'IFU' or 'imager'.")

    l0           = np.unique(np.linspace(l0_min, l0_max, N))
    Dl           = np.unique(np.linspace(Dl_min, Dl_max, N))
    C_raw        = np.unique(np.logspace(np.log10(C_raw_min), np.log10(C_raw_max), N))
    IWA          = np.unique(np.linspace(IWA_min, IWA_max, N))
    trans_instru = np.unique(np.linspace(trans_instru_min, trans_instru_max, N))
    sigma_m      = np.unique(np.logspace(np.log10(sigma_m_min), np.log10(sigma_m_max), N))
    FoV          = np.unique(np.logspace(np.log10(FoV_min), np.log10(FoV_max), N))
    N_l0, N_Dl, N_C_raw, N_IWA, N_TI, N_SM, N_FoV = map(len, (l0, Dl, C_raw, IWA, trans_instru, sigma_m, FoV))

    # --- Detector specifications at l0 (single family per central band) ---
    if detector.startswith("constant_"):
        detector_l0 = np.full(N_l0, detector.replace("constant_", "", 1))
    elif detector == "HWO_conservative":
        detector_l0 = np.select([l0 < 0.30, (l0 >= 0.30) & (l0 < 0.95), l0 >= 0.95], ["UV_MCP", "EMCCD", "H4RG"], default="UNKNOWN")
    elif detector == "HWO_photon_counting":
        detector_l0 = np.select([l0 < 0.30, (l0 >= 0.30) & (l0 < 0.95), l0 >= 0.95], ["UV_MCP", "SKIPPER_CCD", "MKID"], default="UNKNOWN")
    else:
        raise ValueError(f"Unknown HWO detector mode: {detector}")
    if np.any(detector_l0 == "UNKNOWN"):
        raise ValueError("Unassigned detector wavelength band")
    spec_dict    = {name: get_detector_specs(name) for name in np.unique(detector_l0)}
    RON0         = np.array([spec_dict[d]["RON0"]                    for d in detector_l0], dtype=dtype)
    RON_lim      = np.array([spec_dict[d]["RON_lim"]                 for d in detector_l0], dtype=dtype)
    DC0          = np.array([spec_dict[d]["DC0"]                     for d in detector_l0], dtype=dtype)
    saturation_e = np.array([spec_dict[d]["saturation_e"]            for d in detector_l0], dtype=dtype)
    min_DIT      = np.array([spec_dict[d]["min_DIT"]                 for d in detector_l0], dtype=dtype)
    max_DIT      = np.array([spec_dict[d]["max_DIT"]                 for d in detector_l0], dtype=dtype)
    is_UTR       = np.array([spec_dict[d]["family"] == "HgCdTe_HxRG" for d in detector_l0], dtype=bool)

    # --- Pixel scale and photon-counting aperture ---
    l0D          = l0*1e-6 / D * 1000*rad2arcsec # [mas]
    FWHM0        = 1.029 * l0D
    pxscales     = FWHM0 / spatial_sampling      # [mas/px]
    pxscales_rad = pxscales / (rad2arcsec*1000)  # [rad/px]

    # Residuals modulations considered for the post_processing
    residuals = "Systematics" if post_processing == "MM" else "Speckles"

    # Getting labels
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Valid spectral format windows
    if instru_type == "IFU":
        valid_windows_3D = get_valid_IFU_windows(R=R, l0=l0, Dl=Dl, Nl_max=Nl_max)
        if not np.any(valid_windows_3D):
            raise ValueError("No valid IFU configurations with the requested Nl_max.")
    else:
        valid_windows_3D = None



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
    if N_PT == 0:
        raise ValueError("No planets remain after the separation selection")

    # Prints simulation parameters
    # FastYield releases may call the HWO coronagraph axis either C_raw or WFE.
    # Forward only keywords recognized by the installed shared summary helper.
    summary_kwargs = dict(instru=instru, instru_type=instru_type, detector=detector, post_processing=post_processing, D=D, S=S, N_mirror=N_mirror, trans_dust=trans_dust, RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, thermal_model=thermal_model, reflected_model=reflected_model, SNR_thr=SNR_thr, exposure_time=exposure_time, spatial_sampling=spatial_sampling, size_core=spatial_sampling, A_FWHM=A_FWHM, Rc=Rc if instru_type == "IFU" else None, filter_type=filter_type if instru_type == "IFU" else None, table_type=table_type, sep_min=sep_min, sep_max=sep_max, N_PT_raw=N_PT_raw, N_PT=N_PT, force_new_calc=force_new_calc, l0=l0, R=R if instru_type == "IFU" else None, Dl=Dl, C_raw=C_raw, IWA=IWA, IWA_unit="lambda/D", trans_instru=trans_instru, sigma_m=sigma_m, FoV=FoV)
    accepted       = inspect.signature(print_simulation_summary).parameters
    if "WFE" in accepted and "C_raw" not in accepted:
        summary_kwargs["WFE"] = C_raw # legacy summary name; values remain dimensionless raw contrast
    if "r_WFE" in accepted and "C_raw" not in accepted:
        summary_kwargs["r_WFE"] = C_raw
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        summary_kwargs = {key: value for key, value in summary_kwargs.items() if key in accepted}
    print_simulation_summary(**summary_kwargs)

    # HWO PSF response is independent of the host-star magnitude by design.
    # Ensure the raw HWO library is resolved before computing the cache key.
    suffix_PSF = fyh.find_hwo_psf_suffix(psf_dir, instru, apodizer, coronagraph, D, suffix_override=psf_suffix)
    print("\nUsing HWO PSF library:", suffix_PSF)

    # Build meta for YOUR case (IFU or imager)
    meta = dict(code_version=HWO_CODE_VERSION, dtype=str(dtype),

                # instrument / telescope
                instru=instru, D=D, S=S, eps_D=eps_D, N_mirror=N_mirror, trans_dust=trans_dust, detector=detector, detector_l0=detector_l0,

                # detector numeric specs that affect DIT/noise/products
                RON0=RON0, RON_lim=RON_lim, DC0=DC0, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT,

                # science / pipeline
                coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, psf_suffix=suffix_PSF, thermal_model=thermal_model, reflected_model=reflected_model, instru_type=instru_type, post_processing=post_processing, spatial_sampling=spatial_sampling, core_box_width_lD=core_box_width_lD, A_FWHM=A_FWHM, R_model=R_model, table_type=table_type,

                # exploration space
                l0_min=l0_min, l0_max=l0_max, C_raw_min=C_raw_min, C_raw_max=C_raw_max, IWA_min=IWA_min, IWA_max=IWA_max, trans_instru_min=trans_instru_min, trans_instru_max=trans_instru_max, sep_min=sep_min, sep_max=sep_max, N_PT=N_PT,

                # axis used
                l0=l0, Dl=Dl, C_raw=C_raw, IWA=IWA, trans_instru=trans_instru,
                )

    if instru_type == "IFU":
        meta.update(dict(Rc=Rc, filter_type=filter_type, R=R, Nl_max=Nl_max))
    suffix, meta_clean, _payload = make_suffix(meta, n=16)
    sim_dir_concept              = sim_dir / f"{instru_type}_{post_processing}"
    sim_dir_concept.mkdir(parents=True, exist_ok=True)
    write_meta(sim_dir_concept, suffix, meta_clean, suffix_as_prefix=True)
    print("\nFastYield data files suffix:", suffix)

    # File paths derived from suffix
    signal_path                 = sim_dir_concept / f"{suffix}_signal.npy"
    sigma_halo_2_path           = sim_dir_concept / f"{suffix}_sigma_halo_2.npy"
    sigma_bkg_2_path            = sim_dir_concept / f"{suffix}_sigma_bkg_2.npy"
    DIT_path                    = sim_dir_concept / f"{suffix}_DIT.npy"
    if instru_type == "IFU":
        sigma_syst_base_2_path  = sim_dir_concept / f"{suffix}_sigma_syst_base_2.npy"
    else:
        sigma_syst_base_2_path  = None
    PSF_profile_4D_tmp_path     = sim_dir_concept / f"{suffix}_tmp_PSF_profile_4D.npy"
    fraction_core_4D_tmp_path   = sim_dir_concept / f"{suffix}_tmp_fraction_core_4D.npy"
    PSF_profile_max_3D_tmp_path = sim_dir_concept / f"{suffix}_tmp_PSF_profile_max_3D.npy"

    for p in [PSF_profile_4D_tmp_path, fraction_core_4D_tmp_path, PSF_profile_max_3D_tmp_path]:
        if p.exists():
            p.unlink()

    do_new = force_new_calc or (not signal_path.exists()) or (not sigma_halo_2_path.exists()) or (not sigma_bkg_2_path.exists()) or (not DIT_path.exists())
    if instru_type == "IFU":
        do_new = do_new or (not sigma_syst_base_2_path.exists())



    ####################
    # Opening existing #
    ####################
    if not do_new:
        # For IFU (7D):    (Planets, R, l0, Dl, C_raw, IWA, trans_instru)
        # For Imager (6D): (Planets,    l0, Dl, C_raw, IWA, trans_instru)
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
            print(f"\nNew FastYield {instru} calculations (forcing_new_calc=True).\n")
        else:
            print(f"\nNew FastYield {instru} calculations (missing files).\n")

        # --- HWO PSF products (l0, C_raw, IWA_lD, separation_mas) ---
        wave0                   = fits.getdata(psf_dir / f"{instru}_wave_{suffix_PSF}.fits")
        C_raw0                  = fits.getdata(psf_dir / f"{instru}_C_{suffix_PSF}.fits")
        IWA0                    = fits.getdata(psf_dir / f"{instru}_IWA_lD_{suffix_PSF}.fits")
        separation0             = fits.getdata(psf_dir / f"{instru}_separation_mas_{suffix_PSF}.fits")
        PSF_profile_density0_4D = fits.getdata(psf_dir / f"{instru}_PSF_profile_density_4D_{suffix_PSF}.fits")
        fraction_core0_4D       = fits.getdata(psf_dir / f"{instru}_fraction_core_4D_{suffix_PSF}.fits")
        radial_transmission0_4D = fits.getdata(psf_dir / f"{instru}_radial_transmission_4D_{suffix_PSF}.fits")

        # Source FITS semantics inherited from the original HWO PSF pipeline:
        # density = stellar halo fraction / mas2, normalized before radial throughput;
        # core = off-axis planetary fraction in the HWO 1-FWHM BOX;
        # radial transmission = planet throughput versus angular offset.
        axes_psf      = (wave0, C_raw0, IWA0, separation0)
        expected_shape = tuple(map(len, axes_psf))
        for axis in axes_psf:
            if axis.ndim != 1 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError("Invalid HWO PSF axis (must be finite and strictly increasing).")
        for arr in (PSF_profile_density0_4D, fraction_core0_4D, radial_transmission0_4D):
            if arr.shape != expected_shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"Invalid HWO PSF cube, expected {expected_shape}, found {arr.shape}.")
        for requested, available, name in ((l0, wave0, "l0"), (C_raw, C_raw0, "C_raw"), (IWA, IWA0, "IWA")):
            if requested[0] < available[0]*(1-1e-6) or requested[-1] > available[-1]*(1+1e-6):
                raise ValueError(f"Requested {name} range [{requested[0]:g}, {requested[-1]:g}] outside HWO PSF range [{available[0]:g}, {available[-1]:g}].")
        if sep_min < separation0[0] or sep_max > separation0[-1]:
            raise ValueError("Requested planet separation range exceeds the HWO PSF library.")
        if not np.isclose(separation0[0], 0):
            raise ValueError("HWO PSF separation grid must start at zero for radial-throughput normalization.")
        if np.any(PSF_profile_density0_4D < 0) or np.any((fraction_core0_4D < 0) | (fraction_core0_4D > 1)) or np.any((radial_transmission0_4D < 0) | (radial_transmission0_4D > 1)):
            raise ValueError("HWO PSF products must be physically non-negative, core and throughput in [0,1].")

        separation = separation0.astype(dtype, copy=False)
        N_sep      = len(separation)
        PSF_profile_4D     = create_memmap_with_log(PSF_profile_4D_tmp_path,     shape=(N_l0, N_C_raw, N_IWA, N_sep), dtype=dtype, mode="w+")
        fraction_core_4D   = create_memmap_with_log(fraction_core_4D_tmp_path,   shape=(N_l0, N_C_raw, N_IWA, N_sep), dtype=dtype, mode="w+")
        PSF_profile_max_3D = create_memmap_with_log(PSF_profile_max_3D_tmp_path, shape=(N_l0, N_C_raw, N_IWA),        dtype=dtype, mode="w+")

        # Logarithmic interpolation on strictly positive quantities; log(C) is
        # natural for HWO's multi-decade contrast grid. Boundary zeros are
        # preserved using linear interpolation for core and throughput, avoiding
        # positive leakage introduced by a logit epsilon at an exactly dark pixel.
        eps_pos           = np.finfo(dtype).tiny
        logC0             = np.log10(C_raw0)
        logC              = np.log10(C_raw)
        interp_psf        = RegularGridInterpolator((wave0, logC0, IWA0, separation0), np.log(np.clip(PSF_profile_density0_4D, eps_pos, None)), bounds_error=True)
        interp_core       = RegularGridInterpolator((wave0, logC0, IWA0, separation0), fraction_core0_4D, bounds_error=True)
        interp_rt         = RegularGridInterpolator((wave0, logC0, IWA0, separation0), radial_transmission0_4D, bounds_error=True)
        l0_chunk          = max(1, min(chunksize, N_l0))
        for j0 in tqdm(range(0, N_l0, l0_chunk), desc="Building temporary HWO PSF memmaps"):
            j1                  = min(j0 + l0_chunk, N_l0)
            pts                 = np.stack(np.meshgrid(l0[j0:j1], logC, IWA, separation, indexing="ij"), axis=-1)
            density_blk         = np.exp(interp_psf(pts))
            fraction_core_blk   = np.clip(interp_core(pts), 0.0, 1.0)
            radial_trans_blk    = np.clip(interp_rt(pts),   0.0, 1.0)
            if not all(np.all(np.isfinite(x)) for x in (density_blk, fraction_core_blk, radial_trans_blk)):
                raise ValueError("Non-finite interpolated HWO PSF products.")
            PSF_profile_blk     = density_blk * pxscales[j0:j1, None, None, None]**2
            PSF_profile_blk    *= radial_trans_blk[..., :1] # on-axis stellar attenuation (original FITS convention)
            fraction_core_blk  *= radial_trans_blk          # off-axis planetary throughput
            PSF_profile_4D[j0:j1]     = PSF_profile_blk.astype(dtype,   copy=False)
            fraction_core_4D[j0:j1]   = fraction_core_blk.astype(dtype, copy=False)
            PSF_profile_max_3D[j0:j1] = np.max(PSF_profile_blk, axis=-1).astype(dtype, copy=False)
            del pts, density_blk, fraction_core_blk, radial_trans_blk, PSF_profile_blk

        PSF_profile_4D.flush()
        fraction_core_4D.flush()
        PSF_profile_max_3D.flush()
        del wave0, C_raw0, IWA0, separation0, PSF_profile_density0_4D, fraction_core0_4D, radial_transmission0_4D
        del interp_psf, interp_core, interp_rt, PSF_profile_4D, fraction_core_4D, PSF_profile_max_3D
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

        # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotationnal broadening with Vsini)
        # Cover every requested IFU spectral window, including the widest one
        # at the boundaries of the l0 axis; retain the original imager handling.
        if l0_min - Dl_max / 2 <= 0:
            raise ValueError("The spectral bandwidth extends to nonphysical wavelengths; reduce Dl_max.")
        lmin_model = 0.98 * (l0_min - Dl_max / 2)                # [µm]
        lmax_model = 1.02 * (l0_max + Dl_max / 2)                # [µm]
        dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
        wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)

        # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
        wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
        dl_instru   = np.gradient(wave_instru)                                                    # [µm/bin] Nyquist sampling of a spectrum with resolving power R_model: 2 samples per resolution element across the whole axis

        # Effective model range
        lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin
        lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax

        # --- Space telescope: no tellurics, no ground-sky background ---
        # Load HWO aluminium reflectivity, including ONLY the OTA N_mirror surfaces.
        # All other optical/detector losses remain in trans_instru.
        reflectivity_paths = [instru_dir / "aluminium_refelectivity.csv", instru_dir / "aluminium_reflectivity.csv"]
        reflectivity_path  = next((p for p in reflectivity_paths if p.is_file()), None)
        if reflectivity_path is None:
            raise FileNotFoundError(f"Missing HWO aluminium reflectivity CSV in {instru_dir}")
        data_alu_ref = np.genfromtxt(reflectivity_path, delimiter=",", names=True, dtype=float, encoding=None)
        wave_ref     = data_alu_ref["wave"]*1e-3 # [nm] -> [um]
        alu_ref      = data_alu_ref["reflectivity"]
        if np.any((alu_ref < 0) | (alu_ref > 1)):
            raise ValueError("HWO reflectivity must lie between zero and one")
        trans_tel    = trans_dust * alu_ref**N_mirror
        trans_tel    = Spectrum(wavelength=wave_ref, flux=trans_tel).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tel[0], trans_tel[-1]))

        # HWO currently includes ZERO zodiacal/exozodiacal/thermal backgrounds.
        # This matches the old HWO model and is NOT a full physical noise budget.
        background = Spectrum(wavelength=wave_instru, flux=np.zeros_like(wave_instru)) # [ph/mn/mas2/um]


        #
        # IFU path
        #
        if instru_type == "IFU":

            # --- Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/µm] ---
            scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
            scale_spectrum *= S * 60                   # [ph/s/m2/µm] => [ph/mn/µm]

            # --- Preparing for wave axis, range, trans_tel, background for each point ---
            trans_tel_1D_list = [None] * len(R) # (R)
            background_1D_list     = [None] * len(R) # (R)     [ph/px/mn/bin]
            wave_1D_list           = [None] * len(R) # (R)     [µm]
            dwave_1D_list          = [None] * len(R) # (R)     [µm/bin]
            range_1D_list          = [None] * len(R) # (R)
            l0_2D                  = l0[:, None]     # (l0, 1) [µm]
            Dl_2D                  = Dl[None, :]     # (1, Dl) [um]
            # Symmetric wavelength intervals, exactly as in the imager case.
            l0_2D_min = l0_2D - Dl_2D / 2
            l0_2D_max = l0_2D + Dl_2D / 2
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

                # Find spectral channels whose centres lie in
                # [l0 - Dl/2, l0 + Dl/2]; the effective channel count depends
                # on R, l0 and Dl and is no longer an independent axis.
                # Using side="left"/"right" ensures we include the full requested wavelength span.
                idx_lo           = np.searchsorted(wave_res, l0_2D_min, side="left").astype(np.int32)
                idx_hi           = np.searchsorted(wave_res, l0_2D_max, side="right").astype(np.int32)
                idx_lo           = np.clip(idx_lo, 0, len(wave_res)-1)
                idx_hi           = np.clip(idx_hi, 1, len(wave_res))
                idx_hi           = np.maximum(idx_hi, idx_lo + 1)
                range_1D_list[i] = (idx_lo, idx_hi)

                # Tellurics x Telescope spectrum on the grid at this resolution
                trans_tel_1D_list[i] = trans_tel.degrade_resolution(wave_res, renorm=False, R_output=res, filter_type=filter_type).flux

                # Background spectrum on this grid in [ph/px/mn/bin]
                background_res        = background.degrade_resolution(wave_res, renorm=False, R_output=res, filter_type=filter_type).flux # (len(wave_res))     [ph/mn/µm/mas2]
                background_res       *= dwave_res                                                                                         # (len(wave_res))     [ph/mn/µm/mas2]  => [ph/mn/bin/mas2]
                background_1D_list[i] = background_res.astype(dtype, copy=False)                                                          # (len(wave_res))     [ph/mn/bin/mas2]


            # --- Global context worker ---
            _IFU_CTX = dict(planet_table=planet_table, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_model=wave_model, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, scale_spectrum=scale_spectrum, N_R=N_R, N_l0=N_l0, N_Dl=N_Dl, R=R, wave_1D_list=wave_1D_list, dwave_1D_list=dwave_1D_list, trans_tel_1D_list=trans_tel_1D_list, trans_instru=trans_instru, background_1D_list=background_1D_list, pxscales=pxscales, Rc=Rc, filter_type=filter_type, range_1D_list=range_1D_list, valid_windows_3D=valid_windows_3D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_4D_path=str(PSF_profile_4D_tmp_path), fraction_core_4D_path=str(fraction_core_4D_tmp_path), PSF_profile_max_3D_path=str(PSF_profile_max_3D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), sigma_syst_base_2_path=str(sigma_syst_base_2_path), dtype=dtype, D=D, pxscales_rad=pxscales_rad)


            # --- Arguments for Pool ---
            init_worker  = fyh.init_worker_IFU
            _CTX         = _IFU_CTX
            func_process = fyh.process_IFU
            shape_SNR    = (N_PT, N_R, N_l0, N_Dl, N_C_raw, N_IWA, N_TI)
            tqdm_desc    = f"{instru} ({instru_type} and {post_processing}) multiprocessing over planets of signal, stellar halo photon noise, DIT length and base for {residuals.lower()} noise"


        #
        # Imager path
        #
        elif instru_type == "imager":

            # --- Pre-computing the scale factor to convert from [J/s/m²/µm] to [ph/mn/bin] ---
            scale_spectrum  = wave_instru*1e-6 / (h*c) # [J/s/m²/µm]  => [ph/s/m2/µm]
            scale_spectrum *= S * 60 * dl_instru       # [ph/s/m2/µm] => [ph/mn/bin]

            # Preparing for bandwidth range for each point
            l0_2D     = l0[:, None]     # [µm] (l0, 1)
            Dl_2D     = Dl[None, :]     # [µm] (1, Dl)
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
            background_flux_2D  = fyh.window_sums_1d(values=background, idx_lo=idx_lo, idx_hi=idx_hi).astype(dtype) # [ph/mn/mas2]
            background_flux_2D *= pxscales[:, None]**2                                                          # [ph/mn/mas2] => [ph/px/mn]

            # --- Global context worker ---
            _IM_CTX = dict(planet_table=planet_table, A_FWHM=A_FWHM, separation=separation, thermal_model=thermal_model, reflected_model=reflected_model, post_processing=post_processing, wave_instru=wave_instru, wave_K=wave_K, counts_vega_K=counts_vega_K, scale_spectrum=scale_spectrum, N_l0=N_l0, N_Dl=N_Dl, trans_instru=trans_instru, trans_tel=trans_tel.flux, background_flux_2D=background_flux_2D, range_1D=range_1D, saturation_e=saturation_e, min_DIT=min_DIT, max_DIT=max_DIT, PSF_profile_4D_path=str(PSF_profile_4D_tmp_path), fraction_core_4D_path=str(fraction_core_4D_tmp_path), PSF_profile_max_3D_path=str(PSF_profile_max_3D_tmp_path), signal_path=str(signal_path), sigma_halo_2_path=str(sigma_halo_2_path), sigma_bkg_2_path=str(sigma_bkg_2_path), DIT_path=str(DIT_path), dtype=dtype)

            # --- Arguments for Pool ---
            init_worker  = fyh.init_worker_IM
            _CTX         = _IM_CTX
            func_process = fyh.process_IM
            shape_SNR    = (N_PT, N_l0, N_Dl, N_C_raw, N_IWA, N_TI)
            tqdm_desc    = f"{instru} ({instru_type} and {post_processing}) multiprocessing over planets of signal, stellar halo photon noise and DIT length"



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
        print(f"\nCreating memmap files in {sim_dir_concept}")
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
        with ctx_mp.Pool(processes=min(nproc, N_PT), initializer=init_worker, initargs=(_CTX,)) as pool:
            print(f'{tqdm_desc}:')
            for _ in tqdm(pool.imap_unordered(func_process, range(N_PT), chunksize=chunksize), total=N_PT):
                pass
        signal_planets.flush()
        sigma_halo_2_planets.flush()
        sigma_bkg_2_planets.flush()
        DIT_planets.flush()
        if instru_type == "IFU":
            sigma_syst_base_2_planets.flush()

        for p in [PSF_profile_4D_tmp_path, fraction_core_4D_tmp_path, PSF_profile_max_3D_tmp_path]:
            if p.exists():
                p.unlink()



    # %%
    # Plot general parameters
    fontsize           = 22                  # Standard fontsize
    ms                 = 15                  # Marker size
    ss                 = 400                 # Scatter size
    lw                 = 2                   # Linewidth
    alpha              = 0.9                 # Opacity
    gain               = True                # True for probability gain (normalized yield), False for absolute yields
    ptypes             = ["Jupiter",                 "Saturn",                "Neptune",                 "Earth"]
    marker_ptypes      = {"Jupiter": "s",            "Saturn": "v",           "Neptune": "P",            "Earth": "o"}
    label_ptypes       = {"Jupiter": "Jupiter-like", "Saturn": "Saturn-like", "Neptune": "Neptune-like", "Earth": "Earth-like"}

    # Define parameters and their names
    if instru_type == "IFU":
        params         = [R,                           l0,                                                Dl,                                           C_raw,                                 IWA,                                                       trans_instru,                                    100*sigma_m,                                        FoV]                     # params axis
        params_ranges  = [R_plot,                      l0_plot,                                           Dl_plot,                                      C_raw_plot,                            IWA_plot,                                                  trans_instru_plot,                               sigma_m_plot,                                       FoV_plot]                # Plot parameter selections (scalar, range or None)
        params_priors  = ["log-uniform",               "uniform",                                         "uniform",                                    "log-uniform",                         "log-uniform",                                             "uniform",                                       "log-uniform",                                      "log-uniform"]           # params priors for marginalization
        params_islog   = [True,                        False,                                             False,                                        True,                                  True,                                                      False,                                           True,                                               True]                    # params space (linspace or logspace)
        params_isint   = [True,                        False,                                             False,                                        False,                                 False,                                                     False,                                           False,                                              True]                    # params rounded format
        params_names   = ["R",                         "l0 [µm]",                                         "Dl [µm]",                                    "C_raw",                               "IWA [lambda/D]",                                         "trans_instru [e-/ph]",                          "sigma_m [%]",                                      "FoV [mas]"]             # params names
        params_names_l = [r"$R$",                      r"$\lambda_0$  [µm]",                              r"$\Delta\lambda$ [µm]",                      r"$C_{\mathrm{raw}}$",                 r"IWA [$\lambda/D$]",                                        r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                  r"$FoV$ [mas]"]          # params labels
        params_names_L = [r"Spectral resolution $R$",  r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Spectral bandwidth $\Delta\lambda$ [µm]",  r"Raw dark-hole contrast $C_{\mathrm{raw}}$", r"APLC inner working angle [$\lambda/D$]", r"Instrumental transmission $\gamma_{instru}$",  rf"{residuals} residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]"]  # params detailed labels

    elif instru_type == "imager":
        params         = [l0,                                                Dl,                                         C_raw,                                 IWA,                                                        trans_instru,                                    100*sigma_m,                                        FoV]                    # params axis
        params_ranges  = [l0_plot,                                           Dl_plot,                                    C_raw_plot,                            IWA_plot,                                                   trans_instru_plot,                               sigma_m_plot,                                       FoV_plot]               # Plot parameter selections (scalar, range or None)
        params_priors  = ["uniform",                                         "uniform",                                  "log-uniform",                         "log-uniform",                                              "uniform",                                       "log-uniform",                                      "log-uniform"]          # params priors for marginalization
        params_islog   = [False,                                             False,                                      True,                                  True,                                                       False,                                           True,                                               True]                   # params space (linspace or logspace)
        params_isint   = [False,                                             False,                                      False,                                 False,                                                      False,                                           False,                                              False]                  # params rounded format
        params_names   = ["l0 [µm]",                                         "Dl [µm]",                                  "C_raw",                               "IWA [lambda/D]",                                          "trans_instru",                                  "sigma_m [%]",                                      "FoV"]                  # params labels
        params_names_l = [r"$\lambda_0$ [µm]",                               r"$\Delta\lambda$ [µm]",                    r"$C_{\mathrm{raw}}$",                 r"$IWA$ [$\lambda/D$]",                   r"$\gamma_{instru}$ [e-/ph]",                    r"$\sigma_m$ [%]",                                  r"FoV [mas]"]           # params labels
        params_names_L = [r"Bandwidth central wavelength $\lambda_0$ [µm]",  r"Spectral coverage $\Delta\lambda$ [µm]",  r"Raw dark-hole contrast $C_{\mathrm{raw}}$", r"APLC inner working angle [$\lambda/D$]",  r"Instrumental transmission $\gamma_{instru}$",  rf"{residuals} residual amplitude $\sigma_m$ [%]",  r"Field of View [mas]"] # params detailed labels

    # Preparing parameter ranges for global and population plots
    population_axes = params.copy()
    plot_names      = ("R", "l0", "Dl", "C_raw", "IWA", "trans_instru", "sigma_m", "FoV") if instru_type == "IFU" else ("l0", "Dl", "C_raw", "IWA", "trans_instru", "sigma_m", "FoV")
    plot_prepared   = {}
    print("\nPlot parameter ranges:")
    if instru_type == "imager" and R_plot is not None:
        fyh.print_range("R", R_plot, warning="(WARNING: not used for imager)")
    for i, (name, axis, prior) in enumerate(zip(plot_names, population_axes, params_priors)):
        prepared            = fyh.prepare_population_axis(axis=axis, value=params_ranges[i], prior=prior, name=name, verbose=True)
        plot_prepared[name] = prepared
        used                = prepared[3]
        params_ranges[i]    = (used, used) if prepared[2] else used
    print()



    # %%
    # COMPUTING MAIN QUANTITIES

    if plot_global_results:
        # Computing the SNR and detection mask (SNR > SNR_thr) for each planet
        # For IFU (8D):    (planets, R, l0, Dl, C_raw, IWA, trans_instru, sigma_m)
        # For Imager (7D): (planets,    l0, Dl, C_raw, IWA, trans_instru, sigma_m)
        SNR_planets     = fyh.get_SNR(instru=instru, instru_type=instru_type, post_processing=post_processing, exposure_time=exposure_time, min_DIT=min_DIT, RON0=RON0, RON_lim=RON_lim, is_UTR=is_UTR, DC0=DC0, A_FWHM=A_FWHM, Rc=Rc, filter_type=filter_type, signal_planets=signal_planets, sigma_halo_2_planets=sigma_halo_2_planets, sigma_bkg_2_planets=sigma_bkg_2_planets, sigma_syst_base_2_planets=sigma_syst_base_2_planets, DIT_planets=DIT_planets, R=R if instru_type == "IFU" else None, Dl=Dl if instru_type == "IFU" else None, sigma_m=sigma_m, sim_dir=sim_dir_concept, suffix=suffix, dtype=dtype)
        mask_detections = fyh.get_mask_detections(SNR_planets=SNR_planets, SNR_thr=SNR_thr, sim_dir=sim_dir_concept, suffix=suffix, nproc=nproc, chunksize=chunksize)

        # Precompute FoV gating once
        p0_FoV = np.searchsorted(FoV / 2, separation_planets, side="left").astype(np.int32)

        # Getting ndim detection probability
        # For IFU (8D):    (R, l0, Dl, C_raw, IWA, trans_instru, sigma_m, FoV)
        # For Imager (7D): (   l0, Dl, C_raw, IWA, trans_instru, sigma_m, FoV)
        Pdet = fyh.get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, valid_windows_3D=valid_windows_3D, nproc=nproc)

        # --- Removing useless dimensions ---
        # Identify dimensions to remove (if size == 1)
        orig_params   = params.copy()
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
        if instru_type == "IFU":
            valid_windows_reduced = valid_windows_3D.copy()
            for idim in (2, 1, 0):
                if orig_sizes[idim] == 1:
                    valid_windows_reduced = np.take(valid_windows_reduced, 0, axis=idim)

        # Plot order: display C_raw and IWA in the last panels when present
        dims_last  = [next((idim for idim, name in enumerate(params_names) if key in name), None) for key in ("C_raw", "IWA")]
        dims_last  = [idim for idim in dims_last if idim is not None]
        plot_order = [idim for idim in range(Ndim) if idim not in dims_last] + dims_last

        # Thermal/Reflected regime split of the planet table
        # Considering both thermal and reflected contribution
        if thermal_model != "None" and reflected_model != "None":
            mag_th   = np.asarray(planet_table[f"Planet{band_plot}mag(thermal)"],   dtype=float)
            mag_re   = np.asarray(planet_table[f"Planet{band_plot}mag(reflected)"], dtype=float)
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
        print(f"\nPlanet-light regime split in {band_plot}-band:")
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
                Pdet_ptype_thermal = fyh.get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, indices=idx_ptype_thermal, jump_before_tqdm=False, valid_windows_3D=valid_windows_3D, nproc=nproc)
            else:
                Pdet_ptype_thermal = None
            if "reflected" in light_regime_plot:
                print(f"\n{label_ptypes[ptype]} in reflected light:")
                idx_ptype_reflected  = np.where(mask_ptype & mask_reflected)[0]
                Pdet_ptype_reflected = fyh.get_Pdet(mask_detections=mask_detections, FoV=FoV, p0_FoV=p0_FoV, dtype=dtype, indices=idx_ptype_reflected, jump_before_tqdm=False, valid_windows_3D=valid_windows_3D, nproc=nproc)
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



        # %%
        # 2D CORNER PLOT

        # Param values at max pdet_1D
        params_max  = np.zeros((Ndim))
        params_imax = np.zeros((Ndim), dtype=int)
        for idim in range(Ndim):
            pdet_1D           = fyh.reduce_hcube(hcube=Pdet_plot, dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names)
            params_imax[idim] = pdet_1D.argmax()
            params_max[idim]  = params[idim][pdet_1D.argmax()]

        xmin       = np.array([np.nanmin(param) for param in params])
        xmax       = np.array([np.nanmax(param) for param in params])
        levels     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cmap       = plt.get_cmap("plasma_r")
        fig, axes  = plt.subplots(Ndim, Ndim, figsize=(1.6 * Ndim, 1.6*Ndim), dpi=dpi_fig)
        axes       = np.atleast_2d(axes)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for ipanel, idim in enumerate(plot_order):
            for jpanel, jdim in enumerate(plot_order):
                ax = axes[ipanel, jpanel]
                ax.tick_params(axis="both", which="both", labelsize=12)
                if jpanel > ipanel:
                    ax.axis("off")
                    continue

                elif ipanel == jpanel:
                    pdet_1D = fyh.reduce_hcube(hcube=Pdet_plot, dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    ax.step(params[idim], pdet_1D, color="k", where="mid")
                    ax.axvline(params_max[idim], color="k", linestyle="--")
                    ax.set_yticks([])
                    if ipanel == Ndim - 1:
                        ax.set_xlabel(params_names_l[idim], fontsize=14)
                    if params_isint[idim]:
                        ax.set_title(f"{params_names_l[idim]} = {round(params_max[idim], -2):.0f}", fontsize=11)
                    elif params_islog[idim] and params_names[idim] != "IWA [lambda/D]":
                        ax.set_title(f"{params_names_l[idim]} = {params_max[idim]:.1e}", fontsize=11)
                    else:
                        ax.set_title(f"{params_names_l[idim]} = {params_max[idim]:.2f}", fontsize=11)
                    ax.set_xlim(xmin[idim], xmax[idim])
                    if params_islog[idim]:
                        ax.set_xscale("log")
                    if params_names[idim] == "sigma_m [%]":
                        fyh.set_log_decade_ticks(ax.xaxis, xmin[idim], xmax[idim])

                elif jpanel < ipanel:
                    pdet_2D = fyh.reduce_hcube(hcube=Pdet_plot, dims_to_keep=[jdim, idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
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
                    if params_names[jdim] == "sigma_m [%]":
                        fyh.set_log_decade_ticks(ax.xaxis, xmin[jdim], xmax[jdim])
                    if params_names[idim] == "sigma_m [%]":
                        fyh.set_log_decade_ticks(ax.yaxis, xmin[idim], xmax[idim])

                # Hide major and minor tick labels (log axes spanning < 1 decade label their minor ticks)
                if ipanel < Ndim - 1:
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if jpanel > 0:
                    ax.tick_params(axis="y", which="both", labelleft=False)

        title  = f"{instru} detection probability"
        title += f"\n{exposure_time/60:.0f}hr - {instru_type} with {post_processing}"
        title += f"\n{regime_label} light regime"
        title += f"\n{N_PT_plot} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} planets"
        fig.suptitle(title, fontsize=18, weight="bold", x=0.63, y=0.89, linespacing=1.6)
        fig.savefig(sim_dir_concept / f"HWO_{instru}_{instru_type}_{post_processing}_corner_plot_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi_fig)
        plt.show(block=False)



        # %%
        # PLOT : 1D MARGINALIZED DETECTION YIELD/PROBABILITY GAIN PER PARAM, TYPE AND REGIME

        # Layout
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=dpi_fig, sharey=True)
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
                    Pdet_thermal = fyh.reduce_hcube(hcube=Pdet_ptypes_thermal[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_thermal, N_PT_ptypes_thermal[ipt], ptype, "C3"))
                if "reflected" in light_regime_plot and N_PT_ptypes_reflected[ipt] > 0:
                    Pdet_reflected = fyh.reduce_hcube(hcube=Pdet_ptypes_reflected[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_reflected, N_PT_ptypes_reflected[ipt], ptype, "C0"))
            ymax_panel = fyh.get_panel_ymax([curve[0] for curve in curves]) if gain else None
            for Pdet_1D, N_PT_curve, ptype, color in curves:
                y = fyh.convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_1D, N_PT=N_PT_curve, gain=gain, ymax_panel=ymax_panel)
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
                leg_regime = ax.legend(handles=handles_regime, fontsize=fontsize, loc="lower right", frameon=True, edgecolor="gray", facecolor="white")#, title="Planet-light regime", title_fontsize=fontsize + 4)
                ax.add_artist(leg_regime)

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
        title         = f"{instru} {quantity_name} in {exposure_time/60:.0f}hr - {instru_type} with {post_processing}"
        title        += f"\n{N_PT} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} planets"
        fig.suptitle(title, fontsize=fontsize + 6, weight="bold", y=1.00, linespacing=1.6)
        fig.tight_layout(h_pad=3.0, w_pad=3.0)
        fig.savefig(sim_dir_concept / f"HWO_{instru}_{instru_type}_{post_processing}_detection_{table_type}_{light_regime_plot}_Pdet.png", bbox_inches="tight", dpi=dpi_fig)
        plt.show(block=False)



        # %%
        # Plot : 1D MARGINALIZED DETECTION YIELD/PROBABILITY GAIN PER PARAM, TYPE, REGIME AND BANDS

        ptypes_plot = ["Jupiter", "Saturn", "Neptune", "Earth"]
        bands_plot  = ["V", "R", "I", "z", "Y", "J", "H", "K"] # Choose the spectral bands to show

        # Identify lambda0 axis (required for this plot) and identify a l0 for each considered band
        idx_l0 = [idx for idx, param_name in enumerate(params_names) if "l0" in param_name]
        try:
            idx_l0 = idx_l0[0]
        except:
            raise RuntimeError("Could not identify the lambda0 axis in params_names / params_names_L.")
        l0_axis_values = params[idx_l0]
        band_l0_values = np.array([(lmin_bands[band]+lmax_bands[band])/2 for band in bands_plot], dtype=float)

        # Keep only the bands whose central wavelength is within the sampled lambda0 range
        in_range = (band_l0_values >= l0_axis_values[0]) & (band_l0_values <= l0_axis_values[-1])
        if not np.all(in_range):
            dropped = [band for band, ok in zip(bands_plot, in_range) if not ok]
            print(f"Warning: removing band(s) {dropped} from the plot (central wavelength outside the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm).")
            bands_plot     = [band for band, ok in zip(bands_plot, in_range) if ok]
            band_l0_values = band_l0_values[in_range]
        if len(bands_plot) == 0:
            raise ValueError(f"None of the selected bands has a central wavelength within the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm.")
        NbBand      = len(bands_plot)
        band_labels = [f"{band}\n$\\lambda_0$={l0_band:.2f} µm" for band, l0_band in zip(bands_plot, band_l0_values)]
        cmap        = plt.get_cmap("rainbow", NbBand)

        # loop over the different types of planets to make one plot for each
        for ptype in ptypes_plot:

            # Layout
            fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), dpi=dpi_fig, sharey=True)
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
                    ipt = ptypes.index(ptype)
                    if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                        continue
                    Pdet_1D = fyh.reduce_hcube(hcube=Pdet_ptypes_plot[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                    curves.append((Pdet_1D, N_PT_ptypes_plot[ipt], ptype, "k"))
                else:
                    ipt = ptypes.index(ptype)
                    if Pdet_ptypes_plot[ipt] is None or N_PT_ptypes_plot[ipt] == 0:
                        continue
                    for iband, l0_band in enumerate(band_l0_values):
                        params_ranges_band         = [tuple(rng) for rng in params_ranges]
                        params_ranges_band[idx_l0] = (l0_band, l0_band)
                        Pdet_1D                    = fyh.reduce_hcube(hcube=Pdet_ptypes_plot[ipt], dims_to_keep=[idim], params=params, params_ranges=params_ranges_band, params_priors=params_priors, params_names=params_names, verbose=False)
                        curves.append((Pdet_1D, N_PT_ptypes_plot[ipt], ptype, cmap(iband)))
                ymax_panel = fyh.get_panel_ymax([curve[0] for curve in curves]) if gain else None
                for Pdet_1D, N_PT_curve, ptype, color in curves:
                    y = fyh.convert_Pdet_to_plot_quantity(Pdet_curve=Pdet_1D, N_PT=N_PT_curve, gain=gain, ymax_panel=ymax_panel)
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
                # if ipanel == 0:
                #     handles_ptype = [Line2D([], [], ls="", marker=marker_ptypes[ptype], ms=ms, markerfacecolor="white", markeredgewidth=1.5, color="k", label=label_ptypes[ptype]) for ptype in ptypes_plot]
                #     leg_ptype     = ax.legend(handles=handles_ptype, fontsize=fontsize + 2, loc="upper right", frameon=True, edgecolor="gray", facecolor="white", title="Planet type", title_fontsize=fontsize + 4)
                #     ax.add_artist(leg_ptype)

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
            N_PT_ptype_plot_1D = int(N_PT_ptypes_plot[ptypes.index(ptype)])
            title  = f"{instru} {quantity_name} in {exposure_time/60:.0f}hr - {instru_type} with {post_processing}"
            title += f"\n{light_regime_plot} planet-light regime"
            title += f"\n{N_PT_ptype_plot_1D} {table_type.replace('Archive', 'known').replace('Simulated', 'simulated')} {ptype}-like planets"
            fig.suptitle(title, fontsize=fontsize + 6, weight="bold", y=1.00, linespacing=1.6)
            fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.88, wspace=0.15, hspace=0.3)
            fig.savefig(sim_dir_concept / f"HWO_{instru}_{instru_type}_{post_processing}_detection_band_{table_type}_{light_regime_plot}_{ptype}_Pdet.png", bbox_inches="tight", dpi=dpi_fig)
            plt.show(block=False)



        #%%
        # HEATMAP: BAND x RESOLUTION / BANDWIDTH HEATMAPS PER PLANET TYPE AND LIGHT REGIME

        ptypes_heatmap = ["Jupiter", "Saturn", "Neptune", "Earth"]
        bands_plot     = ["V", "R", "I", "z", "Y", "J", "H", "K"]
        heatmap_mode   = "marginalized" # "marginalized" or "max"
        regimes_plot   = [regime for regime in ["thermal", "reflected"] if regime in light_regime_plot]
        if not regimes_plot:
            raise RuntimeError("No valid regime to plot. light_regime_plot must contain 'thermal' and/or 'reflected'.")

        # Choose the second heatmap axis depending on the instrument type
        if post_processing == "MM": #instru_type == "IFU":
            spec_plot      = [1_000, 3_000, 10_000, 30_000, 100_000]
            spec_plot      = [50_000, 75_000, 100_000, 150_000, 200_000]
            idx_spec       = next((idx for idx, name in enumerate(params_names) if "R" in name), None)
            spec_axis_name = "Resolution"
        else:
            spec_plot      = [0.1, 0.3, 1.0, 3.0, 10.0]
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
        in_range         = (band_l0_values >= l0_axis_values[0]) & (band_l0_values <= l0_axis_values[-1])
        if not np.all(in_range):
            dropped = [band for band, ok in zip(bands_plot, in_range) if not ok]
            print(f"Warning: removing band(s) {dropped} from the heatmap (central wavelength outside the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm).")
            bands_plot     = [band for band, ok in zip(bands_plot, in_range) if ok]
            band_l0_values = band_l0_values[in_range]
        if len(bands_plot) == 0:
            raise ValueError(f"None of the selected bands has a central wavelength within the sampled lambda0 range [{l0_axis_values[0]:.3f}, {l0_axis_values[-1]:.3f}] µm.")
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
                                Pdet_value            = fyh.reduce_hcube(hcube=Pdet_cube, dims_to_keep=[], params=params, params_ranges=ranges, params_priors=params_priors, params_names=params_names, verbose=False)
                                heatmap[ispec, iband] = float(np.asarray(Pdet_value)) * N_PT_reg
                            else:
                                Pdet_subcube          = np.asarray(fyh.reduce_hcube(hcube=Pdet_cube, dims_to_keep=dims_keep_max, params=params, params_ranges=ranges, params_priors=params_priors, params_names=params_names, verbose=False), dtype=float)
                                heatmap[ispec, iband] = float(np.nanmax(Pdet_subcube)) * N_PT_reg if np.any(np.isfinite(Pdet_subcube)) else np.nan
                heatmaps[(ptype, regime)] = heatmap

        panel_keys   = [(ptype, regime) for ptype in ptypes_heatmap for regime in regimes_plot]
        panel_labels = {(ptype, regime): f"{label_ptypes[ptype]} — {regime.capitalize()}" for ptype, regime in panel_keys}
        title        = f"{instru} band × {'resolution' if instru_type == 'IFU' else 'speckle residuals'} yield - {instru_type} with {post_processing}"#"\nheatmap mode: {heatmap_mode}"

        yield_heatmap_ELT(instru=instru, exposure_time=exposure_time, heatmaps=heatmaps, ptypes_heatmap=panel_keys, bands_plot=bands_plot, config_labels=spec_plot_labels, config_axis_name=spec_axis_name, x_axis_name="Spectral band", panel_labels=panel_labels, title=title, save_dir=sim_dir_concept, filename=f"HWO_{instru}_{instru_type}_{post_processing}_band_spectral_heatmap_grid_{table_type}_{light_regime_plot}_{heatmap_mode}.png")



    if plot_populations:
        # %%
        # POPULATION DIAGNOSTICS
        population_results    = fyh.get_population_results(instru_type=instru_type, signal_planets=signal_planets, sigma_halo_2_planets=sigma_halo_2_planets, sigma_bkg_2_planets=sigma_bkg_2_planets, sigma_syst_base_2_planets=sigma_syst_base_2_planets, DIT_planets=DIT_planets, separation_planets=separation_planets, axes=population_axes, prepared=plot_prepared, detector_l0=detector_l0, RON0=RON0, RON_lim=RON_lim, DC0=DC0, min_DIT=min_DIT, is_UTR=is_UTR, exposure_time=exposure_time, A_FWHM=A_FWHM, valid_windows_3D=valid_windows_3D, nproc=nproc, chunksize=chunksize)
        SNR_plot              = population_results["SNR"]
        dominant_noise        = population_results["dominant_noise"]
        dominant_noise[dominant_noise == "Systematics"] = str(residuals)

        yield_population_plot(table=table_type, instru=instru, instru_type=instru_type, post_processing=post_processing, thermal_model=thermal_model, reflected_model=reflected_model, exposure_time=exposure_time, band_contrast_plot=band_plot, band_regime_plot=band_plot, planet_table=planet_table, SNR_plot=SNR_plot, dominant_noise=dominant_noise, SNR_thr=SNR_thr, save_dir=sim_dir_concept, DL_mas=np.nan, show_golden_sample=False, show_det_regions=False, show_ptype_regions=False, space=planet_population_space)



    # %%
    # --- Clean temporary memmaps generated for global plots, if any ---
    if plot_global_results:
        snr_tmp_path  = Path(SNR_planets.filename)
        mask_tmp_path = Path(mask_detections.filename)
        del SNR_planets, mask_detections
        gc.collect()
        for path in (snr_tmp_path, mask_tmp_path):
            if path.exists():
                path.unlink()

    del signal_planets, sigma_halo_2_planets, sigma_bkg_2_planets, sigma_syst_base_2_planets, DIT_planets
    gc.collect()

    return locals()



# %%

if __name__ == "__main__":
    __spec__ = None
    if sys.platform.startswith("win"):
        mp.freeze_support() # for Windows
    _out = main()
    globals().update(_out)

