# import FastYield modules
from src.config import instrus, bands, rad2arcsec
from src.get_specs import _load_corona_profile, _load_corr_factor, _load_bkg_flux, _get_transmission, get_PSF_profile, get_config_data, get_wa, get_band_lims, get_R_instru
from src.utils import airy_profile, get_r_core
from src.spectrum import filtered_flux, get_mag, get_resolution, get_spectrum_instru, get_spectrum_band, load_star_spectrum, load_planet_spectrum, load_vega_spectrum, get_counts_from_density
from src.signal_noise import get_DIT_RON, get_delta_cos_theta_syst, get_alpha_cos_theta_syst, get_beta, get_fn_MM, get_systematics, compute_P_al_spat_numba, compute_P_al_spec_numba, compute_P_speck_numba, compress_h_for_P
from src.data_processing import parameters_retrieval
from src.prints_helpers import print_header, print_subheader, print_info, print_metric, print_warning, print_time, sci

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit

# import other modules
import time
import warnings
import re



#--------------------
# FastCurves Function
#--------------------
def FastCurves_process(calculation, instru, exposure_time, mag_star, band0_star, band0_planet, planet_spectrum, star_spectrum, tellurics, apodizer, strehl, coronagraph, systematics, speckles, PCA, PCA_mask, N_PCA, channel, planet_name, separation_planet, mag_planet, show_plot, verbose, post_processing, sep_unit, background, Rc, filter_type, input_DIT, band_only, return_FastYield, return_quantity):
    """
    See the function "FastCurves" below.
    """
        
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive*", category=UserWarning)
    #warnings.filterwarnings('ignore', category=UserWarning, append=True)
    mpl.rcParams['figure.max_open_warning'] = 0
    
    # ----------------------------------
    # Hard-coded toggles for diagnostics
    # ----------------------------------
    cos_theta_p_MM     = 1     # Intrisic model mismatch with MM (on the HF content)
    cos_theta_p_DI     = 1     # Intrisic model mismatch with MM (on the full content)
    cos_theta_est_data = None  # Estimated correlation in the data (for show_cos_theta_est=True)
    show_cos_theta_est = False # To see the impact of the noise on the estimated correlation in order to retrieve the true mismatch
    show_t_syst        = False # To see the systematics time domination
    show_contributions = True  # To see the noise contributions plots for contrast calculations
    show_syst_budget   = True # To see the speckle noise budget for Differential Imaging techniques to be more advantageous than molecular mapping and systematics noise budget
    
    # ------------------
    # Config & constants
    # ------------------
    if instru == "MIRIMRS" and channel and (calculation in {"SNR", "corner plot"}):
        # Each SNR “per channel” splits time over its 3 sub-bands
        exposure_time = exposure_time / 3

    for_planet_name = f" for {planet_name}" if planet_name is not None else ""
    
    config_data    = get_config_data(instru)
    telescope_name = config_data["telescope_name"]           # Telescope name (e.g. ELT, VLT, JWST, etc.)
    D              = config_data["telescope"]["diameter"]    # Telescope diameter [m]
    eps            = config_data["telescope"]["eps"]         # Geometric central obstruction (diameter of the secondary / diameter of the primary)
    instru_type    = config_data["type"]                     # Instrument's type (e.g. imager, IFU, IFU_fiber, etc.)
    NbBand         = len(config_data["gratings"])            # Number of bands
    size_core      = config_data["size_core"]                # Aperture size on which the signal is integrated [pixels]
    A_FWHM         = size_core**2                            # Box aperture 
    saturation_e   = config_data["detector"]["saturation_e"] # Full well capacity of the detector [e-]
    min_DIT        = config_data["detector"]["minDIT"]       # Minimal integration time [in mn]
    max_DIT        = config_data["detector"]["maxDIT"]       # Maximal integration time [mn]
    RON            = config_data["detector"]["RON"]          # Read out noise [e-/px/read]
    RON_lim        = config_data["detector"]["RON_lim"]      # Read out noise limit [e-/px/read]
    dark_current   = config_data["detector"]["dark_current"] # Dark current [e-/px/s]
    IWA, OWA       = get_wa(config_data=config_data, band="instru", sep_unit=sep_unit) # Inner and Outer Working Angle in 'sep_unit' 
    if show_plot:
        cmap = plt.get_cmap("rainbow", NbBand)

    # ------------
    # Accumulators
    # ------------
    contrast_bands        = [] # Final contrast as function of separation for exposure_time
    SNR_bands             = [] # Final S/N as function of separation for exposure_time
    name_bands            = [] # Band labels
    separation_bands      = [] # [arcsec] or [mas]
    DIT_bands             = [] # [mn/DIT]
    signal_bands          = [] # [e-/FWHM/DIT]
    sigma_fund_2_bands    = [] # [e-/FWHM/DIT]
    sigma_halo_2_bands    = [] # [e-/FWHM/DIT]
    sigma_det_2_bands     = [] # [e-/FWHM/DIT]
    sigma_bkg_2_bands     = [] # [e-/FWHM/DIT]
    sigma_speck_2_bands   = [] # [e-/FWHM/DIT]
    sigma_syst_2_bands    = [] # [e-/FWHM/DIT]
    planet_flux_bands     = [] # [e-/DIT] (total e- over the FoV)
    star_flux_bands       = [] # [e-/DIT] (total e- over the FoV)
    wave_bands            = [] # [µm]
    trans_bands           = [] # [e-/ph]
    uncertainties_bands   = [] # [K], [dex(cm/s2)], [km/s] and [km/s]
    iwa_FPM_bands         = [] # [arcsec] or [mas]
    sigma_m_syst_bands    = [] # [%]
    sigma_m_speck_bands   = [] # [%]
    sigma_phi_speck_bands = [] # [nm RMS]
    
    SNR_max_planet = 0.0
    band_SNR_max   = ""
    pca            = None
    
    T_planet     = planet_spectrum.T     # [K]
    T_star       = star_spectrum.T       # [K]
    lg_planet    = planet_spectrum.lg    # [dex(cm/s2)]
    lg_star      = star_spectrum.lg      # [dex(cm/s2)]
    model_planet = planet_spectrum.model # Planet's model
    model_star   = star_spectrum.model   # Star's model
    rv_star      = star_spectrum.rv      # [km/s]
    rv_planet    = planet_spectrum.rv    # [km/s]
    vsini_star   = star_spectrum.vsini   # [km/s]
    vsini_planet = planet_spectrum.vsini # [km/s]
    lmin_instru  = config_data["lambda_range"]["lambda_min"] # [µm]
    lmax_instru  = config_data["lambda_range"]["lambda_max"] # [µm]
    R_planet     = np.nanmedian(planet_spectrum.R[(planet_spectrum.wavelength >= lmin_instru) & (planet_spectrum.wavelength <= lmax_instru)]) # Planet's model resolution
    R_star       = np.nanmedian(star_spectrum.R[(star_spectrum.wavelength     >= lmin_instru) & (star_spectrum.wavelength   <= lmax_instru)]) # Star's model resolution
    R_instru     = get_R_instru(config_data=config_data) # Max instrument resolution (factor 2 to be sure to not loose spectral information)
    lmin_instru  = config_data["lambda_range"]["lambda_min"]
    lmax_instru  = config_data["lambda_range"]["lambda_max"]
    
    # Incoming separation is in arcsec; if instrument expects mas, convert once.
    if sep_unit == "mas" and separation_planet is not None:
        separation_planet = separation_planet * 1e3 # [arcsec] => [mas]
    
    # -----------------------------------------------------------------------------------
    # Spectra on instrument bandwidth (intermediate spectra) + magnitudes in [ph/bin/mn] (total ph over the FoV)
    # -----------------------------------------------------------------------------------
    star_spectrum_instru, star_spectrum_density = get_spectrum_instru(band0=band0_star, R=R_instru, config_data=config_data, mag=mag_star, spectrum=star_spectrum) # [ph/bin/mn] and [J/s/m2/µm]
    if mag_planet is not None:
        planet_spectrum_instru, planet_spectrum_density = get_spectrum_instru(band0=band0_planet, R=R_instru, config_data=config_data, mag=mag_planet, spectrum=planet_spectrum) # [ph/bin/mn] and [J/s/m2/µm]
    else:
        if calculation in {"SNR", "corner plot"}:
            raise KeyError(f"Please input 'mag_planet' for the {calculation} calculation !")
        # The planet spectra are not adjusted to the correct magnitude
        planet_spectrum_instru, planet_spectrum_density = get_spectrum_instru(band0=band0_star, R=R_instru, config_data=config_data, mag=mag_star, spectrum=planet_spectrum) # [ph/bin/mn] and [J/s/m2/µm]
        # For plotting purposes only: normalize planetary density to star density mean level
        planet_spectrum_density.flux *= np.nanmean(star_spectrum_density.flux) / np.nanmean(planet_spectrum_density.flux)
    
    # Wavelength axis
    wave_instru = planet_spectrum_instru.wavelength
    mask_instru = (wave_instru >= lmin_instru) & (wave_instru <= lmax_instru) # wave_instru is slightly broader than the instrumental range

    # Computing the planet-to-star and star-to-planet flux ratio in [ph/bin/mn] on the instrumental bandwidth to renormalize the signal with (by doing so, it will give a contrast in photons counts and not in energy on this bandwidth, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
    planet_to_star_ratio = np.nansum(planet_spectrum_instru.flux[mask_instru]) / np.nansum(star_spectrum_instru.flux[mask_instru])
    star_to_planet_ratio = 1 / planet_to_star_ratio
    
    # Instrumental magnitudes (computed from densities)
    vega_spectrum_density = load_vega_spectrum()                                                    # [J/s/m2/µm]
    vega_spectrum_density = vega_spectrum_density.interpolate_wavelength(wave_instru, renorm=False) # [J/s/m2/µm]
    counts_vega           = get_counts_from_density(wave=wave_instru[mask_instru], density=vega_spectrum_density.flux[mask_instru])
    mag_star_instru       = get_mag(wave=wave_instru[mask_instru], density_obs=star_spectrum_density.flux[mask_instru], density_vega=None, counts_vega=counts_vega)
    if mag_planet is not None:
        mag_planet_instru = get_mag(wave=wave_instru[mask_instru], density_obs=planet_spectrum_density.flux[mask_instru], density_vega=None, counts_vega=counts_vega)
    else:
        mag_planet_instru = None
    
    # ---------------------------------------------------------
    # Optional overview plot on the instrument band [J/s/m2/µm]
    # ---------------------------------------------------------
    if show_plot and band_only is None:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.yscale("log")
        plt.xlim(wave_instru[0], wave_instru[-1])  
        plt.ylim(min(np.nanmin(planet_spectrum_density.flux), np.nanmin(star_spectrum_density.flux)), max(np.nanmax(planet_spectrum_density.flux), np.nanmax(star_spectrum_density.flux)))
        plt.xlabel("Wavelength [µm]", fontsize=14)
        plt.ylabel(r"Flux [J/s/$m^2$/µm]", fontsize=14)
        plt.title(f"Star and planet spectra on the instrumental bandwidth (R = {round(round(R_instru, -2))})\nwith $rv_*$ = {round(rv_star, 1)} km/s and $rv_p$ = {round(rv_planet, 1)} km/s", fontsize=16)
        plt.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        plt.plot(wave_instru, planet_spectrum_density.flux, color='seagreen', linestyle='-', linewidth=2, alpha=0.7, label=f'Planet, {model_planet} with $T$={int(round(T_planet))}K\nmag(instru)={round(mag_planet_instru, 1) if mag_planet_instru is not None else "Unknown"}')        
        plt.plot(wave_instru, star_spectrum_density.flux, color='crimson', linestyle='-', linewidth=2, alpha=0.7, label=f'Star, {model_star} with $T$={int(round(T_star))}K\nmag(instru)={round(mag_star_instru, 1)}')        
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        plt.gca().yaxis.set_ticks_position('both')
        plt.minorticks_on()
        plt.tight_layout()        
        plt.show()
    
    # --------------
    # Verbose header
    # --------------
    if verbose:
        
        print("\n\n\n")
        print_header(f"FastCurves calculation{for_planet_name} with {telescope_name}/{instru} (from {lmin_instru:.2f} to {lmax_instru:.2f}µm)")
        
        print_info(f"  {telescope_name} is a {config_data['base']}-based observatory.", sub=True)            
        
        print_metric("Type of calculation",    "calculation",     f"{calculation}",       "")
        print_metric("Total exposure time",    "exposure_time",   f"{exposure_time:.2f}", "mn")
        
        if strehl != "NO_JQ": # for ground-based observatory 
            print_metric("Strehl regime",    "strehl",   f"{strehl}",   "")
        print_metric("Apodizer/shaped pupil configuration", "apodizer",    f"{apodizer.replace('NO_SP', 'None').replace('_', ' ')}", "")
        print_metric("Coronagraphic configuration",         "coronagraph", f"{str(coronagraph)}",                                    "")
        print_metric("Post-processing method", "post_processing", f"{post_processing}", "")
        if post_processing.lower() in {"molecular mapping", "mm"}:
            print_metric("MM cut-off resolution", "Rc",          f"{Rc}",          "")
            print_metric("MM type of filter",     "filter_type", f"{filter_type}", "")
        print_metric("Systematic noise",             "systematics", f"{str(systematics)}", "")
        print_metric("Speckle noise",                "speckles",    f"{str(speckles)}",    "")
        print_metric("Principal component analysis", "PCA",         f"{str(PCA)}",         "")
        if PCA:
            print_metric("Number of principal components", "N_PCA", f"{N_PCA}", "")
        
        print()
        print_subheader("Planetary spectrum:")
        print_metric("Model family",                  "model_planet", f"{model_planet}",            "")
        print_metric("Model spectral resolution",     "R_planet",     f"{round(R_planet, -3):.0f}", "")
        print_metric("Effective temperature",         "T_planet",     f"{T_planet:.0f}",            "K")
        print_metric("Surface gravity",               "lg_planet",    f"{lg_planet:.0f}",           "dex(cm/s2)")
        print_metric("Radial velocity",               "rv_planet",    f"{rv_planet:.0f}",           "km/s")
        print_metric("Projected rotational velocity", "vsini_planet", f"{vsini_planet:.0f}",        "km/s")
        if separation_planet is not None:
            print_metric("Projected angular separation", "separation_planet", f"{separation_planet:.2f}", f"{sep_unit}")
        if mag_planet is not None:
            print_metric(f"Magnitude in {band0_planet}-band", "mag_planet",          f"{mag_planet:.2f}",           "")
            print_metric("Magnitude in INSTRU-band",          "mag_planet_instru",   f"{mag_planet_instru:.2f}",    "")
            print_metric("Planet-to-star flux ratio",         "Sp_instru/Ss_instru", f"{sci(planet_to_star_ratio)}", "")
        else:
            print_metric("Magnitude", "mag_planet", "'Unknwon'", "")

        print()
        print_subheader("Stellar spectrum:")
        print_metric("Model family",                    "model_star",      f"{model_star}",            "")
        print_metric("Model spectral resolution",       "R_star",          f"{round(R_star, -3):.0f}", "")
        print_metric("Effective temperature",           "T_star",          f"{T_star:.0f}",            "K")
        print_metric("Surface gravity",                 "lg_star",         f"{lg_star:.0f}",           "dex(cm/s2)")
        print_metric("Radial velocity",                 "rv_star",         f"{rv_star:.0f}",           "km/s")
        print_metric("Projected rotational velocity",   "vsini_star",      f"{vsini_star:.0f}",        "km/s")
        print_metric(f"Magnitude in {band0_star}-band", "mag_star",        f"{mag_star:.2f}",          "")
        print_metric("Magnitude in INSTRU-band",        "mag_star_instru", f"{mag_star_instru:.2f}",   "")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Optional overview plot per band [e-/bin/mn] (total e- over the FoV): plotting the planet spectrum (if SNR calculation) or the star spectrum (if contrast calculation) on each band
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if show_plot:
        fig_band_flux = plt.figure(figsize=(10, 6), dpi=300)
        ax_flux_band  = fig_band_flux.gca()
        ax_flux_band.set_yscale("log")
        ax_flux_band.set_xlabel("Wavelength [µm]", fontsize=14)
        ax_flux_band.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        ax_flux_band.yaxis.set_ticks_position('both')
        ax_flux_band.minorticks_on()
        ymin = +np.inf
        ymax = -np.inf
        if mag_planet is not None and separation_planet is not None:
            ax_flux_band.set_title(f"{telescope_name}/{instru} star and planet flux at the position of {'the planet' if planet_name is None else planet_name}", fontsize=16)
            ax_flux_band.set_ylabel("Flux [e-/bin/FWHM/mn]", fontsize=14)
        else:
            ax_flux_band.set_title(f"Star flux ({model_star}) through {telescope_name}/{instru} bands with {band0_star}={mag_star:.2f}", fontsize=16)
            ax_flux_band.set_ylabel("Total flux over the FoV [e-/bin/mn]", fontsize=14)

    # ========================
    # Loop over spectral bands
    # ========================
    for nb, band in enumerate(config_data["gratings"]):
        if (band_only is not None and band != band_only) or (instru=="HARMONI" and strehl=="MED" and band not in ["H", "K"]):
            continue # If you want to calculate for band_only only
                
        # -----------------------------------------------------------------------
        # Spectra on the band + magnitudes in [ph/bin/mn] (total ph over the FoV)
        # -----------------------------------------------------------------------
        star_spectrum_band   = get_spectrum_band(spectrum_instru=star_spectrum_instru,   config_data=config_data, band=band, verbose=verbose) # [ph/bin/mn] (total ph over the FoV)
        planet_spectrum_band = get_spectrum_band(spectrum_instru=planet_spectrum_instru, config_data=config_data, band=band, verbose=verbose) # [ph/bin/mn] (total ph over the FoV)
        wave_band            = planet_spectrum_band.wavelength      # [µm]
        lmin_band            = config_data['gratings'][band].lmin   # Lambda min of the considered band [µm]
        lmax_band            = config_data['gratings'][band].lmax   # Lambda max of the considered band [µm]
        R_band               = config_data['gratings'][band].R      # Spectral resolution of the band
        R_nyquist            = get_resolution(wavelength=wave_band, func=np.nanmedian)       # Spectral resolution assuming Nyquist sampling (should be R_nyquist ~ R_band)
        iwa, owa             = get_wa(config_data=config_data, band=band, sep_unit=sep_unit) # Inner and Outer Working Angle in 'sep_unit' 

        # Band's magnitudes (computed from densities)
        mask_band     = (wave_instru >= lmin_band) & (wave_instru <= lmax_band)
        counts_vega   = get_counts_from_density(wave=wave_instru[mask_band], density=vega_spectrum_density.flux[mask_band])
        mag_star_band = get_mag(wave=wave_instru[mask_band], density_obs=star_spectrum_density.flux[mask_band], density_vega=None, counts_vega=counts_vega)
        if mag_planet is not None:
            mag_planet_band = get_mag(wave=wave_instru[mask_band], density_obs=planet_spectrum_density.flux[mask_band], density_vega=None, counts_vega=counts_vega)
        else:
            mag_planet_band = None
            
        # # Spectral Resolutions sanity check plot
        # plt.figure(dpi=300, figsize=(10, 6))
        # plt.title(f"planet {band}-band at R = {R_band:.0f}", fontsize=14)
        # plt.xlabel("Wavelength [µm]", fontsize=12)
        # plt.ylabel("Spectral resolution", fontsize=12)
        # plt.plot(planet_spectrum.wavelength, planet_spectrum.R,        c="crimson",   lw=3, label=f"planet (raw), R={np.nanmedian(planet_spectrum.R):.0f}")
        # plt.plot(wave_instru,                planet_spectrum_instru.R, c="seagreen",  lw=3, label=f"planet (instru), R={np.nanmedian(planet_spectrum_instru.R):.0f}")
        # plt.plot(wave_band,                  planet_spectrum_band.R,   c="steelblue", lw=3, label=f"planet (band), R={np.nanmedian(planet_spectrum_band.R):.0f}")
        # plt.axhline(R_band, c="k", label="band")
        # plt.plot(wave_band, get_resolution(wave_band, func=np.array), c="k", ls="--", label="Nyquist")
        # plt.xlim(lmin_band, lmax_band)
        # plt.yscale('log')
        # plt.legend()
        # plt.show()
        
        # plt.figure(dpi=300, figsize=(10, 6))
        # plt.title(f"star {band}-band at R = {R_band:.0f}", fontsize=14)
        # plt.xlabel("Wavelength [µm]", fontsize=12)
        # plt.ylabel("Spectral resolution", fontsize=12)
        # plt.plot(star_spectrum.wavelength, star_spectrum.R,        c="crimson",   lw=3, label=f"star (raw), R={np.nanmedian(star_spectrum.R):.0f}")
        # plt.plot(wave_instru,              star_spectrum_instru.R, c="seagreen",  lw=3, label=f"star (instru), R={np.nanmedian(star_spectrum_instru.R):.0f}")
        # plt.plot(wave_band,                star_spectrum_band.R,   c="steelblue", lw=3, label=f"star (band), R={np.nanmedian(star_spectrum_band.R):.0f}")
        # plt.axhline(R_band, c="k", label="band")
        # plt.plot(wave_band, get_resolution(wave_band, func=np.array), c="k", ls="--", label="Nyquist")
        # plt.xlim(lmin_band, lmax_band)
        # plt.yscale('log')
        # plt.legend()
        # plt.show()
        
        # ------------------------------------
        # Total system transmission in [e-/ph]
        # ------------------------------------
        trans = _get_transmission(instru=instru, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, fill_value=np.nan)
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # PSF profiles (radial mean of Ms) [FoV flux fraction/px], fraction_core [FoV flux fraction/FWHM], separations [sep_unit], pxscale [sep_unit/px] (+ coronagraphic radial transmission, if any)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        PSF_profile, fraction_core, separation, pxscale, hdr_PSF = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, instru=instru, separation_planet=separation_planet, return_FastYield=return_FastYield, return_hdr=True, sampling=2 if show_syst_budget else 10)
                
        # Index of the separation of the planet
        if separation_planet is not None:
            idx_planet_sep = np.nanargmin(np.abs(separation - separation_planet))
                        
        # Coronagraphic radial transmission and fraction core [FoV flux fraction/FWHM] as function of separation
        if coronagraph is not None:
            raw_sep, raw_fraction_core, raw_radial_transmission = _load_corona_profile(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)
            logit_fraction_core_interp                          = interp1d(raw_sep, logit(raw_fraction_core),       bounds_error=False, fill_value=(logit(raw_fraction_core[0]),       logit(raw_fraction_core[-1])))
            logit_radial_transmission_interp                    = interp1d(raw_sep, logit(raw_radial_transmission), bounds_error=False, fill_value=(logit(raw_radial_transmission[0]), logit(raw_radial_transmission[-1])))
            fraction_core                                       = expit(logit_fraction_core_interp(separation))       # Fraction of flux of a PSF inside the FWHM as function of the separation
            radial_transmission                                 = expit(logit_radial_transmission_interp(separation)) # Transmission of a PSF as function of the separation
            star_transmission                                   = expit(logit_radial_transmission_interp(0))          # Star coronagraphic transmission factor when the star is perfectly aligned with it (i.e. at 0 separation)
            fraction_core[separation > raw_sep[-1]]             = raw_fraction_core[-1]                               # Flat extrapolation
            radial_transmission[separation > raw_sep[-1]]       = raw_radial_transmission[-1]                         # Flat extrapolation 
            fraction_core                                      *= radial_transmission                                 # Fraction core through the coronagraph
            
            # Multiplying the coronagraphic PSF profile by the star coronagraphic transmission factor
            PSF_profile *= star_transmission
        
        # -------------------------------------------------------
        # DIT in [mn] and effective read-out noise in [e-/px/DIT] 
        # -------------------------------------------------------
        N_DIT, DIT, DIT_saturation, DIT_saturation_planet, RON_eff, iwa_FPM = get_DIT_RON(config_data=config_data, instru_type=instru_type, apodizer=apodizer, PSF_profile=PSF_profile, separation=separation, star_spectrum_band=star_spectrum_band, exposure_time=exposure_time, min_DIT=min_DIT, max_DIT=max_DIT, trans=trans, RON=RON, RON_lim=RON_lim, saturation_e=saturation_e, input_DIT=input_DIT, separation_planet=separation_planet)
        
        # Star and planet spectra integrated over the DIT in [ph/bin/DIT] (total ph over the FoV)
        Ss = star_spectrum_band.flux   * DIT # [ph/bin/DIT] (total ph over the FoV)
        Sp = planet_spectrum_band.flux * DIT # [ph/bin/DIT] (total ph over the FoV)
        
        # -----------------------------------------------------------------------------------------------------------
        # Fiber-fed IFUs: mean injection correction (e.g. ANDES): the fact that the position of the planet is unknown
        # -----------------------------------------------------------------------------------------------------------
        if "fiber" in instru_type:
            try:
                fraction_core *= config_data["injection"][band]
            except:
                pass
        
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # Corrective factor for fundamental noises (per separation): due to potential dithering (impacting the noise statistics, i.e. covariance, etc.)
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        R_corr = np.zeros_like(separation) + 1.
        if instru in {"MIRIMRS", "NIRSpec"}: # Dithering for MIRIMRS and NIRSpec
            separation_R_corr, r_corr                  = _load_corr_factor(instru=instru, band=band)
            valid                                      = np.isfinite(r_corr)
            R_corr                                     = interp1d(separation_R_corr[valid], r_corr[valid], bounds_error=False, fill_value="extrapolate")(separation)
            R_corr[separation > separation_R_corr[-1]] = r_corr[-1] # Flat extrapolation
        else:
            try:
                R_corr *= config_data["R_corr"]
            except:
                pass
        
        # ===============================================================================================================
        # Systematic estimations (if needed): systematic noise profile [e-/FWHM/DIT] and systematic modulations [no unit]
        # ===============================================================================================================
        sigma_syst_2       = np.zeros_like(separation) # [e-/FWHM/DIT]
        sigma_syst_prime_2 = np.zeros_like(separation) # [e-/FWHM/DIT]
        
        if systematics:
            # -----------
            # Imager path
            # -----------
            if instru_type == "imager":
                raise KeyError("Undefined !")   
    
            # --------
            # IFU path
            # --------
            else:
                
                # ---------------------------
                # Molecular Mapping (MM) path
                # ---------------------------
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    
                    # Systematics noise profile and modulations
                    if systematics:
                        
                        # [e-/FHWM/mn]**2, [sep_unit], [no unit], [no unit], [µm], object, str
                        sigma_syst_prime_2, separation_syst, Mp, M_pca, wave, pca, PCA_verbose = get_systematics(config_data=config_data, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, R_band=R_band, Rc=Rc, filter_type=filter_type, star_spectrum_instru=star_spectrum_instru, planet_spectrum_instru=planet_spectrum_instru, wave_band=wave_band, size_core=size_core, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, mag_planet=mag_planet, separation_planet=separation_planet, mag_star=mag_star, exposure_time=exposure_time, target_name=planet_name)
                        
                        # Interpolation + extrapolation (if needed) on the current separation axis
                        sigma_syst_prime = np.exp(interp1d(separation_syst, np.log(np.sqrt(sigma_syst_prime_2)), bounds_error=False, fill_value="extrapolate")(separation))
                        if separation[-1] > separation_syst[-1]: # Systematic profile extrapolation
                            from src.utils import power_law_extrapolation    
                            slope                       = hdr_PSF["slope"]
                            mask_tail                   = separation >= separation_syst[-1]
                            sigma_syst_prime[mask_tail] = power_law_extrapolation(x=separation[mask_tail], x0=separation_syst[-1], y0=np.sqrt(sigma_syst_prime_2)[-1], slope=slope)
                        
                        # Converting in [e-/FWHM/DIT]
                        sigma_syst_prime_2  = sigma_syst_prime**2 # [e-/FWHM/mn]**2
                        sigma_syst_prime_2 *= DIT**2              # [e-/FWHM/DIT]**2
                        
                        # Effective wavelength axis (from data)
                        lmin_band = max(lmin_band, wave[0])
                        lmax_band = min(lmax_band, wave[-1])
                        star_spectrum_band.crop(lmin_band, lmax_band)
                        planet_spectrum_band.crop(lmin_band, lmax_band)
                        mask_wave = (wave_band >= lmin_band) & (wave_band <= lmax_band)
                        Ss        = Ss[mask_wave]
                        Sp        = Sp[mask_wave]
                        trans     = trans[mask_wave]
                        wave_band = wave_band[mask_wave]
                        Mp        = Mp[mask_wave]
                        
                # ------------------------------
                # Differential Imaging (DI) path
                # ------------------------------
                elif post_processing.lower() in {"differential imaging", "di"}:
                    raise KeyError("Undefined !")   
        
        # ===================================================================
        # Signal estimations in [e-/FWHM/DIT] and computing templates for IFU
        # ===================================================================
        
        # -----------
        # Imager path
        # -----------
        if instru_type == "imager":
            
            # Systematic modulations of the spectra are taken into account (mostly insignificant effect on detection capability)
            if systematics:
                Mp_Sp = Mp  * Sp # [ph/bin/DIT] (total ph over the FoV)
            else:
                Mp_Sp = 1.0 * Sp # [ph/bin/DIT] (total ph over the FoV)
            
            signal = np.zeros_like(separation) + np.nansum(trans*Mp_Sp) # Integrated flux [e-/DIT] (total e- over the FoV)
        
        # --------
        # IFU path
        # --------
        else:
            
            # ---------------------------
            # Molecular Mapping (MM) path
            # ---------------------------
            if post_processing.lower() in {"molecular mapping", "mm"}:
                
                # Building MM template
                template_MM, _ = filtered_flux(flux=Sp, R=R_band, Rc=Rc, filter_type=filter_type) # [Sp]_HF
                template_MM    = trans * template_MM                                              # trans * [Sp]_HF
                template_MM   /= np.sqrt(np.nansum(template_MM**2))                               # Normalizing the template
                template       = template_MM
                
                # Systematic modulations of the spectra are taken into account (mostly insignificant effect on detection capability)
                if systematics:
                    Mp_Sp = Mp  * Sp # [ph/bin/DIT] (total ph over the FoV)
                else:
                    Mp_Sp = 1.0 * Sp # [ph/bin/DIT] (total ph over the FoV)
                
                # Useful signal (α) and self-subtraction (β) in [e-/DIT] (total e- over the FoV) (with systematic modulations, if any)
                alpha, cos_theta_syst_MM = get_alpha_cos_theta_syst(Mp_Sp=Mp_Sp, trans=trans, template=template_MM, R=R_nyquist, Rc=Rc, filter_type=filter_type)
                beta                     = get_beta(Ss=Ss,          Mp_Sp=Mp_Sp, trans=trans, template=template_MM, R=R_nyquist, Rc=Rc, filter_type=filter_type)
                
                # For prints (computing DI signal in [e-/DIT] (total e- over the FoV) and template)
                if verbose or show_syst_budget:
                    template_DI              = trans * Sp                         # trans * Sp
                    template_DI             /= np.sqrt(np.nansum(template_DI**2)) # Normalizing the template
                    delta, cos_theta_syst_DI = get_delta_cos_theta_syst(Mp_Sp=Mp_Sp, template=template_DI, trans=trans)

                # Computing MM signal in the CCF in [e-/DIT] (total e- over the FoV)
                signal = np.zeros_like(separation) + (alpha * cos_theta_syst_MM * cos_theta_p_MM - beta) # Useful MM signal in the CCF in [e-/DIT] (total e- over the FoV)
                
            # ------------------------------
            # Differential Imaging (DI) path
            # ------------------------------
            elif post_processing.lower() in {"differential imaging", "di"}:
                
                # Building DI template
                template_DI  = trans * Sp                         # trans * Sp
                template_DI /= np.sqrt(np.nansum(template_DI**2)) # Normalizing the template
                template     = template_DI
                
                # Systematic modulations of the spectra are taken into account (mostly insignificant effect on detection capability)
                if systematics:
                    Mp_Sp = Mp  * Sp # [ph/bin/DIT] (total ph over the FoV)
                else:
                    Mp_Sp = 1.0 * Sp # [ph/bin/DIT] (total ph over the FoV)
                
                # Useful signal (δ) in [e-/mn] (total e- over the FoV) (with systematics modulations, if any)
                delta, cos_theta_syst_DI = get_delta_cos_theta_syst(Mp_Sp=Mp_Sp, template=template_DI, trans=trans)
                
                # Computing DI signal in the CCF in [e-/DIT] (total e- over the FoV)
                signal = np.zeros_like(separation) + delta * cos_theta_syst_DI * cos_theta_p_DI # Useful DI signal in the CCF in [e-/DIT] (total e- over the FoV)
        
        # Focal Plane Mask: flux attenuated by a factor of 1e-4 for HARMONI (FPM)
        if instru=="HARMONI" and iwa_FPM > 0:
            PSF_profile[separation < iwa_FPM] *= 1e-4
            signal[separation < iwa_FPM]      *= 1e-4
        
        # Signal inside FWHM aperture in [e-/FWHM/DIT]
        signal *= fraction_core
        
        # Renormalizing the signal with the planet-to-star ratio (total received photons) on the instrumental bandwidth (by doing so, it will give a contrast in photons and not in energy on this bandwidth, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
        if calculation == "contrast":
            signal *= star_to_planet_ratio
            
        # Signal loss ratio due to the PCA (if required)
        if systematics:
            signal *= M_pca
            
        # For fiber_injection_HRS instruments, and separation_planet < FWHM, MM is impossible
        if instru_type == "fiber_injection_HRS" and separation_planet is not None and separation_planet < iwa and calculation == "SNR":
            signal *= 0

        # If saturation is reached at the planet separation even with the smallest DIT, detection is impossible
        if DIT_saturation_planet is not None and DIT_saturation_planet < min_DIT and calculation == "SNR":
            signal *= 0

        # ==================================================
        # Noise estimations in [e-/px/DIT] and [e-/FWHM/DIT]
        # ==================================================
        
        # -----------------------------------------------------------------------------------------------------------
        # Detector noises, DC and RON in [e-/px/DIT] (does not need CCF projection since it is wavelength indepedant)
        # -----------------------------------------------------------------------------------------------------------
        sigma_dc_2  = dark_current * 60*DIT # Dark current photon noise [e-/px/DIT]
        sigma_ron_2 = RON_eff**2            # Effective read out noise  [e-/px/DIT]
        
        if "fiber" in instru_type: # Detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
            NbPixel      = config_data['pixel_detector_projection'] # Number of detector px per cube px
            sigma_dc_2  *= NbPixel                                  # Adds quadratically [e-/px/DIT]
            sigma_ron_2 *= NbPixel                                  # Adds quadratically [e-/px/DIT]
        
        # ----------------------------------------
        # Stellar halo photon noise in [e-/px/DIT]
        # ----------------------------------------
        if instru_type == "imager": # Integrated flux
            sigma_halo_2 = PSF_profile * np.nansum(trans*Ss) # Stellar photon noise per spectral channel in [e-/px/DIT] for each separation
        
        else: # Projection onto the CCF
            sigma_halo_prime_2 = PSF_profile * np.nansum(trans*Ss * template**2) # Stellar photon noise projected in the CCF [e-/px/DIT] for each separation
        
        # --------------------------------------
        # Background photon noise in [e-/px/DIT]
        # --------------------------------------
        if background is None:
            sigma_bkg_2       = 0. # [e-/px/DIT]
            sigma_bkg_prime_2 = 0. # [e-/px/DIT]
        
        else:
            wave_raw, bkg_raw = _load_bkg_flux(instru, band, background) # [µm] and [e-/bin/px/s]  
            bkg_raw_density   = bkg_raw / np.gradient(wave_raw)          # [e-/µm/px/s]
            bkg_flux          = interp1d(wave_raw, bkg_raw_density, bounds_error=False, fill_value=(bkg_raw_density[0], bkg_raw_density[-1]))(wave_band) # [e-/µm/px/s]
            bkg_flux         *= np.gradient(wave_band) * 60*DIT          # [e-/bin/px/DIT]
            sigma_bkg_2       = bkg_flux                                 # Background photon noise per spectral channel in [e-/bin/px/DIT] for each separation
            
            if instru_type == "imager": # Integrated flux
                sigma_bkg_2 = np.nansum(sigma_bkg_2) # Background photon noise per spectral channel in [e-/px/DIT]

            else: # Projection onto the CCF
                sigma_bkg_prime_2 = np.nansum(sigma_bkg_2 * template**2) # Background photon noise projected in the CCF in [e-/px/DIT]

        # ----------------------------------------
        # Total fundamental noise in [e-/FWHM/DIT]
        # ----------------------------------------
        if instru_type == "imager":
            sigma_fund_2 = R_corr * A_FWHM * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2) # Fundamental noise profile integrated over the FWHM and the exposure time in [e-/FWHM/DIT]
        
        else:
            sigma_fund_prime_2 = R_corr * A_FWHM * (sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2) # Fundamental noise profile projected in the CCF and integrated over the FWHM and the exposure time in [e-/FWHM/DIT]
            
            # ---------------------------------------------------------------------------------------------------------------------
            # MM effective power fraction of the white noise in the CCF: fn_MM = sigma_fund_MM_2 / sigma_fund_2
            # ---------------------------------------------------------------------------------------------------------------------                
            if post_processing.lower() in {"molecular mapping", "mm"}:
                #fn_MM, _            = get_fn_HF_LF(N=len(wave_band), R=R_band, Rc=Rc, filter_type=filter_type, empirical=False)
                fn_MM               = get_fn_MM(template=template_MM, R=R_band, Rc=Rc, filter_type=filter_type)
                sigma_fund_prime_2 *= fn_MM
        
        # ------------------------------
        # Speckle noise in [e-/FWHM/DIT]
        # ------------------------------
        sigma_speck_2       = np.zeros_like(separation) # [e-/FWHM/DIT]
        sigma_speck_prime_2 = np.zeros_like(separation) # [e-/FWHM/DIT]
        
        if speckles:
            
            # -----------
            # Imager path
            # -----------
            if instru_type == "imager":
                raise KeyError("Undefined !")   
            
            # --------
            # IFU path
            # --------
            else:
                
                # ---------------------------
                # Molecular Mapping (MM) path
                # ---------------------------
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    pass # Speckles are assumed to be completely subtracted with MM: This assumption is more or less valid depending on the value of Rc chosen, but a value of Rc >~ 100 is sufficient for this assumption to always be true.
                
                # ------------------------------
                # Differential Imaging (DI) path
                # ------------------------------
                elif post_processing.lower() in {"differential imaging", "di"}:
                    raise KeyError("Undefined !")
                
        # ===================================================================================================================================================================
        # Optional overview plot per band in [e-/bin/mn] (total e- over the FoV): plotting the planet (if SNR calculation) or the star (if contrast calculation) on each band
        # ===================================================================================================================================================================
        if show_plot:
            if mag_planet is not None and separation_planet is not None:
                if coronagraph is not None:
                    flux_band = trans*fraction_core[idx_planet_sep]*Mp_Sp/DIT # Planet flux integrated over the FWHM in [e-/bin/FWHM/mn] at the planet's separation
                else:
                    flux_band = trans*fraction_core*Mp_Sp/DIT                      # Planet flux integrated over the FWHM [e-/bin/FWHM/mn]
                flux_band_star = A_FWHM*PSF_profile[idx_planet_sep] * trans*Ss/DIT # Stellar flux integrated over the FWHM in [e-/bin/FWHM/mn] at the planet's separation
                ax_flux_band.plot(wave_band, flux_band_star, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.3)
                ymin = min(ymin, np.nanmin(flux_band), np.nanmin(flux_band_star))
                ymax = max(ymax, np.nanmax(flux_band), np.nanmax(flux_band_star))
            else:
                if coronagraph is not None:
                    flux_band = trans*Mp_Sp*star_transmission/DIT # [e-/bin/mn] (total e- over the FoV) through coronagraph
                else:
                    flux_band = trans*Mp_Sp/DIT # [e-/bin/mn] (total e- over the FoV)
                ymin = min(ymin, np.nanmin(flux_band))
                ymax = max(ymax, np.nanmax(flux_band))
            if instru_type == "imager":
                label_band = band.replace('_', ' ') + f" ({round(np.nansum(flux_band))} e-/mn)"
            else:
                label_band = band.replace('_', ' ') + f" (R={int(round(R_band))})"
            ax_flux_band.plot(wave_band, flux_band, color=cmap(nb), linestyle='-', linewidth=2, alpha=1.0, label=label_band)
            if band_only is not None or nb == NbBand-1:   
                ax_flux_band.set_ylim(max(ymin, 1e-2), ymax)
                legend1 = ax_flux_band.legend(fontsize=12, loc="center right", frameon=True, fancybox=True, shadow=True, borderpad=1)
                ax_flux_band.add_artist(legend1)
                if mag_planet is not None and separation_planet is not None:
                    h_planet = ax_flux_band.plot([], [], color="k", linestyle='-', linewidth=2, alpha=1.0, label=f"Planet: {model_planet}, $T$={T_planet:.0f}K, $lg$={lg_planet:.1f} and {band0_planet}={mag_planet:.2f}")[0]
                    h_star   = ax_flux_band.plot([], [], color="k", linestyle='-', linewidth=2, alpha=0.3, label=f"Star:   {model_star}, $T$={T_star:.0f}K, $lg$={lg_star:.1f} and {band0_star}={mag_star:.2f}")[0]
                    legend2  = ax_flux_band.legend(handles=[h_planet, h_star], fontsize=12, loc="lower center", frameon=True, fancybox=True, shadow=True, borderpad=1)
                    ax_flux_band.add_artist(legend2)
            xmin, xmax = ax_flux_band.get_xlim()
            xmin       = min(xmin, lmin_band)
            xmax       = max(xmax, lmax_band)
            ax_flux_band.axvspan(lmin_band, lmax_band, color=cmap(nb), alpha=0.1, lw=0)
            ax_flux_band.set_xlim(xmin, xmax)
            
        # ===================================
        # Optional verbose band-level summary
        # ===================================
        if verbose:
            
            print()
            if instru_type == "imager":
                title = f"{band.replace('_', ' ')}-BAND (from {lmin_band:.2f} to {lmax_band:.2f} µm)"
            else:
                title = f"{band.replace('_', ' ')}-BAND (from {lmin_band:.2f} to {lmax_band:.2f} µm at R={R_band:.0f})"
            print_header(title, sub=True)
            
            if instru_type == "fiber_injection_HRS" and separation_planet is not None and separation_planet < iwa:
                print_warning(f"The planet's separation ({separation_planet:.2f} {sep_unit}) is smaller than the IWA ({iwa:.2f} {sep_unit}), no spectral diversity is possible with {instru}; setting the S/N to 0.")
            
            if DIT_saturation < min_DIT:
                print_warning(f"Saturation would occur even at the shortest DIT; using min_DIT ({min_DIT*60:.2f} s)")
            if DIT_saturation_planet is not None and DIT_saturation_planet < min_DIT:
                print_warning(f"Saturation would occur even at the shortest DIT at the planet's location (at {separation_planet:.2f} {sep_unit}); setting the S/N to 0.")
            
            if PCA and PCA_verbose is not None:
                print_info("  "+PCA_verbose, sub=True)
            if iwa_FPM > 0:
                print_info(f"  Using a focal plane mask under {iwa_FPM:.1f} {sep_unit} to avoid saturation", sub=True)
            
            print_metric("Star magnitude", "mag_star_band", f"{mag_star_band:.2f}", "")
            if mag_planet_band is not None:
                print_metric("Planet magnitude",          "mag_planet_band", f"{mag_planet_band:.2f}",                "")
                print_metric("Planet-to-star flux ratio", "Sp_band/Ss_band", f"{sci(np.nansum(Sp) / np.nansum(Ss))}", "")
            else:
                print_metric("Planet magnitude", "mag_planet_band", "Unknown", "")
            if instru_type != "imager":
                print_metric("Number of spectral channels", "N_λ", f"{len(wave_band)}", "channels")
            print_metric("Detector Integration Time",       "DIT",        f"{DIT:.3f}",                   "mn")
            print_metric("Saturating DIT",                  "DIT_sat",    f"{DIT_saturation:.3f}",        "mn")
            print_metric("Effective post-UTR RON",          "RON_eff",    f"{RON_eff:.3f}",               "e-/DIT")
            print_metric("Mean total system transmission",  "trans",      f"{100*np.nanmean(trans):.1f}", "%")
            if separation_planet is not None and coronagraph is not None:
                print_metric("Fraction of flux in the FWHM", "f_FWHM", f"{100*fraction_core[idx_planet_sep]:.1f}", f"% (at {separation_planet:.1f} {sep_unit})")
            elif coronagraph is None:
                print_metric("Fraction of flux in the FWHM", "f_FWHM", f"{100*fraction_core:.1f}", "%")
            print_metric("Diffraction limited FWHM (IWA)",       "iwa",      f"{iwa:.2f}",           f"{sep_unit}")
            if calculation in {"SNR", "corner plot"}:
                expression = "α - β" if post_processing.lower() in {"molecular mapping", "mm"} else "δ"
                pp         = "MM"    if post_processing.lower() in {"molecular mapping", "mm"} else "DI"
                if separation_planet is not None:
                    print_metric(f"Useful {pp} signal", expression, f"{signal[idx_planet_sep] / DIT:.1f}", f"e-/FWHM/mn (at {separation_planet:.1f} {sep_unit})")
                elif coronagraph is None:
                    print_metric(f"Useful {pp} signal", expression, f"{np.nanmean(signal / DIT):.1f}", "e-/FWHM/mn",)
            if post_processing.lower() in {"molecular mapping", "mm"}:
                hp_loss  = 100 * (1 - (alpha * cos_theta_syst_MM * cos_theta_p_MM - beta) / (delta * cos_theta_syst_DI * cos_theta_p_DI))
                self_sub = 100 * beta / alpha
                print_metric("Effective white noise power post-MM",    "fn_MM",     f"{100*fn_MM:.1f}", "%")
                print_metric("Signal loss due to MM", "1 - (α - β)/δ", f"{hp_loss:.1f}",   "%")
                print_metric("Signal loss due to self-subtraction",    "β/α",       f"{self_sub:.1f}",  "%")
            if instru_type != "imager" and systematics:
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    syst_loss = 100 * (1 - cos_theta_syst_MM)
                    print_metric("Signal loss due to systematics", "1 - cosθ_syst", f"{syst_loss:.1f}", "%")
                elif post_processing.lower() in {"differential imaging", "di"}:
                    syst_loss = 100 * (1 - cos_theta_syst_DI)
                    print_metric("Signal loss due to systematics", "1 - cosθ_syst", f"{syst_loss:.1f}", "%")
            if PCA and systematics:
                print_metric("Signal loss due to PCA", "f_PCA", f"{100*(1 - M_pca):.1f}", "%")
        
        # =================
        # Saving quantities
        # =================
        name_bands.append(band)               # Band's name
        separation_bands.append(separation)   # [sep_unit]
        DIT_bands.append(DIT)                 # [mn/DIT]
        signal_bands.append(signal)           # [e-/FWHM/DIT]
        planet_flux_bands.append(trans*Mp_Sp) # [e-/bin/DIT] (total e- over the FoV)
        star_flux_bands.append(trans*Ss)      # [e-/bin/DIT] (total e- over the FoV)
        wave_bands.append(wave_band)          # [µm]
        trans_bands.append(trans)             # [e-/ph]
        iwa_FPM_bands.append(iwa_FPM)         # [sep_unit]
        
        # Saving noise variances in [e-/FWHM/DIT]
        if instru_type == "imager":
            sigma_fund_2_bands.append(sigma_fund_2)
            sigma_halo_2_bands.append(R_corr * A_FWHM * (sigma_halo_2))
            sigma_det_2_bands.append(R_corr  * A_FWHM * (sigma_ron_2 + sigma_dc_2))
            sigma_bkg_2_bands.append(R_corr  * A_FWHM * (sigma_bkg_2))
            sigma_syst_2_bands.append(sigma_syst_2)
        
        else:
            if post_processing.lower() in {"molecular mapping", "mm"}:
                sigma_fund_2_bands.append(sigma_fund_prime_2)
                sigma_halo_2_bands.append(fn_MM * R_corr * A_FWHM * (sigma_halo_prime_2))
                sigma_det_2_bands.append(fn_MM  * R_corr * A_FWHM * (sigma_ron_2 + sigma_dc_2))
                sigma_bkg_2_bands.append(fn_MM  * R_corr * A_FWHM * (sigma_bkg_prime_2))
                sigma_syst_2_bands.append(sigma_syst_prime_2)
            
            elif post_processing.lower() in {"differential imaging", "di"}:
                sigma_fund_2_bands.append(sigma_fund_prime_2)
                sigma_halo_2_bands.append(1 * R_corr * A_FWHM * (sigma_halo_prime_2))
                sigma_det_2_bands.append(1  * R_corr * A_FWHM * (sigma_ron_2 + sigma_dc_2))
                sigma_bkg_2_bands.append(1  * R_corr * A_FWHM * (sigma_bkg_prime_2))
                sigma_speck_2_bands.append(sigma_speck_prime_2)
                sigma_syst_2_bands.append(sigma_syst_prime_2)
        
        # =======================
        # 5σ contrast computation
        # =======================
        if calculation == "contrast":
            
            if instru_type == "imager":
                contrast = 5 * np.sqrt(sigma_fund_2 + N_DIT*sigma_syst_2 ) / ( np.sqrt(N_DIT) * signal )
            
            else: # See Eq. (11) of Martos et al. 2025
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    contrast = 5 * np.sqrt( sigma_fund_prime_2 + N_DIT*sigma_syst_prime_2 ) / ( np.sqrt(N_DIT) * signal )
                
                elif post_processing.lower() in {"differential imaging", "di"}:
                    contrast = 5 * np.sqrt( sigma_fund_prime_2 + N_DIT*sigma_speck_prime_2 + N_DIT*sigma_syst_prime_2 ) / ( np.sqrt(N_DIT) * signal )
                
            # Adding the contrast curve of the band to the list
            contrast_bands.append(contrast)
            
            # ------------------------------------------
            # Noise contributions plots (in 5σ contrast)
            # ------------------------------------------
            if show_plot and show_contributions:
                mask_iwa_FPM = separation >= iwa_FPM
                fig_contrast = plt.figure(figsize=(10, 6), dpi=300)        
                ax_contrast  = fig_contrast.gca()
                ax_contrast.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
                ax_contrast.minorticks_on()
                ax_contrast.tick_params(axis='both', labelsize=12)      
                if iwa_FPM > 0:
                    ax_contrast.axvspan(0, iwa_FPM, color='black', alpha=0.3, lw=0)
                ax_contrast.set_yscale('log')
                ax_contrast.set_xlim(0, separation[-1])
                ax_contrast.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                if instru_type == "imager":
                    ax_contrast.set_title(f"{telescope_name}/{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name} \n with "r"$t_{exp}$="f"{round(exposure_time)}mn and "r"$mag_*$"f"({band0_star})={round(mag_star, 2)}", fontsize=16)        
                    ax_contrast.set_ylabel(r'Contrast 5$\sigma$ / $F_{p}$', fontsize=14)
                    ax_contrast.plot(separation[mask_iwa_FPM], contrast[mask_iwa_FPM], 'k-', label=r"$\sigma_{tot}$")
                    ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(R_corr * A_FWHM * sigma_halo_2) / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="crimson",   ls="--", label=r"$\sigma_{halo}$")
                    ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(R_corr * A_FWHM * sigma_ron_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                    ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(R_corr * A_FWHM * sigma_dc_2)   / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                    ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(R_corr * A_FWHM * sigma_bkg_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="royalblue", ls="--", label=r"$\sigma_{bkg}$")                    
                    if systematics:
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(N_DIT * sigma_syst_2)/((np.sqrt(N_DIT) * signal)))[mask_iwa_FPM], c="cyan", ls="--", label=r"$\sigma_{syst}$")
                    if speckles:
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(N_DIT * sigma_speck_2)/((np.sqrt(N_DIT) * signal)))[mask_iwa_FPM], c="gray", ls="--", label=r"$\sigma_{speck}$")
                else:
                    if post_processing.lower() in {"molecular mapping", "mm"}:
                        ax_contrast.set_title(f"{telescope_name}/{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name}\n with "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                        ax_contrast.set_ylabel(r"Contrast 5$\sigma_{CCF}$ / $\alpha_0$", fontsize=14)
                        ax_contrast.plot(separation[mask_iwa_FPM], contrast[mask_iwa_FPM], 'k-', label=r"$\sigma_{CCF}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(fn_MM * R_corr * A_FWHM * sigma_halo_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="crimson",   ls="--", label=r"$\sigma'_{halo}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(fn_MM * R_corr * A_FWHM * sigma_ron_2)        / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(fn_MM * R_corr * A_FWHM * sigma_dc_2)         / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(fn_MM * R_corr * A_FWHM * sigma_bkg_prime_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="royalblue", ls="--", label=r"$\sigma'_{bkg}$")
                        if systematics:
                            ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(N_DIT * sigma_syst_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="cyan", ls="--", label=r"$\sigma'_{syst}$")
                    elif post_processing.lower() in {"differential imaging", "di"}:
                        ax_contrast.set_title(f"{telescope_name}/{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name}\n with "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K", fontsize=16)        
                        ax_contrast.set_ylabel(r"Contrast 5$\sigma_{CCF}$ / $\delta_0$", fontsize=14)
                        ax_contrast.plot(separation[mask_iwa_FPM], contrast[mask_iwa_FPM], 'k-', label=r"$\sigma_{CCF}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_halo_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="crimson",   ls="--", label=r"$\sigma'_{halo}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_ron_2)        / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_dc_2)         / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                        ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_bkg_prime_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="royalblue", ls="--", label=r"$\sigma'_{bkg}$")
                        if systematics:
                            ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(N_DIT * sigma_syst_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="cyan", ls="--", label=r"$\sigma'_{syst}$")
                        if speckles:
                            ax_contrast.plot(separation[mask_iwa_FPM], (5*np.sqrt(N_DIT * sigma_speck_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa_FPM], c="cyan", ls="--", label=r"$\sigma'_{speck}$")
                if separation_planet is not None:
                    if separation_planet > 2 * OWA:
                        ax_contrast.set_xscale('log')
                        ax_contrast.set_xlim(IWA, separation[-1])
                    if mag_planet is None:
                        ax_contrast.axvline(separation_planet, color="black", linestyle="--", label=f"{planet_name}" if planet_name is not None else "planet")
                        leg_loc = "upper right"
                    else:
                        if planet_to_star_ratio > ax_contrast.get_ylim()[1] or (planet_to_star_ratio > ax_contrast.get_ylim()[0] and planet_to_star_ratio < ax_contrast.get_ylim()[1]):
                            y_text = planet_to_star_ratio/1.5
                            if separation_planet > (IWA+OWA)/2:
                                x_text    = separation_planet - 0.1 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "center"
                            else:
                                x_text    = separation_planet + 0.025 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "right"
                        else :
                            y_text = planet_to_star_ratio*1.5
                            if separation_planet > (IWA+OWA)/2:
                                x_text    = separation_planet - 0.1 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "right"
                            else:
                                x_text    = separation_planet + 0.025 * separation[-1]
                                leg_y_pos = "lower"
                                leg_x_pos = "right"
                        leg_loc = leg_y_pos + " " + leg_x_pos
                        ax_contrast.plot([separation_planet, separation_planet], [planet_to_star_ratio, planet_to_star_ratio], 'ko')
                        ax_contrast.annotate(f"{planet_name}" if planet_name is not None else "planet", (x_text, y_text), fontsize=12)
                else:
                    leg_loc = "upper right"
                ax_contrast.legend(loc=leg_loc, fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
                ax_contrast_mag = ax_contrast.twinx()
                ax_contrast_mag.invert_yaxis()
                ax_contrast_mag.set_ylabel(r'$\Delta$mag', fontsize=14, labelpad=20, rotation=270)
                ax_contrast_mag.tick_params(axis='y', labelsize=12)        
                ymin_c_band, ymax_c_band = ax_contrast.get_ylim()
                ymin_c_band = max(ymin_c_band, 1e-10)
                ymax_c_band = min(ymax_c_band, 1e-1)
                ax_contrast.set_ylim(ymin_c_band, ymax_c_band)
                ax_contrast_mag.set_ylim(-2.5 * np.log10(ymin_c_band), -2.5 * np.log10(ymax_c_band))        
                fig_contrast.tight_layout()
                fig_contrast.show()
        
        # ===============
        # S/N computation
        # ===============
        elif calculation in {"SNR", "corner plot"}:
            
            if instru_type == "imager":
                SNR = np.sqrt(N_DIT) * signal / np.sqrt( sigma_fund_2 + N_DIT*sigma_syst_2 )
            
            else: # See Eq. (10) of Martos et al. 2025
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    SNR = np.sqrt(N_DIT) * signal / np.sqrt( sigma_fund_prime_2 + N_DIT*sigma_syst_prime_2 )
                
                elif post_processing.lower() in {"differential imaging", "di"}:
                    SNR = np.sqrt(N_DIT) * signal / np.sqrt( sigma_fund_prime_2 + N_DIT*sigma_speck_prime_2 + N_DIT*sigma_syst_prime_2 )
                
            # SNR values at the planet's separation
            if separation_planet is not None:
                if verbose:
                    if instru_type == "imager":
                        print_metric("S/N in the post-processed image", "S/N", f"{SNR[idx_planet_sep]:.1f}", f"(at {separation_planet:.1f} {sep_unit})")

                    else:
                        print_metric("S/N in the CCF", "S/N_CCF", f"{SNR[idx_planet_sep]:.1f}", f"(at {separation_planet:.1f} {sep_unit})")
                        if coronagraph is not None:
                            Mp_Sp_tot = fraction_core[idx_planet_sep] * N_DIT * Mp_Sp                                                                # Planet flux integrated over the FWHM and the exposure time [ph/bin/FWHM] (with modulations, if any)
                        else:
                            Mp_Sp_tot = fraction_core * N_DIT * Mp_Sp                                                                                # Planet flux integrated over the FWHM and the exposure time [ph/bin/FWHM] (with modulations, if any)
                        sigma_halo_2 = PSF_profile[idx_planet_sep] * trans*Ss                                                                        # Stellar flux at the considered separation in [e-/bin/px/DIT]
                        sigma_fund   = np.sqrt( R_corr[idx_planet_sep] * A_FWHM * N_DIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2 ) ) # [e-/bin/FWHM]
                        if post_processing.lower() in {"molecular mapping", "mm"}:
                            sigma_fund *= np.sqrt(fn_MM)
                        snr_per_ch = np.nanmedian(trans*Mp_Sp_tot / sigma_fund)
                        print_metric("Per-channel S/N", "S/N_λ", f"{snr_per_ch:.1f}", f"(at {separation_planet:.1f} {sep_unit})")
                if SNR[idx_planet_sep] > SNR_max_planet:
                    SNR_max_planet = SNR[idx_planet_sep]
                    band_SNR_max   = band
            
            # Adding the SNR curve of the band to the list
            SNR_bands.append(SNR)
        
            # -----------
            # Corner plot
            # -----------
            if calculation == "corner plot" and instru_type != "imager":
                planet_spectrum_instru_tot = planet_spectrum_instru.copy()
                if coronagraph is not None:
                    Mp_Sp_tot                        = fraction_core[idx_planet_sep] * N_DIT * Mp_Sp # Planet flux integrated over the FWHM and the exposure time [ph/bin/FWHM] (with modulations, if any)
                    planet_spectrum_instru_tot.flux *= fraction_core[idx_planet_sep] * exposure_time # [ph/bin/mn] => [ph/bin/FWHM]
                else:
                    Mp_Sp_tot                        = fraction_core * N_DIT * Mp_Sp # Planet flux integrated over the FWHM and the exposure time [ph/bin/FWHM] (with modulations, if any)
                    planet_spectrum_instru_tot.flux *= fraction_core * exposure_time # [ph/bin/mn] => [ph/bin/FWHM]
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp_tot, R_band, Rc, filter_type) # Filtered planet flux in [ph/bin/FWHM]
                Ss_HF, Ss_LF       = filtered_flux(Ss, R_band, Rc, filter_type)        # Filtered star flux in [ph/bin/DIT] (total e- over th FoV)
                d_planet           = trans*Mp_Sp_HF - trans*Ss_HF*Mp_Sp_LF/Ss_LF       # Flux in [e-/bin/FWHM] at the planet's location in the filtered cube: see Eq.(18) of Martos et al. 2025
                SNR_CCF            = SNR[idx_planet_sep]                               # Expected CCF SNR of the planet
                
                # Total noise in [e-/bin/FWHM] at the planet position
                sigma_halo_2 = PSF_profile[idx_planet_sep] * trans*Ss # Stellar flux at the planet position in [e-/bin/px/DIT]
                sigma_fund   = np.sqrt( R_corr[idx_planet_sep] * A_FWHM * N_DIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2) ) # Fundamental noise at the planet position integrated over the FWHM and the exposure time [e-/bin/FWHM]
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    sigma_fund *= np.sqrt(fn_MM)
                
                # Priors
                N         = 20
                T_arr     = np.linspace(T_planet-200,  T_planet+200,  N)
                lg_arr    = np.linspace(lg_planet-0.5, lg_planet+0.5, N)
                vsini_arr = np.linspace(max(vsini_planet-5, 0), vsini_planet+5, N)
                rv_arr    = np.linspace(rv_planet-10, rv_planet+10, N)
                
                # airmass value, if model == "tellurics"
                match = re.search(r"airmass=([^)]+)", model_planet)
                if match:
                    airmass = float(match.group(1))
                else:
                    airmass = None
                
                # Running retrieval and displaying corner plot
                lmin_model    = 0.98*lmin_instru                            # [µm]
                lmax_model    = 1.02*lmax_instru                            # [µm] (a bit larger than the instrumental bandwidth to avoid edge effects)
                R_model       = R_instru                                    # Spectral resolution
                dl_model      = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
                wave_model    = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
                star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)
                optimal_values, uncertainties = parameters_retrieval(instru=instru, band=band, target_name=planet_name, d=d_planet, trans_Ss=trans*Ss, wave=wave_band, trans=trans, model=model_planet, R=R_band, Rc=Rc, filter_type=filter_type, calc_logL=True, method_logL="classic", sigma_l=sigma_fund, weight=None, pca=pca, stellar_component=True, degrade_resolution=True, SNR_estimate=False, T_arr=T_arr, lg_arr=lg_arr, vsini_arr=vsini_arr, rv_arr=rv_arr, T=T_planet, lg=lg_planet, vsini=vsini_planet, rv=rv_planet, SNR_CCF=SNR_CCF, calc_d_sim=True, template=planet_spectrum_instru_tot, renorm_d_sim=False, fastcurves=True, Ss_HF=Ss_HF, Ss_LF=Ss_LF, wave_model=wave_model, epsilon=0.8, fastbroad=True, airmass=airmass, star_spectrum=star_spectrum, force_new_est=True, save=False, exposure_time=exposure_time, show=show_plot, verbose=verbose)
                                
                # Adding the uncertainties of the band to the list
                uncertainties_bands.append(uncertainties)
            
            # -------------------------------------------------------------------------------------------------------------------------------------------------
            # Diagnostic: impact of noise + systematics on correlation estimation (incl. auto-subtraction)
            # -------------------------------------------------------------------------------------------------------------------------------------------------
            if show_plot and show_cos_theta_est and post_processing.lower() in {"molecular mapping", "mm"}:
            
                # Planet flux and HF/LF split in [ph/bin/FWHM]
                if coronagraph is not None:
                    Mp_Sp_tot = fraction_core[idx_planet_sep] * N_DIT * Mp_Sp # Planet flux integrated over the FWHM and the exposure time [ph/bin/FWHM] (with modulations, if any)
                else:
                    Mp_Sp_tot = fraction_core * N_DIT * Mp_Sp                 # Planet flux integrated over the FWHM and the exposure time [ph/bin/FWHM] (with modulations, if any)
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp_tot, R_band, Rc, filter_type) # [ph/bin/FWHM]
                
                # Star flux and HF/LF split in [ph/bin] (total ph over the FoV)
                Ss_tot       = N_DIT * Ss                                     # Star flux integrated over the exposure time in [ph/bin] (total ph over the FoV)
                Ss_HF, Ss_LF = filtered_flux(Ss_tot, R_band, Rc, filter_type) # [ph/bin] (total ph over the FoV)                
                
                # Deterministic spectrum at planet location (Eq. 18 in Martos+2025)
                d_planet      = trans*Mp_Sp_HF - trans*Ss_HF*Mp_Sp_LF/Ss_LF # [e-/bin/FWHM]
                norm_d_planet = np.sqrt(np.nansum(d_planet**2))             # [e-/bin/FWHM]
                
                norm_d = np.zeros_like(separation, dtype=float)
                for i, s in enumerate(separation):
                    
                    # Per-channel noise at separation i (scalar per channel)
                    sigma_halo_2 = PSF_profile[i] * trans*Ss                                                                        # Stellar flux at the considered separation in [e-/bin/px/DIT]
                    sigma_fund   = np.sqrt( R_corr[i] * A_FWHM * N_DIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2 ) ) # [e-/bin/FWHM]
                    if post_processing.lower() in {"molecular mapping", "mm"}:
                        sigma_fund *= np.sqrt(fn_MM)
                    
                    # Norms and cos(theta_est)
                    norm_sigma_fund = np.sqrt(np.nansum(sigma_fund**2))
                    norm_d[i]       = np.sqrt( norm_d_planet**2 + norm_sigma_fund**2 )
            
                # Estimated correlation as function of the separation assuming cos_theta_p = 1
                cos_theta_est = np.nansum(d_planet * template) / norm_d
                
                # Pure fundamental-noise correlation limit
                cos_theta_n = alpha*N_DIT*fraction_core / norm_d
            
                # Plot
                fig_cos_theta_est = plt.figure(dpi=300, figsize=(10, 6))
                ax_cos_theta_est  = fig_cos_theta_est.gca()
                ax_cos_theta_est.plot(separation, cos_theta_est, 'k')
                ax_cos_theta_est.set_ylabel(r"cos $\theta_{\rm est}$", fontsize=14)
                ax_cos_theta_est.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_cos_theta_est.set_xlim(separation[0], separation[-1])
                ax_cos_theta_est.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax_cos_theta_est.minorticks_on()
                ax_cos_theta_est.set_title(f"Effect of noise and stellar subtraction on the correlation\nbetween template and planetary spectrum for {telescope_name}/{instru} on {band}\n(assuming the template equals the observed planet spectrum, i.e. "r"$cos\theta_{p}$ = 1)", fontsize=16)
                if (separation_planet is not None) and (separation_planet < np.nanmax(separation)):
                    print(f" beta/alpha = {beta/alpha:.3f} | cos_theta_n = {cos_theta_n[idx_planet_sep]:.3f} | cos_theta_syst = {cos_theta_syst_MM:.3f}")
                    if cos_theta_est_data is not None:
                        # Recover intrinsic mismatch estimate cos_theta_p from an observed cos_theta_est
                        cos_theta_p_data = (cos_theta_est_data / cos_theta_n[idx_planet_sep] + beta/alpha) / cos_theta_syst_MM
                        cos_theta_p_data = np.clip(a=cos_theta_p_data, a_min=-1, a_max=1)
                        print(f" cos_theta_est_data = {cos_theta_est_data:.3f} => cos_theta_p_data = {cos_theta_p_data:.3f}")
                    ax_cos_theta_est.axvline(separation_planet, c='k', ls="--", label=f'Angular separation{for_planet_name}')
                    ax_cos_theta_est.plot([separation_planet, separation_planet], [cos_theta_est[idx_planet_sep], cos_theta_est[idx_planet_sep]], 'rX', ms=11, label=rf"cos $\theta_{{est}}${for_planet_name} ({cos_theta_est[idx_planet_sep]:.2f})")
                    ax_cos_theta_est.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
                fig_cos_theta_est.tight_layout()
                fig_cos_theta_est.show()
            
        # =====================================================================
        # Optional t_syst in [mn] diagnostic: see Eq.(14) of Martos et al. 2025
        # =====================================================================
        if show_plot and show_t_syst and systematics:
            t_syst = DIT * R_corr * A_FWHM * ( sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2 ) / sigma_syst_prime_2 # [mn]
            if post_processing.lower() in {"molecular mapping", "mm"}:
                t_syst *= fn_MM
            fig_t_syst    = plt.figure(figsize=(10, 6), dpi=300)
            ax_fig_t_syst = fig_t_syst.gca()
            ax_fig_t_syst.plot(separation, t_syst, c="crimson", ls="-", label="$t_{syst}$")
            ax_fig_t_syst.set_ylabel('$t_{syst}$ [mn]', fontsize = 14)
            ax_fig_t_syst.set_xlabel(f'Separation [{sep_unit}]', fontsize = 14)
            ax_fig_t_syst.set_xlim(separation[0], separation[-1])
            ax_fig_t_syst.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_fig_t_syst.minorticks_on()
            ax_fig_t_syst.set_title(r"$t_{syst}$"f" on {band}\nwith "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}", fontsize = 14)
            ax_fig_t_syst.plot([separation[0], separation[-1]], [exposure_time, exposure_time], c="black", ls="--", label="$t_{exp}$ ="+f"{round(exposure_time)} mn")
            ax_fig_t_syst.set_yscale('log')
            ax_fig_t_syst.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
            fig_t_syst.tight_layout()  
            fig_t_syst.show()
        
        # =======================
        # Optional Systematic (+speckles) noise budget
        # =======================
        if show_syst_budget and post_processing.lower() in {"molecular mapping", "mm"} and instru_type != "imager" and not return_FastYield:
            
            
            
            # Computing the stellar halo profile in [e-/bin/FWHM] (if there is a coronagraph, PSF_profile already contain the coronagraphic transmission for the star)
            H         = PSF_profile[None, :] * (trans*Ss)[:, None] # (wave, separation): Stellar halo [e-/bin/px/DIT], H = Ms * trans * Ss
            H        *= A_FWHM * N_DIT                             # (wave, separation): Stellar halo [e-/bin/FWHM] (integrated over the FWHM and exposure time)
            H[H <= 0] = np.nanmin(H[H > 0])                        # (wave, separation): Filling invalid values
            # "Undo" FPM effect 
            if instru == "HARMONI" and iwa_FPM > 0:
                H[:, separation < iwa_FPM] /= 1e-4
                        
            # Ponderated halo: h = H * t (wave, separation)
            h_MM = np.nan_to_num(H * template_MM[:, None], nan=0.0, posinf=0.0, neginf=0.0)
            h_DI = np.nan_to_num(H * template_DI[:, None], nan=0.0, posinf=0.0, neginf=0.0)
            
            # Converting in float32 to fasten the computations
            D    = np.float32(D)
            wave = np.ascontiguousarray(wave_band, dtype=np.float32)
            H    = np.ascontiguousarray(H,         dtype=np.float32)
            h_MM = np.ascontiguousarray(h_MM,      dtype=np.float32)
            h_DI = np.ascontiguousarray(h_DI,      dtype=np.float32)
                             
            # Computation parameters
            n_sigma_cut = 4.0
            fast_comp   = True
                        
            # Separation in [rad]
            if sep_unit == "mas":
                separation_rad            = np.ascontiguousarray(separation / (rad2arcsec * 1000), dtype=np.float32)
                pxscale_rad               = np.float32(pxscale / (rad2arcsec * 1000))
                if separation_planet is not None:
                    separation_planet_rad = np.float32(separation_planet / (rad2arcsec * 1000))
            else:
                separation_rad            = np.ascontiguousarray(separation / rad2arcsec, dtype=np.float32)
                pxscale_rad               = np.float32(pxscale / rad2arcsec)
                if separation_planet is not None:
                    separation_planet_rad = np.float32(separation_planet / rad2arcsec)
            
            # Conservative reference separation for wave compression
            if fast_comp:
                if return_quantity and separation_planet is not None:
                    separation_ref = separation_planet
                else:
                    separation_ref = np.nanmax(separation)
                wave, h_DI, h_MM, n_merge = compress_h_for_P(wave=wave_band, h_DI=h_DI, h_MM=h_MM, D=D, separation_ref=separation_ref, sep_unit=sep_unit, oversample=3.0, max_merge=64)
                if verbose:
                    print_info(f"  P coarse compression: merged {n_merge} native channels per coarse bin ({len(wave_band)} -> {len(wave)} channels)", sub=True)

            # Computing P_DI and P_MM    
            if return_quantity and separation_planet is not None: # making the computations faster
                P_DI      = compute_P_speck_numba(  h=h_DI[:, idx_planet_sep:idx_planet_sep+1], wave=wave, separation_rad=np.array([separation_planet_rad], dtype=np.float32), D=D,                     n_sigma_cut=n_sigma_cut)
                P_al_spat = compute_P_al_spat_numba(h=h_MM[:, idx_planet_sep:idx_planet_sep+1], wave=wave, separation_rad=np.array([separation_planet_rad], dtype=np.float32), pxscale_rad=pxscale_rad, n_sigma_cut=n_sigma_cut)
                P_al_spec = compute_P_al_spec_numba(h=h_MM[:, idx_planet_sep:idx_planet_sep+1], wave=wave,                                                                                              n_sigma_cut=n_sigma_cut)
                P_MM      = 0.5 * (P_al_spec + P_al_spat)
                P_DI      = np.full_like(separation, P_DI[0], dtype=float)
                P_MM      = np.full_like(separation, P_MM[0], dtype=float)
                
            else:
                P_DI      = compute_P_speck_numba(  h=h_DI, wave=wave, separation_rad=separation_rad, D=D,                     n_sigma_cut=n_sigma_cut)
                P_al_spat = compute_P_al_spat_numba(h=h_MM, wave=wave, separation_rad=separation_rad, pxscale_rad=pxscale_rad, n_sigma_cut=n_sigma_cut)
                P_al_spec = compute_P_al_spec_numba(h=h_MM, wave=wave,                                                         n_sigma_cut=n_sigma_cut)
                P_MM      = 0.5 * (P_al_spec + P_al_spat)
                                    
            # Fundamental noise profile, projected in the CCF domain with MM and DI in [e-/FWHM]
            if np.isscalar(sigma_bkg_2):
                sigma_bkg_2_vec = np.zeros_like(wave)
            else:
                sigma_bkg_2_vec = sigma_bkg_2
            sigma_halo_2        = H / A_FWHM / N_DIT                                                                                      # (wave, separation): Stellar halo [e-/bin/px/DIT], H = Ms * trans * Ss
            sigma_fund_2        = R_corr[None, :] * A_FWHM * N_DIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2_vec[:, None]) # (wave, separation): Fundamental noise profile integrated over the FWHM and the exposure time in [e-/bin/FWHM]
            sigma_fund_prime_MM = np.sqrt( fn_MM * np.nansum(sigma_fund_2 * template_MM[:, None]**2, axis=0) )                            # (separation):       Fundamental noise profile projected in the MM CCF integrated over the FWHM and the exposure time in [e-/FWHM]
            sigma_fund_prime_DI = np.sqrt( 1     * np.nansum(sigma_fund_2 * template_DI[:, None]**2, axis=0) )                            # (separation):       Fundamental noise profile projected in the DI CCF integrated over the FWHM and the exposure time in [e-/FWHM]
            
            # Systematic noise profile, projected in the CCF domain with MM and DI in [e-/FWHM]
            sigma_syst_prime    = np.sqrt(N_DIT**2 * sigma_syst_prime_2) # (separation): Systematic noise profile projected in the MM CCF integrated over the exposure time in [e-/FWHM]
            sigma_syst_prime_MM = sigma_syst_prime
            sigma_syst_prime_DI = sigma_syst_prime # assuming that sigma_syst_prime_DI ~ sigma_syst_prime_MM ~ sigma_syst_prime_DI
            
            # -----------------------
            # Systematic noise budget
            # -----------------------
            sigma_m_syst = sigma_fund_prime_MM / np.sqrt(P_MM)

            if show_plot:
                mask_iwa_FPM     = separation >= iwa_FPM
                fig_sigma_m_syst = plt.figure(figsize=(10, 6), dpi=300)
                ax_sigma_m_syst  = fig_sigma_m_syst.gca()
                ax_sigma_m_syst.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax_sigma_m_syst.set_xlim(0, separation[-1])
                ax_sigma_m_syst.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_sigma_m_syst.set_ylabel("Systematics modulation budget [%]", fontsize=14)
                ax_sigma_m_syst.set_title(f"{telescope_name}/{instru} systematics budget with {post_processing} on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax_sigma_m_syst.tick_params(axis='both', labelsize=12)
                if iwa_FPM > 0:
                    ax_sigma_m_syst.axvspan(0, iwa_FPM, color='black', alpha=0.3, lw=0)
                ax_sigma_m_syst.plot(separation[mask_iwa_FPM], 100*sigma_m_syst[mask_iwa_FPM], c="black")
                if systematics:
                    ax_sigma_m_syst.plot(separation[mask_iwa_FPM], 100*np.nanmedian( np.sqrt(N_DIT**2*sigma_syst_prime_2) / H, axis=0)[mask_iwa_FPM], c="black", ls="--", label="Real systematic level") # assuming that sigma_syst_prime_2 ~ mean( sigma_syst_2 )
                    ax_sigma_m_syst.legend(fontsize=12, loc="upper right", frameon=True, fancybox=True, shadow=True, borderpad=1)
                ymin_s, ymax_s = ax_sigma_m_syst.get_ylim()
                ymin_s         = max(ymin_s, 0.)
                ax_sigma_m_syst.set_ylim(ymin_s, ymax_s)
                ax_sigma_m_syst.fill_between(separation[mask_iwa_FPM], 100*sigma_m_syst[mask_iwa_FPM], y2=ymax_s, color='crimson', alpha=0.3)
                ax_sigma_m_syst.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "Systematic-dominated regime", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_syst.fill_between(separation[mask_iwa_FPM], 100*sigma_m_syst[mask_iwa_FPM], y2=ymin_s, color='royalblue', alpha=0.3)
                ax_sigma_m_syst.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "Fundamental-dominated regime", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_syst.minorticks_on()
                fig_sigma_m_syst.tight_layout()
                fig_sigma_m_syst.show()
            
            # --------------------
            # Speckle noise budget
            # --------------------
            
            # sigma_speck_prime = sigma_speck,CCF
            sigma_speck_prime = np.sqrt( (delta*cos_theta_p_DI*cos_theta_syst_DI / (alpha*cos_theta_p_MM*cos_theta_syst_MM - beta))**2 * (sigma_fund_prime_MM**2 + sigma_syst_prime_MM**2) - sigma_fund_prime_DI**2 - sigma_syst_prime_DI**2 )
            sigma_m_speck     = sigma_speck_prime / np.sqrt(P_DI)
            
            if show_plot:
                fig_sigma_m_peckles = plt.figure(figsize=(10, 6), dpi=300)
                ax_sigma_m_peckles  = fig_sigma_m_peckles.gca()
                ax_sigma_m_peckles.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax_sigma_m_peckles.set_xlim(0, separation[-1])
                ax_sigma_m_peckles.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_sigma_m_peckles.set_ylabel("Speckles stability budget [%]", fontsize=14)
                ax_sigma_m_peckles.set_title(f"{telescope_name}/{instru} speckles stability budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax_sigma_m_peckles.tick_params(axis='both', labelsize=12)    
                if iwa_FPM > 0:
                    ax_sigma_m_peckles.axvspan(0, iwa_FPM, color='black', alpha=0.3, lw=0)
                ax_sigma_m_peckles.plot(separation[mask_iwa_FPM], 100*sigma_m_speck[mask_iwa_FPM], c="black")
                ymin_s, ymax_s = ax_sigma_m_peckles.get_ylim()
                ymin_s         = max(ymin_s, 0.)
                ax_sigma_m_peckles.set_ylim(ymin_s, ymax_s)
                ax_sigma_m_peckles.fill_between(separation[mask_iwa_FPM], 100*sigma_m_speck[mask_iwa_FPM], y2=ymax_s, color='crimson', alpha=0.3)
                ax_sigma_m_peckles.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "MM > DI", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_peckles.fill_between(separation[mask_iwa_FPM], 100*sigma_m_speck[mask_iwa_FPM], y2=ymin_s, color='royalblue', alpha=0.3)
                ax_sigma_m_peckles.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "MM < DI", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_peckles.minorticks_on()
                fig_sigma_m_peckles.tight_layout()
                fig_sigma_m_peckles.show()
            
            # ---- Non-coronagraphic PSF ---- 
            if coronagraph is not None:
                H_ref, _, _, _ = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=None, instru=instru, separation_planet=separation_planet, return_FastYield=return_FastYield, sampling=2 if show_syst_budget else 10)
            else:
                H_ref = np.copy(PSF_profile)

                # "Undo" FPM effect 
                if instru == "HARMONI" and iwa_FPM > 0:
                    H_ref[separation < iwa_FPM] /= 1e-4
            
            # ---- Estimating SR_ref and phi_ref with diffraction model ----
            ww          = trans * np.gradient(wave_band) # (wave)
            r           = separation
            dr          = np.gradient(r) # [mas/px]
            edges       = np.empty(len(r) + 1, dtype=float)
            edges[1:-1] = 0.5 * (r[:-1] + r[1:])
            edges[0]    = 0.0
            edges[-1]   = r[-1] + 0.5 * (r[-1] - r[-2]) if len(r) > 1 else r[0]
            area        = np.pi * (edges[1:]**2 - edges[:-1]**2)
            try:
                H_diff_ref, _, _, _ = get_PSF_profile(band=band, strehl="DL", apodizer=apodizer, coronagraph=None, instru=instru, separation_planet=separation_planet, return_FastYield=return_FastYield, sampling=2 if show_syst_budget else 10)
            except Exception as e:
                print_warning(f"WARNING: Estimating the DL PSF as the Airy profile (with central obstruction): {e}")
                H_diff_ref = airy_profile(wave=wave_band, separation=separation, sep_unit=sep_unit, D=D, eps=eps) # (wave, separation)
                H_diff_ref = np.nansum(H_diff_ref * ww[:, None], axis=0) / np.nansum(ww)                          # (separation)
                H_diff_ref = H_diff_ref * np.nansum(H_ref * area) / np.nansum(H_diff_ref * area)                  # Same flux inside FoV
            
            r_core    = get_r_core(separation=separation, profile=H_diff_ref) # separation where H_diff_ref(r_core) = 0.5 * max(H_diff_ref)
            mask_core = (separation <= r_core)
            w         = area
            SR_ref    = np.nansum(w[mask_core]*H_ref[mask_core]*H_diff_ref[mask_core]) / np.nansum(w[mask_core]*H_diff_ref[mask_core]**2)
            SR_ref    = np.clip(SR_ref, 1e-8, 1.0)
            phi_ref   = np.sqrt(-np.log(SR_ref))
            if verbose:
                wave_eff = np.nansum(wave_band * ww) / np.nansum(ww)  # µm
                WFE_nm   = 1e3*wave_eff / (2*np.pi) * phi_ref
                print_metric("Estimated Strehl Ratio", "SR_ref", f"{100*SR_ref:.1f}", "%")
                print_metric("Estimated WFE",          "WFE_nm", f"{WFE_nm:.1f}",     "nm RMS")
            
            # ---- Estimating eta ----
            if coronagraph is not None:
                eta = 1
            else:
                eta_raw       = ( H_ref - SR_ref * H_diff_ref ) / H_ref # = H_speck_ref / H_ref (H_ref = SR_ref*H_diff_ref + H_speck_ref)
                valid         = np.isfinite(eta_raw) & (eta_raw > 0) & (eta_raw < 1)
                eta_interp    = interp1d(separation[valid], eta_raw[valid], bounds_error=False, fill_value=np.nan)(separation)
                eta_interp    = np.nan_to_num(eta_interp)
                sigma_bins    = np.nanmedian(2*r_core / dr) # [px]
                eta           = gaussian_filter1d(eta_interp, sigma=sigma_bins) 
                eta[eta <= 0] = np.nanmin(eta[eta > 0])
                
                # plt.figure(dpi=300, figsize=(10, 6))
                # plt.title(f"{telescope_name}/{instru} with {apodizer}-apodizer in {band}-band", fontsize=14)
                # plt.xlabel("separation [mas]", fontsize=12)
                # plt.ylabel("eta = H_speck_ref / H_ref")
                # plt.xlim(separation[0], separation[-1])
                # plt.ylim(-1, 1)
                # plt.axhline(0, c="k", lw=3)
                # plt.plot(separation, ( H_ref - SR_ref * H_diff_ref ) / H_ref, label="eta (raw)")
                # plt.plot(separation, eta_interp,                    label="eta (interp)")
                # plt.plot(separation, eta,                           label="eta (LF)")
                # plt.legend()
                # plt.grid(True)
                # plt.show()
                
                # H_speck_ref = eta * H_ref 
                # plt.figure(dpi=300, figsize=(10, 6))
                # plt.xlim(separation[0], separation[-1])
                # plt.title(f"{telescope_name}/{instru} with {apodizer}-apodizer in {band}-band", fontsize=14)
                # plt.xlabel("separation [mas]", fontsize=12)
                # plt.ylabel("Profiles")
                # plt.yscale('log')
                # plt.plot(separation, H_ref,   label="H_ref")
                # plt.plot(separation, H_diff_ref,    label="H_diff_ref")
                # plt.plot(separation, H_speck_ref, label="H_speck_ref")
                # plt.legend()
                # plt.grid(True)
                # plt.show()
            
            # ---- Computing sigma_m_speck equivalent phase stability ---- 
            rho_phi = 0.0
            
            # With the approximation: SR_ref * I_diff,ref ~ SR_obs * I_diff,obs
            sigma_phi_speck = phi_ref * (-rho_phi + np.sqrt(rho_phi**2 + sigma_m_speck/eta))
            
            # phase [rad] -> WFE [nm RMS]
            sigma_phi_speck = 1e3*np.nanmean(wave_band) / (2*np.pi) * sigma_phi_speck
            
            if show_plot:
                fig_sigma_phi_speck = plt.figure(figsize=(10, 6), dpi=300)
                ax_sigma_phi_speck  = fig_sigma_phi_speck.gca()
                ax_sigma_phi_speck.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax_sigma_phi_speck.set_xlim(0, separation[-1])
                ax_sigma_phi_speck.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_sigma_phi_speck.set_ylabel("Wavefront RMS stability [nm]", fontsize=14)
                ax_sigma_phi_speck.set_title(f"{telescope_name}/{instru} wavefront stability budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax_sigma_phi_speck.tick_params(axis='both', labelsize=12)        
                if iwa_FPM > 0:
                    ax_sigma_phi_speck.axvspan(0, iwa_FPM, color='black', alpha=0.3, lw=0)
                ax_sigma_phi_speck.plot(separation[mask_iwa_FPM], sigma_phi_speck[mask_iwa_FPM], c="black")
                ymin_s, ymax_s = ax_sigma_phi_speck.get_ylim()
                ymin_s         = max(ymin_s, 0.)
                ax_sigma_phi_speck.set_ylim(ymin_s, ymax_s)
                ax_sigma_phi_speck.fill_between(separation[mask_iwa_FPM], sigma_phi_speck[mask_iwa_FPM], y2=ymax_s, color='crimson', alpha=0.3)
                ax_sigma_phi_speck.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "MM > DI", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_phi_speck.fill_between(separation[mask_iwa_FPM], sigma_phi_speck[mask_iwa_FPM], y2=ymin_s, color='royalblue', alpha=0.3)
                ax_sigma_phi_speck.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "MM < DI", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_phi_speck.minorticks_on()
                fig_sigma_phi_speck.tight_layout()
                fig_sigma_phi_speck.show()
            
            # Saving quantities
            sigma_m_syst_bands.append(sigma_m_syst)
            sigma_m_speck_bands.append(sigma_m_speck)
            sigma_phi_speck_bands.append(sigma_phi_speck)
            
    sigma_budget_bands = [sigma_m_syst_bands, sigma_m_speck_bands, sigma_phi_speck_bands]
    
    # =======================
    # Final plot
    # ======================= 
    
    # Figure cleanup
    if show_plot:
        plt.show()
        
    IWA_FPM = np.max(iwa_FPM_bands)
        
    # ---------------------------------------
    # 5σ contrast
    # ---------------------------------------
    if calculation == "contrast" and show_plot and band_only is None:
        fig_contrast = plt.figure(figsize=(10, 6), dpi=300)        
        ax_contrast  = fig_contrast.gca()
        ax_contrast.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
        ax_contrast.minorticks_on()
        ax_contrast.tick_params(axis='both', labelsize=12)  
        ax_contrast.set_yscale('log')
        ax_contrast.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        if coronagraph is not None:
            mask_title = f" with {coronagraph} coronagraph,"
        elif apodizer != "NO_SP":
            mask_title = f" with {apodizer.replace('_', ' ')} apodizer,"
        else:
            mask_title = ""
        ax_contrast.set_title(f"{telescope_name}/{instru} {post_processing} contrast curves{for_planet_name} with $t_{{exp}}$ = {int(round(exposure_time))} mn," + mask_title + f"\n $mag_*$({band0_star}) = {round(mag_star, 1)} and $T_p$ = {int(round(T_planet))} K ({model_planet.replace('_Earth-like', '(Earth-like)')} model)", fontsize=16)        
        ax_contrast.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
        ax_contrast.set_ylabel(r'5$\sigma$ contrast (on instru-band)', fontsize=14)
        if IWA_FPM > 0:
            ax_contrast.axvspan(0, IWA_FPM, color='black', alpha=0.3, lw=0)
        for i in range(len(contrast_bands)):
            mask_IWA_FPM = separation_bands[i] >= IWA_FPM
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax_contrast.plot(separation_bands[i][mask_IWA_FPM], contrast_bands[i][mask_IWA_FPM], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax_contrast.plot(separation_bands[i][mask_IWA_FPM], contrast_bands[i][mask_IWA_FPM], label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
        if separation_planet is not None:
            if separation_planet > 2 * OWA:
                ax_contrast.set_xscale('log')
                ax_contrast.set_xlim(IWA, max(np.max(arr) for arr in separation_bands))
            if mag_planet is None:
                ax_contrast.axvline(separation_planet, color="black", linestyle="--", label=f"{planet_name}" if planet_name is not None else "planet")
            else:
                if planet_to_star_ratio > ax_contrast.get_ylim()[1] or (planet_to_star_ratio > ax_contrast.get_ylim()[0] and planet_to_star_ratio < ax_contrast.get_ylim()[1]):
                    y_text = planet_to_star_ratio/1.5
                    if separation_planet > (IWA+OWA)/2:
                        x_text    = separation_planet - 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "center"
                    else:
                        x_text    = separation_planet + 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "right"
                else :
                    y_text = planet_to_star_ratio*1.5
                    if separation_planet > (IWA+OWA)/2:
                        x_text    = separation_planet - 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "right"
                    else:
                        x_text    = separation_planet + 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "lower"
                        leg_x_pos = "right"
                leg_loc = leg_y_pos + " " + leg_x_pos
                ax_contrast.plot([separation_planet, separation_planet], [planet_to_star_ratio, planet_to_star_ratio], 'ko')
                ax_contrast.annotate(f"{planet_name}" if planet_name is not None else "planet", (x_text, y_text), fontsize=12)
        else:
            leg_loc = "upper right"
        ax_contrast.legend(loc=leg_loc, fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        ax_contrast_mag = ax_contrast.twinx()
        ax_contrast_mag.invert_yaxis()
        ax_contrast_mag.set_ylabel(r'$\Delta$mag', fontsize=14, labelpad=20, rotation=270)
        ax_contrast_mag.tick_params(axis='y', labelsize=12)        
        ymin, ymax = ax_contrast.get_ylim()
        ax_contrast_mag.set_ylim(-2.5 * np.log10(ymin), -2.5 * np.log10(ymax))        
        fig_contrast.tight_layout()        
        plt.show()
    
    # ---------------------------------------
    # SNR
    # ---------------------------------------     
    elif calculation == "SNR" and show_plot:
        
        if channel and instru == "MIRIMRS":
            # Each MIRIMRS channel consists of 3 sub-bands (SHORT, MEDIUM, LONG).
            # To combine them, scale exposure time accordingly.
            exposure_time     = exposure_time * 3  
            NbChannels        = 4
            bands_per_channel = 3
            SNR_channels        = []  # Combined SNR per channel
            separation_channels = []  # Separation grid per channel
            name_channels       = []  # Channel labels
            for ch in range(NbChannels):
                start_idx = ch * bands_per_channel
                end_idx   = start_idx + bands_per_channel
                # Combine SNR curves from the 3 sub-bands via quadrature sum
                snr_combined = np.sqrt(np.nansum(np.array(SNR_bands[start_idx:end_idx])**2, axis=0))
                SNR_channels.append(snr_combined)
                # Use the separation values from the first sub-band (SHORT) as reference
                separation_channels.append(separation_bands[start_idx])
                # Assign channel name for later display
                name_channels.append(f"Channel {ch + 1}")
            # If a planet separation is specified, find its maximum SNR
            SNR_max_planet = 0.0
            band_SNR_max   = ""
            if separation_planet is not None:
                for i, snr_curve in enumerate(SNR_channels):
                    # Ensure the separation exists in the grid before indexing
                    idx_planet_sep = np.nanargmin(np.abs(separation_channels[i] - separation_planet))
                    if snr_curve[idx_planet_sep] > SNR_max_planet:
                        SNR_max_planet = snr_curve[idx_planet_sep]
                        band_SNR_max   = name_channels[i]
            # Update bands and separations for consistency downstream
            separation_bands = separation_channels
            SNR_bands        = SNR_channels
            name_bands       = name_channels

        if separation_planet is not None and verbose:
            print()
            print_info(f"MAX S/N (at {separation_planet:.1f} {sep_unit}) = {SNR_max_planet:.1f} for {band_SNR_max.replace('_', ' ')}.")
        
        fig_SNR = plt.figure(figsize=(10, 6), dpi=300) 
        ax_SNR  = fig_SNR.gca()
        ax_SNR.grid(which='both', linestyle=':', color='gray', alpha=0.5)     
        ax_SNR.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        if coronagraph is not None:
            mask_title = f" with {coronagraph} coronagraph,"
        elif apodizer != "NO_SP":
            mask_title = f" with {apodizer} apodizer,"
        else:
            mask_title = ""
        ax_SNR.set_title(f"{telescope_name}/{instru} {post_processing} S/N curves{for_planet_name} with $t_{{exp}}$ = {int(round(exposure_time))} mn," + mask_title + f"\n $mag_*$({band0_star}) = {round(mag_star, 1)}, $mag_p$({band0_planet}) = {round(mag_planet, 1)}, $T_p$ = {int(round(T_planet))}K ({model_planet} model)", fontsize=16)
        ax_SNR.set_xlabel(f"separation [{sep_unit}]", fontsize=14)
        ax_SNR.set_ylabel('S/N', fontsize=14)
        if IWA_FPM > 0:
            ax_SNR.axvspan(0, IWA_FPM, color='black', alpha=0.3, lw=0)
        for i in range(len(SNR_bands)):
            mask_IWA_FPM = separation_bands[i] >= IWA_FPM
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax_SNR.plot(separation_bands[i][mask_IWA_FPM], SNR_bands[i][mask_IWA_FPM], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax_SNR.plot(separation_bands[i][mask_IWA_FPM], SNR_bands[i][mask_IWA_FPM], label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
        if separation_planet is not None:
            if separation_planet > 2 * OWA:
                ax_SNR.set_xscale('log')
                ax_SNR.set_xlim(IWA, max(np.max(arr) for arr in separation_bands))
            ax_SNR.axvline(x=separation_planet, color='k', linestyle='--', linewidth=1.5)
            ax_SNR.plot([separation_planet], [SNR_max_planet], 'rX', ms=11)        
            ax_legend = ax_SNR.twinx()
            ax_legend.plot([], [], '--', c='k', label=f'Angular separation{for_planet_name}')
            ax_legend.plot([], [], 'X', c='r', label=f'Max S/N{for_planet_name} ({round(SNR_max_planet, 2)})')
            ax_legend.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
            ax_legend.tick_params(axis='y', colors='w')
        ax_SNR.set_ylim(0)
        ax_SNR.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        ax_SNR.minorticks_on()
        ax_SNR.yaxis.set_ticks_position('both')
        ax_SNR.tick_params(axis='both', labelsize=12)  
        fig_SNR.tight_layout()        
        plt.show()
    
    # Figure cleanup
    if show_plot:
        plt.show()
    
    # ---------------------------------------
    # RETURNS
    # ---------------------------------------
    if calculation == "contrast":
        results_bands = contrast_bands
    elif calculation == "SNR":
        results_bands = SNR_bands
    elif calculation == "corner plot":
        results_bands = uncertainties_bands
    return name_bands, separation_bands, results_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands, trans_bands, sigma_budget_bands



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves init
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation=None, instru=None, exposure_time=None, mag_star=None, mag_planet=None, band0=None, band0_star=None, band0_planet=None, model_planet=None, T_planet=None, lg_planet=None, model_star="BT-NextGen", T_star=None, lg_star=None, rv_star=0, rv_planet=0, vsini_star=0, vsini_planet=0, 
             apodizer="NO_SP", strehl="NO_JQ", coronagraph=None, systematics=False, speckles=False, PCA=False, PCA_mask=False, N_PCA=20, channel=False, planet_name=None, separation_planet=None, show_plot=True, verbose=True, 
             post_processing="molecular mapping", background="medium", Rc=100, filter_type="gaussian", input_DIT=None, band_only=None, 
             star_spectrum=None, planet_spectrum=None, return_FastYield=False, return_quantity=False):
    """
    Orchestrates inputs (spectra, models, config) and delegates to FastCurves_main
    to compute contrast/SNR/uncertainty for the selected instrument/setup.
    
    Note: 'sigma_prime' detones the 'sigma' per spectral channel projected into the CCF 
float
    Parameters
    ----------
    calculation: {"contrast","SNR","corner plot"}
        Type of calculation to perform.
    instru: str
        Instrument key (must be in 'instrus' and supported by config).
    exposure_time: float
        Total exposure time [minutes].
    mag_star, mag_planet: float
        Magnitudes of star/planet. Planet magnitude required for SNR/corner-plot.
    band0, band0_star, band0_planet: str
        Photometric bands for magnitudes. If only 'band0' is given, it's used for both.
    model_planet, T_planet, lg_planet: str, float, 
        Planet atmospheric model + parameters (Teff [K], log g [dex(cm/s^2)]).
    model_star, T_star, lg_star: str, float, float
        Star atmospheric model + parameters (Teff [K], log g [dex(cm/s^2)]).
    rv_*, vsini_*: float
        Radial velocity and projected rotation [km/s] for star/planet.
    apodizer, strehl, coronagraph: str | None
        Optical setup (validated against the instrument config).
    systematics, speckles, PCA, PCA_mask, N_PCA :
        Systematics and speckles noise modeling and PCA removal configuration.
    channel: bool
        For MIRIMRS, combine SNR by channel (3 sub-bands per channel).
    planet_name: str
        Label only (for plots/prints).
    separation_planet: float
        Planet separation in **arcsec** (converted internally if needed).
    post_processing: {"molecular mapping","differential imaging"} or None
        Defaults based on instrument type (IFU → molecular mapping, imager → differential imaging).
    background: {"low","medium","high"} or None
        Background level.
    Rc: float or None
        High-pass cutoff resolution (None → no filtering).
    filter_type: str
        Filtering kernel ("gaussian", "step", "smoothstep", ...).
    input_DIT: float or None
        Force a particular DIT [minutes].
    band_only: str or None
        If provided, compute only that band.
    star_spectrum, planet_spectrum: Spectrum-like or None
        Provide pre-loaded spectra; otherwise they are loaded from models.
    return_FastYield: bool
        FastYield helper: returns per-band SNR and components at planet separation.
    return_quantity: bool
        Returns deeper per-band quantities (signals/noises) for analysis.

    Returns
    -------
    - If return_FastYield:
        (name_bands, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_bands)
    - Elif return_quantity:
        (name_bands, separation, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands,
         sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands,
         planet_flux_bands, star_flux_bands, wave_bands)
    - Else (standard):
        (name_bands, separation, results_bands)
    """
    time1       = time.time()
    config_data = get_config_data(instru)
    
    # ---- Validate calculation entry ----
    if calculation not in {"contrast", "SNR", "corner plot"}:
        raise KeyError(f"calculation={calculation} is not valid. Available: 'contrast', 'SNR', 'corner plot'.")
    if instru not in instrus:
        raise KeyError(f"instru={instru} is not valid. Available: {instrus}")
        
    # ---- Validate post-processing entry ----
    if post_processing.lower() not in {"molecular mapping", "mm", "differential imaging", "di"}:
        raise KeyError(f"post_processing={post_processing} is not valid. Available: 'Molecular Mapping', 'MM', 'Differential Imaging', 'DI'")
    if config_data["type"] == "imager" and post_processing.lower() not in {"differential imaging", "di"}:
        post_processing = "differential imaging"
    if config_data["type"] == "fiber_injection_HRS" and post_processing.lower() in {"differential imaging", "di"}:
        raise KeyError("Fiber-fed spectrograph cannot use Differential Imaging (DI) as post-processing")
    
    # ---- Validate optics options against config ----
    if apodizer not in config_data["apodizers"]:
        raise KeyError(f"No PSF profiles for apodizer '{apodizer}' with {instru}. Available: {config_data.get('apodizers', [])}")
    if strehl not in config_data["strehls"]:
        raise KeyError(f"No PSF profiles for strehl '{strehl}' with {instru}. Available: {config_data.get('strehls', [])}")
    if coronagraph not in config_data["coronagraphs"]:
        raise KeyError(f"No PSF profiles for coronagraph '{coronagraph}' with {instru}. Available: {config_data.get('coronagraphs', [])}")
    
    # ---- Validate band_only ----
    if band_only is not None and band_only not in [band for band in config_data["gratings"]]:
        raise KeyError(f"band_only={band_only} is not valid. Available: {[band for band in config_data['gratings']]}")
    
    # ---- Config & observing mode (space vs ground) ----
    if config_data["base"] == "space":
        tellurics = False
        strehl    = "NO_JQ"
    elif config_data["base"] == "ground":
        tellurics = True 
    
    # Angular separation unit expected by the instrument (e.g., 'arcsec' or 'mas')
    sep_unit = config_data["sep_unit"]
    
    # ---- band0 logic ----
    if band0 is None and band0_star is None and band0_planet is None:
        raise KeyError("Please define at least one of 'band0', 'band0_star', or 'band0_planet'.")
    if band0_star is None:
        band0_star = band0
    if band0_star.lower() != "instru" and band0_star not in bands:
        raise KeyError(f"{band0_star} is not a recognized magnitude band. Choose among: {bands} or 'instru'")
    if band0_planet is None:
        band0_planet = band0
    if band0_planet.lower() != "instru" and band0_planet not in bands:
        raise KeyError(f"{band0_planet} is not a recognized magnitude band. Choose among: {bands} or 'instru'")
    
    # Wavelength ranges [µm]
    lmin_instru = config_data["lambda_range"]["lambda_min"] # [µm]
    lmax_instru = config_data["lambda_range"]["lambda_max"] # [µm]
    lmin_band0_star, lmax_band0_star     = get_band_lims(band=instru if band0_star.lower()   == "instru" else band0_star)   # [µm]
    lmin_band0_planet, lmax_band0_planet = get_band_lims(band=instru if band0_planet.lower() == "instru" else band0_planet) # [µm]
    if band_only is not None:
        lmin_band_only, lmax_band_only = get_band_lims(band=band_only) # [µm]

    # ---- Load / prepare star spectrum if not provided ----
    if star_spectrum is None:
        if (model_star is None) or (T_star is None) or (lg_star is None):
            raise KeyError("Please define model_star, T_star, and lg_star to load the star spectrum.")
        star_spectrum = load_star_spectrum(T_star, lg_star, model=model_star)
        # Crop around photometric & instrument ranges
        lmin_star = 0.98*min(lmin_instru, lmin_band0_star)
        lmax_star = 1.02*max(lmax_instru, lmax_band0_star)
        star_spectrum.crop(lmin_star, lmax_star)
        # Rotational broadening of the spectrum [km/s]
        if vsini_star > 0: # the wavelength axis needs to be evenly spaced before broadening
            star_spectrum = star_spectrum.evenly_spaced(lmin=lmin_star, lmax=lmax_star, renorm=False)
            star_spectrum = star_spectrum.broad(vsini_star)
        # Doppler shifting the spectrum [km/s]
        star_spectrum = star_spectrum.doppler_shift(rv_star)
    
    # Checking star spectrum wavelength range
    if band_only is None:
        if (star_spectrum.wavelength[0] > lmin_instru) or (star_spectrum.wavelength[-1] < lmax_instru):
            raise ValueError(f"'star_spectrum' does not fully cover the {instru} range ({lmin_instru}–{lmax_instru} µm).")
    else:
        if (star_spectrum.wavelength[0] > lmin_band_only) or (star_spectrum.wavelength[-1] < lmax_band_only):
            raise ValueError(f"'star_spectrum' does not fully cover the {band0_star}-band0-star range ({lmin_band_only}–{lmax_band_only} µm).")
    if (star_spectrum.wavelength[0] > lmin_band0_star) or (star_spectrum.wavelength[-1] < lmax_band0_star):
        raise ValueError(f"'star_spectrum' does not fully cover the {band_only}-'band_only' range ({lmin_band0_star}–{lmax_band0_star} µm).")
    
    # ---- Load / prepare planet spectrum if not provided ----
    if planet_spectrum is None:
        if (model_planet is None) or (T_planet is None) or (lg_planet is None):
            raise KeyError("Please define model_planet, T_planet, and lg_planet to load the planet spectrum.")
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model_planet, instru=instru)
        # Crop around photometric & instrument ranges
        lmin_planet = 0.98*min(lmin_instru, lmin_band0_planet)
        lmax_planet = 1.02*max(lmax_instru, lmax_band0_planet)
        planet_spectrum.crop(lmin_planet, lmax_planet)
        # Rotational broadening of the spectrum [km/s]
        if vsini_planet > 0: # the wavelength axis needs to be evenly spaced before broadening
            planet_spectrum = planet_spectrum.evenly_spaced(lmin=lmin_planet, lmax=lmax_planet, renorm=False)
            planet_spectrum = planet_spectrum.broad(vsini_planet)
        # Doppler shifting the spectrum [km/s]
        planet_spectrum = planet_spectrum.doppler_shift(rv_planet)
    
    # Checking planet spectrum wavelength range
    if band_only is None:
        if (planet_spectrum.wavelength[0] > lmin_instru) or (planet_spectrum.wavelength[-1] < lmax_instru):
            raise ValueError(f"'planet_spectrum' does not fully cover the {instru} range ({lmin_instru}–{lmax_instru} µm).")
    else:
        if (planet_spectrum.wavelength[0] > lmin_band_only) or (planet_spectrum.wavelength[-1] < lmax_band_only):
            raise ValueError(f"'planet_spectrum' does not fully cover the {band0_planet}-band0-planet range ({lmin_band_only}–{lmax_band_only} µm).")
    if (planet_spectrum.wavelength[0] > lmin_band0_planet) or (planet_spectrum.wavelength[-1] < lmax_band0_planet):
        raise ValueError(f"'planet_spectrum' does not fully cover the {band_only}-'band_only' range ({lmin_band0_planet}–{lmax_band0_planet} µm).")
    
    # ---- Delegate to FastCurves_main ----
    name_bands, separation_bands, results_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands, trans_bands, sigma_budget_bands = FastCurves_process(calculation=calculation, instru=instru, exposure_time=exposure_time, mag_star=mag_star, band0_star=band0_star, band0_planet=band0_planet, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, speckles=speckles, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, channel=channel, planet_name=planet_name, separation_planet=separation_planet, mag_planet=mag_planet, show_plot=show_plot, verbose=verbose, post_processing=post_processing, sep_unit=sep_unit, background=background, Rc=Rc, filter_type=filter_type, input_DIT=input_DIT, band_only=band_only, return_FastYield=return_FastYield, return_quantity=return_quantity)

    if verbose:
        print()
        print_time(f"FastCurves {calculation} calculation took {round(time.time()-time1, 1)} s.")

    # ---- FASTYIELD branch: per-band values at the planet separation ----
    if return_FastYield:
        
        if calculation != "SNR":
            raise KeyError("For return_FastYield=True, set calculation='SNR'.")
        if separation_planet is None:
            raise KeyError("Please provide 'separation_planet' for the SNR calculation.")

        # Convert the requested separation from arcsec to mas if the instrument works in mas
        if sep_unit == "mas":
            separation_planet = separation_planet * 1e3 # [arcsec] => [mas]
        
        SNR_planet        = np.zeros((len(name_bands)))
        signal_planet     = np.zeros((len(name_bands)))
        sigma_fund_planet = np.zeros((len(name_bands)))
        sigma_syst_planet = np.zeros((len(name_bands)))
        
        # Retrieving the values at the planet separation
        for nb, band in enumerate(name_bands):
            idx_planet_sep        = np.nanargmin(np.abs(separation_bands[nb] - separation_planet))
            SNR_planet[nb]        = results_bands[nb][idx_planet_sep]
            signal_planet[nb]     = signal_bands[nb][idx_planet_sep]
            sigma_fund_planet[nb] = np.sqrt(sigma_fund_2_bands[nb][idx_planet_sep])
            sigma_syst_planet[nb] = np.sqrt(sigma_syst_2_bands[nb][idx_planet_sep])
        
        return name_bands, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, np.array(DIT_bands)
    
    # ---- Deep analysis branch ----
    if return_quantity:
        return name_bands, separation_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands, trans_bands, sigma_budget_bands

    # ---- Standard return ----
    return name_bands, separation_bands, results_bands


