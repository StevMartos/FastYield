from src.signal_noise_estimate import *
from src.utils import _load_corona_profile, _load_corr_factor, _load_bkg_flux
from src.signal_noise_estimate import _get_transmission
path_file = os.path.dirname(__file__)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves Function
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves_main(calculation, instru, exposure_time, mag_star, band0_star, band0_planet, planet_spectrum, star_spectrum, tellurics, apodizer, strehl, coronagraph, systematics, speckles, PCA, PCA_mask, N_PCA, channel, planet_name, separation_planet, mag_planet, show_plot, verbose, post_processing, sep_unit, background, Rc, filter_type, input_DIT, band_only, return_SNR_planet):
    """
    See the function "FastCurves" below.
    """

    # ---------------------------------------
    # Hard-coded toggles for diagnostics
    # ---------------------------------------
    cos_theta_p        = 1     # Intrisic mismatch 
    cos_theta_est      = None  # Estimated correlation in the data (for show_cos_theta_est=True)
    show_cos_theta_est = False # To see the impact of the noise on the estimated correlation in order to retrieve the true mismatch
    show_t_syst        = False # To see the systematics time domination
    show_contributions = True  # To see the noise contributions plots for contrast calculations
    show_syst_budget   = True  # To see the speckle noise budget for Differential Imaging techniques to be more advantageous than molecular mapping and systematics noise budget
    
    # ---------------------------------------
    # Config & constants
    # ---------------------------------------
    if instru == "MIRIMRS" and channel and (calculation in {"SNR", "corner plot"}):
        # Each SNR “per channel” splits time over its 3 sub-bands
        exposure_time = exposure_time / 3

    for_planet_name = f" for {planet_name}" if planet_name else ""
    
    config_data  = get_config_data(instru)
    instru_type  = config_data["type"]                 # Instrument's type (e.g. imager, IFU, IFU_fiber, etc.)
    NbBand       = len(config_data["gratings"])        # Number of bands
    size_core    = config_data["size_core"]            # Aperture size on which the signal is integrated [pixels]
    A_FWHM       = size_core**2                        # Box aperture 
    saturation_e = config_data["spec"]["saturation_e"] # Full well capacity of the detector [e-]
    min_DIT      = config_data["spec"]["minDIT"]       # Minimal integration time [in mn]
    max_DIT      = config_data["spec"]["maxDIT"]       # Maximal integration time [mn]
    RON          = config_data["spec"]["RON"]          # Read out noise [e-/px/DIT]
    RON_lim      = config_data["spec"]["RON_lim"]      # Read out noise limit [e-/px/DIT]
    dark_current = config_data["spec"]["dark_current"] # Dark current [e-/px/s]
    IWA, OWA     = get_wa(config_data, sep_unit)       # Inner and Outer Working Angle in 'sep_unit' 
    if show_plot:
        cmap = plt.get_cmap("Spectral_r", NbBand + 1 if NbBand % 2 != 0 else NbBand)

    # ---------------------------------------
    # Accumulators
    # ---------------------------------------    
    contrast_bands        = [] # Final contrast as function of separation for exposure_time
    SNR_bands             = [] # Final S/N as function of separation for exposure_time
    name_bands            = [] # Band labels
    separation_bands      = [] # [arcsec] or [mas]
    signal_bands          = [] # [e-/FWHM/DIT]
    DIT_bands             = [] # [mn/DIT]
    sigma_fund_2_bands    = [] # [e-/FWHM/DIT]
    sigma_halo_2_bands    = [] # [e-/FWHM/DIT]
    sigma_det_2_bands     = [] # [e-/FWHM/DIT]
    sigma_bkg_2_bands     = [] # [e-/FWHM/DIT]
    sigma_speck_2_bands   = [] # [e-/FWHM/DIT]
    sigma_syst_2_bands    = [] # [e-/FWHM/DIT]
    planet_flux_bands     = [] # [e-/DIT]
    star_flux_bands       = [] # [e-/DIT]
    wave_bands            = [] # [µm]
    uncertainties_bands   = [] # [K], [dex(cm/s2)], [km/s] and [km/s]
    iwa_bands             = [] # [arcsec] or [mas]
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
    R_planet     = planet_spectrum.R     # Planet's model resolution
    R_star       = star_spectrum.R       # Star's model resolution
    R            = min(max(R_planet, R_star), R0_max) # Spectra resolution
    
    # Incoming separation is in arcsec; if instrument expects mas, convert once.
    if sep_unit == "mas" and separation_planet is not None:
        separation_planet *= 1e3
    
    # ---------------------------------------
    # Verbose header
    # ---------------------------------------
    if verbose:
        print("\n"+"\033[1m"+f"FastCurves {calculation} calculation{for_planet_name} with {exposure_time:.0f} mn exposure on {instru}:"+"\033[0m")
        print("\n"+"\033[4m"+f"Planetary spectrum ({model_planet}):"+"\033[0m")
        print(f"  R       = {round(R_planet, -3):.0f}")
        print(f"  Teff    = {T_planet:.0f} K")
        print(f"  log(g)  = {lg_planet:.1f} dex(cm/s2)")
        print(f"  rv      = {rv_planet:.1f} km/s")
        print(f"  Vsin(i) = {vsini_planet:.1f} km/s")
        if separation_planet is not None:
            print(f"  sep     = {separation_planet:.1f} {sep_unit}")
        if mag_planet is not None:
            print(f"  mag({band0_planet})  = {mag_planet:.1f}")
        print("\n"+"\033[4m"+f"Stellar spectrum ({model_star}):"+"\033[0m")
        print(f"  R       = {round(R_star, -3):.0f}")
        print(f"  Teff    = {T_star:.0f} K")
        print(f"  log(g)  = {lg_star:.1f} dex(cm/s2)")
        print(f"  rv      = {rv_star:.1f} km/s")
        print(f"  Vsin(i) = {vsini_star:.1f} km/s")
        print(f"  mag({band0_star})  = {mag_star:.1f}")
        if post_processing.lower() in {"molecular mapping", "mm"}:
            print(f'\nMolecular mapping considered as post-processing method with Rc = {Rc} and {filter_type} filtering')
        elif post_processing.lower() in {"differential imaging", "di"}:
            print('\nDifferential imaging considered as post-processing method')
        if systematics:
            if PCA:
                print(f'With systematics + PCA (with {N_PCA} components)')
            else:
                print('With systematics')
        else:
            print('Without systematics')
        if speckles:
            if PCA:
                print(f'With speckles + PCA (with {N_PCA} components)')
            else:
                print('With speckles')
        else:
            print('Without speckles')
        if strehl != "NO_JQ":
            print(f"With {strehl} strehl")
        if apodizer != "NO_SP":
            print(f"With {apodizer} apodizer")
        if coronagraph is not None:
            print(f"With {coronagraph} coronagraph")
        if tellurics:
            print("With tellurics absorption (ground-based observation)")
            
    # ---------------------------------------
    # Spectra on instrument band + magnitudes
    # ---------------------------------------
    star_spectrum_instru, star_spectrum_density = get_spectrum_instru(band0_star, R, config_data, mag_star, star_spectrum) # [ph/mn] and [J/s/m²/µm]
    if mag_planet is not None:
        planet_spectrum_instru, planet_spectrum_density = get_spectrum_instru(band0_planet, R, config_data, mag_planet, planet_spectrum) # [ph/mn] and [J/s/m²/µm]
        planet_to_star_ratio                            = np.nansum(planet_spectrum_instru.flux) / np.nansum(star_spectrum_instru.flux)
    else:
        if calculation in {"SNR", "corner plot"}:
            raise KeyError(f"Please input 'mag_planet' for the {calculation} calculation!")
        # The planet spectra are not adjusted to the correct magnitude
        planet_spectrum_instru, planet_spectrum_density = get_spectrum_instru(band0_star, R, config_data, mag_star, planet_spectrum) # [ph/mn] and [J/s/m²/µm]
        planet_to_star_ratio                            = None
        # For plotting only: normalize planetary density to star density mean level
        planet_spectrum_density.flux *= np.nanmean(star_spectrum_density.flux) / np.nanmean(planet_spectrum_density.flux)
    wave_instru = planet_spectrum_instru.wavelength

    # Computing the planet-to-star ratio (in total received ph/mn) on the instrumental bandwidth to renormalize the signal with (by doing so, it will give a contrast in photons and not in energy on this bandwidth, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
    if calculation == "contrast":
        star_to_planet_ratio = np.nansum(star_spectrum_instru.flux) / np.nansum(planet_spectrum_instru.flux)
        
    # Instrumental magnitudes (computed from densities)
    vega_spectrum   = load_vega_spectrum() # [J/s/m²/µm]
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_instru, renorm=False)
    mask_instru     = (wave_instru >= config_data["lambda_range"]["lambda_min"]) & (wave_instru <= config_data["lambda_range"]["lambda_max"]) # wave_instru is slightly broader than the instrumental range
    mag_star_instru = get_mag(flux_obs=star_spectrum_density.flux[mask_instru], flux_ref=vega_spectrum.flux[mask_instru])
    if mag_planet is not None:
        mag_planet_instru = get_mag(flux_obs=planet_spectrum_density.flux[mask_instru], flux_ref=vega_spectrum.flux[mask_instru])
    else:
        mag_planet_instru = None
    
    # ---------------------------------------
    # Optional overview plot on the instrument band [J/s/m²/µm]
    # ---------------------------------------
    if show_plot and band_only is None:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.yscale("log")
        plt.xlim(wave_instru[0], wave_instru[-1])  
        plt.ylim(max(np.nanmin(planet_spectrum_density.flux) / 10, np.nanmin(star_spectrum_density.flux) * 1e-12), max(np.nanmax(planet_spectrum_density.flux), np.nanmax(star_spectrum_density.flux))*10)
        plt.xlabel("Wavelength [µm]", fontsize=14)
        plt.ylabel(r"Flux [J/s/$m^2$/µm]", fontsize=14)
        plt.title(f"Star and planet spectra on the instrumental bandwidth (R = {round(round(R, -3))})\nwith $rv_*$ = {round(rv_star, 1)} km/s and $rv_p$ = {round(rv_planet, 1)} km/s", fontsize=16)
        plt.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        plt.plot(wave_instru, planet_spectrum_density.flux, color='seagreen', linestyle='-', linewidth=2, alpha=0.7, label=f'Planet, {model_planet} with $T$={int(round(T_planet))}K\nmag(instru)={round(mag_planet_instru, 1) if mag_planet_instru is not None else "Unknown"}')        
        plt.plot(wave_instru, star_spectrum_density.flux, color='crimson', linestyle='-', linewidth=2, alpha=0.7, label=f'Star, {model_star} with $T$={int(round(T_star))}K\nmag(instru)={round(mag_star_instru, 1)}')        
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        plt.gca().yaxis.set_ticks_position('both')
        plt.minorticks_on()
        plt.tight_layout()        
        plt.show()
    
    if verbose:
        print("\n"+"\033[4m" + "ON THE INSTRUMENTAL BANDWIDTH"+":"+"\033[0m")
        if mag_planet is not None:
            print(f" Magnitudes: mag_star = {round(mag_star_instru, 2)} and mag_planet = {round(mag_planet_instru, 2)}" + " => Contrast(photons) = {0:.2e}".format(planet_to_star_ratio))
        else:
            print(f" Magnitudes: mag_star = {round(mag_star_instru, 2)} and mag_planet = 'Unknown'")

    # ---------------------------------------
    # Optional overview plot per band [e-/mn]: plotting the planet spectrum (if SNR calculation) or the star spectrum (if contrast calculation) on each band (in e-/mn)
    # ---------------------------------------
    if show_plot:
        fig_band_flux = plt.figure(figsize=(10, 6), dpi=300)
        ax_band_flux  = fig_band_flux.gca()
        ax_band_flux.set_yscale("log")
        ax_band_flux.set_xlim(wave_instru[0], wave_instru[-1])        
        ax_band_flux.set_xlabel("Wavelength [µm]", fontsize=14)
        ax_band_flux.set_ylabel("Flux [e-/mn]", fontsize=14)
        ax_band_flux.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        ax_band_flux.yaxis.set_ticks_position('both')
        ax_band_flux.minorticks_on()
        ymin = 1e9
        ymax = 1
        if calculation == "SNR" or calculation == "corner plot":
            ax_band_flux.set_title(f"Planet flux ({model_planet}) through {instru} bands \n with $T_p$={int(round(T_planet))}K, $lg_p$={round(lg_planet, 1)} and $mag_p$({band0_planet})={round(mag_planet, 2)}", fontsize=16)
        elif calculation == "contrast":
            ax_band_flux.set_title(f"Star flux ({model_star}) through {instru} bands with $mag_*$({band0_star})={round(mag_star, 2)}", fontsize=16)
    
    # ==============================================================================================================
    # Loop over spectral bands
    # ==============================================================================================================
    for nb, band in enumerate(config_data["gratings"]):
        if (band_only is not None and band != band_only) or (instru=="HARMONI" and strehl=="MED" and band not in ["H", "K"]):
            continue # If you want to calculate for band_only only
        
        iwa = IWA
        
        # Adding the band's name to the list
        name_bands.append(band)
          
        # Degradation at instrumental resolution and restriction of the wavelength range in the considered band        
        star_spectrum_band   = get_spectrum_band(config_data, band, star_spectrum_instru)
        planet_spectrum_band = get_spectrum_band(config_data, band, planet_spectrum_instru)
        wave_band            = planet_spectrum_band.wavelength
        R_band               = config_data['gratings'][band].R
                
        # Band's magnitudes (computed from densities)
        mask_band     = (wave_instru >= config_data['gratings'][band].lmin) & (wave_instru <= config_data['gratings'][band].lmax)
        mag_star_band = get_mag(flux_obs=star_spectrum_density.flux[mask_band], flux_ref=vega_spectrum.flux[mask_band])
        if mag_planet is not None:
            mag_planet_band = get_mag(flux_obs=planet_spectrum_density.flux[mask_band], flux_ref=vega_spectrum.flux[mask_band])
        else:
            mag_planet_band = None

        # System transmission        
        trans = _get_transmission(instru, band, tellurics, apodizer, strehl, coronagraph)
        
        # PSF profiles, fraction_core (fraction of photons in the PSF core/FWHM), separations (instrument unit), pxscale
        PSF_profile, fraction_core, separation, pxscale, iwa_FPM = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, instru=instru, separation_planet=separation_planet, return_SNR_planet=return_SNR_planet)
        
        # Adding the band's separation axis to the list
        separation_bands.append(separation)
        
        # Index of the separation of the planet
        if separation_planet is not None:
            idx_planet = np.where(separation==separation_planet)[0][0]
                        
        # Coronagraphic radial transmission & core fraction vs separation        
        if coronagraph is not None:
            raw_sep, raw_fraction_core, raw_radial_transmission = _load_corona_profile(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)
            fraction_core_interp        = interp1d(raw_sep, raw_fraction_core,        bounds_error=False, fill_value="extrapolate")
            radial_transmission_interp = interp1d(raw_sep, raw_radial_transmission, bounds_error=False, fill_value="extrapolate")
            fraction_core        = fraction_core_interp(separation)        # Fraction of flux of a PSF inside the FWHM as function of the separation
            radial_transmission  = radial_transmission_interp(separation) # Transmission of a PSF as function of the separation
            star_transmission    = radial_transmission_interp(0)          # Total stellar flux transmitted by the coronagraph (+Lyot stop) when the star is perfectly aligned with it (i.e. at 0 separation)
            fraction_core[separation > raw_sep[-1]]       = raw_fraction_core[-1]        # Flat extrapolation
            radial_transmission[separation > raw_sep[-1]] = raw_radial_transmission[-1] # Flat extrapolation 
            PSF_profile *= star_transmission
                
        # Fiber-fed IFUs: mean injection correction (e.g. ANDES): the fact that the position of the planet is unknown
        if instru_type == "IFU_fiber":
            try:
                fraction_core *= config_data["injection"][band]
            except:
                pass
        
        # Corrective factor for fundamental noises (per separation): due to potential dithering (impacting the noise statistics, i.e. covariance, etc.)       
        R_corr = np.zeros_like(separation) + 1.
        if instru in {"MIRIMRS", "NIRSpec"}: # Dithering for MIRIMRS and NIRSpec
            sep, r_corr = _load_corr_factor(instru, band)
            valid       = np.isfinite(r_corr)
            R_corr      = interp1d(sep[valid], r_corr[valid], bounds_error=False, fill_value="extrapolate")(separation)
            R_corr[separation > sep[-1]] = r_corr[-1] # flat extrapolation
        else:
            try:
                R_corr *= config_data["R_corr"]
            except:
                pass
                
        # =======================
        # Non-imager path
        # =======================
        if instru_type != "imager":
            
            # --------------------------
            # Molecular Mapping (MM) path
            # --------------------------
            if post_processing.lower() in {"molecular mapping", "mm"}:
                
                # Power fraction of the white noise filtered by the high-pass filtering (<1)
                fn_HF = get_fraction_noise_filtered(N=len(wave_band), R=R_band, Rc=Rc, filter_type=filter_type)[0]
                
                # Systematics noise profile and modulations
                if systematics:
                    if instru in {"MIRIMRS", "NIRSpec"}:
                        sigma_syst_prime_2_per_tot, sep, m_HF, Ms, Mp, M_pca, wave, pca, PCA_verbose = get_systematic_profile(config_data=config_data, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, R=R_band, star_spectrum_instru=star_spectrum_instru, planet_spectrum_instru=planet_spectrum_instru, planet_spectrum=planet_spectrum, wave_band=wave_band, size_core=size_core, filter_type=filter_type, show_cos_theta_est=show_cos_theta_est, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, mag_planet=mag_planet, band0_planet=band0_planet, separation_planet=separation_planet, mag_star=mag_star, target_name=planet_name, exposure_time=exposure_time)
                        sigma_syst_prime = interp1d(sep, np.sqrt(sigma_syst_prime_2_per_tot), bounds_error=False, fill_value="extrapolate")(separation)
                        if separation[-1] > sep[-1]: # Systematic profile extrapolation
                            mask_outside                   = (separation >= sep[-1])
                            extension                      = PSF_profile[mask_outside].copy()               # Same extrapolation profile as the PSF profile one (propto stellar flux)
                            extension                     *= np.sqrt(sigma_syst_prime_2_per_tot[-1]) / extension[0] # Forcing continuity
                            sigma_syst_prime[mask_outside] = extension
                        sigma_syst_prime_2_per_tot = sigma_syst_prime**2 # Systematic noise profile projected in the CCF in [e-/FWHM/total stellar flux]
                        planet_spectrum_band.crop(wave[0], wave[-1])
                        star_spectrum_band.crop(wave[0], wave[-1])
                        mask_M    = (wave_band >= wave[0]) & (wave_band <= wave[-1]) # Effective wavelength axis (from data)
                        trans     = trans[mask_M]
                        wave_band = wave_band[mask_M]
                        Ms        = Ms[mask_M]
                        Mp        = Mp[mask_M]
                    else:
                        raise KeyError("Undefined !")   
            
                # Build template = trans * planet_HF (normalized)
                template, _ = filtered_flux(planet_spectrum_band.flux, R_band, Rc, filter_type) # [Sp]_HF
                template    = trans * template                                                  # gamma * [Sp]_HF
                template    = template / np.sqrt(np.nansum(template**2))                        # Normalizing the template
                
                # Systematic modulations of the spectra are taken into account (mostly insignificant effect)
                if systematics:
                    star_spectrum_band.flux   = Ms * star_spectrum_band.flux
                    planet_spectrum_band.flux = Mp * planet_spectrum_band.flux
                
                # Useful signal (α) and self-subtraction (β) in [e-/mn] (with systematic modulations, if any)
                alpha = get_alpha(planet_spectrum_band, template, Rc, R_band, trans, filter_type)
                beta  = get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R_band, trans, filter_type)
                
                # For prints
                if verbose or show_syst_budget:
                    template_full = trans * planet_spectrum_band.flux                    # gamma * Sp
                    template_full = template_full / np.sqrt(np.nansum(template_full**2)) # Normalizing the template
                    delta = get_delta(planet_spectrum_band, template_full, trans)

            # --------------------------
            # Differential Imaging (DI) path
            # --------------------------
            elif post_processing.lower() in {"differential imaging", "di"}:

                # Speckles noise profile
                if speckles:
                    raise KeyError("Undefined !")  
                
                # Systematics noise profile and modulations
                if systematics:
                    raise KeyError("Undefined !")
            
                # Build template = trans * planet (normalized)
                template = trans * planet_spectrum_band.flux          # gamma * Sp
                template = template / np.sqrt(np.nansum(template**2)) # Normalizing the template
                
                # Systematics modulations of the spectra are taken into account (mostly insignificant effect)
                if systematics:
                    star_spectrum_band.flux   = Ms * star_spectrum_band.flux
                    planet_spectrum_band.flux = Mp * planet_spectrum_band.flux
                
                # Useful signal (δ) in [e-/mn] (with systematics modulations, if any)
                delta = get_delta(planet_spectrum_band, template, trans)
        
        # --------------------------
        # DIT in [mn] and effective read-out noise in [e-/px/DIT] 
        # --------------------------
        N_DIT, DIT, DIT_saturation, RON_eff, iwa_FPM = get_DIT_RON(instru=instru, instru_type=instru_type, apodizer=apodizer, PSF_profile=PSF_profile, separation=separation, star_spectrum_band=star_spectrum_band, exposure_time=exposure_time, min_DIT=min_DIT, max_DIT=max_DIT, trans=trans, RON=RON, RON_lim=RON_lim, saturation_e=saturation_e, input_DIT=input_DIT, iwa_FPM=iwa_FPM)
        
        # Adding the band's DIT value to the list
        DIT_bands.append(DIT)
        
        # --------------------------
        # Spectra through system in [e-/DIT]
        # --------------------------
        star_spectrum_band.flux   *= trans*DIT
        planet_spectrum_band.flux *= trans*DIT
                
        # ---------------------------------------
        # Optional overview plot per band [e-/mn]: plotting the planet (if SNR calculation) or the star (if contrast calculation) on each band (in e-/mn)
        # ---------------------------------------
        if show_plot:
            if calculation == "SNR" or calculation == "corner plot":
                flux_band = planet_spectrum_band.flux/DIT # [e-/mn]
            elif calculation == "contrast":
                if coronagraph is not None:
                    flux_band = star_spectrum_band.flux*star_transmission/DIT # [e-/mn] through coronagraph
                else:
                    flux_band = star_spectrum_band.flux/DIT # [e-/mn]
            if instru_type == "imager":
                label_band = band.replace('_', ' ') + f" ({round(np.nansum(flux_band))} e-/mn)"
            else:
                label_band = band.replace('_', ' ') + f" (R={int(round(R_band))})"
            ax_band_flux.plot(wave_band, flux_band, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=label_band)
            ax_band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
            ymin = min(ymin, np.nanmin(flux_band)/2)
            ymax = max(ymax, np.nanmax(flux_band)*2)
            ax_band_flux.set_ylim(ymin=max(ymin, 1e-2), ymax=ymax)
        
        # =======================
        # Signal estimations in [e-/FWHM/DIT]
        # =======================
        
        if instru_type == "imager":
            signal = np.zeros_like(separation) + np.nansum(planet_spectrum_band.flux) # Integrated flux  [e-/FWHM/DIT]
        
        else:
            if post_processing.lower() in {"molecular mapping", "mm"}:
                signal = np.zeros_like(separation) + (alpha * cos_theta_p - beta) * DIT # Useful signal in the CCF [e-/FWHM/DIT]
            elif post_processing.lower() in {"differential imaging", "di"}:
                signal = np.zeros_like(separation) + delta  * cos_theta_p         * DIT # Useful signal in the CCF [e-/FWHM/DIT]
            
        # Focal Plane Mask: flux attenuated by a factor of 1e-4 for HARMONI (FPM)
        if instru=="HARMONI" and iwa_FPM is not None:
            PSF_profile[separation < iwa_FPM] *= 1e-4
            signal[separation < iwa_FPM]      *= 1e-4
            iwa                                = iwa_FPM
        
        # Signal inside FWHM aperture in [e-/FWHM/DIT]
        signal *= fraction_core

        # Non-spatial-homegenous tranmission of the coronagraph
        if coronagraph is not None:
            signal *= radial_transmission
        
        # Renormalizing the signal with the planet-to-star ratio (in total received photons) on the instrumental bandwidth (by doing so, it will give a contrast in photons and not in energy on this bandwidth, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
        if calculation == "contrast":
            signal *= star_to_planet_ratio
            
        # Signal loss ratio due to the PCA (if required)
        if systematics:
            signal *= M_pca
        
        # =======================
        # Noise estimations
        # =======================
        
        # --------------------------
        # Detector noises, DC and RON in [e-/spaxel/DIT] for IFU and [e-/px/DIT] for imager
        # --------------------------
        sigma_dc_2  = dark_current * 60*DIT # Dark current photon noise [e-/px/DIT]
        sigma_ron_2 = RON_eff**2            # Effective read out noise  [e-/px/DIT]
        
        if instru_type == "IFU_fiber": # Detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
            NbPixel      = config_data['pixel_detector_projection'] # Number of detector px per spaxel
            sigma_dc_2  *= NbPixel # Adds quadratically [e-/spaxel/DIT]
            sigma_ron_2 *= NbPixel # Adds quadratically [e-/spaxel/DIT]
        
        # --------------------------
        # Stellar halo photon noise in [e-/spaxel/DIT] for IFU and [e-/px/DIT] for imager
        # --------------------------
        if instru_type == "imager": # Integrated flux
            sigma_halo_2 = PSF_profile * np.nansum(star_spectrum_band.flux) # Stellar photon noise per spectral channel in [e-/px/DIT] for each separation
        
        else: # Projection onto the CCF
            sigma_halo_prime_2 = PSF_profile * np.nansum(star_spectrum_band.flux * template**2) # Stellar photon noise projected in the CCF [e-/spaxel/DIT] for each separation
        
        # --------------------------
        # Background photon noise in [e-/spaxel/DIT] for IFU and [e-/px/DIT] for imager
        # --------------------------
        sigma_bkg_2       = 0. # [e-/px/DIT]
        sigma_bkg_prime_2 = 0. # [e-/spaxel/DIT]

        if background is not None:
            
            raw_wave, raw_bkg = _load_bkg_flux(instru, band, background)
            bkg_flux          = interp1d(raw_wave, raw_bkg, bounds_error=False, fill_value=np.nan)(wave_band) # [e-/px/s]
            tot_bkg_flux      = np.nansum(bkg_flux) # [e-/s]
            if tot_bkg_flux != 0:
                # We have to renormalize because we interpolated (flux conservation) in [e-/px/DIT]
                bkg_flux   *= np.nansum(raw_bkg[(raw_wave >= wave_band[0]) & (raw_wave <= wave_band[-1])]) / tot_bkg_flux * 60*DIT
                sigma_bkg_2 = bkg_flux # Background photon noise per spectral channel in [e-/px/DIT] for each separation
            
            if instru_type == "imager": # Integrated flux
                sigma_bkg_2 = np.nansum(sigma_bkg_2) # Background photon noise per spectral channel in [e-/px/DIT] for each separation
                if coronagraph is not None: # Non-homogenous sky transmission
                    sigma_bkg_2 *= radial_transmission
            
            else: # Projection onto the CCF
                sigma_bkg_prime_2 = np.nansum(sigma_bkg_2 * template**2) # Background photon noise projected in the CCF in [e-/spaxel/DIT]
                if coronagraph is not None: # Non-homogenous sky transmission
                    sigma_bkg_prime_2 *= radial_transmission # Background photon noise projected in the CCF in [e-/spaxel/DIT]
        
        # --------------------------
        # Fundamental noise in [e-/FWHM/DIT]
        # --------------------------
        
        if instru_type == "imager":
            sigma_fund_2 = R_corr * A_FWHM * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2) # Fundamental noise profile integrated over the FWHM in [e-/FWHM/DIT]
        
        else:
            if post_processing.lower() in {"molecular mapping", "mm"}:
                sigma_fund_prime_2 = fn_HF * R_corr * A_FWHM * (sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2) # Fundamental noise profile projected in the CCF and integrated over the FWHM in [e-/FWHM/DIT]
            
            elif post_processing.lower() in {"differential imaging", "di"}:
                sigma_fund_prime_2 =  1    * R_corr * A_FWHM * (sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2) # Fundamental noise profile projected in the CCF and integrated over the FWHM in [e-/FWHM/DIT]
        
        # --------------------------
        # Speckle noise in [e-/FWHM/DIT]
        # --------------------------
        sigma_speck_2       = np.zeros_like(separation)
        sigma_speck_prime_2 = np.zeros_like(separation)
        
        if speckles:
            
            if instru_type == "imager":
                raise KeyError("Undefined !")   
                
            elif post_processing.lower() in {"differential imaging", "di"}:
                raise KeyError("Undefined !")
        
        # --------------------------
        # Systematic noise in [e-/FWHM/DIT]
        # --------------------------
        sigma_syst_2       = np.zeros_like(separation)
        sigma_syst_prime_2 = np.zeros_like(separation)
        
        if systematics:
            
            if instru_type == "imager":
                raise KeyError("Undefined !")   
                
            else:
                if post_processing.lower() in {"molecular mapping", "mm"}:
                    sigma_syst_prime_2 = sigma_syst_prime_2_per_tot * np.nansum(star_spectrum_band.flux)**2 # Systematic noise profile projected in the CCF in [e-/spaxel/DIT]
                
                elif post_processing.lower() in {"differential imaging", "di"}:
                    raise KeyError("Undefined !")
        
        # =======================
        # Verbose band-level summary
        # =======================
        if verbose:
            if instru_type == "imager":
                print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm):"+"\033[0m")
            else:
                print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm with R={R_band:.0f}):"+"\033[0m")
            
            if DIT_saturation < min_DIT:
                print(f" Saturation would occur even at the shortest DIT; using min_DIT ({min_DIT*60:.2f} s).")
            print(f" DIT = {DIT*60:.1f} s / Saturating DIT = {DIT_saturation:.2f} mn / RON_eff = {RON_eff:.3f} e-/DIT")
            
            print(f" Mean total system transmission = {100*np.nanmean(trans):.1f} %")
            
            if "IFU" in instru_type:
                print(f" Number of spectral channels: {len(wave_band)}")
            
            if mag_planet is not None:
                print(f" Magnitudes: mag_star = {mag_star_band:.2f} and mag_planet = {mag_planet_band:.2f}" + " => Contrast(photons) = {0:.2e}".format(np.nansum(planet_spectrum_band.flux*trans)/np.nansum(star_spectrum_band.flux*trans)))
            else:
                print(f" Magnitudes: mag_star = {mag_star_band:.2f} and mag_planet = 'Unknown'")
            
            if separation_planet is not None and coronagraph is not None:
                print(f" Fraction of flux in the core (FWHM) of the PSF (at {separation_planet:.1f} {sep_unit}): f = {100*fraction_core[idx_planet]:.1f} %")
            elif coronagraph is None:
                print(f" Fraction of flux in the core (FWHM) of the PSF: f = {100*fraction_core:.1f} %")
            
            if calculation == "SNR" or calculation == "corner plot":
                if separation_planet is not None:
                    print(f" Useful {post_processing} signal from the planet (at {separation_planet:.1f} {sep_unit}): {signal[idx_planet]/DIT:.1f} e-/FWHM/mn")
                else:
                    print(f" Useful {post_processing} signal from the planet: {signal/DIT:.1f} e-/FWHM/mn")
            if post_processing.lower() in {"molecular mapping", "mm"}:
                print(f" Signal loss due to high-pass filtering: (α - β)/δ = {100*(alpha-beta)/delta:.1f} %")
                print(f" Signal loss due to self-subtraction:    β/α       = {100*beta/alpha:.1f} %")
            
            if PCA and systematics:
                if PCA_verbose is not None:
                    print(PCA_verbose)
                print(f" Signal loss due to PCA = {100*(1-M_pca):.1f} %")
                
            if iwa_FPM is not None:
                print(f" Using a FPM under {iwa_FPM:.1f} {sep_unit} to avoid saturation")
        
        # =======================
        # Saving quantities
        # =======================
        signal_bands.append(signal)                         # [e-/FWHM/DIT]
        planet_flux_bands.append(planet_spectrum_band.flux) # [e-/DIT]
        star_flux_bands.append(star_spectrum_band.flux)     # [e-/DIT]
        wave_bands.append(wave_band)                        # [µm]
        iwa_bands.append(iwa)
        
        # Saving noises in [e-/FWHM/DIT]
        if instru_type == "imager":
            sigma_fund_2_bands.append(sigma_fund_2)
            sigma_halo_2_bands.append(R_corr * A_FWHM * (sigma_halo_2))
            sigma_det_2_bands.append(R_corr  * A_FWHM * (sigma_ron_2 + sigma_dc_2))
            sigma_bkg_2_bands.append(R_corr  * A_FWHM * (sigma_bkg_2))
            sigma_syst_2_bands.append(sigma_syst_2)
        else:
            if post_processing.lower() in {"molecular mapping", "mm"}:
                sigma_fund_2_bands.append(sigma_fund_prime_2)
                sigma_halo_2_bands.append(fn_HF * R_corr * A_FWHM * (sigma_halo_prime_2))
                sigma_det_2_bands.append(fn_HF  * R_corr * A_FWHM * (sigma_ron_2 + sigma_dc_2))
                sigma_bkg_2_bands.append(fn_HF  * R_corr * A_FWHM * (sigma_bkg_prime_2))
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
            
            # --------------------------
            # Noise contributions plots (in 5σ contrast)
            # --------------------------
            if show_plot and show_contributions:
                mask_iwa     = separation >= iwa
                fig_contrast = plt.figure(figsize=(10, 6), dpi=300)        
                ax_contrast  = fig_contrast.gca()
                ax_contrast.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
                ax_contrast.minorticks_on()
                ax_contrast.tick_params(axis='both', labelsize=12)        
                ax_contrast.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax_contrast.set_yscale('log')
                ax_contrast.set_xlim(0, separation[-1])
                ax_contrast.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                if instru_type == "imager":
                    ax_contrast.set_title(f"{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name} \n with "r"$t_{exp}$="f"{round(exposure_time)}mn and "r"$mag_*$"f"({band0_star})={round(mag_star, 2)}", fontsize=16)        
                    ax_contrast.set_ylabel(r'Contrast 5$\sigma$ / $F_{p}$', fontsize=14)
                    ax_contrast.plot(separation[mask_iwa], contrast[mask_iwa], 'k-', label=r"$\sigma_{tot}$")
                    ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(R_corr * A_FWHM * sigma_halo_2) / (np.sqrt(N_DIT) * signal))[mask_iwa], c="crimson",   ls="--", label=r"$\sigma_{halo}$")
                    ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(R_corr * A_FWHM * sigma_ron_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa], c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                    ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(R_corr * A_FWHM * sigma_dc_2)   / (np.sqrt(N_DIT) * signal))[mask_iwa], c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                    ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(R_corr * A_FWHM * sigma_bkg_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa], c="royalblue", ls="--", label=r"$\sigma_{bkg}$")                    
                    if systematics:
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(N_DIT * sigma_syst_2)/((np.sqrt(N_DIT) * signal)))[mask_iwa], c="cyan", ls="--", label=r"$\sigma_{syst}$")
                    if speckles:
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(N_DIT * sigma_speck_2)/((np.sqrt(N_DIT) * signal)))[mask_iwa], c="gray", ls="--", label=r"$\sigma_{speck}$")
                    
                else:
                    if post_processing.lower() in {"molecular mapping", "mm"}:
                        ax_contrast.set_title(f"{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name}\n with "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                        ax_contrast.set_ylabel(r"Contrast 5$\sigma_{CCF}$ / $\alpha_0$", fontsize=14)
                        ax_contrast.plot(separation[mask_iwa], contrast[mask_iwa], 'k-', label=r"$\sigma_{CCF}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(fn_HF * R_corr * A_FWHM * sigma_halo_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa], c="crimson",   ls="--", label=r"$\sigma'_{halo}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(fn_HF * R_corr * A_FWHM * sigma_ron_2)        / (np.sqrt(N_DIT) * signal))[mask_iwa], c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(fn_HF * R_corr * A_FWHM * sigma_dc_2)         / (np.sqrt(N_DIT) * signal))[mask_iwa], c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(fn_HF * R_corr * A_FWHM * sigma_bkg_prime_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa], c="royalblue", ls="--", label=r"$\sigma'_{bkg}$")
                        if systematics:
                            ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(N_DIT * sigma_syst_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa], c="cyan", ls="--", label=r"$\sigma'_{syst}$")
                    elif post_processing.lower() in {"differential imaging", "di"}:
                        ax_contrast.set_title(f"{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name}\n with "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K", fontsize=16)        
                        ax_contrast.set_ylabel(r"Contrast 5$\sigma_{CCF}$ / $\delta_0$", fontsize=14)
                        ax_contrast.plot(separation[mask_iwa], contrast[mask_iwa], 'k-', label=r"$\sigma_{CCF}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_halo_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa], c="crimson",   ls="--", label=r"$\sigma'_{halo}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_ron_2)        / (np.sqrt(N_DIT) * signal))[mask_iwa], c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_dc_2)         / (np.sqrt(N_DIT) * signal))[mask_iwa], c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                        ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(1 * R_corr * A_FWHM * sigma_bkg_prime_2)  / (np.sqrt(N_DIT) * signal))[mask_iwa], c="royalblue", ls="--", label=r"$\sigma'_{bkg}$")
                        if systematics:
                            ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(N_DIT * sigma_syst_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa], c="cyan", ls="--", label=r"$\sigma'_{syst}$")
                        if speckles:
                            ax_contrast.plot(separation[mask_iwa], (5*np.sqrt(N_DIT * sigma_speck_prime_2) / (np.sqrt(N_DIT) * signal))[mask_iwa], c="cyan", ls="--", label=r"$\sigma'_{speck}$")
                if separation_planet is not None:
                    if separation_planet > 2 * OWA:
                        ax_contrast.set_xscale('log')
                        ax_contrast.set_xlim(iwa, separation[-1])
                    if mag_planet is None:
                        ax_contrast.axvline(separation_planet, color="black", linestyle="--", label=f"{planet_name}" if planet_name is not None else "planet")
                        leg_loc = "upper right"
                    else:
                        if planet_to_star_ratio > ax_contrast.get_ylim()[1] or (planet_to_star_ratio > ax_contrast.get_ylim()[0] and planet_to_star_ratio < ax_contrast.get_ylim()[1]):
                            y_text = planet_to_star_ratio/1.5
                            if separation_planet > (iwa+OWA)/2:
                                x_text    = separation_planet - 0.1 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "center"
                            else:
                                x_text    = separation_planet + 0.025 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "right"
                        else :
                            y_text = planet_to_star_ratio*1.5
                            if separation_planet > (iwa+OWA)/2:
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
                ax_contrast_mag.set_ylim(-2.5 * np.log10(ymin_c_band), -2.5 * np.log10(ymax_c_band))        
                fig_contrast.tight_layout()
                fig_contrast.show()
        
        # =======================
        # S/N computation
        # =======================
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
                    print(f" S/N at {separation_planet:.1f} {sep_unit} = {SNR[idx_planet]:.1f}")
                if  SNR[idx_planet] > SNR_max_planet:
                    SNR_max_planet = SNR[idx_planet]
                    band_SNR_max   = band
            
            # Adding the SNR curve of the band to the list
            SNR_bands.append(SNR)
        
            # --------------------------
            # Corner plot
            # --------------------------
            if calculation == "corner plot":
                Mp_Sp              = fraction_core * N_DIT * planet_spectrum_band.flux          # Planet flux (with modulations, if any) in [e-/FWHM]
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp / trans, R_band, Rc, filter_type)     # Filtered planet flux in [ph/FWHM]
                star_flux          = PSF_profile[idx_planet] * N_DIT * star_spectrum_band.flux  # Stellar flux in [e-/px] at separation of the planet
                star_HF, star_LF   = filtered_flux(star_flux / trans, R_band, Rc, filter_type) # Filtered star flux
                d_planet           = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF           # Flux in [e-/FWHM] at the planet's location: see Eq.(18) of Martos et al. 2025
                # Total noise in [e-/FWHM/spectral channel] at the planet's location
                sigma_halo_2       = PSF_profile[idx_planet] * star_spectrum_band.flux
                sigma_l            = np.sqrt( fn_HF * R_corr[idx_planet] * A_FWHM * N_DIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2)  )
                SNR_CCF            = SNR[idx_planet]
                
                N             = 20
                T_arr         = np.linspace(T_planet-200, T_planet+200, N)
                lg_arr        = np.linspace(lg_planet-0.5, lg_planet+0.5, N)
                vsini_arr     = np.linspace(max(vsini_planet-5, 0), vsini_planet+5, N)
                vsini_arr     = np.array([vsini_planet]) # R too low for retrieving this parameter
                rv_arr        = np.linspace(rv_planet-10, rv_planet+10, N)
                uncertainties = parameters_estimation(instru=instru, band=band_only, target_name=planet_name, d_planet=d_planet, star_flux=star_flux, wave=wave_band, trans=trans, model=model_planet, R=R_band, Rc=Rc, filter_type=filter_type, logL=True, method_logL="classic", sigma_l=sigma_l, weight=None, pca=pca, stellar_component=True, degrade_resolution=True, SNR_estimate=False, T_arr=T_arr, lg_arr=lg_arr, vsini_arr=vsini_arr, rv_arr=rv_arr, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, SNR_CCF=SNR_CCF, d_planet_sim=False, template=planet_spectrum_instru, renorm_d_planet_sim=False, fastcurves=True, star_HF=star_HF, star_LF=star_LF, wave_interp=wave_instru, epsilon=0.8, fastbroad=True, force_new_est=True, save=False, exposure_time=exposure_time, show=True, verbose=True)
                
                uncertainties_bands.append(uncertainties)
            
            # ---------------------------------------------
            # Diagnostic: impact of noise + systematics on correlation estimation
            # (incl. auto-subtraction). Optionally uses m_HF from the systematics model.
            # ---------------------------------------------
            if show_plot and show_cos_theta_est and post_processing == "molecular mapping":
            
                # High-frequency modulation matrix M_HF(sep, lambda)
                # Defaults to zeros unless 'systematics' provided m_HF at sampled separations.
                M_HF = np.zeros((len(separation), len(wave_band)))
                if systematics:
                    # 'sep' & 'm_HF' come from get_systematic_profile; mask_M restricts to valid wavelengths
                    for i, s in enumerate(separation):
                        idx_sep = np.abs(s - sep).argmin()
                        M_HF[i] = m_HF[idx_sep][mask_M]
            
                # ---------- Build planet/star 1D spectra (per spectral channel) ----------
                # Units reminders:
                # - planet_spectrum_band.flux, star_spectrum_band.flux are in [e-/DIT/px] (after throughput)
                # - Multiplying by N_DIT gives [e-/px] (per observation)
                # - Multiplying planet by fraction_core -> [e-/FWHM]
            
                # Planet (e-/FWHM), then split into HF/LF on *pre-throughput* and re-apply 'trans'
                Mp_Sp              = fraction_core * N_DIT * planet_spectrum_band.flux  # [e-/FWHM]
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp / trans, R_band, Rc, filter_type)
                
                # Star, HF/LF of star without PSF factor (spatial term added per separation below)
                star_flux        = N_DIT * star_spectrum_band.flux                         # [e-]
                star_HF, star_LF = filtered_flux(star_flux / trans, R_band, Rc, filter_type)
            
                # Effective signal and self-subtraction
                alpha         = np.sqrt( np.nansum( (trans*Mp_Sp_HF)**2 ) )            # [e-/FWHM]
                beta          = np.nansum( trans*star_HF*Mp_Sp_LF/star_LF * template ) # [e-/FWHM]
                cos_theta_lim = np.nansum( trans*Mp_Sp_HF * template ) / alpha
            
                # ---------- Monte Carlo over noise for cos(theta_est) ----------
                cos_theta_est = np.zeros_like(separation, dtype=float)
                norm_d        = np.zeros_like(separation, dtype=float)
            
                # Number of noise realizations; vectorized per separation
                N   = 1_000
                rng = np.random.default_rng() # set a seed if you need reproducibility
            
                for i, s in enumerate(separation):
                    # Stellar flux *inside* the FWHM box at separation i (sum over pixels ~ A_FWHM)
                    # PSF_profile[i] is a per-pixel scaling; A_FWHM maps px -> FWHM box.
                    star_flux_FWHM = A_FWHM * PSF_profile[i] * star_flux # [e-/FWHM]
                    sigma_halo_2   = PSF_profile[i] * star_spectrum_band.flux
                    
                    # Per-channel noise (photometric variance form) at separation i (scalar per channel)
                    sigma_l = np.sqrt( fn_HF * R_corr[i] * A_FWHM * N_DIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2 ) )  # [e-/FWHM/channel]
            
                    # Deterministic spectrum at planet location (Eq. 18 in Martos+2025)
                    d_planet = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF + M_HF[i]*star_flux_FWHM  # [e-/FWHM/channel]
                    
                    # Vectorized N noise realizations
                    noise   = rng.normal(loc=0.0, scale=sigma_l, size=(N, len(wave_band))) # [N, channels]
                    d_n_all = d_planet[None, :] + noise                                    # [N, channels]
            
                    # Norms and cos(theta_est)
                    norm_d_n = np.linalg.norm(d_n_all, axis=1)
                    cos_n    = (d_n_all @ template) / norm_d_n
            
                    # Averages over N draws
                    norm_d[i]        = np.nanmean(norm_d_n)
                    cos_theta_est[i] = np.nanmean(cos_n)
            
                    # Optional info at the *planet* separation
                    if (separation_planet is not None) and (i == idx_planet):
                        snr_per_ch = np.nanmean(Mp_Sp / sigma_l)
                        print(f" S/N per spectral channel = {snr_per_ch:.1f}")
            
                # Pure fundamental-noise correlation limit
                cos_theta_n = alpha / norm_d
            
                # ---------- Plot ----------
                fig_cos_theta_est = plt.figure(dpi=300, figsize=(10, 6))
                ax_cos_theta_est  = fig_cos_theta_est.gca()
                ax_cos_theta_est.plot(separation, cos_theta_est, 'k')
                ax_cos_theta_est.set_ylabel(r"cos $\theta_{\rm est}$", fontsize=14)
                ax_cos_theta_est.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_cos_theta_est.set_xlim(separation[0], separation[-1])
                ax_cos_theta_est.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax_cos_theta_est.minorticks_on()
                ax_cos_theta_est.set_title(f"Effect of noise and stellar subtraction on the correlation\nbetween template and planetary spectrum for {instru} on {band}\n(assuming the template equals the observed planet spectrum)", fontsize=16)
                if (separation_planet is not None) and (separation_planet < np.nanmax(separation)):
                    print(f" beta/alpha = {beta/alpha:.3f} | cos_theta_n = {cos_theta_n[idx_planet]:.3f} | cos_theta_lim = {cos_theta_lim:.3f}")
                    if cos_theta_est is not None:
                        # Recover intrinsic mismatch estimate cos_theta_p from an observed cos_theta_est
                        cos_theta_p = (cos_theta_est / cos_theta_n[idx_planet] + beta/alpha) / cos_theta_lim
                        print(f" cos_theta_est = {cos_theta_est:.3f}  =>  cos_theta_p = {cos_theta_p:.3f}")
                    ax_cos_theta_est.axvline(separation_planet, c='k', ls="--", label=f'Angular separation{for_planet_name}')
                    ax_cos_theta_est.plot([separation_planet, separation_planet], [cos_theta_est[idx_planet], cos_theta_est[idx_planet]], 'rX', ms=11, label=rf"cos $\theta_{{est}}${for_planet_name} ({cos_theta_est[idx_planet]:.2f})")
                    ax_cos_theta_est.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
                fig_cos_theta_est.tight_layout()
                fig_cos_theta_est.show()
            
        # =======================
        # Optional t_syst in [mn] diagnostic: see Eq.(14) of Martos et al. 2025
        # =======================
        if show_plot and show_t_syst and systematics:
            t_syst = DIT * fn_HF * R_corr * A_FWHM * ( sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2 ) / sigma_syst_prime_2
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
        if show_syst_budget and post_processing == "molecular mapping":
            PSF, S_star     = np.meshgrid(PSF_profile, star_spectrum_band.flux)
            S_star          = N_DIT * S_star                 # total e-/FOV/spectral channel
            PSF_S_star      = A_FWHM * PSF * S_star          # total e-/FWHM/spectral channel
            mean_PSF_S_star = np.nanmean(PSF_S_star, axis=0) # TODO: mean_PSF_S_star = np.nanmedian(PSF_S_star, axis=0)
            mask_iwa        = separation >= iwa
            
            # --------------------------
            # Systematic noise budget
            # --------------------------
            sigma_fund_prime = np.sqrt(N_DIT * sigma_fund_prime_2 / fn_HF) # total projected e-/FWHM (without fn_HF)
            sigma_syst_prime = sigma_fund_prime
            sigma_m_syst     = sigma_syst_prime / mean_PSF_S_star
            
            if show_plot:
                fig_sigma_m_syst = plt.figure(figsize=(10, 6), dpi=300)
                ax_sigma_m_syst  = fig_sigma_m_syst.gca()
                ax_sigma_m_syst.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax_sigma_m_syst.set_xlim(0, separation[-1])
                ax_sigma_m_syst.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_sigma_m_syst.set_ylabel("Systematics modulation budget [%]", fontsize=14)
                ax_sigma_m_syst.set_title(f"{instru} systematics budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax_sigma_m_syst.tick_params(axis='both', labelsize=12)        
                ax_sigma_m_syst.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax_sigma_m_syst.plot(separation[mask_iwa], 100*sigma_m_syst[mask_iwa], c="black")
                if systematics:
                    ax_sigma_m_syst.plot(separation[mask_iwa], 100*np.nanmedian( np.sqrt(N_DIT**2*sigma_syst_prime_2) / PSF_S_star, axis=0)[mask_iwa], c="black", ls="--", label="Real systematic level") # assuming that sigma_syst_prime_2 ~ mean( sigma_syst_2 )
                    ax_sigma_m_syst.legend(fontsize=12, loc="upper right", frameon=True, fancybox=True, shadow=True, borderpad=1)
                ymin_s, ymax_s = ax_sigma_m_syst.get_ylim()
                ymin_s         = max(ymin_s, 0.)
                ax_sigma_m_syst.set_ylim(ymin_s, ymax_s)
                ax_sigma_m_syst.fill_between(separation[mask_iwa], 100*sigma_m_syst[mask_iwa], y2=ymax_s, color='crimson', alpha=0.3)
                ax_sigma_m_syst.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "Systematic-dominated regime", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_syst.fill_between(separation[mask_iwa], 100*sigma_m_syst[mask_iwa], y2=ymin_s, color='royalblue', alpha=0.3)
                ax_sigma_m_syst.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "Fundamental-dominated regime", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_syst.minorticks_on()
                fig_sigma_m_syst.tight_layout()
                fig_sigma_m_syst.show()
            
            # --------------------------
            # Speckle noise budget
            # --------------------------
            sigma_syst_prime  = np.sqrt(N_DIT**2 * sigma_syst_prime_2) # total projected e-/FWHM
            sigma_speck_prime = np.sqrt( (delta**2 / (alpha - beta)**2 * fn_HF**2 - 1) * (sigma_fund_prime**2 + sigma_syst_prime**2) ) # assuming that sigma_syst_prime_2 ~ mean( sigma_syst_2 )
            sigma_m_speck     = sigma_speck_prime / mean_PSF_S_star
            
            if show_plot:
                fig_sigma_m_peckles = plt.figure(figsize=(10, 6), dpi=300)
                ax_sigma_m_peckles  = fig_sigma_m_peckles.gca()
                ax_sigma_m_peckles.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax_sigma_m_peckles.set_xlim(0, separation[-1])
                ax_sigma_m_peckles.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_sigma_m_peckles.set_ylabel("Speckles stability budget [%]", fontsize=14)
                ax_sigma_m_peckles.set_title(f"{instru} speckles stability budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax_sigma_m_peckles.tick_params(axis='both', labelsize=12)        
                ax_sigma_m_peckles.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax_sigma_m_peckles.plot(separation[mask_iwa], 100*sigma_m_speck[mask_iwa], c="black")
                ymin_s, ymax_s = ax_sigma_m_peckles.get_ylim()
                ymin_s         = max(ymin_s, 0.)
                ax_sigma_m_peckles.set_ylim(ymin_s, ymax_s)
                ax_sigma_m_peckles.fill_between(separation[mask_iwa], 100*sigma_m_speck[mask_iwa], y2=ymax_s, color='crimson', alpha=0.3)
                ax_sigma_m_peckles.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "MM > DI", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_peckles.fill_between(separation[mask_iwa], 100*sigma_m_speck[mask_iwa], y2=ymin_s, color='royalblue', alpha=0.3)
                ax_sigma_m_peckles.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "MM < DI", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_m_peckles.minorticks_on()
                fig_sigma_m_peckles.tight_layout()
                fig_sigma_m_peckles.show()
            
            # ---- Non-coronagraphic PSF ---- 
            if coronagraph is not None:
                I_PSF, _, _, _, _ = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=None, instru=instru, separation_planet=separation_planet, return_SNR_planet=return_SNR_planet)
            else:
                I_PSF = np.copy(PSF_profile)

                # "Undo" FPM effect in the core
                if instru == "HARMONI" and iwa_FPM is not None:
                    I_PSF[separation < iwa_FPM] /= 1e-4
            
            # ---- Diffraction-only model (normalized) ----      
            D_m    = config_data["telescope"]["diameter"]
            S_m2   = config_data["telescope"]["area"] # S = pi/4 * D^2 * (1 - eps^2)
            ratio  = 4.0 * S_m2 / (np.pi * D_m**2)    # should be <= 1
            ratio  = np.clip(ratio, 0.0, 1.0)
            eps    = np.sqrt(1.0 - ratio)
            I_diff = airy_profile_norm(sep=separation, sep_unit=sep_unit, wave_um=wave_band, D_m=D_m, eps=eps)
            
            # ---- Estimating SR and phi0 ----      
            r       = separation
            dr      = np.gradient(r)
            r_in    = np.clip(r - dr/2, 0, None)
            r_out   = r + dr/2
            area    = np.pi*(r_out**2 - r_in**2) # aire du disque central incluse
            I_PSF  /= np.nansum(I_PSF  * area)   # F∝∫2πrI(r)dr
            I_diff /= np.nansum(I_diff * area)
            lam_um         = np.nanmean(wave_band) # [µm]
            lam_m          = lam_um * 1e-6         # [m]
            lam_over_D_rad = lam_m / D_m           # [rad]
            if sep_unit == "arcsec":
                r_core = lam_over_D_rad * rad2arcsec
            elif sep_unit == "mas":
                r_core = lam_over_D_rad * rad2arcsec * 1e3
            mask_core = (separation <= 1.22*r_core)
            w         = area
            SR        = np.nansum(w[mask_core]*I_PSF[mask_core]*I_diff[mask_core]) / np.nansum(w[mask_core]*I_diff[mask_core]**2)
            SR        = np.clip(SR, 1e-8, 1.0)
            phi0      = np.sqrt(-np.log(SR))
            if verbose:
                print(f" Estimated SR:  {100*SR:.1f} %")
                print(f" Estimated WFE: {1e3*np.nanmean(wave_band) / (2*np.pi) * phi0:.1f} nm RMS")
            
            # ---- Computing sigma_m_speck equivalent phase stability ---- 
            # We adopt rho = 1 (or at least rho > 0.5) because sigma_m_speck is intended to represent a quasi-static
            # speckle stability requirement over minute-long (or longer) DITs: once fast AO residuals are time-averaged,
            # the dominant variability is set by slow NCPA/thermal–mechanical drifts, so the speckle pattern remains
            # largely correlated between exposures and mainly “breathes” in amplitude (coherent drift → rho ~ 1).
            # Using rho = 1 is also the conservative choice for an instrument requirement, as it yields the most
            # stringent wavefront-stability budget for a given intensity-modulation budget.
            rho             = 0.5
            sigma_phi_speck = phi0 * (-rho + np.sqrt(rho**2 + sigma_m_speck))

            # phase [rad] -> WFE [nm RMS]
            sigma_phi_speck = 1e3*np.nanmean(wave_band) / (2*np.pi) * sigma_phi_speck
            
            if show_plot:
                fig_sigma_phi_speck = plt.figure(figsize=(10, 6), dpi=300)
                ax_sigma_phi_speck  = fig_sigma_phi_speck.gca()
                ax_sigma_phi_speck.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax_sigma_phi_speck.set_xlim(0, separation[-1])
                ax_sigma_phi_speck.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax_sigma_phi_speck.set_ylabel("Wavefront RMS stability [nm]", fontsize=14)
                ax_sigma_phi_speck.set_title(f"{instru} wavefront stability budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax_sigma_phi_speck.tick_params(axis='both', labelsize=12)        
                ax_sigma_phi_speck.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax_sigma_phi_speck.plot(separation[mask_iwa], sigma_phi_speck[mask_iwa], c="black")
                ymin_s, ymax_s = ax_sigma_phi_speck.get_ylim()
                ymin_s         = max(ymin_s, 0.)
                ax_sigma_phi_speck.set_ylim(ymin_s, ymax_s)
                ax_sigma_phi_speck.fill_between(separation[mask_iwa], sigma_phi_speck[mask_iwa], y2=ymax_s, color='crimson', alpha=0.3)
                ax_sigma_phi_speck.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "MM > DI", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_phi_speck.fill_between(separation[mask_iwa], sigma_phi_speck[mask_iwa], y2=ymin_s, color='royalblue', alpha=0.3)
                ax_sigma_phi_speck.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "MM < DI", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                ax_sigma_phi_speck.minorticks_on()
                fig_sigma_phi_speck.tight_layout()
                fig_sigma_phi_speck.show()
            
            # Saving quantities
            sigma_m_syst_bands.append(sigma_m_syst)
            sigma_m_speck_bands.append(sigma_m_speck)
            sigma_phi_speck_bands.append(sigma_phi_speck)
    
    sigma_syst_budget = [sigma_m_syst_bands, sigma_m_speck_bands, sigma_phi_speck_bands]
        
    # =======================
    # Final plot
    # ======================= 
    
    # Figure cleanup
    if show_plot:
        fig_band_flux.show()
        
    IWA = np.max(iwa_bands)
        
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
            mask_title = f" with {apodizer} apodizer,"
        else:
            mask_title = ""
        ax_contrast.set_title(f"{instru} {post_processing} contrast curves{for_planet_name} with $t_{{exp}}$ = {int(round(exposure_time))} mn," + mask_title + f"\n $mag_*$({band0_star}) = {round(mag_star, 1)} and $T_p$ = {int(round(T_planet))} K ({model_planet} model)", fontsize=16)        
        ax_contrast.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
        ax_contrast.set_ylabel(r'5$\sigma$ contrast (on instru-band)', fontsize=14)
        ax_contrast.axvspan(0, IWA, color='black', alpha=0.3, lw=0)
        for i in range(len(contrast_bands)):
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax_contrast.plot(separation_bands[i][separation_bands[i] >= IWA], contrast_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax_contrast.plot(separation_bands[i][separation_bands[i] >= IWA], contrast_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
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
                        x_text    = separation_planet - 0.1 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "center"
                    else:
                        x_text    = separation_planet + 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "right"
                else :
                    y_text = planet_to_star_ratio*1.5
                    if separation_planet > (IWA+OWA)/2:
                        x_text    = separation_planet - 0.1 * max(np.max(arr) for arr in separation_bands)
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
        fig_contrast.show()
    
    # ---------------------------------------
    # SNR
    # ---------------------------------------     
    elif calculation == "SNR" and show_plot:
        
        if channel and instru == "MIRIMRS":
            # Each MIRIMRS channel consists of 3 sub-bands (SHORT, MEDIUM, LONG).
            # To combine them, scale exposure time accordingly.
            exposure_time    *= 3  
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
            # Find the global maximum SNR across all channels
            SNR_max = max(np.nanmax(snr) for snr in SNR_channels)
            # If a planet separation is specified, find its maximum SNR
            SNR_max_planet = 0.0
            band_SNR_max   = ""
            if separation_planet is not None:
                for i, snr_curve in enumerate(SNR_channels):
                    # Ensure the separation exists in the grid before indexing
                    if separation_planet in separation_channels[i]:
                        idx_planet = np.where(separation_channels[i] == separation_planet)[0][0]
                        if snr_curve[idx_planet] > SNR_max_planet:
                            SNR_max_planet = snr_curve[idx_planet]
                            band_SNR_max   = name_channels[i]
            # Update bands and separations for consistency downstream
            separation_bands = separation_channels
            SNR_bands        = SNR_channels
            name_bands       = name_channels

        if separation_planet is not None and verbose:
            print(f"\nMAX S/N (at {separation_planet:.1f} {sep_unit}) = {SNR_max_planet:.1f} for {band_SNR_max.replace('_', ' ')}")
        
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
        ax_SNR.set_title(f"{instru} {post_processing} S/N curves{for_planet_name} with $t_{{exp}}$ = {int(round(exposure_time))} mn," + mask_title + f"\n $mag_*$({band0_star}) = {round(mag_star, 1)}, $mag_p$({band0_planet}) = {round(mag_planet, 1)}, $T_p$ = {int(round(T_planet))}K ({model_planet} model)", fontsize=16)
        ax_SNR.set_xlabel(f"separation [{sep_unit}]", fontsize=14)
        ax_SNR.set_ylabel('S/N', fontsize=14)        
        ax_SNR.axvspan(0, IWA, color='black', alpha=0.3, lw=0)
        for i in range(len(SNR_bands)):
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax_SNR.plot(separation_bands[i][separation_bands[i] >= IWA], SNR_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax_SNR.plot(separation_bands[i][separation_bands[i] >= IWA], SNR_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
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
        fig_SNR.show()
        
    # ---------------------------------------
    # RETURNS
    # ---------------------------------------
    if calculation == "contrast":
        results_bands = contrast_bands
    elif calculation == "SNR":
        results_bands = SNR_bands
    elif calculation == "corner plot":
        results_bands = uncertainties_bands
    return name_bands, separation_bands, results_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands, sigma_syst_budget



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves init
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation=None, instru=None, exposure_time=None, mag_star=None, mag_planet=None, band0=None, band0_star=None, band0_planet=None, model_planet=None, T_planet=None, lg_planet=None, model_star="BT-NextGen", T_star=None, lg_star=None, rv_star=0, rv_planet=0, vsini_star=0, vsini_planet=0, 
             apodizer="NO_SP", strehl="NO_JQ", coronagraph=None, systematics=False, speckles=False, PCA=False, PCA_mask=False, N_PCA=20, channel=False, planet_name=None, separation_planet=None, show_plot=True, verbose=True, 
             post_processing="molecular mapping", background="medium", Rc=100, filter_type="gaussian", input_DIT=None, band_only=None, 
             star_spectrum=None, planet_spectrum=None, return_SNR_planet=False, return_quantity=False):
    """
    Orchestrates inputs (spectra, models, config) and delegates to FastCurves_main
    to compute contrast/SNR/uncertainty for the selected instrument/setup.
    
    Note: 'sigma_prime' detones the 'sigma' per spectral channel projected into the CCF 

    Parameters
    ----------
    calculation : {"contrast","SNR","corner plot"}
        Type of calculation to perform.
    instru : str
        Instrument key (must be in 'instru_name_list' and supported by config).
    exposure_time : float
        Total exposure time [minutes].
    mag_star, mag_planet : float
        Magnitudes of star/planet. Planet magnitude required for SNR/corner-plot.
    band0, band0_star, band0_planet : str
        Photometric bands for magnitudes. If only 'band0' is given, it's used for both.
    model_planet, T_planet, lg_planet : str, float, float
        Planet atmospheric model + parameters (Teff [K], log g [dex(cm/s^2)]).
    model_star, T_star, lg_star : str, float, float
        Star atmospheric model + parameters (Teff [K], log g [dex(cm/s^2)]).
    rv_*, vsini_* : float
        Radial velocity and projected rotation [km/s] for star/planet.
    apodizer, strehl, coronagraph : str | None
        Optical setup (validated against the instrument config).
    systematics, speckles, PCA, PCA_mask, N_PCA :
        Systematics and speckles noise modeling and PCA removal configuration.
    channel : bool
        For MIRIMRS, combine SNR by channel (3 sub-bands per channel).
    planet_name : str
        Label only (for plots/prints).
    separation_planet : float
        Planet separation in **arcsec** (converted internally if needed).
    post_processing : {"molecular mapping","differential imaging"} or None
        Defaults based on instrument type (IFU → molecular mapping, imager → differential imaging).
    background : {"low","medium","high"} or None
        Background level.
    Rc : float or None
        High-pass cutoff resolution (None → no filtering).
    filter_type : str
        Filtering kernel ("gaussian", "step", "smoothstep", ...).
    input_DIT : float or None
        Force a particular DIT [minutes].
    band_only : str or None
        If provided, compute only that band.
    star_spectrum, planet_spectrum : Spectrum-like or None
        Provide pre-loaded spectra; otherwise they are loaded from models.
    return_SNR_planet : bool
        FastYield helper: returns per-band SNR and components at planet separation.
    return_quantity : bool
        Returns deeper per-band quantities (signals/noises) for analysis.

    Returns
    -------
    - If return_SNR_planet:
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
    if instru not in instru_name_list:
        raise KeyError(f"instru={instru} is not valid. Available: {instru_name_list}")
        
    # ---- Validate post-processing entry ----
    if post_processing.lower() not in {"molecular mapping", "mm", "differential imaging", "di"}:
        raise KeyError(f"post_processing={post_processing} is not valid. Available: 'Molecular Mapping', 'MM', 'Differential Imaging', 'DI'")
    if config_data["type"] == "imager" and post_processing.lower() not in {"differential imaging", "di"}:
        post_processing = "differential imaging"
        
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
    lmin_instru = config_data["lambda_range"]["lambda_min"]
    lmax_instru = config_data["lambda_range"]["lambda_max"]
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
    if band0_star != "instru" and band0_star not in bands:
        raise KeyError(f"{band0_star} is not a recognized magnitude band. Choose among: {bands} or 'instru'")
    if band0_planet is None:
        band0_planet = band0
    if band0_planet != "instru" and band0_planet not in bands:
        raise KeyError(f"{band0_planet} is not a recognized magnitude band. Choose among: {bands} or 'instru'")

    # ---- Load / prepare star spectrum if not provided ----
    if star_spectrum is None:
        if (model_star is None) or (T_star is None) or (lg_star is None):
            raise KeyError("Please define model_star, T_star, and lg_star to load the star spectrum.")
        star_spectrum = load_star_spectrum(T_star, lg_star, model=model_star).copy()
        # Crop around photometric & instrument ranges
        star_spectrum.crop(0.98 * min(globals()[f"lmin_{band0_star}"], globals()[f"lmin_{instru}"]), 1.02 * max(globals()[f"lmax_{band0_star}"], globals()[f"lmax_{instru}"]))
        # Rotational broadening of the spectrum [km/s]
        if vsini_star > 0: # the wavelength axis needs to be evenly spaced before broadening
            star_spectrum = star_spectrum.evenly_spaced(renorm=False)
            star_spectrum = star_spectrum.broad(vsini_star)
        # Doppler shifting the spectrum [km/s]
        star_spectrum = star_spectrum.doppler_shift(rv_star)
    if (star_spectrum.wavelength[0] > lmin_instru) or (star_spectrum.wavelength[-1] < lmax_instru):
        raise ValueError(f"'star_spectrum' does not fully cover the {instru} range ({lmin_instru}–{lmax_instru} µm).")
  
    # ---- Load / prepare planet spectrum if not provided ----
    if planet_spectrum is None:
        if (model_planet is None) or (T_planet is None) or (lg_planet is None):
            raise KeyError("Please define model_planet, T_planet, and lg_planet to load the planet spectrum.")
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model_planet, instru=instru).copy()
        # Crop around photometric & instrument ranges
        planet_spectrum.crop(0.98 * min(globals()[f"lmin_{band0_planet}"], globals()[f"lmin_{instru}"]), 1.02 * max(globals()[f"lmax_{band0_planet}"], globals()[f"lmax_{instru}"]))
        # Rotational broadening of the spectrum [km/s]
        if vsini_planet > 0: # the wavelength axis needs to be evenly spaced before broadening
            planet_spectrum = planet_spectrum.evenly_spaced(renorm=False)
            planet_spectrum = planet_spectrum.broad(vsini_planet)
        # Doppler shifting the spectrum [km/s]
        planet_spectrum = planet_spectrum.doppler_shift(rv_planet)
    if (planet_spectrum.wavelength[0] > lmin_instru) or (planet_spectrum.wavelength[-1] < lmax_instru):
        raise ValueError(f"'planet_spectrum' does not fully cover the {instru} range ({lmin_instru}–{lmax_instru} µm).")

    # ---- Delegate to FastCurves_main ----
    name_bands, separation, results_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands, sigma_syst_budget = FastCurves_main(calculation=calculation, instru=instru, exposure_time=exposure_time, mag_star=mag_star, band0_star=band0_star, band0_planet=band0_planet, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, speckles=speckles, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, channel=channel, planet_name=planet_name, separation_planet=separation_planet, mag_planet=mag_planet, show_plot=show_plot, verbose=verbose, post_processing=post_processing, sep_unit=sep_unit, background=background, Rc=Rc, filter_type=filter_type, input_DIT=input_DIT, band_only=band_only, return_SNR_planet=return_SNR_planet)

    if verbose:
        print(f'\nFastCurves {calculation} calculation took {round(time.time()-time1, 1)} s')

    # ---- FASTYIELD branch: per-band values at the planet separation ----
    if return_SNR_planet:
        
        if calculation != "SNR":
            raise KeyError("For return_SNR_planet=True, set calculation='SNR'.")
        if separation_planet is None:
            raise KeyError("Please provide 'separation_planet' for the SNR calculation.")

        # Convert the requested separation from arcsec to mas if the instrument works in mas
        if sep_unit == "mas":
            separation_planet *= 1e3 # switching the angular separation unit (from arcsec to mas)
        
        SNR_planet        = np.zeros((len(name_bands)))
        signal_planet     = np.zeros((len(name_bands)))
        sigma_fund_planet = np.zeros((len(name_bands)))
        sigma_syst_planet = np.zeros((len(name_bands)))
        
        # Retrieving the values at the planet separation
        for nb, band in enumerate(name_bands):
            idx_planet = np.where(separation[nb]==separation_planet)[0][0]
            SNR_planet[nb]        = results_bands[nb][idx_planet]
            signal_planet[nb]     = signal_bands[nb][idx_planet]
            sigma_fund_planet[nb] = np.sqrt(sigma_fund_2_bands[nb][idx_planet])
            sigma_syst_planet[nb] = np.sqrt(sigma_syst_2_bands[nb][idx_planet])
        
        return name_bands, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, np.array(DIT_bands)
    
    # ---- Deep analysis branch ----
    if return_quantity:
        return name_bands, separation, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands, sigma_syst_budget

    # ---- Standard return ----
    return name_bands, separation, results_bands


