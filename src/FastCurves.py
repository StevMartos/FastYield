from src.signal_noise_estimate import *
from src.utils import _load_corona_profile, _load_corr_factor, _load_bkg_flux
from src.signal_noise_estimate import _get_transmission
path_file = os.path.dirname(__file__)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves Function
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves_main(calculation, instru, exposure_time, mag_star, band0_star, band0_planet, planet_spectrum, star_spectrum, tellurics, apodizer, strehl, coronagraph, systematic, PCA, PCA_mask, N_PCA, channel, planet_name, separation_planet, mag_planet, show_plot, verbose, post_processing, sep_unit, background, Rc, filter_type, input_DIT, band_only, return_SNR_planet):
    """
    See the function "FastCurves" below.
    """

    # ---------------------------------------
    # Hard-coded toggles for diagnostics
    # ---------------------------------------
    cos_p              = 1     # Intrisic mismatch 
    cos_est            = None  # Estimated correlation in the data
    show_cos_theta_est = False # To see the impact of the noise on the estimated correlation in order to retrieve the true mismatch
    show_t_syst        = False # To see the systematic time domination
    show_contributions = True  # To see the noise contributions plots for contrast calculations
    show_syst_budget   = True  # To see the speckle noise budget for Differential Imaging techniques to be more advantageous than molecular mapping
    
    # ---------------------------------------
    # Config & constants
    # ---------------------------------------
    if instru == "MIRIMRS" and channel and (calculation in {"SNR", "corner plot"}):
        # Each SNR “per channel” splits time over its 3 sub-bands
        exposure_time = exposure_time / 3

    for_planet_name = f" for {planet_name}" if planet_name else ""
    
    config_data  = get_config_data(instru)
    NbBand       = len(config_data["gratings"])
    size_core    = config_data["size_core"]            # Aperture size on which the signal is integrated [pixels]
    A_FWHM       = size_core**2                        # Box aperture 
    saturation_e = config_data["spec"]["saturation_e"] # Full well capacity of the detector [e-]
    min_DIT      = config_data["spec"]["minDIT"]       # Minimal integration time [in mn]
    max_DIT      = config_data["spec"]["maxDIT"]       # Maximal integration time [mn]
    RON          = config_data["spec"]["RON"]          # Read out noise [e-/px/DIT]
    dark_current = config_data["spec"]["dark_current"] # Dark current [e-/px/s]
    IWA, OWA     = get_wa(config_data, sep_unit)       # Inner and Outer Working Angle in 'sep_unit' 
    if show_plot:
        cmap = plt.get_cmap("Spectral_r", NbBand + 1 if NbBand % 2 != 0 else NbBand)

    # ---------------------------------------
    # Accumulators
    # ---------------------------------------    
    contrast_bands      = [] # Final contrast as function of separation for exposure_time
    SNR_bands           = [] # Final S/N as function of separation for exposure_time
    name_bands          = [] # Band labels
    separation_bands    = [] # [arcsec] or [mas]
    signal_bands        = [] # [e-/FWHM/DIT]
    DIT_bands           = [] # [mn/DIT]
    sigma_syst_2_bands  = [] # [e-/FWHM/DIT]
    sigma_fund_2_bands  = [] # [e-/FWHM/DIT]
    sigma_halo_2_bands  = [] # [e-/FWHM/DIT]
    sigma_det_2_bands   = [] # [e-/FWHM/DIT]
    sigma_bkg_2_bands   = [] # [e-/FWHM/DIT]
    planet_flux_bands   = [] # [e-/DIT]
    star_flux_bands     = [] # [e-/DIT]
    wave_bands          = [] # [µm]
    uncertainties_bands = [] # [K], [dex(cm/s2)], [km/s] and [km/s]
    iwa_bands           = []
    
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
        if post_processing == "molecular mapping":
            print(f'\nMolecular mapping considered as post-processing method with Rc = {Rc} and {filter_type} filtering')
        elif post_processing == "ADI+RDI":
            print('\nADI and/or RDI considered as post-processing method')
        if systematic:
            if PCA:
                print(f'With systematics + PCA (with {N_PCA} components)')
            else:
                print('With systematics')
        else:
            print('Without systematics')
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
            raise KeyError(f"PLEASE INPUT A MAGNITUDE FOR THE PLANET FOR THE {calculation} CALCULATION !")
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
    mask_instru     = (wave_instru >= config_data["lambda_range"]["lambda_min"]) & (wave_instru <= config_data["lambda_range"]["lambda_max"])
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
    # Optional overview plot per band [e-/mn]: plotting the planet (if SNR calculation) or the star (if contrast calculation) on each band (in e-/mn)
    # ---------------------------------------
    if show_plot:
        f1 = plt.figure(figsize=(10, 6), dpi=300)
        band_flux = f1.add_subplot(111)
        band_flux.set_yscale("log")
        band_flux.set_xlim(wave_instru[0], wave_instru[-1])        
        band_flux.set_xlabel("Wavelength [µm]", fontsize=14)
        band_flux.set_ylabel("Flux [e-/mn]", fontsize=14)
        band_flux.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        band_flux.yaxis.set_ticks_position('both')
        band_flux.minorticks_on()
        ymin = 1e9
        ymax = 1
        if calculation == "SNR" or calculation == "corner plot":
            band_flux.set_title(f"Planet flux ({model_planet}) through {instru} bands \n with $T_p$={int(round(T_planet))}K, $lg_p$={round(lg_planet, 1)} and $mag_p$({band0_planet})={round(mag_planet, 2)}", fontsize=16)
        elif calculation == "contrast":
            band_flux.set_title(f"Star flux ({model_star}) through {instru} bands with $mag_*$({band0_star})={round(mag_star, 2)}", fontsize=16)
    
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
        R_band               = config_data['gratings'][band].R # Spectral resolution of the band
        
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
        PSF_profile, fraction_core, separation, pxscale, iwa_FPM = get_PSF_profile(band, strehl, apodizer, coronagraph, instru, separation_planet, return_SNR_planet)
        
        # Adding the band's separation axis to the list
        separation_bands.append(separation)
        
        # Index of the separation of the planet
        if separation_planet is not None:
            idx_planet = np.where(separation==separation_planet)[0][0]
                        
        # Coronagraphic radial transmission & core fraction vs separation        
        if coronagraph is not None:
            raw_sep, raw_fraction_PSF, raw_radial_transmission = _load_corona_profile(instru, band, strehl, apodizer, coronagraph)
            fraction_PSF_interp        = interp1d(raw_sep, raw_fraction_PSF,        bounds_error=False, fill_value="extrapolate")
            radial_transmission_interp = interp1d(raw_sep, raw_radial_transmission, bounds_error=False, fill_value="extrapolate")
            fraction_core        = fraction_PSF_interp(separation)        # Fraction of flux of a PSF inside the FWHM as function of the separation
            radial_transmission  = radial_transmission_interp(separation) # Transmission of a PSF as function of the separation
            star_transmission    = radial_transmission_interp(0)          # Total stellar flux transmitted by the coronagraph (+Lyot stop) when the star is perfectly aligned with it (i.e. at 0 separation)
            fraction_core[separation > raw_sep[-1]]       = raw_fraction_PSF[-1]        # Flat extrapolation
            radial_transmission[separation > raw_sep[-1]] = raw_radial_transmission[-1] # Flat extrapolation 
            PSF_profile *= star_transmission
                
        # Fiber-fed IFUs: mean injection correction (e.g. ANDES): the fact that the position of the planet is unknown
        if config_data["type"]=="IFU_fiber":
            try:
                fraction_core *= config_data["injection"][band]
            except:
                pass
        
        # Corrective factor (per separation): due to potential dithering (impacting the noise statistics, i.e. covariance, etc.)       
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
        # Molecular Mapping path
        # =======================
        if post_processing == "molecular mapping":
            
            # Power fraction of the white noise filtered by the high-pass filtering (<1)
            fn_HF   = get_fraction_noise_filtered(N=len(wave_band), R=R_band, Rc=Rc, filter_type=filter_type)[0]
            R_corr *= fn_HF
            
            # Systematic noise profile  
            if systematic:
                sigma_syst_prime_2, sep, m_HF, Ms, Mp, M_pca, wave, pca, PCA_verbose = get_systematic_profile(config_data=config_data, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, R=R_band, star_spectrum_instru=star_spectrum_instru, planet_spectrum_instru=planet_spectrum_instru, planet_spectrum=planet_spectrum, wave_band=wave_band, size_core=size_core, filter_type=filter_type, show_cos_theta_est=show_cos_theta_est, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, mag_planet=mag_planet, band0_planet=band0_planet, separation_planet=separation_planet, mag_star=mag_star, target_name=planet_name, exposure_time=exposure_time)
                sigma_syst_prime = interp1d(sep, np.sqrt(sigma_syst_prime_2), bounds_error=False, fill_value="extrapolate")(separation)
                if separation[-1] > sep[-1]: # Systematic profile extrapolation
                    mask_outside                   = (separation >= sep[-1])
                    extension                      = PSF_profile[mask_outside].copy()               # Same extrapolation profile as the PSF profile one (propto stellar flux)
                    extension                     *= np.sqrt(sigma_syst_prime_2[-1]) / extension[0] # Forcing continuity
                    sigma_syst_prime[mask_outside] = extension
                sigma_syst_prime_2_per_tot = sigma_syst_prime**2 # Systematic noise profile projected in the CCF in [e-/FWHM/total stellar flux]
                planet_spectrum_band.crop(wave[0], wave[-1])
                star_spectrum_band.crop(wave[0], wave[-1])
                mask_M    = (wave_band >= wave[0]) & (wave_band <= wave[-1]) # Effective wavelength axis (from data)
                trans     = trans[mask_M]
                wave_band = wave_band[mask_M]
                Ms        = Ms[mask_M]
                Mp        = Mp[mask_M]
        
            # Build template = trans * planet_HF (normalized)
            template, _ = filtered_flux(planet_spectrum_band.flux, R_band, Rc, filter_type) # [Sp]_HF
            template    = trans * template                                             # gamma * [Sp]_HF
            template    = template/np.sqrt(np.nansum(template**2))                     # Normalizing the template
            
            # Systematic modulations of the spectra are taken into account (mostly insignificant effect)
            if systematic:
                star_spectrum_band.flux   = Ms * star_spectrum_band.flux
                planet_spectrum_band.flux = Mp * planet_spectrum_band.flux

            # Self-subtraction (β) and useful signal (α) in [e-/mn] as function of separation (with systematic modulations, if any)
            beta  = get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R_band, trans, filter_type)
            alpha = get_alpha(planet_spectrum_band, template, Rc, R_band, trans, filter_type)
        
        # --------------------------
        # DIT in [mn] and effective read-out noise in [e-/px/DIT] 
        # --------------------------
        NDIT, DIT, DIT_saturation, RON_eff, iwa_FPM = get_DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, RON, saturation_e, input_DIT, iwa_FPM=iwa_FPM)
        
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
            if post_processing == "molecular mapping":
                label_band = band.replace('_', ' ') + f" (R={int(round(R_band))})"
            elif post_processing == "ADI+RDI":
                label_band = band.replace('_', ' ') + f" ({round(np.nansum(flux_band))} e-/mn)"
            band_flux.plot(wave_band, flux_band, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=label_band)
            band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
            ymin = min(ymin, np.nanmin(flux_band)/2)
            ymax = max(ymax, np.nanmax(flux_band)*2)
            band_flux.set_ylim(ymin=max(ymin, 1e-2), ymax=ymax)
        
        # =======================
        # Signal estimations
        # =======================
        
        if post_processing == "molecular mapping": # Projection onto the CCF
            # Total number of useful e-/DIT
            signal = np.zeros_like(separation) + (alpha*cos_p - beta) * DIT
            # Focal Plane Mask: flux attenuated by a factor of 1e-4 for HARMONI (FPM)
            if instru=="HARMONI" and iwa_FPM is not None:
                iwa                            = iwa_FPM
                PSF_profile[separation < iwa] *= 1e-4
                signal[separation < iwa]      *= 1e-4
                
        elif post_processing == "ADI+RDI": # Integrated flux
            # Total number of useful e-/DIT
            signal = np.zeros_like(separation) + np.nansum(planet_spectrum_band.flux)
        
        # Signal inside FWHM aperture in [e-/FWHM/DIT]
        signal *= fraction_core

        # Non-spatial-homegenous tranmission of the coronagraph
        if coronagraph is not None:
            signal *= radial_transmission
        
        # Renormalizing the signal with the planet-to-star ratio (in total received photons) on the instrumental bandwidth (by doing so, it will give a contrast in photons and not in energy on this bandwidth, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
        if calculation == "contrast":
            signal *= star_to_planet_ratio
            
        # Signal loss ratio due to the PCA (if required)
        if systematic:
            signal *= M_pca
        
        # =======================
        # Noise estimations
        # =======================
        
        # --------------------------
        # Detector noises, DC and RON in [e-/px/DIT]
        # --------------------------
        sigma_dc_2  = dark_current * 60*DIT # Dark current photon noise [e-/px/DIT]
        sigma_ron_2 = RON_eff**2            # Effective read out noise  [e-/px/DIT]
        
        if config_data["type"] == "IFU_fiber": # Detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
            NbPixel      = config_data['pixel_detector_projection'] # Number of detector px per spaxel
            sigma_dc_2  *= NbPixel # Adds quadratically [e-/spaxel/DIT]
            sigma_ron_2 *= NbPixel # Adds quadratically [e-/spaxel/DIT]
        
        # --------------------------
        # Stellar halo photon noise in [e-/spaxel/DIT] for IFU and [e-/px/DIT] for imager
        # --------------------------
        if post_processing == "molecular mapping": # Projection onto the CCF
            sigma_halo_prime_2 = PSF_profile * np.nansum(star_spectrum_band.flux * template**2) # Stellar photon noise projected in the CCF [e-/spaxel/DIT] for each separation
        
        elif post_processing == "ADI+RDI": # Integrated flux
            sigma_halo_2 = PSF_profile * np.nansum(star_spectrum_band.flux) # Stellar photon noise per spectral channel in [e-/px/DIT] for each separation

        # --------------------------
        # Background photon noise in [e-/spaxel/DIT] for IFU and [e-/px/DIT] for imager
        # --------------------------
        sigma_bkg_2       = 0.
        sigma_bkg_prime_2 = 0.

        if background is not None:
            
            raw_wave, raw_bkg  = _load_bkg_flux(instru, band, background)
            bkg_flux           = interp1d(raw_wave, raw_bkg, bounds_error=False, fill_value=np.nan)(wave_band) # [e-/px/s]
            tot_bkg_flux       = np.nansum(bkg_flux) # [e-/s]
            if tot_bkg_flux != 0:
                # We have to renormalize because we interpolated (flux conservation) in [e-/px/DIT]
                bkg_flux   *= np.nansum(raw_bkg[(raw_wave >= wave_band[0]) & (raw_wave <= wave_band[-1])]) / tot_bkg_flux * 60*DIT
                sigma_bkg_2 = bkg_flux # Background photon noise per spectral channel in [e-/px/DIT] for each separation
        
            if post_processing == "molecular mapping": # Projection onto the CCF
                sigma_bkg_prime_2 = np.nansum(sigma_bkg_2 * template**2) # Background photon noise projected in the CCF in [e-/spaxel/DIT]
                if coronagraph is not None: # Non-homogenous sky transmission
                    sigma_bkg_prime_2 *= radial_transmission # Background photon noise projected in the CCF in [e-/spaxel/DIT]

            elif post_processing == "ADI+RDI": # Integrated flux
                sigma_bkg_2 = np.nansum(sigma_bkg_2) # Background photon noise per spectral channel in [e-/px/DIT] for each separation
                if coronagraph is not None: # Non-homogenous sky transmission
                    sigma_bkg_2 *= radial_transmission
        
        # --------------------------
        # Systematic noise in [e-/FWHM/DIT] for IFU and [e-/FWHM/DIT] for imager
        # --------------------------
        sigma_syst_2       = np.zeros_like(separation)
        sigma_syst_prime_2 = np.zeros_like(separation)
        
        if systematic:
            
            if post_processing == "molecular mapping": # Projection onto the CCF
                sigma_syst_prime_2 = sigma_syst_prime_2_per_tot * np.nansum(star_spectrum_band.flux)**2 # Systematic noise profile projected in the CCF in [e-/spaxel/DIT]
            
            elif post_processing == "ADI+RDI": # Integrated flux
                raise KeyError("Undefined !")    
        
        # =======================
        # Verbose band-level summary
        # =======================
        if verbose:
            if config_data["type"] == "imager":
                print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm):"+"\033[0m")
            else:
                print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm with R={R_band:.0f}):"+"\033[0m")
            
            if DIT_saturation < min_DIT:
                print(f" Saturation would occur even at the shortest DIT; using min_DIT ({min_DIT*60:.2f} s).")
            print(f" DIT = {DIT*60:.1f} s / Saturating DIT = {DIT_saturation:.2f} mn / RON_eff = {RON_eff:.3f} e-/DIT")
            
            print(f" Mean total system transmission = {100*np.nanmean(trans):.1f} %")
            
            if "IFU" in config_data["type"]:
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
            if post_processing == "molecular mapping":
                print(f" Signal loss due to self-subtraction: β/α = {100*beta/alpha:.1f} %")
            
            if PCA and systematic:
                if PCA_verbose is not None:
                    print(PCA_verbose)
                print(f" Signal loss due to PCA = {100*(1-M_pca):.1f} %")
                
            if iwa_FPM is not None:
                print(f" Using a FPM under {iwa_FPM:.1f} mas to avoid saturation")
        
        # =======================
        # Saving quantities
        # =======================
        signal_bands.append(signal)                         # [e-/FWHM/DIT]
        planet_flux_bands.append(planet_spectrum_band.flux) # [e-/DIT]
        star_flux_bands.append(star_spectrum_band.flux)     # [e-/DIT]
        wave_bands.append(wave_band)                        # [µm]
        iwa_bands.append(iwa)
        
        # Saving noises in [e-/FWHM/DIT]
        if post_processing == "molecular mapping":
            sigma_fund_2_bands.append(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2))
            sigma_halo_2_bands.append(R_corr*A_FWHM*(sigma_halo_prime_2))
            sigma_det_2_bands.append(R_corr*A_FWHM*(sigma_ron_2 + sigma_dc_2))
            sigma_bkg_2_bands.append(R_corr*A_FWHM*(sigma_bkg_prime_2))
            sigma_syst_2_bands.append(sigma_syst_prime_2)

        elif post_processing == "ADI+RDI":
            sigma_fund_2_bands.append(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2))
            sigma_halo_2_bands.append(R_corr*A_FWHM*(sigma_halo_2))
            sigma_det_2_bands.append(R_corr*A_FWHM*(sigma_ron_2 + sigma_dc_2))
            sigma_bkg_2_bands.append(R_corr*A_FWHM*(sigma_bkg_2))
            sigma_syst_2_bands.append(sigma_syst_2)
        
        # =======================
        # 5σ contrast computation
        # =======================
        if calculation == "contrast":
            
            if post_processing == "molecular mapping": # See Eq. (11) of Martos et al. 2025
                contrast = 5 * np.sqrt( R_corr * A_FWHM * (sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2) + NDIT*sigma_syst_prime_2 ) / ( np.sqrt(NDIT) * signal )
            
            elif post_processing == "ADI+RDI":
                contrast = 5 * np.sqrt( R_corr * A_FWHM * ( sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2) + NDIT*sigma_syst_2 ) / ( np.sqrt(NDIT) * signal )
            
            # Adding the contrast curve of the band to the list
            contrast_bands.append(contrast)
            
            # --------------------------
            # Noise contributions plots (in 5σ contrast)
            # --------------------------
            if show_plot and show_contributions:
                mask_iwa = separation >= iwa
                plt.figure(figsize=(10, 6), dpi=300)        
                ax1 = plt.gca()
                ax1.set_yscale('log')        
                ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax1.set_xlim(0, separation[-1])
                ax1.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax1.tick_params(axis='both', labelsize=12)        
                ax2 = ax1.twinx()
                ax2.set_yscale('log')
                ax2.set_xlim(0, separation[-1])
                ax2.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax2.get_yaxis().set_visible(False)
                ax2.get_xaxis().set_visible(False)        
                if post_processing == "molecular mapping":
                    ax1.set_title(f"{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name}\n with "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                    ax1.set_ylabel(r"Contrast 5$\sigma_{CCF}$ / $\alpha_0$", fontsize=14)
                    ax1.plot(separation[mask_iwa], contrast[mask_iwa], 'k-', label=r"$\sigma_{CCF}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2)*(NDIT))/(signal*NDIT))[mask_iwa], c="crimson",   ls="--", label=r"$\sigma'_{halo}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_ron_2)*(NDIT))/(signal*NDIT))[mask_iwa],        c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_dc_2)*(NDIT))/(signal*NDIT))[mask_iwa],         c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_bkg_prime_2)*(NDIT))/(signal*NDIT))[mask_iwa],  c="royalblue", ls="--", label=r"$\sigma'_{bkg}$")
                    if systematic:
                        ax1.plot(separation[mask_iwa], (5*np.sqrt(sigma_syst_prime_2)/(signal))[mask_iwa], c="cyan", ls="--", label=r"$\sigma'_{syst}$")
                elif post_processing == "ADI+RDI":
                    ax1.set_title(f"{instru} noise contributions on {band.replace('_', ' ')}{for_planet_name} \n with "r"$t_{exp}$="f"{round(exposure_time)}mn and "r"$mag_*$"f"({band0_star})={round(mag_star, 2)}", fontsize=16)        
                    ax1.set_ylabel(r'Contrast 5$\sigma$ / $F_{p}$', fontsize=14)
                    ax1.plot(separation[mask_iwa], contrast[mask_iwa], 'k-', label=r"$\sigma_{tot}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_halo_2)*(NDIT))/(signal*NDIT))[mask_iwa], c="crimson",   ls="--", label=r"$\sigma_{halo}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_ron_2)*(NDIT))/(signal*NDIT))[mask_iwa],  c="seagreen",  ls="--", label=r"$\sigma_{ron}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_dc_2)*(NDIT))/(signal*NDIT))[mask_iwa],   c="magenta",   ls="--", label=r"$\sigma_{dc}$")
                    ax1.plot(separation[mask_iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_bkg_2)*(NDIT))/(signal*NDIT))[mask_iwa],  c="royalblue", ls="--", label=r"$\sigma_{bkg}$")                    
                    if systematic:
                        ax1.plot(separation[mask_iwa], (5*np.sqrt(sigma_syst_2)/(signal))[mask_iwa], c="cyan", ls="--", label=r"$\sigma_{syst}$")
                if separation_planet is not None:
                    if separation_planet > 2 * OWA:
                        ax1.set_xscale('log')
                        ax1.set_xlim(iwa, separation[-1])
                    if mag_planet is None:
                        ax1.axvline(separation_planet, color="black", linestyle="--", label=f"{planet_name}" if planet_name is not None else "planet")
                        leg_loc = "upper right"
                    else:
                        if planet_to_star_ratio > ax1.get_ylim()[1] or (planet_to_star_ratio > ax1.get_ylim()[0] and planet_to_star_ratio < ax1.get_ylim()[1]):
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
                        ax1.plot([separation_planet, separation_planet], [planet_to_star_ratio, planet_to_star_ratio], 'ko')
                        ax1.annotate(f"{planet_name}" if planet_name is not None else "planet", (x_text, y_text), fontsize=12)
                else:
                    leg_loc = "upper right"
                ax3 = ax1.twinx()
                ax3.invert_yaxis()
                ax3.set_ylabel(r'$\Delta$mag', fontsize=14, labelpad=20, rotation=270)
                ax3.tick_params(axis='y', labelsize=12)        
                ymin_c_band, ymax_c_band = ax1.get_ylim()
                ax3.set_ylim(-2.5 * np.log10(ymin_c_band), -2.5 * np.log10(ymax_c_band))        
                ax1.legend(loc=leg_loc, fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
                plt.minorticks_on()
                plt.tight_layout()
                
            # --------------------------
            # Systematic (+speckles+ADI/RDI) noise budget
            # --------------------------
            if show_plot and show_syst_budget and post_processing == "molecular mapping" and not systematic:
                
                PSF, S_star  = np.meshgrid(PSF_profile, star_spectrum_band.flux)
                sigma_halo_2 = PSF * S_star         # Stellar photon noise [e-/px/DIT] for each separation
                sigma_bkg_2  = np.nanmean(bkg_flux) # Bkg photon noise [e-/px/DIT]
                if coronagraph is not None: # Non-homogenous sky transmission
                    sigma_bkg_2 *= radial_transmission
                sigma_fund = np.sqrt( NDIT * A_FWHM * R_corr * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2 ) ) # total e-/FWHM
                S_star     = NDIT * S_star
                PSF_S_star = A_FWHM * PSF * S_star
                sigma_m    = np.nanmedian(sigma_fund / (PSF_S_star), axis=0)
                                
                mask_iwa = separation >= iwa
                plt.figure(figsize=(10, 6), dpi=300)
                ax1 = plt.gca()
                ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax1.set_xlim(0, separation[-1])
                ax1.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax1.set_ylabel("Systematics modulation budget [%]", fontsize=14)
                ax1.set_title(f"{instru} systematic budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}", fontsize=16)        
                ax1.tick_params(axis='both', labelsize=12)        
                ax1.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax1.plot(separation[mask_iwa], 100*sigma_m[mask_iwa], c="black")
                ymin_s, ymax_s = ax1.get_ylim()
                #ymin_s, ymax_s = 0., 0.1
                ymin_s         = max(ymin_s, 0.)
                ax1.set_ylim(ymin_s, ymax_s)
                ax1.fill_between(separation[mask_iwa], 100*sigma_m[mask_iwa], y2=ymax_s, color='crimson', alpha=0.3)
                ax1.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "Systematic-dominated regime", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax1.fill_between(separation[mask_iwa], 100*sigma_m[mask_iwa], y2=ymin_s, color='royalblue', alpha=0.3)
                ax1.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "Fundamental-dominated regime", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                plt.minorticks_on()
                plt.tight_layout()
                
                Sp               = planet_spectrum_band.flux                                 # [e-/FWHM]
                Sp_HF, Sp_LF     = filtered_flux(Sp / trans, R_band, Rc, filter_type)                
                star_flux        = star_spectrum_band.flux                                   # [e-]
                star_HF, star_LF = filtered_flux(star_flux / trans, R_band, Rc, filter_type)            
                alpha            = np.nansum( trans*Sp_HF * template )                       # [e-/FWHM]
                beta             = np.nansum( trans*star_HF*Sp_LF/star_LF * template )       # [e-/FWHM]
                alpha0           = alpha - beta
                delta0           = np.sqrt( np.nansum( Sp**2 ) )
                sigma_m_speckles = sigma_m * delta0 / alpha0
                
                plt.figure(figsize=(10, 6), dpi=300)
                ax1 = plt.gca()
                ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax1.set_xlim(0, separation[-1])
                ax1.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax1.set_ylabel("Speckles stability budget [%]", fontsize=14)
                ax1.set_title(f"{instru} speckles stability budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}\n"r"$\alpha_0$/$\delta_0$ = "f"{100*alpha0/delta0:.0f} %", fontsize=16)        
                ax1.tick_params(axis='both', labelsize=12)        
                ax1.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax1.plot(separation[mask_iwa], 100*sigma_m_speckles[mask_iwa], c="black")
                ymin_s, ymax_s = ax1.get_ylim()
                ymin_s = max(ymin_s, 0.)
                ax1.set_ylim(ymin_s, ymax_s)
                ax1.fill_between(separation[mask_iwa], 100*sigma_m_speckles[mask_iwa], y2=ymax_s, color='crimson', alpha=0.3)
                ax1.text(separation[len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "MM > ADI/RDI", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax1.fill_between(separation[mask_iwa], 100*sigma_m_speckles[mask_iwa], y2=ymin_s, color='royalblue', alpha=0.3)
                ax1.text(separation[2*len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "MM < ADI/RDI", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                plt.minorticks_on()
                plt.tight_layout()
                
                I_max = PSF_profile[separation==0][0]
                if instru == "HARMONI" and iwa_FPM is not None: # FPM
                    I_max *= 1e4
                if coronagraph is not None:
                    I_max /= star_transmission
                C0        = PSF_profile / I_max
                phi0      = np.sqrt(C0 / (1 + C0))                             # approx np.sqrt(C0), since C0 << 1,  small-aberration: ~ sqrt(C0)
                rho       = 0.                                                 # Pearson correlation between phi0 and delta_phi (here hard coded at 0)
                delta_phi = phi0 * (-rho + np.sqrt(rho**2 + sigma_m_speckles)) # delta_phi [rad] induce sigma_m_speckles
                delta_phi = 1e3*np.nanmedian(wave_band) / (2*np.pi) * delta_phi  # nm RMS stability required

                plt.figure(figsize=(10, 6), dpi=300)
                ax1 = plt.gca()
                ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax1.set_xlim(0, separation[-1])
                ax1.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
                ax1.set_ylabel("Wavefront RMS stability [nm]", fontsize=14)
                ax1.set_title(f"{instru} wavefront stability budget on {band.replace('_', ' ')}{for_planet_name}\n for "r"$t_{exp}$="f"{round(exposure_time)}mn, "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}, "r"$T_p$="f"{int(round(T_planet))}K and "r"$R_c$="f"{Rc}\n"r"$\alpha_0$/$\delta_0$ = "f"{100*alpha0/delta0:.0f} %", fontsize=16)        
                ax1.tick_params(axis='both', labelsize=12)        
                ax1.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax1.plot(separation[mask_iwa], delta_phi[mask_iwa], c="black")
                ymin_s, ymax_s = ax1.get_ylim()
                ymin_s = max(ymin_s, 0.)
                ax1.set_ylim(ymin_s, ymax_s)
                ax1.fill_between(separation[mask_iwa], delta_phi[mask_iwa], y2=ymax_s, color='crimson', alpha=0.3)
                ax1.text(separation[2*len(separation)//3], ymax_s - 0.1*(ymax_s - ymin_s), "MM > ADI/RDI", color='crimson', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='crimson', alpha=0.8, boxstyle='round,pad=0.3'))
                ax1.fill_between(separation[mask_iwa], delta_phi[mask_iwa], y2=ymin_s, color='royalblue', alpha=0.3)
                ax1.text(separation[len(separation)//3], ymin_s + 0.15*(ymax_s - ymin_s), "MM < ADI/RDI", color='royalblue', fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='royalblue', alpha=0.8, boxstyle='round,pad=0.3'))
                plt.minorticks_on()
                plt.tight_layout()
        
        # =======================
        # S/N computation
        # =======================
        elif calculation in {"SNR", "corner plot"}:
            
            # See Eq. (10) of Martos et al. 2025
            if post_processing == "molecular mapping":
                SNR = np.sqrt(NDIT) * signal / np.sqrt( R_corr * A_FWHM * ( sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2) + NDIT*sigma_syst_prime_2 )
            
            elif post_processing == "ADI+RDI":
                SNR = np.sqrt(NDIT) * signal / np.sqrt( R_corr * A_FWHM * ( sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2) + NDIT*sigma_syst_2 )
                
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
                Mp_Sp              = fraction_core * NDIT * planet_spectrum_band.flux          # Planet flux (with modulations, if any) in [e-/FWHM]
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp / trans, R_band, Rc, filter_type)     # Filtered planet flux in [ph/FWHM]
                star_flux          = PSF_profile[idx_planet] * NDIT * star_spectrum_band.flux  # Stellar flux in [e-/px] at separation of the planet
                star_HF, star_LF   = filtered_flux(star_flux / trans, R_band, Rc, filter_type) # Filtered star flux
                d_planet           = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF           # Flux in [e-/FWHM] at the planet's location: see Eq.(18) of Martos et al. 2025
                # Total noise in [e-/FWHM/spectral channel] at the planet's location
                sigma_halo_2       = PSF_profile[idx_planet] * star_spectrum_band.flux
                sigma_l            = np.sqrt( R_corr[idx_planet] * A_FWHM * NDIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2)  )
                SNR_CCF            = SNR[idx_planet]
                
                print("R_corr     = ", R_corr[idx_planet])
                print("sigma_halo = ", np.nanmean(np.sqrt(NDIT*sigma_halo_2)))
                print("sigma_ron  = ", np.nanmean(np.sqrt(NDIT*sigma_ron_2)))
                print("sigma_dc   = ", np.nanmean(np.sqrt(NDIT*sigma_dc_2)))
                print("sigma_bkg  = ", np.nanmean(np.sqrt(NDIT*sigma_bkg_2)))
                print("sigma_l    = ", np.nanmean(sigma_l))
                N         = 20
                T_arr     = np.linspace(T_planet-200, T_planet+200, N)
                lg_arr    = np.linspace(lg_planet-0.5, lg_planet+0.5, N)
                vsini_arr = np.linspace(max(vsini_planet-5, 0), vsini_planet+5, N)
                vsini_arr = np.array([vsini_planet]) # R too low for retrieving this parameter
                rv_arr    = np.linspace(rv_planet-10, rv_planet+10, N)
                uncertainties      = parameters_estimation(instru=instru, band=band_only, target_name=planet_name, d_planet=d_planet, star_flux=star_flux, wave=wave_band, trans=trans, model=model_planet, R=R_band, Rc=Rc, filter_type=filter_type, logL=True, method_logL="classic", sigma_l=sigma_l, weight=None, pca=pca, stellar_component=True, degrade_resolution=True, SNR_estimate=False, T_arr=T_arr, lg_arr=lg_arr, vsini_arr=vsini_arr, rv_arr=rv_arr, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, SNR_CCF=SNR_CCF, d_planet_sim=False, template=planet_spectrum_instru, renorm_d_planet_sim=False, fastcurves=True, star_HF=star_HF, star_LF=star_LF, wave_interp=wave_instru, epsilon=0.8, fastbroad=True, force_new_est=True, save=False, exposure_time=exposure_time, show=True, verbose=True)

                
                #uncertainties      = parameters_estimation(instru=instru, band=band_only, target_name=planet_name, d_planet=d_planet, star_flux=star_flux, wave=wave_band, trans=trans, model=model_planet, R=R_band, Rc=Rc, filter_type=filter_type, logL=True, method_logL="classic", sigma_l=sigma_l, weight=None, pca=pca, stellar_component=True, degrade_resolution=True, SNR_estimate=False, T_arr=None, lg_arr=None, vsini_arr=None, rv_arr=None, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, SNR_CCF=SNR_CCF, d_planet_sim=False, template=planet_spectrum_instru, renorm_d_planet_sim=False, fastcurves=True, star_HF=star_HF, star_LF=star_LF, wave_interp=wave_instru, epsilon=0.8, fastbroad=True, force_new_est=True, save=False, exposure_time=exposure_time, show=True, verbose=True)
                uncertainties_bands.append(uncertainties)
            
            # ---------------------------------------------
            # Diagnostic: impact of noise + systematics on correlation estimation
            # (incl. auto-subtraction). Optionally uses m_HF from the systematic model.
            # ---------------------------------------------
            if show_plot and show_cos_theta_est and post_processing == "molecular mapping":
            
                # High-frequency modulation matrix M_HF(sep, lambda)
                # Defaults to zeros unless 'systematic' provided m_HF at sampled separations.
                M_HF = np.zeros((len(separation), len(wave_band)))
                if systematic:
                    # 'sep' & 'm_HF' come from get_systematic_profile; mask_M restricts to valid wavelengths
                    for i, s in enumerate(separation):
                        idx_sep = np.abs(s - sep).argmin()
                        M_HF[i] = m_HF[idx_sep][mask_M]
            
                # ---------- Build planet/star 1D spectra (per spectral channel) ----------
                # Units reminders:
                # - planet_spectrum_band.flux, star_spectrum_band.flux are in [e-/DIT/px] (after throughput)
                # - Multiplying by NDIT gives [e-/px] (per observation)
                # - Multiplying planet by fraction_core -> [e-/FWHM]
            
                # Planet (e-/FWHM), then split into HF/LF on *pre-throughput* and re-apply 'trans'
                Mp_Sp              = fraction_core * NDIT * planet_spectrum_band.flux  # [e-/FWHM]
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp / trans, R_band, Rc, filter_type)
                
                # Star, HF/LF of star without PSF factor (spatial term added per separation below)
                star_flux        = NDIT * star_spectrum_band.flux                         # [e-]
                star_HF, star_LF = filtered_flux(star_flux / trans, R_band, Rc, filter_type)
            
                # Effective signal and self-subtraction
                alpha         = np.sqrt( np.nansum( (trans*Mp_Sp_HF)**2 ) )            # [e-/FWHM]
                beta          = np.nansum( trans*star_HF*Mp_Sp_LF/star_LF * template ) # [e-/FWHM]
                cos_theta_lim = np.nansum( trans*Mp_Sp_HF * template ) / alpha
            
                # ---------- Monte Carlo over noise for cos(theta_est) ----------
                cos_theta_est = np.zeros_like(separation, dtype=float)
                norm_d        = np.zeros_like(separation, dtype=float)
            
                # Number of noise realizations;  vectorized per separation
                N   = 1_000
                rng = np.random.default_rng() # set a seed if you need reproducibility
            
                for i, s in enumerate(separation):
                    # Stellar flux *inside* the FWHM box at separation i (sum over pixels ~ A_FWHM)
                    # PSF_profile[i] is a per-pixel scaling; A_FWHM maps px -> FWHM box.
                    star_flux_FWHM = A_FWHM * PSF_profile[i] * star_flux # [e-/FWHM]
                    sigma_halo_2   = PSF_profile[i] * star_spectrum_band.flux
                    
                    
                    # Per-channel noise (photometric variance form) at separation i (scalar per channel)
                    sigma_l = np.sqrt( R_corr[i] * A_FWHM * NDIT * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_2 ) )  # [e-/FWHM/channel]
            
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
                plt.figure(dpi=300, figsize=(10, 6))
                plt.plot(separation, cos_theta_est, 'k')
                plt.ylabel(r"cos $\theta_{\rm est}$", fontsize=14)
                plt.xlabel(f"Separation [{sep_unit}]", fontsize=14)
                plt.xlim(separation[0], separation[-1])
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.minorticks_on()
                plt.title(f"Effect of noise and stellar subtraction on the correlation\nbetween template and planetary spectrum for {instru} on {band}\n(assuming the template equals the observed planet spectrum)", fontsize=16)
                if (separation_planet is not None) and (separation_planet < np.nanmax(separation)):
                    print(f" beta/alpha = {beta/alpha:.3f} | cos_theta_n = {cos_theta_n[idx_planet]:.3f} | cos_theta_lim = {cos_theta_lim:.3f}")
                    if cos_est is not None:
                        # Recover intrinsic mismatch estimate cos_theta_p from an observed cos_est
                        cos_theta_p = (cos_est / cos_theta_n[idx_planet] + beta/alpha) / cos_theta_lim
                        print(f" cos_theta_est = {cos_est:.3f}  =>  cos_theta_p = {cos_theta_p:.3f}")
                    plt.axvline(separation_planet, c='k', ls="--", label=f'Angular separation{for_planet_name}')
                    plt.plot([separation_planet, separation_planet], [cos_theta_est[idx_planet], cos_theta_est[idx_planet]], 'rX', ms=11, label=rf"cos $\theta_{{est}}${for_planet_name} ({cos_theta_est[idx_planet]:.2f})")
                    plt.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
                plt.tight_layout()
            
        # --------------------------
        # Optional t_syst in [mn] diagnostic: see Eq.(14) of Martos et al. 2025
        # --------------------------
        if show_plot and show_t_syst and systematic:
            t_syst = DIT * R_corr * A_FWHM * ( sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkg_prime_2 ) / sigma_syst_prime_2
            plt.figure(figsize=(10, 6), dpi=300)        
            plt.plot(separation, t_syst, c="crimson", ls="-", label="$t_{syst}$")
            plt.ylabel('$t_{syst}$ [mn]', fontsize = 14)
            plt.xlabel(f'Separation [{sep_unit}]', fontsize = 14)
            plt.xlim(separation[0], separation[-1])
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.minorticks_on()
            plt.title(r"$t_{syst}$"f" on {band}\nwith "r"$mag_*$("f"{band0_star})={round(mag_star, 2)}", fontsize = 14)
            plt.plot([separation[0], separation[-1]], [exposure_time, exposure_time], c="black", ls="--", label="$t_{exp}$ ="+f"{round(exposure_time)} mn")
            plt.yscale('log')
            plt.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
            plt.tight_layout()        
            
    # =======================
    # Final plot
    # ======================= 
    
    # Figure cleanup
    if show_plot:
        plt.show()
        
    IWA = np.max(iwa_bands)
        
    # ---------------------------------------
    # 5σ contrast
    # ---------------------------------------        
    if calculation == "contrast" and show_plot and band_only is None:
        plt.figure(figsize=(10, 6), dpi=300)        
        ax1 = plt.gca()
        ax1.set_yscale('log')        
        ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
        ax1.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        if coronagraph is not None:
            mask_title = f" with {coronagraph} coronagraph,"
        elif apodizer != "NO_SP":
            mask_title = f" with {apodizer} apodizer,"
        else:
            mask_title = ""
        ax1.set_title(f"{instru} {post_processing} contrast curves{for_planet_name} with $t_{{exp}}$ = {int(round(exposure_time))} mn," + mask_title + f"\n $mag_*$({band0_star}) = {round(mag_star, 1)} and $T_p$ = {int(round(T_planet))} K ({model_planet} model)", fontsize=16)        
        ax1.set_xlabel(f"Separation [{sep_unit}]", fontsize=14)
        ax1.set_ylabel(r'5$\sigma$ contrast (on instru-band)', fontsize=14)
        ax1.tick_params(axis='both', labelsize=12)        
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        ax2.axvspan(0, IWA, color='black', alpha=0.3, lw=0)
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)        
        for i in range(len(contrast_bands)):
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax1.plot(separation_bands[i][separation_bands[i] >= IWA], contrast_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax1.plot(separation_bands[i][separation_bands[i] >= IWA], contrast_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
        if separation_planet is not None:
            if separation_planet > 2 * OWA:
                ax1.set_xscale('log')
                ax1.set_xlim(IWA, max(np.max(arr) for arr in separation_bands))
            if mag_planet is None:
                ax1.axvline(separation_planet, color="black", linestyle="--", label=f"{planet_name}" if planet_name is not None else "planet")
            else:
                if planet_to_star_ratio > ax1.get_ylim()[1] or (planet_to_star_ratio > ax1.get_ylim()[0] and planet_to_star_ratio < ax1.get_ylim()[1]):
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
                ax1.plot([separation_planet, separation_planet], [planet_to_star_ratio, planet_to_star_ratio], 'ko')
                ax1.annotate(f"{planet_name}" if planet_name is not None else "planet", (x_text, y_text), fontsize=12)
        else:
            leg_loc = "upper right"
        ax3 = ax1.twinx()
        ax3.invert_yaxis()
        ax3.set_ylabel(r'$\Delta$mag', fontsize=14, labelpad=20, rotation=270)
        ax3.tick_params(axis='y', labelsize=12)        
        ymin, ymax = ax1.get_ylim()
        ax3.set_ylim(-2.5 * np.log10(ymin), -2.5 * np.log10(ymax))        
        ax1.legend(loc=leg_loc, fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        plt.minorticks_on()
        plt.tight_layout()        
        plt.show()
    
    # ---------------------------------------
    # SNR
    # ---------------------------------------     
    elif calculation == "SNR" and show_plot:
        
        if channel and instru == "MIRIMRS":
            # Each MIRIMRS channel consists of 3 sub-bands (SHORT, MEDIUM, LONG).
            # To combine them, scale exposure time accordingly.
            exposure_time    *= 3  
            n_channels        = 4
            bands_per_channel = 3
            SNR_channels        = []  # Combined SNR per channel
            separation_channels = []  # Separation grid per channel
            name_channels       = []  # Channel labels
            for ch in range(n_channels):
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
                
        plt.figure(figsize=(10, 6), dpi=300) 
        ax1 = plt.gca()
        ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)     
        ax1.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        if coronagraph is not None:
            mask_title = f" with {coronagraph} coronagraph,"
        elif apodizer != "NO_SP":
            mask_title = f" with {apodizer} apodizer,"
        else:
            mask_title = ""
        ax1.set_title(f"{instru} {post_processing} S/N curves{for_planet_name} with $t_{{exp}}$ = {int(round(exposure_time))} mn," + mask_title + f"\n $mag_*$({band0_star}) = {round(mag_star, 1)}, $mag_p$({band0_planet}) = {round(mag_planet, 1)}, $T_p$ = {int(round(T_planet))}K ({model_planet} model)", fontsize=16)
        ax1.set_xlabel(f"separation [{sep_unit}]", fontsize=14)
        ax1.set_ylabel('S/N', fontsize=14)        
        ax1.axvspan(0, IWA, color='black', alpha=0.3, lw=0)
        for i in range(len(SNR_bands)):
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax1.plot(separation_bands[i][separation_bands[i] >= IWA], SNR_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax1.plot(separation_bands[i][separation_bands[i] >= IWA], SNR_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
        if separation_planet is not None:
            if separation_planet > 2 * OWA:
                ax1.set_xscale('log')
                ax1.set_xlim(IWA, max(np.max(arr) for arr in separation_bands))
            ax1.axvline(x=separation_planet, color='k', linestyle='--', linewidth=1.5)
            ax1.plot([separation_planet], [SNR_max_planet], 'rX', ms=11)        
            ax_legend = ax1.twinx()
            ax_legend.plot([], [], '--', c='k', label=f'Angular separation{for_planet_name}')
            ax_legend.plot([], [], 'X', c='r', label=f'Max S/N{for_planet_name} ({round(SNR_max_planet, 2)})')
            ax_legend.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
            ax_legend.tick_params(axis='y', colors='w')
        ax1.set_ylim(0)
        ax1.legend(loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        ax1.minorticks_on()
        ax1.yaxis.set_ticks_position('both')
        ax1.tick_params(axis='both', labelsize=12)  
        plt.tight_layout()        
        plt.show()
        
    # ---------------------------------------
    # RETURNS
    # ---------------------------------------
    if calculation == "contrast":
        return name_bands, separation_bands, contrast_bands,      signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    elif calculation == "SNR":
        return name_bands, separation_bands, SNR_bands,           signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    elif calculation == "corner plot":
        return name_bands, separation_bands, uncertainties_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves init
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation=None, instru=None, exposure_time=None, mag_star=None, mag_planet=None, band0=None, band0_star=None, band0_planet=None, model_planet=None, T_planet=None, lg_planet=None, model_star="BT-NextGen", T_star=None, lg_star=None, rv_star=0, rv_planet=0, vsini_star=0, vsini_planet=0, 
             apodizer="NO_SP", strehl="NO_JQ", coronagraph=None, systematic=False, PCA=False, PCA_mask=False, N_PCA=20, channel=False, planet_name=None, separation_planet=None, show_plot=True, verbose=True, 
             post_processing=None, background="medium", Rc=100, filter_type="gaussian", input_DIT=None, band_only=None, 
             star_spectrum=None, planet_spectrum=None, return_SNR_planet=False, return_quantity=False):
    """
    Orchestrates inputs (spectra, models, config) and delegates to FastCurves_main
    to compute contrast/SNR/uncertainty for the selected instrument/setup.

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
    systematic, PCA, PCA_mask, N_PCA :
        Systematic noise modeling and PCA removal configuration.
    channel : bool
        For MIRIMRS, combine SNR by channel (3 sub-bands per channel).
    planet_name : str
        Label only (for plots/prints).
    separation_planet : float
        Planet separation in **arcsec** (converted internally if needed).
    post_processing : {"molecular mapping","ADI+RDI"} or None
        Defaults based on instrument type (IFU → molecular mapping, imager → ADI+RDI).
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
        FASTYIELD helper: returns per-band SNR and components at planet separation.
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
        (name_bands, separation, curves)
    """
    time1 = time.time()
    
    # ---- Validation ----
    if calculation not in {"contrast", "SNR", "corner plot"}:
        raise KeyError(f"{calculation} is not valid. Choose among: 'contrast', 'SNR', 'corner plot'.")
    if instru not in instru_name_list:
        raise KeyError(f"{instru} is not supported. Available: {instru_name_list}")

    # Only one coronagraphic mode is considered for NIRCam → set default if missing
    if instru == "NIRCam" and coronagraph is None:
        coronagraph = "MASK335R"

    # ---- Config & observing mode (space vs ground) ----
    config_data = get_config_data(instru)
    lmin_instru = config_data["lambda_range"]["lambda_min"]
    lmax_instru = config_data["lambda_range"]["lambda_max"]
    if config_data["base"] == "space":
        tellurics = False
        strehl    = "NO_JQ"
    elif config_data["base"] == "ground":
        tellurics = True 
    
    # ---- Validate optics options against config ----
    if apodizer not in config_data["apodizers"]:
        raise KeyError(f"No PSF profiles for apodizer '{apodizer}' with {instru}. Available: {config_data.get('apodizers', [])}")
    if strehl not in config_data["strehls"]:
        raise KeyError(f"No PSF profiles for strehl '{strehl}' with {instru}. Available: {config_data.get('strehls', [])}")
    if coronagraph not in config_data["coronagraphs"]:
        raise KeyError(f"No PSF profiles for coronagraph '{coronagraph}' with {instru}. Available: {config_data.get('coronagraphs', [])}")

    # Angular separation unit expected by the instrument (e.g., 'arcsec' or 'mas')
    sep_unit = config_data["sep_unit"]
        
    # ---- Default post-processing based on instrument type ----
    if post_processing is None:
        if "IFU" in config_data["type"]:
            post_processing = "molecular mapping"
        elif config_data["type"] == "imager":
            post_processing = "ADI+RDI"
    
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
    name_bands, separation, curves, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands = FastCurves_main(calculation=calculation, instru=instru, exposure_time=exposure_time, mag_star=mag_star, band0_star=band0_star, band0_planet=band0_planet, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=systematic, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, channel=channel, planet_name=planet_name, separation_planet=separation_planet, mag_planet=mag_planet, show_plot=show_plot, verbose=verbose, post_processing=post_processing, sep_unit=sep_unit, background=background, Rc=Rc, filter_type=filter_type, input_DIT=input_DIT, band_only=band_only, return_SNR_planet=return_SNR_planet)

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
            SNR_planet[nb]        = curves[nb][idx_planet]
            signal_planet[nb]     = signal_bands[nb][idx_planet]
            sigma_fund_planet[nb] = np.sqrt(sigma_fund_2_bands[nb][idx_planet])
            sigma_syst_planet[nb] = np.sqrt(sigma_syst_2_bands[nb][idx_planet])
        
        return name_bands, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, np.array(DIT_bands)
    
    # ---- Deep analysis branch ----
    if return_quantity:
        return name_bands, separation, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands

    # ---- Standard return ----
    return name_bands, separation, curves


