from src.signal_noise_estimate import *
path_file = os.path.dirname(__file__)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves Function
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves_main(calculation, instru, exposure_time, mag_star, band0, planet_spectrum, star_spectrum, tellurics, apodizer, strehl, coronagraph, systematic, PCA, PCA_mask, Nc, channel, planet_name, separation_planet, mag_planet, show_plot, verbose, post_processing, sep_unit, bkgd, Rc, filter_type, input_DIT, band_only):
    """
    See the function "FastCurves" below.
    """

    # HARD CODED: 
    cos_p              = 1     # mismatch 
    cos_est            = None  # estimated correlation in the data # 0.29 for CT Cha b on 1SHORT band of MIRIMRS
    show_cos_theta_est = False # to see the impact of the noise on the estimated correlation in order to retrieve the true mismatch
    show_t_syst        = False # to see the systematic time 
    show_contributions = True  # to see the noise contributions plots for contrast calculations
    
    #------------------------------------------------------------------------------------------------
    
    if  calculation == "corner plot" and band_only is None:
        raise KeyError("band_only NEEDS TO BE SET ON A BAND!")
    if instru == "MIRIMRS" and channel and (calculation == "SNR" or calculation == "corner plot"):
        exposure_time /= 3 # dividing the exposure time budget by 3 to observe the 3 band (SHORT, MEDIUM, LONG) in order to have a SNR per channel
    if planet_name is None:
        for_planet_name = "" # for plot purposes
    else:
        for_planet_name = " for "+planet_name

    # LOADING INSTRUMENTS SPECS
    
    config_data = get_config_data(instru)
    if len(config_data["gratings"])%2 != 0: # for plot colors purposes
        cmap = plt.get_cmap("Spectral", len(config_data["gratings"])+1)
    else:
        cmap = plt.get_cmap("Spectral", len(config_data["gratings"]))
    size_core          = config_data["size_core"] ; # aperture size on which the signal is integrated (A_FWHM = Afwhm) (in pixels)
    A_FWHM             = size_core**2 ; 
    saturation_e       = config_data["spec"]["saturation_e"] # full well capacity of the detector (in e-)
    min_DIT            = config_data["spec"]["minDIT"] # minimal integration time (in mn)
    max_DIT            = config_data["spec"]["maxDIT"] # maximal integration time (in mn)
    quantum_efficiency = config_data["spec"]["Q_eff"] # quantum efficiency (in e-/ph)
    RON                = config_data["spec"]["RON"] # read out noise (in e-/DIT)
    dark_current       = config_data["spec"]["dark_current"] # dark current (in e-/s)
    
    #------------------------------------------------------------------------------------------------
    
    contrast_bands = [] ; SNR_bands = [] ; SNR_planet_bands = [] ; name_bands = [] ; separation_bands = [] # contrast, SNR, name and separation of each band
    signal_bands = [] ; DIT_bands = [] ; sigma_s_2_bands = [] ; sigma_ns_2_bands = [] ; planet_flux_bands = [] ; star_flux_bands = [] ; wave_bands = []
    SNR_max = 0. ; SNR_max_planet = 0. # for plot and print purposes
    T_planet = planet_spectrum.T ; T_star = star_spectrum.T # planet and star temperature
    lg_planet = planet_spectrum.lg ; lg_star = star_spectrum.lg # planet and star surface gravity
    model = planet_spectrum.model ; star_model = star_spectrum.model # planet and star model
    rv_star = star_spectrum.rv ; rv_planet = planet_spectrum.rv # radial velocity of the star and the planet
    vsini_star = star_spectrum.vsini ; vsini_planet = planet_spectrum.vsini # rotationnal velocity of the star and the planet
    R_planet = planet_spectrum.R ; R_star = star_spectrum.R  ; R = max(R_planet, R_star) # spectra resolutions
    if verbose:
        print('\n Planetary spectrum ('+model+'): R =', round(round(R_planet, -3)), ', T =', round(T_planet), "K, lg =", round(lg_planet, 1))
        print(' Star spectrum ('+star_model+'): R =', round(round(R_star, -3)), ', T =', round(T_star), "K & lg =", round(lg_star, 1))
        print(' rv_star =', round(rv_star, 2), 'km/s & rv_planet =', round(rv_planet, 2), 'km/s')
        print(' vsini_star =', round(vsini_star, 2), 'km/s & vsini_planet =', round(vsini_planet, 2), 'km/s')
        if post_processing == "molecular mapping":
            print(f'\n Molecular mapping considered as post-processing method with Rc = {Rc} and {filter_type} filtering')
        elif post_processing == "ADI+RDI":
            print('\n ADI and/or RDI considered as post-processing method')
        if systematic:
            if PCA:
                print(' With systematics + PCA')
            else:
                print(' With systematics')
        else:
            print(' Without systematics')
        if strehl != "NO_JQ":
            print(f" With {strehl} strehl")
        if apodizer != "NO_SP":
            print(f" With {apodizer} apodizer")
        if coronagraph is not None:
            print(f" With {coronagraph} coronagraph")
        if tellurics:
            print(" With tellurics absorption (ground-based observation)")

    if sep_unit == "mas" and separation_planet is not None:
        separation_planet *= 1e3 # switching the angular separation unit (arcsec => mas)
    
    #------------------------------------------------------------------------------------------------

    # Restricting spectra to the instrumental range and normalizing spectra to the correct magnitude
    
    star_spectrum_instru, star_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, star_spectrum) # star spectrum in photons/min adjusted to the correct magnitude
    if calculation == "contrast":
        planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, planet_spectrum) # planet spectrum in photons/min total received
        planet_spectrum_instru.set_flux(np.nansum(star_spectrum_instru.flux)) # setting the planetary spectrum to the same flux (in total received ph/mn) as the stellar spectrum on the instrumental band of interest (by doing this, we will have a contrast in photons and not in energy, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
        planet_spectrum_density.flux *= np.nanmean(star_spectrum_density.flux)/np.nanmean(planet_spectrum_density.flux) # not really useful, only for flux densities (in energy) to have the same magnitude
    elif calculation == "SNR" or calculation == "corner plot":
        if mag_planet == None:
            raise KeyError(f"PLEASE INPUT A MAGNITUDE FOR THE PLANET FOR THE {calculation} CALCULATION !")
        else:
            planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_planet, planet_spectrum) # planet spectrum in photons/min adjusted to the correct magnitude
    wave_instru = planet_spectrum_instru.wavelength
    vega_spectrum = load_vega_spectrum() # vega spectrum in J/s/m²/µm
    vega_spectrum = vega_spectrum.interpolate_wavelength(wave_instru, renorm = False) # interpolating the vega spectrum on the instrumental wavelength axis
    mag_star_instru = -2.5*np.log10(np.nanmean(star_spectrum_density.flux)/np.nanmean(vega_spectrum.flux)) # star magnitude on the instrumental band
    mag_planet_instru = -2.5*np.log10(np.nanmean(planet_spectrum_density.flux)/np.nanmean(vega_spectrum.flux)) # planet magnitude on the instrumental band
    if show_plot and band_only is None: # plot stellar and planetary spectra on the instrument band 
        plt.figure(dpi=300) ; plt.yscale("log") ; plt.plot(planet_spectrum_instru.wavelength, planet_spectrum_instru.flux, 'g', label=f'planet, {model} with $T$={round(T_planet)}K and mag(instru)={np.round(mag_planet_instru, 1)}') ; plt.plot(star_spectrum_instru.wavelength, star_spectrum_instru.flux, 'k', label=f'star, {star_model} with $T$={T_star}K and mag(instru)={np.round(mag_star_instru, 1)}') ; plt.title(f"Star and planet spectra on the instrumental bandwitdh (R = {round(round(R, -3))})"+'\n with $rv_{*}$'+f' = {round(rv_star, 1)} km/s & '+'$rv_{p}$'+f' = {round(rv_planet, 1)} km/s', fontsize = 14) ; plt.xlabel("wavelength [µm]", fontsize = 14) ; plt.ylabel("flux [ph/mn]", fontsize = 14) ; plt.legend() ; plt.show()
    if verbose:
        print("\n"+"\033[4m" + "ON THE INSTRUMENTAL BANDWIDTH"+":"+"\033[0m")
        print(" Cp(ph) = {0:.2e}".format(np.nansum(planet_spectrum_instru.flux)/np.nansum(star_spectrum_instru.flux)), ' & \u0394'+'mag = ', round(mag_planet_instru-mag_star_instru, 2))
        print(" mag(star) = ", round(mag_star_instru, 2), " & mag(planet) = ", round(mag_planet_instru, 2), "\n"  )

    #------------------------------------------------------------------------------------------------
    # For each spectral band of the instrument under consideration:
    #------------------------------------------------------------------------------------------------
    
    if show_plot: # plotting the planet (if SNR calculation) or the star (if contrast calculation) on each band (in e-/mn)
        f1 = plt.figure(dpi=300) ; band_flux = f1.add_subplot(111) ; band_flux.set_yscale("log") ; band_flux.set_xlabel("wavelength [µm]") ;  band_flux.set_ylabel("flux [e-/mn]") ; band_flux.grid(True)
        if calculation == "SNR" or calculation == "corner plot":
            band_flux.set_title(f"Planet flux ({model}) through {instru} bands \n with $T_p$ = {int(round(planet_spectrum_instru.T))}K, $lg_p$ = {round(planet_spectrum_instru.lg,1)} and $mag_p$({band0}) = {round(mag_planet, 2)}")
        elif calculation == "contrast":
            band_flux.set_title(f"Star flux through {instru} with $mag_*$({band0}) = {round(mag_star, 2)}")
    for nb, band in enumerate(config_data['gratings']): # For each band
        if band_only is not None and band != band_only :
            continue # If you want to calculate for band_only only
        if instru == "HARMONI" and apodizer == "SP_Prox" and band != "H" and band != "H_high" and band != "J":
            continue # If you choose the apodizer for Proxima cen b, specially designed for the H-band, you ignore the other bands.
        if instru == "HARMONI" and (apodizer == "SP3" or apodizer == "SP4") and band != "K":
            continue
        if instru == "HARMONI" and strehl == "MED":
            if apodizer == "SP1" and band=="J":
                continue
            elif apodizer == "NO_SP" and band != "H" and band != "K":
                continue
        
        name_bands.append(band) # adds the band name to the list
        
        #------------------------------------------------------------------------------------------------
        
        # Degradation at instrumental resolution and restriction of the wavelength range in the considered band
        
        star_spectrum_band   = spectrum_band(config_data, band, star_spectrum_instru)
        planet_spectrum_band = spectrum_band(config_data, band, planet_spectrum_instru)
        wave_band            = planet_spectrum_band.wavelength
        R = config_data['gratings'][band].R # spectral resolution of the band
        if Rc is None:
            sigma = None # sigma width of gaussian convolution of the filtering
        else:
            sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2)
        mag_star_band   = -2.5*np.log10(np.nanmean(star_spectrum_density.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])/np.nanmean(vega_spectrum.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])) # star magnitude on the band
        mag_planet_band = -2.5*np.log10(np.nanmean(planet_spectrum_density.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])/np.nanmean(vega_spectrum.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])) # planet magnitude on the band

        #------------------------------------------------------------------------------------------------
        
        # System transmissions for the considered band
        
        trans = transmission(instru, wave_band, band, tellurics, apodizer)
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Reading PSF profiles (fraction_PSF = fraction of photons in the PSF core/FWHM)
        
        PSF_profile, fraction_PSF, separation, pxscale = PSF_profile_fraction_separation(band, strehl, apodizer, coronagraph, instru, config_data, sep_unit)
        separation_bands.append(separation) # adds the separation axis to the list
        if separation_planet is not None:
            idx = (np.abs(separation - separation_planet)).argmin()
                
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Coronagraph
        
        if coronagraph is not None and instru == "NIRCam":
            data = fits.getdata("sim_data/PSF/PSF_"+instru+"/fraction_PSF_"+band+"_"+coronagraph+"_"+strehl+"_"+apodizer+".fits") # flux fraction at the PSF peak as a function of separation 
            f = interp1d(data[0], data[1], bounds_error=False, fill_value=np.nan) ; g = interp1d(data[0], data[2], bounds_error=False, fill_value=np.nan)
            correction_transmission_ETC = 0.9 # correction factor (relative to ETC)
            fraction_PSF = f(separation) ; radial_transmission = g(separation)*correction_transmission_ETC
            star_transmission = g(0)*correction_transmission_ETC # = total stellar flux transmitted by the coronagraph (+Lyot stop) when the star is perfectly aligned with it (i.e. at 0 separation)
            PSF_profile *= star_transmission
            
        #------------------------------------------------------------------------------------------------
        
        # corrective factor R_corr (due to potential dithering and taking into account the power fraction of the noise being filtered)
        
        R_corr = np.zeros_like(separation) + 1.
        if instru == "MIRIMRS" or instru == "NIRSpec": # 4pt dithering
            sep, r_corr = fits.getdata("sim_data/R_corr/R_corr_"+instru+"/R_corr_" + band + ".fits")
            f           = interp1d(sep, r_corr, bounds_error=False, fill_value="extrapolate")
            R_corr      = f(separation)
        
        if post_processing == "molecular mapping": # power fraction of the white noise filtered
            fn_HF, _ = get_fraction_noise_filtered(wave=wave_band, R=R, Rc=Rc, filter_type=filter_type)
            R_corr  *= fn_HF
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        if post_processing == "molecular mapping": # Molecular Mapping
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Systematic noise
            
            if systematic: # calculating systematic noise profiles
                sigma_syst_prime_2, sep, m_HF, Mp, M_pca, wave, pca = systematic_profile(config_data, band, trans, Rc, R, star_spectrum_instru, planet_spectrum_instru, planet_spectrum, wave_band, size_core, filter_type, show_cos_theta_est=show_cos_theta_est, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc, mag_planet=mag_planet, band0=band0, separation_planet=separation_planet, mag_star=mag_star, target_name=planet_name, verbose=verbose)
                planet_spectrum_band.flux *= M_pca # M_pca = signal loss ratio due to the PCA (if wanted)
                f = interp1d(sep, np.sqrt(sigma_syst_prime_2), bounds_error=False, fill_value="extrapolate")
                sigma_syst_prime_2 = f(separation)**2 # systematic noise profile projected in the CCF (in e-/Flux_stell_tot/spaxel)
                mask_M = (wave_band>=wave[0]) & (wave_band<=wave[-1]) # effective wavelength axis (from data)
                planet_spectrum_band.crop(wave[0], wave[-1]) ; star_spectrum_band.crop(wave[0], wave[-1])
                trans = trans[mask_M] ; wave_band = wave_band[mask_M] ; Mp = Mp[mask_M]
                if show_cos_theta_est: # for cos theta est. : high-frequency modulations (creating systematic noise...)
                    M_HF = np.zeros((len(separation), len(wave_band)))
                    for i in range(len(separation)):
                        idx_sep = (np.abs(separation[i] - sep)).argmin()
                        M_HF[i] = m_HF[idx_sep][mask_M] 
            else:
                M_pca = 1 ; pca = None

            #------------------------------------------------------------------------------------------------------------------------------------------------
        
            # Template calculation: assuming that template = observed spectrum (cos theta p = 1)
            
            template, _ = filtered_flux(planet_spectrum_band.flux, R, Rc, filter_type) # [Sp]_HF
            template   *= trans # gamma * [Sp]_HF
            template    = template/np.sqrt(np.nansum(template**2)) # normalizing the template
            if systematic:
                planet_spectrum_band.flux *= Mp # Systematic modulations of the planetary spectrum are taken into account (mostly insignificant effect).

            #------------------------------------------------------------------------------------------------------------------------------------------------
        
            # Beta calculation (with systematic modulation, if any)
            
            if Rc is None:
                beta = 0
            else:
                beta = get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, filter_type) # self-subtraction term (in ph/mn)
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Calculation of alpha (number of useful photons/min at molecular mapping on the band under consideration) (with systematic modulations, if any)
            
            alpha = get_alpha(planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, filter_type) # molecular mapping useful signal (in ph/mn)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Printing interest quantities
        
        if verbose:
            print('\n')
            print("\033[4m" + "BAND = "+f'{band}'+f" (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm):"+"\033[0m")
            print(" Mean total system transmission =", round(np.nanmean(trans), 3))
            if calculation == "SNR" or calculation == "corner plot":
                if post_processing == "molecular mapping":
                    c = np.nansum(planet_spectrum_band.flux)/np.nansum(star_spectrum_band.flux) / M_pca
                else:
                    c = np.nansum(planet_spectrum_band.flux)/np.nansum(star_spectrum_band.flux)
                print(" Cp(ph) = {0:.2e}".format(c)  , ' => \u0394'+'mag = ', round(mag_planet_band-mag_star_band, 2), f"\n Magnitudes: mag_star = {round(mag_star_band, 2)} & mag_planet = {round(mag_planet_band, 2)}")
            if coronagraph is None:
                print(" Fraction of flux in the core of the PSF: f =", round(100*fraction_PSF, 1), "%")
            if post_processing == "molecular mapping":
                print(" Number of spectral channels:", len(wave_band), "& R =", round(R))
                if calculation == "SNR" or calculation == "corner plot":
                    print(" Useful molecular mapping signal from the planet: α =", round(np.nanmean(alpha)), "ph/mn")
                print(" Signal loss due to self-subtraction: β/α =", round(100*np.nanmean(beta)/np.nanmean(alpha), 1), "%")
                if PCA and systematic:
                    print(" Signal loss due to PCA =", round(100*(1-M_pca), 1), "%")

                            
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # DIT and RON_eff calculation
        
        DIT, RON_eff = DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, quantum_efficiency, RON, saturation_e, input_DIT, verbose)
        NDIT         = exposure_time/DIT # number of integrations
        DIT_bands.append(DIT)            # adds the DIT value of the band to the list
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Spectra through the system
        
        star_spectrum_band.flux   *= trans*DIT*quantum_efficiency # stellar spectrum through the system in the considered band (in e-/DIT)
        planet_spectrum_band.flux *= trans*DIT*quantum_efficiency # planet spectrum through the system in the considered band (in e-/DIT)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # plotting the spectrum on each band
        if show_plot:
            if post_processing == "molecular mapping": # plotting the spectrum on each band
                if calculation == "SNR" or calculation == "corner plot":
                    band_flux.plot(wave_band, planet_spectrum_band.flux/DIT, c=cmap(nb), label=band+f" (R={int(round(R))})") ; band_flux.legend()
                elif calculation == "contrast":
                    band_flux.plot(wave_band, star_spectrum_band.flux/DIT, c=cmap(nb), label=band+f" (R={int(round(R))})") ; band_flux.legend()
            elif post_processing == "ADI+RDI":
                if calculation == "SNR":
                    band_flux.plot(planet_spectrum_band.wavelength, planet_spectrum_band.flux/DIT, label=band+f" ({round(np.nansum(planet_spectrum_band.flux/DIT))} e-/mn)") ; band_flux.legend(loc='upper right')
                elif calculation == "contrast":
                    if coronagraph is not None:
                        band_flux.plot(star_spectrum_band.wavelength, star_spectrum_band.flux*star_transmission/DIT, label=band+f" ({round(np.nansum(star_transmission*star_spectrum_band.flux/DIT))} e-/mn)") ; band_flux.legend(loc='upper right')
                    else:
                        band_flux.plot(star_spectrum_band.wavelength, star_spectrum_band.flux/DIT, label=band+f" ({round(np.nansum(star_spectrum_band.flux/DIT))} e-/mn)") ; band_flux.legend(loc='upper right')
            band_flux.set_ylim(ymin=1e-3, ymax=None)
            
        #------------------------------------------------------------------------------------------------------------------------------------------------
        # Calculation of band contrast or SNR curves:
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Signal and detector noises
        
        contrast = np.zeros_like(separation) ; contrast_wo_syst = np.zeros_like(separation) ; SNR = np.zeros_like(separation) ; t_syst = np.zeros_like(separation) ; cos_theta_est = np.zeros_like(separation) ; norm_d = np.zeros_like(separation)
        
        if instru=="MIRIMRS" or instru == "ANDES": # for instruments with varying pxscales
            sep_min = min(config_data["pxscale"].values()) # mas
            if config_data["sep_unit"] == "mas":
                sep_min *= 1e3
        else:
            sep_min = config_data["apodizers"][str(apodizer)].sep
            sep_min = max(sep_min, pxscale)
                    
        if post_processing == "molecular mapping": # Molecular Mapping
            signal = (alpha*cos_p - beta)*DIT*quantum_efficiency # total number of useful e- for molecular mapping /DIT (in the FWHM or "fraction_core") (in e-/DIT/FWHM)
            if instru=="HARMONI" or apodizer != "NO_SP":
                PSF_profile[separation <= sep_min] *= 1e-4
                signal[separation <= sep_min] *= 1e-4 # Flux attenuated by a factor of 1e-4 due to Focal Plane Mask for HARMONI 
                    
        elif post_processing == "ADI+RDI": # ADI+RDI
            if coronagraph is not None:
                if calculation == "contrast":
                    signal = np.nansum(star_spectrum_band.flux)*fraction_PSF*radial_transmission ###
                elif calculation == "SNR":
                    signal = np.nansum(planet_spectrum_band.flux)*fraction_PSF*radial_transmission # planet flux in the PSF peak as a function of separation (in e-/DIT/pixel)
                    
        sigma_dc_2  = dark_current * DIT * 60 # dark current photon noise (in e-/DIT/pixel)
        sigma_ron_2 = RON_eff**2 # effective read out noise (in e-/DIT/pixel)
        
        if config_data["type"] == "IFU_fiber": # detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
            sigma_dc_2  *= config_data['pixel_detector_projection']  # adds quadratically
            sigma_ron_2 *= config_data['pixel_detector_projection'] # (in e-/DIT/spaxel)
        
        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Stellar halo: calculation of the number of stellar e-/DIT at each separation as a function of the spectral channel in the considered band.
        
        sf, psf_profile = np.meshgrid(star_spectrum_band.flux, PSF_profile)
        star_flux = psf_profile * sf # star flux normalized by the PSF profile (mean flux) for each separation
        
        if post_processing == "molecular mapping": # Molecular Mapping   
            star_flux[star_flux>saturation_e] = saturation_e # if saturation
            sigma_halo_2 = star_flux # stellar photon noise per spectral channel (in e-/DIT/pixel) for each separation
            t, _ = np.meshgrid(template, PSF_profile)
            sigma_halo_prime_2 = np.nansum(sigma_halo_2 * t**2, axis=1) # stellar photon noise projected in the CCF (in e-/DIT/spaxel)
        
        elif post_processing == "ADI+RDI": # ADI+RDI
            star_flux = np.nansum(star_flux, axis=1) # integrated/photometric stellar flux
            star_flux[star_flux>saturation_e] = saturation_e # if saturation
            sigma_halo_2 = star_flux # stellar photon noise per spectral channel (in e-/DIT/pixel) for each separation
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Background
        
        if bkgd is not None:
            bkgd_flux = fits.getdata("sim_data/Background/"+instru+"/"+bkgd+"/background_"+band+".fits") # (in e-/s/pixel)
            f = interp1d(bkgd_flux[0], bkgd_flux[1], bounds_error=False, fill_value=np.nan)
            bkgd_flux_band = f(wave_band)
            bkgd_flux_band_tot = np.nansum(bkgd_flux_band)
            if bkgd_flux_band_tot==0:
                sigma_bkgd_2 = 0.
            else:
                bkgd_flux_band *= 60 * DIT * np.nansum(bkgd_flux[1]) / bkgd_flux_band_tot # (in e-/DIT/pixel) + we have to renormalize because we interpolate (flux conservation)
                sigma_bkgd_2    = bkgd_flux_band                                          # background photon noise per spectral channel (in e-/DIT/pixel) for each separation
        else:
            sigma_bkgd_2 = 0.
        if post_processing == "molecular mapping":
            sigma_bkgd_prime_2 = np.nansum(sigma_bkgd_2 * template**2) # background photon noise projected in the CCF (in e-/DIT/spaxel)
        elif post_processing == "ADI+RDI": # ADI+RDI
            if coronagraph is not None:
                sigma_bkgd_2 = np.nansum(sigma_bkgd_2)*radial_transmission # background photon noise per spectral channel (in e-/DIT/pixel) for each separation  (radial_transmission due to the coronagraph)
            else:
                sigma_bkgd_2 = np.nansum(sigma_bkgd_2) # background photon noise per spectral channel (in e-/DIT/pixel) for each separation  (radial_transmission due to the coronagraph)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Systematic
        
        if systematic:
            sigma_syst_prime_2 *= np.nansum(star_spectrum_band.flux)**2 # systematic noise projected in the CCF (in e-/DIT/spaxel)
            
        #------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Saving interest quantities (in e-/DIT/FWHM)
        
        signal_bands.append(signal) ; planet_flux_bands.append(planet_spectrum_band.flux) ; star_flux_bands.append(star_spectrum_band.flux) ; wave_bands.append(wave_band)
        if post_processing == "molecular mapping":
            sigma_ns_2_bands.append(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))
            if systematic:
                sigma_s_2_bands.append(sigma_syst_prime_2)
            else:
                sigma_s_2_bands.append(0) 
        elif post_processing == "ADI+RDI": # ADI+RDI
            sigma_ns_2_bands.append(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))
            if systematic:
                sigma_s_2_bands.append(sigma_syst_2)
            else:
                sigma_s_2_bands.append(0) 
                
        #------------------------------------------------------------------------------------------------------------------------------------------------
        # Contrast calculation

        if calculation == "contrast":
            
            if post_processing == "molecular mapping": # See Eq. (11) of Martos et al. 2024
                if systematic:
                    contrast         = 5*np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2 + sigma_syst_prime_2*NDIT/(R_corr*A_FWHM)))/(signal*np.sqrt(NDIT))
                    contrast_wo_syst = 5*np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))/(signal*np.sqrt(NDIT))
                else:
                    contrast = 5*np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))/(signal*np.sqrt(NDIT))

            elif post_processing == "ADI+RDI":
                if systematic:
                    contrast         = 5*np.sqrt(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2 + sigma_syst_2*NDIT/(R_corr*A_FWHM)))/(signal*np.sqrt(NDIT))
                    contrast_wo_syst = 5*np.sqrt(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))/(signal*np.sqrt(NDIT)) 
                else:
                    contrast = 5*np.sqrt(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))/(signal*np.sqrt(NDIT))

            if show_plot and show_contributions: # NOISE CONTRIBUTIONS PLOTS
                if post_processing == "molecular mapping": # Molecular Mapping
                    plt.figure(dpi=300) ; plt.xlabel(f'separation [{sep_unit}]', fontsize = 14) ; plt.ylabel(r'contrast 5$\sigma_{CCF}$ / $\alpha_0$', fontsize = 14) ; plt.title(instru+f" noise contributions on {band}"+for_planet_name+"\n with "+"$t_{exp}$=" + str(round(exposure_time)) + "mn, $mag_*$("+band0+")=" + str(round(mag_star, 2)) + f', $T_p$={int(round(T_planet))}K and $R_c$ = {Rc}', fontsize = 14)
                    plt.plot(separation[separation>sep_min], contrast[separation>sep_min], 'k-', label="$\sigma_{CCF}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'r--', label="$\sigma'_{halo}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_ron_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'g--', label="$\sigma_{ron}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_dc_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'm--', label="$\sigma_{dc}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_bkgd_prime_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'b--', label="$\sigma'_{bkgd}$")
                    if systematic:
                        plt.plot(separation[separation>sep_min], (5*np.sqrt(sigma_syst_prime_2)/(signal))[separation>sep_min], 'c--', label="$\sigma'_{syst}$")
                    plt.xlim(0, separation[-1]) ; plt.legend(fontsize=12) ; plt.yscale('log') ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
                elif post_processing == "ADI+RDI": # RDI+ADI+RDI
                    plt.figure(dpi=300) ; plt.xlabel(f'separation [{sep_unit}]', fontsize = 14) ; plt.ylabel(r'contrast 5$\sigma$ / $F_{max}$', fontsize = 14) ; plt.title(instru+f" noise contributions on {band}"+for_planet_name+" \n with "+"$t_{exp}$=" + str(round(exposure_time)) + "mn and $mag_*$("+band0+")=" + str(round(mag_star, 2)), fontsize = 14)
                    plt.plot(separation[separation>sep_min], contrast[separation>sep_min], 'k-', label="$\sigma_{tot}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_halo_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'r--', label="$\sigma_{halo}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_ron_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'g--', label="$\sigma_{ron}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_dc_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'm--', label="$\sigma_{dc}$")
                    plt.plot(separation[separation>sep_min], (5*np.sqrt(R_corr*A_FWHM*(sigma_bkgd_2)*(NDIT))/(signal*NDIT))[separation>sep_min], 'b--', label="$\sigma_{bkgd}$")
                    plt.xlim(0, separation[-1]) ; plt.legend(fontsize=12) ; plt.yscale('log') ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
            contrast_bands.append(contrast) # adds the contrast curve of the band to the list
        
        # SNR calculation
        
        elif calculation == "SNR" or calculation == "corner plot":
            
            if post_processing == "molecular mapping": # See Eq. (10) of Martos et al. 2024
                if systematic:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2 + sigma_syst_prime_2*NDIT/(R_corr*A_FWHM))) # avec systématiques
                else:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2)) # sans systématiques
            
            elif post_processing == "ADI+RDI":
                if systematic:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2 + sigma_syst_2*NDIT/(R_corr*A_FWHM))) # avec systématiques
                else:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) # sans systématiques

            if separation_planet is not None: # SNR values at the planet's separation
                SNR_planet_bands.append(SNR[idx])
                if verbose:
                    print(f' S/N at {round(separation_planet, 1)} {sep_unit} = ', round(SNR[idx], 2))
                if np.max(SNR) > SNR_max: 
                    SNR_max = np.max(SNR) 
                if  SNR[idx] > SNR_max_planet:
                    SNR_max_planet = SNR[idx] ; name_max_SNR = band
            
            # Effect of noise on correlation estimation
            
            if show_cos_theta_est and post_processing == "molecular mapping": # calculation of cos theta est (impact of noise+systematics (and auto-subtraction) on correlation estimation)
                Mp_Sp              = NDIT*planet_spectrum_band.flux*fraction_PSF # planet flux (with modulations, if any) in e-/FWHM
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp/trans, R, Rc, filter_type) # filtered planet flux
                star_HF, star_LF   = filtered_flux(star_spectrum_band.flux/trans, R, Rc, filter_type) # filtered star flux
                alpha              = np.sqrt(np.nansum((trans*Mp_Sp_HF)**2)) # true effective signal
                beta               = np.nansum(trans*star_HF*Mp_Sp_LF/star_LF * template) # self subtraction
                cos_theta_lim      = np.nansum( trans*Mp_Sp_HF * template ) / alpha # loss of correlation due to systematics
                for i in range(len(separation)): # for each separation
                        star_flux = PSF_profile[i]*star_spectrum_band.flux # stellar flux in e-/DIT/pixel at separation i
                        star_flux[star_flux>saturation_e] = saturation_e # if saturation
                        sigma_tot = np.sqrt(R_corr[i]*A_FWHM*NDIT*(sigma_halo_2[i] + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) # total noise per spectral channel (in e-/Afwhm/bin)
                        N = 1000 ; cos_theta_est_n = np.zeros((N)) ; norm_d_n = np.zeros((N))
                        for n in range(len(cos_theta_est_n)): # N noise simulations
                            noise              = np.random.normal(0, sigma_tot, len(wave_band)) # noise realisation
                            d                  = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF + noise + M_HF[i]*NDIT*A_FWHM*star_flux # spectrum at the planet's location: see Eq.(18) of Martos et al. 2024
                            norm_d_n[n]        = np.sqrt(np.nansum(d**2))
                            cos_theta_est_n[n] = np.nansum( d * template ) / norm_d_n[n]
                        norm_d[i]        = np.nanmean(norm_d_n)
                        cos_theta_est[i] = np.nanmean(cos_theta_est_n)
                        if separation_planet is not None and i == idx:
                            print(" S/N PER SPECTRAL CHANNEL = ", round(np.nanmean(Mp_Sp/sigma_tot), 3))
                cos_theta_n = alpha/norm_d # loss of correlation due to fundamental noises
                plt.figure(dpi=300) ; plt.plot(separation, cos_theta_est, 'k') ; plt.ylabel(r"cos $\theta_{est}$", fontsize=14) ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.title(f"Effect of noise and stellar substraction on the estimation of correlation between \n template and planetary spectrum for {instru} on {band} \n (assuming that the template is the same as the observed spectrum)")
                plt.xlabel(f'separation [{sep_unit}]', fontsize=14)
                if separation_planet is not None and separation_planet < np.nanmax(separation):
                    print(" beta/alpha = ", round(beta/alpha, 3), "\n cos_theta_n = ", round(cos_theta_n[idx], 3), "\n cos_theta_lim = ", round(cos_theta_lim, 3))
                    if cos_est is not None:
                        cos_theta_p = (cos_est/cos_theta_n[idx] + beta/alpha)/cos_theta_lim                   
                        print(" cos_theta_est = ", round(cos_est, 3), " => cos_theta_p = ", round(cos_theta_p, 3))
                    plt.plot([separation_planet, separation_planet], [np.nanmin(cos_theta_est), np.nanmax(cos_theta_est)], 'k--', label=f'angular separation{for_planet_name}')
                    plt.plot([separation_planet, separation_planet], [cos_theta_est[idx], cos_theta_est[idx]], 'rX', ms=11, label=r"cos $\theta_{est}$"+for_planet_name+f' ({round(cos_theta_est[idx], 2)})')
                    plt.legend()
                plt.show()

            SNR_bands.append(SNR) # adds the SNR curve of the band to the list
            
        if show_plot and show_t_syst and systematic: # plot of t_syst : see Eq.(14) of Martos et al. 2024
            t_syst = DIT*R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2)/sigma_syst_prime_2 # en min 
            plt.figure(dpi=300) ; plt.plot(separation, t_syst, 'b') ; plt.ylabel('$t_{syst}$ [mn]', fontsize = 14) ; plt.xlabel(f'separation [{sep_unit}]', fontsize = 14) ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.title("$t_{syst}$"+f" on {band}"+"\n with $mag_*$("+band0+")=" + str(round(mag_star, 2)), fontsize = 14) ; plt.plot([separation[0], separation[-1]], [exposure_time, exposure_time], 'r-') ; plt.yscale('log') ; plt.legend(["$t_{syst}$", "$t_{exp}$ ="+f"{round(exposure_time)} mn"]) 

    #------------------------------------------------------------------------------------------------
    # PLOTS:
    #------------------------------------------------------------------------------------------------
    
    # Contrast:
        
    if calculation == "contrast" and show_plot:
        plt.figure(dpi=300) ; ax1 = plt.gca() ; ax1.grid(True) ; ax1.set_yscale('log') ; ax1.axvspan(0, sep_min, color='k', alpha=0.5, lw=0) ; ax1.set_xlim(0, separation[-1])
        ax1.set_title(instru + " contrast curves"+for_planet_name+f" with {post_processing}"+" \n with $t_{exp}$=" + str(round(exposure_time, 1)) + "mn, $mag_*$("+band0+")=" + str(round(mag_star, 1)) + f' and $T_p$={int(round(T_planet))}K', fontsize = 14) ; ax1.set_xlabel(f"separation [{sep_unit}]", fontsize=12) ; ax1.set_ylabel(r'5$\sigma$ contrast (on instru-band)', fontsize=12)
        for i in range(len(contrast_bands)):
            if band_only is not None:
                ax1.plot(separation_bands[i][separation_bands[i]>sep_min], contrast_bands[i][separation_bands[i]>sep_min], label=name_bands[i], color=cmap([nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]))
            else:
                ax1.plot(separation_bands[i][separation_bands[i]>sep_min], contrast_bands[i][separation_bands[i]>sep_min], label=name_bands[i], color=cmap(i))
        if separation_planet is not None  and separation_planet < np.nanmax(separation):
            if mag_planet is None:
                ax1.axvline(separation_planet, c="k", linestyle="--", label=f'{planet_name}')
            else:
                star_spectrum_instru, star_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, star_spectrum) # spectre de l'étoile en photons/min ajusté à la bonne magnitude
                planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_planet, planet_spectrum) # spectre de la planète en ph/min ajusté à la bonne magnitude
                flux_ratio = np.nanmean(planet_spectrum_instru.flux/star_spectrum_instru.flux)
                ax1.plot([separation_planet, separation_planet], [flux_ratio, flux_ratio], 'ko')
                ax1.annotate(f"{planet_name}", (separation_planet+0.025*separation[-1], flux_ratio))
        ax2 = ax1.twinx() ; ax2.invert_yaxis() ; ax2.set_ylabel(r'$\Delta$mag', fontsize=12, labelpad=20, rotation=270) ; ax2.tick_params(axis='y')
        ymin, ymax = ax1.get_ylim() ; ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))
        ax1.legend(loc="upper right") ; ax1.set_zorder(1) ; plt.show()
    
    #------------------------------------------------------------------------------------------------

    # SNR:
    
    elif calculation == "SNR" and show_plot:
        if separation_planet is not None  and verbose:
            print('\n') ; print(f' MAX S/N (at {round(separation_planet, 1)} {sep_unit}) = ', round(SNR_max_planet, 1), "for "+name_max_SNR)
        if channel and instru == 'MIRIMRS':
            exposure_time *= 3
            SNR_chan=[] ; separation_chan=[] ; separation_chan.append(separation_bands[0]) ; separation_chan.append(separation_bands[3]) ; SNR_chan1=np.zeros(len(SNR_bands[0])) ; SNR_chan2=np.zeros(len(SNR_bands[3])) ; name_bands[0]="channel 1" ; name_bands[1]="channel 2" ; SNR_max_planet=0.
            SNR_chan.append(np.sqrt(np.nansum(np.array(SNR_bands[:3])**2, 0))) ; SNR_chan.append(np.sqrt(np.nansum(np.array(SNR_bands[3:])**2, 0))) ; SNR_max=max(max(SNR_chan[0]), max(SNR_chan[1]))
            if separation_planet is not None:
                for i in range(len(SNR_chan)):
                    idx = (np.abs(separation_chan[i] - separation_planet)).argmin()
                    if  SNR_chan[i][idx] > SNR_max_planet:
                        SNR_max_planetb= SNR_chan[i][idx] ; name_max_SNR = name_bands[i]
                print(f'MAX S/N (at {separation_planet}") = ', round(SNR_max_planet, 2), "for "+name_max_SNR)
            separation_bands=separation_chan ; SNR_bands=SNR_chan
        plt.figure(dpi=300) ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on()
        plt.title(instru + " S/N curves"+for_planet_name+" with $t_{exp}$=" + str(round(exposure_time)) + "mn, \n $mag_*$("+band0+")=" + str(round(mag_star, 2)) + ", $mag_p$("+band0+")=" + str(round(mag_planet, 2))+f' and $T_p$ = {int(round(T_planet))}K ('+model+')', fontsize = 14)
        for i in range(len(SNR_bands)):
            if band_only is not None:
                plt.plot(separation_bands[i][separation_bands[i]>sep_min], SNR_bands[i][separation_bands[i]>sep_min], label=name_bands[i], color=cmap([nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]))
            else:
                plt.plot(separation_bands[i][separation_bands[i]>sep_min], SNR_bands[i][separation_bands[i]>sep_min], label=name_bands[i], color=cmap(i))
        plt.legend(loc='upper left') ; plt.xlim(0, separation[-1]) ; plt.ylim(0) ; plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0) ; plt.xlabel(f"separation [{sep_unit}]", fontsize=12) ; plt.ylabel('S/N', fontsize=14) ; ax = plt.gca() ; ax.yaxis.set_ticks_position('both') ; plt.tight_layout()
        if separation_planet is not None  and separation_planet < np.nanmax(separation):
            plt.plot([separation_planet, separation_planet], [0., SNR_max], 'k--') ; plt.plot([separation_planet, separation_planet], [SNR_max_planet, SNR_max_planet], 'rX', ms=11)
            ax_legend = ax.twinx() ; ax_legend.plot([], [], '--', c='k', label=f'angular separation{for_planet_name}'); ax_legend.plot([], [], 'X', c='r', label='max S/N'+for_planet_name+' ('+str(round(SNR_max_planet, 2))+')') ; ax_legend.legend(loc='lower right') ; ax_legend.tick_params(axis='y', colors='w')
        plt.show()
    
    #------------------------------------------------------------------------------------------------

    # Corner plot:
        
    if calculation == "corner plot":
        Mp_Sp              = NDIT*planet_spectrum_band.flux*fraction_PSF        # planet flux (with modulations, if any) in e-/FWHM
        Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp/trans, R, Rc, filter_type)     # filtered planet flux
        star_flux          = PSF_profile[idx]*star_spectrum_band.flux*NDIT      # stellar flux in e-/pixel at separation idx (of the planet)
        star_HF, star_LF   = filtered_flux(star_flux/trans, R, Rc, filter_type) # filtered star flux
        d_planet           = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF    # spectrum at the planet's location: see Eq.(18) of Martos et al. 2024        
        sigma_l            = np.sqrt(R_corr[idx]*A_FWHM*NDIT*(sigma_halo_2[idx] + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) # total noise per spectral channel (in e-/Afwhm/spectral canal)
        vsini_planet       = planet_spectrum.vsini
        rv_planet          = planet_spectrum.rv
        SNR_CCF            = SNR[idx]
        parameters_estimation(instru=instru, band=band_only, target_name=planet_name, wave=wave_band, d_planet=d_planet, star_flux=star_flux, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=model, logL=True, method_logL="classic", sigma_l=sigma_l, weight=None, pca=pca, precise_estimate=False, SNR_estimate=False, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, T_arr=None, lg_arr=None, vsini_arr=None, rv_arr=None, SNR_CCF=SNR_CCF, show=False, verbose=True, stellar_component=True, degrade_resolution=True, force_new_est=False, d_planet_sim=False, save=False, fastcurves=True, exposure_time=exposure_time, star_HF=star_HF, star_LF=star_LF, wave_inter=wave_instru)
        
        
        
    #------------------------------------------------------------------------------------------------
    # RETURNS:
    #------------------------------------------------------------------------------------------------
    
    if calculation == "contrast":
        return name_bands, separation_bands, contrast_bands, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    elif calculation == "SNR" or calculation == "corner plot":
        return name_bands, separation_bands, SNR_bands, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves init
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation="contrast", instru="HARMONI", exposure_time=120, mag_star=7, mag_planet=None, band0="K", model="BT-Settl", T_planet=1000, lg_planet=4.0, T_star=6000, lg_star=4.0, rv_star=0, rv_planet=0, vsini_star=0, vsini_planet=0, 
             apodizer="NO_SP", strehl="JQ1", coronagraph=None, systematic=False, PCA=False, PCA_mask=False, Nc=20, channel=False, planet_name=None, separation_planet=None, show_plot=True, verbose=True, 
             post_processing=None, bkgd="medium", Rc=100, filter_type="gaussian", input_DIT=None, band_only=None, 
             star_spectrum=None, planet_spectrum=None, return_SNR_planet=False, return_quantity=False):
    """
    Function for calculating contrast or SNR curves, depending on the instrument and of planet/star selected 

    Parameters
    ----------
    calculation (str)        : calculation between "SNR" and "contrast". Default: "contrast".
    instru (str)             : instrument selected. Default: "HARMONI".
    exposure_time (float)    : exposure time selected (in mn). Default: 120.
    mag_star (float)         : star's magnitude. Default: 7.
    mag_planet (float)       : planet's magnitude (necessary for SNR calculation). Default: None.
    band0 (str)              : band where the magnitudes are given. Default: "K".
    model (str)              : planet's model. Default: "BT-Settl".
    T_planet (float)         : planet's temperature (in K). Default: 1000.
    lg_planet (float)        : planet's gravity surface (in dex(cm/s2)). Default: 4.0.
    T_star (float)           : star's temperature (in K). Default: 6000.
    lg_star (float)          : star's gravity surface (in dex(cm/s2)). Default: 4.0.
    rv_star (float)          : star's radial velocity [km/s]. Default: 0.
    rv_planet (float)        : planet's radial velocity [km/s]. Default: 0.
    vsini_star (float)       : star's rotation speed [km/s]. Default: 0.
    vsini_planet (float)     : planet's rotation speed [km/s]. Default: 0.
    apodizer (str)           : apodizer selected, if any. Default: "NO_SP" ("NO_SP" means NO Shaped Pupil => no apodizer).
    strehl (str)             : strehl selected, if ground-based observation. Default: "JQ1". ("NO_JQ" means NO J.. Quartile => no strehl for space-based observation).
    coronagraph (str)        : coronagraph selected, if any. Default: None.
    systematic (bool)        : to take systematic noise (other than speckles) into account (True or False) if it can be estimated (only for MIRIMRS and NIRSpec for now). Default: False.
    PCA (bool)               : to use PCA as systematic removal
    PCA_mask (bool)          : to consider a mask on the planet while estimating the components of the PCA
    Nc (int)                 : Number of PCA components subtracted. Default: 20. (If Nc = 0, there will be no PCA)
    channel (bool)           : for MIRI/MRS, SNR curves can be combined by channel (not by band) (True or False). Default: False.
    planet_name (str)        : planet's name (for plot purposes only). Default: None.
    separation_planet (float): planet's separation (in arcsec), to find the planet's SNR or contrast. Default: None.
    post_processing (str)    : post-processing method ("molecular mapping" or "ADI+RDI"). Default: None.
    bkgd (str)               : background level (None, "low", "medium", "high"). Default: "medium".
    Rc (float)               : cut-off resolution for molecular mapping post-processing. Default: 100. (If Rc = None, there will be no filtering)
    filter_type (str)        : type of filter used ("gaussian", "step" or "smoothstep"). Default: "gaussian".
    """
    time1 = time.time() ; warnings.filterwarnings('ignore', category=UserWarning, append=True)
    
    # checking if the instrument is considered in FastCurves
    if instru not in instru_name_list:
        raise KeyError(f"{instru} is not yet considered in FastCurves. Available instruments: {instru_name_list}")
    
    # only one coronographic mask is yet considered in FastCurves for NIRCam
    if instru == "NIRCam" and coronagraph is None:
        coronagraph = "MASK335R"
        
    # getting the instruments specs
    config_data = get_config_data(instru)
    
    # Space- or Ground-based observations
    if config_data["base"] == "space": # space-based observations => no tellurics and no strehl
        tellurics = False ; strehl = "NO_JQ"
    elif config_data["base"] == "ground": # space-based observations => tellurics and strehl
        tellurics = True 
        
    # checking if the apodizer is considered for the instrument
    if apodizer not in config_data["apodizers"]:
        raise KeyError(f"No PSF profiles for {apodizer} strehl with {instru}. Available apodizers: {config_data['apodizers']}")
    
    # checking if the strehl is considered for the instrument
    if strehl not in config_data["strehls"]:
        raise KeyError(f"No PSF profiles for {strehl} strehl with {instru}. Available strehl values: {config_data['strehls']}")
    
    # angular separation unit (arcsec or mas)
    sep_unit = config_data["sep_unit"]
    
    # post-processing method considered
    if post_processing is None:
        if "IFU" in config_data["type"]:
            post_processing = "molecular mapping"
        elif config_data["type"] == "imager":
            post_processing = "ADI+RDI"
    
    # if not input, load the star spectrum
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star) # star spectrum (BT-NextGen GNS93)
        try:
            star_spectrum.crop(0.98*min(globals()["lmin_"+band0], globals()["lmin_"+instru]), 1.02*max(globals()["lmax_"+band0], globals()["lmax_"+instru]))
        except:
            raise KeyError(f"{band0} is not a considered band to define the magnitude, please choose among: {bands}, {instrus}")
        # Rotational broadening of the spectrum [km/s]
        if vsini_star > 0: # the wavelength axis needs to be evenly spaced
            dl_star       = star_spectrum.wavelength - np.roll(star_spectrum.wavelength, 1) ; dl_star[0] = dl_star[1] ; dl_star[dl_star == 0] = np.nanmean(dl_star) # delta lambda array
            R_star        = np.nanmax(star_spectrum.wavelength/(2*dl_star)) # interpolation Resolution (need to be the max res to avoid nan with cg.downvin)
            if R_star > 300_000: # fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
                R_star    = 300_000
            dl_star       = ((star_spectrum.wavelength[0]+star_spectrum.wavelength[-1])/2)/(2*R_star)
            wave_star     = np.arange(star_spectrum.wavelength[0], star_spectrum.wavelength[-1], dl_star) # constant and linear wavelength array on the considered band
            star_spectrum = star_spectrum.interpolate_wavelength(wave_star, renorm=False)
        star_spectrum = star_spectrum.broad(vsini_star)
        # Doppler shifting the spectrum [km/s]
        star_spectrum = star_spectrum.doppler_shift(rv_star)
            
    # if not input, load the planet spectrum
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru) # planet spectrum: class Spectrum(wavel, flux, R, T) in J/s/m²/µm according to the considered model
        try:
            planet_spectrum.crop(0.98*min(globals()["lmin_"+band0], globals()["lmin_"+instru]), 1.02*max(globals()["lmax_"+band0], globals()["lmax_"+instru]))
        except:
            raise KeyError(f"{band0} is not a considered band to define the magnitude, please choose among: {bands}, {instrus}")
        # Rotational broadening of the spectrum [km/s]
        if vsini_planet > 0: # the wavelength axis needs to be evenly spaced
            dl_planet       = planet_spectrum.wavelength - np.roll(planet_spectrum.wavelength, 1) ; dl_planet[0] = dl_planet[1] ; dl_planet[dl_planet == 0] = np.nanmean(dl_planet) # delta lambda array
            R_planet        = np.nanmax(planet_spectrum.wavelength/(2*dl_planet)) # interpolation Resolution (need to be the max res to avoid nan with cg.downvin)
            if R_planet > 300_000: # fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
                R_planet    = 300_000
            dl_planet       = ((planet_spectrum.wavelength[0]+planet_spectrum.wavelength[-1])/2)/(2*R_planet)
            wave_planet     = np.arange(planet_spectrum.wavelength[0], planet_spectrum.wavelength[-1], dl_planet) # constant and linear wavelength array on the considered band
            planet_spectrum = planet_spectrum.interpolate_wavelength(wave_planet, renorm=False)
        planet_spectrum = planet_spectrum.broad(vsini_planet)
        # Doppler shifting the spectrum [km/s]
        planet_spectrum = planet_spectrum.doppler_shift(rv_planet)

    # FastCurves calculation
    name_bands, separation, curves, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands = FastCurves_main(calculation=calculation, instru=instru, exposure_time=exposure_time, mag_star=mag_star, band0=band0, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, tellurics=tellurics, 
                                                                                                                                                                 apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=systematic, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc, channel=channel, planet_name=planet_name, separation_planet=separation_planet, mag_planet=mag_planet, show_plot=show_plot, verbose=verbose, 
                                                                                                                                                                 post_processing=post_processing, sep_unit=sep_unit, bkgd=bkgd, Rc=Rc, filter_type=filter_type, input_DIT=input_DIT, band_only=band_only)
    
    if verbose:
        print(f'\n FastCurves {calculation} calculation took {round(time.time()-time1, 1)} s')

    # For FASTYIELD
    if return_SNR_planet:
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1e3 # switching the angular separation unit
        SNR_planet = np.zeros((len(name_bands))) ; signal_planet = np.zeros((len(name_bands))) ; sigma_ns_planet = np.zeros((len(name_bands))) ; sigma_s_planet = np.zeros((len(name_bands)))
        for nb, band in enumerate(name_bands): # retrieving the values at the planet separation
            idx = np.abs(separation[nb]-separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_bands[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_bands[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_bands[nb][idx])
        return name_bands, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_bands)
    
    # For deep analysis
    elif return_quantity:
        return name_bands, separation, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    
    # Standard returns
    else:
        return name_bands, separation, curves






