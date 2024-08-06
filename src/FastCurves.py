from src.spectrum import *
path_file = os.path.dirname(__file__)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves Function
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves_main(calculation, instru, exposure_time, mag_star, band0, planet_spectrum, star_spectrum, tellurics, apodizer, strehl, coronagraph, systematic, PCA, PCA_mask, Nc, channel, name_planet, separation_planet, mag_planet, show_plot, print_value, post_processing, sep_unit, bkgd, Rc, used_filter, input_DIT, band_only):
    """
    See the function "FastCurves" below.
    """
        
    # HARD CODED: 
        
    cos_p = 1 # mismatch 
    cos_est = None # estimated correlation in the data # 0.28 for CT Cha b on 1SHORT band of MIRIMRS
    show_cos_theta_est = False # to see the impact of the noise on the estimated correlation in order to retrieve the true mismatch
    show_t_syst = False # to see the systematic time 
    show_contributions = True # to see the noise contributions plots for contrast calculations
    
    #------------------------------------------------------------------------------------------------
    
    if instru == "MIRIMRS" and channel and calculation == "SNR":
        exposure_time /= 3 # dividing the exposure time budget by 3 to observe the 3 band (SHORT, MEDIUM, LONG) in order to have SNR per channel
    if name_planet is None:
        for_name_planet = "" # for plot purposes
    else:
        for_name_planet = " for "+name_planet

    # LOADING INSTRUMENTS SPECS
    
    config_data = get_config_data(instru)
    if len(config_data["gratings"])%2 != 0: # for plot colors purposes
        cmap = plt.get_cmap("Spectral", len(config_data["gratings"])+1)
    else:
        cmap = plt.get_cmap("Spectral", len(config_data["gratings"]))
    size_core = config_data["size_core"] ; # aperture size on which the signal is integrated (size_core**2 = Afwhm) (in pixels)
    saturation_e = config_data["spec"]["saturation_e"] # full well capacity of the detector (in e-)
    min_DIT = config_data["spec"]["minDIT"] # minimal integration time (in mn)
    max_DIT = config_data["spec"]["maxDIT"] # maximal integration time (in mn)
    quantum_efficiency = config_data["spec"]["Q_eff"] # quantum efficiency (in e-/ph)
    RON = config_data["spec"]["RON"] # read out noise (in e-/DIT)
    dark_current = config_data["spec"]["dark_current"] # dark current (in e-/s)

    #------------------------------------------------------------------------------------------------
    
    contrast_bands = [] ; SNR_bands = [] ; SNR_planet_bands = [] ; name_bands = [] ; separation_bands = [] # contrast, SNR, name and separation of each band
    signal_bands = [] ; DIT_bands = [] ; sigma_s_2_bands = [] ; sigma_ns_2_bands = [] ; planet_flux_bands = [] ; star_flux_bands = [] ; wave_bands = []
    SNR_max = 0. ; SNR_max_planet = 0. # for plot and print purposes
    T_planet = planet_spectrum.T ; T_star = star_spectrum.T # planet and star temperature
    lg_planet = planet_spectrum.lg ; lg_star = star_spectrum.lg # planet and star surface gravity
    model = planet_spectrum.model ; star_model = star_spectrum.model # planet and star model
    star_rv = star_spectrum.star_rv ; delta_rv = planet_spectrum.delta_rv # radial velocity of the star and shift between planet and the star
    R_planet = planet_spectrum.R ; R_star = star_spectrum.R  ; R = max(R_planet, R_star) # spectra resolutions
    if print_value:
        print('\n Planetary spectrum ('+model+'): R =', round(round(R_planet, -3)), ', T =', round(T_planet), "K, lg =", round(lg_planet, 1))
        print(' Star spectrum ('+star_model+'): R =', round(round(R_star, -3)), ', T =', round(T_star), "K & lg =", round(lg_star, 1))
        print(' star_rv = ', round(star_rv, 2), ' km/s & Δrv = ', round(delta_rv, 2), ' km/s')
        if post_processing == "molecular mapping":
            print(f'\n Molecular mapping considered as post-processing method with Rc = {Rc} and {used_filter} filtering')
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

    if sep_unit == "mas":
        if separation_planet is not None:
            separation_planet *= 1e3 # switching the angular separation unit (arcsec => mas)
    
    #------------------------------------------------------------------------------------------------

    # Restricting spectra to the instrumental range and normalizing spectra to the correct magnitude
    
    star_spectrum_instru, star_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, star_spectrum) # star spectrum in photons/min adjusted to the correct magnitude
    if calculation == "contrast":
        planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, planet_spectrum) # planet spectrum in photons/min total received
        planet_spectrum_instru.set_flux(np.nansum(star_spectrum_instru.flux)) # setting the planetary spectrum to the same flux (in total received ph/mn) as the stellar spectrum on the instrumental band of interest (by doing this, we will have a contrast in photons and not in energy, otherwise we would have had to set it to the same received energy).
        planet_spectrum_density.flux *= np.nanmean(star_spectrum_density.flux)/np.nanmean(planet_spectrum_density.flux) # not really useful, only for flux densities (in energy) to have the same magnitude
    elif calculation == "SNR":
        if mag_planet == None:
            raise KeyError("PLEASE INPUT A MAGNITUDE FOR THE PLANET FOR THE SNR CALCULATION !")
        else:
            planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_planet, planet_spectrum) # planet spectrum in photons/min adjusted to the correct magnitude
    wave_instru = planet_spectrum_instru.wavelength
    vega_spectrum = load_vega_spectrum() # vega spectrum in J/s/m²/µm
    vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave_instru, renorm = False) # interpolating the vega spectrum on the instrumental wavelength axis
    mag_star_instru = -2.5*np.log10(np.nanmean(star_spectrum_density.flux)/np.nanmean(vega_spectrum.flux)) # star magnitude on the instrumental band
    mag_planet_instru = -2.5*np.log10(np.nanmean(planet_spectrum_density.flux)/np.nanmean(vega_spectrum.flux)) # planet magnitude on the instrumental band
    if show_plot and band_only is None: # plot stellar and planetary spectra on the instrument band 
        plt.figure(dpi=300) ; plt.yscale("log") ; plt.plot(planet_spectrum_instru.wavelength, planet_spectrum_instru.flux, 'g', label=f'planet, {model} with $T$={T_planet}K and mag(instru)={np.round(mag_planet_instru, 1)}') ; plt.plot(star_spectrum_instru.wavelength, star_spectrum_instru.flux, 'k', label=f'star, {star_model} with $T$={T_star}K and mag(instru)={np.round(mag_star_instru, 1)}') ; plt.title(f"Star and planet spectra on the instrumental bandwitdh (R = {round(round(R, -3))})"+'\n with $rv_{syst}$'+f' = {round(star_rv, 1)} km/s & Δrv = {round(delta_rv, 1)} km/s', fontsize = 14) ; plt.xlabel("wavelength (in µm)", fontsize = 14) ; plt.ylabel("flux (in ph/mn)", fontsize = 14) ; plt.legend() ; plt.show()
    if print_value:
        print("\n"+"\033[4m" + "ON THE INSTRUMENTAL BANDWIDTH"+":"+"\033[0m")
        print(" Cp(ph) = {0:.2e}".format(np.nansum(planet_spectrum_instru.flux)/np.nansum(star_spectrum_instru.flux)), ' & \u0394'+'mag = ', round(mag_planet_instru-mag_star_instru, 3))
        print(" mag(star) = ", round(mag_star_instru, 3), " & mag(planet) = ", round(mag_planet_instru, 3), "\n"  )

    #------------------------------------------------------------------------------------------------
    # For each spectral band of the instrument under consideration:
    #------------------------------------------------------------------------------------------------
    
    if show_plot: # plotting the planet (if SNR calculation) or the star (if contrast calculation) on each band (in e-/mn)
        f1 = plt.figure(dpi=300) ; band_flux = f1.add_subplot(111) ; band_flux.set_yscale("log") ; band_flux.set_xlabel("wavelength (in µm)") ;  band_flux.set_ylabel("flux (in e-/mn)") ; band_flux.grid(True)
        if post_processing == "ADI+RDI":
            band_flux.set_title(f"Star flux through coronagraph with {coronagraph} and mag({band0}) = {round(mag_star, 2)}")
        elif post_processing == "molecular mapping":
            if calculation == "SNR":
                band_flux.set_title(f"Planet flux ({model}) through {instru} bands \n with $T_p$ = {int(round(planet_spectrum_instru.T))}K and $mag_p$({band0}) = {round(mag_planet, 2)}")
            elif calculation == "contrast":
                band_flux.set_title(f"Star flux through {instru} with $mag_*$({band0}) = {round(mag_star, 2)}")
    for nb, band in enumerate(config_data['gratings']): # For each band
        if band_only is not None and band != band_only :
            continue # If you want to calculate for band_only only
        if instru == "HARMONI" and apodizer == "SP_Prox" and band != "H" and band != "H_high" and band != "J":
            continue # If you choose the apodizer for Proxima cen b, specially designed for the H-band, you ignore the other bands.
        
        name_bands.append(band) # adds the band name to the list
        
        #------------------------------------------------------------------------------------------------
        
        # Degradation at instrumental resolution and restriction of the wavelength range in the considered band
        
        star_spectrum_band = spectrum_band(config_data, band, star_spectrum_instru)
        planet_spectrum_band = spectrum_band(config_data, band, planet_spectrum_instru)
        wave_band = planet_spectrum_band.wavelength
        R = planet_spectrum_band.R # spectral resolution of the band
        if Rc is None:
            sigma = None # sigma width of gaussian convolution of the filtering
        else:
            sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2)
        mag_star_band = -2.5*np.log10(np.nanmean(star_spectrum_density.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])/np.nanmean(vega_spectrum.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])) # star magnitude on the band
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
            peak_PSF = fits.getdata("sim_data/PSF/PSF_"+instru+"/peak_PSF_"+band+"_"+coronagraph+"_"+strehl+"_"+apodizer+".fits") # flux fraction at the PSF peak as a function of separation 
            f = interp1d(peak_PSF[0], peak_PSF[1], bounds_error=False, fill_value=np.nan) ; g = interp1d(peak_PSF[0], peak_PSF[2], bounds_error=False, fill_value=np.nan)
            correction_transmission_ETC = 0.9 # correction factor (relative to ETC)
            peak_PSF_interp = f(separation)*correction_transmission_ETC ; radial_transmission_interp = g(separation)*correction_transmission_ETC
            fraction_PSF *= correction_transmission_ETC # = total stellar flux transmitted by the coronagraph (+Lyot stop) when the star is perfectly aligned with it
            PSF_profile *= fraction_PSF
            if show_plot:
                band_flux.plot(star_spectrum_band.wavelength, star_spectrum_band.flux*fraction_PSF*trans, label=band+f" ({round(np.nansum(fraction_PSF*star_spectrum_band.flux*trans/60))} e-/s)") ; band_flux.legend(loc='upper right')
        
        #------------------------------------------------------------------------------------------------
        
        # corrective factor R_corr (due to potential dithering and taking into account the power fraction of the noise being filtered)
        
        R_corr = np.zeros_like(separation)+1.
        if instru == "MIRIMRS" or instru == "NIRSpec": # 4pt dithering
            sep, r_corr = fits.getdata("sim_data/R_corr/R_corr_"+instru+"/R_corr_" + band + ".fits")
            f = interp1d(sep, r_corr, bounds_error=False, fill_value="extrapolate")
            R_corr = f(separation)
        
        if post_processing == "molecular mapping":
            fn_HF, _ = get_fraction_noise_filtered(wave=wave_band, R=R, Rc=Rc, used_filter=used_filter)
            R_corr *= fn_HF
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        if post_processing == "molecular mapping": # Molecular Mapping
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Systematic noise
            
            if systematic: # calculating systematic noise profiles
                sigma_syst_prime_2, sep, m_HF, Mp, M_pca, wave = systematic_profile(config_data, band, trans, Rc, R, star_spectrum_instru, planet_spectrum, wave_band, size_core, used_filter, show_cos_theta_est=show_cos_theta_est, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc, mag_planet=mag_planet, band0=band0, separation_planet=separation_planet, mag_star=mag_star)
                planet_spectrum_band.flux *= M_pca # M_pca = signal loss ratio due to the PCA (if wanted)
                f = interp1d(sep, np.sqrt(sigma_syst_prime_2), bounds_error=False, fill_value=np.nan)
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
                M_pca = 1

            #------------------------------------------------------------------------------------------------------------------------------------------------
        
            # Template calculation: assuming that tempalte = observed spectrum (cos theta p = 1)
            
            template, _ = filtered_flux(planet_spectrum_band.flux, R, Rc, used_filter) # [Sp]_HF
            template *= trans # gamma * [Sp]_HF
            template = template/np.sqrt(np.nansum(template**2)) # normalizing the template
            if systematic:
                planet_spectrum_band.flux *= Mp # Systematic modulations of the planetary spectrum are taken into account (mostly insignificant effect).
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
        
            # Beta calculation (with systematic modulation, if any)
            
            if Rc is None:
                beta = 0
            else:
                beta = get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, used_filter) # self-subtraction term (in ph/mn)
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Calculation of alpha (number of useful photons/min at molecular mapping on the band under consideration) (with systematic modulations, if any)
            
            alpha = get_alpha(planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, used_filter) # molecular mapping useful signal (in ph/mn)
            
            # if calculation == "contrast": # TO HAVE THE CONTRAST PER BAND INSTEAD OF A CONTRAST ON THE INSTRUMENTAL BANDWIDTH
            #     alpha *= np.nansum(star_spectrum_band.flux)/np.nansum(planet_spectrum_band.flux)
            #     beta *= np.nansum(star_spectrum_band.flux)/np.nansum(planet_spectrum_band.flux)                
            
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Printing interest quantities
        
        if print_value:
            print('\n')
            print("\033[4m" + "BAND = "+f'{band}'+f" (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm):"+"\033[0m")
            print(" mean total system transmission =", round(np.nanmean(trans), 3))
            if calculation == "SNR":
                print(" Cp(ph) = {0:.2e}".format(np.nansum(planet_spectrum_band.flux)/np.nansum(star_spectrum_band.flux))  , ' => \u0394'+'mag = ', round(mag_planet_instru-mag_star_instru, 3), f"\n Magnitudes: mag_star = {round(mag_star_band, 3)} & mag_planet = {round(mag_planet_band, 3)}")
            if post_processing == "molecular mapping":
                print(" Number of spectral pixels:", len(wave_band), " & R =", round(R))
                if Rc is not None:
                    print(" Cut-off resolution: Rc =", Rc)
                print(" Fraction in the heart of the PSF: fraction_PSF =", round(fraction_PSF, 3), 
                      "\n Useful photons (molecular mapping)/mn from the planet: alpha =", round(np.nanmean(alpha)), 
                      "\n Signal loss due to self-subtraction: beta/alpha =", round(100*np.nanmean(beta)/np.nanmean(alpha), 3), "%")
                if PCA:
                    print(" Signal loss due to PCA = ", round(100*(1-M_pca), 1), "%")

                            
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # DIT and RON_eff calculation
        
        DIT, RON_eff = DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, quantum_efficiency, RON, saturation_e, input_DIT, print_value)
        NDIT = exposure_time/DIT # number of integrations
        DIT_bands.append(DIT) # adds the DIT value of the band to the list
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Spectra through the system
        
        star_spectrum_band.flux = star_spectrum_band.flux*trans*DIT*quantum_efficiency # stellar spectrum through the system in the considered band (in e-/DIT)
        planet_spectrum_band.flux = planet_spectrum_band.flux*trans*DIT*quantum_efficiency # planet spectrum through the system in the considered band (in e-/DIT)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        if show_plot and post_processing == "molecular mapping": # plotting the spectrum on each band
            if calculation == "SNR":
                band_flux.plot(wave_band, planet_spectrum_band.flux/DIT, c=cmap(nb), label=band+f" (R={int(round(R))})") ; band_flux.legend()
            elif calculation == "contrast":
                band_flux.plot(wave_band, star_spectrum_band.flux/DIT, c=cmap(nb), label=band+f" (R={int(round(R))})") ; band_flux.legend()
        
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
                PSF_profile[separation < sep_min] *= 1e-4
                signal[separation < sep_min] *= 1e-4 # Flux attenuated by a factor of 1e-4 due to Focal Plane Mask for HARMONI 
                    
        elif post_processing == "ADI+RDI": # ADI+RDI
            if calculation == "contrast":
                signal = np.nansum(star_spectrum_band.flux)*peak_PSF_interp
            elif calculation == "SNR":
                signal = np.nansum(planet_spectrum_band.flux)*peak_PSF_interp # planet flux in the PSF peak as a function of separation (in e-/DIT/pixel)
                
        sigma_dc_2 = dark_current * DIT * 60 # dark current photon noise (in e-/DIT/pixel)
        sigma_ron_2 = RON_eff**2 # effective read out noise (in e-/DIT/pixel)
        
        if config_data["type"] == "IFU_fiber": # detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
            sigma_dc_2 *= config_data['pixel_detector_projection'] # adds quadratically
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
            bkgd_flux=fits.getdata("sim_data/Background/"+instru+"/"+bkgd+"/background_"+band+".fits") # (in e-/s/pixel)
            f = interp1d(bkgd_flux[0], bkgd_flux[1], bounds_error=False, fill_value=np.nan)
            bkgd_flux_band = f(wave_band)
            bkgd_flux_band *= 60 * DIT * np.nansum(bkgd_flux[1])/np.nansum(bkgd_flux_band) # (in e-/DIT/pixel) + we have to renormalize because we interpolate (flux conservation)
            sigma_bkgd_2 = bkgd_flux_band # background photon noise per spectral channel (in e-/DIT/pixel) for each separation
        else:
            sigma_bkgd_2 = 0
        if post_processing == "molecular mapping":
            sigma_bkgd_prime_2 = np.nansum(sigma_bkgd_2 * template**2) # background photon noise projected in the CCF (in e-/DIT/spaxel)
        elif post_processing == "ADI+RDI": # ADI+RDI
            sigma_bkgd_2 = np.nansum(sigma_bkgd_2)*radial_transmission_interp # background photon noise per spectral channel (in e-/DIT/pixel) for each separation  (radial_transmission_interp due to the coronagraph)
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Systematic
        
        if systematic:
            sigma_syst_prime_2 *= np.nansum(star_spectrum_band.flux)**2 # systematic noise projected in the CCF (in e-/DIT/spaxel)
            
        #------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Saving interest quantities (in e-/DIT/FWHM)
        
        signal_bands.append(signal) ; planet_flux_bands.append(planet_spectrum_band.flux) ; star_flux_bands.append(star_spectrum_band.flux) ; wave_bands.append(wave_band)
        if post_processing == "molecular mapping":
            sigma_ns_2_bands.append(R_corr*size_core**2*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))
            if systematic:
                sigma_s_2_bands.append(sigma_syst_prime_2)
            else:
                sigma_s_2_bands.append(0) 
        elif post_processing == "ADI+RDI": # ADI+RDI
            sigma_ns_2_bands.append(R_corr*size_core**2*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))
            if systematic:
                sigma_s_2_bands.append(sigma_syst_2)
            else:
                sigma_s_2_bands.append(0) 
                
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Contrast calculation

        if calculation == "contrast":
            
            if post_processing == "molecular mapping": # See Eq. (11) of Martos et al. 2024
                if systematic:
                    contrast = 5*np.sqrt(R_corr*size_core**2*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2 + sigma_syst_prime_2*NDIT/(R_corr*size_core**2)))/(signal*np.sqrt(NDIT))
                    contrast_wo_syst = 5*np.sqrt(R_corr*size_core**2*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))/(signal*np.sqrt(NDIT))
                else:
                    contrast = 5*np.sqrt(R_corr*size_core**2*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))/(signal*np.sqrt(NDIT))

            elif post_processing == "ADI+RDI":
                if systematic:
                    contrast = 5*np.sqrt(R_corr*size_core**2*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2 + sigma_syst_2*NDIT/(R_corr*size_core**2)))/(signal*np.sqrt(NDIT))
                    contrast_wo_syst = 5*np.sqrt(R_corr*size_core**2*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))/(signal*np.sqrt(NDIT)) 
                else:
                    contrast = 5*np.sqrt(R_corr*size_core**2*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))/(signal*np.sqrt(NDIT))

            if show_plot and show_contributions: # NOISE CONTRIBUTIONS PLOTS
                if post_processing == "molecular mapping": # Molecular Mapping
                    plt.figure(dpi=300) ; plt.xlabel(f'separation (in {sep_unit})', fontsize = 14) ; plt.ylabel(r'contrast 5$\sigma_{CCF}$ / $\alpha_0$', fontsize = 14) ; plt.title(instru+f" noise contributions on {band}"+for_name_planet+"\n with "+"$t_{exp}$=" + str(round(exposure_time)) + "mn, $mag_*$("+band0+")=" + str(round(mag_star, 2)) + f', $T_p$={T_planet}K and $R_c$ = {Rc}', fontsize = 14)
                    plt.plot(separation[separation>=sep_min], contrast[separation>=sep_min], 'k-', label="$\sigma_{CCF}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_halo_prime_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'r--', label="$\sigma'_{halo}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_ron_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'g--', label="$\sigma_{ron}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_dc_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'm--', label="$\sigma_{dc}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_bkgd_prime_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'b--', label="$\sigma'_{bkgd}$")
                    if systematic:
                        plt.plot(separation[separation>=sep_min], (5*np.sqrt(sigma_syst_prime_2)/(signal))[separation>=sep_min], 'c--', label="$\sigma'_{syst}$")
                    plt.xlim(0, separation[-1]) ; plt.legend(fontsize=12) ; plt.yscale('log') ; plt.grid(True) ; plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
                elif post_processing == "ADI+RDI": # RDI+ADI+RDI
                    plt.figure(dpi=300) ; plt.xlabel(f'separation (in {sep_unit})', fontsize = 14) ; plt.ylabel(r'contrast 5$\sigma$ / $F_{max}$', fontsize = 14) ; plt.title(instru+f" noise contributions on {band}"+for_name_planet+" \n with "+"$t_{exp}$=" + str(round(exposure_time)) + "mn and $mag_*$("+band0+")=" + str(round(mag_star, 2)), fontsize = 14)
                    plt.plot(separation[separation>=sep_min], contrast[separation>=sep_min], 'k-', label="$\sigma_{tot}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_halo_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'r--', label="$\sigma_{halo}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_ron_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'g--', label="$\sigma_{ron}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_dc_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'm--', label="$\sigma_{dc}$")
                    plt.plot(separation[separation>=sep_min], (5*np.sqrt(R_corr*size_core**2*(sigma_bkgd_2)*(NDIT))/(signal*NDIT))[separation>=sep_min], 'b--', label="$\sigma_{bkgd}$")
                    plt.xlim(0, separation[-1]) ; plt.legend(fontsize=12) ; plt.yscale('log') ; plt.grid(True) ; plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
            contrast_bands.append(contrast) # adds the contrast curve of the band to the list
        
        # SNR calculation
        
        elif calculation == "SNR":
            
            if post_processing == "molecular mapping": # See Eq. (10) of Martos et al. 2024
                if systematic:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*size_core**2*(sigma_halo_prime_2+sigma_ron_2+sigma_dc_2+sigma_bkgd_prime_2+sigma_syst_prime_2*NDIT/(R_corr*size_core**2))) # avec systématiques
                else:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*size_core**2*(sigma_halo_prime_2+sigma_ron_2+sigma_dc_2+sigma_bkgd_prime_2)) # sans systématiques
            
            elif post_processing == "ADI+RDI":
                if systematic:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*size_core**2*(sigma_halo_2+sigma_ron_2+sigma_dc_2+sigma_bkgd_2+sigma_syst_2*NDIT/(R_corr*size_core**2))) # avec systématiques
                else:
                    SNR = signal*np.sqrt(NDIT)/np.sqrt(R_corr*size_core**2*(sigma_halo_2+sigma_ron_2+sigma_dc_2+sigma_bkgd_2)) # sans systématiques

            if separation_planet is not None: # SNR values at the planet's separation
                SNR_planet_bands.append(SNR[idx])
                if print_value:
                    print(f' S/N at {round(separation_planet, 1)} {sep_unit} = ', round(SNR[idx], 2))
                if np.max(SNR) > SNR_max: 
                    SNR_max=np.max(SNR) 
                if  SNR[idx] > SNR_max_planet:
                    SNR_max_planet=SNR[idx] ; name_max_SNR = band
            
            # Effect of noise on correlation estimation
            
            if show_cos_theta_est and post_processing == "molecular mapping": # calculation of cos theta est (impact of noise+systematics (and auto-subtraction) on correlation estimation)
                Mp_Sp = NDIT*planet_spectrum_band.flux*fraction_PSF # planet flux (with modulations, if any) in e-/FWHM
                Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp/trans, R, Rc, used_filter) # filtered planet flux
                star_HF, star_LF = filtered_flux(star_spectrum_band.flux/trans, R, Rc, used_filter) # filtered star flux
                alpha = np.sqrt(np.nansum((trans*Mp_Sp_HF)**2)) # true effective signal
                beta = np.nansum(trans*star_HF*Mp_Sp_LF/star_LF * template) # self subtraction
                cos_theta_lim = np.nansum( trans*Mp_Sp_HF * template ) / alpha # loss of correlation due to systematics
                for i in range(len(separation)): # for each separation
                        star_flux = PSF_profile[i]*star_spectrum_band.flux # stellar flux in e-/DIT/pixel at separation i
                        star_flux[star_flux>saturation_e] = saturation_e # if saturation
                        sigma_tot = np.sqrt(R_corr[i]*size_core**2*NDIT*(sigma_halo_2[i] + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) # total noise per spectral channel (in e-/Afwhm/bin)
                        N = 1000 ; cos_theta_est_n = np.zeros((N)) ; norm_d_n = np.zeros((N))
                        for n in range(len(cos_theta_est_n)): # N noise simulations
                            noise  = np.random.normal(0, sigma_tot, len(wave_band)) # noise realisation
                            d = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF + noise + M_HF[i]*NDIT*size_core**2*star_flux # spectrum at the planet's location: see Eq.(18) of Martos et al. 2024
                            norm_d_n[n] = np.sqrt(np.nansum(d**2))
                            cos_theta_est_n[n] = np.nansum( d * template ) / norm_d_n[n]
                        norm_d[i] = np.nanmean(norm_d_n)
                        cos_theta_est[i] = np.nanmean(cos_theta_est_n)
                        if separation_planet is not None and i == idx:
                            print(" S/N PER SPECTRAL CHANNEL = ", round(np.nanmean(Mp_Sp/sigma_tot), 3))
                cos_theta_n = alpha/norm_d # loss of correlation due to fundamental noises
                plt.figure(dpi=300) ; plt.plot(separation, cos_theta_est, 'k') ; plt.ylabel(r"cos $\theta_{est}$", fontsize=14) ; plt.grid(True) ; plt.title(f"Effect of noise and stellar substraction on the estimation of correlation between \n template and planetary spectrum for {instru} on {band} \n (assuming that the template is the same as the observed spectrum)")
                plt.xlabel(f'separation (in {sep_unit})', fontsize=14)
                if separation_planet is not None and separation_planet < np.nanmax(separation):
                    print(" beta/alpha = ", round(beta/alpha, 3), "\n cos_theta_n = ", round(cos_theta_n[idx], 3), "\n cos_theta_lim = ", round(cos_theta_lim, 3))
                    if cos_est is not None:
                        cos_theta_p = (cos_est/cos_theta_n[idx] + beta/alpha)/cos_theta_lim                   
                        print(" cos_theta_est = ", round(cos_est, 3), " => cos_theta_p = ", round(cos_theta_p, 3))
                    plt.plot([separation_planet, separation_planet], [np.nanmin(cos_theta_est), np.nanmax(cos_theta_est)], 'k--', label=f'angular separation{for_name_planet}')
                    plt.plot([separation_planet, separation_planet], [cos_theta_est[idx], cos_theta_est[idx]], 'rX', ms=11, label=r"cos $\theta_{est}$"+for_name_planet+f' ({round(cos_theta_est[idx], 2)})')
                    plt.legend()
                plt.show()

            SNR_bands.append(SNR) # adds the SNR curve of the band to the list
            
        if show_plot and show_t_syst and systematic: # plot of t_syst : see Eq.(14) of Martos et al. 2024
            t_syst = DIT*R_corr*size_core**2*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2)/sigma_syst_prime_2 # en min 
            plt.figure(dpi=300) ; plt.plot(separation, t_syst, 'b') ; plt.ylabel('$t_{syst}$ (in mn)', fontsize = 14) ; plt.xlabel(f'separation (in {sep_unit})', fontsize = 14) ; plt.grid(True) ; plt.title("$t_{syst}$"+f" on {band}"+"\n with $mag_*$("+band0+")=" + str(round(mag_star, 2)), fontsize = 14) ; plt.plot([separation[0], separation[-1]], [exposure_time, exposure_time], 'r-') ; plt.yscale('log') ; plt.legend(["$t_{syst}$", "$t_{exp}$ ="+f"{exposure_time} mn"]) 

    #------------------------------------------------------------------------------------------------
    # PLOTS:
    #------------------------------------------------------------------------------------------------
    
    # Contrast:
        
    if calculation == "contrast" and show_plot:
        plt.figure(dpi=300) ; ax1 = plt.gca() ; ax1.grid(True) ; ax1.set_yscale('log') ; ax1.axvspan(0, sep_min, color='k', alpha=0.5, lw=0) ; ax1.set_xlim(0, separation[-1])
        ax1.set_title(instru + " contrast curves"+for_name_planet+f" with {post_processing}"+" \n with $t_{exp}$=" + str(round(exposure_time, 1)) + "mn, $mag_*$("+band0+")=" + str(round(mag_star, 1)) + f' and $T_p$={round(T_planet)}K', fontsize = 14) ; ax1.set_xlabel(f"separation (in {sep_unit})", fontsize=12) ; ax1.set_ylabel(r'5$\sigma$ contrast (on instru-band)', fontsize=12)
        for i in range(len(contrast_bands)):
            if band_only is not None:
                ax1.plot(separation_bands[i][separation_bands[i]>=sep_min], contrast_bands[i][separation_bands[i]>=sep_min], label=name_bands[i], color=cmap([nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]))
            else:
                ax1.plot(separation_bands[i][separation_bands[i]>=sep_min], contrast_bands[i][separation_bands[i]>=sep_min], label=name_bands[i], color=cmap(i))
        if separation_planet is not None  and separation_planet < np.nanmax(separation):
            if mag_planet is None:
                ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin, ymax)
                ax1.plot([separation_planet, separation_planet], [ymin, ymax], 'k--', label=f'{name_planet}')
            else:
                star_spectrum_instru, star_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, star_spectrum) # spectre de l'étoile en photons/min ajusté à la bonne magnitude
                planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_planet, planet_spectrum) # spectre de la planète en ph/min ajusté à la bonne magnitude
                flux_ratio = np.nanmean(planet_spectrum_instru.flux/star_spectrum_instru.flux)
                ax1.plot([separation_planet, separation_planet], [flux_ratio, flux_ratio], 'ko')
                ax1.annotate(f"{name_planet}", (separation_planet+0.025*separation[-1], flux_ratio))
        ax2 = ax1.twinx() ; ax2.invert_yaxis() ; ax2.set_ylabel(r'$\Delta$mag', fontsize=12, labelpad=20, rotation=270) ; ax2.tick_params(axis='y')   
        for i in range(len(contrast_bands)):
            if band_only is not None:
                ax2.plot(separation_bands[i][separation_bands[i]>=sep_min], -2.5*np.log10(contrast_bands[i])[separation_bands[i]>=sep_min], color=cmap([nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]))
            else:
                ax2.plot(separation_bands[i][separation_bands[i]>=sep_min], -2.5*np.log10(contrast_bands[i])[separation_bands[i]>=sep_min], color=cmap(i))
        if separation_planet is not None  and separation_planet < np.nanmax(separation) and mag_planet is not None:
            ax2.plot([separation_planet, separation_planet], [-2.5*np.log10(flux_ratio), -2.5*np.log10(flux_ratio)], 'ko')
            ax2.annotate(f"{name_planet}", (separation_planet+0.025*separation[-1], -2.5*np.log10(flux_ratio)))
        ax1.legend(loc="upper right") ; ax1.set_zorder(1) ; plt.show()
    
    #------------------------------------------------------------------------------------------------

    # SNR:
    
    elif calculation == "SNR" and show_plot:
        if separation_planet is not None  and print_value:
            print('\n') ; print(f' MAX S/N (at {round(separation_planet, 1)} {sep_unit}) = ', round(SNR_max_planet, 1), "for "+name_max_SNR)
        if channel and instru == 'MIRIMRS':
            exposure_time *= 3
            SNR_chan=[] ; separation_chan=[] ; separation_chan.append(separation_bands[0]) ; separation_chan.append(separation_bands[3]) ; SNR_chan1=np.zeros(len(SNR_bands[0])) ; SNR_chan2=np.zeros(len(SNR_bands[3])) ; name_bands[0]="channel 1" ; name_bands[1]="channel 2" ; SNR_max_planet=0.
            SNR_chan.append(np.sqrt(np.nansum(np.array(SNR_bands[:3])**2, 0))) ; SNR_chan.append(np.sqrt(np.nansum(np.array(SNR_bands[3:])**2, 0))) ; SNR_max=max(max(SNR_chan[0]), max(SNR_chan[1]))
            if separation_planet is not None:
                for i in range(len(SNR_chan)):
                    idx = (np.abs(separation_chan[i] - separation_planet)).argmin()
                    if  SNR_chan[i][idx] > SNR_max_planet:
                        SNR_max_planet=SNR_chan[i][idx] ; name_max_SNR = name_bands[i]
                print(f'MAX S/N (at {separation_planet}") = ', round(SNR_max_planet, 2), "for "+name_max_SNR)
            separation_bands=separation_chan ; SNR_bands=SNR_chan
        plt.figure(dpi=300) ; plt.grid(True)
        plt.title(instru + " S/N curves"+for_name_planet+" with $t_{exp}$=" + str(round(exposure_time)) + "mn, \n $mag_*$("+band0+")=" + str(round(mag_star, 2)) + ", $mag_p$("+band0+")=" + str(round(mag_planet, 2))+f' and $T_p$ = {round(T_planet)}K ('+model+')', fontsize = 14)
        for i in range(len(SNR_bands)):
            if band_only is not None:
                plt.plot(separation_bands[i][separation_bands[i]>=sep_min], SNR_bands[i][separation_bands[i]>=sep_min], label=name_bands[i], color=cmap([nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]))
            else:
                plt.plot(separation_bands[i][separation_bands[i]>=sep_min], SNR_bands[i][separation_bands[i]>=sep_min], label=name_bands[i], color=cmap(i))
        plt.legend(loc='upper left') ; plt.xlim(0, separation[-1]) ; plt.ylim(0) ; plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0) ; plt.xlabel(f"separation (in {sep_unit})", fontsize=12) ; plt.ylabel('S/N', fontsize=14) ; ax = plt.gca() ; ax.yaxis.set_ticks_position('both') ; plt.tight_layout()
        if separation_planet is not None  and separation_planet < np.nanmax(separation):
            plt.plot([separation_planet, separation_planet], [0., SNR_max], 'k--') ; plt.plot([separation_planet, separation_planet], [SNR_max_planet, SNR_max_planet], 'rX', ms=11)
            ax_legend = ax.twinx() ; ax_legend.plot([], [], '--', c='k', label=f'angular separation{for_name_planet}'); ax_legend.plot([], [], 'X', c='r', label='max S/N'+for_name_planet+' ('+str(round(SNR_max_planet, 2))+')') ; ax_legend.legend(loc='lower right') ; ax_legend.tick_params(axis='y', colors='w')
        plt.show()
        
    #------------------------------------------------------------------------------------------------
    # RETURNS:
    #------------------------------------------------------------------------------------------------
    
    if calculation == "contrast":
        return name_bands, separation_bands, contrast_bands, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    elif calculation == "SNR":
        return name_bands, separation_bands, SNR_bands, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves init
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation="contrast", instru="HARMONI", exposure_time=120, mag_star=7, mag_planet=None, band0="K", model="BT-Settl", T_planet=1000, lg_planet=4.0, T_star=6000, lg_star=4.0, star_rv=0, delta_rv=0, vsini_star=0, vsini_planet=0, 
             apodizer="NO_SP", strehl="JQ1", coronagraph=None, systematic=False, PCA=False, PCA_mask=False, Nc=20, channel=False, name_planet=None, separation_planet=None, show_plot=True, print_value=True, 
             post_processing=None, bkgd="medium", Rc=100, used_filter="gaussian", input_DIT=None, band_only=None, 
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
    star_rv (float)          : star's radial velocity (in km/s). Default: 0.
    delta_rv (float)         : radial velocity shift between the planet and the star (in km/s). Default: 0.
    vsini_star (float)       : star's rotation speed (in km/s). Default: 0.
    vsini_planet (float)     : planet's rotation speed (in km/s). Default: 0.
    apodizer (str)           : apodizer selected, if any. Default: "NO_SP" ("NO_SP" means NO Shaped Pupil => no apodizer).
    strehl (str)             : strehl selected, if ground-based observation. Default: "JQ1". ("NO_JQ" means NO J.. Quartile => no strehl for space-based observation).
    coronagraph (str)        : coronagraph selected, if any. Default: None.
    systematic (bool)        : to take systematic noise (other than speckles) into account (True or False) if it can be estimated (only for MIRIMRS and NIRSpec for now). Default: False.
    PCA (bool)               : to use PCA as systematic removal
    PCA_mask (bool)          : to consider a mask on the planet while estimating the components of the PCA
    Nc (int)                 : Number of PCA components subtracted. Default: 20. (If Nc = 0, there will be no PCA)
    channel (bool)           : for MIRI/MRS, SNR curves can be combined by channel (not by band) (True or False). Default: False.
    name_planet (str)        : planet's name (for plot purposes only). Default: None.
    separation_planet (float): planet's separation (in arcsec), to find the planet's SNR or contrast. Default: None.
    post_processing (str)    : post-processing method ("molecular mapping" or "ADI+RDI"). Default: None.
    bkgd (str)               : background level (None, "low", "medium", "high"). Default: "medium".
    Rc (float)               : cut-off resolution for molecular mapping post-processing. Default: 100. (If Rc = None, there will be no filtering)
    used_filter (str)        : type of filter used ("gaussian", "step" or "smoothstep"). Default: "gaussian".
    """
    time1 = time.time() ; warnings.filterwarnings('ignore', category=UserWarning, append=True)
    
    if instru not in instru_name_list: # checking if the instru is in FastCurves
        raise KeyError(f"{instru} is not yet considered in FastCurves. Available instruments: {instru_name_list}")
    if instru == "NIRCam" and coronagraph is None: # only one coronographic mask is yet considered in FastCurves for NIRCam
        coronagraph = "MASK335R"
    config_data = get_config_data(instru) # getting the instruments specs
    if config_data["base"] == "space": # space-based observations => no tellurics and no strehl
        tellurics = False ; strehl = "NO_JQ"
    elif config_data["base"] == "ground": # space-based observations => tellurics and strehl
        tellurics = True 
        if strehl not in config_data["strehl"]: # checking if the strehl is considered for the instrument
            raise KeyError(f"No PSF profiles for {strehl} strehl with {instru}. Available strehl values: {config_data['strehl']}")
    if apodizer not in config_data["apodizers"]:
        raise KeyError(f"No PSF profiles for {apodizer} strehl with {instru}. Available apodizers: {config_data['apodizers']}")
    sep_unit = config_data["sep_unit"] # angular separation unit (arcsec or mas)
    if post_processing is None: # post-processing method considered
        if "IFU" in config_data["type"]:
            post_processing = "molecular mapping"
        elif config_data["type"] == "imager":
            post_processing = "ADI+RDI"
    
    if star_spectrum is None: # if not input, load the star spectrum
        star_spectrum = load_star_spectrum(T_star, lg_star) # star spectrum (BT-NextGen GNS93)
        if vsini_star != 0: # rotational broadening (in km/s)
            star_spectrum = star_spectrum.broad(vsini_star)
        if star_rv != 0: # Doppler shifting (in km/s)
            star_spectrum = star_spectrum.doppler_shift(star_rv)
            star_spectrum.star_rv = star_rv
            
    if planet_spectrum is None: # if not input, load the planet spectrum
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru) #  planet spectrum: class Spectrum(wavel, flux, R, T) in J/s/m²/µm according to the considered model
        if vsini_planet != 0: # rotational broadening (in km/s)
            planet_spectrum = planet_spectrum.broad(vsini_planet)
        if delta_rv != 0 or star_rv != 0: # Doppler shifting (in km/s)
            planet_spectrum = planet_spectrum.doppler_shift(star_rv+delta_rv)
            planet_spectrum.delta_rv = delta_rv

    name_bands, separation, curves, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands = FastCurves_main(calculation=calculation, instru=instru, exposure_time=exposure_time, mag_star=mag_star, band0=band0, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, tellurics=tellurics, 
                   apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=systematic, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc, channel=channel, name_planet=name_planet, separation_planet=separation_planet, mag_planet=mag_planet, show_plot=show_plot, print_value=print_value, 
                   post_processing=post_processing, sep_unit=sep_unit, bkgd=bkgd, Rc=Rc, used_filter=used_filter, input_DIT=input_DIT, band_only=band_only)
    
    if print_value:
        print(f'\n FastCurves {calculation} calculation took {round(time.time()-time1, 1)} s')

    if return_SNR_planet: # For FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1e3 # switching the angular separation unit
        SNR_planet = np.zeros((len(config_data["gratings"]))) ; signal_planet = np.zeros((len(config_data["gratings"]))) ; sigma_ns_planet = np.zeros((len(config_data["gratings"]))) ; sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]): # retrieving the values at the planet separation
            idx = np.abs(separation[nb]-separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_bands[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_bands[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_bands[nb][idx])
        return name_bands, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_bands)
    elif return_quantity:
        return name_bands, separation, signal_bands, sigma_s_2_bands, sigma_ns_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    else:
        return name_bands, separation, curves








