from src.signal_noise_estimate import *
path_file = os.path.dirname(__file__)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves Function
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves_main(calculation, instru, exposure_time, mag_star, band0, planet_spectrum, star_spectrum, tellurics, apodizer, strehl, coronagraph, systematic, PCA, PCA_mask, Nc, channel, planet_name, separation_planet, mag_planet, show_plot, verbose, post_processing, sep_unit, bkgd, Rc, filter_type, input_DIT, band_only, return_SNR_planet):
    """
    See the function "FastCurves" below.
    """

    # HARD CODED (variables for specific analysis): 
    cos_p              = 1     # mismatch 
    cos_est            = None  # estimated correlation in the data # 0.29 for CT Cha b on 1SHORT band of MIRIMRS
    show_cos_theta_est = False # to see the impact of the noise on the estimated correlation in order to retrieve the true mismatch
    show_t_syst        = False # to see the systematic time 
    show_contributions = True  # to see the noise contributions plots for contrast calculations
    
    #------------------------------------------------------------------------------------------------
    
    if instru == "MIRIMRS" and channel and (calculation == "SNR" or calculation == "corner plot"):
        exposure_time /= 3 # dividing the exposure time budget by 3 to observe the 3 band (SHORT, MEDIUM, LONG) in order to have a SNR per channel
    if planet_name is None:
        for_planet_name = "" # for plot purposes
    else:
        for_planet_name = " for "+planet_name

    # LOADING INSTRUMENTS SPECS
    
    config_data  = get_config_data(instru)
    n_gratings   = len(config_data["gratings"])
    cmap         = plt.get_cmap("Spectral", n_gratings + 1 if n_gratings % 2 != 0 else n_gratings)
    size_core    = config_data["size_core"]            # aperture size on which the signal is integrated (in pixels)
    A_FWHM       = size_core**2                        # box aperture 
    saturation_e = config_data["spec"]["saturation_e"] # full well capacity of the detector (in e-)
    min_DIT      = config_data["spec"]["minDIT"]       # minimal integration time (in mn)
    max_DIT      = config_data["spec"]["maxDIT"]       # maximal integration time (in mn)
    RON          = config_data["spec"]["RON"]          # read out noise (in e-/px/DIT)
    dark_current = config_data["spec"]["dark_current"] # dark current (in e-/px/s)
    IWA, OWA     = get_wa(config_data=config_data, band="INSTRU", apodizer=apodizer, sep_unit=config_data["sep_unit"])
    
    #------------------------------------------------------------------------------------------------
    
    contrast_bands = [] ; SNR_bands = [] ; SNR_planet_bands = [] ; name_bands = [] ; separation_bands = [] # contrast, SNR, name and separation of each band
    signal_bands = [] ; DIT_bands = [] ; sigma_syst_2_bands = [] ; sigma_fund_2_bands = [] ; sigma_halo_2_bands = [] ; sigma_det_2_bands = [] ; sigma_bkg_2_bands = [] ; planet_flux_bands = [] ; star_flux_bands = [] ; wave_bands = []
    uncertainties_bands = []
    SNR_max = 0. ; SNR_max_planet = 0. # for plot and print purposes
    T_planet = planet_spectrum.T ; T_star = star_spectrum.T # planet and star temperature
    lg_planet = planet_spectrum.lg ; lg_star = star_spectrum.lg # planet and star surface gravity
    model_planet = planet_spectrum.model ; model_star = star_spectrum.model # planet and star model
    rv_star = star_spectrum.rv ; rv_planet = planet_spectrum.rv # radial velocity of the star and the planet
    vsini_star = star_spectrum.vsini ; vsini_planet = planet_spectrum.vsini # rotationnal velocity of the star and the planet
    R_planet = planet_spectrum.R ; R_star = star_spectrum.R  ; R = max(R_planet, R_star) # spectra resolutions
    if sep_unit == "mas" and separation_planet is not None:
        separation_planet *= 1e3 # switching the angular separation unit (arcsec => mas)
    
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
            print(f"  mag({band0})  = {mag_planet:.1f}")
        print("\n"+"\033[4m"+f"Stellar spectrum ({model_star}):"+"\033[0m")
        print(f"  R       = {round(R_star, -3):.0f}")
        print(f"  Teff    = {T_star:.0f} K")
        print(f"  log(g)  = {lg_star:.1f} dex(cm/s2)")
        print(f"  rv      = {rv_star:.1f} km/s")
        print(f"  Vsin(i) = {vsini_star:.1f} km/s")
        print(f"  mag({band0})  = {mag_star:.1f}")
        if post_processing == "molecular mapping":
            print(f'\nMolecular mapping considered as post-processing method with Rc = {Rc} and {filter_type} filtering')
        elif post_processing == "ADI+RDI":
            print('\nADI and/or RDI considered as post-processing method')
        if systematic:
            if PCA:
                print(f'With systematics + PCA (with {Nc} components)')
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

    #------------------------------------------------------------------------------------------------

    # Restricting spectra to the instrumental range and normalizing spectra to the correct magnitude
    
    star_spectrum_instru, star_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, star_spectrum) # star spectrum in photons/min adjusted to the correct magnitude
    if mag_planet is not None:
        planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_planet, planet_spectrum) # planet spectrum in photons/min adjusted to the correct magnitude
        planet_to_star_flux_ratio = np.nanmean(planet_spectrum_instru.flux / star_spectrum_instru.flux)
    else:
        if calculation == "SNR" or calculation == "corner plot":
            raise KeyError(f"PLEASE INPUT A MAGNITUDE FOR THE PLANET FOR THE {calculation} CALCULATION !")
        planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_star, planet_spectrum) # planet spectrum in photons/min adjusted to the correct magnitude
    if calculation == "contrast": # setting the planetary spectrum to the same flux (in total received ph/mn) as the stellar spectrum on the instrumental band of interest (by doing this, we will have a contrast in photons and not in energy, otherwise we would have had to set it to the same received energy) + the contrast is then for all over the instrumental bandwidth
        planet_spectrum_instru.set_flux(np.nansum(star_spectrum_instru.flux))
        planet_spectrum_density.flux *= np.nanmean(star_spectrum_density.flux)/np.nanmean(planet_spectrum_density.flux) # not really useful, only for flux densities (in energy) to have the same magnitude
    wave_instru = planet_spectrum_instru.wavelength
    vega_spectrum = load_vega_spectrum() # vega spectrum in J/s/m²/µm
    vega_spectrum = vega_spectrum.interpolate_wavelength(wave_instru, renorm = False) # interpolating the vega spectrum on the instrumental wavelength axis
    mag_star_instru   = -2.5*np.log10(np.nanmean(star_spectrum_density.flux)/np.nanmean(vega_spectrum.flux)) # star magnitude on the instrumental band
    mag_planet_instru = -2.5*np.log10(np.nanmean(planet_spectrum_density.flux)/np.nanmean(vega_spectrum.flux)) # planet magnitude on the instrumental band
    
    if show_plot and band_only is None: # plot stellar and planetary spectra on the instrument band 
        plt.figure(figsize=(10, 6), dpi=300)
        plt.yscale("log")
        plt.xlim(wave_instru[0], wave_instru[-1])        
        plt.xlabel("wavelength [µm]", fontsize=14)
        plt.ylabel(r"flux [J/s/$m^2$/µm]", fontsize=14)
        plt.title(f"Star and planet spectra on the instrumental bandwidth (R = {round(round(R, -3))})" + f"\n with $rv_*$ = {round(rv_star, 1)} km/s & $rv_p$ = {round(rv_planet, 1)} km/s", fontsize=16)
        plt.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        plt.plot(wave_instru, planet_spectrum_density.flux, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'Planet, {model_planet} with $T$={int(round(T_planet))}K\nmag (instru)={round(mag_planet_instru, 1)}')        
        plt.plot(wave_instru, star_spectrum_density.flux, color='black', linestyle='-', linewidth=2, alpha=0.8, label=f'Star, {model_star} with $T$={int(round(T_star))}K\nmag (instru)={round(mag_star_instru, 1)}')        
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)  
        plt.gca().yaxis.set_ticks_position('both')
        plt.minorticks_on()
        plt.tight_layout()        
        plt.show()
    
    if verbose:
        print("\n"+"\033[4m" + "ON THE INSTRUMENTAL BANDWIDTH"+":"+"\033[0m")
        print(" Cp(ph) = {0:.2e}".format(np.nansum(planet_spectrum_instru.flux)/np.nansum(star_spectrum_instru.flux)), ' & \u0394'+'mag = ', round(mag_planet_instru-mag_star_instru, 2))
        print(" mag(star) = ", round(mag_star_instru, 2), " & mag(planet) = ", round(mag_planet_instru, 2))

    #------------------------------------------------------------------------------------------------
    # For each spectral band of the instrument under consideration:
    #------------------------------------------------------------------------------------------------
    
    if show_plot: # plotting the planet (if SNR calculation) or the star (if contrast calculation) on each band (in e-/mn)
        f1 = plt.figure(figsize=(10, 6), dpi=300)
        band_flux = f1.add_subplot(111)
        band_flux.set_yscale("log")
        band_flux.set_xlim(wave_instru[0], wave_instru[-1])        
        band_flux.set_xlabel("wavelength [µm]", fontsize=14)
        band_flux.set_ylabel("flux [e-/mn]", fontsize=14)
        band_flux.grid(which='both', linestyle=':', color='gray', alpha=0.5) 
        band_flux.yaxis.set_ticks_position('both')
        band_flux.minorticks_on()
        ymin = 1e9 ; ymax = 1
        if calculation == "SNR" or calculation == "corner plot":
            band_flux.set_title(f"Planet flux ({model_planet}) through {instru} bands \n with $T_p$ = {int(round(planet_spectrum_instru.T))}K, $lg_p$ = {round(planet_spectrum_instru.lg,1)} and $mag_p$({band0}) = {round(mag_planet, 2)}", fontsize=16)
        elif calculation == "contrast":
            band_flux.set_title(f"Star flux through {instru} with $mag_*$({band0}) = {round(mag_star, 2)}", fontsize=16)
    
    for nb, band in enumerate(config_data['gratings']): # For each band
        if band_only is not None and band != band_only :
            continue # If you want to calculate for band_only only
        
        name_bands.append(band) # adds the band name to the list
        
        iwa, owa = get_wa(config_data=config_data, band=band, apodizer=apodizer, sep_unit=config_data["sep_unit"])
        
        #------------------------------------------------------------------------------------------------
        
        # Degradation at instrumental resolution and restriction of the wavelength range in the considered band
        
        star_spectrum_band   = spectrum_band(config_data, band, star_spectrum_instru)
        planet_spectrum_band = spectrum_band(config_data, band, planet_spectrum_instru)
        wave_band            = planet_spectrum_band.wavelength
        R                    = config_data['gratings'][band].R # spectral resolution of the band
        mag_star_band        = -2.5*np.log10(np.nanmean(star_spectrum_density.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])/np.nanmean(vega_spectrum.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)]))   # star magnitude on the band
        mag_planet_band      = -2.5*np.log10(np.nanmean(planet_spectrum_density.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])/np.nanmean(vega_spectrum.flux[(wave_instru>config_data['gratings'][band].lmin)&(wave_instru<config_data['gratings'][band].lmax)])) # planet magnitude on the band

        #------------------------------------------------------------------------------------------------
        
        # System transmissions for the considered band
        
        trans = get_transmission(instru, wave_band, band, tellurics, apodizer, strehl, coronagraph)
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Reading PSF profiles (fraction_PSF = fraction of photons in the PSF core/FWHM)
        
        PSF_profile, fraction_PSF, separation, pxscale = get_PSF_profile(band, strehl, apodizer, coronagraph, instru, config_data, sep_unit, separation_planet, return_SNR_planet)
        separation_bands.append(separation) # adds the separation axis to the list
        
        if separation_planet is not None: # index of the separation of the planet
            idx = np.where(separation==separation_planet)[0][0]
                
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Coronagraph
        
        if coronagraph is not None:
            data = fits.getdata(f"sim_data/PSF/PSF_{instru}/fraction_PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits") # flux fraction at the PSF peak as a function of separation 
            f = interp1d(data[0], data[1], bounds_error=False, fill_value=np.nan)
            g = interp1d(data[0], data[2], bounds_error=False, fill_value=np.nan)
            f_interp = f(separation)
            g_interp = g(separation)
            f_interp[separation > data[0][-1]] = data[1][-1] # flat extrapolation
            g_interp[separation > data[0][-1]] = data[2][-1] # flat extrapolation 
            fraction_PSF        = f_interp
            radial_transmission = g_interp
            star_transmission   = g(0) # = total stellar flux transmitted by the coronagraph (+Lyot stop) when the star is perfectly aligned with it (i.e. at 0 separation)
            PSF_profile *= star_transmission
            
        #------------------------------------------------------------------------------------------------
        
        # corrective factor R_corr (due to potential dithering and taking into account the power fraction of the noise being filtered)
        
        R_corr = np.zeros_like(separation) + 1.
        if instru == "MIRIMRS" or instru == "NIRSpec": # dithering for MIRIMRS and NIRSpec
            sep, r_corr = fits.getdata(f"sim_data/R_corr/R_corr_{instru}/R_corr_{band}.fits")
            sep    = sep[~np.isnan(r_corr)]
            r_corr = r_corr[~np.isnan(r_corr)]
            f      = interp1d(sep, r_corr, bounds_error=False, fill_value="extrapolate")
            R_corr = f(separation)
            R_corr[separation > sep[-1]] = r_corr[-1] # flat extrapolation
        
        if post_processing == "molecular mapping": # power fraction of the white noise filtered
            fn_HF, _ = get_fraction_noise_filtered(wave=wave_band, R=R, Rc=Rc, filter_type=filter_type)
            R_corr  *= fn_HF
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        if post_processing == "molecular mapping": # Molecular Mapping
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Systematic noise
            
            if systematic: # calculating systematic noise profiles
                sigma_syst_prime_2, sep, m_HF, Mp, M_pca, wave, pca = get_systematic_profile(config_data, band, trans, Rc, R, star_spectrum_instru, planet_spectrum_instru, planet_spectrum, wave_band, size_core, filter_type, show_cos_theta_est=show_cos_theta_est, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc, mag_planet=mag_planet, band0=band0, separation_planet=separation_planet, mag_star=mag_star, target_name=planet_name, exposure_time=exposure_time, band_only=band_only, verbose=verbose)
                planet_spectrum_band.flux *= M_pca # M_pca = signal loss ratio due to the PCA (if wanted)
                sep                = sep[~np.isnan(sigma_syst_prime_2)]
                sigma_syst_prime_2 = sigma_syst_prime_2[~np.isnan(sigma_syst_prime_2)]
                f = interp1d(sep, np.sqrt(sigma_syst_prime_2), bounds_error=False, fill_value="extrapolate")
                f_interp = f(separation)
                f_interp[separation < sep[0]] = np.sqrt(sigma_syst_prime_2[0])
                if separation[-1] > sep[-1]: # systematic profile extrapolation
                    f_extension  = PSF_profile[separation >= sep[-1]]               # same extrapolation profile (propto stellar flux)
                    f_extension *= np.sqrt(sigma_syst_prime_2[-1]) / f_extension[0] # forcing continuity
                    f_interp[separation >= sep[-1]] = f_extension
                sigma_syst_prime_2 = f_interp**2 # systematic noise profile projected in the CCF (in e-/Flux_stell_tot/spaxel)
                mask_M = (wave_band>=wave[0]) & (wave_band<=wave[-1]) # effective wavelength axis (from data)
                planet_spectrum_band.crop(wave[0], wave[-1])
                star_spectrum_band.crop(wave[0], wave[-1])
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
            template   *= trans                                                        # gamma * [Sp]_HF
            template    = template/np.sqrt(np.nansum(template**2))                     # normalizing the template
            if systematic:
                planet_spectrum_band.flux *= Mp # Systematic modulations of the planetary spectrum are taken into account (mostly insignificant effect).

            #------------------------------------------------------------------------------------------------------------------------------------------------
        
            # Beta calculation (with systematic modulation, if any)
            
            beta = get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, filter_type) # self-subtraction term (in ph/mn)
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            # Calculation of alpha (number of useful photons/min at molecular mapping on the band under consideration) (with systematic modulations, if any)
            
            alpha = get_alpha(planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, filter_type) # molecular mapping useful signal (in ph/mn)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Printing interest quantities
        
        if verbose:
            if config_data["type"] == "imager":
                print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm):"+"\033[0m")
            else:
                print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave_band[0], 2)} to {round(wave_band[-1], 2)} µm with R={R:.0f}):"+"\033[0m")
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
                print(" Number of spectral channels:", len(wave_band))
                if calculation == "SNR" or calculation == "corner plot":
                    print(" Useful molecular mapping signal from the planet: α =", round(np.nanmean(alpha)), "ph/mn")
                print(" Signal loss due to self-subtraction: β/α =", round(100*np.nanmean(beta)/np.nanmean(alpha), 1), "%")
                if PCA and systematic:
                    print(" Signal loss due to PCA =", round(100*(1-M_pca), 1), "%")
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # DIT and RON_eff calculation
        
        DIT, RON_eff = get_DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, RON, saturation_e, input_DIT, verbose)
        NDIT         = exposure_time/DIT # number of integrations
        DIT_bands.append(DIT)            # adds the DIT value of the band to the list
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Spectra through the system
        
        star_spectrum_band.flux   *= trans*DIT # stellar spectrum through the system in the considered band (in e-/DIT)
        planet_spectrum_band.flux *= trans*DIT # planet spectrum through the system in the considered band (in e-/DIT)
        
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Plotting the spectrum on each band
        if show_plot:
            if post_processing == "molecular mapping": # plotting the spectrum on each band
                if calculation == "SNR" or calculation == "corner plot":
                    band_flux.plot(wave_band, planet_spectrum_band.flux/DIT, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=band.replace('_', ' ')+f" (R={int(round(R))})") ; band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1) ; ymin = min(ymin, np.nanmin(planet_spectrum_band.flux/DIT)/2) ; ymax = max(ymax, np.nanmax(planet_spectrum_band.flux/DIT)*2) ; band_flux.set_ylim(ymin=max(ymin, 1e-3), ymax=ymax)
                elif calculation == "contrast":
                    band_flux.plot(wave_band, star_spectrum_band.flux/DIT, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=band.replace('_', ' ')+f" (R={int(round(R))})") ; band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1) ; ymin = min(ymin, np.nanmin(star_spectrum_band.flux/DIT)/2) ; ymax = max(ymax, np.nanmax(star_spectrum_band.flux/DIT)*2) ; band_flux.set_ylim(ymin=max(ymin, 1e-3), ymax=ymax)
            elif post_processing == "ADI+RDI":
                if calculation == "SNR":
                    band_flux.plot(wave_band, planet_spectrum_band.flux/DIT, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=band.replace('_', ' ')+f" ({round(np.nansum(planet_spectrum_band.flux/DIT))} e-/mn)") ; band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1) ; ymin = min(ymin, np.nanmin(planet_spectrum_band.flux/DIT)/2) ; ymax = max(ymax, np.nanmax(planet_spectrum_band.flux/DIT)*2) ; band_flux.set_ylim(ymin=max(ymin, 1e-3), ymax=ymax)
                elif calculation == "contrast":
                    if coronagraph is not None:
                        band_flux.plot(wave_band, star_spectrum_band.flux*star_transmission/DIT, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=band.replace('_', ' ')+f" ({round(np.nansum(star_transmission*star_spectrum_band.flux/DIT))} e-/mn)") ; band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1) ; ymin = min(ymin, np.nanmin(star_spectrum_band.flux*star_transmission/DIT)/2) ; ymax = max(ymax, np.nanmax(star_spectrum_band.flux*star_transmission/DIT)*2) ; band_flux.set_ylim(ymin=max(ymin, 1e-3), ymax=ymax)
                    else:
                        band_flux.plot(wave_band, star_spectrum_band.flux/DIT, color=cmap(nb), linestyle='-', linewidth=2, alpha=0.8, label=band.replace('_', ' ')+f" ({round(np.nansum(star_spectrum_band.flux/DIT))} e-/mn)") ; band_flux.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1) ; ymin = min(ymin, np.nanmin(star_spectrum_band.flux/DIT)/2) ; ymax = max(ymax, np.nanmax(star_spectrum_band.flux/DIT)*2) ; band_flux.set_ylim(ymin=max(ymin, 1e-3), ymax=ymax)
            
        #------------------------------------------------------------------------------------------------------------------------------------------------
        # Calculation of band contrast or SNR curves:
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Signal
        
        contrast = np.zeros_like(separation) ; contrast_wo_syst = np.zeros_like(separation) ; SNR = np.zeros_like(separation) ; t_syst = np.zeros_like(separation) ; cos_theta_est = np.zeros_like(separation) ; norm_d = np.zeros_like(separation)
                                    
        if post_processing == "molecular mapping": # Molecular Mapping
            signal = (alpha*cos_p - beta)*DIT # total number of useful e- for molecular mapping /DIT (in the FWHM or "fraction_core") (in e-/DIT/FWHM)
            if instru=="HARMONI" or apodizer != "NO_SP":
                PSF_profile[separation < iwa] *= 1e-4
                signal[separation < iwa] *= 1e-4 # Flux attenuated by a factor of 1e-4 due to Focal Plane Mask for HARMONI 
                    
        elif post_processing == "ADI+RDI": # ADI+RDI
            if coronagraph is not None:
                if calculation == "contrast":
                    signal = np.nansum(star_spectrum_band.flux) * fraction_PSF * radial_transmission ###
                elif calculation == "SNR":
                    signal = np.nansum(planet_spectrum_band.flux) * fraction_PSF * radial_transmission # planet flux in the PSF peak as a function of separation (in e-/DIT/pixel)

        # Detector noises
        
        sigma_dc_2  = dark_current * DIT * 60 # dark current photon noise (in e-/DIT/pixel)
        sigma_ron_2 = RON_eff**2              # effective read out noise (in e-/DIT/pixel)
        
        if config_data["type"] == "IFU_fiber": # detector noises must be multiplied by the number on which the fiber's signal is projected and integrated along the diretion perpendicular to the spectral dispersion of the detector
            sigma_dc_2  *= config_data['pixel_detector_projection'] # adds quadratically
            sigma_ron_2 *= config_data['pixel_detector_projection'] # (in e-/DIT/spaxel)
        
        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Stellar halo: calculation of the number of stellar e-/DIT at each separation as a function of the spectral channel in the considered band.
        
        sf, psf_profile = np.meshgrid(star_spectrum_band.flux, PSF_profile)
        star_flux = psf_profile * sf # star flux normalized by the PSF profile (mean flux) for each separation
        
        if post_processing == "molecular mapping": # Molecular Mapping   
            star_flux[star_flux>saturation_e] = saturation_e            # if saturation
            sigma_halo_2       = star_flux                              # stellar photon noise per spectral channel (in e-/DIT/pixel) for each separation
            t, _               = np.meshgrid(template, PSF_profile)
            sigma_halo_prime_2 = np.nansum(sigma_halo_2 * t**2, axis=1) # stellar photon noise projected in the CCF (in e-/DIT/spaxel)
        
        elif post_processing == "ADI+RDI": # ADI+RDI
            star_flux = np.nansum(star_flux, axis=1)         # integrated/photometric stellar flux
            star_flux[star_flux>saturation_e] = saturation_e # if saturation
            sigma_halo_2 = star_flux                         # stellar photon noise per spectral channel (in e-/DIT/pixel) for each separation

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Background
        
        if bkgd is not None:
            bkgd_flux = fits.getdata(f"sim_data/Background/{instru}/{bkgd}/background_{band}.fits") # (in e-/s/pixel)
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
            sigma_bkgd_prime_2 = np.nansum(sigma_bkgd_2 * template**2)       # background photon noise projected in the CCF (in e-/DIT/spaxel)
        elif post_processing == "ADI+RDI": # ADI+RDI
            if coronagraph is not None:
                sigma_bkgd_2 = np.nansum(sigma_bkgd_2) * radial_transmission # background photon noise per spectral channel (in e-/DIT/pixel) for each separation (radial_transmission due to the coronagraph)
            else:
                sigma_bkgd_2 = np.nansum(sigma_bkgd_2)                       # background photon noise per spectral channel (in e-/DIT/pixel) for each separation

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Systematic
        
        if systematic:
            sigma_syst_prime_2 *= np.nansum(star_spectrum_band.flux)**2 # systematic noise projected in the CCF (in e-/DIT/spaxel)
            
        #------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Saving quantities (in e-/DIT/FWHM)
        
        signal_bands.append(signal) ; planet_flux_bands.append(planet_spectrum_band.flux) ; star_flux_bands.append(star_spectrum_band.flux) ; wave_bands.append(wave_band)
        if post_processing == "molecular mapping":
            sigma_fund_2_bands.append(R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))
            sigma_halo_2_bands.append(R_corr*A_FWHM*(sigma_halo_prime_2))
            sigma_det_2_bands.append(R_corr*A_FWHM*(sigma_ron_2 + sigma_dc_2))
            sigma_bkg_2_bands.append(R_corr*A_FWHM*(sigma_bkgd_prime_2))

            if systematic:
                sigma_syst_2_bands.append(sigma_syst_prime_2)
            else:
                sigma_syst_2_bands.append(0) 
        elif post_processing == "ADI+RDI": # ADI+RDI
            sigma_fund_2_bands.append(R_corr*A_FWHM*(sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))
            sigma_halo_2_bands.append(R_corr*A_FWHM*(sigma_halo_2))
            sigma_det_2_bands.append(R_corr*A_FWHM*(sigma_ron_2 + sigma_dc_2))
            sigma_bkg_2_bands.append(R_corr*A_FWHM*(sigma_bkgd_2))

            if systematic:
                sigma_syst_2_bands.append(sigma_syst_2)
            else:
                sigma_syst_2_bands.append(0) 
                
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        #
        # Contrast calculation
        #
        if calculation == "contrast":
            
            if post_processing == "molecular mapping": # See Eq. (11) of Martos et al. 2025
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
                plt.figure(figsize=(10, 6), dpi=300)        
                ax1 = plt.gca()
                ax1.set_yscale('log')        
                ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
                ax1.set_xlim(0, separation[-1])
                ax1.set_xlabel(f"separation [{sep_unit}]", fontsize=14)
                ax1.tick_params(axis='both', labelsize=12)        
                ax2 = ax1.twinx()
                ax2.set_yscale('log')
                ax2.set_xlim(0, separation[-1])
                ax2.axvspan(0, iwa, color='black', alpha=0.3, lw=0)
                ax2.get_yaxis().set_visible(False)
                ax2.get_xaxis().set_visible(False)        
                if post_processing == "molecular mapping":
                    ax1.set_title(instru+f" noise contributions on {band.replace('_', ' ')}"+for_planet_name+"\n with "+"$t_{exp}$=" + str(round(exposure_time)) + "mn, $mag_*$("+band0+")=" + str(round(mag_star, 2)) + f', $T_p$={int(round(T_planet))}K and $R_c$ = {Rc}', fontsize=16)        
                    ax1.set_ylabel(r'contrast 5$\sigma_{CCF}$ / $\alpha_0$', fontsize=14)
                    ax1.plot(separation[separation>=iwa], contrast[separation>=iwa], 'k-', label=r"$\sigma_{CCF}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_halo_prime_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'r--', label=r"$\sigma'_{halo}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_ron_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'g--', label=r"$\sigma_{ron}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_dc_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'm--', label=r"$\sigma_{dc}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_bkgd_prime_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'b--', label=r"$\sigma'_{bkgd}$")
                    if systematic:
                        ax1.plot(separation[separation>=iwa], (5*np.sqrt(sigma_syst_prime_2)/(signal))[separation>=iwa], 'c--', label=r"$\sigma'_{syst}$")
                elif post_processing == "ADI+RDI":
                    ax1.set_title(instru+f" noise contributions on {band.replace('_', ' ')}"+for_planet_name+" \n with "+"$t_{exp}$=" + str(round(exposure_time)) + "mn and $mag_*$("+band0+")=" + str(round(mag_star, 2)), fontsize=16)        
                    ax1.set_ylabel(r'contrast 5$\sigma$ / $F_{p}$', fontsize=14)
                    ax1.plot(separation[separation>=iwa], contrast[separation>=iwa], 'k-', label=r"$\sigma_{tot}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_halo_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'r--', label=r"$\sigma_{halo}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_ron_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'g--', label=r"$\sigma_{ron}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_dc_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'm--', label=r"$\sigma_{dc}$")
                    ax1.plot(separation[separation>=iwa], (5*np.sqrt(R_corr*A_FWHM*(sigma_bkgd_2)*(NDIT))/(signal*NDIT))[separation>=iwa], 'b--', label=r"$\sigma_{bkgd}$")                    
                    if systematic:
                        ax1.plot(separation[separation>=iwa], (5*np.sqrt(sigma_syst_2)/(signal))[separation>=iwa], 'c--', label=r"$\sigma_{syst}$")
                if separation_planet is not None:
                    if separation_planet > 2 * owa:
                        ax1.set_xscale('log')
                        ax1.set_xlim(iwa, separation[-1])
                    if mag_planet is None:
                        ax1.axvline(separation_planet, color="black", linestyle="--", label=f'{planet_name}')
                    else:
                        if planet_to_star_flux_ratio > ax1.get_ylim()[1] or (planet_to_star_flux_ratio > ax1.get_ylim()[0] and planet_to_star_flux_ratio < ax1.get_ylim()[1]):
                            y_text = planet_to_star_flux_ratio/1.5
                            if separation_planet > (iwa+owa)/2:
                                x_text    = separation_planet - 0.1 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "center"
                            else:
                                x_text    = separation_planet + 0.025 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "right"
                        else :
                            y_text = planet_to_star_flux_ratio*1.5
                            if separation_planet > (iwa+owa)/2:
                                x_text    = separation_planet - 0.1 * separation[-1]
                                leg_y_pos = "upper"
                                leg_x_pos = "right"
                            else:
                                x_text    = separation_planet + 0.025 * separation[-1]
                                leg_y_pos = "lower"
                                leg_x_pos = "right"
                        leg_loc = leg_y_pos + " " + leg_x_pos
                        ax1.plot([separation_planet, separation_planet], [planet_to_star_flux_ratio, planet_to_star_flux_ratio], 'ko')
                        ax1.annotate(f"{planet_name}", (x_text, y_text), fontsize=12)
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
            
            # Adding the contrast curve of the band to the list
            contrast_bands.append(contrast)
        
        #
        # SNR calculation
        #
        elif calculation == "SNR" or calculation == "corner plot":
            
            if post_processing == "molecular mapping": # See Eq. (10) of Martos et al. 2025
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
                    print(f" S/N at {separation_planet:.1f} {sep_unit} = {SNR[idx]:.1f}")
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
                            d                  = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF + noise + M_HF[i]*NDIT*A_FWHM*star_flux # spectrum at the planet's location: see Eq.(18) of Martos et al. 2025
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

            # Adding the SNR curve of the band to the list
            SNR_bands.append(SNR)
        
        #
        # Corner plot:
        #
        if calculation == "corner plot":
            Mp_Sp              = NDIT*planet_spectrum_band.flux*fraction_PSF        # planet flux (with modulations, if any) in e-/FWHM
            Mp_Sp_HF, Mp_Sp_LF = filtered_flux(Mp_Sp/trans, R, Rc, filter_type)     # filtered planet flux
            star_flux          = PSF_profile[idx]*star_spectrum_band.flux*NDIT      # stellar flux in e-/pixel at separation idx (of the planet)
            star_HF, star_LF   = filtered_flux(star_flux/trans, R, Rc, filter_type) # filtered star flux
            d_planet           = trans*Mp_Sp_HF - trans*star_HF*Mp_Sp_LF/star_LF    # spectrum at the planet's location: see Eq.(18) of Martos et al. 2025        
            sigma_l            = np.sqrt(R_corr[idx]*A_FWHM*NDIT*(sigma_halo_2[idx] + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) # total noise per spectral channel (in e-/Afwhm/spectral canal)
            vsini_planet       = planet_spectrum.vsini
            rv_planet          = planet_spectrum.rv
            SNR_CCF            = SNR[idx]
            uncertainties      = parameters_estimation(instru=instru, band=band_only, target_name=planet_name, wave=wave_band, d_planet=d_planet, star_flux=star_flux, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=model_planet, logL=True, method_logL="classic", sigma_l=sigma_l, weight=None, pca=pca, precise_estimate=False, SNR_estimate=False, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, T_arr=None, lg_arr=None, vsini_arr=None, rv_arr=None, SNR_CCF=SNR_CCF, show=show_plot, verbose=show_plot, stellar_component=True, degrade_resolution=True, force_new_est=False, d_planet_sim=False, save=False, fastcurves=True, exposure_time=exposure_time, star_HF=star_HF, star_LF=star_LF, wave_inter=wave_instru)
            uncertainties_bands.append(uncertainties)
            
        # plot of t_syst : see Eq.(14) of Martos et al. 2025
        if show_plot and show_t_syst and systematic:
            t_syst = DIT*R_corr*A_FWHM*(sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2)/sigma_syst_prime_2 # en min 
            plt.figure(dpi=300) ; plt.plot(separation, t_syst, 'b') ; plt.ylabel('$t_{syst}$ [mn]', fontsize = 14) ; plt.xlabel(f'separation [{sep_unit}]', fontsize = 14) ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.title("$t_{syst}$"+f" on {band}"+"\n with $mag_*$("+band0+")=" + str(round(mag_star, 2)), fontsize = 14) ; plt.plot([separation[0], separation[-1]], [exposure_time, exposure_time], 'r-') ; plt.yscale('log') ; plt.legend(["$t_{syst}$", "$t_{exp}$ ="+f"{round(exposure_time)} mn"]) 

    #------------------------------------------------------------------------------------------------
    # PLOTS:
    #------------------------------------------------------------------------------------------------
        
    # Contrast:
        
    if calculation == "contrast" and show_plot:
        plt.figure(figsize=(10, 6), dpi=300)        
        ax1 = plt.gca()
        ax1.set_yscale('log')        
        ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)        
        ax1.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        ax1.set_title(f"{instru} contrast curves{for_planet_name} with {post_processing}\n" + f"with $t_{{exp}}$ = {int(round(exposure_time))} mn, $mag_*$({band0}) = {round(mag_star, 1)}" + f" and $T_p$ = {int(round(T_planet))} K", fontsize=16)        
        ax1.set_xlabel(f"separation [{sep_unit}]", fontsize=14)
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
                ax1.axvline(separation_planet, color="black", linestyle="--", label=f'{planet_name}')
            else:
                if planet_to_star_flux_ratio > ax1.get_ylim()[1] or (planet_to_star_flux_ratio > ax1.get_ylim()[0] and planet_to_star_flux_ratio < ax1.get_ylim()[1]):
                    y_text = planet_to_star_flux_ratio/1.5
                    if separation_planet > (IWA+OWA)/2:
                        x_text    = separation_planet - 0.1 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "center"
                    else:
                        x_text    = separation_planet + 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "right"
                else :
                    y_text = planet_to_star_flux_ratio*1.5
                    if separation_planet > (IWA+OWA)/2:
                        x_text    = separation_planet - 0.1 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "upper"
                        leg_x_pos = "right"
                    else:
                        x_text    = separation_planet + 0.025 * max(np.max(arr) for arr in separation_bands)
                        leg_y_pos = "lower"
                        leg_x_pos = "right"
                leg_loc = leg_y_pos + " " + leg_x_pos
                ax1.plot([separation_planet, separation_planet], [planet_to_star_flux_ratio, planet_to_star_flux_ratio], 'ko')
                ax1.annotate(f"{planet_name}", (x_text, y_text), fontsize=12)
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
    
    #------------------------------------------------------------------------------------------------

    # SNR:
            
    elif calculation == "SNR" and show_plot:
        
        if channel and instru == 'MIRIMRS':
            # Multiply exposure time by 3 to account for the 3 sub-bands per channel
            exposure_time *= 3
            SNR_chan        = []
            separation_chan = []
            name_chan       = []
            # MIRIMRS has 4 channels, each composed of 3 bands (SHORT, MEDIUM, LONG)
            n_channels        = 4
            bands_per_channel = 3
            for ch in range(n_channels):
                start_idx = ch * bands_per_channel
                end_idx   = start_idx + bands_per_channel
                # Combine SNR for each channel using quadrature sum
                snr_combined = np.sqrt(np.nansum(np.array(SNR_bands[start_idx:end_idx])**2, axis=0))
                SNR_chan.append(snr_combined)
                # Use the separation values from the SHORT band as reference for each channel
                separation_chan.append(separation_bands[start_idx])
                # Name each channel for later display
                name_chan.append(f"channel {ch + 1}")
            # Find the global maximum SNR
            SNR_max = max(np.nanmax(snr) for snr in SNR_chan)
            # If a planet separation is provided, find the corresponding maximum SNR
            SNR_max_planet = 0.
            name_max_SNR = ""
            if separation_planet is not None:
                for i in range(n_channels):
                    idx = np.where(separation_chan[i] == separation_planet)[0][0]
                    if SNR_chan[i][idx] > SNR_max_planet:
                        SNR_max_planet = SNR_chan[i][idx]
                        name_max_SNR   = name_chan[i]
            # Update bands and separations for further processing or plotting
            separation_bands = separation_chan
            SNR_bands        = SNR_chan
            name_bands       = name_chan

        if separation_planet is not None  and verbose:
            print(f"\nMAX S/N (at {separation_planet:.1f} {sep_unit}) = {SNR_max_planet:.1f} for {name_max_SNR.replace('_', ' ')}")
                
        plt.figure(figsize=(10, 6), dpi=300) 
        ax1 = plt.gca()
        ax1.grid(which='both', linestyle=':', color='gray', alpha=0.5)     
        ax1.set_xlim(0, max(np.max(arr) for arr in separation_bands))
        ax1.set_title(f"{instru} S/N curves{for_planet_name} with $t_{{exp}}$ = {round(exposure_time)} mn,\n" + f"$mag_*$({band0}) = {round(mag_star, 2)}, $mag_p$({band0}) = {round(mag_planet, 2)}, " + f"$T_p$ = {int(round(T_planet))}K ({model_planet})", fontsize=16)
        ax1.set_xlabel(f"separation [{sep_unit}]", fontsize=14)
        ax1.set_ylabel('S/N', fontsize=14)        
        ax1.axvspan(0, IWA, color='black', alpha=0.3, lw=0)
        for i in range(len(SNR_bands)):
            if band_only is not None:
                color_idx = [nb for nb, band in enumerate(config_data["gratings"]) if band == band_only][0]
                ax1.plot(separation_bands[i][separation_bands[i] >= IWA], SNR_bands[i][separation_bands[i] >= IWA], label=name_bands[i].replace('_', ' '), color=cmap(color_idx), linewidth=2, alpha=0.8)
            else:
                ax1.plot(separation_bands[i][separation_bands[i] >= IWA], SNR_bands[i][separation_bands[i] >= IWA],label=name_bands[i].replace('_', ' '), color=cmap(i), linewidth=2, alpha=0.8)
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
        
    #------------------------------------------------------------------------------------------------
    # RETURNS:
    #------------------------------------------------------------------------------------------------
    
    if calculation == "contrast":
        return name_bands, separation_bands, contrast_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    elif calculation == "SNR":
        return name_bands, separation_bands, SNR_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    elif calculation == "corner plot":
        return name_bands, separation_bands, uncertainties_bands, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves init
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation="contrast", instru="HARMONI", exposure_time=120, mag_star=7, mag_planet=None, band0="K", model_planet="BT-Settl", T_planet=1000, lg_planet=4.0, model_star="BT-NextGen", T_star=6000, lg_star=4.0, rv_star=0, rv_planet=0, vsini_star=0, vsini_planet=0, 
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
    model_planet (str)       : planet's model. Default: "BT-Settl".
    T_planet (float)         : planet's temperature (in K). Default: 1000.
    lg_planet (float)        : planet's gravity surface (in dex(cm/s2)). Default: 4.0.
    model_star (str)         : star's model. Default: "BT-NextGen".
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
    
    if calculation not in ["contrast", "SNR", "corner plot"]:
        raise KeyError(f"{calculation} is not a valid value for calculation. Available values: 'contrast', 'SNR' or 'corner plot'")
    
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
        star_spectrum = load_star_spectrum(T_star, lg_star, model="BT-NextGen") # star spectrum (BT-NextGen GNS93)
        try:
            star_spectrum.crop(0.98*min(globals()["lmin_"+band0], globals()["lmin_"+instru]), 1.02*max(globals()["lmax_"+band0], globals()["lmax_"+instru]))
        except:
            raise KeyError(f"{band0} is not a considered band to define the magnitude, please choose among: {bands}, {instrus}")
        # Rotational broadening of the spectrum [km/s]
        if vsini_star > 0: # the wavelength axis needs to be evenly spaced
            star_spectrum = star_spectrum.evenly_spaced(renorm=False)
            star_spectrum = star_spectrum.broad(vsini_star)
        # Doppler shifting the spectrum [km/s]
        star_spectrum = star_spectrum.doppler_shift(rv_star)
            
    # if not input, load the planet spectrum
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model_planet, instru=instru) # planet spectrum: class Spectrum(wavel, flux, R, T) in J/s/m²/µm according to the considered model
        try:
            planet_spectrum.crop(0.98*min(globals()["lmin_"+band0], globals()["lmin_"+instru]), 1.02*max(globals()["lmax_"+band0], globals()["lmax_"+instru]))
        except:
            raise KeyError(f"{band0} is not a considered band to define the magnitude, please choose among: {bands}, {instrus}")
        # Rotational broadening of the spectrum [km/s]
        if vsini_planet > 0: # the wavelength axis needs to be evenly spaced
            planet_spectrum = planet_spectrum.evenly_spaced(renorm=False)
            planet_spectrum = planet_spectrum.broad(vsini_planet)
        # Doppler shifting the spectrum [km/s]
        planet_spectrum = planet_spectrum.doppler_shift(rv_planet)

    # FastCurves calculation
    name_bands, separation, curves, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands = FastCurves_main(calculation=calculation, instru=instru, exposure_time=exposure_time, mag_star=mag_star, band0=band0, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, tellurics=tellurics, 
                                                                                                                                                                 apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=systematic, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc, channel=channel, planet_name=planet_name, separation_planet=separation_planet, mag_planet=mag_planet, show_plot=show_plot, verbose=verbose, 
                                                                                                                                                                 post_processing=post_processing, sep_unit=sep_unit, bkgd=bkgd, Rc=Rc, filter_type=filter_type, input_DIT=input_DIT, band_only=band_only, return_SNR_planet=return_SNR_planet)
    
    if verbose:
        print(f'\nFastCurves {calculation} calculation took {round(time.time()-time1, 1)} s')

    # For FASTYIELD
    if return_SNR_planet:
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1e3 # switching the angular separation unit (from arcsec to mas)
        SNR_planet = np.zeros((len(name_bands))) ; signal_planet = np.zeros((len(name_bands))) ; sigma_fund_planet = np.zeros((len(name_bands))) ; sigma_syst_planet = np.zeros((len(name_bands)))
        for nb, band in enumerate(name_bands): # retrieving the values at the planet separation
            idx = np.where(separation[nb]==separation_planet)[0][0]
            SNR_planet[nb]        = curves[nb][idx]
            signal_planet[nb]     = signal_bands[nb][idx]
            sigma_fund_planet[nb] = np.sqrt(sigma_fund_2_bands[nb][idx])
            if systematic:
                sigma_syst_planet[nb] = np.sqrt(sigma_syst_2_bands[nb][idx])
        return name_bands, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, np.array(DIT_bands)
    
    # For deep analysis (regimes, t_syst, etc. calculations)
    elif return_quantity:
        return name_bands, separation, signal_bands, sigma_syst_2_bands, sigma_fund_2_bands, sigma_halo_2_bands, sigma_det_2_bands, sigma_bkg_2_bands, DIT_bands, planet_flux_bands, star_flux_bands, wave_bands
    
    # Standard returns
    else:
        return name_bands, separation, curves






