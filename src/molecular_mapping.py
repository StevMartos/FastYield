from src.spectrum import *



def stellar_high_filtering(cube, R, Rc, filter_type, outliers=False, sigma_outliers=5, verbose=True, renorm_cube_res=False, only_high_pass=False, stell=None, show=False):
    """
    Post-processing filtering method according to molecular mapping, see Appendix B of Martos et al. (2025)

    Parameters
    ----------
    cube : 3d-array
        Data cube.
    R : float
        Spectral resolution of the cube.
    Rc : float
        Cut-off resolution of the filter. If Rc is None, no filter will be applied.
    filter_type : str
        Type of filter used.
    outliers : bool, optional
        To filter outliers. The default is False.
    sigma_outliers : float, optional
        Sigma value of the outliers filtering method (sigma clipping). The default is 5.
    verbose : bool, optional
        If True, prints additional information. The default is True.
    renorm_cube_res : bool, optional
        If True, renormalizes every spaxel of the final cube product by their norm, directly giving the correlation intensity. The default is False.
    only_high_pass : bool, optional
        If True, applies only a high-pass filter on the cube instead of also subtracting the stellar component. The default is False.
    stell : 1D-array, optional
        Stellar spectrum. If None, it is estimated by summing the cube across spatial dimensions. The default is None.
    show : bool, optional
        If True, shows plots of the filtering process for each spaxel. The default is False.

    Returns
    -------
    cube_res : 3d-array
        Residual data cube after filtering.
    cube_M : 3d-array
        Estimated stellar modulation function.
    """
    cube = np.copy(cube)
    NbChannel, NbLine, NbColumn = cube.shape
    if stell is None:
        stell = np.nansum(cube, (1, 2)) # estimated stellar spectrum
    S = np.reshape(cube, (NbChannel, NbLine*NbColumn))
    cube_M = np.copy(S) ; m = 0
    for k in range(S.shape[1]):
        if not all(np.isnan(S[:, k])):
            if show:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3), layout="constrained", gridspec_kw={'wspace': 0.05, 'hspace':0}, dpi=300) ; ax[0].set_xlabel("wavelength axis") ; ax[0].set_ylabel("modulation (normalized)") ; ax[1].set_yscale('log') ; ax[1].set_xscale('log') ; ax[1].set_xlabel("resolution frequency R") ; ax[1].set_ylabel("PSD") ; ax[0].plot(S[:, k]/np.sqrt(np.nansum(S[:, k]**2)), 'darkred', zorder=10) ; S_spectrum = Spectrum(None, S[:, k][~np.isnan(S[:, k])], R, None) ; res, psd = S_spectrum.get_psd(smooth=0) ; ax[1].plot(res, psd, "r", label="raw (LF+HF)") ; S_HF = filtered_flux(S[:, k], R=R, Rc=Rc, filter_type=filter_type)[0] ; ax[0].plot(S_HF/np.sqrt(np.nansum(S_HF**2)), 'seagreen') ; S_spectrum = Spectrum(None, S_HF[~np.isnan(S_HF)], R, None) ; res_HF, psd_HF = S_spectrum.get_psd(smooth=0) ; ax[1].plot(res_HF, psd_HF, "g", label="HF") ; plt.title(f" F = {round(psd_HF[np.abs(res_HF-Rc).argmin()]/psd[np.abs(res-Rc).argmin()], 3)}")
            if only_high_pass:
                S_LF = filtered_flux(S[:, k], R, Rc, filter_type)[1]
                M    = S_LF / stell
            else:
                M = filtered_flux(S[:, k], R, Rc, filter_type)[1] / filtered_flux(stell, R, Rc, filter_type)[1]
            cube_M[:, k] = M #  Estimated modulations, assuming that stell is the real observed stellar spectrum
            if outliers:
                sg      = sigma_clip(S[:, k] - stell * M, sigma=sigma_outliers)
                S[:, k] = np.array(np.ma.masked_array(sg, mask=sg.mask).filled(np.nan))
            else:
                S[:, k] = S[:, k] - stell * M
            m += np.nansum(M)
            if show:
                ax[0].plot(S[:, k]/np.sqrt(np.nansum(S[:, k]**2)), "b") ; S_spectrum = Spectrum(None, S[:, k][~np.isnan(S[:, k])], R, None) ; res, psd = S_spectrum.get_psd(smooth=0) ; ax[1].plot(res, psd, "b", label="post MM filtering method") ; plt.legend() ; plt.show()
    if verbose:
        print("\n sum(M) =", round(m/(NbChannel), 3)) # must be ~ 1 (sanity check)
    cube_res = S.reshape((NbChannel, NbLine, NbColumn))
    cube_M   = cube_M.reshape((NbChannel, NbLine, NbColumn))
    cube_res[cube_res == 0] = np.nan
    cube_M[cube_M == 0]     = np.nan
    if renorm_cube_res: # renormalizing the spectra of every spaxel in order to directly have the correlation strength
        for i in range(NbLine):
            for j in range(NbColumn): # for every spaxel
                if not all(np.isnan(cube_res[:, i, j])): # ignoring nan values
                    cube_res[:, i, j] = cube_res[:, i, j]/np.sqrt(np.nansum(cube_res[:, i, j]**2))
    return cube_res, cube_M



def molecular_mapping_rv(instru, S_res, star_flux, T_planet, lg_planet, model, wave, trans, R, Rc, filter_type, rv=None, vsini_planet=0, verbose=True, template=None, pca=None, degrade_resolution=True, stellar_component=True, epsilon=0.8, fastbroad=True):
    """
    Cross-correlating the residual cube S_res with templates, giving the CCF.

    Parameters
    ----------
    instru : str
        Instrument's name.
    S_res : 3d-array
        Residual cube.
    star_flux : 1d-array
        Flux of the star.
    T_planet : float
        Planet's temperature.
    lg_planet : float
        Planet's surface gravity.
    model : str
        Planet's spectrum model.
    wave : 1d-array
        Wavelength axis.
    trans : 1d-array
        Total system transmission.
    R : float
        Spectral resolution of the cube.
    Rc : float
        Cut-off resolution of the filter. If Rc = None, no filter will be applied.
    filter_type : str
        Type of filter used.
    rv : float or array-like, optional
        Planet's radial velocity. If None, a range of values will be used. Default is None.
    vsini_planet : float, optional
        Planet's rotational speed. Default is 0.
    verbose : bool, optional
        Whether to print progress messages. Default is True.
    template : object, optional
        Template to use for cross-correlation. If None, a template will be generated. Default is None.
    pca : object, optional
        PCA object used to subtract components from the template. Default is None.
    epsilon : float, optional
        Parameter for template generation. Default is 0.8.
    fastbroad : bool, optional
        Whether to use fast broadening for template generation. Default is True.

    Returns
    -------
    CCF : 3d-array or 2d-array
        Cross-correlation function array.
    rv : array-like, optional
        Radial velocity values, if multiple values are used.
    """
    # Generate template if not provided
    if template is None:
        template = get_template(instru=instru, wave=wave, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=0, R=R, Rc=Rc, filter_type=filter_type, epsilon=epsilon, fastbroad=fastbroad)
    # Degrade template to instrumental resolution if required
    if degrade_resolution:
        template = template.degrade_resolution(wave, renorm=False)
    else:
        template = template.interpolate_wavelength(wave, renorm=False)
    # Return if template only contains NaNs (e.g. if the crop of the molecular templates left only NaNs)
    if all(np.isnan(template.flux)):
        return rv, CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet
    # Filter the template
    template_HF = template.copy() ; template_LF = template.copy()
    template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type)
    # Handle stellar component if needed
    if stellar_component and Rc is not None:
        if star_flux is None:
            raise KeyError("star_flux is not defined for the stellar component!")
        f = interp1d(wave[~np.isnan(star_flux / trans)], (star_flux / trans)[~np.isnan(star_flux / trans)], bounds_error=False, fill_value=np.nan)
        sf = f(wave)
        star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type)
    NbChannel, NbLine, NbColumn = S_res.shape
    if rv is None:
        rv = np.linspace(-50, 50, 201)
    elif not isinstance(rv, np.ndarray):
        rv = np.array([rv])
    CCF = np.zeros((len(rv), NbLine, NbColumn))
    for k in range(len(rv)):
        if verbose:
            print(" CCF for: rv = ", rv[k], " km/s & Tp =", T_planet, "K & lg = ", lg_planet)
        template_HF_shift = template_HF.doppler_shift(rv[k], renorm=False).flux # shifting the tempalte
        template_LF_shift = template_LF.doppler_shift(rv[k], renorm=False).flux # shifting the tempalte
        template_shift = trans*template_HF_shift # it should be almost the same thing with or without the residual star flux, for the 2D CCF it is not considered by default beacause this residual term could project on systematics and accentuate the spatial systematic noise contribution
        if stellar_component and Rc is not None: # adding the stellar component (if required)
            template_shift += -trans * star_HF * template_LF_shift / star_LF
        if pca is not None: # subtraction of the PCA's modes to the template
            template0_shift = np.copy(template_shift)
            n_comp_sub = pca.n_components
            for nk in range(n_comp_sub): 
                template_shift -= np.nan_to_num(np.nansum(template0_shift * pca.components_[nk]) * pca.components_[nk])
        template_shift /= np.sqrt(np.nansum(template_shift**2))  # normalizing the template
        for i in range(NbLine):
            for j in range(NbColumn): # for every spaxel
                if not all(np.isnan(S_res[:, i, j])): # ignoring all nan values spaxels
                    d = np.copy(S_res[:, i, j])
                    t = np.copy(template_shift)
                    t[np.isnan(d)] = np.nan
                    t /= np.sqrt(np.nansum(t**2))  # normalizing the template
                    CCF[k, i, j] = np.nansum(d*t) # cross-correlation between the residual signal and the template
                    #plt.figure(dpi=300) ; plt.title(f"CCF[k, i, j] = {CCF[k, i, j]}"); plt.plot(wave, d / np.sqrt(np.nansum(d**2))) ; plt.plot(wave, t / np.sqrt(np.nansum(t**2))) ; plt.show()
    CCF[CCF == 0] = np.nan
    if len(rv) == 1:
        return CCF[0], template_shift
    else:
        return CCF, rv



########################################################################################################################################################################################################################################################################################################################################################################################################



def correlation_rv(instru, d_planet, d_bkg, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, large_rv=False, rv=None, rv_planet=None, vsini_planet=None, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, stellar_component=True, degrade_resolution=True, disable_tqdm=False, show=True, smooth_PSD=1, target_name=None, compare_data=False, cut_fringes=False, Rmin=None, Rmax=None):
    """
    Perform radial velocity (RV) correlation analysis between a template and the observed data.

    Parameters:
    instru (str): Instrument name.
    d_planet (array-like): Planet signal data.
    d_bkg (array-like): Background data.
    star_flux (array-like): Stellar flux data.
    wave (array-like): Wavelength array.
    trans (array-like): Transmission spectrum.
    T_planet (float): Temperature of the planet.
    lg_planet (float): Logarithmic gravity of the planet.
    model (str): Model name for the template.
    R (float): Spectral resolution.
    Rc (float): Cutoff resolution for filtering.
    filter_type (str): Type of filter used in high pass filtering.
    large_rv (bool, optional): If True, considers a large RV range. Defaults to False.
    rv (array-like, optional): RV values for correlation. Defaults to None.
    rv_planet (float, optional): Known radial velocity of the planet. Defaults to None.
    vsini_planet (float, optional): Rotational broadening of the planet. Defaults to None.
    pca (object, optional): PCA model for subtracting modes. Defaults to None.
    template (object, optional): Precomputed template spectrum. Defaults to None.
    epsilon (float, optional): Limb-darkening coefficient. Defaults to 0.8.
    fastbroad (bool, optional): Use fast broadening method. Defaults to True.
    logL (bool, optional): Calculate log-likelihood. Defaults to False.
    method_logL (str, optional): Method for log-likelihood calculation. Defaults to "classic".
    sigma_l (array-like, optional): Noise standard deviation. Defaults to None.
    weight (array-like, optional): Weighting function for the data. Defaults to None.
    stellar_component (bool, optional): Include stellar component in the template. Defaults to True.
    degrade_resolution (bool, optional): Degrade template to instrumental resolution. Defaults to True.
    disable_tqdm (bool, optional): Disable progress bar. Defaults to False.
    show (bool, optional): Show resulting plots. Defaults to True.
    smooth_PSD (int, optional): Smoothing parameter for PSD plots. Defaults to 1.
    target_name (str, optional): Name of the target. Defaults to None.
    compare_data (bool, optional): If True, compare data sets. Defaults to False.
    cut_fringes (bool, optional): Apply fringe cutting to the data. Defaults to False.
    Rmin (float, optional): Minimum resolution for fringe cutting. Defaults to None.
    Rmax (float, optional): Maximum resolution for fringe cutting. Defaults to None.

    Returns:
    tuple: RV array, cross-correlation functions (CCF) for planet and background, correlations, log-likelihood values, and noise estimates.
    """
    # Define RV grid if not provided
    if rv is None:
        if large_rv:
            rv = np.linspace(-10000, 10000, 4001)
            #rv = np.linspace(-100, 100, 401)
        else:
            rv = np.linspace(-1000, 1000, 4001)
    # Initialize correlation arrays
    CCF_planet = np.zeros((len(rv))) ; corr_planet = np.zeros((len(rv))) ; corr_auto = np.zeros((len(rv))) ; corr_auto_noise = np.zeros((len(rv))) ; logL_planet = np.zeros((len(rv))) ; sigma_planet = None
    if d_bkg is not None: # Initialize background CCF if background data is provided
        CCF_bkg = np.zeros((len(d_bkg), len(rv)))
    else:
        CCF_bkg = None
    if not compare_data: # If not comparing data, calculating a template
        # Generate template if not provided
        if template is None:
            template_raw = get_template(instru=instru, wave=wave, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=0, R=R, Rc=Rc, filter_type=filter_type, epsilon=epsilon, fastbroad=fastbroad)
            template     = template_raw.copy()
        else:
            template_raw = template.copy()
        # Degrade template to instrumental resolution if required
        if degrade_resolution:
            template = template.degrade_resolution(wave, renorm=False)
        else:
            template = template.interpolate_wavelength(wave, renorm=False)
        # Return if template only contains NaNs (i.e. if the crop of the molecular templates left only NaNs)
        if all(np.isnan(template.flux)):
            return rv, CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet
        # Filter the template
        template_HF = template.copy()
        template_LF = template.copy()
        template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type)
        
        # Handle stellar component if needed
        if stellar_component and Rc is not None:
            if star_flux is None:
                raise KeyError("star_flux is not defined for the stellar component!")
            f  = interp1d(wave[~np.isnan(star_flux / trans)], (star_flux / trans)[~np.isnan(star_flux / trans)], bounds_error=False, fill_value=np.nan)
            sf = f(wave)
            if instru == "HiRISE": # Handle filtering edge effects due to gaps between orders
                _, star_LF = filtered_flux(sf, R, Rc, filter_type)
                nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005 / np.nanmean(np.diff(wave)))
                f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan)
                sf[nan_values] = f(wave[nan_values])
            star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type)
        # Shift the template at the known planet RV in order to mimic the data and to calculate/estimate the auto-correlation
        if rv_planet is not None:
            template_shift = template_raw.doppler_shift(rv_planet, renorm=False)
            if degrade_resolution:
                template_shift = template_shift.degrade_resolution(wave, renorm=False)
            else:
                template_shift = template_shift.interpolate_wavelength(wave, renorm=False)
            template_HF_shift, template_LF_shift = filtered_flux(template_shift.flux, R, Rc, filter_type)
            template_shift = trans * template_HF_shift
            if stellar_component and Rc is not None: # adding the stellar component (if required)
                template_shift += -trans * star_HF * template_LF_shift / star_LF
            if pca is not None: # Subtract PCA components (if required)
                template_shift0 = np.copy(template_shift)
                n_comp_sub = pca.n_components
                for nk in range(n_comp_sub):
                    template_shift -= np.nan_to_num(np.nansum(template_shift0 * pca.components_[nk]) * pca.components_[nk])
            if cut_fringes: # Cut fringes frequencies (if required)
                template_shift = cut_spectral_frequencies(template_shift, R, Rmin, Rmax, target_name=target_name)
            if weight is not None:
                template_shift *= weight
            template_shift[np.isnan(d_planet)] = np.nan # Masking invalid regions
    else: # The data to compare with is assimiled to the "template"
        if template is None:
            raise KeyError("You need to input the data to compare with: template=None")
        template_HF = Spectrum(wave, template, R, T_planet, lg_planet)
        template_shift = np.copy(template)
        if weight is not None:
            template_shift *= weight
    # Loop through RV values and compute correlations
    for i in tqdm(range(len(rv)), desc="correlation_rv()", disable=disable_tqdm):
        template_HF_shift = template_HF.doppler_shift(rv[i], renorm=False).flux
        if not compare_data:
            t = trans * template_HF_shift
            # adding the stellar component (if required)
            if stellar_component and Rc is not None:
                template_LF_shift = template_LF.doppler_shift(rv[i], renorm=False).flux
                t += -trans * star_HF * template_LF_shift / star_LF
            # Subtract PCA components (if required)
            if pca is not None:
                t0 = np.copy(t)
                n_comp_sub = pca.n_components
                for nk in range(n_comp_sub):
                    t -= np.nan_to_num(np.nansum(t0 * pca.components_[nk]) * pca.components_[nk])
            # Cut fringes frequencies (if required)
            if cut_fringes:
                t = cut_spectral_frequencies(t, R, Rmin, Rmax, target_name=target_name)
        else:
            t = template_HF_shift
        d_p = np.copy(d_planet)
        # Taking the weight function into account (if required)
        if weight is not None:
            d_p *= weight
            t   *= weight
        # Masking invalid regions
        d_p[d_p == 0]    = np.nan
        t[t == 0]        = np.nan
        t[np.isnan(d_p)] = np.nan
        d_p[np.isnan(t)] = np.nan
        # Computations 
        t /= np.sqrt(np.nansum(t**2)) # normalizing the template
        CCF_planet[i]  = np.nansum(d_p * t) # CCF signal
        corr_planet[i] = np.nansum(d_p * t) / np.sqrt(np.nansum(d_p**2)) # correlation strength
        # auto-correlation (if possible)
        if rv_planet is not None: 
            corr_auto[i] = np.nansum(template_shift * t) / np.sqrt(np.nansum(template_shift[~np.isnan(t)]**2))
        # CCF noise
        if d_bkg is not None:
            for j in range(len(d_bkg)):
                CCF_bkg[j, i] = np.nansum(d_bkg[j] * t)
        # logL computation
        if logL:
            logL_planet[i] = get_logL(d_p, t, sigma_l, method=method_logL)
        # Sanity check
        #plt.figure(dpi=300) ; plt.plot(wave, d_p / np.sqrt(np.nansum(d_p**2))) ; plt.plot(wave, t / np.sqrt(np.nansum(t**2))) ; plt.title(f" N = {len(d_p[(~np.isnan(d_p))&(~np.isnan(t))])}") ; plt.show()
    
    # END OF THE LOOP
    if rv_planet is not None:
        # Compute sigma for the planet if RV is known
        if sigma_l is not None:
            sigma_planet = np.sqrt(np.nansum(sigma_l**2 * template_shift**2)) / np.sqrt(np.nansum(template_shift**2))
        
        # Display template and extracted planet spectrum
        if show:
            
            d_planet = np.copy(d_planet)
            if weight is not None:
                d_planet *= weight
            d_planet[d_planet == 0] = np.nan ; template_shift[template_shift == 0] = np.nan ; template_shift[np.isnan(d_planet)] = np.nan ; d_planet[np.isnan(template_shift)] = np.nan
            
            # Set template to approximately the same signal as the data
            res_d_planet, psd_d_planet             = calc_psd(wave, d_planet, R, smooth=0)
            res_template_shift, psd_template_shift = calc_psd(wave, template_shift, R, smooth=0)
            if Rc is not None:
                if R / 10 > Rc:
                    ratio = np.sqrt(np.nansum(psd_d_planet[(res_d_planet > Rc) & (res_d_planet < R / 10)]) / np.nansum(psd_template_shift[(res_template_shift > Rc) & (res_template_shift < R / 10)]))
                else:
                    ratio = np.sqrt(np.nansum(psd_d_planet[(res_d_planet > Rc) & (res_d_planet < R)]) / np.nansum(psd_template_shift[(res_template_shift > Rc) & (res_template_shift < R)]))
            else:
                ratio = np.nansum(d_planet*template_shift) / np.sqrt(np.nansum(template_shift**2)) / np.sqrt(np.nansum(template_shift**2))  # assuming cos_p = 1
            template_shift *= ratio
            
            # Create subplots for data visualization
            fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=300, gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
            if target_name is not None:
                tn = target_name.replace("_", " ") + " "
            if compare_data:
                fig.suptitle(f"{instru} {tn}data sets, with R={int(R)} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d_planet * template_shift) / np.sqrt(np.nansum(d_planet**2) * np.nansum(template_shift**2)), 3)}", fontsize=20)
            else:
                if "mol_" in model:
                    fig.suptitle(f"{instru} {tn}data and {model} template, with $T$={round(T_planet)}K, rv={round(rv_planet, 1)}km/s, vsini={round(vsini_planet, 1)}km/s, R={int(R)} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d_planet * template_shift) / np.sqrt(np.nansum(d_planet**2) * np.nansum(template_shift**2)), 3)}", fontsize=20)
                else:
                    fig.suptitle(f"{instru} {tn}data and {model} template, with $T$={round(T_planet)}K, lg={round(lg_planet, 1)}, rv={round(rv_planet, 1)}km/s, vsini={round(vsini_planet, 1)}km/s, R={int(R)} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d_planet * template_shift) / np.sqrt(np.nansum(d_planet**2) * np.nansum(template_shift**2)), 3)}", fontsize=20)

            # Plot high-pass filtered data and template
            axs[0, 0].set_ylabel("high-pass flux (normalized)", fontsize=14)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
            axs[0, 0].plot(wave, d_planet, 'darkred', label=f"{tn}data")
            if compare_data:
                axs[0, 0].plot(wave, template_shift, 'steelblue', label=model)
            else:
                axs[0, 0].plot(wave, template_shift, 'steelblue', label=model + " template")
            if sigma_l is not None:
                template_shift_noise = np.copy(template_shift)
                noise = np.random.normal(0, sigma_l, len(wave))
                if degrade_resolution and not instru=="HARMONI":
                    noise = Spectrum(wave, noise, None, None).degrade_resolution(wave, renorm=False).flux
                if Rc is not None:
                    noise = filtered_flux(noise, R=R, Rc=Rc, filter_type=filter_type)[0]
                if cut_fringes:
                    noise = cut_spectral_frequencies(noise, R, Rmin, Rmax, target_name=target_name)
                noise[np.isnan(d_planet)] = np.nan
                template_shift_noise += noise
                if not compare_data:
                    axs[0, 0].plot(wave, template_shift_noise, 'steelblue', label=model + " template w/ expected noise " + r"($cos\theta_n$ = " + f"{round(np.nansum(template_shift_noise * template_shift) / np.sqrt(np.nansum(template_shift_noise**2) * np.nansum(template_shift**2)), 3)})", alpha=0.5)
                    res_template_shift_noise, psd_template_shift_noise = calc_psd(wave, template_shift_noise, R, smooth=smooth_PSD)
                    axs[0, 1].plot(res_template_shift_noise, psd_template_shift_noise, 'steelblue', alpha=0.5, zorder=10)
                axs[1, 0].plot(wave, noise, 'seagreen', label="expected noise", zorder=3, alpha=0.8)
                res_noise, psd_noise = calc_psd(wave, noise, R, smooth=smooth_PSD)
                axs[1, 1].plot(res_noise, psd_noise, 'seagreen', zorder=10, alpha=0.8)
            axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[0, 0].minorticks_on()
            axs[0, 0].legend(fontsize=14, loc="upper left")
            axs[0, 0].set_ylim(2 * np.nanmin(d_planet), 2 * np.nanmax(d_planet))
            
            #Zoom
            if (instru=="HiRISE" or instru=="VIPA"):
                if model == "mol_H2O":
                    zoom_xmin, zoom_xmax = 1.43, 1.46
                elif target_name == "CH4":
                    zoom_xmin, zoom_xmax = 1.665, 1.668
                elif target_name == "Jupiter":
                    zoom_xmin, zoom_xmax = 1.67, 1.69
                else:
                    zoom_xmin, zoom_xmax = 1.57, 1.58
                axins = inset_axes(axs[0, 0], width="40%", height="40%", loc="lower right", borderpad=2)
                axins.plot(wave, d_planet, 'darkred', label=f"{tn}data")
                axins.plot(wave, template_shift, 'steelblue')
                axins.set_xlim(zoom_xmin, zoom_xmax)
                axins.set_ylim(min(np.nanmin(d_planet[(wave >= zoom_xmin) & (wave <= zoom_xmax)]), np.nanmin(template_shift[(wave >= zoom_xmin) & (wave <= zoom_xmax)])), max(np.nanmax(d_planet[(wave >= zoom_xmin) & (wave <= zoom_xmax)]), np.nanmax(template_shift[(wave >= zoom_xmin) & (wave <= zoom_xmax)])))
                axins.tick_params(axis='both', which='major', labelsize=10)            
                mark_inset(axs[0, 0], axins, loc1=1, loc2=3, fc="none", ec="black", linestyle="--")
                axins.set_xticklabels([])
                axins.set_yticklabels([])

            # Calculate PSDs
            res_planet, psd_planet = calc_psd(wave, d_planet, R, smooth=smooth_PSD)
            res_template, psd_template = calc_psd(wave, template_shift, R, smooth=smooth_PSD)
            
            # Plot PSDs
            axs[0, 1].set_ylabel("PSD", fontsize=14)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_xlim(10, 2*R)
            axs[0, 1].plot(res_planet, psd_planet, 'darkred')
            axs[0, 1].plot(res_template, psd_template, 'steelblue')
            axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[0, 1].minorticks_on()
            if Rc is not None:
                axs[0, 1].axvline(Rc, c='k', ls="--", label=f"$R_c$ = {int(Rc)}")
            axs[0, 1].axvline(R, c='k', ls="-", label=f"$R$ = {int(R)}")
            axs[0, 1].legend(fontsize=14, loc="upper left")
            
            # Calculate residuals
            residuals = d_planet - template_shift
            
            # Plot residuals
            axs[1, 0].set_xlabel("wavelength [µm]", fontsize=14) ; axs[1, 0].set_ylabel("residuals", fontsize=14)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
            axs[1, 0].set_xlim(np.nanmin(wave[~np.isnan(residuals)]), np.nanmax(wave[~np.isnan(residuals)])) ; axs[1, 0].set_ylim(-5 * np.nanstd(residuals), 5 * np.nanstd(residuals))
            if compare_data:
                axs[1, 0].plot(wave, residuals, 'k', label="data1 - data2", alpha=0.8)
            else:
                axs[1, 0].plot(wave, residuals, 'k', label="data - template", alpha=0.8)
            axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[1, 0].minorticks_on()
            axs[1, 0].legend(fontsize=14, loc="upper left")
            
            # Calculate PSD of residuals
            res_residuals, psd_residuals = calc_psd(wave, residuals, R, smooth=smooth_PSD)
            
            # Plot PSD of residuals
            axs[1, 1].set_xlabel("resolution R", fontsize=14) ; axs[1, 1].set_ylabel("PSD", fontsize=14)
            axs[1, 1].set_xscale('log') ; axs[1, 1].set_yscale('log')
            axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
            axs[1, 1].plot(res_residuals, psd_residuals, 'k', alpha=0.8)
            axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[1, 1].minorticks_on()
            if Rc is not None:
                axs[1, 1].axvline(Rc, c='k', ls="--", label=f"$R_c$ = {Rc}")
            axs[1, 1].axvline(R, c='k', ls="-", label=f"$R$ = {R}")
            
            plt.tight_layout()
            plt.show()
            
            if sigma_l is not None:
                print("std(residuals) / std(noise) = ", np.nanstd(residuals) / np.nanstd(noise))
                print("psd(residuals) / psd(noise) = ", np.sqrt(np.nansum(psd_residuals)) / np.sqrt(np.nansum(psd_noise)))
            
    return rv, CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet



def plot_CCF_rv(instru, band, target_name, d_planet, d_bkg, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, large_rv=True, rv=None, rv_planet=None, vsini_planet=0, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, stellar_component=True, degrade_resolution=True, verbose=True, show=True, smooth_PSD=1, compare_data=False, cut_fringes=False, Rmin=None, Rmax=None, rv_star=None, zoom_CCF=True):
    """
    Plots the Cross-Correlation Function (CCF) for the given parameters.

    Parameters:
    instru (str): Instrument used for the observation.
    band (str): Spectral band used for the observation.
    target_name (str): Name of the target.
    d_planet (array-like): Data of the planet.
    d_bkg (array-like): Background data.
    star_flux (array-like): Flux of the star.
    wave (array-like): Wavelength array.
    trans (array-like): Transmission data.
    T_planet (float): Temperature of the planet.
    lg_planet (float): Logarithmic gravity of the planet.
    model (str): Model used for the correlation.
    R (float): Spectral resolution.
    Rc (float): Resolution cutoff.
    filter_type (str): Type of filter used.
    rv (array-like, optional): Radial velocity array.
    rv_planet (float, optional): Radial velocity of the planet.
    vsini_planet (float, optional): Rotational velocity of the planet. Default is 0.
    pca (array-like, optional): PCA components.
    template (array-like, optional): Template for correlation.
    epsilon (float, optional): Limb darkening coefficient. Default is 0.8.
    fastbroad (bool, optional): Whether to use fast broadening. Default is True.
    logL (bool, optional): Whether to plot log-likelihood. Default is False.
    method_logL (str, optional): Method for log-likelihood calculation. Default is "classic".
    sigma_l (float, optional): Sigma level.
    weight (array-like, optional): Weighting array.
    stellar_component (bool, optional): Whether to include stellar component. Default is True.
    degrade_resolution (bool, optional): Whether to degrade resolution. Default is True.
    show (bool, optional): Whether to show the plot. Default is True.
    smooth_PSD (int, optional): Smoothing parameter for the power spectral density. Default is 1.
    compare_data (bool, optional): Whether to show compare_data plots. Default is False.
    cut_fringes (bool, optional): Whether to cut fringes. Default is False.
    Rmin (float, optional): Minimum resolution for fringe cutting.
    Rmax (float, optional): Maximum resolution for fringe cutting.

    Returns:
    tuple: Contains radial velocity array, SNR of the planet, SNR of the background, and maximum SNR.
    """
    # Compute the radial velocity and cross-correlation functions for the planet and background
    rv, CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet = correlation_rv(instru=instru, d_planet=d_planet, d_bkg=d_bkg, star_flux=star_flux, wave=wave, trans=trans, T_planet=T_planet, lg_planet=lg_planet, model=model, R=R, Rc=Rc, filter_type=filter_type, large_rv=large_rv, rv=rv, rv_planet=rv_planet, vsini_planet=vsini_planet, pca=pca, template=template, epsilon=epsilon, fastbroad=fastbroad, logL=logL, method_logL=method_logL, sigma_l=sigma_l, weight=weight, stellar_component=stellar_component, degrade_resolution=degrade_resolution, disable_tqdm=not verbose, show=show, smooth_PSD=smooth_PSD, target_name=target_name, compare_data=compare_data, cut_fringes=cut_fringes, Rmin=Rmin, Rmax=Rmax)
    # Refine the radial velocity estimate for the planet
    rv_planet = rv[(rv < rv_planet + 25) & (rv > rv_planet - 25)][np.abs(CCF_planet[(rv < rv_planet + 25) & (rv > rv_planet - 25)]).argmax()]
    if np.nanmax(np.abs(rv))/2 > 200: # https://arxiv.org/pdf/2405.13469: std(rv_planet +- 200 km/s)
        mask_rv = (rv > rv_planet + 200) | (rv < rv_planet - 200)
        # Remove the offset introduced by the residual stellar component and systematic effects
        CCF_planet -= np.nanmean(CCF_planet[mask_rv])
        #corr_planet -= np.nanmean(corr_planet[mask_rv]) # not needed since we are only interested on the max correlation value (without subtracting the potential offset)
        corr_auto -= np.nanmean(corr_auto[mask_rv])
        if d_bkg is not None: # Remove offset from the background CCF if background data is provided
            for i in range(len(d_bkg)):
                CCF_bkg[i] -= np.nanmean(CCF_bkg[i])
    else:
        mask_rv = np.full((len(rv)), True)
    # Estimating the CCF noise
    sigma2_tot = np.nanvar(CCF_planet[mask_rv])  # Total variance
    sigma2_auto = np.nanvar(corr_auto[mask_rv] * np.nanmax(CCF_planet[(rv < rv_planet + 25) & (rv > rv_planet - 25)]) / np.nanmax(corr_auto))    
    if sigma2_auto < sigma2_tot and not compare_data:
        sigma_CCF = np.sqrt(sigma2_tot - sigma2_auto)  # sqrt(var(signal) - var(auto-correlation))
    else:
        sigma_CCF = np.sqrt(sigma2_tot)
    # Calculate Signal-to-Noise Ratio (SNR)
    SNR_planet    = CCF_planet / sigma_CCF
    signal_planet = np.nanmax(CCF_planet)
    max_SNR       = np.nanmax(SNR_planet[(rv < rv_planet + 25) & (rv > rv_planet - 25)])
    if d_bkg is not None: # Plot background SNR if background data is provided
        SNR_bkg = np.zeros((len(d_bkg), len(rv)))
        for i in range(len(d_bkg)):
            SNR_bkg[i] = CCF_bkg[i] / np.nanstd(CCF_bkg[i][mask_rv])
    else:
        SNR_bkg = None
    if show:
        # Plot the CCF in S/N and correlation units
        plt.figure(figsize=(10, 6), dpi=300)
        ax1 = plt.gca()        
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.minorticks_on()
        ax1.set_xlim(rv[0], rv[-1])        
        title = f'CCF of {target_name} on {band}-band of {instru} with {model} template'
        if compare_data:
            title += f"\nwith $R_c$ = {Rc}"
        else:
            if "mol_" in model:
                title += f"\nat $T$ = {round(T_planet)}K and Vsini = {round(vsini_planet, 1)} km/s\nwith $R_c$ = {Rc}"
            else:
                title += f"\nat $T$ = {round(T_planet)}K, "+r"$\log g$ = "+f"{round(lg_planet, 1)} and Vsini = {round(vsini_planet, 1)} km/s\nwith $R_c$ = {Rc}"
        ax1.set_title(title, fontsize=18, pad=15)        
        ax1.set_xlabel("Observed Radial Velocity [km/s]", fontsize=14, labelpad=10)
        ax1.set_ylabel("CCF [S/N]", fontsize=14, labelpad=10)
        
        ax1.plot([], [], 'gray', label="Noise", alpha=0.5)        
        if d_bkg is not None:  # Ajout du fond si disponible
            for i in range(len(d_bkg)):
                ax1.plot(rv, SNR_bkg[i], 'gray', alpha=min(max(0.1, 1 / len(d_bkg)) + 0.1, 1))
        
        ax1.plot(rv, corr_auto * max_SNR / np.nanmax(corr_auto), "k", label="Auto-correlation")
        ax1.plot(rv, SNR_planet, label=f"{target_name.replace('_', ' ')}", c="C0", zorder=3, linewidth=2)
        
        if rv_star is not None:
            ax1.axvline(rv_star, c='g', ls='--', label=f"$rv_{{star}}$ = {round(rv_star, 1)} km/s", alpha=0.5, linewidth=1.5)
        ax1.axvline(rv_planet, c='r', ls='--', label=f"$rv_{{obs}}$ = {round(rv_planet, 1)} km/s", alpha=0.5, linewidth=1.5)
        ax1.axvline(0, c='k', ls='-', alpha=0.5, linewidth=1)
        
        ymax = 1.2 * np.nanmax(SNR_planet)
        ymin = -1.5 * np.abs(np.nanmin(SNR_planet))
        ax1.set_ylim(ymin, ymax)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Correlation Strength", fontsize=14, labelpad=20, rotation=270)
        ax2.tick_params(axis='y')
        ax2.set_ylim(ymin * corr_planet[SNR_planet==max_SNR][0] / max_SNR, ymax * corr_planet[SNR_planet==max_SNR][0] / max_SNR)

        if zoom_CCF:
            zoom_xmin, zoom_xmax = rv_planet - np.nanmax(np.abs(rv))/10, rv_planet + np.nanmax(np.abs(rv))/10
            axins = inset_axes(ax1, width="40%", height="40%", loc="upper left", borderpad=2)  
            if d_bkg is not None:  # Ajout du fond si disponible
                for i in range(len(d_bkg)):
                    axins.plot(rv, SNR_bkg[i], 'gray', alpha=min(max(0.1, 1 / len(d_bkg)) + 0.1, 1))
            axins.plot(rv, corr_auto * max_SNR / np.nanmax(corr_auto), "k")
            axins.plot(rv, SNR_planet, c='C0', zorder=10)
            if rv_star is not None:
                axins.axvline(rv_star, c='g', ls='--', alpha=0.5, linewidth=1.5)
            axins.axvline(rv_planet, c='r', ls='--', alpha=0.5, linewidth=1.5)
            axins.axvline(0, c='k', ls='-', alpha=0.5, linewidth=1)
            axins.set_xlim(zoom_xmin, zoom_xmax)
            axins.set_ylim(-1.25 * np.abs(np.nanmin(SNR_planet)), 1.1*np.nanmax(SNR_planet))                
            axins.tick_params(axis='both', which='major', labelsize=10)
            mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="black", linestyle="--")
            #axins.set_xticklabels([])
            axins.set_yticklabels([])
        
        ax1.legend(loc="upper right", fontsize=12, frameon=True, facecolor='white', framealpha=0.8)        
        plt.tight_layout()
        plt.show()
        
        if verbose: # Print maximum S/N and correlation
            if sigma_l is not None: # Print error on sigma (if required):
                print(" error on sigma (sigma CCF/sigma planet) = ", round(sigma_CCF / sigma_planet, 3))
            # Print maximum S/N and correlation
            print(f" CCF: max S/N ({round(max_SNR, 1)}) and correlation ({round(np.nanmax(corr_planet), 5)}) for rv = {round(rv_planet, 2)} km/s")
            
        # Plot log-likelihood (if required)
        if logL:
            rv_planet = rv[(rv < rv_planet + 25) & (rv > rv_planet - 25)][logL_planet[(rv < rv_planet + 25) & (rv > rv_planet - 25)].argmax()]
            plt.figure(dpi=300)
            ax1 = plt.gca()
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.minorticks_on()
            ax1.set_xlim(rv[0], rv[-1])
            ax1.set_title(f'logL ({method_logL}) of {target_name} on {band}-band of {instru} with {model} \n'f'at $T_p$ = {round(T_planet)}K and Vsin(i) = {vsini_planet} km/s with $R_c$ = {Rc}', fontsize=16)
            ax1.set_xlabel("radial velocity [km/s]", fontsize=14)
            ax1.set_ylabel("logL", fontsize=14)
            ax1.plot(rv, logL_planet, label=f"planet", zorder=3)
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(ymin, ymax)
            ax1.plot([rv_planet, rv_planet], [ymin, ymax], 'r:', label=f"rv = {round(rv_planet, 1)} km/s")
            ax1.legend(loc="upper right")
            ax1.set_zorder(1)
            plt.show()
            # Print maximum log-likelihood
            print(f" max logL for rv = {round(rv_planet, 2)} km/s")
    return rv, SNR_planet, SNR_bkg, corr_planet, signal_planet, sigma_CCF, sigma_planet, corr_auto



########################################################################################################################################################################################################################################################################################################################################################################################################



def correlation_vsini(instru, d_planet, d_bkg, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, vsini, rv_planet=None, vsini_planet=None, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, stellar_component=True, degrade_resolution=True, disable_tqdm=False, compare_data=False, cut_fringes=False, Rmin=None, Rmax=None, target_name=None):
    # Initialize correlation arrays
    CCF_planet = np.zeros_like(vsini) ; corr_planet = np.zeros_like(vsini) ; corr_auto = np.zeros_like(vsini) ; corr_auto_noise = np.zeros_like(vsini) ; logL_planet = np.zeros_like(vsini) ; sigma_planet = None 
    if d_bkg is not None: # Initialize background CCF if background data is provided
        CCF_bkg = np.zeros((len(d_bkg), len(vsini)))
    else:
        CCF_bkg = None
    if not compare_data: # If not comparing data, calculating a template
        # Generate template if not provided
        if template is None:
            template_raw = get_template(instru=instru, wave=wave, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=0, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, epsilon=epsilon, fastbroad=fastbroad)
            template = template_raw.copy()
        else:
            template_raw = template.copy()
        # Degrade template to instrumental resolution if required
        if degrade_resolution:
            template = template.degrade_resolution(wave, renorm=False)
        else:
            template = template.interpolate_wavelength(wave, renorm=False)
        # Return if template only contains NaNs (i.e. if the crop of the molecular templates left only NaNs)
        if all(np.isnan(template.flux)):
            return CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet
        # Filter the template
        template_HF = template.copy() ; template_LF = template.copy()
        template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type)
        # Handle stellar component if needed
        if stellar_component and Rc is not None:
            if star_flux is None:
                raise KeyError("star_flux is not defined for the stellar component!")
            f = interp1d(wave[~np.isnan(star_flux / trans)], (star_flux / trans)[~np.isnan(star_flux / trans)], bounds_error=False, fill_value=np.nan)
            sf = f(wave)
            if instru == "HiRISE": # Handle filtering edge effects due to gaps between orders
                _, star_LF = filtered_flux(sf, R, Rc, filter_type)
                nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005 / np.nanmean(np.diff(wave)))
                f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan)
                sf[nan_values] = f(wave[nan_values])
            star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type)
        # Broad the template at the known planet Vsini in order to mimic the data and to calculate/estimate the auto-correlation
        if vsini_planet is not None: # if the vsini of the planet is known
            template_broad = template_raw.broad(vsini_planet, epsilon=epsilon, fastbroad=fastbroad)
            if degrade_resolution:
                template_broad = template_broad.degrade_resolution(wave, renorm=False)
            else:
                template_broad = template_broad.interpolate_wavelength(wave, renorm=False)
            template_HF_broad, template_LF_broad = filtered_flux(template_broad.flux, R, Rc, filter_type)
            template_broad = trans * template_HF_broad
            if stellar_component and Rc is not None:
                template_broad += - trans * star_HF * template_LF_broad/star_LF # better without the residual stellar contributions for the auto-correlation
            if pca is not None: # Subtract PCA components (if required)
                template_broad0 = np.copy(template_broad)
                n_comp_sub = pca.n_components
                for nk in range(n_comp_sub):
                    template_broad -= np.nan_to_num(np.nansum(template_broad0 * pca.components_[nk]) * pca.components_[nk])
            if cut_fringes: # filter fringes frequencies
                template_broad = cut_spectral_frequencies(template_broad, R, Rmin, Rmax, target_name=target_name)
            if weight is not None:
                template_broad *= weight    
            template_broad[np.isnan(d_planet)] = np.nan 
    else: # The data to compare with is assimiled to the "template"
        if template is None:
            raise KeyError("You need to input the data to compare with: template=None")
        template_HF = Spectrum(wave, template, R, T_planet, lg_planet)
        template_broad = np.copy(template)
        if weight is not None:
            template_broad *= weight    
    # Loop through Vsini values and compute correlations
    for i in tqdm(range(len(vsini)), desc="correlation_vsini()"): # for each rv value
        if vsini[i] > 0: # broadening a template at her known rv in order to mimic it (for the auto-correlation)
            template_HF_broad = template_HF.broad(vsini[i], epsilon=epsilon, fastbroad=fastbroad).flux # for the auto-correlation
            if not compare_data:
                template_LF_broad = template_LF.broad(vsini[i], epsilon=epsilon, fastbroad=fastbroad).flux # for the auto-correlation
        else:
            template_HF_broad = template_HF.copy().flux
            if not compare_data:
                template_LF_broad = template_LF.copy().flux
        if not compare_data:
            t = trans * template_HF_broad
            # adding the stellar component (if required)
            if stellar_component and Rc is not None:
                t += - trans * star_HF * template_LF_broad/star_LF
            # Subtract PCA components (if required)
            if pca is not None:
                t0 = np.copy(t)
                n_comp_sub = pca.n_components
                for nk in range(n_comp_sub): 
                    t -= np.nan_to_num(np.nansum(t0*pca.components_[nk])*pca.components_[nk])
            # Cut fringes frequencies (if required)
            if cut_fringes:
                t = cut_spectral_frequencies(t, R, Rmin, Rmax, target_name=target_name)
        else:
            t = template_HF_broad
        d_p = np.copy(d_planet)
        # Taking the weight function into account (if required)
        if weight is not None:
            d_p *= weight ; t *= weight
        # Masking invalid regions
        d_p[d_p == 0] = np.nan ; t[t == 0] = np.nan ; t[np.isnan(d_p)] = np.nan ; d_p[np.isnan(t)] = np.nan 
        # Computations 
        t /= np.sqrt(np.nansum(t**2)) # normalizing the tempalte
        CCF_planet[i] = np.nansum(d_p*t) # CCF_signal
        corr_planet[i] = np.nansum(d_p*t) / np.sqrt(np.nansum(d_p**2)) # correlation strength
        if vsini_planet is not None: # auto correlation
            corr_auto[i] = np.nansum(template_broad*t) / np.sqrt(np.nansum(template_broad[~np.isnan(t)]**2)) # auto-correlation
        if d_bkg is not None: # CCF noise
            for j in range(len(d_bkg)):
                CCF_bkg[j, i] = np.nansum(d_bkg[j]*t)
        if logL: # logL computation
            logL_planet[i] = get_logL(d_p, t, sigma_l, method=method_logL)
    if vsini_planet is not None and sigma_l is not None:
        sigma_planet = np.sqrt(np.nansum(sigma_l**2*template_broad**2)) / np.sqrt(np.nansum(template_broad**2))
    return CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet


    
def plot_CCF_vsini(instru, band, target_name, d_planet, d_bkg, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, vsini=None, rv_planet=None, vsini_planet=0, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, show=True, stellar_component=True, degrade_resolution=True, compare_data=False, cut_fringes=False, Rmin=None, Rmax=None):
    CCF_planet, corr_planet, CCF_bkg, corr_auto, logL_planet, sigma_planet = correlation_vsini(instru=instru, d_planet=d_planet, d_bkg=d_bkg, star_flux=star_flux, wave=wave, trans=trans, T_planet=T_planet, lg_planet=lg_planet, model=model, R=R, Rc=Rc, filter_type=filter_type, vsini=vsini, rv_planet=rv_planet, vsini_planet=vsini_planet, pca=pca, template=template, epsilon=epsilon, fastbroad=fastbroad, logL=logL, method_logL=method_logL, sigma_l=sigma_l, weight=weight, stellar_component=stellar_component, degrade_resolution=degrade_resolution, compare_data=compare_data, cut_fringes=cut_fringes, Rmin=Rmin, Rmax=Rmax, target_name=target_name)
    SNR_planet = CCF_planet/sigma_planet
    vsini_planet = vsini[SNR_planet.argmax()]
    plt.figure(dpi=300) # CCF plot (in S/N and correlation units) 
    ax1 = plt.gca() ; ax1.grid(True) ; ax1.set_xlim(vsini[0], vsini[-1])
    if compare_data:
        ax1.set_title(f'CCF of {target_name} on {band}-band of {instru} with {model} as template \n with $R_c$ = {Rc}', fontsize=16)
    else:
        ax1.set_title(f'CCF of {target_name} on {band}-band of {instru} with {model} \n at $T_p$ = {T_planet}K and RV = {round(rv_planet, 1)} km/s with $R_c$ = {Rc}', fontsize=16)
    ax1.set_xlabel("rotational broadening [km/s]", fontsize=14)
    ax1.set_ylabel("CCF [S/N]", fontsize=14)
    ax1.plot([], [], 'gray', label=f"noise", alpha=0.5)
    for i in range(len(d_bkg)):
        CCF_bkg[i] -= np.nanmean(CCF_bkg[i])
        ax1.plot(vsini, CCF_bkg[i]/np.nanstd(CCF_bkg[i]), 'gray', alpha=max(0.1, 1/len(d_bkg)))
    ax1.plot(vsini, corr_auto*np.nanmax(SNR_planet)/np.nanmax(corr_auto), "k", label=f"auto-correlation")
    ax1.plot(vsini, SNR_planet, label=f"planet", zorder=3)
    ax1.set_xlim(0, vsini[-1])
    ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin, ymax)
    ax1.plot([vsini_planet, vsini_planet], [ymin, ymax], 'r:', label=f"Vsini = {round(vsini_planet, 1)} km/s")
    ax2 = ax1.twinx() ; ax2.set_ylabel('correlation strength', fontsize=14, labelpad=20, rotation=270) ; ax2.tick_params(axis='y')  
    ax2.set_ylim(ymin*np.nanmax(corr_planet)/np.nanmax(SNR_planet), ymax*np.nanmax(corr_planet)/np.nanmax(SNR_planet)) 
    # pas vraiment correct: SNR_planet = signal_CCF / sigma_CCF sauf que sigma_CCF ici semble dépendre de manière non négligeable de Vsin(i), donc il ne suffit pas de renormaliser à une valeur pour avoir toutes les valeurs de corr (contrairement à la CCF en fonction de rv, où sigma_CCF = cste par définition)
    # cf. : plt.plot(vsini, SNR_planet/np.nanmax(SNR_planet)) ; plt.plot(vsini, corr_planet/np.nanmax(corr_planet)) # par contre CCF_planet et corr_planet vsini sont evidémment identiques (à un facteur de normalisation près)
    ax1.legend(loc="lower right") ; ax1.set_zorder(1) ; plt.show()
    print(f" CCF: max S/N ({round(np.nanmax(SNR_planet), 1)}) and correlation ({round(np.nanmax(corr_planet), 5)}) for Vsini = {round(vsini_planet, 1)} km/s")
    if logL:
        vsini_planet = vsini[logL_planet.argmax()]
        plt.figure(dpi=300) # CCF plot (in S/N and correlation units) 
        ax1 = plt.gca() ; ax1.grid(True) ; ax1.set_xlim(vsini[0], vsini[-1]) ; ax1.set_title(f'logL ({method_logL}) of {target_name} on {band}-band of {instru} with {model} \n at $T_p$ = {T_planet}K and RV = {round(rv_planet, 1)} km/s with $R_c$ = {Rc}', fontsize=16) ; ax1.set_xlabel("rotational broadening [km/s]", fontsize=14) ; ax1.set_ylabel("logL", fontsize=14)
        ax1.plot(vsini, logL_planet, label=f"planet", zorder=3)
        ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin, ymax)
        ax1.plot([vsini_planet, vsini_planet], [ymin, ymax], 'r:', label=f"Vsini = {round(vsini_planet, 1)} km/s")
        ax1.legend(loc="lower right") ; ax1.set_zorder(1) ; plt.show()
        print(f" max for logL: max for Vsini = {round(vsini_planet, 1)} km/s")
    return SNR_planet



########################################################################################################################################################################################################################################################################################################################################################################################################



def SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, verbose=True, snr_calc=True):
        NbLine, NbColumn = CCF.shape
        R_planet = int(round(np.sqrt((y0-NbLine//2)**2 + (x0-NbColumn//2)**2)))
        if R_planet == 0: 
            CCF = CCF * annular_mask(0, NbLine//2, value=np.nan, size=(NbLine, NbColumn))
            CCF_noise = CCF_wo_planet * annular_mask(3*size_core, 4*size_core, value=np.nan, size=(NbLine, NbColumn))
        else:
            CCF = CCF * annular_mask(max(R_planet-3*size_core-1, 0), R_planet+3*size_core, value=np.nan, size=(NbLine, NbColumn))
            if snr_calc:
                CCF_noise = CCF_wo_planet * annular_mask(max(1, R_planet-size_core-1), R_planet+size_core+1, value=np.nan, size=(NbLine, NbColumn))
            else:
                CCF_noise = CCF_wo_planet * annular_mask(max(1, R_planet-1), max(2, R_planet), value=np.nan, size=(NbLine, NbColumn))
        CCF_signal = CCF[y0, x0]
        noise = np.sqrt(np.nanvar(CCF_noise))
        if verbose:
            print(" E[<n, t>]/Std[<n, t>] = ", round(100*np.nanmean(CCF_noise)/np.nanstd(CCF_noise), 2), "%")
        signal = CCF_signal-np.nanmean(CCF_noise)
        SNR = signal/noise
        return SNR, CCF, CCF_signal, CCF_noise
    


########################################################################################################################################################################################################################################################################################################################################################################################################



def SNR_T_rv(instru, Sres, Sres_wo_planet, x0, y0, size_core, d_planet, rv_planet, wave, trans, R, Rc, filter_type, target_name, band, model="BT-Settl", vsini_planet=0, pca=None, template=None, weight=None):
    T_planet, lg_planet = model_T_lg(model)
    SNR_2d = np.zeros((len(lg_planet), len(T_planet))) ; noise_2d = np.zeros_like(SNR_2d) + np.nan ; signal_2d = np.zeros_like(SNR_2d) + np.nan
    for j in tqdm(range(len(T_planet))):
        for i in range(len(lg_planet)):
            if model[:4] == "mol_":
                model = model[:4] + lg_planet[i]
            CCF, _ = molecular_mapping_rv(instru, Sres, T_planet=T_planet[j], lg_planet=lg_planet[i], model=model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, rv=rv_planet, verbose=False) 
            CCF_wo_planet, _ = molecular_mapping_rv(instru, Sres_wo_planet, T_planet=T_planet[j], lg_planet=lg_planet[i], model=model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, rv=rv_planet, verbose=False) 
            SNR_2d[i, j], CCF, CCF_signal, CCF_noise = SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, verbose=False)
            noise_2d[i, j] = np.nanstd(CCF_noise)
            signal_2d[i, j] = CCF_signal - np.nanmean(CCF_noise)
    SNR_2d = np.nan_to_num(SNR_2d)
    idx_max_snr = np.unravel_index(np.argmax(SNR_2d, axis=None), SNR_2d.shape)
    plt.figure()
    plt.pcolormesh(T_planet, lg_planet, SNR_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(SNR_2d), vmax=np.nanmax(SNR_2d))
    cbar = plt.colorbar() ; cbar.set_label("S/N", fontsize=14, labelpad=20, rotation=270)
    if model[:4] == "mol_":
        print(f"maximum S/N value of {round(np.nanmax(SNR_2d), 2)} for T = {T_planet[idx_max_snr[1]]} K, {lg_planet[idx_max_snr[0]]} and rv = {rv_planet} km/s")
        plt.ylabel("molecule", fontsize=12)
        plt.title(f'S/N with different molecular template for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T_planet[idx_max_snr[1]], T_planet[idx_max_snr[1]]], [lg_planet[idx_max_snr[0]], lg_planet[idx_max_snr[0]]], 'kX', ms=10, label=r"$S/N_{max}$ = "+f"{round(np.nanmax(SNR_2d), 1)} for T = {T_planet[idx_max_snr[1]]} K, \n {lg_planet[idx_max_snr[0]]} and rv = {rv_planet} km/s")
    else:
        print(f"maximum S/N value of {round(np.nanmax(SNR_2d), 2)} for T = {T_planet[idx_max_snr[1]]} K, lg_planet = {lg_planet[idx_max_snr[0]]} and rv_planet = {rv_planet} km/s")
        plt.ylabel("template's gravity surface", fontsize=12)
        plt.title(f'S/N with different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T_planet[idx_max_snr[1]], T_planet[idx_max_snr[1]]], [lg_planet[idx_max_snr[0]], lg_planet[idx_max_snr[0]]], 'kX', ms=10, label=r"$S/N_{max}$ = "+f"{round(np.nanmax(SNR_2d), 1)} for T = {T_planet[idx_max_snr[1]]} K, \n lg = {lg_planet[idx_max_snr[0]]} and rv = {rv_planet} km/s")
        plt.contour(T_planet, lg_planet, SNR_2d, linewidths=0.1, colors='k')
        plt.ylim(lg_planet[0], lg_planet[-1])
    plt.xlabel("template's temperature [K]", fontsize=12)
    plt.xlim(T_planet[0], T_planet[-1])
    plt.legend(fontsize=12) ; plt.show()
    if model[:4] == "mol_":
        model = model[:4] + lg_planet[idx_max_snr[0]]
    correlation_rv(instru=instru, d_planet=d_planet, d_bkg=None, wave=wave, trans=trans, T_planet=T_planet[idx_max_snr[1]], lg_planet=lg_planet[idx_max_snr[0]], model=model, R=R, Rc=Rc, filter_type=filter_type, show=True, large_rv=False, rv=np.array([rv_planet]), rv_planet=rv_planet, vsini_planet=vsini_planet, pca=pca, template=template, weight=weight)
    if 1 == 0:
        plt.figure() ; plt.pcolormesh(T_planet, lg_planet, noise_2d, cmap=plt.get_cmap('rainbow'))
        plt.xlabel("planet's temperature [K]", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(f'Noise value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        cbar = plt.colorbar() ; cbar.set_label("noise [e-]", fontsize=14, labelpad=20, rotation=270) ; plt.show()
        plt.figure() ; plt.pcolormesh(T_planet, lg_planet, signal_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(signal_2d), vmax=np.nanmax(signal_2d))
        plt.xlabel("planet's temperature [K]", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(f'Signal value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        cbar = plt.colorbar() ; cbar.set_label("signal [e-]", fontsize=14, labelpad=20, rotation=270) ; plt.show()
    


########################################################################################################################################################################################################################################################################################################################################################################################################



def get_template(instru, wave, model, T_planet, lg_planet, vsini_planet, rv_planet, R, Rc, filter_type, epsilon=0.8, fastbroad=True, wave_inter=None):
    if model == "tellurics":
        sky_transmission_path = os.path.join("sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans             = fits.getdata(sky_transmission_path)
        template              = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
    elif model == "BT-NextGen":
        template = load_star_spectrum(T_planet, lg_planet, model=model, interpolated_spectrum=True) # loading the template model
    else:
        template = load_planet_spectrum(T_planet, lg_planet, model=model, instru=instru, interpolated_spectrum=True) # loading the template model
    if model[:4] == "mol_" and Rc is not None: # to crop empty features regions in molecular templates
        _, template_LF = filtered_flux(template.flux, R=template.R, Rc=Rc, filter_type=filter_type)
        sg = sigma_clip(template_LF, sigma=1)
        template.flux[~sg.mask] = np.nan 
    if wave_inter is None:
        template.crop(0.98*wave[0], 1.02*wave[-1])
        dl = template.wavelength - np.roll(template.wavelength, 1) ; dl[0] = dl[1] # delta lambda array
        Rold = np.nanmax(template.wavelength/(2*dl))
        Rold = max(Rold, 2*R)
        if Rold > R0_max: # takes too much time otherwise (limiting then the resolution to 200 000 => does not seem to change anything)
            Rold = R0_max
        dl = np.nanmean(template.wavelength/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist sampling (Shannon)
        wave_inter = np.arange(0.98*wave[0], 1.02*wave[-1], dl) # regularly sampled template
    template = template.interpolate_wavelength(wave_inter, renorm=False)
    # Broadening the spectrum
    template = template.broad(vsini_planet, epsilon=epsilon, fastbroad=fastbroad)
    # Doppler shifting the spectrum [km/s]
    template = template.doppler_shift(rv_planet, renorm=False)
    if model[:4] != "mol_": # if it is not a molecular template
        template.flux *= wave_inter # for the template to be homogenous to ph or e- or ADU
    return template



# CALCULATING d_planet_sim
def get_d_planet_sim(d_planet, wave, trans, model, T_planet, lg_planet, vsini_planet, rv_planet, R, Rc, filter_type, degrade_resolution, stellar_component, pca=None, star_flux=None, instru=None, epsilon=0.8, fastbroad=True, cut_fringes=False, Rmin=None, Rmax=None, target_name=None, corner_plot=False, wave_inter=None):
    """
    Compute the simulated planetary spectrum after high-pass filtering and optional corrections.
    """
    # Generate planetary template
    planet = get_template(instru=instru, wave=wave, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, epsilon=epsilon, fastbroad=fastbroad, wave_inter=wave_inter)
    
    # Degrade resolution (if required)
    if degrade_resolution:
        planet = planet.degrade_resolution(wave, renorm=False)
    else:
        planet = planet.interpolate_wavelength(wave, renorm=False)
    
    # High-pass filtering
    planet_HF, planet_LF = filtered_flux(planet.flux, R, Rc, filter_type)
    template             = trans * planet_HF
    
    # Stellar component correction
    if stellar_component and Rc is not None:
        if star_flux is None:
            raise KeyError("star_flux is not defined for the stellar component!")
        f  = interp1d(wave[~np.isnan(star_flux / trans)], (star_flux / trans)[~np.isnan(star_flux / trans)], bounds_error=False, fill_value=np.nan)
        sf = f(wave)
        if instru == "HiRISE": # Handling filtering edge effects due to the gaps bewteen the orders
            _, star_LF     = filtered_flux(sf, R, Rc, filter_type)
            nan_values     = keep_true_chunks(np.isnan(d_planet), N=0.005 / np.nanmean(np.diff(wave)))
            f              = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan)
            sf[nan_values] = f(wave[nan_values])
        star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type)
        template        += -trans * star_HF * planet_LF / star_LF
    
    # PCA component subtraction
    if pca is not None:
        template0 = np.copy(template)
        for nk in range(pca.n_components):
            template -= np.nan_to_num(np.nansum(template0 * pca.components_[nk]) * pca.components_[nk])
    
    # Filtering fringes frequencies
    if cut_fringes:
        template = cut_spectral_frequencies(template, R, Rmin, Rmax, target_name=target_name)
    
    # Normalize and prepare output
    template[np.isnan(d_planet)] = np.nan
    template /= np.sqrt(np.nansum(template**2))
    d_planet_sim = np.copy(template)
    
    # Compute power spectral density (PSD)
    res_d_planet, psd_d_planet         = calc_psd(wave, d_planet, R, smooth=0)
    res_d_planet_sim, psd_d_planet_sim = calc_psd(wave, d_planet_sim, R, smooth=0)
    
    # Compute ratio for normalization
    if corner_plot:
        ratio = np.nansum(d_planet * template) / np.sqrt(np.nansum(d_planet_sim**2)) # assuming cos_p = 1
        #ratio = np.sqrt(np.nansum(psd_d_planet[(res_d_planet>Rc)&(res_d_planet<R/10)]) / np.nansum(psd_d_planet_sim[(res_d_planet_sim>Rc)&(res_d_planet_sim<R/10)]))
    else:
        ratio = np.sqrt(np.nansum(psd_d_planet[(res_d_planet > Rc) & (res_d_planet < R / 10)]) / np.nansum(psd_d_planet_sim[(res_d_planet_sim > Rc) & (res_d_planet_sim < R / 10)]))
    d_planet_sim *= ratio
    
    return d_planet_sim



########################################################################################################################################################################################################################################################################################################################################################################################################



def parameters_estimation(instru, band, target_name, wave, d_planet, star_flux, trans, R, Rc, filter_type, model, logL=False, method_logL="classic", sigma_l=None, weight=None, pca=None, precise_estimate=False, SNR_estimate=False, T_planet=None, lg_planet=None, vsini_planet=None, rv_planet=None, T_arr=None, lg_arr=None, vsini_arr=None, rv_arr=None, SNR_CCF=None, show=True, verbose=True, stellar_component=True, degrade_resolution=True, force_new_est=False, d_planet_sim=False, save=True, fastcurves=False, exposure_time=None, star_HF=None, star_LF=None, wave_inter=None):
    """
    Estimates astrophysical parameters based on cross-correlation analysis.

    Parameters:
        instru : str
            Name of the instrument.
        band : str
            Spectral band used for observation.
        target_name : str
            Name of the observed target.
        wave : numpy.ndarray
            Wavelength grid.
        d_planet : numpy.ndarray
            Observed planetary spectrum.
        star_flux : numpy.ndarray
            Stellar flux.
        trans : numpy.ndarray
            Transmission function.
        R : float
            Spectral resolution.
        Rc : float
            Calibration spectral resolution.
        filter_type : str
            Type of filtering applied.
        model : str
            Atmospheric model used.
        logL : bool, optional
            Whether to compute log-likelihood. Default is False.
        method_logL : str, optional
            Log-likelihood computation method. Default is "classic".
        sigma_l : float, optional
            Noise level.
        weight : numpy.ndarray, optional
            Weight function for the likelihood computation.
        pca : bool, optional
            Whether to use PCA for noise reduction.
        precise_estimate : bool, optional
            Whether to perform a precise estimate. Default is False.
        SNR_estimate : bool, optional
            Whether to compute the signal-to-noise ratio. Default is False.
        T_planet, lg_planet, vsini_planet, rv_planet : float, optional
            True planetary parameters (temperature, log gravity, rotational velocity, radial velocity).
        T_arr, lg_arr, vsini_arr, rv_arr : numpy.ndarray, optional
            Grids of parameters.
        SNR_CCF : numpy.ndarray, optional
            Cross-correlation function's signal-to-noise ratio.
        show : bool, optional
            Whether to display results. Default is True.
        verbose : bool, optional
            Whether to print logs. Default is True.
        stellar_component : bool, optional
            Whether to include the stellar component. Default is True.
        degrade_resolution : bool, optional
            Whether to degrade resolution for calculations. Default is True.
        force_new_est : bool, optional
            Whether to force new calculations. Default is False.
        d_planet_sim : bool, optional
            Whether to simulate a theoretical planetary spectrum. Default is False.
        save : bool, optional
            Whether to save results to disk. Default is True.
        fastcurves : bool, optional
            Whether to use FastCurves for fast parameter estimation. Default is False.
        exposure_time : float, optional
            Exposure time in minutes. Default is None.
        star_HF, star_LF : numpy.ndarray, optional
            High-frequency and low-frequency stellar components.
        wave_inter : numpy.ndarray, optional
            Intermediate wavelength grid.

    Returns:
        T_arr, lg_arr, vsini_arr, rv_arr : numpy.ndarray
            Parameter grids.
        corr_4D, SNR_4D, logL_4D, logL_4D_sim : numpy.ndarray
            Computed correlation, SNR, and log-likelihood cubes.
    """
    
    epsilon=0.8 ; fastbroad=True
    
    # Attempt to load existing parameter estimations
    try:
        if force_new_est:  # If forcing new calculations
            raise ValueError("force_new_est = True")
        
        # Load parameter grids
        T_arr     = fits.getdata(f"utils/parameters estimation/parameters_estimation_T_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        lg_arr    = fits.getdata(f"utils/parameters estimation/parameters_estimation_lg_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        vsini_arr = fits.getdata(f"utils/parameters estimation/parameters_estimation_vsini_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        rv_arr    = fits.getdata(f"utils/parameters estimation/parameters_estimation_rv_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        
        # Load computed matrices
        if SNR_estimate:
            SNR_4D = fits.getdata(f"utils/parameters estimation/parameters_estimation_SNR_4D_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
            corr_4D = None ; logL_4D = None ; logL_4D_sim = None
        else:
            corr_4D = fits.getdata(f"utils/parameters estimation/parameters_estimation_corr_4D_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
            if logL:
                logL_4D = fits.getdata(f"utils/parameters estimation/parameters_estimation_logL_4D_{method_logL}_{instru}_{band}_R{R}_Rc{Rc}_{target_name}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
                if d_planet_sim:
                    logL_4D_sim = fits.getdata(f"utils/parameters estimation/parameters_estimation_logL_4D_sim_{method_logL}_{instru}_{band}_R{R}_Rc{Rc}_{target_name}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
                else:
                    d_planet_sim = None ; logL_4D_sim = None
            else:
                logL_4D = None ; logL_4D_sim = None
            SNR_4D = None
        if verbose:
            print("Loading existing parameters calculations...")
            
    # Compute new estimations if files are missing or forced update
    except Exception as e:
        if verbose and not fastcurves:
            print(f"New parameters estimation: {e}")
            
        # Initialize parameter grids if not provided
        if T_arr is None or lg_arr is None or vsini_arr is None or rv_arr is None or SNR_estimate:
            DT     = max(2*T_planet / SNR_CCF, 10)
            Dlg    = max(2*lg_planet / SNR_CCF, 0.5)
            Dvsini = min(20 * c*1e-3 / (2*R), 50)
            Drv    = min(20 * c*1e-3 / (2*R) / SNR_CCF, 50)
            T_arr, lg_arr = get_model_grid(model)
            if precise_estimate:
                N = 20
                DT     = min(DT, 100)
                Dlg    = min(Dlg, 1)
                Dvsini = min(Dvsini, 20)
                Drv    = min(Drv, 20)
            elif not (R > 5000 or instru=="HARMONI"):
                N = 20
            else:
                N = 10
            T_arr  = np.linspace(max(T_arr[0], T_planet-DT/2), min(T_arr[-1], T_planet + DT/2), N+1).astype(np.float32) # dvsini = 0.5 km/s
            if "mol_" not in model:
                lg_arr = np.linspace(max(lg_arr[0], lg_planet-Dlg/2), min(lg_arr[-1], lg_planet + Dlg/2), N+1).astype(np.float32) # dvsini = 0.5 km/s
            if R > 5000 or instru=="HARMONI":
                vsini_arr = np.linspace(max(0, vsini_planet-Dvsini/2), min(80, vsini_planet + Dvsini/2), N+1).astype(np.float32) # dvsini = 0.5 km/s
            else:
                vsini_arr = np.array([vsini_planet])
            if SNR_estimate: # for S/N estimations
                rv_arr    = np.append(np.linspace(-1000, rv_planet-3*Drv/4, 100), np.append(np.linspace(rv_planet-Drv/2, rv_planet+Drv/2, int(Drv/0.5)), np.linspace(rv_planet+3*Drv/4, 1000, 100)))
            else:
                rv_arr = np.linspace(rv_planet-Drv/2, rv_planet+Drv/2, N+1).astype(np.float32)
                
        # Calculating the high and low frequency content of the star if necessary and if missing 
        if stellar_component and Rc is not None:
            if star_HF is None and star_LF is None:
                if star_flux is None:
                    raise KeyError("star_flux is not defined for the stellar component !")
                f = interp1d(wave[~np.isnan(star_flux/trans)], (star_flux/trans)[~np.isnan(star_flux/trans)], bounds_error=False, fill_value=np.nan) 
                sf = f(wave) 
                if instru=="HiRISE": # Handling filtering edge effects due to the gaps bewteen the orders
                    _, star_LF = filtered_flux(sf, R, Rc, filter_type) # high pass filtering
                    nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
                    f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
                    sf[nan_values] = f(wave[nan_values])
                star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type) # high pass filtering
        
        # CALCULATING d_planet_sim (theoritical data we should have w/o noise) (if wanted)
        if d_planet_sim and logL:
            d_planet_sim = get_d_planet_sim(d_planet=d_planet, wave=wave, trans=trans, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, degrade_resolution=degrade_resolution, stellar_component=stellar_component, pca=pca, star_flux=star_flux, instru=instru, epsilon=epsilon, fastbroad=fastbroad, target_name=target_name, corner_plot=True, wave_inter=wave_inter)
        else:
            d_planet_sim = None
            
        # Compute likelihood, correlation and S/N matrices in parallel
        corr_4D     = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        SNR_4D      = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        logL_4D     = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        logL_4D_sim = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        with Pool(processes=cpu_count()) as pool: 
            results = list(tqdm(pool.imap(process_parameters_estimation, [(i, j, T_arr, lg_arr, vsini_arr, rv_arr, d_planet, weight, pca, model, instru, wave, trans, epsilon, fastbroad, R, Rc, filter_type, sigma_l, logL, method_logL, star_HF, star_LF, SNR_estimate, rv_planet, stellar_component, degrade_resolution, d_planet_sim, wave_inter) for i in range(len(T_arr)) for j in range(len(lg_arr))]), total=len(T_arr)*len(lg_arr), disable=not verbose, desc="Parameters estimation"))
            for (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim) in results:
                corr_4D[i, j, :, :]     = corr_2D
                SNR_4D[i, j, :, :]      = SNR_2D
                logL_4D[i, j, :, :]     = logL_2D
                logL_4D_sim[i, j, :, :] = logL_2D_sim
                
        # Saving data (if wanted)
        if save:
            fits.writeto(f"utils/parameters estimation/parameters_estimation_T_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", T_arr, overwrite=True)
            if "mol_" not in model:
                fits.writeto(f"utils/parameters estimation/parameters_estimation_lg_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", lg_arr, overwrite=True)
            fits.writeto(f"utils/parameters estimation/parameters_estimation_vsini_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", vsini_arr, overwrite=True)
            fits.writeto(f"utils/parameters estimation/parameters_estimation_rv_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", rv_arr, overwrite=True)
            if SNR_estimate:
                fits.writeto(f"utils/parameters estimation/parameters_estimation_SNR_4D_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", SNR_4D, overwrite=True)
            else:
                fits.writeto(f"utils/parameters estimation/parameters_estimation_corr_4D_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", corr_4D, overwrite=True)
                if logL:
                    fits.writeto(f"utils/parameters estimation/parameters_estimation_logL_4D_{method_logL}_{instru}_{band}_R{R}_Rc{Rc}_{target_name}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", logL_4D, overwrite=True)
                    if d_planet_sim is not None:
                        fits.writeto(f"utils/parameters estimation/parameters_estimation_logL_4D_sim_{method_logL}_{instru}_{band}_R{R}_Rc{Rc}_{target_name}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", logL_4D_sim, overwrite=True)         
    
    # Show 2D (T, lg) matrices (if wanted and if possible)
    if show and len(T_arr) > 2 and len(lg_arr) > 2 and not fastcurves:
        
        if SNR_estimate:
            idx_max_SNR = np.unravel_index(np.argmax(np.nan_to_num(SNR_4D[:, :, :, (rv_arr>-50)&(rv_arr<50)]), axis=None), SNR_4D[:, :, :, (rv_arr>-50)&(rv_arr<50)].shape)
            SNR_2D      = np.nan_to_num(SNR_4D[:, :, :, (rv_arr>-50)&(rv_arr<50)][:, :, idx_max_SNR[2], idx_max_SNR[3]].transpose())
            T_SNR_found     = T_arr[idx_max_SNR[0]]
            lg_SNR_found    = lg_arr[idx_max_SNR[1]]
            vsini_SNR_found = vsini_arr[idx_max_SNR[2]]
            rv_SNR_found    = rv_arr[(rv_arr>-50)&(rv_arr<50)][idx_max_SNR[3]]
            print(f"maximum S/N for T = {T_SNR_found:.0f} K, lg = {lg_SNR_found:.2f} and rv = {rv_SNR_found:.1f} km/s")
            plt.figure(dpi=300) ; plt.ylabel("surface gravity [dex]", fontsize=12) ; plt.xlabel("temperature [K]", fontsize=12) ; plt.title(f'S/N with {model} spectra for {target_name}\n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}', fontsize=14)
            plt.pcolormesh(T_arr, lg_arr, SNR_2D, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(SNR_2D), vmax=np.nanmax(SNR_2D))
            cbar = plt.colorbar() ; cbar.set_label("S/N", fontsize=12, labelpad=20, rotation=270)
            if "mol_" in model:
                plt.plot([T_SNR_found, T_SNR_found], [lg_SNR_found, lg_SNR_found], 'kX', ms=10, label=r"$S/N_{max}$ "+f"for T = {round(T_SNR_found)}K, {lg_SNR_found}, \n Vsin(i) = {vsini_SNR_found:.1f}km/s and RV = {rv_SNR_found:.1f}km/s")
            else:
                plt.plot([T_SNR_found, T_SNR_found], [lg_SNR_found, lg_SNR_found], 'kX', ms=10, label=r"$S/N_{max}$ "+f"for T = {round(T_SNR_found)}K, lg = {lg_SNR_found:.2f}, \n Vsin(i) = {vsini_SNR_found:.1f}km/s and RV = {rv_SNR_found:.1f}km/s")
            plt.contour(T_arr, lg_arr, SNR_2D, linewidths=0.1, colors='k')
            plt.ylim(lg_arr[0], lg_arr[-1]) ; plt.xlim(T_arr[0], T_arr[-1]) ; plt.legend(fontsize=10) ; plt.show()
        
        else:
            idx_max_corr = np.unravel_index(np.argmax(corr_4D, axis=None), corr_4D.shape)
            corr_2D      = np.nan_to_num(corr_4D[:, :, idx_max_corr[2], idx_max_corr[3]].transpose())
            T_corr_found     = T_arr[idx_max_corr[0]]
            lg_corr_found    = lg_arr[idx_max_corr[1]]
            vsini_corr_found = vsini_arr[idx_max_corr[2]]
            rv_corr_found    = rv_arr[idx_max_corr[3]]
            print(f"maximum correlation ({np.nanmax(corr_4D):.3f}) for T = {T_corr_found:.0f} K, lg = {lg_corr_found:.2f} and rv = {rv_corr_found:.1f} km/s")
            fig, ax = plt.subplots(dpi=300, figsize=(8, 6))            
            ax.set_title(f'Correlation between {model} spectra and {target_name}\n{band}-band spectrum from {instru} at $R_c$ = {Rc}', fontsize=16, pad=20)            
            ax.set_xlabel(r"$T_\mathrm{eff}$ [K]", fontsize=14)
            ax.set_ylabel(r"$\log\,g$ [dex (cm/s²)]", fontsize=14)            
            pcm = ax.pcolormesh(T_arr, lg_arr, corr_2D, cmap='coolwarm', shading='auto', vmin=np.nanmin(corr_2D), vmax=np.nanmax(corr_2D))            
            cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046, format='%.3f')
            cbar.set_label("Correlation strength", fontsize=14, labelpad=18, rotation=270)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.minorticks_on()            
            label_text = rf"max at $T_\mathrm{{eff}}$ = {T_corr_found:.0f} K, $\log\,g$ = {lg_corr_found:.2f}, $V\sin i$ = {vsini_corr_found:.1f} km/s, $RV$ = {rv_corr_found:.1f} km/s" if "mol_" not in model else rf"max at $T_\mathrm{{eff}}$ = {T_corr_found:.0f} K, {lg_corr_found}, $V\sin i$ = {vsini_corr_found:.1f} km/s, $RV$ = {rv_corr_found:.1f} km/s"
            ax.plot(T_corr_found, lg_corr_found, marker='X', color='white', markersize=9, markeredgecolor='black', markeredgewidth=1.5, label=label_text)            
            ax.contour(T_arr, lg_arr, corr_2D, levels=12, linewidths=0.8, colors='white', alpha=0.7)            
            ax.set_xlim(T_arr[0], T_arr[-1])
            ax.set_ylim(lg_arr[0], lg_arr[-1])
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.minorticks_on()            
            ax.legend(fontsize=11, loc='lower left', frameon=True, facecolor='whitesmoke', edgecolor='gray')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)            
            plt.tight_layout()
            plt.show()
            
            if logL:
                idx_max_logL = np.unravel_index(np.argmax(logL_4D, axis=None), logL_4D.shape)
                logL_2D      = logL_4D[:, :, idx_max_logL[2], idx_max_logL[3]].transpose()
                logL_2D      = (logL_2D - np.nanmin(logL_2D)) / np.nanmax(logL_2D - np.nanmin(logL_2D))
                logL_2D[np.isnan(logL_2D)] = np.nanmin(logL_2D)
                T_logL_found     = T_arr[idx_max_logL[0]]
                lg_logL_found    = lg_arr[idx_max_logL[1]]
                vsini_logL_found = vsini_arr[idx_max_logL[2]]
                rv_logL_found    = rv_arr[idx_max_logL[3]]
                print(f"maximum logL for T = {T_logL_found:.0f} K, lg = {lg_logL_found} and rv = {rv_logL_found:.1f} km/s")
                plt.figure(dpi=300) ; plt.ylabel("surface gravity [dex]", fontsize=12) ; plt.xlabel("temperature [K]", fontsize=12) ; plt.title(f'logL ({method_logL}) with {model} spectra for {target_name}\n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}', fontsize=14)
                plt.pcolormesh(T_arr, lg_arr, logL_2D, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(logL_2D), vmax=np.nanmax(logL_2D))
                cbar = plt.colorbar() ; cbar.set_label("logL", fontsize=12, labelpad=20, rotation=270)
                if "mol_" in model:
                    plt.plot([T_logL_found, T_logL_found], [lg_logL_found, lg_logL_found], 'kX', ms=10, label=r"$logL_{max}$ "+f" for T = {round(T_logL_found)}K, {lg_logL_found}, \n Vsin(i) = {vsini_logL_found:.1f}km/s and RV = {rv_logL_found:.1f}km/s")
                else:
                    plt.plot([T_logL_found, T_logL_found], [lg_logL_found, lg_logL_found], 'kX', ms=10, label=r"$logL_{max}$ "+f" for T = {round(T_logL_found)}K, lg = {lg_logL_found:.2f}, \n Vsin(i) = {vsini_logL_found:.1f}km/s and RV = {rv_logL_found:.1f}km/s")
                plt.contour(T_arr, lg_arr, logL_2D, linewidths=0.1, colors='k')
                plt.ylim(lg_arr[0], lg_arr[-1]) ; plt.xlim(T_arr[0], T_arr[-1]) ; plt.legend(fontsize=10) ; plt.show()
                
                if d_planet_sim is not None:
                    idx_max_logL_sim = np.unravel_index(np.argmax(logL_4D_sim, axis=None), logL_4D_sim.shape)
                    logL_sim_2D = logL_4D_sim[:, :, idx_max_logL_sim[2], idx_max_logL_sim[3]].transpose()
                    logL_sim_2D = (logL_sim_2D - np.nanmin(logL_sim_2D)) / np.nanmax(logL_sim_2D - np.nanmin(logL_sim_2D))
                    T_logL_sim_found     = T_arr[idx_max_logL_sim[0]]
                    lg_logL_sim_found    = lg_arr[idx_max_logL_sim[1]]
                    vsini_logL_sim_found = vsini_arr[idx_max_logL_sim[2]]
                    rv_logL_sim_found    = rv_arr[idx_max_logL_sim[3]]
                    plt.figure(dpi=300) ; plt.ylabel("surface gravity [dex]", fontsize=12) ; plt.xlabel("temperature [K]", fontsize=12) ; plt.title(f'logL_sim ({method_logL}) with {model} spectra for {target_name}\n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}', fontsize=14)
                    plt.pcolormesh(T_arr, lg_arr, logL_sim_2D, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(logL_sim_2D), vmax=np.nanmax(logL_sim_2D))
                    cbar = plt.colorbar() ; cbar.set_label("logL_sim", fontsize=12, labelpad=20, rotation=270)
                    plt.plot([T_logL_sim_found, T_logL_sim_found], [lg_logL_sim_found, lg_logL_sim_found], 'kX', ms=10, label=r"${logL_{sim}}_{max}$ "+f" for T = {round(T_logL_sim_found)}K, lg = {lg_logL_sim_found:.2f}, \n Vsin(i) = {vsini_logL_sim_found:.1f}km/s and RV = {rv_logL_sim_found:.1f}km/s")
                    plt.contour(T_arr, lg_arr, logL_sim_2D, linewidths=0.1, colors='k')
                    plt.ylim(lg_arr[0], lg_arr[-1]) ; plt.xlim(T_arr[0], T_arr[-1]) ; plt.legend(fontsize=10) ; plt.show()
                
    # Corner plot :
    
    # FastCurves estiamtion
    if fastcurves:
        p_4D = np.exp(logL_4D - np.nanmax(logL_4D))
        uncertainties_1sigma = custom_corner_plot(p_4D, T_arr, lg_arr, vsini_arr, rv_arr, target_name, band, instru, model, R, Rc, sim=False, exposure_time=exposure_time, show=show)
        return uncertainties_1sigma
    else:
        if show and logL:
            p_4D = np.exp(logL_4D - np.nanmax(logL_4D))
            custom_corner_plot(p_4D, T_arr, lg_arr, vsini_arr, rv_arr, target_name, band, instru, model, R, Rc, sim=False, exposure_time=exposure_time, show=show)
            if d_planet_sim is not None:
                p_4D_sim = np.exp(logL_4D_sim - np.nanmax(logL_4D_sim))
                custom_corner_plot(p_4D_sim, T_arr, lg_arr, vsini_arr, rv_arr, target_name, band, instru, model, R, Rc, sim=True, exposure_time=exposure_time, show=show)
        return T_arr, lg_arr, vsini_arr, rv_arr, corr_4D, SNR_4D, logL_4D, logL_4D_sim



def process_parameters_estimation(args):
    i, j, T_arr, lg_arr, vsini_arr, rv_arr, d_planet, weight, pca, model, instru, wave, trans, epsilon, fastbroad, R, Rc, filter_type, sigma_l, logL, method_logL, star_HF, star_LF, SNR_estimate, rv_planet, stellar_component, degrade_resolution, d_planet_sim, wave_inter = args
    corr_2D     = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    SNR_2D      = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    auto_2D     = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    logL_2D     = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    logL_2D_sim = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    if "mol_" in model:
        model = model[:4]+lg_arr[j]
    if len(vsini_arr) == 1: # il est préférable de faire l'élargissement Doppler avant la dégradation en résolution
        template = get_template(instru=instru, wave=wave, model=model, T_planet=T_arr[i], lg_planet=lg_arr[j], vsini_planet=vsini_arr[0], rv_planet=0, R=R, Rc=Rc, filter_type=filter_type, epsilon=epsilon, fastbroad=fastbroad, wave_inter=wave_inter)
    else:
        template = get_template(instru=instru, wave=wave, model=model, T_planet=T_arr[i], lg_planet=lg_arr[j], vsini_planet=0, rv_planet=0, R=R, Rc=Rc, filter_type=filter_type, epsilon=epsilon, fastbroad=fastbroad, wave_inter=wave_inter)
    if degrade_resolution:
        template = template.degrade_resolution(wave, renorm=False)
    else:
        template = template.interpolate_wavelength(wave, renorm=False)
    if all(np.isnan(template.flux)):
        corr_2D     += np.nan
        SNR_2D      += np.nan
        auto_2D     += np.nan
        logL_2D     += np.nan
        logL_2D_sim += np.nan
        return (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim)

    template_HF = template.copy() ; template_LF = template.copy()
    template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type)
    for k in range(len(vsini_arr)):
        if len(vsini_arr) == 1: # l'élargissement a déja été fait
            template_HF_broad = template_HF.copy()
            template_LF_broad = template_LF.copy()
        else:
            template_HF_broad = template_HF.broad(vsini_arr[k], epsilon=epsilon, fastbroad=fastbroad)
            if stellar_component and Rc is not None:
                template_LF_broad = template_LF.broad(vsini_arr[k], epsilon=epsilon, fastbroad=fastbroad)
        if SNR_estimate: # needed to calculate sigma_auto_correlation (and subtract it later)
            template_auto = trans*template_HF_broad.flux
            if stellar_component and Rc is not None:
                template_auto += - trans * star_HF * template_LF_broad.flux/star_LF # better without the residual stellar contributions for the auto-correlation
            template_auto[np.isnan(d_planet)] = np.nan 
            if weight is not None:
                template_auto *= weight
            template_auto /= np.sqrt(np.nansum(template_auto**2)) # normalizing
        for l in range(len(rv_arr)):
            template_HF_broad_shift = template_HF_broad.doppler_shift(rv_arr[l], renorm=False).flux
            t = trans*template_HF_broad_shift
            if stellar_component and Rc is not None:
                template_LF_broad_shift = template_LF_broad.doppler_shift(rv_arr[l], renorm=False).flux
                t += - trans * star_HF * template_LF_broad_shift/star_LF
            if pca is not None: # subtracting PCA modes (if any)
                t0 = np.copy(t)
                n_comp_sub = pca.n_components
                for nk in range(n_comp_sub): 
                    t -= np.nan_to_num(np.nansum(t0*pca.components_[nk])*pca.components_[nk])
            d_p = np.copy(d_planet)
            if weight is not None:
                d_p *= weight
                t   *= weight
            d_p[d_p == 0]    = np.nan
            t[t == 0]        = np.nan
            t[np.isnan(d_p)] = np.nan
            d_p[np.isnan(t)] = np.nan
            t /= np.sqrt(np.nansum(t**2)) # normalizing the tempalte
            signal_CCF = np.nansum(d_p*t)
            if SNR_estimate:
                SNR_2D[k, l] = signal_CCF
                auto_2D[k, l] = np.nansum(template_auto*t)
            else:
                corr_2D[k, l] = signal_CCF / np.sqrt(np.nansum(d_p**2))
                if logL:
                    logL_2D[k, l] = get_logL(d_p, t, sigma_l, method=method_logL)
                    if d_planet_sim is not None:
                        d_p_sim = np.copy(d_planet_sim)
                        if weight is not None:
                            d_p_sim *= weight
                        d_p_sim[np.isnan(t)] = np.nan
                        logL_2D_sim[k, l] = get_logL(d_p_sim, t, sigma_l, method=method_logL)
        if SNR_estimate:
            SNR_2D[k, :]  -= np.nanmean(SNR_2D[k][(rv_arr>rv_planet+200)|(rv_arr<rv_planet-200)])
            auto_2D[k, :] -= np.nanmean(auto_2D[k][(rv_arr>200)|(rv_arr<-200)])
            sigma2_tot  = np.nanvar(SNR_2D[k][(rv_arr>rv_planet+200)|(rv_arr<rv_planet-200)]) # Variance
            sigma2_auto = np.nanvar(auto_2D[k][(rv_arr>200)|(rv_arr<-200)]*np.nanmax(SNR_2D[k][(rv_arr<rv_planet+25)&(rv_arr>rv_planet-25)])/np.nanmax(auto_2D[k]))
            
            if sigma2_auto < sigma2_tot:
                sigma_CCF = np.sqrt(sigma2_tot - sigma2_auto) # NOISE ESTIMATION = sqrt(var(signal) - var(auto correlation))
            else:
                sigma_CCF = np.sqrt(sigma2_tot)
            SNR_2D[k, :] /= sigma_CCF
    return (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim)



########################################################################################################################################################################################################################################################################################################################################################################################################




def estimate_uncertainties_1sigma(p_4D, *params):
    """
    Estimates the 1σ uncertainties and optimal values for each parameter.

    Parameters:
        p_4D : numpy.ndarray
            Probability cube (or hypercube).
        *params : list of numpy.ndarray
            Parameter grids (T, lg, vsini, rv) or a subset if a dimension is ignored.

    Returns:
        uncertainties : list
            List of 1σ uncertainties for each parameter.
        optimal_values : list
            List of optimal values (maximum probability) for each parameter.
    """

    uncertainties  = []
    optimal_values = []
    ndim = len(params)

    for i, param_values in enumerate(params):
        # Marginalize the probability distribution
        marginalized_p  = np.nansum(p_4D, axis=tuple(j for j in range(ndim) if j != i))
        marginalized_p /= np.nansum(marginalized_p)  # Normalization

        # Interpolation to estimate the maximum probability value
        f_interp      = interp1d(param_values, marginalized_p, kind='cubic', bounds_error=False, fill_value=np.nan)
        values_fine   = np.linspace(param_values[0], param_values[-1], 1000)
        p_fine        = f_interp(values_fine)
        idx_max_fine  = np.argmax(p_fine)
        optimal_value = values_fine[idx_max_fine]
        optimal_values.append(optimal_value)

        # Compute the 1σ limits from the cumulative probability distribution
        cdf  = np.cumsum(marginalized_p)
        cdf /= cdf[-1]  # Normalization

        # Interpolation to find the 16% and 84% bounds (1σ)
        f_interp_cdf = interp1d(cdf, param_values, kind='linear', bounds_error=False, fill_value=(param_values[0], param_values[-1]))
        lower_bound  = f_interp_cdf(0.16)
        upper_bound  = f_interp_cdf(0.84)
        uncertainty  = (upper_bound - lower_bound) / 2
        uncertainties.append(uncertainty)

    return uncertainties, optimal_values



def custom_corner_plot(p_4D, T_arr, lg_arr, vsini_arr, rv_arr, target_name, band, instru, model, R, Rc, sim=False, exposure_time=None, show=True):
    """
    Generates a corner plot for parameter estimation based on a probability cube.

    Parameters:
        p_4D : numpy.ndarray
            Probability cube (or hypercube).
        T_arr, lg_arr, vsini_arr, rv_arr : numpy.ndarray
            Parameter grids for temperature, log gravity, rotational velocity, and radial velocity.
        target_name : str
            Name of the observed target.
        band : str
            Spectral band used for observation.
        instru : str
            Instrument name.
        model : str
            Model name.
        R : float
            Spectral resolution.
        Rc : float
            Calibration spectral resolution.
        sim : bool, optional
            Whether the data is simulated. Default is False.
        exposure_time : float, optional
            Exposure time in minutes. Default is None.
    """

    # Define chi² levels and labels
    levels_chi2 = [0, 2.30, 6.17, 11.83, 19.35, 28.74]  # 1σ, 2σ, ..., 5σ
    labels = ["1σ", "2σ", "3σ", "4σ", "5σ"]
    
    # Define parameters and their names
    params = [T_arr, lg_arr, vsini_arr, rv_arr]
    param_names = [r"$T \, [\mathrm{K}]$", r"$\lg \, [\mathrm{dex}]$", r"$Vsin(i) \, [\mathrm{km/s]}$", r"$RV \, [\mathrm{km/s]}$"]
    
    # Identify dimensions to remove (if size == 1)
    axes_to_marginalize = []
    params_to_remove = []
    
    if len(T_arr) == 1:
        axes_to_marginalize.append(0)
        params_to_remove.append(0)
    if len(lg_arr) == 1:
        axes_to_marginalize.append(1)
        params_to_remove.append(1)
    if len(vsini_arr) == 1:
        axes_to_marginalize.append(2)
        params_to_remove.append(2)
    if len(rv_arr) == 1:
        axes_to_marginalize.append(3)
        params_to_remove.append(3)
    
    # Remove elements in reverse order to avoid index shift issues
    for idx in sorted(params_to_remove, reverse=True):
        params.pop(idx)
        param_names.pop(idx)
    
    # Reduce the p_4D array by marginalizing over dimensions of size 1
    for axis in sorted(axes_to_marginalize, reverse=True):
        p_4D = np.nansum(p_4D, axis=axis)
    
    # Number of remaining dimensions
    ndim = len(params)

    # Estimate uncertainties
    uncertainties_1sigma, optimal_values = estimate_uncertainties_1sigma(p_4D, *params)
        
    xmin = np.array([np.nanmin(param) for param in params])
    xmax = np.array([np.nanmax(param) for param in params])
    
    # Plot
    if show:
        fig, axes = plt.subplots(ndim, ndim, figsize=(10, 10), dpi=300)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
        for i in range(ndim):
            for j in range(ndim):
                ax = axes[i, j]
                if j > i:  # Upper triangular part of the matrix
                    ax.axis("off")
                    continue
                elif i == j:  # Histograms on the diagonal
                    marginalized_p  = np.nansum(p_4D, axis=tuple(k for k in range(ndim) if k != i))
                    marginalized_p /= np.nansum(marginalized_p)
                    ax.step(params[i], marginalized_p, color="k", where="mid")
                    ax.axvline(optimal_values[i], color="k", linestyle="--")
                    sigma = uncertainties_1sigma[i]
                    if sigma is not None:
                        ax.axvline(optimal_values[i] - sigma, color="k", linestyle="--")
                        ax.axvline(optimal_values[i] + sigma, color="k", linestyle="--")
                    ax.set_yticks([])
                    ax.set_xlabel(param_names[i], fontsize=10)
                    ax.set_title(f"{param_names[i]} = {optimal_values[i]:.2f} ± {max(sigma, 0.01):.2f}", fontsize=10)
                    ax.set_xlim(xmin[i], xmax[i])
                elif j < i:  # Contour plots in the lower triangular part
                    marginalized_p  = np.nansum(p_4D, axis=tuple(k for k in range(ndim) if k != i and k != j))
                    marginalized_p /= np.nansum(marginalized_p)
                    marginalized_p[marginalized_p == 0] = np.nanmin(marginalized_p[marginalized_p != 0])
                    marginalized_p = marginalized_p.transpose()
                    marginalized_chi2  = -2 * np.log(marginalized_p)
                    marginalized_chi2 -= np.nanmin(marginalized_chi2)
                    cmap = plt.get_cmap("plasma")
                    filled_contour = ax.contourf(params[j], params[i], marginalized_chi2, levels=levels_chi2, cmap=cmap, alpha=0.8)
                    linewidths = [3 / (n / 2 + 2) for n in range(len(levels_chi2))]
                    contour = ax.contour(params[j], params[i], marginalized_chi2, levels=levels_chi2, colors="black", linewidths=linewidths)
                    fmt = {level: label for level, label in zip(levels_chi2[1:], labels)}
                    ax.clabel(contour, inline=True, fontsize=10, fmt=fmt)
                    ax.axvline(optimal_values[j], color="k", linestyle="--")
                    ax.axhline(optimal_values[i], color="k", linestyle="--")
                    ax.plot(optimal_values[j], optimal_values[i], "X", color="black")
                    if j == 0:
                        ax.set_ylabel(param_names[i], fontsize=10)
                    if i == ndim - 1:
                        ax.set_xlabel(param_names[j], fontsize=10)
                    ax.set_xlim(xmin[j], xmax[j])
                    ax.set_ylim(xmin[i], xmax[i])
    
                if i < ndim - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])
        
        # Set the title
        if sim:
            if exposure_time is None:
                fig.suptitle(f"Parameter estimation of {target_name} on {band}-band of {instru}\n with {model} model\n (with R = {round(R)} and $R_c$ = {Rc}) \n Simulation", fontsize=16, y=0.95)
            else:
                fig.suptitle(f"Parameter estimation of {target_name} on {band}-band of {instru}\n with {model} model\n (with "+"$t_{exp}$"+f" = {round(exposure_time)} min, R = {round(R)} and $R_c$ = {Rc}) \n Simulation", fontsize=16, y=0.95)
        else:
            if exposure_time is None:
                fig.suptitle(f"Parameter estimation of {target_name} on {band}-band of {instru}\n with {model} model\n (with R = {round(R)} and $R_c$ = {Rc})", fontsize=16, y=0.95)
            else:
                fig.suptitle(f"Parameter estimation of {target_name} on {band}-band of {instru}\n with {model} model\n (with "+"$t_{exp}$"+f" = {round(exposure_time)} min, R = {round(R)} and $R_c$ = {Rc})", fontsize=16, y=0.95)
        
        plt.show()
    
    return uncertainties_1sigma



    
    
    
    