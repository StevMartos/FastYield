from src.spectrum import *



def stellar_high_filtering(cube, R, Rc, filter_type, outliers=False, sigma_outliers=5, verbose=True, renorm_cube_res=False, only_high_pass=False, stell=None, show=False):
    """
    Post-processing filtering method according to molecular mapping, see Appendix B of Martos et al. (2024)

    Parameters
    ----------
     cube: 3d-array
        Data cube.
    renorm_cube_res: bool
        To renorm every spaxel of the final cube product by their norm, if True the CCF calculated will directly gives the correlation intensity
    R: float
        Spectral resolution of the cube.
    Rc: float
        Cut-off resolution of the filter. If Rc = None, no filter will be applyied
    filter_type: str
        type of filter used.
    outliers: bool, optional
        To filter outliers. The default is False.
    sigma_outliers: float, optional
        sigma value of the outliers filtering method (sigma clipping). The default is 5.
    only_high_pass: bool, optional
        In order to apply only a high-pass filter on the cube (whether than also subtracting the stellar component). The default is False.
    stell: 1D-array, optional
        To input the stellar spectrum. The default is False.

    Returns
    -------
    cube_res: 3d-array
        S_res.
    cube_M: 3d-array
        Estimated stellar modulation function, not really usefull.
    """
    cube = np.copy( cube)
    NbChannel, NbLine, NbColumn = cube.shape
    if stell is None:
        stell = np.nansum(cube, (1, 2)) # estimated stellar spectrum
    Y = np.reshape(cube, (NbChannel, NbLine*NbColumn))
    cube_M = np.copy(Y) ; m = 0
    for k in range(Y.shape[1]):
        if not all(np.isnan(Y[:, k])):
            if show:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3), layout="constrained", gridspec_kw={'wspace': 0.05,'hspace':0}, dpi=300) ; ax[0].set_xlabel("wavelength axis") ; ax[0].set_ylabel("modulation (normalized)") ; ax[1].set_yscale('log') ; ax[1].set_xscale('log') ; ax[1].set_xlabel("resolution frequency R") ; ax[1].set_ylabel("PSD") ; ax[0].plot(Y[:, k]/np.sqrt(np.nansum(Y[:,k]**2)),'r',zorder=10) ; Y_spectrum = Spectrum(None, Y[:, k][~np.isnan(Y[:,k])], R, None) ; res, psd = Y_spectrum.get_psd(smooth=0) ; ax[1].plot(res, psd,"r",label="raw (LF+HF)") ; Y_HF = filtered_flux(Y[:,k], R=R, Rc=Rc, filter_type=filter_type)[0] ; ax[0].plot(Y_HF/np.sqrt(np.nansum(Y_HF**2)),'g') ; Y_spectrum = Spectrum(None, Y_HF[~np.isnan(Y_HF)], R, None) ; res_HF, psd_HF = Y_spectrum.get_psd(smooth=0) ; ax[1].plot(res_HF, psd_HF,"g",label="HF") ; plt.title(f" F = {round(psd_HF[np.abs(res_HF-Rc).argmin()]/psd[np.abs(res-Rc).argmin()],3)}")
            if only_high_pass:
                _, Y_LF = filtered_flux(Y[:, k], R, Rc, filter_type)
                M = Y_LF/stell
            else:
                _, M = filtered_flux(Y[:, k]/stell, R, Rc, filter_type)
            cube_M[:, k] = Y[:, k]/stell #  True modulations (with noise), assuming that stell is the real observed stellar spectrum
            if outliers:
                sg = sigma_clip(Y[:, k]-stell*M, sigma=sigma_outliers)
                Y[:, k] = np.array(np.ma.masked_array(sg, mask=sg.mask).filled(np.nan))
            else:
                Y[:, k] = Y[:, k] - stell*M
            m += np.nansum(M)
            if show:
                ax[0].plot(Y[:, k]/np.sqrt(np.nansum(Y[:,k]**2)),"b") ; Y_spectrum = Spectrum(None, Y[:, k][~np.isnan(Y[:,k])], R, None) ; res, psd = Y_spectrum.get_psd(smooth=0) ; ax[1].plot(res, psd,"b",label="post MM filtering method") ; plt.legend() ; plt.show()
    if verbose:
        print("\n norme de M =", round(m/(NbChannel), 3)) # must be ~ 1
    cube_res =  Y.reshape((NbChannel, NbLine, NbColumn))
    cube_M = cube_M.reshape((NbChannel, NbLine, NbColumn))
    cube_res[cube_res == 0] = np.nan ; cube_M[cube_M == 0] = np.nan
    if renorm_cube_res: # renormalizing the spectra of every spaxel in order to directly have the correlation strength
        for i in range(NbLine):
            for j in range(NbColumn): # for every spaxel
                if not all(np.isnan(cube_res[:, i, j])): # ignoring nan values
                    cube_res[:, i, j] = cube_res[:, i, j]/np.sqrt(np.nansum(cube_res[:, i, j]**2))
    return cube_res, cube_M



def molecular_mapping_rv(instru, S_res, star_flux, T_planet, lg_planet, model, wave, trans, R, Rc, filter_type, rv=None, vsini_planet=0, verbose=True, template=None, pca=None):
    """
    Cross-correlating the residual cube S_res with templates, giving the CCF

    Parameters
    ----------
    instru : str
        Instrument's name.
    S_res : 3d-array
        Residual cube.
    T_planet : float
        PLanet's temperature.
    lg_planet : float
        Planet's surface gravity.
    model : str
        Planet's spectrum model.
    wave : 1d-array
        Wavelength axis.
    trans : 1d-ar-ay
        Total system transmission.
    R: float
        Spectral resolution of the cube.
    Rc: float
        Cut-off resolution of the filter. If Rc = None, no filter will be applyied
    filter_type: str
        type of filter used.
    rv : float, optional
        Planet's radial velocity. The default is None.
    vsini_planet : float, optional
        Planet's rotationnal speed. The default is 0.
    pca : pca, optional
        If PCA is applied, it is necessary to also subtract the components to the template. The default is None.
    """
    if template is None:
        template = load_planet_spectrum(T_planet, lg_planet, model=model, instru=instru, interpolated_spectrum=True) # loading the template
        if model[:4] == "mol_" and Rc is not None: # to crop empty features regions in molecular templates
            _, planet_spectrum_LF = filtered_flux(template.flux, R=template.R, Rc=Rc, filter_type=filter_type)
            sg = sigma_clip(planet_spectrum_LF, sigma=1)
            template.flux[~sg.mask] = np.nan 
        template.crop(0.98*wave[0], 1.02*wave[-1])
        dl = template.wavelength - np.roll(template.wavelength, 1) ; dl[0] = dl[1] # delta lambda array
        Rold = np.nanmax(template.wavelength/(2*dl))
        if Rold > 200000: # takes too much time otherwise (limiting then the resolution to 200 000 => does not change anything)
            Rold = 200000
        dl = np.nanmean(template.wavelength/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist sampling (Shannon)
        wave_inter = np.arange(0.98*wave[0], 1.02*wave[-1], dl) # regularly sampled template
        template = template.interpolate_wavelength(wave_inter, renorm=False)
        if vsini_planet > 0: # broadening the spectrum
            template = template.broad(vsini_planet)
        if model[:4] != "mol_":
            template.flux *= wave_inter # for the template to be homogenous to photons or e- or ADU
    template = template.degrade_resolution(wave, renorm=False)
    template_HF = template.copy() ; template_LF = template.copy()
    template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type) # high pass filtering
    f = interp1d(wave[~np.isnan(star_flux/trans)], (star_flux/trans)[~np.isnan(star_flux/trans)], bounds_error=False, fill_value=np.nan) 
    star_flux = f(wave) 
    star_HF, star_LF = filtered_flux(star_flux, R, Rc, filter_type) # high pass filtering
    NbChannel, NbLine, NbColumn = S_res.shape
    if rv is None:
        rv = np.linspace(-50, 50, 201)
    else:
        rv = np.array([rv])
    CCF = np.zeros((len(rv), NbLine, NbColumn))
    for k in range(len(rv)):
        if verbose:
            print(" CCF for: rv = ", rv[k], " km/s & Tp =", T_planet, "K & lg = ", lg_planet)
        template_HF_shift = template_HF.doppler_shift(rv[k]).flux # shifting the tempalte
        template_LF_shift = template_LF.doppler_shift(rv[k]).flux # shifting the tempalte
        template_shift = trans*template_HF_shift# - trans * star_HF * template_LF_shift/star_LF # it should be almost the same thing with or without the residual star flux, for the 2D CCF it is not considered by default beacause this residual term could project on systematics and accentuate the spatial systematic noise contribution
        if pca is not None: # subtraction of the PCA's modes to the template
            template0_shift = np.copy(template_shift)
            n_comp_sub = pca.n_components
            for nk in range(n_comp_sub): 
                template_shift -= np.nan_to_num(np.nansum(template0_shift*pca.components_[nk])*pca.components_[nk])
        for i in range(NbLine):
            for j in range(NbColumn): # for every spaxel
                if not all(np.isnan(S_res[:, i, j])): # ignoring all nan values spaxels
                    d = np.copy(S_res[:, i, j])
                    t = np.copy(template_shift)
                    t[np.isnan(d)] = np.nan
                    t /= np.sqrt(np.nansum(t**2))  # normalizing the template
                    CCF[k, i, j] = np.nansum(d*t) # cross-correlation between the residual signal and the template
    CCF[CCF == 0] = np.nan
    if len(rv) == 1:
        return CCF[0], template_shift
    else:
        return CCF, rv



########################################################################################################################################################################################################################################################################################################################################################################################################



def correlation_rv(instru, d_planet, d_bkgd, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, large_rv=False, rv=None, rv_planet=None, vsini_planet=None, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, stellar_component=True, degrade_resolution=True, disable_tqdm=False, show=True, smooth_PSD=1, target_name=None, comparison=False):
    if rv is None:
        if large_rv:
            rv = np.linspace(-10000, 10000, 4001)
        else:
            rv = np.linspace(-1000, 1000, 4001)
    CCF_planet = np.zeros_like(rv) ; corr_planet = np.zeros_like(rv) ; corr_auto = np.zeros_like(rv) ;  corr_auto_noise = np.zeros_like(rv) ; logL_planet = np.zeros_like(rv) ; sigma_planet = None 
    if d_bkgd is not None:
        CCF_bkgd = np.zeros((len(d_bkgd), len(rv)))
    else:
        CCF_bkgd = None
    if template is None: # if no template injected
        template = load_planet_spectrum(T_planet, lg_planet, model=model, instru=instru, interpolated_spectrum=True) # loading the template model
        if model[:4] == "mol_" and Rc is not None : # to crop empty features regions in molecular templates
            _, template_LF = filtered_flux(template.flux, R=template.R, Rc=Rc, filter_type=filter_type)
            sg = sigma_clip(template_LF, sigma=1)
            template.flux[~sg.mask] = np.nan 
        template.crop(0.98*wave[0], 1.02*wave[-1])
        dl = template.wavelength - np.roll(template.wavelength, 1) ; dl[0] = dl[1] # delta lambda array
        Rold = np.nanmax(template.wavelength/(2*dl))
        if Rold > 200000: # takes too much time otherwise (limiting then the resolution to 200 000 => does not seem to change anything)
            Rold = 200000
        dl = np.nanmean(template.wavelength/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist sampling (Shannon)
        wave_inter = np.arange(0.98*wave[0], 1.02*wave[-1], dl) # regularly sampled template
        template = template.interpolate_wavelength(wave_inter, renorm=False)
        if vsini_planet > 0: # broadening the spectrum
            template = template.broad(vsini_planet, epsilon=epsilon, fastbroad=fastbroad)
        if model[:4] != "mol_":
            template.flux *= wave_inter # for the template to be homogenous to photons or e- or ADU
    if degrade_resolution:
        template = template.degrade_resolution(wave, renorm=False) # degrating the template to the instrumental resolution
    else:
        template = template.interpolate_wavelength(wave, renorm=False)
    if all(np.isnan(template.flux)): # if the crop of the molecular templates left only NaNs
        return rv, CCF_planet, corr_planet, CCF_bkgd, corr_auto, logL_planet, sigma_planet
    template_HF = template.copy() ; template_LF = template.copy()
    template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type) # high pass filtering
    if stellar_component: # trans needs to be in star_flux ! 
        if star_flux is None:
            raise KeyError("star_flux is not defined for the stellar component !")
        f = interp1d(wave[~np.isnan(star_flux/trans)], (star_flux/trans)[~np.isnan(star_flux/trans)], bounds_error=False, fill_value=np.nan) 
        sf = f(wave) 
        if instru=="HiRISE" and 1==1: # Handling filtering edge effects due to the gaps bewteen the orders
            _, star_LF = filtered_flux(sf, R, Rc, filter_type) # high pass filtering
            nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
            f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
            sf[nan_values] = f(wave[nan_values])
        star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type) # high pass filtering
        #plt.figure(dpi=300); plt.title("Star's signal") ; plt.plot(wave, sf, "r", label="LF+HF") ; plt.plot(wave, star_LF, "g", label="LF") ; plt.plot(wave, star_HF, "b", label="HF") ; plt.xlabel("wavelength [µm]") ; plt.ylabel("flux") ; plt.grid(True) ; plt.legend() ; plt.show()
    if rv_planet is not None: # if the radial of the planet is known, 
        d_planet_sim = get_d_planet_sim(d_planet=d_planet, wave=wave, trans=trans, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, degrade_resolution=degrade_resolution, stellar_component=stellar_component, star_flux=star_flux, instru=instru, epsilon=epsilon, fastbroad=fastbroad)
        if weight is not None:
            d_planet_sim *= weight
    for i in tqdm(range(len(rv)), desc="correlation_rv()", disable=disable_tqdm): # for each rv value
        template_HF_shift = template_HF.doppler_shift(rv[i]).flux # shift a template at her known rv in order to mimic it (for the auto-correlation)
        t = trans*template_HF_shift
        if stellar_component:
            template_LF_shift = template_LF.doppler_shift(rv[i]).flux # shift a template at her known rv in order to mimic it (for the auto-correlation)
            t += - trans * star_HF * template_LF_shift/star_LF # the residual star flux need to be taken into account in order to not avoid bias in the parameters retrieval
        if pca is not None: # subtracting PCA modes (if any)
            t0 = np.copy(t)
            n_comp_sub = pca.n_components
            for nk in range(n_comp_sub): 
                t -= np.nan_to_num(np.nansum(t0*pca.components_[nk])*pca.components_[nk])
        d_p = np.copy(d_planet)
        if weight is not None:
            d_p *= weight ; t *= weight
        d_p[d_p == 0] = np.nan
        t[t == 0] = np.nan
        t[np.isnan(d_p)] = np.nan
        d_p[np.isnan(t)] = np.nan
        t /= np.sqrt(np.nansum(t**2)) # normalizing the tempalte
        CCF_planet[i] = np.nansum(d_p*t) # projection of the planet's signal
        corr_planet[i] = np.nansum(d_p*t) / np.sqrt(np.nansum(d_p**2)) # projection of the planet's signal
        #plt.figure(dpi=300) ; plt.title(f" rv = {round(rv[i], 1)} km/s") ; plt.plot(wave, d_p / np.sqrt(np.nansum(d_p**2)), 'r', label="data") ; plt.plot(wave, t, 'b', label="template") ; plt.legend() ; plt.xlabel("wavelength [µm]") ; plt.ylabel("high-pass filtered flux") ; plt.show()
        if rv_planet is not None: # auto correlation
            corr_auto[i] = np.nansum(d_planet_sim*t) / np.sqrt(np.nansum(d_planet_sim[~np.isnan(t)]**2)) # auto-correlation
        if d_bkgd is not None: # background projections
            for j in range(len(d_bkgd)):
                CCF_bkgd[j,i] = np.nansum(d_bkgd[j]*t)
        if logL: # likelihood calculations 
            logL_planet[i] = get_logL(d_p, t, sigma_l, method=method_logL)
    if rv_planet is not None: # projection of sigma 
        if sigma_l is not None:
            sigma_planet = np.sqrt(np.nansum(sigma_l**2*d_planet_sim**2)) / np.sqrt(np.nansum(d_planet_sim**2))
        if show: # showing the template and the extracted planet spectrum at rv_planet
            if target_name is not None:
                target_name = target_name.replace("_"," ") + " "
            d_planet = np.copy(d_planet)
            if weight is not None:
                d_planet *= weight
            # Creating subplots
            fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=300, gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
            if comparison:
                fig.suptitle(f"{instru} {target_name}data sets, with R={int(R)} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d_planet * d_planet_sim) / np.sqrt(np.nansum(d_planet**2)*np.nansum(d_planet_sim**2)), 2)}", fontsize=20)
            else:
                fig.suptitle(f"{instru} {target_name}data and {model} template, with $T$={round(T_planet)}K, lg={round(lg_planet, 1)}, rv={round(rv_planet, 1)}km/s, R={int(R)} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d_planet * d_planet_sim) / np.sqrt(np.nansum(d_planet**2)*np.nansum(d_planet_sim**2)), 2)}", fontsize=20)
            # Plotting high-pass filtered data and template (top left subplot)
            axs[0, 0].set_ylabel("high-pass flux (normalized)", fontsize=14)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
            axs[0, 0].plot(wave, d_planet / np.sqrt(np.nansum(d_planet ** 2)), 'r', label=f"{target_name}data")
            if comparison:
                axs[0, 0].plot(wave, d_planet_sim / np.sqrt(np.nansum(d_planet_sim**2)), 'b', label=model)
            else:
                axs[0, 0].plot(wave, d_planet_sim / np.sqrt(np.nansum(d_planet_sim**2)), 'b', label=model + " template")
            if sigma_l is not None:
                d_planet_sim_noise = np.copy(d_planet_sim)                
                noise = np.random.normal(0, sigma_l, len(wave))
                if degrade_resolution:
                    noise = Spectrum(wave, noise, None, None).degrade_resolution(wave, renorm=False).flux
                d_planet_sim_noise += noise
                noise /= np.sqrt(np.nansum(d_planet_sim_noise ** 2)) # for plot purposes
                d_planet_sim_noise /= np.sqrt(np.nansum(d_planet_sim_noise ** 2))  # normalizing
                if not comparison:
                    axs[0, 0].plot(wave, d_planet_sim_noise, 'b', label=model + " template w/ expected noise "+r"($cos\theta_n$ = "+f"{round(np.nansum(d_planet_sim_noise * d_planet_sim) / np.sqrt(np.nansum(d_planet_sim_noise**2)*np.nansum(d_planet_sim**2)), 2)})", alpha=0.5)
                    res_template_noise, psd_template_noise = calc_psd(wave, d_planet_sim_noise, R, smooth=smooth_PSD)
                    axs[0, 1].plot(res_template_noise, psd_template_noise, 'b', alpha=0.5, zorder=10)
                axs[1, 0].plot(wave, noise, 'g', label="expected noise", zorder=3, alpha=0.5)
                res_noise, psd_noise = calc_psd(wave, noise, R, smooth=smooth_PSD)
                axs[1, 1].plot(res_noise, psd_noise, 'g', zorder=10, alpha=0.5)
            axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
            axs[0, 0].minorticks_on()
            axs[0, 0].legend(fontsize=14)
            axs[0, 0].set_ylim(2*np.nanmin(d_planet / np.sqrt(np.nansum(d_planet ** 2))), 2*np.nanmax(d_planet / np.sqrt(np.nansum(d_planet ** 2))))
            # Calculating PSDs
            res_planet, psd_planet = calc_psd(wave, d_planet / np.sqrt(np.nansum(d_planet ** 2)), R, smooth=smooth_PSD)
            res_template, psd_template = calc_psd(wave, d_planet_sim / np.sqrt(np.nansum(d_planet_sim ** 2)), R, smooth=smooth_PSD)
            # Plotting PSDs (top right subplot)
            axs[0, 1].set_ylabel("PSD", fontsize=14)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_xlim(10, R)
            axs[0, 1].plot(res_planet, psd_planet, 'r')
            axs[0, 1].plot(res_template, psd_template, 'b')
            axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
            axs[0, 1].minorticks_on()
            # Calculating residuals
            residuals = d_planet / np.sqrt(np.nansum(d_planet ** 2)) - d_planet_sim / np.sqrt(np.nansum(d_planet_sim ** 2))
            # Plotting residuals (bottom left subplot)
            axs[1, 0].set_xlabel("wavelength [µm]", fontsize=14) ; axs[1, 0].set_ylabel("residuals", fontsize=14)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
            axs[1, 0].set_xlim(np.nanmin(wave[~np.isnan(residuals)]), np.nanmax(wave[~np.isnan(residuals)]))
            #axs[1, 0].set_xlim(1.665, 1.668)
            #axs[1, 0].set_xlim(1.648, 1.654)
            axs[1, 0].set_ylim(-5*np.nanstd(residuals), 5*np.nanstd(residuals))
            if comparison:
                axs[1, 0].plot(wave, residuals, 'k', label="data1 - data2")
            else:
                axs[1, 0].plot(wave, residuals, 'k', label="data - template")
            axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[1, 0].minorticks_on()
            axs[1, 0].legend(fontsize=14)
            # Calculating PSD of residuals
            res_residuals, psd_residuals = calc_psd(wave, residuals, R, smooth=smooth_PSD)
            # Plotting PSD of residuals (bottom right subplot)
            axs[1, 1].set_xlabel("resolution R", fontsize=14) ; axs[1, 1].set_ylabel("PSD", fontsize=14)
            axs[1, 1].set_xscale('log')
            axs[1, 1].set_yscale('log')
            axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
            axs[1, 1].set_xlim(10, R)
            axs[1, 1].plot(res_residuals, psd_residuals, 'k')
            axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[1, 1].minorticks_on()
            # Display the combined figure
            plt.tight_layout()
            plt.show()
    return rv, CCF_planet, corr_planet, CCF_bkgd, corr_auto, logL_planet, sigma_planet



def plot_CCF_rv(instru, band, target_name, d_planet, d_bkgd, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, rv=None, rv_planet=None, vsini_planet=0, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, stellar_component=True, degrade_resolution=True, show=True, smooth_PSD=1, comparison=False): 
    rv, CCF_planet, corr_planet, CCF_bkgd, corr_auto, logL_planet, sigma_planet = correlation_rv(instru=instru, d_planet=d_planet, d_bkgd=d_bkgd, star_flux=star_flux, wave=wave, trans=trans, T_planet=T_planet, lg_planet=lg_planet, model=model, R=R, Rc=Rc, filter_type=filter_type, large_rv=True, rv=rv, rv_planet=rv_planet, vsini_planet=vsini_planet, pca=pca, template=template, epsilon=epsilon, fastbroad=fastbroad, logL=logL, method_logL=method_logL, sigma_l=sigma_l, weight=weight, stellar_component=stellar_component, degrade_resolution=degrade_resolution, show=show, smooth_PSD=smooth_PSD, target_name=target_name, comparison=comparison)
    rv_planet = rv[(rv<rv_planet+25)&(rv>rv_planet-25)][CCF_planet[(rv<rv_planet+25)&(rv>rv_planet-25)].argmax()]
    # The residual stellar component and systematic effects can introduce an offset in the CCFs.
    CCF_planet -= np.nanmean(CCF_planet[(rv>rv_planet+200)|(rv<rv_planet-200)])
    # corr_planet -= np.nanmean(corr_planet[(rv>rv_planet+200)|(rv<rv_planet-200)]) # not needed since we are only interested on the max correlation value (without subtracting the potential offset)
    corr_auto -= np.nanmean(corr_auto[(rv>rv_planet+200)|(rv<rv_planet-200)])
    if d_bkgd is not None:
        for i in range(len(d_bkgd)):
            CCF_bkgd[i] -= np.nanmean(CCF_bkgd[i])
    # NOISE ESTIMATIONS as function of RV # https://arxiv.org/pdf/2405.13469: std(rv_planet +- 200 km/s)
    sigma2_tot = np.nanvar(CCF_planet[(rv>rv_planet+200)|(rv<rv_planet-200)]) # Variance
    sigma2_auto = np.nanvar(corr_auto[(rv>rv_planet+200)|(rv<rv_planet-200)]*np.nanmax(CCF_planet[(rv<rv_planet+25)&(rv>rv_planet-25)])/np.nanmax(corr_auto))
    if sigma2_auto < sigma2_tot:
        sigma_CCF = np.sqrt(sigma2_tot - sigma2_auto) # NOISE ESTIMATION = sqrt(var(signal) - var(auto correlation))
    else:
        sigma_CCF = np.sqrt(sigma2_tot) # NOISE ESTIMATION = sqrt(var(signal) - var(auto correlation))
    SNR_planet = CCF_planet / sigma_CCF
    max_SNR = np.nanmax(SNR_planet[(rv<rv_planet+25)&(rv>rv_planet-25)])
    plt.figure(dpi=300) # CCF plot (in S/N and correlation units) 
    ax1 = plt.gca() ; ax1.grid(True, which='both', linestyle='--', linewidth=0.5) ; ax1.minorticks_on() ; ax1.set_xlim(rv[0], rv[-1]) ; ax1.set_title(f'CCF of {target_name} on {band}-band of {instru} with {model} template \n at $T$ = {round(T_planet)}K, $lg$ = {round(lg_planet,1)} and Vsini = {round(vsini_planet,1)} km/s with $R_c$ = {Rc}',fontsize=16) ; ax1.set_xlabel("radial velocity [km/s]",fontsize=14) ; ax1.set_ylabel("CCF [S/N]",fontsize=14)
    ax1.plot([],[],'gray',label=f"noise",alpha=0.5)
    if d_bkgd is not None:
        SNR_bkgd = np.zeros((len(d_bkgd), len(rv)))
        for i in range(len(d_bkgd)):
            SNR_bkgd[i] = CCF_bkgd[i]/np.nanstd(CCF_bkgd[i])
            ax1.plot(rv, SNR_bkgd[i], 'gray', alpha=max(0.1,1/len(d_bkgd)))
    else:
        SNR_bkgd = None
    ax1.plot(rv, corr_auto*max_SNR/np.nanmax(corr_auto), "k", label=f"auto-correlation")
    ax1.plot(rv, SNR_planet, label=f"{target_name}", zorder=3)
    ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin,ymax)
    ax1.plot([rv_planet, rv_planet], [ymin, ymax], 'r:', label=f"rv = {round(rv_planet, 1)} km/s")
    ax2 = ax1.twinx() ; ax2.set_ylabel("correlation strength", fontsize=14, labelpad=20, rotation=270) ; ax2.tick_params(axis='y')  
    ax2.set_ylim(ymin*np.nanmax(corr_planet)/max_SNR, ymax*np.nanmax(corr_planet)/max_SNR)
    ax1.legend(loc="upper right") ; ax1.set_zorder(1) ; plt.show()
    if sigma_l is not None:
        print(" error on sigma (sigma CCF/sigma planet) = ", round(sigma_CCF/sigma_planet,3))
    print(f" CCF: max S/N ({round(max_SNR, 1)}) and correlation ({round(np.nanmax(corr_planet), 5)}) for rv = {round(rv_planet,2)} km/s")
    if logL:
        rv_planet = rv[(rv<rv_planet+25)&(rv>rv_planet-25)][logL_planet[(rv<rv_planet+25)&(rv>rv_planet-25)].argmax()]
        plt.figure(dpi=300)
        ax1 = plt.gca() ; ax1.grid(True, which='both', linestyle='--', linewidth=0.5) ; ax1.minorticks_on() ; ax1.set_xlim(rv[0],rv[-1]) ; ax1.set_title(f'logL ({method_logL}) of {target_name} on {band}-band of {instru} with {model} \n at $T_p$ = {round(T_planet)}K and Vsin(i) = {vsini_planet} km/s with $R_c$ = {Rc}',fontsize=16) ; ax1.set_xlabel("radial velocity [km/s]",fontsize=14) ; ax1.set_ylabel("logL",fontsize=14)
        ax1.plot(rv, logL_planet, label=f"planet", zorder=3)
        ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin, ymax)
        ax1.plot([rv_planet, rv_planet], [ymin, ymax], 'r:', label=f"rv = {round(rv_planet, 1)} km/s")
        ax1.legend(loc="upper right") ; ax1.set_zorder(1) ; plt.show()
        print(f" max logL for rv = {round(rv_planet, 2)} km/s")
    return rv, SNR_planet, SNR_bkgd, max_SNR



########################################################################################################################################################################################################################################################################################################################################################################################################



def correlation_vsini(instru, d_planet, d_bkgd, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, vsini, rv_planet=None, vsini_planet=None, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, stellar_component=True, degrade_resolution=True, disable_tqdm=False):
    CCF_planet = np.zeros_like(vsini) ; corr_planet = np.zeros_like(vsini) ; corr_auto = np.zeros_like(vsini) ; corr_auto_noise = np.zeros_like(vsini) ; logL_planet = np.zeros_like(vsini) ; sigma_planet = None 
    if d_bkgd is not None:
        CCF_bkgd = np.zeros((len(d_bkgd), len(vsini)))
    else:
        CCF_bkgd = None
    if template is None: # if no template injected
        template = load_planet_spectrum(T_planet, lg_planet, model=model, instru=instru, interpolated_spectrum=True) # loading the template model
        if model[:4] == "mol_" and Rc is not None: # to crop empty features regions in molecular templates
            _, template_LF = filtered_flux(template.flux, R=template.R, Rc=Rc, filter_type=filter_type)
            sg = sigma_clip(template_LF, sigma=1)
            template.flux[~sg.mask] = np.nan 
        template.crop(0.98*wave[0], 1.02*wave[-1])
        dl = template.wavelength - np.roll(template.wavelength, 1) ; dl[0] = dl[1] # delta lambda array
        Rold = np.nanmax(template.wavelength/(2*dl))
        if Rold > 200000: # takes too much time otherwise (limiting then the resolution to 200 000 => does not seem to change anything)
            Rold = 200000
        dl = np.nanmean(template.wavelength/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist sampling (Shannon)
        wave_inter = np.arange(0.98*wave[0], 1.02*wave[-1], dl) # regularly sampled template
        template = template.interpolate_wavelength(wave_inter, renorm=False)
        if rv_planet != 0: # Doppler shifting [km/s]
            template = template.doppler_shift(rv_planet)
        if model[:4] != "mol_":
            template.flux *= wave_inter # for the template to be homogenous to photons or e- or ADU
    if degrade_resolution:
        template = template.degrade_resolution(wave, renorm=False) # degrating the template to the instrumental resolution
    else:
        template = template.interpolate_wavelength(wave, renorm=False)
    if all(np.isnan(template.flux)): # if the crop of the molecular templates left only NaNs
        return CCF_planet, corr_planet, CCF_bkgd, corr_auto, logL_planet, sigma_planet
    template_HF = template.copy() ; template_LF = template.copy()
    template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type) # high pass filtering
    if stellar_component: # trans needs to be in star_flux ! 
        if star_flux is None:
            raise KeyError("star_flux is not defined for the stellar component !")
        f = interp1d(wave[~np.isnan(star_flux/trans)], (star_flux/trans)[~np.isnan(star_flux/trans)], bounds_error=False, fill_value=np.nan) 
        sf = f(wave) 
        if instru=="HiRISE" and 1==1: # Handling filtering edge effects due to the gaps bewteen the orders
            _, star_LF = filtered_flux(sf, R, Rc, filter_type) # high pass filtering
            nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
            f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
            sf[nan_values] = f(wave[nan_values])
        star_HF, star_LF = filtered_flux(sf, R, Rc, filter_type) # high pass filtering
        #plt.figure(dpi=300); plt.title("Star's signal") ; plt.plot(wave, sf, "r", label="LF+HF") ; plt.plot(wave, star_LF, "g", label="LF") ; plt.plot(wave, star_HF, "b", label="HF") ; plt.xlabel("wavelength [µm]") ; plt.ylabel("flux") ; plt.grid(True) ; plt.legend() ; plt.show()
    if rv_planet is not None: # if the radial of the planet is known, 
        d_planet_sim = get_d_planet_sim(d_planet=d_planet, wave=wave, trans=trans, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, degrade_resolution=degrade_resolution, stellar_component=stellar_component, star_flux=star_flux, instru=instru, epsilon=epsilon, fastbroad=fastbroad)
        if weight is not None:
            d_planet_sim *= weight
    for i in tqdm(range(len(vsini)), desc="correlation_vsini()"): # for each rv value
        if vsini[i] > 0: # broadening a template at her known rv in order to mimic it (for the auto-correlation)
            template_HF_broad = template_HF.broad(vsini[i], epsilon=epsilon, fastbroad=fastbroad).flux # for the auto-correlation
            template_LF_broad = template_LF.broad(vsini[i], epsilon=epsilon, fastbroad=fastbroad).flux # for the auto-correlation
        else:
            template_HF_broad = template_HF.copy().flux
            template_LF_broad = template_LF.copy().flux
        t = trans*template_HF_broad
        if stellar_component:
            t += - trans * star_HF * template_LF_broad/star_LF # the residual star flux need to be taken into account in order to not avoid bias in the parameters retrieval
        if pca is not None: # subtracting PCA modes (if any)
            t0 = np.copy(t)
            n_comp_sub = pca.n_components
            for nk in range(n_comp_sub): 
                t -= np.nan_to_num(np.nansum(t0*pca.components_[nk])*pca.components_[nk])
        d_p = np.copy(d_planet)
        if weight is not None:
            d_p *= weight ; t *= weight
        d_p[d_p == 0] = np.nan
        t[t == 0] = np.nan
        t[np.isnan(d_p)] = np.nan
        d_p[np.isnan(t)] = np.nan
        t /= np.sqrt(np.nansum(t**2)) # normalizing the tempalte
        CCF_planet[i] = np.nansum(d_p*t) # projection of the planet's signal
        corr_planet[i] = np.nansum(d_p*t) / np.sqrt(np.nansum(d_p**2)) # projection of the planet's signal
        if vsini_planet is not None: # auto correlation
            corr_auto[i] = np.nansum(d_planet_sim*t) / np.sqrt(np.nansum(d_planet_sim[~np.isnan(t)]**2)) # auto-correlation
        if d_bkgd is not None: # background projections
            for j in range(len(d_bkgd)):
                CCF_bkgd[j,i] = np.nansum(d_bkgd[j]*t)
        if logL: # likelihood calculations 
            logL_planet[i] = get_logL(d_p, t, sigma_l, method=method_logL)
    if vsini_planet is not None and sigma_l is not None:
        sigma_planet = np.sqrt(np.nansum(sigma_l**2*d_planet_sim**2)) / np.sqrt(np.nansum(d_planet_sim**2))
    return CCF_planet, corr_planet, CCF_bkgd, corr_auto, logL_planet, sigma_planet


    
def plot_CCF_vsini(instru, band, target_name, d_planet, d_bkgd, star_flux, wave, trans, T_planet, lg_planet, model, R, Rc, filter_type, vsini=None, rv_planet=None, vsini_planet=0, pca=None, template=None, epsilon=0.8, fastbroad=True, logL=False, method_logL="classic", sigma_l=None, weight=None, show=True, stellar_component=True, degrade_resolution=True):
    CCF_planet, corr_planet, CCF_bkgd, corr_auto, logL_planet, sigma_planet = correlation_vsini(instru=instru, d_planet=d_planet, d_bkgd=d_bkgd, star_flux=star_flux, wave=wave, trans=trans, T_planet=T_planet, lg_planet=lg_planet, model=model, R=R, Rc=Rc, filter_type=filter_type, vsini=vsini, rv_planet=rv_planet, vsini_planet=vsini_planet, pca=pca, template=template, epsilon=epsilon, fastbroad=fastbroad, logL=logL, method_logL=method_logL, sigma_l=sigma_l, weight=weight, stellar_component=stellar_component, degrade_resolution=degrade_resolution)
    SNR_planet = CCF_planet/sigma_planet
    vsini_planet = vsini[SNR_planet.argmax()]
    plt.figure(dpi=300) # CCF plot (in S/N and correlation units) 
    ax1 = plt.gca() ; ax1.grid(True) ; ax1.set_xlim(vsini[0], vsini[-1]) ; ax1.set_title(f'CCF of {target_name} on {band}-band of {instru} with {model} \n at $T_p$ = {T_planet}K and RV = {round(rv_planet,1)} km/s with $R_c$ = {Rc}',fontsize=16) ; ax1.set_xlabel("rotational broadening (in km/s)",fontsize=14) ; ax1.set_ylabel("CCF (in S/N)",fontsize=14)
    ax1.plot([], [], 'gray', label=f"noise", alpha=0.5)
    for i in range(len(d_bkgd)):
        CCF_bkgd[i] -= np.nanmean(CCF_bkgd[i])
        ax1.plot(vsini, CCF_bkgd[i]/np.nanstd(CCF_bkgd[i]), 'gray', alpha=max(0.1,1/len(d_bkgd)))
    ax1.plot(vsini, corr_auto*np.nanmax(SNR_planet)/np.nanmax(corr_auto), "k", label=f"auto-correlation")
    ax1.plot(vsini, SNR_planet, label=f"planet", zorder=3)
    ax1.set_xlim(0, vsini[-1])
    ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin, ymax)
    ax1.plot([vsini_planet, vsini_planet], [ymin, ymax], 'r:', label=f"Vsini = {round(vsini_planet,1)} km/s")
    ax2 = ax1.twinx() ; ax2.set_ylabel('correlation strength', fontsize=14, labelpad=20, rotation=270) ; ax2.tick_params(axis='y')  
    ax2.set_ylim(ymin*np.nanmax(corr_planet)/np.nanmax(SNR_planet), ymax*np.nanmax(corr_planet)/np.nanmax(SNR_planet)) 
    # pas vraiment correct: SNR_planet = signal_CCF / sigma_CCF sauf que sigma_CCF ici semble dépendre de manière non négligeable de Vsin(i), donc il ne suffit pas de renormaliser à une valeur pour avoir toutes les valeurs de corr (contrairement à la CCF en fonction de rv, où sigma_CCF = cste par définition)
    # cf. : plt.plot(vsini, SNR_planet/np.nanmax(SNR_planet)) ; plt.plot(vsini, corr_planet/np.nanmax(corr_planet)) # par contre CCF_planet et corr_planet vsini sont evidémment identiques (à un facteur de normalisation près)
    ax1.legend(loc="lower right") ; ax1.set_zorder(1) ; plt.show()
    print(f" CCF: max S/N ({round(np.nanmax(SNR_planet),1)}) and correlation ({round(np.nanmax(corr_planet),2)}) for Vsini = {round(vsini_planet,1)} km/s")
    if logL:
        vsini_planet = vsini[logL_planet.argmax()]
        plt.figure(dpi=300) # CCF plot (in S/N and correlation units) 
        ax1 = plt.gca() ; ax1.grid(True) ; ax1.set_xlim(vsini[0], vsini[-1]) ; ax1.set_title(f'logL ({method_logL}) of {target_name} on {band}-band of {instru} with {model} \n at $T_p$ = {T_planet}K and RV = {round(rv_planet,1)} km/s with $R_c$ = {Rc}',fontsize=16) ; ax1.set_xlabel("rotational broadening (in km/s)",fontsize=14) ; ax1.set_ylabel("logL",fontsize=14)
        ax1.plot(vsini, logL_planet, label=f"planet", zorder=3)
        ymin, ymax = ax1.get_ylim() ; ax1.set_ylim(ymin, ymax)
        ax1.plot([vsini_planet, vsini_planet], [ymin, ymax], 'r:', label=f"Vsini = {round(vsini_planet,1)} km/s")
        ax1.legend(loc="lower right") ; ax1.set_zorder(1) ; plt.show()
        print(f" max for logL: max for Vsini = {round(vsini_planet,1)} km/s")
    return SNR_planet



########################################################################################################################################################################################################################################################################################################################################################################################################



def SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, verbose=True, snr_calc=True):
        NbLine, NbColumn = CCF.shape
        R_planet = int(round(np.sqrt((y0-NbLine//2)**2+(x0-NbColumn//2)**2)))
        if R_planet == 0: 
            CCF = CCF*annular_mask(0, NbLine//2, value=np.nan, size=(NbLine, NbColumn))
            CCF_noise = np.copy(CCF_wo_planet)*annular_mask(3*size_core, 4*size_core, value=np.nan, size=(NbLine, NbColumn))
        else:
            CCF = CCF*annular_mask(max(R_planet-3*size_core-1, 0), R_planet+3*size_core, value=np.nan, size=(NbLine, NbColumn))
            if snr_calc:
                CCF_noise = np.copy(CCF_wo_planet)*annular_mask(max(1, R_planet-size_core-1), R_planet+size_core+1, value=np.nan, size=(NbLine, NbColumn))
            else:
                CCF_noise = np.copy(CCF_wo_planet)*annular_mask(max(1, R_planet-1), max(2, R_planet), value=np.nan, size=(NbLine, NbColumn))
        CCF_signal = CCF[y0, x0]
        noise = np.sqrt(np.nanvar(CCF_noise))
        if verbose:
            print(" E[<n, t>]/Std[<n, t>] = ", round(100*np.nanmean(CCF_noise)/np.nanstd(CCF_noise), 2), "%")
        signal = CCF_signal-np.nanmean(CCF_noise)
        SNR = signal/noise
        return SNR, CCF, CCF_signal, CCF_noise
    


########################################################################################################################################################################################################################################################################################################################################################################################################



def correlation_T_lg(instru, d_planet, star_flux, wave, trans, R, Rc, filter_type, target_name, band, model="BT-Settl", rv_planet=0, vsini_planet=0, pca=None, template=None, weight=None, stellar_component=True):
    T_planet, lg_planet = get_model_grid(model)
    rv = np.linspace(rv_planet-25, rv_planet+25, 100)
    corr_3d = np.zeros((len(lg_planet), len(T_planet), len(rv)))
    for j in tqdm(range(len(T_planet)), desc="correlation_T_lg()"):
        for i in range(len(lg_planet)):
            T = T_planet[j]
            if model[:4] == "mol_":
                model = model[:4] + lg_planet[i]
                lg = 4
            else:
                lg = lg_planet[i]
            _, _, corr_planet, _, _, _, _ = correlation_rv(instru=instru, d_planet=d_planet, d_bkgd=None, star_flux=star_flux, wave=wave, trans=trans, T_planet=T, lg_planet=lg, model=model, R=R, Rc=Rc, filter_type=filter_type, show=False, large_rv=False, rv=rv, rv_planet=None, vsini_planet=vsini_planet, pca=pca, template=template, weight=weight, stellar_component=stellar_component, disable_tqdm=True)
            corr_3d[i, j, :] = corr_planet
    idx_max_corr = np.unravel_index(np.argmax(corr_3d, axis=None), corr_3d.shape)
    corr_2d = corr_3d[:, :, idx_max_corr[2]] # on se place à la vitesse radiale donnant le plus grand SNR
    plt.figure(dpi=300)
    plt.pcolormesh(T_planet, lg_planet, corr_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(corr_2d), vmax=np.nanmax(corr_2d))
    cbar = plt.colorbar() ; cbar.set_label("correlation strength", fontsize=14, labelpad=20, rotation=270)
    if model[:4] == "mol_":
        print(f"maximum correlation value of {round(np.nanmax(corr_2d), 2)} for T = {T_planet[idx_max_corr[1]]} K, {lg_planet[idx_max_corr[0]]} and rv = {round(rv[idx_max_corr[2]],1)} km/s")
        plt.ylabel("molecule", fontsize=12)
        plt.title(f'Correlation between molecular template and {target_name} \n data spectrum on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T_planet[idx_max_corr[1]], T_planet[idx_max_corr[1]]], [lg_planet[idx_max_corr[0]], lg_planet[idx_max_corr[0]]], 'kX', ms=10, label=f"max for T = {T_planet[idx_max_corr[1]]} K, \n {lg_planet[idx_max_corr[0]]} and rv = {round(rv[idx_max_corr[2]],1)} km/s")
    else:
        print(f"maximum correlation value of {round(np.nanmax(corr_2d), 2)} for T = {T_planet[idx_max_corr[1]]} K, lg = {lg_planet[idx_max_corr[0]]} and rv = {round(rv[idx_max_corr[2]],1)} km/s")
        plt.ylabel("template's gravity surface", fontsize=12)
        plt.title(f'Correlation between {model} spectra and {target_name} \n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T_planet[idx_max_corr[1]], T_planet[idx_max_corr[1]]], [lg_planet[idx_max_corr[0]], lg_planet[idx_max_corr[0]]], 'kX', ms=10, label=f"max for T = {T_planet[idx_max_corr[1]]} K, \n lg = {lg_planet[idx_max_corr[0]]} and rv = {round(rv[idx_max_corr[2]],1)} km/s")
        plt.contour(T_planet, lg_planet, corr_2d, linewidths=0.1, colors='k')
        plt.ylim(lg_planet[0], lg_planet[-1])
    plt.xlabel("template's temperature [K]", fontsize=12)
    plt.xlim(T_planet[0], T_planet[-1])
    plt.legend(fontsize=12) ; plt.show()
    T = T_planet[idx_max_corr[1]]
    if model[:4] == "mol_":
        model = model[:4] + lg_planet[idx_max_corr[0]]
        lg = 4
    else:
        lg = lg_planet[idx_max_corr[0]]
    correlation_rv(instru=instru, d_planet=d_planet, d_bkgd=None, star_flux=star_flux, wave=wave, trans=trans, T_planet=T, lg_planet=lg, model=model, R=R, Rc=Rc, filter_type=filter_type, show=True, large_rv=False, rv=np.array([rv[idx_max_corr[2]]]), rv_planet=rv[idx_max_corr[2]], vsini_planet=vsini_planet, pca=pca, template=template, weight=weight, stellar_component=stellar_component, disable_tqdm=True)
    return T_planet[idx_max_corr[1]], lg_planet[idx_max_corr[0]], rv[idx_max_corr[2]]



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
    correlation_rv(instru=instru, d_planet=d_planet, d_bkgd=None, wave=wave, trans=trans, T_planet=T_planet[idx_max_snr[1]], lg_planet=lg_planet[idx_max_snr[0]], model=model, R=R, Rc=Rc, filter_type=filter_type, show=True, large_rv=False, rv=np.array([rv_planet]), rv_planet=rv_planet, vsini_planet=vsini_planet, pca=pca, template=template, weight=weight)
    if 1 == 0:
        plt.figure() ; plt.pcolormesh(T_planet, lg_planet, noise_2d, cmap=plt.get_cmap('rainbow'))
        plt.xlabel("planet's temperature [K]", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(f'Noise value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        cbar = plt.colorbar() ; cbar.set_label("noise (in e-)", fontsize=14, labelpad=20, rotation=270) ; plt.show()
        plt.figure() ; plt.pcolormesh(T_planet, lg_planet, signal_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(signal_2d), vmax=np.nanmax(signal_2d))
        plt.xlabel("planet's temperature [K]", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(f'Signal value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        cbar = plt.colorbar() ; cbar.set_label("signal (in e-)", fontsize=14, labelpad=20, rotation=270) ; plt.show()
    







########################################################################################################################################################################################################################################################################################################################################################################################################


# CALCULATING d_planet_sim
def get_d_planet_sim(d_planet, wave, trans, model, T_planet, lg_planet, vsini_planet, rv_planet, R, Rc, filter_type, degrade_resolution, stellar_component, star_flux=None, instru=None, epsilon=0.8, fastbroad=True):
    planet = load_planet_spectrum(T_planet, lg_planet, model=model, instru=instru, interpolated_spectrum=True) # loading the planet model
    if model[:4] == "mol_" and Rc is not None: # to crop empty features regions in molecular planets
        _, planet_LF = filtered_flux(planet.flux, R=planet.R, Rc=Rc, filter_type=filter_type)
        sg = sigma_clip(planet_LF, sigma=1)
        planet.flux[~sg.mask] = np.nan 
    planet.crop(0.98*wave[0], 1.02*wave[-1])
    dl = planet.wavelength - np.roll(planet.wavelength, 1) ; dl[0] = dl[1] # delta lambda array
    Rold = np.nanmax(planet.wavelength/(2*dl))
    if Rold > 200000: # takes too much time otherwise (limiting then the resolution to 200 000 => does not seem to change anything)
        Rold = 200000
    dl = np.nanmean(planet.wavelength/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist sampling (Shannon)
    wave_inter = np.arange(0.98*wave[0], 1.02*wave[-1], dl) # regularly sampled planet
    planet = planet.interpolate_wavelength(wave_inter, renorm=False)
    if vsini_planet > 0: # broadening the spectrum
        planet = planet.broad(vsini_planet, epsilon=epsilon, fastbroad=fastbroad)
    if rv_planet != 0:
        planet = planet.doppler_shift(rv_planet) # for the auto-correlation
    if model[:4] != "mol_":
        planet.flux *= wave_inter # for the planet to be homogenous to photons or e- or ADU
    if degrade_resolution:
        planet = planet.degrade_resolution(wave, renorm=False) # degrating the planet to the instrumental resolution
    else:
        planet = planet.interpolate_wavelength(wave, renorm=False) # degrating the planet to the instrumental resolution
    planet_HF, planet_LF = filtered_flux(planet.flux,R,Rc,filter_type) # high pass filtering
    template = trans*planet_HF 
    if stellar_component:
        if star_flux is None:
            raise KeyError("star_flux is not defined for the stellar component !")
        f = interp1d(wave[~np.isnan(star_flux/trans)], (star_flux/trans)[~np.isnan(star_flux/trans)], bounds_error=False, fill_value=np.nan) 
        star_flux = f(wave) 
        if instru=="HiRISE" and 1==1: # Handling filtering edge effects due to the gaps bewteen the orders
            _, star_LF = filtered_flux(star_flux, R, Rc, filter_type) # high pass filtering
            nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
            f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
            star_flux[nan_values] = f(wave[nan_values])
        star_HF, star_LF = filtered_flux(star_flux, R, Rc, filter_type) # high pass filtering
        template += - trans * star_HF * planet_LF/star_LF
    template[np.isnan(d_planet)] = np.nan 
    template /= np.sqrt(np.nansum(template**2)) # normalizing
    d_planet_sim = np.copy(template)
    res_d_planet, psd_d_planet = calc_psd(wave, d_planet, R, smooth=0)
    res_d_planet_sim, psd_d_planet_sim = calc_psd(wave, d_planet_sim, R, smooth=0)
    ratio = np.sqrt(np.nansum(psd_d_planet[(res_d_planet>Rc)&(res_d_planet<R/10)]) / np.nansum(psd_d_planet_sim[(res_d_planet_sim>Rc)&(res_d_planet_sim<R/10)]))
    ratio = np.nansum(d_planet*template) / np.sqrt(np.nansum(d_planet_sim**2))  # assuming cos_p = 1
    d_planet_sim *= ratio
    # d_planet_sim = np.random.normal(d_planet_sim, sigma_l, len(wave))
    return d_planet_sim


########################################################################################################################################################################################################################################################################################################################################################################################################



def parameters_estimation(instru, band, target_name, wave, d_planet, star_flux, trans, R, Rc, filter_type, model, logL=False, method_logL="classic", sigma_l=None, weight=None, pca=None, precise_estimate=False, SNR_estimate=False, T_planet=None, lg_planet=None, vsini_planet=None, rv_planet=None, T_arr=None, lg_arr=None, vsini_arr=None, rv_arr=None, show=True, verbose=True, stellar_component=True, force_new_est=False, d_planet_sim=False):
    epsilon=0.8 ; fastbroad=True
    try:
        if force_new_est:
            raise ValueError("force_new_est = True")
        T_arr = fits.getdata(f"utils/parameters estimation/parameters_estimation_T_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        lg_arr = fits.getdata(f"utils/parameters estimation/parameters_estimation_lg_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        vsini_arr = fits.getdata(f"utils/parameters estimation/parameters_estimation_vsini_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        rv_arr = fits.getdata(f"utils/parameters estimation/parameters_estimation_rv_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
        if SNR_estimate:
            SNR_4D = fits.getdata(f"utils/parameters estimation/parameters_estimation_SNR_4D_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
            corr_4D = None ; logL_4D = None
        else:
            corr_4D = fits.getdata(f"utils/parameters estimation/parameters_estimation_corr_4D_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
            if logL:
                logL_4D = fits.getdata(f"utils/parameters estimation/parameters_estimation_logL_4D_{method_logL}_{instru}_{band}_R{R}_Rc{Rc}_{target_name}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
                if d_planet_sim:
                    logL_4D_sim = fits.getdata(f"utils/parameters estimation/parameters_estimation_logL_4D_sim_{method_logL}_{instru}_{band}_R{R}_Rc{Rc}_{target_name}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
            else:
                logL_4D = None
            SNR_4D = None
        if verbose:
            print("Loading existing parameters calculations...")
    except Exception as e:
        if verbose:
            print(f"New parameters estimation: {e}")
        DT = 100 ; Dlg = 1 ; Dvsini = 20 ; Drv = 20
        if T_arr is None and lg_arr is None and vsini_arr is None and rv_arr is None:
            if SNR_estimate:
                T_arr, lg_arr = get_model_grid(model)
                T_arr = T_arr[T_arr>=500]
                lg_arr = lg_arr[lg_arr>=3]
                if precise_estimate:
                    T_arr = np.linspace(max(500, T_planet-DT/2), min(3000, T_planet+DT/2), int(DT/10)).astype(np.float32)
                    lg_arr = np.linspace(max(3, lg_planet-Dlg/2), min(5, lg_planet+Dlg/2), int(Dlg/0.1)).astype(np.float32)
                    rv_arr = np.append(np.linspace(-1000, rv_planet-3*Drv/4, 100), np.append(np.linspace(rv_planet-Drv/2, rv_planet+Drv/2, int(Drv/0.5)), np.linspace(rv_planet+3*Drv/4, 1000, 100)))
                else:
                    rv_arr = np.append(np.linspace(-1000, rv_planet-3*Dv/4, 50), np.append(np.linspace(rv_planet-Drv/2, rv_planet+Drv/2, int(Drv/1)), np.linspace(rv_planet+3*Drv/4, 1000, 50)))
                vsini_arr = np.append(np.linspace(1, vsini_planet-3*Dvsini/4, int(vsini_planet/5)), np.append(np.linspace(vsini_planet-Dvsini/2, vsini_planet+Dvsini/2, int(Dvsini/1)), np.linspace(vsini_planet+3*Dvsini/4, 80, int((80-vsini_planet-3*Dvsini/4)/5))))
            else:
                if precise_estimate:
                    T_arr = np.linspace(max(500, T_planet-DT/2), min(3000, T_planet+DT/2), int(DT/5)).astype(np.float32) # dT = 5 K
                    lg_arr = np.linspace(max(3, lg_planet-Dlg/2), min(5, lg_planet+Dlg/2), int(Dlg/0.05)).astype(np.float32) # dlg = 2
                    vsini_arr = np.linspace(max(0, vsini_planet-Dvsini/2), min(80, vsini_planet + Dvsini/2), int(Dvsini/0.5)).astype(np.float32) # dvsini = 0.5 km/s
                    rv_arr = np.linspace(rv_planet-Drv/2, rv_planet+Drv/2, int(Drv/0.5)).astype(np.float32) # drv = 0.5 km/s
                else:
                    T_arr, lg_arr = get_model_grid(model)
                    T_arr = T_arr[T_arr>=500]
                    lg_arr = lg_arr[lg_arr>=3]
                    T_arr = np.interp(np.arange(0, len(T_arr)-0.5, 0.5), np.arange(0, len(T_arr), 1), T_arr)
                    lg_arr = np.arange(3, 5.1, 0.1)
                    if R > 10000:
                        vsini_arr = np.linspace(max(0, vsini_planet-Dvsini/2), min(80, vsini_planet + Dvsini/2), int(Dvsini/1)).astype(np.float32) # dvsini = 0.5 km/s
                    else:
                        vsini_arr = np.array([0])
                    rv_arr = np.linspace(rv_planet-Drv/2, rv_planet+Drv/2, int(Drv/1)).astype(np.float32) # drv = 0.5 km/s
        if stellar_component:
            if star_flux is None:
                raise KeyError("star_flux is not defined for the stellar component !")
            f = interp1d(wave[~np.isnan(star_flux/trans)], (star_flux/trans)[~np.isnan(star_flux/trans)], bounds_error=False, fill_value=np.nan) 
            star_flux = f(wave) 
            if instru=="HiRISE" and 1==1: # Handling filtering edge effects due to the gaps bewteen the orders
                _, star_LF = filtered_flux(star_flux, R, Rc, filter_type) # high pass filtering
                nan_values = keep_true_chunks(np.isnan(d_planet), N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
                f = interp1d(wave[~nan_values], star_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
                star_flux[nan_values] = f(wave[nan_values])
            star_HF, star_LF = filtered_flux(star_flux, R, Rc, filter_type) # high pass filtering
        else:
            star_HF = None ; star_LF = None
        if d_planet_sim: # CALCULATING d_planet_sim
            d_planet_sim = get_d_planet_sim(d_planet=d_planet, wave=wave, trans=trans, model=model, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, degrade_resolution=degrade_resolution, stellar_component=stellar_component, star_flux=star_flux, instru=instru, epsilon=epsilon, fastbroad=fastbroad)
        else:
            d_planet_sim = None
        corr_4D = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        SNR_4D = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        logL_4D = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        logL_4D_sim = np.zeros((len(T_arr), len(lg_arr), len(vsini_arr), len(rv_arr)), dtype=np.float32)
        from numba import njit, prange
        from multiprocessing import Pool, cpu_count
        with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            results = list(tqdm(pool.imap(process_parameters_estimation, [(i, j, T_arr, lg_arr, vsini_arr, rv_arr, d_planet, weight, pca, model, instru, wave, trans, epsilon, fastbroad, R, Rc, filter_type, sigma_l, logL, method_logL, star_HF, star_LF, SNR_estimate, rv_planet, stellar_component, d_planet_sim) for i in range(len(T_arr)) for j in range(len(lg_arr))]), total=len(T_arr)*len(lg_arr), disable=not verbose))
            for (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim) in results: # Remplissage des matrices 5D avec les résultats
                corr_4D[i, j, :, :] = corr_2D
                SNR_4D[i, j, :, :] = SNR_2D
                logL_4D[i, j, :, :] = logL_2D
                logL_4D_sim[i, j, :, :] = logL_2D_sim
        fits.writeto(f"utils/parameters estimation/parameters_estimation_T_arr_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", T_arr, overwrite=True)
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
    if show:
        if SNR_estimate:
            idx_max_SNR = np.unravel_index(np.argmax(SNR_4D, axis=None), SNR_4D.shape)
            SNR_2D = SNR_4D[:, :, idx_max_SNR[2], idx_max_SNR[3]].transpose()
            T_SNR_found = T_arr[idx_max_SNR[0]]
            lg_SNR_found = lg_arr[idx_max_SNR[1]]
            vsini_SNR_found = vsini_arr[idx_max_SNR[2]]
            rv_SNR_found = rv_arr[idx_max_SNR[3]]
            print(f"maximum S/N for T = {round(T_SNR_found)} K, lg = {lg_SNR_found:.2f} and rv = {rv_SNR_found:.1f} km/s")
            plt.figure(dpi=300) ; plt.ylabel("surface gravity (in dex)" , fontsize=12) ; plt.xlabel("temperature [K]",fontsize=12) ; plt.title(f'S/N with {model} spectra for {target_name}\n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}',fontsize=14)
            plt.pcolormesh(T_arr, lg_arr, SNR_2D, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(SNR_2D), vmax=np.nanmax(SNR_2D))
            cbar = plt.colorbar() ; cbar.set_label("S/N", fontsize=12, labelpad=20, rotation=270)
            plt.plot([T_SNR_found, T_SNR_found], [lg_SNR_found, lg_SNR_found], 'kX', ms=10, label=r"$S/N_{max}$ "+f"for T = {round(T_SNR_found)}K, lg = {lg_SNR_found:.2f},\n Vsin(i) = {vsini_SNR_found:.1f}km/s and RV = {rv_SNR_found:.1f}km/s")
            plt.contour(T_arr, lg_arr, SNR_2D, linewidths=0.1,colors='k')
            plt.ylim(lg_arr[0], lg_arr[-1]) ; plt.xlim(T_arr[0], T_arr[-1]) ; plt.legend(fontsize=10) ; plt.show()
        else:
            idx_max_corr = np.unravel_index(np.argmax(corr_4D, axis=None), corr_4D.shape)
            corr_2D = corr_4D[:, :, idx_max_corr[2], idx_max_corr[3]].transpose()
            T_corr_found = T_arr[idx_max_corr[0]]
            lg_corr_found = lg_arr[idx_max_corr[1]]
            vsini_corr_found = vsini_arr[idx_max_corr[2]]
            rv_corr_found = rv_arr[idx_max_corr[3]]
            print(f"maximum correlation for T = {round(T_corr_found)} K, lg = {lg_corr_found:.2f} and rv = {rv_corr_found:.1f} km/s")
            plt.figure(dpi=300) ; plt.ylabel("surface gravity (in dex)" , fontsize=12) ; plt.xlabel("temperature [K]",fontsize=12) ; plt.title(f'Correlation between {model} spectra and {target_name}\n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}',fontsize=14)
            plt.pcolormesh(T_arr, lg_arr, corr_2D, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(corr_2D), vmax=np.nanmax(corr_2D))
            cbar = plt.colorbar() ; cbar.set_label("correlation strength", fontsize=12, labelpad=20, rotation=270)
            plt.plot([T_corr_found, T_corr_found], [lg_corr_found, lg_corr_found], 'kX', ms=10, label=f"max for T = {round(T_corr_found)}K, lg = {lg_corr_found:.2f},\n Vsin(i) = {vsini_corr_found:.1f}km/s and RV = {rv_corr_found:.1f}km/s")
            plt.contour(T_arr, lg_arr, corr_2D, linewidths=0.1,colors='k')
            plt.ylim(lg_arr[0], lg_arr[-1]) ; plt.xlim(T_arr[0], T_arr[-1]) ; plt.legend(fontsize=10) ; plt.show()
            if logL:
                idx_max_logL = np.unravel_index(np.argmax(logL_4D, axis=None), logL_4D.shape)
                logL_2D = logL_4D[:, :, idx_max_logL[2], idx_max_logL[3]].transpose()
                T_logL_found = T_arr[idx_max_logL[0]]
                lg_logL_found = lg_arr[idx_max_logL[1]]
                vsini_logL_found = vsini_arr[idx_max_logL[2]]
                rv_logL_found = rv_arr[idx_max_logL[3]]
                print(f"maximum logL for T = {round(T_logL_found)} K, lg = {lg_logL_found:.2f} and rv = {rv_logL_found:.1f} km/s")
                plt.figure(dpi=300) ; plt.ylabel("surface gravity (in dex)" , fontsize=12) ; plt.xlabel("temperature [K]",fontsize=12) ; plt.title(f'logL ({method_logL}) with {model} spectra for {target_name}\n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}',fontsize=14)
                plt.pcolormesh(T_arr, lg_arr, logL_2D, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(logL_2D), vmax=np.nanmax(logL_2D))
                cbar = plt.colorbar() ; cbar.set_label("logL", fontsize=12, labelpad=20, rotation=270)
                plt.plot([T_logL_found, T_logL_found], [lg_logL_found, lg_logL_found], 'kX', ms=10, label=r"$logL_{max}$ "+f" for T = {round(T_logL_found)}K, lg = {lg_logL_found:.2f},\n Vsin(i) = {vsini_logL_found:.1f}km/s and RV = {rv_logL_found:.1f}km/s")
                plt.contour(T_arr, lg_arr, logL_2D, linewidths=0.1,colors='k')
                plt.ylim(lg_arr[0], lg_arr[-1]) ; plt.xlim(T_arr[0], T_arr[-1]) ; plt.legend(fontsize=10) ; plt.show()
                # Corner plot :
                try:
                    if force_new_est:
                        raise ValueError("force_new_est = True")
                    samples = fits.getdata(f"utils/parameters estimation/parameters_estimation_samples_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
                    if d_planet_sim is not None:
                        samples_sim = fits.getdata(f"utils/parameters estimation/parameters_estimation_samples_sim_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits")
                except:
                    samples = parameters_estimation_MCMC(logL_4D, T_arr, lg_arr, vsini_arr, rv_arr, T_planet=T_logL_found, lg_planet=lg_logL_found, vsini_planet=vsini_logL_found, rv_planet=rv_logL_found)
                    fits.writeto(f"utils/parameters estimation/parameters_estimation_samples_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", samples, overwrite=True)
                    if d_planet_sim is not None:
                        samples_sim = parameters_estimation_MCMC(logL_4D_sim, T_arr, lg_arr, vsini_arr, rv_arr, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet)
                        fits.writeto(f"utils/parameters estimation/parameters_estimation_samples_sim_{instru}_{band}_{target_name}_R{R}_Rc{Rc}_{model}_precise_estimate_{precise_estimate}_stellar_component_{stellar_component}.fits", samples_sim, overwrite=True)
                parameters_estimation_corner_plot(instru, band, target_name, R, Rc, samples, logL_4D, T_arr, lg_arr, vsini_arr, rv_arr, T_planet=T_logL_found, lg_planet=lg_logL_found, vsini_planet=vsini_logL_found, rv_planet=rv_logL_found, d_planet_sim=False)
                parameters_estimation_corner_plot(instru, band, target_name, R, Rc, samples_sim, logL_4D_sim, T_arr, lg_arr, vsini_arr, rv_arr, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, d_planet_sim=True)
    return T_arr, lg_arr, vsini_arr, rv_arr, corr_4D, SNR_4D, logL_4D

def process_parameters_estimation(args):
    i, j, T_arr, lg_arr, vsini_arr, rv_arr, d_planet, weight, pca, model, instru, wave, trans, epsilon, fastbroad, R, Rc, filter_type, sigma_l, logL, method_logL, star_HF, star_LF, SNR_estimate, rv_planet, stellar_component, d_planet_sim = args
    corr_2D = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    SNR_2D = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    auto_2D = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    logL_2D = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    logL_2D_sim = np.zeros((len(vsini_arr), len(rv_arr)), dtype=np.float32)
    template = load_planet_spectrum(T_arr[i], lg_arr[j], model=model, instru=instru, interpolated_spectrum=True)
    template = template.degrade_resolution(wave, renorm=False)
    template.flux *= wave
    template_HF = template.copy() ; template_LF = template.copy()
    template_HF.flux, template_LF.flux = filtered_flux(template.flux, R, Rc, filter_type)
    for k in range(len(vsini_arr)):
        if vsini_arr[k] > 0:
            template_HF_broad = template_HF.broad(vsini_arr[k], epsilon=epsilon, fastbroad=fastbroad)
            template_LF_broad = template_LF.broad(vsini_arr[k], epsilon=epsilon, fastbroad=fastbroad)
        else:
            template_HF_broad = template_HF.copy()
            template_LF_broad = template_LF.copy()
        if SNR_estimate: # needed to calculate sigma_auto_correlation (and subtract it later)
            template_auto = trans*template_HF_broad.flux
            if stellar_component:
                template_auto += - trans * star_HF * template_LF_broad.flux/star_LF # better without the residual stellar contributions for the auto-correlation
            template_auto[np.isnan(d_planet)] = np.nan 
            if weight is not None:
                template_auto *= weight
            template_auto /= np.sqrt(np.nansum(template_auto**2)) # normalizing
        for l in range(len(rv_arr)):
            template_HF_broad_shift = template_HF_broad.doppler_shift(rv_arr[l]).flux
            template_LF_broad_shift = template_LF_broad.doppler_shift(rv_arr[l]).flux
            t = trans*template_HF_broad_shift
            if stellar_component:
                t += - trans * star_HF * template_LF_broad_shift/star_LF
            if pca is not None: # subtracting PCA modes (if any)
                t0 = np.copy(t)
                n_comp_sub = pca.n_components
                for nk in range(n_comp_sub): 
                    t -= np.nan_to_num(np.nansum(t0*pca.components_[nk])*pca.components_[nk])
            d_p = np.copy(d_planet)
            if weight is not None:
                d_p *= weight ; t *= weight
            d_p[d_p == 0] = np.nan
            t[t == 0] = np.nan
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
            SNR_2D[k, :] -= np.nanmean(SNR_2D[k][(rv_arr>rv_planet+200)|(rv_arr<rv_planet-200)])
            auto_2D[k, :] -= np.nanmean(auto_2D[k][(rv_arr>200)|(rv_arr<-200)])
            sigma2_tot = np.nanvar(SNR_2D[k][(rv_arr>rv_planet+200)|(rv_arr<rv_planet-200)]) # Variance
            sigma2_auto = np.nanvar(auto_2D[k][(rv_arr>200)|(rv_arr<-200)]*np.nanmax(SNR_2D[k][(rv_arr<rv_planet+25)&(rv_arr>rv_planet-25)])/np.nanmax(auto_2D[k]))
            if sigma2_auto < sigma2_tot:
                sigma_CCF = np.sqrt(sigma2_tot - sigma2_auto) # NOISE ESTIMATION = sqrt(var(signal) - var(auto correlation))
            else:
                sigma_CCF = np.sqrt(sigma2_tot)
            SNR_2D[k, :] /= sigma_CCF
    return (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim)







def logL_function(theta, logL_interpolator, T_arr, lg_arr, vsini_arr, rv_arr):
    T, lg, vsini, rv = theta
    if (T < T_arr.min() or T > T_arr.max() or
        lg < lg_arr.min() or lg > lg_arr.max() or
        vsini < vsini_arr.min() or vsini > vsini_arr.max() or
        rv < rv_arr.min() or rv > rv_arr.max()):
        return -np.inf  # Retourne -inf si les paramètres sont hors des limites
    return logL_interpolator([T, lg, vsini, rv])

def log_probability(theta, logL_interpolator, T_arr, lg_arr, vsini_arr, rv_arr):
    logL_val = logL_function(theta, logL_interpolator, T_arr, lg_arr, vsini_arr, rv_arr)
    if np.isnan(logL_val):  # En cas de problème, renvoyer -inf
        return -np.inf
    return logL_val

def parameters_estimation_MCMC(logL_4D, T_arr, lg_arr, vsini_arr, rv_arr, T_planet, lg_planet, vsini_planet, rv_planet):
    import emcee
    from scipy.interpolate import RegularGridInterpolator
    ndim = 4  # T, lg, vsini, rv
    logL_interpolator = RegularGridInterpolator((T_arr, lg_arr, vsini_arr, rv_arr), logL_4D)

    # MCMC parameters
    nwalkers = 100  # Nombre de walkers
    initial_position = [T_planet, lg_planet, vsini_planet, rv_planet]  # Position initiale des paramètres
    initial_spread = [100, 0.1, 1, 1]  # Écart-type pour les walkers
    p0 = [initial_position + initial_spread * np.random.randn(ndim) for i in range(nwalkers)]
    nsteps = 5000  # Nombre d'itérations du MCMC
    burn_in = 1000  # Nombre d'étapes pour le burn-in

    # Exécution du MCMC avec multiprocessing
    print("Running MCMC...")
    with Pool(processes=cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(logL_interpolator, T_arr, lg_arr, vsini_arr, rv_arr), pool=pool
        )
        with tqdm(total=nsteps) as pbar:
            for sample in sampler.sample(p0, iterations=nsteps):
                pbar.update()

    # Extraire les échantillons après burn-in
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    return samples



def parameters_estimation_corner_plot(instru, band, target_name, R, Rc, samples, logL_4D, T_arr, lg_arr, vsini_arr, rv_arr, T_planet, lg_planet, vsini_planet, rv_planet, d_planet_sim):
    import corner
    from scipy.integrate import quad
    import matplotlib.lines as mlines
    ndim = 4  # T, lg, vsini, rv
    smooth_corner = 0
    alpha_value = 0.5
    if d_planet_sim:
        color_value = "r"
    else:
        color_value = "g"
    color_L = "b"
    target_name = target_name.replace("_", " ")
    T_found, lg_found, vsini_found, rv_found = samples.T

    # Plot des résultats avec corner
    data = np.zeros((samples.shape[0], ndim))
    data[:,0] = T_found
    data[:,1] = lg_found
    data[:,2] = vsini_found
    data[:,3] = rv_found
    data_bins = np.int64(np.zeros((ndim)))
    data_bins[0] = len(T_arr[(T_arr>np.nanmin(T_found))&(T_arr<np.nanmax(T_found))])
    data_bins[1] = len(lg_arr[(lg_arr>np.nanmin(lg_found))&(lg_arr<np.nanmax(lg_found))])
    data_bins[2] = len(vsini_arr[(vsini_arr>np.nanmin(vsini_found))&(vsini_arr<np.nanmax(vsini_found))])
    data_bins[3] = len(rv_arr[(rv_arr>np.nanmin(rv_found))&(rv_arr<np.nanmax(rv_found))])
    injected_values = np.array([T_planet, lg_planet, vsini_planet, rv_planet])
    # https://corner.readthedocs.io/en/latest/api/
    figure = corner.corner(
        data,
        bins = data_bins,
        labels = [r"$T \, [\mathrm{K}]$", r"$lg \, [\mathrm{dex}]$", r"$Vsin(i) \, [\mathrm{km/s}]$", r"$RV \, [\mathrm{km/s}]$"],
        quantiles = [0.16, 0.5, 0.84], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = True,
        title_kwargs = {"fontsize": 12},
        top_ticks=False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        smooth = smooth_corner,
        smooth1d = smooth_corner)
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(injected_values[i], color=color_value, alpha=alpha_value)
        p = np.exp(logL_4D - np.max(logL_4D))  # Pour éviter les overflow numériques
        if i==0:
            p = np.nansum(p, axis=(1,2,3))
            f = interp1d(T_arr, p, bounds_error=False, fill_value=np.nan)
        elif i==1:
            p = np.nansum(p, axis=(0,2,3))
            f = interp1d(lg_arr, p, bounds_error=False, fill_value=np.nan)
        elif i==2:
            p = np.nansum(p, axis=(0,1,3))
            f = interp1d(vsini_arr, p, bounds_error=False, fill_value=np.nan)
        elif i==3:
            p = np.nansum(p, axis=(0,1,2))
            f = interp1d(rv_arr, p, bounds_error=False, fill_value=np.nan)
        data_param = data[:, i]
        x = np.linspace(data_param.min(), data_param.max(), 1000)
        pdf = f(x)
        integral, _ = quad(f, np.nanmin(x[~np.isnan(pdf)]), np.nanmax(x[~np.isnan(pdf)]))
        pdf /= integral
        ax.plot(x, pdf * len(data_param) * np.diff(ax.get_xlim())[0] / data_bins[i], color=color_L, lw=2, label="logL", alpha=0.5)
        if i==0:
            ax.legend()
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(injected_values[xi], color=color_value, alpha=alpha_value)
            ax.axhline(injected_values[yi], color=color_value, alpha=alpha_value)
            ax.plot(injected_values[xi], injected_values[yi], "s"+color_value)
    figure.suptitle(f"Parameters estimation of {target_name} on {band}-band of {instru}\n (with R = {round(R)} and $R_c$ = {Rc})", fontsize=16, y=1.05)
    if d_planet_sim:
        plt.legend(handles=[mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"T = {round(T_planet)} K"), mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"lg = {lg_planet:.2f}"), mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"Vsini(i) = {vsini_planet:.1f} km/s"), mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"RV = {rv_planet:.1f} km/s"), ], frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=12, title="Injected parameters:", title_fontsize=16)    
    else:
        plt.legend(handles=[mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"T = {round(T_planet)} K"), mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"lg = {lg_planet:.2f}"), mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"Vsini(i) = {vsini_planet:.1f} km/s"), mlines.Line2D([], [], color="w", label=""), mlines.Line2D([], [], linestyle="-", marker="s", color=color_value, label=f"RV = {rv_planet:.1f} km/s"), ], frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=12, title=r"$logL_{max}$ parameters:", title_fontsize=16)    
    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()







    
    
    
    
    
    
    
    