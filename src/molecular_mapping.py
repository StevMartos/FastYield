from src.spectrum import *



def stellar_high_filtering( cube, renorm_cube_res, R, Rc, used_filter, cosmic=False, sigma_cosmic=5, print_value=True, only_high_pass=False):
    """
    Post-processing filtering method according to molecular mapping, see Appendix B of Martos et al. (2024)

    Parameters
    ----------
     cube: 3d-array
        Data cube.
    renorm_cube_res: bool
        To renorm every spaxel of the final cube product by their norm, if True the CCF calculated will directly gives the correlation intensity (cos_theta_est)
    R: float
        Spectral resolution of the cube.
    Rc: float
        Cut-off resolution of the filter. If Rc = None, no filter will be applyied
    used_filter: str
        type of filter used.
    cosmic: bool, optional
        TO filter outliers. The default is False.
    sigma_cosmic: float, optional
        sigma value of the outliers filtering method (sigma clipping). The default is 5.
    only_high_pass: bool, optional
        In order to apply only a high-pass filter on the cube (whether than also subtracting the stellar component). The default is False.

    Returns
    -------
    cube_res: 3d-array
        S_res.
    cube_M: 3d-array
        Estimated stellar modulation function, not really usefull.
    """
    cube = np.copy( cube)
    NbChannel, NbLine, NbColumn = cube.shape
    stell = np.nansum(cube, (1, 2)) # estimated stellar spectrum
    Y = np.reshape(cube, (NbChannel, NbLine*NbColumn))
    cube_M = np.copy(Y) ; m = 0
    for k in range(Y.shape[1]):
        if not all(np.isnan(Y[:, k])):
            if only_high_pass:
                _, Y_BF = filtered_flux(Y[:, k], R, Rc, used_filter)
                M = Y_BF/stell
            else:
                _, M = filtered_flux(Y[:, k]/stell, R, Rc, used_filter)
            cube_M[:, k] = Y[:, k]/stell #  True modulations (with noise), assuming that stell is the real observed stellar spectrum
            if cosmic:
                sg = sigma_clip(Y[:, k]-stell*M, sigma=sigma_cosmic)
                Y[:, k] = np.array(np.ma.masked_array(sg, mask=sg.mask).filled(np.nan))
            else:
                Y[:, k] = Y[:, k] - stell*M
            m += np.nansum(M) 
    if print_value:
        print("\n norme de M =", round(m/(NbChannel), 3)) # must be ~ 1
    cube_res =  Y.reshape((NbChannel, NbLine, NbColumn))
    cube_M = cube_M.reshape((NbChannel, NbLine, NbColumn))
    cube_res[cube_res == 0] = np.nan ; cube_M[cube_M == 0] = np.nan
    if renorm_cube_res: # renormalizing the spectra of every spaxel in order to directly have cos_theta_est
        for i in range(NbLine):
            for j in range(NbColumn): # for every spaxel
                if not all(np.isnan(cube_res[:, i, j])): # ignoring nan values
                    cube_res[:, i, j] = cube_res[:, i, j]/np.sqrt(np.nansum(cube_res[:, i, j]**2))
    return cube_res, cube_M



def molecular_mapping_rv(instru, S_res, T, lg, model, wave, trans, R, Rc, used_filter, rv=None, vsini=0, print_value=True, planet_spectrum=None, pca=None):
    """
    Cross-correlating the residual cube S_res with templates, giving the CCF

    Parameters
    ----------
    instru : str
        Instrument's name.
    S_res : 3d-array
        Residual cube.
    T : float
        PLanet's temperature.
    lg : float
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
    used_filter: str
        type of filter used.
    rv : float, optional
        Planet's radial velocity. The default is None.
    vsini : float, optional
        Planet's rotationnal speed. The default is 0.
    pca : pca, optional
        If PCA is applied, it is necessary to also subtract the components to the template. The default is None.
    """
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T, lg, model=model, instru=instru)
        if model[:4] == "mol_" and Rc is not None:
            _, planet_spectrum_BF = filtered_flux(planet_spectrum.flux, R=planet_spectrum.R, Rc=Rc, used_filter=used_filter)
            sg = sigma_clip(planet_spectrum_BF, sigma=1)
            planet_spectrum.flux[~sg.mask] = np.nan
        planet_spectrum.crop(0.98*wave[0], 1.02*wave[-1])
        if vsini > 0:
            planet_spectrum = planet_spectrum.broad(vsini)
        if model[:4] != "mol_":
            planet_spectrum.flux *= planet_spectrum.wavelength # homogenous to a number of photon
    planet_spectrum = planet_spectrum.degrade_resolution(wave, renorm=False)
    NbChannel, NbLine, NbColumn = S_res.shape
    if rv is None:
        rv = np.linspace(-50, 50, 101)
    else:
        rv = np.array([rv])
    CCF = np.zeros((len(rv), NbLine, NbColumn))
    for k in range(len(rv)):
        if print_value:
            print(" CCF for: rv = ", rv[k], " km/s & Tp =", T, "K & lg = ", lg)
        if rv[k]!=0:
            planet_shift = planet_spectrum.doppler_shift(rv[k]) # en ph/mn
            template, _ = filtered_flux(planet_shift.flux, R, Rc, used_filter)
        else: 
            template, _ = filtered_flux(planet_spectrum.flux, R, Rc, used_filter)
        template *= trans 
        if pca is not None: # subtraction of the PCA's modes to the template
            template0 = np.copy(template)
            n_comp_sub = pca.n_components
            for nk in range(n_comp_sub): 
                template -= np.nan_to_num(np.nansum(template0*pca.components_[nk])*pca.components_[nk])
        for i in range(NbLine):
            for j in range(NbColumn): # for every spaxel
                if not all(np.isnan(S_res[:, i, j])): # ignoring nan values
                    d = np.copy(S_res[:, i, j])
                    t = np.copy(template)
                    d[d == 0] = np.nan
                    t[np.isnan(d)] = np.nan
                    t[t == 0] = np.nan
                    d[np.isnan(t)] = np.nan
                    t /= np.sqrt(np.nansum(t**2))  # normalizing the template
                    CCF[k, i, j] = np.nansum(d*t) # cross-correlation between the residual signal and the template
    CCF[CCF == 0] = np.nan
    if len(rv) == 1:
        return CCF[0], template
    else:
        return CCF, rv, template



def correlation_rv(instru, d, wave, trans, T, lg, model, R, Rc, used_filter, show=True, large_rv=False, vsini=0, rv=None, pca=None, correlation=True, noise_estimation=False):
    planet_spectrum = load_planet_spectrum(T, lg, model=model, instru=instru)
    if model[:4] == "mol_" and Rc is not None:
        _, planet_spectrum_BF = filtered_flux(planet_spectrum.flux, R=planet_spectrum.R, Rc=Rc, used_filter=used_filter)
        sg = sigma_clip(planet_spectrum_BF, sigma=1)
        planet_spectrum.flux[~sg.mask] = np.nan
    planet_spectrum.crop(0.98*wave[0], 1.02*wave[-1])
    dl = planet_spectrum.wavelength - np.roll(planet_spectrum.wavelength, 1) ; dl[0] = dl[1] # array de delta Lambda
    Rold = np.nanmax(planet_spectrum.wavelength/(2*dl))
    if Rold > 200000: 
        Rold = 200000
    dl = np.nanmean(planet_spectrum.wavelength/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist samplé (Shannon)
    wave_inter = np.arange(0.98*wave[0], 1.02*wave[-1], dl)
    planet_pectrum = planet_spectrum.interpolate_wavelength(planet_spectrum.flux, planet_spectrum.wavelength, wave_inter, renorm=False)
    if vsini > 0:
        planet_spectrum = planet_spectrum.broad(vsini)
    planet_spectrum = planet_spectrum.degrade_resolution(wave, renorm=False)
    if model[:4] != "mol_":
        planet_spectrum.flux *= wave
    if rv is None:
        rv = np.linspace(-50, 50, 201)
        if large_rv:
            rv = np.linspace(-1000, 1000, 401)
    cos_theta = np.zeros_like(rv)
    for i in range(len(rv)):
        d = np.copy(d)
        planet_shift = planet_spectrum.doppler_shift(rv[i]) # en ph/mn
        t, _ = filtered_flux(planet_shift.flux, R, Rc, used_filter)
        t *= trans
        if pca is not None: # soustraction des modes de PCA au template
            t0 = np.copy(t)
            n_comp_sub = pca.n_components
            for nk in range(n_comp_sub): 
                t -= np.nan_to_num(np.nansum(t0*pca.components_[nk])*pca.components_[nk])
        d[d == 0] = np.nan
        t[np.isnan(d)] = np.nan
        t[t == 0] = np.nan
        d[np.isnan(t)] = np.nan
        cos_theta[i] = np.nansum(d*t) / (np.sqrt(np.nansum(d**2)*np.nansum(t**2)))
    cos_theta[np.isnan(cos_theta)] = 0.
    if show:     
        plt.figure(dpi=300) ; plt.plot(wave, d/np.sqrt(np.nansum(d**2)), 'r', label="data")  ; plt.plot(wave, t/np.sqrt(np.nansum(t**2)), 'b', label="template") ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("high-pass flux (normalized)") ; plt.legend() ; plt.title(f"High-pass filtered data and {model} template flux, \n with $T_p$={T}K, R={int(R)} and $R_c$={Rc}") ; plt.show()
    return cos_theta, rv



def plot_CCF(instru, d, bs_HF, wave, trans, T, lg, model, R, Rc, used_filter, target_name, band, rv0, vsini=0, large_rv=True, pca=None): 
    plt.figure(dpi=300) ; plt.xlabel("radial velocity (in km/s)", fontsize=14) ; plt.ylabel(r"cos $\theta_{est}$", fontsize=14) ; plt.grid(True)
    cos_signal, rv = correlation_rv(instru, d, wave, trans, T=T, lg=lg, model=model, R=R, Rc=Rc, used_filter=used_filter, show=False, large_rv=large_rv, vsini=vsini, pca=pca)
    plt.plot(rv, cos_signal, 'k', label=f"on-planet")
    cos_noise, rv = correlation_rv(instru, bs_HF, wave, trans, T=T, lg=lg, model=model, R=R, Rc=Rc, used_filter=used_filter, show=False, large_rv=large_rv, vsini=vsini, pca=pca)
    plt.plot(rv, cos_noise, "k:", label=f"off-planet")   
    
    # planet_spectrum = load_planet_spectrum(T, lg, model=model, instru=instru)
    # if model[:4] == "mol_" and Rc is not None:
    #     _, planet_spectrum_BF = filtered_flux(planet_spectrum.flux, R=planet_spectrum.R, Rc=Rc, used_filter=used_filter)
    #     sg = sigma_clip(planet_spectrum_BF, sigma=1)
    #     planet_spectrum.flux[~sg.mask] = np.nan
    # planet_spectrum.crop(0.98*wave[0], 1.02*wave[-1])
    # planet_spectrum = planet_spectrum.degrade_resolution(wave, renorm=False)
    # if model[:4] != "mol_":
    #     planet_spectrum.flux *= wave
    # planet_spectrum = planet_spectrum.doppler_shift(rv0) # en ph/mn
    # Sp_HF, _ = filtered_flux(planet_spectrum.flux, R, Rc, used_filter)
    # Sp_HF *= trans 
    # Sp_HF[np.isnan(d)] = np.nan ; Sp_HF[Sp_HF == 0] = np.nan
    # cos_auto, rv = correlation_rv(instru, Sp_HF, wave, trans, T=T, lg=lg, model=model, R=R, Rc=Rc, used_filter=used_filter, show=False, large_rv=large_rv, vsini=0, pca=pca)
    # plt.plot(rv, cos_auto*np.nanmax(cos_signal), "k--", label=f"auto-correlation (normalized)")
    
    plt.legend(fontsize=12)
    plt.title(f'Correlation between {model} spectrum and {target_name} data spectrum \n on {band}-band of {instru} with $T_p$ = {T}K and $R_c$ = {Rc}', fontsize=16)
    plt.show()


def correlation_PSF(cube_M, CCF):
    PSF = np.nanmean(cube_M, axis=0)
    PSF = PSF/np.sqrt(np.nansum(PSF**2))
    idx_PSF_centroid = np.unravel_index(np.nanargmax(PSF, axis=None), PSF.shape)    
    i_PSF_centroid = idx_PSF_centroid[0] ; j_PSF_centroid = idx_PSF_centroid[1]
    PSF_shift = np.copy(PSF)*0
    CCF_conv = np.copy(CCF)*np.nan
    for i_shift in range(CCF_conv.shape[0]):
        for j_shift in range(CCF_conv.shape[1]): 
            if not np.isnan(CCF[i_shift, j_shift]):
                for i in range(PSF_shift.shape[0]):
                    for j in range(PSF_shift.shape[1]):
                        if i+i_PSF_centroid-i_shift>=0 and j+j_PSF_centroid-j_shift>=0 and i+i_PSF_centroid-i_shift<PSF_shift.shape[0] and j+j_PSF_centroid-j_shift<PSF_shift.shape[1]:
                            PSF_shift[i, j] = PSF[i+i_PSF_centroid-i_shift, j+j_PSF_centroid-j_shift]
                        else:
                            PSF_shift[i, j] = np.nan
                #plt.figure() ; plt.imshow(PSF_shift, extent=[-(cube.shape[2]+1)//2*pxscale, (cube.shape[2]+1)//2*pxscale, -(cube.shape[2]+1)//2*pxscale, (cube.shape[2]+1)//2*pxscale]) ; plt.title(f'MIRIMRS PSF of {target_name} \n on {band}', fontsize=14) ; plt.ylabel('y offset (in ")', fontsize=14) ; plt.xlabel('x offset (in ")', fontsize=14) ; plt.show()
                #plt.figure() ; plt.imshow(CCF, extent=[-(cube.shape[2]+1)//2*pxscale, (cube.shape[2]+1)//2*pxscale, -(cube.shape[2]+1)//2*pxscale, (cube.shape[2]+1)//2*pxscale]) ; plt.title(f'MIRIMRS PSF of {target_name} \n on {band}', fontsize=14) ; plt.ylabel('y offset (in ")', fontsize=14) ; plt.xlabel('x offset (in ")', fontsize=14) ; plt.show()
                CCF_conv[i_shift, j_shift] = np.nansum(PSF_shift*CCF)
    return CCF_conv



def SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, print_value=True, snr_calc=True):
        planet_sep = int(round(np.sqrt((y0-CCF.shape[0]//2)**2+(x0-CCF.shape[1]//2)**2)))
        if planet_sep == 0: 
            CCF = CCF*annular_mask(0, CCF.shape[0]//2, value=np.nan, size=CCF.shape)
            CCF_noise = np.copy(CCF_wo_planet)*annular_mask(3*size_core, 4*size_core, value=np.nan, size=CCF.shape)
        else:
            CCF = CCF*annular_mask(max(planet_sep-3*size_core-1, 0), planet_sep+3*size_core, value=np.nan, size=CCF.shape)
            if snr_calc:
                CCF_noise = np.copy(CCF_wo_planet)*annular_mask(max(1, planet_sep-size_core-1), planet_sep+size_core+1, value=np.nan, size=CCF_wo_planet.shape)
            else:
                CCF_noise = np.copy(CCF_wo_planet)*annular_mask(max(1, planet_sep-1), planet_sep, value=np.nan, size=CCF_wo_planet.shape)
        CCF_signal = CCF[y0, x0]
        noise = np.sqrt(np.nanvar(CCF_noise))
        if print_value:
            print(" E[<n, t>]/Std[<n, t>] = ", round(100*np.nanmean(CCF_noise)/np.nanstd(CCF_noise), 2), "%")
        signal = CCF_signal-np.nanmean(CCF_noise)
        SNR = signal/noise
        return SNR, CCF, CCF_signal, CCF_noise



def correlation_T_rv(instru, d, wave, trans, R, Rc, used_filter, target_name, band, model="BT-Settl", vsini=0, pca=None):
    T, lg = model_T_lg_array(model)
    rv = np.linspace(-50, 50, 201) 
    cos_3d = np.zeros((len(lg), len(T), len(rv)))
    k=0
    for i in range(len(lg)):
        if model[:4] == "mol_":
            model = model[:4] + lg[i]
        for j in range(len(T)):
            print(round(100*(k+1)/(len(T)*len(lg)), 2), "%")
            cos_rv, rv = correlation_rv(instru=instru, d=d, wave=wave, trans=trans, T=T[j], lg=lg[i], model=model, R=R, Rc=Rc, used_filter=used_filter, show=False, large_rv=False, vsini=vsini, rv=rv, pca=pca)
            cos_3d[i, j, :] = cos_rv
            k+=1
    idx_max_corr = np.unravel_index(np.argmax(cos_3d, axis=None), cos_3d.shape)
    cos_2d = cos_3d[:, :, idx_max_corr[2]] # on se place à la vitesse radiale donnant le plus grand SNR
    plt.figure(dpi=300)
    plt.pcolormesh(T, lg, cos_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(cos_2d), vmax=np.nanmax(cos_2d))
    cbar = plt.colorbar() ; cbar.set_label(r"cos $\theta_{est}$", fontsize=14, labelpad=20, rotation=270)
    if model[:4] == "mol_":
        print(f"maximum correlation value of {round(np.nanmax(cos_2d), 2)} for T = {T[idx_max_corr[1]]} K, {lg[idx_max_corr[0]]} and rv = {rv[idx_max_corr[2]]} km/s")
        plt.ylabel("molecule", fontsize=12)
        plt.title(f'Correlation between molecular template and {target_name} \n data spectrum on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T[idx_max_corr[1]], T[idx_max_corr[1]]], [lg[idx_max_corr[0]], lg[idx_max_corr[0]]], 'kX', ms=10, label=r"cos $\theta_{max}$ = "+f"{round(np.nanmax(cos_2d), 2)} for T = {T[idx_max_corr[1]]} K, \n {lg[idx_max_corr[0]]} and rv = {rv[idx_max_corr[2]]} km/s")
    else:
        print(f"maximum correlation value of {round(np.nanmax(cos_2d), 2)} for T = {T[idx_max_corr[1]]} K, lg = {lg[idx_max_corr[0]]} and rv = {rv[idx_max_corr[2]]} km/s")
        plt.ylabel("template's gravity surface", fontsize=12)
        plt.title(f'Correlation between {model} spectra and {target_name} \n data spectrum on {band}-band of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T[idx_max_corr[1]], T[idx_max_corr[1]]], [lg[idx_max_corr[0]], lg[idx_max_corr[0]]], 'kX', ms=10, label=r"cos $\theta_{max}$ = "+f"{round(np.nanmax(cos_2d), 2)} for T = {T[idx_max_corr[1]]} K, \n lg = {lg[idx_max_corr[0]]} and rv = {rv[idx_max_corr[2]]} km/s")
        plt.contour(T, lg, cos_2d, linewidths=0.1, colors='k')
        plt.ylim(lg[0], lg[-1])
    plt.xlabel("template's temperature (in K)", fontsize=12)
    plt.xlim(T[0], T[-1])
    plt.legend(fontsize=12) ; plt.show()
    
    if model[:4] == "mol_":
        model = model[:4] + lg[idx_max_corr[0]]
    correlation_rv(instru=instru, d=d, wave=wave, trans=trans, T=T[idx_max_corr[1]], lg=lg[idx_max_corr[0]], model=model, R=R, Rc=Rc, used_filter=used_filter, show=True, large_rv=False, vsini=vsini, rv=rv, pca=pca)

    
    return T[idx_max_corr[1]], lg[idx_max_corr[0]], rv[idx_max_corr[2]]




def SNR_T_rv(instru, Sres, Sres_wo_planet, x0, y0, size_core, d, rv, wave, trans, R, Rc, used_filter, target_name, band, model="BT-Settl", vsini=0, pca=None):
    T, lg = model_T_lg_array(model)
    SNR_2d = np.zeros((len(lg), len(T)))
    noise_2d = np.zeros_like(SNR_2d) + np.nan
    signal_2d = np.zeros_like(SNR_2d) + np.nan
    k=0
    for i in range(len(lg)):
        if model[:4] == "mol_":
            model = model[:4] + lg[i]
        for j in range(len(T)):
            print(round(100*(k+1)/(len(T)*len(lg)), 2), "%")
            CCF, _ = molecular_mapping_rv(instru, Sres, T=T[j], lg=lg[i], model=model, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, rv=rv, print_value=False) 
            CCF_wo_planet, _ = molecular_mapping_rv(instru, Sres_wo_planet, T=T[j], lg=lg[i], model=model, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, rv=rv, print_value=False) 
            SNR_2d[i, j], CCF, CCF_signal, CCF_noise = SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, print_value=False)
            noise_2d[i, j] = np.nanstd(CCF_noise)
            signal_2d[i, j] = CCF_signal - np.nanmean(CCF_noise)
            k+=1
    SNR_2d = np.nan_to_num(SNR_2d)
    idx_max_snr = np.unravel_index(np.argmax(SNR_2d, axis=None), SNR_2d.shape)
    plt.figure()
    plt.pcolormesh(T, lg, SNR_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(SNR_2d), vmax=np.nanmax(SNR_2d))
    cbar = plt.colorbar() ; cbar.set_label("S/N", fontsize=14, labelpad=20, rotation=270)
    if model[:4] == "mol_":
        print(f"maximum S/N value of {round(np.nanmax(SNR_2d), 2)} for T = {T[idx_max_snr[1]]} K, {lg[idx_max_snr[0]]} and rv = {rv} km/s")
        plt.ylabel("molecule", fontsize=12)
        plt.title(f'S/N with different molecular template for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T[idx_max_snr[1]], T[idx_max_snr[1]]], [lg[idx_max_snr[0]], lg[idx_max_snr[0]]], 'kX', ms=10, label=r"$S/N_{max}$ = "+f"{round(np.nanmax(SNR_2d), 1)} for T = {T[idx_max_snr[1]]} K, \n {lg[idx_max_snr[0]]} and rv = {rv} km/s")
    else:
        print(f"maximum S/N value of {round(np.nanmax(SNR_2d), 2)} for T = {T[idx_max_snr[1]]} K, lg = {lg[idx_max_snr[0]]} and rv = {rv} km/s")
        plt.ylabel("template's gravity surface", fontsize=12)
        plt.title(f'S/N with different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        plt.plot([T[idx_max_snr[1]], T[idx_max_snr[1]]], [lg[idx_max_snr[0]], lg[idx_max_snr[0]]], 'kX', ms=10, label=r"$S/N_{max}$ = "+f"{round(np.nanmax(SNR_2d), 1)} for T = {T[idx_max_snr[1]]} K, \n lg = {lg[idx_max_snr[0]]} and rv = {rv} km/s")
        plt.contour(T, lg, SNR_2d, linewidths=0.1, colors='k')
        plt.ylim(lg[0], lg[-1])
    plt.xlabel("template's temperature (in K)", fontsize=12)
    plt.xlim(T[0], T[-1])
    plt.legend(fontsize=12) ; plt.show()
    
    if model[:4] == "mol_":
        model = model[:4] + lg[idx_max_snr[0]]
    correlation_rv(instru=instru, d=d, wave=wave, trans=trans, T=T[idx_max_snr[1]], lg=lg[idx_max_snr[0]], model=model, R=R, Rc=Rc, used_filter=used_filter, show=True, large_rv=False, vsini=vsini, pca=pca)

    
    if 1 == 0:
        plt.figure() ; plt.pcolormesh(T, lg, noise_2d, cmap=plt.get_cmap('rainbow'))
        plt.xlabel("planet's temperature (in K)", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(f'Noise value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        cbar = plt.colorbar() ; cbar.set_label("noise (in e-)", fontsize=14, labelpad=20, rotation=270) ; plt.show()
        plt.figure() ; plt.pcolormesh(T, lg, signal_2d, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(signal_2d), vmax=np.nanmax(signal_2d))
        plt.xlabel("planet's temperature (in K)", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(f'Signal value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
        cbar = plt.colorbar() ; cbar.set_label("signal (in e-)", fontsize=14, labelpad=20, rotation=270) ; plt.show()
    


def chi2_rv(instru, d, wave, trans, T, lg, R, Rc, used_filter, sigma_l, model="BT-Settl", show=True, large_rv=False, vsini=0):
    config_data = get_config_data(instru)
    planet_spectrum = load_planet_spectrum(T, lg, model=model, instru=instru)
    planet_spectrum = planet_spectrum.degrade_resolution(wave, renorm=True)
    if vsini > 0:
        planet_spectrum = planet_spectrum.broad(vsini)
    if model[:4] != "mol_":
        planet_spectrum = planet_spectrum.set_nbphotons_min(config_data, planet_spectrum.wavelength)
    rv = np.linspace(-50, 50, 101)
    if large_rv:
        rv = np.linspace(-2000, 2000, 200)
    chi2 = np.zeros_like(rv)
    for i in range(len(rv)):
        planet_shift = planet_spectrum.doppler_shift(rv[i]) # en ph/mn
        valid = ~np.isnan(planet_shift.flux)
        planet_shift.flux = planet_shift.flux[valid] ; planet_shift.wavelength = planet_shift.wavelength[valid]
        template, _ = filtered_flux(planet_shift.flux, R, Rc, used_filter) # en ph
        f = interp1d(planet_shift.wavelength, template, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        template = f(wave)*trans # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
        template[np.isnan(d)] = np.nan
        template = template / np.sqrt(np.nansum(template**2)) # On suppose qu'on a le template "parfait"
        R = np.nansum(d*template/sigma_l**2) / np.nansum(template**2/sigma_l**2) # voir Malin et al. 2023
        chi2[i] = np.nansum(((d-R*template)/sigma_l)**2)
    if show:     
        plt.figure() ; plt.plot(rv, cos_theta, 'k') ; plt.xlabel("Doppler velocity (in km/s)", fontsize=14) ; plt.ylabel(r"cos($\theta$)", fontsize=14) ; plt.grid(True)
        plt.figure() ; plt.plot(wave, planet_spectrum_model.high_pass_flux, 'r') ; plt.plot(wave, d, 'b')
    return chi2, rv



def chi2_T_rv(instru, d, wave, trans, R, Rc, used_filter, sigma_l, target_name, band, model="BT-Settl", vsini=0):
    T, lg = model_T_lg_array(model)
    chi2_2d = np.zeros((len(lg), len(T)))
    rv_2d = np.zeros((len(lg), len(T)))
    k=0
    for i in range(len(lg)):
        for j in range(len(T)):
            print(round(100*(k+1)/(len(T)*len(lg)), 2), "%")
            chi2, rv = chi2_rv(instru, d, wave, trans, T=T[j], lg=lg[i], R=R, Rc=Rc, used_filter=used_filter, sigma_l=sigma_l, model=model, show=False, large_rv=False, vsini=vsini)
            chi2_2d[i, j] = np.nanmin(chi2)
            rv_2d[i, j] = rv[chi2.argmin()]
            k+=1
    idx_min_chi2 = np.unravel_index(np.argmin(chi2_2d, axis=None), chi2_2d.shape) ; chi2_min = np.nanmin(chi2_2d)
    chi2_2d = chi2_2d-chi2_min
    print(f"minimum chi2 value is given for T = {T[idx_min_chi2[1]]} K, lg = {lg[idx_min_chi2[0]]} rv = {rv_2d[idx_min_chi2[0], idx_min_chi2[1]]} km/s")
    plt.figure() ; plt.pcolormesh(T, lg, chi2_2d, cmap=plt.get_cmap('rainbow_r'))
    plt.xlabel("planet's temperature (in K)", fontsize=12) ; plt.ylabel("planet's gravity surface", fontsize=12) ; plt.title(r'$\chi^2$'+f' between {model} spectra and {target_name} \n data spectrum on {band} of {instru} with $R_c$ = {Rc}', fontsize=14)
    cbar = plt.colorbar() ; cbar.set_label(r"$\chi^2$ - $\chi^2_{min}$", fontsize=14, labelpad=20, rotation=270)
    plt.plot([T[idx_min_chi2[1]], T[idx_min_chi2[1]]], [lg[idx_min_chi2[0]], lg[idx_min_chi2[0]]], 'kX', ms=10, label=r"$\chi^2_{min}$ "+f"for T = {T[idx_min_chi2[1]]} K, \n lg = {lg[idx_min_chi2[0]]} and rv = {round(rv_2d[idx_min_chi2[0], idx_min_chi2[1]], 2)} km/s")
    plt.contour(T, lg, chi2_2d, linewidths=0.1, colors='k') ; plt.ylim(lg[0], lg[-1]) ; plt.xlim(T[0], T[-1])
    plt.legend() ; plt.show()