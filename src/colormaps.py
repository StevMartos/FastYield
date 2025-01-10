from src.spectrum import *

path_file = os.path.dirname(__file__)
save_path_colormap = os.path.join(os.path.dirname(path_file), "plots/colormaps/")
plots = ["SNR", "lost_signal"]



############################################################################################################################################################################################################################################"
############################################################################################################################################################################################################################################"
############################################################################################################################################################################################################################################"



def colormap_bandwidth_resolution_with_constant_Nlambda(T_planet, T_star, lg_planet=4.0, lg_star=4.0, star_rv=0, planet_rv=25, vsini_planet=3, vsini_star=7, 
                                  spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True):
    config_data = get_config_data(instru)
    if config_data["base"]=="space" or instru=="all":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    if instru=="all":
        Npx = 4096 # number of pixels (spectral channels) considered to sample a spectrum
    else :
        Npx = 0
        for band in config_data["gratings"] :
            Npx += ( (config_data["gratings"][band].lmax - config_data["gratings"][band].lmin) * 2*config_data["gratings"][band].R/((config_data["gratings"][band].lmax + config_data["gratings"][band].lmin)/2))/len(config_data["gratings"])
        Npx = int(round(Npx, -2))
    R = np.logspace(np.log10(500), np.log10(200000), num=100)
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm
    lambda_0 = np.linspace(lmin, lmax, 2*len(R))
    SNR = np.zeros((len(R), len(lambda_0)), dtype=float)
    lost_signal = np.zeros_like(SNR)

    res_model = 1e6
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star = star.broad(vsini_star)
        
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet, lg_planet)
            albedo = albedo.interpolate_wavelength(wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
        planet = Spectrum(wave, albedo*star.flux, star.R, T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    planet = planet.broad(vsini_planet)
    star = star.doppler_shift(star_rv)
    planet = planet.doppler_shift(planet_rv)
    delta_rv = planet_rv - star_rv # by definition
    star.flux *= wave # pour être homogène à des photons
    planet.flux *= wave # pour être homogène à des photons 
    
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
        
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_bandwidth_resolution_with_constant_Nlambda, [(i, R, lmin, lmax, Npx, wave, planet, star, sky_trans, tellurics, lambda_0, Rc, filter_type, SNR, lost_signal, stellar_halo_photon_noise_limited) for i in range(len(R))]), total=len(R)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :] = SNR_1D
            lost_signal[i, :] = lost_signal_1D

    SNR /= np.nanmax(SNR)

    for plot in plots:
        plt.figure(dpi=300) ; plt.yscale('log') ; plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14) ; plt.ylabel("instrumental resolution R", fontsize=14) ; plt.ylim([R[0], R[-1]]) ; plt.xlim(lmin, lmax)
        if plot=="SNR":
            plt.contour(lambda_0, R, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(lambda_0, R, 100*SNR, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
                elif spectrum_contributions=="reflected" :
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
                    elif model=="tellurics" or model=="flat" :
                        plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
            else :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
                elif spectrum_contributions=="reflected" :
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
                    elif model=="tellurics" or model=="flat" :
                        plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
            filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{planet.T}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        elif plot=="lost_signal" :
            plt.contour(lambda_0, R, 100*lost_signal, linewidths=0.5, colors='k') ; plt.pcolormesh(lambda_0, R, 100*lost_signal, cmap=plt.get_cmap('rainbow_r'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ [%]', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"Lost signal fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
            else :
                plt.title(f"Lost signal fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}", fontsize=14, pad=14)
            filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_lost_signal_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{planet.T}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        if instru=="all":
            markers = ["o", "v", "s", "p", "*", "d", "P", "X"]
            for i, instrument in enumerate(config_data_list) :
                instrument = instrument["name"]
                if spectrum_contributions == "reflected" and instrument == "MIRIMRS":
                    pass
                else :
                    if instrument != "NIRCam":
                        x_instrument, y_instrument = [], []
                        config_data = get_config_data(instrument)
                        for band in config_data['gratings']:
                            x_instrument.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
                            y_instrument.append(config_data['gratings'][band].R)
                        plt.scatter(x_instrument, y_instrument, c='black', marker=markers[i], label=instrument)
        else :
            x_instru, y_instru, labels, x_dl = [], [], [], []
            config_data = get_config_data(instru)
            for band in config_data['gratings']:
                if (instru=="MIRIMRS" or instru=="HARMONI") and  band != "H_high":
                    labels.append(band[:2])
                elif instru=="ANDES":
                    if band=="YJH_HR_5mas":
                        labels.append(band[:3]) ; x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2) ; y_instru.append(config_data['gratings'][band].R) ; x_dl.append((config_data['gratings'][band].lmax - config_data['gratings'][band].lmin)/2)
                    continue
                else:
                    labels.append(band)
                x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
                y_instru.append(config_data['gratings'][band].R)
                x_dl.append((config_data['gratings'][band].lmax - config_data['gratings'][band].lmin)/2)
            plt.scatter(x_instru, y_instru, c='black', marker='o', label=instru+' bands')
            plt.errorbar(x_instru, y_instru, xerr=x_dl, fmt='o', color='k', linestyle='None', capsize=5)
            for i, l in enumerate(labels):
                plt.annotate(l, (x_instru[i], 1.2*y_instru[i]))
        plt.legend()
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    return lambda_0, R, SNR, lost_signal

def process_colormap_bandwidth_resolution_with_constant_Nlambda(args):
    i, R, lmin, lmax, Npx, wave, planet, star, sky_trans, tellurics, lambda_0, Rc, filter_type, SNR, lost_signal, stellar_halo_photon_noise_limited = args
    res = R[i]
    dl = (lmin+lmax)/2 / (2 * res)
    wav = np.arange(max(lmin-(Npx/2)*dl, wave[0]), min(lmax+(Npx/2)*dl, wave[-1]), dl)
    planet_R = planet.degrade_resolution(wav)
    star_R = star.degrade_resolution(wav)  
    SNR_1D = np.zeros((len(lambda_0)))
    lost_signal_1D = np.zeros((len(lambda_0)))
    if tellurics :
        sky_R = sky_trans.degrade_resolution(wav, renorm=False)
    for j, l0 in enumerate(lambda_0):
        umin = l0-(Npx/2)*dl
        umax = l0+(Npx/2)*dl
        valid = np.where(((wav<umax)&(wav>umin)))
        if tellurics:
            trans = sky_R.flux[valid]
        else:
            trans = 1 
        star_R_crop = Spectrum(wav[valid], star_R.flux[valid], res, None)
        planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, None)
        planet_HF, planet_BF = filtered_flux(planet_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF = filtered_flux(star_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        template = trans*planet_HF / np.sqrt(np.nansum((trans*planet_HF)**2))
        alpha = np.nansum(trans*planet_HF * template)
        beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if stellar_halo_photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_R_crop.flux * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength-independent limiting noise (e.g. RON domination)
        SNR_1D[j] = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    return i, SNR_1D, lost_signal_1D



############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################"
############################################################################################################################################################################################################################################"



def colormap_bandwidth_resolution_with_constant_Dlambda(T_planet, T_star, lg_planet=4.0, lg_star=4.0, star_rv=0, planet_rv=30, vsini_planet=3, vsini_star=7, 
                                  spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=False):
    config_data = get_config_data(instru)
    if config_data["base"]=="space" or instru=="all":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    Dl = 0
    for band in config_data["gratings"] :
        Dl += ( config_data['gratings'][band].lmax - config_data['gratings'][band].lmin )/len(config_data["gratings"])
    R = np.logspace(np.log10(500), np.log10(200000), num=100)
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm
    
    res_model = 1e6
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(min(0.1, lmin-Dl), lmax+Dl, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star = star.broad(vsini_star)
        
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet, lg_planet)
            albedo = albedo.interpolate_wavelength(wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
        planet = Spectrum(wave, albedo*star.flux, star.R, T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    planet = planet.broad(vsini_planet)
    star = star.doppler_shift(star_rv)
    planet = planet.doppler_shift(planet_rv)
    delta_rv = planet_rv - star_rv
    star.flux *= wave # pour être homogène à des photons
    planet.flux *= wave # pour être homogène à des photons 
    
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    
    lambda_0 = np.linspace(lmin, lmax, 2*len(R))
    SNR = np.zeros((len(R), len(lambda_0)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    
    
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_bandwidth_resolution_with_constant_Dlambda, [(i, R, lmin, lmax, Dl, wave, planet, star, sky_trans, tellurics, lambda_0, Rc, filter_type, SNR, lost_signal, stellar_halo_photon_noise_limited) for i in range(len(R))]), total=len(R)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :] = SNR_1D
            lost_signal[i, :] = lost_signal_1D

    SNR /= np.nanmax(SNR)

    for plot in plots:
        plt.figure(dpi=300) ; plt.yscale('log') ; plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14) ; plt.ylabel("instrumental resolution R", fontsize=14) ; plt.ylim([R[0], R[-1]]) ; plt.xlim(lmin, lmax)
        if plot=="SNR":
            plt.contour(lambda_0, R, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(lambda_0, R, 100*SNR, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
                elif spectrum_contributions=="reflected" :
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
                    elif model=="tellurics" or model=="flat" :
                        plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
            else :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
                elif spectrum_contributions=="reflected" :
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
                    elif model=="tellurics" or model=="flat" :
                        plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
            filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{planet.T}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Dl{Dl}"
        elif plot=="lost_signal" :
            plt.contour(lambda_0, R, 100*lost_signal, linewidths=0.5, colors='k') ; plt.pcolormesh(lambda_0, R, 100*lost_signal, cmap=plt.get_cmap('rainbow_r'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ [%]', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"Lost signal fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
            else :
                plt.title(f"Lost signal fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K, \n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$\Delta\lambda$="+f"{round(Dl, 1)}µm", fontsize=14, pad=14)
            filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_lost_signal_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{planet.T}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Dl{Dl}"
        if instru=="all":
            markers = ["o", "v", "s", "p", "*", "d", "P", "X"]
            for i, instrument in enumerate(config_data_list) :
                instrument = instrument["name"]
                if spectrum_contributions == "reflected" and instrument == "MIRIMRS":
                    pass
                else :
                    if instrument != "NIRCam":
                        x_instrument, y_instrument = [], []
                        config_data = get_config_data(instrument)
                        for band in config_data['gratings']:
                            x_instrument.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
                            y_instrument.append(config_data['gratings'][band].R)
                        plt.scatter(x_instrument, y_instrument, c='black', marker=markers[i], label=instrument)
        else :
            x_instru, y_instru, labels, x_dl = [], [], [], []
            config_data = get_config_data(instru)
            for band in config_data['gratings']:
                x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
                y_instru.append(config_data['gratings'][band].R)
                x_dl.append((config_data['gratings'][band].lmax - config_data['gratings'][band].lmin)/2)
                if (instru=="MIRIMRS" or instru=="HARMONI") and  band != "H_high":
                    labels.append(band[:2])
                else:
                    labels.append(band)
            plt.scatter(x_instru, y_instru, c='black', marker='o', label=instru+' bands')
            plt.errorbar(x_instru, y_instru, xerr=x_dl, fmt='o', color='k', linestyle='None', capsize=5)
            for i, l in enumerate(labels):
                plt.annotate(l, (x_instru[i], 1.2*y_instru[i]))
        plt.legend()
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    return lambda_0, R, SNR, lost_signal

def process_colormap_bandwidth_resolution_with_constant_Dlambda(args):
    i, R, lmin, lmax, Dl, wave, planet, star, sky_trans, tellurics, lambda_0, Rc, filter_type, SNR, lost_signal, stellar_halo_photon_noise_limited = args
    res = R[i]
    dl = (lmin+lmax)/2 / (2 * res)
    wav = np.arange(max(lmin-Dl/2, wave[0]), min(lmax+Dl/2, wave[-1]), dl)
    planet_R = planet.degrade_resolution(wav)
    star_R = star.degrade_resolution(wav)  
    SNR_1D = np.zeros((len(lambda_0)))
    lost_signal_1D = np.zeros((len(lambda_0)))
    if tellurics :
        sky_R = sky_trans.degrade_resolution(wav, renorm=False)
    for j, l0 in enumerate(lambda_0):
        umin = l0-Dl/2
        umax = l0+Dl/2
        valid = np.where(((wav<umax)&(wav>umin)))
        if tellurics:
            trans = sky_R.flux[valid]
        else:
            trans = 1 
        star_R_crop = Spectrum(wav[valid], star_R.flux[valid], res, None)
        planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, None)    
        planet_HF, planet_BF = filtered_flux(planet_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF = filtered_flux(star_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        template = trans*planet_HF / np.sqrt(np.nansum((trans*planet_HF)**2))
        alpha = np.nansum(trans*planet_HF * template)
        beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if stellar_halo_photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_R_crop.flux * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength-independent limiting noise (e.g. RON domination)
        SNR_1D[j] = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    return i, SNR_1D, lost_signal_1D



############################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################"
############################################################################################################################################################################################################################################"



def colormap_bandwidth_Tp(instru, T_star, lg_planet=4.0, lg_star=4.0, star_rv=0, planet_rv=25, vsini_planet=3, vsini_star=7, 
                          spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True):    
    config_data = get_config_data(instru)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    R = 0. ; Npx = 0.
    for band in config_data["gratings"] :
        R += config_data["gratings"][band].R/len(config_data["gratings"])
        Npx += ( (config_data["gratings"][band].lmax - config_data["gratings"][band].lmin) * 2*config_data["gratings"][band].R/((config_data["gratings"][band].lmax + config_data["gratings"][band].lmin)/2))/len(config_data["gratings"])
    R = int(round(R, -2)) ; Npx = int(round(Npx, -2))
    lmin = 0.6
    if spectrum_contributions=="reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm
        
    res_model = 1e6
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    if vsini_star != 0:
        star = star.broad(vsini_star)
    star.flux *= wave # pour être homogène à des photons

    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    
    T_arr = np.arange(300, 3100, 100)
    lambda_0 = np.linspace(lmin, lmax, 4*len(T_arr))
    SNR = np.zeros((len(T_arr), len(lambda_0)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    opti_l0 = np.zeros((len(T_arr)))
    
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_bandwidth_Tp, [(i, T_arr, lg_planet, star_rv, planet_rv, vsini_planet, star, tellurics, sky_trans, spectrum_contributions, model, instru, wave, lambda_0, Npx, R, Rc, filter_type, stellar_halo_photon_noise_limited) for i in range(len(T_arr))]), total=len(T_arr)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :] = SNR_1D / np.nanmax(SNR_1D)
            lost_signal[i, :] = lost_signal_1D
            opti_l0[i] = lambda_0[SNR_1D.argmax()]
    
    delta_rv = planet_rv - star_rv
    for plot in plots :
        plt.figure(dpi=300) ; plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14) ; plt.ylabel("planet's temperature [K]", fontsize=14) ; plt.ylim([T_arr[0], T_arr[-1]]) ; plt.xlim(lmin, lmax)
        if plot=="SNR":
            plt.pcolormesh(lambda_0, T_arr, 100*SNR, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ", fontsize=14, pad=14)
            else :
                plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ", fontsize=14, pad=14)
            filename = f"colormaps_bandwidth_Tp/Colormap_bandwitdth_Tp_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        elif plot=="lost_signal":
            plt.pcolormesh(lambda_0, T_arr, 100*lost_signal, cmap=plt.get_cmap('rainbow_r'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ [%]', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"Lost signal fluctuations for {instru} (with tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ", fontsize=14, pad=14)
            else :
                plt.title(f"Lost signal fluctuations for {instru} (without tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ", fontsize=14, pad=14)
            filename = f"colormaps_bandwidth_Tp/Colormap_bandwitdth_Tp_lost_signal_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        bands = []
        for nb, band in enumerate(config_data['gratings']):
            x = (config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2 # center lambda
            if instru=="MIRIMRS" or instru=="NIRSpec" :
                plt.plot([x, x], [T_arr[0], 0.95*np.nanmean(T_arr)], "k")
                plt.plot([x, x], [1.1*np.nanmean(T_arr), T_arr[-1]], "k")
                plt.annotate(band[:2], (x-0.2, np.nanmean(T_arr)))
            elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS" or instru=="HiRISE":
                if band=="HK" or band=="YJH":
                    if band not in bands :
                        plt.plot([x, x], [T_arr[0], 0.95*np.nanmean(T_arr)], "k")
                        plt.plot([x, x], [1.1*np.nanmean(T_arr), T_arr[-1]], "k")
                        plt.annotate(band, (x-0.15, np.nanmean(T_arr)))
                        bands.append(band)
                elif band[0]=="Y" or band[0]=="J" or band[0]=="H"  or band[0]=="K":
                    if band[0] not in bands:
                        plt.plot([x, x], [T_arr[0], 0.95*np.nanmean(T_arr)], "k")
                        plt.plot([x, x], [1.1*np.nanmean(T_arr), T_arr[-1]], "k")
                        plt.annotate(band[0], (x-0.07, np.nanmean(T_arr)))
                        bands.append(band[0])
        plt.plot([], [], "k", label=instru+' bands') ; plt.plot(opti_l0, T_arr, 'k:', label=r"optimum $\lambda_0$") ; plt.legend()
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    return lambda_0, T_arr, SNR, lost_signal

def process_colormap_bandwidth_Tp(args):
    i, T_arr, lg_planet, star_rv, planet_rv, vsini_planet, star, tellurics, sky_trans, spectrum_contributions, model, instru, wave, lambda_0, Npx, R, Rc, filter_type, stellar_halo_photon_noise_limited = args
    T_planet = T_arr[i]
    SNR_1D = np.zeros((len(lambda_0)))
    lost_signal_1D = np.zeros((len(lambda_0)))
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet, lg_planet)
            albedo = albedo.interpolate_wavelength(wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
        planet = Spectrum(wave, albedo*star.flux, star.R, T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
        planet.flux *= wave # pour être homogène à des photons 
    star = star.doppler_shift(star_rv)
    planet = planet.doppler_shift(planet_rv)
    planet = planet.broad(vsini_planet)
    for j, l0 in enumerate(lambda_0):
        dl = l0/(2*R)
        umin = l0-(Npx/2)*dl
        umax = l0+(Npx/2)*dl
        wav = np.arange(umin, umax, dl)
        star_R = star.degrade_resolution(wav)  
        planet_R = planet.degrade_resolution(wav)
        if tellurics:
            trans = sky_trans.degrade_resolution(wav, renorm=False).flux
        else:
            trans = 1
        planet_HF, planet_BF = filtered_flux(planet_R.flux, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF = filtered_flux(star_R.flux, R=R, Rc=Rc, filter_type=filter_type)
        template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
        alpha = np.nansum(trans*planet_HF * template)
        beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if stellar_halo_photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_R.flux * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength-independent limiting noise (e.g. RON domination)
        lost_signal_1D[j] = beta / alpha
        SNR_1D[j] = (alpha-beta) / noise
    return i, SNR_1D, lost_signal_1D


############################################################################################################################################################################################################################################"
############################################################################################################################################################################################################################################"
############################################################################################################################################################################################################################################"



def colormap_rv(T_planet, T_star, lg_planet=4.0, lg_star=4.0, vsini_planet=3, vsini_star=7, 
                spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", band="H", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True):
    config_data = get_config_data(instru)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    lmin = config_data['gratings'][band].lmin # lambda_min de la bande considérée
    lmax = config_data['gratings'][band].lmax # lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    dl = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wav = np.arange(lmin, lmax, dl) # axe de longueur d'onde de la bande considérée
    
    res_model = max(100000, R)
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star = star.broad(vsini_star)
        
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet, lg_planet)
            albedo = albedo.interpolate_wavelength(wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
        planet = Spectrum(wave, albedo*star.flux, star.R, T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    planet = planet.broad(vsini_planet)
    star.flux *= wave # pour être homogène à des photons
    planet.flux *= wave # pour être homogène à des photons 
    
    trans = transmission(instru, wav, band, tellurics, apodizer="NO_SP")
    
    star_rv = np.linspace(-100, 100, 201)
    delta_rv = np.copy(star_rv)
    SNR = np.zeros((len(star_rv), len(delta_rv)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    
    
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_rv, [(i, star, planet, trans, star_rv, wav, delta_rv, R, Rc, filter_type, stellar_halo_photon_noise_limited) for i in range(len(star_rv))]), total=len(star_rv)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :] = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    SNR /= np.nanmax(SNR)
    
    for plot in plots :
        plt.figure(dpi=300) ; plt.xlabel("delta radial velocity [km/s]", fontsize=14) ; plt.ylabel("star radial velocity [km/s]", fontsize=14)
        if plot=="SNR":
            plt.contour(delta_rv, star_rv, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(delta_rv, star_rv, 100*SNR, cmap=plt.get_cmap('rainbow'))
            if tellurics :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light with {model} model, \n $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
                elif spectrum_contributions=="reflected":
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light with {model} albedo, \n $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
                    elif model=="tellurics" or model=="flat":
                        plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light with {model} albedo and $T_*$={T_star}K", fontsize=14, pad=14)
            else :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light with {model} model, \n $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
                elif spectrum_contributions=="reflected":
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light with {model} albedo, \n $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
                    elif model=="tellurics" or model=="flat":
                        plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light with {model} albedo and $T_*$={T_star}K", fontsize=14, pad=14)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_rv/Colormap_rv_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        elif plot=="lost_signal":
            plt.contour(delta_rv, star_rv, 100*lost_signal, linewidths=0.333, colors='k') ; plt.pcolormesh(delta_rv, star_rv, 100*lost_signal, cmap=plt.get_cmap('rainbow_r'))
            if tellurics :
                plt.title(f"Lost signal fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R, -2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14)
            else :
                plt.title(f"Lost signal fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ [%]', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_rv/Colormap_rv_lost_signal_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    return delta_rv, star_rv, SNR, lost_signal

def process_colormap_rv(args):
    i, star, planet, trans, star_rv, wav, delta_rv, R, Rc, filter_type, stellar_halo_photon_noise_limited = args
    SNR_1D = np.zeros((len(delta_rv)))
    lost_signal_1D = np.zeros((len(delta_rv)))
    star_shift = star.doppler_shift(star_rv[i])
    star_shift = star_shift.degrade_resolution(wav, renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
    for j in range(len(delta_rv)):
        planet_shift = planet.doppler_shift(star_rv[i] + delta_rv[j])
        planet_shift = planet_shift.degrade_resolution(wav, renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
        planet_HF, planet_BF = filtered_flux(planet_shift.flux, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF = filtered_flux(star_shift.flux, R=R, Rc=Rc, filter_type=filter_type)
        template = trans*planet_HF / np.sqrt(np.nansum((trans*planet_HF)**2))
        alpha = np.nansum(trans*planet_HF * template)
        beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if stellar_halo_photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_shift.flux * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength-independent limiting noise (e.g. RON domination)
        lost_signal_1D[j] = beta / alpha
        SNR_1D[j] = (alpha-beta) / noise
    return i, SNR_1D, lost_signal_1D


############################################################################################################################################################################################################################################"



def colormap_vsini(T_planet, T_star, lg_planet=4.0, lg_star=4.0, delta_rv=30, 
                   spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", band="H", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True):
    """
    https://www.aanda.org/articles/aa/pdf/2022/03/aa42314-21.pdf
    """
    if instru=="MIRIMRS" or instru=="NIRSpec" or instru=="NIRCam" :
        tellurics = False
    elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS":
        tellurics = True
    config_data=get_config_data(instru)
    lmin = config_data['gratings'][band].lmin # lambda_min de la bande considérée
    lmax = config_data['gratings'][band].lmax # lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    dl = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wav = np.arange(lmin, lmax, dl) # axe de longueur d'onde de la bande considérée
    
    res_model = max(100000, R)
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star.flux *= wave # pour être homogène à des photons

    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet, lg_planet)
            albedo = albedo.interpolate_wavelength(wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
        planet = None
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
        planet = planet.doppler_shift(delta_rv)
        planet.flux *= wave # pour être homogène à des photons
        albedo = None
    
    trans = transmission(instru, wav, band, tellurics, apodizer="NO_SP")
    
    vsini_star = np.linspace(0.1, 100, 201)
    vsini_planet = np.copy(vsini_star)
    SNR = np.zeros((len(vsini_star), len(vsini_planet)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_vsini, [(i, star, planet, albedo, spectrum_contributions, trans, vsini_star, vsini_planet, wav, delta_rv, R, Rc, filter_type, stellar_halo_photon_noise_limited) for i in range(len(vsini_star))]), total=len(vsini_star)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :] = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    SNR /= np.nanmax(SNR)
    
    for plot in plots:
        plt.figure(dpi=300) ; plt.xlabel("planet Vsini [km/s]", fontsize=14) ; plt.ylabel("star Vsini [km/s]", fontsize=14)
        if plot=="SNR":
            plt.contour(vsini_planet, vsini_star, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(vsini_planet, vsini_star, 100*SNR, cmap=plt.get_cmap('rainbow'))
            if tellurics :
                plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R, -2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
            else :
                plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_vsini/Colormap_vsini_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        elif plot=="lost_signal":
            plt.contour(vsini_planet, vsini_star, 100*lost_signal, linewidths=0.333, colors='k') ; plt.pcolormesh(vsini_planet, vsini_star, 100*lost_signal, cmap=plt.get_cmap('rainbow_r'))
            if tellurics :
                plt.title(f"Lost signal fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R, -2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
            else :
                plt.title(f"Lost signal fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R, 2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K", fontsize=14, pad=14)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ [%]', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_vsini/Colormap_vsini_lost_signal_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    return vsini_planet, vsini_star, SNR, lost_signal

def process_colormap_vsini(args):
    i, star, planet, albedo, spectrum_contributions, trans, vsini_star, vsini_planet, wav, delta_rv, R, Rc, filter_type, stellar_halo_photon_noise_limited = args
    SNR_1D = np.zeros((len(vsini_planet)))
    lost_signal_1D = np.zeros((len(vsini_planet)))
    star_broad = star.broad(vsini_star[i])
    if spectrum_contributions=="reflected" :
        planet = Spectrum(star.wavelength, albedo*star_broad.flux, star.R, None)
        if delta_rv != 0:
            planet = planet.doppler_shift(delta_rv)
    star_broad = star_broad.degrade_resolution(wav, renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
    for j in range(len(vsini_planet)):
        planet_broad = planet.broad(vsini_planet[j])
        planet_broad = planet_broad.degrade_resolution(wav, renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
        planet_HF, planet_BF = filtered_flux(planet_broad.flux, R, Rc, filter_type)
        star_HF, star_BF = filtered_flux(star_broad.flux, R, Rc, filter_type)
        template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
        alpha = np.nansum(trans*planet_HF * template)
        beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if stellar_halo_photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_broad.flux * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength-independent limiting noise (e.g. RON domination)
        lost_signal_1D[j] = beta / alpha
        SNR_1D[j] = np.abs(alpha-beta)/noise
    return i, SNR_1D, lost_signal_1D



############################################################################################################################################################################################################################################"



def colormap_maxsep_phase(instru="HARMONI", band="H", inc=90):
    if instru=="MIRIMRS" or instru=="NIRSpec" or instru=="NIRCam" :
        sep_unit = "arcsec" ; strehl = None
    elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS":
        if instru=="ANDES" or instru=="HARMONI":
            strehl = "JQ1"
        elif instru=="ERIS" :
            strehl = "JQ0"
        sep_unit = "mas"
    config_data=get_config_data(instru)
    if instru == "HARMONI":
        iwa = config_data["apodizers"]["NO_SP"].sep
    else :
        lambda_c = (config_data["lambda_range"]["lambda_min"]+config_data["lambda_range"]["lambda_max"])/2 *1e-6*u.m #(config_data["lambda_range"]["lambda_max"]+config_data["lambda_range"]["lambda_min"])/2 *1e-6*u.m
        diameter = config_data['telescope']['diameter'] *u.m
        iwa = lambda_c/diameter*u.rad ; iwa = iwa.to(u.mas) ; iwa = iwa.value

    PSF_profile, fraction_PSF, separation, pxscale = PSF_profile_fraction_separation(band=band, strehl=strehl, apodizer="NO_SP", coronagraph=None, instru=instru, config_data=config_data, sep_unit=sep_unit)
    
    phase = np.linspace(0, np.pi, 1000)
    a = np.arccos(- np.sin(inc*np.pi/180) * np.cos(phase) ) 
    g_a = ( np.sin(a) + (np.pi - a) * np.cos(a) ) / np.pi # phase function
    maxsep = np.copy(separation)[separation>iwa]
    PSF_profile = PSF_profile[separation>iwa]
    separation = separation[separation>iwa]
    SNR = np.zeros((len(phase), len(maxsep)), dtype=float)
    SNR_syst = np.zeros((len(phase), len(maxsep)), dtype=float)
        
    plt.figure(dpi=300)
    plt.yscale('log')
    plt.plot(maxsep, 100*1/(np.sqrt(PSF_profile))/np.nanmax(1/(np.sqrt(PSF_profile))), 'r', label="$1/\sqrt{PSF}$ ($\sigma_{\gamma}$)")
    plt.plot(maxsep, 100*1/(PSF_profile)/np.nanmax(1/(PSF_profile)), 'r--', label="$1/PSF$ ($\sigma_{syst}$)")
    plt.plot(maxsep, 100*1/maxsep**2/max(1/maxsep**2), 'g', label="$1/R^2$ (scaling factor)")
    plt.plot(maxsep, 100*1/(np.sqrt(PSF_profile)*maxsep**2)/max(1/(np.sqrt(PSF_profile)*maxsep**2)), 'b', label="$S/N \propto 1/R^2$ x $1/\sqrt{PSF}$ (w/o $\sigma_{syst}$)")
    plt.plot(maxsep, 100*1/(PSF_profile*maxsep**2)/max(1/(PSF_profile*maxsep**2)), 'b--', label="$S/N \propto 1/R^2$ x $1/PSF$ ($\sigma_{syst}$ domination)")
    plt.legend()
    plt.xlabel(f"maximum elongation (in {sep_unit})")
    plt.ylabel(r"$GAIN_{S/N}$ [%]")
    plt.grid(True)
    plt.show()
    
    for i in tqdm(range(len(phase))):
        for j in range(len(maxsep)):
            sep = maxsep[j] * np.sqrt( np.sin(phase[i])**2 + np.cos(phase[i])**2 * np.cos(inc*np.pi/180)**2 ) # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/74/pdf
            frac_PSF_sep = PSF_profile[np.abs(separation-sep).argmin()]
            SNR[i, j] = g_a[i] * 1/maxsep[j]**2 * 1/np.sqrt(frac_PSF_sep)
            SNR_syst[i, j] = g_a[i] * 1/maxsep[j]**2 * 1/frac_PSF_sep

    SNR /= np.nanmax(SNR)
    SNR_syst /= np.nanmax(SNR_syst)
    
    plt.figure(dpi=300) ; plt.xlabel(f"maximum elongation (in {sep_unit})", fontsize=14) ; plt.ylabel("phase (in rad)", fontsize=14)
    plt.contour(maxsep, phase, 100*SNR, linewidths=0.333, colors='k')
    plt.pcolormesh(maxsep, phase, 100*SNR, cmap=plt.get_cmap('rainbow'))
    plt.title(f"S/N fluctuations for {instru} on {band}-band\n for planets in reflected light with inc = {int(round(inc))}° \n with photon noise", fontsize=14, pad=14)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    filename = f"colormaps_maxsep_phase/Colormap_maxsep_phase_SNR_{instru}_{band}_reflected_inc{inc}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
        
    plt.figure(dpi=300) ; plt.xlabel(f"maximum elongation (in {sep_unit})", fontsize=14) ; plt.ylabel("phase (in rad)", fontsize=14)
    plt.contour(maxsep, phase, 100*SNR_syst, linewidths=0.333, colors='k')
    plt.pcolormesh(maxsep, phase, 100*SNR_syst, cmap=plt.get_cmap('rainbow'))
    plt.title(f"S/N fluctuations for {instru} on {band}-band\n for planets in reflected light with inc = {int(round(inc))}° \n with systematic noise", fontsize=14, pad=14)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    filename = f"colormaps_maxsep_phase/Colormap_maxsep_phase_SNR_{instru}_{band}_reflected_inc{inc}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    
    return maxsep, phase, SNR, SNR_syst




############################################################################################################################################################################################################################################"



def colormap_best_parameters_earth(Npx=10000, T_planet=288, T_star=5800, lg_planet=3.0, lg_star=4.4, delta_rv=30, vsini_planet=0.5, vsini_star=2, SMA=1, planet_radius=1, star_radius=1, distance=1, 
                                  thermal_model="BT-Settl", reflected_model="tellurics", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True, norm_plot = "star"):
        
    d = distance * u.pc # parsec
    SMA = SMA * u.AU # AU
    planet_radius = planet_radius * u.earthRad # earth radius
    star_radius = star_radius * u.solRad # star radius
    g_a = 0.32 # elongation max phase function
    
    D_space = 8 # m
    D_ground = 40 # m
    
    S_space = (D_space/2)**2 * np.pi # m
    S_ground = (D_ground/2)**2 * np.pi # m
    
    N = 200
    R = np.logspace(np.log10(1000), np.log10(100000), num=N)
    lmin = 0.6
    lmax = 6 # en µm
    lambda_0 = np.linspace(lmin, lmax, N)

    res_model = 2e5
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star.flux *= float((star_radius/d).decompose()**2)
    star = star.broad(vsini_star)
        
    planet_thermal = load_planet_spectrum(T_planet, lg_planet, model=thermal_model)
    planet_thermal = planet_thermal.interpolate_wavelength(wave, renorm = False)
    planet_thermal.flux *= float((planet_radius/d).decompose()**2)

    albedo = load_albedo(planet_thermal.T, planet_thermal.lg)
    albedo = albedo.interpolate_wavelength(wave, renorm = False)
    if reflected_model == "PICASO":
        planet_reflected = star.flux * albedo.flux * g_a * (planet_radius/SMA).decompose()**2
    elif reflected_model == "flat":
        planet_reflected = star.flux * np.nanmean(albedo.flux)*1 * g_a * (planet_radius/SMA).decompose()**2
    elif reflected_model == "tellurics":
        wave_tell, tell = fits.getdata("sim_data/Transmission/sky_transmission_airmass_2.5.fits")
        f = interp1d(wave_tell, tell, bounds_error=False, fill_value=np.nan)
        tell = f(wave)
        planet_reflected = star.flux * np.nanmean(albedo.flux)/np.nanmean(tell)*tell * g_a * (planet_radius/SMA).decompose()**2
    else :
        raise KeyError(reflected_model+" IS NOT A VALID REFLECTED MODEL : tellurics, flat, or PICASO")
    planet_reflected = Spectrum(wave, np.nan_to_num(np.array(planet_reflected.value)), max(star.R, albedo.R), albedo.T, lg_planet, reflected_model)
    planet_thermal = planet_thermal.doppler_shift(delta_rv)
    planet_reflected = planet_reflected.doppler_shift(delta_rv)
    planet_thermal = planet_thermal.broad(vsini_planet)
    planet_reflected = planet_reflected.broad(vsini_planet)
        
    bb_star = (2*const.h * const.c**2 / (wave*u.micron)**5).decompose() / np.expm1((const.h * const.c/(wave * u.micron * const.k_B * T_star * u.K)).decompose())
    bb_star = bb_star.to(u.J/u.s/u.m**2/u.micron)
    bb_star = np.pi * bb_star.value * (1/1)**2 # on suppose que le facteur de dilution vaut 1
    bb_star *= float((star_radius/d).decompose()**2)
    
    bb_planet_thermal = (2*const.h * const.c**2 / (wave*u.micron)**5).decompose() / np.expm1((const.h * const.c/(wave * u.micron * const.k_B * T_planet * u.K)).decompose())
    bb_planet_thermal = bb_planet_thermal.to(u.J/u.s/u.m**2/u.micron)
    bb_planet_thermal = np.pi * bb_planet_thermal.value * float((planet_radius/d).decompose()**2) # facteur de dilution = (R/d)^2
    
    bb_planet_reflected = bb_star * np.nanmean(albedo.flux)*1 * g_a * (planet_radius/SMA).decompose()**2
    bb_planet_reflected = np.nan_to_num(np.array(bb_planet_reflected.value))
    
    star.flux *= wave # pour être homogène à des photons
    planet_thermal.flux *= wave # pour être homogène à des photons
    planet_reflected.flux *= wave # pour être homogène à des photons      
    bb_star *= wave
    bb_planet_thermal *= wave
    bb_planet_reflected *= wave

    sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
    sky_trans = fits.getdata(sky_transmission_path)
    sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
    sky_trans = sky_trans.interpolate_wavelength(wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    
    if norm_plot == "star":
        norm = star.flux ; norm_bb = bb_star
    if norm_plot == "1":
        norm = 1 ; norm_bb = 1
    bb_planet = bb_planet_thermal + bb_planet_reflected
    
    plt.figure(dpi=300)
    plt.title("Eart-like spectrum")
    plt.plot(wave, planet_thermal.flux/norm, 'r-', label=f"thermal ({thermal_model})")
    plt.plot(wave, planet_reflected.flux/norm, 'b-', label=f"reflected ({reflected_model})")
    print("HERE = ", bb_planet)
    plt.plot(wave, bb_planet/norm_bb, 'k-', label=f"blackbody ({round(100*np.nansum(planet_thermal.flux+planet_reflected.flux) / np.nansum(bb_planet), 1)} %)")
    plt.legend()
    if norm_plot == "1":
        plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("wavelength [µm]")
    plt.ylabel("flux [contrast unit]")
    plt.xlim(lmin, lmax)
    if norm_plot == "star":
        plt.ylim(1e-15, np.nanmax((planet_reflected.flux+planet_thermal.flux)/star.flux)*10)
    elif norm_plot == "1":
        plt.ylim(1e-30, np.nanmax((planet_reflected.flux+planet_thermal.flux))*10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on()
    plt.show()
    
    SNR_space_thermal = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_space_reflected = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_ground_thermal = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_ground_reflected = np.zeros((len(R), len(lambda_0)), dtype=float)
    
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_best_parameters_earth, [(i, R, lmin, lmax, planet_thermal, planet_reflected, star, sky_trans, lambda_0, Npx, Rc, filter_type, S_space, S_ground, stellar_halo_photon_noise_limited) for i in range(len(R))]), total=len(R)))
        for (i, SNR_space_thermal_1D, SNR_space_reflected_1D, SNR_ground_thermal_1D, SNR_ground_reflected_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR_space_thermal[i, :] = SNR_space_thermal_1D
            SNR_space_reflected[i, :] = SNR_space_reflected_1D
            SNR_ground_thermal[i, :] = SNR_ground_thermal_1D
            SNR_ground_reflected[i, :] = SNR_ground_reflected_1D
    max_SNR = max(np.nanmax(SNR_space_thermal), np.nanmax(SNR_space_reflected), np.nanmax(SNR_ground_thermal), np.nanmax(SNR_ground_reflected))    
    SNR_space_thermal /= max_SNR
    SNR_space_reflected /= max_SNR
    SNR_ground_thermal /= max_SNR
    SNR_ground_reflected /= max_SNR
    
    SNR_space_thermal[np.isnan(SNR_space_thermal)] = 0. ; SNR_space_reflected[np.isnan(SNR_space_reflected)] = 0. ; SNR_ground_thermal[np.isnan(SNR_ground_thermal)] = 0. ; SNR_ground_reflected[np.isnan(SNR_ground_reflected)] = 0.
    
    fig, axs = plt.subplots(2, 2, dpi=300, sharex=True, sharey=True, figsize=(14, 9))
    fig.suptitle(f"S/N fluctuations for earth-like with "+r"$N_{\lambda}$="+f"{round(round(Npx, -3))} and $R_c$={Rc}", fontsize=20)
    for i, base in enumerate(["space", "ground"]):
        for j, contribution in enumerate(["thermal", "reflected"]):
            SNR = locals()["SNR_" + base + "_" + contribution]
            model = locals()[contribution + "_model"]
            idx_max_snr = np.unravel_index(np.argmax(np.nan_to_num(SNR), axis=None), SNR.shape)
            ax = axs[i, j]
            ax.set_yscale('log')
            if i == 1:
                ax.set_xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=12)
            if j == 0:
                ax.set_ylabel("instrumental resolution R", fontsize=12)
            ax.set_ylim([R[0], R[-1]])
            ax.set_xlim(lmin, lmax)
            contour = ax.contour(lambda_0, R, 100 * SNR / np.nanmax(SNR), linewidths=0.333, colors='k')
            pcm = ax.pcolormesh(lambda_0, R, 100 * SNR / np.nanmax(SNR), cmap='rainbow', vmin=0, vmax=100)
            ax.plot(lambda_0[idx_max_snr[1]], R[idx_max_snr[0]], 'kX', label=f"max for $\lambda_0$ = {round(lambda_0[idx_max_snr[1]], 1)}\u00b5m and R = {int(round(R[idx_max_snr[0]], -2))}")
            dl = (lmin + lmax) / 2 / (2 * R[idx_max_snr[0]])
            umin = lambda_0[idx_max_snr[1]] - (Npx / 2) * dl
            umax = lambda_0[idx_max_snr[1]] + (Npx / 2) * dl
            ax.errorbar(lambda_0[idx_max_snr[1]], R[idx_max_snr[0]], xerr=(umax - umin)/2, fmt='X', color='k', linestyle='None', capsize=5)
            ax.set_title(f"{base.capitalize()} / {contribution.capitalize()}", fontsize=14, pad=14)
            ax.legend(fontsize=10)
    cbar = fig.colorbar(pcm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04) ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    filename = f"colormaps_best_parameters_earth_like/colormaps_best_parameters_earth_like_Npx_{round(round(Npx, -3))}_Rc_{Rc}_thermal_{thermal_model}_reflected_{reflected_model}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    
    idx_max_snr_space_thermal = np.unravel_index(np.argmax(np.nan_to_num(SNR_space_thermal), axis=None), SNR_space_thermal.shape)
    idx_max_snr_space_reflected = np.unravel_index(np.argmax(np.nan_to_num(SNR_space_reflected), axis=None), SNR_space_reflected.shape)
    idx_max_snr_ground_thermal = np.unravel_index(np.argmax(np.nan_to_num(SNR_ground_thermal), axis=None), SNR_ground_thermal.shape)
    idx_max_snr_ground_reflected = np.unravel_index(np.argmax(np.nan_to_num(SNR_ground_reflected), axis=None), SNR_ground_reflected.shape)
    
    plt.figure(dpi=300)
    plt.title(f"S/N fluctuations for Earth-like with "+r"$N_{\lambda}$="+f"{round(round(Npx, -3))} and $R_c$={Rc}", fontsize=14, pad=14)
    plt.plot(lambda_0, SNR_space_thermal[idx_max_snr_space_thermal[0], :], "r--", label=f"space/thermal: R = {int(round(R[idx_max_snr_space_thermal[0]], -3))}")
    plt.plot(lambda_0, SNR_space_reflected[idx_max_snr_space_reflected[0], :], "b--", label=f"space/reflected: R = {int(round(R[idx_max_snr_space_reflected[0]], -3))}")
    plt.plot(lambda_0, SNR_ground_thermal[idx_max_snr_ground_thermal[0], :], "r", label=f"ground & thermal: R = {int(round(R[idx_max_snr_ground_thermal[0]], -3))}")
    plt.plot(lambda_0, SNR_ground_reflected[idx_max_snr_ground_reflected[0], :], "b", label=f"ground & reflected for R = {int(round(R[idx_max_snr_ground_reflected[0]], -3))}")
    plt.legend(loc="upper left")
    plt.ylabel("S/N [normalized]") ; plt.xlabel("central wavelength range $\lambda_0$ [µm]")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on()
    plt.xlim(lmin, lmax)
    plt.ylim(1e-5, 2)    
    filename = f"colormaps_best_parameters_earth_like/plot_best_parameters_earth_like_Npx_{round(round(Npx, -3))}_Rc_{Rc}_thermal_{thermal_model}_reflected_{reflected_model}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    
    return lambda_0, R, Npx, SNR_space_thermal, SNR_space_reflected, SNR_ground_thermal, SNR_ground_reflected

def process_colormap_best_parameters_earth(args):
    i, R, lmin, lmax, planet_thermal, planet_reflected, star, sky_trans, lambda_0, Npx, Rc, filter_type, S_space, S_ground, stellar_halo_photon_noise_limited = args
    res = R[i]
    dl = (lmin+lmax)/2 / (2 * res)
    wav = np.arange(lmin, lmax, dl)
    planet_thermal_R = planet_thermal.degrade_resolution(wav)
    planet_reflected_R = planet_reflected.degrade_resolution(wav)
    star_R = star.degrade_resolution(wav)   
    sky_R = sky_trans.degrade_resolution(wav, renorm=False)
    SNR_space_thermal_1D = np.zeros((len(lambda_0)))
    SNR_space_reflected_1D = np.zeros((len(lambda_0)))
    SNR_ground_thermal_1D = np.zeros((len(lambda_0)))
    SNR_ground_reflected_1D = np.zeros((len(lambda_0)))
    for j, l0 in enumerate(lambda_0):
        umin = l0-(Npx/2)*dl
        umax = l0+(Npx/2)*dl
        valid = np.where(((wav<umax)&(wav>umin)))
        trans = sky_R.flux[valid]

        star_R_crop = Spectrum(wav[valid], star_R.flux[valid], res, None)
        planet_thermal_R_crop = Spectrum(wav[valid], planet_thermal_R.flux[valid], res, None)   
        planet_reflected_R_crop = Spectrum(wav[valid], planet_reflected_R.flux[valid], res, None)
        star_HF, star_BF = filtered_flux(star_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        planet_thermal_HF, planet_thermal_BF = filtered_flux(planet_thermal_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        planet_reflected_HF, planet_reflected_BF = filtered_flux(planet_reflected_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
        
        template_space_thermal = planet_thermal_HF/np.sqrt(np.nansum((planet_thermal_HF)**2))
        alpha_space_thermal = np.sqrt(np.nansum((planet_thermal_HF)**2))
        beta_space_thermal = np.nansum(star_HF*planet_thermal_BF/star_BF * template_space_thermal)
        template_space_reflected = planet_reflected_HF/np.sqrt(np.nansum((planet_reflected_HF)**2))
        alpha_space_reflected = np.sqrt(np.nansum((planet_reflected_HF)**2))
        beta_space_reflected = np.nansum(star_HF*planet_reflected_BF/star_BF * template_space_reflected)
        template_ground_thermal = trans*planet_thermal_HF/np.sqrt(np.nansum((trans*planet_thermal_HF)**2))
        alpha_ground_thermal = np.sqrt(np.nansum((trans*planet_thermal_HF)**2))
        beta_ground_thermal = np.nansum(trans*star_HF*planet_thermal_BF/star_BF * template_ground_thermal)
        template_ground_reflected = trans*planet_reflected_HF/np.sqrt(np.nansum((trans*planet_reflected_HF)**2))
        alpha_ground_reflected = np.sqrt(np.nansum((trans*planet_reflected_HF)**2))
        beta_ground_reflected = np.nansum(trans*star_HF*planet_reflected_BF/star_BF * template_ground_reflected)
        
        if stellar_halo_photon_noise_limited:
            SNR_space_thermal_1D[j] = np.sqrt(S_space) * (alpha_space_thermal - beta_space_thermal) / np.sqrt(np.nansum(star_R_crop.flux * template_space_thermal**2))
            SNR_space_reflected_1D[j] = np.sqrt(S_space) * (alpha_space_reflected - beta_space_reflected) / np.sqrt(np.nansum(star_R_crop.flux * template_space_reflected**2))
            SNR_ground_thermal_1D[j] = np.sqrt(S_ground) * (alpha_ground_thermal - beta_ground_thermal) / np.sqrt(np.nansum(trans*star_R_crop.flux * template_ground_thermal**2))
            SNR_ground_reflected_1D[j] = np.sqrt(S_ground) * (alpha_ground_reflected - beta_ground_reflected) / np.sqrt(np.nansum(trans*star_R_crop.flux * template_ground_reflected**2))
        else:
            SNR_space_thermal_1D[j] = np.sqrt(S_space) * (alpha_space_thermal - beta_space_thermal) / 1.
            SNR_space_reflected_1D[j] = np.sqrt(S_space) * (alpha_space_reflected - beta_space_reflected) / 1.
            SNR_ground_thermal_1D[j] = np.sqrt(S_ground) * (alpha_ground_thermal - beta_ground_thermal) / 1.
            SNR_ground_reflected_1D[j] = np.sqrt(S_ground) * (alpha_ground_reflected - beta_ground_reflected) / 1.
    return i, SNR_space_thermal_1D, SNR_space_reflected_1D, SNR_ground_thermal_1D, SNR_ground_reflected_1D



############################################################################################################################################################################################################################################"












