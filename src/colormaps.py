from src.FastYield import *

path_file          = os.path.dirname(__file__)
save_path_colormap = os.path.join(os.path.dirname(path_file), "plots/colormaps/")
plots              = ["SNR", "lost_signal"]
plots              = ["SNR"]
cmap_colormaps     = "rainbow"
contour_levels     = np.linspace(0, 100, 11)
R_model            = 1_000_000
R_model            = R0_max



######################## Bandwidth vs Resolution (with constant Nlambda or Dlambda) ###########################################################################################################################################################################################



def colormap_bandwidth_resolution_with_constant_Nlambda(instru="HARMONI", T_planet=300, T_star=6000, lg_planet=3.0, lg_star=4.44, delta_rv=30, vsini_planet=3, vsini_star=7, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True, Nlambda=None, num=100):
    
    # Get instru specs
    if instru != "all" and instru != "PCS":
        config_data = get_config_data(instru)
        if config_data["type"]=="imager":
            raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")
    
    # tellurics (or not)
    if instru=="all":
        tellurics = False
    elif instru=="PCS":
        tellurics = True
    elif config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
        
    # Number of pixels (spectral channels/bins) considered to sample a spectrum
    if instru=="all":
        Npx = 4096
    elif instru=="PCS":
        Npx = Nlambda
    else :
        Npx = np.zeros((len(config_data["gratings"])))
        for iband, band in enumerate(config_data["gratings"]):
            R_band     = config_data["gratings"][band].R
            lmax_band  = config_data["gratings"][band].lmax
            lmin_band  = config_data["gratings"][band].lmin
            DELTA_band = lmax_band - lmin_band
            l0_band    = (lmax_band + lmin_band) / 2
            delta_band = l0_band / (2*R_band)
            Npx[iband] = DELTA_band/delta_band
        Npx = int(round(np.nanmean(Npx), -2))
    
    # Raw wavelength axis
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm    
    lmin_model = 0.9*lmin
    lmax_model = 1.1*lmax
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)
    
    # Getting star spectrum
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave_model, renorm=False) 
    star = star.broad(vsini_star) # Broadening the spectrum
    
    # Getting planet spectrum
    if spectrum_contributions=="reflected" :
        albedo = load_albedo(T_planet, lg_planet, model=model, airmass=2.5)
        albedo = albedo.interpolate_wavelength(wave_model, renorm=False)
        planet = Spectrum(wave_model, albedo.flux*star.flux)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave_model, renorm=False)
    planet = planet.broad(vsini_planet)     # Broadening the spectrum
    planet = planet.doppler_shift(delta_rv) # Shifting the spectrum
    
    # To be homogenous to photons
    star.flux   *= wave_model
    planet.flux *= wave_model
    
    # Geting tellurics model (or not)
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave_model, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    else:
        sky_trans = None
    
    # Defining arrays
    R_arr       = np.logspace(np.log10(1000), np.log10(200000), num=num)
    l0_arr      = np.linspace(lmin, lmax, len(R_arr))
    SNR         = np.zeros((len(R_arr), len(l0_arr)))
    lost_signal = np.zeros((len(R_arr), len(l0_arr)))
    
    # Parallel calculations
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_colormap_bandwidth_resolution_with_constant_Nlambda, [(i, R_arr[i], lmin, lmax, Npx, lmin_model, lmax_model, planet, star, sky_trans, l0_arr, Rc, filter_type, photon_noise_limited) for i in range(len(R_arr))]), total=len(R_arr)))
        for (i, SNR_1D, lost_signal_1D) in results:
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    SNR /= np.nanmax(SNR)
    
    # Plots
    for plot in plots:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.yscale('log')
        plt.xlabel(r"Central wavelength $\lambda_0$ [$\mu$m]", fontsize=14)
        plt.ylabel("Spectral resolution $R$", fontsize=14)
        plt.ylim([R_arr[0], R_arr[-1]])
        plt.xlim(lmin, lmax)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        
        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
        else:
            data = 100 * lost_signal
            cmap = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
        
        # Heatmap with pcolormesh
        mesh = plt.pcolormesh(l0_arr, R_arr, data, cmap=cmap, shading='auto', vmin=0, vmax=100)
        
        # Contours
        cs = plt.contour(l0_arr, R_arr, data, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar photon noise regime" if photon_noise_limited else "detector noise regime"
        title_text   = (f"{'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, $T_p$={T_planet}K, "r"$\Delta$rv="f"{delta_rv}km/s, "r"$N_\lambda$="f"{Npx}")
        #plt.title(title_text, fontsize=16, pad=14)
        plt.title(r"$N_\lambda$="f"{np.round(Npx, -2):.0f} channels", fontsize=20, pad=14)

        # Scatter & errorbars for bands
        if instru=="all" or instru=="PCS":
            markers = ["o", "v", "s", "p", "*", "d", "P", "X"]
            for i, config_data in enumerate(config_data_list):
                lmin_instru = config_data["lambda_range"]["lambda_min"]
                lmax_instru = config_data["lambda_range"]["lambda_max"]
                if config_data["type"] != "imager" and ((lmin < lmin_instru < lmax) or (lmin < lmax_instru < lmax)):
                    instrument                 = config_data["name"]
                    x_instrument, y_instrument = [], []
                    for band in config_data['gratings']:
                        R_band    = config_data['gratings'][band].R
                        lmin_band = config_data['gratings'][band].lmin
                        lmax_band = config_data['gratings'][band].lmax
                        l0_band   = (lmax_band + lmin_band) / 2
                        x_instrument.append(l0_band)
                        y_instrument.append(R_band)
                    plt.scatter(x_instrument, y_instrument, c='black', marker=markers[i], s=50, label=instrument)
        else :
            x_instru, y_instru, labels, x_dl = [], [], [], []
            for iband, band in enumerate(config_data['gratings']):
                if (instru=="MIRIMRS" or instru=="HARMONI") and  band != "H_high":
                    labels.append(band[:2])
                elif instru=="ANDES" and iband==0:
                    labels.append(band[:3])
                else:
                    labels.append(band.replace('_', ' '))
                R_band     = config_data['gratings'][band].R
                lmin_band  = config_data['gratings'][band].lmin
                lmax_band  = config_data['gratings'][band].lmax
                l0_band    = (lmax_band + lmin_band) / 2
                DELTA_band = lmax_band - lmin_band
                x_instru.append(l0_band)
                y_instru.append(R_band)
                x_dl.append(DELTA_band / 2)
            plt.errorbar(x_instru, y_instru, xerr=x_dl, fmt='o', color='k', linestyle='None', capsize=5, label=f"{instru}")
            for i, l in enumerate(labels):
                plt.annotate(l, (x_instru[i], 1.2*y_instru[i]), ha='center', fontsize=12, fontweight="bold")

        plt.legend(fontsize=12, loc="upper right", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_with_constant_Nlambda_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}_{noise_regime.replace(' ', '_')}"
        plt.savefig(save_path_colormap + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
    return l0_arr, R_arr, SNR, lost_signal

def process_colormap_bandwidth_resolution_with_constant_Nlambda(args):
    i, R, lmin, lmax, Npx, lmin_model, lmax_model, planet, star, sky_trans, l0_arr, Rc, filter_type, photon_noise_limited = args
    dl_R     = (lmin + lmax)/2 / (2 * R)
    lmin_R   = max(lmin - (Npx/2)*dl_R, lmin_model)
    lmax_R   = min(lmax + (Npx/2)*dl_R, lmax_model)
    wave_R   = np.arange(lmin_R, lmax_R, dl_R)
    planet_R = planet.degrade_resolution(wave_R, renorm=True).flux
    star_R   = star.degrade_resolution(wave_R, renorm=True).flux
    SNR_1D         = np.zeros((len(l0_arr)))
    lost_signal_1D = np.zeros((len(l0_arr)))
    if sky_trans is not None:
        sky_R = sky_trans.degrade_resolution(wave_R, renorm=False)
    for j, l0 in enumerate(l0_arr):
        umin  = l0 - (Npx/2)*dl_R
        umax  = l0 + (Npx/2)*dl_R
        valid = np.where(( (wave_R<umax) & (wave_R>umin) ))
        if sky_trans is not None:
            trans = sky_R.flux[valid]
        else:
            trans = 1 
        star_R_crop          = star_R[valid]
        planet_R_crop        = planet_R[valid]
        planet_HF, planet_BF = filtered_flux(planet_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF     = filtered_flux(star_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        template             = trans*planet_HF 
        template            /= np.sqrt(np.nansum(template**2))
        alpha                = np.nansum(trans*planet_HF * template)
        beta                 = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_R_crop * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    return i, SNR_1D, lost_signal_1D



def colormap_bandwidth_resolution_with_constant_Dlambda(instru="HARMONI", T_planet=300, T_star=6000, lg_planet=3.0, lg_star=4.44, delta_rv=30, vsini_planet=3, vsini_star=7, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=False):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    
    # Bandwidth considered to sample a spectrum
    Dl = np.zeros((len(config_data["gratings"])))
    for iband, band in enumerate(config_data["gratings"]):
        Dl[iband] = config_data["gratings"][band].lmax - config_data["gratings"][band].lmin
    Dl = round(np.nanmean(Dl), 3)
    
    # Raw wavelength axis
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm    
    lmin_model = max(lmin - Dl/2, 0.1)
    lmax_model = lmax + Dl/2
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)

    # Getting star spectrum
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave_model, renorm=False) 
    star = star.broad(vsini_star) # Broadening the spectrum
    
    # Getting planet spectrum
    if spectrum_contributions=="reflected" :
        albedo = load_albedo(T_planet, lg_planet, model=model, airmass=2.5)
        albedo = albedo.interpolate_wavelength(wave_model, renorm=False)
        planet = Spectrum(wave_model, albedo.flux*star.flux)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave_model, renorm=False)
    planet = planet.broad(vsini_planet)     # Broadening the spectrum
    planet = planet.doppler_shift(delta_rv) # Shifting the spectrum
    
    # To be homogenous to photons
    star.flux   *= wave_model
    planet.flux *= wave_model
    
    # Geting tellurics model (or not)
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave_model, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    else:
        sky_trans = None
    
    # Defining arrays
    R_arr       = np.logspace(np.log10(1000), np.log10(200000), num=100)
    l0_arr      = np.linspace(lmin, lmax, len(R_arr))
    SNR         = np.zeros((len(R_arr), len(l0_arr)))
    lost_signal = np.zeros((len(R_arr), len(l0_arr)))
    
    # Parallel calculations
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_colormap_bandwidth_resolution_with_constant_Dlambda, [(i, R_arr[i], lmin, lmax, Dl, lmin_model, lmax_model, planet, star, sky_trans, l0_arr, Rc, filter_type, photon_noise_limited) for i in range(len(R_arr))]), total=len(R_arr)))
        for (i, SNR_1D, lost_signal_1D) in results:
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    SNR /= np.nanmax(SNR)
    
    # Plots
    for plot in plots:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.yscale('log')
        plt.xlabel(r"Central wavelength $\lambda_0$ [$\mu$m]", fontsize=14)
        plt.ylabel("Spectral resolution $R$", fontsize=14)
        plt.ylim([R_arr[0], R_arr[-1]])
        plt.xlim(lmin, lmax)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        
        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
        else:
            data = 100 * lost_signal
            cmap = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
        
        # Heatmap with pcolormesh
        mesh = plt.pcolormesh(l0_arr, R_arr, data, cmap=cmap, shading='auto', vmin=0, vmax=100)
        
        # Contours
        cs = plt.contour(l0_arr, R_arr, data, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar photon noise regime" if photon_noise_limited else "detector noise regime"
        title_text   = (f"{'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, $T_p$={T_planet}K, "r"$\Delta$rv="f"{delta_rv}km/s, "r"$\Delta\lambda$="f"{Dl:.2f}µm")
        plt.title(title_text, fontsize=16, pad=14)
        
        # Scatter & errorbars for bands
        if instru=="all":
            markers = ["o", "v", "s", "p", "*", "d", "P", "X"]
            for i, config_data in enumerate(config_data_list) :
                if config_data["type"]!="imager":
                    instrument                 = config_data["name"]
                    x_instrument, y_instrument = [], []
                    for band in config_data['gratings']:
                        R_band    = config_data['gratings'][band].R
                        lmin_band = config_data['gratings'][band].lmin
                        lmax_band = config_data['gratings'][band].lmax
                        l0_band   = (lmax_band + lmin_band) / 2
                        x_instrument.append(l0_band)
                        y_instrument.append(R_band)
                    plt.scatter(x_instrument, y_instrument, c='black', marker=markers[i], s=50, label=instrument)
        else :
            x_instru, y_instru, labels, x_dl = [], [], [], []
            for iband, band in enumerate(config_data['gratings']):
                if (instru=="MIRIMRS" or instru=="HARMONI") and  band != "H_high":
                    labels.append(band[:2])
                elif instru=="ANDES" and iband==0:
                    labels.append(band[:3])
                else:
                    labels.append(band.replace('_', ' '))
                R_band     = config_data['gratings'][band].R
                lmin_band  = config_data['gratings'][band].lmin
                lmax_band  = config_data['gratings'][band].lmax
                l0_band    = (lmax_band + lmin_band) / 2
                DELTA_band = lmax_band - lmin_band
                x_instru.append(l0_band)
                y_instru.append(R_band)
                x_dl.append(DELTA_band / 2)
            plt.errorbar(x_instru, y_instru, xerr=x_dl, fmt='o', color='k', linestyle='None', capsize=5, label=f"{instru} bands")
            for i, l in enumerate(labels):
                plt.annotate(l, (x_instru[i], 1.2*y_instru[i]), ha='center', fontsize=12)

        plt.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_with_constant_Dlambda_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Dl{Dl}µm_{noise_regime.replace(' ', '_')}"
        plt.savefig(save_path_colormap + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
    return l0_arr, R_arr, SNR, lost_signal

def process_colormap_bandwidth_resolution_with_constant_Dlambda(args):
    i, R, lmin, lmax, Dl, lmin_model, lmax_model, planet, star, sky_trans, l0_arr, Rc, filter_type, photon_noise_limited = args
    dl_R     = (lmin + lmax)/2 / (2 * R)
    lmin_R   = max(lmin-Dl/2, lmin_model)
    lmax_R   = min(lmax+Dl/2, lmax_model)
    wave_R   = np.arange(lmin_R, lmax_R, dl_R)
    planet_R = planet.degrade_resolution(wave_R, renorm=True).flux
    star_R   = star.degrade_resolution(wave_R, renorm=True).flux
    SNR_1D         = np.zeros((len(l0_arr)))
    lost_signal_1D = np.zeros((len(l0_arr)))
    if sky_trans is not None:
        sky_R = sky_trans.degrade_resolution(wave_R, renorm=False)
    for j, l0 in enumerate(l0_arr):
        umin  = l0 - Dl/2
        umax  = l0 + Dl/2
        valid = np.where(( (wave_R<umax) & (wave_R>umin) ))
        if sky_trans is not None:
            trans = sky_R.flux[valid]
        else:
            trans = 1 
        star_R_crop          = star_R[valid]
        planet_R_crop        = planet_R[valid]
        planet_HF, planet_BF = filtered_flux(planet_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF     = filtered_flux(star_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        template             = trans*planet_HF 
        template            /= np.sqrt(np.nansum(template**2))
        alpha                = np.nansum(trans*planet_HF * template)
        beta                 = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_R_crop * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    return i, SNR_1D, lost_signal_1D



######################## Bandwidth (or bands) vs Planet temperature ###########################################################################################################################################################################################



def colormap_bandwidth_Tp(instru, T_star=6000, lg_planet=3.0, lg_star=4.44, delta_rv=30, vsini_planet=3, vsini_star=7, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True):    
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    
    # Mean instru specs (R, Npx)
    R   = np.zeros((len(config_data["gratings"])))
    Npx = np.zeros((len(config_data["gratings"])))
    for iband, band in enumerate(config_data["gratings"]):
        R_band     = config_data["gratings"][band].R
        R[iband]   = R_band
        lmax_band  = config_data["gratings"][band].lmax
        lmin_band  = config_data["gratings"][band].lmin
        DELTA_band = lmax_band - lmin_band
        l0_band    = (lmax_band + lmin_band) / 2
        delta_band = l0_band / (2*R_band)
        Npx[iband] = DELTA_band/delta_band
    R   = int(round(np.nanmean(R), -2))
    Npx = int(round(np.nanmean(Npx), -2))
    
    # Raw wavelength axis
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm
    dl         = (lmin+lmax)/2 / (2*R)
    lmin_model = max(lmin - Npx/2*dl, 0.1)
    lmax_model = lmax + Npx/2*dl
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)
    
    # Getting star spectrum
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave_model, renorm=False) 
    star = star.broad(vsini_star) # Broadening the spectrum
    
    # To be homogenous to photons
    star.flux *= wave_model
    
    # Geting tellurics model (or not)
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave_model, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    else:
        sky_trans = None
        
    # Defining arrayrs
    N           = 100
    T_arr       = np.linspace(300, 2000, N)
    l0_arr      = np.linspace(lmin, lmax, len(T_arr))
    SNR         = np.zeros((len(T_arr), len(l0_arr)))
    lost_signal = np.zeros((len(T_arr), len(l0_arr)))
    opti_l0     = np.zeros((len(T_arr)))
    
    # Calculating matrices
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_bandwidth_Tp, [(i, T_arr[i], lg_planet, delta_rv, vsini_planet, star, sky_trans, spectrum_contributions, model, instru, wave_model, l0_arr, Npx, R, Rc, filter_type, photon_noise_limited) for i in range(len(T_arr))]), total=len(T_arr)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :]         = SNR_1D / np.nanmax(SNR_1D) # Normalizing each row
            lost_signal[i, :] = lost_signal_1D
            opti_l0[i]        = l0_arr[SNR_1D.argmax()]

    # Plots
    for plot in plots:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel(r"Central wavelength $\lambda_0$ [$\mu$m]", fontsize=14)
        plt.ylabel("Planet temperature [K]", fontsize=14)
        plt.ylim([T_arr[0], T_arr[-1]])
        plt.xlim(lmin, lmax)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        
        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
        else:
            data = 100 * lost_signal
            cmap = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
        
        # Heatmap with pcolormesh
        mesh = plt.pcolormesh(l0_arr, T_arr, data, cmap=cmap, shading='auto', vmin=0, vmax=100)
        
        # Contours
        cs = plt.contour(l0_arr, T_arr, data, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar photon noise regime" if photon_noise_limited else "detector noise regime"
        title_text   = (f"{'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, "r"$\Delta$rv="f"{delta_rv}km/s, R={R}, "r"$N_\lambda$="f"{Npx}")
        plt.title(title_text, fontsize=16, pad=14)
        
        # Bandes spectrales
        ax.plot([], [], "k", label=f"{instru} bands")
        bands_done = []
        y_center   = np.nanmean(T_arr)
        for nb, band in enumerate(config_data['gratings']):
            x = (config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2
            draw_band = False
            band_label = None
        
            if instru in ["MIRIMRS", "NIRSpec"]:
                draw_band  = True
                band_label = band[:2]
            elif instru in ["ANDES", "HARMONI", "ERIS", "HiRISE"]:
                if band in ["HK", "YJH"] and band not in bands_done:
                    draw_band  = True
                    band_label = band
                elif band[0] in ["Y", "J", "H", "K"] and band[0] not in bands_done:
                    draw_band  = True
                    band_label = band[0]
        
            if draw_band:
                ax.plot([x, x], [T_arr[0], 0.95*y_center], "k", lw=1.2)
                ax.plot([x, x], [1.05*y_center, T_arr[-1]], "k", lw=1.2)
                ax.annotate(band_label, (x, y_center), ha='center', va='center', fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
                bands_done.append(band_label)
        
        # # Optimum lambda_0
        # ax.plot(opti_l0, T_arr, 'k:', lw=1.2, label=r"Optimum $\lambda_0$")

        plt.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        filename = f"colormaps_bandwidth_Tp/Colormap_bandwidth_Tp_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}_R{R}_{noise_regime.replace(' ', '_')}"
        plt.savefig(save_path_colormap + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
    return l0_arr, T_arr, SNR, lost_signal

def process_colormap_bandwidth_Tp(args):
    i, T_planet, lg_planet, delta_rv, vsini_planet, star, sky_trans, spectrum_contributions, model, instru, wave_model, l0_arr, Npx, R, Rc, filter_type, photon_noise_limited = args
    
    SNR_1D         = np.zeros((len(l0_arr)))
    lost_signal_1D = np.zeros((len(l0_arr)))
    
    # Getting planet spectrum
    if spectrum_contributions=="reflected" :
        albedo = load_albedo(T_planet, lg_planet, model=model, airmass=2.5)
        albedo = albedo.interpolate_wavelength(wave_model, renorm=False)
        planet = Spectrum(wave_model, albedo.flux*star.flux)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave_model, renorm=False)
        # To be homogenous to photons
        planet.flux *= wave_model        
    planet = planet.broad(vsini_planet)     # Broadening the spectrum
    planet = planet.doppler_shift(delta_rv) # Shifting the spectrum
    
    # Calculation for each lambda0
    for j, l0 in enumerate(l0_arr):
        
        # Degrading the spectra on wave
        dl       = l0 / (2*R)
        umin     = l0 - (Npx/2)*dl
        umax     = l0 + (Npx/2)*dl
        wave     = np.arange(umin, umax, dl)
        star_R   = star.degrade_resolution(wave, renorm=True).flux
        planet_R = planet.degrade_resolution(wave, renorm=True).flux
        if sky_trans is not None:
            trans = sky_trans.degrade_resolution(wave, renorm=False).flux
        else:
            trans = 1
            
        # High- and low-pass filtering the spectra
        planet_HF, planet_BF = filtered_flux(planet_R, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_BF     = filtered_flux(star_R, R=R, Rc=Rc, filter_type=filter_type)
        # S/N and signal loss calculations
        template  = trans*planet_HF 
        template /= np.sqrt(np.nansum(template**2))
        alpha     = np.nansum(trans*planet_HF * template)
        beta      = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_R * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    
    return i, SNR_1D, lost_signal_1D



def colormap_bands_Tp(instru, T_star=6000, lg_planet=3.0, lg_star=4.44, delta_rv=30, vsini_planet=3, vsini_star=7, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True):    
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    
    # Raw wavelength axis
    lmin_model = 0.98*config_data["lambda_range"]["lambda_min"]
    lmax_model = 1.02*config_data["lambda_range"]["lambda_max"]
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)
    
    # Getting star spectrum
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave_model, renorm=False) 
    star = star.broad(vsini_star) # Broadening the spectrum
    
    # To be homogenous to photons
    star.flux *= wave_model
    
    # Geting tellurics model (or not)
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(wave_model, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    else:
        sky_trans = None
    
    # Definig arrays
    N            = 100
    T_arr        = np.linspace(300, 2000, N)
    bands_labels = np.full(0, "", dtype=object)
    for nb, band in enumerate(config_data["gratings"]):
        if instru=="ANDES":
            if "10mas" in band:
                bands_labels = np.append(bands_labels, band)
        else:
            bands_labels = np.append(bands_labels, band)
    SNR          = np.zeros((len(bands_labels), len(T_arr)))
    lost_signal  = np.zeros((len(bands_labels), len(T_arr)))
        
    # Calculating matrices
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_bands_Tp, [(nb, band, config_data, T_arr, lg_planet, delta_rv, vsini_planet, star, sky_trans, spectrum_contributions, model, instru, wave_model, Rc, filter_type, photon_noise_limited) for nb, band in enumerate(bands_labels)]), total=len(bands_labels)))
        for (nb, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[nb, :]         = SNR_1D
            lost_signal[nb, :] = lost_signal_1D
    
    # Normalizing each column (for each temperature/scientific case)
    for nt in range(len(T_arr)):
        SNR[:, nt] /= np.nanmax(SNR[:, nt])
    
    for nb in range(len(bands_labels)):
        bands_labels[nb] = bands_labels[nb].replace('_', ' ').replace('10mas ', '')
    
    # Plots
    cmap                  = plt.get_cmap('rainbow')
    bands_idx             = np.arange(len(bands_labels))
    bands_idx_extended    = np.arange(-1, len(bands_labels)+1)
    bands_labels_extended = np.append(np.append(bands_labels[0], bands_labels), bands_labels[-1])
    
    for plot in plots:
                
        plt.figure(figsize=(10, 6), dpi=300)
        plt.ylabel("Bands", fontsize=14)
        plt.xlabel("Planet temperature [K]", fontsize=14)
        plt.xlim([T_arr[0], T_arr[-1]])
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()

        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
        else:
            data = 100 * lost_signal
            cmap = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
        data_extended = np.vstack([data[0], data, data[-1]])

        # Heatmap with pcolormesh        
        mesh = plt.pcolormesh(T_arr, bands_idx_extended, data_extended, cmap=cmap, shading='auto', vmin=0, vmax=100) 
        
        # Contours
        T_mesh, bands_mesh = np.meshgrid(T_arr, bands_idx_extended, indexing='xy')
        cs                 = plt.contour(T_mesh, bands_mesh, data_extended, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar photon noise regime" if photon_noise_limited else "detector noise regime"
        title_text   = (f"{instru} {'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, "r"$\Delta$rv="f"{delta_rv}km/s")
        plt.title(title_text, fontsize=16, pad=14)
        
        plt.ylim(-0.5, len(bands_labels)-0.5)
        plt.yticks(bands_idx, bands_labels)
        plt.tight_layout()
        filename = f"colormaps_bands_Tp/Colormap_bands_Tp_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_{noise_regime.replace(' ', '_')}"
        plt.savefig(save_path_colormap + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
def process_colormap_bands_Tp(args):
    nb, band, config_data, T_arr, lg_planet, delta_rv, vsini_planet, star, sky_trans, spectrum_contributions, model, instru, wave_model, Rc, filter_type, photon_noise_limited = args

    R_band         = config_data["gratings"][band].R
    lmin_band      = config_data["gratings"][band].lmin
    lmax_band      = config_data["gratings"][band].lmax
    dl_band        = (lmin_band+lmax_band)/2 / (2*R_band)
    wave_band      = np.arange(lmin_band, lmax_band, dl_band)    
    SNR_1D         = np.zeros((len(T_arr)))
    lost_signal_1D = np.zeros((len(T_arr)))
    
    # Degrading the spectra on wave_band
    star_band = star.degrade_resolution(wave_band, renorm=True).flux
    if sky_trans is not None:
        trans = sky_trans.degrade_resolution(wave_band, renorm=False).flux
    else:
        trans = 1
    
    # High- and low-pass filtering the spectra
    star_HF, star_BF = filtered_flux(star_band, R=R_band, Rc=Rc, filter_type=filter_type)
    
    # Calculation for each planet's temperature
    for nt, T_planet in enumerate(T_arr):
        
        # Getting planet spectrum
        if spectrum_contributions=="reflected" :
            albedo = load_albedo(T_planet, lg_planet, model=model, airmass=2.5)
            albedo = albedo.interpolate_wavelength(wave_model, renorm=False)
            planet = Spectrum(wave_model, albedo.flux*star.flux)
        elif spectrum_contributions=="thermal" :
            planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
            planet = planet.interpolate_wavelength(wave_model, renorm=False)
            # To be homogenous to photons
            planet.flux *= wave_model        
        planet = planet.broad(vsini_planet)     # Broadening the spectrum
        planet = planet.doppler_shift(delta_rv) # Shifting the spectrum
        
        # Degrading the spectra on wave_band
        planet_band = planet.degrade_resolution(wave_band, renorm=True).flux
        
        # High- and low-pass filtering the spectra
        planet_HF, planet_BF = filtered_flux(planet_band, R=R_band, Rc=Rc, filter_type=filter_type)
        
        # S/N and signal loss calculations
        template  = trans*planet_HF 
        template /= np.sqrt(np.nansum(template**2))
        alpha     = np.nansum(trans*planet_HF * template)
        beta      = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_band * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[nt]         = (alpha - beta) / noise
        lost_signal_1D[nt] = beta / alpha

    return nb, SNR_1D, lost_signal_1D



######################## Bands vs Planet types (SNR or parameters uncertainties) ###########################################################################################################################################################################################



def colormap_bands_planets_SNR(mode="multi", instru="HARMONI", thermal_model="BT-Settl", reflected_model="PICASO", exposure_time=120, strehl="JQ1", systematic=False, PCA=False, planet_types=planet_types):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True

    # get spectrum contribution + name model
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Load the planet tables and retrieving SNR
    table           = "Archive"
    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    SNR       = []
    SNR_bands = {}
    for band in config_data["gratings"]:
        SNR_bands[f"{band}"] = []
    for apodizer in config_data["apodizers"]:
        for coronagraph in config_data["coronagraphs"]:
            coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
            filename        = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}"
            planet_table    = load_planet_table(f"{filename}.ecsv")    
            SNR.append(SNR_from_table(table=planet_table, exposure_time=exposure_time, band="INSTRU"))
            for band in config_data["gratings"]:
                SNR_bands[f"{band}"].append(SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band))
    
    # Keeping only the best SNR
    planet_table["SNR"] = np.nanmax(np.stack(SNR, axis=0), axis=0)
    for band in config_data["gratings"]:
        planet_table[f"SNR_{band}"] = np.nanmax(np.stack(SNR_bands[f"{band}"], axis=0), axis=0)

    # Find planets based on mode
    planet_table_pd  = planet_table.to_pandas() # convert to pandas to find matching types
    selected_planets = set()                    # Set to store already assigned planets for mode = "unique"
    matching_planets = {planet_type: find_matching_planets(criteria, planet_table_pd, mode, selected_planets) for planet_type, criteria in planet_types.items()}

    # Table plot
    plot_matching_planets(matching_planets, exposure_time, mode, planet_types=planet_types)

    # Definig arrays
    planet_types_arr           = np.array([])
    list_planets               = []
    planet_table["PlanetType"] = np.array(planet_table["PlanetType"], dtype="<U32")

    # Saving all planet entries, valid types and bands
    for ptype, planets in matching_planets.items():
        if len(planets) > 0:
            planet_types_arr = np.append(planet_types_arr, ptype)
            for planet in planets:
                planet_raw = copy.deepcopy(planet_table[get_planet_index(planet_table, planet["PlanetName"])])
                planet_raw["PlanetType"] = ptype
                list_planets.append(planet_raw)
    bands_labels = np.full(0, "", dtype=object)
    for nb, band in enumerate(config_data["gratings"]):
        if instru == "ANDES" and len(config_data["gratings"]) > 3:
            if "10mas" in band:
                bands_labels = np.append(bands_labels, band)
        else:
            bands_labels = np.append(bands_labels, band)
    
    # Retrieving SNR
    SNR = np.zeros((len(bands_labels), len(planet_types_arr)), dtype=float)
    for planet in list_planets:
        SNR_1D = np.zeros((len(bands_labels)))
        for nb, band in enumerate(bands_labels):
            SNR_1D[nb] = planet[f"SNR_{band}"]
        SNR_1D /= np.nanmax(SNR_1D) # In order to have a relative SNR gain across bands
        itype   = np.where(planet["PlanetType"]==planet_types_arr)[0][0]
        SNR[:, itype] += SNR_1D # Doing a mean over all itype planets
    
    # Normalizing the SNR gain for each planet type
    for itype in range(len(planet_types_arr)):
        SNR[:, itype] /= np.nanmax(SNR[:, itype])
        
    for nb in range(len(bands_labels)):
        bands_labels[nb] = bands_labels[nb].replace('_', ' ')
        if instru=="ANDES" and len(config_data["gratings"]) > 3:
            bands_labels[nb] = bands_labels[nb].replace(' 10mas', '').replace('YJH', 'R =')+' 000'
    
    # Plot
    cmap = plt.get_cmap(cmap_colormaps)
    planet_types_arr_idx          = np.arange(len(planet_types_arr))
    planet_types_arr_idx_extended = np.arange(-1, len(planet_types_arr)+1)
    planet_types_arr_extended     = np.append(np.append(planet_types_arr[0], planet_types_arr), planet_types_arr[-1])
    bands_idx             = np.arange(len(bands_labels))
    bands_idx_extended    = np.arange(-1, len(bands_labels)+1)
    bands_labels_extended = np.append(np.append(bands_labels[0], bands_labels), bands_labels[-1])
    data          = 100 * SNR
    data_extended = np.vstack([data[0], data, data[-1]])
    data_extended = np.hstack([data_extended[:, [0]], data_extended, data_extended[:, [-1]]])
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.xlabel("Various planetary types", fontsize=16, weight='bold')
    plt.ylabel("Various spectral modes", fontsize=16, weight='bold')
    plt.title(f"{instru} S/N fluctuations\nin {spectrum_contributions} light ({name_model}-model)", fontsize=18, pad=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Heatmap with pcolormesh
    if instru == "ANDES" and len(config_data["gratings"]) > 3:
        mesh = plt.pcolormesh(planet_types_arr_idx_extended, bands_idx_extended, data_extended, cmap=cmap, shading='auto', vmin=np.nanmin(data), vmax=100) 
    else:
        mesh = plt.pcolormesh(planet_types_arr_idx_extended, bands_idx_extended, data_extended, cmap=cmap, shading='auto', vmin=0, vmax=100) 
    
    # Contours
    planet_types_arr_mesh, bands_mesh = np.meshgrid(planet_types_arr_idx_extended, bands_idx_extended, indexing='xy')
    if instru == "ANDES" and len(config_data["gratings"]) > 3:
        cs = plt.contour(planet_types_arr_mesh, bands_mesh, data_extended, colors='k', linewidths=0.5, alpha=0.7)
    else:  
        cs = plt.contour(planet_types_arr_mesh, bands_mesh, data_extended, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")

    # Colorbar
    ax = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.05, shrink=1)
    cbar.minorticks_on()
    if instru != "ANDES" and len(config_data["gratings"]) > 3:
        cbar.set_ticks(contour_levels)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('$GAIN_{S/N}$ [%]', rotation=270, labelpad=14, fontsize=14)
    cbar.ax.text(1.2, 1.05,  'High signal', ha='center', va='bottom', fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='red')
    cbar.ax.text(1.2, -0.05, 'Poor signal', ha='center', va='top',    fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='blue')

    plt.xlim(-0.5, len(planet_types_arr)-0.5)
    plt.xticks(planet_types_arr_idx, planet_types_arr, rotation=45, ha="right")
    plt.ylim(-0.5, len(bands_labels)-0.5)
    plt.yticks(bands_idx, bands_labels)
    plt.tight_layout()
    filename = f"colormaps_bands_planets_snr/Colormap_bands_planets_SNR_{filename}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight')
    plt.show()



def colormap_bands_planets_parameters(mode="multi", instru="HARMONI", thermal_model="BT-Settl", reflected_model="PICASO", exposure_time = 120, apodizer="NO_SP", strehl="JQ1", coronagraph=None, Nmax=10, systematic=False, PCA=False, PCA_mask=False, Nc=20, Rc=100, filter_type="gaussian"):

    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
    
    # get spectrum contribution + name model
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Load the planet table and convert it from QTable to pandas DataFrame
    table           = "Archive"
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename            = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}"
    planet_table        = load_planet_table(f"{filename}.ecsv")    
    planet_table["SNR"] = SNR_from_table(table=planet_table, exposure_time=exposure_time, band="INSTRU")
    
    # Find planets based on mode
    planet_table_pd  = planet_table.to_pandas() # convert to pandas to find matching types
    selected_planets = set()                    # Set to store already assigned planets for mode = "unique"
    matching_planets = {planet_type: find_matching_planets(criteria, planet_table_pd, mode, selected_planets, Nmax=Nmax) for planet_type, criteria in planet_types.items()}

    # Table plot
    plot_matching_planets(matching_planets, exposure_time, mode)

    # Definig arrays
    planet_types_arr           = np.array([])
    list_planets               = []
    planet_table["PlanetType"] = np.array(planet_table["PlanetType"], dtype="<U32")

    # Raw wavelength axis
    lmin_model = 0.98*config_data["lambda_range"]["lambda_min"]
    lmax_model = 1.02*config_data["lambda_range"]["lambda_max"]
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)
    
    # Saving all planet entries, valid types and bands
    for ptype, planets in matching_planets.items():
        if len(planets) > 0:
            planet_types_arr = np.append(planet_types_arr, ptype)
            for planet in planets:
                planet_raw = copy.deepcopy(planet_table[get_planet_index(planet_table, planet["PlanetName"])])
                planet_raw["PlanetType"] = ptype
                list_planets.append(planet_raw)
    bands_labels = np.full(0, "", dtype=object)
    for nb, band in enumerate(config_data["gratings"]):
        if instru=="ANDES":
            if "10mas" in band:
                bands_labels = np.append(bands_labels, band)
        else:
            bands_labels = np.append(bands_labels, band)
            
    # Calculating uncertainties
    sigma_T     = np.zeros((len(bands_labels), len(planet_types_arr)), dtype=float)
    sigma_lg    = np.zeros((len(bands_labels), len(planet_types_arr)), dtype=float)
    sigma_vsini = np.zeros((len(bands_labels), len(planet_types_arr)), dtype=float)
    sigma_rv    = np.zeros((len(bands_labels), len(planet_types_arr)), dtype=float)   
    for planet in tqdm(list_planets, total=len(list_planets), desc="Processing planets", unit="planet"):
        
        itype = np.where(planet["PlanetType"]==planet_types_arr)[0][0]
        
        planet_spectrum, _, _, star_spectrum = thermal_reflected_spectrum(planet, instru, thermal_model=thermal_model, reflected_model=reflected_model, wave_instru=wave_model, wave_K=None, vega_spectrum_K=None, show=False)
        mag_p                                = float(planet[f"PlanetINSTRUmag({instru})({spectrum_contributions})"])
        mag_s                                = float(planet[f"StarINSTRUmag({instru})"])
        planet_spectrum.model                = thermal_model
        name_bands, _, uncertainties         = FastCurves(instru=instru, calculation="corner plot", systematic=systematic, T_planet=float(planet["PlanetTeq"].value), lg_planet=float(planet["PlanetLogg"].value), mag_star=mag_s, band0="instru", T_star=float(planet["StarTeff"].value), lg_star=float(planet["StarLogg"].value), exposure_time=exposure_time, model_planet=thermal_model, mag_planet=mag_p, separation_planet=float(planet["AngSep"].value/1000), planet_name="None", return_SNR_planet=False, show_plot=False, verbose=False, planet_spectrum=planet_spectrum.copy(), star_spectrum=star_spectrum.copy(), apodizer=apodizer, strehl=strehl, PCA=PCA, PCA_mask=PCA_mask, Nc=Nc)
        
        sigma_T_1D     = np.zeros((len(bands_labels)))
        sigma_lg_1D    = np.zeros((len(bands_labels)))
        sigma_vsini_1D = np.zeros((len(bands_labels)))
        sigma_rv_1D    = np.zeros((len(bands_labels)))
        
        for iband, band in enumerate(bands_labels):
            sigma_T_1D[iband]     = uncertainties[name_bands.index(band)][0]
            sigma_lg_1D[iband]    = uncertainties[name_bands.index(band)][1]
            sigma_vsini_1D[iband] = uncertainties[name_bands.index(band)][2]
            sigma_rv_1D[iband]    = uncertainties[name_bands.index(band)][3]
                    
        if np.nanmax(sigma_T_1D)!=0:
            sigma_T_1D     /= np.nanmax(sigma_T_1D)
        if np.nanmax(sigma_lg_1D)!=0:
            sigma_lg_1D    /= np.nanmax(sigma_lg_1D)
        if np.nanmax(sigma_vsini_1D)!=0:
            sigma_vsini_1D /= np.nanmax(sigma_vsini_1D)
        if np.nanmax(sigma_rv_1D)!=0:
            sigma_rv_1D    /= np.nanmax(sigma_rv_1D)
        
        sigma_T[:, itype]     += sigma_T_1D     / len(matching_planets[planet_types_arr[itype]])
        sigma_lg[:, itype]    += sigma_lg_1D    / len(matching_planets[planet_types_arr[itype]])
        sigma_vsini[:, itype] += sigma_vsini_1D / len(matching_planets[planet_types_arr[itype]])
        sigma_rv[:, itype]    += sigma_rv_1D    / len(matching_planets[planet_types_arr[itype]])

    for itype in range(len(planet_types_arr)):
        sigma_T[:, itype][sigma_T[:, itype]<0]         = np.nanmin(sigma_T[:, itype][sigma_T[:, itype]>0])
        sigma_lg[:, itype][sigma_lg[:, itype]<0]       = np.nanmin(sigma_lg[:, itype][sigma_lg[:, itype]>0])
        sigma_vsini[:, itype][sigma_vsini[:, itype]<0] = np.nanmin(sigma_vsini[:, itype][sigma_vsini[:, itype]>0])
        sigma_rv[:, itype][sigma_rv[:, itype]<0]       = np.nanmin(sigma_rv[:, itype][sigma_rv[:, itype]>0])

    gain_sigma_T     = np.copy(sigma_T)
    gain_sigma_lg    = np.copy(sigma_lg)
    gain_sigma_vsini = np.copy(sigma_vsini)
    gain_sigma_rv    = np.copy(sigma_rv)
    
    for itype in range(len(planet_types_arr)):
        gain_sigma_T[:, itype]     = np.nanmin(sigma_T[:, itype])     / sigma_T[:, itype]
        gain_sigma_lg[:, itype]    = np.nanmin(sigma_lg[:, itype])    / sigma_lg[:, itype]
        gain_sigma_vsini[:, itype] = np.nanmin(sigma_vsini[:, itype]) / sigma_vsini[:, itype]
        gain_sigma_rv[:, itype]    = np.nanmin(sigma_rv[:, itype])    / sigma_rv[:, itype]
    
    # Plot
    cmap = plt.get_cmap(cmap_colormaps)
    planet_types_arr_idx          = np.arange(len(planet_types_arr))
    planet_types_arr_idx_extended = np.arange(-1, len(planet_types_arr)+1)
    planet_types_arr_extended     = np.append(np.append(planet_types_arr[0], planet_types_arr), planet_types_arr[-1])
    bands_idx             = np.arange(len(bands_labels))
    bands_idx_extended    = np.arange(-1, len(bands_labels)+1)
    bands_labels_extended = np.append(np.append(bands_labels[0], bands_labels), bands_labels[-1])

    # Création de la figure avec subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300, sharex=True, sharey=True)
    axes      = axes.flatten()
    
    # Définition des données et labels
    data_list = [gain_sigma_T, gain_sigma_lg, gain_sigma_vsini, gain_sigma_rv]
    titles    = [r'$T_{\rm eff}$', r'$\log g$', r'$V \sin i$', r'$RV$']
    cmap      = plt.get_cmap(cmap_colormaps)
    
    # Boucle sur les 4 subplots
    for i, ax in enumerate(axes):
        data          = 100 * data_list[i]
        data_extended = np.pad(data, pad_width=((1, 1), (1, 1)), mode='edge')  # Ajout d'une bordure pour éviter le chevauchement
        
        # Heatmap with pcolormesh
        mesh = ax.pcolormesh(planet_types_arr_idx_extended, bands_idx_extended, data_extended, cmap=cmap, shading='auto', vmin=0, vmax=100)
    
        # Contours 
        planet_types_arr_mesh, bands_mesh = np.meshgrid(planet_types_arr_idx_extended, bands_idx_extended, indexing='xy')
        contours                          = ax.contour(planet_types_arr_mesh, bands_mesh, data_extended, levels=contour_levels, coloes='k', linewidths=1., alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=10, fmt="%d%%")  # Augmenté pour plus de lisibilité
        
        # Axes et labels améliorés
        ax.set_xlim(-0.5, len(planet_types_arr)-0.5)
        ax.set_xticks(planet_types_arr_idx)
        ax.set_xticklabels(planet_types_arr, rotation=40, ha="right", fontsize=18)  # Augmentation de la taille
    
        ax.set_ylim(-0.5, len(bands_labels)-0.5)
        ax.set_yticks(bands_idx)
        ax.set_yticklabels(bands_labels, fontsize=18)  # Augmentation de la taille
            
        # Titres des subplots
        ax.set_title(titles[i], pad=14, fontsize=24)
    
    # Ajustement des marges et disposition (plus d'espace pour la colorbar)
    plt.subplots_adjust(left=0.08, right=0.83, top=0.92, bottom=0.12, hspace=0.15, wspace=0.1)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # Alignement optimisé
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='vertical')
    cbar.minorticks_on()
    cbar.set_ticks(contour_levels)
    cbar.set_label(r'$GAIN_{\sigma}$ [%]', rotation=270, labelpad=25, fontsize=28)
    cbar.ax.tick_params(labelsize=18)    
    cbar.ax.text(1.2, 1.05, 'High precision', ha='center', va='bottom', fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='red')
    cbar.ax.text(1.2, -0.05, 'Poor precision', ha='center', va='top', fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='blue')
    
    fig.suptitle(f"Error fluctuations for {instru} ({'with' if tellurics else 'without'} tellurics absorption)\nin {spectrum_contributions} light with {thermal_model}+{reflected_model} models", fontsize=28, y=1.05)  # Position du titre remontée    
    filename = f"colormaps_bands_planets_snr/Colormap_bands_planets_SNR_{filename}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight')
    plt.show()



######################## Star RV vs Delta RV ###########################################################################################################################################################################################



def colormap_rv(instru="HARMONI", band="H", T_planet=300, T_star=6000, lg_planet=3.0, lg_star=4.44, vsini_planet=3, vsini_star=7,  spectrum_contributions="thermal", model="BT-Settl", airmass=2.5, Rc=100, filter_type="gaussian", photon_noise_limited=True):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
            
    # Raw and bandwidth wavelength axis
    R_band     = config_data['gratings'][band].R
    lmin_band  = config_data['gratings'][band].lmin
    lmax_band  = config_data['gratings'][band].lmax
    dl_band    = (lmin_band+lmax_band)/2 / (2*R_band)
    wave_band  = np.arange(lmin_band, lmax_band, dl_band)
    lmin_model = 0.98*lmin_band
    lmax_model = 1.02*lmax_band
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)
    
    # Getting star spectrum
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave_model, renorm=False) 
    star = star.broad(vsini_star) # Broadening the spectrum
    
    # Getting planet spectrum
    if spectrum_contributions=="reflected" :
        albedo = load_albedo(T_planet, lg_planet, model=model, airmass=airmass)
        albedo = albedo.interpolate_wavelength(wave_model, renorm=False)
        planet = Spectrum(wave_model, albedo.flux*star.flux)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave_model, renorm=False)
    planet = planet.broad(vsini_planet)     # Broadening the spectrum
    
    # To be homogenous to photons
    star.flux   *= wave_model
    planet.flux *= wave_model
        
    # Geting trans model (with tellurics or not)
    trans = get_transmission(instru, wave_band, band, tellurics, apodizer="NO_SP")
    
    # Defining arrays
    N           = 100
    rv_star     = np.linspace(-100, 100, N)
    delta_rv    = np.linspace(-100, 100, N)
    SNR         = np.zeros((len(rv_star), len(delta_rv)))
    lost_signal = np.zeros((len(rv_star), len(delta_rv)))
    
    # Parallel calculations
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_rv, [(i, rv_star, delta_rv, star, planet, trans, wave_band, R_band, Rc, filter_type, photon_noise_limited) for i in range(len(rv_star))]), total=len(rv_star)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    SNR /= np.nanmax(SNR)
        
    # Plots
    for plot in plots:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel(r"Projected orbital velocity [km/s]", fontsize=16)
        plt.xlabel(r"$\Delta$RV [km/s]", fontsize=16)
        plt.ylabel("Star RV [km/s]",                     fontsize=16)
        plt.xlim(delta_rv[0], delta_rv[-1])
        plt.ylim(rv_star[0], rv_star[-1])
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        
        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
        else:
            data = 100 * lost_signal
            cmap = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
        vmin = 0
        vmax = 100
                
        # Heatmap with pcolormesh
        mesh = plt.pcolormesh(delta_rv, rv_star, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        # Contours
        cs = plt.contour(delta_rv, rv_star, data, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        #cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=16, fontsize=16)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar photon noise regime" if photon_noise_limited else "detector noise regime"
        title_text   = (f"{instru} {band}-band {'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, $T_p$={T_planet}K")
        #plt.title(title_text, fontsize=16, pad=14)

        plt.tight_layout()
        filename = f"colormaps_rv/Colormap_rv_{plot}_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_broad{vsini_planet}kms_{noise_regime.replace(' ', '_')}"
        plt.savefig(save_path_colormap + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
    
    return delta_rv, rv_star, SNR, lost_signal

def process_colormap_rv(args):
    i, rv_star, delta_rv, star, planet, trans, wave_band, R_band, Rc, filter_type, photon_noise_limited = args
    SNR_1D         = np.zeros((len(delta_rv)))
    lost_signal_1D = np.zeros((len(delta_rv)))
    # Preparing the star spectrum
    star_shift       = star.doppler_shift(rv_star[i])
    star_shift       = star_shift.degrade_resolution(wave_band, renorm=True).flux
    star_HF, star_BF = filtered_flux(star_shift, R=R_band, Rc=Rc, filter_type=filter_type)
    for j in range(len(delta_rv)):
        # Preparing the planet spectrum
        planet_shift         = planet.doppler_shift(rv_star[i] + delta_rv[j])
        planet_shift         = planet_shift.degrade_resolution(wave_band, renorm=True).flux
        planet_HF, planet_BF = filtered_flux(planet_shift, R=R_band, Rc=Rc, filter_type=filter_type)
        # S/N and signal loss calculations
        template  = trans*planet_HF 
        template /= np.sqrt(np.nansum(template**2))
        alpha     = np.nansum(trans*planet_HF * template)
        beta      = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_shift * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    return i, SNR_1D, lost_signal_1D



######################## Star Vsini vs Planet Vsini ###########################################################################################################################################################################################



def colormap_vsini(instru="HARMONI", band="H", T_planet=300, T_star=6000, lg_planet=3.0, lg_star=4.44, delta_rv=30,  spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True):
    """
    https://www.aanda.org/articles/aa/pdf/2022/03/aa42314-21.pdf
    """
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
            
    # Raw and bandwidth wavelength axis
    R_band     = config_data['gratings'][band].R
    lmin_band  = config_data['gratings'][band].lmin
    lmax_band  = config_data['gratings'][band].lmax
    dl_band    = (lmin_band+lmax_band)/2 / (2*R_band)
    wave_band  = np.arange(lmin_band, lmax_band, dl_band)
    lmin_model = 0.9*lmin_band
    lmax_model = 1.1*lmax_band
    dl_model   = (lmin_model+lmax_model)/2 / (2*R_model)
    wave_model = np.arange(lmin_model, lmax_model, dl_model)
    
    # Getting star spectrum
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave_model, renorm=False)
    
    # Getting planet spectrum
    if spectrum_contributions=="reflected" :
        albedo = load_albedo(T_planet, lg_planet, model=model, airmass=2.5)
        albedo = albedo.interpolate_wavelength(wave_model, renorm=False).flux
        planet = None
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet = planet.interpolate_wavelength(wave_model, renorm=False)
        planet = planet.doppler_shift(delta_rv)
        # To be homogenous to photons
        planet.flux *= wave_model
        albedo = None 
        
    # Geting trans model (with tellurics or not)
    trans = get_transmission(instru, wave_band, band, tellurics, apodizer="NO_SP")
    
    # Defining arrays
    N            = 100
    vsini_star   = np.linspace(0, 100, N)
    vsini_planet = np.linspace(0, 100, N)
    SNR          = np.zeros((len(vsini_star), len(vsini_planet)))
    lost_signal  = np.zeros((len(vsini_star), len(vsini_planet)))
        
    # Parallel calculations
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_vsini, [(i, vsini_star, vsini_planet, star, planet, albedo, trans, wave_band, delta_rv, R_band, Rc, filter_type, photon_noise_limited) for i in range(len(vsini_star))]), total=len(vsini_star)))
        for (i, SNR_1D, lost_signal_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    if np.nanmin(SNR)<0:
        SNR += np.abs(np.nanmin(SNR))
    SNR /= np.nanmax(SNR)
    
    # Plots
    for plot in plots:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel("Planet rotationnal velocity [km/s]", fontsize=14)
        plt.ylabel("Star rotationnal velocity [km/s]", fontsize=14)
        plt.xlim(vsini_planet[0], vsini_planet[-1])
        plt.ylim(vsini_star[0], vsini_star[-1])
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        
        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
            vmin = np.nanmin(data)
            vmax = 100
        else:
            data = 100 * lost_signal
            cmap = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
                
        # Heatmap with pcolormesh
        mesh = plt.pcolormesh(vsini_planet, vsini_star, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        # Contours
        cs = plt.contour(vsini_planet, vsini_star, data, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax   = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        #cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar photon noise regime" if photon_noise_limited else "detector noise regime"
        title_text   = (f"{instru} {band}-band {'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, $T_p$={T_planet}K, "r"$\Delta$rv="f"{delta_rv}km/s")
        plt.title(title_text, fontsize=16, pad=14)

        plt.tight_layout()
        filename = f"colormaps_vsini/Colormap_vsini_{plot}_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_drv{delta_rv}kms_{noise_regime.replace(' ', '_')}"
        plt.savefig(save_path_colormap + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
    
    return vsini_planet, vsini_star, SNR, lost_signal

def process_colormap_vsini(args):
    i, vsini_star, vsini_planet, star, planet, albedo, trans, wave_band, delta_rv, R_band, Rc, filter_type, photon_noise_limited = args
    SNR_1D         = np.zeros((len(vsini_planet)))
    lost_signal_1D = np.zeros((len(vsini_planet)))    
    # Preparing star spectrum (and planet if necessary)
    star_broad = star.broad(vsini_star[i])
    if planet is None:
        planet = Spectrum(star.wavelength, albedo*star_broad.flux)
        planet = planet.doppler_shift(delta_rv)
    # To be homogenous to photons
    star_broad.flux *= star_broad.wavelength
    star_broad       = star_broad.degrade_resolution(wave_band, renorm=True).flux
    star_HF, star_BF = filtered_flux(star_broad, R=R_band, Rc=Rc, filter_type=filter_type)

    for j in range(len(vsini_planet)):
        # Preparing the planet spectrum
        planet_broad         = planet.broad(vsini_planet[j])
        # To be homogenous to photons
        planet_broad.flux   *= planet.wavelength
        planet_broad         = planet_broad.degrade_resolution(wave_band, renorm=True).flux
        planet_HF, planet_BF = filtered_flux(planet_broad, R=R_band, Rc=Rc, filter_type=filter_type)
        # S/N and signal loss calculations
        template  = trans*planet_HF 
        template /= np.sqrt(np.nansum(template**2))
        alpha     = np.nansum(trans*planet_HF * template)
        beta      = np.nansum(trans*star_HF*planet_BF/star_BF * template)
        if photon_noise_limited:
            noise = np.sqrt(np.nansum(trans*star_broad * template**2)) # stellar halo photon noise
        else:
            noise = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / noise
        lost_signal_1D[j] = beta / alpha
    return i, SNR_1D, lost_signal_1D



############################################################################################################################################################################################################################################"



def colormap_maxsep_phase(instru="HARMONI", band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None, inc=90):

    config_data = get_config_data(instru)
    sep_unit    = config_data["sep_unit"]
    
    iwa, _ = get_wa(config_data=config_data, band=band, apodizer=apodizer, sep_unit=sep_unit)
    
    PSF_profile, fraction_PSF, separation, pxscale = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, instru=instru, config_data=config_data, sep_unit=sep_unit, sampling=100 if instru=="ANDES" else 10, OWA=60 if instru=="ANDES" else None)
    
    if coronagraph is None:
        fraction_PSF = np.zeros(len(separation)) + fraction_PSF
    else:
        data                               = fits.getdata(f"sim_data/PSF/PSF_{instru}/fraction_PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits") # flux fraction at the PSF peak as a function of separation 
        f                                  = interp1d(data[0], data[1], bounds_error=False, fill_value=np.nan)
        f_interp                           = f(separation)
        f_interp[separation > data[0][-1]] = data[1][-1] # flat extrapolation
        fraction_PSF                       = f_interp

    # Phase function
    N     = 1000
    phase = np.linspace(0, np.pi, N)
    a     = np.arccos(- np.sin(inc*np.pi/180) * np.cos(phase) ) 
    g_a   = ( np.sin(a) + (np.pi - a) * np.cos(a) ) / np.pi # phase function
    
    # Defining arrays
    maxsep = np.copy(separation)
    signal = np.zeros((len(separation))) + np.nan
    
    # Calculating basic metrics
    sigma_photon      = np.sqrt(PSF_profile)
    sigma_syst        = PSF_profile
    signal[maxsep!=0] = fraction_PSF[maxsep!=0] / maxsep[maxsep!=0]**2
    SNR_photon        = signal / sigma_photon
    SNR_syst          = signal / sigma_syst
    
    # Masking
    mask = maxsep > iwa
    maxsep       = maxsep[mask]
    sigma_photon = sigma_photon[mask]
    sigma_syst   = sigma_syst[mask]
    signal       = signal[mask]
    SNR_photon   = SNR_photon[mask]
    SNR_syst     = SNR_syst[mask]
    
    # Normalizing
    sigma_photon /= np.nanmax(sigma_photon)
    sigma_syst   /= np.nanmax(sigma_syst)
    signal       /= np.nanmax(signal)
    SNR_photon   /= np.nanmax(SNR_photon)
    SNR_syst     /= np.nanmax(SNR_syst)
    
    # Metrics dependancy
    plt.figure(figsize=(10, 6), dpi=300)
    plt.yscale('log')
    plt.xlabel(f"Maximum elongation [{sep_unit}]", fontsize=14)
    plt.ylabel(r"$GAIN$ [%]", fontsize=14)
    plt.xlim(maxsep[0], maxsep[-1])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    plt.plot(maxsep, 100*sigma_photon, c='crimson',   ls='-',  label=r"$\sigma_{\gamma} \propto \sqrt{PSF}$")
    plt.plot(maxsep, 100*sigma_syst,   c='crimson',   ls='--', label=r"$\sigma_{syst} \propto PSF$")
    plt.plot(maxsep, 100*signal,       c='seagreen',  ls='-',  label=r"$signal \propto 1/R^2$ (scaling factor)")
    plt.plot(maxsep, 100*SNR_photon,   c='steelblue', ls='-',  label=r"$S/N$ ($\sigma_{\gamma}$ domination)")
    plt.plot(maxsep, 100*SNR_syst,     c='steelblue', ls='--', label=r"$S/N$ ($\sigma_{syst}$ domination)")
    plt.title(f"{instru} {band}-band reflected light planet fluctuations at phase="r"$\pi/2$"f"\nwith {apodizer.replace('_', ' ')} apodizer and {strehl.replace('_', ' ')} strehl", fontsize=16, pad=14)
    plt.legend(fontsize=14, loc="lower left", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=0.7)    
    plt.show()
    
    # Defining arrays
    SNR_photon = np.zeros((len(phase), len(maxsep)))
    SNR_syst   = np.zeros((len(phase), len(maxsep)))
    
    # Parallel calculations
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_maxsep_phase, [(i, maxsep, phase, inc, PSF_profile, fraction_PSF, separation, g_a) for i in range(len(phase))]), total=len(phase)))
        for (i, SNR_photon_1D, SNR_syst_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR_photon[i, :] = SNR_photon_1D
            SNR_syst[i, :]   = SNR_syst_1D
    
    # Normalizing
    SNR_photon *= 100/np.nanmax(SNR_photon)
    SNR_syst   *= 100/np.nanmax(SNR_syst)
    # Normalizing each SMA
    for j in range(len(maxsep)):
        SNR_photon[:, j] *= 100/np.nanmax(SNR_photon[:, j])
        SNR_syst[:, j]   *= 100/np.nanmax(SNR_syst[:, j])
    
    # Plots
    cmap       = plt.get_cmap(cmap_colormaps)
    cbar_label = '$GAIN_{S/N}$ [%]'
    vmin       = 0
    vmax       = 100
    
    # SNR_photon plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xlabel(f"Maximum elongation [{sep_unit}]", fontsize=14)
    plt.ylabel("Phase [rad]", fontsize=14)
    plt.xlim(maxsep[0], maxsep[-1])
    plt.ylim(phase[0], phase[-1])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    mesh = plt.pcolormesh(maxsep, phase, SNR_photon, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cs   = plt.contour(maxsep, phase, SNR_photon, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
    ax   = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
    cbar.minorticks_on()
    cbar.set_ticks(contour_levels)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
    plt.title(f"{instru} {band}-band S/N fluctuations for reflected light planet at inc={inc}°\nwith {apodizer.replace('_', ' ')} apodizer and {strehl.replace('_', ' ')} strehl in stellar photon noise regime", fontsize=16, pad=14)
    plt.tight_layout()
    filename = f"colormaps_maxsep_phase/Colormap_maxsep_phase_SNR_photon_{instru}_{band}_reflected_inc{inc}_{strehl}_{apodizer}_{coronagraph}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight')
    plt.show()
    
    # SNR_syst plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xlabel(f"Maximum elongation [{sep_unit}]", fontsize=14)
    plt.ylabel("Phase [rad]", fontsize=14)
    plt.xlim(maxsep[0], maxsep[-1])
    plt.ylim(phase[0], phase[-1])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    mesh = plt.pcolormesh(maxsep, phase, SNR_syst, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cs   = plt.contour(maxsep, phase, SNR_syst, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
    ax   = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
    cbar.minorticks_on()
    cbar.set_ticks(contour_levels)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
    plt.title(f"{instru} {band}-band S/N fluctuations for reflected light planet at inc={inc}°\nwith {apodizer.replace('_', ' ')} apodizer and {strehl.replace('_', ' ')} strehl in systematic noise regime", fontsize=16, pad=14)
    plt.tight_layout()
    filename = f"colormaps_maxsep_phase/Colormap_maxsep_phase_SNR_syst_{instru}_{band}_reflected_inc{inc}_{strehl}_{apodizer}_{coronagraph}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight')
    plt.show()
    
    return maxsep, phase, SNR_photon, SNR_syst

def process_colormap_maxsep_phase(args):
    i, maxsep, phase, inc, PSF_profile, fraction_PSF, separation, g_a = args
    SNR_photon_1D = np.zeros((len(maxsep)))
    SNR_syst_1D   = np.zeros((len(maxsep)))
    for j in range(len(maxsep)):
        sep              = maxsep[j] * np.sqrt( np.sin(phase[i])**2 + np.cos(phase[i])**2 * np.cos(inc*np.pi/180)**2 ) # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/74/pdf
        PSF_sep          = PSF_profile[np.abs(separation-sep).argmin()]
        fraction_PSF_sep = fraction_PSF[np.abs(separation-sep).argmin()]
        SNR_photon_1D[j] = g_a[i]*fraction_PSF_sep/maxsep[j]**2 * 1/np.sqrt(PSF_sep)
        SNR_syst_1D[j]   = g_a[i]*fraction_PSF_sep/maxsep[j]**2 * 1/PSF_sep
    return i, SNR_photon_1D, SNR_syst_1D



def colormap_maxsep_inc(instru="HARMONI", band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None):

    config_data = get_config_data(instru)
    sep_unit    = config_data["sep_unit"]
    
    iwa, _ = get_wa(config_data=config_data, band=band, apodizer=apodizer, sep_unit=sep_unit)
    
    PSF_profile, fraction_PSF, separation, pxscale = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, instru=instru, config_data=config_data, sep_unit=sep_unit, sampling=100 if instru=="ANDES" else 10, OWA=60 if instru=="ANDES" else None)
    
    if coronagraph is None:
        fraction_PSF = np.zeros(len(separation)) + fraction_PSF
    else:
        data                               = fits.getdata(f"sim_data/PSF/PSF_{instru}/fraction_PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits") # flux fraction at the PSF peak as a function of separation 
        f                                  = interp1d(data[0], data[1], bounds_error=False, fill_value=np.nan)
        f_interp                           = f(separation)
        f_interp[separation > data[0][-1]] = data[1][-1] # flat extrapolation
        fraction_PSF                       = f_interp

    # Phase function
    N     = 100
    phase = np.linspace(0, np.pi, N)
    inc   = np.linspace(0, 90, N)
    a     = np.arccos(- np.sin(inc*np.pi/180) * np.cos(phase) ) 
    g_a   = ( np.sin(a) + (np.pi - a) * np.cos(a) ) / np.pi # phase function
    
    # Defining arrays
    maxsep = np.copy(separation)
    signal = np.zeros((len(separation))) + np.nan
    
    # Calculating basic metrics
    sigma_photon      = np.sqrt(PSF_profile)
    sigma_syst        = PSF_profile
    signal[maxsep!=0] = fraction_PSF[maxsep!=0] / maxsep[maxsep!=0]**2
    SNR_photon        = signal / sigma_photon
    SNR_syst          = signal / sigma_syst
    
    # Masking
    mask = maxsep > iwa
    maxsep       = maxsep[mask]
    sigma_photon = sigma_photon[mask]
    sigma_syst   = sigma_syst[mask]
    signal       = signal[mask]
    SNR_photon   = SNR_photon[mask]
    SNR_syst     = SNR_syst[mask]
    
    # Normalizing
    sigma_photon /= np.nanmax(sigma_photon)
    sigma_syst   /= np.nanmax(sigma_syst)
    signal       /= np.nanmax(signal)
    SNR_photon   /= np.nanmax(SNR_photon)
    SNR_syst     /= np.nanmax(SNR_syst)
    
    # Metrics dependancy
    plt.figure(figsize=(10, 6), dpi=300)
    plt.yscale('log')
    plt.xlabel(f"Maximum elongation [{sep_unit}]", fontsize=14)
    plt.ylabel(r"$GAIN$ [%]", fontsize=14)
    plt.xlim(maxsep[0], maxsep[-1])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    plt.plot(maxsep, 100*sigma_photon, c='crimson',   ls='-',  label=r"$\sigma_{\gamma} \propto \sqrt{PSF}$")
    plt.plot(maxsep, 100*sigma_syst,   c='crimson',   ls='--', label=r"$\sigma_{syst} \propto PSF$")
    plt.plot(maxsep, 100*signal,       c='seagreen',  ls='-',  label=r"$signal \propto 1/R^2$ (scaling factor)")
    plt.plot(maxsep, 100*SNR_photon,   c='steelblue', ls='-',  label=r"$S/N$ ($\sigma_{\gamma}$ domination)")
    plt.plot(maxsep, 100*SNR_syst,     c='steelblue', ls='--', label=r"$S/N$ ($\sigma_{syst}$ domination)")
    plt.title(f"{instru} {band}-band reflected light planet fluctuations at phase="r"$\pi/2$"f"\nwith {apodizer.replace('_', ' ')} apodizer and {strehl.replace('_', ' ')} strehl", fontsize=16, pad=14)
    plt.legend(fontsize=14, loc="lower left", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', alpha=0.5, linewidth=0.7)    
    plt.show()
    
    
    # Defining arrays
    optimum_phase_photon = np.zeros((len(inc), len(maxsep)))
    optimum_phase_syst   = np.zeros((len(inc), len(maxsep)))
    
    for k in tqdm(range(len(inc)), desc="Colormap: maxsep VS inc"):
    
        # Defining arrays
        SNR_photon = np.zeros((len(phase), len(maxsep)))
        SNR_syst   = np.zeros((len(phase), len(maxsep)))
        
        # Parallel calculations
        with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            results = list(pool.imap(process_colormap_maxsep_phase, [(i, maxsep, phase, inc[k], PSF_profile, fraction_PSF, separation, g_a) for i in range(len(phase))]))
            for (i, SNR_photon_1D, SNR_syst_1D) in results: # Remplissage des matrices 5D avec les résultats
                SNR_photon[i, :] = SNR_photon_1D
                SNR_syst[i, :]   = SNR_syst_1D
        
        # Finding optimum phase for each maxsep and inc
        for j in range(len(maxsep)):
            optimum_phase_photon[k, j] = phase[SNR_photon[:, j].argmax()]
            optimum_phase_syst[k, j]   = phase[SNR_syst[:, j].argmax()]

    # Plots
    cmap       = plt.get_cmap('inferno')
    cbar_label = 'Optimum phase [rad]'
    vmin       = phase[0]
    vmax       = phase[-1]
    
    # SNR photon plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xlabel(f"Maximum elongation [{sep_unit}]", fontsize=14)
    plt.ylabel("Inclination [°]", fontsize=14)
    plt.xlim(maxsep[0], maxsep[-1])
    plt.ylim(inc[0], inc[-1])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    mesh = plt.pcolormesh(maxsep, inc, optimum_phase_photon, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cs   = plt.contour(maxsep, inc, optimum_phase_photon, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8)
    ax   = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
    cbar.minorticks_on()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
    plt.title(f"{instru} {band}-band optimum phase for reflected light planet\nwith {apodizer.replace('_', ' ')} apodizer and {strehl.replace('_', ' ')} strehl in stellar photon noise regime", fontsize=16, pad=14)
    plt.tight_layout()
    filename = f"colormaps_maxsep_inc/Colormap_maxsep_inc_optimum_phase_photon_{instru}_{band}_reflected_{strehl}_{apodizer}_{coronagraph}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight')
    plt.show()
    
    # SNR syst plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xlabel(f"Maximum elongation [{sep_unit}]", fontsize=14)
    plt.ylabel("Inclination [°]", fontsize=14)
    plt.xlim(maxsep[0], maxsep[-1])
    plt.ylim(inc[0], inc[-1])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    mesh = plt.pcolormesh(maxsep, inc, optimum_phase_syst, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cs   = plt.contour(maxsep, inc, optimum_phase_syst, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8)
    ax   = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
    cbar.minorticks_on()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
    plt.title(f"{instru} {band}-band optimum phase for reflected light planet\nwith {apodizer.replace('_', ' ')} apodizer and {strehl.replace('_', ' ')} strehl in systematic noise regime", fontsize=16, pad=14)
    plt.tight_layout()
    filename = f"colormaps_maxsep_inc/Colormap_maxsep_inc_optimum_phase_syst_{instru}_{band}_reflected_{strehl}_{apodizer}_{coronagraph}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight')
    plt.show()
    
    return maxsep, inc, optimum_phase_photon, optimum_phase_syst

############################################################################################################################################################################################################################################"



def colormap_best_parameters_earth(Npx=10000, T_planet=288, T_star=5800, lg_planet=3.0, lg_star=4.4, delta_rv=30, vsini_planet=0.5, vsini_star=2, SMA=1, planet_radius=1, star_radius=1, distance=1, thermal_model="BT-Settl", reflected_model="tellurics", Rc=100, filter_type="gaussian", photon_noise_limited=True, norm_plot="star"):
        
    d = distance * u.pc # parsec
    SMA = SMA * u.AU # AU
    planet_radius = planet_radius * u.earthRad # earth radius
    star_radius = star_radius * u.solRad # star radius
    g_a = 0.32 # elongation max phase function
    
    D_space = 8 # m
    D_ground = 40 # m
    
    S_space = (D_space/2)**2 * np.pi # m2
    S_ground = (D_ground/2)**2 * np.pi # m2
    
    N = 200
    R = np.logspace(np.log10(1000), np.log10(100000), num=N)
    lmin = 0.6
    lmax = 6 # en µm
    l0_arr = np.linspace(lmin, lmax, N)

    R_model = 2e5
    dl_model = (lmin+lmax)/2 / (2*R_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, dl_model)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(wave, renorm=False)
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
    
    SNR_space_thermal = np.zeros((len(R), len(l0_arr)), dtype=float)
    SNR_space_reflected = np.zeros((len(R), len(l0_arr)), dtype=float)
    SNR_ground_thermal = np.zeros((len(R), len(l0_arr)), dtype=float)
    SNR_ground_reflected = np.zeros((len(R), len(l0_arr)), dtype=float)
    
    with Pool(processes=cpu_count()) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
        results = list(tqdm(pool.imap(process_colormap_best_parameters_earth, [(i, R, lmin, lmax, planet_thermal, planet_reflected, star, sky_trans, l0_arr, Npx, Rc, filter_type, S_space, S_ground, photon_noise_limited) for i in range(len(R))]), total=len(R)))
        for (i, SNR_space_thermal_1D, SNR_space_reflected_1D, SNR_ground_thermal_1D, SNR_ground_reflected_1D) in results: # Remplissage des matrices 5D avec les résultats
            SNR_space_thermal[i, :] = SNR_space_thermal_1D
            SNR_space_reflected[i, :] = SNR_space_reflected_1D
            SNR_ground_thermal[i, :] = SNR_ground_thermal_1D
            SNR_ground_reflected[i, :] = SNR_ground_reflected_1D
    max_SNR = max(np.nanmax(SNR_space_thermal), np.nanmax(SNR_space_reflected), np.nanmax(SNR_ground_thermal), np.nanmax(SNR_ground_reflected))    
    SNR_space_thermal    /= max_SNR
    SNR_space_reflected  /= max_SNR
    SNR_ground_thermal   /= max_SNR
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
                ax.set_xlabel(r"central wavelength range $\lambda_0$ [µm]", fontsize=12)
            if j == 0:
                ax.set_ylabel("instrumental resolution R", fontsize=12)
            ax.set_ylim([R[0], R[-1]])
            ax.set_xlim(lmin, lmax)
            contour = ax.contour(l0_arr, R, 100 * SNR / np.nanmax(SNR), linewidths=0.333, colors='k')
            pcm = ax.pcolormesh(l0_arr, R, 100 * SNR / np.nanmax(SNR), cmap='rainbow', vmin=0, vmax=100)
            ax.plot(l0_arr[idx_max_snr[1]], R[idx_max_snr[0]], 'kX', label=r"max for $\lambda_0$ = "+f"{round(l0_arr[idx_max_snr[1]], 1)}\u00b5m and R = {int(round(R[idx_max_snr[0]], -2))}")
            dl = (lmin + lmax) / 2 / (2 * R[idx_max_snr[0]])
            umin = l0_arr[idx_max_snr[1]] - (Npx / 2) * dl
            umax = l0_arr[idx_max_snr[1]] + (Npx / 2) * dl
            ax.errorbar(l0_arr[idx_max_snr[1]], R[idx_max_snr[0]], xerr=(umax - umin)/2, fmt='X', color='k', linestyle='None', capsize=5)
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
    plt.plot(l0_arr, SNR_space_thermal[idx_max_snr_space_thermal[0], :], "r--", label=f"space/thermal: R = {int(round(R[idx_max_snr_space_thermal[0]], -3))}")
    plt.plot(l0_arr, SNR_space_reflected[idx_max_snr_space_reflected[0], :], "b--", label=f"space/reflected: R = {int(round(R[idx_max_snr_space_reflected[0]], -3))}")
    plt.plot(l0_arr, SNR_ground_thermal[idx_max_snr_ground_thermal[0], :], "r", label=f"ground & thermal: R = {int(round(R[idx_max_snr_ground_thermal[0]], -3))}")
    plt.plot(l0_arr, SNR_ground_reflected[idx_max_snr_ground_reflected[0], :], "b", label=f"ground & reflected for R = {int(round(R[idx_max_snr_ground_reflected[0]], -3))}")
    plt.legend(loc="upper left")
    plt.ylabel("S/N [normalized]") ; plt.xlabel(r"central wavelength range $\lambda_0$ [µm]")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on()
    plt.xlim(lmin, lmax)
    plt.ylim(1e-5, 2)    
    filename = f"colormaps_best_parameters_earth_like/plot_best_parameters_earth_like_Npx_{round(round(Npx, -3))}_Rc_{Rc}_thermal_{thermal_model}_reflected_{reflected_model}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    
    return l0_arr, R, Npx, SNR_space_thermal, SNR_space_reflected, SNR_ground_thermal, SNR_ground_reflected

def process_colormap_best_parameters_earth(args):
    i, R, lmin, lmax, planet_thermal, planet_reflected, star, sky_trans, l0_arr, Npx, Rc, filter_type, S_space, S_ground, photon_noise_limited = args
    res = R[i]
    dl = (lmin+lmax)/2 / (2 * res)
    wav = np.arange(lmin, lmax, dl)
    planet_thermal_R = planet_thermal.degrade_resolution(wav, renorm=True)
    planet_reflected_R = planet_reflected.degrade_resolution(wav, renorm=True)
    star_R = star.degrade_resolution(wav, renorm=True)   
    sky_R = sky_trans.degrade_resolution(wav, renorm=False)
    SNR_space_thermal_1D = np.zeros((len(l0_arr)))
    SNR_space_reflected_1D = np.zeros((len(l0_arr)))
    SNR_ground_thermal_1D = np.zeros((len(l0_arr)))
    SNR_ground_reflected_1D = np.zeros((len(l0_arr)))
    for j, l0 in enumerate(l0_arr):
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
        
        if photon_noise_limited:
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












