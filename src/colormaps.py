from src.spectrum import *

path_file = os.path.dirname(__file__)
save_path_colormap = os.path.join(os.path.dirname(path_file), "plots/colormaps/")
plots = ["SNR","lost_signal"]

# All colormaps assume photon noise limitation of the stellar halo
############################################################################################################################################################################################################################################"



def colormap_bandwidth_resolution(T_planet, T_star, lg_planet=4.0,lg_star=4.0, delta_rv=25, vsini_planet=3, vsini_star=7,
                                  spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", Rc=100, used_filter="gaussian"):
    """
    Creating figure for IFS Trade-off between Bandwidth/Resolution
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param Npx: number of pixels considered to sample a spectrum
    :param tellurics: tellurics absorption considered if True
    :param vsini_planet: rotational vsini_planet (km/s)
    :param save: save the figure if True
    :param show: displays the figures if True
    :param thermal_model: name of the template library ("BT-Settl" or "Exo-REM")
    :param instru: name of the instrument considered
    """
    if instru=="MIRIMRS" or instru=="NIRSpec" or instru=="NIRCam" or instru=="all":
        tellurics = False
    elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS" or instru=="HiRISE":
        tellurics = True
    if instru=="all":
        Npx = 4096
    elif instru=="HiRISE":
        Npx = 50000
    else :
        config_data = get_config_data(instru) ; Npx = 0
        for band in config_data["gratings"] :
            Npx += ( (config_data["gratings"][band].lmax - config_data["gratings"][band].lmin) * 2*config_data["gratings"][band].R/((config_data["gratings"][band].lmax + config_data["gratings"][band].lmin)/2))/len(config_data["gratings"])
        Npx = int(round(Npx,-2))
    R = np.logspace(np.log10(500), np.log10(200000), num=100)
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # en µm
    else :
        lmax = 12 # en µm
    lambda_0 = np.linspace(lmin, lmax, 4*len(R))
    SNR = np.zeros((len(R), len(lambda_0)), dtype=float)
    lost_signal = np.zeros_like(SNR)

    res_model = 1e6
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(star.flux, star.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    if vsini_star != 0:
        star = star.broad(vsini_star)
        
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet,lg_planet)
            albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
        planet = Spectrum(wave,albedo*star.flux,star.R,T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet,lg_planet,model,instru=instru)
        planet = planet.interpolate_wavelength(planet.flux, planet.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    if delta_rv != 0:
        planet = planet.doppler_shift(delta_rv)
    if vsini_planet != 0:
        planet = planet.broad(vsini_planet)
    star.flux *= wave # pour être homogène à des photons
    planet.flux *= wave # pour être homogène à des photons 
    
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        sky_trans = sky_trans.interpolate_wavelength(sky_trans.flux, sky_trans.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    
    for i, res in enumerate(R):
        print(100*(i+1)/len(R),"%")
        dl = (lmin+lmax)/2 / (2 * res)
        wav = np.arange(lmin-(Npx/2)*dl, lmax+(Npx/2)*dl, dl)
        planet_R = planet.degrade_resolution(wav)
        star_R = star.degrade_resolution(wav)   
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
            star_R_crop = Spectrum(wav[valid], star_R.flux[valid], res, T_star)
            planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, T_planet)    
            planet_HF,planet_BF = filtered_flux(planet_R_crop.flux,R=res,Rc=Rc,used_filter=used_filter)
            star_HF,star_BF = filtered_flux(star_R_crop.flux,R=res,Rc=Rc,used_filter=used_filter)
            template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
            alpha = np.sqrt(np.nansum((trans*planet_HF)**2))
            beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
            noise = np.sqrt(np.nansum(trans*star_R_crop.flux * template**2))
            lost_signal[i,j] = beta / alpha
            SNR[i,j] = (alpha-beta) / noise
    SNR /= np.nanmax(SNR)

    for plot in plots:
        plt.figure(dpi=300) ; plt.yscale('log') ; plt.xlabel("central wavelength range $\lambda_0$ (in µm)",fontsize=14) ; plt.ylabel("instrumental resolution $R_{inst}$" , fontsize=14) ; plt.ylim([R[0], R[-1]]) ; plt.xlim(lmin,lmax)
        if plot=="SNR":
            plt.contour(lambda_0, R, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(lambda_0, R, 100*SNR, cmap=plt.plt.get_cmap('rainbow'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ (in %)', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
                elif spectrum_contributions=="reflected" :
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, $T_p$={planet.T}K,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
                    elif model=="tellurics" or model=="flat" :
                        plt.title(f"S/N fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
            else :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
                elif spectrum_contributions=="reflected" :
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo, $T_p$={planet.T}K,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
                    elif model=="tellurics" or model=="flat" :
                        plt.title(f"S/N fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} albedo,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
            filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{planet.T}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        elif plot=="lost_signal" :
            plt.contour(lambda_0, R, 100*lost_signal, linewidths=0.5,colors='k') ; plt.pcolormesh(lambda_0, R, 100*lost_signal, cmap=plt.plt.get_cmap('rainbow_r'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ (in %)', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"Lost signal fluctuations (with tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
            else :
                plt.title(f"Lost signal fluctuations (without tellurics absoprtion)\n in {spectrum_contributions} light with {model} model, $T_p$={planet.T}K,\n $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s and "+r"$N_{\lambda}$="+f"{Npx}",fontsize=14,pad=14)
            filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_lost_signal_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{planet.T}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        if instru=="all":
            markers = ["o","v","s","p","*","d","P","X"]
            for i,instrument in enumerate(config_data_list) :
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
        elif instru=="HiRISE":
            plt.scatter(1.6, 100000, c='black', marker='o', label="HiRISE")
        else :
            x_instru, y_instru, labels = [], [], []
            config_data = get_config_data(instru)
            for band in config_data['gratings']:
                x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
                y_instru.append(config_data['gratings'][band].R)
                if instru=="MIRIMRS" or instru=="HARMONI" :
                    if band != "H_high":
                        labels.append(band[:2])
                    else:
                        labels.append(band)
                else:
                    labels.append(band)
            plt.scatter(x_instru, y_instru, c='black', marker='o', label=instru+' bands')
            for i, l in enumerate(labels):
                plt.annotate(l, (x_instru[i] , 1.2*y_instru[i]))
        plt.legend()
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
        
    return lambda_0, R, SNR, lost_signal



############################################################################################################################################################################################################################################"



def colormap_bandwidth_Tp(instru,T_star,lg_planet=4.0,lg_star=4.0, delta_rv=25, vsini_planet=3, vsini_star=7,
                          spectrum_contributions="thermal", model="BT-Settl",Rc=100,used_filter="gaussian"):    
    if instru=="MIRIMRS" or instru=="NIRSpec" or instru=="NIRCam" :
        tellurics = False
    elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS":
        tellurics = True
    config_data = get_config_data(instru)
    R = 0. ; Npx = 0.
    for band in config_data["gratings"] :
        R += config_data["gratings"][band].R/len(config_data["gratings"])
        Npx += ( (config_data["gratings"][band].lmax - config_data["gratings"][band].lmin) * 2*config_data["gratings"][band].R/((config_data["gratings"][band].lmax + config_data["gratings"][band].lmin)/2))/len(config_data["gratings"])
    R = int(round(R,-2)) ; Npx = int(round(Npx,-2))
    lmin = 0.8
    if spectrum_contributions=="reflected" or tellurics :
        lmax = 6 # en µm
    else :
        lmax = 12 # en µm
    Tp = np.arange(300,3100,100)
    lambda_0 = np.linspace(lmin, lmax, 4*len(Tp))
    SNR = np.zeros((len(Tp), len(lambda_0)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    opti_l0 = np.zeros((len(Tp)))
    
    res_model = 1e6
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star,lg_star)
    star = star.interpolate_wavelength(star.flux, star.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    if vsini_star != 0:
        star = star.broad(vsini_star)
      
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0,:], sky_trans[1,:], None, None)
        sky_trans = sky_trans.interpolate_wavelength(sky_trans.flux, sky_trans.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    
    for i, T_planet in enumerate(Tp):
        print(round(100*(i+1)/len(Tp)),"%")
        if spectrum_contributions=="reflected" :
            if model=="PICASO":
                albedo = load_albedo(T_planet,lg_planet)
                albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, wave, renorm = False).flux
            elif model=="tellurics":
                albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
                f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
                albedo = f(wave)
            elif model=="flat":
                albedo = np.zeros_like(wave) + 1.
            planet = Spectrum(wave,albedo*star.flux,star.R,T_planet)
        elif spectrum_contributions=="thermal" :
            planet = load_planet_spectrum(T_planet,lg_planet,model,instru=instru)
            planet = planet.interpolate_wavelength(planet.flux, planet.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
        if delta_rv != 0:
            planet = planet.doppler_shift(delta_rv)
        if vsini_planet != 0:
            planet = planet.broad(vsini_planet)
        star.flux *= wave # pour être homogène à des photons
        planet.flux *= wave # pour être homogène à des photons 
        planet.flux /= np.nansum(planet.flux) # disabling flux dependency with the temperature
        for j, l0 in enumerate(lambda_0):
            dl = l0/(2*R)
            umin = l0-(Npx/2)*dl
            umax = l0+(Npx/2)*dl
            wav = np.arange(umin, umax, dl)
            star_R = star.degrade_resolution(wav)  
            planet_R = planet.degrade_resolution(wav)
            #planet_R.flux /= np.nansum(planet_R.flux) # disabling flux dependency with the temperature
            if tellurics:
                trans = sky_trans.degrade_resolution(wav, renorm=False).flux
            else:
                trans = 1
            planet_HF,planet_BF = filtered_flux(planet_R.flux,R=R,Rc=Rc,used_filter=used_filter)
            star_HF,star_BF = filtered_flux(star_R.flux,R=R,Rc=Rc,used_filter=used_filter)
            template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
            alpha = np.sqrt(np.nansum((trans*planet_HF)**2))
            beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
            noise = np.sqrt(np.nansum(trans*star_R.flux * template**2))
            lost_signal[i,j] = beta / alpha
            SNR[i,j] = (alpha-beta) / noise
        SNR[i,:] /= np.nanmax(SNR[i,:])
        opti_l0[i] = lambda_0[SNR[i,:].argmax()]
    
    for plot in plots :
        plt.figure(dpi=300) ; plt.xlabel("central wavelength range $\lambda_0$ (in µm)",fontsize=14) ; plt.ylabel("planet's temperature (in K)" , fontsize=14) ; plt.ylim([Tp[0], Tp[-1]]) ; plt.xlim(lmin,lmax)
        if plot=="SNR":
            plt.pcolormesh(lambda_0, Tp, 100*SNR,cmap=plt.plt.get_cmap('rainbow'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ (in %)', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ",fontsize=14,pad=14)
            else :
                plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ",fontsize=14,pad=14)
            filename = f"colormaps_bandwidth_Tp/Colormap_bandwitdth_Tp_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        elif plot=="lost_signal":
            plt.pcolormesh(lambda_0, Tp, 100*lost_signal,cmap=plt.plt.get_cmap('rainbow_r'), vmin=0, vmax=100)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ (in %)', fontsize=14, labelpad=20, rotation=270)
            if tellurics :
                plt.title(f"Lost signal fluctuations for {instru} (with tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ",fontsize=14,pad=14)
            else :
                plt.title(f"Lost signal fluctuations for {instru} (without tellurics absorption)\n in {spectrum_contributions} light with {model} models, $T_*$={T_star}K, $\Delta$$v_r$={delta_rv}km/s\n $R_c$={Rc}, R={R} and $N_\lambda$={Npx} ",fontsize=14,pad=14)
            filename = f"colormaps_bandwidth_Tp/Colormap_bandwitdth_Tp_lost_signal_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Npx{Npx}"
        bands = []
        for nb,band in enumerate(config_data['gratings']):
            x = (config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2
            if instru=="MIRIMRS" or instru=="NIRSpec" :
                plt.plot([x,x],[Tp[0],0.95*np.nanmean(Tp)],"k")
                plt.plot([x,x],[1.1*np.nanmean(Tp),Tp[-1]],"k")
                plt.annotate(band[:2], (x-0.2, np.nanmean(Tp)))
            elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS":
                if band=="HK" or band=="YJH":
                    if band not in bands :
                        plt.plot([x,x],[Tp[0],0.95*np.nanmean(Tp)],"k")
                        plt.plot([x,x],[1.1*np.nanmean(Tp),Tp[-1]],"k")
                        plt.annotate(band, (x-0.15, np.nanmean(Tp)))
                        bands.append(band)
                elif band[0]=="Y" or band[0]=="J" or band[0]=="H"  or band[0]=="K":
                    if band[0] not in bands:
                        plt.plot([x,x],[Tp[0],0.95*np.nanmean(Tp)],"k")
                        plt.plot([x,x],[1.1*np.nanmean(Tp),Tp[-1]],"k")
                        plt.annotate(band[0], (x-0.07, np.nanmean(Tp)))
                        bands.append(band[0])
        plt.plot([],[],"k",label=instru+' bands') ; plt.plot(opti_l0,Tp,'k:',label=r"optimum $\lambda_0$") ; plt.legend()
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    
    return lambda_0, Tp, SNR, lost_signal



############################################################################################################################################################################################################################################"



def colormap_rv(T_planet, T_star, lg_planet=4.0, lg_star=4.0,
                spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI",band="H",Rc=100,used_filter="gaussian"):
    if instru=="MIRIMRS" or instru=="NIRSpec" or instru=="NIRCam" :
        tellurics = False
    elif instru=="ANDES" or instru=="HARMONI" or instru=="ERIS":
        tellurics = True
    config_data=get_config_data(instru)
    lmin = config_data['gratings'][band].lmin # lambda_min de la bande considérée
    lmax = config_data['gratings'][band].lmax # lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    dl = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wav = np.arange(lmin,lmax,dl) # axe de longueur d'onde de la bande considérée
    
    star = load_star_spectrum(T_star, lg_star)
    star.crop(0.9*lmin,1.1*lmax)

    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet,lg_planet)
            albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, star.wavelength, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(star.wavelength)
        elif model=="flat":
            albedo = np.zeros_like(star.wavelength) + 1.
        planet = Spectrum(star.wavelength,albedo*star.flux,star.R,T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet,lg_planet,model,instru=instru)
        
    star.flux *= star.wavelength
    planet.flux *= planet.wavelength
    planet.crop(0.9*lmin,1.1*lmax)
    
    trans = transmission(instru,wav,band,tellurics,apodizer="NO_SP")
    
    star_rv = np.linspace(-100,100,101)
    delta_rv = np.copy(star_rv)
    SNR = np.zeros((len(star_rv), len(delta_rv)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    for i in range(len(star_rv)):
        print(round(100*(i+1)/len(star_rv),1),"%")
        star_shift = star.doppler_shift(star_rv[i])
        star_shift = star_shift.degrade_resolution(wav,renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
        for j in range(len(delta_rv)):
            planet_shift = planet.doppler_shift(star_rv[i] + delta_rv[j])
            planet_shift = planet_shift.degrade_resolution(wav,renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
            planet_HF,planet_BF = filtered_flux(planet_shift.flux,R=R,Rc=Rc,used_filter=used_filter)
            star_HF,star_BF = filtered_flux(star_shift.flux,R=R,Rc=Rc,used_filter=used_filter)
            template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
            alpha = np.sqrt(np.nansum((trans*planet_HF)**2))
            beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
            noise = np.sqrt(np.nansum(trans*star_shift.flux * template**2))
            lost_signal[i,j] = beta / alpha
            SNR[i,j] = (alpha-beta)/noise
    SNR /= np.nanmax(SNR)
    
    for plot in plots :
        plt.figure(dpi=300) ; plt.xlabel("delta radial velocity (in km/s)",fontsize=14) ; plt.ylabel("star radial velocity (in km/s)" , fontsize=14)
        if plot=="SNR":
            plt.contour(delta_rv, star_rv, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(delta_rv, star_rv, 100*SNR,cmap=plt.plt.get_cmap('rainbow'))
            if tellurics :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light with {model} model,\n $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
                elif spectrum_contributions=="reflected":
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light with {model} albedo,\n $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
                    elif model=="tellurics" or model=="flat":
                        plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light with {model} albedo and $T_*$={T_star}K",fontsize=14,pad=14)
            else :
                if spectrum_contributions=="thermal":
                    plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light with {model} model,\n $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
                elif spectrum_contributions=="reflected":
                    if model=="PICASO":
                        plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light with {model} albedo,\n $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
                    elif model=="tellurics" or model=="flat":
                        plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light with {model} albedo and $T_*$={T_star}K",fontsize=14,pad=14)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ (in %)', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_rv/Colormap_rv_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        elif plot=="lost_signal":
            plt.contour(delta_rv, star_rv, 100*lost_signal, linewidths=0.333, colors='k') ; plt.pcolormesh(delta_rv, star_rv, 100*lost_signal,cmap=plt.plt.get_cmap('rainbow_r'))
            if tellurics :
                plt.title(f"Lost signal fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R,-2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14)
            else :
                plt.title(f"Lost signal fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ (in %)', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_rv/Colormap_rv_lost_signal_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
        
    return delta_rv, star_rv, SNR, lost_signal



############################################################################################################################################################################################################################################"



def colormap_vsini(T_planet,T_star,lg_planet=4.0,lg_star=4.0, delta_rv=3,
                   spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI",band="H",Rc=100,used_filter="gaussian"):
    """
    https://www.aanda.org/articles/aa/pdf/2022/03/aa42314-21.pdf
    
    Creating figure for IFS Trade-off between Bandwidth/Resolution
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param Npx: number of pixels considered to sample a spectrum
    :param tellurics: tellurics absorption considered if True
    :param vsini_planet: rotational vsini_planet (km/s)
    :param save: save the figure if True
    :param show: displays the figures if True
    :param thermal_model: name of the template library ("BT-Settl" or "Exo-REM")
    :param instru: name of the instrument considered
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
    wav = np.arange(lmin,lmax,dl) # axe de longueur d'onde de la bande considérée
    
    res_model = 200000
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(lmin, lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(star.flux, star.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star.flux *= wave # pour être homogène à des photons

    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet,lg_planet)
            albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, wave, renorm = False).flux
        elif model=="tellurics":
            albedo = fits.getdata(os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2.5.fits"))
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0)
            albedo = f(wave)
        elif model=="flat":
            albedo = np.zeros_like(wave) + 1.
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet,lg_planet,model,instru=instru)
        planet = planet.interpolate_wavelength(planet.flux, planet.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
        planet.flux *= wave # pour être homogène à des photons     
        if delta_rv != 0:
            planet = planet.doppler_shift(delta_rv)

    trans = transmission(instru,wav,band,tellurics,apodizer="NO_SP")
    vsini_star = np.linspace(0.1,100,101)
    vsini_planet = np.copy(vsini_star)
    SNR = np.zeros((len(vsini_star), len(vsini_planet)), dtype=float)
    lost_signal = np.zeros_like(SNR)
    for i in range(len(vsini_star)):
        print(round(100*(i+1)/len(vsini_star),1),"%")
        star_broad = star.broad(vsini_star[i])
        if spectrum_contributions=="reflected" :
            planet = Spectrum(wave,albedo*star_broad.flux,star.R,T_planet)
            if delta_rv != 0:
                planet = planet.doppler_shift(delta_rv)
        star_broad = star_broad.degrade_resolution(wav,renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
        for j in range(len(vsini_planet)):
            planet_broad = planet.broad(vsini_planet[j])
            planet_broad = planet_broad.degrade_resolution(wav,renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
            planet_HF,planet_BF = filtered_flux(planet_broad.flux,R,Rc,used_filter)
            star_HF,star_BF = filtered_flux(star_broad.flux,R,Rc,used_filter)
            template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
            alpha = np.sqrt(np.nansum((trans*planet_HF)**2))
            beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
            noise = np.sqrt(np.nansum(trans*star_broad.flux * template**2))
            lost_signal[i,j] = beta / alpha
            SNR[i,j] = np.abs(alpha-beta)/noise
    SNR /= np.nanmax(SNR)
    
    for plot in plots:
        plt.figure(dpi=300) ; plt.xlabel("planet Vsini (in km/s)",fontsize=14) ; plt.ylabel("star Vsini (in km/s)" , fontsize=14)
        if plot=="SNR":
            plt.contour(vsini_planet, vsini_star, 100*SNR, linewidths=0.333, colors='k') ; plt.pcolormesh(vsini_planet, vsini_star, 100*SNR,cmap=plt.plt.get_cmap('rainbow'))
            if tellurics :
                plt.title(f"S/N fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R,-2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
            else :
                plt.title(f"S/N fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
            cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ (in %)', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_vsini/Colormap_vsini_SNR_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        elif plot=="lost_signal":
            plt.contour(vsini_planet, vsini_star, 100*lost_signal, linewidths=0.333, colors='k') ; plt.pcolormesh(vsini_planet, vsini_star, 100*lost_signal,cmap=plt.plt.get_cmap('rainbow_r'))
            if tellurics :
                plt.title(f"Lost signal fluctuations for {instru} (with tellurics absorption)\n on {band} band (R={int(round(R,-2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
            else :
                plt.title(f"Lost signal fluctuations for {instru} (without tellurics absorption)\n on {band} band (R={int(round(R,2))}) in {spectrum_contributions} light\n with {model} model, $R_c$={Rc}, $T_p$={T_planet}K and $T_*$={T_star}K",fontsize=14,pad=14)
            cbar = plt.colorbar() ; cbar.set_label(r'lost signal $\beta/\alpha$ (in %)', fontsize=14, labelpad=20, rotation=270)
            filename = f"colormaps_vsini/Colormap_vsini_lost_signal_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K"
        if tellurics:
            filename += "_with_tellurics"
        plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
    
    return vsini_planet, vsini_star, SNR, lost_signal




############################################################################################################################################################################################################################################"



def colormap_maxsep_phase(instru="HARMONI",band="H",inc=90):
    """
    Creating figure for IFS Trade-off between Bandwidth/Resolution
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param Npx: number of pixels considered to sample a spectrum
    :param tellurics: tellurics absorption considered if True
    :param vsini_planet: rotational vsini_planet (km/s)
    :param save: save the figure if True
    :param show: displays the figures if True
    :param thermal_model: name of the template library ("BT-Settl" or "Exo-REM")
    :param instru: name of the instrument considered
    """
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

    PSF_profile , fraction_PSF , separation = PSF_profile_fraction_separation(band=band,strehl=strehl,apodizer="NO_SP",coronagraph=None,instru=instru,config_data=config_data,sep_unit=sep_unit,star_pos="center")
    
    phase = np.linspace(0,np.pi,1000)
    alph = np.arccos(- np.sin(inc*np.pi/180) * np.cos(phase) )
    g_alpha = ( np.sin(alph) + (np.pi - alph) * np.cos(alph) ) / np.pi
    maxsep = np.linspace(iwa,np.nanmax(separation),len(phase))
    SNR = np.zeros((len(phase), len(maxsep)), dtype=float)
    
    f = interp1d(separation,PSF_profile,bounds_error=False,fill_value="extrapolate")
    PSF_profile = f(maxsep)
    separation = maxsep
        
    plt.figure()
    plt.yscale('log')
    plt.plot(separation,1/np.sqrt(PSF_profile)/np.nanmax(1/np.sqrt(PSF_profile)),'r',label="$1/\sqrt{PSF}$")
    plt.plot(separation,1/separation**2/max(1/separation**2),'g',label="$1/R^2$")
    plt.plot(separation,1/(np.sqrt(PSF_profile)*separation**2)/max(1/(np.sqrt(PSF_profile)*separation**2)),'b',label="$1/R^2$ x $1/\sqrt{PSF}$")
    plt.legend()
    plt.show()
    
    for i in range(len(phase)):
        print(round(100*(i+1)/len(phase),1),"%")
        for j in range(len(maxsep)):
            sep = maxsep[j] * np.sin(phase[i])
            sep = maxsep[j] * np.sqrt( np.sin(phase[i])**2 + np.cos(phase[i])**2 * np.cos(inc*np.pi/180)**2 ) # https://iopscience.iop.org/article/10.1088/0004-637X/729/1/74/pdf
            frac_PSF_sep = PSF_profile[np.abs(separation-sep).argmin()]
            SNR[i,j] = g_alpha[i]/maxsep[j]**2 / np.sqrt(frac_PSF_sep)
    SNR /= np.nanmax(SNR)
    
    plt.figure(dpi=300) ; plt.xlabel(f"maximum elongation (in {sep_unit})",fontsize=14) ; plt.ylabel("phase (in rad)" , fontsize=14)
    plt.contour(maxsep, phase, 100*SNR, linewidths=0.333, colors='k')
    plt.pcolormesh(maxsep, phase, 100*SNR,cmap=plt.plt.get_cmap('rainbow'))
    plt.title(f"S/N fluctuations for {instru} on {band}-band\n for planets in reflected light with inc = {int(round(inc))}°",fontsize=14,pad=14)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ (in %)', fontsize=14, labelpad=20, rotation=270)
    filename = f"colormaps_maxsep_phase/Colormap_maxsep_phase_SNR_{instru}_{band}_reflected_inc{inc}"
    plt.savefig(save_path_colormap + filename + ".png", format='png', bbox_inches='tight') ; plt.show()
        
    return maxsep, phase, SNR




############################################################################################################################################################################################################################################"





def colormap_best_parameters_earth(T_planet=300, T_star=5800, lg_planet=3.0, lg_star=4.4, delta_rv=30, vsini_planet=0.5, vsini_star=2, SMA=1, planet_radius=1, star_radius=1, distance=1,
                                  thermal_model="BT-Settl", reflected_model="tellurics", Rc=100, used_filter="gaussian"):
    
    d = distance * u.pc
    SMA = SMA * u.AU
    planet_radius = planet_radius * u.earthRad
    star_radius = star_radius * u.solRad
    g_alpha = 0.32 # elongation max
    
    S_space = (8/2)**2 * np.pi # m
    S_ground = (40/2)**2 * np.pi # m
    
    N = 10
    Npx = 10000
    R = np.logspace(np.log10(1000), np.log10(200000), num=N)
    lmin = 0.6
    lmax = 6 # en µm

    lambda_0 = np.linspace(lmin, lmax, N)
    SNR_space_thermal = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_space_reflected = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_ground_thermal = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_ground_reflected = np.zeros((len(R), len(lambda_0)), dtype=float)

    res_model = 1e6
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    star = load_star_spectrum(T_star, lg_star)
    star = star.interpolate_wavelength(star.flux, star.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    star.flux *= float((star_radius/d).decompose()**2)
    if vsini_star != 0:
        star = star.broad(vsini_star)
        
    planet_thermal = load_planet_spectrum(T_planet,lg_planet,model=thermal_model,instru=instru)
    planet_thermal = planet_thermal.interpolate_wavelength(planet_thermal.flux, planet_thermal.wavelength, wave, renorm = False)
    planet_thermal.flux *= float((planet_radius/d).decompose()**2)

    albedo = load_albedo(planet_thermal.T,planet_thermal.lg)
    albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, wave, renorm = False)
    if reflected_model == "PICASO":
        planet_reflected = star.flux * albedo.flux * g_alpha * (planet_radius_radius/SMA).decompose()**2
    elif reflected_model == "flat":
        planet_reflected = star.flux * np.nanmean(albedo.flux)*1 * g_alpha * (planet_radius_radius/SMA).decompose()**2
    elif reflected_model == "tellurics":
        wave_tell,tell = fits.getdata("sim_data/Transmission/sky_transmission_airmass_2.5.fits")
        f = interp1d(wave_tell,tell,bounds_error=False,fill_value=np.nan)
        tell = f(wave)
        planet_reflected = star.flux * np.nanmean(albedo.flux)/np.nanmean(tell)*tell * g_alpha * (planet_radius/SMA).decompose()**2
    else :
        raise KeyError(reflected_model+" IS NOT A VALID REFLECTED MODEL : tellurics, flat, or PICASO")
    planet_reflected = Spectrum(wave,np.nan_to_num(np.array(planet_reflected.value)),max(star.R,albedo.R),albedo.T,lg_planet,reflected_model)
        
    if delta_rv != 0 :
        planet_thermal = planet_thermal.doppler_shift(delta_rv)
        planet_reflected = planet_reflected.doppler_shift(delta_rv)
    if vsini_planet != 0:
        planet_thermal = planet_thermal.broad(vsini_planet)
        planet_reflected = planet_reflected.broad(vsini_planet)
    planet_thermal.flux *= wave # pour être homogène à des photons
    planet_reflected.flux *= wave # pour être homogène à des photons      
    star.flux *= wave # pour être homogène à des photons

    sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.0.fits")
    sky_trans = fits.getdata(sky_transmission_path)
    sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
    sky_trans = sky_trans.interpolate_wavelength(sky_trans.flux, sky_trans.wavelength, wave, renorm=False) # on réinterpole le flux (en densité (énergie)) sur wave_band
    
    plt.figure()
    plt.plot(wave,planet_thermal.flux/star.flux,'r-',label=f"reflected {reflected_model}")
    plt.plot(wave,planet_reflected.flux/star.flux,'b-',label=f"thermal {thermal_model}")
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("wavelength (in µm)")
    plt.ylabel("flux (in contrast unit)")
    plt.ylim(1e-12,max(np.nanmax(planet_reflected.flux/star.flux),2*np.nanmax(planet_thermal.flux/star.flux)))
    plt.show()
    
    for i, res in enumerate(R):
        print(100*(i+1)/len(R),"%")
        dl = (lmin+lmax)/2 / (2 * res)
        wav = np.arange(lmin, lmax, dl)
        planet_thermal_R = planet_thermal.degrade_resolution(wav)
        planet_reflected_R = planet_reflected.degrade_resolution(wav)
        star_R = star.degrade_resolution(wav)   
        sky_R = sky_trans.degrade_resolution(wav, renorm=False)
        for j, l0 in enumerate(lambda_0):
            umin = l0-(Npx/2)*dl
            umax = l0+(Npx/2)*dl
            valid = np.where(((wav<umax)&(wav>umin)))
            trans = sky_R.flux[valid]

            star_R_crop = Spectrum(wav[valid], star_R.flux[valid], res, T_star)
            planet_thermal_R_crop = Spectrum(wav[valid], planet_thermal_R.flux[valid], res, T_planet)    
            planet_thermal_HF,planet_thermal_BF = filtered_flux(planet_thermal_R_crop.flux,R=res,Rc=Rc,used_filter=used_filter)
            planet_reflected_R_crop = Spectrum(wav[valid], planet_reflected_R.flux[valid], res, T_planet)    
            planet_reflected_HF,planet_reflected_BF = filtered_flux(planet_reflected_R_crop.flux,R=res,Rc=Rc,used_filter=used_filter)
            star_HF,star_BF = filtered_flux(star_R_crop.flux,R=res,Rc=Rc,used_filter=used_filter)
            
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
                            
            SNR_space_thermal[i,j] = np.sqrt(S_space) * (alpha_space_thermal - beta_space_thermal) / np.sqrt(np.nansum(star_R_crop.flux * template_space_thermal**2))
            SNR_space_reflected[i,j] = np.sqrt(S_space) * (alpha_space_reflected - beta_space_reflected) / np.sqrt(np.nansum(star_R_crop.flux * template_space_reflected**2))
            SNR_ground_thermal[i,j] = np.sqrt(S_ground) * (alpha_ground_thermal - beta_ground_thermal) / np.sqrt(np.nansum(trans*star_R_crop.flux * template_ground_thermal**2))
            SNR_ground_reflected[i,j] = np.sqrt(S_ground) * (alpha_ground_reflected - beta_ground_reflected) / np.sqrt(np.nansum(trans*star_R_crop.flux * template_ground_reflected**2))
            
            # plt.figure()
            # plt.title(f"R = {round(res)} / l0 = {round(l0,1)} µm / Npx = {round(npx)}")
            # plt.plot(wav[valid],template_space_thermal,linestyle="-",label="space thermal")
            # plt.plot(wav[valid],template_space_reflected,linestyle=":",label="space reflected")
            # plt.plot(wav[valid],template_ground_thermal,linestyle="-",label="ground thermal")
            # plt.plot(wav[valid],template_ground_reflected,linestyle=":",label="ground reflected")
            # plt.legend()
            # plt.show()

                
    max_SNR = max(np.nanmax(SNR_space_thermal),np.nanmax(SNR_space_reflected),np.nanmax(SNR_ground_thermal),np.nanmax(SNR_ground_reflected))
    SNR_space_thermal /= max_SNR
    SNR_space_reflected /= max_SNR
    SNR_ground_thermal /= max_SNR
    SNR_ground_reflected /= max_SNR
    
    idx_max_snr_space_thermal = np.unravel_index(np.argmax(SNR_space_thermal, axis=None), SNR_space_thermal.shape)
    idx_max_snr_space_reflected = np.unravel_index(np.argmax(SNR_space_reflected, axis=None), SNR_space_reflected.shape)
    idx_max_snr_ground_thermal = np.unravel_index(np.argmax(SNR_ground_thermal, axis=None), SNR_ground_thermal.shape)
    idx_max_snr_ground_reflected = np.unravel_index(np.argmax(SNR_ground_reflected, axis=None), SNR_ground_reflected.shape)
    
    distance = np.logspace(0,4)
    plt.figure()
    plt.plot(distance,1/(distance/d.value)*np.nanmax(SNR_space_thermal),"r",label=f"space & thermal for: R = {round(R[idx_max_snr_space_thermal[0]])} and $\lambda_0$ = {round(lambda_0[idx_max_snr_space_thermal[1]],2)} µm")
    plt.plot(distance,1/(distance/d.value)*np.nanmax(SNR_space_reflected),"b",label=f"space & reflected for: R = {round(R[idx_max_snr_space_reflected[0]])} and $\lambda_0$ = {round(lambda_0[idx_max_snr_space_reflected[1]],2)} µm")
    plt.plot(distance,1/(distance/d.value)*np.nanmax(SNR_ground_thermal),"r--",label=f"ground & thermal for: R = {round(R[idx_max_snr_ground_thermal[0]])} and $\lambda_0$ = {round(lambda_0[idx_max_snr_ground_thermal[1]],2)} µm")
    plt.plot(distance,1/(distance/d.value)*np.nanmax(SNR_ground_reflected),"b--",label=f"ground & reflected for: R = {round(R[idx_max_snr_ground_reflected[0]])} and $\lambda_0$ = {round(lambda_0[idx_max_snr_ground_reflected[1]],2)} µm")
    plt.legend()
    plt.ylabel("S/N") ; plt.xlabel("distance (in pc)")
    plt.yscale('log') ; plt.xscale('log')
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(lambda_0,SNR_space_thermal[idx_max_snr_space_thermal[0],:],"r",label=f"space & thermal for R = {round(R[idx_max_snr_space_thermal[0]])}")
    plt.plot(lambda_0,SNR_space_reflected[idx_max_snr_space_reflected[0],:],"b",label=f"space & reflected for R = {round(R[idx_max_snr_space_reflected[0]])}")
    plt.plot(lambda_0,SNR_ground_thermal[idx_max_snr_ground_thermal[0],:],"r--",label=f"ground & thermal for R = {round(R[idx_max_snr_ground_thermal[0]])}")
    plt.plot(lambda_0,SNR_ground_reflected[idx_max_snr_ground_reflected[0],:],"b--",label=f"ground & reflected for R = {round(R[idx_max_snr_ground_reflected[0]])}")
    plt.legend()
    plt.ylabel("S/N") ; plt.xlabel("wavelength (in µm)")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    return lambda_0, R, Npx, SNR_space_thermal, SNR_space_reflected, SNR_ground_thermal, SNR_ground_reflected



############################################################################################################################################################################################################################################"












