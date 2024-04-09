from src.spectrum import *
from src.config import config_data_list
from src.FastCurves import get_config_data
from src.FastYield import *

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

path_file = os.path.dirname(__file__)
save_path_colormap = os.path.join(os.path.dirname(path_file), "plots/colormaps/")

############################################################################################################################################################################################################################################"

def colormap_tradeoff_band(T_planet,T_star,lg_planet=4.0,lg_star=4.0, step_l0=0.1, nbPixels=3330, tellurics=True,delta_radial_velocity=0,broadening=0,
             spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI",Rc=100,lost_signal=False):
    """
    Creating figure for IFS Trade-off between Bandwidth/Resolution
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param nbPixels: number of pixels considered to sample a spectrum
    :param tellurics: tellurics absorption considered if True
    :param broadening: rotational broadening (km/s)
    :param save: save the figure if True
    :param show: displays the figures if True
    :param thermal_model: name of the template library ("BT-Settl" or "Exo-REM")
    :param instru: name of the instrument considered
    """
    R = np.logspace(np.log10(500), np.log10(200000), num=100)
    if instru=="all":
        lmin = 1
        if spectrum_contributions == "reflected":
            if model == "tellurics":
                lmax = 3
            else :
                lmax = 5 # en µm
        else :
            lmax = 12 # en µm
    else :
        config_data=get_config_data(instru)
        lmin = 0.9*config_data["lambda_range"]["lambda_min"]
        lmax = 1.1*config_data["lambda_range"]["lambda_max"] # en µm
    lambda_0 = np.arange(lmin, lmax, step_l0)
    alpha_2d = np.zeros((lambda_0.shape[0], R.shape[0]), dtype=float)
    noise = np.zeros_like(alpha_2d, dtype=float)
    star = load_star_spectrum(T_star,lg_star)
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet,lg_planet)
            albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, star.wavelength, renorm = False).flux
        elif model=="tellurics":
            albedo_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2_5.fits")
            albedo = fits.getdata(albedo_path)
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
            albedo = f(star.wavelength)
        elif model=="flat":
            albedo = np.zeros_like(star.wavelength) + 1.
        planet = Spectrum(star.wavelength,albedo*star.flux,star.R,T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet,lg_planet,model)
    res_model = 1000000
    delta_lamb = (lmin+lmax)/2 / (2*res_model)
    wave_output = np.arange(0.1, 30, delta_lamb)
    star = star.interpolate_wavelength(star.flux, star.wavelength, wave_output, renorm=False) # on réinterpole le flux (en densité) sur wave_band
    star.flux *= star.wavelength # homogène à des ph
    planet = planet.interpolate_wavelength(planet.flux, planet.wavelength, wave_output, renorm=False) # on réinterpole le flux (en densité) sur wave_band
    planet.flux *= planet.wavelength
    if broadening != 0:
        planet = planet.broad(broadening)
    if delta_radial_velocity != 0:
        planet = planet.doppler_shift(delta_radial_velocity)
    if tellurics :
        sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
    for j, res in enumerate(R):
        print(100*(j+1)/len(R),"%")
        delta_lamb = (lmin+lmax)/2 / (2 * res)
        wav = np.arange(lmin-(nbPixels//2)*delta_lamb, lmax+(nbPixels//2)*delta_lamb, delta_lamb)
        planet_R = planet.degrade_resolution(wav)
        star_R = star.degrade_resolution(wav)   
        if tellurics :
            sky_R = sky_trans.degrade_resolution(wav, renorm=False)
        sigma = max(1,2*res/(np.pi*Rc)*np.sqrt(np.log(2)/2))
        for i, l0 in enumerate(lambda_0):
            umin = l0-(nbPixels//2)*delta_lamb
            umax = l0+(nbPixels//2)*delta_lamb
            valid = np.where(((wav<umax)&(wav>umin)))
            if tellurics:
                trans = sky_R.flux[valid]
            else:
                trans = 1 
            star_R_crop = Spectrum(wav[valid], star_R.flux[valid], res, T_star)
            planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, T_planet)            
            planet_BF = gaussian_filter(planet_R_crop.flux,sigma=sigma)
            planet_HF = planet_R_crop.flux - planet_BF
            star_BF = gaussian_filter(star_R_crop.flux,sigma=sigma)
            star_HF = star_R_crop.flux - star_BF
            template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
            alpha = np.sqrt(np.nansum((trans*planet_HF)**2))
            beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
            noise[i, j] = np.sqrt(np.nansum(star_R_crop.flux * template**2))
            if lost_signal :
                alpha_2d[i, j] = 100*beta/alpha
            else :
                alpha_2d[i, j] = alpha-beta
    xx, yy = np.meshgrid(lambda_0, R)
    
    plt.figure()
    plt.yscale('log')
    if lost_signal :
        z = alpha_2d.transpose()
    else :
        z = (alpha_2d / noise).transpose() / np.nanmax((alpha_2d / noise))
    plt.contour(xx, yy, z, linewidths=0.5,colors='k')
    if lost_signal :
        plt.pcolormesh(xx, yy, z,cmap=plt.get_cmap('rainbow'))
    else :
        plt.pcolormesh(xx, yy, 100*z,cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
    cbar = plt.colorbar()
    if lost_signal :
        cbar.set_label('lost signal (in %)', fontsize=14, labelpad=20, rotation=270)
    else :
        cbar.set_label('$GAIN_{SNR}$ (in %)', fontsize=14, labelpad=20, rotation=270)
    if instru=="all":
        markers = ["o","v","s","p","*","d","P","X"]
        for i,instru in enumerate(config_data_list) :
            instru = instru["name"]
            if spectrum_contributions == "reflected" and instru == "MIRIMRS":
                pass
            else :
                if instru != "NIRCam":
                    x_instru, y_instru = [], []
                    config_data = get_config_data(instru)
                    for band in config_data['gratings']:
                        x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
                        y_instru.append(config_data['gratings'][band].R)
                    plt.scatter(x_instru, y_instru, c='black', marker=markers[i], label=instru)
    else :
        x_instru, y_instru, labels = [], [], []
        config_data = get_config_data(instru)
        for band in config_data['gratings']:
            x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
            y_instru.append(config_data['gratings'][band].R)
            if instru=="MIRIMRS" :
                labels.append(band[:2])
            else:
                labels.append(band)
        plt.scatter(x_instru, y_instru, c='black', marker='o', label=instru + ' modes')
        for i, l in enumerate(labels):
            plt.annotate(l, (x_instru[i] , 1.2*y_instru[i]))
    plt.legend()
    if lost_signal :
        plt.title(f"Lost signal trade-off in {spectrum_contributions} light \n with $T_p$={planet.T}K, $T_*$={T_star}K and $\Delta$$v_r$={delta_radial_velocity}km/s",fontsize=14)
    else :
        plt.title(f"SNR fluctuations trade-off in {spectrum_contributions} light \n with $T_p$={planet.T}K, $T_*$={T_star}K and $\Delta$$v_r$={delta_radial_velocity}km/s",fontsize=14)
    plt.xlabel("central wavelength range $\lambda_0$ (in µm)",fontsize=14)
    plt.ylabel("instrumental resolution $R_{inst}$" , fontsize=14)
    plt.ylim([R[0], R[-1]])
    plt.xlim(lmin,lmax)
    if lost_signal :
        filename = f"Colormap_lost_signal_{spectrum_contributions}_{model}_T{planet.T}K_drv{delta_radial_velocity}kms"
    else :
        filename = f"Colormap_SNR_{spectrum_contributions}_{model}_T{planet.T}K_drv{delta_radial_velocity}kms"
    if tellurics:
        filename += "_with_tellurics"
    if broadening != 0:
        filename += f"_broadening_{broadening}_kms"
    plt.savefig(save_path_colormap + filename + ".png", format='png')
    
    return xx, yy, (alpha_2d / noise).transpose()



############################################################################################################################################################################################################################################"


def colormap_tradeoff_rv(T_planet,T_star,lg_planet=4.0,lg_star=4.0, tellurics=True,
             spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI",band="H",Rc=100):
    """
    Creating figure for IFS Trade-off between Bandwidth/Resolution
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param nbPixels: number of pixels considered to sample a spectrum
    :param tellurics: tellurics absorption considered if True
    :param broadening: rotational broadening (km/s)
    :param save: save the figure if True
    :param show: displays the figures if True
    :param thermal_model: name of the template library ("BT-Settl" or "Exo-REM")
    :param instru: name of the instrument considered
    """


    config_data=get_config_data(instru)
    
    lmin = config_data['gratings'][band].lmin # lambda_min de la bande considérée
    lmax = config_data['gratings'][band].lmax # lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    if R is None : # dans le cas où il ne s'agit pas d'un spectro-imageur (eg NIRCAM)
        R = spectrum_instru.R
    delta_lambda = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave_inter = np.arange(0.9*lmin,1.1*lmax,delta_lambda) # axe de longueur d'onde de la bande considérée
    
    sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2)
    
    star = load_star_spectrum(T_star,lg_star)
    star.crop(0.9*lmin,1.1*lmax)
    if spectrum_contributions=="reflected" :
        if model=="PICASO":
            albedo = load_albedo(T_planet,lg_planet)
            albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, star.wavelength, renorm = False).flux
        elif model=="tellurics":
            albedo_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_2_5.fits")
            albedo = fits.getdata(albedo_path)
            f = interp1d(albedo[0], albedo[1], bounds_error=False, fill_value=0) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
            albedo = f(star.wavelength)
        elif model=="flat":
            albedo = np.zeros_like(star.wavelength) + 1.
        planet = Spectrum(star.wavelength,albedo*star.flux,star.R,T_planet)
    elif spectrum_contributions=="thermal" :
        planet = load_planet_spectrum(T_planet,lg_planet,model)
        planet.crop(0.9*lmin,1.1*lmax)
   
    trans = transmission(instru,wave_inter[(wave_inter>=lmin)&(wave_inter<=lmax)],band,tellurics,"NO_SP")
    
    syst_rv = np.linspace(-100,100,101)
    delta_rv = np.linspace(-100,100,101)
    
    signal_2d = np.zeros((len(delta_rv), len(syst_rv)), dtype=float)
    noise = np.zeros_like(signal_2d, dtype=float)

    for j in range(len(syst_rv)):
        print(round(100*(j+1)/len(syst_rv),1),"%")
        star_shift = star.doppler_shift(syst_rv[j])
        star_shift.crop(lmin,lmax)
        star_shift = star_shift.degrade_resolution(wave_inter[(wave_inter>=lmin)&(wave_inter<=lmax)],renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
        star_shift.flux *= star_shift.wavelength # star.set_nbphotons_min(config_data,wave_inter) # homgène à un nb de ph
        
        for i in range(len(delta_rv)):
            planet_shift = planet.doppler_shift(syst_rv[j] + delta_rv[i])
            planet_shift.crop(lmin,lmax)       
            planet_shift = planet_shift.degrade_resolution(wave_inter[(wave_inter>=lmin)&(wave_inter<=lmax)],renorm=False) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
            planet_shift.flux *= planet_shift.wavelength # planet.set_nbphotons_min(config_data,wave_inter) # homgène à un nb de ph
            planet_BF = gaussian_filter(planet_shift.flux,sigma=sigma)
            planet_HF = planet_shift.flux - planet_BF
            star_BF = gaussian_filter(star_shift.flux,sigma=sigma)
            star_HF = star_shift.flux - star_BF
            template = trans*planet_HF/np.sqrt(np.nansum((trans*planet_HF)**2))
            alpha = np.sqrt(np.nansum((trans*planet_HF)**2))
            beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
            noise[i, j] = np.sqrt(np.nansum(star_shift.flux * template**2))
            signal_2d[i, j] = (alpha-beta)/(noise[i,j])
            
            
    xx, yy = np.meshgrid(delta_rv, syst_rv)
    
    plt.figure()
    z = (signal_2d).transpose() / np.nanmax((signal_2d))
    plt.contour(xx, yy, z, linewidths=0.333,colors='k')
    plt.pcolormesh(xx, yy, 100*z,cmap=plt.get_cmap('rainbow'))#, vmin=0, vmax=100)

    plt.title(f"SNR fluctuations for $T_p$={planet.T}K in {spectrum_contributions} light ("+model+f') \n with $T_*$={T_star}K',fontsize=14)
    plt.xlabel("delta radial velocity (in km/s)",fontsize=14)
    plt.ylabel("system radial velocity (in km.s)" , fontsize=14)

    cbar = plt.colorbar()
    cbar.set_label('$GAIN_{SNR}$ (in %)', fontsize=14, labelpad=20, rotation=270)

    plt.legend()
    
    filename = f"Colormap_RV_SNR_{spectrum_contributions}_{model}_T{planet.T}K"
    if tellurics:
        filename += "_with_tellurics"
    plt.savefig(save_path_colormap + filename + ".png", format='png')
    
    return xx, yy, (signal_2d).transpose()



############################################################################################################################################################################################################################################"