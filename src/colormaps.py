# import FastYield modules
from src.config import c, R0_max, config_data_list, T_earth, lg_earth, vrot_earth, drv_earth, airmass_earth, T_sun, lg_sun, vrot_sun, M_earth, M_sun, R_sun, AU, G, colormaps_path, sim_data_path
from src.get_specs import _load_tell_trans, get_config_data, get_transmission, get_PSF_profile, get_R_instru
from src.spectrum import get_counts_from_density, load_vega_spectrum, get_wave_K, get_wave_band, get_wavelength_axis_constant_R, filtered_flux, Spectrum, load_star_spectrum, load_planet_spectrum, load_albedo_spectrum, get_spectrum_contribution_name_model, get_thermal_reflected_spectrum
from src.FastCurves import FastCurves
from src.FastYield import planet_types, load_planet_table, get_SNR_from_table, find_matching_planets, plot_matching_planets

# import matplotlib modules
import matplotlib.pyplot as plt

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d

# import astropy modules
from astropy.io import fits

# import other modules
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

plots_colormaps = ["SNR", "lost_signal"]
cmap_colormaps  = "rainbow"
contour_levels  = np.linspace(0, 100, 11)
R_model         = R0_max



#
# Global context worker
#
_CM_CTX = None
def _init_cm_ctx(ctx):
    global _CM_CTX
    _CM_CTX = ctx



#
# GAIN_SNR(Bandwidth vs Resolution) (with constant Nlambda)
#

def colormap_bandwidth_resolution_with_constant_Nlambda(instru="HARMONI", T_planet=T_earth, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, delta_rv=drv_earth, vsini_planet=vrot_earth, vsini_star=vrot_sun, spectrum_contributions="reflected", model="tellurics", airmass=airmass_earth, Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True, Nlambda=None, num=100, title=None):
    
    # Get instru specs
    if instru != "all":
        config_data = get_config_data(instru)
        if config_data["type"] == "imager":
            raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")
    
    # tellurics (or not)
    if instru=="all" or config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
        
    # Number of pixels (spectral channels/bins) considered to sample a spectrum
    if Nlambda is not None:
        Nl = Nlambda
    else:
        if instru=="all":
            Nl = 4096
        else:
            Nl = np.zeros((len(config_data["gratings"])))
            for iband, band in enumerate(config_data["gratings"]):
                Nl[iband] = len(get_wave_band(config_data=config_data, band=band))
            Nl = int(round(np.nanmedian(Nl)))
    
    # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotational broadening with Vsini)
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # [µm]
    else:
        lmax = 12 # [µm]    
    lmin_model = 0.9*lmin                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
    
    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
    
    # Effective model range
    lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin 
    lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax 
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)
    star_spectrum = star_spectrum.broad(vsini_star)                                 # Broadening the spectrum
    star          = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    
    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0)
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    planet_spectrum = planet_spectrum.broad(vsini_planet)                               # Broadening the spectrum
    planet_spectrum = planet_spectrum.doppler_shift(delta_rv)                           # Shifting the spectrum
    planet          = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)

    # To be homogenous to photons
    star.flux   = star.flux   * wave_instru # [J/s/m2/µm] => propto [ph/µm]
    planet.flux = planet.flux * wave_instru # [J/s/m2/µm] => propto [ph/µm]
    
    # Tellurics transmission spectrum (from SkyCalc), if needed  
    if tellurics :
        wave_tell, trans_tell = _load_tell_trans(airmass=1.0)
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1])) 
    else:
        trans_tell = None
    
    # Defining arrays
    R_arr       = np.logspace(np.log10(Rc), np.log10(R_model), num=num)
    l0_arr      = np.linspace(lmin,         lmax,              num=num)
    SNR         = np.zeros((num, num))
    lost_signal = np.zeros((num, num))
    signal      = np.zeros((num, num))

    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(R_arr=R_arr, Nl=Nl, lmin_model=lmin_model, lmax_model=lmax_model, planet=planet, star=star, trans_tell=trans_tell, l0_arr=l0_arr, Rc=Rc, filter_type=filter_type, stellar_halo_photon_noise_limited=stellar_halo_photon_noise_limited)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, SNR_1D, lost_signal_1D, signal_1D) in tqdm(pool.imap_unordered(process_colormap_bandwidth_resolution_with_constant_Nlambda, range(num)), total=num, desc=f"colormap_bandwidth_resolution_with_constant_Nlambda(instru={instru}, model={model}, Rc={Rc})",):
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
            signal[i, :]      = signal_1D

    # Normalizing
    SNR    = SNR / np.nanmax(SNR)
    signal = signal / np.nanmax(signal)
    
    # Plots
    for plot in plots_colormaps:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.yscale('log')
        plt.xlabel(r"Central wavelength $\lambda_0$ [$\mu$m]", fontsize=14)
        plt.ylabel("Spectral resolution $R$", fontsize=14)
        plt.ylim(R_arr[0], R_arr[-1])
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
        
        noise_regime = "stellar halo photon noise regime" if stellar_halo_photon_noise_limited else "detector noise regime"

        # Title
        if title is None:
            tell       = "with tellurics absorption" if tellurics else "without tellurics absorption"
            title_text = (f"{'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star:.0f}K, $T_p$={T_planet:.0f}K, "r"$\Delta$rv="f"{delta_rv:.0f}km/s, "r"$N_\lambda$="f"{Nl:.0f}")
        else:
            title_text = title
        plt.title(title_text, fontsize=16, pad=14)
        #plt.title(r"$N_\lambda$="f"{np.round(Nl, -2):.0f} channels", fontsize=20, pad=14)

        # Scatter & errorbars for bands
        if instru=="all":
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
        else:
            x_instru, y_instru, labels, x_dl = [], [], [], []
            for iband, band in enumerate(config_data['gratings']):
                if (instru=="MIRIMRS" or instru=="HARMONI") and  band != "H_high":
                    labels.append(band[:2])
                elif instru=="ANDES" and iband==0:
                    labels.append(band[:3])
                else:
                    labels.append(band.replace('_', ' '))
                R_band    = config_data['gratings'][band].R
                lmin_band = config_data['gratings'][band].lmin
                lmax_band = config_data['gratings'][band].lmax
                l0_band   = (lmax_band + lmin_band) / 2
                Dl_band   = lmax_band - lmin_band
                x_instru.append(l0_band)
                y_instru.append(R_band)
                x_dl.append(Dl_band / 2)
            plt.errorbar(x_instru, y_instru, xerr=x_dl, fmt='o', color='k', linestyle='None', capsize=5, label=f"{instru}")
            for i, l in enumerate(labels):
                plt.annotate(l, (x_instru[i], 1.2*y_instru[i]), ha='center', fontsize=12, fontweight="bold")

        plt.legend(fontsize=12, loc="upper right", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_with_constant_Nlambda_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Nl{Nl}_{noise_regime.replace(' ', '_')}"
        plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
    return l0_arr, R_arr, SNR, lost_signal, signal

def process_colormap_bandwidth_resolution_with_constant_Nlambda(i):
    
    R_arr                             = _CM_CTX["R_arr"]
    Nl                                = _CM_CTX["Nl"]
    lmin_model                        = _CM_CTX["lmin_model"]
    lmax_model                        = _CM_CTX["lmax_model"]
    planet                            = _CM_CTX["planet"] # propto [ph/µm]
    star                              = _CM_CTX["star"]   # propto [ph/µm]
    trans_tell                        = _CM_CTX["trans_tell"]
    l0_arr                            = _CM_CTX["l0_arr"]
    Rc                                = _CM_CTX["Rc"]
    filter_type                       = _CM_CTX["filter_type"]
    stellar_halo_photon_noise_limited = _CM_CTX["stellar_halo_photon_noise_limited"]

    R = R_arr[i]
    
    # --- Build a wavelength grid with *constant resolving power* R = res ---
    # For a spectrograph with resolving power R, one resolution element is:
    #   Δλ_res = λ / R
    # Nyquist sampling requires 2 samples per resolution element, hence a pixel step:
    #   Δλ_pix ≈ (λ / R) / 2  =>  Δλ_pix / λ ≈ 1 / (2R)
    # This means the natural uniform grid is in ln(λ), with constant step: dln(λ)/dλ = 1/λ => dln(λ) = dλ/λ = 1 / (2R)
    #   Δln(λ) = ln(λ_{i+1}) - ln(λ_i) ≈ Δλ_pix / λ ≈ 1/(2R) = constant
    dln = 1.0 / (2.0 * R)
    
    # Number of samples needed to cover [lmin_model, lmax_model] with that constant Δln(λ)
    # ln(λ_i) = ln(λ_min) + k*Δln(λ) => λ_i = λ_min * exp(k*Δln(λ))
    # If we have n points, it means that we have n - 1 between l_min and l_max
    # because: ln(l_max/l_min) = (n-1) * dln  =>  n ≈ ln(l_max/l_min)/dln + 1
    n = int(np.floor(np.log(lmax_model / lmin_model) / dln)) + 1
    
    # Log-uniform wavelength grid: λ_i = λ_min * exp(i * Δln(λ))
    # This keeps λ/Δλ (i.e., R) approximately constant across the whole band.
    wave_R  = lmin_model * np.exp(np.arange(n) * dln)             # [µm]
    dwave_R = np.gradient(wave_R)                                 # [µm/bin]

    # --- Build, for each (l0, Nl), the index range of a contiguous spectral window ---
    # On a log-λ grid, a window of Nl pixels corresponds to a multiplicative span in λ.
    # Half-window in ln(λ) is: half = (Nl/2) * Δln(λ)
    half = Nl * dln / 2
    
    # Degrading to wave_R and converting from [ph/µm] to [ph]
    planet_R = planet.degrade_resolution(wave_R, renorm=False).flux # [ph/µm]
    star_R   = star.degrade_resolution(wave_R,   renorm=False).flux # [ph/µm]
    planet_R = planet_R * dwave_R                                   # [ph/µm] => [ph/bin]
    star_R   = star_R   * dwave_R                                   # [ph/µm] => [ph/bin]
    
    if trans_tell is not None:
        trans_tell_R = trans_tell.degrade_resolution(wave_R, renorm=False).flux
    
    SNR_1D         = np.zeros((len(l0_arr))) + np.nan
    lost_signal_1D = np.zeros((len(l0_arr))) + np.nan
    signal_1D      = np.zeros((len(l0_arr))) + np.nan
    for j, l0 in enumerate(l0_arr):
        
        # Convert the ±half span in ln(λ) into wavelength bounds around the central wavelength λ0:
        l0_min = l0 * np.exp(-half)
        l0_max = l0 * np.exp(+half)
        
        # # Keeping only the cases inside the model range
        # if (l0_min < lmin_model) or (l0_max > lmax_model):
        #     continue
        
        idx_lo = np.searchsorted(wave_R, l0_min, side="left")
        idx_hi = np.searchsorted(wave_R, l0_max, side="right")
        idx_lo = np.clip(idx_lo, 0, len(wave_R)-1)
        idx_hi = np.clip(idx_hi, 1, len(wave_R))
        idx_hi = np.maximum(idx_hi, idx_lo + 1)
        sl     = slice(idx_lo, idx_hi)        
        
        if trans_tell is not None:
            trans = trans_tell_R[sl]
        else:
            trans = 1 
        star_R_crop          = star_R[sl]
        planet_R_crop        = planet_R[sl]
        planet_HF, planet_LF = filtered_flux(planet_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_LF     = filtered_flux(star_R_crop,   R=R, Rc=Rc, filter_type=filter_type)
        template             = trans*planet_HF
        template             = template / np.sqrt(np.nansum(template**2))
        alpha                = np.nansum(trans*planet_HF * template)
        beta                 = np.nansum(trans*star_HF*planet_LF/star_LF * template)
        if stellar_halo_photon_noise_limited:
            sigma_CCF = np.sqrt(np.nansum(trans*star_R_crop * template**2)) # stellar halo photon noise
            #sigma     = np.sqrt(trans*star_R_crop)
        else:
            sigma_CCF = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)                
            #sigma     = np.ones_like(template)
        
        #from src.signal_noise import get_fn_MM, get_fn_MM_exact, get_fn_HF_LF
        #fn_MM, _          = get_fn_HF_LF(N=len(template), R=R, Rc=Rc, filter_type=filter_type, empirical=False)
        #fn_MM             = get_fn_MM_exact(trans=trans, Ss=star_R_crop, template=template, R=R, Rc=Rc, filter_type=filter_type, sigma=sigma)
        #fn_MM             = get_fn_MM(template=template, R=R, Rc=Rc, filter_type=filter_type, sigma=sigma)
        #sigma_CCF        *= np.sqrt(fn_MM)
        
        SNR_1D[j]         = (alpha - beta) / (sigma_CCF)
        lost_signal_1D[j] = beta / alpha
        signal_1D[j]      = alpha - beta
    
    return i, SNR_1D, lost_signal_1D, signal_1D



#
# GAIN_SNR(Bandwidth vs Resolution) (with constant Dlambda)
#

def colormap_bandwidth_resolution_with_constant_Dlambda(instru="HARMONI", T_planet=T_earth, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, delta_rv=drv_earth, vsini_planet=vrot_earth, vsini_star=vrot_earth, spectrum_contributions="reflected", model="tellurics", airmass=airmass_earth, Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=False, num=100):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
    
    # Bandwidth considered to sample a spectrum
    Dl = np.zeros((len(config_data["gratings"])))
    for iband, band in enumerate(config_data["gratings"]):
        Dl[iband] = config_data["gratings"][band].lmax - config_data["gratings"][band].lmin
    Dl = np.nanmedian(Dl)
    
    # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotational broadening with Vsini)
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3  # [µm]
    else:
        lmax = 12 # [µm]      
    lmin_model = 0.9*lmin                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
    
    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
    
    # Effective model range
    lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin 
    lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax 
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)
    star_spectrum = star_spectrum.broad(vsini_star)                                 # Broadening the spectrum
    star          = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    
    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0)
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    planet_spectrum = planet_spectrum.broad(vsini_planet)                               # Broadening the spectrum
    planet_spectrum = planet_spectrum.doppler_shift(delta_rv)                           # Shifting the spectrum
    planet          = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)

    # To be homogenous to photons
    star.flux   = star.flux   * wave_instru # [J/s/m2/µm] => propto [ph/µm]
    planet.flux = planet.flux * wave_instru # [J/s/m2/µm] => propto [ph/µm]
    
    # Tellurics transmission spectrum (from SkyCalc), if needed  
    if tellurics :
        wave_tell, trans_tell = _load_tell_trans(airmass=1.0)
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1])) 
    else:
        trans_tell = None
    
    # Defining arrays
    R_arr       = np.logspace(np.log10(Rc), np.log10(R_model), num=num)
    l0_arr      = np.linspace(lmin,         lmax,              num=num)
    SNR         = np.zeros((num, num))
    lost_signal = np.zeros((num, num))
    
    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(R_arr=R_arr, Dl=Dl, lmin_model=lmin_model, lmax_model=lmax_model, planet=planet, star=star, trans_tell=trans_tell, l0_arr=l0_arr, Rc=Rc, filter_type=filter_type, stellar_halo_photon_noise_limited=stellar_halo_photon_noise_limited)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, SNR_1D, lost_signal_1D) in tqdm(pool.imap_unordered(process_colormap_bandwidth_resolution_with_constant_Dlambda, range(num)), total=num, desc=f"colormap_bandwidth_resolution_with_constant_Dlambda(instru={instru}, model={model}, Rc={Rc})",):
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    SNR = SNR / np.nanmax(SNR)
    
    # Plots
    for plot in plots_colormaps:
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
        noise_regime = "stellar halo photon noise regime" if stellar_halo_photon_noise_limited else "detector noise regime"
        title_text   = (f"{'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star:.0f}K, $T_p$={T_planet:.0f}K, "r"$\Delta$rv="f"{delta_rv:.0f}km/s, "r"$\Delta\lambda$="f"{Dl:.2f}µm")
        plt.title(title_text, fontsize=16, pad=14)
        
        # Scatter & errorbars for bands
        x_instru, y_instru, labels, x_dl = [], [], [], []
        for iband, band in enumerate(config_data['gratings']):
            if (instru=="MIRIMRS" or instru=="HARMONI") and  band != "H_high":
                labels.append(band[:2])
            elif instru=="ANDES" and iband==0:
                labels.append(band[:3])
            else:
                labels.append(band.replace('_', ' '))
            R_band    = config_data['gratings'][band].R
            lmin_band = config_data['gratings'][band].lmin
            lmax_band = config_data['gratings'][band].lmax
            l0_band   = (lmax_band + lmin_band) / 2
            Dl_band   = lmax_band - lmin_band
            x_instru.append(l0_band)
            y_instru.append(R_band)
            x_dl.append(Dl_band / 2)
        plt.errorbar(x_instru, y_instru, xerr=x_dl, fmt='o', color='k', linestyle='None', capsize=5, label=f"{instru} bands")
        for i, l in enumerate(labels):
            plt.annotate(l, (x_instru[i], 1.2*y_instru[i]), ha='center', fontsize=12)

        plt.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        filename = f"colormaps_bandwidth_resolution/Colormap_bandwidth_resolution_with_constant_Dlambda_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Dl{Dl}um_{noise_regime.replace(' ', '_')}"
        plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
    return l0_arr, R_arr, SNR, lost_signal

def process_colormap_bandwidth_resolution_with_constant_Dlambda(i):
    
    R_arr                             = _CM_CTX["R_arr"]
    Dl                                = _CM_CTX["Dl"]
    lmin_model                        = _CM_CTX["lmin_model"]
    lmax_model                        = _CM_CTX["lmax_model"]
    planet                            = _CM_CTX["planet"] # propto [ph/µm]
    star                              = _CM_CTX["star"]   # propto [ph/µm]
    trans_tell                        = _CM_CTX["trans_tell"]
    l0_arr                            = _CM_CTX["l0_arr"]
    Rc                                = _CM_CTX["Rc"]
    filter_type                       = _CM_CTX["filter_type"]
    stellar_halo_photon_noise_limited = _CM_CTX["stellar_halo_photon_noise_limited"]
    
    R = R_arr[i]
    
    # --- Build a wavelength grid with *constant resolving power* R = res ---
    # For a spectrograph with resolving power R, one resolution element is:
    #   Δλ_res = λ / R
    # Nyquist sampling requires 2 samples per resolution element, hence a pixel step:
    #   Δλ_pix ≈ (λ / R) / 2  =>  Δλ_pix / λ ≈ 1 / (2R)
    # This means the natural uniform grid is in ln(λ), with constant step: dln(λ)/dλ = 1/λ => dln(λ) = dλ/λ = 1 / (2R)
    #   Δln(λ) = ln(λ_{i+1}) - ln(λ_i) ≈ Δλ_pix / λ ≈ 1/(2R) = constant
    dln = 1.0 / (2.0 * R)
    
    # Number of samples needed to cover [lmin_model, lmax_model] with that constant Δln(λ)
    # ln(λ_i) = ln(λ_min) + k*Δln(λ) => λ_i = λ_min * exp(k*Δln(λ))
    # If we have n points, it means that we have n - 1 between l_min and l_max
    # because: ln(l_max/l_min) = (n-1) * dln  =>  n ≈ ln(l_max/l_min)/dln + 1
    n = int(np.floor(np.log(lmax_model / lmin_model) / dln)) + 1
    
    # Log-uniform wavelength grid: λ_i = λ_min * exp(i * Δln(λ))
    # This keeps λ/Δλ (i.e., R) approximately constant across the whole band.
    wave_R  = lmin_model * np.exp(np.arange(n) * dln)             # [µm]
    dwave_R = np.gradient(wave_R)                                 # [µm/bin]
    
    # Degrading to wave_R and converting from [ph/µm] to [ph]
    planet_R = planet.degrade_resolution(wave_R, renorm=False).flux # [ph/µm]
    star_R   = star.degrade_resolution(wave_R,   renorm=False).flux # [ph/µm]
    planet_R = planet_R * dwave_R # [ph/µm] => [ph/bin]
    star_R   = star_R   * dwave_R # [ph/µm] => [ph/bin]
    
    if trans_tell is not None:
        trans_tell_R = trans_tell.degrade_resolution(wave_R, renorm=False).flux
    
    SNR_1D         = np.zeros((len(l0_arr))) + np.nan
    lost_signal_1D = np.zeros((len(l0_arr))) + np.nan
    for j, l0 in enumerate(l0_arr):
        
        # Ranges
        l0_min = l0 - Dl/2
        l0_max = l0 + Dl/2
        
        # # Keeping only the cases inside the model range
        # if (l0_min < lmin_model) or (l0_max > lmax_model):
        #     continue
        
        idx_lo = np.searchsorted(wave_R, l0_min, side="left")
        idx_hi = np.searchsorted(wave_R, l0_max, side="right")
        idx_lo = np.clip(idx_lo, 0, len(wave_R)-1)
        idx_hi = np.clip(idx_hi, 1, len(wave_R))
        idx_hi = np.maximum(idx_hi, idx_lo + 1)
        sl     = slice(idx_lo, idx_hi)        
        
        if trans_tell is not None:
            trans = trans_tell_R[sl]
        else:
            trans = 1 
        star_R_crop          = star_R[sl]
        planet_R_crop        = planet_R[sl]
        planet_HF, planet_LF = filtered_flux(planet_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_LF     = filtered_flux(star_R_crop,   R=R, Rc=Rc, filter_type=filter_type)
        template             = trans*planet_HF 
        template             = template / np.sqrt(np.nansum(template**2))
        alpha                = np.nansum(trans*planet_HF * template)
        beta                 = np.nansum(trans*star_HF*planet_LF/star_LF * template)
        if stellar_halo_photon_noise_limited:
            sigma_CCF = np.sqrt(np.nansum(trans*star_R_crop * template**2)) # stellar halo photon noise
        else:
            sigma_CCF = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / sigma_CCF
        lost_signal_1D[j] = beta / alpha
                    
    return i, SNR_1D, lost_signal_1D



#
# GAIN_SNR(Bandwidth vs Planet Temperature)
#

def colormap_bandwidth_Tp(instru, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, delta_rv=drv_earth, vsini_planet=vrot_earth, vsini_star=vrot_sun, spectrum_contributions="thermal", model="BT-Settl", airmass=airmass_earth, Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True, num=100):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
    
    # Mean instru specs (R, Nl)
    R  = np.zeros((len(config_data["gratings"])))
    Nl = np.zeros((len(config_data["gratings"])))
    for iband, band in enumerate(config_data["gratings"]):
        R[iband]  = config_data["gratings"][band].R
        Nl[iband] = len(get_wave_band(config_data=config_data, band=band))
    R  = np.nanmedian(R)
    Nl = int(round(np.nanmedian(Nl)))
    
    # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotational broadening with Vsini)
    lmin = 0.6
    if spectrum_contributions == "reflected" or tellurics :
        lmax = 3 # [µm]
    else:
        lmax = 12 # [µm]    
    lmin_model = 0.9*lmin                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
    
    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
    
    # Effective model range
    lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin 
    lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax 
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)
    star_spectrum = star_spectrum.broad(vsini_star)                                 # Broadening the spectrum
    star          = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)    
    star.flux     = star.flux * wave_instru                                        # [J/s/m2/µm] => propto [ph/µm]
        
    # Tellurics transmission spectrum (from SkyCalc), if needed  
    if tellurics :
        wave_tell, trans_tell = _load_tell_trans(airmass=1.0)
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1])) 
    else:
        trans_tell = None
    
    # Defining arrays
    T_arr       = np.linspace(300,  2000, num)
    l0_arr      = np.linspace(lmin, lmax, num)
    SNR         = np.zeros((num, num))
    lost_signal = np.zeros((num, num))
    l0_opti     = np.zeros((num))
    
    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(instru=instru, T_arr=T_arr, lg_planet=lg_planet, vsini_planet=vsini_planet, delta_rv=delta_rv, model=model, spectrum_contributions=spectrum_contributions, R=R, Nl=Nl, lmin_model=lmin_model, lmax_model=lmax_model, wave_model=wave_model, wave_instru=wave_instru, star_spectrum=star_spectrum, star=star, trans_tell=trans_tell, l0_arr=l0_arr, Rc=Rc, filter_type=filter_type, stellar_halo_photon_noise_limited=stellar_halo_photon_noise_limited)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, SNR_1D, lost_signal_1D) in tqdm(pool.imap_unordered(process_colormap_bandwidth_Tp, range(num)), total=num, desc=f"colormap_bandwidth_Tp(instru={instru}, model={model}, Rc={Rc})",):
            SNR[i, :]         = SNR_1D / np.nanmax(SNR_1D) # Normalizing each row
            lost_signal[i, :] = lost_signal_1D
            l0_opti[i]        = l0_arr[SNR_1D.argmax()]
    
    # Plots
    for plot in plots_colormaps:
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
        noise_regime = "stellar halo photon noise regime" if stellar_halo_photon_noise_limited else "detector noise regime"
        title_text   = (f"{'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star:.0f}K, "r"$\Delta$rv="f"{delta_rv:.0f}km/s, R={R:.0f}, "r"$N_\lambda$="f"{Nl:.0f}")
        plt.title(title_text, fontsize=16, pad=14)
        
        # Bandes spectrales
        ax.plot([], [], "k", label=f"{instru} bands")
        bands_done = []
        y_center   = np.nanmean(T_arr)
        for ib, band in enumerate(config_data['gratings']):
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
        # ax.plot(l0_opti, T_arr, 'k:', lw=1.2, label=r"Optimum $\lambda_0$")

        plt.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        filename = f"colormaps_bandwidth_Tp/Colormap_bandwidth_Tp_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_Nl{Nl}_R{R}_{noise_regime.replace(' ', '_')}"
        plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
        
    return l0_arr, T_arr, SNR, lost_signal, l0_opti

def process_colormap_bandwidth_Tp(i):
    instru                            = _CM_CTX["instru"]
    T_arr                             = _CM_CTX["T_arr"]
    lg_planet                         = _CM_CTX["lg_planet"]
    vsini_planet                      = _CM_CTX["vsini_planet"]
    delta_rv                          = _CM_CTX["delta_rv"]
    model                             = _CM_CTX["model"]
    spectrum_contributions            = _CM_CTX["spectrum_contributions"]
    airmass                           = _CM_CTX["airmass"]
    R                                 = _CM_CTX["R"]
    Nl                                = _CM_CTX["Nl"]
    lmin_model                        = _CM_CTX["lmin_model"]
    lmax_model                        = _CM_CTX["lmax_model"]
    wave_model                        = _CM_CTX["wave_model"]
    wave_instru                       = _CM_CTX["wave_instru"]
    star_spectrum                     = _CM_CTX["star_spectrum"] # (wave_model)  propto [ph/µm]
    star                              = _CM_CTX["star"]          # (wave_instru) propto [ph/µm]
    trans_tell                        = _CM_CTX["trans_tell"]
    l0_arr                            = _CM_CTX["l0_arr"]
    Rc                                = _CM_CTX["Rc"]
    filter_type                       = _CM_CTX["filter_type"]
    stellar_halo_photon_noise_limited = _CM_CTX["stellar_halo_photon_noise_limited"]
        
    T_planet = T_arr[i]
    
    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0)
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    planet_spectrum = planet_spectrum.broad(vsini_planet)                               # Broadening the spectrum
    planet_spectrum = planet_spectrum.doppler_shift(delta_rv)                           # Shifting the spectrum
    planet          = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    planet.flux     = planet.flux * wave_instru                                         # [J/s/m2/µm] => propto [ph/µm]
    
    # --- Build a wavelength grid with *constant resolving power* R = res ---
    # For a spectrograph with resolving power R, one resolution element is:
    #   Δλ_res = λ / R
    # Nyquist sampling requires 2 samples per resolution element, hence a pixel step:
    #   Δλ_pix ≈ (λ / R) / 2  =>  Δλ_pix / λ ≈ 1 / (2R)
    # This means the natural uniform grid is in ln(λ), with constant step: dln(λ)/dλ = 1/λ => dln(λ) = dλ/λ = 1 / (2R)
    #   Δln(λ) = ln(λ_{i+1}) - ln(λ_i) ≈ Δλ_pix / λ ≈ 1/(2R) = constant
    dln = 1.0 / (2.0 * R)
    
    # Number of samples needed to cover [lmin_model, lmax_model] with that constant Δln(λ)
    # ln(λ_i) = ln(λ_min) + k*Δln(λ) => λ_i = λ_min * exp(k*Δln(λ))
    # If we have n points, it means that we have n - 1 between l_min and l_max
    # because: ln(l_max/l_min) = (n-1) * dln  =>  n ≈ ln(l_max/l_min)/dln + 1
    n = int(np.floor(np.log(lmax_model / lmin_model) / dln)) + 1
    
    # Log-uniform wavelength grid: λ_i = λ_min * exp(i * Δln(λ))
    # This keeps λ/Δλ (i.e., R) approximately constant across the whole band.
    wave_R  = lmin_model * np.exp(np.arange(n) * dln)             # [µm]
    dwave_R = np.gradient(wave_R)                                 # [µm/bin]

    # --- Build, for each (l0, Nl), the index range of a contiguous spectral window ---
    # On a log-λ grid, a window of Nl pixels corresponds to a multiplicative span in λ.
    # Half-window in ln(λ) is: half = (Nl/2) * Δln(λ)
    half = Nl * dln / 2
    
    # Degrading to wave_R and converting from [ph/µm] to [ph]
    planet_R = planet.degrade_resolution(wave_R, renorm=False).flux # [ph/µm]
    star_R   = star.degrade_resolution(wave_R,   renorm=False).flux # [ph/µm]
    planet_R = planet_R * dwave_R # [ph/µm] => [ph/bin]
    star_R   = star_R   * dwave_R # [ph/µm] => [ph/bin]
    
    if trans_tell is not None:
        trans_tell_R = trans_tell.degrade_resolution(wave_R, renorm=False).flux
    
    SNR_1D         = np.zeros((len(l0_arr))) + np.nan
    lost_signal_1D = np.zeros((len(l0_arr))) + np.nan
    for j, l0 in enumerate(l0_arr):
        
        # Convert the ±half span in ln(λ) into wavelength bounds around the central wavelength λ0:
        l0_min = l0 * np.exp(-half)
        l0_max = l0 * np.exp(+half)
        
        # # Keeping only the cases inside the model range
        # if (l0_min < lmin_model) or (l0_max > lmax_model):
        #     continue
        
        idx_lo = np.searchsorted(wave_R, l0_min, side="left")
        idx_hi = np.searchsorted(wave_R, l0_max, side="right")
        idx_lo = np.clip(idx_lo, 0, len(wave_R)-1)
        idx_hi = np.clip(idx_hi, 1, len(wave_R))
        idx_hi = np.maximum(idx_hi, idx_lo + 1)
        sl     = slice(idx_lo, idx_hi)        
        
        if trans_tell is not None:
            trans = trans_tell_R[sl]
        else:
            trans = 1 
        star_R_crop          = star_R[sl]
        planet_R_crop        = planet_R[sl]
        planet_HF, planet_LF = filtered_flux(planet_R_crop, R=R, Rc=Rc, filter_type=filter_type)
        star_HF, star_LF     = filtered_flux(star_R_crop,   R=R, Rc=Rc, filter_type=filter_type)
        template             = trans*planet_HF 
        template             = template / np.sqrt(np.nansum(template**2))
        alpha                = np.nansum(trans*planet_HF * template)
        beta                 = np.nansum(trans*star_HF*planet_LF/star_LF * template)
        if stellar_halo_photon_noise_limited:
            sigma_CCF = np.sqrt(np.nansum(trans*star_R_crop * template**2)) # stellar halo photon noise
        else:
            sigma_CCF = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / sigma_CCF
        lost_signal_1D[j] = beta / alpha
                    
    return i, SNR_1D, lost_signal_1D
    


#
# GAIN_SNR(Bands vs Planet Temperature)
#

def colormap_bands_Tp(instru, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, delta_rv=drv_earth, vsini_planet=vrot_earth, vsini_star=vrot_sun, spectrum_contributions="thermal", model="BT-Settl", airmass=airmass_earth, Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True, num=100):    
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
        
    # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotational broadening with Vsini)
    lmin_model = 0.9*config_data["lambda_range"]["lambda_min"] # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*config_data["lambda_range"]["lambda_max"] # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                      # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model)   # [µm] Model wavelength axis (with constant dl step)
    
    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)
    star_spectrum = star_spectrum.broad(vsini_star)                                 # Broadening the spectrum
    star          = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)    
    star.flux     = star.flux * wave_instru                                        # [J/s/m2/µm] => propto [ph/µm]
        
    # Tellurics transmission spectrum (from SkyCalc), if needed  
    if tellurics :
        wave_tell, trans_tell = _load_tell_trans(airmass=1.0)
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1])) 
    else:
        trans_tell = None

    # Definig arrays
    T_arr       = np.linspace(300, 2000, num)
    bands       = np.array([band.replace("_", " ") for band in config_data["gratings"]])
    SNR         = np.zeros((num, len(bands)))
    lost_signal = np.zeros((num, len(bands)))

    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(instru=instru, config_data=config_data, T_arr=T_arr, lg_planet=lg_planet, vsini_planet=vsini_planet, delta_rv=delta_rv, model=model, spectrum_contributions=spectrum_contributions, airmass=airmass, wave_model=wave_model, wave_instru=wave_instru, star_spectrum=star_spectrum, star=star, trans_tell=trans_tell, Rc=Rc, filter_type=filter_type, stellar_halo_photon_noise_limited=stellar_halo_photon_noise_limited)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, SNR_1D, lost_signal_1D) in tqdm(pool.imap_unordered(process_colormap_bands_Tp, range(num)), total=num, desc=f"colormap_bandwidth_Tp(instru={instru}, model={model}, Rc={Rc})",):
            SNR[i, :]         = SNR_1D / np.nanmax(SNR_1D) # Normalizing each row
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing each line (for each temperature/scientific case)
    for nt in range(num):
        SNR[nt, :] = SNR[nt, :] / np.nanmax(SNR[nt, :])
    # for ib in range(len(bands)):
    #      SNR[:, ib] = SNR[:, ib] / np.nanmax(SNR[:, ib])
    
    # Plots
    bands_idx          = np.arange(len(bands))
    bands_idx_extended = np.arange(-1, len(bands)+1)
    
    for plot in plots_colormaps:
                
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel("Bands", fontsize=14)
        plt.ylabel("Planet temperature [K]", fontsize=14)
        plt.ylim([T_arr[0], T_arr[-1]])
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()

        if plot == "SNR":
            data       = 100 * SNR
            cmap       = plt.get_cmap(cmap_colormaps)
            cbar_label = '$GAIN_{S/N}$ [%]'
        else:
            data       = 100 * lost_signal
            cmap       = plt.get_cmap(cmap_colormaps+'_r')
            cbar_label = r'Lost signal $\beta/\alpha$ [%]'
        data_extended = np.column_stack([data[:, 0], data, data[:, -1]])

        # Heatmap with pcolormesh        
        mesh = plt.pcolormesh(bands_idx_extended, T_arr, data_extended, cmap=cmap, shading='auto', vmin=0, vmax=100) 
        
        # Contours
        bands_mesh, T_mesh = np.meshgrid(bands_idx_extended, T_arr, indexing='xy')
        cs                 = plt.contour(bands_mesh, T_mesh, data_extended, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax   = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # Title
        tell         = "with tellurics absorption" if tellurics else "without tellurics absorption"
        noise_regime = "stellar halo photon noise regime" if stellar_halo_photon_noise_limited else "detector noise regime"
        title_text   = (f"{instru} {'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, "r"$\Delta$rv="f"{delta_rv}km/s")
        plt.title(title_text, fontsize=16, pad=14)
        
        plt.xlim(-0.5, len(bands)-0.5)
        plt.xticks(bands_idx, bands)
        plt.tight_layout()
        filename = f"colormaps_bands_Tp/Colormap_bands_Tp_{plot}_{instru}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms_{noise_regime.replace(' ', '_')}"
        plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
    
    return bands, T_arr, SNR, lost_signal
    
def process_colormap_bands_Tp(i):
    instru                            = _CM_CTX["instru"]
    config_data                       = _CM_CTX["config_data"]
    T_arr                             = _CM_CTX["T_arr"]
    lg_planet                         = _CM_CTX["lg_planet"]
    vsini_planet                      = _CM_CTX["vsini_planet"]
    delta_rv                          = _CM_CTX["delta_rv"]
    model                             = _CM_CTX["model"]
    spectrum_contributions            = _CM_CTX["spectrum_contributions"]
    airmass                           = _CM_CTX["airmass"]
    wave_model                        = _CM_CTX["wave_model"]
    wave_instru                       = _CM_CTX["wave_instru"]
    star_spectrum                     = _CM_CTX["star_spectrum"] # (wave_model)  propto [ph/µm]
    star                              = _CM_CTX["star"]          # (wave_instru) propto [ph/µm]
    trans_tell                        = _CM_CTX["trans_tell"]
    Rc                                = _CM_CTX["Rc"]
    filter_type                       = _CM_CTX["filter_type"]
    stellar_halo_photon_noise_limited = _CM_CTX["stellar_halo_photon_noise_limited"]
    
    T_planet = T_arr[i]
    
    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0)
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    planet_spectrum = planet_spectrum.broad(vsini_planet)                               # Broadening the spectrum
    planet_spectrum = planet_spectrum.doppler_shift(delta_rv)                           # Shifting the spectrum
    planet          = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    planet.flux     = planet.flux * wave_instru                                         # [J/s/m2/µm] => propto [ph/µm]
    
    SNR_1D         = np.zeros((len(config_data["gratings"]))) + np.nan
    lost_signal_1D = np.zeros((len(config_data["gratings"]))) + np.nan
    for ib, band in enumerate(config_data["gratings"]):
        
        wave_band  = get_wave_band(config_data=config_data, band=band)
        dwave_band = np.gradient(wave_band)
        R_band     = config_data["gratings"][band].R
        
        # Degrading to wave_band and converting from [ph/µm] to [ph]
        planet_band = planet.degrade_resolution(wave_band, renorm=False).flux # [ph/µm]
        star_band   = star.degrade_resolution(wave_band,   renorm=False).flux # [ph/µm]
        planet_band = planet_band * dwave_band # [ph/µm] => [ph/bin]
        star_band   = star_band   * dwave_band # [ph/µm] => [ph/bin]
        
        if trans_tell is not None:
            trans = trans_tell.degrade_resolution(wave_band, renorm=False).flux
        else:
            trans = 1.

        planet_HF, planet_LF = filtered_flux(planet_band, R=R_band, Rc=Rc, filter_type=filter_type)
        star_HF, star_LF     = filtered_flux(star_band,   R=R_band, Rc=Rc, filter_type=filter_type)
        template             = trans*planet_HF 
        template             = template / np.sqrt(np.nansum(template**2))
        alpha                = np.nansum(trans*planet_HF * template)
        beta                 = np.nansum(trans*star_HF*planet_LF/star_LF * template)
        if stellar_halo_photon_noise_limited:
            sigma_CCF = np.sqrt(np.nansum(trans*star_band * template**2)) # stellar halo photon noise
        else:
            sigma_CCF = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[ib]         = (alpha - beta) / sigma_CCF
        lost_signal_1D[ib] = beta / alpha
                    
    return i, SNR_1D, lost_signal_1D



#
# GAIN_SNR(Bands vs Planet Types)
#

def colormap_bands_ptypes_SNR(mode="multi", instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="JQ1", systematics=False, PCA=False, planet_types=planet_types):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")
        
    # get spectrum contribution + name model
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Load the planet tables and retrieving SNR
    table = "Archive"
    if systematics:
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
            SNR.append(get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band="INSTRU"))
            for band in config_data["gratings"]:
                SNR_bands[f"{band}"].append(get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band))
    
    # Keeping only the best SNR over the apodizers/coronagraph config
    planet_table["SNR"] = np.nanmax(np.stack(SNR, axis=0), axis=0)
    for band in config_data["gratings"]:
        planet_table[f"SNR_{band}"] = np.nanmax(np.stack(SNR_bands[f"{band}"], axis=0), axis=0)

    # Find planets based on mode
    planet_table_pd  = planet_table.to_pandas() # convert to pandas to find matching types
    selected_planets = set()                    # Set to store already assigned planets for mode = "unique"
    matching_planets = {planet_type: find_matching_planets(criteria, planet_table_pd, mode, selected_planets) for planet_type, criteria in planet_types.items()}

    # Table plot
    plot_matching_planets(matching_planets=matching_planets, exposure_time=exposure_time, mode=mode, planet_types=planet_types)

    # Definig arrays
    planet_types_list = [ptype for ptype, planets in matching_planets.items() if len(planets) > 0]
    planet_types_arr  = np.array(planet_types_list, dtype=object)
    bands             = np.array([band for band in config_data["gratings"]])

    # Map each planet name to its row index to avoid repeated calls to get_planet_index
    name_to_index = {name: i for i, name in enumerate(planet_table["PlanetName"])}
    
    # Build the global band-SNR matrix with shape (N_planets, N_bands)
    snr_by_band = np.vstack([np.asarray(planet_table[f"SNR_{band}"], dtype=float) for band in bands]).T
    
    # Accumulate the relative SNR per planet type
    SNR = np.full((len(bands), len(planet_types_arr)), np.nan, dtype=float)
    for itype, ptype in enumerate(planet_types_arr):
        idx = np.array([name_to_index[planet["PlanetName"]] for planet in matching_planets[ptype]], dtype=int)
    
        snr = snr_by_band[idx, :].copy()  # shape = (N_planets_of_type, N_bands)
    
        # Normalize each planet individually to convert absolute SNR into relative band-to-band SNR
        snr = snr / np.nanmax(snr, axis=1)[:, None]  # shape = (N_planets_of_type, N_bands)
    
        # Average the relative SNR over all planets of the same type
        SNR[:, itype] = np.nanmean(snr, axis=0)  # shape = (N_bands,)
    
    # Renormalize each planet-type column, since the per-type average does not necessarily peak at 1
    SNR = SNR / np.nanmax(SNR, axis=0)[None, :]  # shape = (N_bands, N_planet_types)

    # Plot
    planet_types_arr_idx          = np.arange(len(planet_types_arr))
    planet_types_arr_idx_extended = np.arange(-1, len(planet_types_arr)+1)
    bands_idx                     = np.arange(len(bands))
    bands_idx_extended            = np.arange(-1, len(bands)+1)
    data                          = 100 * SNR
    data_extended                 = np.vstack([data[0], data, data[-1]])
    data_extended                 = np.hstack([data_extended[:, [0]], data_extended, data_extended[:, [-1]]])
    
    cmap = plt.get_cmap(cmap_colormaps)
    plt.figure(figsize=(10, 8), dpi=300)
    plt.xlabel("Various planetary types", fontsize=16, weight='bold')
    plt.ylabel("Various spectral modes",  fontsize=16, weight='bold')
    plt.title(f"{instru} S/N fluctuations\nin {spectrum_contributions} light ({name_model}-model)", fontsize=18, pad=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Heatmap with pcolormesh
    mesh = plt.pcolormesh(planet_types_arr_idx_extended, bands_idx_extended, data_extended, cmap=cmap, shading='auto', vmin=0, vmax=100) 
    
    # Contours
    planet_types_arr_mesh, bands_mesh = np.meshgrid(planet_types_arr_idx_extended, bands_idx_extended, indexing='xy')
    cs                                = plt.contour(planet_types_arr_mesh, bands_mesh, data_extended, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")

    # Colorbar
    ax = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.05, shrink=1)
    cbar.minorticks_on()
    cbar.set_ticks(contour_levels)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('$GAIN_{S/N}$ [%]', rotation=270, labelpad=14, fontsize=14)
    cbar.ax.text(1.2, 1.05,  'High signal', ha='center', va='bottom', fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='red')
    cbar.ax.text(1.2, -0.05, 'Poor signal', ha='center', va='top',    fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='blue')

    plt.xlim(-0.5, len(planet_types_arr)-0.5)
    plt.xticks(planet_types_arr_idx, planet_types_arr, rotation=45, ha="right")
    plt.ylim(-0.5, len(bands)-0.5)
    plt.yticks(bands_idx, bands)
    plt.tight_layout()
    filename = f"colormaps_bands_planets_snr/colormap_bands_ptypes_SNR_{instru}_{strehl}_{suffix}_{name_model}_{mode}"
    plt.savefig(colormaps_path + filename + ".png", format='png', bbox_inches='tight')
    plt.show()
    
    return planet_types_arr, bands, SNR



#
# GAIN_UNCERTAINTIES(Bands vs Planet Types)
#

def colormap_bands_ptypes_parameters(mode="multi", Nmax=10, instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, apodizer="NO_SP", strehl="JQ1", coronagraph=None, systematics=False, PCA=False, PCA_mask=False, N_PCA=20, Rc=100, filter_type="gaussian", planet_types=planet_types):
    
    def normalize_if_possible(x):
        x    = np.asarray(x, dtype=float)
        xmax = np.nanmax(x)
        return x / xmax if np.isfinite(xmax) and xmax > 0 else x
    
    def replace_nonpositive_by_smallest_positive(x):
        x   = np.asarray(x, dtype=float).copy()
        pos = np.isfinite(x) & (x > 0)
        if np.any(pos):
            xmin    = np.nanmin(x[pos])
            x[~pos] = xmin
        else:
            x[:] = np.nan
        return x
    
    def sigma_to_gain(x):
        x     = np.asarray(x, dtype=float)
        valid = np.isfinite(x) & (x > 0)
        out   = np.full_like(x, np.nan)
        if np.any(valid):
            xmin       = np.nanmin(x[valid])
            out[valid] = xmin / x[valid]
        return out
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    elif config_data["base"]=="ground":
        tellurics = True
        
    # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotational broadening with Vsini)
    lmin_model = 0.9*config_data["lambda_range"]["lambda_min"] # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*config_data["lambda_range"]["lambda_max"] # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                      # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model)   # [µm] Model wavelength axis (with constant dl step)
    
    # K-band for photometry
    wave_K = get_wave_K()
    
    # Vega spectrum on K-band and instru-band [J/s/m2/µm]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)
    counts_vega_K   = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
    
    # get spectrum contribution + name model
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Load the planet table and convert it from QTable to pandas DataFrame
    table           = "Archive"
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename            = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}"
    planet_table        = load_planet_table(f"{filename}.ecsv")    
    planet_table["SNR"] = get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band="INSTRU")
    
    # Find planets based on mode
    planet_table_pd  = planet_table.to_pandas() # convert to pandas to find matching types
    selected_planets = set()                    # Set to store already assigned planets for mode = "unique"
    matching_planets = {planet_type: find_matching_planets(criteria, planet_table_pd, mode, selected_planets, Nmax=Nmax) for planet_type, criteria in planet_types.items()}
    
    # Table plot
    plot_matching_planets(matching_planets=matching_planets, exposure_time=exposure_time, mode=mode, planet_types=planet_types)
    
    # Definig arrays
    planet_types_list = [ptype for ptype, planets in matching_planets.items() if len(planets) > 0]
    planet_types_arr  = np.array(planet_types_list, dtype=object)
    bands             = np.array(list(config_data["gratings"].keys()), dtype=object)
    type_to_index     = {ptype: i for i, ptype in enumerate(planet_types_arr)}

    # Map each planet name to its row index in the table
    name_to_index = {name: i for i, name in enumerate(planet_table["PlanetName"])}
    
    # Store only lightweight (row_index, planet_type) pairs instead of copying full table rows
    selected_entries = []
    for ptype in planet_types_arr:
        for planet in matching_planets[ptype]:
            idx = name_to_index[planet["PlanetName"]]
            selected_entries.append((idx, ptype))
    
    # Calculating uncertainties
    sigma_T       = np.zeros((len(bands), len(planet_types_arr)), dtype=float)
    sigma_lg      = np.zeros((len(bands), len(planet_types_arr)), dtype=float)
    sigma_vsini   = np.zeros((len(bands), len(planet_types_arr)), dtype=float)
    sigma_rv      = np.zeros((len(bands), len(planet_types_arr)), dtype=float)           
    for idx, ptype in tqdm(selected_entries, total=len(selected_entries), desc="Processing planets", unit="planet"):
        
        planet = planet_table[idx]
        itype  = type_to_index[ptype]
        
        planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=instru, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_im_mag=True)        
        planet_spectrum.model        = planet_thermal.model # TODO: generalize to reflected models to (with mag(ref) < mag(th))
        mag_p                        = float(planet[f"PlanetINSTRUmag({instru})({spectrum_contributions})"])
        mag_s                        = float(planet[f"StarINSTRUmag({instru})"])
        name_bands, _, uncertainties = FastCurves(instru=instru, calculation="corner plot", systematics=systematics, mag_star=mag_s, band0="instru", exposure_time=exposure_time, mag_planet=mag_p, separation_planet=float(planet["AngSep"].value / 1000), planet_name="None", return_FastYield=False, show_plot=False, verbose=False, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, PCA=PCA, PCA_mask=PCA_mask, N_PCA=N_PCA, Rc=Rc, filter_type=filter_type)
        
        sigma_T_1D     = np.zeros((len(bands)))
        sigma_lg_1D    = np.zeros((len(bands)))
        sigma_vsini_1D = np.zeros((len(bands)))
        sigma_rv_1D    = np.zeros((len(bands)))
        band_to_idx    = {band_name: k for k, band_name in enumerate(name_bands)}
        for iband, band in enumerate(bands):
            u                  = uncertainties[band_to_idx[band]]
            sigma_T_1D[iband]  = u[0]
            sigma_lg_1D[iband] = u[1]
            if len(u) == 3:
                sigma_rv_1D[iband] = u[2]
            else:
                sigma_vsini_1D[iband] = u[2]
                sigma_rv_1D[iband]    = u[3]
        
        sigma_T_1D     = normalize_if_possible(sigma_T_1D)
        sigma_lg_1D    = normalize_if_possible(sigma_lg_1D)
        sigma_vsini_1D = normalize_if_possible(sigma_vsini_1D)
        sigma_rv_1D    = normalize_if_possible(sigma_rv_1D)
        
        sigma_T[:, itype]     += sigma_T_1D     / len(matching_planets[planet_types_arr[itype]])
        sigma_lg[:, itype]    += sigma_lg_1D    / len(matching_planets[planet_types_arr[itype]])
        sigma_vsini[:, itype] += sigma_vsini_1D / len(matching_planets[planet_types_arr[itype]])
        sigma_rv[:, itype]    += sigma_rv_1D    / len(matching_planets[planet_types_arr[itype]])

    for itype in range(len(planet_types_arr)):
        sigma_T[:, itype]     = replace_nonpositive_by_smallest_positive(sigma_T[:, itype])
        sigma_lg[:, itype]    = replace_nonpositive_by_smallest_positive(sigma_lg[:, itype])
        sigma_vsini[:, itype] = replace_nonpositive_by_smallest_positive(sigma_vsini[:, itype])
        sigma_rv[:, itype]    = replace_nonpositive_by_smallest_positive(sigma_rv[:, itype])
    
    gain_sigma_T     = np.zeros_like(sigma_T)
    gain_sigma_lg    = np.zeros_like(sigma_lg)
    gain_sigma_vsini = np.zeros_like(sigma_vsini)
    gain_sigma_rv    = np.zeros_like(sigma_rv)
    
    for itype in range(len(planet_types_arr)):
        gain_sigma_T[:, itype]     = sigma_to_gain(sigma_T[:, itype])
        gain_sigma_lg[:, itype]    = sigma_to_gain(sigma_lg[:, itype])
        gain_sigma_vsini[:, itype] = sigma_to_gain(sigma_vsini[:, itype])
        gain_sigma_rv[:, itype]    = sigma_to_gain(sigma_rv[:, itype])
    
    # Plot
    planet_types_arr_idx          = np.arange(len(planet_types_arr))
    planet_types_arr_idx_extended = np.arange(-1, len(planet_types_arr)+1)
    bands_idx                     = np.arange(len(bands))
    bands_idx_extended            = np.arange(-1, len(bands)+1)

    # Création de la figure avec subplots
    cmap      = plt.get_cmap(cmap_colormaps)
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
        contours                          = ax.contour(planet_types_arr_mesh, bands_mesh, data_extended, levels=contour_levels, colors='k', linewidths=1., alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=10, fmt="%d%%")  # Augmenté pour plus de lisibilité
        
        # Axes et labels améliorés
        ax.set_xlim(-0.5, len(planet_types_arr)-0.5)
        ax.set_xticks(planet_types_arr_idx)
        ax.set_xticklabels(planet_types_arr, rotation=40, ha="right", fontsize=18)  # Augmentation de la taille
    
        ax.set_ylim(-0.5, len(bands)-0.5)
        ax.set_yticks(bands_idx)
        ax.set_yticklabels(bands, fontsize=18)  # Augmentation de la taille
            
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
    cbar.ax.text(1.2,  1.05, 'High precision', ha='center', va='bottom', fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='red')
    cbar.ax.text(1.2, -0.05, 'Poor precision', ha='center', va='top',    fontsize=14, transform=cbar.ax.transAxes, weight='bold', color='blue')
    
    fig.suptitle(f"Error fluctuations for {instru} ({'with' if tellurics else 'without'} tellurics absorption)\nin {spectrum_contributions} light with {thermal_model}+{reflected_model} models", fontsize=28, y=1.05)  # Position du titre remontée    
    filename = f"colormaps_bands_planets_snr/colormap_bands_ptypes_parameters_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}_{mode}"
    plt.savefig(colormaps_path + filename + ".png", format='png', bbox_inches='tight')
    plt.show()
    
    return planet_types_arr, bands, gain_sigma_T, gain_sigma_lg, gain_sigma_vsini, gain_sigma_rv



#
# GAIN_SNR(STAR_RV VS DELTA_RV)
#

def colormap_rv(instru="HARMONI", band="H", T_planet=T_earth, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, vsini_planet=vrot_earth, vsini_star=vrot_sun, spectrum_contributions="reflected", model="tellurics", airmass=airmass_earth, Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True, title=None, title_weight=None, num=100):
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
    
    # Raw and bandwidth wavelength axis
    R_band     = config_data['gratings'][band].R                   # [dimensionless] 
    lmin_band  = config_data['gratings'][band].lmin                # [µm]
    lmax_band  = config_data['gratings'][band].lmax                # [µm]
    wave_band  = get_wave_band(config_data=config_data, band=band) # [µm]
    dwave_band = np.gradient(wave_band)                            # [µm/bin]
    lmin_model = 0.9*lmin_band                                     # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax_band                                     # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                          # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model)       # [µm] Model wavelength axis (with constant dl step)
    
    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)
    star_spectrum = star_spectrum.broad(vsini_star)                                 # Broadening the spectrum
    star          = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    
    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0)
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    planet_spectrum = planet_spectrum.broad(vsini_planet)                               # Broadening the spectrum
    planet          = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)

    # To be homogenous to photons
    star.flux   = star.flux   * wave_instru # [J/s/m2/µm] => propto [ph/µm]
    planet.flux = planet.flux * wave_instru # [J/s/m2/µm] => propto [ph/µm]
    
    # Geting trans model (with tellurics or not)
    trans = get_transmission(instru=instru, wave_band=wave_band, band=band, tellurics=tellurics, apodizer="NO_SP") # the apodizer does not matter
    
    # Defining arrays
    rv_star_arr  = np.linspace(-100, 100, num)
    delta_rv_arr = np.linspace(-100, 100, num)
    SNR           = np.zeros((num, num))
    lost_signal   = np.zeros((num, num))
    
    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(rv_star_arr=rv_star_arr, delta_rv_arr=delta_rv_arr, R_band=R_band, wave_band=wave_band, dwave_band=dwave_band, star=star, planet=planet, trans=trans, Rc=Rc, filter_type=filter_type, stellar_halo_photon_noise_limited=stellar_halo_photon_noise_limited)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, SNR_1D, lost_signal_1D) in tqdm(pool.imap_unordered(process_colormap_rv, range(num)), total=num, desc=f"colormap_rv(instru={instru}, band={band}, model={model}, Rc={Rc})",):
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    SNR = SNR / np.nanmax(SNR)
        
    # Plots
    for plot in plots_colormaps:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel(r" Planet-star differential radial velocity $\Delta rv$ [km/s]", fontsize=16)
        plt.ylabel(r"Star radial velocity $rv_{\star}$ [km/s]",                     fontsize=16)
        plt.xlim(delta_rv_arr[0], delta_rv_arr[-1])
        plt.ylim(rv_star_arr[0], rv_star_arr[-1])
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
        mesh = plt.pcolormesh(delta_rv_arr, rv_star_arr, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        # Contours
        cs = plt.contour(delta_rv_arr, rv_star_arr, data, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        #cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=16, fontsize=16)
        
        # c/R
        c_kms      = c / 1000       # [km/s]
        c_over_R   = c_kms / R_band # [km/s]   
        line_color = "black"
        alpha      = 0.4444
        line_style = "-"
        line_width = 4.444
        plt.axvline(-c_over_R, color=line_color, linestyle=line_style, linewidth=line_width, alpha=alpha, zorder=5)
        plt.axvline(+c_over_R, color=line_color, linestyle=line_style, linewidth=line_width, alpha=alpha, zorder=5)
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5)
        if c_over_R < 10:
            plt.text(-c_over_R, 1.025, r"$-c/R$", color="black", fontsize=12, ha="right", va="center",  fontweight="bold", transform=ax.get_xaxis_transform())
            plt.text(+c_over_R, 1.025, r"$+c/R$", color="black", fontsize=12, ha="left",  va="center",  fontweight="bold", transform=ax.get_xaxis_transform())
        else:
            plt.text(-c_over_R, 1.025, r"$-c/R$", color="black", fontsize=12, ha="center", va="center", fontweight="bold", transform=ax.get_xaxis_transform())
            plt.text(+c_over_R, 1.025, r"$+c/R$", color="black", fontsize=12, ha="center", va="center", fontweight="bold", transform=ax.get_xaxis_transform())
        plt.text(0.98,      0.98,  rf"$c/R \approx {c_over_R:.1f}\,\mathrm{{km\,s^{{-1}}}}$", color="white", fontsize=14, ha="right",  va="top", fontweight="bold", bbox=bbox, transform=plt.gca().transAxes)
            
        noise_regime = "stellar halo photon noise regime" if stellar_halo_photon_noise_limited else "detector noise regime"
        
        # Title
        if title is None:
            tell       = "with tellurics absorption" if tellurics else "without tellurics absorption"
            title_text = (f"{instru} {band}-band {'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, $T_p$={T_planet}K")
        else:
            title_text = title
        plt.title(title_text, fontsize=16, pad=24, weight=title_weight)
        plt.tight_layout()
        filename = f"colormaps_rv/Colormap_rv_{plot}_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_broad{vsini_planet}kms_{noise_regime.replace(' ', '_')}"
        plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
    
    return delta_rv_arr, rv_star_arr, SNR, lost_signal

def process_colormap_rv(i):
    
    rv_star_arr                       = _CM_CTX["rv_star_arr"]  # [km/s]
    delta_rv_arr                      = _CM_CTX["delta_rv_arr"] # [km/s]
    R_band                            = _CM_CTX["R_band"]       # [dimensionless]
    wave_band                         = _CM_CTX["wave_band"]    # [µm]
    dwave_band                        = _CM_CTX["dwave_band"]   # [µm/bin]
    star                              = _CM_CTX["star"]         # propto [ph/µm]
    planet                            = _CM_CTX["planet"]       # propto [ph/µm]
    trans                             = _CM_CTX["trans"]
    Rc                                = _CM_CTX["Rc"]
    filter_type                       = _CM_CTX["filter_type"]
    stellar_halo_photon_noise_limited = _CM_CTX["stellar_halo_photon_noise_limited"]
    
    rv_star = rv_star_arr[i]
    
    # Preparing the star spectrum
    star_shift       = star.doppler_shift(rv_star)                                         # [ph/µm]
    star_shift       = star_shift.degrade_resolution(wave_band, renorm=False).flux         # [ph/µm]
    star_shift       = star_shift * dwave_band                                             # [ph/µm] => [ph/bin]
    star_HF, star_LF = filtered_flux(star_shift, R=R_band, Rc=Rc, filter_type=filter_type) # [ph/bin]
    
    SNR_1D         = np.zeros((len(delta_rv_arr))) + np.nan
    lost_signal_1D = np.zeros((len(delta_rv_arr))) + np.nan
    for j, delta_rv in enumerate(delta_rv_arr):
        
        # Preparing the planet spectrum
        planet_shift         = planet.doppler_shift(rv_star + delta_rv)                              # [ph/µm]
        planet_shift         = planet_shift.degrade_resolution(wave_band, renorm=False).flux         # [ph/µm]
        planet_shift         = planet_shift * dwave_band                                             # [ph/µm] => [ph/bin]
        planet_HF, planet_LF = filtered_flux(planet_shift, R=R_band, Rc=Rc, filter_type=filter_type) # [ph/bin]

        template = trans*planet_HF 
        template = template / np.sqrt(np.nansum(template**2))
        alpha    = np.nansum(trans*planet_HF * template)
        beta     = np.nansum(trans*star_HF*planet_LF/star_LF * template)
        if stellar_halo_photon_noise_limited:
            sigma_CCF = np.sqrt(np.nansum(trans*star_shift * template**2)) # stellar halo photon noise
        else:
            sigma_CCF = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / sigma_CCF
        lost_signal_1D[j] = beta / alpha
                    
    return i, SNR_1D, lost_signal_1D



#
# GAIN_SNR(STAR_Vsini VS PLANET_Vsini)
#

def colormap_vrot(instru="HARMONI", band="H", T_planet=T_earth, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, delta_rv=drv_earth, inc=90, spectrum_contributions="reflected", model="tellurics", airmass=airmass_earth, Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True, title=None, title_weight=None, num=100):
    """
    https://www.aanda.org/articles/aa/pdf/2022/03/aa42314-21.pdf
    """
    
    sini = np.sin(np.deg2rad(inc))
    
    # Get instru specs
    config_data = get_config_data(instru)
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
    
    # Raw and bandwidth wavelength axis
    R_band     = config_data['gratings'][band].R                   # [dimensionless] 
    lmin_band  = config_data['gratings'][band].lmin                # [µm]
    lmax_band  = config_data['gratings'][band].lmax                # [µm]
    wave_band  = get_wave_band(config_data=config_data, band=band) # [µm]
    dwave_band = np.gradient(wave_band)                            # [µm/bin]
    lmin_model = 0.9*lmin_band                                     # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax_band                                     # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                          # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model)       # [µm] Model wavelength axis (with constant dl step)
    
    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)

    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = None
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)  # [J/s/m2/µm]
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = planet_spectrum.doppler_shift(delta_rv)                          # Shifting the spectrum
        albedo_spectrum = None
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    
    # Geting trans model (with tellurics or not)
    trans = get_transmission(instru=instru, wave_band=wave_band, band=band, tellurics=tellurics, apodizer="NO_SP") # the apodizer does not matter
    
    # Defining arrays
    vrot_star_arr   = np.linspace(0, 100, num)
    vrot_planet_arr = np.linspace(0, 100, num)
    SNR              = np.zeros((num, num))
    lost_signal      = np.zeros((num, num))
    
    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(vrot_star_arr=vrot_star_arr, vrot_planet_arr=vrot_planet_arr, delta_rv=delta_rv, sini=sini, R_band=R_band, wave_band=wave_band, dwave_band=dwave_band, wave_model=wave_model, wave_instru=wave_instru, star_spectrum=star_spectrum, planet_spectrum=planet_spectrum, albedo_spectrum=albedo_spectrum, trans=trans, Rc=Rc, filter_type=filter_type, stellar_halo_photon_noise_limited=stellar_halo_photon_noise_limited)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, SNR_1D, lost_signal_1D) in tqdm(pool.imap_unordered(process_colormap_vrot, range(num)), total=num, desc=f"colormap_vrot(instru={instru}, band={band}, model={model}, delta_rv={delta_rv}, inc={inc}, Rc={Rc})",):
            SNR[i, :]         = SNR_1D
            lost_signal[i, :] = lost_signal_1D
    
    # Normalizing
    SNR = np.maximum(SNR, 0)
    SNR = SNR / np.nanmax(SNR)
    
    # Plots
    for plot in plots_colormaps:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel(r"Planet equatorial rotation velocity $V_p$ [km/s]",     fontsize=16)
        plt.ylabel(r"Star equatorial rotation velocity $V_{\star}$ [km/s]", fontsize=16)
        plt.xlim(vrot_planet_arr[0], vrot_planet_arr[-1])
        plt.ylim(vrot_star_arr[0], vrot_star_arr[-1])
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
        mesh = plt.pcolormesh(vrot_planet_arr, vrot_star_arr, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        # Contours
        cs = plt.contour(vrot_planet_arr, vrot_star_arr, data, colors='k', linewidths=0.5, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
        
        # Colorbar
        ax   = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        #cbar.set_ticks(contour_levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar_label, rotation=270, labelpad=14, fontsize=14)
        
        # c/R
        c_kms      = c/1000 # [km/s]
        c_over_R   = c_kms / R_band        
        line_color = "black"
        alpha      = 0.4444
        line_style = "-"
        line_width = 4.444
        plt.axvline(+c_over_R, color=line_color, linestyle=line_style, linewidth=line_width, alpha=alpha, zorder=5)
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5)
        plt.text(+c_over_R, 1.025, r"$c/R$",                                                  color="black", fontsize=12, ha="center", va="center", fontweight="bold", transform=ax.get_xaxis_transform())
        plt.text(0.98,      0.98,  rf"$c/R \approx {c_over_R:.1f}\,\mathrm{{km\,s^{{-1}}}}$", color="white", fontsize=14, ha="right",  va="top",    fontweight="bold", bbox=bbox, transform=plt.gca().transAxes)
        
        noise_regime = "stellar halo photon noise regime" if stellar_halo_photon_noise_limited else "detector noise regime"
        
        # Title
        if title is None:
            tell       = "with tellurics absorption" if tellurics else "without tellurics absorption"
            title_text = (f"{instru} {band}-band {'S/N' if plot=='SNR' else 'Lost signal'} fluctuations in {noise_regime} ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, $T_p$={T_planet}K, "r"$\Delta$rv="f"{delta_rv}km/s and i={inc:.0f}°")
        else:
            title_text = title
        plt.title(title_text, fontsize=16, pad=24, weight=title_weight)

        plt.tight_layout()
        filename = f"colormaps_vrot/Colormap_vrot_{plot}_{instru}_{band}_{spectrum_contributions}_{model}_Rc{Rc}_Tp{T_planet}K_Ts{T_star}K_drv{delta_rv}kms_inc{inc}deg_{noise_regime.replace(' ', '_')}"
        plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
        plt.show()
    
    return vrot_planet_arr, vrot_star_arr, SNR, lost_signal

def process_colormap_vrot(i):
    
    vrot_star_arr                     = _CM_CTX["vrot_star_arr"]    # [km/s]
    vrot_planet_arr                   = _CM_CTX["vrot_planet_arr"]  # [km/s]
    delta_rv                          = _CM_CTX["delta_rv"]         # [km/s]
    sini                              = _CM_CTX["sini"]             # [dimensionless]
    R_band                            = _CM_CTX["R_band"]           # [dimensionless]
    wave_band                         = _CM_CTX["wave_band"]        # [µm]
    dwave_band                        = _CM_CTX["dwave_band"]       # [µm/bin]
    wave_model                        = _CM_CTX["wave_model"]       # [µm]
    wave_instru                       = _CM_CTX["wave_instru"]      # [µm]
    star_spectrum                     = _CM_CTX["star_spectrum"]    # propto [J/µm]
    planet_spectrum                   = _CM_CTX["planet_spectrum"]  # propto [J/µm]
    albedo_spectrum                   = _CM_CTX["albedo_spectrum"]  # [dimensionless]
    trans                             = _CM_CTX["trans"]
    Rc                                = _CM_CTX["Rc"]
    filter_type                       = _CM_CTX["filter_type"]
    stellar_halo_photon_noise_limited = _CM_CTX["stellar_halo_photon_noise_limited"]
    
    vrot_star  = vrot_star_arr[i]
    vsini_star = vrot_star * sini 
    
    # Preparing the star spectrum
    star_spectrum_broad = star_spectrum.broad(vsini_star)                                       # [J/µm]
    star_broad          = star_spectrum_broad.interpolate_wavelength(wave_instru, renorm=False) # [J/µm] Interpolating on wave_instru (constant R_model)
    star_broad.flux     = star_broad.flux * wave_instru                                         # [J/µm] => propto [ph/µm]
    star_broad          = star_broad.degrade_resolution(wave_band, renorm=False).flux           # [ph/µm]
    star_broad          = star_broad * dwave_band                                               # [ph/µm] => [ph/bin]
    star_HF, star_LF    = filtered_flux(star_broad, R=R_band, Rc=Rc, filter_type=filter_type)   # [ph/bin]
    
    # Preparing the planet spectrum (if necessary)
    if planet_spectrum is None:
        star_spectrum_broad_ref = star_spectrum.broad(vrot_star)          # [J/µm]
        planet_spectrum         = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum_broad_ref.flux, R=albedo_spectrum.R, T=albedo_spectrum.T, lg=albedo_spectrum.lg, model=albedo_spectrum.model, rv=0, vsini=0)
        planet_spectrum         = planet_spectrum.doppler_shift(delta_rv) # Shifting the spectrum

    SNR_1D         = np.zeros((len(vrot_planet_arr))) + np.nan
    lost_signal_1D = np.zeros((len(vrot_planet_arr))) + np.nan
    for j, vrot_planet in enumerate(vrot_planet_arr):
        
        vsini_planet = vrot_planet * sini
        
        # Preparing the planet spectrum
        planet_spectrum_broad = planet_spectrum.broad(vsini_planet)                                     # [J/µm]
        planet_broad          = planet_spectrum_broad.interpolate_wavelength(wave_instru, renorm=False) # [J/µm] Interpolating on wave_instru (constant R_model)
        planet_broad.flux     = planet_broad.flux * wave_instru                                         # [J/µm] => propto [ph/µm]
        planet_broad          = planet_broad.degrade_resolution(wave_band, renorm=False).flux           # [ph/µm]
        planet_broad          = planet_broad * dwave_band                                               # [ph/µm] => [ph/bin]
        planet_HF, planet_LF  = filtered_flux(planet_broad, R=R_band, Rc=Rc, filter_type=filter_type)   # [ph/bin]
        
        template = trans*planet_HF 
        template = template / np.sqrt(np.nansum(template**2))
        alpha    = np.nansum(trans*planet_HF * template)
        beta     = np.nansum(trans*star_HF*planet_LF/star_LF * template)
        if stellar_halo_photon_noise_limited:
            sigma_CCF = np.sqrt(np.nansum(trans*star_broad * template**2)) # stellar halo photon noise
        else:
            sigma_CCF = 1. # wavelength and resolution-independent limiting noise (e.g. RON and dark current - detector noise - domination)
        SNR_1D[j]         = (alpha - beta) / (sigma_CCF)
        lost_signal_1D[j] = beta / alpha
                    
    return i, SNR_1D, lost_signal_1D



#
# GAIN_SNR(PHASE VS MAXSEP(SMA) VS INC)
#
def colormap_maxsep_phase_inc(instru="HARMONI", band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None, T_planet=T_earth, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, vrot_planet=vrot_earth, vrot_star=vrot_sun, M_planet=1, M_star=1, d=10, spectrum_contributions="reflected", model="tellurics", airmass=airmass_earth, Rc=100, filter_type="gaussian", title_list=None, title_weight=None, num=100):
    
    sep_min = 10/1000 # [arcsec]
    sep_max = 1       # [arcsec]
    
    # Marginalization helper functions
    def bin_widths(x):
        """Approximate integration weights Δx for a 1D grid x (monotonic)."""
        x           = np.asarray(x)
        edges       = np.empty(x.size + 1)
        edges[1:-1] = 0.5 * (x[1:] + x[:-1])
        edges[0]    = x[0] - (edges[1] - x[0])
        edges[-1]   = x[-1] + (x[-1] - edges[-2])
        w           = edges[1:] - edges[:-1]
        return np.clip(w, 0, np.inf)
    
    def weighted_nanmean(a, axis, w):
        """NaN-safe weighted mean along 'axis'."""
        a           = np.asarray(a)
        w           = np.asarray(w)
        shape       = [1] * a.ndim
        shape[axis] = w.size
        w_b         = w.reshape(shape)
        mask        = np.isfinite(a)
        num         = np.nansum(a * w_b, axis=axis)
        den         = np.nansum(mask * w_b, axis=axis)
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
    
    def marginalize(arr, axis, w_rho=None, w_phi=None, w_inc=None):
        """
        arr axis: 0=rho_max, 1=phase, 2=inc
        Weighted means use w_rho, w_phi, w_inc.
        """
        out = arr
        for ax in sorted(axis, reverse=True):  # reduce from high to low to avoid index shifts
            w   = {0: w_rho, 1: w_phi, 2: w_inc}.get(ax, None)
            out = weighted_nanmean(out, axis=ax, w=w) if w is not None else np.nanmean(out, axis=ax)
        return out
    
    # Get instru specs
    config_data = get_config_data(instru)
    sep_unit    = config_data["sep_unit"]
    if config_data["type"]=="imager":
        raise KeyError(f"{instru} is not a spectrograph but an {config_data['type']}")

    # tellurics (or not)
    if config_data["base"]=="space":
        tellurics = False
    else:
        tellurics = True
        
    # Raw and bandwidth wavelength axis
    R_band     = config_data['gratings'][band].R                   # [dimensionless] 
    lmin_band  = config_data['gratings'][band].lmin                # [µm]
    lmax_band  = config_data['gratings'][band].lmax                # [µm]
    wave_band  = get_wave_band(config_data=config_data, band=band) # [µm]
    dwave_band = np.gradient(wave_band)                            # [µm/bin]
    R_model    = get_R_instru(config_data)                         # [dimensionless]
    lmin_model = 0.9*lmin_band                                     # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax_band                                     # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                          # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model)       # [µm] Model wavelength axis (with constant dl step)
    
    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)

    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = None
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=instru)  # [J/s/m2/µm]
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        albedo_spectrum = None
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    
    # Geting trans model (with tellurics or not)
    trans = get_transmission(instru=instru, wave_band=wave_band, band=band, tellurics=tellurics, apodizer="NO_SP") # the apodizer does not matter
    
    # Load the PSF profile
    PSF_profile, fraction_core, separation, _ = get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, instru=instru, separation_planet=sep_max*1000 if sep_unit=="mas" else sep_max)
    
    # Compute the core flux fraction
    if coronagraph is None:
        fraction_core = np.zeros(len(separation)) + fraction_core
    else:
        separation0, fraction_core0, radial_transmission0 = fits.getdata(f"{sim_data_path}/PSF/PSF_{instru}/fraction_core_radial_transmission_{band}_{coronagraph}_{strehl}_{apodizer}.fits")
        fraction_core0                                    = fraction_core0 * radial_transmission0 # taking into account the coronagraph transmission, not needed for the star, as it is a common scaling factor (star_transmission)
        fraction_core                                     = interp1d(separation0, fraction_core0, bounds_error=False, fill_value=np.nan)(separation)
        fraction_core[separation > separation0[-1]]       = fraction_core0[-1] # flat extrapolation
    
    # Considering in arcsec only
    if sep_unit == "mas":
        separation = separation / 1000 # [mas] => [arcsec] = [AU/pc]
    
    # Build interpolation functions (in logspace)
    valid                    = np.isfinite(separation) & np.isfinite(PSF_profile) & (separation > 0) & (PSF_profile > 0) & (fraction_core > 0)
    log_PSF_profile_interp   = interp1d(np.log10(separation[valid]), np.log10(PSF_profile[valid]),   bounds_error=False, fill_value="extrapolate")
    log_fraction_core_interp = interp1d(np.log10(separation[valid]), np.log10(fraction_core[valid]), bounds_error=False, fill_value="extrapolate")
    
    # Defining arrays
    max_sep_arr = np.logspace(np.log10(sep_min), np.log10(sep_max), num)                 # [arcsec]: dim i
    phase_arr   = np.linspace(0,                 2*np.pi,           num, endpoint=False) # [rad]:    dim j
    inc_arr     = np.linspace(0,                 90,                num)                 # [deg]:    dim k
    SNR_halo_3D = np.zeros((num, num, num))
    SNR_syst_3D = np.zeros((num, num, num))
    
    # SMA axis in [AU] (d in [pc])
    SMA_arr = max_sep_arr * d
            
    # Compute phase angle and Lambert phase function (if needed)
    if spectrum_contributions == "reflected":
        
        star_spectrum_broad_ref  = star_spectrum.broad(vrot_star) # [J/µm]
        
        a_arr = np.arccos(- np.sin(inc_arr[None, :]*np.pi/180) * np.cos(phase_arr[:, None]) ) # Phase angle
        g_arr = ( np.sin(a_arr) + (np.pi - a_arr) * np.cos(a_arr) ) / np.pi                   # Lambert phase function
        
        # Plot the Lambert phase function as a function of phase and inclination
        plt.figure(figsize=(10, 6), dpi=300)
        plt.xlabel(r"Inclination $i$ [°]", fontsize=14)
        plt.ylabel(r"Phase $\varphi$ [rad]", fontsize=14)
        plt.xlim(inc_arr[0],   inc_arr[-1])
        plt.ylim(phase_arr[0], phase_arr[-1])
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        mesh = plt.pcolormesh(inc_arr, phase_arr, g_arr, shading='auto', vmin=np.nanmin(g_arr), vmax=np.nanmax(g_arr))
        cs   = plt.contour(inc_arr,    phase_arr, g_arr, colors='k', linewidths=0.6, alpha=0.7)
        plt.clabel(cs, inline=True, fontsize=8)
        ax   = plt.gca()
        cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
        cbar.minorticks_on()
        cbar.set_ticks(contour_levels/100)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(r"Lambert phase function $g(\alpha)$", rotation=270, labelpad=20, fontsize=14)
        #plt.title("Lambert phase function", fontsize=16, pad=14)
        plt.tight_layout()
        plt.show()
    else:
        star_spectrum_broad_ref = None
        g_arr                   = None
    
    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(SMA_arr=SMA_arr, max_sep_arr=max_sep_arr, phase_arr=phase_arr, inc_arr=inc_arr, g_arr=g_arr, log_PSF_profile_interp=log_PSF_profile_interp, log_fraction_core_interp=log_fraction_core_interp, M_star=M_star, M_planet=M_planet, vrot_star=vrot_star, vrot_planet=vrot_planet, spectrum_contributions=spectrum_contributions, star_spectrum=star_spectrum, star_spectrum_broad_ref=star_spectrum_broad_ref, planet_spectrum=planet_spectrum, albedo_spectrum=albedo_spectrum, wave_model=wave_model, wave_band=wave_band, dwave_band=dwave_band, trans=trans, R_band=R_band, Rc=Rc, filter_type=filter_type)
    
    # Parallel calculations
    print()
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (k, SNR_halo_2D, SNR_syst_2D) in tqdm(pool.imap_unordered(process_colormap_maxsep_phase_inc, range(num)), total=num, desc=f"process_colormap_maxsep_phase_inc(instru={instru}, band={band}, apodizer={apodizer}, coronagraph={coronagraph})",):
            SNR_halo_3D[:, :, k] = SNR_halo_2D
            SNR_syst_3D[:, :, k] = SNR_syst_2D
        
    # Normalize the 3D SNR cubes
    SNR_halo_3D = np.nan_to_num(SNR_halo_3D, nan=0.0, posinf=0.0, neginf=0.0)
    SNR_syst_3D = np.nan_to_num(SNR_syst_3D, nan=0.0, posinf=0.0, neginf=0.0)
    SNR_halo_3D = SNR_halo_3D / np.nanmax(SNR_halo_3D)
    SNR_syst_3D = SNR_syst_3D / np.nanmax(SNR_syst_3D)
    
    # --- Weights ---
    # Uniform SMA distribution (fixed distance) => rho_max uniform in linear space => use Δrho weights
    w_rho = bin_widths(max_sep_arr)
    w_rho = w_rho / np.nansum(w_rho)
    
    # Uniform phase over the orbit => use Δphi weights
    w_phi = bin_widths(phase_arr)
    w_phi = w_phi / np.nansum(w_phi)
    
    # Isotropic orientations => p(i) ∝ sin(i), integrated in radians
    di_rad = np.deg2rad(bin_widths(inc_arr))
    w_inc  = np.sin(np.deg2rad(inc_arr)) * di_rad
    w_inc  = w_inc / np.nansum(w_inc)
    
    # Marginalization wrapper
    marginalize_func = lambda A, axis: marginalize(A, axis, w_rho=w_rho, w_phi=w_phi, w_inc=w_inc)

    # Plot settings
    params       = [max_sep_arr, phase_arr, inc_arr]
    log_param    = [True, False, False]
    names_param  = [r"$\rho_{max}$ [arcsec]",                  r"$\varphi$ [rad]", r"$i$ [°]"]
    labels_param = [r"$\rho_{max}$ [arcsec] or SMA/d [AU/pc]", r"$\varphi$ [rad]", r"$i$ [°]"]
    Ndim         = len(params)
    xmin         = np.array([np.nanmin(p) for p in params])
    xmax         = np.array([np.nanmax(p) for p in params])
    cmap         = plt.get_cmap(cmap_colormaps)
    vmin         = 0
    vmax         = 100
    SNR_3D_list  = [SNR_halo_3D,                                   SNR_syst_3D]
    if title_list is None:
        title_list   = [f"{instru}: stellar-halo-photon-noise regime", f"{instru}: systematics noise regime"]
    
    # Generate the plots
    for idx_snr in range(len(SNR_3D_list)):
        SNR_3D         = SNR_3D_list[idx_snr]
        optimal_values = np.zeros((Ndim))
        for idim in range(Ndim):
            snr                  = marginalize_func(SNR_3D, axis=tuple(kdim for kdim in range(Ndim) if kdim != idim))
            optimal_values[idim] = params[idim][snr.argmax()]
        fig, axes = plt.subplots(Ndim, Ndim, figsize=(10, 10), dpi=300)
        plt.subplots_adjust(wspace=0.07, hspace=0.08)
        for idim in range(Ndim):
            for jdim in range(Ndim):
                ax = axes[idim, jdim]
                if jdim > idim:  # Upper triangular part of the matrix
                    ax.axis("off")
                    continue
                elif idim == jdim:  # Histograms on the diagonal
                    snr = marginalize_func(SNR_3D, axis=tuple(kdim for kdim in range(Ndim) if kdim != idim))
                    snr = snr / np.nanmax(snr)
                    ax.plot(params[idim], 100*snr, color="k")
                    ax.axvline(optimal_values[idim], color="k", linestyle="--")
                    ax.set_title(f"{names_param[idim]} = {optimal_values[idim]:.2f}", fontsize=14)
                    if idim == Ndim-1:
                        ax.set_xlabel(labels_param[idim], fontsize=14)
                    ax.set_xlim(xmin[idim], xmax[idim])
                    ax.set_ylim(vmin, vmax)
                    if idim == Ndim//2:
                        ax.set_ylabel("$GAIN_{S/N}$ [%]", fontsize=14, rotation=270, labelpad=15)
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.tick_params(axis="y", which="both", left=False, right=True)
                    if log_param[idim]:
                        ax.set_xscale("log")
                elif jdim < idim:  # Contour plots in the lower triangular part
                    snr = marginalize_func(SNR_3D, axis=tuple(kdim for kdim in range(Ndim) if kdim not in (idim, jdim)))
                    snr = snr.T / np.nanmax(snr)  # (len(y), len(x))
                    ax.pcolormesh(params[jdim], params[idim], 100*snr, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
                    cs = ax.contour(params[jdim], params[idim], 100*snr, colors='k', linewidths=0.5, alpha=0.7)
                    plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
                    ax.axvline(optimal_values[jdim], color="k", linestyle="--")
                    ax.axhline(optimal_values[idim], color="k", linestyle="--")
                    ax.plot(optimal_values[jdim], optimal_values[idim], "X", color="black")
                    if jdim == 0:
                        ax.set_ylabel(labels_param[idim], fontsize=14)
                    if idim == Ndim - 1:
                        ax.set_xlabel(labels_param[jdim], fontsize=14)
                    ax.set_xlim(xmin[jdim], xmax[jdim])
                    ax.set_ylim(xmin[idim], xmax[idim])
                    if log_param[jdim]:
                        ax.set_xscale("log")
                    if log_param[idim]:
                        ax.set_yscale("log")
                if idim < Ndim - 1:
                    ax.set_xticklabels([])
                if (jdim > 0) and (idim != jdim):
                    ax.set_yticklabels([])
        
        # Title
        fig.suptitle(title_list[idx_snr], weight=title_weight, fontsize=16, x=0.67, y=0.86)
        plt.show()
    
    return max_sep_arr, phase_arr, inc_arr, SNR_halo_3D, SNR_syst_3D

def process_colormap_maxsep_phase_inc(k):
    SMA_arr                  = _CM_CTX["SMA_arr"]
    max_sep_arr              = _CM_CTX["max_sep_arr"]
    phase_arr                = _CM_CTX["phase_arr"]
    inc_arr                  = _CM_CTX["inc_arr"]
    g_arr                    = _CM_CTX["g_arr"]
    log_PSF_profile_interp   = _CM_CTX["log_PSF_profile_interp"]
    log_fraction_core_interp = _CM_CTX["log_fraction_core_interp"]
    M_star                   = _CM_CTX["M_star"]
    M_planet                 = _CM_CTX["M_planet"]
    vrot_star                = _CM_CTX["vrot_star"]
    vrot_planet              = _CM_CTX["vrot_planet"]
    spectrum_contributions   = _CM_CTX["spectrum_contributions"]
    star_spectrum            = _CM_CTX["star_spectrum"]
    star_spectrum_broad_ref  = _CM_CTX["star_spectrum_broad_ref"]
    planet_spectrum          = _CM_CTX["planet_spectrum"]
    albedo_spectrum          = _CM_CTX["albedo_spectrum"]
    wave_model               = _CM_CTX["wave_model"]
    wave_band                = _CM_CTX["wave_band"]
    dwave_band               = _CM_CTX["dwave_band"]
    trans                    = _CM_CTX["trans"]
    R_band                   = _CM_CTX["R_band"]
    Rc                       = _CM_CTX["Rc"]
    filter_type              = _CM_CTX["filter_type"]

    inc          = inc_arr[k]              # [°]   
    sini         = np.sin(np.deg2rad(inc)) # [dimensionless]
    vsini_star   = vrot_star   * sini      # [km/s]
    vsini_planet = vrot_planet * sini      # [km/s]

    # Preparing the star spectrum
    star_broad       = star_spectrum.broad(vsini_star)                                       # [J/µm]
    star_broad.flux  = star_broad.flux * wave_model                                          # [J/µm] => propto [ph/µm]
    star_broad       = star_broad.degrade_resolution(wave_band, renorm=False).flux           # [ph/µm]
    star_broad       = star_broad * dwave_band                                               # [ph/µm] => [ph/bin]
    star_HF, star_LF = filtered_flux(star_broad, R=R_band, Rc=Rc, filter_type=filter_type)   # [ph/bin]
    
    # Preparing the planet spectrum
    if planet_spectrum is None and spectrum_contributions == "reflected":
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum_broad_ref.flux, R=albedo_spectrum.R, T=albedo_spectrum.T, lg=albedo_spectrum.lg, model=albedo_spectrum.model, rv=0, vsini=0)
    planet_broad       = planet_spectrum.broad(vsini_planet)                                            # [J/µm]
    planet_broad.flux *= wave_model
    planet_broad       = planet_broad.degrade_resolution(wave_band, renorm=False)         # [ph/µm]
    planet_broad.flux  = planet_broad.flux * dwave_band                                             # [ph/µm] => [ph/bin]
    
    # HF/LF split
    planet_HF = planet_broad.copy()
    planet_LF = planet_broad.copy()
    planet_HF.flux, planet_LF.flux  = filtered_flux(planet_broad.flux, R=R_band, Rc=Rc, filter_type=filter_type) # [ph/bin]
    
    SNR_halo_2D = np.zeros((len(max_sep_arr), len(phase_arr))) + np.nan
    SNR_syst_2D = np.zeros((len(max_sep_arr), len(phase_arr))) + np.nan
    for i in range(len(max_sep_arr)):
        
        max_sep  = max_sep_arr[i] # [arcsec]
        SMA      = SMA_arr[i]     # [AU]
        
        # A_B      = 0.3
        # T_star   = T_sun  # [K]
        # R_star   = 1
        # T_planet = T_star * np.sqrt( R_star*R_sun / (2*SMA*AU) ) * (1 - A_B)**(1/4)
        
        for j in range(len(phase_arr)):
            
            phase = phase_arr[j] # [rad]
            
            # Computing delta rv (M_star in [M_sun], M_planet in [M_earth], G in [m3/kg/s2] and AU [in m])
            delta_rv = np.sqrt( G*(M_star*M_sun + M_planet*M_earth) / (SMA*AU) ) * sini * np.sin(phase) / 1000 # [km/s]
                        
            # Shifting planet spectra
            planet_HF_shift = planet_HF.doppler_shift(delta_rv).flux # Shifting the spectrum
            planet_LF_shift = planet_LF.doppler_shift(delta_rv).flux # Shifting the spectrum

            # Computing the angular separation
            sep = max_sep * np.sqrt( np.sin(phase)**2 + np.cos(phase)**2 * np.cos(np.deg2rad(inc))**2 ) # [arcsec] https://iopscience.iop.org/article/10.1088/0004-637X/729/1/74/pdf
            if (sep <= 0) or (not np.isfinite(sep)):
                continue
            
            # Computing PSF profile
            PSF_sep           = 10**log_PSF_profile_interp(np.log10(sep))
            fraction_PSF_sep  = 10**log_fraction_core_interp(np.log10(sep))
            if (PSF_sep <= 0) or (fraction_PSF_sep <= 0) or (not np.isfinite(PSF_sep)) or (not np.isfinite(fraction_PSF_sep)):
                continue
            
            # Computing SNR
            template = trans*planet_HF_shift
            template = template / np.sqrt(np.nansum(template**2))
            alpha    = np.nansum(trans*planet_HF_shift * template)
            beta     = np.nansum(trans*star_HF*planet_LF_shift/star_LF * template)
            if spectrum_contributions == "reflected":
                g                 = g_arr[j, k]  # [dimensionless]
                SNR_halo_2D[i, j] = g * fraction_PSF_sep / max_sep**2 * (alpha - beta) / np.sqrt(PSF_sep)
                SNR_syst_2D[i, j] = g * fraction_PSF_sep / max_sep**2 * (alpha - beta) / PSF_sep
            else:
                SNR_halo_2D[i, j] = fraction_PSF_sep * (alpha - beta) / np.sqrt(PSF_sep)
                SNR_syst_2D[i, j] = fraction_PSF_sep * (alpha - beta) / PSF_sep
    
    return k, SNR_halo_2D, SNR_syst_2D









######################## signal(MM) / signal(DI) = alpha - beta / delta: Resolution VS Temperature ###########################################################################################################################################################################################

def colormap_MM_DI_R_Tp(lmin=1, lmax=2.5, Rmin=1_000, Rmax=100_000, Tmin=200, Tmax=2_000, tellurics=True, T_star=T_sun, lg_planet=lg_earth, lg_star=lg_sun, delta_rv=drv_earth, vsini_planet=vrot_earth, vsini_star=vrot_sun, spectrum_contributions="thermal", model="BT-Settl", airmass=airmass_earth, Rc=100, filter_type="gaussian", num=100, title=None):    
    
    # Global model-bandwidth (with constant dl step, must be evenly spaced in order to create the model spectra, for the rotational broadening with Vsini)
    lmin_model = 0.9*lmin                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    lmax_model = 1.1*lmax                                    # [µm] a bit larger for doppler shifts and to avoid edge effects
    dl_model   = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)

    # Global instru-bandwidth (with constant resolution R_model) (intermediate wavelength axis with constant sampling resolution, between wave_model and wave_res)
    wave_instru = get_wavelength_axis_constant_R(lmin=lmin_model, lmax=lmax_model, R=R_model) # [µm] Model wavelength axis (with constant spectral resolution R_model)

    # Effective model range
    lmin_model = max(wave_model[0],  wave_instru[0])  # [µm] effective lmin 
    lmax_model = min(wave_model[-1], wave_instru[-1]) # [µm] effective lmax 

    # Getting star spectrum in [J/s/m2/µm]
    star_spectrum = load_star_spectrum(T_star, lg_star)
    star_spectrum = star_spectrum.interpolate_wavelength(wave_model, renorm=False)  # Interpolating on wave_model (constant dl)
    star_spectrum = star_spectrum.broad(vsini_star)                                 # Broadening the spectrum
    star          = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    star.flux     = star.flux * wave_instru                                         # [J/s/m2/µm] => propto [ph/µm]
    
    # Tellurics transmission spectrum (from SkyCalc), if needed  
    if tellurics :
        wave_tell, trans_tell = _load_tell_trans(airmass=1.0)
        trans_tell            = Spectrum(wavelength=wave_tell, flux=trans_tell).interpolate_wavelength(wave_output=wave_instru, renorm=False, fill_value=(trans_tell[0], trans_tell[-1])) 
    else:
        trans_tell = None
    
    # Defining arrayrs
    T_arr           = np.linspace(Tmin,           Tmax,           num=num)
    R_arr           = np.logspace(np.log10(Rmin), np.log10(Rmax), num=num)
    residual_signal = np.zeros((num, num))
    
    # Precomputing star, trans spectra for each R
    star_R       = [None] * num
    star_R_HF    = [None] * num
    star_R_LF    = [None] * num 
    trans_tell_R = [None] * num
    wave_R       = [None] * num
    dwave_R      = [None] * num
    for j in tqdm(range(num)):
        R                          = R_arr[j]        
        wave_R[j]                  = get_wavelength_axis_constant_R(lmin=lmin, lmax=lmax, R=R)
        dwave_R[j]                 = np.gradient(wave_R[j])
        star_R[j]                  = star.degrade_resolution(wave_R[j], renorm=False).flux
        star_R[j]                  = star_R[j] * dwave_R[j] # propto [ph/µm] => [ph/bin]
        star_R_HF[j], star_R_LF[j] = filtered_flux(star_R[j], R=R, Rc=Rc, filter_type=filter_type)
        if trans_tell is not None:
            trans_tell_R[j] = trans_tell.degrade_resolution(wave_R[j], renorm=False).flux

    # Global context worker
    global _CM_CTX
    _CM_CTX = dict(T_arr=T_arr, lg_planet=lg_planet, delta_rv=delta_rv, vsini_planet=vsini_planet, airmass=airmass, trans_tell=trans_tell, spectrum_contributions=spectrum_contributions, model=model, star_spectrum=star_spectrum, wave_model=wave_model, wave_instru=wave_instru, R_arr=R_arr, Rc=Rc, filter_type=filter_type, wave_R=wave_R, dwave_R=dwave_R, star_R_HF=star_R_HF, star_R_LF=star_R_LF, trans_tell_R=trans_tell_R)
    
    # Parallel calculations
    with Pool(processes=cpu_count(), initializer=_init_cm_ctx, initargs=(_CM_CTX,)) as pool:
        for (i, residual_signal_1D) in tqdm(pool.imap(process_colormap_MM_DI_R_Tp, [(i) for i in range(num)]), total=num, desc=f"rocess_colormap_MM_DI_R_Tp(lmin={lmin}, lmax={lmax}, Rmin={Rmin}, Rmax={Rmax}, Tmin={Tmin}, Tmax={Tmax}, model={model})"):
            residual_signal[i, :] = residual_signal_1D # Normalizing each row
    
    # Plots
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xlabel("Spectral resolution", fontsize=14)
    plt.ylabel("Planet temperature [K]", fontsize=14)
    plt.ylim([T_arr[0], T_arr[-1]])
    plt.xlim(R_arr[0],  R_arr[-1])
    plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.minorticks_on()
    
    data       = 100 * residual_signal
    cmap       = plt.get_cmap(cmap_colormaps)
    cbar_label = r'Residual MM signal fraction $(\alpha - \beta)$ $/$ $\delta$  [%]'

    # Heatmap with pcolormesh
    mesh = plt.pcolormesh(R_arr, T_arr, data, cmap=cmap, shading='auto', vmin=0, vmax=np.nanmax(data))
    
    # Contours
    cs = plt.contour(R_arr, T_arr, data, colors='k', linewidths=0.5, alpha=0.7)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%d%%")
    
    # Colorbar
    ax = plt.gca()
    cbar = plt.colorbar(mesh, ax=ax, pad=0.025, shrink=1)
    cbar.minorticks_on()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=14)
    
    # Title
    if title is None:
        tell       = "with tellurics absorption" if tellurics else "without tellurics absorption"
        title_text = (f"Molecular mapping residual signal fluctuations ({tell}) \n in {spectrum_contributions} light ({model}-model), $T_*$={T_star}K, "r"$\Delta$rv="f"{delta_rv}km/s")    
    else:
        title_text = title
    plt.title(title_text, fontsize=16, pad=14)
    
    plt.tight_layout()
    filename = f"colormaps_bandwidth_Tp/Colormap_MM_DI_R_Tp_residual_signal_{tellurics}_{spectrum_contributions}_{model}_Rc{Rc}_Ts{T_star}K_drv{delta_rv}kms_broad{vsini_planet}kms"
    plt.savefig(colormaps_path + filename + ("_with_tellurics" if tellurics else "") + ".png", format='png', bbox_inches='tight')
    plt.show()
        
    return R_arr, T_arr, residual_signal

def process_colormap_MM_DI_R_Tp(i):
    T_arr                  = _CM_CTX["T_arr"]
    lg_planet              = _CM_CTX["lg_planet"]
    delta_rv               = _CM_CTX["delta_rv"]
    vsini_planet           = _CM_CTX["vsini_planet"]
    airmass                = _CM_CTX["airmass"]
    trans_tell             = _CM_CTX["trans_tell"]
    spectrum_contributions = _CM_CTX["spectrum_contributions"]
    model                  = _CM_CTX["model"]
    star_spectrum          = _CM_CTX["star_spectrum"]
    wave_model             = _CM_CTX["wave_model"]
    wave_instru            = _CM_CTX["wave_instru"]
    R_arr                  = _CM_CTX["R_arr"]
    Rc                     = _CM_CTX["Rc"]
    filter_type            = _CM_CTX["filter_type"]
    wave_R                 = _CM_CTX["wave_R"]
    dwave_R                = _CM_CTX["dwave_R"]
    star_R_HF              = _CM_CTX["star_R_HF"]
    star_R_LF              = _CM_CTX["star_R_LF"]
    trans_tell_R           = _CM_CTX["trans_tell_R"]
    
    T_planet = T_arr[i]
    
    # Getting planet spectrum in [J/s/m2/µm]
    if spectrum_contributions=="reflected":
        albedo_spectrum = load_albedo_spectrum(T_planet, lg_planet, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
        planet_spectrum = Spectrum(wavelength=wave_model, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T_planet, lg=lg_planet, model=model, rv=0, vsini=0)
    elif spectrum_contributions=="thermal":
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model, instru=None)
        planet_spectrum = planet_spectrum.interpolate_wavelength(wave_model, renorm=False) # Interpolating on wave_model (constant dl)
    else:
        raise ValueError("spectrum_contributions must be 'reflected' or 'thermal'")
    planet_spectrum = planet_spectrum.broad(vsini_planet)                               # Broadening the spectrum
    planet_spectrum = planet_spectrum.doppler_shift(delta_rv)                           # Shifting the spectrum
    planet          = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru (constant R_model)
    planet.flux     = planet.flux * wave_instru                                         # [J/s/m2/µm] => propto [ph/µm]
    
    # Calculation for each R
    residual_signal_1D = np.zeros((len(R_arr)))
    for j, R in enumerate(R_arr):
        
        # Degrading the spectra on wave
        planet_R = planet.degrade_resolution(wave_R[j], renorm=False).flux
        planet_R = planet_R * dwave_R[j] # propto [ph/µm] => [ph/bin]

        if trans_tell is not None:
            trans = trans_tell_R[j]
        else:
            trans = 1
        
        # DI CCF signal
        template = trans*planet_R
        template = template / np.sqrt(np.nansum(template**2))
        delta    = np.nansum(trans*planet_R * template)
        
        # High- and low-pass filtering the spectra
        planet_HF, planet_LF = filtered_flux(planet_R, R=R, Rc=Rc, filter_type=filter_type)
        
        # S/N and signal loss calculations
        template = trans*planet_HF 
        template = template / np.sqrt(np.nansum(template**2))
        alpha    = np.nansum(trans*planet_HF * template)
        beta     = np.nansum(trans*star_R_HF[j]*planet_LF/star_R_LF[j] * template)

        # Lost signal calculations
        residual_signal_1D[j] = (alpha - beta) / delta
    
    return i, residual_signal_1D



