from src.config import *
from src.spectrum import *
from matplotlib.cm import get_cmap

path_file = os.path.dirname(__file__)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONTRAST OR SNR CURVES
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FastCurves(calculation, instru, time, name_planet, mag_star, band0, planet_spectrum, star_spectrum, tellurics,
               apodizer=None, strehl=None, coronagraph=None, plot_mag=True, separation_planet=None, mag_planet=None,
               channel=False, systematic=False, show_plot=True, print_value=True,
               post_processing="molecular mapping", sep_unit="arcsec", bkgd=None, star_pos="center", Rc=100,
               band_only=None, used_filter="gaussian"):
    """
    Function for calculating contrast or SNR curves, depending on the instrument and type of planet/star selected 
    
    Parameters
    ----------
    calculation : str
        type of calculation wanted ("contrast" or "SNR" or "Noise")
    instru : str
        name of the considered instrument ("HARMONI" or "ERIS" or "MIRIMRS" or "NIRCam")
    time : float/int
        exposure time
    name_planet : str
        planet name
    mag_star : float/int
        star magnitude
    band0 : str
        spectral band where the magnitude is defined
    planet_spectrum : class Spectrum
        planet spectrum used for the calculation
    star_spectrum : class Spectrum
        planet spectrum used for the calculation
    tellurics : bool (True or False)
        considering (or not) the earth atmosphere (ground or space observations)
    apodizer : str, optional
        apodizer considered. The default is None.
    strehl : str, optional
        strehl ratio considered (when tellurics=True). The default is None.
    coronagraph : str, optional
        coronagraph considered. The default is None.
    plot_mag : bool, optional
        for contrast curves in mag unit. The default is True.
    separation_planet : float/int (in arcsec), optional
        planet separation (if known). The default is None.
    mag_planet : float/int, optional
        planet magnitude (needed for SNR calculation). The default is None.
    channel : bool, optional
        for MIRI/MRS and SNR calculation : True if channel SNR value is wanted. The default is True.
    """

    #------------------------------------------------------------------------------------------------

    cos_p = 1  # mismatch 
    cos_est = 0.28  #None # correlation estimée

    size_core = 1  # Taille de l'ouverture sur laquelle on intègre (size_core**2 = Afwhm)

    band_plot = "1SHORT"
    show_cos_theta_est = False  # pour voir l'impact du bruit sur l'estimation de la corrélation entre le spectre des données et le template
    show_contrast_comparison = False  # pour comparer le contraste estimé et le contraste des données
    show_contrast_comparison_all = False  # pour comparer le contraste estimé et le contraste des données avec toutes les bandes
    if instru == "MIRIMRS" and channel and calculation == "SNR":  # on divise le budget du temps d'exposition par 3 car il faut le répartir dans les bandes XSHORT XMEDIUM XLONG
        time /= 3
    if name_planet is None:
        for_name_planet = " "
    else:
        for_name_planet = " for " + name_planet

    #------------------------------------------------------------------------------------------------

    # On récupère les caractéristiques de l'instrument

    config_data = get_config_data(instru)
    saturation_e = config_data["spec"]["saturation_e"]  # capacité maximale en e- des pixels
    min_DIT = config_data["spec"]["minDIT"]  # temps d'intégration minimale
    max_DIT = config_data["spec"]["maxDIT"]  # temps d'intégration maximale
    quantum_efficiency = config_data["spec"]["Q_eff"]  # efficacité quantique (en e-/ph)
    RON = config_data["spec"]["RON"]  # bruit de lecture du détecteur en e-/DIT
    dark_current = config_data["spec"]["dark_current"]  # "dark current" du détecteur en e-/s

    #------------------------------------------------------------------------------------------------

    contrast_band = []
    SNR_band = []
    SNR_planet_band = []
    name_band = []
    separation_band = []  # contraste,SNR,nom et separation de chaque bande
    noise_band = []
    signal_band = []
    DIT_band = []
    sigma_s_2_band = []
    sigma_ns_2_band = []
    SNR_max = 0.
    SNR_max_planet = 0.  # pour connaitre le maximum de SNR pour les plots et les prints
    T_planet = planet_spectrum.T
    T_star = star_spectrum.T  # températures des spectres
    lg_planet = planet_spectrum.lg
    lg_star = star_spectrum.lg  # gravités de surface des spectres
    model = planet_spectrum.model
    star_model = star_spectrum.model  # modèles des spectres
    syst_rv = star_spectrum.syst_rv
    delta_rv = planet_spectrum.delta_rv
    R_planet = planet_spectrum.R
    R_star = star_spectrum.R
    R = max(R_planet, R_star)  # résolutions des spectres
    if print_value:
        print('\n Planetary spectrum (' + model + ') : R =', round(round(R_planet, -3)), ', T =', round(T_planet),
              "K, lg =", round(lg_planet, 1))
        print(' Star spectrum (' + star_model + ') : R =', round(round(R_star, -3)), ', T =', round(T_star), "K & lg =",
              round(lg_star, 1))
        print(' syst_rv = ', round(syst_rv, 2), ' km/s & Δrv = ', round(delta_rv, 2), ' km/s')
        if post_processing == "molecular mapping":
            print('\n Rc = ', Rc)
        if systematic:
            print('\n With systematics')
    if sep_unit == "mas":
        if separation_planet is not None:
            separation_planet *= 1e3  # separation en arcsec => mas

    #------------------------------------------------------------------------------------------------

    # Restriction des spectres à la gamme instrumentale et normalisation des spectres à la bonne magnitude

    star_spectrum_instru, star_spectrum_density = spectrum_instru(band0, R, config_data, mag_star,
                                                                  star_spectrum)  # spectre de l'étoile en photons/min ajusté à la bonne magnitude
    if calculation == "contrast" or calculation == "Noise":
        planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_star,
                                                                          planet_spectrum)  # spectre de la planète en photons/min total reçu
        planet_spectrum_instru.set_flux(np.nansum(
            star_spectrum_instru.flux))  # on met le spectre planétaire au même flux (en ph/mn total reçu) que le spectre stellaire sur la bande d'intérêt instrumentale (en faisant cela, on aura un contraste en photons et non en énergie, sinon il aurait fallu mettre à la même énergie recu)
        planet_spectrum_density.flux *= np.nanmean(star_spectrum_density.flux) / np.nanmean(
            planet_spectrum_density.flux)  # pas vraiment d'utilité, uniquement pour que les densités de flux (en énergie) aient la même magnitude
    elif calculation == "SNR":
        if mag_planet == None:
            raise KeyError("PLEASE INPUT A MAGNITUDE FOR THE PLANET FOR THE SNR CALCULATION !")
        else:
            planet_spectrum_instru, planet_spectrum_density = spectrum_instru(band0, R, config_data, mag_planet,
                                                                              planet_spectrum)  # spectre de la planète en ph/min ajusté à la bonne magnitude
    wave_instru = planet_spectrum_instru.wavelength
    vega_spectrum = load_vega_spectrum()  # spectre de vega en J/s/m²/µm
    vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave_instru,
                                                         renorm=False)
    mag_star_instru = -2.5 * np.log10(np.nanmean(star_spectrum_density.flux) / np.nanmean(vega_spectrum.flux))
    mag_planet_instru = -2.5 * np.log10(np.nanmean(planet_spectrum_density.flux) / np.nanmean(vega_spectrum.flux))
    if show_plot:  # plot des spectres stellaire et planétaire sur la bande instrumentale 
        plt.figure()
        plt.yscale("log")
        plt.plot(planet_spectrum_instru.wavelength, planet_spectrum_instru.flux, 'g',
                 label=f'planet, {model} with $T$={T_planet}K and mag(instru)={np.round(mag_planet_instru, 1)}')
        plt.plot(star_spectrum_instru.wavelength, star_spectrum_instru.flux, 'k',
                 label=f'star, {star_model} with $T$={T_star}K and mag(instru)={np.round(mag_star_instru, 1)}')
        plt.title(
            f"Star and planet spectra on the instrumental bandwitdh (R = {round(round(R, -3))})" + '\n with $rv_{syst}$' + f' = {round(syst_rv, 1)} km/s & Δrv = {round(delta_rv, 1)} km/s',
            fontsize=14)
        plt.xlabel("wavelength (in µm)", fontsize=14)
        plt.ylabel("flux (in ph/mn)", fontsize=14)
        plt.legend()
        plt.show()
    if print_value:
        print("\n" + "\033[4m" + "ON THE INSTRUMENTAL BANDWIDTH" + " :" + "\033[0m")
        print(" Cp(ph) = {0:.2e}".format(np.nansum(planet_spectrum_instru.flux) / np.nansum(star_spectrum_instru.flux)),
              ' & \u0394' + 'mag = ', round(mag_planet_instru - mag_star_instru, 3))
        print(" mag(star) = ", round(mag_star_instru, 3), " & mag(planet) = ", round(mag_planet_instru, 3), "\n")

    #------------------------------------------------------------------------------------------------
    # Pour chaque bande spectrale de l'instrument considéré :
    #------------------------------------------------------------------------------------------------

    if show_plot:  # on plot le flux stellaire de chaque band pour NIRCam
        f1 = plt.figure()
        band_flux = f1.add_subplot(111)
        band_flux.set_xlabel("wavelength (in µm)")
        band_flux.set_ylabel("flux (in e-/mn)")
        band_flux.set_title(f"Star flux through coronagraph with {coronagraph} and mag({band0}) = {mag_star}")

    if calculation == "contrast" and show_contrast_comparison_all and show_plot:  # comparaison des niveaux de bruit sur toutes les bandes
        fig, axs = plt.subplots(2, figsize=(6, 6), sharex=True, sharey=True, layout="constrained",
                                gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
        fig.suptitle(f'Comparison of noise levels in {name_planet} on-sky data from {instru}', fontsize=14)
        axs[0].set_title("with systematics (analytical) and with spatial variance (empirical)", fontsize=10,
                         fontweight="bold")
        axs[1].set_title("without systematics (analytical) and with ERR extension (empirical)", fontsize=10,
                         fontweight="bold")
        axs[1].set_xlabel(r'mean empirical std per spectral channel (in e-)', fontsize=10)
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[0].set_xscale("log")
        axs[0].grid(True)
        axs[1].grid(True)
        fig.text(-0.03, 0.5, 'mean analytical std per spectral channel (in e-)', va='center', rotation='vertical')
        cmap = get_cmap('Spectral', len(config_data['gratings']))

    for nb, band in enumerate(config_data['gratings']):  # POUR CHAQUE BANDE
        if band_only is not None and band != band_only:
            continue  # Si on choisit l'apodizeur pour Proxima cen b concu spécialement pour la bande H, on ignore les autres bandes
        if instru == "HARMONI" and apodizer == "SP_Prox" and band != "H":
            continue  # Si on choisit l'apodizeur pour Proxima cen b concu spécialement pour la bande H, on ignore les autres bandes

        name_band.append(band)  # on rajoute le nom de la bande dans la liste

        #------------------------------------------------------------------------------------------------

        # Degradation à la resolution instrumentale et restriction de la gamme de longueur d'onde sur la bande considérée

        star_spectrum_inter = spectrum_inter(config_data, band, star_spectrum_instru)
        planet_spectrum_inter = spectrum_inter(config_data, band, planet_spectrum_instru)
        wave_inter = planet_spectrum_inter.wavelength
        R = planet_spectrum_inter.R  # résolution de la bande spectrale considérée
        if Rc is None:
            sigma = None
        else:
            sigma = 2 * R / (np.pi * Rc) * np.sqrt(np.log(2) / 2)
        mag_star_inter = -2.5 * np.log10(np.nanmean(star_spectrum_density.flux[
                                                        (wave_instru > config_data['gratings'][band].lmin) & (
                                                                    wave_instru < config_data['gratings'][
                                                                band].lmax)]) / np.nanmean(vega_spectrum.flux[(
                                                                                                                          wave_instru >
                                                                                                                          config_data[
                                                                                                                              'gratings'][
                                                                                                                              band].lmin) & (
                                                                                                                          wave_instru <
                                                                                                                          config_data[
                                                                                                                              'gratings'][
                                                                                                                              band].lmax)]))
        mag_planet_inter = -2.5 * np.log10(np.nanmean(planet_spectrum_density.flux[
                                                          (wave_instru > config_data['gratings'][band].lmin) & (
                                                                      wave_instru < config_data['gratings'][
                                                                  band].lmax)]) / np.nanmean(vega_spectrum.flux[(
                                                                                                                            wave_instru >
                                                                                                                            config_data[
                                                                                                                                'gratings'][
                                                                                                                                band].lmin) & (
                                                                                                                            wave_instru <
                                                                                                                            config_data[
                                                                                                                                'gratings'][
                                                                                                                                band].lmax)]))

        #------------------------------------------------------------------------------------------------

        # Transmissions du système pour la bande considérée

        trans = transmission(instru, wave_inter, band, tellurics, apodizer)

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Lecture des profils de PSF (fraction_PSF = fraction de photons dans le coeur de la PSF)

        PSF_profile, fraction_PSF, separation = PSF_profile_fraction_separation(band, strehl, apodizer, coronagraph,
                                                                                instru, config_data, sep_unit, star_pos)
        separation_band.append(separation)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------

        # Coronagraph

        if coronagraph is not None and instru == "NIRCam":
            peak_PSF = fits.getdata(
                "sim_data/PSF/star_center/PSF_" + instru + "/peak_PSF_" + band + "_" + coronagraph + ".fits")  # transmission du coronographe en fonction de la séparation de la cible
            f = interp1d(peak_PSF[0], peak_PSF[1], bounds_error=False, fill_value=np.nan)
            g = interp1d(peak_PSF[0], peak_PSF[2], bounds_error=False, fill_value=np.nan)
            correction_transmission_ETC = 0.9  # facteur de correction (par rapport à l'ETC)
            peak_PSF_interp = f(separation) * correction_transmission_ETC
            radial_transmission_interp = g(separation) * correction_transmission_ETC
            fraction_PSF *= correction_transmission_ETC  # = flux stellaire total transmis par le coronographe (+Lyot stop) lorsque l'étoile est parfaitement alignée avec ce dernier
            PSF_profile *= fraction_PSF
            if show_plot:
                band_flux.plot(star_spectrum_inter.wavelength, star_spectrum_inter.flux * fraction_PSF * trans,
                               label=band + f" ({round(np.nansum(fraction_PSF * star_spectrum_inter.flux * trans / 60))} e-/s)")
                band_flux.legend(loc='upper right')

        #------------------------------------------------------------------------------------------------

        # R_corr

        R_corr = np.zeros_like(separation) + 1.
        if instru == "MIRIMRS":  # 4pt dithering
            sep, r_corr = fits.getdata("sim_data/R_corr/star_" + star_pos + "/R_corr_MIRIMRS/R_corr_" + band + ".fits")
            f = interp1d(sep, r_corr, bounds_error=False, fill_value="extrapolate")
            R_corr = f(separation)

        #------------------------------------------------------------------------------------------------

        # Aperture correction ratio

        if instru == "MIRIMRS":
            aper_corr = fits.getheader("sim_data/PSF/star_" + star_pos + "/PSF_" + instru + "/PSF_" + band + ".fits")[
                'AC']
            star_spectrum_inter.flux *= aper_corr
            planet_spectrum_inter.flux *= aper_corr

        #------------------------------------------------------------------------------------------------------------------------------------------------

        if post_processing == "molecular mapping":  # Molecular Mapping

            size_core = 3  #  AFWHM = size_core**2 = nb de pixels sur lesquels on intègre la planète

            #------------------------------------------------------------------------------------------------------------------------------------------------

            # Bruit systématique

            if systematic:
                sigma_syst_2 = np.zeros((len(separation), len(wave_inter)))  # en e-/Flux_stell_tot/Afwhm
                sigma_syst_prime_2 = np.zeros_like(separation)  # en e-/Flux_stell_tot/Afwhm
                M_HF = np.zeros_like(separation)  # modulations haute fréquence (créant le bruit systématique...)
                Mp = np.zeros_like(wave_inter) + 1  # modulations systématiques du spectre planétaire
                if instru == "MIRIMRS":
                    sigma_syst_prime_2, sigma_syst_2, sep, m_HF, Mp, wave = systematic_profile(config_data, band, sigma,
                                                                                               Rc, R,
                                                                                               star_spectrum_instru,
                                                                                               planet_spectrum_instru,
                                                                                               wave_inter, size_core,
                                                                                               star_pos,
                                                                                               used_filter)  # en e-/Flux_tot
                    f = interp1d(sep, np.sqrt(sigma_syst_prime_2), bounds_error=False, fill_value=np.nan)
                    sigma_syst_prime_2 = f(separation) ** 2  # en e-/Flux_stell_tot/pixel
                    f = interp2d(wave_inter[~np.isnan(sigma_syst_2[0])], sep,
                                 np.sqrt(sigma_syst_2[:, ~np.isnan(sigma_syst_2[0])]), bounds_error=False,
                                 fill_value=np.nan)
                    mask_M = (wave_inter >= wave[0]) & (wave_inter <= wave[-1])
                    planet_spectrum_inter.crop(wave[0], wave[-1])
                    star_spectrum_inter.crop(wave[0], wave[-1])
                    trans = trans[mask_M]
                    wave_inter = wave_inter[mask_M]
                    Mp = Mp[mask_M]
                    sigma_syst_2 = f(wave_inter, separation) ** 2  # en e-/Flux_stell_tot/Afwhm
                    M_HF = np.zeros((len(separation), len(wave_inter)))  # pour le cos theta est.
                    for i in range(len(separation)):
                        idx = (np.abs(separation[i] - sep)).argmin()
                        M_HF[i] = m_HF[idx][mask_M]
                        #plt.figure()  plt.plot(separation,sigma_syst_prime_2)  plt.plot(separation,np.nanmean(sigma_syst_2,1))  plt.yscale('log')  plt.title(f"{band}")  plt.show()

            #------------------------------------------------------------------------------------------------------------------------------------------------

            # Calcul calcul du template

            template, _ = filtered_flux(planet_spectrum_inter.flux, R, Rc, used_filter)
            template *= trans
            template = template / np.sqrt(np.nansum(template ** 2))  # On suppose qu'on a le template "parfait" 
            if systematic:
                planet_spectrum_inter.flux *= Mp  # On prend en compte les modulations systématiques du spectre planétaire (effet peu significatif pour MIRIMRS)

            #------------------------------------------------------------------------------------------------------------------------------------------------

            # Calcul du beta (avec modulations systématiques)

            if Rc is None:
                beta = 0
            else:
                beta = beta_calc(star_spectrum_inter, planet_spectrum_inter, template, Rc, R, fraction_PSF, trans,
                                 separation, used_filter)  # en ph/mn

            #------------------------------------------------------------------------------------------------------------------------------------------------

            # Calcul du alpha (nb de photons utiles /min au molecular mapping sur la bande considérée) (avec modulations systématiques)

            alpha = alpha_calc(planet_spectrum_inter, template, Rc, R, fraction_PSF, trans, separation,
                               used_filter)  # en ph/mn

            #if calculation=="contrast": # on renormalise sur chaque bande alpha afin d'avoir un contraste par bande
            #alpha *= np.nansum(star_spectrum_inter.flux)/np.nansum(planet_spectrum_inter.flux)
            #beta *= np.nansum(star_spectrum_inter.flux)/np.nansum(planet_spectrum_inter.flux)                

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Affichage des quantités d'intérêts
        if print_value:
            print('\n')
            print(
                "\033[4m" + "BAND = " + f'{band}' + f" (from {round(wave_inter[0], 2)} to {round(wave_inter[-1], 2)} µm) :" + "\033[0m")
            if calculation == "SNR":
                print(" Cp(ph) = {0:.2e}".format(
                    np.nansum(planet_spectrum_inter.flux) / np.nansum(star_spectrum_inter.flux)),
                      ' => \u0394' + 'mag = ', round(mag_planet_instru - mag_star_instru, 3),
                      f"\n Magnitudes : mag_star = {round(mag_star_inter, 3)} & mag_planet = {round(mag_planet_inter, 3)}")
            if post_processing == "molecular mapping":
                print(" Number of spectral pixels :", len(wave_inter), " & R =", round(R))
                if Rc is not None:
                    print(" Cut-off resolution: Rc =", Rc)
                print(" Fraction in the heart of the PSF : fraction_PSF =", round(fraction_PSF, 3),
                      "\n Useful photons (molecular mapping)/min from the planet : alpha =", round(np.nanmean(alpha)),
                      "\n Signal loss : beta/alpha =", round(100 * np.nanmean(beta) / np.nanmean(alpha), 3), "%")

        # Pour comparer le spectre stellaire des donnéees et le spectre stellaire utilisé pour MIRI/MRS

        if band == band_plot and calculation == "contrast" and show_contrast_comparison and not show_contrast_comparison_all and show_plot:
            plt.figure()
            plt.yscale("log")
            plt.xlabel("wavelength (in µm)", fontsize=14)
            plt.ylabel("Flux (in e-/mn)", fontsize=14)
            plt.title('Star spectra on ' + band + ' (FastCurves VS data)', fontsize=14)
            x, y = fits.getdata('utils/star_spectrum/star_spectrum_' + band + '_data.fits')
            y = y[(x > wave_inter[0]) & (x < wave_inter[-1])]
            x = x[(x > wave_inter[0]) & (x < wave_inter[-1])]
            plt.plot(x, y, 'b')
            plt.ylim(np.nanmin(y), np.nanmax(y))
            plt.plot(x, star_spectrum_inter.interpolate_wavelength(star_spectrum_inter.flux * trans,
                                                                   star_spectrum_inter.wavelength, x, renorm=True).flux,
                     'r')
            plt.legend(["data", "FastCurves"])
            print('ratio = data/model => ', (np.nansum(y)) / np.nansum(star_spectrum_inter.flux * trans))

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Calcul du du DIT et du RON_eff

        DIT, RON_eff = DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_inter, time,
                               min_DIT, max_DIT, trans, quantum_efficiency, RON, saturation_e, print_value)
        NDIT = time / DIT  # nombre d'intégrations
        DIT_band.append(DIT)

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Spectres à travers le système

        star_spectrum_inter.flux = star_spectrum_inter.flux * trans * DIT * quantum_efficiency  # spectre en nb de e-/DIT à travers le système en fonction du canal spectral dans la bande spectrale considérée (H, K, etc.)
        planet_spectrum_inter.flux = planet_spectrum_inter.flux * trans * DIT * quantum_efficiency  # spectre en nb de e-/DIT à travers le système en fonction du canal spectral dans la bande spectrale considérée (H, K, etc.)

        #------------------------------------------------------------------------------------------------------------------------------------------------

        #------------------------------------------------------------------------------------------------------------------------------------------------
        # Calcul des courbes de contraste ou de SNR de la bande :
        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Signal et bruit détecteurs

        contrast = np.zeros_like(separation)
        contrast_wo_syst = np.zeros_like(separation)
        SNR = np.zeros_like(separation)
        t_syst = np.zeros_like(separation)
        cos_theta_est = np.zeros_like(separation)
        norm_d = np.zeros_like(separation)
        if apodizer is not None:
            sep_min = config_data["apodizers"][str(apodizer)].sep
        else:
            sep_min = 0
        if separation_planet is not None:
            idx = (np.abs(separation - separation_planet)).argmin()

        if post_processing == "molecular mapping":  # Molecular Mapping
            signal = (
                                 alpha * cos_p - beta) * DIT * quantum_efficiency  # nombre total d'e- recu utiles au molecular mapping /DIT (dans la FWHM ou "fraction_core")
            PSF_profile[separation < sep_min] *= 1e-4
            signal[separation < sep_min] *= 1e-4  # flux atténué d'un facteur 1e-4 à cause du FPM

        elif post_processing == "ADI+RDI":  # ADI+RDI
            if calculation == "contrast":
                signal = np.nansum(
                    star_spectrum_inter.flux) * peak_PSF_interp  # flux max en e-/DIT en fonction de la séparation
            elif calculation == "SNR":
                signal = np.nansum(
                    planet_spectrum_inter.flux) * peak_PSF_interp  # flux max en e-/DIT en fonction de la séparation

        sigma_dc_2 = dark_current * DIT * 60  # bruit de dark current en  e-/DIT/pixel (dark_current est donné en e-/s)
        sigma_ron_2 = RON_eff ** 2  # bruit de lecture en e-/DIT/pixel

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Halo stellaire : calcul du nb d'e- stellaires /DIT à la séparation i en fonction du canal spectral dans la bande spectrale considérée

        sf, sep = np.meshgrid(star_spectrum_inter.flux, PSF_profile)
        star_flux = sep * sf

        if post_processing == "molecular mapping":  # Molecular Mapping   
            star_flux[star_flux > saturation_e] = saturation_e  # si saturation
            sigma_halo_2 = star_flux
            t, _ = np.meshgrid(template, PSF_profile)
            sigma_halo_prime_2 = np.nansum(sigma_halo_2 * t ** 2,
                                           axis=1)  #  bruit de photon du halo stellaire en e-/DIT/pixel

        elif post_processing == "ADI+RDI":  # ADI+RDI+RDI 
            star_flux = np.nansum(star_flux, axis=1)
            star_flux[star_flux > saturation_e] = saturation_e  # si saturation
            sigma_halo_2 = star_flux  # flux stellaire total e-/DIT/pixel en fonction à la séparation i

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Background

        if bkgd is not None:
            bkgd_flux = fits.getdata(
                "sim_data/Background/" + instru + "/" + bkgd + "/background_" + band + ".fits")  # en e-/s/pixel
            f = interp1d(bkgd_flux[0], bkgd_flux[1], bounds_error=False, fill_value=np.nan)
            bkgd_flux_inter = f(wave_inter)
            bkgd_flux_inter *= 60 * DIT * np.nansum(bkgd_flux[1]) / np.nansum(
                bkgd_flux_inter)  # en e-/DIT/pixel + il faut renormaliser car on interpole 
            sigma_bkgd_2 = bkgd_flux_inter
        else:
            sigma_bkgd_2 = 0
        if post_processing == "molecular mapping":
            sigma_bkgd_prime_2 = np.nansum(sigma_bkgd_2 * template ** 2)  # bruit du background en e-/DIT/pixel
        elif post_processing == "ADI+RDI":  # ADI+RDI
            sigma_bkgd_2 = np.nansum(
                bkgd_flux_inter) * radial_transmission_interp  # bruit/DIT  (radial_transmission_interp = transmission radiale)

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Systematic

        if systematic:
            sigma_syst_prime_2 *= np.nansum(
                star_spectrum_inter.flux) ** 2  # en e-/DIT/boite AFWHM / approx : sigma_syst_prime_2 = np.nanmean(sigma_syst_2,1)
            sigma_syst_2 *= np.nansum(star_spectrum_inter.flux) ** 2  # en e-/DIT/boite AFWHM

        #------------------------------------------------------------------------------------------------------------------------------------------------

        signal_band.append(signal)
        if post_processing == "molecular mapping":
            sigma_ns_2_band.append(
                R_corr * size_core ** 2 * (sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))

            if systematic:
                sigma_s_2_band.append(sigma_syst_prime_2)
        elif post_processing == "ADI+RDI":  # ADI+RDI
            sigma_ns_2_band.append(R_corr * size_core ** 2 * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))
            if systematic:
                sigma_s_2_band.append(sigma_syst_2)

        #------------------------------------------------------------------------------------------------------------------------------------------------

        # Calcul du contraste

        if calculation == "contrast" or calculation == "Noise" :

            if post_processing == "molecular mapping":
                if systematic:
                    contrast = 5 * np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2 + sigma_syst_prime_2 * NDIT / (
                                    R_corr * size_core ** 2))) / (signal * np.sqrt(
                        NDIT))  # AFWHM = 9 = 3x3 nombre de pixels sur lequel on integre le companion
                    contrast_wo_syst = 5 * np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2)) / (
                                                   signal * np.sqrt(
                                               NDIT))  # AFWHM = 9 = 3x3 nombre de pixels sur lequel on integre le companion
                else:
                    contrast = 5 * np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2)) / (
                                           signal * np.sqrt(
                                       NDIT))  # AFWHM = 9 = 3x3 nombre de pixels sur lequel on integre le companion

            elif post_processing == "ADI+RDI":
                if systematic:
                    contrast = 5 * np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2 + sigma_syst_2 * NDIT / (
                                    R_corr * size_core ** 2))) / (signal * np.sqrt(
                        NDIT))  # AFWHM = 9 = 3x3 nombre de pixels sur lequel on integre le companion
                    contrast_wo_syst = 5 * np.sqrt(
                        R_corr * size_core ** 2 * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) / (
                                                   signal * np.sqrt(
                                               NDIT))  # AFWHM = 9 = 3x3 nombre de pixels sur lequel on integre le companion
                else:
                    contrast = 5 * np.sqrt(
                        R_corr * size_core ** 2 * (sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2)) / (
                                           signal * np.sqrt(
                                       NDIT))  # AFWHM = 9 = 3x3 nombre de pixels sur lequel on integre le companion

            if plot_mag:  # calcul en deltaMAG 
                contrast = -2.5 * np.log10(contrast)

            if show_plot:
                if band == band_plot and not show_contrast_comparison_all:  # noise contributions :
                    if post_processing == "molecular mapping":  # Molecular Mapping
                        if systematic:
                            plt.figure()
                            plt.xlim(0, separation[-1])
                            plt.plot(separation, contrast, 'k-')
                            plt.plot(separation,
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_halo_prime_2) * (NDIT)) / (
                                                 signal * NDIT), 'r--')
                            plt.plot(separation,
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_ron_2) * (NDIT)) / (signal * NDIT),
                                     'g--')
                            plt.plot(separation,
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_dc_2) * (NDIT)) / (signal * NDIT),
                                     'm--')
                            plt.plot(separation,
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_bkgd_prime_2) * (NDIT)) / (
                                                 signal * NDIT), 'b--')
                            plt.plot(separation, 5 * np.sqrt(sigma_syst_prime_2) / (signal), 'c--')
                            plt.xlabel('separation (in arcsec)', fontsize=14)
                            plt.ylabel(r'contrast 5$\sigma_{CCF}$ / $\alpha_0$', fontsize=14)
                            plt.title(
                                instru + f" noise contributions on {band}" + for_name_planet + "\n with " + "$t_{exp}$=" + str(
                                    round(time)) + "mn, $mag_*$(" + band0 + ")=" + str(
                                    round(mag_star, 2)) + f', $T_p$={T_planet}K and $R_c$ = {Rc}', fontsize=14)
                            plt.legend(["$\sigma_{CCF}$", "$\sigma'_{halo}$", "$\sigma_{ron}$", "$\sigma_{dc}$",
                                        "$\sigma'_{bkgd}$", "$\sigma'_{syst}$"], fontsize=11)
                            plt.yscale('log')
                            plt.grid(True)  #  plt.ylim(1e-4,1e-1)  plt.show()
                        else:
                            plt.figure()
                            plt.xlim(0, separation[-1])
                            plt.plot(separation[separation > sep_min], contrast[separation > sep_min], 'k-')
                            plt.plot(separation[separation > sep_min], 5 * np.sqrt(
                                R_corr * size_core ** 2 * (sigma_halo_prime_2[separation > sep_min]) * (NDIT)) / (
                                                 signal[separation > sep_min] * NDIT), 'r--')
                            plt.plot(separation[separation > sep_min],
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_ron_2) * (NDIT)) / (
                                                 signal[separation > sep_min] * NDIT), 'g--')
                            plt.plot(separation[separation > sep_min],
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_dc_2) * (NDIT)) / (
                                                 signal[separation > sep_min] * NDIT), 'm--')
                            plt.plot(separation[separation > sep_min],
                                     5 * np.sqrt(R_corr * size_core ** 2 * (sigma_bkgd_prime_2) * (NDIT)) / (
                                                 signal[separation > sep_min] * NDIT), 'b--')
                            plt.xlabel('separation (in mas)', fontsize=14)
                            plt.ylabel(r'contrast 5$\sigma$($\rho$) / $\alpha_0$', fontsize=14)
                            plt.title(
                                instru + f" noise contributions on {band}" + for_name_planet + " \n (with " + "$t_{exp}$=" + str(
                                    round(time)) + "mn, $mag_*$(" + band0 + ")=" + str(
                                    round(mag_star, 2)) + f' and $T_p$={T_planet}K )', fontsize=14)
                            plt.legend(["$\sigma_{CCF}$", "$\sigma_{halo}$", "$\sigma_{RON}$", "$\sigma_{DC}$",
                                        "$\sigma_{bkgd}$"])
                            plt.yscale('log')
                            plt.grid(True)
                            plt.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
                        if show_contrast_comparison:  # Pour comparer avec les données réelles 
                            plt.figure()
                            if systematic:
                                plt.plot(separation, contrast, 'r',
                                         label='$\sigma_{CCF}$ according to FastCurves (with systematics)')
                            plt.plot(separation, contrast_wo_syst, 'r--',
                                     label='$\sigma_{CCF}$ according to FastCurves (without systematics)')
                            sep_data, sigma_spatial_2, sigma_err_2, _, _ = fits.getdata(
                                'utils/noiseMM/noiseMM_' + band + '.fits')
                            f = interp1d(sep_data,
                                         5 * np.sqrt(sigma_spatial_2 * np.nansum(star_spectrum_inter.flux / DIT) ** 2),
                                         bounds_error=False, fill_value=np.nan)
                            contrast_spatial = f(separation) / (signal * NDIT)
                            f = interp1d(sep_data, 5 * np.sqrt(
                                size_core ** 2 * sigma_err_2 * np.nansum(star_spectrum_inter.flux / DIT) ** 2),
                                         bounds_error=False, fill_value=np.nan)
                            contrast_err = f(separation) / (signal * NDIT)
                            contrast_err *= np.nanmean(contrast_wo_syst) / np.nanmean(contrast_err)
                            #sep_data,sigma_spatial_2,sigma_err_2,_,_ = fits.getdata('utils/noiseMM/noiseMM_'+band+'.fits')  f=interp1d(sep_data,5*np.sqrt(sigma_spatial_2), bounds_error=False, fill_value=np.nan)  contrast_spatial=f(separation)/(signal*NDIT)  f=interp1d(sep_data,5*np.sqrt(size_core**2*sigma_err_2), bounds_error=False, fill_value=np.nan)  contrast_err=f(separation)/(signal*NDIT)  contrast_err *= np.nanmean(contrast_wo_syst)/np.nanmean(contrast_err)
                            plt.plot(separation, contrast_spatial, 'b',
                                     label='$\sigma_{CCF}$ according to on-sky data (from spatial variance)')
                            plt.plot(separation, contrast_err, 'b--',
                                     label='$\sigma_{CCF}$ according to on-sky data (from ERR extension)')
                            plt.yscale('log')
                            plt.xlabel('separation (in arcsec)', fontsize=14)
                            plt.ylabel(r'contrast 5$\sigma_{CCF}$ / $\alpha_0$', fontsize=14)
                            plt.title(
                                'FastCurves VS on-sky data' + f" on {band}" + for_name_planet + "\n with " + "$t_{exp}$=" + str(
                                    round(time)) + "mn, $mag_*$(" + band0 + ")=" + str(
                                    round(mag_star, 2)) + f', $T_p$={T_planet}K and $R_c$ = {Rc}', fontsize=14)
                            plt.grid(True)
                            plt.legend(fontsize=11)  #  plt.ylim(1e-4,1e-1)  plt.show()
                        if systematic:  # plot du t_syst
                            t_syst = DIT * R_corr * size_core ** 2 * (
                                        sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2) / sigma_syst_prime_2  # en min 
                            plt.figure()
                            plt.plot(separation, t_syst, 'b')
                            plt.ylabel('$t_{syst}$ (in mn)', fontsize=14)
                            plt.xlabel('separation (in arcsec)', fontsize=14)
                            plt.grid(True)
                            plt.title("$t_{syst}$" + f" on {band}" + "\n with $mag_*$(" + band0 + ")=" + str(
                                round(mag_star, 2)), fontsize=14)
                            plt.plot([separation[0], separation[-1]], [time, time], 'r-')
                            plt.yscale('log')
                            plt.legend(["$t_{syst}$", "$t_{exp}$ =" + f"{time} mn"])
                    elif post_processing == "ADI+RDI":  # RDI+ADI+RDI
                        plt.figure()
                        plt.xlim(0, separation[-1])
                        plt.plot(separation, contrast, 'k-')
                        plt.plot(separation,
                                 5 * np.sqrt(R_corr * size_core ** 2 * (sigma_halo_2) * (NDIT)) / (signal * NDIT),
                                 'r--')
                        plt.plot(separation,
                                 5 * np.sqrt(R_corr * size_core ** 2 * (sigma_ron_2) * (NDIT)) / (signal * NDIT),
                                 'g--')
                        plt.plot(separation,
                                 5 * np.sqrt(R_corr * size_core ** 2 * (sigma_dc_2) * (NDIT)) / (signal * NDIT), 'm--')
                        plt.plot(separation,
                                 5 * np.sqrt(R_corr * size_core ** 2 * (sigma_bkgd_2) * (NDIT)) / (signal * NDIT),
                                 'b--')
                        plt.xlabel('separation (in arcsec)', fontsize=14)
                        plt.ylabel(r'contrast (5$\sigma$ / $F_{max}$)', fontsize=14)
                        plt.title(
                            instru + f" noise contributions on {band}" + for_name_planet + " \n with " + "$t_{exp}$=" + str(
                                round(time)) + "mn and $mag_*$(" + band0 + ")=" + str(round(mag_star, 2)), fontsize=14)
                        plt.legend(
                            ["$\sigma_{tot}$", "$\sigma_{halo}$", "$\sigma_{RON}$", "$\sigma_{DC}$", "$\sigma_{bkgd}$",
                             "$\sigma_{syst}$"])
                        plt.yscale('log')
                        plt.grid(True)
                elif show_contrast_comparison_all:
                    R_corr_px = np.zeros_like(separation) + 1.
                    if instru == "MIRIMRS":  # 4pt dithering
                        sep, r_corr_px = fits.getdata(
                            "sim_data/R_corr/star_" + star_pos + "/R_corr_MIRIMRS/R_corr_perpx_" + band + ".fits")
                        f = interp1d(sep, r_corr_px, bounds_error=False, fill_value='extrapolate')
                        R_corr_px = f(separation)
                    r_corr = np.zeros_like(sigma_halo_2) + 1
                    r_corr_px = np.zeros_like(sigma_halo_2) + 1
                    for i in range(r_corr.shape[1]):
                        r_corr[:, i] = R_corr
                        r_corr_px[:, i] = R_corr_px
                    sigma_tot = np.sqrt(NDIT * r_corr * size_core ** 2 * (
                                sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2 + sigma_syst_2 * NDIT / (
                                    r_corr * size_core ** 2)))  # bruit total en e- dans la boite intégré pendant le temps d'exposition
                    sigma_tot = np.nanmean(sigma_tot, 1)
                    sigma_tot_wo_syst = np.sqrt(NDIT * r_corr_px * size_core ** 2 * (
                                sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))  # bruit total en e- dans la boite intégré pendant le temps d'exposition
                    sigma_tot_wo_syst = np.nanmean(sigma_tot_wo_syst, 1)
                    sep_data, _, _, sigma_lambda_spatial_2, sigma_lambda_ERR_2 = fits.getdata(
                        'utils/noiseMM/noiseMM_' + band + '.fits')
                    f = interp1d(sep_data,
                                 np.sqrt(sigma_lambda_spatial_2 * np.nansum(star_spectrum_inter.flux / DIT) ** 2),
                                 bounds_error=False, fill_value=np.nan)
                    sigma_lambda_spatial = f(separation)
                    f = interp1d(sep_data, np.sqrt(
                        size_core ** 2 * sigma_lambda_ERR_2 * np.nansum(star_spectrum_inter.flux / DIT) ** 2),
                                 bounds_error=False, fill_value=np.nan)
                    sigma_lambda_ERR = f(separation)
                    sigma_lambda_ERR *= np.nanmean(sigma_tot_wo_syst) / np.nanmean(sigma_lambda_ERR)
                    #sep_data,_,_,sigma_lambda_spatial_2,sigma_lambda_ERR_2 = fits.getdata('utils/noiseMM/noiseMM_'+band+'.fits')  f=interp1d(sep_data,np.sqrt(sigma_lambda_spatial_2), bounds_error=False, fill_value=np.nan)  sigma_lambda_spatial = f(separation)  f=interp1d(sep_data,np.sqrt(size_core**2*sigma_lambda_ERR_2), bounds_error=False, fill_value=np.nan)  sigma_lambda_ERR = f(separation)  sigma_lambda_ERR *= np.nanmean(sigma_tot_wo_syst)/np.nanmean(sigma_lambda_ERR)
                    axs[0].scatter(sigma_lambda_spatial, sigma_tot, label=band, c=cmap(nb), zorder=2)
                    axs[1].scatter(sigma_lambda_ERR, sigma_tot_wo_syst, label=band, c=cmap(nb), zorder=2)
                    #plt.figure()  plt.plot(separation,sigma_tot,'r')  plt.plot(separation,sigma_lambda_spatial,'b')  plt.yscale('log')  plt.yscale('log')  plt.title(f"{band}")  plt.show()
                    if 1 == 0:
                        points_syst = np.zeros((2, len(sigma_lambda_spatial)))
                        points_syst[0] = sigma_tot
                        points_syst[1] = sigma_lambda_spatial
                        fits.writeto("utils/points/points_syst_" + band + "_" + name_planet + ".fits", points_syst,
                                     overwrite=True)
                        points_wo_syst = np.zeros((2, len(sigma_lambda_spatial)))
                        points_wo_syst[0] = sigma_tot_wo_syst
                        points_wo_syst[1] = sigma_lambda_ERR
                        fits.writeto("utils/points/points_wo_syst_" + band + "_" + name_planet + ".fits",
                                     points_wo_syst, overwrite=True)

                    if nb == len(config_data["gratings"]) - 1:
                        lims = [np.min([axs[0].get_xlim(), axs[0].get_ylim()]),
                                np.max([axs[0].get_xlim(), axs[0].get_ylim()])]
                        axs[0].plot(lims, lims, 'k-')
                        axs[1].plot(lims, lims, 'k-')
                        axs[1].legend()

            contrast_band.append(contrast)  # on rajoute à la liste la courbe de contraste de la bande



        # Calcul du SNR

        elif calculation == "SNR":

            if post_processing == "molecular mapping":
                if systematic:
                    SNR = signal * np.sqrt(NDIT) / np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2 + sigma_syst_prime_2 * NDIT / (
                                    R_corr * size_core ** 2)))  # avec systématiques
                else:
                    SNR = signal * np.sqrt(NDIT) / np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_prime_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_prime_2))  # sans systématiques

            elif post_processing == "ADI+RDI":
                if systematic:
                    SNR = signal * np.sqrt(NDIT) / np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2 + sigma_syst_2 * NDIT / (
                                    R_corr * size_core ** 2)))  # avec systématiques
                else:
                    SNR = signal * np.sqrt(NDIT) / np.sqrt(R_corr * size_core ** 2 * (
                                sigma_halo_2 + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))  # sans systématiques

            if separation_planet is not None:
                SNR_planet_band.append(SNR[idx])
                if print_value:
                    print(f' SNR at {round(separation_planet, 1)} {sep_unit} = ', round(SNR[idx], 2))
                if np.max(SNR) > SNR_max:
                    SNR_max = np.max(SNR)
                if SNR[idx] > SNR_max_planet:
                    SNR_max_planet = SNR[idx]
                    name_max_SNR = band

            # Effet du bruit sur la corrélation

            if show_cos_theta_est and band == band_plot and calculation == "SNR":  # calcul du cos theta n (impact du bruit (et de l'auto-soustraction) sur l'estimation de la corrélation)
                Sp = NDIT * planet_spectrum_inter.flux * fraction_PSF  # flux de la planète (avec modulations) en e- tot/Afwhm
                Mp_Sp_HF, Mp_Sp_BF = filtered_flux(Sp / trans, R, Rc, used_filter)
                star_HF, star_BF = filtered_flux(star_spectrum_inter.flux / trans, R, Rc, used_filter)
                alpha = np.sqrt(np.nansum((trans * Mp_Sp_HF) ** 2))
                beta = np.nansum(trans * star_HF * Mp_Sp_BF / star_BF * template)
                cos_theta_lim = np.nansum(trans * Mp_Sp_HF * template) / alpha
                for i in range(len(separation)):  # pour chaque separation
                    star_flux = PSF_profile[
                                    i] * star_spectrum_inter.flux  # flux stellaire en e-/DIT/px en fonction de lambda à la séparation i
                    star_flux[star_flux > saturation_e] = saturation_e  # 
                    sigma_tot = np.sqrt(R_corr[i] * size_core ** 2 * NDIT * (
                                sigma_halo_2[i] + sigma_ron_2 + sigma_dc_2 + sigma_bkgd_2))  # bruit en e- tot/Afwhm
                    N = 1000
                    cos_theta_est_n = np.zeros((N))
                    norm_d_n = np.zeros((N))
                    for n in range(len(cos_theta_est_n)):
                        noise = np.random.normal(0, sigma_tot, len(wave_inter))
                        d = trans * Mp_Sp_HF + noise - trans * star_HF * Mp_Sp_BF / star_BF + M_HF[
                            i] * NDIT * size_core ** 2 * star_flux
                        norm_d_n[n] = np.sqrt(np.nansum(d ** 2))
                        cos_theta_est_n[n] = np.nansum(d * template) / norm_d_n[n]
                    norm_d[i] = np.nanmean(norm_d_n)
                    cos_theta_est[i] = np.nanmean(cos_theta_est_n)
                    cos_theta_est[i] = alpha / norm_d[i] * cos_theta_lim - beta / norm_d[i]
                    if separation_planet is not None and i == idx:
                        print(" SNR PER SPECTRAL CHANNEL = ", round(np.nanmean(NDIT * Sp / sigma_tot), 3))
                cos_theta_n = alpha / norm_d
                plt.figure()
                plt.plot(separation, cos_theta_est, 'k')
                plt.ylabel(r"cos $\theta_{est}$", fontsize=14)
                plt.grid(True)
                plt.title(
                    f"Effect of noise and stellar substraction on the estimation of correlation between \n template and planetary spectrum for {instru} on {band} \n (assuming that the template is the same as the observed spectrum)")
                plt.xlabel(f'separation (in {sep_unit})', fontsize=14)
                if separation_planet is not None and separation_planet < np.nanmax(separation):
                    plt.plot([separation_planet, separation_planet],
                             [np.nanmin(cos_theta_est), np.nanmax(cos_theta_est)], 'k--',
                             label=f'angular separation{for_name_planet}')
                    plt.plot([separation_planet, separation_planet], [cos_theta_est[idx], cos_theta_est[idx]], 'rX',
                             ms=11,
                             label=r"cos $\theta_{est}$" + for_name_planet + f' ({round(cos_theta_est[idx], 2)})')
                    plt.legend()
                plt.show()
                if separation_planet is not None and separation_planet < np.nanmax(separation):
                    cos_theta_est = np.linspace(0, 1, N)
                    cos_theta_p = (cos_theta_est / cos_theta_n[idx] + beta / alpha) / cos_theta_lim
                    cos_theta_est = cos_theta_est[cos_theta_p > 0]
                    cos_theta_p = cos_theta_p[cos_theta_p > 0]
                    cos_theta_est = cos_theta_est[cos_theta_p < 1]
                    cos_theta_p = cos_theta_p[cos_theta_p < 1]
                    print(" beta/alpha = ", round(beta / alpha, 3), "\n cos_theta_n = ", round(cos_theta_n[idx], 3),
                          "\n cos_theta_lim = ", round(cos_theta_lim, 3))
                    plt.figure()
                    plt.plot(cos_theta_est, cos_theta_p, 'r')
                    plt.grid(True)
                    plt.ylabel(r"cos $\theta_p$", fontsize=14)
                    plt.xlabel(r"cos $\theta_{est}$", fontsize=14)
                    plt.title(
                        f"Effect of noise and self-subtraction on correlation estimation \n {for_name_planet} on {band} with $R_c$ = {Rc}",
                        fontsize=14)
                    if cos_est is not None:
                        index = np.abs(cos_est - cos_theta_est).argmin()
                        plt.plot([0, cos_theta_est[index]], [cos_theta_p[index], cos_theta_p[index]], 'k--',
                                 label=r'estimated correlation : cos $\theta_{est}$ ' + f'= {cos_est}, \n ' + r'mismatch deduced : cos $\theta_{p}$ = ' + f'{round(cos_theta_p[index], 2)}')
                        plt.plot([cos_theta_est[index], cos_theta_est[index]], [0, cos_theta_p[index]], 'k--')
                        plt.legend(fontsize=12)
                        plt.ylim(0, 1)
                        plt.xlim(0, np.nanmax(cos_theta_est))
                    plt.show()

            SNR_band.append(SNR)



    #------------------------------------------------------------------------------------------------
    # PARTIE PLOT :
    #------------------------------------------------------------------------------------------------

    # Contraste :

    if calculation == "contrast" and show_plot:
        plt.figure()
        plt.grid(True)
        if separation_planet is not None and separation_planet < np.nanmax(separation):
            if apodizer is not None:
                plt.plot([separation_planet, separation_planet],
                         [np.nanmin(contrast_band[0][separation_band[0] > sep_min]),
                          np.nanmax(contrast_band[0][separation_band[0] > sep_min])], 'k--')
            else:
                plt.plot([separation_planet, separation_planet],
                         [np.nanmin(contrast_band[0]), np.nanmax(contrast_band[0])], 'k--')
        if coronagraph is not None:
            plt.title(instru + " contrast curves" + for_name_planet + " with coronagraph \n (with $t_{exp}$=" + str(
                round(time, 1)) + "mn and $mag_*$(" + band0 + ")=" + str(mag_star) + f')', fontsize=14)
        else:
            plt.title(
                instru + " contrast curves" + for_name_planet + f" with {post_processing}" + " \n (with $t_{exp}$=" + str(
                    round(time, 1)) + "mn, $mag_*$(" + band0 + ")=" + str(
                    round(mag_star, 1)) + f' and $T_p$={round(T_planet)}K )', fontsize=14)
        plt.xlabel(f"separation (in {sep_unit})", fontsize=12)
        if plot_mag:
            plt.ylabel(r'5$\sigma$ contrast ($\Delta$mag)', fontsize=12)
        else:
            plt.ylabel(r'5$\sigma$ contrast (on instru-band)', fontsize=12)
        ax = plt.gca()
        if plot_mag:
            ax.invert_yaxis()
        else:
            plt.yscale('log')
        ax.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
        cmap = get_cmap('Spectral', len(contrast_band))
        for i in range(len(contrast_band)):
            ax.plot(separation_band[i][separation_band[i] > sep_min], contrast_band[i][separation_band[i] > sep_min],
                    label="Band " + name_band[i], color=cmap(i))
        ax.legend(loc='upper right')
        ax.set_xlim(0)
        if separation_planet is not None and separation_planet < np.nanmax(separation):
            ax_legend = ax.twinx()
            ax_legend.plot([], [], '--', c='k', label=f'angular separation{for_name_planet}')
            ax_legend.legend(loc='lower left')
            ax_legend.tick_params(axis='y', colors='w')
        ax.yaxis.set_ticks_position('both')
        plt.tight_layout()
        plt.show()

    #------------------------------------------------------------------------------------------------

    # SNR :

    elif calculation == "SNR" and show_plot:
        if separation_planet is not None:
            print('\n')
            print(f'MAX SNR (at {round(separation_planet, 1)} {sep_unit}) = ', round(SNR_max_planet, 1),
                  "for " + name_max_SNR)
        if channel and instru == 'MIRIMRS':
            time *= 3
            SNR_chan = []
            separation_chan = []
            separation_chan.append(separation_band[0])
            separation_chan.append(separation_band[3])
            SNR_chan1 = np.zeros(len(SNR_band[0]))
            SNR_chan2 = np.zeros(len(SNR_band[3]))
            name_band[0] = "channel 1"
            name_band[1] = "channel 2"
            SNR_max_planet = 0.
            SNR_chan.append(np.sqrt(np.nansum(np.array(SNR_band[:3]) ** 2, 0)))
            SNR_chan.append(np.sqrt(np.nansum(np.array(SNR_band[3:]) ** 2, 0)))
            SNR_max = max(max(SNR_chan[0]), max(SNR_chan[1]))

            if separation_planet is not None:
                for i in range(len(SNR_chan)):
                    idx = (np.abs(separation_chan[i] - separation_planet)).argmin()
                    if SNR_chan[i][idx] > SNR_max_planet:
                        SNR_max_planet = SNR_chan[i][idx]
                        name_max_SNR = name_band[i]
                print(f'MAX SNR (at {separation_planet}") = ', round(SNR_max_planet, 2), "for " + name_max_SNR)
            separation_band = separation_chan
            SNR_band = SNR_chan
        plt.figure()
        plt.grid(True)
        if separation_planet is not None and separation_planet < np.nanmax(separation):
            plt.plot([separation_planet, separation_planet], [0., SNR_max], 'k--')
        if coronagraph is not None:
            plt.title(instru + " SNR curves" + for_name_planet + " with coronagraph \n (with $t_{exp}$=" + str(
                round(time)) + "mn, $mag_*$(" + band0 + ")=" + str(
                round(mag_star, 2)) + ", $mag_p$(" + band0 + ")=" + str(
                round(mag_planet, 2)) + f' and $T_p$ = {T_planet}K)', fontsize=14)
        else:
            plt.title(instru + " SNR curves" + for_name_planet + " with $t_{exp}$=" + str(
                round(time)) + "mn, \n $mag_*$(" + band0 + ")=" + str(
                round(mag_star, 2)) + ", $mag_p$(" + band0 + ")=" + str(
                round(mag_planet, 2)) + f' and $T_p$ = {round(T_planet)}K (' + model + ')', fontsize=14)
        plt.xlabel(f"separation (in {sep_unit})", fontsize=14)
        plt.ylabel('SNR', fontsize=14)
        ax = plt.gca()
        ax.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
        cmap = get_cmap("Spectral", len(SNR_band))
        for i in range(len(SNR_band)):
            ax.plot(separation_band[i], SNR_band[i], label=name_band[i], color=cmap(i))
        ax.legend(loc='upper left')
        ax.set_xlim(0)
        ax.set_ylim(0)
        if separation_planet is not None and separation_planet < np.nanmax(separation):
            ax_legend = ax.twinx()
            ax_legend.plot([], [], '--', c='k', label=f'angular separation{for_name_planet}')
            ax_legend.plot([], [], 'X', c='r',
                           label='max SNR' + for_name_planet + ' (' + str(round(SNR_max_planet, 2)) + ')')
            ax_legend.legend(loc='lower right')
            ax_legend.tick_params(axis='y', colors='w')
            ax.plot([separation_planet, separation_planet], [SNR_max_planet, SNR_max_planet], 'rX', ms=11)
        plt.tight_layout()
        plt.show()

    #------------------------------------------------------------------------------------------------
    # RETURNS :
    #------------------------------------------------------------------------------------------------

    if calculation == "contrast":
        return name_band, separation_band, contrast_band, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    elif calculation == "SNR":
        return name_band, separation_band, SNR_band, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    elif calculation == "Noise":
        return name_band, separation_band, sigma_ns_2_band, NDIT


#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################


def mirimrs(calculation, T_planet, lg_planet, mag_star, band0, T_star, lg_star, exposure_time, systematic=True,
            model="BT-Settl", mag_planet=None, separation_planet=None, name_planet=None, syst_radial_velocity=0,
            delta_radial_velocity=0, star_broadening=0, planet_broadening=0, channel=False, plot_mag=False,
            return_SNR_planet=False, return_quantity=False, show_plot=True, print_value=True,
            post_processing="molecular mapping", bkgd="medium", star_pos="center", star_spectrum=None,
            planet_spectrum=None, Rc=100, band_only=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    instrument = "MIRIMRS"  # Instrument considéré
    sep_unit = "arcsec"
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star)  # Spectre de l'étoile GNS93
    if planet_spectrum is None:
        if model == "Exo-REM":
            planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model,
                                                   version="old")  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
        else:
            planet_spectrum = load_planet_spectrum(T_planet, lg_planet, model,
                                                   version="new")  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
    if syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        star_spectrum = star_spectrum.doppler_shift(syst_radial_velocity)
        star_spectrum.syst_rv = syst_radial_velocity
    if delta_radial_velocity != 0 or syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        planet_spectrum = planet_spectrum.doppler_shift(syst_radial_velocity + delta_radial_velocity)
        planet_spectrum.delta_rv = delta_radial_velocity
    if planet_broadening != 0:  # Elargissement rotationnel (en km/s)
        planet_spectrum = planet_spectrum.broad(planet_broadening)
    if star_broadening != 0:  # Elargissement rotationnel (en km/s)
        star_spectrum = star_spectrum.broad(star_broadening)
    tellurics = False  # Effet de l'atmosphère (False)
    name_band, separation, curves, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band = FastCurves(calculation,
                                                                                                       instrument,
                                                                                                       exposure_time,
                                                                                                       name_planet,
                                                                                                       mag_star, band0,
                                                                                                       planet_spectrum,
                                                                                                       star_spectrum,
                                                                                                       tellurics,
                                                                                                       apodizer=None,
                                                                                                       strehl=None,
                                                                                                       mag_planet=mag_planet,
                                                                                                       separation_planet=separation_planet,
                                                                                                       channel=channel,
                                                                                                       systematic=systematic,
                                                                                                       plot_mag=plot_mag,
                                                                                                       show_plot=show_plot,
                                                                                                       print_value=print_value,
                                                                                                       post_processing=post_processing,
                                                                                                       sep_unit=sep_unit,
                                                                                                       bkgd=bkgd,
                                                                                                       star_pos=star_pos,
                                                                                                       Rc=Rc,
                                                                                                       band_only=band_only)
    if return_SNR_planet:  # POUR FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1000  # on passe la séparation de la planète en mas également
        config_data = get_config_data(instrument)
        SNR_planet = np.zeros((len(config_data["gratings"])))
        signal_planet = np.zeros((len(config_data["gratings"])))
        sigma_ns_planet = np.zeros((len(config_data["gratings"])))
        sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]):
            idx = np.abs(separation[nb] - separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_band[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_band[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_band[nb][idx])
        return name_band, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_band)
    elif return_quantity:
        return name_band, separation, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    else:
        return name_band, separation, curves


#################################################################################################################################################################################################


def harmoni(calculation, T_planet, lg_planet, mag_star, band0, T_star, lg_star, exposure_time, systematic=False,
            apodizer="SP1", strehl="MED", model="BT-Settl", mag_planet=None, separation_planet=None, name_planet=None,
            syst_radial_velocity=0, delta_radial_velocity=0, star_broadening=0, planet_broadening=0, plot_mag=False,
            return_SNR_planet=False, return_quantity=False, show_plot=True, print_value=True,
            post_processing="molecular mapping", star_spectrum=None, planet_spectrum=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    instrument = "HARMONI"  # Instrument considéré  # Apodiseur utilisé (NO_SP,SP1,SP2) # Strehl (JQ0,JQ1,MED)
    sep_unit = "mas"
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star)  # Spectre de l'étoile GNS93
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet,
                                               model)  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
    if syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        star_spectrum = star_spectrum.doppler_shift(syst_radial_velocity)
        star_spectrum.syst_rv = syst_radial_velocity
    if delta_radial_velocity != 0 or syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        planet_spectrum = planet_spectrum.doppler_shift(syst_radial_velocity + delta_radial_velocity)
        planet_spectrum.delta_rv = delta_radial_velocity
    if planet_broadening != 0:  # Elargissement rotationnel (en km/s)
        planet_spectrum = planet_spectrum.broad(planet_broadening)
    if star_broadening != 0:  # Elargissement rotationnel (en km/s)
        star_spectrum = star_spectrum.broad(star_broadening)
    tellurics = True  # Effet de l'atmosphère (True)
    name_band, separation, curves, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band = FastCurves(calculation,
                                                                                                       instrument,
                                                                                                       exposure_time,
                                                                                                       name_planet,
                                                                                                       mag_star, band0,
                                                                                                       planet_spectrum,
                                                                                                       star_spectrum,
                                                                                                       tellurics,
                                                                                                       apodizer, strehl,
                                                                                                       mag_planet=mag_planet,
                                                                                                       separation_planet=separation_planet,
                                                                                                       plot_mag=plot_mag,
                                                                                                       show_plot=show_plot,
                                                                                                       print_value=print_value,
                                                                                                       post_processing=post_processing,
                                                                                                       sep_unit=sep_unit,
                                                                                                       bkgd=None)
    if return_SNR_planet:  # POUR FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1000  # on passe la séparation de la planète en mas également
        config_data = get_config_data(instrument)
        SNR_planet = np.zeros((len(config_data["gratings"])))
        signal_planet = np.zeros((len(config_data["gratings"])))
        sigma_ns_planet = np.zeros((len(config_data["gratings"])))
        sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]):
            idx = np.abs(separation[nb] - separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_band[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_band[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_band[nb][idx])
        return name_band, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_band)
    elif return_quantity:
        return name_band, separation, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    else:
        return name_band, separation, curves


#################################################################################################################################################################################################


def eris(calculation, T_planet, lg_planet, mag_star, band0, T_star, lg_star, exposure_time, systematic=False,
         strehl="JQ0", model="BT-Settl", mag_planet=None, separation_planet=None, name_planet=None,
         syst_radial_velocity=0, delta_radial_velocity=0, star_broadening=0, planet_broadening=0, plot_mag=False,
         return_SNR_planet=False, return_quantity=False, show_plot=True, print_value=True,
         post_processing="molecular mapping", star_spectrum=None, planet_spectrum=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    instrument = "ERIS"  # Instrument considéré
    sep_unit = "mas"
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star)  # Spectre de l'étoile GNS93
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet,
                                               model)  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
    if syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        star_spectrum = star_spectrum.doppler_shift(syst_radial_velocity)
        star_spectrum.syst_rv = syst_radial_velocity
    if delta_radial_velocity != 0 or syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        planet_spectrum = planet_spectrum.doppler_shift(syst_radial_velocity + delta_radial_velocity)
        planet_spectrum.delta_rv = delta_radial_velocity
    if planet_broadening != 0:  # Elargissement rotationnel (en km/s)
        planet_spectrum = planet_spectrum.broad(planet_broadening)
    if star_broadening != 0:  # Elargissement rotationnel (en km/s)
        star_spectrum = star_spectrum.broad(star_broadening)
    tellurics = True  # Effet de l'atmosphère (False)
    name_band, separation, curves, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band = FastCurves(calculation,
                                                                                                       instrument,
                                                                                                       exposure_time,
                                                                                                       name_planet,
                                                                                                       mag_star, band0,
                                                                                                       planet_spectrum,
                                                                                                       star_spectrum,
                                                                                                       tellurics,
                                                                                                       apodizer="NO_SP",
                                                                                                       strehl=strehl,
                                                                                                       mag_planet=mag_planet,
                                                                                                       separation_planet=separation_planet,
                                                                                                       plot_mag=plot_mag,
                                                                                                       show_plot=show_plot,
                                                                                                       print_value=print_value,
                                                                                                       post_processing=post_processing,
                                                                                                       sep_unit=sep_unit,
                                                                                                       bkgd=None)
    if return_SNR_planet:  # POUR FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1000  # on passe la séparation de la planète en mas également
        config_data = get_config_data(instrument)
        SNR_planet = np.zeros((len(config_data["gratings"])))
        signal_planet = np.zeros((len(config_data["gratings"])))
        sigma_ns_planet = np.zeros((len(config_data["gratings"])))
        sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]):
            idx = np.abs(separation[nb] - separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_band[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_band[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_band[nb][idx])
        return name_band, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_band)
    elif return_quantity:
        return name_band, separation, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    else:
        return name_band, separation, curves


#################################################################################################################################################################################################


def nircam(calculation, T_planet, lg_planet, mag_star, band0, T_star, lg_star, exposure_time, systematic=False,
           model="BT-Settl", mag_planet=None, separation_planet=None, name_planet=None, syst_radial_velocity=0,
           delta_radial_velocity=0, star_broadening=0, planet_broadening=0, plot_mag=False, return_SNR_planet=False,
           return_quantity=False, show_plot=True, print_value=True, post_processing="ADI+RDI", bkgd="medium",
           star_spectrum=None, planet_spectrum=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    instrument = "NIRCam"  # Instrument considéré 
    sep_unit = "arcsec"
    coronagraph = "MASK335R"  # Coronographe (False)
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star)  # Spectre de l'étoile GNS93
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet,
                                               model)  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
    if syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        star_spectrum = star_spectrum.doppler_shift(syst_radial_velocity)
        star_spectrum.syst_rv = syst_radial_velocity
    if delta_radial_velocity != 0 or syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        planet_spectrum = planet_spectrum.doppler_shift(syst_radial_velocity + delta_radial_velocity)
        planet_spectrum.delta_rv = delta_radial_velocity
    if planet_broadening != 0:  # Elargissement rotationnel (en km/s)
        planet_spectrum = planet_spectrum.broad(planet_broadening)
    if star_broadening != 0:  # Elargissement rotationnel (en km/s)
        star_spectrum = star_spectrum.broad(star_broadening)
    tellurics = False  # Effet de l'atmosphère (True)
    name_band, separation, curves, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band = FastCurves(calculation,
                                                                                                       instrument,
                                                                                                       exposure_time,
                                                                                                       name_planet,
                                                                                                       mag_star, band0,
                                                                                                       planet_spectrum,
                                                                                                       star_spectrum,
                                                                                                       tellurics,
                                                                                                       coronagraph=coronagraph,
                                                                                                       mag_planet=mag_planet,
                                                                                                       separation_planet=separation_planet,
                                                                                                       plot_mag=plot_mag,
                                                                                                       show_plot=show_plot,
                                                                                                       print_value=print_value,
                                                                                                       post_processing=post_processing,
                                                                                                       sep_unit=sep_unit,
                                                                                                       bkgd=bkgd)
    if return_SNR_planet:  # POUR FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1000  # on passe la séparation de la planète en mas également
        config_data = get_config_data(instrument)
        SNR_planet = np.zeros((len(config_data["gratings"])))
        signal_planet = np.zeros((len(config_data["gratings"])))
        sigma_ns_planet = np.zeros((len(config_data["gratings"])))
        sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]):
            idx = np.abs(separation[nb] - separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_band[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_band[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_band[nb][idx])
        return name_band, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_band)
    elif return_quantity:
        return name_band, separation, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    else:
        return name_band, separation, curves


#################################################################################################################################################################################################


def andes(calculation, T_planet, lg_planet, mag_star, band0, T_star, lg_star, exposure_time, systematic=False,
          strehl="JQ1", model="BT-Settl", mag_planet=None, separation_planet=None, name_planet=None,
          syst_radial_velocity=0, delta_radial_velocity=0, star_broadening=0, planet_broadening=0, plot_mag=False,
          return_SNR_planet=False, return_quantity=False, show_plot=True, print_value=True,
          post_processing="molecular mapping", star_spectrum=None, planet_spectrum=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    instrument = "ANDES"  # Instrument considéré
    sep_unit = "mas"
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star)  # Spectre de l'étoile GNS93
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet,
                                               model)  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
    if syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        star_spectrum = star_spectrum.doppler_shift(syst_radial_velocity)
        star_spectrum.syst_rv = syst_radial_velocity
    if delta_radial_velocity != 0 or syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        planet_spectrum = planet_spectrum.doppler_shift(syst_radial_velocity + delta_radial_velocity)
        planet_spectrum.delta_rv = delta_radial_velocity
    if planet_broadening != 0:  # Elargissement rotationnel (en km/s)
        planet_spectrum = planet_spectrum.broad(planet_broadening)
    if star_broadening != 0:  # Elargissement rotationnel (en km/s)
        star_spectrum = star_spectrum.broad(star_broadening)
    tellurics = True  # Effet de l'atmosphère (False)
    name_band, separation, curves, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band = FastCurves(calculation,
                                                                                                       instrument,
                                                                                                       exposure_time,
                                                                                                       name_planet,
                                                                                                       mag_star, band0,
                                                                                                       planet_spectrum,
                                                                                                       star_spectrum,
                                                                                                       tellurics,
                                                                                                       apodizer="NO_SP",
                                                                                                       strehl=strehl,
                                                                                                       mag_planet=mag_planet,
                                                                                                       separation_planet=separation_planet,
                                                                                                       plot_mag=plot_mag,
                                                                                                       show_plot=show_plot,
                                                                                                       print_value=print_value,
                                                                                                       post_processing=post_processing,
                                                                                                       sep_unit=sep_unit,
                                                                                                       bkgd=None)
    if return_SNR_planet:  # POUR FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1000  # on passe la séparation de la planète en mas également
        config_data = get_config_data(instrument)
        SNR_planet = np.zeros((len(config_data["gratings"])))
        signal_planet = np.zeros((len(config_data["gratings"])))
        sigma_ns_planet = np.zeros((len(config_data["gratings"])))
        sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]):
            idx = np.abs(separation[nb] - separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_band[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_band[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_band[nb][idx])
        return name_band, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_band)
    elif return_quantity:
        return name_band, separation, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    else:
        return name_band, separation, curves


#################################################################################################################################################################################################


def nirspec(calculation, T_planet, lg_planet, mag_star, band0, T_star, lg_star, exposure_time, systematic=False,
            model="BT-Settl", mag_planet=None, separation_planet=None, name_planet=None, syst_radial_velocity=0,
            delta_radial_velocity=0, star_broadening=0, planet_broadening=0, plot_mag=False, return_SNR_planet=False,
            return_quantity=False, show_plot=True, print_value=True, post_processing="molecular mapping", bkgd="medium",
            star_spectrum=None, planet_spectrum=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    instrument = "NIRSpec"  # Instrument considéré
    sep_unit = "arcsec"
    if star_spectrum is None:
        star_spectrum = load_star_spectrum(T_star, lg_star)  # Spectre de l'étoile GNS93
    if planet_spectrum is None:
        planet_spectrum = load_planet_spectrum(T_planet, lg_planet,
                                               model)  #  Spectre de la planète : class Spectrum(wavel,flux,R,T) en J/s/m²/µm
    if syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        star_spectrum = star_spectrum.doppler_shift(syst_radial_velocity)
        star_spectrum.syst_rv = syst_radial_velocity
    if delta_radial_velocity != 0 or syst_radial_velocity != 0:  # décalage Doppler (en km/s)
        planet_spectrum = planet_spectrum.doppler_shift(syst_radial_velocity + delta_radial_velocity)
        planet_spectrum.delta_rv = delta_radial_velocity
    if planet_broadening != 0:  # Elargissement rotationnel (en km/s)
        planet_spectrum = planet_spectrum.broad(planet_broadening)
    if star_broadening != 0:  # Elargissement rotationnel (en km/s)
        star_spectrum = star_spectrum.broad(star_broadening)
    tellurics = False  # Effet de l'atmosphère (False)
    name_band, separation, curves, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band = FastCurves(calculation,
                                                                                                       instrument,
                                                                                                       exposure_time,
                                                                                                       name_planet,
                                                                                                       mag_star, band0,
                                                                                                       planet_spectrum,
                                                                                                       star_spectrum,
                                                                                                       tellurics,
                                                                                                       apodizer=None,
                                                                                                       strehl=None,
                                                                                                       mag_planet=mag_planet,
                                                                                                       separation_planet=separation_planet,
                                                                                                       plot_mag=plot_mag,
                                                                                                       show_plot=show_plot,
                                                                                                       print_value=print_value,
                                                                                                       post_processing=post_processing,
                                                                                                       sep_unit=sep_unit,
                                                                                                       bkgd=bkgd)
    if return_SNR_planet:  # POUR FASTYIELD
        if calculation != "SNR":
            raise KeyError("THE CALCULATION NEED TO BE SET ON SNR !")
        if separation_planet is None:
            raise KeyError("PLEASE INPUT A SEPARATION FOR THE PLANET FOR THE SNR CALCULATION !")
        if sep_unit == "mas":
            separation_planet *= 1000  # on passe la séparation de la planète en mas également
        config_data = get_config_data(instrument)
        SNR_planet = np.zeros((len(config_data["gratings"])))
        signal_planet = np.zeros((len(config_data["gratings"])))
        sigma_ns_planet = np.zeros((len(config_data["gratings"])))
        sigma_s_planet = np.zeros((len(config_data["gratings"])))
        for nb, band in enumerate(config_data["gratings"]):
            idx = np.abs(separation[nb] - separation_planet).argmin()
            SNR_planet[nb] = curves[nb][idx]
            signal_planet[nb] = signal_band[nb][idx]
            sigma_ns_planet[nb] = np.sqrt(sigma_ns_2_band[nb][idx])
            if systematic:
                sigma_s_planet[nb] = np.sqrt(sigma_s_2_band[nb][idx])
        return name_band, SNR_planet, signal_planet, sigma_ns_planet, sigma_s_planet, np.array(DIT_band)
    elif return_quantity:
        return name_band, separation, signal_band, sigma_s_2_band, sigma_ns_2_band, DIT_band
    else:
        return name_band, separation, curves
