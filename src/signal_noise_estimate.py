from src.molecular_mapping import *



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def transmission(instru, wave_band, band, tellurics, apodizer, fill_value=np.nan):
    """
    To read and retrieve instrumental and sky (if needed) transmission

    Parameters
    ----------
    instru: str
        considered instrument
    wave_band: 1d array
        wavelength axis of the considered spectral band
    band: str
        considered spectral band
    R: float
        spectral resolution of the considered spectral band
    tellurics: bool (True or False)
        considering (or not) the earth atmosphere (ground or space observations)

    Returns
    -------
    transmission: 1d array (same size as wave_band)
        total system transmission 
    """
    wave, trans = fits.getdata("sim_data/Transmission/"+instru+"/transmission_" + band + ".fits") # instrumental transmission on the considered band
    f = interp1d(wave, trans, bounds_error=False, fill_value=fill_value)
    trans = f(wave_band) # interpolated instrumental transmission on the considered band
    config_data = get_config_data(instru) # get instrument specs
    apodizer_trans = config_data["apodizers"][str(apodizer)].transmission # get apodizer transmission, if any
    trans *= apodizer_trans
    if instru == "MIRIMRS" or instru == "NIRSpec":
        trans *= fits.getheader("sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_NO_JQ_NO_SP.fits")['AC'] # aperture corrective factor (the fact that not all the incident flux reaches the FOV)
    if tellurics: # if ground-based observation
        sky_transmission_path = os.path.join("sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans             = fits.getdata(sky_transmission_path)
        trans_tell_band       = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        trans_tell_band       = trans_tell_band.degrade_resolution(wave_band, renorm=False).flux # degraded tellurics transmission on the considered band
        trans                *= trans_tell_band # total system throughput (instru x atmo)
    trans[trans<=0] = np.nan # ignoring negatives values (if any)
    return trans

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def PSF_profile_fraction_separation(band, strehl, apodizer, coronagraph, instru, config_data, sep_unit):
    """
    Gives the PSF profile, the fraction of flux contained in the PSF core (or coronagraphic transmission) and the separation array.
    
    Parameters
    ----------
    band: str
        considered spectral band
    strehl: str
        strehl ratio
    apodizer: str
        apodizer
    coronagraph: str
        coronagraph
    instru: str
        instrument
    config_data: collections
        gives the parameters of the considered instrument

    Returns
    -------
    PSF_profile: 1d-array
        PSF profile
    fraction_PSF: float
        fraction core or coronagraphic transmission
    separation: 1d-array
        separation vector (in arcsec or mas)
    """
    if coronagraph is None: # Coronographic (stellar) PSF profile
        file = "sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_"+strehl+"_"+apodizer+".fits"
    else:
        file = "sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_"+coronagraph+"_"+strehl+"_"+apodizer+".fits"
    fraction_PSF = fits.getheader(file)['FC'] # fraction of flux contained in the FWHM (or the coronagraphic stellar transmission)
    if instru == "MIRIMRS" or instru == "ANDES":
        pxscale = config_data["pxscale"][band] # in arcsec/px (dithered effective pixelscale)
    else:
        pxscale = config_data["spec"]["pxscale"] # in arcsec/px
    if config_data["type"] == "IFU_fiber":
        try:
            FOV = config_data["FOV_fiber"]*pxscale # in arcsec
        except:
            FOV = config_data["spec"]["FOV"] # in arcsec
    else:
        FOV = config_data["spec"]["FOV"] # in arcsec
    if sep_unit == "mas":
        pxscale *= 1e3 ; FOV *= 1e3 # arcsec => mas (if wanted)
    profile = fits.getdata(file) # in fraction/arcsec**2 or fraction/mas**2
    if instru=="HiRISE":
        separation = np.arange(pxscale, FOV/2+pxscale, pxscale) # in arcsec or mas (/10 doesn't change the result but gives smoother curves)
    else:
        separation = np.arange(pxscale/10, FOV/2+pxscale/10, pxscale/10) # in arcsec or mas (/10 doesn't change the result but gives smoother curves)

    #separation = np.arange(pxscale, FOV/2+pxscale, pxscale) # FOR FASTER CALCULATIONS (t_syst calculations)
    f = interp1d(profile[0], profile[1], bounds_error=False, fill_value=np.nan)
    if instru == "MIRIMRS":
        PSF_profile = f(separation) * config_data["pxscale0"][band[0]]**2 # pxscale (non-dithered) (because all the values are considered in the detector space in the first place, then multiplied by R_corr, to take into account the transformation into 3D cube sapce)
    else:
        PSF_profile = f(separation) * pxscale**2
    return PSF_profile, fraction_PSF, separation, pxscale

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def get_alpha(planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, filter_type):
    """
    Compute the total amount of useful photons/min for molecular mapping detection
    
    Parameters
    ----------
    fraction_PSF: float
        fraction of flux of the PSF in size_core**2 pixels
    transmission: array
        instrumental + sky transmission
    sigma: float
        cut-off frequency = parameter for the high-pass filter
        (plus sigma est grand, plus la fréquence de coupure est petite => moins on coupera les basses fréquences)
    
    Returns
    -------
    alpha: float/int
        amount of useful photons
    """
    Sp = planet_spectrum_band.flux*fraction_PSF # planetary flux integrated in the FWHM (in ph/mn)
    Sp_HF, _ = filtered_flux(Sp, R, Rc, filter_type) # high_pass filtered planetary flux
    Sp_HF *= trans # gamma x [Sp]_HF
    alpha = np.nansum(Sp_HF*template) # alpha x cos theta lim (if systematic)
    alpha = np.zeros_like(separation) + alpha # constant as function of the separation
    return alpha # in ph/mn

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, filter_type):
    """
    Gives the value of the self-subtraction term β

    Parameters
    ----------
    star_spectrum_band: class Spectrum
        band-restricted and resolution-degrated star spectrum in photons/min received
    planet_spectrum_band: class Spectrum
        band-restricted and resolution-degrated planet spectrum in photons/min received
    sigma: float
        cut-off frequency = parameter for the high-pass filter (plus sigma est grand, plus la fréquence de coupure est petite => moins on coupera les basses fréquences)
    fraction_PSF: float
        fraction of flux contained in the PSF core of interest
    trans: 1d-array
        total-systemm transmission
    """
    star_HF, star_LF = filtered_flux(star_spectrum_band.flux, R, Rc, filter_type) # star filtered spectra
    planet_HF, planet_LF = filtered_flux(planet_spectrum_band.flux*fraction_PSF, R, Rc, filter_type) # planet filtered spectra
    beta = np.nansum(trans*star_HF*planet_LF/star_LF * template) # self-subtraction term
    beta = np.zeros_like(separation) + beta # constant as function of the separation
    return beta # in ph/mn

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, quantum_efficiency, RON, saturation_e, input_DIT, verbose=True):
    """
    Gives DIT and effective reading noise

    Parameters
    ----------
    instru: str
        considered instrument
    config_data: collections
        gives the parameters of the considered instrument
    apodizer: str
        apodizer
    PSF_profile: 1d-array
        PSF profile
    star_spectrum_band: class Spectrum
        band-restricted and resolution-degrated star spectrum in photons/min received
    exposure_time: float (in min)
        total exposure_time
    min_DIT: float (in min)
        minimum integration exposure_time
    max_DIT: float (in min)
        maximum integration time
    quantum_efficiency: float
        quantum efficiency (in e-/ph)
    RON: float
        Read-Out Noise (in e-)
    saturation_e: float
        full well capacity (in e-)

    Returns
    -------
    DIT: float (in min)
        integration time
    RON_eff: TYPE
        effective Read-Out Noise (e-)
    """
    if instru == "NIRCam": # Calculating the DIT and the saturing DIT
        max_flux_e = np.nanmax(PSF_profile) * np.nansum(star_spectrum_band.flux) * trans * quantum_efficiency
    else: # maximum number of e-/mn in the considered band
        sep_min = config_data["apodizers"][apodizer].sep # separation where the PSF starts
        max_flux_e = np.nanmax(PSF_profile[separation>sep_min]) * star_spectrum_band.flux * trans * quantum_efficiency  
    saturating_DIT = saturation_e/np.nanmax(max_flux_e) # in mn
    if saturating_DIT > max_DIT: # The max DIT is determined by saturation or smearing DIT (the fact that the planet must not move too much angularly during the integration) or by the maximum detector integration time
        DIT = max_DIT
    elif saturating_DIT < min_DIT: 
        DIT = min_DIT
        if verbose:
            print(" Saturated detector even with the shortest integration time")
    else: # otherwise the DIT is given by the saturating DIT
        DIT = saturating_DIT # mn
    if instru == "HiRISE" or instru == "VIPAPYRUS":
        DIT = max_DIT
    if input_DIT is not None: # except if a DIT is input
        DIT = input_DIT # mn
    if DIT > exposure_time: # The DIT cannot be longer than the total exposure time
        DIT = exposure_time # mn
    nb_min_DIT = 1 # "Up the ramp" reading mode: the pose is sequenced in several non-destructive readings to reduce reading noise (see https://en.wikipedia.org/wiki/Signal_averaging).
    if DIT > nb_min_DIT*min_DIT: # choose 4 min_DIT because if intermittent readings are too short, the detector will heat up too quickly => + dark current
        N_i = DIT/(nb_min_DIT*min_DIT) # number of intermittent readings
        RON_eff = RON/np.sqrt(N_i) # effective read out noise (in e-/DIT)
    else:
        RON_eff = RON
    if instru == 'ERIS' and RON_eff < 7: # effective RON low limit for ERIS
        RON_eff = 7
    if RON_eff < 0.5:
        RON_eff = 0.5
    if verbose:
        print(" DIT =", round(DIT*60, 2), "s / Saturating DIT =", round(saturating_DIT, 1), " mn / ", "RON =", round(RON_eff, 3), "e-")
    return DIT, RON_eff



#######################################################################################################################
##################################### SYSTEMATIC NOISE PROFILE CALCULATION: ###########################################
#######################################################################################################################

def systematic_profile(config_data, band, trans, Rc, R, star_spectrum_instru, planet_spectrum_instru, planet_spectrum, wave_band, size_core, filter_type, show_cos_theta_est=False, PCA=False, PCA_mask=False, Nc=20, mag_planet=None, band0=None, separation_planet=None, mag_star=None, target_name=None, verbose=True):
    """
    Estimates the systematic noise profile projected into the CCF

    Parameters
    ----------
    config_data: dictonnary
        instrument specs
    band: str
        spectral band considered.
    trans: 1d-array
        total system throughput
    Rc: float
        cut-off resolution
    R: float
        spectral resolution of the band
    star_spectrum_instru: class Spectrum()
        star spectrum considered in FastCurves on the instrumental bandwidth
    planet_spectrum_instru: class Spectrum()
        raw planet spectrum considered in FastCurves
    wave_band: 1d-array
        wavelength array of the considered band
    size_core: float
        size of the FWHM boxes
    filter_type: str
        used filter method
    show_cos_theta_est: bool, optional
        to estimate the correlation that would be measured in the data. The default is False.
    PCA: bool, optional
        to apply PCA to reduce the systematic noise level. The default is False.
    PCA_mask: bool, optional
        to mask the planet while applying the PCA. The default is False.
    Nc: int, optional
        number of principal components of the PCA subtracted. The default is 20.
    mag_planet: float, optional
        planet magnitude (for fake planet injection to estimate the PCA performances). The default is None.
    band0: str, optional
        band where the planet magnitude is given. The default is None.
    separation_planet: TYPE, optional
        planet separation (for fake planet injection to estimate the PCA performances). The default is None.
    """
    instru = config_data["name"] ; pca = None ; data = False ; warnings.simplefilter("ignore", category=RuntimeWarning)
    if instru == "MIRIMRS":
        correction = "all_corrected" # correction = "with_fringes_straylight" # correction applied to the simulated MIRISim noiseless data
        T_star_sim_arr = np.array([4000, 6000, 8000]) ; T_star_sim = T_star_sim_arr[np.abs(star_spectrum_instru.T-T_star_sim_arr).argmin()] # available values for the star temperature for MIRSim noiseless data
        file = "data/MIRIMRS/MIRISim/star_center/star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction ; sigma_outliers = None # simulated MIRISim noiseless data file
        #file = 'data/MIRIMRS/MAST/HD 159222_ch'+band[0]+'-shortmediumlong_s3d' ; data = True ; sigma_outliers = 3 # CALIBRATION DATA => High S/N per spectral channel => M_data 
    elif instru == "NIRSpec":
        file = 'data/NIRSpec/MAST/HD 163466_nirspec_'+band+'_s3d' ; data = True ; sigma_outliers = 3 # CALIBRATION DATA => High S/N per spectral channel => M_data: see Section 2.3 + 3.3 of Martos et al (2024)
    try: # if the files already exist
        if data: 
            S_noiseless = fits.getdata("sim_data/Systematics/"+instru+"/S_data_star_center_s3d_"+band+".fits") # on-sky data cube used to estimate the modulations (in e-/mn)
            pxscale = fits.getheader("sim_data/Systematics/"+instru+"/S_data_star_center_s3d_"+band+".fits")['pxscale'] # in arcsec/px
            wave = fits.getdata("sim_data/Systematics/"+instru+"/wave_data_star_center_s3d_"+band+".fits") # wavelength array of the data
        else:
            S_noiseless = fits.getdata("sim_data/Systematics/"+instru+"/S_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits")  # MIRISim noiseless data cube used to estimate the modulations (in e-/mn)
            pxscale = fits.getheader("sim_data/Systematics/"+instru+"/S_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits")['pxscale'] # in arcsec/px
            wave = fits.getdata("sim_data/Systematics/"+instru+"/wave_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits") # wavelength array of the data
    except: # in case they don't, create them (but the raw data are needed)
        file += ".fits"
        S_noiseless, wave, pxscale, _, _, exposure_time, _ = extract_jwst_data(instru, "sim", band, crop_band=True, cosmic=data, sigma_cosmic=sigma_outliers, file=file, X0=None, Y0=None, R_crop=None, verbose=False)
        S_noiseless /= exposure_time # in e-/mn
        hdr = fits.Header() ; hdr['pxscale'] = pxscale
        if data: # writing the data for systematics estimation purposes
            fits.writeto("sim_data/Systematics/"+instru+"/S_data_star_center_s3d_"+band+".fits", S_noiseless, header=hdr, overwrite=True)
            fits.writeto("sim_data/Systematics/"+instru+"/wave_data_star_center_s3d_"+band+".fits", wave, overwrite=True)
        else:
            fits.writeto("sim_data/Systematics/"+instru+"/S_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits", S_noiseless, header=hdr, overwrite=True)
            fits.writeto("sim_data/Systematics/"+instru+"/wave_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits", wave, overwrite=True)

    f = interp1d(wave_band, trans, bounds_error=False, fill_value=np.nan) ; trans = f(wave) # interpolating the transmission on the wavelength array of the data
    NbChannel, NbLine, NbColumn = S_noiseless.shape # size of the cube
    y_center = NbLine//2 ; x_center = NbColumn//2 # spatial center position of the cube
    S_noiseless *= annular_mask(0, int(round(config_data["spec"]["FOV"]/2/pxscale))+1, size=(NbLine,NbColumn))
    star_flux_FC = star_spectrum_instru.degrade_resolution(wave, renorm=True).flux * trans # star spectrum considered in FastCurves (in e-/mn)
    total_flux = np.nansum(star_flux_FC) # total stellar flux (in e-/mn)
    sep = np.zeros((int(round(config_data["spec"]["FOV"]/(2*pxscale))) + 1)) # separation array for the data
    sigma_syst_prime_2 = np.zeros_like(sep) # systematic noise profile array estimated
    M_HF = np.zeros((len(sep), len(wave_band))) # high frequency systematic modulations as function of the separation [M(lambda, rho)]_HF
    star_flux_data = np.nansum(S_noiseless, (1, 2)) # gamma x S* of the data cube (in e-/mn)     
    star_flux_data[star_flux_data==0] = np.nan # en e-/mn

    if instru == "MIRIMRS" and not data and 1==1: # for MIRISim data, the injected star spectra is known
        input_flux = np.loadtxt('sim_data/Systematics/MIRIMRS/star_'+str(T_star_sim)+'_mag7_J.txt', skiprows=1) ; input_flux = Spectrum(input_flux[:, 0], input_flux[:, 1], None, None) ; input_flux = input_flux.degrade_resolution(wave, renorm=False) ; input_flux = input_flux.set_nbphotons_min(config_data, wave) ; input_flux.flux *= trans # en e-/mn
        star_flux_data = input_flux.flux * np.nanmean(star_flux_data) / np.nanmean(input_flux.flux) # S_* en e-/mn
    
    cube_wo_planet = np.zeros_like(S_noiseless) + np.nan
    for i in range(NbChannel): # renormalizing the cube with the star spectra considered
        cube_wo_planet[i] = S_noiseless[i] * star_flux_FC[i] / star_flux_data[i] # M x gamma x S_* # en e-/mn
    
    Sres_wo_planet, _ = stellar_high_filtering(cube=cube_wo_planet, R=R, Rc=Rc, filter_type=filter_type, outliers=data, sigma_outliers=sigma_outliers, verbose=False) # stellar subtracted data
    
    if PCA: # applying PCA as it would be applied on real data
        if Rc==100: # For FastYield calculations (with Rc = 100)
            T_star_t_syst_arr = np.array([3000, 6000, 9000]) ; T_star_t_syst = T_star_t_syst_arr[np.abs(star_spectrum_instru.T-T_star_t_syst_arr).argmin()] 
            T_planet_t_syst_arr = np.arange(500, 3000+100, 100) ; T_planet_t_syst = T_planet_t_syst_arr[np.abs(planet_spectrum_instru.T-T_planet_t_syst_arr).argmin()] 
            t_syst = fits.getdata("sim_data/Systematics/"+instru+"/t_syst/t_syst_"+instru+"_"+band+"_Tp"+str(T_planet_t_syst)+"K_Ts"+str(T_star_t_syst)+"K_Rc"+str(Rc))
            separation_t_syst = fits.getdata("sim_data/Systematics/"+instru+"/t_syst/separation_"+instru+"_"+band+"_Tp"+str(T_planet_t_syst)+"K_Ts"+str(T_star_t_syst)+"K_Rc"+str(Rc))
            mag_star_t_syst = fits.getdata("sim_data/Systematics/"+instru+"/t_syst/mag_star_"+instru+"_"+band+"_Tp"+str(T_planet_t_syst)+"K_Ts"+str(T_star_t_syst)+"K_Rc"+str(Rc))
            idx_mag_star_t_syst = np.abs(mag_star_t_syst - mag_star).argmin()
            idx_separation_t_syst = np.abs(separation_t_syst - separation_planet).argmin()
            if "sim" in target_name or t_syst[idx_mag_star_t_syst, idx_separation_t_syst] < 120: # If the systematics are not dominating for an exoposure time of about 2 hours (~ order of magnitude of the observations generally made), PCA is not necessary (EXCEPTION IS MADE FOR SIMUMATIONS)
                PCA_calc = True
            else:
                PCA_calc = False
        else:
            PCA_calc = True
        if PCA_calc: 
            if verbose and [bv for bv, band_verbose in enumerate(config_data["gratings"]) if band_verbose==band][0] == 0:
                print(f"PCA, with {Nc} principal components subtracted, is included in the FastCurves estimations as a technique for systematic noise removal")
            mask = np.copy(Sres_wo_planet) ; mask[~np.isnan(mask)] = 0
            if mag_planet is not None: # in case of calculation = "contrast", planet_spectrum_instru is not renormalized with mag_planet...
                planet_spectrum_instru, _ = spectrum_instru(band0, 200000, config_data, mag_planet, planet_spectrum)
            planet_flux_FC = trans * planet_spectrum_instru.degrade_resolution(wave, renorm=True).flux # gamma x Sp
            planet_HF, planet_LF = filtered_flux(planet_flux_FC/trans, R=R, Rc=Rc, filter_type=filter_type) # [Sp]_HF, [Sp]_LF
            star_HF, star_LF = filtered_flux(star_flux_FC/trans, R=R, Rc=Rc, filter_type=filter_type) # [S_*]_HF, [S_*]_LF
            # FAKE INJECTION of the planet in order to estimate components that would be estimated on real data and thus estimating the systematic noise and signal reduction 
            if mag_planet is not None:
                y0 = NbColumn//2 + int(round(min(separation_planet, config_data["spec"]["FOV"]/2-pxscale)/pxscale)) ; x0 = NbColumn//2 # planet's position according its separation supposing a position along the vertical spatial axis
                shift = int(round(min(separation_planet, config_data["spec"]["FOV"]/2-pxscale)/pxscale)) # estimating the shift in spaxel for the injection
                cube_shift = np.roll(np.copy(cube_wo_planet)*annular_mask(0, int(round(config_data["spec"]["FOV"]/4/pxscale)), size=(NbLine, NbColumn)), shift, 1) # planet = star PSF shifted
                for i in range(NbChannel):
                    cube_shift[i] *= planet_flux_FC[i]/star_flux_FC[i] # renormalize the star PSF to simulate the planet PSF
                cube = np.copy(cube_wo_planet) + np.nan_to_num(cube_shift) # add the planet PSF to the cube
                Sres, _ = stellar_high_filtering(cube=cube, R=R, Rc=Rc, filter_type=filter_type, outliers=data, sigma_outliers=sigma_outliers, verbose=False) # stellar subtracted data with the fake planet injected
                Sres_pca, pca = PCA_subtraction(np.copy(Sres), n_comp_sub=Nc, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=PCA_mask, PCA_plots=False, wave=wave, R=R) # apply PCA to it
                Sres_pca += mask # retrieve the NaN values
                Sres_wo_planet = np.copy(Sres) ; Sres_wo_planet_pca = np.copy(Sres_pca)
                if separation_planet > size_core*pxscale: # if the planet is further than a FWHM from the star (otherwise hiding the planet will also hide the region used to estimate the noise)
                    Sres_wo_planet[:, y0-size_core:y0+size_core+1, x0-size_core:x0+size_core+1] = np.nan
                    Sres_wo_planet_pca[:, y0-size_core:y0+size_core+1, x0-size_core:x0+size_core+1] = np.nan # hiding the planet as it would be done on real data
                sres = np.zeros_like(Sres)+np.nan ; sres_pca = np.zeros_like(Sres_pca)+np.nan ; sres_wo_planet = np.zeros_like(Sres_wo_planet)+np.nan ; sres_wo_planet_pca = np.zeros_like(Sres_wo_planet_pca)+np.nan 
                for i in range(NbLine):
                    for j in range(NbColumn): # pour chaque spaxel
                        if not (np.isnan(Sres[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)]).all()):
                            sres[:, i, j] = np.nansum(Sres[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
                            sres_pca[:, i, j] = np.nansum(Sres_pca[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
                            sres_wo_planet[:, i, j] = np.nansum(Sres_wo_planet[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
                            sres_wo_planet_pca[:, i, j] = np.nansum(Sres_wo_planet_pca[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
                sres[sres == 0] = np.nan ; Sres = sres ; sres_pca[sres_pca == 0] = np.nan ; Sres_pca = sres_pca ; sres_wo_planet[sres_wo_planet == 0] = np.nan ; Sres_wo_planet = sres_wo_planet ; sres_wo_planet_pca[sres_wo_planet_pca == 0] = np.nan ; Sres_wo_planet_pca = sres_wo_planet_pca
                CCF, _ = molecular_mapping_rv(instru=instru, S_res=Sres, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=None)
                CCF_wo_planet, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=None)
                CCF_pca, _ = molecular_mapping_rv(instru=instru, S_res=Sres_pca, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=pca)
                CCF_wo_planet_pca, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet_pca, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=pca) # the planet's parameters are not needed since the planet spectrum is injected
                _, _, CCF_signal, CCF_noise = SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, verbose=False, snr_calc=False) ; _, _, CCF_signal_pca, CCF_noise_pca = SNR_calculation(CCF_pca, CCF_wo_planet_pca, y0, x0, size_core, verbose=False, snr_calc=False)
                signal = CCF_signal - np.nanmean(CCF_noise) ; signal_pca = CCF_signal_pca - np.nanmean(CCF_noise_pca) # estimating the planet signal at its location (with and without pca)
                M_pca = abs(signal_pca / signal) # signal loss measured
            # NO FAKE INJECTION of the planet 
            else:
                y0 = None ; x0 = None
                Sres_wo_planet_pca, pca = PCA_subtraction(np.copy(Sres_wo_planet), n_comp_sub=Nc, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=False, PCA_plots=False, wave=wave, R=R)
                Sres_wo_planet_pca += mask
                sres_wo_planet_pca = np.zeros_like(Sres_wo_planet_pca)+np.nan 
                for i in range(NbLine):
                    for j in range(NbColumn): # pour chaque spaxel
                        if not (np.isnan(Sres_wo_planet_pca[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)]).all()):
                            sres_wo_planet_pca[:, i, j] = np.nansum(Sres_wo_planet_pca[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
                sres_wo_planet_pca[sres_wo_planet_pca == 0] = np.nan ; Sres_wo_planet_pca = sres_wo_planet_pca
                CCF_wo_planet_pca, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet_pca, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=pca)
                M_pca = 1
            if Rc is None: # another way to estimate the signal loss due to the PCA: substract the PCA components to the planetary spectrum
                d = trans*planet_HF
            else:
                d = trans*planet_HF - trans*star_HF*planet_LF/star_LF # spectrum at the planet's location: see Eq.(18) of Martos et al. 2024
            template = trans*planet_HF
            template /= np.sqrt(np.nansum(template**2))
            d_sub = np.copy(d) ; template_sub = np.copy(template)
            for nk in range(Nc): # subtracting the components 
                d_sub -= np.nan_to_num(np.nansum(d*pca.components_[nk])*pca.components_[nk])
                template_sub -= np.nan_to_num(np.nansum(template*pca.components_[nk])*pca.components_[nk])
            m_pca = abs(np.nansum(d_sub*template_sub) / np.nansum(d*template)) # analytical signal loss
            M_pca = min(M_pca, m_pca, 1) # taking the minimal value between the two methods (and knowing that the signal loss ratio must be lower than 1)
            CCF_wo_planet = CCF_wo_planet_pca
        # NO PCA
        else:
            if verbose and [bv for bv, band_verbose in enumerate(config_data["gratings"]) if band_verbose==band][0] == 0:
                print(f"PCA is not included in the FastCurves estimations, even if desired, because systematic noises are not expected to be dominant")
            M_pca = 1.
            sres_wo_planet = np.zeros_like(Sres_wo_planet) + np.nan 
            for i in range(NbLine):
                for j in range(NbColumn): # pour chaque spaxel
                    if not (np.isnan(Sres_wo_planet[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)]).all()):
                        sres_wo_planet[:, i, j] = np.nansum(Sres_wo_planet[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
            sres_wo_planet[sres_wo_planet == 0] = np.nan ; Sres_wo_planet = sres_wo_planet
            CCF_wo_planet, temp = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=pca)
    # NO PCA
    else:
        M_pca = 1.
        sres_wo_planet = np.zeros_like(Sres_wo_planet) + np.nan 
        for i in range(NbLine):
            for j in range(NbColumn): # pour chaque spaxel
                if not (np.isnan(Sres_wo_planet[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)]).all()):
                    sres_wo_planet[:, i, j] = np.nansum(Sres_wo_planet[:, max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)], axis=(1, 2))
        sres_wo_planet[sres_wo_planet == 0] = np.nan ; Sres_wo_planet = sres_wo_planet
        CCF_wo_planet, temp = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet, star_flux=star_flux_FC, T_planet=planet_spectrum_instru.T, lg_planet=planet_spectrum_instru.lg, rv=0, vsini_planet=0, model=planet_spectrum_instru.model, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, verbose=False, template=planet_spectrum_instru.copy(), pca=pca)
    
    # sigma_syst calculations
    for r in range(1, len(sep)+1):
        sep[r-1] = r*pxscale
        ccf = CCF_wo_planet*annular_mask(max(1, r-1), r, size=(NbLine, NbColumn)) # ring of the cube at separation r
        if not all(np.isnan(ccf.flatten())):
            sigma_syst_prime_2[r-1] = np.nanvar(ccf)/total_flux**2 # systematic noise at separation r (in e-/total stellar flux) 
            if show_cos_theta_est: # to estimate the correlation that would be measured, the high frequency modulation is needed at each separation
                f = interp1d(wave, Sres_wo_planet[:, y_center+1-r, x_center+1]/star_flux_FC, bounds_error=False, fill_value=np.nan)
                M_HF[r-1, :] = f(wave_band)
        
    Mp = np.nanmean(cube_wo_planet[:, y_center-size_core//2:y_center+size_core//2+1, x_center-size_core//2:x_center+size_core//2+1], axis=(1, 2))/star_flux_FC 
    Mp /= np.nanmean(Mp) # estimating the planet modulation function on the FWHM of the star (it's actually a bad estimation, but it's not significant for the performance estimates, only for the estimation of the correlation that would be measured cos_theta_est)
    #Mp = fits.getdata("utils/Mp/Mp_"+band+"_"+str(planet_spectrum_instru.T_planet)+"K.fits")
    f = interp1d(wave, Mp, bounds_error=False, fill_value=np.nan)
    Mp = f(wave_band)
    #plt.figure(dpi=300) ; plt.plot(wave_band, Mp) ; plt.title(f"{band}") ; plt.show()
    return sigma_syst_prime_2, sep, M_HF, Mp, M_pca, wave, pca





