from src.molecular_mapping import *
from src.utils import _load_instru_trans, _load_tell_trans, _load_psf_profile, _load_noiseless_cube



# -------------------------------------------------------------------------
# Transmission
# -------------------------------------------------------------------------

@lru_cache(maxsize=50)
def _get_transmission(instru, band, tellurics, apodizer, strehl=None, coronagraph=None, fill_value=np.nan):
    """
    Cached version of get_transmission()
    """
    config_data = get_config_data(instru)
    lmin_band   = config_data['gratings'][band].lmin       # Lambda_min of the considered band [µm]
    lmax_band   = config_data['gratings'][band].lmax       # Lambda_max of the considered band [µm]
    R_band      = config_data['gratings'][band].R          # Spectral resolution of the band
    dl_band     = (lmin_band+lmax_band)/2 / (2*R_band)     # Delta lambda [µm]
    wave_band   = np.arange(lmin_band, lmax_band, dl_band) # Constant and linear wavelength array on the considered band
    
    trans = get_transmission(instru=instru, wave_band=wave_band, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, fill_value=fill_value)
    return trans

def get_transmission(instru, wave_band, band, tellurics, apodizer, strehl=None, coronagraph=None, fill_value=np.nan):
    """
    Build the total end-to-end transmission vector on a target wavelength grid.

    This multiplies the instrumental throughput, the apodizer transmission,
    (optionally) an aperture-correction factor read from the PSF header,
    and (optionally) the atmospheric transmission for ground-based cases.

    Parameters
    ----------
    instru : str
        Instrument name.
    wave_band : (M,) array_like
        Target wavelength grid for the band (µm).
    band : str
        Instrumental band identifier (e.g., "J", "H", "HK"...).
    tellurics : bool
        If True, multiplies by sky transmission (airmass=1.0).
    apodizer : str
        Apodizer key present in config_data["apodizers"].
    strehl : str or None, optional
        Strehl tag used to select the PSF header (for aperture correction).
    coronagraph : str or None, optional
        Coronagraph tag for PSF header selection.
    fill_value : float, optional
        Extrapolation fill value for interpolation (default: NaN).

    Returns
    -------
    transmission : (M,) ndarray
        Total system transmission sampled on 'wave_band'. Negative values are clipped to 0.
    """
    # 1) Instrumental transmission on the band
    wave, trans = _load_instru_trans(instru=instru, band=band)
    trans       = interp1d(wave, trans, bounds_error=False, fill_value=fill_value)(wave_band)
    
    # 2) Apodizer throughput
    trans *= get_config_data(instru)["apodizers"][apodizer].transmission

    # 3) Aperture correction (if present)
    #    We look for a PSF profile file and read its 'AC' header key when available.
    try:
        hdr, _, _, _ = _load_psf_profile(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)
        trans       *= hdr["AC"]
    except:
        pass
    
    # 4) Telluric transmission (only if necessary, i.e. for ground based observation)
    if tellurics: # if ground-based observation
        wave_tell, tell = _load_tell_trans(airmass=1.0)
        trans          *= Spectrum(wave_tell, tell).degrade_resolution(wave_band, renorm=False).flux # degraded tellurics transmission on the considered band

    # Numerical cleanup
    trans[trans <= 0] = np.nan
    return trans



# -------------------------------------------------------------------------
# PSF radial profile (+ extrapolation if needed)
# -------------------------------------------------------------------------

def get_PSF_profile(band, strehl, apodizer, coronagraph, instru, separation_planet=None, return_SNR_planet=False, new_extrapolation=False, sampling=10):
    """
    Load the azimuthally averaged PSF surface-brightness profile and return it on a
    convenient separation grid, along with the "fraction of core" (header FC) and pxscale.

    Notes
    -----
    - The returned profile has units of "fraction_per_sq(sep_unit)"; multiplying by
      pxscale**2 converts it to "fraction per px area" (done here for convenience).
    - If the requested separation range exceeds the tabulated one, an extrapolation is
      performed using a power-law times an exponential tail; the fitted params are cached
      in the PSF FITS header (keys 'extrapolation_alpha' and 'extrapolation_rc').

    Parameters
    ----------
    band, strehl, apodizer, coronagraph, instru, config_data, sep_unit : see calling code
    separation_planet : float or None
        Specific separation to include in the grid for SNR evaluation (same unit as sep_unit).
    return_SNR_planet : bool, optional
        If True and 'separation_planet' is provided, the returned separation vector will be
        [0, IWA, separation_planet] (sorted) rather than a dense sampling.
    new_extrapolation : bool, optional
        Force a re-fit of the tail extrapolation parameters even if present in the header.
    sampling : int, optional
        Number of samples per pxscale unit for the dense grid (default 10 → smoother curve).
    OWA : float or None
        Override the default outer working angle.

    Returns
    -------
    PSF_profile : (Ns,) ndarray
        Surface-brightness profile vs separation (converted to per-pixel-area).
    fraction_core : float or None
        Fraction of flux in the core (FC header) — None for coronagraphic case if not present.
    separation : (Ns,) ndarray
        Separation axis in sep_unit (mas or arcsec).
    pxscale : float
        Pixel scale in sep_unit/pixel (mas/px if sep_unit=="mas", otherwise arcsec/px).
    """
    config_data = get_config_data(instru)
    sep_unit    = config_data["sep_unit"]
    
    # Pixel scale in output unit
    try:
        pxscale = config_data["pxscale"][band]   # [arcsec/px]
    except Exception:
        pxscale = config_data["spec"]["pxscale"] # [arcsec/px]
    if sep_unit == "mas":
        pxscale *= 1e3  # mas/px
    elif sep_unit != "arcsec":
        raise ValueError("sep_unit must be 'mas' or 'arcsec'.")
    
    iwa, owa = get_wa(config_data=config_data)
    
    # Load PSF
    hdr, raw_separation, raw_profile, psf_file = _load_psf_profile(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)

    fraction_core  = hdr["FC"]                # Fraction of flux contained in the FWHM (or None for coronagraph)
    valid          = np.isfinite(raw_profile)
    raw_separation = raw_separation[valid]    # [arcsec] or [mas]
    raw_profile    = raw_profile[valid]       # [FOV flux fraction/arcsec**2] or [FOV flux fraction/mas**2]
    profile_interp = interp1d(raw_separation, raw_profile, bounds_error=False, fill_value="extrapolate")
    
    # Build separation grid
    if return_SNR_planet and separation_planet is not None:
        separation = np.sort(np.array([0, iwa, separation_planet]))
    else:
        try:
            pxscale_sampling = min(config_data["pxscale"].values()) # arcsec
        except:
            pxscale_sampling = config_data["spec"]["pxscale"] # arcsec
        if sep_unit == "mas":
            pxscale_sampling *= 1e3  # mas/px
        step       = pxscale_sampling / max(1, int(sampling))
        separation = np.arange(0.0, owa + step, step)
        # Ensure IWA is represented exactly
        if iwa not in separation:
            idx        = np.searchsorted(separation, iwa)
            separation = np.insert(separation, idx, iwa)
        # Ensure planet separation represented exactly
        if separation_planet is not None and separation_planet > separation[-1]: # extension of the separation axis to the planet's separation
            extension  = np.linspace(separation[-1] + pxscale_sampling, separation_planet, len(separation))
            separation = np.concatenate([separation, extension, [separation_planet + pxscale_sampling]])
        elif separation_planet is not None and separation_planet not in separation:
            idx        = np.searchsorted(separation, separation_planet)
            separation = np.insert(separation, idx, separation_planet)
    
    # Ensure FPM IWA is represented exactly for HARMONI
    iwa_FPM = None
    if instru == "HARMONI":
        iwa_FPM = config_data["apodizers"][apodizer].sep
        if separation_planet is not None and 1==1: # TODO: adapted IWA for each planet's separation case (it's kinda of cheating)
            iwa_FPM = max(separation_planet - pxscale, 0)
        if iwa_FPM not in separation:
            idx        = np.searchsorted(separation, iwa_FPM)
            separation = np.insert(separation, idx, iwa_FPM)
    
    # Extrapolation of the separation axis to the planet's separation (if needed)
    if separation_planet is not None and separation_planet > raw_separation[-1]:
        
        try: 
            if new_extrapolation:
                raise KeyError(f"new_extrapolation={new_extrapolation}")
            alpha = hdr['extrapolation_alpha'] 
            rc    = hdr['extrapolation_rc']
        except Exception as e:
            # Fit hybrid tail y ~ y0 * (x/x0)^(-|alpha|) * exp(-x/rc)
            # Use last quartile as tail region
            print(f"Extrapolating PSF profile: {e}")
            new_extrapolation = True
            tail_n            = max(8, len(raw_separation) // 4)
            x_tail            = raw_separation[-tail_n:]
            y_tail            = raw_profile[-tail_n:]  
            log_x_tail        = np.log(x_tail)
            log_y_tail        = np.log(y_tail)        
            valid             = np.isfinite(log_x_tail) & np.isfinite(log_y_tail)
            alpha, log_y0     = np.polyfit(log_x_tail[valid], log_y_tail[valid], 1) # Adjustment of exponential power law
            y0                = np.exp(log_y0)        
            rc                = np.median(x_tail) # Exponential decreasing parameter (adaptive)        
            if alpha > 0: # sanity check on alpha coeff sign (needs to be negative for decreasing flux)
                alpha = -alpha
            # Persist in header for next runs
            hdul = fits.open(psf_file)
            hdul[0].header['extrapolation_alpha'] = alpha
            hdul[0].header['extrapolation_rc']    = rc
            hdul.writeto(psf_file, overwrite=True)
            hdul.close()
            _load_psf_profile.cache_clear()
        
        PSF_profile            = profile_interp(separation)
        mask_tail              = separation >= raw_separation[-1]
        PSF_profile[mask_tail] = improved_power_law_extrapolation(x=separation[mask_tail], x0=raw_separation[-1], y0=raw_profile[-1], alpha=alpha, rc=rc)
        
        if new_extrapolation: # Sanity check Plot
            plt.figure(dpi=300, figsize=(10, 6))
            plt.plot(separation, PSF_profile, label="Extrapolation", color='orange')
            plt.plot(separation, profile_interp(separation), label="Profil PSF initial")
            plt.xlabel(f"Separation [{config_data['sep_unit']}]", fontsize=14)
            plt.ylabel("PSF profile", fontsize=14)
            plt.title(f"{instru} - {band} - {apodizer} - {strehl} - {coronagraph}")
            plt.yscale('log')
            plt.xscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.minorticks_on()
            plt.legend()
            plt.show()

    # No extrapolation of the PSF profile
    else:
        PSF_profile = profile_interp(separation)

    if instru == "MIRIMRS":
        PSF_profile *= config_data["pxscale0"][band]**2 # pxscale (non-dithered) (because all the values are considered in the detector space in the first place, then multiplied by R_corr, to take into account the transformation into the 3D cube sapce)
    else:
        PSF_profile *= pxscale**2
                        
    return PSF_profile, fraction_core, separation, pxscale, iwa_FPM



# -------------------------------------------------------------------------
# DIT and effective read-out noise
# -------------------------------------------------------------------------

@lru_cache(maxsize=50)
def estimate_RON_UTR(N, RON0, RON_lim=0.):
    """
    Estimate the effective readout noise (RON) for an Up-the-Ramp (UTR) sampling scheme (see: https://ntrs.nasa.gov/api/citations/20070034922/downloads/20070034922.pdf)

    Parameters
    ----------
    N : int or float
        Number of non-destructive reads (must be ≥ 2).
    RON0 : float, optional
        Read noise per read in electrons (default: global variable RON0).
    RON_lim : float, optional
        Asymptotic read noise floor at large N, representing other limiting sources 
        (e.g., 1/f noise or temporal systematics) in electrons (default: global variable RON_lim).

    Returns
    -------
    RON_eff : float
        Effective readout noise in electrons for the given number of reads.

    Notes
    -----
    The model assumes ideal equispaced Up-the-Ramp (UTR) sampling and combines two components:
    - A term decreasing as 1/N² from averaging uncorrelated read noise.
    - A constant floor from residual correlated noise or systematics.

    The formula used is:
        RON_eff² = RON0² × [12 × (N−1) / (N × (N+1))] + RON_lim²

    Raises
    ------
    ValueError
        If N < 2, since UTR requires at least 2 reads.
    """
    if (type(N) == int or type(N) == float) and N < 2:
        raise ValueError("N must be >= 2 for Up-the-Ramp.")

    RON_eff_2 = (RON0**2) * 12 * (N - 1) / (N * (N + 1)) + RON_lim**2

    return np.sqrt(RON_eff_2)



def get_DIT_RON(instru, instru_type, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, RON, RON_lim, saturation_e, input_DIT, iwa_FPM):
    """
    Compute the detector integration time (DIT) to avoid saturation and the effective read-noise.

    Parameters
    ----------
    instru, instru_type, apodizer
        Instrument description and apodizer name (used to ignore the PSF core inside the IWA for IFU_fiber).
    PSF_profile
        Radial PSF profile (in fraction per pixel^2, already scaled).
    separation
        Separation vector matching 'PSF_profile'.
    star_spectrum_band
        Star spectrum (photons/min) on the considered band and grid.
    exposure_time
        Total exposure time (minutes).
    min_DIT, max_DIT
        Min/max allowed DIT (minutes).
    trans
        Total transmission on the considered band (array).
    RON
        Read-Out Noise (e-/px).
    saturation_e
        Full-well capacity (e-).
    input_DIT
        If provided, forces the DIT (still clamped by exposure time).

    Returns
    -------
    DIT
        Integration time per exposure (minutes).
    RON_eff
        Effective read-out noise per DIT (e-/px).
    """
    # Per-DIT max electron rate estimate (worst-case pixel in the PSF core)
    if instru_type == "imager":
        # Collapse spectrum to band total [e-/mn], then scale by PSF profile peak
        max_flux_e = np.nanmax(PSF_profile) * np.nansum(star_spectrum_band.flux * trans) # [total e-/mn]
        # Detector Integration Time when the saturation is reached on the brightest pixel
        DIT_saturation = saturation_e / max_flux_e # [mn]
    else:        
        max_flux_e = np.nanmax(PSF_profile) * np.nanmax(star_spectrum_band.flux * trans) # [e-/bin/mn]
        # Detector Integration Time when the saturation is reached on the brightest pixel
        DIT_saturation = saturation_e / max_flux_e # [mn]
        # For HARMONI, we can use a Focal Plane Mask to avoid saturation
        if instru == "HARMONI" and DIT_saturation < min_DIT:
            max_flux_e = np.nanmax(PSF_profile[separation>=iwa_FPM]) * np.nanmax(star_spectrum_band.flux * trans) # [e-/bin/mn]
            # Detector Integration Time when the saturation is reached on the brightest pixel
            DIT_saturation = saturation_e / max_flux_e # [mn]
        else:
            iwa_FPM = None
    
    # Apply limits / overrides
    if instru == "HiRISE" or instru == "VIPAPYRUS": # Overriding the DIT
        DIT = max_DIT
    elif input_DIT is not None: # Overriding the DIT
        DIT = input_DIT
    else:
        DIT = np.clip(DIT_saturation, min_DIT, max_DIT)
    DIT = min(DIT, exposure_time) # [mn]: The DIT cannot be longer than the total exposure time
    
    # Up-the-ramp effective read-noise (see https://arxiv.org/pdf/0706.2344)
    if DIT >= 2*min_DIT: # At least 2 reads inside the ramp
        N_i = DIT / min_DIT # Number of intermittent readings
        if instru in {"MIRIMRS", "NIRSpec", "NIRCam"}:
            RON_eff = RON / np.sqrt(N_i)
        else:
            RON_eff = estimate_RON_UTR(N=N_i, RON0=RON, RON_lim=RON_lim) # Effective read out noise [e-/px/DIT]
            RON_eff = min(RON_eff, RON)
    else:
        RON_eff = RON
    
    # Instrument- and laboratory-based floors
    if instru in {"ERIS"} and RON_eff < 7.0:
        RON_eff = 7.0
    if RON_eff < 0.5:
        RON_eff = 0.5
    
    # Total number of integrations
    NDIT = exposure_time / DIT
            
    return NDIT, DIT, DIT_saturation, RON_eff, iwa_FPM


# -------------------------------------------------------------------------
# δ term for differential imaging with non-imager
# -------------------------------------------------------------------------
    
def get_delta(planet_spectrum_band, template, trans):
    """
    Compute the useful photo-electron rate δ (per minute) used in differential imaging.
    δ ≈ sum(  trans*Sp * template ) × fraction_core

    Parameters
    ----------
    planet_spectrum_band
        Planet spectrum on the band grid (photons/min).
    template
        (Unitless) normalized template on the same spectral grid.
    trans
        Total system transmission on the same grid as the spectrum (scalar or array).

    Returns
    -------
    delta
        δ (e-/mn).
    """
    delta = np.nansum(trans*planet_spectrum_band.flux * template) # delta x cos theta lim (if systematics)
    return delta # [e-/mn]



# -------------------------------------------------------------------------
# α and β terms for molecular mapping
# -------------------------------------------------------------------------
    
def get_alpha(planet_spectrum_band, template, Rc, R, trans, filter_type):
    """
    Compute the useful photo-electron rate α (per minute) used in molecular mapping.
    α ≈ sum(  [Sp]_HF * trans * template )
    where _HF denotes the high-pass filtered spectrum.

    Parameters
    ----------
    planet_spectrum_band
        Planet spectrum on the band grid (photons/min).
    template
        (Unitless) normalized template on the same spectral grid.
    Rc
        Cut-off resolving power for the filter; if None, no filtering (HP=Sp, LP=0).
    R
        Resolving power of the input spectrum.
    trans
        Total system transmission on the same grid as the spectrum (scalar or array).
    filter_type
        Filter kind passed to 'filtered_flux' ("gaussian", "step", ...).

    Returns
    -------
    alpha
        α (e-/mn)
    """
    R_nyquist = estimate_resolution(planet_spectrum_band.wavelength)
    Sp        = planet_spectrum_band.flux                                           # Planetary flux (in ph/mn)
    Sp_HF, _  = filtered_flux(flux=Sp, R=R_nyquist, Rc=Rc, filter_type=filter_type) # High_pass filtered planetary flux
    Sp_HF    *= trans                                                               # gamma x [Sp]_HF
    alpha     = np.nansum(Sp_HF*template)                                           # alpha x cos theta lim (if systematics)
    return alpha # [e-/mn]



def get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R, trans, filter_type):
    """
    Compute the self-subtraction term β (per minute).
    β ≈ sum( trans * Star_HF * Planet_LF / Star_LF * template )

    Parameters
    ----------
    star_spectrum_band, planet_spectrum_band
        Star/planet spectra (photons/min) on the band grid.
    template
        (Unitless) normalized template on the same spectral grid.
    Rc
        Cut-off resolving power for HP/LP filter. If None, β = 0.
    R
        Resolving power of the input spectra.
    trans
        Total system transmission on the same grid as the spectrum (scalar or array).
    separation
        Separation array (only used to return a vector of same length).
    filter_type
        Filter kind passed to 'filtered_flux' ("gaussian", "step", ...).

    Returns
    -------
    beta
        β (e-/mn).
    """
    R_nyquist = estimate_resolution(planet_spectrum_band.wavelength)
    if Rc is None or Rc == 0:
        beta = 0
    else:
        star_HF, star_LF     = filtered_flux(star_spectrum_band.flux, R=R_nyquist, Rc=Rc, filter_type=filter_type)   # Star filtered spectra
        planet_HF, planet_LF = filtered_flux(planet_spectrum_band.flux, R=R_nyquist, Rc=Rc, filter_type=filter_type) # Planet filtered spectra
        beta                 = np.nansum(trans*star_HF*planet_LF/star_LF * template)                                 # Self-subtraction term
    return beta # [e-/mn]



# -------------------------------------------------------------------------
# Working angles
# -------------------------------------------------------------------------

def get_wa(config_data, sep_unit=None):
    """
    Return IWA and OWA in the requested separation unit.

    Parameters
    ----------
    config_data
        Instrument configuration dictionary (must include pxscale/spec/FOV/apodizers).

    Returns
    -------
    iwa, owa
        Inner/outer working angles in 'sep_unit'.
    """
    # WORKING ANGLE
    # lambda_c = (config_data["lambda_range"]["lambda_min"]+config_data["lambda_range"]["lambda_max"])/2 * 1e-6 # m
    # diameter = config_data['telescope']['diameter']                                                           # m
    try:
        pxscale = min(config_data["pxscale"].values()) # arcsec
    except:
        pxscale = config_data["spec"]["pxscale"]       # arcsec
    iwa = pxscale                      # arcsec
    owa = config_data["spec"]["FOV"]/2 # arcsec
    
    if sep_unit is None:
        sep_unit = config_data["sep_unit"]
    
    # Returns
    if sep_unit == "arcsec":
        return iwa, owa
    elif sep_unit == "mas":
        return max(iwa * 1e3, 5), owa * 1e3
        #return iwa * 1e3, owa * 1e3
    else:
        raise ValueError("sep_unit must be 'arcsec' or 'mas'.")



#######################################################################################################################
##################################### SYSTEMATIC NOISE PROFILE CALCULATION: ###########################################
#######################################################################################################################

def get_systematic_profile(config_data, band, tellurics, apodizer, strehl, coronagraph, Rc, R, star_spectrum_instru, planet_spectrum_instru, planet_spectrum, wave_band, size_core, filter_type, show_cos_theta_est=False, PCA=False, PCA_mask=False, N_PCA=20, mag_planet=None, band0_planet=None, separation_planet=None, mag_star=None, target_name=None, exposure_time=None, use_data=False, sigma_outliers=3):
    """
    Estimate the systematic-noise radial profile (projected into the CCF), along with
    a few ancillary vectors used in performance predictions.

    The flow:
      1) Load a noiseless (or calibration) 3-D cube S_noiseless (Nchan, Ny, Nx).
      2) Renormalize each channel to the *current* stellar flux model (photons/min).
      3) Apply the standard stellar high-pass filtering to obtain residuals S_res_wo_planet.
      4) Optionally simulate PCA on the filtered cube (with or without fake-planet injection)
         to estimate the signal loss factor M_pca and a PCA-cleaned CCF reference.
      5) Compute the CCF at each annulus and derive σ_syst'(ρ) ∝ Var(CCF)/F_star^2.
      6) Build Mp (planet modulation proxy) and, optionally, the per-ρ HF residual matrix M_HF.

    Parameters
    ----------
    config_data : dict
        Instrument configuration dictionary (includes pxscale, FOV, etc.).
    band : str
        Band identifier (e.g. "1A", "J", "H", ...).
    tellurics : bool
        If True, include atmospheric transmission in 'trans' building.
    apodizer, strehl, coronagraph : str or None
        Optical configuration tags (used for IWA/OWA, PSF choices, etc.).
    Rc : float or None
        Cut-off resolving power for the high-pass filter; None disables filtering.
    R : float
        Resolving power for the *band* being processed (used by filters).
    star_spectrum_instru : Spectrum
        Star spectrum on the instrument-wide grid (photons/min frame).
    planet_spectrum_instru : Spectrum
        Planet spectrum on the instrument-wide grid.
    planet_spectrum : Spectrum
        The raw planet spectrum (used if we reconstruct from magnitude when PCA is on).
    wave_band : (M,) ndarray
        Wavelength array for the *band* (for plotting and HF residual export).
    size_core : int
        Box size (in pixels) used to sum flux within the PSF core (~ FWHM box).
    filter_type : str
        High-pass filter type ("gaussian", "smoothstep", "savitzky_golay", ...).
    show_cos_theta_est : bool, optional
        If True, export a per-ρ HF residual matrix M_HF to estimate measured correlation.
    PCA : bool, optional
        If True, attempt to reduce systematics via PCA on the high-pass filtered cube.
    PCA_mask : bool, optional
        Mask the planet location during PCA (only when a fake injection is done).
    N_PCA : int, optional
        Number of PCA components to subtract.
    mag_planet, band0_planet : optional
        If set (and PCA is True), re-derive a planet spectrum normalized to 'mag_planet' in band 'band0_planet'
        for the fake injection step.
    separation_planet : float or None
        Planet separation (in arcsec or mas depending on config), required if you want a fake injection.
    mag_star, target_name, exposure_time : optional
        Metadata for PCA heuristics; exposure_time helps decide if PCA is beneficial.
    use_data : bool, optional
        If True, use real calibration data (e.g. MAST) instead of simulated noiseless cubes.
    sigma_outliers : float, optional
        Sigma threshold for outlier rejection in stellar filtering steps if 'use_data' is True.

    Returns
    -------
    sigma_syst_prime_2 : (Nr,) ndarray
        Estimated systematic noise power profile (variance of CCF per annulus / F_*^2).
    sep : (Nr,) ndarray
        Separation centers (in the instrument's separation unit).
    M_HF : (Nr, M) ndarray
        Per-annulus high-frequency residuals vs wavelength on 'wave_band' (NaNs if not requested).
    Mp : (M,) ndarray
        Proxy of the planet modulation function vs wavelength on 'wave_band'.
    M_pca : float
        Estimated multiplicative signal-loss factor due to PCA (≤ 1). Equals 1 if PCA not applied.
    wave : (Nchan,) ndarray
        Wavelength grid used by the cube and internal operations.
    pca : object or None
        The fitted PCA object (None if PCA not used or not computed).
    """
    PCA_verbose = None
    instru      = config_data["name"]
    FOV         = float(config_data["spec"]["FOV"]) # arcsec
    pca         = None
    PCA_calc    = False  # we’ll decide below if PCA = True
    
    # -----------------------
    # 1) Load modulation cube (noiseless or calibration)
    # -----------------------
    if instru == "MIRIMRS":
        #use_data       = True
        correction     = "all_corrected" # correction = "with_fringes_straylight" # correction applied to the simulated MIRISim noiseless data
        T_star_sim_arr = np.array([4000, 6000, 8000]) # available values for the star temperature for MIRSim noiseless data
        T_star_sim     = T_star_sim_arr[np.abs(star_spectrum_instru.T - T_star_sim_arr).argmin()]
        if use_data: # CALIBRATION DATA => High S/N per spectral channel => estimation of modulations: M_data 
            file = f"data/MIRIMRS/MAST/HD 159222_ch{band[0]}-shortmediumlong_s3d.fits" 
        else: # Using MIRISim (end-to-end simulation) data: in order to estimation modulations
            file = f"data/MIRIMRS/MIRISim/star_center/star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits"
    
    elif instru == "NIRSpec":
        file       = f"data/NIRSpec/MAST/HD 163466_nirspec_{band}_s3d.fits"
        use_data   = True
        T_star_sim = None
        correction = None
        warnings.simplefilter("ignore", category=RuntimeWarning) # Some slices are filled with NaN
    
    else:
        # TODO : For other instruments, extend here as needed
        file       = None
        T_star_sim = None
        correction = None
        
    try: # if the files already exist
        S_noiseless, wave, pxscale = _load_noiseless_cube(instru=instru, band=band, use_data=use_data, T_star_sim=T_star_sim, correction=correction)
    except Exception as e: # in case they don't, create them (but the raw data are needed)
        print(f"\n [get_systematic_profile] Building modulation cube (first run): {e}")
        if instru in ["MIRIMRS", "NIRSpec"]:
            S_noiseless, wave, pxscale, _, _, _, _ = extract_jwst_data(instru, "sim", band, crop_band=True, outliers=use_data, sigma_outliers=sigma_outliers, file=file, X0=None, Y0=None, R_crop=None, verbose=False)
        else:
            raise # TODO
        hdr            = fits.Header()
        hdr['pxscale'] = pxscale
        if use_data: # writing the data for systematics estimation purposes
            fits.writeto(f"sim_data/Systematics/{instru}/S_data_star_center_s3d_{band}.fits", S_noiseless, header=hdr, overwrite=True)
            fits.writeto(f"sim_data/Systematics/{instru}/wave_data_star_center_s3d_{band}.fits", wave, overwrite=True)
        else:
            fits.writeto(f"sim_data/Systematics/{instru}/S_noiseless_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits", S_noiseless, header=hdr, overwrite=True)
            fits.writeto(f"sim_data/Systematics/{instru}/wave_noiseless_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits", wave, overwrite=True)
    
    NbChannel, NbLine, NbColumn = S_noiseless.shape          # Shape of the cube
    y_center, x_center          = NbLine // 2, NbColumn // 2 # Spatial center position of the cube
    R_nyquist                   = estimate_resolution(wave)
    
    # Get transmission on the cube wavelength grid
    trans = get_transmission(instru, wave, band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph)
    
    # Current stellar model on cube grid [e-/mn]
    star_flux_FC = trans * star_spectrum_instru.degrade_resolution(wave, renorm=True, R_output=R).flux
    
    # For MIRISim data we know the injected star spectrum → re-derive star_flux_data (photons/min)
    if not use_data and instru == "MIRIMRS":
        input_flux       = np.loadtxt(f"sim_data/Systematics/{instru}/star_{T_star_sim}_mag7_J.txt", skiprows=1)
        input_flux       = Spectrum(input_flux[:, 0], input_flux[:, 1])
        input_flux       = input_flux.degrade_resolution(wave, renorm=False, R_output=R)
        input_flux       = input_flux.set_nbphotons_min(config_data, wave)
        input_flux.flux *= trans # propto e-
        # Rescale to match the integrated level measured in S_noiseless
        star_flux_data   = input_flux.flux * np.nansum(S_noiseless) / np.nansum(input_flux.flux)
    
    # Stellar flux from data cube (sum over spatial axes)
    else:
        star_flux_data                      = np.nansum(S_noiseless, axis=(1, 2)) # # gamma x S_* (propto e-)
        star_flux_data[star_flux_data == 0] = np.nan
    
    # Renormalize the cube channel-by-channel to match star_flux_FC (vectorized)
    cube_wo_planet = S_noiseless * (star_flux_FC / star_flux_data)[:, np.newaxis, np.newaxis]  # M x gamma x S_* [e-/mn]
    total_flux     = np.nansum(cube_wo_planet)
    
    # Estimating the star flux from the data cube
    star_flux_data                      = np.nansum(cube_wo_planet, (1, 2))
    star_flux_data[star_flux_data == 0] = np.nan
        
    # -----------------------
    # 2) Stellar filtering (high-pass etc.)
    # -----------------------
    S_res_wo_planet, _ = stellar_high_filtering(wave=wave, cube=cube_wo_planet, Rc=Rc, filter_type=filter_type, outliers=use_data, sigma_outliers=sigma_outliers, renorm_cube_res=False, only_high_pass=False, star_flux=star_flux_data, debug=False)

    # -----------------------
    # 3) Decide whether PCA is beneficial
    # -----------------------
    if PCA:
        if separation_planet is not None and Rc == 100: # For all FastYield calculations (with Rc = 100)
            T_star_t_syst_arr   = np.array([3000, 6000, 9000])
            T_star_t_syst       = T_star_t_syst_arr[np.abs(star_spectrum_instru.T-T_star_t_syst_arr).argmin()] 
            T_planet_t_syst_arr = np.arange(500, 3000+100, 100)
            T_planet_t_syst     = T_planet_t_syst_arr[np.abs(planet_spectrum_instru.T-T_planet_t_syst_arr).argmin()] 
            t_syst            = fits.getdata(f"sim_data/Systematics/{instru}/t_syst/t_syst_{instru}_{band}_Tp{T_planet_t_syst}K_Ts{T_star_t_syst}K_Rc{Rc}")
            separation_t_syst = fits.getdata(f"sim_data/Systematics/{instru}/t_syst/separation_{instru}_{band}_Tp{T_planet_t_syst}K_Ts{T_star_t_syst}K_Rc{Rc}")
            mag_star_t_syst   = fits.getdata(f"sim_data/Systematics/{instru}/t_syst/mag_star_{instru}_{band}_Tp{T_planet_t_syst}K_Ts{T_star_t_syst}K_Rc{Rc}")
            mag_star          = np.clip(mag_star, np.nanmin(mag_star_t_syst), np.nanmax(mag_star_t_syst))
            separation_planet = np.clip(separation_planet, np.nanmin(separation_t_syst), np.nanmax(separation_t_syst))
            interp_func = RegularGridInterpolator((mag_star_t_syst, separation_t_syst), t_syst, method='linear')
            point       = np.array([[mag_star, separation_planet]])
            t_syst      = interp_func(point)[0]
            # If the systematics are not dominating for the given exoposure time, PCA is not necessary (EXCEPTION IS MADE FOR SIMUMATIONS)
            if 120 > t_syst or "sim" in target_name: # 120 mn (typical exposure_time) > t_syst => systematics are dominating
                PCA_calc = True
            if exposure_time < 120 and exposure_time < t_syst and 120 > t_syst: # 2 hours (~ order of magnitude of the observations generally made) is the exposure time considered in FastYield calculations
                PCA_verbose = " PCA is not considered with t_exp = {exposure_time}mn but was considered in FastYield calculations with t_exp = 120mn."
        # Default to True if Rc!=100 (table is missing) or if separation_planet is unknown
        else:
            PCA_calc = True
    
    # -----------------------
    # 4) PCA branch (with optional fake-planet injection)
    # -----------------------
    if PCA_calc: 
        PCA_verbose = f" PCA, with {N_PCA} principal components subtracted, is included in the FastCurves estimations as a technique for systematic noise removal"
        
        # Optionally rebuild planet_spectrum_instru normalized to mag_planet (if known)
        if mag_planet is not None:
            planet_spectrum_instru, _ = get_spectrum_instru(band0_planet, R0_max, config_data, mag_planet, planet_spectrum)
        
        planet_flux_FC       = trans * planet_spectrum_instru.degrade_resolution(wave, renorm=True, R_output=R).flux         # gamma x Sp
        planet_HF, planet_LF = filtered_flux(planet_flux_FC/trans, R=R_nyquist, Rc=Rc, filter_type=filter_type)  # [Sp]_HF, [Sp]_LF
        star_HF, star_LF     = filtered_flux(star_flux_FC/trans, R=R_nyquist, Rc=Rc, filter_type=filter_type)    # [S_*]_HF, [S_*]_LF
        
        # FAKE INJECTION of the planet in order to estimate components that would be estimated on real data and thus estimating the systematic noise and signal reduction 
        if (separation_planet is not None) and (separation_planet < FOV / 2) and (mag_planet is not None):
            
            # Fake injection
            cube_planet = np.copy(cube_wo_planet)
            for i in range(NbChannel):
                cube_planet[i] *= planet_flux_FC[i] / star_flux_FC[i]
            dy                     = int(round( separation_planet/pxscale ))
            y0                     = int(round( (NbLine-1)/2 + dy ))
            x0                     = int(round( (NbColumn-1)/2 ))
            cube_planet            = np.roll(cube_planet, dy, 1)
            cube_planet[:, :dy, :] = np.nan  
            cube                   = cube_wo_planet + np.nan_to_num(cube_planet)
            
            # Stellar filtering + PCA
            S_res, _       = stellar_high_filtering(wave=wave, cube=cube, Rc=Rc, filter_type=filter_type, outliers=use_data, sigma_outliers=sigma_outliers, renorm_cube_res=False, only_high_pass=False, star_flux=star_flux_data, debug=False) # stellar subtracted data with the fake planet injected
            S_res_pca, pca = PCA_subtraction(S_res=np.copy(S_res), N_PCA=N_PCA, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=PCA_mask, PCA_plots=False, wave=wave, R=R) # apply PCA to it
            
            # Masking the planet (if beyond FWHM)
            S_res_wo_planet     = np.copy(S_res)
            S_res_wo_planet_pca = np.copy(S_res_pca)
            if separation_planet > size_core*pxscale: # if the planet is further than a FWHM from the star (otherwise hiding the planet will also hide the region used to estimate the noise)
                planet_mask                            = circular_mask(y0, x0, r=size_core, size=(NbLine, NbColumn))
                S_res_wo_planet[:, planet_mask==1]     = np.nan
                S_res_wo_planet_pca[:, planet_mask==1] = np.nan
            
            # BOX convolutions
            S_res               = box_convolution(data=S_res,               size_core=size_core)
            S_res_pca           = box_convolution(data=S_res_pca,           size_core=size_core)
            S_res_wo_planet     = box_convolution(data=S_res_wo_planet,     size_core=size_core)
            S_res_wo_planet_pca = box_convolution(data=S_res_wo_planet_pca, size_core=size_core)

            # CCF computations
            CCF, _               = molecular_mapping_rv(instru=instru, S_res=S_res,               star_flux=star_flux_data, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, template=planet_spectrum_instru.copy(), pca=None)
            CCF_wo_planet, _     = molecular_mapping_rv(instru=instru, S_res=S_res_wo_planet,     star_flux=star_flux_data, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, template=planet_spectrum_instru.copy(), pca=None)
            CCF_pca, _           = molecular_mapping_rv(instru=instru, S_res=S_res_pca,           star_flux=star_flux_data, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, template=planet_spectrum_instru.copy(), pca=pca)
            CCF_wo_planet_pca, _ = molecular_mapping_rv(instru=instru, S_res=S_res_wo_planet_pca, star_flux=star_flux_data, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, template=planet_spectrum_instru.copy(), pca=pca)
            
            # Signal loss du to PCA (empirical at the injected location)
            r_planet        = int(round(np.sqrt((y0-y_center)**2 + (x0-x_center)**2)))
            mask_sep_planet = annular_mask(max(1, r_planet-1), r_planet+1, value=np.nan, size=(NbLine, NbColumn))
            CCF_signal      = CCF[y0, x0] - np.nanmean(CCF_wo_planet*mask_sep_planet)
            CCF_signal_pca  = CCF_pca[y0, x0] - np.nanmean(CCF_wo_planet_pca*mask_sep_planet)
            M_pca           = min(abs(CCF_signal_pca / CCF_signal), 1) # signal loss measured
        
        # No injection: PCA on the filtered cube without planet
        else:
            
            # No planet
            y0 = None
            x0 = None
            
            # PCA
            S_res_wo_planet_pca, pca = PCA_subtraction(S_res=np.copy(S_res_wo_planet), N_PCA=N_PCA, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=False, PCA_plots=False, wave=wave, R=R)
            
            # BOX convolutions
            S_res_wo_planet_pca = box_convolution(data=S_res_wo_planet_pca, size_core=size_core)

            # CCF computations
            CCF_wo_planet_pca, _ = molecular_mapping_rv(instru=instru, S_res=S_res_wo_planet_pca, star_flux=star_flux_data, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, template=planet_spectrum_instru.copy(), pca=pca)
            
            # No signal loss due to PCA
            M_pca = 1
        
        # Analytical PCA signal-loss proxy
        # Another way to estimate the signal loss due to the PCA: substract the PCA components to the planetary spectrum
        if Rc is None or Rc == 0:
            d = trans*planet_HF
        else:
            d = trans*planet_HF - trans*star_HF*planet_LF/star_LF # Spectrum at the planet's location: see Eq.(18) of Martos et al. 2025
        template     = trans*planet_HF
        template    /= np.sqrt(np.nansum(template**2))
        d_sub        = np.copy(d)
        template_sub = np.copy(template)
        for nk in range(N_PCA): # Subtracting the components 
            d_sub        -= np.nan_to_num(np.nansum(d*pca.components_[nk])*pca.components_[nk])
            template_sub -= np.nan_to_num(np.nansum(template*pca.components_[nk])*pca.components_[nk])
        m_pca = abs(np.nansum(d_sub*template_sub) / np.nansum(d*template)) # Analytical signal loss
        M_pca = min(M_pca, m_pca, 1) # Taking the minimal value between the two methods (and knowing that the signal loss ratio must be lower than 1)
        
        # Assigning the CCF that will be used for estimation of the systematic noise level
        CCF_wo_planet = CCF_wo_planet_pca
    
    # -----------------------
    # 5) No-PCA branch
    # -----------------------
    else:
        
        # No signal loss
        if PCA:
            PCA_verbose = " PCA requested but skipped (systematics not expected to dominate for this case)."
        M_pca = 1.
        
        # BOX convolutions
        S_res_wo_planet = box_convolution(data=S_res_wo_planet, size_core=size_core)
                
        # CCF computations
        CCF_wo_planet, template = molecular_mapping_rv(instru=instru, S_res=S_res_wo_planet, star_flux=star_flux_data, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, template=planet_spectrum_instru.copy(), pca=pca)

    # -----------------------
    # 6) Build radial profile of systematic noise from the CCF
    # -----------------------
    max_r              = int(round((FOV / 2) / pxscale)) + 1
    sep                = np.zeros((max_r)) + np.nan
    sigma_syst_prime_2 = np.zeros((max_r)) + np.nan
    for r in range(len(sep)):
        r_int  = max(1, r - 1) if r > 1 else r
        r_ext  = r + 1 if r==0 else r
        sep[r] = (r_int + r_ext)/2 * pxscale
        amask  = annular_mask(r_int, r_ext, size=(NbLine, NbColumn)) == 1 # ring at separation r
        ccf    = CCF_wo_planet[amask]
        if np.isfinite(ccf).sum() > 1:
            sigma_syst_prime_2[r] = np.nanvar(ccf) # systematic noise at separation r
    sigma_syst_prime_2 /= total_flux**2 # systematic noise (in e-/total stellar flux) 

    # -----------------------
    # 7) Optional per-ρ M_HF matrix for cosθ_est: to estimate the correlation that would be measured, the high frequency modulation is needed at each separation
    # -----------------------
    M_HF = np.full((len(sep), len(wave_band)), np.nan, dtype=float)
    if show_cos_theta_est:
        for r in range(len(sep)):
            M_HF[r, :] = interp1d(wave, S_res_wo_planet[:, y_center+1-r, x_center+1]/star_flux_FC, bounds_error=False, fill_value=np.nan)(wave_band)
    
    # -----------------------
    # 8) Mp (planet modulation proxy on FWHM box): mean within FWHM box around the star core, normalized by star flux
    # -----------------------
    core  = cube_wo_planet[:, y_center - size_core//2 : y_center + size_core//2 + 1, x_center - size_core//2 : x_center + size_core//2 + 1]
    Mp_0  = np.nanmean(core, axis=(1, 2)) / star_flux_FC
    Mp_0 /= np.nanmean(Mp_0)
    Mp    = interp1d(wave, Mp_0, bounds_error=False, fill_value=np.nan)(wave_band)
    
    Ms_0  = np.nanmean(cube_wo_planet, axis=(1, 2)) / star_flux_FC
    Ms_0 /= np.nanmean(Ms_0)
    Ms    = interp1d(wave, Ms_0, bounds_error=False, fill_value=np.nan)(wave_band)
    

    return sigma_syst_prime_2, sep, M_HF, Ms, Mp, M_pca, wave, pca, PCA_verbose



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_dark_noise_budget(instru, noise_level=None):
    RON_min = 0.5 # e-/px/DIT (achieved laboratory limit)
    config_data  = get_config_data(instru)
    min_DIT      = config_data["spec"]["minDIT"]       # minimal integration time (in mn)
    max_DIT      = config_data["spec"]["maxDIT"]       # maximal integration time (in mn)
    RON          = config_data["spec"]["RON"]          # read out noise (in e-/px/DIT)
    dark_current = config_data["spec"]["dark_current"] # dark current (in e-/px/s)
    DIT       = np.logspace(np.log10(min_DIT), np.log10(200*max_DIT), 100)
    sigma_dc  = np.zeros_like(DIT)
    sigma_ron = np.zeros_like(DIT)
    for i in range(len(DIT)):
        dit = DIT[i]
        sigma_dc[i] = dark_current * dit * 60 # e-/px/DOT
        nb_min_DIT = 1 # "Up the ramp" reading mode: the pose is sequenced in several non-destructive readings to reduce reading noise (see https://en.wikipedia.org/wiki/Signal_averaging).
        if dit > nb_min_DIT*min_DIT: # choose 4 min_DIT because if intermittent readings are too short, the detector will heat up too quickly => + dark current
            N_i     = dit / (nb_min_DIT*min_DIT) # number of intermittent readings
            sigma_ron[i] = RON / np.sqrt(N_i) # effective read out noise (in e-/px/DIT)
        else:
            sigma_ron[i] = RON
        if instru == 'ERIS' and sigma_ron[i] < 7: # effective RON lower limit for ERIS
            sigma_ron[i] = 7
        if sigma_ron[i] < RON_min: # achieved lower limit in laboratory
            sigma_ron[i] = RON_min
    sigma_tot = np.sqrt(sigma_ron**2 + sigma_dc**2)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(DIT, sigma_ron, "g--", label=f"RON ($RON_0$ = {RON} e-/px/DIT)")
    plt.plot(DIT, sigma_dc, "m--", label=f"dark current ({dark_current*60:.3f} e-/px/mn)")
    plt.plot(DIT, sigma_tot, "k-", label="total")
    if noise_level is not None:
        plt.axhline(noise_level, color='r', ls='--', label=f"noise level = {noise_level} e-/px/DIT")    
        intersections = []
        for i in range(len(DIT) - 1):
            y0, y1 = sigma_tot[i], sigma_tot[i+1]
            if (y0 - noise_level) * (y1 - noise_level) < 0:
                x0, x1 = DIT[i], DIT[i+1]
                frac = (noise_level - y0) / (y1 - y0)  # interpolation linéaire
                x_cross = x0 + frac * (x1 - x0)
                intersections.append(x_cross)
        for x_cross in intersections:
            plt.axvline(x_cross, color='r', ls=':', alpha=0.7)
            plt.annotate(f"{x_cross:.2f} mn", xy=(x_cross, noise_level), xycoords='data', xytext=(0, 0), textcoords='offset points', ha='center', va='bottom', color='r', rotation=0, bbox=dict(boxstyle="round", fc="white", ec="r", alpha=1))
    plt.axhspan(0, RON_min, facecolor='gray', alpha=0.3, label=f"Laboratory limit (< {RON_min} e-/px/DIT)")
    plt.title(f"Dark noise budget for {instru}", fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.xlabel("DIT [mn]", fontsize=14)
    plt.ylabel("Noise [e-/px/DIT]", fontsize=14)
    plt.xlim(DIT[0], DIT[-1])
    plt.ylim(0)
    plt.grid(which='both', alpha=0.4)
    plt.minorticks_on()
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().tick_params(axis='both', labelsize=12)
    plt.show()