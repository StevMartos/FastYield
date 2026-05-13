# import FastYield modules
from src.config import config_data_list, lmin_bands, lmax_bands, rad2arcsec, sim_data_path, R0_max

# import astropy modules
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d
from scipy.special import expit, logit, j1
from scipy.optimize import brentq

# import other modules
from functools import lru_cache

# For fits warnings

import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", message="Header block contains null bytes*")



# -------------------------------------------------------------------------
# Load & caches
# -------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _load_instru_trans(instru, band):
    """Load *once* the native instrument-throughput curve for (instru, band)."""
    wave, trans = fits.getdata(f"{sim_data_path}/Transmission/{instru}/transmission_{band}.fits")
    return wave, trans

@lru_cache(maxsize=2)
def _load_tell_trans(airmass, return_R=False):
    """Load *once* the native sky transmission curve for a given airmass."""
    wave_tell, tell = fits.getdata(f"{sim_data_path}/Transmission/sky_transmission_airmass_{airmass:.1f}.fits")
    if return_R:
        from src.spectrum import get_resolution
        R_tell = get_resolution(wavelength=wave_tell, func=np.array)
        return wave_tell, tell, R_tell
    else:
        return wave_tell, tell

@lru_cache(maxsize=32)
def _load_psf_profile(instru, band, strehl, apodizer, coronagraph):
    """Load *once* the PSF profile arrays + header for (instru, band, strehl, apodizer, coronagraph)."""
    if coronagraph is None:
        psf_file = f"{sim_data_path}/PSF/PSF_{instru}/PSF_{band}_{strehl}_{apodizer}.fits"
    else:
        psf_file = f"{sim_data_path}/PSF/PSF_{instru}/PSF_{band}_{coronagraph}_{strehl}_{apodizer}.fits"
    hdul      = fits.open(psf_file)
    hdr       = hdul[0].header
    sep, prof = hdul[0].data
    hdul.close()
    return hdr, sep, prof, psf_file

@lru_cache(maxsize=32)
def _load_corona_profile(instru, band, strehl, apodizer, coronagraph):
    """Load *once* the coronagraphic profile arrays."""
    sep, fraction_core, radial_transmission = fits.getdata(f"{sim_data_path}/PSF/PSF_{instru}/fraction_core_radial_transmission_{band}_{coronagraph}_{strehl}_{apodizer}.fits")
    return sep, fraction_core, radial_transmission

@lru_cache(maxsize=32)
def _load_stellar_modulation_function(instru, band, on_sky_data, T_star_sim, correction):
    """Load *once* stellar modulation functions [no unit]."""
    if on_sky_data: 
        Ms      = fits.getdata(f"{sim_data_path}/Systematics/{instru}/Ms_onsky_star_center_s3d_{band}.fits")              # Stellar modulation function [no unit]
        pxscale = fits.getheader(f"{sim_data_path}/Systematics/{instru}/Ms_onsky_star_center_s3d_{band}.fits")['pxscale'] # [arcsec/px]
        wave    = fits.getdata(f"{sim_data_path}/Systematics/{instru}/wave_onsky_star_center_s3d_{band}.fits")            # Wavelength axis of the data
    else:
        Ms      = fits.getdata(f"{sim_data_path}/Systematics/{instru}/Ms_sim_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits")              # Stellar modulation function [no unit]
        pxscale = fits.getheader(f"{sim_data_path}/Systematics/{instru}/Ms_sim_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits")['pxscale'] # [arcsec/px]
        wave    = fits.getdata(f"{sim_data_path}/Systematics/{instru}/wave_sim_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits")            # Wavelength axis of the data
    return Ms, wave, pxscale

@lru_cache(maxsize=32)
def _load_corr_factor(instru, band):
    """Load *once* the corrective factor."""
    sep, r_corr = fits.getdata(f"{sim_data_path}/R_corr/R_corr_{instru}/R_corr_{band}.fits")
    return sep, r_corr

@lru_cache(maxsize=32)
def _load_bkg_flux(instru, band, background):
    """Load *once* the background flux in [e-/px/s]."""
    raw_wave, raw_bkg = fits.getdata(f"{sim_data_path}/Background/{instru}/{background}/background_{band}.fits")
    return raw_wave, raw_bkg

@lru_cache(maxsize=32)
def _load_phi_m(post_processing, Rc):
    """Load *once* the effective correlation drift of systematics/speckles."""
    sep, phi_m = fits.getdata(f"{sim_data_path}/Systematics/rho_m_{post_processing}_Rc{Rc}.fits")
    return sep, phi_m



# -------------------------------------------------------------------------
# Get config data
# -------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_config_data(instru):
    """
    Retrieve the specifications of a given instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument.

    Returns
    -------
    dict
        Configuration parameters of the instrument.

    Raises
    ------
    NameError
        If the instrument name is not defined in config_data_list.
    """
    for cfg in config_data_list:
        if cfg["name"] == instru:
            return cfg
    raise NameError(f"Undefined instrument name: {instru}")



@lru_cache(maxsize=64)
def get_R_instru(instru):
    """
    Compute a conservative upper bound on the instrument spectral resolving power.
            R_instru = 2 * max(R_grating)
    The factor 2 is an intentional margin to avoid undersampling in subsequent
     spectral resampling/interpolation steps.
    
    Parameters
    ----------
    instru : str
        Instrument's name. 
    
    Returns
    -------
    R_instru : float
        Conservative maximum resolving power.
    """
    config_data = get_config_data(instru=instru)
    R_instru    = 2*np.nanmax([config_data["gratings"][band].R for band in config_data["gratings"]])
    R_instru    = min(R_instru, R0_max) # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
    return R_instru



@lru_cache(maxsize=64)
def get_band_lims(band):
    return lmin_bands[band], lmax_bands[band]




# -------------------------------------------------------------------------
# Get working angles
# -------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_annular_pupil_fwhm_factor(eps):
    """
    Return the FWHM coefficient k(eps) such that

        FWHM = k(eps) * lambda / D

    for a diffraction-limited annular aperture with central obstruction
    eps = D_obs / D_tel.

    The PSF intensity is normalized to I(0) = 1.
    """

    eps = float(eps)

    if eps < 0 or eps >= 1:
        raise ValueError("eps must satisfy 0 <= eps < 1.")

    def psf_intensity(u):
        amp = 2.0 * (j1(u) - eps * j1(eps * u)) / ((1.0 - eps**2) * u)
        return amp**2

    def half_max_equation(u):
        return psf_intensity(u) - 0.5

    u_half = brentq(half_max_equation, 1e-10, 3.0)

    return 2.0 * u_half / np.pi



@lru_cache(maxsize=64)
def get_DL_FWHM(instru, sep_unit=None, band="instru"):
    config_data = get_config_data(instru=instru)
    if band.lower() == "instru":
        lmin = config_data["lambda_range"]["lambda_min"] # [µm]
        lmax = config_data["lambda_range"]["lambda_max"] # [µm]
    else:
        lmin = config_data["gratings"][band].lmin # [µm]
        lmax = config_data["gratings"][band].lmax # [µm]
    l0      = (lmin + lmax) / 2                    # [µm]
    D       = config_data["telescope"]["diameter"] # [m]
    eps     = config_data["telescope"]["eps"]      # [no unit] (diameter of the secondary / diameter of the primary))
    k_eps   = get_annular_pupil_fwhm_factor(eps)   # [no unit]
    FWHM_DL = k_eps * l0 * 1e-6 / D * rad2arcsec   # [arcsec]

    # Returns
    if sep_unit is None:
        sep_unit = config_data["sep_unit"]
    if sep_unit == "arcsec":
        return FWHM_DL # [arcsec]
    elif sep_unit == "mas":
        return FWHM_DL*1e3 # [mas]
    else:
        raise ValueError("sep_unit must be 'arcsec' or 'mas'.")


    
@lru_cache(maxsize=64)
def get_wa(instru, sep_unit=None, band="instru"):
    """
    Return IWA and OWA in the requested separation unit.

    Parameters
    ----------
    instru : str
        Instrument's name. 
    
    Returns
    -------
    iwa, owa
        Inner/outer working angles in 'sep_unit'.
    """
    config_data = get_config_data(instru=instru)
    iwa         = get_DL_FWHM(instru=instru, sep_unit="arcsec", band=band) # [arcsec]
    owa         = config_data["FOV"]/2                                     # [arcsec]
    # Returns
    if sep_unit is None:
        sep_unit = config_data["sep_unit"]
    if sep_unit == "arcsec":
        return iwa, owa
    elif sep_unit == "mas":
        return iwa*1e3, owa*1e3
    else:
        raise ValueError("sep_unit must be 'arcsec' or 'mas'.")



# -------------------------------------------------------------------------
# Get transmission
# -------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _get_transmission(instru, band, tellurics, apodizer, strehl=None, coronagraph=None, fill_value=np.nan):
    """
    Cached version of get_transmission()
    """
    from src.spectrum import get_wave_band
    wave_band   = get_wave_band(instru=instru, band=band)
    trans       = get_transmission(instru=instru, wave_band=wave_band, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, fill_value=fill_value)
    return trans



def get_transmission(instru, wave_band, band, tellurics, apodizer, strehl=None, coronagraph=None, fill_value=np.nan, gaussian_filtering=True):
    """
    Build the total end-to-end transmission vector on a target wavelength grid.

    This multiplies the instrumental throughput, the apodizer transmission,
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
        Total system transmission sampled on 'wave_band'. Negative values are clipped to nan.
    """
    # 1) Instrumental transmission on the band
    wave, trans = _load_instru_trans(instru=instru, band=band)
    trans       = np.interp(wave_band, wave, trans, left=np.nan, right=np.nan)

    
    # 2) Apodizer throughput
    trans *= get_config_data(instru)["apodizers"][apodizer].transmission

    # 3) Telluric transmission (only if necessary, i.e. for ground based observation)
    if tellurics: # if ground-based observation
        # import spectrum modules
        from src.spectrum import Spectrum
        wave_tell, tell = _load_tell_trans(airmass=1.0)
        trans          *= Spectrum(wave_tell, tell).degrade_resolution(wave_band, renorm=False, gaussian_filtering=gaussian_filtering).flux # degraded tellurics transmission on the considered band

    # Numerical cleanup
    trans[trans <= 0] = np.nan
    return trans



# -------------------------------------------------------------------------
# Get PSF radial profile (+ extrapolation if needed)
# -------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_pxscale(instru, band, sep_unit):
    config_data = get_config_data(instru=instru)
    # Pixel scale in output unit
    try:
        pxscale = config_data["pxscale"][band] # [arcsec/px]
    except Exception:
        pxscale = config_data["pxscale"]       # [arcsec/px]
    if sep_unit == "mas":
        pxscale = pxscale*1e3                  # [mas/px]
    elif sep_unit != "arcsec":
        raise ValueError("sep_unit must be 'mas' or 'arcsec'.")
    return pxscale



@lru_cache(maxsize=64)
def get_separation_axis(instru, sep_unit, separation_planet, return_FastYield, sampling=10):
    
    config_data = get_config_data(instru=instru)
    
    _, owa = get_wa(instru=instru, sep_unit=sep_unit) # [sep_unit]
    
    if return_FastYield and separation_planet is not None:
        separation = np.array([0, separation_planet])
    else:
        try:
            pxscale_sampling = min(config_data["pxscale"].values()) # [arcsec]
        except:
            pxscale_sampling = config_data["pxscale"]               # [arcsec]
        if sep_unit == "mas":
            pxscale_sampling = pxscale_sampling * 1e3               # [mas/px]
        step       = pxscale_sampling / sampling
        separation = np.arange(0.0, owa + step, step)
        # Ensure planet separation represented exactly
        if separation_planet is not None and separation_planet > separation[-1]: # Extension of the separation axis to the planet's separation
            extension  = np.linspace(separation[-1], separation_planet, len(separation))
            separation = np.concatenate([separation, extension])
        elif separation_planet is not None and separation_planet not in separation:
            separation = np.append(separation, separation_planet)
    
    # Ensure iwa FPM values are represented exactly
    try: 
        iwa_FPM_values = config_data["FPMs"] # list of FPM IWAs in [mas] or [arcsec]
        for iwa_FPM in iwa_FPM_values:
            if iwa_FPM not in separation:
                separation = np.append(separation, iwa_FPM)
    except:
        pass

    # Sorted unique elements of separation
    separation = np.unique(separation)
    
    return separation



@lru_cache(maxsize=64)
def get_log_profile_interp(instru, band, strehl, apodizer, coronagraph, sep_unit):
    
    config_data = get_config_data(instru=instru)
    pxscale     = get_pxscale(instru=instru, band=band, sep_unit=sep_unit) # [sep_unit/px]
    
    # Load PSF
    hdr, raw_separation, raw_profile, psf_file = _load_psf_profile(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)
    fraction_core  = hdr.get("FC", None)                                    # Fraction of flux contained in the FWHM (or None for coronagraph)
    valid          = np.isfinite(raw_separation) & np.isfinite(raw_profile) # Valid mask on the profile
    raw_separation = raw_separation[valid]                                  # [sep_unit] ([arcsec] or [mas])
    raw_profile    = raw_profile[valid]                                     # [FOV flux fraction/arcsec**2] or [FOV flux fraction/mas**2]
    if instru == "MIRIMRS":
        raw_profile = raw_profile * config_data["pxscale0"][band]**2 # pxscale (non-dithered) (because all the values are considered in the detector space in the first place, then multiplied by R_corr, to take into account the transformation into the 3D cube sapce)
    else:
        raw_profile = raw_profile * pxscale**2
    
    # Interpolation of the raw profile
    log_profile_interp = interp1d(raw_separation, np.log(raw_profile), bounds_error=False, fill_value="extrapolate")
    
    return log_profile_interp, fraction_core, raw_profile, raw_separation, psf_file, hdr



def get_PSF_profile(band, strehl, apodizer, coronagraph, instru, separation_planet=None, return_FastYield=False, new_extrapolation=False, sampling=10, return_hdr=False, separation=None):
    """
    Load, interpolate, and optionally extrapolate an azimuthally averaged PSF radial profile.
    
    This function reads the PSF surface-brightness profile associated with a given
    instrumental configuration, interpolates it onto a convenient separation grid,
    and returns the profile converted from surface-brightness units to flux fraction
    per pixel area. It also returns the fraction of flux contained in the PSF core
    (if available in the FITS header), the separation grid, the pixel scale, and
    the focal-plane mask inner working angle (IWA).
    
    Parameters
    ----------
    band : str
        Spectral band associated with the PSF profile.
    strehl : str
        Strehl ratio label identifying the AO correction level or PSF realization.
    apodizer : str
        Name of the apodizer / focal-plane-mask configuration.
    coronagraph : str or None
        Coronagraphic setup used to select the appropriate PSF profile.
    instru : str
        Instrument name.
    separation_planet : float or None, optional
        Planet separation to include explicitly in the returned separation grid,
        expressed in the instrument separation unit ('mas' or 'arcsec' depending on
        'config_data["sep_unit"]'). If larger than the nominal outer working angle,
        the grid is extended up to this value.
    return_FastYield : bool, optional
        If True and 'separation_planet' is provided, return a minimal separation grid
        containing the key evaluation points instead of a densely sampled grid.
        In practice, the returned grid contains at least 0, 'separation_planet',
        and the focal-plane-mask IWA (after sorting and duplicate removal).
    new_extrapolation : bool, optional
        If True, force a refit of the power-law slope used for extrapolation, even if
        a previously fitted value is already stored in the PSF FITS header.
    sampling : int, optional
        Number of samples per pixel-scale element used to build the dense separation
        grid. Higher values produce a finer sampling.
    
    Returns
    -------
    PSF_profile : ndarray of shape (Ns,)
        Interpolated PSF radial profile evaluated on the returned separation grid.
        The output is converted from surface-brightness units
        ('flux fraction / sep_unit^2') to flux fraction per pixel area.
    fraction_core : float or None
        Fraction of total flux contained in the PSF core, as stored in the FITS
        header under the 'FC' keyword. Returns None if unavailable.
    separation : ndarray of shape (Ns,)
        Separation grid in the instrument separation unit ('mas' or 'arcsec').
    pxscale : float
        Pixel scale in the same separation unit per pixel.
    
    Notes
    -----
    - The raw PSF profile is assumed to be azimuthally averaged and expressed as a
      surface-brightness radial profile.
    - If the requested separation grid extends beyond the last tabulated PSF point,
      the profile is extrapolated using a pure power law anchored to the last valid
      tabulated value:
      
          y(x) = y0 * (x / x0)^slope
    
      where 'slope' is fitted in log-log space over the full valid profile
      excluding 'r = 0'.
    - The fitted extrapolation slope is cached in the PSF FITS header under the
      keyword ''slope'' for reuse in subsequent calls.
    - For 'MIRIMRS', the final conversion to per-pixel-area units uses
      'config_data["pxscale0"][band]**2' instead of 'pxscale**2', because the PSF
      profile is handled in detector-space coordinates before cube-space correction.
    """
    config_data = get_config_data(instru)
    sep_unit    = config_data["sep_unit"] # [arcsec] or [mas]    
    pxscale     = get_pxscale(instru=instru, band=band, sep_unit=sep_unit) # [sep_unit/px]

    # Get log interpolation object of the raw profile, fraction core, raw data axis, psf filename and hdr
    log_profile_interp, fraction_core, raw_profile, raw_separation, psf_file, hdr = get_log_profile_interp(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, sep_unit=sep_unit)
    
    # Build separation grid (if not input)
    if separation is None:
        separation = get_separation_axis(instru=instru, sep_unit=sep_unit, separation_planet=separation_planet, return_FastYield=return_FastYield, sampling=sampling)
        
    # Extrapolation of the separation axis to the planet's separation (if needed)
    if separation[-1] > raw_separation[-1]:
        from src.utils import power_law_extrapolation
                
        try: 
            if new_extrapolation:
                raise KeyError(f"new_extrapolation={new_extrapolation}")
            slope = hdr['slope'] 
        except Exception as e:
            # Fit a global effective log-log slope on the full profile (excluding r=0)
            print(f"Extrapolating PSF profile: {e}")
            new_extrapolation = True
            x                 = raw_separation[raw_separation>0]
            y                 = raw_profile[raw_separation>0] 
            log_x             = np.log(x)
            log_y             = np.log(y) 
            valid             = np.isfinite(log_x) & np.isfinite(log_y)
            slope, _          = np.polyfit(log_x[valid], log_y[valid], 1) # Global log-log slope fit
            # Persist in header for next runs
            hdul                    = fits.open(psf_file)
            hdul[0].header['slope'] = slope
            hdul.writeto(psf_file, overwrite=True)
            hdul.close()
            get_log_profile_interp.cache_clear()
            _load_psf_profile.cache_clear()
        
        PSF_profile            = np.exp(log_profile_interp(separation))
        mask_tail              = separation >= raw_separation[-1]
        PSF_profile[mask_tail] = power_law_extrapolation(x=separation[mask_tail], x0=raw_separation[-1], y0=raw_profile[-1], slope=slope)
        
        if new_extrapolation: # Sanity check Plot
            plt.figure(dpi=300, figsize=(10, 6))
            plt.xlim(pxscale, separation[-1])
            plt.ylim(1e-10, 1)
            plt.plot(separation, PSF_profile, label="Extrapolation", color='orange')
            plt.plot(raw_separation, raw_profile, label="Profil PSF initial")
            plt.xlabel(f"Separation [{config_data['sep_unit']}]", fontsize=14)
            plt.ylabel("PSF profile", fontsize=14)
            plt.title(f"{instru} - {band} - {apodizer} - {strehl} - {coronagraph} \n slope = {slope}")
            plt.yscale('log')
            plt.xscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.minorticks_on()
            plt.legend()
            plt.show()

    # No extrapolation of the PSF profile
    else:
        PSF_profile = np.exp(log_profile_interp(separation))
    
    if return_hdr:
        return PSF_profile, fraction_core, separation, pxscale, hdr
    else:
        return PSF_profile, fraction_core, separation, pxscale



@lru_cache(maxsize=64)
def get_logit_coronagraphic_profile_interp(instru, band, strehl, apodizer, coronagraph):
    raw_separation, raw_fraction_core, raw_radial_transmission = _load_corona_profile(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)
    logit_fraction_core_interp                                 = interp1d(raw_separation, logit(raw_fraction_core),       bounds_error=False, fill_value=(logit(raw_fraction_core[0]),       logit(raw_fraction_core[-1])))
    logit_radial_transmission_interp                           = interp1d(raw_separation, logit(raw_radial_transmission), bounds_error=False, fill_value=(logit(raw_radial_transmission[0]), logit(raw_radial_transmission[-1])))
    star_transmission                                          = expit(logit_radial_transmission_interp(0))          # Star coronagraphic transmission factor when the star is perfectly aligned with it (i.e. at 0 separation)
    return logit_fraction_core_interp, logit_radial_transmission_interp, star_transmission, raw_fraction_core, raw_radial_transmission, raw_separation
    

    
@lru_cache(maxsize=64)
def get_R_corr_interp(instru, band):
    raw_separation, raw_R_corr = _load_corr_factor(instru=instru, band=band)
    valid                      = np.isfinite(raw_R_corr)
    R_corr_interp              = interp1d(raw_separation[valid], raw_R_corr[valid], bounds_error=False, fill_value="extrapolate")
    return R_corr_interp, raw_R_corr, raw_separation



@lru_cache(maxsize=64)
def get_bkg_flux_band(instru, band, background_level):
    from src.spectrum import get_wave_band
    wave_band         = get_wave_band(instru=instru, band=band)        # [µm]
    wave_raw, bkg_raw = _load_bkg_flux(instru, band, background_level) # [µm] and [e-/bin/px/s]  
    bkg_raw_density   = bkg_raw / np.gradient(wave_raw)                # [e-/µm/px/s]
    bkg_flux_band     = interp1d(wave_raw, bkg_raw_density, bounds_error=False, fill_value=(bkg_raw_density[0], bkg_raw_density[-1]))(wave_band) # [e-/µm/px/s]
    bkg_flux_band    *= np.gradient(wave_band) * 60                    # [e-/bin/px/mn]
    return bkg_flux_band # [e-/bin/px/mn]


















