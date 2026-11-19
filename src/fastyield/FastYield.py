# import FastYield modules
from .config import dpi_fig, h, c, R0_max, LMIN, LMAX, bands, instrus, colors_instru, get_planet_table_path, ignore_reflected_thresh_um, SNR_thresh, planet_types, planet_types_reduced, instrus_with_systematics, filter_type_default
from .get_specs import get_config_data, get_band_lims, get_wa, get_R_instru, get_DL_FWHM
from .utils import faded
from .spectrum import load_vega_spectrum, get_mag, get_spectrum_contribution_name_model, get_thermal_reflected_spectrum, get_counts_from_density, get_wave_K, get_wave_model
from .FastCurves import FastCurves
from .FastYield_helpers import load_planet_table, get_filename_table, get_planet_type, get_valid_mask, get_invalid_mask, get_SNR_from_table, build_match_dict, get_contrast_from_saved_SNR, get_contrast_from_table, get_size_from_SNR, mass_to_size, get_mask_planet_type, get_planet_index, PlainLogFormatter

# import astropy modules
from astropy.table import QTable
from astropy import units as u
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import numpy modules
import numpy as np

# import other modules
from pathlib import Path
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import corner
import pandas as pd

# For fits warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", message="Header block contains null bytes*")



#######################################################################################################################
#################################################### Magnitudes computations: #########################################
#######################################################################################################################

def _init_magnitudes_worker(ctx):
    """
    Initialize the magnitude-computation context in each worker.
    """
    global _MAG_CTX
    _MAG_CTX = ctx



def process_magnitudes(idx):
    """
    Worker that computes all magnitudes for a single planet.

    Parameters
    ----------
    idx: int
        Row index in the original table.

    Returns
    -------
    idx : int
        Row index in the original table.
    mags : dict[str, float or Quantity]
        Magnitudes keyed by the final column names to be written.
    """

    planet_table    = _MAG_CTX["planet_table"]
    wave_model      = _MAG_CTX["wave_model"]
    wave_K          = _MAG_CTX["wave_K"]
    counts_vega     = _MAG_CTX["counts_vega"]
    counts_vega_K   = _MAG_CTX["counts_vega_K"]
    masks           = _MAG_CTX["masks"]
    use_reflected   = _MAG_CTX["use_reflected"]

    # Planet row
    planet = planet_table[idx]

    # Computing models on wave_model
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model="auto", reflected_model="auto", instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True)

    # Computing the magnitudes
    mags = {}

    # Instrus mags
    for instru in instrus:
        mask_instru = masks[instru]
        mags[f"StarINSTRUmag({instru})"]                      = get_mag(wave=wave_model[mask_instru], density_obs=star_spectrum.flux[mask_instru],   density_vega=None, counts_vega=counts_vega[instru])
        mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_spectrum.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
        mags[f"PlanetINSTRUmag({instru})(thermal)"]           = get_mag(wave=wave_model[mask_instru], density_obs=planet_thermal.flux[mask_instru],  density_vega=None, counts_vega=counts_vega[instru])
        if use_reflected[instru]:
            mags[f"PlanetINSTRUmag({instru})(reflected)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_reflected.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])

    # Bands mags
    for band in bands:
        mask_band = masks[band]
        if band != "K":
            mags[f'Star{band}mag']                  = get_mag(wave=wave_model[mask_band], density_obs=star_spectrum.flux[mask_band],   density_vega=None, counts_vega=counts_vega[band])
        mags[f"Planet{band}mag(thermal+reflected)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_spectrum.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])
        mags[f"Planet{band}mag(thermal)"]           = get_mag(wave=wave_model[mask_band], density_obs=planet_thermal.flux[mask_band],  density_vega=None, counts_vega=counts_vega[band])
        if use_reflected[band]:
            mags[f"Planet{band}mag(reflected)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_reflected.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])

    return idx, mags



def get_planet_table_magnitudes(planet_table):
    """
    Compute star/planet magnitudes for all configured instruments and bands.

    This function:
      1) builds a common wavelength grid and resamples Vega once,
      2) precomputes boolean masks for every instrument/band,
      3) parallelizes per-planet spectral synthesis + photometry,
      4) writes results back column-wise (vectorized, unit-safe).

    Parameters
    ----------
    planet_table : astropy.table.QTable
        Catalog with one row per planet. Columns for output magnitudes should
        already exist (dimensionless).

    Returns
    -------
    planet_table : astropy.table.QTable
        The input table with magnitudes filled in.
    """

    # --- 1) Wavelength grids and Vega spectrum ---

    # Model bandwidth
    wave_model = np.arange(LMIN, LMAX, 1e-4)

    # K-band for photometry
    wave_K = get_wave_K()

    # Vega spectrum on K-band and model-band [J/s/m2/µm]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K,     renorm=False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_model, renorm=False)

    # --- 2) Bandpass masks, Vega flux on bands and “use_reflected” flags ---
    counts_vega_K = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
    counts_vega   = {}
    masks         = {}
    use_reflected = {}
    for instru in instrus:
        lmin, lmax            = get_band_lims(instru)
        masks[instru]         = (wave_model >= lmin) & (wave_model <= lmax)
        use_reflected[instru] = (lmin < ignore_reflected_thresh_um)
        counts_vega[instru]   = get_counts_from_density(wave=wave_model[masks[instru]], density=vega_spectrum.flux[masks[instru]])
    for band in bands:
        lmin, lmax          = get_band_lims(band)
        masks[band]         = (wave_model >= lmin) & (wave_model <= lmax)
        use_reflected[band] = (lmin < ignore_reflected_thresh_um)
        counts_vega[band]   = get_counts_from_density(wave=wave_model[masks[band]], density=vega_spectrum.flux[masks[band]])

    # --- 3) Init global context for workers ---
    mag_ctx = dict(planet_table=planet_table, wave_model=wave_model, wave_K=wave_K, counts_vega=counts_vega, counts_vega_K=counts_vega_K, masks=masks, use_reflected=use_reflected)

    # Magnitude estimations
    print()
    with Pool(processes=max(1, cpu_count() // 2), initializer=_init_magnitudes_worker, initargs=(mag_ctx,)) as pool:
        for (idx, mags) in tqdm(pool.imap(process_magnitudes, [(idx) for idx in range(len(planet_table))]), total=len(planet_table), desc="         Estimating all magnitudes"):
            for instru in instrus:
                planet_table[idx][f"StarINSTRUmag({instru})"]                      = mags[f"StarINSTRUmag({instru})"]
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal+reflected)"] = mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"]
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal)"]           = mags[f"PlanetINSTRUmag({instru})(thermal)"]
                if use_reflected[instru]:
                    planet_table[idx][f"PlanetINSTRUmag({instru})(reflected)"] = mags[f"PlanetINSTRUmag({instru})(reflected)"]
            for band in bands:
                if band != "K":
                    planet_table[idx][f'Star{band}mag']                  = mags[f'Star{band}mag']
                planet_table[idx][f"Planet{band}mag(thermal+reflected)"] = mags[f"Planet{band}mag(thermal+reflected)"]
                planet_table[idx][f"Planet{band}mag(thermal)"]           = mags[f"Planet{band}mag(thermal)"]
                if use_reflected[band]:
                    planet_table[idx][f"Planet{band}mag(reflected)"] = mags[f"Planet{band}mag(reflected)"]

    return planet_table



#######################################################################################################################
########################################### SNR computations: #########################################################
#######################################################################################################################

def warmup_instrument_caches(instru, config_data, apodizer, strehl, coronagraph, background):
    from .get_specs import _get_transmission, get_PSF_profile, get_logit_coronagraphic_profile_interp, get_R_corr_interp, get_bkg_flux_band
    from .spectrum import get_wave_band
    tellurics = config_data["base"] == "ground"
    for band in config_data["gratings"]:
        get_wave_band(instru=instru, band=band)
        _get_transmission(instru=instru, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, fill_value=np.nan)
        get_PSF_profile(band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph, instru=instru, separation_planet=None, return_FastYield=False, return_hdr=True)
        if coronagraph is not None:
            get_logit_coronagraphic_profile_interp(instru=instru, band=band, strehl=strehl, apodizer=apodizer, coronagraph=coronagraph)
        if instru in {"MIRIMRS", "NIRSpec"}:
            get_R_corr_interp(instru=instru, band=band)
        if background is not None:
            get_bkg_flux_band(instru=instru, band=band, background_level=background)

def _init_snr_worker(ctx):
    global _SNR_CTX
    _SNR_CTX    = ctx
    config_data = get_config_data(ctx["instru"])
    warmup_instrument_caches(instru=ctx["instru"], config_data=config_data, apodizer=ctx["apodizer"], strehl=ctx["strehl"], coronagraph=ctx["coronagraph"], background=ctx["background"])

def get_photon_weights(wave):
    """
    Precompute the photon-conversion integration weights associated with a wavelength grid.
    """
    wave    = np.asarray(wave, dtype=float)
    dwave   = np.gradient(wave)                         # [µm/bin]
    weights = wave*1e-6/(h*c) * dwave                   # [ph/J] * [µm/bin]
    return weights

def get_counts_from_weights(density, weights):
    """
    Integrate a spectral energy density using precomputed photon-conversion weights.
    """
    counts = np.nansum(density * weights) # [ph/s/m2]
    if not np.isfinite(counts) or counts <= 0.0:
        return np.nan
    return counts

def get_mag_from_weights(density_obs, weights, counts_vega):
    """
    Compute a Vega magnitude using precomputed photon-conversion weights.
    """
    counts_obs = get_counts_from_weights(density=density_obs, weights=weights) # [ph/s/m2]
    return -2.5 * np.log10(counts_obs / counts_vega)

def process_SNR(idx):
    """
    Worker: compute SNR and band magnitudes for a single planet.

    Parameters
    ----------
    idx: int

    Returns
    -------
    result : tuple
         - idx       : int
         - mags_p    : dict {'Star{instru}mag' : float, 'Planet{instru}mag' : float, 'Star{band}mag' : float, ... }
         - name_band : list[str]
         - signal    : np.ndarray [e-/DIT]  (per name_band entry)
         - sigma_f   : np.ndarray [e-/DIT]
         - sigma_s   : np.ndarray [e-/DIT]
         - DIT       : np.ndarray [mn]
    """

    # Context
    planet_table           = _SNR_CTX["planet_table"]
    instru                 = _SNR_CTX["instru"]
    thermal_model          = _SNR_CTX["thermal_model"]
    reflected_model        = _SNR_CTX["reflected_model"]
    spectrum_contributions = _SNR_CTX["spectrum_contributions"]
    wave_model             = _SNR_CTX["wave_model"]
    wave_K                 = _SNR_CTX["wave_K"]
    counts_vega            = _SNR_CTX["counts_vega"]
    counts_vega_K          = _SNR_CTX["counts_vega_K"]
    band0                  = _SNR_CTX["band0"]
    exposure_time          = _SNR_CTX["exposure_time"]
    apodizer               = _SNR_CTX["apodizer"]
    strehl                 = _SNR_CTX["strehl"]
    coronagraph            = _SNR_CTX["coronagraph"]
    Rc                     = _SNR_CTX["Rc"]
    filter_type            = _SNR_CTX["filter_type"]
    background             = _SNR_CTX["background"]
    systematics            = _SNR_CTX["systematics"]
    PCA                    = _SNR_CTX["PCA"]
    N_PCA                  = _SNR_CTX["N_PCA"]
    masks                  = _SNR_CTX["masks"]
    weights                = _SNR_CTX["weights"]
    bands_valid            = _SNR_CTX["bands_valid"]

    # Planet row
    planet = planet_table[idx]

    # Computing models on wave_model
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True)

    # Re-computing the magnitudes (in case the models is different from the original table)
    mags = {}

    # Instrumental magnitudes
    mask_instru    = masks[instru]
    weights_instru = weights[instru]
    mags[f"StarINSTRUmag({instru})"] = get_mag_from_weights(density_obs=star_spectrum.flux[mask_instru], weights=weights_instru, counts_vega=counts_vega[instru])
    if "thermal" in spectrum_contributions:
        mags[f"PlanetINSTRUmag({instru})(thermal)"] = get_mag_from_weights(density_obs=planet_thermal.flux[mask_instru], weights=weights_instru, counts_vega=counts_vega[instru])
    if "reflected" in spectrum_contributions:
        mags[f"PlanetINSTRUmag({instru})(reflected)"] = get_mag_from_weights(density_obs=planet_reflected.flux[mask_instru], weights=weights_instru, counts_vega=counts_vega[instru])
    if spectrum_contributions == "thermal+reflected":
        mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"] = get_mag_from_weights(density_obs=planet_spectrum.flux[mask_instru], weights=weights_instru, counts_vega=counts_vega[instru])

    # Standard-band magnitudes
    for band in bands_valid:
        mask_band    = masks[band]
        weights_band = weights[band]
        if band != "K":
            mags[f"Star{band}mag"] = get_mag_from_weights(density_obs=star_spectrum.flux[mask_band], weights=weights_band, counts_vega=counts_vega[band])
        if "thermal" in spectrum_contributions:
            mags[f"Planet{band}mag(thermal)"] = get_mag_from_weights(density_obs=planet_thermal.flux[mask_band], weights=weights_band, counts_vega=counts_vega[band])
        if "reflected" in spectrum_contributions:
            mags[f"Planet{band}mag(reflected)"] = get_mag_from_weights(density_obs=planet_reflected.flux[mask_band], weights=weights_band, counts_vega=counts_vega[band])
        if spectrum_contributions == "thermal+reflected":
            mags[f"Planet{band}mag(thermal+reflected)"] = get_mag_from_weights(density_obs=planet_spectrum.flux[mask_band], weights=weights_band, counts_vega=counts_vega[band])

    # Computing the SNR for the planet
    mag_s = mags[f"StarINSTRUmag({instru})"]
    mag_p = mags[f"PlanetINSTRUmag({instru})({spectrum_contributions})"]
    name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = FastCurves(instru=instru, band_only=None, calculation="SNR", mag_star=mag_s, band0=band0, exposure_time=exposure_time, mag_planet=mag_p, separation_planet=planet["AngSep"].value/1000, return_FastYield=True, show_plot=False, verbose=False, planet_name=planet["PlanetName"], planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, background=background, systematics=systematics, PCA=PCA, N_PCA=N_PCA)

    return idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band



def get_planet_table_SNR(instru, table="Archive", thermal_model="None", reflected_model="None", apodizer="NO_SP", strehl="NO_JQ", coronagraph=None, Rc=100, filter_type=filter_type_default, background="medium", systematics=False, PCA=False, N_PCA=20):
    """
    Compute per-planet SNRs for a given instrument and write results to the table.

    Parameters
    ----------
    instru : str
        Instrument name (must be known by your config & globals).
    table : str, optional
        Catalog name, used for both input and output (default: "Archive").
        The input must be <table>_Pull_For_FastYield.ecsv in its directory.
    thermal_model, reflected_model : str, optional
        Spectrum choices for the planet/starlight models.
    apodizer, strehl : str, optional
        Instrument performance flags passed to FastCurves.
    coronagraph : str or None, optional
        Coronagraph choice passed to FastCurves.
    Rc : float or None
        High-pass cutoff resolution (None → no filtering).
    filter_type : str
        Filtering kernel ("gaussian", "step", "smoothstep", ...).
    systematics : bool, optional
        Whether to include systematics noise in FastCurves.
    PCA : bool, optional
        Whether to include PCA post-processing in FastCurves.
    N_PCA : int, optional
        Number of PCA components if PCA=True.
    """
    config_data   = get_config_data(instru)
    exposure_time = 120 # [mn]
    time1         = time.time()

    # --- 1) Input catalog and output directory ---
    planet_table = load_planet_table(f"{table}_Pull_For_FastYield.ecsv", table=table)
    path         = get_planet_table_path(table, create=True)

    # --- 2) Wavelength grids and Vega ---
    # K-band for photometry
    wave_K = get_wave_K()

    # Model bandwidth
    lmin_instru = config_data["lambda_range"]["lambda_min"]                     # [µm]
    lmax_instru = config_data["lambda_range"]["lambda_max"]                     # [µm]
    R_instru    = get_R_instru(instru=instru)                                   # Max instrument resolution (factor 2 to be sure to not loose spectral information)
    R_model     = min(R_instru, R0_max)                                         # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
    wave_model  = get_wave_model(lmin=lmin_instru, lmax=lmax_instru, R=R_model) # [µm] Model wavelength axis (with constant dl step)

    # Vega spectrum on K-band and instru-band [J/s/m2/µm]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K,     renorm=False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_model, renorm=False)

    # --- 3) Create columns for signal, noise and DIT length ---
    planet_table["signal_INSTRU"]     = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["sigma_fund_INSTRU"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["sigma_syst_INSTRU"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["DIT_INSTRU"]        = np.full(len(planet_table), np.nan) # [mn]
    for band in bands:
        planet_table[f"signal_{band}"]     = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"sigma_fund_{band}"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"sigma_syst_{band}"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"DIT_{band}"]        = np.full(len(planet_table), np.nan) # [nm]

    # --- 4) Bandpass masks and Vega flux on bands ---
    counts_vega_K       = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
    counts_vega         = {}
    masks               = {}
    weights             = {}
    bands_valid         = []
    masks[instru]       = (wave_model >= lmin_instru) & (wave_model <= lmax_instru)
    weights[instru]     = get_photon_weights(wave_model[masks[instru]])
    counts_vega[instru] = get_counts_from_weights(density=vega_spectrum.flux[masks[instru]], weights=weights[instru])
    for band in bands:
        lmin_band, lmax_band = get_band_lims(band)
        if lmin_instru <= lmin_band and lmax_instru >= lmax_band: # if this band is inside the instrumental range
            bands_valid.append(band)
            masks[band]       = (wave_model >= lmin_band) & (wave_model <= lmax_band)
            weights[band]     = get_photon_weights(wave_model[masks[band]])
            counts_vega[band] = get_counts_from_weights(density=vega_spectrum.flux[masks[band]], weights=weights[band])

    # Band where magnitudes are defined for the FastCurves computations
    band0 = "instru"

    # Contribution and model labels
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Suffix
    suffix = "with systematics+PCA" if (systematics and PCA) else ("with systematics" if systematics else "without systematics")

    # Print
    print(f"\n {instru} ({apodizer} & {strehl} & {coronagraph}) {suffix} ({thermal_model} & {reflected_model})")

    # --- 5) Init global context for workers ---
    global _SNR_CTX
    _SNR_CTX = dict(planet_table=planet_table, instru=instru, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, wave_model=wave_model, wave_K=wave_K, counts_vega=counts_vega, counts_vega_K=counts_vega_K, band0=band0, exposure_time=exposure_time, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, background=background, systematics=systematics, PCA=PCA, N_PCA=N_PCA, masks=masks, weights=weights, bands_valid=bands_valid)

    # Function to enter the estimations in the planet_table
    def set_planet_table_values(planet_table, idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band):
        planet_table[idx][f"StarINSTRUmag({instru})"] = mags[f"StarINSTRUmag({instru})"]
        if "thermal" in spectrum_contributions:
            planet_table[idx][f"PlanetINSTRUmag({instru})(thermal)"] = mags[f"PlanetINSTRUmag({instru})(thermal)"]
        if "reflected" in spectrum_contributions:
            planet_table[idx][f"PlanetINSTRUmag({instru})(reflected)"] = mags[f"PlanetINSTRUmag({instru})(reflected)"]
        if spectrum_contributions == "thermal+reflected":
            planet_table[idx][f"PlanetINSTRUmag({instru})(thermal+reflected)"] = mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"]
        for band in bands_valid:
            if band != "K":
                planet_table[idx][f"Star{band}mag"] = mags[f"Star{band}mag"]
            if "thermal" in spectrum_contributions:
                planet_table[idx][f"Planet{band}mag(thermal)"] = mags[f"Planet{band}mag(thermal)"]
            if "reflected" in spectrum_contributions:
                planet_table[idx][f"Planet{band}mag(reflected)"] = mags[f"Planet{band}mag(reflected)"]
            if spectrum_contributions == "thermal+reflected":
                planet_table[idx][f"Planet{band}mag(thermal+reflected)"] = mags[f"Planet{band}mag(thermal+reflected)"]
        SNR       = np.sqrt(exposure_time / DIT_band) * signal_planet / np.sqrt(sigma_fund_planet**2 + (exposure_time / DIT_band) * sigma_syst_planet**2)
        valid_snr = np.isfinite(SNR)
        if np.any(valid_snr):
            idx_max = np.where(valid_snr)[0][np.nanargmax(SNR[valid_snr])]
            planet_table[idx]["signal_INSTRU"]     = signal_planet[idx_max]
            planet_table[idx]["sigma_fund_INSTRU"] = sigma_fund_planet[idx_max]
            planet_table[idx]["sigma_syst_INSTRU"] = sigma_syst_planet[idx_max]
            planet_table[idx]["DIT_INSTRU"]        = DIT_band[idx_max]
        else:
            planet_table[idx]["signal_INSTRU"]     = np.nan
            planet_table[idx]["sigma_fund_INSTRU"] = np.nan
            planet_table[idx]["sigma_syst_INSTRU"] = np.nan
            planet_table[idx]["DIT_INSTRU"]        = np.nan
        for nb, band in enumerate(name_band):
            planet_table[idx][f"signal_{band}"]     = signal_planet[nb]
            planet_table[idx][f"sigma_fund_{band}"] = sigma_fund_planet[nb]
            planet_table[idx][f"sigma_syst_{band}"] = sigma_syst_planet[nb]
            planet_table[idx][f"DIT_{band}"]        = DIT_band[nb]

    # --- 6) Run SNR for each planet (parallel if not PCA or serial if PCA) ---
    if PCA: # if PCA, no multiprocessing (otherwise it crashes: TODO ?)
        for idx in tqdm(range(len(planet_table)), desc="Serial"):
            idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = process_SNR(idx)
            set_planet_table_values(planet_table, idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band)

    else: # if no PCA, uses multiprocessing
        nproc     = max(1, cpu_count() - 3)
        chunksize = max(1, len(planet_table) // (20 * nproc))
        with Pool(processes=nproc, initializer=_init_snr_worker, initargs=(_SNR_CTX,),) as pool:
            iterator = pool.imap_unordered(process_SNR, range(len(planet_table)), chunksize=chunksize)
            for result in tqdm(iterator, total=len(planet_table), desc="Multiprocessing"):
                idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = result
                set_planet_table_values(planet_table, idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band)

    print(f"\n Calculating SNR took {(time.time()-time1)/60:.1f} mn")
    filename = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=name_model)
    planet_table.write(path / filename, format='ascii.ecsv', overwrite=True)
    print(f"Table saved: {path / filename}")



def all_SNR_table(table="Archive", instrus=instrus): # takes ~ 13 hours
    """
    To compute SNR for every instruments with different model spectra
    """
    time0 = time.time()
    for instru in instrus:
        config_data = get_config_data(instru)
        if config_data["lambda_range"]["lambda_max"] > ignore_reflected_thresh_um:
            thermal_models   = ["None", "auto"]
            reflected_models = ["None"]
        else:
            thermal_models   = ["None", "auto"]
            reflected_models = ["None", "auto"]
        for apodizer in config_data["apodizers"]:
            for strehl in config_data["strehls"]:
                for coronagraph in config_data["coronagraphs"]:
                    for thermal_model in thermal_models:
                        for reflected_model in reflected_models:
                            if thermal_model == "None" and reflected_model == "None":
                                continue
                            else:
                                get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=False)
                                if instru in instrus_with_systematics:
                                    get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=True)
                                    get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=True, PCA=True, N_PCA=20)

    print('\n Calculating all SNR took {0:.3f} s'.format(time.time()-time0))



#######################################################################################################################
################################################# PLOTS: ##############################################################
#######################################################################################################################

########################################
# Classifications and statistics plots #
########################################

def planet_table_classification(planet_table=None, dpi=dpi_fig):
    if planet_table is None:
        planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")

    radius = np.array(planet_table["PlanetRadius"].value) # R_earth
    mass   = np.array(planet_table["PlanetMass"].value)   # M_earth

    mask_im = planet_table["DiscoveryMethod"]=="Imaging"
    mask_rv = planet_table["DiscoveryMethod"]=="Radial Velocity"
    mask_tr = planet_table["DiscoveryMethod"]=="Transit"
    mask_ot = (~mask_im) & (~mask_rv) & (~mask_tr)

    mask_cold = np.array(planet_table["PlanetTeff"].value < 250)
    mask_temp = np.array(250 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 500)
    mask_warm = np.array(500 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 1000)
    mask_hot  = np.array(1000 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 1500)
    mask_vhot = np.array(1500 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 2000)
    mask_uhot = np.array(2000 <= planet_table["PlanetTeff"].value)

    plt.figure(figsize=(9, 5), dpi=dpi)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Mass $M_\oplus$",   fontsize=18)
    plt.ylabel(r"Radius $R_\oplus$", fontsize=18)
    #plt.title(f"FastYield classification: {len(planet_table)} known exoplanets", fontsize=16)
    plt.xlim(np.nanmin(mass[mass!=0]), np.nanmax(mass[mass!=0]))
    plt.ylim(np.nanmin(radius[radius!=0]), np.nanmax(radius[radius!=0]))

    plt.scatter(mass[mask_im & mask_cold], radius[mask_im & mask_cold], c="#6fb0c9", marker="s")
    plt.scatter(mass[mask_rv & mask_cold], radius[mask_rv & mask_cold], c="#6fb0c9", marker="o")
    plt.scatter(mass[mask_tr & mask_cold], radius[mask_tr & mask_cold], c="#6fb0c9", marker="v")
    plt.scatter(mass[mask_ot & mask_cold], radius[mask_ot & mask_cold], c="#6fb0c9", marker="P")

    plt.scatter(mass[mask_im & mask_temp], radius[mask_im & mask_temp], c="#7ac87a", marker="s")
    plt.scatter(mass[mask_rv & mask_temp], radius[mask_rv & mask_temp], c="#7ac87a", marker="o")
    plt.scatter(mass[mask_tr & mask_temp], radius[mask_tr & mask_temp], c="#7ac87a", marker="v")
    plt.scatter(mass[mask_ot & mask_temp], radius[mask_ot & mask_temp], c="#7ac87a", marker="P")

    plt.scatter(mass[mask_im & mask_warm], radius[mask_im & mask_warm], c="#c8c26f", marker="s")
    plt.scatter(mass[mask_rv & mask_warm], radius[mask_rv & mask_warm], c="#c8c26f", marker="o")
    plt.scatter(mass[mask_tr & mask_warm], radius[mask_tr & mask_warm], c="#c8c26f", marker="v")
    plt.scatter(mass[mask_ot & mask_warm], radius[mask_ot & mask_warm], c="#c8c26f", marker="P")

    plt.scatter(mass[mask_im & mask_hot], radius[mask_im & mask_hot], c="#f0a44f", marker="s")
    plt.scatter(mass[mask_rv & mask_hot], radius[mask_rv & mask_hot], c="#f0a44f", marker="o")
    plt.scatter(mass[mask_tr & mask_hot], radius[mask_tr & mask_hot], c="#f0a44f", marker="v")
    plt.scatter(mass[mask_ot & mask_hot], radius[mask_ot & mask_hot], c="#f0a44f", marker="P")

    plt.scatter(mass[mask_im & mask_vhot], radius[mask_im & mask_vhot], c="#e36c4a", marker="s")
    plt.scatter(mass[mask_rv & mask_vhot], radius[mask_rv & mask_vhot], c="#e36c4a", marker="o")
    plt.scatter(mass[mask_tr & mask_vhot], radius[mask_tr & mask_vhot], c="#e36c4a", marker="v")
    plt.scatter(mass[mask_ot & mask_vhot], radius[mask_ot & mask_vhot], c="#e36c4a", marker="P")

    plt.scatter(mass[mask_im & mask_uhot], radius[mask_im & mask_uhot], c="#c23a3a", marker="s")
    plt.scatter(mass[mask_rv & mask_uhot], radius[mask_rv & mask_uhot], c="#c23a3a", marker="o")
    plt.scatter(mass[mask_tr & mask_uhot], radius[mask_tr & mask_uhot], c="#c23a3a", marker="v")
    plt.scatter(mass[mask_ot & mask_uhot], radius[mask_ot & mask_uhot], c="#c23a3a", marker="P")

    plt.minorticks_on()
    plt.tick_params(axis='both', labelsize=14)
    plt.xlim(1e-1, 1e4)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)

    # LEGEND
    ax = plt.gca()
    temp_colors_merged = {"Cold (<250 K)":          "#6fb0c9",
                          "Temperate (250–500 K)":  "#7ac87a",
                          "Warm (500–1000 K)":      "#c8c26f",
                          "Hot (1000–1500 K)":      "#f0a44f",
                          "Very-hot (1500–2000 K)": "#e36c4a",
                          "Ultra-hot (≥2000 K)":    "#c23a3a"}
    legend_temp = [mlines.Line2D([0], [0], marker='o', linestyle='', markersize=8, label=lbl, markerfacecolor=col, markeredgecolor='none') for lbl, col in temp_colors_merged.items()]
    leg1 = ax.legend(handles=legend_temp, title="Temperature bands", loc='lower right', frameon=True, fontsize=12, title_fontsize=14)
    ax.add_artist(leg1)
    legend_methods = [mlines.Line2D([], [], marker='s', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Imaging'),
                      mlines.Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Radial Velocity'),
                      mlines.Line2D([], [], marker='v', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Transit'),
                      mlines.Line2D([], [], marker='P', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Other')]
    ax.legend(handles=legend_methods, title="Discovery method", loc='upper left', frameon=True, fontsize=12, title_fontsize=14)

    # TYPES ZONES
    def build_type_bounds(planet_types, ax):
        """Retourne une liste {name, m1,m2,r1,r2} en agrégeant toutes les bandes Teff."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        by_type = {}
        for key, crit in planet_types.items():
            tname = key.split(" ", 1)[1]  # ex: "Sub-Earth"
            d = by_type.setdefault(tname, {
                "m_min": np.inf, "m_max": -np.inf,
                "r_min": np.inf, "r_max": -np.inf
            })
            d["m_min"] = min(d["m_min"], crit["mass_min"])
            d["m_max"] = max(d["m_max"], crit["mass_max"])
            d["r_min"] = min(d["r_min"], crit["radius_min"])
            d["r_max"] = max(d["r_max"], crit["radius_max"])

        # borne infinie -> limite d’axe
        bounds = []
        for name, d in by_type.items():
            m1 = max(d["m_min"], xlim[0])
            m2 = min(d["m_max"], xlim[1]) if np.isfinite(d["m_max"]) else xlim[1]
            r1 = max(d["r_min"], ylim[0])
            r2 = min(d["r_max"], ylim[1]) if np.isfinite(d["r_max"]) else ylim[1]
            if np.isfinite([m1,m2,r1,r2]).all() and (m2>m1) and (r2>r1):
                bounds.append({"name": name, "m1": m1, "m2": m2, "r1": r1, "r2": r2})
        return bounds

    def draw_grey_highlight(ax, bounds, face_alpha=0.10, edge_alpha=0.9, edge_lw=1.5, label_fs=14):
        """Surligne en gris (au-dessus de tout)."""
        for b in bounds:
            rect = Rectangle((b["m1"], b["r1"]),
                             b["m2"] - b["m1"], b["r2"] - b["r1"],
                             facecolor=(0,0,0,face_alpha),   # gris translucide
                             edgecolor=(0,0,0,edge_alpha),   # contour gris foncé
                             linewidth=edge_lw,
                             zorder=1e6, clip_on=False)
            ax.add_patch(rect)

            # centre géométrique (axes log)
            cx   = np.sqrt(b["m1"]*b["m2"])
            cy   = np.sqrt(b["r1"]*b["r2"])
            text = b["name"].replace('-', '\n')
            ax.text(cx, cy, text,
                    ha="center", va="center",
                    fontsize=label_fs, color="black", weight="bold",
                    zorder=1e6+1, alpha=1.0,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.8)],
                    )
    bounds = build_type_bounds(planet_types, ax)
    draw_grey_highlight(ax, bounds, face_alpha=0.08, edge_alpha=0.85, edge_lw=1.2, label_fs=13)
    plt.draw()
    plt.tight_layout()
    plt.show(block=False)




def planet_table_classification_histogram(planet_table=None, dpi=dpi_fig):
    """
    Plot the number of planets per planet type with visual circle markers.

    The plot shows:
    - a horizontal histogram of the number of planets per type,
    - one circle per type,
    - circle size proportional to the characteristic radius of the type,
    - circle edge width proportional to the characteristic mass of the type,
    - circle color based on a simplified temperature category:
      Cold (<500 K), Warm (500-1000 K), Hot (>=1000 K).

    If a 'PlanetType' column already exists, it is used directly.
    Otherwise, planet types are computed with get_planet_type().

    Parameters
    ----------
    planet_table : astropy.table.Table, astropy.table.QTable, or pandas.DataFrame
        Table containing either a 'PlanetType' column or the columns needed
        by get_planet_type().

    Returns
    -------
    fig, (ax_circ, ax_bar) : matplotlib Figure and Axes
        Figure and axes of the plot.
    """
    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _has_column(table, name):
        """Return True if the table contains the requested column."""
        if hasattr(table, "colnames"):
            return name in table.colnames
        return name in table.columns

    def _representative_positive_value(vmin, vmax):
        """
        Return a representative value for a positive interval.

        If the lower bound is zero, use a fraction of the upper bound instead
        of the upper bound itself. This avoids overestimating open low-radius
        or low-mass categories such as Sub-Earth.
        """
        if np.isfinite(vmin) and np.isfinite(vmax):
            if vmin > 0 and vmax > 0:
                return np.sqrt(vmin * vmax)
            if vmin <= 0 and vmax > 0:
                return 0.70 * vmax
        if np.isfinite(vmin) and vmin > 0:
            return float(vmin)
        if np.isfinite(vmax) and vmax > 0:
            return float(vmax)
        return np.nan

    def _linear_center(vmin, vmax):
        """Return a representative value for a linear interval."""
        if np.isfinite(vmin) and np.isfinite(vmax):
            return 0.5 * (vmin + vmax)
        if np.isfinite(vmin):
            return float(vmin)
        if np.isfinite(vmax):
            return float(vmax)
        return np.nan

    def _temperature_category(teff):
        """Map a characteristic temperature to one of three broad categories."""
        if not np.isfinite(teff):
            return "Unidentified"
        if teff < 500:
            return "Cold (<500 K)"
        elif teff < 1000:
            return "Warm (500–1000 K)"
        else:
            return "Hot (≥1000 K)"

    def _radius_to_marker_size(radius, rmin, rmax, dmin=5, dmax=40, gamma=1.0):
        """
        Convert a characteristic radius into a scatter-marker area.
        """
        if not np.isfinite(radius):
            radius = 4.0
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin <= 0 or rmax <= 0 or rmin == rmax:
            visual_diameter = 16.0
        else:
            radius = np.clip(radius, rmin, rmax)
            t = (np.log10(radius) - np.log10(rmin)) / (np.log10(rmax) - np.log10(rmin))
            t = np.clip(t, 0, 1)
            # Slight boost around terrestrial radii so that R = 1 R_earth
            # remains visible compared to the smallest categories.
            visual_diameter = dmin + (dmax - dmin) * t**gamma
            if radius <= 2:
                visual_diameter *= 1.2
        return visual_diameter**2

    def _mass_to_linewidth(mass, mmin, mmax, lwmin=0.5, lwmax=5.0, gamma=2.0):
        """Convert a characteristic mass into a marker edge width."""
        if not np.isfinite(mass):
            return 2.0
        if not np.isfinite(mmin) or not np.isfinite(mmax) or mmin <= 0 or mmax <= 0 or mmin == mmax:
            return 1.8
        mass = np.clip(mass, mmin, mmax)
        t    = (np.log10(mass) - np.log10(mmin)) / (np.log10(mmax) - np.log10(mmin))
        t    = np.clip(t, 0, 1)
        return lwmin + (lwmax - lwmin) * t**gamma

    def _format_radius_label(r):
        """Format radius labels for the legend."""
        if r >= 10:
            return f"{r:.0f} R$_\\oplus$"
        return f"{r:.1f} R$_\\oplus$"

    def _format_mass_label(m):
        """Format mass labels for the legend."""
        if m >= 100:
            return f"{m:.0f} M$_\\oplus$"
        if m >= 10:
            return f"{m:.1f} M$_\\oplus$"
        return f"{m:.1f} M$_\\oplus$"

    if planet_table is None:
        planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")

    # ------------------------------------------------------------------
    # Circle colors: simplified temperature categories
    # ------------------------------------------------------------------
    temp_colors = {"Cold (<500 K)": "#5B8DB8", "Warm (500–1000 K)": "#D99C4A", "Hot (≥1000 K)": "#B94A48", "Unidentified": "#BDBDBD"}

    # ------------------------------------------------------------------
    # Retrieve or compute planet types
    # ------------------------------------------------------------------
    if _has_column(planet_table, "PlanetType"):
        planet_types_array = np.asarray(planet_table["PlanetType"], dtype=str)
    else:
        if hasattr(planet_table, "colnames"):
            planet_types_array = np.array([get_planet_type(planet_table[i]) for i in range(len(planet_table))], dtype=str)
        else:
            planet_types_array = np.array([get_planet_type(row) for _, row in planet_table.iterrows()], dtype=str)

    planet_types_array = np.char.strip(planet_types_array)
    planet_types_array[planet_types_array == ""] = "Unidentified"

    # ------------------------------------------------------------------
    # Count planets per type
    # ------------------------------------------------------------------
    counts = pd.Series(planet_types_array).value_counts()

    mask_unidentified = counts.index.str.contains("Unidentified", case=False, regex=False)
    identified_counts = counts[~mask_unidentified].sort_values(ascending=False)
    unknown_counts    = counts[mask_unidentified].sort_values(ascending=False)

    counts = pd.concat([identified_counts, unknown_counts])

    labels = counts.index.to_numpy()
    values = counts.values.astype(int)
    y_pos  = np.arange(len(labels))
    ntypes = len(labels)

    # ------------------------------------------------------------------
    # Characteristic values for each type
    # ------------------------------------------------------------------
    radius_ref = np.full(ntypes, np.nan, dtype=float)
    mass_ref   = np.full(ntypes, np.nan, dtype=float)
    teff_ref   = np.full(ntypes, np.nan, dtype=float)
    temp_cat   = np.empty(ntypes, dtype=object)
    is_unknown = np.array(["Unidentified" in lbl for lbl in labels], dtype=bool)

    for i, ptype in enumerate(labels):
        if is_unknown[i] or ptype not in planet_types:
            temp_cat[i] = "Unidentified"
            continue

        criteria = planet_types[ptype]

        radius_ref[i] = _representative_positive_value(criteria["radius_min"], criteria["radius_max"])
        mass_ref[i]   = _representative_positive_value(criteria["mass_min"], criteria["mass_max"])
        teff_ref[i]   = _linear_center(criteria["teff_min"], criteria["teff_max"])
        temp_cat[i]   = _temperature_category(teff_ref[i])

    finite_r = radius_ref[np.isfinite(radius_ref) & (radius_ref > 0)]
    finite_m = mass_ref[np.isfinite(mass_ref) & (mass_ref > 0)]

    rmin = np.nanmin(finite_r) if len(finite_r) > 0 else 0.6
    rmax = np.nanmax(finite_r) if len(finite_r) > 0 else 12.0
    mmin = np.nanmin(finite_m) if len(finite_m) > 0 else 0.1
    mmax = np.nanmax(finite_m) if len(finite_m) > 0 else 1000.0

    marker_sizes = np.array([_radius_to_marker_size(r, rmin, rmax) for r in radius_ref], dtype=float)
    marker_lws   = np.array([_mass_to_linewidth(m, mmin, mmax) for m in mass_ref], dtype=float)
    marker_cols  = np.array([temp_colors.get(tc, "#BDBDBD") for tc in temp_cat], dtype=object)

    # Unidentified categories: grey and Neptune-like size, to avoid overlap.
    unidentified_radius      = 4.0
    marker_sizes[is_unknown] = _radius_to_marker_size(unidentified_radius, rmin, rmax)
    marker_lws[is_unknown]   = 2.2
    marker_cols[is_unknown]  = temp_colors["Unidentified"]

    # ------------------------------------------------------------------
    # Histogram colors
    # ------------------------------------------------------------------
    cmap                  = plt.get_cmap("magma")
    n_identified          = len(identified_counts)
    identified_bar_colors = cmap(np.linspace(0.9, 0.1, max(n_identified, 1)))
    bar_colors            = []
    j                     = 0
    for i in range(ntypes):
        if is_unknown[i]:
            bar_colors.append("#BDBDBD")
        else:
            bar_colors.append(identified_bar_colors[j])
            j += 1

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig_height             = max(8.2, 0.50 * ntypes + 1.9)
    fig, (ax_circ, ax_bar) = plt.subplots(1, 2, figsize=(16.0, fig_height), dpi=dpi, sharey=True, gridspec_kw={"width_ratios": [2.0, 8.0], "wspace": 0.02})

    # Left panel: planet-type circles
    ax_circ.scatter(np.full(ntypes, 0.58), y_pos, s=marker_sizes, c=marker_cols, edgecolors="black", linewidths=marker_lws, zorder=3)
    ax_circ.set_xlim(0, 1)
    ax_circ.set_xticks([])
    ax_circ.set_yticks(y_pos)
    ax_circ.set_yticklabels(labels, fontsize=14)
    ax_circ.invert_yaxis()
    ax_circ.tick_params(axis="y", length=0)
    for spine in ax_circ.spines.values():
        spine.set_visible(False)

    # Right panel: histogram
    bars = ax_bar.barh(y_pos, values, color=bar_colors, edgecolor="black", linewidth=0.9, alpha=0.88, zorder=2)
    xmax = np.nanmax(values) if len(values) > 0 else 1
    ax_bar.set_xlim(0, round(xmax*1.1, -2))
    ax_bar.set_xlabel("Number of planets", fontsize=18, labelpad=12)
    ax_bar.set_yticks(y_pos)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.grid(axis="x", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="x", labelsize=14)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)

    # Bar labels
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.014 * xmax, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", ha="left", fontsize=12.5, fontweight="bold", color="#1a1a1a")

    # Global title
    total_planets = int(np.sum(values))
    fig.suptitle(f"Planet population by type ({total_planets} planets)", fontsize=23, fontweight="bold", y=0.95)

    # Inset legend: temperature categories
    ax_temp = ax_bar.inset_axes([0.55, 0.55, 0.41, 0.27])
    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, 1)
    ax_temp.set_xticks([])
    ax_temp.set_yticks([])
    ax_temp.set_facecolor((1, 1, 1, 0.97))
    for spine in ax_temp.spines.values():
        spine.set_color("0.70")
        spine.set_linewidth(1.2)
    ax_temp.text(0.50, 0.88, "Temperature bands", ha="center", va="center", fontsize=18, fontweight="bold")
    temp_items = [("Cold (<500 K)", temp_colors["Cold (<500 K)"]), ("Warm (500–1000 K)", temp_colors["Warm (500–1000 K)"]), ("Hot (≥1000 K)", temp_colors["Hot (≥1000 K)"])]
    y_items    = [0.65, 0.44, 0.23]
    for (label, color), yy in zip(temp_items, y_items):
        ax_temp.scatter(0.15, yy, s=800, facecolor=color, edgecolor="black", linewidth=1.3)
        ax_temp.text(0.29, yy, label, ha="left", va="center", fontsize=18)

    # Inset legend: circle encoding
    ax_enc = ax_bar.inset_axes([0.52, 0.10, 0.45, 0.38])
    ax_enc.set_xlim(0, 1)
    ax_enc.set_ylim(0, 1)
    ax_enc.set_xticks([])
    ax_enc.set_yticks([])
    ax_enc.set_facecolor((1, 1, 1, 0.97))
    for spine in ax_enc.spines.values():
        spine.set_color("0.70")
        spine.set_linewidth(1.2)
    ax_enc.text(0.50, 0.92, "Circle encoding", ha="center", va="center", fontsize=18, fontweight="bold")

    # Radius legend
    y0_rad = 0.75
    ax_enc.text(0.5, y0_rad, "Size → radius", ha="center", va="center", fontsize=18, color="dimgray", fontweight="bold")
    radius_examples = np.array([1.0, 2.0, 4.0, 8.0])
    radius_examples = radius_examples[(radius_examples >= rmin * 0.8) & (radius_examples <= rmax * 1.1)]
    if len(radius_examples) == 0:
        radius_examples = np.array([rmin, 0.5 * (rmin + rmax), rmax])
    x_rad = np.linspace(0.12, 0.88, len(radius_examples))
    y_rad = y0_rad-0.1
    for r, xx in zip(radius_examples, x_rad):
        ax_enc.scatter(xx, y_rad, s=_radius_to_marker_size(r, rmin, rmax), facecolor="white", edgecolor="black", linewidth=1.3)
        ax_enc.text(xx, y_rad-0.1, _format_radius_label(r), ha="center", va="center", fontsize=14)

    # Mass legend
    y0_mass = 0.35
    ax_enc.text(0.5, y0_mass, "Edge width → mass", ha="center", va="center", fontsize=18, color="dimgray", fontweight="bold")
    mass_examples = np.array([1.0, 10.0, 100.0, 600.0])
    mass_examples = mass_examples[(mass_examples >= mmin * 0.8) & (mass_examples <= mmax * 1.2)]
    if len(mass_examples) == 0:
        mass_examples = np.array([mmin, np.sqrt(mmin * mmax), mmax])
    x_mass = np.linspace(0.14, 0.86, len(mass_examples))
    y_mass = y0_mass-0.1
    for m, xx in zip(mass_examples, x_mass):
        ax_enc.scatter(xx, y_mass, s=800, facecolor="white", edgecolor="black", linewidth=_mass_to_linewidth(m, mmin, mmax))
        ax_enc.text(xx, y_mass-0.1, _format_mass_label(m), ha="center", va="center", fontsize=14)

    # Final layout
    fig.subplots_adjust(left=0.27, right=0.97, top=0.93, bottom=0.06, wspace=0.02)
    plt.show(block=False)



def planet_table_statistics(planet_table=None):

    if planet_table is None:
        planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")

    smooth_corner = 1
    planet_table  = planet_table[get_valid_mask(planet_table["PlanetMass"])]

    ndim       = 5 # Mp, Rp, Tp, d, a
    data       = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeff"].value)
    data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
    data[:, 4] = np.log10(np.array(planet_table["Distance"].value))

    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data           = data,
        bins           = 30,
        labels         = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$"],
        quantiles      = [0.16, 0.5, 0.84], # below -+1 sigma
        levels         = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles    = True,
        title_kwargs   = {"fontsize": 14, "pad": 10},
        top_ticks      = False,
        plot_density   = True,
        plot_contours  = True,
        fill_contours  = True,
        smooth         = smooth_corner,
        smooth1d       = smooth_corner,
        color          = "gray",
        contour_kwargs = {"colors": ["black"], "alpha": 0.85, "linewidths": 1.3},
        hist_kwargs    = {"color":   "black",  "alpha": 0.85, "linewidth":  1.3},
        label_kwargs   = {"fontsize": 22})
    #figure.suptitle(f"Archive table statistics (with {len(planet_table)} known exoplanets)", fontsize=18, y=1.05, fontweight="bold")

    tick_labelsize = 14
    for ax in figure.get_axes():
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_labelsize)
        ax.xaxis.get_offset_text().set_fontsize(tick_labelsize)
        ax.yaxis.get_offset_text().set_fontsize(tick_labelsize)
    figure.set_dpi(300)
    plt.show(block=False)




###############
# Yield plots #
###############

def yield_plot_instrus_texp(thermal_model="auto", reflected_model="auto", fraction=False, dpi=dpi_fig):

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    planet_table_harmoni          = load_planet_table(f"Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_harmoni_sp       = load_planet_table(f"Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_andes            = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_without_systematics_{name_model}.ecsv")
    planet_table_andes_coro       = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_{name_model}.ecsv")
    planet_table_eris             = load_planet_table(f"Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_{name_model}.ecsv")
    planet_table_hirise           = load_planet_table(f"Archive_Pull_HiRISE_NO_SP_MED_without_systematics_{name_model}.ecsv")
    planet_table_mirimrs_non_syst = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst     = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nircam           = load_planet_table(f"Archive_Pull_NIRCam_NO_SP_NO_JQ_MASK335R_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_non_syst = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst     = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    #planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")

    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    yield_harmoni          = np.zeros(len(exposure_time))
    yield_andes            = np.zeros(len(exposure_time))
    yield_eris             = np.zeros(len(exposure_time))
    yield_hirise           = np.zeros(len(exposure_time))
    yield_mirimrs_non_syst = np.zeros(len(exposure_time))
    yield_mirimrs_syst     = np.zeros(len(exposure_time))
    yield_mirimrs_syst_pca = np.zeros(len(exposure_time))
    yield_nircam           = np.zeros(len(exposure_time))
    yield_nirspec_non_syst = np.zeros(len(exposure_time))
    yield_nirspec_syst     = np.zeros(len(exposure_time))
    yield_nirspec_syst_pca = np.zeros(len(exposure_time))

    if fraction:
        ratio        = 100
        norm_harmoni = len(planet_table_harmoni)
        norm_andes   = len(planet_table_andes)
        norm_eris    = len(planet_table_eris)
        norm_hirise  = len(planet_table_hirise)
        norm_mirimrs = len(planet_table_mirimrs_non_syst)
        norm_nircam  = len(planet_table_nircam)
        norm_nirspec = len(planet_table_nirspec_non_syst)
    else:
        ratio = norm_harmoni = norm_andes = norm_eris = norm_hirise = norm_mirimrs = norm_nircam = norm_nirspec = 1

    for i in range(len(exposure_time)):
        SNR_harmoni          = get_SNR_from_table(planet_table=planet_table_harmoni,          exposure_time=exposure_time[i], band="INSTRU")
        SNR_harmoni_sp       = get_SNR_from_table(planet_table=planet_table_harmoni_sp,       exposure_time=exposure_time[i], band="INSTRU")
        SNR_andes            = get_SNR_from_table(planet_table=planet_table_andes,            exposure_time=exposure_time[i], band="INSTRU")
        SNR_andes_coro       = get_SNR_from_table(planet_table=planet_table_andes_coro,       exposure_time=exposure_time[i], band="INSTRU")
        SNR_eris             = get_SNR_from_table(planet_table=planet_table_eris,             exposure_time=exposure_time[i], band="INSTRU")
        SNR_hirise           = get_SNR_from_table(planet_table=planet_table_hirise,           exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_non_syst = get_SNR_from_table(planet_table=planet_table_mirimrs_non_syst, exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_syst     = get_SNR_from_table(planet_table=planet_table_mirimrs_syst,     exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_syst_pca = get_SNR_from_table(planet_table=planet_table_mirimrs_syst_pca, exposure_time=exposure_time[i], band="INSTRU")
        SNR_nircam           = get_SNR_from_table(planet_table=planet_table_nircam,           exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_non_syst = get_SNR_from_table(planet_table=planet_table_nirspec_non_syst, exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_syst     = get_SNR_from_table(planet_table=planet_table_nirspec_syst,     exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_syst_pca = get_SNR_from_table(planet_table=planet_table_nirspec_syst_pca, exposure_time=exposure_time[i], band="INSTRU")

        yield_harmoni[i]          = ratio * len(planet_table_harmoni[(SNR_harmoni > SNR_thresh)|(SNR_harmoni_sp > SNR_thresh)]) / norm_harmoni
        yield_andes[i]            = ratio * len(planet_table_andes[(SNR_andes > SNR_thresh)|(SNR_andes_coro > SNR_thresh)])     / norm_andes
        yield_eris[i]             = ratio * len(planet_table_eris[SNR_eris > SNR_thresh])                                       / norm_eris
        yield_hirise[i]           = ratio * len(planet_table_hirise[SNR_hirise > SNR_thresh])                                   / norm_hirise
        yield_mirimrs_non_syst[i] = ratio * len(planet_table_mirimrs_non_syst[SNR_mirimrs_non_syst > SNR_thresh])               / norm_mirimrs
        yield_mirimrs_syst[i]     = ratio * len(planet_table_mirimrs_syst[SNR_mirimrs_syst > SNR_thresh])                       / norm_mirimrs
        yield_mirimrs_syst_pca[i] = ratio * len(planet_table_mirimrs_syst_pca[SNR_mirimrs_syst_pca > SNR_thresh])               / norm_mirimrs
        yield_nircam[i]           = ratio * len(planet_table_nircam[SNR_nircam > SNR_thresh])                                   / norm_nircam
        yield_nirspec_non_syst[i] = ratio * len(planet_table_nirspec_non_syst[SNR_nirspec_non_syst > SNR_thresh])               / norm_nirspec
        yield_nirspec_syst[i]     = ratio * len(planet_table_nirspec_syst[SNR_nirspec_syst > SNR_thresh])                       / norm_nirspec
        yield_nirspec_syst_pca[i] = ratio * len(planet_table_nirspec_syst_pca[SNR_nirspec_syst_pca > SNR_thresh])               / norm_nirspec

    lw = 2
    plt.figure(dpi=dpi, figsize=(13, 7))
    plt.plot(exposure_time, yield_harmoni,          lw=lw, c=colors_instru["HARMONI"], label="ELT/HARMONI")
    plt.plot(exposure_time, yield_andes,            lw=lw, c=colors_instru["ANDES"],   label="ELT/ANDES")
    plt.plot(exposure_time, yield_eris,             lw=lw, c=colors_instru["ERIS"],    label="VLT/ERIS")
    #plt.plot(exposure_time, yield_hirise,           lw=lw, c=colors_instru["HiRISE"],  label="VLT/HiRISE")
    plt.plot(exposure_time, yield_mirimrs_non_syst, lw=lw, c=colors_instru["MIRIMRS"], label="JWST/MIRI/MRS")
    plt.plot(exposure_time, yield_mirimrs_syst,     lw=lw, c=colors_instru["MIRIMRS"], ls='--')
    plt.plot(exposure_time, yield_mirimrs_syst_pca, lw=lw, c=colors_instru["MIRIMRS"], ls=':')
    plt.plot(exposure_time, yield_nircam,           lw=lw, c=colors_instru["NIRCam"],  label="JWST/NIRCam")
    plt.plot(exposure_time, yield_nirspec_non_syst, lw=lw, c=colors_instru["NIRSpec"], label="JWST/NIRSpec/IFU")
    plt.plot(exposure_time, yield_nirspec_syst,     lw=lw, c=colors_instru["NIRSpec"], ls='--')
    plt.plot(exposure_time, yield_nirspec_syst_pca, lw=lw, c=colors_instru["NIRSpec"], ls=':')
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure time per target [mn]', fontsize=20)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=20)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=20)
        plt.yscale('log')
    plt.xlim(exposure_time[0], exposure_time[-1])
    plt.title('Known exoplanets detection yield', fontsize=22, weight='bold', pad=22)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=16, ncols=2, loc="upper left", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Instruments", title_fontsize=18)
    ax        = plt.gca()
    ax_legend = ax.twinx()
    ax_legend.plot([], [], 'k-',  label='Without Systematics',    linewidth=lw)
    ax_legend.plot([], [], 'k--', label='With Systematics',       linewidth=lw)
    ax_legend.plot([], [], 'k:',  label='With Systematics + PCA', linewidth=lw)
    ax_legend.legend(fontsize=16, loc="lower right", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Systematic assumption", title_fontsize=18)
    ax_legend.tick_params(axis='y', colors='w')  # Masking ticks
    plt.tight_layout()
    plt.show(block=False)



def yield_plot_bands_texp(table="Archive", instru="HARMONI", thermal_model="auto", reflected_model="auto", systematics=False, PCA=False, fraction=False, dpi=dpi_fig):

    ls_modes    = ["-", "--", ":"]
    config_data = get_config_data(instru)
    apodizer    = "NO_SP"
    coronagraph = None
    strehl      = config_data["strehls"][0]
    bands       = config_data["gratings"]
    NbBand      = len(bands)
    cmap        = plt.get_cmap("rainbow", NbBand)

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    if instru == "HARMONI":
        modes         = ["NO_SP", "SP1", "SP_Prox"]
        planet_tables = []
        NbPlanet      = np.zeros(len(modes), dtype=int)
        for im, mode in enumerate(modes):
            filename_pt  = get_filename_table(table=table, instru=instru, apodizer=mode, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=name_model)
            pt           = load_planet_table(filename_pt, table=table)
            NbPlanet[im] = len(pt)
            planet_tables.append(pt)

    elif instru == "ANDES":
        modes         = [None, "LYOT"]
        planet_tables = []
        NbPlanet      = np.zeros(len(modes), dtype=int)
        for im, mode in enumerate(modes):
            filename_pt  = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=mode, systematics=systematics, PCA=PCA, name_model=name_model)
            pt           = load_planet_table(filename_pt, table=table)
            NbPlanet[im] = len(pt)
            planet_tables.append(pt)

    else:
        filename_pt   = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=name_model)
        pt            = load_planet_table(filename_pt, table=table)
        planet_tables = [pt]
        NbPlanet      = np.array([len(pt)])
        modes         = [""]

    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    yields        = np.zeros((NbBand, len(modes), len(exposure_time)))
    for ib, band in enumerate(bands):
        for im in range(len(modes)):
            for it in range(len(exposure_time)):
                SNR = get_SNR_from_table(planet_table=planet_tables[im], exposure_time=exposure_time[it], band=band)
                if fraction :
                    yields[ib, im, it] = len(planet_tables[im][SNR > SNR_thresh]) * 100/NbPlanet[im]
                else:
                    yields[ib, im, it] = len(planet_tables[im][SNR > SNR_thresh])

    plt.figure(dpi=dpi, figsize=(14, 8))
    for ib, band in enumerate(bands):
        for im in range(len(modes)):
            if not (yields[ib, im]==0).all():
                if im == 0:
                    plt.plot(exposure_time, yields[ib, im], color=cmap(ib), label=band.replace("_", " "), ls=ls_modes[im], lw=3)
                else:
                    plt.plot(exposure_time, yields[ib, im], color=cmap(ib), ls=ls_modes[im], lw=3)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure time per target [mn]', fontsize=16)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=16)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=16)
        plt.yscale('log')
    plt.xlim(exposure_time[0], exposure_time[-1])
    if config_data["base"] == "space":
        plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above", fontsize=18, weight='bold')
    else:
        plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above for {strehl} strehl", fontsize=18, weight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=16, loc="upper left", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Bands", title_fontsize=18)
    if len(modes) > 1:
        ax = plt.gca()
        ax_legend = ax.twinx()
        for im, mode in enumerate(modes):
            ax_legend.plot([], [], c="k", ls=ls_modes[im], label=str(mode).replace("_", " ").replace("None", "w/o coronagraph"), lw=3)
        if instru=="HARMONI":
            ax_legend.legend(fontsize=16, loc="upper center", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Apodizers", title_fontsize=18)
        elif instru=="ANDES":
            ax_legend.legend(fontsize=16, loc="upper center", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Coronagraphs", title_fontsize=18)
        ax_legend.tick_params(axis='y', colors='w') # Masking ticks
    plt.tight_layout()
    plt.show(block=False)



####################
# Yield histograms #
####################

def yield_hist_instrus_ptypes(exposure_time=10*60, thermal_model="auto", reflected_model="auto", planet_types=planet_types_reduced, fraction=False, dpi=dpi_fig):

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    planet_table_harmoni          = load_planet_table(f"Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_harmoni_sp       = load_planet_table(f"Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_andes            = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_without_systematics_{name_model}.ecsv")
    planet_table_andes_coro       = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_{name_model}.ecsv")
    planet_table_eris             = load_planet_table(f"Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_{name_model}.ecsv")
    planet_table_mirimrs_non_syst = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst     = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nircam           = load_planet_table(f"Archive_Pull_NIRCam_NO_SP_NO_JQ_MASK335R_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_non_syst = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst     = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    #planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")

    planet_table_harmoni["SNR"]          = np.fmax(get_SNR_from_table(planet_table=planet_table_harmoni,  exposure_time=exposure_time, band="INSTRU"), get_SNR_from_table(planet_table=planet_table_harmoni_sp, exposure_time=exposure_time, band="INSTRU"))
    planet_table_andes["SNR" ]           = np.fmax(get_SNR_from_table(planet_table=planet_table_andes,    exposure_time=exposure_time, band="INSTRU"), get_SNR_from_table(planet_table=planet_table_andes_coro, exposure_time=exposure_time, band="INSTRU"))
    planet_table_eris["SNR"]             = get_SNR_from_table(planet_table=planet_table_eris,             exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_non_syst["SNR"] = get_SNR_from_table(planet_table=planet_table_mirimrs_non_syst, exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_syst["SNR"]     = get_SNR_from_table(planet_table=planet_table_mirimrs_syst,     exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_syst_pca["SNR"] = get_SNR_from_table(planet_table=planet_table_mirimrs_syst_pca, exposure_time=exposure_time, band="INSTRU")
    planet_table_nircam["SNR"]           = get_SNR_from_table(planet_table=planet_table_nircam,           exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_non_syst["SNR"] = get_SNR_from_table(planet_table=planet_table_nirspec_non_syst, exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_syst["SNR"]     = get_SNR_from_table(planet_table=planet_table_nirspec_syst,     exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_syst_pca["SNR"] = get_SNR_from_table(planet_table=planet_table_nirspec_syst_pca, exposure_time=exposure_time, band="INSTRU")

    # Dictionnaries giving the list of planets for mp[type]
    mp_harmoni          = build_match_dict(planet_table=planet_table_harmoni,          planet_types=planet_types)
    mp_andes            = build_match_dict(planet_table=planet_table_andes,            planet_types=planet_types)
    mp_eris             = build_match_dict(planet_table=planet_table_eris,             planet_types=planet_types)
    mp_mirimrs_non_syst = build_match_dict(planet_table=planet_table_mirimrs_non_syst, planet_types=planet_types)
    mp_mirimrs_syst     = build_match_dict(planet_table=planet_table_mirimrs_syst,     planet_types=planet_types)
    mp_mirimrs_syst_pca = build_match_dict(planet_table=planet_table_mirimrs_syst_pca, planet_types=planet_types)
    mp_nircam           = build_match_dict(planet_table=planet_table_nircam,           planet_types=planet_types)
    mp_nirspec_non_syst = build_match_dict(planet_table=planet_table_nirspec_non_syst, planet_types=planet_types)
    mp_nirspec_syst     = build_match_dict(planet_table=planet_table_nirspec_syst,     planet_types=planet_types)
    mp_nirspec_syst_pca = build_match_dict(planet_table=planet_table_nirspec_syst_pca, planet_types=planet_types)

    planet_types_array     = np.array(list(planet_types.keys()))
    yield_harmoni          = np.zeros(len(planet_types_array))
    yield_andes            = np.zeros(len(planet_types_array))
    yield_eris             = np.zeros(len(planet_types_array))
    yield_mirimrs_non_syst = np.zeros(len(planet_types_array))
    yield_mirimrs_syst     = np.zeros(len(planet_types_array))
    yield_mirimrs_syst_pca = np.zeros(len(planet_types_array))
    yield_nircam           = np.zeros(len(planet_types_array))
    yield_nirspec_non_syst = np.zeros(len(planet_types_array))
    yield_nirspec_syst     = np.zeros(len(planet_types_array))
    yield_nirspec_syst_pca = np.zeros(len(planet_types_array))
    N_harmoni              = np.zeros(len(planet_types_array))
    N_andes                = np.zeros(len(planet_types_array))
    N_eris                 = np.zeros(len(planet_types_array))
    N_mirimrs              = np.zeros(len(planet_types_array))
    N_nircam               = np.zeros(len(planet_types_array))
    N_nirspec              = np.zeros(len(planet_types_array))

    for i in range(len(planet_types_array)):
        ptype = planet_types_array[i]

        N_harmoni[i] = len(mp_harmoni[ptype])
        N_andes[i]   = len(mp_andes[ptype])
        N_eris[i]    = len(mp_eris[ptype])
        N_mirimrs[i] = len(mp_mirimrs_non_syst[ptype])
        N_nircam[i]  = len(mp_nircam[ptype])
        N_nirspec[i] = len(mp_nirspec_non_syst[ptype])

        if fraction:
            ratio = 100
            norm_harmoni = N_harmoni[i]
            norm_andes   = N_andes[i]
            norm_eris    = N_eris[i]
            norm_mirimrs = N_mirimrs[i]
            norm_nircam  = N_nircam[i]
            norm_nirspec = N_nirspec[i]
        else:
            ratio = norm_harmoni = norm_andes = norm_eris = norm_mirimrs = norm_nircam = norm_nirspec = 1

        yield_harmoni[i]          = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_harmoni[ptype])          / norm_harmoni if norm_harmoni > 0 else 0
        yield_andes[i]            = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_andes[ptype])            / norm_andes   if norm_andes > 0   else 0
        yield_eris[i]             = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_eris[ptype])             / norm_eris    if norm_eris > 0    else 0
        yield_mirimrs_non_syst[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_mirimrs_non_syst[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst[i]     = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_mirimrs_syst[ptype])     / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst_pca[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_mirimrs_syst_pca[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_nircam[i]           = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nircam[ptype])           / norm_nircam  if norm_nircam > 0  else 0
        yield_nirspec_non_syst[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nirspec_non_syst[ptype]) / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst[i]     = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nirspec_syst[ptype])     / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst_pca[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nirspec_syst_pca[ptype]) / norm_nirspec if norm_nirspec > 0 else 0

    N_table   = 10
    bar_width = 1 / (N_table+2)
    linewidth = 1
    indices   = np.arange(len(planet_types_array))

    plt.figure(figsize=(14, 8), dpi=dpi)

    for i in range(len(planet_types_array)):
        if i % 2 == 0:  # shade every other type
            plt.axvspan(i - 0.5, i + 0.5, facecolor="gray", alpha=0.1, zorder=-10)
    for x in np.arange(0.5, len(planet_types_array), 1.0):
        plt.axvline(x, color="gray", lw=linewidth, ls=":", zorder=-10)  # dashed, light grey

    if not fraction:
        plt.bar(indices - 4.5*bar_width, N_harmoni, bar_width, edgecolor=colors_instru["HARMONI"], color=faded(colors_instru["HARMONI"]), linewidth=linewidth, zorder=1)
        plt.bar(indices - 3.5*bar_width, N_andes,   bar_width, edgecolor=colors_instru["ANDES"],   color=faded(colors_instru["ANDES"]),   linewidth=linewidth, zorder=1)
        plt.bar(indices - 2.5*bar_width, N_eris,    bar_width, edgecolor=colors_instru["ERIS"],    color=faded(colors_instru["ERIS"]),    linewidth=linewidth, zorder=1)
        plt.bar(indices - 1.5*bar_width, N_mirimrs, bar_width, edgecolor=colors_instru["MIRIMRS"], color=faded(colors_instru["MIRIMRS"]), linewidth=linewidth, zorder=1)
        plt.bar(indices - 0.5*bar_width, N_mirimrs, bar_width, edgecolor=colors_instru["MIRIMRS"], color=faded(colors_instru["MIRIMRS"]), linewidth=linewidth, zorder=1)
        plt.bar(indices + 0.5*bar_width, N_mirimrs, bar_width, edgecolor=colors_instru["MIRIMRS"], color=faded(colors_instru["MIRIMRS"]), linewidth=linewidth, zorder=1)
        plt.bar(indices + 1.5*bar_width, N_nircam,  bar_width, edgecolor=colors_instru["NIRCam"],  color=faded(colors_instru["NIRCam"]),  linewidth=linewidth, zorder=1)
        plt.bar(indices + 2.5*bar_width, N_nirspec, bar_width, edgecolor=colors_instru["NIRSpec"], color=faded(colors_instru["NIRSpec"]), linewidth=linewidth, zorder=1)
        plt.bar(indices + 3.5*bar_width, N_nirspec, bar_width, edgecolor=colors_instru["NIRSpec"], color=faded(colors_instru["NIRSpec"]), linewidth=linewidth, zorder=1)
        plt.bar(indices + 4.5*bar_width, N_nirspec, bar_width, edgecolor=colors_instru["NIRSpec"], color=faded(colors_instru["NIRSpec"]), linewidth=linewidth, zorder=1)

    plt.bar(indices - 4.5*bar_width, yield_harmoni,          bar_width, edgecolor="black", color=colors_instru["HARMONI"], linewidth=linewidth, label="ELT/HARMONI")
    plt.bar(indices - 3.5*bar_width, yield_andes,            bar_width, edgecolor="black", color=colors_instru["ANDES"],   linewidth=linewidth, label="ELT/ANDES")
    plt.bar(indices - 1.5*bar_width, yield_mirimrs_non_syst, bar_width, edgecolor="black", color=colors_instru["MIRIMRS"], linewidth=linewidth, label="JWST/MIRI/MRS")
    plt.bar(indices - 0.5*bar_width, yield_mirimrs_syst,     bar_width, edgecolor="black", color=colors_instru["MIRIMRS"], linewidth=linewidth, label="JWST/MIRI/MRS (w/ syst)",        hatch='//')
    plt.bar(indices + 0.5*bar_width, yield_mirimrs_syst_pca, bar_width, edgecolor="black", color=colors_instru["MIRIMRS"], linewidth=linewidth, label="JWST/MIRI/MRS (w/ syst+PCA)",    hatch='xx')
    plt.bar(indices - 2.5*bar_width, yield_eris,             bar_width, edgecolor="black", color=colors_instru["ERIS"],    linewidth=linewidth, label="VLT/ERIS/IFU")
    plt.bar(indices + 1.5*bar_width, yield_nircam,           bar_width, edgecolor="black", color=colors_instru["NIRCam"],  linewidth=linewidth, label="JWST/NIRCam")
    plt.bar(indices + 2.5*bar_width, yield_nirspec_non_syst, bar_width, edgecolor="black", color=colors_instru["NIRSpec"], linewidth=linewidth, label="JWST/NIRSpec/IFU")
    plt.bar(indices + 3.5*bar_width, yield_nirspec_syst,     bar_width, edgecolor="black", color=colors_instru["NIRSpec"], linewidth=linewidth, label="JWST/NIRSpec/IFU (w/ syst)",     hatch='//')
    plt.bar(indices + 4.5*bar_width, yield_nirspec_syst_pca, bar_width, edgecolor="black", color=colors_instru["NIRSpec"], linewidth=linewidth, label="JWST/NIRSpec/IFU (w/ syst+PCA)", hatch='xx')

    plt.xticks(indices, planet_types_array, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=18)
    plt.xlim(-0.5, len(planet_types_array) - 0.5)
    plt.ylim(0.5, 2_000)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=18)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=18)
        plt.yscale('log')
    plt.title(f'Known exoplanets detection yield for {int(round(exposure_time/60))} h per target', fontsize=20, pad=20, fontweight='bold')
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)
    plt.legend(title="Instruments", title_fontsize=16, fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke", ncol=2)
    plt.tight_layout()
    plt.show(block=False)



def yield_hist_instrus_ptypes_ELT(exposure_time=10*60, thermal_model="auto", reflected_model="auto", planet_types=planet_types, fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox", "ANDES", "ANDES+LYOT"], dpi=dpi_fig):

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    planet_table_harmoni      = load_planet_table("Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_harmoni_prox = load_planet_table("Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_andes        = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_without_systematics_"+name_model+".ecsv")
    planet_table_andes_lyot   = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_"+name_model+".ecsv")

    planet_table_harmoni["SNR"]      = get_SNR_from_table(planet_table=planet_table_harmoni,      exposure_time=exposure_time, band="INSTRU")
    planet_table_harmoni_prox["SNR"] = get_SNR_from_table(planet_table=planet_table_harmoni_prox, exposure_time=exposure_time, band="INSTRU")
    planet_table_andes["SNR"]        = get_SNR_from_table(planet_table=planet_table_andes,        exposure_time=exposure_time, band="INSTRU")
    planet_table_andes_lyot["SNR"]   = get_SNR_from_table(planet_table=planet_table_andes_lyot,   exposure_time=exposure_time, band="INSTRU")

    mp_harmoni      = build_match_dict(planet_table=planet_table_harmoni,      planet_types=planet_types)
    mp_harmoni_prox = build_match_dict(planet_table=planet_table_harmoni_prox, planet_types=planet_types)
    mp_andes        = build_match_dict(planet_table=planet_table_andes,        planet_types=planet_types)
    mp_andes_lyot   = build_match_dict(planet_table=planet_table_andes_lyot,   planet_types=planet_types)

    planet_types_array = np.array(list(planet_types.keys()))
    yield_harmoni      = np.zeros(len(planet_types_array))
    yield_harmoni_prox = np.zeros(len(planet_types_array))
    yield_andes        = np.zeros(len(planet_types_array))
    yield_andes_lyot   = np.zeros(len(planet_types_array))
    N_harmoni          = np.zeros(len(planet_types_array))
    N_harmoni_prox     = np.zeros(len(planet_types_array))
    N_andes            = np.zeros(len(planet_types_array))
    N_andes_lyot       = np.zeros(len(planet_types_array))

    for i in range(len(planet_types_array)):
        ptype = planet_types_array[i]

        N_harmoni[i]      = len(mp_harmoni[ptype])
        N_harmoni_prox[i] = len(mp_harmoni_prox[ptype])
        N_andes[i]        = len(mp_andes[ptype])
        N_andes_lyot[i]   = len(mp_andes_lyot[ptype])

        if fraction:
            ratio = 100
            norm_harmoni = N_harmoni[i]
            norm_andes   = N_andes[i]
        else:
            ratio = 1
            norm_harmoni = norm_harmoni_prox = norm_andes = norm_andes_lyot = 1

        yield_harmoni[i]      = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_harmoni[ptype])      / norm_harmoni      if norm_harmoni > 0      else 0
        yield_harmoni_prox[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_harmoni_prox[ptype]) / norm_harmoni_prox if norm_harmoni_prox > 0 else 0
        yield_andes[i]        = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_andes[ptype])        / norm_andes        if norm_andes > 0        else 0
        yield_andes_lyot[i]   = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_andes_lyot[ptype])   / norm_andes_lyot   if norm_andes_lyot > 0   else 0

    N_table   = len(instrus)
    bar_width = 1 / (N_table+2)
    linewidth = 1
    indices   = np.arange(len(planet_types_array))

    plt.figure(figsize=(14, 8), dpi=dpi)

    # Shade for every type
    for i in range(len(planet_types_array)):
        if i % 2 == 0:
            plt.axvspan(i - 0.5, i + 0.5, facecolor="gray", alpha=0.1, zorder=-10)
    for x in np.arange(0.5, len(planet_types_array), 1.0):
        plt.axvline(x, color="gray", lw=linewidth, ls=":", zorder=-10)  # dashed, light grey

    # Shade for every temperature type
    tags     = [s.split(' ', 1)[0] for s in planet_types_array]  # "Cold Sub-Earth" -> "Cold"
    segments = []
    start    = 0
    for i in range(1, len(tags)+1):
        if i == len(tags) or tags[i] != tags[i-1]:
            segments.append((tags[start], start, i-1))  # (tag, i0, i1)
            start = i
    band = {'Cold': '#E6F0FF',  'Warm': '#FFF1D6',  'Hot': '#FFE2E2'}   # bleu / orange / rouge pâles
    sep  = {'Cold': '#B7D3FF',  'Warm': '#FFD89A',  'Hot': '#FFB4B4'}   # ton un peu plus soutenu
    for tag, i0, i1 in segments:
        plt.axvspan(i0 - 0.5, i1 + 0.5, facecolor=band.get(tag, '#EEEEEE'), alpha=0.3, zorder=-20)
        plt.axvline(i1 + 0.5, color=sep.get(tag, '0.8'), lw=linewidth, ls=(0, (4, 3)), alpha=0.9, zorder=-15)
        xmid = 0.5 * (i0 + i1)
        plt.text(xmid, 1.01, tag.upper(), ha='center', va='bottom', transform=plt.gca().get_xaxis_transform(), fontsize=18, color=sep.get(tag, '0.4'), fontweight='bold')
    plt.axvline(segments[0][1] - 0.5, color=sep.get(segments[0][0], '0.8'), lw=linewidth, ls=(0, (4, 3)), alpha=0.9, zorder=-15)

    for i, instru in enumerate(instrus):
        idx = i - N_table//2 + 0.5

        if "harmoni" in instru.lower() and "prox" not in instru.lower():
            plt.bar(indices + idx*bar_width, yield_harmoni, bar_width, zorder=10, edgecolor="black", color=colors_instru["HARMONI"], linewidth=linewidth, label="ELT/HARMONI (w/o apodizer)")
            if not fraction:
                plt.bar(indices + idx*bar_width, N_harmoni, bar_width, edgecolor=colors_instru["HARMONI"], color=faded(colors_instru["HARMONI"]), linewidth=linewidth, zorder=1)

        if "harmoni" in instru.lower() and "prox" in instru.lower():
            plt.bar(indices + idx*bar_width, yield_harmoni_prox, bar_width, zorder=10, edgecolor="black", color=colors_instru["HARMONI"], linewidth=linewidth, label="ELT/HARMONI (w/ SP Prox)", hatch='//')
            if not fraction:
                plt.bar(indices + idx*bar_width, N_harmoni_prox, bar_width, edgecolor=colors_instru["HARMONI"], color=faded(colors_instru["HARMONI"]), linewidth=linewidth, zorder=1)

        if "andes" in instru.lower() and "lyot" not in instru.lower():
            plt.bar(indices + idx*bar_width, yield_andes,        bar_width, zorder=10, edgecolor="black", color=colors_instru["ANDES"],   linewidth=linewidth, label="ELT/ANDES (w/o coronagraph)")
            if not fraction:
                plt.bar(indices + idx*bar_width, N_andes,        bar_width, edgecolor=colors_instru["ANDES"],   color=faded(colors_instru["ANDES"]),   linewidth=linewidth, zorder=1)

        if "andes" in instru.lower() and "lyot" in instru.lower():
            plt.bar(indices + idx*bar_width, yield_andes_lyot,   bar_width, zorder=10, edgecolor="black", color=colors_instru["ANDES"],   linewidth=linewidth, label="ELT/ANDES (w/ Lyot)", hatch='//')
            if not fraction:
                plt.bar(indices + idx*bar_width, N_andes_lyot,   bar_width, edgecolor=colors_instru["ANDES"],   color=faded(colors_instru["ANDES"]),   linewidth=linewidth, zorder=1)

    plt.xticks(indices, planet_types_array, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=16)
    plt.xlim(-0.5, len(planet_types_array) - 0.5)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=18, labelpad=18)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=18, labelpad=18)
        plt.yscale('log')
    plt.title(f'Known exoplanets detection yield for {exposure_time/60:.0f} h per target', fontsize=20, fontweight='bold', pad=36)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)
    plt.legend(title="Instruments", title_fontsize=16, fontsize=14, loc="upper center", frameon=True, edgecolor="gray", facecolor="whitesmoke", ncol=2)
    plt.tight_layout()
    plt.show(block=False)



#################
# Yield corners #
#################

def yield_corner_instru(instru="HARMONI", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", coronagraph=None, band="INSTRU", systematics=False, PCA=False):
    smooth_corner = 1
    ndim          = 6 # Mp, Rp, Tp, a, d, sep

    # WORKING ANGLE
    iwa, owa = get_wa(instru=instru, sep_unit="mas")

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeff"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]

    # SETTING DATA
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeff"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))

    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data          = data_raw,
        bins          = 20,
        labels        = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles     = [0.16, 0.5, 0.84],   # below -+1 sigma
        levels        = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles   = True,
        title_kwargs  = {"fontsize": 14, "pad": 10},
        top_ticks     = False,
        plot_density  = True,
        plot_contours = True,
        fill_contours = True,
        smooth        = smooth_corner,
        smooth1d      = smooth_corner,
        color          = "gray",
        contour_kwargs = {"colors": ["black"], "alpha": 0.85, "linewidths": 1.3},
        hist_kwargs    = {"color": "black", "alpha": 0.85, "linewidth": 1.3},
        label_kwargs  = {"fontsize": 16})

    # DETECTIONS TABLE
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv")
    planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeff"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
    planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
    SNR          = get_SNR_from_table(planet_table=planet_table, exposure_time=exposure_time, band=band)
    planet_table = planet_table[SNR > SNR_thresh]
    data       = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeff"].value)
    data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
    data[:, 4] = np.log10(np.array(planet_table["Distance"].value))
    data[:, 5] = np.log10(np.array(planet_table["AngSep"].value))
    corner.corner(
        data           = data,
        fig            = figure,
        bins           = 20,
        quantiles      = [0.5], # below -+1 sigma
        levels         = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles    = False,
        top_ticks      = False,
        plot_density   = True,
        plot_contours  = True,
        fill_contours  = True,
        color          ='crimson',
        contour_kwargs = {"colors": ["crimson"], "alpha": 0.85, "linewidths": 1.3},
        hist_kwargs    = {"color": "crimson", "alpha": 0.85, "linewidth": 1.3},
        smooth         = smooth_corner,
        smooth1d       = smooth_corner)

    figure.suptitle(f"{instru} re-detections statistics with {len(planet_table)} / {len(planet_table_raw)} detections between {round(iwa)} and {round(owa)} mas\nfor {round(exposure_time/60)} hours per target (with {spectrum_contributions} light with {name_model})", fontsize=18, y=1.05, fontweight="bold")
    plt.gcf().set_dpi(300)
    plt.show(block=False)



def yield_corner_instrus(instru1="HARMONI", instru2="ANDES", band1="INSTRU", band2="INSTRU", apodizer1="NO_SP", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", coronagraph1=None, coronagraph2=None, exposure_time=6*60, thermal_model="auto", reflected_model="auto", systematics=False, PCA=False, all_configs_if_different=True):

    instrus       = [instru1,      instru2]
    bands         = [band1,        band2]
    apodizers     = [apodizer1,    apodizer2]
    strehls       = [strehl1,      strehl2]
    coronagraphs  = [coronagraph1, coronagraph2]
    colors_instru = ["crimson", "royalblue"]
    smooth_corner = 1
    ndim          = 6 # Mp, Rp, Tp, a, d, sep
    alpha_earth   = 0.5
    color_earth   = "seagreen"
    earth_values  = np.array([np.log10(1), np.log10(1), 300, np.log10(1), None, None], dtype=object)
    all_configs_mode = all_configs_if_different and (instru1 != instru2)

    fontsize_tick     = 15
    fontsize_label    = 20
    fontsize_title    = 18
    fontsize_legend   = 18
    fontsize_suptitle = 22

    PlanetMassMin   = None  # [M_earth]
    PlanetMassMax   = None  # [M_earth]
    PlanetRadiusMin = None  # [R_earth]
    PlanetRadiusMax = None  # [R_earth]
    PlanetTeffMin   = None   # [K]
    PlanetTeffMax   = 3000  # [K]
    SMAMin          = None  # [AU]
    SMAMax          = None  # [AU]
    DistanceMin     = None  # [pc]
    DistanceMax     = None  # [pc]
    AngSepMin       = None  # [mas]
    AngSepMax       = None  # [mas]

    parameter_bounds = {
        "PlanetMass":   (PlanetMassMin,   PlanetMassMax),
        "PlanetRadius": (PlanetRadiusMin, PlanetRadiusMax),
        "PlanetTeff":   (PlanetTeffMin,   PlanetTeffMax),
        "SMA":          (SMAMin,          SMAMax),
        "Distance":     (DistanceMin,     DistanceMax),
        "AngSep":       (AngSepMin,       AngSepMax),
    }

    # Small helpers
    def get_col_array(planet_table, colname):
        col = planet_table[colname]
        return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)

    def get_bounds(bounds):
        if np.isscalar(bounds):
            return None, bounds
        return bounds

    def get_valid_bounds_mask(planet_table):
        cols = ["PlanetMass", "PlanetRadius", "PlanetTeff", "SMA", "Distance", "AngSep"]
        positive_cols = ["PlanetMass", "PlanetRadius", "PlanetTeff", "SMA", "Distance", "AngSep"]
        mask = np.ones(len(planet_table), dtype=bool)
        for col in cols:
            x = get_col_array(planet_table, col)
            mask &= ~get_invalid_mask(planet_table[col])
            mask &= np.isfinite(x)
            if col in positive_cols:
                mask &= x > 0
        for col, bounds in parameter_bounds.items():
            vmin, vmax = get_bounds(bounds)
            x = get_col_array(planet_table, col)
            if vmin is not None:
                mask &= x >= vmin
            if vmax is not None:
                mask &= x <= vmax
        return mask

    def table_to_corner_data(planet_table):
        data = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(get_col_array(planet_table, "PlanetMass"))
        data[:, 1] = np.log10(get_col_array(planet_table, "PlanetRadius"))
        data[:, 2] = get_col_array(planet_table, "PlanetTeff")
        data[:, 3] = np.log10(get_col_array(planet_table, "SMA"))
        data[:, 4] = np.log10(get_col_array(planet_table, "Distance"))
        data[:, 5] = np.log10(get_col_array(planet_table, "AngSep"))
        return data

    def get_corner_ranges(data):
        ranges = []
        for idim in range(data.shape[1]):
            xmin, xmax = np.nanmin(data[:, idim]), np.nanmax(data[:, idim])
            if not np.isfinite(xmin) or not np.isfinite(xmax):
                xmin, xmax = 0.0, 1.0
            if xmin == xmax:
                dx = 0.5 if xmin == 0 else 0.05 * abs(xmin)
                xmin, xmax = xmin - dx, xmax + dx
            ranges.append((xmin, xmax))
        return ranges

    def get_table_filename(instru, apodizer, strehl, coronagraph):
        coronagraph_str = "_" + str(coronagraph) if coronagraph is not None else ""
        suffix = "with_systematics+PCA" if (systematics and PCA) else ("with_systematics" if systematics else "without_systematics")
        return f"Archive_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"

    def get_detected_table_fixed(instru, band, apodizer, strehl, coronagraph):
        filename = get_table_filename(instru, apodizer, strehl, coronagraph)
        planet_table = load_planet_table(filename)
        SNR = get_SNR_from_table(planet_table=planet_table, exposure_time=exposure_time, band=band)
        mask = get_valid_bounds_mask(planet_table) & np.isfinite(SNR) & (SNR > SNR_thresh)
        return planet_table[mask]

    def get_detected_table_all_configs(instru):
        config_data = get_config_data(instru)
        detected_names = set()
        N_loaded = 0
        bands_all = list(config_data["gratings"].keys())
        strehls_all = list(config_data["strehls"])
        apodizers_all = list(config_data["apodizers"].keys())
        coronagraphs_all = list(config_data["coronagraphs"])
        for strehl in strehls_all:
            for apodizer in apodizers_all:
                for coronagraph in coronagraphs_all:
                    filename = get_table_filename(instru, apodizer, strehl, coronagraph)
                    try:
                        planet_table = load_planet_table(filename)
                    except Exception:
                        continue
                    N_loaded += 1
                    names = np.asarray(planet_table["PlanetName"], dtype=str)
                    mask_valid = get_valid_bounds_mask(planet_table)

                    for band in bands_all:
                        required_cols = [f"DIT_{band}", f"signal_{band}", f"sigma_fund_{band}", f"sigma_syst_{band}"]
                        if np.any([col not in planet_table.colnames for col in required_cols]):
                            continue

                        SNR = get_SNR_from_table(planet_table=planet_table, exposure_time=exposure_time, band=band)
                        detected_names.update(names[mask_valid & np.isfinite(SNR) & (SNR > SNR_thresh)])

        if N_loaded == 0:
            raise FileNotFoundError(f"No table found for {instru} in all_configs_if_different mode.")

        raw_names = np.asarray(planet_table_raw["PlanetName"], dtype=str)
        return planet_table_raw[np.isin(raw_names, list(detected_names))]

    def get_instru_label(instru, apodizer, coronagraph):
        if all_configs_mode:
            return f"{instru}"
        if instru == "HARMONI":
            return "HARMONI w/o apodizer" if apodizer == "NO_SP" else f"HARMONI w/ {apodizer.replace('_', ' ')}"
        if instru == "ANDES":
            return "ANDES w/o coronagraph" if coronagraph is None else f"ANDES w/ {coronagraph}"
        return instru

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table_raw = planet_table_raw[get_valid_bounds_mask(planet_table_raw)]

    if len(planet_table_raw) == 0:
        raise ValueError("No planet remains after applying validity and parameter_bounds cuts.")

    data_raw = table_to_corner_data(planet_table_raw)
    corner_ranges = get_corner_ranges(data_raw)

    labels = [
        r"$\log_{10}(M_p/M_{\oplus})$",
        r"$\log_{10}(R_p/R_{\oplus})$",
        r"$T_p \, [\mathrm{K}]$",
        r"$\log_{10}(a/\mathrm{AU})$",
        r"$\log_{10}(d/\mathrm{pc})$",
        r"$\log_{10}(\mathrm{sep}/\mathrm{mas})$",
    ]

    figure = corner.corner(
        data=data_raw,
        bins=20,
        labels=labels,
        range=corner_ranges,
        quantiles=[0.5],
        levels=[0.68, 0.95, 0.997],
        show_titles=False,
        title_kwargs={"fontsize": fontsize_title, "pad": 10},
        top_ticks=False,
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        smooth=smooth_corner,
        smooth1d=smooth_corner,
        color="gray",
        contour_kwargs={"colors": ["black"], "alpha": 0.85, "linewidths": 1.4},
        hist_kwargs={"color": "black", "alpha": 0.85, "linewidth": 1.4},
        label_kwargs={"fontsize": fontsize_label},
    )
    axes           = np.array(figure.axes).reshape((ndim, ndim))
    diag_ylims_raw = [axes[i, i].get_ylim() for i in range(ndim)]

    # DETECTIONS TABLES
    yields = np.zeros((len(instrus)))
    for ni, instru in enumerate(instrus):
        band = bands[ni]
        apodizer = apodizers[ni]
        strehl = strehls[ni]
        coronagraph = coronagraphs[ni]

        if all_configs_mode:
            planet_table = get_detected_table_all_configs(instru)
        else:
            planet_table = get_detected_table_fixed(instru, band, apodizer, strehl, coronagraph)

        yields[ni] = len(planet_table)

        if len(planet_table) == 0:
            continue

        data = table_to_corner_data(planet_table)

        corner.corner(
            data=data,
            fig=figure,
            bins=20,
            range=corner_ranges,
            quantiles=[0.5],
            levels=[0.68, 0.95, 0.997],
            show_titles=False,
            top_ticks=False,
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            color=colors_instru[ni],
            contour_kwargs={"colors": [colors_instru[ni]], "alpha": 0.85, "linewidths": 1.5},
            hist_kwargs={"color": colors_instru[ni], "alpha": 0.85, "linewidth": 1.5},
            smooth=smooth_corner,
            smooth1d=smooth_corner,
        )

    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(ndim):
        axes[i, i].set_xlim(corner_ranges[i])
        axes[i, i].set_ylim(diag_ylims_raw[i])

    # Earth points
    for i in range(ndim):
        if earth_values[i] is not None:
            ax = axes[i, i]
            ax.axvline(earth_values[i], color=color_earth, alpha=alpha_earth, lw=3)
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if earth_values[xi] is not None:
                ax.axvline(earth_values[xi], color=color_earth, alpha=alpha_earth, lw=3)
            if earth_values[yi] is not None:
                ax.axhline(earth_values[yi], color=color_earth, alpha=alpha_earth, lw=3)
            if earth_values[xi] is not None and earth_values[yi] is not None:
                ax.plot(earth_values[xi], earth_values[yi], marker="s", c=color_earth, ms=10)

    # Font sizes on axes
    for ax in figure.axes:
        ax.tick_params(axis="both", which="major", labelsize=fontsize_tick)
        ax.xaxis.label.set_size(fontsize_label)
        ax.yaxis.label.set_size(fontsize_label)
        ax.title.set_size(fontsize_title)

    # Legends
    handles = []
    for ni, instru in enumerate(instrus):
        handles.append(mlines.Line2D([], [], color="w", label=""))
        instru_label = get_instru_label(instru, apodizers[ni], coronagraphs[ni])
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=colors_instru[ni], label=instru_label + f" ({round(yields[ni])} / {len(planet_table_raw)})"))
    fig = plt.gcf()
    fig.legend(handles=handles, frameon=True, bbox_to_anchor=(0.95, 0.7), loc="upper right", fontsize=fontsize_legend, title="Instruments:", title_fontsize=fontsize_legend+2)
    handles_earth = [mlines.Line2D([], [], linestyle="-", marker="s", color=color_earth, label="Earth")]
    fig.legend(handles=handles_earth, frameon=True, bbox_to_anchor=(0.82, 0.75), loc="upper right", fontsize=fontsize_legend)

    # Title
    if instru1 == instru2:
        title = f"ELT/{instru1} re-detections statistics\n\nwith {len(planet_table_raw)} known planets for {round(exposure_time/60)} hr per target"
    else:
        title = f"ELT/{instru1} VS ELT/{instru2} re-detections statistics\n\nwith {len(planet_table_raw)} known planets for {round(exposure_time/60)} hr per target"

    figure.suptitle(title, fontsize=fontsize_suptitle, fontweight="bold", x=0.63, y=0.93)

    plt.gcf().set_dpi(300)
    plt.show(block=False)





##################
# Yield contrast #
##################

def yield_plot_instrus_contrast(table="Archive", exposure_time=10*60, thermal_model="auto", reflected_model="auto", band_quantity="INSTRU", band_contrast="H", mode="best", percentile=None, nbins=18, min_per_bin=3, dpi=dpi_fig):
    """
    Plot rebinned 5-sigma contrast limits for several instruments.

    mode="best"   : minimum contrast in each separation bin.
    mode="median" : median contrast with the 16th-84th percentile interval.
    """

    if mode not in ("best", "median"):
        raise ValueError("mode must be 'best' or 'median'.")

    if band_contrast == "INSTRU":
        raise ValueError("For a multi-instrument comparison, band_contrast must be a common photometric band, e.g. 'H'.")

    # For PCS
    exposure_label = f"{exposure_time/60:g}".replace(".", "p")
    SNR_col        = f"SNR_INSTRU_{exposure_label}h"

    spectrum_contributions, name_model                 = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    spectrum_contributions_thermal, name_model_thermal = get_spectrum_contribution_name_model(thermal_model, "None")

    # Each configuration contains:
    # instru, apodizer, strehl, coronagraph, systematics, PCA, name_model, spectrum_contributions
    lw0 = 6
    curve_specs = [
        {
            "key": "PCS",
            "label": "ELT/PCS-like w/ Lyot",
            "color": colors_instru["PCS"],
            "ls": "-",
            "marker": "o",
            "mfc": "none",
            "mec": colors_instru["PCS"],
            "mew": lw0/2,
            "lw": lw0,
            "use_saved_SNR": True,
            "configs": [
                ("PCS", "NO_SP", "Q2", "LYOT", False, False, name_model, spectrum_contributions),
            ],
        },
        {
            "key": "HARMONI",
            "label": "ELT/HARMONI",
            "color": colors_instru["HARMONI"],
            "ls": "-",
            "lw": lw0,
            "configs": [
                ("HARMONI", "NO_SP", "JQ1", None, False, False, name_model, spectrum_contributions),
                #("HARMONI", "SP_Prox", "JQ1", None, False, False, name_model, spectrum_contributions),
            ],
        },
        {
            "key": "ANDES",
            "label": "ELT/ANDES w/o coronagraph",
            "color": colors_instru["ANDES"],
            "ls": "-",
            "lw": lw0,
            "configs": [
                ("ANDES", "NO_SP", "MED", None, False, False, name_model, spectrum_contributions),
            ],
        },
        {
            "key": "ANDES_LYOT",
            "label": "ELT/ANDES w/ Lyot",
            "color": colors_instru["ANDES"],
            "ls": "-",
            "marker": "o",
            "mfc": "none",
            "mec": colors_instru["ANDES"],
            "mew": lw0/2,
            "lw": lw0,
            "configs": [
                ("ANDES", "NO_SP", "MED", "LYOT", False, False, name_model, spectrum_contributions),
            ],
        },
        # {
        #     "key": "ERIS",
        #     "label": "VLT/ERIS/IFU",
        #     "color": colors_instru["ERIS"],
        #     "ls": "-",
        #     "lw": lw0,
        #     "configs": [
        #         ("ERIS", "NO_SP", "JQ0", None, False, False, name_model, spectrum_contributions),
        #     ],
        # },
        {
            "key": "MIRIMRS",
            "label": "JWST/MIRI/MRS",
            "color": colors_instru["MIRIMRS"],
            "ls": "-",
            "lw": lw0,
            "configs": [
                ("MIRIMRS", "NO_SP", "NO_JQ", None, False, False, name_model_thermal, spectrum_contributions_thermal),
            ],
        },
        {
            "key": "MIRIMRS_syst",
            "label": None,
            "color": colors_instru["MIRIMRS"],
            "ls": "--",
            "lw": lw0,
            "configs": [
                ("MIRIMRS", "NO_SP", "NO_JQ", None, True, False, name_model_thermal, spectrum_contributions_thermal),
            ],
        },
        {
            "key": "MIRIMRS_PCA",
            "label": None,
            "color": colors_instru["MIRIMRS"],
            "ls": ":",
            "lw": lw0,
            "configs": [
                ("MIRIMRS", "NO_SP", "NO_JQ", None, True, True, name_model_thermal, spectrum_contributions_thermal),
            ],
        },
        # {
        #     "key": "NIRCam",
        #     "label": "JWST/NIRCam",
        #     "color": colors_instru["NIRCam"],
        #     "ls": "-",
        #     "lw": lw0,
        #     "configs": [
        #         ("NIRCam", "NO_SP", "NO_JQ", "MASK335R", False, False, name_model, spectrum_contributions),
        #     ],
        # },
        {
            "key": "NIRSpec",
            "label": "JWST/NIRSpec/IFU",
            "color": colors_instru["NIRSpec"],
            "ls": "-",
            "lw": lw0,
            "configs": [
                ("NIRSpec", "NO_SP", "NO_JQ", None, False, False, name_model_thermal, spectrum_contributions_thermal),
            ],
        },
        {
            "key": "NIRSpec_syst",
            "label": None,
            "color": colors_instru["NIRSpec"],
            "ls": "--",
            "lw": lw0,
            "configs": [
                ("NIRSpec", "NO_SP", "NO_JQ", None, True, False, name_model_thermal, spectrum_contributions_thermal),
            ],
        },
        # {
        #     "key": "NIRSpec_PCA",
        #     "label": None,
        #     "color": colors_instru["NIRSpec"],
        #     "ls": ":",
        #     "lw": lw0,
        #     "configs": [
        #         ("NIRSpec", "NO_SP", "NO_JQ", None, True, True, name_model_thermal, spectrum_contributions_thermal),
        #     ],
        # },
    ]

    def load_config(config):
        instru, apodizer, strehl, coronagraph, systematics, PCA, model_name, _ = config
        filename = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=model_name)
        return load_planet_table(filename, table=table)

    def get_binned_curve(separation, contrast, bins, mode):
        xbin  = np.sqrt(bins[:-1]*bins[1:])
        y     = np.full(len(xbin), np.nan)
        ylow  = np.full(len(xbin), np.nan)
        yhigh = np.full(len(xbin), np.nan)
        for ibin in range(len(xbin)):
            mask = np.isfinite(separation) & np.isfinite(contrast) & (separation >= bins[ibin]) & (separation < bins[ibin+1]) & (separation > 0) & (contrast > 0)
            if np.sum(mask) < (1 if mode == "best" else min_per_bin):
                continue
            if mode == "best":
                y[ibin] = np.nanpercentile(contrast[mask], 10)
            elif mode == "median":
                y[ibin] = np.nanmedian(contrast[mask])
                y[ibin] = np.nanpercentile(contrast[mask], 50)
                ylow[ibin], yhigh[ibin] = np.nanpercentile(contrast[mask], [16, 84])
            elif mode == "percentile":
                y[ibin] = np.nanpercentile(contrast[mask], percentile)
        return xbin, y, ylow, yhigh

    def get_planet_contrast(planet_table, instru, contributions):
        if band_contrast == "INSTRU":
            star_mag_col   = f"StarINSTRUmag({instru})"
            planet_mag_col = f"PlanetINSTRUmag({instru})({contributions})"
        else:
            star_mag_col   = f"Star{band_contrast}mag"
            planet_mag_col = f"Planet{band_contrast}mag({contributions})"
        missing_cols = [col for col in [star_mag_col, planet_mag_col] if col not in planet_table.colnames]
        if missing_cols:
            raise KeyError(f"Missing magnitude columns: {missing_cols}")
        mag_star   = np.asarray(planet_table[star_mag_col], dtype=float)
        mag_planet = np.asarray(planet_table[planet_mag_col], dtype=float)

        return 10**(-0.4*(mag_planet - mag_star))

    # Load tables and compute one contrast value per planet
    curves            = {}
    planet_table_ref  = None
    instru_ref        = None
    contributions_ref = None

    for spec in curve_specs:
        configs = spec["configs"]
        tables  = [load_config(config) for config in configs]
        if spec.get("use_saved_SNR", False):
            contrasts = [
                get_contrast_from_saved_SNR(
                    planet_table=planet_table,
                    band_contrast=band_contrast,
                    instru=config[0],
                    spectrum_contributions="thermal+reflected",#config[7],
                    SNR_col=SNR_col,
                )
                for planet_table, config in zip(tables, configs)
            ]
        else:
            contrasts = [
                get_contrast_from_table(
                    planet_table=planet_table,
                    exposure_time=exposure_time,
                    band_quantity=band_quantity,
                    band_contrast=band_quantity,#band_contrast,
                    instru=config[0],
                    spectrum_contributions="thermal+reflected",#config[7],
                )
                for planet_table, config in zip(tables, configs)
            ]

        names_ref = np.asarray(tables[0]["PlanetName"], dtype=str)
        for planet_table in tables[1:]:
            names = np.asarray(planet_table["PlanetName"], dtype=str)
            if len(names) != len(names_ref) or np.any(names != names_ref):
                raise ValueError(f"Inconsistent planet ordering for {spec['key']}.")

        # For HARMONI: min(NO_SP, SP_Prox) planet by planet
        contrast = contrasts[0].copy()
        for contrast_config in contrasts[1:]:
            contrast = np.fmin(contrast, contrast_config)

        separation          = np.asarray(tables[0]["AngSep"].value if hasattr(tables[0]["AngSep"], "value") else tables[0]["AngSep"], dtype=float)
        curves[spec["key"]] = {"separation": separation, "contrast": contrast, **spec}

        if planet_table_ref is None:
            planet_table_ref  = tables[0]
            instru_ref        = configs[0][0]
            contributions_ref = configs[0][7]

    # Common logarithmic separation bins
    all_sep          = np.concatenate([curve["separation"][np.isfinite(curve["separation"]) & (curve["separation"] > 0)] for curve in curves.values()])
    sep_min, sep_max = np.nanmin(all_sep), np.nanmax(all_sep)
    bins             = np.logspace(np.log10(sep_min), np.log10(sep_max), nbins+1)

    # Figure
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=dpi)

    # All known planets
    separation_planets = np.asarray(planet_table_ref["AngSep"].value if hasattr(planet_table_ref["AngSep"], "value") else planet_table_ref["AngSep"], dtype=float)
    planet_contrast    = get_planet_contrast(planet_table_ref, instru_ref, contributions_ref)
    PlanetTeff         = np.asarray(planet_table_ref["PlanetTeff"].value if hasattr(planet_table_ref["PlanetTeff"], "value") else planet_table_ref["PlanetTeff"], dtype=float)
    discovery_method   = np.asarray(planet_table_ref["DiscoveryMethod"], dtype=str)

    cmap    = plt.get_cmap("plasma")
    norm    = mpl.colors.Normalize(vmin=200, vmax=3000, clip=True)
    methods = {
        "Transit":         ("v", "Transit"),
        "Radial Velocity": ("o", "Radial velocity"),
        "Imaging":         ("s", "Direct imaging"),
        "Other":           ("P", "Other"),
    }

    for method, (marker, label) in methods.items():
        method_mask = discovery_method == method if method != "Other" else ~np.isin(discovery_method, ["Transit", "Radial Velocity", "Imaging"])
        mask        = method_mask & np.isfinite(separation_planets) & np.isfinite(planet_contrast) & np.isfinite(PlanetTeff) & (separation_planets > 0) & (planet_contrast > 0) & (PlanetTeff > 0)

        ax.scatter(
            separation_planets[mask],
            planet_contrast[mask],
            c=PlanetTeff[mask],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=100,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.15,
            zorder=1,
        )

    # Rebinned contrast curves
    results = {}
    instrument_handles = []

    for key, curve in curves.items():
        xbin, y, ylow, yhigh = get_binned_curve(curve["separation"], curve["contrast"], bins, mode)
        line, = ax.plot(
                        xbin, y,
                        color=curve["color"],
                        ls=curve["ls"],
                        lw=curve["lw"],
                        marker=curve.get("marker", None),
                        ms=21,
                        markevery=2,
                        mfc=curve.get("mfc", curve["color"]),
                        mec=curve.get("mec", curve["color"]),
                        mew=curve.get("mew", 1.5),
                        label=curve["label"] if curve["label"] is not None else "_nolegend_",
                        zorder=5,
                    )

        # if mode == "median":
        #     ax.fill_between(xbin, ylow, yhigh, color=curve["color"], alpha=0.15, linewidth=0, zorder=4)

        if curve["label"] is not None:
            instrument_handles.append(line)

        results[key] = {"separation": xbin, "contrast": y, "contrast_low": ylow, "contrast_high": yhigh}

    # Discovery-method legend
    handles_methods = [Line2D([], [], ls="", marker=marker, ms=11, markerfacecolor="white", markeredgecolor="k", color="k", label=label) for marker, label in methods.values()]
    legend_methods = ax.legend(handles=handles_methods, loc="upper right", fontsize=16, frameon=True, edgecolor="gray", facecolor="white", framealpha=1.0)#, title="Discovery method", title_fontsize=16)
    ax.add_artist(legend_methods)

    # Instrument legend
    legend_instrus = ax.legend(handles=instrument_handles, loc="lower left", fontsize=16, frameon=True, edgecolor="gray", facecolor="white", framealpha=1.0)#, title="Instruments", title_fontsize=16)
    ax.add_artist(legend_instrus)

    # Configuration/systematics legend
    handles_systematics = [
        Line2D([], [], color="k", ls="-",  lw=3, label="w/o systematics"),
        Line2D([], [], color="k", ls="--", lw=3, label="w/ systematics"),
        Line2D([], [], color="k", ls=":",  lw=3, label="w/ systematics + PCA"),
    ]
    ax.legend(handles=handles_systematics, loc="upper center", fontsize=16, frameon=True, edgecolor="gray", facecolor="white", framealpha=1.0)#, title="Configuration", title_fontsize=16)

    # Planet-temperature colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Planet $T_\mathrm{eff}$ [K]", fontsize=19, rotation=270, labelpad=27)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e3)
    ax.set_ylim(1e-9, 1e-2)
    ax.set_xlabel("Angular separation [mas]", fontsize=21)
    ax.set_ylabel(r"5$\sigma$ contrast", fontsize=21)
    ax.set_title(rf"Exoplanet {mode} 5σ contrast limits in {exposure_time/60:.0f} hr with molecular mapping", fontsize=23, weight="bold", pad=18)
    ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=18)

    fig.tight_layout()
    plt.show(block=False)

    return results, fig, ax



def yield_contrast_instru(instru="ANDES", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", coronagraph=None, systematics=False, PCA=False, table="Archive", band="INSTRU", dpi=dpi_fig):
    config_data = get_config_data(instru)

    # WORKING ANGLE
    iwa, owa = get_wa(instru=instru)

    # Specs of the instru
    lmin = config_data["lambda_range"]["lambda_min"]
    lmax = config_data["lambda_range"]["lambda_max"]
    R = 0.
    N = len(config_data["gratings"])
    for b in config_data["gratings"]:
        R += config_data["gratings"][b].R/N # mean resolution

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    tablename = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"

    planet_table = load_planet_table(tablename, table=table)
    SNR          = get_SNR_from_table(planet_table=planet_table, exposure_time=exposure_time, band=band)
    planet_table = planet_table[SNR > SNR_thresh]
    SNR          = SNR[SNR > SNR_thresh]

    x = np.array(planet_table["AngSep"].value) # sep axis [mas]
    if config_data["sep_unit"]=="arcsec":
        x = x/1000
    mag_p = planet_table[f"PlanetINSTRUmag({instru})({spectrum_contributions})"]
    mag_s = planet_table[f"StarINSTRUmag({instru})"]
    y     = 10**(-(mag_p-mag_s)/2.5)
    z     = np.array(planet_table["PlanetTeff"].value) # color axis [K]

    im_mask = planet_table["DiscoveryMethod"]=="Imaging"
    tr_mask = planet_table["DiscoveryMethod"]=="Transit"
    rv_mask = planet_table["DiscoveryMethod"]=="Radial Velocity"
    ot_mask = (planet_table["DiscoveryMethod"]!="Imaging") & (planet_table["DiscoveryMethod"]!="Radial Velocity") & (planet_table["DiscoveryMethod"]!="Transit")
    
    # Taille des points (capés) en fonction du SNR
    s = get_size_from_SNR(SNR=SNR, s0=50, ds=200, SNR_min=SNR_thresh, SNR_max=1_000)

    # Colormap + normalisation (Teff en log)
    cmap = plt.get_cmap("rainbow")
    znanmin = np.nanmin(z[np.isfinite(z)])
    znanmax = np.nanmax(z[np.isfinite(z)])
    norm = LogNorm(vmin=znanmin, vmax=znanmax)

    # Figure/axes
    fig = plt.figure(figsize=(13.5, 6.5), dpi=dpi)
    ax1 = plt.gca()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(np.nanmin(x), np.nanmax(x))
    ax1.set_ylim(np.nanmin(y), np.nanmax(y))
    ax1.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    ax1.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)
    ax1.minorticks_on()
    if coronagraph is not None:
        ax1.set_title(f'{instru} re-detections with {coronagraph} coronagraph ({spectrum_contributions} light with {name_model})\n (from {round(lmin, 1):.1f} to {round(lmax, 1):.1f} µm with R ≈ {int(round(R, -2))}) with '+r'$t_{exp}$='+f'{round(exposure_time/60):.0f}h ', fontsize=16)
    elif apodizer != "NO_SP":
        ax1.set_title(f'{instru} re-detections with {apodizer.replace("_", " ")} apodizer ({spectrum_contributions} light with {name_model})\n (from {round(lmin, 1):.1f} to {round(lmax, 1):.1f} µm with R ≈ {int(round(R, -2))}) with '+r'$t_{exp}$='+f'{round(exposure_time/60):.0f}h ', fontsize=16)
    else:
        ax1.set_title(f'{instru} re-detections ({spectrum_contributions} light with {name_model})\n (from {round(lmin, 1):.1f} to {round(lmax, 1):.1f} µm with R ≈ {int(round(R, -2))}) with '+r'$t_{exp}$='+f'{round(exposure_time/60):.0f}h ', fontsize=16)
    ax1.set_xlabel(f"Separation [{config_data['sep_unit']}]", fontsize=14)
    ax1.set_ylabel("Planet-to-star flux ratio (on instru-band)", fontsize=14)
    ax1.axvspan(iwa, owa, color='k', alpha=0.3, lw=0, label="Working angle", zorder=2)
    scatter_kwargs = dict(cmap=cmap, norm=norm, edgecolors="k", linewidths=0.6, alpha=0.95, zorder=3)
    ax1.scatter(x[rv_mask], y[rv_mask], s=s[rv_mask], c=z[rv_mask], marker="o", **scatter_kwargs)
    ax1.scatter(x[tr_mask], y[tr_mask], s=s[tr_mask], c=z[tr_mask], marker="v", **scatter_kwargs)
    ax1.scatter(x[im_mask], y[im_mask], s=s[im_mask], c=z[im_mask], marker="s", **scatter_kwargs)
    ax1.scatter(x[ot_mask], y[ot_mask], s=s[ot_mask], c=z[ot_mask], marker="P", **scatter_kwargs)
    ax1.plot([], [], 'ko', ms=10, label="Radial Velocity")
    ax1.plot([], [], 'kv', ms=10, label="Transit")
    ax1.plot([], [], 'ks', ms=10, label="Direct Imaging")
    ax1.plot([], [], 'kP', ms=10, label="Other")
    ax1.legend(fontsize=14, loc="lower right", frameon=True, edgecolor="gray", facecolor="whitesmoke")

    # Cbar
    norm = LogNorm(vmin=np.nanmin(z), vmax=np.nanmax(z))
    sm   =  ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.065)
    cbar.set_label('$T_{eff}$ [K]', fontsize=14, labelpad=18, rotation=270)
    cbar.minorticks_on()

    # Delta mag axis
    ax2 = ax1.twinx()
    ax2.invert_yaxis()
    ax2.set_ylabel(r'$\Delta$mag', fontsize=14, labelpad=18, rotation=270)
    ax2.tick_params(axis='y')
    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))
    ax2.minorticks_on()

    # Show
    plt.tight_layout()
    plt.show(block=False)



##########
# Others #
##########

def Vrot_plots(dpi=dpi_fig):
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")

    nbins = 50
    pm_mask = np.logical_not(get_invalid_mask(planet_table["PlanetMass"]))
    plt.figure(dpi=dpi)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(300*u.Mearth<planet_table["PlanetMass"])]), bins=nbins, edgecolor='black', alpha=0.666, label="Super-Jupiters", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(100*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=300*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Jupiters-like", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(20*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=100*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Neptunes-like", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(5*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=20*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Super-Earths", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(planet_table["PlanetMass"]<=5*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Earths-like", zorder=3)
    plt.ylabel("Occurences", fontsize=14)
    plt.xlabel("Vsin(i) [km/s]", fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show(block=False)

    st_mask = np.logical_not(get_invalid_mask(planet_table["StarTeff"]))
    plt.figure(dpi=dpi)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (10000 * u.K < planet_table["StarTeff"])]), bins=nbins, edgecolor='black', alpha=0.666, label="Very hot stars", zorder=3)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (6000 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Hot stars", zorder=3)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (3500 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Solar-type stars", zorder=3)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (planet_table["StarTeff"] <= 3500 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Cool stars", zorder=3)
    plt.ylabel("Occurrences", fontsize=14)
    plt.xlabel("Vsin(i) [km/s]", fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1)
    plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show(block=False)



################################
# ELT exo-earth yield contrast #
################################

def process_contrast(args):
    """
    Worker: compute contrast and band magnitudes for a single planet.
    """
    # idx, planet row
    idx, planet = args

    # Context
    instru                 = _CONTRAST_CTX["instru"]
    config_data            = _CONTRAST_CTX["config_data"]
    thermal_model          = _CONTRAST_CTX["thermal_model"]
    reflected_model        = _CONTRAST_CTX["reflected_model"]
    wave_model             = _CONTRAST_CTX["wave_model"]
    wave_K                 = _CONTRAST_CTX["wave_K"]
    counts_vega            = _CONTRAST_CTX["counts_vega"]
    counts_vega_K          = _CONTRAST_CTX["counts_vega_K"]
    band0                  = _CONTRAST_CTX["band0"]
    exposure_time          = _CONTRAST_CTX["exposure_time"]
    Rc                     = _CONTRAST_CTX["Rc"]
    filter_type            = _CONTRAST_CTX["filter_type"]
    systematics            = _CONTRAST_CTX["systematics"]
    PCA                    = _CONTRAST_CTX["PCA"]
    N_PCA                  = _CONTRAST_CTX["N_PCA"]
    masks                  = _CONTRAST_CTX["masks"]
    sep_max                = _CONTRAST_CTX["sep_max"]
    spectrum_contributions = _CONTRAST_CTX["spectrum_contributions"]

    # Computing models on wave_model
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=instru, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False)

    # Recalculates the magnitude in case the thermal model is no longer BT-Settl or the reflected model is no longer PICASO (the mag changes with regards to the raw archive table with the estimated magnitudes)
    mask_instru     = masks[instru]
    mag_s           = get_mag(wave=wave_model[mask_instru], density_obs=star_spectrum.flux[mask_instru],    density_vega=None, counts_vega=counts_vega[instru])
    mag_p_thermal   = get_mag(wave=wave_model[mask_instru], density_obs=planet_thermal.flux[mask_instru],   density_vega=None, counts_vega=counts_vega[instru])
    mag_p_reflected = get_mag(wave=wave_model[mask_instru], density_obs=planet_reflected.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
    mag_p_total     = get_mag(wave=wave_model[mask_instru], density_obs=planet_spectrum.flux[mask_instru],  density_vega=None, counts_vega=counts_vega[instru])

    if spectrum_contributions=="thermal":
        planet_spectrum = planet_thermal.copy()
        mag_p           = mag_p_thermal
    elif spectrum_contributions=="reflected":
        planet_spectrum = planet_reflected.copy()
        mag_p           = mag_p_reflected
    elif spectrum_contributions=="thermal+reflected":
        planet_spectrum = planet_spectrum.copy()
        mag_p           = mag_p_total

    # Computing the contrast for the planet
    contrasts_min = []
    for apodizer in config_data["apodizers"]:
        for strehl in config_data["strehls"]:
            # strehl = "MED"
            # if apodizer not in ["NO_SP", "SP1"]:
            #     continue
            for coronagraph in config_data["coronagraphs"]:
                _, separation, curves = FastCurves(instru=instru, calculation="contrast", mag_star=mag_s, band0=band0, exposure_time=exposure_time, mag_planet=mag_p, separation_planet=sep_max/1000, show_plot=False, verbose=False, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematics=systematics, PCA=PCA, N_PCA=N_PCA)
                contrasts_min.append(np.nanmin(np.stack(curves, axis=0), axis=0))

    contrast_min = np.nanmin(np.stack(contrasts_min, axis=0), axis=0)

    return idx, mag_s, mag_p_thermal, mag_p_reflected, mag_p_total, separation[0], contrast_min



def get_planet_table_contrast(instru, planet_table, exposure_time, table="Archive", thermal_model="None", reflected_model="None", spectrum_contributions=None, Rc=100, filter_type=filter_type_default, systematics=False, PCA=False, N_PCA=20, force_table_calc=False, sep_max=None):

    # Contribution and model labels
    _, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # Suffix
    suffix = "with systematics+PCA" if (systematics and PCA) else ("with systematics" if systematics else "without systematics")

    # Filename
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename = str(get_planet_table_path(table, create=True) / f"{table}_ELT_Contrast_Earth_Like_Pull_{instru}_{suffix}_{name_model}_{spectrum_contributions}_t{exposure_time}mn_Rc{Rc}.ecsv")

    # Retrieving or computing the table
    try:
        print(f"\nOpening existing table: {filename}")
        if force_table_calc:
            raise ValueError("Forcing table calculation: force_table_calc=True")
        planet_table = QTable.read(filename, format="ascii.ecsv")
        separation   = fits.getdata(filename.replace(".ecsv", "_separation.fits"))

    except Exception as e:

        print(f"\nComputing the table: {e}...")

        planet_table = planet_table.copy()
        config_data  = get_config_data(instru)
        time1        = time.time()
        if systematics and filter_type == "gaussian_fast":
            filter_type = "gaussian" # "gaussian_fast" is bad for handling systematics estimations

        # --- 4) Wavelength grids and Vega ---
        # K-band for photometry
        wave_K = get_wave_K()

        # Model bandwidth
        lmin_instru = config_data["lambda_range"]["lambda_min"]                    # [µm]
        lmax_instru = config_data["lambda_range"]["lambda_max"]                    # [µm]
        wave_model  = get_wave_model(lmin=lmin_instru, lmax=lmax_instru, R=R0_max) # [µm]

        # Vega spectrum on K-band and instru-band [J/s/m2/µm]
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K,      renorm=False)
        vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_model, renorm=False)

        # --- 5) Bandpass masks and Vega flux on bands ---
        counts_vega_K       = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
        counts_vega         = {}
        masks               = {}
        masks[instru]       = (wave_model >= lmin_instru) & (wave_model <= lmax_instru)
        counts_vega[instru] = get_counts_from_density(wave=wave_model[masks[instru]], density=vega_spectrum.flux[masks[instru]])

        # Band where magnitudes are defined for the FastCurves computations
        band0 = "instru"

        # Print
        print(f"\n {instru} {suffix} ({thermal_model} & {reflected_model})")

        # --- 6) Init global context for workers ---
        global _CONTRAST_CTX
        _CONTRAST_CTX = dict(instru=instru, config_data=config_data, thermal_model=thermal_model, reflected_model=reflected_model, wave_model=wave_model, wave_K=wave_K, counts_vega=counts_vega, counts_vega_K=counts_vega_K, band0=band0, exposure_time=exposure_time, Rc=Rc, filter_type=filter_type, systematics=systematics, PCA=PCA, N_PCA=N_PCA, masks=masks, sep_max=sep_max, spectrum_contributions=spectrum_contributions)

        # --- 7) Run contrast---
        with Pool(processes=cpu_count()//2) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            for (idx, mag_s, mag_p_thermal, mag_p_reflected, mag_p_total, separation, contrast_5sigma) in tqdm(pool.imap(process_contrast, [(idx, planet_table[idx]) for idx in range(len(planet_table))]), total=len(planet_table), desc="Multiprocessing"):
                if idx==0:
                    planet_table["contrast_5sigma"] = np.full((len(planet_table), len(separation)), np.nan)
                planet_table[idx][f"StarINSTRUmag({instru})"]                      = mag_s
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal)"]           = mag_p_thermal
                planet_table[idx][f"PlanetINSTRUmag({instru})(reflected)"]         = mag_p_reflected
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal+reflected)"] = mag_p_total
                planet_table[idx]["contrast_5sigma"]                               = contrast_5sigma

        print(f"\n Calculating 5sigma-contrasts took {(time.time()-time1)/60:.1f} mn")
        planet_table.write(filename, format='ascii.ecsv', overwrite=True)
        fits.writeto(filename.replace(".ecsv", "_separation.fits"), separation, overwrite=True)
        print(f"\nSaving table: {filename}")

    return separation, planet_table



def yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", spectrum_contributions="reflected", force_table_calc=False, exposure_time=10*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3, dpi=dpi_fig):

    from math import erf

    # --- Archive table of known exoplanets
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")

    # --- Ranges "Earth-like"
    R_min, R_max       = 0, 2     # [R_earth]
    M_min, M_max       = 0, 10    # [M_earth]
    Teff_min, Teff_max = 0, 500   # [K]

    def col(name):
        return planet_table[name].value if name in planet_table.colnames else None

    R    = col("PlanetRadius")        # R_earth
    M    = col("PlanetMass")          # M_earth
    Teff = col("PlanetTeff")          # K
    #a_AU = col("SMA")                 # AU
    n    = len(planet_table)
    mask_earth = np.ones(n, dtype=bool)
    print("Filtering Earth-Like planets")

    # Rayon & masse & Teff
    if R is not None:
        before      = int(mask_earth.sum())
        mask_earth &= np.isfinite(R) & (R >= R_min) & (R <= R_max)
        after       = int(mask_earth.sum())
        print(f" After radius filtering:      {after} / {n} (-{before - after})")
    if M is not None:
        before      = int(mask_earth.sum())
        mask_earth &= np.isfinite(M) & (M >= M_min) & (M <= M_max)
        after       = int(mask_earth.sum())
        print(f" After mass filtering:        {after} / {n} (-{before - after})")
    if Teff is not None:
        before      = int(mask_earth.sum())
        mask_earth &= np.isfinite(Teff) & (Teff >= Teff_min) & (Teff <= Teff_max)
        after       = int(mask_earth.sum())
        print(f" After temperature filtering: {after} / {n} (-{before - after})")

    planet_table = planet_table[mask_earth]

    # ---Retrieving tables
    separation_HARMONI, planet_table_HARMONI = get_planet_table_contrast(instru="HARMONI", planet_table=planet_table, exposure_time=exposure_time, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, Rc=Rc, filter_type=filter_type_default, systematics=False, PCA=False, N_PCA=20, force_table_calc=force_table_calc, sep_max=sep_max)
    separation_ANDES, planet_table_ANDES     = get_planet_table_contrast(instru="ANDES",   planet_table=planet_table, exposure_time=exposure_time, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, Rc=Rc, filter_type=filter_type_default, systematics=False, PCA=False, N_PCA=20, force_table_calc=force_table_calc, sep_max=sep_max)

    # --- 5 sigma contrast
    # HARMONI
    contrast_5sigma_HARMONI = np.zeros((len(planet_table_HARMONI), len(separation_HARMONI)))
    for i in range(len(planet_table_HARMONI)):
        contrast_5sigma_HARMONI[i]         = planet_table_HARMONI[i]["contrast_5sigma"]
        valid                              = np.isfinite(contrast_5sigma_HARMONI[i])
        contrast_5sigma_HARMONI[i][~valid] = contrast_5sigma_HARMONI[i][valid][0]

    # ANDES
    contrast_5sigma_ANDES = np.zeros((len(planet_table_ANDES), len(separation_ANDES)))
    for i in range(len(planet_table_ANDES)):
        contrast_5sigma_ANDES[i] = planet_table_ANDES[i]["contrast_5sigma"]

    # --- Contrasts / Magnitudes
    # HARMONI
    planet_magnitude_thermal   = planet_table_HARMONI["PlanetINSTRUmag(HARMONI)(thermal)"].value           # thermal contribution
    planet_magnitude_reflected = planet_table_HARMONI["PlanetINSTRUmag(HARMONI)(reflected)"].value         # reflected contribution
    #planet_magnitude_total     = planet_table_HARMONI["PlanetINSTRUmag(HARMONI)(thermal+reflected)"].value # thermal + reflected contribution
    star_magnitude             = planet_table_HARMONI["StarINSTRUmag(HARMONI)"].value
    contrast_thermal_HARMONI   = 10**(-(planet_magnitude_thermal-star_magnitude)/2.5)
    contrast_reflected_HARMONI = 10**(-(planet_magnitude_reflected-star_magnitude)/2.5)

    # ANDES
    planet_magnitude_thermal   = planet_table_ANDES["PlanetINSTRUmag(ANDES)(thermal)"].value           # thermal contribution
    planet_magnitude_reflected = planet_table_ANDES["PlanetINSTRUmag(ANDES)(reflected)"].value         # reflected contribution
    #planet_magnitude_total     = planet_table_ANDES["PlanetINSTRUmag(ANDES)(thermal+reflected)"].value # thermal + reflected contribution
    star_magnitude             = planet_table_ANDES["StarINSTRUmag(ANDES)"].value
    contrast_thermal_ANDES     = 10**(-(planet_magnitude_thermal-star_magnitude)/2.5)
    contrast_reflected_ANDES   = 10**(-(planet_magnitude_reflected-star_magnitude)/2.5)

    # Mean over both instruments
    contrast_thermal   = (contrast_thermal_HARMONI + contrast_thermal_ANDES) / 2
    contrast_reflected = (contrast_reflected_HARMONI + contrast_reflected_ANDES) / 2
    contrast           = contrast_thermal + contrast_reflected

    # --- Axis & masks
    separation     = planet_table["AngSep"].value # sep axis [mas]
    planet_mass    = planet_table["PlanetMass"].value # size axis [M_earth]
    thermal_mask   = contrast_thermal >= contrast_reflected
    reflected_mask = contrast_thermal < contrast_reflected

    # --- Plot
    # Taille des points en fonction de la masse
    pm_valid  = planet_mass[np.isfinite(planet_mass) & (planet_mass > 0)]
    mass_minG = pm_valid.min()
    mass_maxG = pm_valid.max()
    s         = mass_to_size(planet_mass, s0=s0, ds=ds, mass_min=mass_minG, mass_max=mass_maxG)
    levels    = (0.10, 0.25, 0.50, 0.75)      # niveaux de probabilité P_det
    x_label   = 40.0                   # position des clabels (en mas)
    lw        = 3



    #######################################################################
    # V1
    #######################################################################

    def _lighten(color, amount):
        """Blend 'color' toward white by 'amount' (0→no change, 1→white)."""
        c = np.array(mpl.colors.to_rgb(color))
        return tuple(c + (1 - c)*amount)

    # Percentiles correspondant à ±kσ (k=1,2,3)
    def band_k_sigma(arr, k):
        low  = 50.0 * (1.0 - erf(k/np.sqrt(2.0)))      # ex: k=1 -> 15.866%
        high = 100.0 - low                              # ex: k=1 -> 84.134%
        lo = np.nanpercentile(arr, low,  axis=0)
        hi = np.nanpercentile(arr, high, axis=0)
        return lo, hi

    # Figure/axes
    plt.figure(figsize=(13.5, 6.5), dpi=dpi)
    ax1 = plt.gca()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(np.nanmin(separation)/2, sep_max)
    ax1.set_ylim(1e-9, 1e-5)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.minorticks_on()
    ax1.set_xlabel("Separation [mas]", fontsize=16)
    ax1.set_ylabel("Contrast", fontsize=16)

    # ---------- bandes 5σ TOUT EN BAS ----------

    # pré-calcul des bandes (évite tout mélange d’ordre)
    dsig = 0.666
    loH1, hiH1 = band_k_sigma(contrast_5sigma_HARMONI, 1*dsig)
    loA1, hiA1 = band_k_sigma(contrast_5sigma_ANDES,   1*dsig)
    loH2, hiH2 = band_k_sigma(contrast_5sigma_HARMONI, 2*dsig)
    loA2, hiA2 = band_k_sigma(contrast_5sigma_ANDES,   2*dsig)
    loH3, hiH3 = band_k_sigma(contrast_5sigma_HARMONI, 3*dsig)
    loA3, hiA3 = band_k_sigma(contrast_5sigma_ANDES,   3*dsig)

    z0 = 0  # très bas pour être sous la grille
    # ax1.fill_between(separation_HARMONI, loH1, hiH1, color=_lighten(colors_instru["HARMONI"], 1*alpha_sig), zorder=z0+5, edgecolor="gray")
    # ax1.fill_between(separation_ANDES,   loA1, hiA1, color=_lighten(colors_instru["ANDES"],   1*alpha_sig), zorder=z0+4, edgecolor="gray")
    # ax1.fill_between(separation_HARMONI, loH2, hiH2, color=_lighten(colors_instru["HARMONI"], 2*alpha_sig), zorder=z0+3, edgecolor="gray")
    # ax1.fill_between(separation_ANDES,   loA2, hiA2, color=_lighten(colors_instru["ANDES"],   2*alpha_sig), zorder=z0+2, edgecolor="gray")
    # ax1.fill_between(separation_HARMONI, loH3, hiH3, color=_lighten(colors_instru["HARMONI"], 3*alpha_sig), zorder=z0+1, edgecolor="gray")
    # ax1.fill_between(separation_ANDES,   loA3, hiA3, color=_lighten(colors_instru["ANDES"],   3*alpha_sig), zorder=z0,   edgecolor="gray")

    ax1.fill_between(separation_HARMONI, loH1, hiH1, color=colors_instru["HARMONI"], alpha=alpha_sig, zorder=z0+5, edgecolor="gray")
    ax1.fill_between(separation_ANDES,   loA1, hiA1, color=colors_instru["ANDES"],   alpha=alpha_sig, zorder=z0+4, edgecolor="gray")
    ax1.fill_between(separation_HARMONI, loH2, hiH2, color=colors_instru["HARMONI"], alpha=alpha_sig, zorder=z0+3, edgecolor="gray")
    ax1.fill_between(separation_ANDES,   loA2, hiA2, color=colors_instru["ANDES"],   alpha=alpha_sig, zorder=z0+2, edgecolor="gray")
    ax1.fill_between(separation_HARMONI, loH3, hiH3, color=colors_instru["HARMONI"], alpha=alpha_sig, zorder=z0+1, edgecolor="gray")
    ax1.fill_between(separation_ANDES,   loA3, hiA3, color=colors_instru["ANDES"],   alpha=alpha_sig, zorder=z0,   edgecolor="gray")

    # ---------- grille en dessous des bandes ----------
    ax1.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=z0-1)
    ax1.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=z0-1)

    # ---------- points AU-DESSUS de la grille ----------
    scatter_kwargs = dict(marker="o", edgecolors="k", linewidths=0.5, alpha=0.9, zorder=10)
    ax1.scatter(separation[reflected_mask], contrast[reflected_mask], s=s[reflected_mask], c="C0", **scatter_kwargs)
    ax1.scatter(separation[thermal_mask],   contrast[thermal_mask],   s=s[thermal_mask],   c="C3", **scatter_kwargs)
    ax1.plot([], [], ls="", marker='o', ms=15, label="Reflected", c="C0")
    ax1.plot([], [], ls="", marker='o', ms=15, label="Thermal",   c="C3")

    # ---------- légendes TOUT EN HAUT ----------
    leg_planets = ax1.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Planet-light regime", title_fontsize=16)
    ax1.add_artist(leg_planets)

    handles_contours = [Patch(facecolor=_lighten(colors_instru["HARMONI"], 1*alpha_sig), edgecolor=colors_instru["HARMONI"], label="HARMONI bands"),
                        Patch(facecolor=_lighten(colors_instru["ANDES"],   1*alpha_sig), edgecolor=colors_instru["ANDES"],   label="ANDES bands"),]
    leg_bands = ax1.legend(handles=handles_contours, loc="lower left", fontsize=14, frameon=True, edgecolor="gray", facecolor="white", title=r"$5\sigma$-contrast", title_fontsize=16)

    leg_planets.set_zorder(100)
    leg_bands.set_zorder(100)

    # ---------- axe Δmag ----------
    ax2 = ax1.twinx()
    ax2.invert_yaxis()
    ax2.set_ylabel(r'$\Delta$mag', fontsize=16, labelpad=20, rotation=270)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))
    ax2.minorticks_on()

    # ---------- légende taille ----------
    mass_ticks   = np.array([0.1, 1, 10])
    size_ticks   = mass_to_size(mass_ticks, s0=s0, ds=ds, mass_min=mass_minG, mass_max=mass_maxG)
    size_labels  = [fr"{m:.1f} $M_\oplus$" for m in mass_ticks]
    size_handles = [ax2.scatter([], [], s=sz, edgecolors='k', facecolors="none") for sz in size_ticks]
    leg_size     = ax2.legend(size_handles, size_labels, loc="center left", frameon=True, edgecolor="gray", facecolor="white", title="Planet mass", title_fontsize=16, fontsize=14, scatterpoints=1)

    for lg in (leg_planets, leg_bands, leg_size):
        lg.set_zorder(100)
        lg.get_frame().set_alpha(1.0)       # <-- opaque
        lg.get_frame().set_facecolor("white")
        lg.get_frame().set_edgecolor("gray")

    import matplotlib.patheffects as pe
    # ---- Diffraction limit line + on-line label (bottom) ----
    DL_mas = 7.0
    # the line
    ax1.axvline(DL_mas, color="k", lw=lw, ls=(0, (6, 4)), zorder=9)
    # y-position: a touch above the bottom on a log scale
    ymin, ymax = ax1.get_ylim()
    y_text     = ymin * (ymax / ymin)**0.32
    # label along the line, anchored at bottom
    txt = ax1.text(DL_mas*1.02, y_text, "Diffraction limit", rotation=-90, rotation_mode="anchor", va="bottom", ha="left", fontsize=14, color="k", zorder=10, fontweight="bold")
    # white halo for readability
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])

    plt.tight_layout()
    plt.show(block=False)



    #######################################################################
    # V2
    #####################################################################

    import matplotlib.patheffects as pe

    def pdet_grid(sep_axis, contrast_curves, y_grid):
        """
        Construit P_det(contrast, sep) à partir d'un faisceau de courbes 5σ.
        - sep_axis: (M,) séparations
        - contrast_curves: (N, M) faisceau de courbes 5σ
        - y_grid: (K,) grille de contraste (log-espacée conseillé)
        Retourne P: (K, M) dans [0, 1].
        """
        curves = np.asarray(contrast_curves)
        K, M = len(y_grid), curves.shape[1]
        P = np.empty((K, M), dtype=float)
        for j in range(M):
            col = np.sort(curves[:, j])                # ECDF à la séparation j
            idx = np.searchsorted(col, y_grid, 'right')
            P[:, j] = idx / col.size
        return P

    def y_at_prob_at_x(sep, P, y_grid, level, x0):
        """
        Donne la valeur de contraste y telle que P_det(y, x0) ~ level.
        Interpole dans la colonne de P la plus proche de x0.
        """
        j = int(np.argmin(np.abs(sep - x0)))
        colP = P[:, j]
        # borne dans [min,max] si besoin pour éviter les extrapolations
        lv = np.clip(level, colP.min(), colP.max())
        # P(y) est croissante avec y -> interpolation dans l'espace log(y)
        return float(10**np.interp(lv, colP, np.log10(y_grid)))

    def draw_prob_contours(ax, sep, P, y_grid, color, levels=(0.25,0.5,0.75),
                           lw=2.0, z=0.5):
        """
        Trace les contours de P_det et renvoie l'objet QuadContourSet.
        """
        CS = ax.contour(sep, y_grid, P, levels=levels, colors=color,
                        linestyles='-', linewidths=lw, zorder=z)
        return CS

    def clabel_all_at_x(ax, CS, sep, P, y_grid, levels, x0, fontsize=12):
        """
        Place un label par niveau 'levels' exactement à x = x0 mas.
        """
        manual = [(x0, y_at_prob_at_x(sep, P, y_grid, L, x0)) for L in levels]
        labs = ax.clabel(CS, levels=levels, manual=manual,
                         fmt=lambda v: f"{int(round(100*v))}%", inline=True,
                         fontsize=fontsize)
        # halo blanc + au-dessus des lignes
        for t in labs:
            t.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
            t.set_clip_on(False)
            t.set_zorder(20)
        return labs

    import matplotlib.patheffects as pe

    # ---------------------- Ta taille de points ----------------------
    pm_valid  = planet_mass[np.isfinite(planet_mass) & (planet_mass > 0)]
    mass_minG = pm_valid.min()
    mass_maxG = pm_valid.max()
    s         = mass_to_size(planet_mass, s0=s0, ds=ds,
                             mass_min=mass_minG, mass_max=mass_maxG)

    # ---------------------- Figure / Axes ----------------------
    plt.figure(figsize=(13.5, 6.5), dpi=dpi)
    ax1 = plt.gca()
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(np.nanmin(separation)/2, sep_max)
    ax1.set_ylim(1e-9, 1e-5)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.minorticks_on()
    ax1.set_xlabel("Separation [mas]", fontsize=16)
    ax1.set_ylabel("Contrast", fontsize=16)

    # Grille sous les contours
    ax1.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
    ax1.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)

    # ---------------------- Probabilité de détection ----------------------
    ymin, ymax = ax1.get_ylim()
    y_grid = np.logspace(np.log10(ymin), np.log10(ymax), 400)

    # Champs P_det
    P_H = pdet_grid(separation_HARMONI, contrast_5sigma_HARMONI, y_grid)
    P_A = pdet_grid(separation_ANDES,   contrast_5sigma_ANDES,   y_grid)

    # Contours
    CS_H = draw_prob_contours(ax1, separation_HARMONI, P_H, y_grid, colors_instru["HARMONI"], levels=levels, lw=lw, z=0.4)
    CS_A = draw_prob_contours(ax1, separation_ANDES,   P_A, y_grid, colors_instru["ANDES"],   levels=levels, lw=lw, z=0.5)

    # Labels forcés à x = 40 mas
    clabel_all_at_x(ax1, CS_H, separation_HARMONI, P_H, y_grid, levels, x_label, fontsize=12)
    clabel_all_at_x(ax1, CS_A, separation_ANDES,   P_A, y_grid, levels, x_label, fontsize=12)

    # ---------------------- Points (au-dessus) ----------------------
    scatter_kwargs = dict(marker="o", edgecolors="k", linewidths=0.5, alpha=0.9, zorder=10)
    ax1.scatter(separation[reflected_mask], contrast[reflected_mask],
                s=s[reflected_mask], c="C0", **scatter_kwargs)
    ax1.scatter(separation[thermal_mask],   contrast[thermal_mask],
                s=s[thermal_mask],   c="C3", **scatter_kwargs)
    ax1.plot([], [], ls="", marker='o', ms=15, label="Reflected", c="C0")
    ax1.plot([], [], ls="", marker='o', ms=15, label="Thermal",   c="C3")

    # ---------------------- Légendes ----------------------
    leg_planets = ax1.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Planet-light regime", title_fontsize=16)
    ax1.add_artist(leg_planets)

    leg_inst = ax1.legend(handles=[Line2D([0],[0], color=colors_instru["HARMONI"], lw=lw, ls='-',  label="HARMONI"), Line2D([0],[0], color=colors_instru["ANDES"],   lw=lw, ls='-',  label="ANDES")],
        loc="lower left", fontsize=14, frameon=True, edgecolor="gray",
        facecolor="white", title="Detection probability", title_fontsize=16)
    ax1.add_artist(leg_inst)

    for lg in (leg_planets, leg_inst):
        lg.set_zorder(100)
        lg.get_frame().set_alpha(1.0)

    # ---------------------- Axe Δmag ----------------------
    ax2 = ax1.twinx()
    ax2.invert_yaxis()
    ax2.set_ylabel(r'$\Delta$mag', fontsize=16, labelpad=20, rotation=270)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.minorticks_on()
    ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))

    # ---------------------- Légende taille ----------------------
    mass_ticks   = np.array([0.1, 1, 10])
    size_ticks   = mass_to_size(mass_ticks, s0=s0, ds=ds,
                                mass_min=mass_minG, mass_max=mass_maxG)
    size_labels  = [fr"{m:.1f} $M_\oplus$" for m in mass_ticks]
    size_handles = [ax2.scatter([], [], s=sz, edgecolors='k', facecolors="none")
                    for sz in size_ticks]
    leg_size = ax2.legend(size_handles, size_labels, loc="center left", frameon=True,
                          edgecolor="gray", facecolor="white", title="Planet mass",
                          title_fontsize=16, fontsize=14, scatterpoints=1)
    leg_size.set_zorder(100); leg_size.get_frame().set_alpha(1.0)

    # ---- Diffraction limit line + on-line label (bottom) ----
    DL_mas = 7.0
    # the line
    ax1.axvline(DL_mas, color="k", lw=lw, ls=(0, (6, 4)), zorder=9)
    # y-position: a touch above the bottom on a log scale
    ymin, ymax = ax1.get_ylim()
    y_text     = ymin * (ymax / ymin)**0.32
    # label along the line, anchored at bottom
    txt = ax1.text(DL_mas*1.02, y_text, "Diffraction limit", rotation=-90, rotation_mode="anchor", va="bottom", ha="left", fontsize=14, color="k", zorder=10, fontweight="bold")
    # white halo for readability
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])

    plt.tight_layout()
    plt.show(block=False)



def yield_heatmap_ELT(table="Archive", instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl=None, systematics=False, PCA=False, split_ptypes_by_regime=False, band_regime_plot=None, fraction=False, SNR_thresh=SNR_thresh, heatmaps=None, ptypes_heatmap=None, bands_plot=None, config_labels=None, config_axis_name=None, x_axis_name=None, panel_labels=None, title=None, save_dir=None, filename=None, dpi=dpi_fig):
    """Plot ELT yield heatmaps from FastYield tables or already-computed heatmaps."""

    # Optional mode using already-computed heatmaps
    precomputed = heatmaps is not None
    if precomputed:
        if ptypes_heatmap is None or bands_plot is None or config_labels is None:
            raise ValueError("ptypes_heatmap, bands_plot and config_labels must be provided with precomputed heatmaps.")
        ptypes_heatmap   = list(ptypes_heatmap)
        bands_plot       = list(bands_plot)
        config_labels    = list(config_labels)
        panel_labels     = {} if panel_labels is None else panel_labels
        config_axis_name = "Configuration" if config_axis_name is None else config_axis_name
        x_axis_name      = "Spectral band" if x_axis_name is None else x_axis_name
        colorbar_label   = "Detected [%]" if fraction  else "Detected"
        for ptype in ptypes_heatmap:
            if ptype not in heatmaps:
                raise KeyError(f"Missing heatmap for {ptype!r}.")
            if np.shape(heatmaps[ptype]) != (len(config_labels), len(bands_plot)):
                raise ValueError(f"Heatmap {ptype!r} has shape {np.shape(heatmaps[ptype])}, expected {(len(config_labels), len(bands_plot))}.")

    # Mode using FastYield tables
    else:
        config_data                        = get_config_data(instru)
        spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
        bands_plot                         = list(config_data["gratings"].keys())
        base_ptypes                        = ["Jupiter", "Saturn", "Neptune", "Earth"]
        ptype_specs                        = sum(([{"key": f"{ptype}-like - Reflected", "base": ptype, "regime": "reflected"}, {"key": f"{ptype}-like - Thermal", "base": ptype, "regime": "thermal"}] for ptype in base_ptypes), []) if split_ptypes_by_regime else [{"key": ptype, "base": ptype, "regime": None} for ptype in base_ptypes]
        ptypes_heatmap                     = [spec["key"] for spec in ptype_specs]
        regime_by_current_band             = split_ptypes_by_regime and band_regime_plot is None

        if instru == "HARMONI":
            config_axis_name = "Apodizer"
            configs          = [{"label": str(cfg).replace("_", " "), "apodizer": cfg, "coronagraph": config_data["coronagraphs"][0]} for cfg in config_data["apodizers"]]
        elif instru == "ANDES":
            config_axis_name = "Coronagraph"
            configs          = [{"label": "No coronagraph" if cfg is None else str(cfg), "apodizer": "NO_SP", "coronagraph": cfg} for cfg in config_data["coronagraphs"]]
        else:
            raise ValueError("yield_heatmap_ELT is designed for HARMONI and ANDES unless precomputed heatmaps are provided.")
        config_labels  = [cfg["label"] for cfg in configs]
        x_axis_name    = "Spectral resolution" if instru == "ANDES" else "Spectral band"
        colorbar_label = "Detected [%]" if fraction else "Detected"

        def get_ptype_mask(planet_table, ptype):
            if ptype.lower() in ("all", "all planets", "total"):
                return np.ones(len(planet_table), dtype=bool)
            return np.asarray(get_mask_planet_type(planet_table=planet_table, planet_type=ptype), dtype=bool)

        def get_regime_masks(planet_table, band):
            col_th = f"Planet{band}mag({instru})(thermal)"   if band == "INSTRU" else f"Planet{band}mag(thermal)"
            col_re = f"Planet{band}mag({instru})(reflected)" if band == "INSTRU" else f"Planet{band}mag(reflected)"
            if col_th not in planet_table.colnames or col_re not in planet_table.colnames:
                raise KeyError(f"Missing reflected/thermal columns for band {band}: {col_th}, {col_re}")
            mag_th = np.asarray(planet_table[col_th], dtype=float)
            mag_re = np.asarray(planet_table[col_re], dtype=float)
            valid  = np.isfinite(mag_th) & np.isfinite(mag_re)
            return {"thermal": valid & (mag_th < mag_re), "reflected": valid & (mag_re < mag_th)}

        heatmaps             = {ptype: np.full((len(configs), len(bands_plot)), np.nan, dtype=float) for ptype in ptypes_heatmap}
        detected_names_total = {ptype: set() for ptype in ptypes_heatmap}
        for icfg, cfg in enumerate(configs):
            filename_table = get_filename_table(table=table, instru=instru, apodizer=cfg["apodizer"], strehl=strehl, coronagraph=cfg["coronagraph"], systematics=systematics, PCA=PCA, name_model=name_model)
            planet_table   = load_planet_table(filename_table, table=table)
            base_masks     = {ptype: get_ptype_mask(planet_table, ptype) for ptype in base_ptypes}
            if split_ptypes_by_regime and not regime_by_current_band:
                regime_masks = get_regime_masks(planet_table, band_regime_plot)
                masks_fixed  = {spec["key"]: base_masks[spec["base"]] & regime_masks[spec["regime"]] for spec in ptype_specs}
            elif not split_ptypes_by_regime:
                masks_fixed = {spec["key"]: base_masks[spec["base"]] for spec in ptype_specs}

            for iband, band in enumerate(bands_plot):
                missing_cols = [col for col in [f"DIT_{band}", f"signal_{band}", f"sigma_fund_{band}", f"sigma_syst_{band}"] if col not in planet_table.colnames]
                if missing_cols:
                    raise KeyError(f"Missing columns for band {band!r} in {filename_table}: {missing_cols}")
                SNR      = get_SNR_from_table(planet_table=planet_table, exposure_time=exposure_time, band=band)
                detected = np.isfinite(SNR) & (SNR >= SNR_thresh)
                if regime_by_current_band:
                    regime_masks = get_regime_masks(planet_table, band)
                    masks        = {spec["key"]: base_masks[spec["base"]] & regime_masks[spec["regime"]] for spec in ptype_specs}
                else:
                    masks = masks_fixed
                planet_names = np.asarray(planet_table["PlanetName"], dtype=str)
                for ptype in ptypes_heatmap:
                    N_ptype        = int(np.sum(masks[ptype]))
                    mask_det_ptype = detected & masks[ptype]
                    N_det          = int(np.sum(mask_det_ptype))
                    detected_names_total[ptype].update(planet_names[mask_det_ptype])
                    heatmaps[ptype][icfg, iband] = 100.0*N_det/N_ptype if fraction and N_ptype > 0 else (np.nan if fraction else float(N_det))

        panel_labels = {ptype: f"{str(ptype).replace('_', ' ')} (total: {len(detected_names_total[ptype])})" for ptype in ptypes_heatmap}
        if title is None:
            axis_text = "resolution × configuration" if instru == "ANDES" else "band × configuration"
            title     = f"ELT/{instru}: {axis_text} {'fractional yield' if fraction else 'yield'} with {exposure_time/60:.0f} hr per target"

    # Plot
    nptype     = len(ptypes_heatmap)
    ncols      = 2
    nrows      = int(np.ceil(nptype / ncols))
    fig, axes  = plt.subplots(nrows, ncols, figsize=(max(5.2*ncols, 6.0), max((2.2 + 0.45*len(config_labels))*nrows, 3.8)), dpi=dpi, squeeze=False)
    cmap       = plt.get_cmap("inferno")
    fontsize   = 13
    local_vmax = True

    if local_vmax:
        global_vmax = None
    else:
        values      = np.concatenate([np.asarray(heatmaps[ptype], dtype=float).ravel() for ptype in ptypes_heatmap])
        global_vmax = np.nanmax(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 1.0
        global_vmax = 1.0 if not np.isfinite(global_vmax) or global_vmax <= 0 else global_vmax

    for ipt, ptype in enumerate(ptypes_heatmap):
        ax   = axes[ipt//ncols, ipt%ncols]
        hmap = np.asarray(heatmaps[ptype], dtype=float)
        vmax = np.nanmax(hmap) if local_vmax and np.any(np.isfinite(hmap)) else global_vmax
        vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
        im   = ax.imshow(hmap, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)

        for icfg in range(len(config_labels)):
            for iband in range(len(bands_plot)):
                val = hmap[icfg, iband]
                if not np.isfinite(val):
                    continue
                text = f"{val:.0f}%" if fraction else (f"{val:.1f}" if instru == "PCS" else f"{int(round(val))}")
                ax.text(iband, icfg, text, ha="center", va="center", fontsize=fontsize-1, color="white" if val < 0.55*vmax else "black", weight="bold")

        ax.set_title(panel_labels.get(ptype, str(ptype)), fontsize=fontsize+3, pad=fontsize+3, weight="bold")
        ax.set_xticks(np.arange(len(bands_plot)))
        if instru == "ANDES":
            ax.set_xticklabels([str(value).replace("YJH_5mas_", "").replace("YJH_10mas_", "").replace("YJH_16mas_", "")+" 000" for value in bands_plot], fontsize=fontsize, rotation=45)
        else:
            ax.set_xticklabels([str(value).replace("_", " ") for value in bands_plot], fontsize=fontsize, rotation=45)
        ax.set_yticks(np.arange(len(config_labels)))
        ax.set_yticklabels([str(value) for value in config_labels], fontsize=fontsize)
        if ipt//ncols == nrows-1:
            ax.set_xlabel(x_axis_name, fontsize=fontsize+5, labelpad=fontsize)
        else:
            ax.tick_params(labelbottom=False)
        if ipt%ncols == 0:
            ax.set_ylabel(config_axis_name, fontsize=fontsize+5, labelpad=fontsize-3)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xticks(np.arange(-0.5, len(bands_plot), 1),    minor=True)
        ax.set_yticks(np.arange(-0.5, len(config_labels), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.6, alpha=0.18)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", which="major", length=4, width=0.8, pad=3)

        cbar = fig.colorbar(im, ax=ax, pad=0.03)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(colorbar_label, fontsize=fontsize+3, labelpad=16, rotation=270)
        if not fraction:
            cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, integer=instru != "PCS"))

    for ipt in range(nptype, nrows*ncols):
        axes[ipt//ncols, ipt%ncols].axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=fontsize+8, weight="bold", y=0.96)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.88, wspace=0.16, hspace=0.30)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ELT_{instru}_yield_heatmap.png" if filename is None else filename
        fig.savefig(save_dir/filename, bbox_inches="tight", dpi=dpi)
    plt.show(block=False)
    return fig, axes



def yield_population_plot(table="Archive", instru="HARMONI", instru_type=None, post_processing=None, thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="JQ1", config_mode="best", apodizer="NO_SP", coronagraph=None, band_snr="INSTRU", band_contrast_plot="INSTRU", band_regime_plot="INSTRU", systematics=False, PCA=False, planet_table=None, SNR_plot=None, dominant_noise=None, SNR_thr=None, save_dir=None, DL_mas=None, DL_text="Diffraction limit", show_golden_sample=True, show_det_regions=True, show_ptype_regions=True, show_VLT_DL=False, space="both"):
    """
    Plot in one call:
      1) detected population, color coded in different spaces (star Teff, planet Teff, SNR)
      2) dominant noise regime,
      3) thermal/reflected complementarity.
      4) detection fraction in different bins of the parameter space.

    Detection is computed from get_SNR_from_table().
    If config_mode="fixed", one configuration is used.
    If config_mode="max",   the best SNR over all apodizer/coronagraph configurations is used planet by planet.
    If space="obs", only the observational space is plotted.
    If space="phys",  only the physical space is plotted.
    If space="both",  both spaces are plotted side by side.
    """

    golden_sample  = ["Proxima Cen b", "Wolf 1061 b", "GJ 682 b"]#, "GJ 273 b", "Ross 128 b"]
    golden_offsets = {"obs":  {"Proxima Cen b": (100, 50),  "GJ 273 b": (25, -45), "Wolf 1061 b": (-50, 50), "GJ 682 b": (40, -80), "Ross 128 b": (25, 45)},
                      "phys": {"Proxima Cen b": (100, 30), "GJ 273 b": (55, 20),  "Wolf 1061 b": (-30, 80), "GJ 682 b": (50, 40),  "Ross 128 b": (35, 55)}}

    detection_method_region_text_positions = {"Transit": (None, 1e-5), "Radial Velocity": (None, 5e-10), "Imaging": (None, None)}

    ptype_region_text_positions = {"Earth": (None, 7e-1), "Neptune": (None, 2e1), "Saturn": (None, None), "Jupiter": (None, None)}

    if instru is None:
        band_contrast_plot = "H" if band_contrast_plot == "INSTRU" else band_contrast_plot
        band_regime_plot   = "H" if band_regime_plot   == "INSTRU" else band_regime_plot

    if instru_type is not None and post_processing is not None:
        type_and_pp = f'_{instru_type}_{post_processing}'
    else:
        type_and_pp = ''

    space = space.lower()
    if space not in ("obs", "phys", "both"):
        raise ValueError("space must be 'obs', 'phys' or 'both'.")

    if instru == "PCS":
        obs_xlim      = (1, 1e3)
    else:
        obs_xlim      = (1e-2, 1e6)
    obs_ylim          = (1e-10, 1e-1)
    phys_xlim         = (5e-3, 1e3) # [SMA]
    phys_ylim         = (5e-1, 1e4) # [M_earth]
    ss_detected       = 180
    ss_nondetected    = 90
    alpha_detected    = 0.5 if instru is None else 0.9
    alpha_nondetected = 0.05
    fontsize          = 22
    dpi               = dpi_fig
    cbar_frac         = 0.03 if space == "both" else 0.06 # colorbar width and pad, relative to the panel(s) width

    # Optional mode using already-computed FastYield results
    precomputed = instru is not None and (planet_table is not None or SNR_plot is not None or dominant_noise is not None)
    if precomputed and (planet_table is None or SNR_plot is None):
        raise ValueError("planet_table and SNR_plot must both be provided in precomputed mode.")
    SNR_thr  = None if instru is None else (SNR_thresh if SNR_thr is None else float(SNR_thr))
    save_dir = None if save_dir is None else Path(save_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    if instru is None:
        planet_table = load_planet_table(f'{table}_Pull_For_FastYield.ecsv', table=table)
        config_data  = None
        configs      = None
    elif precomputed:
        config_data = None
        configs     = None
        DL_mas      = get_DL_FWHM(instru="ANDES", sep_unit="mas", band="INSTRU") if DL_mas is None else float(DL_mas) # ANDES DL PROXY FOR PCS
    else:
        config_data  = get_config_data(instru)
        apodizers    = list(config_data["apodizers"].keys())
        coronagraphs = list(config_data["coronagraphs"])
        DL_mas       = get_DL_FWHM(instru=instru, sep_unit="mas", band="INSTRU") if DL_mas is None else float(DL_mas)
        if config_mode.lower() in ("max", "best", "all"):
            configs = [{"apodizer": apo, "coronagraph": coro, "label": str(apo).replace("_", " ") + (f" + {coro}" if coro is not None else "")} for apo in apodizers for coro in coronagraphs]
        elif config_mode.lower() in ("fixed", "single"):
            apodizer    = apodizers[0] if apodizer is None else apodizer
            coronagraph = coronagraphs[0] if coronagraph is None else coronagraph
            configs     = [{"apodizer": apodizer, "coronagraph": coronagraph, "label": str(apodizer).replace("_", " ") + (f" + {coronagraph}" if coronagraph is not None else "")}]
        else:
            raise ValueError("config_mode must be 'fixed' or 'max'.")

    ptypes_plot         = ["Jupiter", "Saturn", "Neptune", "Earth"]
    ptype_region_colors = {"Earth": "tab:green", "Neptune": "tab:blue", "Saturn": "tab:orange", "Jupiter": "tab:red"}
    ptype_region_labels = {"Earth": "Earth-like", "Neptune": "Neptune-like", "Saturn": "Saturn-like", "Jupiter": "Jupiter-like"}

    detection_methods_plot          = ["Transit", "Radial Velocity", "Imaging", "Other"]
    marker_detection_methods        = {"Transit": "v", "Radial Velocity": "o", "Imaging": "s", "Other": "P"}
    label_detection_methods         = {"Transit": "Transit", "Radial Velocity": "Radial velocity", "Imaging": "Direct imaging", "Other": "Other"}
    detection_methods_regions       = ["Transit", "Radial Velocity", "Imaging"]
    detection_method_region_colors  = {"Transit": "tab:green", "Radial Velocity": "tab:blue", "Imaging": "tab:red"}

    # Helpers
    def arr(colname):
        col = planet_table[colname]
        return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)

    def get_ptype_mask(ptype):
        if ptype.lower() in ("all", "all planets", "total"):
            return np.ones(len(planet_table), dtype=bool)
        return np.asarray(get_mask_planet_type(planet_table=planet_table, planet_type=ptype), dtype=bool)

    def get_combined_planet_mag(band):
        if band=="INSTRU":
            col_tot, col_th, col_re = f"PlanetINSTRUmag({instru})(thermal+reflected)", f"PlanetINSTRUmag({instru})(thermal)", f"PlanetINSTRUmag({instru})(reflected)"
        else:
            col_tot, col_th, col_re = f"Planet{band}mag(thermal+reflected)", f"Planet{band}mag(thermal)", f"Planet{band}mag(reflected)"
        if col_tot in planet_table.colnames:
            return np.asarray(planet_table[col_tot], dtype=float)
        mag  = np.full(len(planet_table), np.nan, dtype=float)
        flux = np.zeros(len(planet_table), dtype=float)
        if col_th in planet_table.colnames:
            mag_th = np.asarray(planet_table[col_th], dtype=float)
            flux[np.isfinite(mag_th)] += 10**(-0.4 * mag_th[np.isfinite(mag_th)])
        if col_re in planet_table.colnames:
            mag_re = np.asarray(planet_table[col_re], dtype=float)
            flux[np.isfinite(mag_re)] += 10**(-0.4 * mag_re[np.isfinite(mag_re)])
        mag[np.isfinite(flux) & (flux > 0)] = -2.5 * np.log10(flux[np.isfinite(flux) & (flux > 0)])
        return mag

    def get_dominant_noise(planet_table, band):
        DIT = np.asarray(planet_table[f"DIT_{band}"], dtype=float)
        DIT = np.where(np.isfinite(DIT) & (DIT > 0), DIT, np.nan)
        if all(col in planet_table.colnames for col in [f"sigma_fund_{band}", f"sigma_syst_{band}"]):
            var_fund = exposure_time / DIT * np.asarray(planet_table[f"sigma_fund_{band}"], dtype=float)**2
            var_syst = (exposure_time / DIT)**2 * np.asarray(planet_table[f"sigma_syst_{band}"], dtype=float)**2
            labels   = np.array(["Fundamental noise", "Systematics" if systematics else "Residuals"], dtype=object)
            idx      = np.nanargmax(np.vstack([var_fund, var_syst]), axis=0)
            return labels[idx]
        return np.full(len(planet_table), "Unknown", dtype=object)

    def setup_axes(ax, xlabel, ylabel, xlim, ylim, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=fontsize+2, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=fontsize+2, labelpad=10)
        ax.set_title(title, fontsize=fontsize+2, weight="bold", pad=10)
        ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
        ax.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25, zorder=-10)
        ax.tick_params(axis="both", which="major", labelsize=fontsize-2, length=6, width=1.0)
        ax.tick_params(axis="both", which="minor", labelsize=fontsize-2, length=3, width=0.8)

    def add_detection_method_legend(ax, loc="lower right"):
        handles = [Line2D([], [], ls="", marker=marker_detection_methods[method], ms=13, markerfacecolor="white", markeredgecolor="k", markeredgewidth=1.5, color="k", label=label_detection_methods[method]) for method in detection_methods_plot]
        ax.add_artist(ax.legend(handles=handles, fontsize=fontsize-2, loc=loc, frameon=True, edgecolor="gray", facecolor="white", title="Discovery method", title_fontsize=fontsize))

    def add_light_regime_legend(ax, loc="lower right"):
        groups, masks, markers = get_regime_marker_groups()
        handles = [Line2D([], [], ls="", marker=markers[label], ms=13, markerfacecolor="white", markeredgecolor="k", markeredgewidth=1.5, color="k", label=f"{label} ({int(np.sum(detected & masks[label]))})") for label in groups]
        ax.add_artist(ax.legend(handles=handles, fontsize=fontsize-2, loc=loc, frameon=True, edgecolor="gray", facecolor="white", title=f"Planet-light regime ({band_regime_plot})", title_fontsize=fontsize))

    def add_detection_legend(ax, loc="lower left"):
        handles = [
            Line2D([], [], ls="", marker="o", ms=13, markerfacecolor="0.65", markeredgecolor="none", alpha=0.35, color="0.65", label="Non-detected"),
            Line2D([], [], ls="", marker="o", ms=13, markerfacecolor="0.65", markeredgecolor="k", markeredgewidth=1.5, alpha=alpha_detected, color="0.65", label="Detected"),
        ]
        leg = ax.add_artist(ax.legend(handles=handles, fontsize=fontsize-2, loc=loc, frameon=True, edgecolor="gray", facecolor="white"))
        leg.set_zorder(100_000)

    def add_detection_method_regions(ax):
        for method in detection_methods_regions:
            mask = masks_detection_method[method] & np.isfinite(AngSep) & np.isfinite(contrast) & (AngSep > 0) & (contrast > 0)
            if np.sum(mask) < 2:
                continue
            xmin, xmax   = 10**np.nanpercentile(np.log10(AngSep[mask]), [2, 98])
            ymin, ymax   = 10**np.nanpercentile(np.log10(contrast[mask]), [2, 98])
            dx, dy       = max(np.log10(xmax/xmin), 0.1), max(np.log10(ymax/ymin), 0.1)
            xmin, xmax   = xmin/10**(0.05*dx), xmax*10**(0.05*dx)
            ymin, ymax   = ymin/10**(0.08*dy), ymax*10**(0.08*dy)
            color, label = detection_method_region_colors[method], label_detection_methods[method]
            text_pos     = detection_method_region_text_positions.get(method)
            xtext        = np.sqrt(xmin*xmax) if text_pos is None or text_pos[0] is None else text_pos[0]
            ytext        = np.sqrt(ymin*ymax) if text_pos is None or text_pos[1] is None else text_pos[1]
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor=mpl.colors.to_rgba(color, 0.25), edgecolor=mpl.colors.to_rgba(color, 0.5), linewidth=3.0, zorder=1000))
            ax.text(xtext, ytext, label, color=color, fontsize=fontsize-1, fontweight="bold", ha="center", va="center", zorder=1001, bbox=dict(boxstyle="round,pad=0.25", facecolor=mpl.colors.to_rgba("white", 0.92), edgecolor=mpl.colors.to_rgba(color, 0.95), linewidth=2.0))

    def add_ptype_regions(ax):
        for ptype in ptypes_plot:
            mask = masks_ptype[ptype] & np.isfinite(SMA) & np.isfinite(PlanetMass) & (SMA > 0) & (PlanetMass > 0)
            if np.sum(mask) < 2:
                continue
            xmin, xmax   = 10**np.nanpercentile(np.log10(SMA[mask]), [2, 98])
            ymin, ymax   = 10**np.nanpercentile(np.log10(PlanetMass[mask]), [2, 98])
            dx, dy       = max(np.log10(xmax/xmin), 0.1), max(np.log10(ymax/ymin), 0.1)
            xmin, xmax   = xmin/10**(0.05*dx), xmax*10**(0.05*dx)
            ymin, ymax   = ymin/10**(0.08*dy), ymax*10**(0.08*dy)
            color, label = ptype_region_colors[ptype], ptype_region_labels[ptype]
            text_pos     = ptype_region_text_positions.get(ptype)
            xtext        = np.sqrt(xmin*xmax) if text_pos is None or text_pos[0] is None else text_pos[0]
            ytext        = np.sqrt(ymin*ymax) if text_pos is None or text_pos[1] is None else text_pos[1]
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor=mpl.colors.to_rgba(color, 0.25), edgecolor=mpl.colors.to_rgba(color, 0.5), linewidth=3.0, zorder=1000))
            ax.text(xtext, ytext, label, color=color, fontsize=fontsize-1, fontweight="bold", ha="center", va="center", zorder=1001, bbox=dict(boxstyle="round,pad=0.25", facecolor=mpl.colors.to_rgba("white", 0.92), edgecolor=mpl.colors.to_rgba(color, 0.95), linewidth=2.0))

    def add_golden_sample(ax, x, y, valid, space):
        if not show_golden_sample:
            return
        for planet_name, idx in golden_indices.items():
            if not valid[idx] or not np.isfinite(x[idx]) or not np.isfinite(y[idx]) or x[idx] <= 0 or y[idx] <= 0:
                continue
            method = next((method for method in detection_methods_plot if masks_detection_method[method][idx]), "Other")
            marker = marker_detection_methods[method]
            dx, dy = golden_offsets[space].get(planet_name, (20, 20))
            ax.scatter(x[idx], y[idx], marker=marker, s=2*ss_detected, facecolors="none", edgecolors="white", linewidths=5.0, zorder=2000)
            ax.scatter(x[idx], y[idx], marker=marker, s=2*ss_detected, facecolors="none", edgecolors="gold",  linewidths=2.5, zorder=2001)
            ax.annotate(planet_name, xy=(x[idx], y[idx]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize-4, fontweight="bold", color="0.10", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="goldenrod", linewidth=2, alpha=1.0), arrowprops=dict(arrowstyle="-", color="goldenrod", linewidth=4, shrinkA=2, shrinkB=5), zorder=2002)

    def get_marker_groups():
        return detection_methods_plot, masks_detection_method, marker_detection_methods

    def get_regime_marker_groups():
        groups = [label for label in ["Thermal", "Reflected", "Unclassified"] if np.any(light_class == label)]
        masks  = {label: light_class == label for label in groups}
        return groups, masks, marker_light_regimes

    def plot_population_panel(ax, x, y, valid, xlabel, ylabel, xlim, ylim, title, space):
        groups, masks, markers = get_marker_groups()
        for group in groups:
            mask = valid & not_detected & masks[group]
            if np.any(mask):
                #ax.scatter(x[mask], y[mask], c="0.70", marker=markers[group], s=ss_nondetected, edgecolors="none", alpha=alpha_nondetected, zorder=1)
                ax.scatter(x[mask], y[mask], c=Teff_star[mask], cmap=cmap_teff, norm=norm_teff, marker=markers[group], s=ss_nondetected, edgecolors="none", alpha=alpha_nondetected, zorder=1)
        for group in groups:
            mask = valid & detected & masks[group]
            if np.any(mask):
                ax.scatter(x[mask], y[mask], c=Teff_star[mask], cmap=cmap_teff, norm=norm_teff, marker=markers[group], s=ss_detected, edgecolors="k", linewidths=0.75, alpha=alpha_detected, zorder=3)
        setup_axes(ax, xlabel, ylabel, xlim, ylim, title)

    def plot_population_planet_teff_panel(ax, x, y, valid, xlabel, ylabel, xlim, ylim, title, space):
        groups, masks, markers = get_marker_groups()
        for group in groups:
            mask = valid & not_detected & masks[group]
            if np.any(mask):
                #ax.scatter(x[mask], y[mask], c="0.70", marker=markers[group], s=ss_nondetected, edgecolors="none", alpha=alpha_nondetected, zorder=1)
                ax.scatter(x[mask], y[mask], c=PlanetTeff[mask], cmap=cmap_planet_teff, norm=norm_planet_teff, marker=markers[group], s=ss_nondetected, edgecolors="none", alpha=alpha_nondetected, zorder=1)
        for group in groups:
            mask = valid & detected & masks[group]
            if np.any(mask):
                ax.scatter(x[mask], y[mask], c=PlanetTeff[mask], cmap=cmap_planet_teff, norm=norm_planet_teff, marker=markers[group], s=ss_detected, edgecolors="k", linewidths=0.75, alpha=alpha_detected, zorder=3)
        setup_axes(ax, xlabel, ylabel, xlim, ylim, title)

    def plot_class_panel(ax, x, y, valid, classes, labels, colors, alphas, xlabel, ylabel, xlim, ylim, title, space):
        groups, masks, markers = get_marker_groups()
        for label in labels:
            for group in groups:
                mask = valid & masks[group] & (classes == label)
                mask_non = mask & not_detected
                mask_det = mask & detected
                if np.any(mask_non):
                    ax.scatter(x[mask_non], y[mask_non], c=colors[label], marker=markers[group], s=ss_nondetected, edgecolors="none", alpha=alpha_nondetected, zorder=1)
                if np.any(mask_det):
                    ax.scatter(x[mask_det], y[mask_det], c=colors[label], marker=markers[group], s=ss_detected, edgecolors="k", linewidths=0.75, alpha=alpha_detected, zorder=3)
        setup_axes(ax, xlabel, ylabel, xlim, ylim, title)

    def plot_fraction_panel(ax, x, y, valid, xlabel, ylabel, xlim, ylim, title, space):
        edges_x = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), int(round(bins_per_decade_x * np.log10(xlim[1] / xlim[0]))) + 1)
        edges_y = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), int(round(bins_per_decade_y * np.log10(ylim[1] / ylim[0]))) + 1)
        N_tot, _, _ = np.histogram2d(x[valid],            y[valid],            bins=[edges_x, edges_y])
        N_det, _, _ = np.histogram2d(x[valid & detected], y[valid & detected], bins=[edges_x, edges_y])
        keep        = N_tot >= max(min_planets_per_bin, 1)
        frac        = np.where(keep, N_det / np.where(keep, N_tot, 1), np.nan)
        mesh        = ax.pcolormesh(edges_x, edges_y, np.ma.masked_invalid(frac.T), cmap=cmap_frac, norm=norm_frac, edgecolors="0.6", linewidth=0.5, zorder=1)
        for ix, iy in zip(*np.nonzero(keep)):
            xc, yc = np.sqrt(edges_x[ix] * edges_x[ix+1]), np.sqrt(edges_y[iy] * edges_y[iy+1])
            ax.text(xc, yc, f"{100*frac[ix, iy]:.0f}%\n{int(N_det[ix, iy])}/{int(N_tot[ix, iy])}", ha="center", va="center", fontsize=fontsize-8, color="w" if frac[ix, iy] < 0.6 else "k", zorder=3)
        setup_axes(ax, xlabel, ylabel, xlim, ylim, title)
        ax.grid(False)

        # Marginal histograms of detected planets along both axes
        divider = make_axes_locatable(ax)
        ax_top  = divider.append_axes("top",   size="18%", pad=0.08, sharex=ax)
        ax_right = divider.append_axes("right", size="18%", pad=0.08, sharey=ax)
        hist_x_det = N_det.sum(axis=1)
        hist_y_det = N_det.sum(axis=0)
        ax_top.stairs(np.maximum(hist_x_det, 0.9), edges_x, baseline=0.9, fill=True, color=cmap_frac(0.6), edgecolor="k", linewidth=1)
        ax_right.stairs(np.maximum(hist_y_det, 0.9), edges_y, baseline=0.9, orientation="horizontal", fill=True, color=cmap_frac(0.6), edgecolor="k", linewidth=1)
        ax.set_title("")
        ax_top.set_title(title, fontsize=fontsize+2, weight="bold", pad=10)
        for a in (ax_top, ax_right):
            a.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45, zorder=-10)
            a.tick_params(axis="both", which="major", labelsize=fontsize-4, length=4)
            a.tick_params(axis="both", which="minor", labelbottom=False, labelleft=False)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_top.set_ylabel("N", fontsize=fontsize)
        ax_right.set_xlabel("N", fontsize=fontsize)
        ax_top.set_yscale("log")
        ax_top.set_ylim(0.9, max(hist_x_det.max(), 1) * 1.5)
        ax_right.set_xscale("log")
        ax_right.set_xlim(0.9, max(hist_y_det.max(), 1) * 1.5)
        return mesh

    def plot_snr_panel(ax, x, y, valid, xlabel, ylabel, xlim, ylim, title, space):
        groups, masks, markers = get_regime_marker_groups()
        for group in groups:
            mask = valid & not_detected & masks[group]
            if np.any(mask):
                ax.scatter(x[mask], y[mask], c="0.70", marker=markers[group], s=ss_nondetected, edgecolors="none", alpha=alpha_nondetected, zorder=1)
        for group in groups:
            idx = np.flatnonzero(valid & detected & masks[group])
            if idx.size == 0:
                continue
            idx = idx[np.argsort(SNR_plot[idx])] # highest SNR drawn on top
            ax.scatter(x[idx], y[idx], c=SNR_plot[idx], cmap=cmap_snr, norm=norm_snr, marker=markers[group], s=ss_detected, edgecolors="k", linewidths=0.75, alpha=alpha_detected, zorder=3)
        setup_axes(ax, xlabel, ylabel, xlim, ylim, title)

    def make_panels(title, plot_func, valid_obs_plot=None, valid_phys_plot=None, subtitle=None, legends=True, marker_legend="method"):
        # One panel per requested space: wide figure for both spaces, narrower one for a single space
        valid_obs_plot  = valid_obs if valid_obs_plot is None else valid_obs_plot
        valid_phys_plot = valid_phys if valid_phys_plot is None else valid_phys_plot
        if space == "both":
            fig, axes = plt.subplots(1, 2, figsize=(22, 10) if instru is not None else (21, 9), dpi=dpi)
            ax_obs, ax_phys = axes
            margins   = dict(left=0.07, right=0.97, wspace=0.23)
        else:
            fig, ax   = plt.subplots(1, 1, figsize=(12, 10) if instru is not None else (11, 9), dpi=dpi)
            axes      = np.array([ax])
            ax_obs    = ax if space == "obs" else None
            ax_phys   = ax if space == "phys"  else None
            margins   = dict(left=0.13, right=0.95)
        if ax_obs is not None:
            plot_func(ax_obs, AngSep, contrast, valid_obs_plot, "Angular separation [mas]", f"{band_contrast_plot}-band contrast", obs_xlim, obs_ylim, "Observational space", "obs")
            if DL_mas is not None:
                ax_obs.axvline(DL_mas, color="k", linestyle="--", linewidth=2.5, zorder=4)
                ax_obs.annotate(DL_text, xy=(DL_mas, 0.8), xycoords=ax_obs.get_xaxis_transform(), xytext=(10, 0), textcoords="offset points", rotation=270, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="k", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2}, zorder=5)
            if show_VLT_DL:
                ax_obs.axvline(30, color="k", linestyle="--", linewidth=2, alpha=0.5, zorder=4)
                ax_obs.annotate("VLT", xy=(30, 0.8), xycoords=ax_obs.get_xaxis_transform(), xytext=(10, 0), textcoords="offset points", rotation=270, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="k", alpha=0.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2}, zorder=5)
            if show_det_regions:
                add_detection_method_regions(ax_obs)
            add_golden_sample(ax_obs, AngSep, contrast, valid_obs_plot, space="obs")
        if ax_phys is not None:
            plot_func(ax_phys, SMA, PlanetMass, valid_phys_plot, "Semi-major axis [AU]", r"Planet mass [$M_\oplus$]", phys_xlim, phys_ylim, "Physical space", "phys")
            if show_ptype_regions:
                add_ptype_regions(ax_phys)
            add_golden_sample(ax_phys, SMA, PlanetMass, valid_phys_plot, space="phys")
        if legends:
            # if instru is not None:
            #     add_detection_legend(axes[0],    loc="lower left")
            loc = "upper left" if space == "obs" else "lower right"
            if marker_legend == "regime":
                add_light_regime_legend(axes[-1], loc=loc)
            else:
                add_detection_method_legend(axes[-1], loc=loc)
        if instru is not None:
            fig.suptitle(title, fontsize=fontsize+5, weight="bold", y=0.98)
            if subtitle is not None:
                fig.text(0.5, 0.905, subtitle, ha="center", va="center", fontsize=fontsize+2, fontweight="normal")
        fig.subplots_adjust(bottom=0.10, top=0.82 if instru is not None else 0.92, **margins)
        return fig, axes

    def save_and_show(fig, filename):
        if save_dir is not None:
            if space != "both":
                filename = f"{Path(filename).stem}_{space}{Path(filename).suffix}"
            fig.savefig(save_dir / filename, bbox_inches="tight", dpi=dpi)
        plt.show(block=False)

    # Load tables and compute SNR, unless quantities were already computed
    if instru is None:
        SNR_plot = None
    elif not precomputed:
        loaded_labels, SNR_configs, dominant_configs = [], [], []
        SNR_bands_configs = {band: [] for band in config_data["gratings"]}
        ref_names         = None
        for cfg in configs:
            filename         = get_filename_table(table=table, instru=instru, apodizer=cfg["apodizer"], strehl=strehl, coronagraph=cfg["coronagraph"], systematics=systematics, PCA=PCA, name_model=name_model)
            planet_table_cfg = load_planet_table(filename, table=table)
            names            = np.asarray(planet_table_cfg["PlanetName"], dtype=str)
            if ref_names is None:
                ref_names    = names.copy()
                planet_table = planet_table_cfg.copy()
            elif len(names) != len(ref_names) or np.any(names != ref_names):
                raise ValueError(f"Inconsistent planet ordering/content in {filename}.")
            SNR_configs.append(get_SNR_from_table(planet_table=planet_table_cfg, exposure_time=exposure_time, band=band_snr))
            dominant_configs.append(get_dominant_noise(planet_table_cfg, band_snr))
            loaded_labels.append(cfg["label"])
            for band in config_data["gratings"]:
                SNR_bands_configs[band].append(get_SNR_from_table(planet_table=planet_table_cfg, exposure_time=exposure_time, band=band))
        SNR_stack     = np.stack(SNR_configs, axis=0)
        SNR_stack_max = np.where(np.isfinite(SNR_stack), SNR_stack, -np.inf)
        best_idx      = np.argmax(SNR_stack_max, axis=0)
        SNR_plot      = np.max(SNR_stack_max, axis=0)
        SNR_plot[~np.isfinite(SNR_plot) | (SNR_plot == -np.inf)] = np.nan
        planet_table["SNR"] = SNR_plot
        if len(loaded_labels) == 1:
            planet_table["BestConfig"] = np.full(len(planet_table), loaded_labels[0], dtype="U128")
        else:
            planet_table["BestConfig"] = np.asarray([loaded_labels[i] if np.isfinite(SNR_plot[ip]) else "" for ip, i in enumerate(best_idx)], dtype="U128")
        for band in config_data["gratings"]:
            SNR_band_stack = np.stack(SNR_bands_configs[band], axis=0)
            SNR_band       = np.max(np.where(np.isfinite(SNR_band_stack), SNR_band_stack, -np.inf), axis=0)
            SNR_band[~np.isfinite(SNR_band) | (SNR_band == -np.inf)] = np.nan
            planet_table[f"SNR_{band}"] = SNR_band
        dominant_stack = np.stack(dominant_configs, axis=0)
        dominant_noise = np.asarray([dominant_stack[best_idx[ip], ip] if np.isfinite(SNR_plot[ip]) else "Unknown" for ip in range(len(planet_table))], dtype=object)
    else:
        planet_table = planet_table.copy()
        SNR_plot = np.asarray(SNR_plot, dtype=float)
        dominant_noise = np.full(len(planet_table), "Unknown", dtype=object) if dominant_noise is None else np.asarray(dominant_noise, dtype=object)
        if SNR_plot.shape != (len(planet_table),):
            raise ValueError(f"SNR_plot must have shape ({len(planet_table)},), got {SNR_plot.shape}.")
        if dominant_noise.shape != (len(planet_table),):
            raise ValueError(f"dominant_noise must have shape ({len(planet_table)},), got {dominant_noise.shape}.")
        planet_table["SNR"]           = SNR_plot
        planet_table["DominantNoise"] = dominant_noise
    detected                      = np.ones(len(planet_table), dtype=bool)  if instru is None else np.isfinite(SNR_plot) & (SNR_plot >= SNR_thr)
    not_detected                  = np.zeros(len(planet_table), dtype=bool) if instru is None else np.isfinite(SNR_plot) & (SNR_plot <  SNR_thr)
    if instru is not None:
        planet_table["Detected"]      = detected
        planet_table["DominantNoise"] = dominant_noise

    golden_indices = {}
    if show_golden_sample:
        planet_names = np.asarray(planet_table["PlanetName"], dtype=str)
        for planet_name in golden_sample:
            if planet_name in planet_names:
                golden_indices[planet_name] = get_planet_index(planet_table=planet_table, planet_name=planet_name)

    # Data and masks
    AngSep     = arr("AngSep")
    SMA        = arr("SMA")
    PlanetMass = arr("PlanetMass")
    Teff_star  = arr("StarTeff")
    PlanetTeff = arr("PlanetTeff")
    Pmag       = get_combined_planet_mag(band_contrast_plot)
    if band_contrast_plot == "INSTRU":
        Smag   = np.asarray(planet_table[f"StarINSTRUmag({instru})"], dtype=float)
    else:
        Smag   = np.asarray(planet_table[f"Star{band_contrast_plot}mag"], dtype=float)
    contrast   = 10**(-(Pmag - Smag) / 2.5)

    valid_obs              = np.isfinite(AngSep) & np.isfinite(contrast) & np.isfinite(Teff_star) & (AngSep > 0) & (contrast > 0)
    valid_phys             = np.isfinite(SMA) & np.isfinite(PlanetMass) & np.isfinite(Teff_star) & (SMA > 0) & (PlanetMass > 0)
    valid_obs_planet_teff  = np.isfinite(AngSep) & np.isfinite(contrast) & np.isfinite(PlanetTeff) & (AngSep > 0) & (contrast > 0) & (PlanetTeff > 0)
    valid_phys_planet_teff = np.isfinite(SMA) & np.isfinite(PlanetMass) & np.isfinite(PlanetTeff) & (SMA > 0) & (PlanetMass > 0) & (PlanetTeff > 0)
    masks_ptype            = {ptype: get_ptype_mask(ptype) for ptype in ptypes_plot}

    if "DiscoveryMethod" in planet_table.colnames:
        discovery_method = np.asarray(planet_table["DiscoveryMethod"], dtype=str)
        masks_detection_method = {
            "Transit": discovery_method == "Transit",
            "Radial Velocity": discovery_method == "Radial Velocity",
            "Imaging": discovery_method == "Imaging",
            "Other": ~np.isin(discovery_method, ["Transit", "Radial Velocity", "Imaging"]),
        }
    else:
        detection_methods_plot   = ["Other"]
        marker_detection_methods = {"Other": "o"}
        label_detection_methods  = {"Other": "All planets"}
        masks_detection_method   = {"Other": np.ones(len(planet_table), dtype=bool)}

    if band_regime_plot == "INSTRU":
        mag_th     = np.asarray(planet_table[f"PlanetINSTRUmag({instru})(thermal)"],   dtype=float)
        mag_re     = np.asarray(planet_table[f"PlanetINSTRUmag({instru})(reflected)"], dtype=float)
    else:
        mag_th     = np.asarray(planet_table[f"Planet{band_regime_plot}mag(thermal)"],   dtype=float)
        mag_re     = np.asarray(planet_table[f"Planet{band_regime_plot}mag(reflected)"], dtype=float)
    mask_thermal   = np.isfinite(mag_th) & np.isfinite(mag_re) & (mag_th < mag_re)
    mask_reflected = np.isfinite(mag_th) & np.isfinite(mag_re) & (mag_re < mag_th)

    light_class                 = np.full(len(planet_table), "Unclassified", dtype=object)
    light_class[mask_thermal]   = "Thermal"
    light_class[mask_reflected] = "Reflected"
    planet_table["LightRegime"] = light_class

    # Plot style
    cmap_teff = plt.get_cmap("afmhot")
    norm_teff = mpl.colors.Normalize(vmin=2000, vmax=10_000, clip=True)
    mask_planet_teff_color = detected & np.isfinite(PlanetTeff) & (PlanetTeff > 0)
    if not np.any(mask_planet_teff_color):
        raise ValueError("No detected planet has a finite strictly positive PlanetTeff.")
    PlanetTeffMin    = np.nanmin(PlanetTeff[mask_planet_teff_color])
    PlanetTeffMax    = 3000 # np.nanmax(PlanetTeff[mask_planet_teff_color])
    cmap_planet_teff = plt.get_cmap("plasma")
    norm_planet_teff = mpl.colors.Normalize(vmin=PlanetTeffMin, vmax=PlanetTeffMax, clip=True)

    N_total        = len(planet_table)
    N_det          = int(np.sum(detected))

    # 1) Detected population
    fig_pop, axes_pop = make_panels(title=f"ELT/{instru} detected population in {exposure_time/60:.0f} hr per target", subtitle=f"{N_det:.0f}/{N_total:.0f} detections ({N_det/N_total*100:.1f}%)", plot_func=plot_population_panel)
    sm                = mpl.cm.ScalarMappable(norm=norm_teff, cmap=cmap_teff)
    sm.set_array([])
    cbar = fig_pop.colorbar(sm, ax=axes_pop, pad=cbar_frac, fraction=cbar_frac)
    cbar.set_label(r"Star $T_\mathrm{eff}$ [K]", fontsize=fontsize+2, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=fontsize-2)
    save_and_show(fig_pop, f"ELT_{instru}{type_and_pp}_detected_population_{table}_{band_contrast_plot}band.png")

    # 2) Detected population colored by planet temperature
    fig_planet_teff, axes_planet_teff = make_panels(title=f"ELT/{instru} detected population in {exposure_time/60:.0f} hr per target", subtitle=f"{N_det:.0f}/{N_total:.0f} detections ({N_det/N_total*100:.1f}%)", plot_func=plot_population_planet_teff_panel, valid_obs_plot=valid_obs_planet_teff, valid_phys_plot=valid_phys_planet_teff)
    sm = mpl.cm.ScalarMappable(norm=norm_planet_teff, cmap=cmap_planet_teff)
    sm.set_array([])
    cbar = fig_planet_teff.colorbar(sm, ax=axes_planet_teff, pad=cbar_frac, fraction=cbar_frac)
    cbar.set_label(r"Planet $T_\mathrm{eff}$ [K]", fontsize=fontsize+2, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=fontsize-2)
    save_and_show(fig_planet_teff, f"ELT_{instru}{type_and_pp}_detected_population_planet_teff_{table}_{band_contrast_plot}band.png")

    # 3) Dominant noise regime
    if instru is not None:
        noise_class  = dominant_noise.copy()
        noise_order  = ["Stellar halo",               "Background",             "Read noise",               "Dark current",              "Speckles",            "Residuals",            "Systematics",            "Fundamental noise",               "Unknown"]
        noise_colors = {"Stellar halo": "tab:orange", "Background": "tab:blue", "Read noise": "tab:purple", "Dark current": "tab:green", "Speckles": "tab:red", "Residuals": "tab:red", "Systematics": "tab:red", "Fundamental noise": "tab:orange", "Unknown": "0.50"}
        noise_labels = [label for label in noise_order if np.any(noise_class == label)]
        noise_alphas              = {label: 0.20 if label == "Non-detected" else alpha_detected for label in noise_labels}
        fig_noise, axes_noise     = make_panels(title=f"ELT/{instru} dominant noise regime in {exposure_time/60:.0f} hr per target", subtitle=f"{N_det:.0f}/{N_total:.0f} detections ({N_det/N_total*100:.1f}%)", plot_func=lambda ax, x, y, valid, xlabel, ylabel, xlim, ylim, title, space: plot_class_panel(ax, x, y, valid, noise_class, noise_labels, noise_colors, noise_alphas, xlabel, ylabel, xlim, ylim, title, space))
        N_det_noise               = {label: int(np.sum(detected & (noise_class == label))) for label in noise_labels}
        handles_noise             = [Line2D([], [], ls="", marker="o", ms=11, markerfacecolor=noise_colors[lab], markeredgecolor="k", color=noise_colors[lab], label=f"{lab} ({N_det_noise[lab]})") for lab in noise_labels if lab != "Non-detected"]
        loc = "upper right" if space == "obs" else "center right"
        axes_noise[-1].add_artist(axes_noise[-1].legend(handles=handles_noise, fontsize=fontsize-2, loc=loc, frameon=True, edgecolor="gray", facecolor="white", title="Dominant noise", title_fontsize=fontsize))
        save_and_show(fig_noise, f"ELT_{instru}{type_and_pp}_dominant_noise_{table}_{band_contrast_plot}band.png")

    # 4) Thermal/reflected complementarity
    light_labels = ["Thermal", "Reflected"] + (["Unclassified"] if np.any(light_class == "Unclassified") else [])
    light_colors = {"Thermal": "C3", "Reflected": "C0", "Unclassified": "0.50"}
    light_alphas = {"Non-detected": 0.20, "Detected, thermal-dominated": alpha_detected, "Detected, reflected-dominated": alpha_detected, "Detected, unclassified": alpha_detected}
    fig_light, axes_light = make_panels(title=f"ELT/{instru} thermal/reflected complementarity in {exposure_time/60:.0f} hr", subtitle=f"{N_det:.0f}/{N_total:.0f} detections ({N_det/N_total*100:.1f}%)", plot_func=lambda ax, x, y, valid, xlabel, ylabel, xlim, ylim, title, space: plot_class_panel(ax, x, y, valid, light_class, light_labels, light_colors, light_alphas, xlabel, ylabel, xlim, ylim, title, space))
    N_det_light           = {label: int(np.sum(detected & (light_class == label))) for label in light_labels}
    handles_light         = [Line2D([], [], ls="", marker="o", ms=11, markerfacecolor=light_colors["Thermal"], markeredgecolor="k", color=light_colors["Thermal"], label=f"Thermal ({N_det_light['Thermal']})"),  Line2D([], [], ls="", marker="o", ms=11, markerfacecolor=light_colors["Reflected"], markeredgecolor="k", color=light_colors["Reflected"], label=f"Reflected ({N_det_light['Reflected']})")]
    loc = "upper right" if space == "obs" else "center right"
    axes_light[-1].add_artist(axes_light[-1].legend(handles=handles_light, fontsize=fontsize-2, loc=loc, frameon=True, edgecolor="gray", facecolor="white", title=f"Planet-light regime ({band_regime_plot})", title_fontsize=fontsize))
    save_and_show(fig_light, f"ELT_{instru}{type_and_pp}_thermal_reflected_{table}_{band_regime_plot}band.png")

    # 5) Detection fraction per bin (log grid), in observational and physical spaces
    bins_per_decade_x   = 1 # bins per decade along x
    bins_per_decade_y   = 1 # bins per decade along y
    min_planets_per_bin = 1 # bins with fewer planets are left blank
    cmap_frac           = plt.get_cmap("inferno").copy()
    cmap_frac.set_bad("white")
    norm_frac           = mpl.colors.Normalize(vmin=0, vmax=1)
    fig_frac, axes_frac = make_panels(title=f"ELT/{instru} detection fraction in {exposure_time/60:.0f} hr per target", subtitle=f"{N_det:.0f}/{N_total:.0f} detections ({N_det/N_total*100:.1f}%)", plot_func=plot_fraction_panel, legends=False)
    sm                  = mpl.cm.ScalarMappable(norm=norm_frac, cmap=cmap_frac)
    sm.set_array([])
    cbar = fig_frac.colorbar(sm, ax=axes_frac, pad=cbar_frac, fraction=cbar_frac)
    cbar.set_label("Fraction of detected planets", fontsize=fontsize+2, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=fontsize-2)
    save_and_show(fig_frac, f"ELT_{instru}{type_and_pp}_detection_fraction_{table}_{band_contrast_plot}band.png")

    # 6) Population colored by final SNR
    if instru is not None:
        mask_snr_color = np.isfinite(SNR_plot) & (SNR_plot > 0)
        if np.any(mask_snr_color):
            SNR_min  = 4
            SNR_max  = max(np.nanpercentile(SNR_plot[mask_snr_color], 99), 10 * SNR_min)
            cmap_snr = plt.get_cmap("rainbow")
            norm_snr = mpl.colors.LogNorm(vmin=SNR_min, vmax=SNR_max, clip=True)
            marker_light_regimes = {"Thermal": "o", "Reflected": "D", "Unclassified": "X"}
            fig_snr, axes_snr = make_panels(title=f"ELT/{instru} final SNR in {exposure_time/60:.0f} hr per target", subtitle=f"{N_det:.0f}/{N_total:.0f} detections ({N_det/N_total*100:.1f}%)", plot_func=plot_snr_panel, marker_legend="regime")
            sm   = mpl.cm.ScalarMappable(norm=norm_snr, cmap=cmap_snr)
            sm.set_array([])
            cbar = fig_snr.colorbar(sm, ax=axes_snr, pad=cbar_frac, fraction=cbar_frac, extend="max")
            cbar.set_label("SNR", fontsize=fontsize+2, rotation=270, labelpad=30)
            cbar.ax.yaxis.set_major_formatter(PlainLogFormatter())
            cbar.ax.yaxis.set_minor_formatter(PlainLogFormatter())
            cbar.ax.tick_params(labelsize=fontsize-2)
            if SNR_min <= SNR_thr <= SNR_max:
                cbar.ax.axhline(SNR_thr, color="k", linestyle="--", linewidth=2.5)
            save_and_show(fig_snr, f"ELT_{instru}{type_and_pp}_SNR_{table}_{band_contrast_plot}band.png")


