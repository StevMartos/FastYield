# import FastYield modules
from .config import c, R0_max
from .utils import cut_spectral_frequencies, get_logL
from .spectrum import get_wavelength_axis_constant_dl, Spectrum, get_resolution, filtered_flux, get_psd, load_star_spectrum, load_planet_spectrum, load_albedo_spectrum, load_mol_spectrum, get_model_grid, get_spectrum_band

# import astropy modules
from astropy.io import fits
from astropy.stats import sigma_clip

# import matplotlib modules
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d

# import other modules
from math import erf
import hashlib
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# For fits warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore",   category=VerifyWarning)
warnings.filterwarnings("ignore", message="Header block contains null bytes*")



# -------------------------------------------------------------------------
# Data spectra modelisations
# -------------------------------------------------------------------------

def get_valid_mask(data, template, weight=None):
    # Creating mask
    valid = (data !=0 ) & np.isfinite(data) & np.isfinite(template)
    if weight is not None:
        valid &= np.isfinite(weight) & (weight != 0)
    return valid



def get_masked_quantity(data, template, weight=None):
    # Creating mask
    valid = get_valid_mask(data=data, template=template, weight=weight)
    # Masking
    data_masked             = np.copy(data)
    template_masked         = np.copy(template)
    data_masked[~valid]     = np.nan
    template_masked[~valid] = np.nan
    return data_masked, template_masked



def get_pca_subtracted_data(data, pca):
    d  = np.copy(data)
    d0 = np.copy(d)
    for nk in range(pca.n_components):
        d -= np.nan_to_num(np.nansum(d0 * pca.components_[nk]) * pca.components_[nk])
    return d



def get_Ss_HF_LF(trans_Ss, trans, wave, R_sampling, Rc, filter_type):
    Ss           = trans_Ss / trans
    valid        = np.isfinite(Ss)
    Ss           = interp1d(wave[valid], Ss[valid], bounds_error=False, fill_value=np.nan)(wave) # Handling NaN
    Ss_HF, Ss_LF = filtered_flux(flux=Ss, R=R_sampling, Rc=Rc, filter_type=filter_type)
    return Ss_HF, Ss_LF



def get_template(instru, wave, R, model, T, lg, rv, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, to_counts=True):
    """
    Build a high-resolution template Spectrum, then (optionally) broaden and Doppler-shift it.

    Steps
    -----
    1) Load a model spectrum (tellurics / stellar / planetary).
    2) (Molecular models only) Optionally crop featureless regions by sigma-clipping the LF
       component after a temporary high/low-pass at the native resolution.
    3) Interpolate to a *regular* wavelength grid 'wave_model' whose sampling corresponds to
       ~Nyquist at min(max(template.R, 2*R), R0_max).
    4) Apply rotational broadening (vsini) and Doppler shift (rv).
    5) For non-molecular models, multiply by wavelength to make the template homogeneous to
       photon-like units.

    Parameters
    ----------
    instru : str
        Instrument name (passed through to planetary model loader if needed).
    wave : (N,) array_like
        Target wavelength span (µm). Only the min/max are used to define 'wave_model' if not
        provided.
    model : str
        Model identifier: "tellurics", any of {"BT-NextGen", "Husser"} (stellar),
        or planetary/molecular names (e.g., "mol_CO", "BT-Settl", etc).
    T : float
        Effective temperature used to select the model (K).
    lg : float
        Surface gravity log10(g[cm s^-2]) used to select the model.
    rv : float
        Doppler shift (km/s).
    vsini : float
        Rotational broadening (km/s).
    R : float
        Instrumental resolving power of the *data* (used to choose a working interpolation grid).
    epsilon : float, optional
        Limb-darkening parameter for rotational broadening.
    fastbroad : bool, optional
        Use fast convolutional broadening.
    wave_model : (M,) array_like or None, optional
        Explicit regular wavelength grid (µm) to interpolate the template onto. If None, a
        regular grid is built from 0.98*min(wave) .. 1.02*max(wave) with Nyquist sampling at
        R_interp = min(max(template.R, 2*R), R0_max).
    
    Returns
    -------
    template : Spectrum
        High-resolution template, interpolated, broadened and Doppler-shifted. For non-molecular
        models, its flux is multiplied by wavelength.
    """
    
    # 1) Load model
    if model in {"BT-NextGen", "Husser"}:
        template = load_star_spectrum(T_star=T, lg_star=lg, model=model)
    elif model[:4] == "mol_":
        template = load_mol_spectrum(T_mol=T, P_mol=1.0, model=model)
    elif "albedo" in model.lower():
        albedo_spectrum = load_albedo_spectrum(T_planet=T, lg_planet=lg, model=model, airmass=airmass)
        albedo_spectrum = albedo_spectrum.interpolate_wavelength(star_spectrum.wavelength, renorm=False) # Interpolating on wave_model (constant dl)
        template        = Spectrum(wavelength=star_spectrum.wavelength, flux=albedo_spectrum.flux*star_spectrum.flux, R=albedo_spectrum.R, T=T, lg=lg, model=model, rv=0, vsini=0)
    else:
        template = load_planet_spectrum(T_planet=T, lg_planet=lg, model=model, instru=instru)
    
    # 2) Optional LF-based cropping for molecular grids: to crop empty features regions in molecular templates
    if model[:4] == "mol_":
        _, template_LF          = filtered_flux(template.flux, R=get_resolution(wavelength=template.wavelength, func=np.nanmedian), Rc=1_000, filter_type="gaussian")
        sg                      = sigma_clip(np.ma.masked_invalid(template_LF), sigma=1)
        template.flux[~sg.mask] = np.nan 
    
    # 3) Interpolation grid (regular, Nyquist-like)
    if wave_model is None:
        R_interp    = min(max(np.nanmax(template.R), 2*np.nanmax(R)), R0_max)
        lmin_interp = 0.98*wave[0]
        lmax_interp = 1.02*wave[-1]        
        wave_model  = get_wavelength_axis_constant_dl(lmin=lmin_interp, lmax=lmax_interp, R=R_interp) # Regularly sampled template
    
    # Interpolate to the interpolation grid
    template = template.interpolate_wavelength(wave_model, renorm=False)
    
    # 4) Broadening & Doppler shift
    template = template.broad(vsini, epsilon=epsilon, fastbroad=fastbroad)    
    template = template.doppler_shift(rv, renorm=False)
    
    # 5) To be homogenous to counts: ph/bin or e-/bin or ADU/bin
    if to_counts:
        template.flux *= wave_model * np.gradient(wave_model)
    
    return template



def get_d_sim(instru, d, wave, trans, R, Rc, filter_type, model, T, lg, rv, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, pca=None, cut_fringes=False, Rmin=None, Rmax=None, target_name=None, renorm_d_sim=True, sigma_l=None, Mp=None, verbose=True):
    """"
    Build a *simulated* high-pass signal vector 'd_sim' matched to the data.

    Pipeline
    --------
    1) Build template at high resolution (or use provided 'template'), broaden and Doppler shift.
    2) Degrade to instrumental resolution (or interpolate to the data grid).
    3) Split into HF/LF with the chosen filter; start from HF component scaled by 'trans'.
    4) (Optional) Add the residual stellar term (MM formalism) using 'trans_Ss'.
    5) (Optional) Project out PCA modes from the simulated vector.
    6) (Optional) Notch out fringe bands in R-space.
    7) Mask like the data, L2-normalize, then rescale so that its PSD matches the data PSD
       in the HF band (Rc .. R/10 by default).

    Parameters
    ----------
    d : (N,) array_like
        Observed **high-pass** data vector (used for masking and PSD matching).
    wave : (N,) array_like
        Wavelength grid (µm).
    trans : (N,) array_like
        Total end-to-end transmission on 'wave'.
    model, T, lg, vsini, rv : see 'get_template'.
    R : float
        Instrument resolving power for the data.
    Rc : float or None
        High/low-pass cut-off in resolving-power units (consistent with the data processing).
    filter_type : str
        Filter name passed to 'filtered_flux'.
    degrade_resolution : bool
        If True, call 'Spectrum.degrade_resolution(..., R_output=R)', else only interpolate to 'wave'.
    stellar_component : bool
        If True and 'Rc' is not None, add the residual stellar term using 'trans_Ss'.
    pca : PCA-like or None, optional
        If provided, subtract its components from the simulated vector (template-side projection).
    trans_Ss : (N,) array_like or None, optional
        Stellar flux on 'wave' before transmission; mandatory if 'stellar_component=True'.
    instru : str or None, optional
        Used for instrument-specific tweaks (e.g., HiRISE order gaps handling).
    epsilon, fastbroad : see 'get_template'.
    cut_fringes : bool, optional
        If True, apply 'cut_spectral_frequencies' with (Rmin, Rmax).
    Rmin, Rmax : float or None, optional
        Notch limits in resolving-power units for 'cut_fringes'.
    target_name : str or None, optional
        Only used for logging/plotting inside helpers.
    corner_plot : bool, optional
        If True, use a simple dot-product scaling; otherwise use PSD-based scaling in (Rc, R/10).
    template : Spectrum or None, optional
        Pre-built template to start from (before instrument degradation).
    wave_model : (M,) array_like or None, optional
        If 'template is None', forwarded to 'get_template' to control the high-res working grid.
    sigma_l : (N,) array_like or None, optional
        Per-channel noise (same grid as 'wave'). Needed to return analytical 'sigma'.

    Returns
    -------
    d_sim : (N,) ndarray
        Simulated, masked, normalized, and PSD-matched vector to compare against 'd'.
    """
    
    R_sampling = get_resolution(wavelength=wave, func=np.array)
    
    if verbose:
        print()
    
    # 1) Template creation (or copy if injected)
    if template is None:
        template = get_template(instru=instru, wave=wave, R=R, model=model, T=T, lg=lg, rv=rv, vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model)
        if verbose:
            print(f" get_d_sim: get_template: T={template.T:.0f}K, lg={template.lg:.2f}, rv={template.rv:.2f}km/s, vinsi={template.vsini:.2f}km/s...")
    else:
        template = template.copy()
        
    # 2) Match instrument sampling
    if verbose:
        print(f" get_d_sim: Degrading the resolution on 'wave' (degrade_resolution={degrade_resolution})...")
    Sp = get_spectrum_band(spectrum_instru=template, wave_band=wave, R_output=R, degrade_resolution=degrade_resolution, verbose=False)
    
    # 2') If the modulation Mp is given, taking it into account
    if Mp is not None:
        Sp.flux *= Mp
    
    # 3) HF/LF split of the planetary spectrum
    if verbose:
        print(f" get_d_sim: HF/LF split (Rc={Rc}, filter_type={filter_type}...")
    Sp_HF, Sp_LF = filtered_flux(flux=Sp.flux, R=R_sampling, Rc=Rc, filter_type=filter_type)
    d_sim        = trans * Sp_HF

    # 4) Residual stellar term (MM formalism), if requested
    if stellar_component and Rc is not None:
        if verbose:
            print(" get_d_sim: Adding the residual stellar component...")
        Ss_HF, Ss_LF = get_Ss_HF_LF(trans_Ss=trans_Ss, trans=trans, wave=wave, R_sampling=R_sampling, Rc=Rc, filter_type=filter_type)
        d_sim       += - trans * Ss_HF * Sp_LF / Ss_LF
    
    # 5) Optional notch (fringe) filtering
    if cut_fringes:
        if verbose:
            print(" get_d_sim: Cutting fringes...")
        d_sim = cut_spectral_frequencies(d_sim, R, Rmin, Rmax, target_name=target_name)
    
    # 6) Optional mask like the data (if d is given)
    if d is not None:
        if verbose:
            print(" get_d_sim: Mask like the data...")
        d, d_sim = get_masked_quantity(data=d, template=d_sim, weight=None)
        
    # 7) Optional PCA modes removing (template-side projection)
    if pca is not None:
        if verbose:
            print(" get_d_sim: Subtracting PCA modes...")
        d_sim = get_pca_subtracted_data(data=d_sim, pca=pca)
        if d is not None:
            d, d_sim = get_masked_quantity(data=d, template=d_sim, weight=None)
    
    # 8) Optionnal renormalization
    if renorm_d_sim:
        if (sigma_l is not None) and (np.nansum(d**2) - np.nansum(sigma_l**2) >= 0):
            if verbose:
                print(" get_d_sim: Renormalizing with sigma_l...")
            norm_d_sim = np.sqrt(np.nansum(d**2) - np.nansum(sigma_l**2))
            scale      = norm_d_sim / np.sqrt(np.nansum(d_sim**2))
        else:
            if verbose:
                print(" get_d_sim: Renormalizing from PSD..")
            # # PSD-based amplitude matching (or dot-product scaling for corner plots)
            # res, psd_data = get_psd(wave, d,     smooth=0)
            # res, psd_sim  = get_psd(wave, d_sim, smooth=0)
            # # Match power between Rc and R/10 (or as close as possible within available range)
            # R_hi   = np.nanmedian(R) / 10
            # R_lo   = Rc if Rc else 0
            # mask_d = (res >= R_lo) & (res <= R_hi)
            # mask_s = (res >= R_lo) & (res <= R_hi)
            # # Use the common region if both masks are non-empty; otherwise fall back to all finite
            # p_data = np.nansum(psd_data[mask_d])
            # p_sim  = np.nansum(psd_sim[mask_s])
            # scale  = np.sqrt(p_data / p_sim)
            
            # TODO: test
            norm_d_sim = np.sqrt(np.nansum(d**2))
            scale      = norm_d_sim / np.sqrt(np.nansum(d_sim**2))
            
        d_sim *= scale
    
    return d_sim



# -------------------------------------------------------------------------
# Stellar filtering method
# -------------------------------------------------------------------------

def get_S_res(wave, S, Rc, filter_type, trans_Ss=None, outliers=False, sigma_outliers=5, renorm_S_res=False, only_high_pass=False, debug=False):
    """
    Stellar filtering for molecular mapping (Appendix B of Martos et al., 2025).

    Steps (per spaxel):
      - Estimate stellar LF modulation: M = S_LF / Ss_LF  (or M = S_LF / trans_Ss if Rc=None)
      - Residual: S_res = S - M * trans_Ss
      - Optional sigma-clipping on residuals
      - Optional normalization of residuals (per spaxel) to unit L2-norm

    Parameters
    ----------
    wave : (NbChannel,) array_like
        Wavelength grid (µm).
    S : ndarray, shape (NbChannel, NbLine, NbColumn)
        Data cube (can contain NaNs).
    Rc : float or None
        Cut-off resolving power. If None, no spectral filter is applied (LF=raw).
    filter_type : {"gaussian","gaussian_bis","step","smoothstep","savitzky_golay"}
        Low/High-pass filter family (see 'filtered_flux').
    trans_Ss : (NbChannel,) array_like or None
        Stellar reference spectrum. If None, it is estimated by summing 'cube' over (y,x).
    outliers : bool, optional
        If True, apply sigma clipping on the residual spectrum of each spaxel.
    sigma_outliers : float, optional
        Sigma threshold for clipping, if 'outliers=True'.
    renorm_S_res : bool, optional
        If True, L2-normalize the residual spectrum of each spaxel.
    only_high_pass : bool, optional
        If True, do NOT subtract a stellar component; only estimate LF to return M.
    debug : bool, optional
        If True, plot a decomposition/PSD example for a representative spaxel.

    Returns
    -------
    S_res : ndarray, shape (NbChannel, NbLine, NbColumn)
        Residual cube after filtering (NaN where input was empty).
    M : ndarray, shape (NbChannel, NbLine, NbColumn)
        Estimated stellar modulation per spaxel (LF / reference).
    """
    
    R_sampling                   = get_resolution(wavelength=wave, func=np.nanmedian)
    S                           = np.copy(S)
    NbChannel, NbLine, NbColumn = S.shape
    
    # Stellar reference spectrum
    if trans_Ss is None:
        trans_Ss = np.nansum(S, (1, 2)) # estimated stellar spectrum
    else:
        trans_Ss = trans_Ss
    
    # Flatten spatial dimensions to vectorize most operations
    S     = np.reshape(S, (NbChannel, NbLine*NbColumn))
    S_res = np.zeros_like(S) + np.nan
    M     = np.zeros_like(S) + np.nan
    
    # Work spaxel by spaxel (fast enough; per-column 1D filter)
    for i in range(S.shape[1]):
        
        S_i = S[:, i]
        
        if not all(~np.isfinite(S_i)):
            
            # Estimated modulations, assuming that trans_Ss is the real observed stellar spectrum
            if only_high_pass:
                M[:, i] = filtered_flux(flux=S_i, R=R_sampling, Rc=Rc, filter_type=filter_type)[1] / trans_Ss
            else:
                M[:, i] = filtered_flux(flux=S_i / trans_Ss, R=R_sampling, Rc=Rc, filter_type=filter_type)[1]
            
            # Filtered cube
            S_res[:, i] = S_i - trans_Ss * M[:, i]
            
            # Optional outlier clipping (on residual)
            if outliers and np.any(np.isfinite(S_res[:, i])):
                sg          = sigma_clip(np.ma.masked_invalid(S_res[:, i]), sigma=sigma_outliers)
                S_res[:, i] = np.array(np.ma.masked_array(sg, mask=sg.mask).filled(np.nan))
                
            # Sanity check of the filtering method
            if debug:
                d          = S_i.copy()
                d_HF, d_LF = filtered_flux(flux=d, R=R_sampling, Rc=Rc, filter_type=filter_type)
                d_res      = S_res[:, i].copy()
                d     /= np.sqrt(np.nansum(d**2))
                d_LF  /= np.sqrt(np.nansum(d_LF**2))
                d_HF  /= np.sqrt(np.nansum(d_HF**2))
                d_res /= np.sqrt(np.nansum(d_res**2))
                res, psd         = get_psd(wave=None, flux=d,     R=R_sampling, smooth=0)
                res_LF, psd_LF   = get_psd(wave=None, flux=d_LF,  R=R_sampling, smooth=0)
                res_HF, psd_HF   = get_psd(wave=None, flux=d_HF,  R=R_sampling, smooth=0)
                res_res, psd_res = get_psd(wave=None, flux=d_res, R=R_sampling, smooth=0)
                fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", gridspec_kw={'wspace': 0.05, 'hspace': 0}, dpi=300)
                ax[0].plot(d, color='steelblue', lw=1, label="Raw (LF+HF)", alpha=0.5)
                ax[0].plot(d_LF, color='crimson', lw=1, label="LF", alpha=0.5)
                ax[0].plot(d_HF, color='seagreen', lw=1, label="HF", alpha=0.5)
                ax[0].plot(d_res, color='black', lw=1, label="Post-MM filtering", alpha=0.5)
                ax[0].minorticks_on()
                ax[0].set_xlim(0, len(d))
                ax[0].set_xlabel("Wavelength axis", fontsize=12, labelpad=10)
                ax[0].set_ylabel("Modulation (normalized)", fontsize=12, labelpad=10)
                ax[0].tick_params(axis='both', labelsize=10)
                ax[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)                
                ax[1].plot(res, psd, color='steelblue', lw=1, label="Raw (LF+HF)", alpha=0.5)
                ax[1].plot(res_LF, psd_LF, color='crimson', lw=1, label="LF", alpha=0.5)
                ax[1].plot(res_HF, psd_HF, color='seagreen', lw=1, label="HF", alpha=0.5)
                ax[1].plot(res_res, psd_res, color='black', lw=1, label="Post-MM filtering", alpha=0.5)
                ax[1].set_xlim(res[res>0].min(), res[res>0].max())
                ax[1].set_xscale('log')
                ax[1].set_yscale('log')
                ax[1].set_xlabel("Resolution frequency R", fontsize=12, labelpad=10)
                ax[1].set_ylabel("Power Spectral Density (PSD)", fontsize=12, labelpad=10)
                ax[1].tick_params(axis='both', labelsize=10)
                ax[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)                
                ax[1].legend(fontsize=10, loc="lower left", frameon=True, edgecolor="gray", facecolor="whitesmoke")
                plt.suptitle("Signal decomposition and PSD analysis", fontsize=14, y=1.05)
                plt.show()
    
    # Reshape back to cube and turn exact zeros to NaN
    S_res             = S_res.reshape((NbChannel, NbLine, NbColumn))
    M                 = M.reshape((NbChannel, NbLine, NbColumn))
    S_res[S_res == 0] = np.nan
    M[M == 0]         = np.nan
    
    # Optional per-spaxel L2 normalization of residual spectra
    if renorm_S_res:
        # Compute ||S_res|| per spaxel -> (NbLine*NbColumn,)
        flat  = S_res.reshape(NbChannel, -1)
        norms = np.sqrt(np.nansum(flat**2, axis=0))
        norms = np.where(np.isfinite(norms) & (norms > 0), norms, np.nan)
        flat /= norms
        S_res = flat.reshape(NbChannel, NbLine, NbColumn)

    return S_res, M



# -------------------------------------------------------------------------
# 2D CCF maps computations (for different RV)
# -------------------------------------------------------------------------

def get_CCF_2D_rv(instru, S_res, wave, trans, R, Rc, filter_type, model, T, lg, rv_arr, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template_wo_shift=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, Ss_HF=None, Ss_LF=None, R_sampling=None, pca=None):
    """
    Cross-correlate the residual cube against Doppler-shifted templates (vectorized over pixels).

    Parameters
    ----------
    instru : str
        Instrument name (passed to template_wo_shift generator if needed).
    S_res : ndarray, shape (NbChannel, NbLine, NbColumn)
        Residual cube after stellar filtering (NaNs allowed).
    trans_Ss : (NbChannel,) array_like
        Stellar flux in electrons/min (or consistent units).
    T : float
        Temperature (K).
    lg : float
        log g [dex(cm/s^2)].
    model : str
        Model name (used by 'get_template').
    wave : (NbChannel,) array_like
        Wavelength grid (µm).
    trans : (NbChannel,) array_like
        Total system transmission.
    R : float
        Instrument resolving power.
    Rc : float or None
        Cut-off resolving power for HF/LF split in filtering.
    filter_type : str
        Filter family for HF/LF.
    rv_arr : float or array_like or None
        Radial-velocity grid (km/s). If None, defaults to [-50..50] km/s, step 0.5 (201 pts).
    vsini : float, optional
        Rotational broadening for the template_wo_shift (km/s).
    pca : fitted PCA or None, optional
        If provided, project and subtract PCA components from the template_wo_shift (to match data processing).
    degrade_resolution : bool, optional
        If True, degrade template_wo_shift to instrumental resolution R before matching.
    stellar_component : bool, optional
        If True and Rc is not None, include the stellar self-subtraction component in the template_wo_shift.
    epsilon : float, optional
        Template-generation parameter for 'get_template'.
    fastbroad : bool, optional
        Whether to use the fast broadening.
    template_wo_shift : Spectrum or None, optional
        Precomputed template_wo_shift. If None, it is generated.

    Returns
    -------
    CCF : ndarray
        If rv_arr is scalar: shape (NbLine, NbColumn)
        If rv_arr has multiple values: shape (Nrv, NbLine, NbColumn)
    rv_grid : None or ndarray
        None if rv_arr scalar, else the rv_arr grid used (km/s).
    """
    NbChannel, NbLine, NbColumn = S_res.shape
    Npix                        = NbLine * NbColumn

    if R_sampling is None:
        R_sampling = get_resolution(wavelength=wave, func=np.array)

    # 1) Template generation (if not provided) / alignment on wavelength grid
    if template_wo_shift is None:
        template_wo_shift = get_template(instru=instru, wave=wave, R=R, model=model, T=T, lg=lg, rv=0, vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model)
    
    # Degrade template_wo_shift to target grid
    Sp = get_spectrum_band(spectrum_instru=template_wo_shift, wave_band=wave, R_output=R, degrade_resolution=degrade_resolution, verbose=False)
    
    # Return if Sp only contains NaNs (e.g. if the crop of the molecular templates left only NaNs)
    if np.all(~np.isfinite(Sp.flux)):
        return None, None

    # 2) Build HF/LF parts of the Sp
    Sp_HF                  = Sp.copy()
    Sp_LF                  = Sp.copy()
    Sp_HF.flux, Sp_LF.flux = filtered_flux(flux=Sp.flux, R=R_sampling, Rc=Rc, filter_type=filter_type)
    
    # 3) Stellar HF/LF if needed
    if stellar_component and Rc is not None and (Ss_HF is None or Ss_LF is None):
        Ss_HF, Ss_LF = get_Ss_HF_LF(trans_Ss=trans_Ss, trans=trans, wave=wave, R_sampling=R_sampling, Rc=Rc, filter_type=filter_type)

    # 4) RV grid handling
    scalar_rv = np.ndim(rv_arr) == 0
    rv_arr    = np.atleast_1d(np.asarray(rv_arr, dtype=float))
    Nrv       = len(rv_arr)

    # 5) Prepare data matrix once
    S2D        = S_res.reshape(NbChannel, Npix)
    valid_data = np.isfinite(S2D)
    S0         = np.nan_to_num(S2D, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    M0         = valid_data.astype(np.float64, copy=False)

    # 6) Build all RV-shifted templates
    Tmat = np.full((Nrv, NbChannel), np.nan, dtype=np.float64)
    
    # 7) Loop over RV values (vectorize across all pixels inside)
    for k in range(len(rv_arr)):
        
        # Doppler shift Sp
        Sp_HF_shift = Sp_HF.doppler_shift(rv_arr[k], renorm=False).flux
        
        # Build final Sp including stellar component if requested
        t = trans * Sp_HF_shift 
        if stellar_component and Rc is not None: # Adding the stellar component (if required)
            Sp_LF_shift = Sp_LF.doppler_shift(rv_arr[k], renorm=False).flux    
            t          += - trans * Ss_HF * Sp_LF_shift / Ss_LF # it should be almost the same thing with or without the residual star flux
        
        # PCA projection and subtraction on the template (components are assumed on the *same* wavelength grid)
        if pca is not None:
            t = get_pca_subtracted_data(data=t, pca=pca)

        # Normalize global template once (will be re-masked per pixel)
        norm = np.sqrt(np.nansum(t**2))
        if not (np.isfinite(norm) and norm > 0):
            # Degenerate template at this RV => skip
            continue
        t /= norm  # normalized global template
        
        Tmat[k, :] = t.astype(np.float64)
    
    # 8) Remove invalid RV templates
    valid_rv = np.isfinite(Tmat).any(axis=1)
    if not np.any(valid_rv):
        return None, None
    T0 = np.nan_to_num(Tmat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    # 9) Fast matrix products
    numerator   = T0 @ S0
    denominator = np.sqrt((T0**2) @ M0)
    CCF2D       = np.full_like(numerator, np.nan, dtype=np.float64)
    good        = denominator > 0
    CCF2D[good] = numerator[good] / denominator[good]
    CCF         = CCF2D.reshape(Nrv, NbLine, NbColumn)
    
    # If rv_arr was scalar, return 2D and the (normalized) template used for that RV
    if scalar_rv:
        return CCF[0], t
    else:
        return CCF, rv_arr



# -------------------------------------------------------------------------
# 1D CCF along rv computation
# -------------------------------------------------------------------------

def init_rv_worker(ctx):
    global _RV_CTX
    _RV_CTX = ctx

def process_correlation_rv(args):
    
    i, rv_i = args
    
    rv                = _RV_CTX['rv']
    d                 = _RV_CTX['d']
    sigma_l           = _RV_CTX['sigma_l']
    d_bkg             = _RV_CTX['d_bkg']
    w                 = _RV_CTX['w']
    trans             = _RV_CTX['trans']
    template_HF       = _RV_CTX['template_HF']
    template_LF       = _RV_CTX['template_LF']
    Ss_HF             = _RV_CTX['Ss_HF']
    Ss_LF             = _RV_CTX['Ss_LF']
    template_shift    = _RV_CTX['template_shift']
    Rc                = _RV_CTX['Rc']
    pca               = _RV_CTX['pca']
    cut_fringes       = _RV_CTX['cut_fringes']
    R                 = _RV_CTX['R']
    Rmin              = _RV_CTX['Rmin']
    Rmax              = _RV_CTX['Rmax']
    target_name       = _RV_CTX['target_name']
    stellar_component = _RV_CTX['stellar_component']
    compare_data      = _RV_CTX['compare_data']
    calc_logL         = _RV_CTX['calc_logL']
    method_logL       = _RV_CTX['method_logL']
    mask_dpx_rv       = _RV_CTX['mask_dpx_rv']

    CCF_value       = np.nan
    corr_value      = np.nan
    corr_auto_value = np.nan
    logL_value      = np.nan
    CCF_bkg_value   = None if d_bkg is None else (np.zeros((len(d_bkg))) + np.nan)
    
    # Shifting the template
    template_HF_shift = template_HF.doppler_shift(rv_i, renorm=False).flux
    
    # The data to compare with is assimiled to the "template"
    if compare_data:
        t = template_HF_shift
    
    # If not comparing data, calculating the template
    else:
        t = trans * template_HF_shift
        # Adding the stellar component (if required)
        if stellar_component and Rc is not None:
            template_LF_shift = template_LF.doppler_shift(rv_i, renorm=False).flux
            t                += - trans * Ss_HF * template_LF_shift / Ss_LF
    
    # Apply weights & masks consistently
    t          *= w
    d_masked, t = get_masked_quantity(data=d, template=t, weight=w)
    
    # Optionnal notch (fringe) filtering and PCA modes removing (template-side projection)
    if not compare_data:
        # Cut fringes frequencies (if required)
        if cut_fringes:
            t           = cut_spectral_frequencies(t, R, Rmin, Rmax, target_name=target_name)
            d_masked, t = get_masked_quantity(data=d_masked, template=t, weight=w)
        # Subtract PCA components (if required)
        if pca is not None:
            t           = get_pca_subtracted_data(data=t, pca=pca)
            d_masked, t = get_masked_quantity(data=d_masked, template=t, weight=w)
    
    # Normalizing the template
    norm_t = np.sqrt(np.nansum(t**2))
    if not np.isfinite(norm_t) or norm_t == 0:
        return i, CCF_value, corr_value, corr_auto_value, logL_value, CCF_bkg_value
    t /= norm_t

    # CCF computations
    CCF_value  = np.nansum(d_masked * t) # CCF signal
    corr_value = CCF_value / np.sqrt(np.nansum(d_masked**2))

    # Auto-correlation (if possible)
    if rv is not None: 
        corr_auto_value = np.nansum(template_shift * t)
    
    # CCF noises (backgrounds)
    if d_bkg is not None:
        for j in range(len(d_bkg)):
            CCF_bkg_value[j] = np.nansum(d_bkg[j]*w * t)
    
    # logL computation
    if calc_logL and sigma_l is not None:
        logL_value = get_logL(d_masked[mask_dpx_rv], t[mask_dpx_rv], sigma_l[mask_dpx_rv], method=method_logL)
    
    # Sanity check plots
    #plt.figure(dpi=300) ; plt.plot(np.isnan(d)) ; plt.plot(np.isnan(t)) ; plt.title(f"rv = {rv_i} km/S | N_d = {np.isfinite(d).sum()} | N_t = {np.isfinite(t).sum()}") ; plt.show()
    #plt.figure(dpi=300) ; plt.plot(np.isnan(d[mask_dpx_rv])) ; plt.plot(np.isnan(t[mask_dpx_rv])) ; plt.title(f"logL = {logL_value:.0f} | rv = {rv_i:.2f} km/s | N_d = {np.isfinite(d[mask_dpx_rv]).sum()} | N_t = {np.isfinite(t[mask_dpx_rv]).sum()}") ; plt.show()
    #plt.figure(dpi=300) ; plt.plot(d / np.sqrt(np.nansum(d**2))) ; plt.plot(t) ; plt.title(f"rv = {rv_i} km/S | N_d = {np.isfinite(d).sum()} | N_t = {np.isfinite(t).sum()}") ; plt.show()
    
    return i, CCF_value, corr_value, corr_auto_value, logL_value, CCF_bkg_value

def get_CCF_1D_rv(instru, band, d, d_bkg, wave, trans, R, Rc, filter_type, model, T, lg, rv_arr, rv, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template_wo_shift=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, pca=None, cut_fringes=False, Rmin=None, Rmax=None, target_name=None, renorm_d_sim=True, sigma_l=None, calc_logL=False, method_logL="classic", weight=None, compare_data=False, verbose=True, show=True, smooth_PSD=1, noise=None):
    """
    Correlate a 1D high–pass spectrum with a Doppler-shifted template over a radial-velocity grid.

    This routine computes, for each trial radial velocity (RV), the cross-correlation between the
    (optionally weighted) observed high–pass data and a template spectrum shifted to that RV.
    It supports adding a residual stellar component (MM formalism), optional PCA-mode projection
    from the template, optional fringe band-stop, and returns several diagnostics:
    the CCF curve, correlation strength, an auto-correlation proxy, optional log-likelihood, and
    analytical noise at the expected RV if per-channel noise is provided.

    Parameters
    ----------
    instru : str
        Instrument name (only used for a few instrument-specific tweaks, e.g. "HiRISE").
    d : (N,) array_like
        Observed **high-pass** spectrum on 'wave' (can contain NaNs, zeros are treated as NaNs).
        If 'weight' is provided, the data are multiplied by it internally.
    d_bkg : None or (Nbkg, N) array_like
        Optional set of background spectra (same grid as 'd'). Each row is treated like a
        background CCF channel (e.g., from spatial background extraction).
        If 'None', background CCFs are skipped.
    trans_Ss : (N,) array_like
        Stellar flux on 'wave' **before** applying the total transmission. Required if
        'stellar_component=True'. Ignored otherwise.
    wave : (N,) array_like
        Wavelength grid in microns (monotonic).
    trans : (N,) array_like
        Total end-to-end transmission sampled on 'wave' (unitless). Used to scale the template.
    T : float
        Effective temperature used for building the template (K).
    lg : float
        Surface gravity 'log10(g[cm s^-2])' used for building the template.
    model : str
        Template model name (e.g. "BT-Settl", "mol_CO", ...).
    R : float
        Spectral resolving power of the data on 'wave' (used by filters and resampling).
    Rc : float or None
        High/low-pass cut-off in resolving-power units. If 'None', no filtering is applied.
    filter_type : {"gaussian", "gaussian_bis", "step", "smoothstep", "savitzky_golay"}
        High/low-pass filter type, passed to 'filtered_flux'.
    rv_arr : (Nrv,) array_like or None, optional
        Custom RV grid in km/s.
    rv : float or None, optional
        Expected RV in km/s (e.g., from ephemeris). Enables auto-correlation proxy and
        analytical sigma for that RV if 'sigma_l' is provided.
    vsini : float or None, optional
        Template rotational broadening parameter (km/s). Passed to 'get_template'.
    pca : object or None, optional
        PCA object with attributes 'components_' (2D array) and 'n_components' (int). If provided,
        the template is orthogonally projected to remove these modes before correlation.
    template_wo_shift : Spectrum or (N,) array_like or None, optional
        If 'compare_data=False': a 'Spectrum' object to use directly (bypasses 'get_template') before
        filtering. If 'compare_data=True': a 1D array on 'wave' representing the *other* dataset
        to compare with 'd'.
        If 'None', a template is generated from 'model', 'T', 'lg', etc.
    renorm_d_sim : bool, optional
        Used only for plotting in 'plot_CCF_1D_rv'; ignored here. Default True.
    epsilon : float, optional
        Limb-darkening coefficient for broadening kernels (if used by 'get_template'). Default 0.8.
    fastbroad : bool, optional
        Use fast broadening (implementation detail of 'get_template'). Default True.
    calc_logL : bool, optional
        If True, compute log-likelihood vs RV using 'get_logL(d, t, sigma_l, method_logL)'. Default False.
    method_logL : {"classic", ...}, optional
        Log-likelihood method name forwarded to 'calc_logL'. Default "classic".
    sigma_l : (N,) array_like or None, optional
        Per-channel noise (same grid as 'wave'). Needed to return analytical 'sigma'.
    weight : (N,) array_like or None, optional
        Optional weights applied to both data and template prior to normalization (e.g. mask or inverse-variance).
    stellar_component : bool, optional
        If True and 'Rc' is not None, include the residual stellar term in the template:
        't = t - trans * Ss_HF * template_LF / Ss_LF'. Requires 'trans_Ss'. Default True.
    degrade_resolution : bool, optional
        Degrade template to R using 'Spectrum.degrade_resolution(..., R_output=R)' prior to filtering.
        If False, only interpolate to 'wave'. Default True.
    compare_data : bool, optional
        If True, interpret 'template' as another data vector to be compared to 'd'.
        Skips model generation and stellar-component handling. Default False.
    cut_fringes : bool, optional
        If True, pass the (weighted) template vector through 'cut_spectral_frequencies'
        with '(Rmin, Rmax)'. Default False.
    Rmin, Rmax : float or None, optional
        Band-stop limits in resolving-power space for 'cut_fringes'. Default None.
    target_name : str or None, optional
        Target label for plotting/logging only. Default None.
    verbose : bool, optional
        Disable the progress bar. Default True.
    show : bool, optional
        If True and 'rv' is provided, this function calls 'plot_CCF_1D_rv' for display. Default True.
    smooth_PSD : int, optional
        Smoothing factor for PSDs shown in the plot (if 'show=True'). Default 1.
    noise : (N,) array_like or None, optional
        Realistic noise realisation.
        
    Returns
    -------
    rv : (Nrv,) ndarray
        The RV grid (km/s).
    CCF : (Nrv,) ndarray
        Cross-correlation vs RV (dot product with a **unit-norm** template over valid channels).
    corr : (Nrv,) ndarray
        Correlation strength vs RV: 'CCF / ||d||' over the valid masked region.
    CCF_bkg : None or (Nbkg, Nrv) ndarray
        Background CCFs vs RV if 'd_bkg' is provided; otherwise 'None'.
    corr_auto : (Nrv,) ndarray
        Auto-correlation proxy (template vs template shifted at 'rv') scaled to unit data norm.
        If 'rv=None', returns zeros.
    logL : (Nrv,) ndarray
        Log-likelihood vs RV (zeros if 'calc_logL=False').
    sigma_CCF : float or None
        Analytical 1-sigma CCF uncertainty at 'rv' if 'sigma_l' is provided;
        otherwise 'None'.
    """
    
    R_sampling = get_resolution(wavelength=wave, func=np.array)
    
    # --- RV grid handling
    if rv_arr is None:
        rv_arr = np.linspace(-1000, 1000, 4001, dtype=float)
    else:
        rv_arr = np.asarray(rv_arr, dtype=float)
    if rv is not None and rv not in rv_arr:
        rv_arr = np.sort(np.append(rv_arr, rv))
        
    # --- Initialize data
    Nrv       = len(rv_arr)
    CCF       = np.zeros((Nrv)) + np.nan
    corr      = np.zeros((Nrv)) + np.nan
    corr_auto = np.zeros((Nrv)) + np.nan
    logL      = np.zeros((Nrv)) + np.nan
    sigma_CCF = None
    CCF_bkg   = np.zeros((len(d_bkg), Nrv)) + np.nan if d_bkg is not None else None
    
    # --- Templates preparation

    # The data to compare with is assimiled to the "template"
    if compare_data:
        if template_wo_shift is None:
            raise ValueError("When compare_data=True, provide 'template_wo_shift' array on the same grid.")
        template_HF = Spectrum(wavelength=wave, flux=template_wo_shift, R=R)
        template_LF = Ss_HF = Ss_LF = None
        
    # If not comparing data, calculating the raw template
    else:
        # Generate template if not provided
        if template_wo_shift is None:
            template_wo_shift = get_template(instru=instru, wave=wave, R=R, model=model, T=T, lg=lg, rv=0, vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model)

        # Degrade template to target grid
        template = get_spectrum_band(spectrum_instru=template_wo_shift, wave_band=wave, R_output=R, degrade_resolution=degrade_resolution)
        
        # Return if template only contains NaNs (i.e. if the crop of the molecular templates left only NaNs)
        if np.all(np.isnan(template.flux)): # Nothing to correlate
            return rv_arr, CCF, corr, CCF_bkg, corr_auto, logL, sigma_CCF
        
        # Filter the template
        template_HF                        = template.copy()
        template_LF                        = template.copy()
        template_HF.flux, template_LF.flux = filtered_flux(template.flux, R=R_sampling, Rc=Rc, filter_type=filter_type)
        
        # Handle stellar component (if needed)
        if stellar_component and Rc is not None:
            Ss_HF, Ss_LF = get_Ss_HF_LF(trans_Ss=trans_Ss, trans=trans, wave=wave, R_sampling=R_sampling, Rc=Rc, filter_type=filter_type)
        else:
            Ss_HF = Ss_LF = None
    
    # --- Data preparation: apply weighting and mask zeros as NaN
    d_raw     = np.copy(d)
    w         = np.ones_like(wave) if weight is None else weight
    d         = d_raw * w
    d[d == 0] = np.nan
    if sigma_l is not None:
        sigma_l_raw = np.copy(sigma_l)
        sigma_l     = sigma_l_raw * w
    else:
        sigma_l_raw = sigma_l = None

    # --- Reference for auto-correlation
    if rv is not None:
        # Already prepared "template"
        if compare_data:
            d_sim = np.copy(template_HF.flux)
        # Preparing the template
        else:
            template = template_wo_shift.doppler_shift(rv)
            d_sim    = get_d_sim(instru=instru, d=d_raw, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=None, T=None, lg=None, rv=None, vsini=None, epsilon=None, fastbroad=None, airmass=None, star_spectrum=None, wave_model=None, template=template, degrade_resolution=degrade_resolution, stellar_component=stellar_component, trans_Ss=trans_Ss, pca=pca, cut_fringes=cut_fringes, Rmin=Rmin, Rmax=Rmax, target_name=target_name, renorm_d_sim=renorm_d_sim, sigma_l=sigma_l_raw, verbose=verbose)
        d_sim         *= w
        d, d_sim       = get_masked_quantity(data=d, template=d_sim, weight=w)
        template_shift = d_sim / np.sqrt(np.nansum( d_sim**2 ))
    else:
        d_sim          = None
        template_shift = None
    
    # Keeping same mask across RV for logL computationsNone
    if calc_logL:
        dlambda = 10 * wave/R
        #dlambda = wave * 1000*np.nanmax(np.abs((rv_arr))) / c
        dwave   = np.gradient(wave)
        dpx_rv  = int(np.ceil(np.nanmax(np.abs(dlambda / dwave))))
        if verbose:
            print(f"\nRV padding for logL computation: dpx_rv = {dpx_rv} bins")
        if 2*dpx_rv >= len(wave):
            raise ValueError("RV padding removes the full spectrum. Check rv_arr or wavelength sampling.")
        mask_dpx_rv = np.ones(len(wave), dtype=bool)
        if dpx_rv > 0:
            mask_dpx_rv[:dpx_rv]  = False
            mask_dpx_rv[-dpx_rv:] = False
    else:
        mask_dpx_rv = None
    
    # ---------- Parallel sweep across (rv) ----------
    # Init global context for workers
    global _RV_CTX
    _RV_CTX = dict(rv=rv, d=d, sigma_l=sigma_l, d_bkg=d_bkg, w=w, trans=trans, template_HF=template_HF, template_LF=template_LF, Ss_HF=Ss_HF, Ss_LF=Ss_LF, template_shift=template_shift, Rc=Rc, pca=pca, cut_fringes=cut_fringes, R=R, Rmin=Rmin, Rmax=Rmax, target_name=target_name, stellar_component=stellar_component, compare_data=compare_data, calc_logL=calc_logL, method_logL=method_logL, mask_dpx_rv=mask_dpx_rv)
    
    if verbose:
        print()
    with Pool(processes=max(1, cpu_count() // 2), initializer=init_rv_worker, initargs=(_RV_CTX,)) as pool:    
        results = list(tqdm(pool.imap(process_correlation_rv, ((i, rv_arr[i]) for i in range(Nrv))), total=Nrv, desc=" get_CCF_1D_rv()", disable=not verbose))
        for (i, CCF_value, corr_value, corr_auto_value, logL_value, CCF_bkg_value) in results:
            CCF[i]        = CCF_value
            corr[i]       = corr_value
            corr_auto[i]  = corr_auto_value
            logL[i]       = logL_value
            if CCF_bkg is not None:
                CCF_bkg[:, i] = CCF_bkg_value

    # # Sanity check plots
    # for i in range(Nrv):
    #     args = (i, rv_arr[i])
    #     _, CCF[i], corr[i], corr_auto[i], logL[i], CCF_bkg[:, i] = process_correlation_rv(args)
    
    # --- END OF THE LOOP (computing sigma_CCF and plot)
    if rv is not None:
        
        # --- Optional sigma at rv (analytical)
        if sigma_l is not None:
            sigma_CCF = np.sqrt(np.nansum(sigma_l**2 * template_shift**2))
        
        # --- Optional display
        if show:
            
            fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=300, gridspec_kw={"height_ratios": [3, 1]}, sharex="col", constrained_layout=True)
            if target_name is not None:
                tn = target_name.replace("_", " ")
            if compare_data:
                fig.suptitle(f"{instru} {tn} data sets, with R={np.nanmedian(R):.0f} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d * template_shift) / np.sqrt(np.nansum(d**2) * np.nansum(template_shift**2)), 3)}", fontsize=20)
            else:
                if lg is None:
                    fig.suptitle(f"{instru} {tn} data and {model} template on {band}-band,\n with $T$={T:.0f}K, rv={rv:.1f}km/s, vsini={vsini:.1f}km/s, R={int(np.nanmedian(R)):.0f} and $R_c$={Rc:.0f} (correlation strength = {np.nansum(d * template_shift) / np.sqrt(np.nansum(d**2) * np.nansum(template_shift**2)):.3f})", fontsize=20)
                else:
                    fig.suptitle(f"{instru} {tn} data and {model} template on {band}-band,\n with $T$={T:.0f}K, lg={lg:.1f}, rv={rv:.1f}km/s, vsini={vsini:.1f}km/s, R={int(np.nanmedian(R)):.0f} and $R_c$={Rc:.0f} (correlation strength = {np.nansum(d * template_shift) / np.sqrt(np.nansum(d**2) * np.nansum(template_shift**2)):.3f})", fontsize=20)

            # Plot high-pass filtered data and template
            axs[0, 0].set_ylabel("High-pass flux", fontsize=14)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=14)
            axs[0, 0].plot(wave, d, c='crimson', label=f"{tn} data")
            
            # Plotting template
            if compare_data:
                axs[0, 0].plot(wave, d_sim, c='steelblue', label=model)
            else:
                axs[0, 0].plot(wave, d_sim, c='steelblue', label=model + " template")
                
            # Simulating expected white noise
            if sigma_l is not None:
                if noise is None:
                    noise = np.random.normal(0, sigma_l, len(wave))
                    if instru not in {"HARMONI", "ANDES"}: # Filtering noise at R
                        noise = filtered_flux(flux=noise, R=R_sampling, Rc=R,  filter_type=filter_type)[1]
                    if Rc is not None: # Filtering noise under Rc
                        noise = filtered_flux(flux=noise, R=R_sampling, Rc=Rc, filter_type=filter_type)[0]
                    if cut_fringes:
                        noise = cut_spectral_frequencies(noise, R=R, Rmin=Rmin, Rmax=Rmax, target_name=target_name)
                d_sim_noise = d_sim + noise
                if not compare_data:
                    axs[0, 0].plot(wave, d_sim_noise, c='steelblue', label=model + " template w/ expected noise " + r"($cos\theta_n$ = " + f"{round(np.nansum(d_sim_noise * template_shift) / np.sqrt(np.nansum(d_sim_noise**2)), 3)})", alpha=0.5)
                    res_d_sim_noise, psd_d_sim_noise = get_psd(wave, d_sim_noise, smooth=smooth_PSD)
                    axs[0, 1].plot(res_d_sim_noise, psd_d_sim_noise, 'steelblue', alpha=0.5, zorder=10)
                axs[1, 0].plot(wave, noise, c='seagreen', label="expected noise", zorder=3, alpha=0.8)
                res_noise, psd_noise = get_psd(wave, noise, smooth=smooth_PSD)
                axs[1, 1].plot(res_noise, psd_noise, 'seagreen', zorder=10, alpha=0.8)
            axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[0, 0].minorticks_on()
            axs[0, 0].legend(fontsize=14, loc="upper left")
            axs[0, 0].set_ylim(2 * np.nanmin(d), 2 * np.nanmax(d))
            
            #Zoom
            if instru in {"HiRISE", "VIPA"}:
                if "CH4" in target_name:
                    zoom_xmin, zoom_xmax = 1.665, 1.668
                else:
                    zoom_xmin, zoom_xmax = 1.686, 1.692
                axins = inset_axes(axs[0, 0], width="40%", height="40%", loc="lower right", borderpad=2)
                axins.plot(wave, d,     c='crimson', label=f"{tn} data")
                axins.plot(wave, d_sim, c='steelblue')
                axins.set_xlim(zoom_xmin, zoom_xmax)
                axins.set_ylim(min(np.nanmin(d[(wave >= zoom_xmin) & (wave <= zoom_xmax)]), np.nanmin(d_sim[(wave >= zoom_xmin) & (wave <= zoom_xmax)])), max(np.nanmax(d[(wave >= zoom_xmin) & (wave <= zoom_xmax)]), np.nanmax(d_sim[(wave >= zoom_xmin) & (wave <= zoom_xmax)])))
                axins.tick_params(axis='both', which='major', labelsize=10)            
                mark_inset(axs[0, 0], axins, loc1=1, loc2=3, fc="none", ec="black", linestyle="--")
                axins.set_xticklabels([])
                axins.set_yticklabels([])

            # Calculate PSDs
            res, psd_d     = get_psd(wave, d,     smooth=smooth_PSD)
            res, psd_d_sim = get_psd(wave, d_sim, smooth=smooth_PSD)
            
            # Plot PSDs
            axs[0, 1].set_ylabel("PSD", fontsize=14)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=14)
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_xlim(10, 2*np.nanmedian(R_sampling))
            axs[0, 1].plot(res, psd_d,     c='crimson')
            axs[0, 1].plot(res, psd_d_sim, c='steelblue')
            axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[0, 1].minorticks_on()
            if Rc is not None:
                axs[0, 1].axvline(Rc,                   c='k', ls="--", label=f"$R_c$ = {Rc:.0f}")
            axs[0, 1].axvline(np.nanmedian(R),          c='k', ls="-",  label=f"$R$ = {np.nanmedian(R):.0f}", alpha=0.5)
            axs[0, 1].axvline(np.nanmedian(R_sampling), c='k', ls="-",  label=f"$R$ (sampling) = {np.nanmedian(R_sampling):.0f}")
            axs[0, 1].legend(fontsize=14, loc="upper left")
            
            # Calculate residuals
            residuals = d - d_sim
            
            # Plot residuals
            axs[1, 0].set_xlabel("Wavelength [µm]", fontsize=16)
            axs[1, 0].set_ylabel("Residuals",       fontsize=16)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=14)
            axs[1, 0].set_xlim(np.nanmin(wave[np.isfinite(residuals)]), np.nanmax(wave[np.isfinite(residuals)]))
            axs[1, 0].set_ylim(-5 * np.nanstd(residuals), 5 * np.nanstd(residuals))
            if compare_data:
                axs[1, 0].plot(wave, residuals, 'k', label="data1 - data2", alpha=0.8)
            else:
                axs[1, 0].plot(wave, residuals, 'k', label="data - template", alpha=0.8)
            axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[1, 0].minorticks_on()
            axs[1, 0].legend(fontsize=14, loc="upper left")
            
            # Calculate PSD of residuals
            res_residuals, psd_residuals = get_psd(wave, residuals, smooth=smooth_PSD)
            
            # Plot PSD of residuals
            axs[1, 1].set_xlabel("Resolution R", fontsize=16)
            axs[1, 1].set_ylabel("PSD",          fontsize=16)
            axs[1, 1].set_xscale('log')
            axs[1, 1].set_yscale('log')
            axs[1, 1].tick_params(axis='both', which='major', labelsize=14)
            axs[1, 1].plot(res_residuals, psd_residuals, 'k', alpha=0.8)
            axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5) ; axs[1, 1].minorticks_on()
            if Rc is not None:
                axs[1, 1].axvline(Rc,                   c='k', ls="--", label=f"$R_c$ = {Rc:.0f}")
            axs[1, 1].axvline(np.nanmedian(R),          c='k', ls="-",  label=f"$R$ = {np.nanmedian(R):.0f}", alpha=0.5)
            axs[1, 1].axvline(np.nanmedian(R_sampling), c='k', ls="-",  label=f"$R$ (sampling) = {np.nanmedian(R_sampling):.0f}")
            axs[1, 1].set_ylim(np.nanmin(psd_residuals[res_residuals>10]), np.nanmax(psd_residuals[res_residuals>10]))

            plt.show()
            
            if sigma_l is not None:
                sigma_eff = np.sqrt(np.nanmean(sigma_l**2))
                print(f"  sigma(empirical) / sigma(analytical) = {100*np.nanstd(residuals)/sigma_eff:.1f} %") # print(" psd(noise) / psd(residuals) = ", np.sqrt(np.nansum(psd_residuals)) / np.sqrt(np.nansum(psd_noise)))
            
            # plt.figure(figsize=(12, 6), dpi=300)
            # plt.plot(wave, d, c='crimson', label=f"{tn} data")
            # if compare_data:
            #     plt.plot(wave, d_sim, c='steelblue', label=model)
            # else:
            #     plt.plot(wave, d_sim, c='steelblue', label=model + " template")
            # plt.xlabel('Wavelength [nm]', fontsize=14)
            # plt.ylabel("High-pass flux",  fontsize=14)
            # if compare_data:
            #     plt.title(f"{instru} {tn} data sets, with R={int(R)} and $R_c$={Rc} \n correlation strength = {round(np.nansum(d * template_shift) / np.sqrt(np.nansum(d**2) * np.nansum(template_shift**2)), 3)}", fontsize=14)
            # else:
            #     plt.title(f"{instru} {tn} data and {model} template on {band}-band,\n with $T$={round(T)}K, lg={round(lg, 1)}, rv={round(rv, 1)}km/s, vsini={round(vsini, 1)}km/s, R={int(np.nanmedian(R)):.0f} and $R_c$={Rc} (correlation strength = {round(np.nansum(d * template_shift) / np.sqrt(np.nansum(d**2) * np.nansum(template_shift**2)), 3)})", fontsize=14)
            # plt.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.45)
            # plt.grid(which="minor", linestyle=":",  linewidth=0.4, alpha=0.25)
            # plt.xlim(wave[0], wave[-1])
            # plt.ylim(2 * np.nanmin(d), 2 * np.nanmax(d))
            # plt.legend(fontsize=12, loc="upper right", frameon=True, edgecolor="gray", facecolor="whitesmoke")
            # plt.minorticks_on()
            # plt.tight_layout()
            # plt.show()
            
    return rv_arr, CCF, corr, CCF_bkg, corr_auto, logL, sigma_CCF



def plot_CCF_1D_rv(instru, band, target_name, d, d_bkg, wave, trans, R, Rc, filter_type, model, T, lg, rv_arr, rv, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template_wo_shift=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, pca=None, cut_fringes=False, Rmin=None, Rmax=None, renorm_d_sim=True, sigma_l=None, calc_logL=False, method_logL="classic", weight=None, compare_data=False, verbose=True, show=True, smooth_PSD=1, noise=None, rv_star=None, zoom_CCF=True):
    """
    Compute and plot the CCF (S/N and correlation units) — optional log-likelihood — and return metrics.

    This is a convenience wrapper around 'get_CCF_1D_rv' that also:
    (1) recenters on the peak near the expected RV (if provided),
    (2) removes large-scale offsets for very wide RV ranges,
    (3) converts the CCF to empirical S/N using the high-RV wings as noise,
    (4) optionally plots a zoom on the peak and a separate log-likelihood panel.

    Parameters
    ----------
    instru : str
        Instrument name (passed to 'get_CCF_1D_rv').
    band : str
        Spectral band label (for the plot title only).
    target_name : str
        Target label (for the plot title only).
    d : (N,) array_like
        Observed **high-pass** spectrum on 'wave' (NaNs allowed).
    d_bkg : None or (Nbkg, N) array_like
        Optional background spectra (for background CCFs/SNRs).
    trans_Ss : (N,) array_like
        Stellar flux on 'wave' (needed if 'stellar_component=True').
    wave : (N,) array_like
        Wavelength grid (µm).
    trans : (N,) array_like
        Total end-to-end transmission on 'wave'.
    T : float
        Effective temperature for the template (K).
    lg : float
        Surface gravity 'log10(g[cm s^-2])' for the template.
    model : str
        Template model name.
    R : float
        Spectral resolving power of the data.
    Rc : float or None
        High/low-pass cutoff in resolving-power units; if 'None', no filtering.
    filter_type : {"gaussian", "gaussian_bis", "step", "smoothstep", "savitzky_golay"}
        Filter type for 'filtered_flux'.
    rv_arr : (Nrv,) array_like or None, optional
        Custom RV grid in km/s.
    rv : float or None, optional
        Expected RV (km/s). Used to refine the peak estimate and define the wing region (±200 km/s).
    vsini : float, optional
        Rotational broadening for the template (km/s). Default 0.
    pca : object or None, optional
        PCA object with 'components_' and 'n_components'. If provided, modes are projected out from the template.
    template_wo_shift : Spectrum or (N,) array_like or None, optional
        See 'get_CCF_1D_rv'. If 'compare_data=True', provide a 1D array.
    renorm_d_sim : bool, optional
        Only used when 'show=True' and 'rv' provided: rescales the template to match the
        data HF power in a robust band (useful for illustrative plots). Default True.
    epsilon : float, optional
        Limb-darkening coefficient for broadening kernels. Default 0.8.
    fastbroad : bool, optional
        Use fast broadening inside 'get_template'. Default True.
    calc_logL : bool, optional
        If True, also plot log-likelihood vs RV. Default False.
    method_logL : {"classic", ...}, optional
        Method passed to 'calc_logL'. Default "classic".
    sigma_l : (N,) array_like or None, optional
        Per-channel noise. Enables analytical sigma in the printout (if available).
    weight : (N,) array_like or None, optional
        Optional weight array applied to data and template.
    stellar_component : bool, optional
        Include residual stellar term in the template when 'Rc' is not None. Default True.
    degrade_resolution : bool, optional
        Degrade the template to R before filtering. Default True.
    compare_data : bool, optional
        Treat 'template' as another dataset rather than a physical model. Default False.
    cut_fringes : bool, optional
        Apply fringe band-stop ('Rmin', 'Rmax') to the template. Default False.
    Rmin, Rmax : float or None, optional
        Band-stop limits in resolving-power space for 'cut_fringes'.
    verbose : bool, optional
        Print summary (peak S/N, correlation, sigma ratios). Default True.
    show : bool, optional
        Display the figure(s). Default True.
    smooth_PSD : int, optional
        PSD smoothing factor for visual diagnostics. Default 1.
    rv_star : float or None, optional
        Stellar RV for reference line in the plot. Default None.
    zoom_CCF : bool, optional
        Add an inset zoom around the detected peak. Default True.
    noise : (N,) array_like or None, optional
        Realistic noise realisation.

    Returns
    -------
    rv_arr : (Nrv,) ndarray
        RV grid (km/s).
    SNR : (Nrv,) ndarray
        Empirical S/N of the CCF vs RV (noise estimated from RV wings).
    SNR_bkg : None or (Nbkg, Nrv) ndarray
        Background CCFs converted to S/N by their wing standard deviation; 'None' if no backgrounds.
    corr : (Nrv,) ndarray
        Correlation strength vs RV (normalized by ||data||).
    signal : float
        CCF peak value (not in S/N).
    sigma_CCF : float
        Empirical standard deviation of the CCF (wing-based).
    sigma_CCF : float or None
        Analytical sigma at 'rv' if 'sigma_l' available; else 'None'.
    corr_auto : (Nrv,) ndarray
        Auto-correlation proxy (see 'get_CCF_1D_rv').
    """
    
    # Compute the radial velocity and cross-correlation functions for the data and background
    rv_arr, CCF, corr, CCF_bkg, corr_auto, logL, sigma_CCF = get_CCF_1D_rv(instru=instru, band=band, d=d, d_bkg=d_bkg, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=model, T=T, lg=lg, rv_arr=rv_arr, rv=rv, vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model, template_wo_shift=template_wo_shift, degrade_resolution=degrade_resolution, stellar_component=stellar_component, trans_Ss=trans_Ss, pca=pca, cut_fringes=cut_fringes, Rmin=Rmin, Rmax=Rmax, target_name=target_name, renorm_d_sim=renorm_d_sim, sigma_l=sigma_l, calc_logL=calc_logL, method_logL=method_logL, weight=weight, compare_data=compare_data, verbose=verbose, show=show, smooth_PSD=smooth_PSD, noise=noise)

    # RV prior
    if rv is None:
        if calc_logL:
            rv = rv_arr[np.nanargmax(logL)]
        else:
            rv = rv_arr[np.nanargmax(CCF)]
    prior_width = np.nanmedian(10 * 1e-3*c/R)  # [km/s]
    mask_prior  = (rv_arr >= rv - prior_width) & (rv_arr <= rv + prior_width)
    if np.count_nonzero(mask_prior) < 3:
        raise ValueError(f"RV prior contains too few points: {np.count_nonzero(mask_prior)}. Increase prior_width or use a finer rv_arr grid.")
    
    # Estimating RV
    if calc_logL:
        P                                            = np.exp(logL[mask_prior] - np.nanmax(logL[mask_prior]))
        dv                                           = np.gradient(rv_arr[mask_prior])
        norm                                         = np.nansum(P * dv)
        P                                            = P / norm
        optimal_values, uncertainties, bounds_1sigma = estimate_uncertainties_1sigma(P=P, func_marg=np.nansum, params=[rv_arr[mask_prior]])
        rv                                           = optimal_values[0]
        sigma_rv                                     = uncertainties[0]
    else:
        rv       = rv_arr[mask_prior][np.nanargmax(CCF[mask_prior])]
        sigma_rv = None
    
    # Estimating the CCF noise (sigma_CCF_emp)
    if np.nanmax(np.abs(rv_arr))/2 > 200: # https://arxiv.org/pdf/2405.13469: std(rv +- 200 km/s)
        mask_noise = (rv_arr > rv + 200) | (rv_arr < rv - 200)
        # Remove the offset introduced by the residual stellar component and systematic effects
        CCF -= np.nanmean(CCF[mask_noise])
        #corr -= np.nanmean(corr[mask_noise]) # not needed since we are only interested on the max correlation value (without subtracting the potential offset)
        corr_auto -= np.nanmean(corr_auto[mask_noise])
        if d_bkg is not None: # Remove offset from the background CCF if background data is provided
            for i in range(len(d_bkg)):
                CCF_bkg[i] -= np.nanmean(CCF_bkg[i])
    else:
        mask_noise = np.full((len(rv_arr)), True)    
    sigma2_tot  = np.nanvar(CCF[mask_noise])  # Total variance
    sigma2_auto = np.nanvar(corr_auto[mask_noise] * np.nanmax(CCF[mask_prior]) / np.nanmax(corr_auto))    
    if sigma2_auto < sigma2_tot and not compare_data:
        sigma_CCF_emp = np.sqrt(sigma2_tot - sigma2_auto)  # sqrt(var(signal) - var(auto-correlation))
    else:
        sigma_CCF_emp = np.sqrt(sigma2_tot)
    
    # Calculating SNR
    SNR     = CCF / sigma_CCF_emp
    signal  = np.nanmax(CCF[mask_prior])
    max_SNR = np.nanmax(SNR[mask_prior])
    if d_bkg is not None: # Plot background SNR if background data is provided
        SNR_bkg = np.zeros((len(d_bkg), len(rv_arr)))
        for i in range(len(d_bkg)):
            SNR_bkg[i] = CCF_bkg[i] / np.nanstd(CCF_bkg[i][mask_noise])
    else:
        SNR_bkg = None
    
    if show:
        
        if rv > 0:
            loc_legend = "upper right"
            loc_zoom   = "upper left"
        else:
            loc_legend = "upper left"
            loc_zoom   = "upper right"
        
        # Plot the CCF in S/N and correlation units
        fig = plt.figure(figsize=(12, 7), dpi=300)
        ax1 = fig.add_subplot(111)
        ax1.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.35, color="0.45")
        ax1.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25, color="0.55")
        ax1.minorticks_on()    
        ax1.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.1)
        ax1.tick_params(axis="both", which="minor", length=3.5, width=0.8)
        ax1.set_xlim(rv_arr[0], rv_arr[-1])        
        title = f'CCF of {target_name.replace("_", " ")} on {band}-band of {instru} with {model} template'
        if compare_data:
            title += f"\nwith $R_c$ = {Rc}"
        else:
            if lg is None:
                title += f"\nat $T$ = {T:.0f}K and Vsini = {vsini:.2f} km/s with $R_c$ = {Rc}"
            else:
                title += f"\nat $T$ = {T:.0f}K, "+r"$\log g$ = "+f"{lg:.2f} and Vsini = {vsini:.2f} km/s with $R_c$ = {Rc}"
        ax1.set_title(title, fontsize=16, pad=15)        
        ax1.set_xlabel("Observed radial velocity [km/s]", fontsize=16, labelpad=10)
        ax1.set_ylabel("CCF [S/N]",                       fontsize=16, labelpad=10)
        ax1.plot([], [], 'gray', label="Noise", alpha=0.5)        
        if d_bkg is not None:  # Ajout du fond si disponible
            for i in range(len(d_bkg)):
                ax1.plot(rv_arr, SNR_bkg[i], 'gray', alpha=min(max(0.1, 1 / len(d_bkg)) + 0.1, 1))
        ax1.plot(rv_arr, corr_auto * max_SNR / np.nanmax(corr_auto), "k", label="Auto-correlation")
        ax1.plot(rv_arr, SNR, label=f"{target_name.replace('_', ' ')}", c="steelblue", zorder=3, linewidth=2)
        if rv_star is not None:
            ax1.axvline(rv_star, c='seagreen', ls='--', label=f"$rv_{{star}}$ = {rv_star:.2f} km/s", alpha=0.7, linewidth=1.5)
        ax1.axvline(rv, c='crimson', ls='--', label=f"$rv_{{obs}}$ = {rv:.2f} km/s", alpha=0.7, linewidth=1.5)
        ax1.axvline(0,  c='k',       ls='-', alpha=0.5, linewidth=1)
        ymax = +1.2 * np.nanmax(SNR)
        ymin = -1.5 * np.abs(np.nanmin(SNR))
        ax1.set_ylim(ymin, ymax)
        ax2 = ax1.twinx()
        ax2.minorticks_on()
        ax2.set_ylabel("Correlation strength", fontsize=16, labelpad=20, rotation=270)
        ax2.tick_params(axis="y", which="major", labelsize=12, length=6, width=1.1)
        ax2.tick_params(axis="y", which="minor", length=3.5, width=0.8)
        ax2.set_ylim(ymin * corr[SNR==max_SNR][0] / max_SNR, ymax * corr[SNR==max_SNR][0] / max_SNR)
        if zoom_CCF:
            zoom_xmin, zoom_xmax = rv - np.nanmax(np.abs(rv_arr))/10, rv + np.nanmax(np.abs(rv_arr))/10
            axins = inset_axes(ax1, width="40%", height="40%", loc=loc_zoom, borderpad=2)  
            if d_bkg is not None:
                for i in range(len(d_bkg)):
                    axins.plot(rv_arr, SNR_bkg[i], 'gray', alpha=min(max(0.1, 1 / len(d_bkg)) + 0.1, 1))
            axins.plot(rv_arr, corr_auto * max_SNR / np.nanmax(corr_auto), "k")
            axins.plot(rv_arr, SNR, c='steelblue', zorder=10)
            if rv_star is not None:
                axins.axvline(rv_star, c='seagreen', ls='--', alpha=0.5, linewidth=1.5)
            axins.axvline(rv, c='crimson', ls='--', alpha=0.5, linewidth=1.5)
            axins.axvline(0, c='k', ls='-', alpha=0.5, linewidth=1)
            axins.set_xlim(zoom_xmin, zoom_xmax)
            axins.set_ylim(-1.25 * np.abs(np.nanmin(SNR)), 1.1*np.nanmax(SNR))                
            axins.tick_params(axis='both', which='major', labelsize=10)
            mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="black", linestyle="--")
            #axins.set_xticklabels([])
            axins.set_yticklabels([])
        ax1.legend(loc=loc_legend, fontsize=14, frameon=True, fancybox=True, framealpha=0.92, edgecolor="gray", facecolor="white")
        fig.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.78)
        plt.show()
        
        if verbose: # Print maximum S/N and correlation
            if sigma_l is not None: # Print error on sigma (if required):
                print(f"  sigma_CCF(empirical) / sigma_CCF(analytical) = {round(100*sigma_CCF_emp / sigma_CCF, 1)} %")
            # Print maximum S/N and correlation
            print(f"  CCF: max S/N ({max_SNR:.1f}) and correlation ({np.nanmax(corr[mask_prior]):.4f}) for rv = {rv_arr[mask_prior][np.nanargmax(SNR[mask_prior])]:.3f} km/s")
            
        # Plot log-likelihood (if required)
        if calc_logL:
            rv_lo, rv_hi = bounds_1sigma[0]
            rv_min       = max(rv_arr[mask_prior][0],  rv-5*sigma_rv)
            rv_max       = min(rv_arr[mask_prior][-1], rv+5*sigma_rv)

            # Plot the logL
            plt.figure(figsize=(12, 7), dpi=300)
            ax1 = plt.gca()        
            ax1.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.35, color="0.45")
            ax1.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25, color="0.55")
            ax1.minorticks_on()    
            ax1.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.1)
            ax1.tick_params(axis="both", which="minor", length=3.5, width=0.8)
            ax1.set_xlim(rv_min, rv_max)        
            ax1.set_ylim(0,      1.1)
            title = title.replace('CCF', 'Radial velocity probability')
            ax1.set_title(title, fontsize=16, pad=15)        
            ax1.set_xlabel("Observed Radial Velocity [km/s]",                     fontsize=16, labelpad=10)
            ax1.set_ylabel("Density probability function from logL [normalized]", fontsize=16, labelpad=10)
            ax1.plot(rv_arr[mask_prior], P/np.nanmax(P), label=f"{target_name.replace('_', ' ')}", c="steelblue", zorder=3, linewidth=2)
            if rv_star is not None:
                ax1.axvline(rv_star, c='seagreen', ls='--', label=f"$rv_{{star}}$ = {rv_star:.2f} km/s", alpha=0.7, linewidth=1.5)
            ax1.axvline(rv,    c='crimson', ls='--', label=rf"$rv_{{obs}}$ = {rv:.2f} $\pm$ {sigma_rv:.2f} km/s", alpha=0.7, linewidth=1.5)
            ax1.axvline(rv_lo, c='crimson', ls=':',                                                            alpha=0.7, linewidth=1.5)
            ax1.axvline(rv_hi, c='crimson', ls=':',                                                            alpha=0.7, linewidth=1.5)
            ax1.axvline(0, c='k', ls='-', alpha=0.5, linewidth=1)
            ax1.legend(loc=loc_legend, fontsize=14, frameon=True, fancybox=True, framealpha=0.92, edgecolor="gray", facecolor="white")
            plt.tight_layout()
            plt.show()
            
            if verbose: # Print maximum log-likelihood
                print(f"  max logL for rv = {rv:.3f} ± {sigma_rv:.3f} km/s")
    
    return rv_arr, SNR, SNR_bkg, corr, signal, sigma_CCF_emp, sigma_CCF, corr_auto, logL, rv, sigma_rv



# -------------------------------------------------------------------------
# 1D CCF along Vsini computation
# -------------------------------------------------------------------------

def init_vsini_worker(ctx):
    global _VSINI_CTX
    _VSINI_CTX = ctx

def process_correlation_vsini(args):
    
    i, vsini_i = args
    
    vsini             = _VSINI_CTX['vsini']
    epsilon           = _VSINI_CTX['epsilon']
    fastbroad         = _VSINI_CTX['fastbroad']
    d                 = _VSINI_CTX['d']
    sigma_l           = _VSINI_CTX['sigma_l']
    d_bkg             = _VSINI_CTX['d_bkg']
    w                 = _VSINI_CTX['w']
    trans             = _VSINI_CTX['trans']
    template_HF       = _VSINI_CTX['template_HF']
    template_LF       = _VSINI_CTX['template_LF']
    Ss_HF             = _VSINI_CTX['Ss_HF']
    Ss_LF             = _VSINI_CTX['Ss_LF']
    template_broad    = _VSINI_CTX['template_broad']
    Rc                = _VSINI_CTX['Rc']
    pca               = _VSINI_CTX['pca']
    cut_fringes       = _VSINI_CTX['cut_fringes']
    R                 = _VSINI_CTX['R']
    Rmin              = _VSINI_CTX['Rmin']
    Rmax              = _VSINI_CTX['Rmax']
    target_name       = _VSINI_CTX['target_name']
    stellar_component = _VSINI_CTX['stellar_component']
    compare_data      = _VSINI_CTX['compare_data']
    calc_logL         = _VSINI_CTX['calc_logL']
    method_logL       = _VSINI_CTX['method_logL']
    
    CCF_value       = np.nan
    corr_value      = np.nan
    corr_auto_value = np.nan
    logL_value      = np.nan
    CCF_bkg_value   = None if d_bkg is None else (np.zeros((len(d_bkg))) + np.nan)
    
    # Broadening the template
    template_HF_broad = template_HF.broad(vsini_i, epsilon=epsilon, fastbroad=fastbroad).flux
    
    # The data to compare with is assimiled to the "template"
    if compare_data:
        t = template_HF_broad
    
    # If not comparing data, calculating the template
    else:
        t = trans * template_HF_broad
        # Adding the stellar component (if required)
        if stellar_component and Rc is not None:
            template_LF_broad = template_LF.broad(vsini_i, epsilon=epsilon, fastbroad=fastbroad).flux
            t                += - trans * Ss_HF * template_LF_broad / Ss_LF
    
    # Apply weights & masks consistently
    t          *= w
    d_masked, t = get_masked_quantity(data=d, template=t, weight=w)
    
    # Optionnal notch (fringe) filtering and PCA modes removing (template-side projection)
    if not compare_data:
        # Cut fringes frequencies (if required)
        if cut_fringes:
            t           = cut_spectral_frequencies(t, R, Rmin, Rmax, target_name=target_name)
            d_masked, t = get_masked_quantity(data=d_masked, template=t, weight=w)
        # Subtract PCA components (if required)
        if pca is not None:
            t           = get_pca_subtracted_data(data=t, pca=pca)
            d_masked, t = get_masked_quantity(data=d_masked, template=t, weight=w)
    
    # Normalizing the template
    norm_t = np.sqrt(np.nansum(t**2))
    if not np.isfinite(norm_t) or norm_t == 0:
        return i, CCF_value, corr_value, corr_auto_value, logL_value, CCF_bkg_value
    t /= norm_t

    # CCF computations
    CCF_value  = np.nansum(d_masked * t) # CCF signal
    corr_value = CCF_value / np.sqrt(np.nansum(d_masked**2))

    # Auto-correlation (if possible)
    if vsini is not None: 
        corr_auto_value = np.nansum(template_broad * t)
    
    # CCF noises (backgrounds)
    if d_bkg is not None:
        for j in range(len(d_bkg)):
            CCF_bkg_value[j] = np.nansum(d_bkg[j]*w * t)
    
    # logL computation
    if calc_logL and sigma_l is not None:
        logL_value = get_logL(d_masked, t, sigma_l, method=method_logL)
    
    return i, CCF_value, corr_value, corr_auto_value, logL_value, CCF_bkg_value

def get_CCF_1D_vsini(instru, d, d_bkg, wave, trans, R, Rc, filter_type, model, T, lg, rv, vsini_arr, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template_wo_broad=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, pca=None, cut_fringes=False, Rmin=None, Rmax=None, target_name=None, renorm_d_sim=True, sigma_l=None, calc_logL=False, method_logL="classic", weight=None, compare_data=False, verbose=True):
    """
    Correlate a 1D high–pass spectrum with a rotational broaded template over a vsini grid.

    This routine computes, for each trial Vsini, the cross-correlation between the
    (optionally weighted) observed high–pass data and a template spectrum broaded to that Vsini.
    It supports adding a residual stellar component (MM formalism), optional PCA-mode projection
    from the template, optional fringe band-stop, and returns several diagnostics:
    the CCF curve, correlation strength, an auto-correlation proxy, optional log-likelihood, and
    analytical noise at the known Vsini if per-channel noise is provided (sigma_l).

    Parameters
    ----------
    instru : str
        Instrument name (only used for a few instrument-specific tweaks, e.g. "HiRISE").
    d : (N,) array_like
        Observed **high-pass** spectrum on 'wave' (can contain NaNs, zeros are treated as NaNs).
        If 'weight' is provided, the data are multiplied by it internally.
    d_bkg : None or (Nbkg, N) array_like
        Optional set of background spectra (same grid as 'd'). Each row is treated like a
        background CCF channel (e.g., from spatial background extraction).
        If 'None', background CCFs are skipped.
    trans_Ss : (N,) array_like
        Stellar flux on 'wave' **before** applying the total transmission. Required if
        'stellar_component=True'. Ignored otherwise.
    wave : (N,) array_like
        Wavelength grid in microns (monotonic).
    trans : (N,) array_like
        Total end-to-end transmission sampled on 'wave' (unitless). Used to scale the template.
    T : float
        Effective temperature used for building the template (K).
    lg : float
        Surface gravity 'log10(g[cm s^-2])' used for building the template.
    model : str
        Template model name (e.g. "BT-Settl", "mol_CO", ...).
    R : float
        Spectral resolving power of the data on 'wave' (used by filters and resampling).
    Rc : float or None
        High/low-pass cut-off in resolving-power units. If 'None', no filtering is applied.
    filter_type : {"gaussian", "gaussian_bis", "step", "smoothstep", "savitzky_golay"}
        High/low-pass filter type, passed to 'filtered_flux'.
    vsini_arr : (Nvsini,) array_like or None, optional
        Custom Vsini grid in km/s. If 'None', generated.
    rv : float or None, optional
        Expected RV in km/s (e.g., from ephemeris). Enables auto-correlation proxy and
        analytical sigma for that RV if 'sigma_l' is provided.
    vsini : float or None, optional
        Template rotational broadening parameter (km/s). Passed to 'get_template'.
    pca : object or None, optional
        PCA object with attributes 'components_' (2D array) and 'n_components' (int). If provided,
        the template is orthogonally projected to remove these modes before correlation.
    template_wo_broad : Spectrum or (N,) array_like or None, optional
        If 'compare_data=False': a 'Spectrum' object to use directly (bypasses 'get_template') before
        filtering. If 'compare_data=True': a 1D array on 'wave' representing the *other* dataset
        to compare with 'd'.
        If 'None', a template is generated from 'model', 'T', 'lg', etc.
    epsilon : float, optional
        Limb-darkening coefficient for broadening kernels (if used by 'get_template'). Default 0.8.
    fastbroad : bool, optional
        Use fast broadening (implementation detail of 'get_template'). Default True.
    calc_logL : bool, optional
        If True, compute log-likelihood vs Vsini using 'get_logL(d, t, sigma_l, method_logL)'. Default False.
    method_logL : {"classic", ...}, optional
        Log-likelihood method name forwarded to 'calc_logL'. Default "classic".
    sigma_l : (N,) array_like or None, optional
        Per-channel noise (same grid as 'wave'). Needed to return analytical 'sigma'.
    weight : (N,) array_like or None, optional
        Optional weights applied to both data and template prior to normalization (e.g. mask or inverse-variance).
    stellar_component : bool, optional
        If True and 'Rc' is not None, include the residual stellar term in the template:
        't = t - trans * Ss_HF * template_LF / Ss_LF'. Requires 'trans_Ss'. Default True.
    degrade_resolution : bool, optional
        Degrade template to R using 'Spectrum.degrade_resolution(..., R_output=R)' prior to filtering.
        If False, only interpolate to 'wave'. Default True.
    compare_data : bool, optional
        If True, interpret 'template' as another data vector to be compared to 'd'.
        Skips model generation and stellar-component handling. Default False.
    cut_fringes : bool, optional
        If True, pass the (weighted) template vector through 'cut_spectral_frequencies'
        with '(Rmin, Rmax)'. Default False.
    Rmin, Rmax : float or None, optional
        Band-stop limits in resolving-power space for 'cut_fringes'. Default None.
    target_name : str or None, optional
        Target label for plotting/logging only. Default None.
    verbose : bool, optional
        Disable the progress bar. Default True.
        
    Returns
    -------
    vsini_arr : (Nvsini,) ndarray
        The Vsini grid (km/s).
    CCF : (Nvsini,) ndarray
        Cross-correlation vs Vsini (dot product with a **unit-norm** template over valid channels).
    corr : (Nvsini,) ndarray
        Correlation strength vs Vsini: 'CCF / ||d||' over the valid masked region.
    CCF_bkg : None or (Nbkg, Nvsini) ndarray
        Background CCFs vs Vsini if 'd_bkg' is provided; otherwise 'None'.
    corr_auto : (Nvsini,) ndarray
        Auto-correlation proxy (template vs template broaded at 'vsini') scaled to unit data norm.
        If 'vsini=None', returns zeros.
    logL : (Nvsini,) ndarray
        Log-likelihood vs Vsini (zeros if 'calc_logL=False').
    sigma_CCF : float or None
        Analytical 1-sigma CCF uncertainty at 'vsini' if 'sigma_l' is provided;
        otherwise 'None'.
    """
    
    R_sampling = get_resolution(wavelength=wave, func=np.array)
    
    # --- Vsini grid
    if vsini_arr is None:
        vsini_arr = np.linspace(0, 100, 10_000)
    
    if vsini is not None and vsini not in vsini_arr:
        vsini_arr = np.sort(np.append(vsini_arr, vsini))
    
    # --- Initialize arrays
    Nvsini    = len(vsini_arr)
    CCF       = np.zeros((Nvsini)) + np.nan
    corr      = np.zeros((Nvsini)) + np.nan
    corr_auto = np.zeros((Nvsini)) + np.nan
    logL      = np.zeros((Nvsini)) + np.nan
    sigma_CCF = None
    CCF_bkg   = np.zeros((len(d_bkg), Nvsini)) + np.nan if d_bkg is not None else None
    
    # --- Templates preparation

    # The data to compare with is assimiled to the "template"
    if compare_data:
        if template_wo_broad is None:
            raise ValueError("When compare_data=True, provide 'template_wo_broad' array on the same grid.")
        template_HF = Spectrum(wavelength=wave, flux=template_wo_broad)
        template_LF = Ss_HF = Ss_LF = None
        
    # If not comparing data, calculating the raw template
    else:
        # Generate template if not provided
        if template_wo_broad is None:
            template_wo_broad = get_template(instru=instru, wave=wave, R=R, model=model, T=T, lg=lg, rv=rv, vsini=0, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model)

        # Degrade template to target grid
        template = get_spectrum_band(spectrum_instru=template_wo_broad, wave_band=wave, R_output=R, degrade_resolution=degrade_resolution)

        # Return if template only contains NaNs (i.e. if the crop of the molecular templates left only NaNs)
        if np.all(np.isnan(template.flux)): # Nothing to correlate
            return vsini_arr, CCF, corr, CCF_bkg, corr_auto, logL, sigma_CCF
        
        # Filter the template
        template_HF                        = template.copy()
        template_LF                        = template.copy()
        template_HF.flux, template_LF.flux = filtered_flux(template.flux, R=R_sampling, Rc=Rc, filter_type=filter_type)
        
        # Handle stellar component if needed
        if stellar_component and Rc is not None:
            Ss_HF, Ss_LF = get_Ss_HF_LF(trans_Ss=trans_Ss, trans=trans, wave=wave, R_sampling=R_sampling, Rc=Rc, filter_type=filter_type)
        else:
            Ss_HF = Ss_LF = None
    
    # --- Data preparation: apply weighting and mask zeros as NaN
    d_raw     = np.copy(d)
    w         = np.ones_like(wave) if weight is None else weight
    d         = d_raw * w
    d[d == 0] = np.nan
    if sigma_l is not None:
        sigma_l_raw = np.copy(sigma_l)
        sigma_l     = sigma_l_raw * w
    else:
        sigma_l_raw = sigma_l = None

    # --- Reference for auto-correlation
    if vsini is not None:
        # Already prepared "template"
        if compare_data:
            d_sim = np.copy(template_HF.flux)
        # Preparing the template
        else:
            template = template_wo_broad.broad(vsini, epsilon=epsilon, fastbroad=fastbroad)
            d_sim    = get_d_sim(instru=instru, d=d_raw, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=None, T=None, lg=None, rv=None, vsini=None, epsilon=None, fastbroad=None, airmass=None, star_spectrum=None, wave_model=None, template=template, degrade_resolution=degrade_resolution, stellar_component=stellar_component, trans_Ss=trans_Ss, pca=pca, cut_fringes=cut_fringes, Rmin=Rmin, Rmax=Rmax, target_name=target_name, renorm_d_sim=renorm_d_sim, sigma_l=sigma_l_raw, verbose=True)
        d_sim         *= w
        d, d_sim       = get_masked_quantity(data=d, template=d_sim, weight=w)
        template_broad = d_sim / np.sqrt(np.nansum(d_sim**2))
    else:
        d_sim          = None
        template_broad = None
        
    # ---------- Parallel sweep across (vsini) ----------
    # Init global context for workers
    global _VSINI_CTX
    _VSINI_CTX = dict(vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, d=d, sigma_l=sigma_l, d_bkg=d_bkg, w=w, trans=trans, template_HF=template_HF, template_LF=template_LF, Ss_HF=Ss_HF, Ss_LF=Ss_LF, template_broad=template_broad, Rc=Rc, pca=pca, cut_fringes=cut_fringes, R=R, Rmin=Rmin, Rmax=Rmax, target_name=target_name, stellar_component=stellar_component, compare_data=compare_data, calc_logL=calc_logL, method_logL=method_logL)

    print()
    with Pool(processes=max(1, cpu_count() // 2), initializer=init_vsini_worker, initargs=(_VSINI_CTX,)) as pool:    
        results = list(tqdm(pool.imap(process_correlation_vsini, ((i, vsini_arr[i]) for i in range(Nvsini))), total=Nvsini, desc=" get_CCF_1D_vsini()", disable=not verbose))
        for (i, CCF_value, corr_value, corr_auto_value, logL_value, CCF_bkg_value) in results:
            CCF[i]        = CCF_value
            corr[i]       = corr_value
            corr_auto[i]  = corr_auto_value
            logL[i]       = logL_value
            if CCF_bkg is not None:
                CCF_bkg[:, i] = CCF_bkg_value
        
    # --- Optional sigma at vsini (analytical)
    if vsini is not None and sigma_l is not None:
        sigma_CCF = np.sqrt(np.nansum(sigma_l**2 * template_broad**2))
        
    return vsini_arr, CCF, corr, CCF_bkg, corr_auto, logL, sigma_CCF



def plot_CCF_vsini(instru, band, target_name, d, d_bkg, wave, trans, R, Rc, filter_type, model, T, lg, rv, vsini_arr, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template_wo_broad=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, pca=None, cut_fringes=False, Rmin=None, Rmax=None, renorm_d_sim=True, sigma_l=None, calc_logL=False, method_logL="classic", weight=None, compare_data=False, verbose=True, show=True):    
    """
    Compute and plot the CCF (S/N and correlation units) — optional log-likelihood — and return metrics.

    This is a convenience wrapper around 'get_CCF_1D_vsini' that also
    converts the CCF to empirical S/N using propagated sigma_l.

    Parameters
    ----------
    instru : str
        Instrument name (passed to 'get_CCF_1D_vsini').
    band : str
        Spectral band label (for the plot title only).
    target_name : str
        Target label (for the plot title only).
    d : (N,) array_like
        Observed **high-pass** spectrum on 'wave' (NaNs allowed).
    d_bkg : None or (Nbkg, N) array_like
        Optional background spectra (for background CCFs/SNRs).
    trans_Ss : (N,) array_like
        Stellar flux on 'wave' (needed if 'stellar_component=True').
    wave : (N,) array_like
        Wavelength grid (µm).
    trans : (N,) array_like
        Total end-to-end transmission on 'wave'.
    T : float
        Effective temperature for the template (K).
    lg : float
        Surface gravity 'log10(g[cm s^-2])' for the template.
    model : str
        Template model name.
    R : float
        Spectral resolving power of the data.
    Rc : float or None
        High/low-pass cutoff in resolving-power units; if 'None', no filtering.
    filter_type : {"gaussian", "gaussian_bis", "step", "smoothstep", "savitzky_golay"}
        Filter type for 'filtered_flux'.
    vsini_arr : (Nvsini,) array_like or None, optional
        Custom Vsini grid in km/s.
    rv : float or None, optional
        Expected RV (km/s). Used to refine the peak estimate and define the wing region (±200 km/s).
    vsini : float, optional
        Rotational broadening for the template (km/s). Default 0.
    pca : object or None, optional
        PCA object with 'components_' and 'n_components'. If provided, modes are projected out from the template.
    template_wo_broad : Spectrum or (N,) array_like or None, optional
        See 'get_CCF_1D_vsini'. If 'compare_data=True', provide a 1D array.
    epsilon : float, optional
        Limb-darkening coefficient for broadening kernels. Default 0.8.
    fastbroad : bool, optional
        Use fast broadening inside 'get_template'. Default True.
    calc_logL : bool, optional
        If True, also plot log-likelihood vs RV. Default False.
    method_logL : {"classic", ...}, optional
        Method passed to 'calc_logL'. Default "classic".
    sigma_l : (N,) array_like or None, optional
        Per-channel noise. Enables analytical sigma in the printout (if available).
    weight : (N,) array_like or None, optional
        Optional weight array applied to data and template.
    stellar_component : bool, optional
        Include residual stellar term in the template when 'Rc' is not None. Default True.
    degrade_resolution : bool, optional
        Degrade the template to R before filtering. Default True.
    compare_data : bool, optional
        Treat 'template' as another dataset rather than a physical model. Default False.
    cut_fringes : bool, optional
        Apply fringe band-stop ('Rmin', 'Rmax') to the template. Default False.
    Rmin, Rmax : float or None, optional
        Band-stop limits in resolving-power space for 'cut_fringes'.
    verbose : bool, optional
        Print summary (peak S/N, correlation, sigma ratios). Default True.
    show : bool, optional
        Display the figure(s). Default True.

    Returns
    -------
    vsini_arr : (Nvsini,) ndarray
        RV grid (km/s).
    SNR : (Nvsini,) ndarray
        Empirical S/N of the CCF vs RV (noise estimated from RV wings).
    SNR_bkg : None or (Nbkg, Nvsini) ndarray
        Background CCFs converted to S/N by their wing standard deviation; 'None' if no backgrounds.
    corr : (Nvsini,) ndarray
        Correlation strength vs RV (normalized by ||data||).
    signal : float
        CCF peak value (not in S/N).
    sigma_CCF : float
        Empirical standard deviation of the CCF (wing-based).
    sigma_CCF : float or None
        Analytical sigma at 'rv' if 'sigma_l' available; else 'None'.
    corr_auto : (Nvsini,) ndarray
        Auto-correlation proxy (see 'get_CCF_1D_vsini').
    """
    
    # Compute the radial velocity and cross-correlation functions for the data and background
    vsini_arr, CCF, corr, CCF_bkg, corr_auto, logL, sigma_CCF = get_CCF_1D_vsini(instru=instru, d=d, d_bkg=d_bkg, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=model, T=T, lg=lg, rv=rv, vsini_arr=vsini_arr, vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model, template_wo_broad=template_wo_broad, degrade_resolution=degrade_resolution, stellar_component=stellar_component, trans_Ss=trans_Ss, pca=pca, cut_fringes=cut_fringes, Rmin=Rmin, Rmax=Rmax, target_name=target_name, renorm_d_sim=renorm_d_sim, sigma_l=sigma_l, calc_logL=calc_logL, method_logL=method_logL, weight=weight, compare_data=compare_data, verbose=verbose)
    
    # Vsini prior
    if vsini is None:
        if calc_logL:
            vsini = vsini_arr[np.nanargmax(logL)]
        else:
            vsini = vsini_arr[np.nanargmax(CCF)]
    prior_width = np.nanmedian(10 * 1e-3*c/R)  # [km/s]
    mask_prior  = (vsini_arr >= vsini - prior_width) & (vsini_arr <= vsini + prior_width)
    if np.count_nonzero(mask_prior) < 3:
        raise ValueError(f"Vsini prior contains too few points: {np.count_nonzero(mask_prior)}. Increase prior_width or use a finer vsini_arr grid.")
    
    # Estimating Vsini
    if calc_logL:
        P                                            = np.exp(logL[mask_prior] - np.nanmax(logL[mask_prior]))
        dv                                           = np.gradient(vsini_arr[mask_prior])
        norm                                         = np.nansum(P * dv)
        P                                            = P / norm
        optimal_values, uncertainties, bounds_1sigma = estimate_uncertainties_1sigma(P=P, func_marg=np.nansum, params=[vsini_arr[mask_prior]])
        vsini                                        = optimal_values[0]
        sigma_vsini                                  = uncertainties[0]
    else:
        vsini       = vsini_arr[mask_prior][np.nanargmax(CCF[mask_prior])]
        sigma_vsini = None
    
    # Calculate Signal-to-Noise Ratio (SNR) from sigma_CCF (analytical noise)
    SNR     = CCF / sigma_CCF
    signal  = np.nanmax(CCF)
    max_SNR = np.nanmax(SNR[mask_prior])
    if d_bkg is not None: # Plot background SNR if background data is provided
        SNR_bkg = np.zeros((len(d_bkg), len(vsini_arr)))
        for i in range(len(d_bkg)):
            SNR_bkg[i] = CCF_bkg[i] / np.nanstd(CCF_bkg[i])
    else:
        SNR_bkg = None
        
    if show:
        
        # Plot the CCF in S/N and correlation units
        plt.figure(figsize=(10, 6), dpi=300)
        ax1 = plt.gca()        
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.minorticks_on()
        ax1.set_xlim(vsini_arr[0], vsini_arr[-1])        
        title = f'CCF of {target_name.replace("_", " ")} on {band}-band of {instru} with {model} template'
        if compare_data:
            title += f"\nwith $R_c$ = {Rc}"
        else:
            if lg is None:
                title += f"\nat $T$ = {T:.0f}K and RV = {rv:.1f} km/s\nwith $R_c$ = {Rc:.0f}"
            else:
                title += f"\nat $T$ = {T:.0f}K, "+r"$\log g$ = "+f"{lg:.1f} and RV = {rv:.1f} km/s\nwith $R_c$ = {Rc:.0f}"
        ax1.set_title(title, fontsize=18, pad=15)        
        ax1.set_xlabel("Rotational Velocity [km/s]", fontsize=14, labelpad=10)
        ax1.set_ylabel("CCF [S/N]", fontsize=14, labelpad=10)
        ax1.plot([], [], 'gray', label="Noise", alpha=0.5)        
        if d_bkg is not None:  # Ajout du fond si disponible
            for i in range(len(d_bkg)):
                ax1.plot(vsini_arr, SNR_bkg[i], 'gray', alpha=min(max(0.1, 1 / len(d_bkg)) + 0.1, 1))
        ax1.plot(vsini_arr, corr_auto * max_SNR / np.nanmax(corr_auto), "k", label="Auto-correlation")
        ax1.plot(vsini_arr, SNR, label=f"{target_name.replace('_', ' ')}", c="steelblue", zorder=3, linewidth=2)
        ax1.axvline(vsini, c='crimson', ls='--', label=f"Vsini = {round(vsini, 1)} km/s", alpha=0.5, linewidth=1.5)
        ax1.axvline(0, c='k', ls='-', alpha=0.5, linewidth=1)
        ymin, ymax = ax1.get_ylim()
        ax2 = ax1.twinx()
        ax2.minorticks_on()
        ax2.set_ylabel("Correlation Strength", fontsize=14, labelpad=20, rotation=270)
        ax2.tick_params(axis='y')
        ax2.set_ylim(ymin * corr[SNR==max_SNR][0] / max_SNR, ymax * corr[SNR==max_SNR][0] / max_SNR)
        ax1.legend(fontsize=12, loc="upper right", frameon=True, edgecolor="gray", facecolor="whitesmoke")        
        plt.tight_layout()
        plt.show()
        
        if verbose: # Print maximum S/N and correlation
            print(f"  CCF: max S/N ({max_SNR:.1f}) and correlation ({np.nanmax(corr[mask_prior]):.4f}) for rv = {vsini_arr[mask_prior][np.nanargmax(SNR[mask_prior])]:.3f} km/s")
            
        # Plot log-likelihood (if required)
        if calc_logL:
            vsini_lo, vsini_hi = bounds_1sigma[0]
            vsini_min       = max(vsini_arr[mask_prior][0],  vsini-5*sigma_vsini)
            vsini_max       = min(vsini_arr[mask_prior][-1], vsini+5*sigma_vsini)

            # Plot the logL
            plt.figure(figsize=(12, 7), dpi=300)
            ax1 = plt.gca()        
            ax1.grid(True, which="major", linestyle="-", linewidth=0.55, alpha=0.35, color="0.45")
            ax1.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25, color="0.55")
            ax1.minorticks_on()    
            ax1.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.1)
            ax1.tick_params(axis="both", which="minor", length=3.5, width=0.8)
            ax1.set_xlim(vsini_min, vsini_max)        
            ax1.set_ylim(0,         1.1)
            title = title.replace('CCF', 'Rotational velocity probability')
            ax1.set_title(title, fontsize=16, pad=15)        
            ax1.set_xlabel("Obsevsinied Rotational Velocity [km/s]",                     fontsize=16, labelpad=10)
            ax1.set_ylabel("Density probability function from logL [normalized]", fontsize=16, labelpad=10)
            ax1.plot(vsini_arr[mask_prior], P/np.nanmax(P), label=f"{target_name.replace('_', ' ')}", c="steelblue", zorder=3, linewidth=2)
            ax1.axvline(vsini,    c='crimson', ls='--', label=rf"$vsini_{{obs}}$ = {vsini:.2f} $\pm$ {sigma_vsini:.2f} km/s", alpha=0.7, linewidth=1.5)
            ax1.axvline(vsini_lo, c='crimson', ls=':',                                                            alpha=0.7, linewidth=1.5)
            ax1.axvline(vsini_hi, c='crimson', ls=':',                                                            alpha=0.7, linewidth=1.5)
            ax1.axvline(0, c='k', ls='-', alpha=0.5, linewidth=1)
            ax1.legend(loc="upper right", fontsize=14, frameon=True, fancybox=True, framealpha=0.92, edgecolor="gray", facecolor="white")
            plt.tight_layout()
            plt.show()
            
            if verbose: # Print maximum log-likelihood
                print(f"  max logL for vsini = {vsini:.3f} ± {sigma_vsini:.3f} km/s")
    
    return vsini_arr, SNR, SNR_bkg, corr, signal, sigma_CCF, corr_auto, logL, vsini, sigma_vsini




# -------------------------------------------------------------------------
# Parameters retrieval computations
# -------------------------------------------------------------------------

def get_priors(SNR_CCF, SNR_estimate, wave, d, R, model, T, lg, rv, vsini):
    # --- Half-widths for parameter grids (better scaling with SNR and R) ---
    if SNR_CCF is None or not np.isfinite(SNR_CCF):
        SNR_CCF = 1.0
    else:
        SNR_CCF = max(SNR_CCF, 1.0)
    
    # Effective number of resolution elements
    valid = np.isfinite(d)
    wv    = wave[valid]                         # [µm]
    dw    = np.gradient(wv)                     # [µm/px] 
    FWHM  = wv / np.nanmedian(R)                              # [µm] width of a resolution element
    N_eff = np.nansum(dw / FWHM)
    N_eff = max(N_eff, 1.0)
    
    # --- Temperature ---
    # Rule of thumb: uncertainty scales ~ T / (SNR * sqrt(N_eff))
    sigma_T = (T / SNR_CCF) / np.sqrt(N_eff)
    DT      = max(5 * sigma_T, 10.0)   # 5σ half-width, min 10 K
    
    # --- log g ---
    sigma_lg = (1.0 / SNR_CCF) / np.sqrt(N_eff) # dex
    Dlg      = max(5 * sigma_lg, 0.5)           # 5σ half-width, min 0.5 dex
    
    # Instrumental resolution element in velocity [km/s]
    delta_v_res = np.nanmedian(c * 1e-3 / R) # [km/s]
    
    # --- v sin i ---
    w_inst      = delta_v_res
    w_rot       = max(vsini, 1.0)   # rough proxy
    w_tot       = np.sqrt(w_inst**2 + w_rot**2)
    sigma_vsini = (w_tot**2 / (0.66 * SNR_CCF * np.sqrt(N_eff) * np.sqrt(max(w_tot**2 - w_inst**2, 1e-6))))
    Dvsini      = 5 * max(delta_v_res, sigma_vsini)  # ≥ resolution elements
    
    # --- RV ---
    sigma_rv = delta_v_res / (SNR_CCF * np.sqrt(N_eff))
    Drv      = 5 * max(delta_v_res, sigma_rv)
    
    # Base grids from the model bounds
    T_grid, lg_grid = get_model_grid(model)

    if np.nanmedian(R) < 5000: # No vsini retrieval
        N = 30
    else:
        N = 20

    T_arr  = np.linspace(max(T_grid[0],   T  - DT),  min(T_grid[-1],  T  + DT),  N + 1, dtype=np.float64)
    if lg is None:
        lg_arr = lg_grid
    else:
        lg_arr = np.linspace(max(lg_grid[0],  lg - Dlg), min(lg_grid[-1], lg + Dlg), N + 1, dtype=np.float64)
    
    if np.nanmedian(R) > 5000: # Enough resolution to retrieve Vsini
        vsini_arr = np.linspace(max(0, vsini - Dvsini), min(80, vsini + Dvsini), N + 1, dtype=np.float64)
    else:
        vsini_arr = np.array([vsini], dtype=np.float64)

    if SNR_estimate:
        # refined sampling near rv + large wings
        left   = np.linspace(-1000,           rv - 0.5 * Drv,  100,                     dtype=np.float64)
        core   = np.linspace(rv - 0.5 * Drv,  rv + 0.5 * Drv,  max(10, int(Drv / 0.5)), dtype=np.float64)
        right  = np.linspace(rv + 0.5 * Drv,  1000,            100,                     dtype=np.float64)
        rv_arr = np.concatenate([left, core, right])
    else:
        rv_arr = np.linspace(rv - Drv, rv + Drv, N + 1, dtype=np.float64)
    
    return T_arr, lg_arr, rv_arr, vsini_arr, Drv



def doppler_shift_flux_same_grid(wave, flux, rv):
    """
    Fast equivalent of Spectrum.doppler_shift(rv, renorm=False).flux
    for a fixed wavelength grid.
    """
    if np.isclose(rv, 0.0, atol=1e-12):
        return flux  # no copy needed if caller does not modify it
    beta = 1000.0 * rv / c
    if abs(beta) >= 1:
        raise ValueError(f"Invalid RV={rv} km/s gives |beta| >= 1.")
    doppler  = np.sqrt((1.0 + beta) / (1.0 - beta))
    wave_src = wave / doppler
    return np.interp(wave_src, wave, flux, left=np.nan, right=np.nan)

def init_worker(ctx):
    global _TLG_CTX
    _TLG_CTX = ctx

def process_parameters_estimation(args):
    """
    Worker: compute (corr/SNR/logL) on a (vsini, RV) 2D subgrid for a fixed (T, log g).

    Parameters
    ----------
    args : tuple
        Packed arguments produced in 'parameters_retrieval' (kept as-is to stay picklable).

    Returns
    -------
    i, j : int
        Indices in (T, log g).
    corr_2D, SNR_2D, logL_2D, logL_2D_sim : (Nvs, Nrv) arrays
        2D slices to be inserted into the final 4D cubes.
    """
    i, j, T, lg = args
    
    instru                 = _TLG_CTX['instru']
    R_sampling              = _TLG_CTX['R_sampling']
    vsini_arr              = _TLG_CTX['vsini_arr']
    rv_arr                 = _TLG_CTX['rv_arr']
    d                      = _TLG_CTX['d']
    C_pca                  = _TLG_CTX['C_pca']
    model                  = _TLG_CTX['model']
    wave                   = _TLG_CTX['wave']
    trans_w                = _TLG_CTX['trans_w']
    epsilon                = _TLG_CTX['epsilon']
    fastbroad              = _TLG_CTX['fastbroad']
    airmass                = _TLG_CTX['airmass']
    star_spectrum          = _TLG_CTX['star_spectrum']
    R                      = _TLG_CTX['R']
    Rc                     = _TLG_CTX['Rc']
    filter_type            = _TLG_CTX['filter_type']
    sigma_l                = _TLG_CTX['sigma_l']
    calc_logL              = _TLG_CTX['calc_logL']
    method_logL            = _TLG_CTX['method_logL']
    trans_w_Ss_HF_LF_ratio = _TLG_CTX['trans_w_Ss_HF_LF_ratio']
    SNR_estimate           = _TLG_CTX['SNR_estimate']
    mask_noise             = _TLG_CTX['mask_noise']
    mask_prior              = _TLG_CTX['mask_prior']
    stellar_component      = _TLG_CTX['stellar_component']
    degrade_resolution     = _TLG_CTX['degrade_resolution']
    d_sim                  = _TLG_CTX['d_sim']
    wave_model             = _TLG_CTX['wave_model']
    valid_base             = _TLG_CTX['valid_base']
    valid_logL_base        = _TLG_CTX['valid_logL_base']

    Nvsini      = len(vsini_arr)
    Nrv         = len(rv_arr)
    corr_2D     = np.full((Nvsini, Nrv), np.nan, dtype=np.float64)
    SNR_2D      = np.full((Nvsini, Nrv), np.nan, dtype=np.float64)
    auto_2D     = np.full((Nvsini, Nrv), np.nan, dtype=np.float64)
    logL_2D     = np.full((Nvsini, Nrv), np.nan, dtype=np.float64)
    logL_2D_sim = np.full((Nvsini, Nrv), np.nan, dtype=np.float64)
    
    # Build a base template at (T, lg) with/without pre-broadening/pre-shift
    template    = get_template(instru=instru, wave=wave, R=R, model=model, T=T, lg=lg, rv=0, vsini=0, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model, to_counts=False)
    wave_model  = template.wavelength     # [µm]
    dwave_model = np.gradient(wave_model) # [µm/bin]
    for k, vsini in enumerate(vsini_arr):
            
        # Broadening
        template_broad = template.broad(vsini, epsilon=epsilon, fastbroad=fastbroad)    
        
        # To be homogenous to counts: ph/bin or e-/bin or ADU/bin
        template_broad.flux *= wave_model * dwave_model
        
        # Degrade to target grid
        template_broad = get_spectrum_band(spectrum_instru=template_broad, wave_band=wave, R_output=R, degrade_resolution=degrade_resolution)
        
        if np.all(~np.isfinite(template_broad.flux)):
            continue
        
        # HF/LF split once (then reuse with Doppler/broadening ops)
        template_broad_HF, template_broad_LF = filtered_flux(flux=template_broad.flux, R=R_sampling, Rc=Rc, filter_type=filter_type)
        
        # For S/N computation: reference for sigma_auto_correlation (and subtract it later)
        if SNR_estimate:
            template_auto = trans_w*template_broad_HF
            if stellar_component and Rc is not None:
                template_auto = template_auto - trans_w_Ss_HF_LF_ratio * template_broad_LF # better without the residual stellar contributions for the auto-correlation
            template_auto[~valid_base] = np.nan
            if C_pca is not None:
                ta0            = np.nan_to_num(template_auto, nan=0.0, posinf=0.0, neginf=0.0)
                template_auto -= (ta0 @ C_pca.T) @ C_pca
            norm_auto = np.sqrt(np.nansum(template_auto**2))
            if np.isfinite(norm_auto) and norm_auto > 0:
                template_auto /= norm_auto
            else:
                template_auto = None
        
        for l, rv in enumerate(rv_arr):

            # Shifting
            template_broad_HF_shift = doppler_shift_flux_same_grid(wave=wave, flux=template_broad_HF, rv=rv)
            if stellar_component and Rc is not None:
                template_broad_LF_shift = doppler_shift_flux_same_grid(wave=wave, flux=template_broad_LF, rv=rv)
            
            # Computing the template
            t = trans_w * template_broad_HF_shift
            # Adding the stellar component (if required)
            if stellar_component and Rc is not None:
                t += - trans_w_Ss_HF_LF_ratio * template_broad_LF_shift
            
            # Masking like data
            t[np.isnan(d)] = np.nan
            
            # Optionnal PCA modes removing (template-side projection)
            if C_pca is not None:
                t0 = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                t -= (t0 @ C_pca.T) @ C_pca
            
            # Creating mask
            valid = valid_base & np.isfinite(t)
            
            # Normalizing the template
            norm_t = np.sqrt(np.sum(t[valid]**2))
            if not np.isfinite(norm_t) or norm_t == 0:
                continue
            t[valid] /= norm_t
            
            # CCF computations
            signal_CCF = np.sum( d[valid] * t[valid] )
            if SNR_estimate:
                SNR_2D[k, l] = signal_CCF
                if template_auto is not None:
                    valid_auto = valid & np.isfinite(template_auto)
                    if np.any(valid_auto):
                        auto_2D[k, l] = np.sum(template_auto[valid_auto] * t[valid_auto])  
            else:
                d_norm = np.sqrt(np.sum(d[valid]**2))
                if np.isfinite(d_norm) and d_norm > 0:
                    corr_2D[k, l] = signal_CCF / d_norm
                if calc_logL:
                    valid_logL = valid & valid_logL_base
                    if valid_logL.sum() >= 3:
                        logL_2D[k, l] = get_logL(d[valid_logL], t[valid_logL], sigma_l[valid_logL], method=method_logL)
                        if d_sim is not None:
                            valid_logL_sim = valid_logL & np.isfinite(d_sim)
                            if valid_logL_sim.sum() >= 3:
                                logL_2D_sim[k, l] = get_logL(d_sim[valid_logL_sim], t[valid_logL_sim], sigma_l[valid_logL_sim], method=method_logL)
        
        # S/N computation
        if SNR_estimate:
            finite_noise = mask_noise & np.isfinite(SNR_2D[k, :])
            if np.sum(finite_noise) < 3:
                continue
            SNR_2D[k, :] -= np.nanmean(SNR_2D[k, finite_noise])
            if np.any(np.isfinite(auto_2D[k, finite_noise])):
                auto_2D[k, :] -= np.nanmean(auto_2D[k, finite_noise])
            sigma2_tot = np.nanvar(SNR_2D[k, finite_noise])
            auto_max       = np.nanmax(auto_2D[k, :]) if np.any(np.isfinite(auto_2D[k, :])) else np.nan
            peak_max       = np.nanmax(SNR_2D[k, mask_prior]) if np.any(np.isfinite(SNR_2D[k, mask_prior])) else np.nan
            if (np.isfinite(auto_max) and auto_max != 0 and np.isfinite(peak_max) and np.isfinite(sigma2_tot)):
                sigma2_auto = np.nanvar(auto_2D[k, finite_noise] * peak_max / auto_max)
            else:
                sigma2_auto = np.nan
            if np.isfinite(sigma2_auto) and sigma2_auto < sigma2_tot:
                sigma_CCF = np.sqrt(sigma2_tot - sigma2_auto)
            else:
                sigma_CCF = np.sqrt(sigma2_tot)
            if np.isfinite(sigma_CCF) and sigma_CCF > 0:
                SNR_2D[k, :] /= sigma_CCF
            else:
                SNR_2D[k, :] = np.nan

    return (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim)



def parameters_retrieval(instru, band, target_name, d, wave, trans, R, Rc, filter_type, model, T_arr, lg_arr, rv_arr, vsini_arr, T, lg, rv, vsini, epsilon=0.8, fastbroad=True, airmass=2.0, star_spectrum=None, wave_model=None, template=None, degrade_resolution=True, stellar_component=True, trans_Ss=None, pca=None, calc_d_sim=False, renorm_d_sim=True, sigma_l=None, calc_logL=False, method_logL="classic", weight=None, SNR_estimate=False, SNR_CCF=None, force_new_est=False, save=True, fastcurves=False, exposure_time=None, show=True, verbose=True):
        
    R_sampling = get_resolution(wavelength=wave, func=np.array)
    
    if calc_logL and sigma_l is None:
        raise ValueError("sigma_l must be provided when calc_logL=True.")
    
    # ---------- Try to load cached results ----------
    try:
        
        # Prefix and suffix
        def make_hash_dict(d):
            payload = json.dumps(d, sort_keys=True, default=str).encode()
            return hashlib.md5(payload).hexdigest()[:10]
        outdir    = Path(f"parameters retrieval/{instru}")
        outdir.mkdir(parents=True, exist_ok=True)
        prefix    = f"{outdir}/parameters_retrieval"
        grid_hash = make_hash_dict({"T_arr": np.asarray(T_arr).tolist() if T_arr is not None else None, "lg_arr": np.asarray(lg_arr).tolist() if lg_arr is not None else None, "rv_arr": np.asarray(rv_arr).tolist() if rv_arr is not None else None, "vsini_arr": np.asarray(vsini_arr).tolist() if vsini_arr is not None else None, "T_center": T, "lg_center": lg, "rv_center": rv, "vsini_center": vsini, "SNR_CCF": SNR_CCF, "R": R, "Rc": Rc, "filter_type": filter_type, "epsilon": epsilon, "fastbroad": fastbroad, "airmass": airmass, "calc_logL": calc_logL, "method_logL": method_logL, "SNR_estimate": SNR_estimate, "stellar_component": stellar_component, "degrade_resolution": degrade_resolution})
        suffix    = f"_{instru}_{band}_{target_name}_{np.nanmedian(R):.0f}_{Rc}_{model}_{grid_hash}.fits"
        
        if force_new_est:
            raise RuntimeError(f"Forced recomputation requested: force_new_est = {force_new_est}")
        
        # Grids
        T_arr     = fits.getdata(prefix + "_T_arr"     + suffix)
        lg_arr    = fits.getdata(prefix + "_lg_arr"    + suffix)
        vsini_arr = fits.getdata(prefix + "_vsini_arr" + suffix)
        rv_arr    = fits.getdata(prefix + "_rv_arr"    + suffix)
        NT        = len(T_arr)
        Nlg       = len(lg_arr)
        Nvsini    = len(vsini_arr)
        Nrv       = len(rv_arr)
        
        # Hyper-cubes
        if SNR_estimate:
            SNR_4D  = fits.getdata(prefix + "_SNR_4D" + suffix)
            corr_4D = logL_4D = logL_sim_4D = None
        else:
            corr_4D = fits.getdata(prefix + "_corr_4D" + suffix)
            if calc_logL:
                logL_4D = fits.getdata(prefix + f"_logL_4D_{method_logL}" + suffix)
                if calc_d_sim:
                    logL_sim_4D = fits.getdata(prefix + f"_logL_sim_4D_{method_logL}" + suffix)
                else:
                    d_sim = logL_sim_4D = None
            else:
                logL_4D = logL_sim_4D = None
            SNR_4D = None
        
        if verbose:
            print("\nLoaded cached parameter-estimation products...")
            
    # Compute new estimations if files are missing or forced update
    except Exception as e:
        if verbose and not fastcurves:
            print(f"\nNew parameter grid computation: {e}")
            
        # ---------- Construct grids if needed ----------
        if any(arr is None for arr in (T_arr, lg_arr, vsini_arr, rv_arr)):
            if any(x is None for x in (T, lg, vsini, rv)):
                if "mol" in model and lg is None:
                    pass
                else:
                    raise ValueError("If explicit grids are not provided, you must supply T, lg, vsini, rv to center the search.")
            T_arr, lg_arr, rv_arr, vsini_arr, Drv = get_priors(SNR_CCF=SNR_CCF, SNR_estimate=SNR_estimate, wave=wave, d=d, R=R, model=model, T=T, lg=lg, rv=rv, vsini=vsini)
            
        # Adding x in x_arr if necessary
        T_arr, lg_arr, vsini_arr, rv_arr = [np.sort(np.append(arr, val)) if (val is not None and val not in arr) else arr for arr, val in [(T_arr, T), (lg_arr, lg), (vsini_arr, vsini), (rv_arr, rv)]]
        
        # ---------- Build stellar HF/LF terms if required ----------
        if stellar_component and Rc is not None:
            Ss_HF, Ss_LF = get_Ss_HF_LF(trans_Ss=trans_Ss, trans=trans, wave=wave, R_sampling=R_sampling, Rc=Rc, filter_type=filter_type)
        else:
            Ss_HF = Ss_LF = None
            
        # --- Data preparation: apply weighting and mask zeros as NaN
        d_raw     = np.copy(d)
        w         = np.ones_like(wave) if weight is None else weight
        d         = d_raw * w
        d[d == 0] = np.nan
        if sigma_l is not None:
            sigma_l_raw = np.copy(sigma_l)
            sigma_l     = sigma_l_raw * w
        else:
            sigma_l_raw = sigma_l = None
        
        # ---------- (Optional) Noiseless simulated data for logL_sim ----------
        if calc_d_sim and calc_logL:
            d_sim    = get_d_sim(instru=instru, d=d_raw, wave=wave, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=model, T=T, lg=lg, rv=rv, vsini=vsini, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, wave_model=wave_model, template=template, degrade_resolution=degrade_resolution, stellar_component=stellar_component, trans_Ss=trans_Ss, pca=pca, cut_fringes=None, Rmin=None, Rmax=None, target_name=None, renorm_d_sim=renorm_d_sim, sigma_l=sigma_l_raw, verbose=verbose)
            d_sim   *= w
            d, d_sim = get_masked_quantity(data=d, template=d_sim, weight=w)
        else:
            d_sim = None
        
        #  ---------- Keeping same mask across RV for logL computations (if needed)
        if calc_logL:
            dlambda = wave[:, None] * 1000*rv_arr / c
            dwave   = np.nanmean(np.gradient(wave))
            dpx_rv  = int(np.ceil(np.nanmax(np.abs(dlambda / dwave))))
            if 2*dpx_rv >= len(wave):
                raise ValueError("RV padding removes the full spectrum. Check rv_arr or wavelength sampling.")
            mask_dpx_rv = np.ones(len(wave), dtype=bool)
            if dpx_rv > 0:
                mask_dpx_rv[:dpx_rv]  = False
                mask_dpx_rv[-dpx_rv:] = False
        else:
            mask_dpx_rv = None
            
        # --- Pre-compute RV masks for SNR_estimate (if needed)
        if SNR_estimate:
            mask_noise = (rv_arr > rv+200) | (rv_arr < rv-200)
            mask_prior  = (rv_arr < rv+25)  & (rv_arr > rv-25)
        else:
            mask_noise = mask_prior = None
        
        # --- Pre-computing trans*w
        trans_w = trans * w
        
        # --- Pre-computing trans_w_Ss_HF_LF_ratio
        if stellar_component and Rc is not None:
            valid_ratio                         = np.isfinite(trans_w) & np.isfinite(Ss_HF) & np.isfinite(Ss_LF) & (Ss_LF != 0)
            trans_w_Ss_HF_LF_ratio              = np.full((len(Ss_HF)), np.nan, dtype=np.float64)
            trans_w_Ss_HF_LF_ratio[valid_ratio] = trans_w[valid_ratio] * Ss_HF[valid_ratio] / Ss_LF[valid_ratio]
        else:
            trans_w_Ss_HF_LF_ratio = None
        
        # --- Pre-computing the base valid mask
        valid_base = (d !=0 ) & np.isfinite(d) & np.isfinite(w) & (w != 0)
        if calc_logL:
            valid_logL_base = valid_base & mask_dpx_rv & np.isfinite(sigma_l) & (sigma_l > 0)
        else:
            valid_logL_base = None
                    
        # --- Pre-computing the PCA matrix components
        if pca is not None:
            C_pca = np.asarray(pca.components_[:pca.n_components], dtype=np.float64)
        else:
            C_pca = None
        
        # ---------- Parallel sweep across (T, lg) ----------
        global _TLG_CTX
        _TLG_CTX    = dict(instru=instru, R_sampling=R_sampling, vsini_arr=vsini_arr, rv_arr=rv_arr, d=d, C_pca=C_pca, model=model, wave=wave, trans_w=trans_w, epsilon=epsilon, fastbroad=fastbroad, airmass=airmass, star_spectrum=star_spectrum, R=R, Rc=Rc, filter_type=filter_type, sigma_l=sigma_l, calc_logL=calc_logL, method_logL=method_logL, trans_w_Ss_HF_LF_ratio=trans_w_Ss_HF_LF_ratio, SNR_estimate=SNR_estimate, mask_noise=mask_noise, mask_prior=mask_prior, stellar_component=stellar_component, degrade_resolution=degrade_resolution, d_sim=d_sim, wave_model=wave_model, valid_base=valid_base, valid_logL_base=valid_logL_base)
        NT          = len(T_arr)
        Nlg         = len(lg_arr)
        Nvsini      = len(vsini_arr)
        Nrv         = len(rv_arr)
        corr_4D     = np.full((NT, Nlg, Nvsini, Nrv), np.nan, dtype=np.float64)
        SNR_4D      = np.full((NT, Nlg, Nvsini, Nrv), np.nan, dtype=np.float64)
        logL_4D     = np.full((NT, Nlg, Nvsini, Nrv), np.nan, dtype=np.float64)
        logL_sim_4D = np.full((NT, Nlg, Nvsini, Nrv), np.nan, dtype=np.float64)
        
        if verbose:
            print()
        nproc = max(1, cpu_count() // 2)            
        with Pool(processes=nproc, initializer=init_worker, initargs=(_TLG_CTX,),) as pool:
            for (i, j, corr_2D, SNR_2D, logL_2D, logL_2D_sim) in tqdm(pool.imap_unordered(process_parameters_estimation, [(i, j, T_arr[i], lg_arr[j]) for i in range(NT) for j in range(Nlg)]), total=NT*Nlg, disable=not verbose, desc="Parameters estimation"):
                corr_4D[i, j, :, :]     = corr_2D
                SNR_4D[i, j, :, :]      = SNR_2D
                logL_4D[i, j, :, :]     = logL_2D
                logL_sim_4D[i, j, :, :] = logL_2D_sim
                
        # ---------- Save ----------
        if save:
            fits.writeto(prefix + "_T_arr"     + suffix, T_arr,     overwrite=True)
            fits.writeto(prefix + "_lg_arr"    + suffix, lg_arr,    overwrite=True)
            fits.writeto(prefix + "_rv_arr"    + suffix, rv_arr,    overwrite=True)
            fits.writeto(prefix + "_vsini_arr" + suffix, vsini_arr, overwrite=True)
            if SNR_estimate:
                fits.writeto(prefix + "_SNR_4D" + suffix, SNR_4D, overwrite=True)
            else:
                fits.writeto(prefix + "_corr_4D" + suffix, corr_4D, overwrite=True)
                if calc_logL:
                    fits.writeto(prefix + f"_logL_4D_{method_logL}" + suffix, logL_4D, overwrite=True)
                    if calc_d_sim:
                        fits.writeto(prefix + f"_logL_sim_4D_{method_logL}" + suffix, logL_sim_4D, overwrite=True)

    # ---------- Optional visualization ----------
    if show and NT > 2 and Nlg > 2 and not fastcurves:
        
        # SNR Map
        if SNR_estimate:
            R_med       = float(np.nanmedian(R))
            plot_window = 5.0 * c * 1e-3 / R_med  # [km/s]
            valid_rv    = (rv_arr >= rv - plot_window) & (rv_arr <= rv + plot_window)
            if np.sum(valid_rv) == 0:
                raise ValueError("No RV point remains inside the S/N plotting window.")
            SNR_4D_plot = SNR_4D[:, :, :, valid_rv]
            rv_arr_plot = rv_arr[valid_rv]
            if not np.any(np.isfinite(SNR_4D_plot)):
                raise ValueError("SNR_4D contains no finite value in the selected RV plotting window.")
            idx_max_SNR     = np.unravel_index(np.nanargmax(SNR_4D_plot), SNR_4D_plot.shape)
            SNR_2D          = np.nan_to_num(SNR_4D_plot[:, :, idx_max_SNR[2], idx_max_SNR[3]].transpose())
            T_SNR_found     = T_arr[idx_max_SNR[0]]
            lg_SNR_found    = lg_arr[idx_max_SNR[1]]
            vsini_SNR_found = vsini_arr[idx_max_SNR[2]]
            rv_SNR_found    = rv_arr_plot[idx_max_SNR[3]]
            print(f" {'Maximum S/N:':<25} {np.nanmax(SNR_4D_plot):>6.3f}   =>  T = {T_SNR_found:.0f} K,  log(g) = {lg_SNR_found:.2f},  RV = {rv_SNR_found:.1f} km/s")
            label_text = rf"max at $T_\mathrm{{eff}}$ = {T_SNR_found:.0f} K, {lg_SNR_found}, $V\sin i$ = {vsini_SNR_found:.1f} km/s, $RV$ = {rv_SNR_found:.1f} km/s"
            y_for_plot = lg_arr
            y_point    = lg_SNR_found
            ylabel     = r"$\log\,g$ [dex (cm/s²)]"
            fig, ax    = plt.subplots(dpi=300, figsize=(10, 6))            
            ax.set_title(f'S/N of {target_name.replace("_", " ")} with {model} models\n{band}-band spectrum from {instru} at $R_c$ = {Rc}', fontsize=16, pad=20)            
            ax.set_xlabel(r"$T_\mathrm{eff}$ [K]", fontsize=14)
            ax.set_ylabel(r"$\log\,g$ [dex (cm/s²)]", fontsize=14)            
            pcm  = ax.pcolormesh(T_arr, y_for_plot, SNR_2D, cmap='rainbow', shading='auto', vmin=np.nanmin(SNR_2D), vmax=np.nanmax(SNR_2D))            
            cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046, format='%.1f')
            cbar.set_label("S/N", fontsize=14, labelpad=18, rotation=270)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.minorticks_on()            
            ax.plot(T_SNR_found, y_point, marker='X', color='white', markersize=9, markeredgecolor='black', markeredgewidth=1.5, label=label_text)
            ax.contour(T_arr, lg_arr, SNR_2D, levels=12, linewidths=0.8, colors='white', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.minorticks_on()            
            ax.legend(fontsize=11, loc='lower left', frameon=True, facecolor='whitesmoke', edgecolor='gray')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)            
            plt.tight_layout()
            plt.show()

        # Correlation Map
        else:
            idx_max_corr     = np.unravel_index(np.nanargmax(corr_4D), corr_4D.shape)
            corr_2D          = np.nan_to_num(corr_4D[:, :, idx_max_corr[2], idx_max_corr[3]].transpose())
            T_corr_found     = T_arr[idx_max_corr[0]]
            lg_corr_found    = lg_arr[idx_max_corr[1]]
            vsini_corr_found = vsini_arr[idx_max_corr[2]]
            rv_corr_found    = rv_arr[idx_max_corr[3]]
            print(f" {'Maximum correlation:':<25} {np.nanmax(corr_4D):>6.3f}     =>  T = {T_corr_found:.0f} K,  log(g) = {lg_corr_found:.2f},  RV = {rv_corr_found:.1f} km/s")
            label_text = rf"max at $T_\mathrm{{eff}}$ = {T_corr_found:.0f} K, $\log\,g$ = {lg_corr_found:.2f}, $V\sin i$ = {vsini_corr_found:.1f} km/s, $RV$ = {rv_corr_found:.1f} km/s"
            y_for_plot = lg_arr
            y_point    = lg_corr_found
            ylabel     = r"$\log\,g$ [dex (cm/s²)]"
            fig, ax    = plt.subplots(dpi=300, figsize=(10, 6))            
            ax.set_title(f'Correlation between {model} spectra and {target_name.replace("_", " ")}\n{band}-band spectrum from {instru} at $R_c$ = {Rc}', fontsize=16, pad=20)            
            ax.set_xlabel(r"$T_\mathrm{eff}$ [K]", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)            
            pcm  = ax.pcolormesh(T_arr, y_for_plot, corr_2D, cmap='coolwarm', shading='auto', vmin=np.nanmin(corr_2D), vmax=np.nanmax(corr_2D))            
            cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046, format='%.3f')
            cbar.set_label("Correlation strength", fontsize=14, labelpad=18, rotation=270)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.minorticks_on()            
            ax.plot(T_corr_found, y_point, marker='X', color='white', markersize=9, markeredgecolor='black', markeredgewidth=1.5, label=label_text)            
            ax.contour(T_arr, lg_arr, corr_2D, levels=12, linewidths=0.8, colors='white', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.minorticks_on()            
            ax.legend(fontsize=11, loc='lower left', frameon=True, facecolor='whitesmoke', edgecolor='gray')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)            
            plt.tight_layout()
            plt.show()
            
            # logL map
            if calc_logL:
                idx_max_logL     = np.unravel_index(np.nanargmax(logL_4D), logL_4D.shape)
                logL_2D          = logL_4D[:, :, idx_max_logL[2], idx_max_logL[3]].transpose()
                T_logL_found     = T_arr[idx_max_logL[0]]
                lg_logL_found    = lg_arr[idx_max_logL[1]]
                vsini_logL_found = vsini_arr[idx_max_logL[2]]
                rv_logL_found    = rv_arr[idx_max_logL[3]]
                print(f" {'Maximum log-likelihood:':<25} {'':>6}     =>  T = {T_logL_found:.0f} K,  log(g) = {lg_logL_found:.2f},  RV = {rv_logL_found:.1f} km/s")
                label_text = rf"max at $T_\mathrm{{eff}}$ = {T_logL_found:.0f} K, $\log\,g$ = {lg_logL_found:.2f}, $V\sin i$ = {vsini_logL_found:.1f} km/s, $RV$ = {rv_logL_found:.1f} km/s"
                y_for_plot = lg_arr
                y_point    = lg_logL_found
                ylabel     = r"$\log\,g$ [dex (cm/s²)]"
                fig, ax    = plt.subplots(dpi=300, figsize=(10, 6))            
                ax.set_title(f'logL ({method_logL}) between {model} spectra and {target_name.replace("_", " ")}\n{band}-band spectrum from {instru} at $R_c$ = {Rc}', fontsize=16, pad=20)            
                ax.set_xlabel(r"$T_\mathrm{eff}$ [K]", fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)            
                pcm  = ax.pcolormesh(T_arr, y_for_plot, logL_2D, cmap='coolwarm', shading='auto', vmin=np.nanmin(logL_2D), vmax=np.nanmax(logL_2D))            
                cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
                cbar.set_label("logL", fontsize=14, labelpad=18, rotation=270)
                cbar.ax.tick_params(labelsize=10)
                cbar.ax.minorticks_on()            
                ax.plot(T_logL_found, y_point, marker='X', color='white', markersize=9, markeredgecolor='black', markeredgewidth=1.5, label=label_text)            
                ax.contour(T_arr, lg_arr, logL_2D, levels=12, linewidths=0.8, colors='white', alpha=0.7)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.minorticks_on()            
                ax.legend(fontsize=11, loc='lower left', frameon=True, facecolor='whitesmoke', edgecolor='gray')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)            
                plt.tight_layout()
                plt.show()

                # Theoritical/"perfect" logL map
                if calc_d_sim:
                    idx_max_logL_sim     = np.unravel_index(np.nanargmax(logL_sim_4D), logL_sim_4D.shape)
                    logL_sim_2D          = logL_sim_4D[:, :, idx_max_logL_sim[2], idx_max_logL_sim[3]].transpose()
                    #logL_sim_2D          = (logL_sim_2D - np.nanmin(logL_sim_2D)) / np.nanmax(logL_sim_2D - np.nanmin(logL_sim_2D))
                    T_logL_sim_found     = T_arr[idx_max_logL_sim[0]]
                    lg_logL_sim_found    = lg_arr[idx_max_logL_sim[1]]
                    vsini_logL_sim_found = vsini_arr[idx_max_logL_sim[2]]
                    rv_logL_sim_found    = rv_arr[idx_max_logL_sim[3]]
                    print(f" {'Maximum log-likelihood (sim):':<25} {'':>6} =>  T = {T_logL_sim_found:.0f} K,  log(g) = {lg_logL_sim_found:.2f},  RV = {rv_logL_sim_found:.1f} km/s")
                    label_text = rf"max at $T_\mathrm{{eff}}$ = {T_logL_sim_found:.0f} K, $\log\,g$ = {lg_logL_sim_found:.2f}, $V\sin i$ = {vsini_logL_sim_found:.1f} km/s, $RV$ = {rv_logL_sim_found:.1f} km/s"
                    y_for_plot = lg_arr
                    y_point    = lg_logL_sim_found
                    ylabel     = r"$\log\,g$ [dex (cm/s²)]"
                    fig, ax    = plt.subplots(dpi=300, figsize=(10, 6))            
                    ax.set_title(f'SIMULATION: logL ({method_logL}) between {model} spectra and {target_name.replace("_", " ")}\n{band}-band spectrum from {instru} at $R_c$ = {Rc}', fontsize=16, pad=20)            
                    ax.set_xlabel(r"$T_\mathrm{eff}$ [K]", fontsize=14)
                    ax.set_ylabel(ylabel, fontsize=14)            
                    pcm  = ax.pcolormesh(T_arr, y_for_plot, logL_sim_2D, cmap='coolwarm', shading='auto', vmin=np.nanmin(logL_sim_2D), vmax=np.nanmax(logL_sim_2D))            
                    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
                    cbar.set_label("logL", fontsize=14, labelpad=18, rotation=270)
                    cbar.ax.tick_params(labelsize=10)
                    cbar.ax.minorticks_on()            
                    ax.plot(T_logL_sim_found, y_point, marker='X', color='white', markersize=9, markeredgecolor='black', markeredgewidth=1.5, label=label_text)            
                    ax.contour(T_arr, lg_arr, logL_sim_2D, levels=12, linewidths=0.8, colors='white', alpha=0.7)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.minorticks_on()            
                    ax.legend(fontsize=11, loc='lower left', frameon=True, facecolor='whitesmoke', edgecolor='gray')
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)            
                    plt.tight_layout()
                    plt.show()

    # ---------- Corner plot / 1σ uncertainties ----------
    param_names = BF_values = optimal_values = uncertainties_1sigma = bounds_1sigma = None

    if calc_logL and sigma_l is not None and not SNR_estimate:
        
        params      = [T_arr, lg_arr, vsini_arr, rv_arr]
        param_names = ["T", "logg", "Vsini", "RV"]
        param_units = ["K", "dex", "km/s", "km/s"]
        
        param_names, BF_values, optimal_values, uncertainties_1sigma, bounds_1sigma = custom_corner_plot(logL_4D, params, param_names, param_units, target_name, band, instru, model, R, Rc, sim=False, exposure_time=exposure_time, show=show)
    
        if calc_d_sim:
            custom_corner_plot(logL_sim_4D, params, param_names, param_units, target_name, band, instru, model, R, Rc, sim=True, exposure_time=exposure_time, show=show)
        
        if fastcurves:
            return uncertainties_1sigma
    
    return T_arr, lg_arr, vsini_arr, rv_arr, corr_4D, SNR_4D, logL_4D, logL_sim_4D, param_names, BF_values, optimal_values, uncertainties_1sigma, bounds_1sigma



def estimate_uncertainties_1sigma(P, func_marg, params):
    """
    Estimate marginalized best-fit values and 1-sigma credible intervals
    for each parameter from a multidimensional posterior probability array.

    Parameters
    ----------
    P : ndarray
        Multidimensional posterior probability array. Its dimensions must match
        the parameter grids given in params.
    func_marg : callable
        Function used to marginalize the posterior over unwanted axes.
        Typically np.nansum.
    params : list of ndarray
        List of parameter grids, one per dimension of P.

    Returns
    -------
    optimal_values : list
        Median value of each marginalized 1D posterior.
    uncertainties : list
        Symmetric 1-sigma uncertainty, defined as half of the 16th-84th percentile interval.
    bounds_1sigma : list of tuple
        Lower and upper bounds of the 68% credible interval.
    """

    # Output containers
    optimal_values = []
    uncertainties  = []
    bounds_1sigma  = []

    # Number of dimensions / parameters in the posterior
    ndim = len(params)

    # Loop over each parameter axis
    for i, param_values in enumerate(params):

        # Convert the parameter grid to a clean float array
        param_values = np.asarray(param_values, dtype=float)

        # Marginalize the posterior over all axes except the current parameter axis
        axes           = tuple(j for j in range(ndim) if j != i)
        marginalized_p = func_marg(P, axis=axes)

        # Clean invalid values in the marginalized posterior
        marginalized_p = np.asarray(marginalized_p, dtype=float)
        marginalized_p = np.nan_to_num(marginalized_p, nan=0.0, posinf=0.0, neginf=0.0)

        # Safety check: the marginalized posterior must match the parameter grid size
        if marginalized_p.size != param_values.size:
            raise ValueError(f"Shape mismatch for parameter axis {i}: marginalized_p has size {marginalized_p.size}, but param_values has size {param_values.size}.")

        # Sort the parameter grid and reorder the posterior consistently
        order          = np.argsort(param_values)
        param_values   = param_values[order]
        marginalized_p = marginalized_p[order]

        # Remove duplicated parameter values, which would break interpolation
        unique_values, unique_indices = np.unique(param_values, return_index=True)
        param_values   = unique_values
        marginalized_p = marginalized_p[unique_indices]

        # If the parameter grid or posterior is not usable, return NaNs for this parameter
        if param_values.size < 2 or np.nansum(marginalized_p) <= 0:
            optimal_values.append(np.nan)
            uncertainties.append(np.nan)
            bounds_1sigma.append((np.nan, np.nan))
            continue

        # Estimate the typical parameter sampling step
        dparam = np.nanmedian(np.diff(param_values))

        # Reject invalid or non-positive sampling steps
        if not np.isfinite(dparam) or dparam <= 0:
            optimal_values.append(np.nan)
            uncertainties.append(np.nan)
            bounds_1sigma.append((np.nan, np.nan))
            continue

        # Interpolate the marginalized posterior on a finer linear grid
        f_interp    = interp1d(param_values, marginalized_p, kind="linear", bounds_error=False, fill_value=0.0)
        values_fine = np.linspace(param_values[0], param_values[-1], max(1000, 10 * param_values.size))
        p_fine      = np.nan_to_num(f_interp(values_fine), nan=0.0, posinf=0.0, neginf=0.0)

        # If the interpolated posterior is empty, return NaNs for this parameter
        if np.nansum(p_fine) <= 0:
            optimal_values.append(np.nan)
            uncertainties.append(np.nan)
            bounds_1sigma.append((np.nan, np.nan))
            continue
        
        # Compute the normalization of the interpolated 1D posterior
        dx_fine   = np.gradient(values_fine)
        norm_fine = np.nansum(p_fine * dx_fine)

        # Reject invalid normalization
        if not np.isfinite(norm_fine) or norm_fine <= 0:
            optimal_values.append(np.nan)
            uncertainties.append(np.nan)
            bounds_1sigma.append((np.nan, np.nan))
            continue

        # Build the cumulative distribution function
        cdf_fine = np.cumsum(p_fine * dx_fine) / norm_fine

        # Enforce monotonicity to avoid numerical interpolation issues
        cdf_fine = np.maximum.accumulate(cdf_fine)

        # Normalize exactly to one at the end
        cdf_fine = cdf_fine / cdf_fine[-1]

        # Extract the 16th, 50th, and 84th percentiles
        lower_bound   = np.interp(0.16, cdf_fine, values_fine)
        optimal_value = np.interp(0.50, cdf_fine, values_fine)
        upper_bound   = np.interp(0.84, cdf_fine, values_fine)

        # Define a symmetric 1-sigma uncertainty from the percentile interval
        uncertainty = max(0.5 * (upper_bound - lower_bound), 0.001)

        # Store results for the current parameter
        optimal_values.append(optimal_value)
        uncertainties.append(uncertainty)
        bounds_1sigma.append((lower_bound, upper_bound))

    return optimal_values, uncertainties, bounds_1sigma



def gaussian_equivalent_credible_levels(sigmas=(1, 2, 3, 4, 5), ndim=2):
    sigmas = np.asarray(sigmas, dtype=float)
    if ndim == 1:
        return [erf(s / np.sqrt(2.0)) for s in sigmas]
    if ndim == 2:
        return list(1.0 - np.exp(-0.5 * sigmas**2))
    raise ValueError("Only ndim=1 and ndim=2 are implemented.")



def get_hpd_levels_2d(P2D, x, y, sigmas=(1, 2, 3, 4, 5)):
    """
    Compute 2D Highest Posterior Density (HPD) contour levels corresponding to Gaussian-equivalent
    enclosed probability masses.

    The returned contour thresholds can be labelled as 1σ, 2σ, etc.
    """

    P2D = np.asarray(P2D, dtype=float)
    P2D = np.nan_to_num(P2D, nan=0.0, posinf=0.0, neginf=0.0)

    if np.nansum(P2D) <= 0:
        return [], {}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = np.gradient(x)
    dy = np.gradient(y)

    # Probability mass per grid cell
    dA   = dy[:, None] * dx[None, :]
    mass = P2D * dA

    valid = np.isfinite(P2D) & np.isfinite(mass) & (P2D > 0) & (mass > 0)

    if not np.any(valid):
        return [], {}

    density_valid = P2D[valid]
    mass_valid    = mass[valid]

    order          = np.argsort(density_valid)[::-1]
    density_sorted = density_valid[order]
    mass_sorted    = mass_valid[order]

    cdf = np.cumsum(mass_sorted)
    cdf = cdf / cdf[-1]

    cred_levels = gaussian_equivalent_credible_levels(sigmas, ndim=2)

    levels = []
    labels = []

    for sigma, cred in zip(sigmas, cred_levels):
        idx   = np.searchsorted(cdf, cred, side="left")
        idx   = min(idx, len(density_sorted) - 1)
        level = density_sorted[idx]

        if np.isfinite(level):
            levels.append(level)
            labels.append(rf"{sigma}$\sigma$")

    # Matplotlib needs increasing contour levels
    pairs          = sorted(zip(levels, labels), key=lambda p: p[0])
    contour_levels = []
    fmt            = {}
    for level, label in pairs:
        if len(contour_levels) > 0 and level <= contour_levels[-1]:
            level = np.nextafter(contour_levels[-1], np.inf)
        contour_levels.append(level)
        fmt[level] = label

    return contour_levels, fmt



def custom_corner_plot(logL, params, param_names, param_units, target_name, band, instru, model, R, Rc, func_marg=np.nansum, sim=False, exposure_time=None, show=True):

    # Avoid modifying external lists
    params      = [np.asarray(p, dtype=float) for p in params]
    param_names = list(param_names)
    param_units = list(param_units)
    
    # Relative posterior, assuming flat priors
    P = np.exp(logL - np.nanmax(logL))
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop size-1 dimensions
    dims_to_drop = [idim for idim, p in enumerate(params) if len(p) == 1]
    for idim in sorted(dims_to_drop, reverse=True):
        P = np.nansum(P, axis=idim)
        params.pop(idim)
        param_names.pop(idim)
        param_units.pop(idim)

    ndim = len(params)

    if ndim == 0:
        raise ValueError("No parameter dimension remains after dropping size-1 axes.")
    
    # Best-fit and marginalized 1D intervals
    optimal_values, uncertainties_1sigma, bounds_1sigma = estimate_uncertainties_1sigma(P, func_marg, params)
    
    if np.any(np.isfinite(P)) and np.nanmax(P) > 0:
        idx_ml    = np.unravel_index(np.nanargmax(P), P.shape)
        BF_values = [params[idim][idx_ml[idim]] for idim in range(ndim)]
    else:
        BF_values = [np.nan] * ndim
    
    xmin = np.array([np.nanmin(param) for param in params])
    xmax = np.array([np.nanmax(param) for param in params])
    
    # Plot
    if show:

        print("\n=== Marginalized parameter estimates ===")
        for idim in range(ndim):
            lower, upper = bounds_1sigma[idim]
            err_minus    = optimal_values[idim] - lower
            err_plus     = upper - optimal_values[idim]
            print(f"{param_names[idim]:<7} = {optimal_values[idim]:>8.3f} -{err_minus:.3f} +{err_plus:.3f} {param_units[idim]}")
        print("\n=== Global maximum-likelihood grid point (best-fit) ===")
        for idim in range(ndim):
            print(f"{param_names[idim]:<7} = {BF_values[idim]:>8.3f} ± {uncertainties_1sigma[idim]:.3f} {param_units[idim]}")
        
        sigmas      = (1, 2, 3, 4, 5)
        linewidths  = [0.8, 1.0, 1.2, 1.5, 1.8]  
        fill_colors = plt.get_cmap("plasma_r")(np.linspace(0.05, 0.95, len(sigmas)))
        fig, axes   = plt.subplots(ndim, ndim, figsize=(12, 12), dpi=300)
        axes        = np.atleast_2d(axes)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        for i in range(ndim):
            for j in range(ndim):

                ax = axes[i, j]
                ax.tick_params(axis="both", which="major", labelsize=10)
                ax.tick_params(axis="both", which="minor", labelsize=8)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
                ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

                if j > i:
                    ax.axis("off")
                    continue

                elif i == j:
                    marginalized_p = func_marg(P, axis=tuple(idim for idim in range(ndim) if idim != i))
                    marginalized_p = np.nan_to_num(marginalized_p, nan=0.0, posinf=0.0, neginf=0.0)
                    norm_p         = np.nansum(marginalized_p)
                    if norm_p <= 0 or not np.isfinite(norm_p):
                        ax.axis("off")
                        continue
                    marginalized_p = marginalized_p / norm_p
                    lower, upper   = bounds_1sigma[i]
                    ax.step(params[i], marginalized_p, color="k", where="mid")
                    ax.axvline(optimal_values[i], color="k", linestyle="--")
                    if np.isfinite(lower):
                        ax.axvline(lower, color="k", linestyle=":")
                    if np.isfinite(upper):
                        ax.axvline(upper, color="k", linestyle=":")
                    ax.set_yticks([])
                    ax.set_xlabel(f"{param_names[i]} [{param_units[i]}]", fontsize=14)
                    ax.set_title(f"{param_names[i]} = {optimal_values[i]:.3f} ± {uncertainties_1sigma[i]:.3f} {param_units[i]}", fontsize=13)
                    ax.set_xlim(xmin[i], xmax[i])

                elif j < i:
                    # Marginalizing
                    marginalized_p = func_marg(P, axis=tuple(idim for idim in range(ndim) if idim != i and idim != j))
                    marginalized_p = np.nan_to_num(marginalized_p, nan=0.0, posinf=0.0, neginf=0.0)
                
                    # Normalizing
                    norm_p = np.nansum(marginalized_p)
                    if norm_p <= 0 or not np.isfinite(norm_p):
                        ax.axis("off")
                        continue
                
                    marginalized_p = marginalized_p / norm_p
                
                    # Remaining axes are ordered as (j, i), so transpose for contour:
                    # x = params[j], y = params[i]
                    marginalized_p = marginalized_p.transpose()
                    
                    # Keeping only positive values
                    positive = marginalized_p[marginalized_p > 0]
                    if positive.size == 0:
                        ax.axis("off")
                        continue
                    marginalized_p[marginalized_p <= 0] = np.nanmin(positive)
                    
                    # HPD probability filled contours
                    hpd_levels, fmt = get_hpd_levels_2d(marginalized_p, params[j], params[i], sigmas=sigmas)
                    if len(hpd_levels) == 0:
                        ax.axis("off")
                        continue
                    pmax        = max(np.nextafter(hpd_levels[-1], np.inf), np.nanmax(marginalized_p))
                    fill_levels = hpd_levels + [pmax]
                    ax.contourf(params[j], params[i], marginalized_p, levels=fill_levels, colors=fill_colors, alpha=0.8)
                    
                    contour = ax.contour(params[j], params[i], marginalized_p, levels=hpd_levels, colors="black", linewidths=linewidths)
                    ax.clabel(contour, inline=True, fontsize=12, fmt=fmt)
                    
                    ax.axvline(optimal_values[j], color="k", linestyle="--")
                    ax.axhline(optimal_values[i], color="k", linestyle="--")
                    ax.plot(optimal_values[j], optimal_values[i], "o", color="black")
            
                    if j == 0:
                        ax.set_ylabel(f"{param_names[i]} [{param_units[i]}]", fontsize=14)
                    if i == ndim - 1:
                        ax.set_xlabel(f"{param_names[j]} [{param_units[j]}]", fontsize=14)
                
                    ax.set_xlim(xmin[j], xmax[j])
                    ax.set_ylim(xmin[i], xmax[i])

                if i < ndim - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

        suffix = "\nSimulation" if sim else ""
        if exposure_time is None:
            exp_text = f"(R = {np.nanmedian(R):.0f}, $R_c$ = {Rc})"
        else:
            exp_text = f"($t_{{exp}}$ = {round(exposure_time)}mn, R = {np.nanmedian(R):.0f}, $R_c$ = {Rc})"
        fig.suptitle(f"Parameter estimation of {target_name.replace('_', ' ')} \n\n on {band}-band of {instru} with {model} model \n\n {exp_text}{suffix}", fontsize=18, x=0.63, y=0.88)
        plt.show()

    return param_names, BF_values, optimal_values, uncertainties_1sigma, bounds_1sigma


