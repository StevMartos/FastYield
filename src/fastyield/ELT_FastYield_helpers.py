# import FastYield modules
from fastyield.config import rad2arcsec
from fastyield.FastYield_helpers import interp_flat_last_axis, effective_number_of_DIT, sigma_RON_2_from_DIT
from fastyield.signal_noise import compress_h_for_sigma_base_2, compute_sigma_base_2_speck_numba, compute_sigma_base_2_al_spat_numba, compute_sigma_base_2_al_spec_fast
from fastyield.spectrum import get_thermal_reflected_spectrum, filtered_flux


# import numpy modules
import numpy as np
from numpy.lib.format import open_memmap

# import other modules
from tqdm import tqdm



# ---------------------------------------------------
# Main functions for ELT instruments (HARMONI, ANDES)
# ---------------------------------------------------

def metrics_slice(sl, post_processing, wave_i, trans_tell_i, star_i, planet_i, background_i, star_HF_i, star_LF_i, planet_HF_i, planet_LF_i, valid_i, separation_planet_rad, separation_planet_rad_arr, pxscale_rad, D, background_scale, fast_comp=True):
    if sl.stop <= sl.start or sl.stop - sl.start < 2:
        return (np.nan,) * 5
    wave_ijk       = wave_i[sl]                          # [µm]
    trans_tell_ijk = trans_tell_i[sl]                    # [no unit]
    star_ijk       = star_i[sl]                          # [ph/mn/bin]
    planet_ijk     = planet_i[sl]                        # [ph/mn/bin]
    background_ijk = background_i[sl] * background_scale # [ph/mas2/mn/bin] => [ph/px/mn/bin]
    
    # Valid range (if force_inside_bands)
    if valid_i is not None:
        valid_ijk = valid_i[sl]
        if valid_ijk.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        wave_ijk       = wave_ijk[valid_ijk]
        trans_tell_ijk = trans_tell_ijk[valid_ijk]
        star_ijk       = star_ijk[valid_ijk]
        planet_ijk     = planet_ijk[valid_ijk]
        background_ijk = background_ijk[valid_ijk]

    # MM post-processing method
    if post_processing == "MM":
        star_HF_ijk   = star_HF_i[sl]   # [ph/mn/bin]
        star_LF_ijk   = star_LF_i[sl]   # [ph/mn/bin]
        planet_HF_ijk = planet_HF_i[sl] # [ph/mn/bin]
        planet_LF_ijk = planet_LF_i[sl] # [ph/mn/bin]
        # Valid range (if force_inside_bands)
        if valid_i is not None:
            star_HF_ijk   = star_HF_ijk[valid_ijk]
            star_LF_ijk   = star_LF_ijk[valid_ijk]
            planet_HF_ijk = planet_HF_ijk[valid_ijk]
            planet_LF_ijk = planet_LF_ijk[valid_ijk]
        
        # Template computation
        template = trans_tell_ijk*planet_HF_ijk
        norm2    = np.nansum(template**2)        
        if (not np.isfinite(norm2)) or (norm2 <= 0):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        template /= np.sqrt(norm2)
        
        # CCF signal computation in [ph/mn] (not normalized with PSF data and transmission)
        alpha                    = np.nansum(trans_tell_ijk * planet_HF_ijk * template)
        ratio_planet_star_LF_ijk = np.divide(planet_LF_ijk, star_LF_ijk, out=np.zeros_like(planet_LF_ijk), where=np.isfinite(star_LF_ijk) & (star_LF_ijk != 0))
        beta                     = np.nansum(trans_tell_ijk * star_HF_ijk * ratio_planet_star_LF_ijk * template)
        signal                   = alpha - beta
    
    # DI post-processing method
    elif post_processing == "DI":
        # Template computation
        template = trans_tell_ijk*planet_ijk
        norm2    = np.nansum(template**2)
        if (not np.isfinite(norm2)) or (norm2 <= 0):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        template /= np.sqrt(norm2)
        
        # CCF signal computation in [ph/mn] (not normalized with PSF data and transmission)
        delta  = np.nansum(trans_tell_ijk*planet_ijk * template)
        signal = delta
    
    if not np.any(np.isfinite(star_ijk)):
        return (np.nan,) * 5

    # Sigma stellar halo photon noise computation in [ph/mn] (not normalized with PSF data and transmission)
    sigma_halo_2 = np.nansum(trans_tell_ijk*star_ijk * template**2)
    
    # Max star flux in [ph/mn] (not normalized with PSF data and transmission)
    max_star_flux = np.nanmax(trans_tell_ijk * star_ijk)       
    
    # Sigma background photon noise computation in [ph/px/mn] (without transmission)
    sigma_bkg_2 = np.nansum(background_ijk * template**2)
    
    # Sigma stellar halo systematic noise (assuming sigma_m = 1) computation in [ph/mn]**2 (not normalized with PSF data and transmission)
    # This implements:
    #   sigma_m,base^2 = h^T C_m h
    # with, for MM:
    #   C_m = 0.5 * (C_al_spec + C_al_spat).
    # Therefore:
    #   sigma_m,base^2 = 0.5 * (h^T C_al_spec h + h^T C_al_spat h).
    h_PP = (trans_tell_ijk*star_ijk * template)[:, None] # (wave, sep) Ponderated halo, H*t
    if fast_comp:
        wave_ijk, h_PP_c, n_merge = compress_h_for_sigma_base_2(wave=wave_ijk, h=h_PP, post_processing=post_processing, D=D, pxscale=pxscale_rad, separation_ref=separation_planet_rad, sep_unit="rad")        
    if post_processing == "DI":
        sigma_syst_base_2 = compute_sigma_base_2_speck_numba(     h=h_PP_c if fast_comp else h_PP, wave=wave_ijk, separation_rad=separation_planet_rad_arr, D=D)[0]
    elif post_processing == "MM":
        sigma_base_2_al_spat = compute_sigma_base_2_al_spat_numba(h=h_PP_c if fast_comp else h_PP, wave=wave_ijk, separation_rad=separation_planet_rad_arr, pxscale_rad=pxscale_rad)[0]
        sigma_base_2_al_spec = compute_sigma_base_2_al_spec_fast(   h=h_PP)[0]
        sigma_syst_base_2    = 0.5 * (sigma_base_2_al_spec + sigma_base_2_al_spat)

    return signal, sigma_halo_2, max_star_flux, sigma_bkg_2, sigma_syst_base_2




# --- Globals used by multiprocessing workers ---
_CTX       = None
_OUT_NAMES = [
    "signal_4D_R", "signal_4D_l0", "signal_4D_Dl", "signal_4D_TI", "signal_4D_WFE", "signal_4D_IWA",
    "sigma_halo_2_4D_R", "sigma_halo_2_4D_l0", "sigma_halo_2_4D_Dl", "sigma_halo_2_4D_TI", "sigma_halo_2_4D_WFE", "sigma_halo_2_4D_IWA",
    "sigma_syst_base_2_4D_R", "sigma_syst_base_2_4D_l0", "sigma_syst_base_2_4D_Dl", "sigma_syst_base_2_4D_TI", "sigma_syst_base_2_4D_WFE", "sigma_syst_base_2_4D_IWA",
    "sigma_bkg_2_4D_R", "sigma_bkg_2_4D_l0", "sigma_bkg_2_4D_Dl", "sigma_bkg_2_4D_TI", "sigma_bkg_2_4D_WFE", "sigma_bkg_2_4D_IWA",
    "DIT_4D_R", "DIT_4D_l0", "DIT_4D_Dl", "DIT_4D_TI", "DIT_4D_WFE", "DIT_4D_IWA",
]
def init_worker(ctx):
    global _CTX
    _CTX = dict(ctx)

    # Open PSF memmaps in read-only mode inside each worker
    _CTX["PSF_profile_6D"]   = open_memmap(_CTX["PSF_profile_6D_path"], mode="r")
    _CTX["fraction_core_6D"] = open_memmap(_CTX["fraction_core_6D_path"], mode="r")

    # Open output memmaps in read/write mode inside each worker
    for name in _OUT_NAMES:
        _CTX[f"{name}_mm"] = open_memmap(_CTX[f"{name}_path"], mode="r+")



# --- IFU function ---
def process_ELT(idx):
    
    # --- Context ---
    planet_table       = _CTX["planet_table"]
    separation         = _CTX["separation"]
    pxscales           = _CTX["pxscales"]
    PSF_profile_6D     = _CTX["PSF_profile_6D"]
    fraction_core_6D   = _CTX["fraction_core_6D"]
    idx_R_ref          = _CTX["idx_R_ref"]
    idx_l0_ref         = _CTX["idx_l0_ref"]
    idx_Dl_ref         = _CTX["idx_Dl_ref"]
    idx_WFE_ref        = _CTX["idx_WFE_ref"]
    idx_IWA_ref        = _CTX["idx_IWA_ref"]
    R                  = _CTX["R"]
    trans_instru       = _CTX["trans_instru"]
    wave_1D_list       = _CTX["wave_1D_list"]
    dwave_1D_list      = _CTX["dwave_1D_list"]
    range_1D_list      = _CTX["range_1D_list"]
    valid_2D_list      = _CTX["valid_2D_list"]
    trans_tell_1D_list = _CTX["trans_tell_1D_list"]
    background_1D_list = _CTX["background_1D_list"]
    wave_model         = _CTX["wave_model"]
    wave_instru        = _CTX["wave_instru"]
    wave_K             = _CTX["wave_K"]
    counts_vega_K      = _CTX["counts_vega_K"]
    scale_spectrum     = _CTX["scale_spectrum"]
    thermal_model      = _CTX["thermal_model"]
    reflected_model    = _CTX["reflected_model"]
    post_processing    = _CTX["post_processing"]
    Rc                 = _CTX["Rc"]
    filter_type        = _CTX["filter_type"]
    D                  = _CTX["D"]
    saturation_e       = _CTX["saturation_e"]
    min_DIT            = _CTX["min_DIT"]
    max_DIT            = _CTX["max_DIT"]
    A_FWHM             = _CTX["A_FWHM"]
    trans_instru_ref   = _CTX["trans_instru_ref"]
    N_band             = _CTX["N_band"]
    N_config           = _CTX["N_config"]
    N_R                = _CTX["N_R"]
    N_l0               = _CTX["N_l0"]
    N_Dl               = _CTX["N_Dl"]
    dtype              = _CTX["dtype"]

    
    # --- Planet row ---
    planet                    = planet_table[idx]
    separation_planet         = float(planet["AngSep"].value)                  # [mas]
    separation_planet_rad     = separation_planet / (rad2arcsec * 1000)        # [rad]
    separation_planet_rad_arr = np.array([separation_planet_rad], dtype=dtype) # [rad]
    
    # --- PSF data linearly interpolated at the planet's separation ---
    # PSF_profile_6D   : (band, config, l0, WFE, IWA, separation)
    # fraction_core_6D : (band, config, l0, WFE, IWA, separation)
    PSF_profile_sep_5D   = interp_flat_last_axis(separation, PSF_profile_6D,   separation_planet, dtype=dtype) # (band, config, l0, WFE, IWA)
    fraction_core_sep_5D = interp_flat_last_axis(separation, fraction_core_6D, separation_planet, dtype=dtype) # (band, config, l0, WFE, IWA)
    
    
    # --- PSF data at ref values (bands, configs, *) ---
    PSF_profile_sep_3D_l0    = PSF_profile_sep_5D[:,   :, :, idx_WFE_ref, idx_IWA_ref]                          # [star flux fraction/px]     (band, config, l0)
    fraction_core_sep_3D_l0  = fraction_core_sep_5D[:, :, :, idx_WFE_ref, idx_IWA_ref]                          # [planet flux fraction/FWHM] (band, config, l0)
    PSF_profile_sep_3D_WFE   = PSF_profile_sep_5D[np.arange(N_band),   :, idx_l0_ref, :, idx_IWA_ref]           # [star flux fraction/px]     (band, config, WFE)
    fraction_core_sep_3D_WFE = fraction_core_sep_5D[np.arange(N_band), :, idx_l0_ref, :, idx_IWA_ref]           # [planet flux fraction/FWHM] (band, config, WFE)
    PSF_profile_sep_3D_IWA   = PSF_profile_sep_5D[np.arange(N_band),   :, idx_l0_ref, idx_WFE_ref, :]           # [star flux fraction/px]     (band, config, IWA)
    fraction_core_sep_3D_IWA = fraction_core_sep_5D[np.arange(N_band), :, idx_l0_ref, idx_WFE_ref, :]           # [planet flux fraction/FWHM] (band, config, IWA)
    PSF_profile_sep_2D_ref   = PSF_profile_sep_5D[np.arange(N_band),   :, idx_l0_ref, idx_WFE_ref, idx_IWA_ref] # [star flux fraction/px]     (band, config)
    fraction_core_sep_2D_ref = fraction_core_sep_5D[np.arange(N_band), :, idx_l0_ref, idx_WFE_ref, idx_IWA_ref] # [planet flux fraction/FWHM] (band, config)
    
    
    # --- Computing the planet and star models for the planet row in [J/s/m²/µm] ---
    planet_spectrum, _, _, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_planet_mag=True, in_star_mag=True, interpolated_spectrum=True)
    star_spectrum                        = star_spectrum.interpolate_wavelength(wave_instru,   renorm=False) # Interpolating on wave_instru [J/s/m2/µm]
    planet_spectrum                      = planet_spectrum.interpolate_wavelength(wave_instru, renorm=False) # Interpolating on wave_instru [J/s/m2/µm]
        
    # --- Converting in [ph/mn/µm] ---
    star_spectrum.flux   *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]
    planet_spectrum.flux *= scale_spectrum # [J/s/m²/µm] => [ph/mn/µm]
    
    
    # --- Pre-computing downgraded spectra and HF/LF splits
    star_1D_list      = [None] * N_R # [ph/mn/bin]
    planet_1D_list    = [None] * N_R # [ph/mn/bin]
    star_HF_1D_list   = [None] * N_R # [ph/mn/bin]
    star_LF_1D_list   = [None] * N_R # [ph/mn/bin]
    planet_HF_1D_list = [None] * N_R # [ph/mn/bin]
    planet_LF_1D_list = [None] * N_R # [ph/mn/bin]
    for i, res in enumerate(R):
        wave_i   = wave_1D_list[i]  # [µm]
        dwave_i  = dwave_1D_list[i] # [µm/bin]
        star_i   = star_spectrum.degrade_resolution(wave_i,   renorm=False, R_output=res).flux # [ph/mn/µm]
        planet_i = planet_spectrum.degrade_resolution(wave_i, renorm=False, R_output=res).flux # [ph/mn/µm]
    
        # Converting in [ph/mn/bin]
        star_i   *= dwave_i # [ph/mn/µm] => [ph/mn/bin]
        planet_i *= dwave_i # [ph/mn/µm] => [ph/mn/bin]
        
        # HF/LF split for MM
        if post_processing == "MM": # by definition res == R_nyquist (from get_resolution(wave_i))
            star_HF_i, star_LF_i     = filtered_flux(flux=star_i,   R=res, Rc=Rc, filter_type=filter_type) # [ph/mn/bin]
            planet_HF_i, planet_LF_i = filtered_flux(flux=planet_i, R=res, Rc=Rc, filter_type=filter_type) # [ph/mn/bin]
        
        # Saving in the lists
        star_1D_list[i]   = star_i   # [ph/mn/bin]
        planet_1D_list[i] = planet_i # [ph/mn/bin]
        if post_processing == "MM":
            star_HF_1D_list[i]   = star_HF_i   # [ph/mn/bin]
            star_LF_1D_list[i]   = star_LF_i   # [ph/mn/bin]
            planet_HF_1D_list[i] = planet_HF_i # [ph/mn/bin]
            planet_LF_1D_list[i] = planet_LF_i # [ph/mn/bin]
    
    
    # --- Computing data (signal, stellar halo photon noise variance, max star flux and base systematic noise) depending only on R, l0 and Dl ---
    signal_2D_R            = np.full((N_band, N_R), np.nan, dtype=dtype) # alpha - beta                         [ph/mn]
    sigma_halo_2_2D_R      = np.full((N_band, N_R), np.nan, dtype=dtype) # sum ( trans * S_star * template**2 ) [ph/mn]
    max_star_flux_2D_R     = np.full((N_band, N_R), np.nan, dtype=dtype) # max ( trans * S_star )               [ph/mn] in the brightest spectral bin
    sigma_bkg_2_2D_R       = np.full((N_band, N_R), np.nan, dtype=dtype) # sum ( background * template**2 )     [ph/px/mn]
    sigma_syst_base_2_2D_R = np.full((N_band, N_R), np.nan, dtype=dtype) # sigma_syst_base**2                   [ph/mn]**2 (assuming sigma_m = 1)
    
    signal_2D_l0            = np.full((N_band, N_l0), np.nan, dtype=dtype) # alpha - beta                         [ph/mn]
    sigma_halo_2_2D_l0      = np.full((N_band, N_l0), np.nan, dtype=dtype) # sum ( trans * S_star * template**2 ) [ph/mn]
    max_star_flux_2D_l0     = np.full((N_band, N_l0), np.nan, dtype=dtype) # max ( trans * S_star )               [ph/mn] in the brightest spectral bin
    sigma_bkg_2_2D_l0       = np.full((N_band, N_l0), np.nan, dtype=dtype) # sum ( background * template**2 )     [ph/px/mn]
    sigma_syst_base_2_2D_l0 = np.full((N_band, N_l0), np.nan, dtype=dtype) # sigma_syst_base**2                   [ph/mn]**2 (assuming sigma_m = 1)
    
    signal_2D_Dl            = np.full((N_band, N_Dl), np.nan, dtype=dtype) # alpha - beta                         [ph/mn]
    sigma_halo_2_2D_Dl      = np.full((N_band, N_Dl), np.nan, dtype=dtype) # sum ( trans * S_star * template**2 ) [ph/mn]
    max_star_flux_2D_Dl     = np.full((N_band, N_Dl), np.nan, dtype=dtype) # max ( trans * S_star )               [ph/mn] in the brightest spectral bin
    sigma_bkg_2_2D_Dl       = np.full((N_band, N_Dl), np.nan, dtype=dtype) # sum ( background * template**2 )     [ph/px/mn]
    sigma_syst_base_2_2D_Dl = np.full((N_band, N_Dl), np.nan, dtype=dtype) # sigma_syst_base**2                   [ph/mn]**2 (assuming sigma_m = 1)
    
    # --- Global band loop ---
    for ib in range(N_band):
        
        # --- Reference index for this band ---
        i_ref            = idx_R_ref[ib]
        j_ref            = idx_l0_ref[ib]
        k_ref            = idx_Dl_ref[ib]
        pxscale          = pxscales[ib]                  # [mas/px]
        pxscale_rad      = pxscale / (rad2arcsec * 1000) # [rad/px]
        background_scale = pxscales[ib]**2               # [mas/px]**2

        
        # --- Resolution loop for *_R quantities ---
        for i, res in enumerate(R):
            wave_i         = wave_1D_list[i]                       # [µm]
            trans_tell_i   = trans_tell_1D_list[i]                 # [no unit]
            background_i   = background_1D_list[i]                 # [ph/mas2/mn/bin]
            star_i         = star_1D_list[i]                       # [ph/mn/bin]
            planet_i       = planet_1D_list[i]                     # [ph/mn/bin]
            star_HF_i      = star_HF_1D_list[i]                    # [ph/mn/bin]
            star_LF_i      = star_LF_1D_list[i]                    # [ph/mn/bin]
            planet_HF_i    = planet_HF_1D_list[i]                  # [ph/mn/bin]
            planet_LF_i    = planet_LF_1D_list[i]                  # [ph/mn/bin]
            idx_lo, idx_hi = range_1D_list[i]
            sl             = slice(idx_lo[j_ref, k_ref], idx_hi[j_ref, k_ref]) # Cropped range for (l0, Dl)
            valid_i        = valid_2D_list[ib][i]                  #  Valid range (usefull if force_inside_bands)
            signal_2D_R[ib, i], sigma_halo_2_2D_R[ib, i], max_star_flux_2D_R[ib, i], sigma_bkg_2_2D_R[ib, i], sigma_syst_base_2_2D_R[ib, i] = metrics_slice(sl=sl, post_processing=post_processing, wave_i=wave_i, trans_tell_i=trans_tell_i, star_i=star_i, planet_i=planet_i, background_i=background_i, star_HF_i=star_HF_i, star_LF_i=star_LF_i, planet_HF_i=planet_HF_i, planet_LF_i=planet_LF_i, valid_i=valid_i, separation_planet_rad=separation_planet_rad, separation_planet_rad_arr=separation_planet_rad_arr, pxscale_rad=pxscale_rad, D=D, background_scale=background_scale, fast_comp=True)

    
        # --- Data at R_ref (for l0 and Dl loops) ---
        wave_i         = wave_1D_list[i_ref]                       # [µm]
        trans_tell_i   = trans_tell_1D_list[i_ref]                 # [no unit]
        background_i   = background_1D_list[i_ref]                 # [ph/mas2/mn/bin]
        star_i         = star_1D_list[i_ref]                       # [ph/mn/bin]
        planet_i       = planet_1D_list[i_ref]                     # [ph/mn/bin]
        star_HF_i      = star_HF_1D_list[i_ref]                    # [ph/mn/bin]
        star_LF_i      = star_LF_1D_list[i_ref]                    # [ph/mn/bin]
        planet_HF_i    = planet_HF_1D_list[i_ref]                  # [ph/mn/bin]
        planet_LF_i    = planet_LF_1D_list[i_ref]                  # [ph/mn/bin]
        idx_lo, idx_hi = range_1D_list[i_ref]
        valid_i        = valid_2D_list[ib][i_ref]                  #  Valid range (usefull if force_inside_bands)

    
        # --- l0 loop for *_l0 quantities ---
        for j in range(N_l0): # for each l0
            sl = slice(idx_lo[j, k_ref], idx_hi[j, k_ref]) # cropped range for (l0, Dl)
            signal_2D_l0[ib, j], sigma_halo_2_2D_l0[ib, j], max_star_flux_2D_l0[ib, j], sigma_bkg_2_2D_l0[ib, j], sigma_syst_base_2_2D_l0[ib, j] = metrics_slice(sl=sl, post_processing=post_processing, wave_i=wave_i, trans_tell_i=trans_tell_i, star_i=star_i, planet_i=planet_i, background_i=background_i, star_HF_i=star_HF_i, star_LF_i=star_LF_i, planet_HF_i=planet_HF_i, planet_LF_i=planet_LF_i, valid_i=valid_i, separation_planet_rad=separation_planet_rad, separation_planet_rad_arr=separation_planet_rad_arr, pxscale_rad=pxscale_rad, D=D, background_scale=background_scale, fast_comp=True)
    
        
        # --- Dl loop for *_Dl quantities ---
        for k in range(N_Dl): # for each Dl
            sl = slice(idx_lo[j_ref, k], idx_hi[j_ref, k]) # cropped range for (l0, Dl)
            signal_2D_Dl[ib, k], sigma_halo_2_2D_Dl[ib, k], max_star_flux_2D_Dl[ib, k], sigma_bkg_2_2D_Dl[ib, k], sigma_syst_base_2_2D_Dl[ib, k] = metrics_slice(sl=sl, post_processing=post_processing, wave_i=wave_i, trans_tell_i=trans_tell_i, star_i=star_i, planet_i=planet_i, background_i=background_i, star_HF_i=star_HF_i, star_LF_i=star_LF_i, planet_HF_i=planet_HF_i, planet_LF_i=planet_LF_i, valid_i=valid_i, separation_planet_rad=separation_planet_rad, separation_planet_rad_arr=separation_planet_rad_arr, pxscale_rad=pxscale_rad, D=D, background_scale=background_scale, fast_comp=True)
        
    
    # --- Computing DIT length in [mn/DIT] --- 
    
    # max_star_flux at the brightest pixel in [ph/px/mn] in the brightest pixel (through coronagraph, if any):
    # [ph/mn] in the brightest bin over the FoV * [star flux fraction/px] in the brightest pixel => [ph/px/mn] in the brightest pixel
    # (bands, *)                                * (bands, configs, l0)                           => (bands, configs, *)
    max_star_flux_1D_ref = max_star_flux_2D_R[np.arange(N_band), idx_R_ref]                         # (bands)
    max_star_flux_3D_R   = max_star_flux_2D_R[:, None, :]      * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, R)
    max_star_flux_3D_l0  = max_star_flux_2D_l0[:, None, :]     * PSF_profile_sep_3D_l0[:, :, :]     # (bands, configs, l0)
    max_star_flux_3D_Dl  = max_star_flux_2D_Dl[:, None, :]     * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, Dl)
    max_star_flux_3D_WFE = max_star_flux_1D_ref[:, None, None] * PSF_profile_sep_3D_WFE[:, :, :]    # (bands, configs, WFE)
    max_star_flux_3D_IWA = max_star_flux_1D_ref[:, None, None] * PSF_profile_sep_3D_IWA[:, :, :]    # (bands, configs, IWA)
    max_star_flux_2D_ref = max_star_flux_3D_R[np.arange(N_band), :, idx_R_ref]                      # (bands, configs)
    
    # max_star_flux at the brightest pixel in [e-/px/mn] in the brightest pixel with instrumental transmission (through coronagraph, if any):
    # [ph/px/mn] in the brightest pixel * [e-/ph]          => [e-/px/mn] in the brightest pixel
    # (bands, configs, *)               * (bands, configs) => (bands, configs, *)
    max_star_flux_3D_R   = max_star_flux_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    max_star_flux_3D_l0  = max_star_flux_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    max_star_flux_3D_Dl  = max_star_flux_3D_Dl              * trans_instru_ref[:, :, None] # (bands, configs, Dl)
    max_star_flux_3D_TI  = max_star_flux_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    max_star_flux_3D_WFE = max_star_flux_3D_WFE             * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    max_star_flux_3D_IWA = max_star_flux_3D_IWA             * trans_instru_ref[:, :, None] # (bands, configs, IWA)

    # Computing DIT length in [mn/DIT] (+ clipping between min and max DIT)
    DIT_3D_R   = saturation_e / max_star_flux_3D_R     # (bands, configs, R)   [mn/DIT]
    DIT_3D_l0  = saturation_e / max_star_flux_3D_l0    # (bands, configs, l0)  [mn/DIT]
    DIT_3D_Dl  = saturation_e / max_star_flux_3D_Dl    # (bands, configs, Dl)  [mn/DIT]
    DIT_3D_TI  = saturation_e / max_star_flux_3D_TI    # (bands, configs, TI)  [mn/DIT]
    DIT_3D_WFE = saturation_e / max_star_flux_3D_WFE   # (bands, configs, WFE) [mn/DIT]
    DIT_3D_IWA = saturation_e / max_star_flux_3D_IWA   # (bands, configs, IWA) [mn/DIT]
    DIT_3D_R   = np.clip(DIT_3D_R, min_DIT, max_DIT)   # (bands, configs, R)   [mn/DIT]
    DIT_3D_l0  = np.clip(DIT_3D_l0, min_DIT, max_DIT)  # (bands, configs, l0)  [mn/DIT]
    DIT_3D_Dl  = np.clip(DIT_3D_Dl, min_DIT, max_DIT)  # (bands, configs, Dl)  [mn/DIT]
    DIT_3D_TI  = np.clip(DIT_3D_TI, min_DIT, max_DIT)  # (bands, configs, TI)  [mn/DIT]
    DIT_3D_WFE = np.clip(DIT_3D_WFE, min_DIT, max_DIT) # (bands, configs, WFE) [mn/DIT]
    DIT_3D_IWA = np.clip(DIT_3D_IWA, min_DIT, max_DIT) # (bands, configs, IWA) [mn/DIT]

    
    # --- Computing signal in [e-/FWHM/DIT] (R, l0, Dl, WFE, IWA, TI) ---
    
    # Signal at the planet's separation inside the FWHM in [ph/FWHM/mn] (throught coronagraph, if any)
    # [ph/mn] over the FoV * [planet flux fraction/FWHM] => [ph/FWHM/mn]
    # (bands, *)           * (bands, configs, l0)                           => (bands, configs, *)
    signal_1D_ref = signal_2D_R[np.arange(N_band), idx_R_ref]                           # (bands)
    signal_3D_R   = signal_2D_R[:, None, :]      * fraction_core_sep_2D_ref[:, :, None] # (bands, configs, R)
    signal_3D_l0  = signal_2D_l0[:, None, :]     * fraction_core_sep_3D_l0[:, :, :]     # (bands, configs, l0)
    signal_3D_Dl  = signal_2D_Dl[:, None, :]     * fraction_core_sep_2D_ref[:, :, None] # (bands, configs, Dl)
    signal_3D_WFE = signal_1D_ref[:, None, None] * fraction_core_sep_3D_WFE[:, :, :]    # (bands, configs, WFE)
    signal_3D_IWA = signal_1D_ref[:, None, None] * fraction_core_sep_3D_IWA[:, :, :]    # (bands, configs, IWA)
    signal_2D_ref = signal_3D_R[np.arange(N_band), :, idx_R_ref]                        # (bands, configs)
    
    # Signal at the planet's separation inside the FWHM in [e-/FWHM/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/FWHM/mn]        * [e-/ph]          => [e-/FWHM/mn]
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    signal_3D_R   = signal_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    signal_3D_l0  = signal_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    signal_3D_Dl  = signal_3D_Dl              * trans_instru_ref[:, :, None] # (bands, configs, Dl)
    signal_3D_TI  = signal_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    signal_3D_WFE = signal_3D_WFE             * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    signal_3D_IWA = signal_3D_IWA             * trans_instru_ref[:, :, None] # (bands, configs, IWA)


    # Signal at the planet's separation inside the FWHM in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT (throught coronagraph, if any)
    # [e-/FWHM/mn]        * [mn/DIT]            => [e-/FWHM/DIT]
    # (bands, configs, *) * (bands, configs, *) => (bands, configs, *)
    signal_3D_R   = signal_3D_R   * DIT_3D_R   # (bands, configs, R)
    signal_3D_l0  = signal_3D_l0  * DIT_3D_l0  # (bands, configs, l0)
    signal_3D_Dl  = signal_3D_Dl  * DIT_3D_Dl  # (bands, configs, Dl)
    signal_3D_TI  = signal_3D_TI  * DIT_3D_TI  # (bands, configs, TI)
    signal_3D_WFE = signal_3D_WFE * DIT_3D_WFE # (bands, configs, WFE)
    signal_3D_IWA = signal_3D_IWA * DIT_3D_IWA # (bands, configs, IWA)

    
    # --- Computing stellar halo photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Dl, WFE, IWA, TI) ---
    
    # Stellar halo photon noise variance (Poisson) at the planet's separation in [ph/px/mn] (throught coronagraph, if any)
    # [ph/mn] over the FoV * [star flux fraction/px] => [ph/px/mn]
    # (bands, *)           * (bands, configs, l0)    => (bands, configs, *)
    sigma_halo_2_1D_ref = sigma_halo_2_2D_R[np.arange(N_band), idx_R_ref]
    sigma_halo_2_3D_R   = sigma_halo_2_2D_R[:, None, :]      * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, R)
    sigma_halo_2_3D_l0  = sigma_halo_2_2D_l0[:, None, :]     * PSF_profile_sep_3D_l0[:, :, :]     # (bands, configs, l0)
    sigma_halo_2_3D_Dl  = sigma_halo_2_2D_Dl[:, None, :]     * PSF_profile_sep_2D_ref[:, :, None] # (bands, configs, Dl)
    sigma_halo_2_3D_WFE = sigma_halo_2_1D_ref[:, None, None] * PSF_profile_sep_3D_WFE[:, :, :]    # (bands, configs, WFE)
    sigma_halo_2_3D_IWA = sigma_halo_2_1D_ref[:, None, None] * PSF_profile_sep_3D_IWA[:, :, :]    # (bands, configs, IWA)
    sigma_halo_2_2D_ref = sigma_halo_2_3D_R[np.arange(N_band), :, idx_R_ref]                      # (bands, configs)
    
    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/px/mn] with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]          * [e-/ph]          => [e-/px/mn]
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    sigma_halo_2_3D_R   = sigma_halo_2_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    sigma_halo_2_3D_l0  = sigma_halo_2_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    sigma_halo_2_3D_Dl  = sigma_halo_2_3D_Dl              * trans_instru_ref[:, :, None] # (bands, configs, Dl)
    sigma_halo_2_3D_TI  = sigma_halo_2_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    sigma_halo_2_3D_WFE = sigma_halo_2_3D_WFE             * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    sigma_halo_2_3D_IWA = sigma_halo_2_3D_IWA             * trans_instru_ref[:, :, None] # (bands, configs, IWA)

    # Stellar halo photon noise variance (Poisson) at the planet's separation in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance) (throught coronagraph, if any)
    # [e-/px/mn]          * [px/FWHM]*[mn/DIT]  => [e-/FWHM/DIT]
    # (bands, configs, *) * (bands, configs, *) => (bands, configs, *)
    sigma_halo_2_3D_R   = sigma_halo_2_3D_R   * A_FWHM*DIT_3D_R   # (bands, configs, R)
    sigma_halo_2_3D_l0  = sigma_halo_2_3D_l0  * A_FWHM*DIT_3D_l0  # (bands, configs, l0)
    sigma_halo_2_3D_Dl  = sigma_halo_2_3D_Dl  * A_FWHM*DIT_3D_Dl  # (bands, configs, Dl)
    sigma_halo_2_3D_TI  = sigma_halo_2_3D_TI  * A_FWHM*DIT_3D_TI  # (bands, configs, TI)
    sigma_halo_2_3D_WFE = sigma_halo_2_3D_WFE * A_FWHM*DIT_3D_WFE # (bands, configs, WFE)
    sigma_halo_2_3D_IWA = sigma_halo_2_3D_IWA * A_FWHM*DIT_3D_IWA # (bands, configs, IWA)

    
    # --- Computing stellar halo systematic noise variance (assuming sigma_m = 1) in [e-/FWHM/DIT]**2 (R, l0, Dl, WFE, IWA, TI) (all quantities are squared since sigma_syst**2 propto stellar_halo**2) ---
    
    # Stellar halo systematic noise variance at the planet's separation in [ph/px/mn]**2 (throught coronagraph, if any)
    # [ph/mn]**2 over the FoV * [star flux fraction/px]**2 => [ph/px/mn]**2
    # (bands, *)              * (bands, configs, l0)       => (bands, configs, *)
    sigma_syst_base_2_1D_ref = sigma_syst_base_2_2D_R[np.arange(N_band), idx_R_ref]
    sigma_syst_base_2_3D_R   = sigma_syst_base_2_2D_R[:, None, :]      * PSF_profile_sep_2D_ref[:, :, None]**2 # (bands, configs, R)
    sigma_syst_base_2_3D_l0  = sigma_syst_base_2_2D_l0[:, None, :]     * PSF_profile_sep_3D_l0[:, :, :]**2     # (bands, configs, l0)
    sigma_syst_base_2_3D_Dl  = sigma_syst_base_2_2D_Dl[:, None, :]     * PSF_profile_sep_2D_ref[:, :, None]**2 # (bands, configs, Dl)
    sigma_syst_base_2_3D_WFE = sigma_syst_base_2_1D_ref[:, None, None] * PSF_profile_sep_3D_WFE[:, :, :]**2    # (bands, configs, WFE)
    sigma_syst_base_2_3D_IWA = sigma_syst_base_2_1D_ref[:, None, None] * PSF_profile_sep_3D_IWA[:, :, :]**2    # (bands, configs, IWA)
    sigma_syst_base_2_2D_ref = sigma_syst_base_2_3D_R[np.arange(N_band), :, idx_R_ref]                         # (bands, configs)
    
    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission (throught coronagraph, if any)
    # [ph/px/mn]**2       * [e-/ph]**2       => [e-/px/mn]**2
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    sigma_syst_base_2_3D_R   = sigma_syst_base_2_3D_R               * trans_instru_ref[:, :, None]**2 # (bands, configs, R)
    sigma_syst_base_2_3D_l0  = sigma_syst_base_2_3D_l0              * trans_instru_ref[:, :, None]**2 # (bands, configs, l0)
    sigma_syst_base_2_3D_Dl  = sigma_syst_base_2_3D_Dl              * trans_instru_ref[:, :, None]**2 # (bands, configs, Dl)
    sigma_syst_base_2_3D_TI  = sigma_syst_base_2_2D_ref[:, :, None] * trans_instru[None, None, :]**2  # (bands, configs, TI)
    sigma_syst_base_2_3D_WFE = sigma_syst_base_2_3D_WFE             * trans_instru_ref[:, :, None]**2 # (bands, configs, WFE)
    sigma_syst_base_2_3D_IWA = sigma_syst_base_2_3D_IWA             * trans_instru_ref[:, :, None]**2 # (bands, configs, WFE)

    # Stellar halo systematic noise variance at the planet's separation in [e-/px/mn]**2 with instrumental transmission integrated over the DIT and the FWHM (throught coronagraph, if any) (we take A_FWHM**2 * DIT**2 since sigma_syst_base_2 propto H_integrated**2 propto A_FWHM**2*DIT**2, where H_integrated = the halo integrated over FWHM boxes and the DIT)
    # [e-/px/mn]**2       * [px/FWHM]**2*[mn/DIT]**2  => [e-/FWHM/DIT]**2
    # (bands, configs, *) * (bands, configs, *)       => (bands, configs, *)
    sigma_syst_base_2_3D_R   = sigma_syst_base_2_3D_R   * A_FWHM**2*DIT_3D_R**2   # (bands, configs, R)
    sigma_syst_base_2_3D_l0  = sigma_syst_base_2_3D_l0  * A_FWHM**2*DIT_3D_l0**2  # (bands, configs, l0)
    sigma_syst_base_2_3D_Dl  = sigma_syst_base_2_3D_Dl  * A_FWHM**2*DIT_3D_Dl**2  # (bands, configs, Dl)
    sigma_syst_base_2_3D_TI  = sigma_syst_base_2_3D_TI  * A_FWHM**2*DIT_3D_TI**2  # (bands, configs, TI)
    sigma_syst_base_2_3D_WFE = sigma_syst_base_2_3D_WFE * A_FWHM**2*DIT_3D_WFE**2 # (bands, configs, WFE)
    sigma_syst_base_2_3D_IWA = sigma_syst_base_2_3D_IWA * A_FWHM**2*DIT_3D_IWA**2 # (bands, configs, WFE)

    
    # --- Computing background photon noise variance (Poisson) in [e-/FWHM/DIT] (R, l0, Dl, WFE, IWA, TI) ---
    
    # (bands, *) => (bands, configs, *)
    sigma_bkg_2_1D_ref = sigma_bkg_2_2D_R[np.arange(N_band), idx_R_ref]
    sigma_bkg_2_3D_R   = sigma_bkg_2_2D_R[:, None, :]      * np.ones((1, N_config, 1)) # (bands, configs, R)
    sigma_bkg_2_3D_l0  = sigma_bkg_2_2D_l0[:, None, :]     * np.ones((1, N_config, 1)) # (bands, configs, l0)
    sigma_bkg_2_3D_Dl  = sigma_bkg_2_2D_Dl[:, None, :]     * np.ones((1, N_config, 1)) # (bands, configs, Dl)
    sigma_bkg_2_3D_WFE = sigma_bkg_2_1D_ref[:, None, None] * np.ones((1, N_config, 1)) # (bands, configs, WFE)
    sigma_bkg_2_3D_IWA = sigma_bkg_2_1D_ref[:, None, None] * np.ones((1, N_config, 1)) # (bands, configs, IWA)
    sigma_bkg_2_2D_ref = sigma_bkg_2_3D_R[np.arange(N_band), :, idx_R_ref]             # (bands, configs)

    # Background photon noise variance (Poisson) in [e-/px/mn] with instrumental transmission
    # [ph/px/mn]          * [e-/ph]          => [e-/px/mn]
    # (bands, configs, *) * (bands, configs) => (bands, configs, *)
    sigma_bkg_2_3D_R   = sigma_bkg_2_3D_R               * trans_instru_ref[:, :, None] # (bands, configs, R)
    sigma_bkg_2_3D_l0  = sigma_bkg_2_3D_l0              * trans_instru_ref[:, :, None] # (bands, configs, l0)
    sigma_bkg_2_3D_Dl  = sigma_bkg_2_3D_Dl              * trans_instru_ref[:, :, None] # (bands, configs, Dl)
    sigma_bkg_2_3D_TI  = sigma_bkg_2_2D_ref[:, :, None] * trans_instru[None, None, :]  # (bands, configs, TI)
    sigma_bkg_2_3D_WFE  = sigma_bkg_2_3D_WFE            * trans_instru_ref[:, :, None] # (bands, configs, WFE)
    sigma_bkg_2_3D_IWA  = sigma_bkg_2_3D_IWA            * trans_instru_ref[:, :, None] # (bands, configs, IWA)

    # Background photon noise variance (Poisson) in [e-/FWHM/DIT] with instrumental transmission integrated over the DIT and the FWHM (quadrature addition assuming statistical independance)
    # [e-/px/mn]          * [px/FWHM]*[mn/DIT]  => [e-/FWHM/DIT]
    # (bands, configs, *) * (bands, configs, *) => (bands, configs, *)
    sigma_bkg_2_3D_R   = sigma_bkg_2_3D_R   * A_FWHM*DIT_3D_R   # (bands, configs, R)
    sigma_bkg_2_3D_l0  = sigma_bkg_2_3D_l0  * A_FWHM*DIT_3D_l0  # (bands, configs, l0)
    sigma_bkg_2_3D_Dl  = sigma_bkg_2_3D_Dl  * A_FWHM*DIT_3D_Dl  # (bands, configs, Dl)
    sigma_bkg_2_3D_TI  = sigma_bkg_2_3D_TI  * A_FWHM*DIT_3D_TI  # (bands, configs, TI)
    sigma_bkg_2_3D_WFE = sigma_bkg_2_3D_WFE * A_FWHM*DIT_3D_WFE # (bands, configs, WFE)
    sigma_bkg_2_3D_IWA = sigma_bkg_2_3D_IWA * A_FWHM*DIT_3D_IWA # (bands, configs, IWA)

    
    _CTX["signal_4D_R_mm"][idx]   = signal_3D_R
    _CTX["signal_4D_l0_mm"][idx]  = signal_3D_l0
    _CTX["signal_4D_Dl_mm"][idx]  = signal_3D_Dl
    _CTX["signal_4D_TI_mm"][idx]  = signal_3D_TI
    _CTX["signal_4D_WFE_mm"][idx] = signal_3D_WFE
    _CTX["signal_4D_IWA_mm"][idx] = signal_3D_IWA

    _CTX["sigma_halo_2_4D_R_mm"][idx]   = sigma_halo_2_3D_R
    _CTX["sigma_halo_2_4D_l0_mm"][idx]  = sigma_halo_2_3D_l0
    _CTX["sigma_halo_2_4D_Dl_mm"][idx]  = sigma_halo_2_3D_Dl
    _CTX["sigma_halo_2_4D_TI_mm"][idx]  = sigma_halo_2_3D_TI
    _CTX["sigma_halo_2_4D_WFE_mm"][idx] = sigma_halo_2_3D_WFE
    _CTX["sigma_halo_2_4D_IWA_mm"][idx] = sigma_halo_2_3D_IWA

    _CTX["sigma_syst_base_2_4D_R_mm"][idx]   = sigma_syst_base_2_3D_R
    _CTX["sigma_syst_base_2_4D_l0_mm"][idx]  = sigma_syst_base_2_3D_l0
    _CTX["sigma_syst_base_2_4D_Dl_mm"][idx]  = sigma_syst_base_2_3D_Dl
    _CTX["sigma_syst_base_2_4D_TI_mm"][idx]  = sigma_syst_base_2_3D_TI
    _CTX["sigma_syst_base_2_4D_WFE_mm"][idx] = sigma_syst_base_2_3D_WFE
    _CTX["sigma_syst_base_2_4D_IWA_mm"][idx] = sigma_syst_base_2_3D_IWA

    _CTX["sigma_bkg_2_4D_R_mm"][idx]   = sigma_bkg_2_3D_R
    _CTX["sigma_bkg_2_4D_l0_mm"][idx]  = sigma_bkg_2_3D_l0
    _CTX["sigma_bkg_2_4D_Dl_mm"][idx]  = sigma_bkg_2_3D_Dl
    _CTX["sigma_bkg_2_4D_TI_mm"][idx]  = sigma_bkg_2_3D_TI
    _CTX["sigma_bkg_2_4D_WFE_mm"][idx] = sigma_bkg_2_3D_WFE
    _CTX["sigma_bkg_2_4D_IWA_mm"][idx] = sigma_bkg_2_3D_IWA

    _CTX["DIT_4D_R_mm"][idx]   = DIT_3D_R
    _CTX["DIT_4D_l0_mm"][idx]  = DIT_3D_l0
    _CTX["DIT_4D_Dl_mm"][idx]  = DIT_3D_Dl
    _CTX["DIT_4D_TI_mm"][idx]  = DIT_3D_TI
    _CTX["DIT_4D_WFE_mm"][idx] = DIT_3D_WFE
    _CTX["DIT_4D_IWA_mm"][idx] = DIT_3D_IWA
    
    return idx




# --------------------------
# Function for computing S/N
# --------------------------

def get_SNR(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, NbPixel, SNR_thr, separation_planets,
             idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Dl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
             signal_4D_R,   sigma_halo_2_4D_R,   sigma_syst_base_2_4D_R,   sigma_bkg_2_4D_R,   DIT_4D_R,
             signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_syst_base_2_4D_l0,  sigma_bkg_2_4D_l0,  DIT_4D_l0,
             signal_4D_Dl,  sigma_halo_2_4D_Dl,  sigma_syst_base_2_4D_Dl,  sigma_bkg_2_4D_Dl,  DIT_4D_Dl,
             signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_syst_base_2_4D_TI,  sigma_bkg_2_4D_TI,  DIT_4D_TI,
             signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_syst_base_2_4D_WFE, sigma_bkg_2_4D_WFE, DIT_4D_WFE,
             signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_syst_base_2_4D_IWA, sigma_bkg_2_4D_IWA, DIT_4D_IWA,
             dtype):
    
    # Data shape = (planets, bands, configs, *)
    
    # --- Number of DIT ---
    N_DIT_4D_R   = effective_number_of_DIT(DIT_4D_R,   exposure_time, dtype)
    N_DIT_4D_l0  = effective_number_of_DIT(DIT_4D_l0,  exposure_time, dtype)
    N_DIT_4D_Dl  = effective_number_of_DIT(DIT_4D_Dl,  exposure_time, dtype)
    N_DIT_4D_TI  = effective_number_of_DIT(DIT_4D_TI,  exposure_time, dtype)
    N_DIT_4D_WFE = effective_number_of_DIT(DIT_4D_WFE, exposure_time, dtype)
    N_DIT_4D_IWA = effective_number_of_DIT(DIT_4D_IWA, exposure_time, dtype)

    # --- Effective RON0 variance in Up-The-Ramp read-out mode (UTR) in [e-/FWHM/DIT]**2---
    sigma_RON_2_4D_R   = sigma_RON_2_from_DIT(DIT_4D_R,   min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_l0  = sigma_RON_2_from_DIT(DIT_4D_l0,  min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_Dl  = sigma_RON_2_from_DIT(DIT_4D_Dl,  min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_TI  = sigma_RON_2_from_DIT(DIT_4D_TI,  min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_WFE = sigma_RON_2_from_DIT(DIT_4D_WFE, min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)
    sigma_RON_2_4D_IWA = sigma_RON_2_from_DIT(DIT_4D_IWA, min_DIT, RON0, RON_lim, A_FWHM, dtype=dtype)

    # --- Dark current photon noise variance (Poisson) [e-/FWHM/DIT] ---
    sigma_DC_2_4D_R   = (DC0 * DIT_4D_R   * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_l0  = (DC0 * DIT_4D_l0  * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_Dl  = (DC0 * DIT_4D_Dl  * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_TI  = (DC0 * DIT_4D_TI  * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_WFE = (DC0 * DIT_4D_WFE * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    sigma_DC_2_4D_IWA = (DC0 * DIT_4D_IWA * A_FWHM).astype(dtype) # [e-/FWHM/DIT]
    
    # --- Pixel detector projection factor, adds quadratically (NbPixel = Number of detector px per cube px) ---
    if NbPixel is not None:
        sigma_RON_2_4D_R   *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_l0  *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_Dl  *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_TI  *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_WFE *= NbPixel # [e-/FWHM/DIT]
        sigma_RON_2_4D_IWA *= NbPixel # [e-/FWHM/DIT]
        
        sigma_DC_2_4D_R   *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_l0  *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_Dl  *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_TI  *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_WFE *= NbPixel # [e-/FWHM/DIT]
        sigma_DC_2_4D_IWA *= NbPixel # [e-/FWHM/DIT]
    
    # --- Reference quantities (for sigma_m axis) ---
    idx = idx_R_ref[None, :, None, None]                    # (1, band, 1, 1)
    idx = np.broadcast_to(idx, (N_PT, N_band, N_config, 1)) # (planet, band, config, 1)
    N_DIT_4D_ref             = np.take_along_axis(N_DIT_4D_R,             idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    signal_4D_ref            = np.take_along_axis(signal_4D_R,            idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_halo_2_4D_ref      = np.take_along_axis(sigma_halo_2_4D_R,      idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_bkg_2_4D_ref       = np.take_along_axis(sigma_bkg_2_4D_R,       idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_RON_2_4D_ref       = np.take_along_axis(sigma_RON_2_4D_R,       idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_DC_2_4D_ref        = np.take_along_axis(sigma_DC_2_4D_R,        idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
    sigma_syst_base_2_4D_ref = np.take_along_axis(sigma_syst_base_2_4D_R, idx, axis=3)[..., 0][..., None]  # (planets, bands, configs, sigma_m)
        
    # --- sigma_syst^2 = sigma_m^2 * sigma_syst_base_2, by definition of sigma_syst_base_2 (sigma_m = residuals amplitude over the FWHM and over the entire integration exposure_time) ---
    sigma_syst_2_4D_R   = sigma_m_ref**2                  * sigma_syst_base_2_4D_R    # (planets, bands, configs, R)
    sigma_syst_2_4D_l0  = sigma_m_ref**2                  * sigma_syst_base_2_4D_l0   # (planets, bands, configs, l0)
    sigma_syst_2_4D_Dl  = sigma_m_ref**2                  * sigma_syst_base_2_4D_Dl   # (planets, bands, configs, Dl)
    sigma_syst_2_4D_TI  = sigma_m_ref**2                  * sigma_syst_base_2_4D_TI   # (planets, bands, configs, TI)
    sigma_syst_2_4D_SM  = sigma_m[None, None, None, :]**2 * sigma_syst_base_2_4D_ref  # (planets, bands, configs, SM)
    sigma_syst_2_4D_WFE = sigma_m_ref**2                  * sigma_syst_base_2_4D_WFE  # (planets, bands, configs, WFE)
    sigma_syst_2_4D_IWA = sigma_m_ref**2                  * sigma_syst_base_2_4D_IWA  # (planets, bands, configs, IWA)

    # --- SNR ---
    SNR_4D_R   = N_DIT_4D_R   * signal_4D_R   / np.sqrt( N_DIT_4D_R   * (sigma_halo_2_4D_R   + sigma_bkg_2_4D_R   + sigma_RON_2_4D_R   + sigma_DC_2_4D_R)   + N_DIT_4D_R**2   * sigma_syst_2_4D_R   )
    SNR_4D_l0  = N_DIT_4D_l0  * signal_4D_l0  / np.sqrt( N_DIT_4D_l0  * (sigma_halo_2_4D_l0  + sigma_bkg_2_4D_l0  + sigma_RON_2_4D_l0  + sigma_DC_2_4D_l0)  + N_DIT_4D_l0**2  * sigma_syst_2_4D_l0  )
    SNR_4D_Dl  = N_DIT_4D_Dl  * signal_4D_Dl  / np.sqrt( N_DIT_4D_Dl  * (sigma_halo_2_4D_Dl  + sigma_bkg_2_4D_Dl  + sigma_RON_2_4D_Dl  + sigma_DC_2_4D_Dl)  + N_DIT_4D_Dl**2  * sigma_syst_2_4D_Dl  )
    SNR_4D_TI  = N_DIT_4D_TI  * signal_4D_TI  / np.sqrt( N_DIT_4D_TI  * (sigma_halo_2_4D_TI  + sigma_bkg_2_4D_TI  + sigma_RON_2_4D_TI  + sigma_DC_2_4D_TI)  + N_DIT_4D_TI**2  * sigma_syst_2_4D_TI  )
    SNR_4D_SM  = N_DIT_4D_ref * signal_4D_ref / np.sqrt( N_DIT_4D_ref * (sigma_halo_2_4D_ref + sigma_bkg_2_4D_ref + sigma_RON_2_4D_ref + sigma_DC_2_4D_ref) + N_DIT_4D_ref**2 * sigma_syst_2_4D_SM  )
    SNR_4D_WFE = N_DIT_4D_WFE * signal_4D_WFE / np.sqrt( N_DIT_4D_WFE * (sigma_halo_2_4D_WFE + sigma_bkg_2_4D_WFE + sigma_RON_2_4D_WFE + sigma_DC_2_4D_WFE) + N_DIT_4D_WFE**2 * sigma_syst_2_4D_WFE )
    SNR_4D_IWA = N_DIT_4D_IWA * signal_4D_IWA / np.sqrt( N_DIT_4D_IWA * (sigma_halo_2_4D_IWA + sigma_bkg_2_4D_IWA + sigma_RON_2_4D_IWA + sigma_DC_2_4D_IWA) + N_DIT_4D_IWA**2 * sigma_syst_2_4D_IWA )
    
    return SNR_4D_R, SNR_4D_l0, SNR_4D_Dl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA




# --------------------------------------------
# Function for computing detection probability
# --------------------------------------------

def get_Pdet(post_processing, exposure_time, min_DIT, DC0, RON0, RON_lim, A_FWHM, SNR_thr, separation_planets,
             idx_R_ref, sigma_m, sigma_m_ref, FoV, FoV_ref, N_band, N_config, N_PT, N_R, N_l0, N_Dl, N_TI, N_SM, N_FoV, N_WFE, N_IWA,
             SNR_4D_R, SNR_4D_l0, SNR_4D_Dl, SNR_4D_TI, SNR_4D_SM, SNR_4D_WFE, SNR_4D_IWA,
             dtype, indices=None, idx_l0_ref=None, idx_Dl_ref=None):
    
    if indices is None:
        indices = np.arange(N_PT, dtype=int)
    else:
        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.where(indices)[0]
        indices = indices.astype(int)
    N_PT_sel = len(indices)
    if N_PT_sel == 0:
        return (
            np.full((N_band, N_config, N_R),   np.nan, dtype=dtype),
            np.full((N_band, N_config, N_l0),  np.nan, dtype=dtype),
            np.full((N_band, N_config, N_Dl),  np.nan, dtype=dtype),
            np.full((N_band, N_config, N_TI),  np.nan, dtype=dtype),
            np.full((N_band, N_config, N_SM),  np.nan, dtype=dtype),
            np.full((N_band, N_config, N_FoV), np.nan, dtype=dtype),
            np.full((N_band, N_config, N_WFE), np.nan, dtype=dtype),
            np.full((N_band, N_config, N_IWA), np.nan, dtype=dtype),
        )
    
    FoV_half = FoV/2
    
    # Detection counting : nb of detected planet above the threshold for each point
    N_detections_3D_R   = np.zeros((N_band, N_config, N_R),   dtype=np.uint32)
    N_detections_3D_l0  = np.zeros((N_band, N_config, N_l0),  dtype=np.uint32)
    N_detections_3D_Dl  = np.zeros((N_band, N_config, N_Dl),  dtype=np.uint32)
    N_detections_3D_TI  = np.zeros((N_band, N_config, N_TI),  dtype=np.uint32)
    N_detections_3D_SM  = np.zeros((N_band, N_config, N_SM),  dtype=np.uint32)
    N_detections_3D_FoV = np.zeros((N_band, N_config, N_FoV), dtype=np.uint32)
    N_detections_3D_WFE = np.zeros((N_band, N_config, N_WFE), dtype=np.uint32)
    N_detections_3D_IWA = np.zeros((N_band, N_config, N_IWA), dtype=np.uint32)

    for idx in tqdm(indices, total=N_PT_sel, desc="SNR planets"):

        # --- SNR per planet ---
        SNR_3D_R   = SNR_4D_R[idx]
        SNR_3D_l0  = SNR_4D_l0[idx]
        SNR_3D_Dl  = SNR_4D_Dl[idx]
        SNR_3D_TI  = SNR_4D_TI[idx]
        SNR_3D_SM  = SNR_4D_SM[idx]
        SNR_3D_WFE = SNR_4D_WFE[idx]
        SNR_3D_IWA = SNR_4D_IWA[idx]
        
        # # For impossible off-axis      
        # if separation_planets[idx] <= FoV_ref/2:

        # --- Counting detections, whatever the ref FoV value (assuming off-axis is possible) ---     
        N_detections_3D_R   += (np.nan_to_num(SNR_3D_R,   nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_l0  += (np.nan_to_num(SNR_3D_l0,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_Dl  += (np.nan_to_num(SNR_3D_Dl,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_TI  += (np.nan_to_num(SNR_3D_TI,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_SM  += (np.nan_to_num(SNR_3D_SM,  nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_WFE += (np.nan_to_num(SNR_3D_WFE, nan=-np.inf) >= SNR_thr).astype(np.uint32)
        N_detections_3D_IWA += (np.nan_to_num(SNR_3D_IWA, nan=-np.inf) >= SNR_thr).astype(np.uint32)

        # --- FoV axis: update only if the planet can enter the FoV for some values ---
        p0 = np.searchsorted(FoV_half, separation_planets[idx], side="left")  # first FoV that includes the planet
        if p0 < N_FoV:
            # Reference SNR for FoV axis: choose your "reference point"
            # Here: R_ref per band
            SNR_2D_ref = SNR_3D_R[np.arange(N_band), :, idx_R_ref]                             # (bands, configs)
            det_ref    = (np.nan_to_num(SNR_2D_ref, nan=-np.inf) >= SNR_thr).astype(np.uint32) # (bands, configs)
    
            # The planet is inside FoV for all f >= p0
            N_detections_3D_FoV[:, :, p0:] += det_ref[:, :, None]
            
    # --- Computing detection probabilities --- 
    Pdet_3D_R   = N_detections_3D_R.astype(dtype)   / N_PT_sel # [0..1]
    Pdet_3D_l0  = N_detections_3D_l0.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_Dl  = N_detections_3D_Dl.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_TI  = N_detections_3D_TI.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_SM  = N_detections_3D_SM.astype(dtype)  / N_PT_sel # [0..1]
    Pdet_3D_FoV = N_detections_3D_FoV.astype(dtype) / N_PT_sel # [0..1]
    Pdet_3D_WFE = N_detections_3D_WFE.astype(dtype) / N_PT_sel # [0..1]
    Pdet_3D_IWA = N_detections_3D_IWA.astype(dtype) / N_PT_sel # [0..1]
    
    return Pdet_3D_R, Pdet_3D_l0, Pdet_3D_Dl, Pdet_3D_TI, Pdet_3D_SM, Pdet_3D_FoV, Pdet_3D_WFE, Pdet_3D_IWA




# ---------------------------------------------
# Function for computing required exposure time
# ---------------------------------------------

def get_exposure_time(min_DIT, DC0, RON0, RON_lim, A_FWHM, NbPixel, SNR_thr, idx_R_ref, sigma_m, sigma_m_ref,
                      signal_4D_R,   sigma_halo_2_4D_R,   sigma_syst_base_2_4D_R,   sigma_bkg_2_4D_R,   DIT_4D_R,
                      signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_syst_base_2_4D_l0,  sigma_bkg_2_4D_l0,  DIT_4D_l0,
                      signal_4D_Dl,  sigma_halo_2_4D_Dl,  sigma_syst_base_2_4D_Dl,  sigma_bkg_2_4D_Dl,  DIT_4D_Dl,
                      signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_syst_base_2_4D_TI,  sigma_bkg_2_4D_TI,  DIT_4D_TI,
                      signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_syst_base_2_4D_WFE, sigma_bkg_2_4D_WFE, DIT_4D_WFE,
                      signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_syst_base_2_4D_IWA, sigma_bkg_2_4D_IWA, DIT_4D_IWA,
                      dtype):

    # Data shape = (planets, bands, configs, *)
    N_PT, N_band, N_config, _ = signal_4D_R.shape

    # --- Detector variances per DIT ---
    def detector_variances(DIT):
        sigma_RON_2 = sigma_RON_2_from_DIT(DIT, min_DIT, RON0, RON_lim, A_FWHM, NbPixel=NbPixel, dtype=dtype)
        sigma_DC_2  = (DC0 * DIT * A_FWHM).astype(dtype)
        if NbPixel is not None:
            sigma_DC_2 *= NbPixel
        return sigma_RON_2, sigma_DC_2

    sigma_RON_2_4D_R,   sigma_DC_2_4D_R   = detector_variances(DIT_4D_R)
    sigma_RON_2_4D_l0,  sigma_DC_2_4D_l0  = detector_variances(DIT_4D_l0)
    sigma_RON_2_4D_Dl,  sigma_DC_2_4D_Dl  = detector_variances(DIT_4D_Dl)
    sigma_RON_2_4D_TI,  sigma_DC_2_4D_TI  = detector_variances(DIT_4D_TI)
    sigma_RON_2_4D_WFE, sigma_DC_2_4D_WFE = detector_variances(DIT_4D_WFE)
    sigma_RON_2_4D_IWA, sigma_DC_2_4D_IWA = detector_variances(DIT_4D_IWA)

    # --- Reference quantities (for sigma_m axis) ---
    idx = np.broadcast_to(idx_R_ref[None, :, None, None], (N_PT, N_band, N_config, 1))
    DIT_4D_ref             = np.take_along_axis(DIT_4D_R,             idx, axis=3)[..., 0][..., None]
    signal_4D_ref          = np.take_along_axis(signal_4D_R,          idx, axis=3)[..., 0][..., None]
    sigma_halo_2_4D_ref    = np.take_along_axis(sigma_halo_2_4D_R,    idx, axis=3)[..., 0][..., None]
    sigma_bkg_2_4D_ref     = np.take_along_axis(sigma_bkg_2_4D_R,     idx, axis=3)[..., 0][..., None]
    sigma_RON_2_4D_ref     = np.take_along_axis(sigma_RON_2_4D_R,     idx, axis=3)[..., 0][..., None]
    sigma_DC_2_4D_ref      = np.take_along_axis(sigma_DC_2_4D_R,      idx, axis=3)[..., 0][..., None]
    sigma_syst_base_2_ref  = np.take_along_axis(sigma_syst_base_2_4D_R, idx, axis=3)[..., 0][..., None]

    # --- Systematic-noise variances ---
    sigma_syst_2_4D_R   = sigma_m_ref**2                  * sigma_syst_base_2_4D_R
    sigma_syst_2_4D_l0  = sigma_m_ref**2                  * sigma_syst_base_2_4D_l0
    sigma_syst_2_4D_Dl  = sigma_m_ref**2                  * sigma_syst_base_2_4D_Dl
    sigma_syst_2_4D_TI  = sigma_m_ref**2                  * sigma_syst_base_2_4D_TI
    sigma_syst_2_4D_SM  = sigma_m[None, None, None, :]**2 * sigma_syst_base_2_ref
    sigma_syst_2_4D_WFE = sigma_m_ref**2                  * sigma_syst_base_2_4D_WFE
    sigma_syst_2_4D_IWA = sigma_m_ref**2                  * sigma_syst_base_2_4D_IWA

    # --- Required exposure time ---
    def time_to_SNR(signal, sigma_halo_2, sigma_bkg_2, sigma_RON_2, sigma_DC_2, sigma_syst_2, DIT):
        signal, sigma_halo_2, sigma_bkg_2, sigma_RON_2, sigma_DC_2, sigma_syst_2, DIT = np.broadcast_arrays(signal, sigma_halo_2, sigma_bkg_2, sigma_RON_2, sigma_DC_2, sigma_syst_2, DIT)
        signal       = signal.astype(np.float64, copy=False)
        sigma_fund_2 = (sigma_halo_2 + sigma_bkg_2 + sigma_RON_2 + sigma_DC_2).astype(np.float64, copy=False)
        sigma_syst_2 = sigma_syst_2.astype(np.float64, copy=False)
        DIT          = DIT.astype(np.float64, copy=False)
        Texp         = np.full(signal.shape, np.nan, dtype=np.float64)
        valid        = np.isfinite(signal) & np.isfinite(sigma_fund_2) & np.isfinite(sigma_syst_2) & np.isfinite(DIT) & (signal > 0) & (sigma_fund_2 >= 0) & (sigma_syst_2 >= 0) & (DIT > 0)
        denominator  = signal**2 - SNR_thr**2 * sigma_syst_2
        unreachable  = valid & (denominator <= 0)
        reachable    = valid & (denominator > 0)
        Texp[unreachable] = np.inf
        N_req_cont        = np.zeros(signal.shape, dtype=np.float64)
        N_req_cont[reachable] = SNR_thr**2 * sigma_fund_2[reachable] / denominator[reachable]
        N_req = np.maximum(1.0, np.ceil(N_req_cont))
        # Numerical safety: make sure the rounded integer number of DITs really reaches SNR_thr
        snr_req = np.full(signal.shape, np.nan, dtype=np.float64)
        snr_req[reachable] = N_req[reachable] * signal[reachable] / np.sqrt(N_req[reachable] * sigma_fund_2[reachable] + N_req[reachable]**2 * sigma_syst_2[reachable])
        N_req[reachable & (snr_req < SNR_thr)] += 1
        Texp[reachable] = N_req[reachable] * DIT[reachable]
        return Texp.astype(dtype)

    Texp_4D_R   = time_to_SNR(signal_4D_R,   sigma_halo_2_4D_R,   sigma_bkg_2_4D_R,   sigma_RON_2_4D_R,   sigma_DC_2_4D_R,   sigma_syst_2_4D_R,   DIT_4D_R)
    Texp_4D_l0  = time_to_SNR(signal_4D_l0,  sigma_halo_2_4D_l0,  sigma_bkg_2_4D_l0,  sigma_RON_2_4D_l0,  sigma_DC_2_4D_l0,  sigma_syst_2_4D_l0,  DIT_4D_l0)
    Texp_4D_Dl  = time_to_SNR(signal_4D_Dl,  sigma_halo_2_4D_Dl,  sigma_bkg_2_4D_Dl,  sigma_RON_2_4D_Dl,  sigma_DC_2_4D_Dl,  sigma_syst_2_4D_Dl,  DIT_4D_Dl)
    Texp_4D_TI  = time_to_SNR(signal_4D_TI,  sigma_halo_2_4D_TI,  sigma_bkg_2_4D_TI,  sigma_RON_2_4D_TI,  sigma_DC_2_4D_TI,  sigma_syst_2_4D_TI,  DIT_4D_TI)
    Texp_4D_SM  = time_to_SNR(signal_4D_ref, sigma_halo_2_4D_ref, sigma_bkg_2_4D_ref, sigma_RON_2_4D_ref, sigma_DC_2_4D_ref, sigma_syst_2_4D_SM,  DIT_4D_ref)
    Texp_4D_WFE = time_to_SNR(signal_4D_WFE, sigma_halo_2_4D_WFE, sigma_bkg_2_4D_WFE, sigma_RON_2_4D_WFE, sigma_DC_2_4D_WFE, sigma_syst_2_4D_WFE, DIT_4D_WFE)
    Texp_4D_IWA = time_to_SNR(signal_4D_IWA, sigma_halo_2_4D_IWA, sigma_bkg_2_4D_IWA, sigma_RON_2_4D_IWA, sigma_DC_2_4D_IWA, sigma_syst_2_4D_IWA, DIT_4D_IWA)

    return Texp_4D_R, Texp_4D_l0, Texp_4D_Dl, Texp_4D_TI, Texp_4D_SM, Texp_4D_WFE, Texp_4D_IWA

