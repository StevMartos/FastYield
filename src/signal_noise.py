# import FastYield modules
from src.config import sim_data_path, rad2arcsec
from src.get_specs import _load_stellar_modulation_function, get_config_data, get_transmission
from src.utils import extract_jwst_data, PCA_subtraction, circular_mask, annular_mask, box_convolution, fill_nan_linear
from src.spectrum import Spectrum, get_resolution, filtered_flux, _fft_filter_response, get_spectrum_band
from src.data_processing import get_S_res, get_CCF_2D_rv

# import astropy modules
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import RegularGridInterpolator

# import other modules
from functools import lru_cache
from numba import njit, prange
import math

# For fits warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", message="Header block contains null bytes*")



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



def get_DIT_RON(config_data, instru_type, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, RON, RON_lim, saturation_e, input_DIT, separation_planet=None):
    """
    Compute the detector integration time (DIT) to avoid saturation and the effective read-noise.

    Parameters
    ----------
    config_data: dict
        Instrument specs.
    PSF_profile
        Radial PSF profile (in fraction per pixel^2, already scaled).
    separation
        Separation vector matching 'PSF_profile'.
    star_spectrum_band
        Star spectrum [ph/mn] on the considered band and grid.
    exposure_time
        Total exposure time [mn].
    min_DIT, max_DIT
        Min/max allowed DIT [mn].
    trans
        Total transmission on the considered band (array).
    RON
        Read-Out Noise [e-/px].
    saturation_e
        Full-well capacity [e-].
    input_DIT
        If provided, forces the DIT (still clamped by exposure time).

    Returns
    -------
    DIT
        Integration time per exposure [mn].
    RON_eff
        Effective read-out noise per DIT [e-/px].
    """
    # Per-DIT max electron rate estimate (worst-case pixel in the PSF core)
    if instru_type == "imager":
        func = np.nansum
    else:
        func = np.nanmax
    try:
        iwa_FPM_values = config_data["FPMs"]
        iwa_FPM        = max((x for x in iwa_FPM_values if x < separation_planet), default=0.)
    except:
        iwa_FPM = 0.
    
    # DIT when the saturation is reached on the brightest pixel
    max_flux_e     = np.nanmax(PSF_profile[separation>=iwa_FPM])              * func(trans*star_spectrum_band.flux) # brightest pixel [e-/bin/mn] or [e-/mn]
    DIT_saturation = saturation_e / max_flux_e # [mn]

    # DIT when the saturation is reached at the planet position (if needed)
    if separation_planet is not None:
        max_flux_e_planet     = np.nanmax(PSF_profile[separation==separation_planet][0]) * func(trans*star_spectrum_band.flux) # flux at the planet position [e-/bin/mn] or [e-/mn]
        DIT_saturation_planet = saturation_e / max_flux_e_planet # [mn]
    else:
        DIT_saturation_planet = None
        
    if "fiber" in instru_type and DIT_saturation_planet is not None:
        DIT_saturation = DIT_saturation_planet

    # Apply limits / overrides
    if input_DIT is not None:
        DIT = input_DIT
    else:
        DIT = np.clip(DIT_saturation, min_DIT, max_DIT)
    DIT = min(DIT, exposure_time) # [mn]: The DIT cannot be longer than the total exposure time
    
    # Up-the-ramp effective read-noise (see https://arxiv.org/pdf/0706.2344)
    if DIT >= 2*min_DIT: # At least 2 reads inside the ramp
        N_read  = DIT / min_DIT                                         # Number of intermittent non-destructive readings
        RON_eff = estimate_RON_UTR(N=N_read, RON0=RON, RON_lim=RON_lim) # Effective read out noise [e-/px/DIT]
        RON_eff = min(RON_eff, RON)
    else:
        RON_eff = RON
    
    # Total number of integrations
    NDIT = exposure_time / DIT
            
    return NDIT, DIT, DIT_saturation, DIT_saturation_planet, RON_eff, iwa_FPM


# -------------------------------------------------------------------------
# δ term for differential imaging with non-imager
# -------------------------------------------------------------------------
    
def get_delta_cos_theta_syst(Mp_Sp, template, trans):
    """
    Compute the useful photo-electron rate δ (per DIT) used in differential imaging.
    δ = norm( trans*Sp )

    Parameters
    ----------
    Sp: 1d array
        Planet spectrum on the band grid [ph/bin/DIT] (total ph over the FoV).
    template: 1d array
        (Unitless) normalized template on the same spectral grid.
    trans: 1d array
        Total system transmission on the same grid as the spectrum (scalar or array).

    Returns
    -------
    delta
        δ [e-/DIT] (total e- over the FoV).
    """
    delta          = np.sqrt( np.nansum( (trans*Mp_Sp)**2 ) )     # DI signal term [e-/DIT] (total e- over the FoV)
    cos_theta_syst = np.nansum( trans*Mp_Sp * template ) / delta  # Correlation signal loss induced by systematic modulations
    cos_theta_syst = np.clip(a=cos_theta_syst, a_min=-1, a_max=1) # cos_theta_syst in [-1, 1]
    return delta, cos_theta_syst # [e-/DIT] (total e- over the FoV), [no unit]
    


# -------------------------------------------------------------------------
# α and β terms for molecular mapping
# -------------------------------------------------------------------------
    
def get_alpha_cos_theta_syst(Mp_Sp, trans, template, R, Rc, filter_type):
    """
    Compute the useful photo-electron rate α (per DIT) used in molecular mapping.
    α = norm( trans * [Mp*Sp]_HF )
    where _HF denotes the high-pass filtered spectrum.

    Parameters
    ----------
    Mp_Sp: 1d array
        Planet spectrum on the band grid [ph/bin/DIT] (total ph over the FoV).
    trans: 1d array
        Total system transmission on the same grid as the spectrum (scalar or array).
    template: 1d array
        (Unitless) normalized template on the same spectral grid.
    R: float
        Effective spectral resolution of the input array (assuming Nyquist sampling).
    Rc: float
        Cut-off resolving power for the filter; if None, no filtering (HP=Sp, LP=0).
    filter_type: str
        Filter kind passed to 'filtered_flux' ("gaussian", "step", ...).

    Returns
    -------
    alpha
        α [e-/DIT] (total e- over the FoV).
    """
    Mp_Sp_HF, _    = filtered_flux(flux=Mp_Sp, R=R, Rc=Rc, filter_type=filter_type) # High_pass filtered planetary flux in [e-/bin/DIT] (total e- over the FoV)
    alpha          = np.sqrt( np.nansum( (trans*Mp_Sp_HF)**2 ) )                     # MM signal term [e-/DIT] (total e- over the FoV)
    cos_theta_syst = np.nansum( trans*Mp_Sp_HF * template ) / alpha                  # Correlation signal loss induced by systematic modulations
    cos_theta_syst = np.clip(a=cos_theta_syst, a_min=-1, a_max=1)                    # cos_theta_syst in [-1, 1]
    return alpha, cos_theta_syst # [e-/DIT] (total e- over the FoV), [no unit]



def get_beta(Ss, Mp_Sp, trans, template, R, Rc, filter_type):
    """
    Compute the self-subtraction term β (per DIT).
    β ≈ sum( trans * Ss_HF * Mp_Sp_LF / Ss_LF * template )

    Parameters
    ----------
    Ss, Sp: 1d array
        Star/planet spectra on the band grid in [ph/bin/DIT] (total ph over the FoV).    
    trans: 1d array
        Total system transmission on the same grid as the spectrum (scalar or array).
    template: 1d array
        (Unitless) normalized template on the same spectral grid.
    R: float
        Effective spectral resolution of the input array (assuming Nyquist sampling).
    Rc: float
        Cut-off resolving power for HP/LP filter. If None, β = 0.
    filter_type: str
        Filter kind passed to 'filtered_flux' ("gaussian", "step", ...).

    Returns
    -------
    beta
        β [e-/DIT] (total e- over the FoV).
    """
    if Rc is None or Rc == 0:
        beta = 0
    else:
        
        Ss_HF, Ss_LF = filtered_flux(Ss,   R=R, Rc=Rc, filter_type=filter_type)  # Star filtered spectra in [ph/bin/DIT] (total ph over the FoV)
        _, Mp_Sp_LF  = filtered_flux(Mp_Sp, R=R, Rc=Rc, filter_type=filter_type) # Planet filtered spectra in [ph/bin/DIT] (total ph over the FoV)
        beta         = np.nansum(trans*Ss_HF*Mp_Sp_LF/Ss_LF * template)          # MM self-subtraction term in [e-/mn] (total e- over the FoV)
    return beta # [e-/DIT] (total e- over the FoV)



# --------------------------------------------------------------------------------------------------------------------------
# Power fraction of effective white noise after high-pass filtering fn_HF = sigma_HF**2/sigma**2 (mean per spectral channel)
# --------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=128)
def get_fn_HF_LF(N, R, Rc, filter_type, empirical=False):
    """
    Return the high-pass (HF) and low-pass (LF) white-noise variance fractions
    associated with the spectral filter split defined by
    ''_fft_filter_response''.

    This function evaluates the dimensionless factors ''fn_HF'' and ''fn_LF''
    such that, for an input white-noise vector ''n'' with per-channel variance
    ''sigma^2'',

        Var([n]_HF) = sigma^2 * fn_HF
        Var([n]_LF) = sigma^2 * fn_LF

    where ''[.]_HF'' and ''[.]_LF'' denote the high-pass and low-pass branches
    of the filter, and the variances are understood here as per-spectral-channel
    variances, not as variances after projection into a cross-correlation
    function (CCF).

    Parameters
    ----------
    N : int
        Number of uniformly sampled spectral channels.
    R : float
        Sampling resolving power of the high-resolution grid (Nyquist
        convention). This is used internally to map FFT frequencies to the
        resolution domain.
    Rc : float or None
        Cut-off resolution used to define the LF/HF split. If ''Rc'' is
        ''None'' or ''0'', no filtering is applied and the function returns
        ''(1.0, 0.0)''.
    filter_type : {'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep'}
        Shape of the low-pass transfer function used to construct the LF/HF
        filter pair.
    empirical : bool, optional
        If ''False'' (default), compute the variance fractions analytically from
        the FFT-domain transfer functions. If ''True'', estimate them
        empirically by Monte Carlo using Gaussian white-noise realizations.

    Returns
    -------
    fn_HF : float
        Fraction of the input white-noise variance transmitted to the HF
        component, i.e.

            fn_HF = sigma_HF^2 / sigma^2.

    fn_LF : float
        Fraction of the input white-noise variance transmitted to the LF
        component, i.e.

            fn_LF = sigma_LF^2 / sigma^2.

    Notes
    -----
    In the analytical branch, the function retrieves the FFT-domain transfer
    functions ''H_HF'' and ''H_LF'' and uses the standard result for white
    noise filtered by a linear circulant operator:

        fn_HF = mean(|H_HF|^2)
        fn_LF = mean(|H_LF|^2).

    These quantities correspond to the per-channel output variances for a white,
    independent, identically distributed input noise.

    In the empirical branch, the function generates ''1000'' Gaussian white-
    noise realizations with unit variance, filters each realization with
    ''filtered_flux'', and estimates

        fn_HF = <var(noise_HF)>
        fn_LF = <var(noise_LF)>,

    where the average is taken over the Monte Carlo realizations.

    Caution
    -------
    The returned quantities describe the variance of the filtered noise itself,
    channel by channel. They do not represent the variance of a projected
    quantity such as a CCF or matched-filter output.

    Also note that, for a generic complementary split defined by
    ''H_HF = 1 - H_LF'', one does not necessarily have

        fn_HF + fn_LF = 1,

    unless the filter pair is power-complementary.
    """
    if Rc is None or Rc == 0:
        return 1., 0.
    
    # --- Analytical: white-noise power fractions for the LF/HF split, cached by (N, R, Rc, type).
    if not empirical:
        H_HF, H_LF = _fft_filter_response(N=N, R=R, Rc=Rc, filter_type=filter_type)
        # For white noise, output variance ∝ mean(|H|^2) over frequencies.
        fn_LF = np.nanmean( np.abs(H_LF)**2 )
        fn_HF = np.nanmean( np.abs(H_HF)**2 )

    # --- Empirical (debug) path: slower, but straightforward ---
    else:
        # Keep sample count modest; analytic path is preferred.
        n     = 1000
        fn_HF = 0.
        fn_LF = 0.
        var   = 1
        sig   = np.sqrt(var)
        for i in range(n):
            noise              = np.random.normal(0, sig, N)
            noise_HF, noise_LF = filtered_flux(flux=noise, R=R, Rc=Rc, filter_type=filter_type)
            fn_HF             += np.nanvar(noise_HF) / var / n
            fn_LF             += np.nanvar(noise_LF) / var / n
        
    return fn_HF, fn_LF



# -------------------------------------------------------------------------------
# Power fraction of effective MM white noise in the CCF fn = sigma_MM**2/sigma**2
# -------------------------------------------------------------------------------

def get_fn_MM(template, R, Rc, filter_type, sigma=None):
    """
    Compute the relative variance propagation factor of spectral noise into the
    molecular-mapping (MM) cross-correlation function (CCF), assuming that the
    MM residual noise is reduced to the high-frequency filtered noise term only.

    The function explicitly treats non-finite spectral channels as rejected
    channels. Let M denote the diagonal mask operator selecting the valid
    channels, and let H_HF denote the FFT-based high-pass filtering operator
    returned by '_fft_filter_response'. The effective operator considered here is

        B = M H_HF M.

    The effective template used in the CCF is the masked and L2-normalized
    template

        t_eff = M t / ||M t||.

    The projected MM noise is then written as

        CCF_noise = <B n, t_eff> = <n, B* t_eff>,

    where B* is the adjoint of the masked operator. Defining

        q = B* t_eff,

    the function returns the relative variance propagation factor

        fn_MM = Var(<B n, t_eff>) / Var(<n, t_eff>).

    For independent spectral channels with per-channel standard deviation
    sigma_i, this becomes

        fn_MM = sum_i sigma_i^2 |q_i|^2 / sum_i sigma_i^2 |t_eff,i|^2.

    If 'sigma' is not provided, the input noise is assumed to be white and
    homoscedastic over the valid channels, so that

        fn_MM = ||q||^2,

    because 't_eff' is normalized to unit L2 norm.

    Parameters
    ----------
    template : 1D ndarray
        Spectral template used in the CCF projection. Non-finite entries are
        treated as rejected channels and excluded through the validity mask.
    R : float
        Spectral resolution of the input spectrum.
    Rc : float
        Cutoff resolution defining the high-pass / low-pass spectral
        decomposition.
    filter_type : str
        Type of spectral filter passed to '_fft_filter_response'.
    sigma : 1D ndarray, optional
        Per-channel standard deviation of the input spectral noise. If provided,
        it must have the same shape as 'template'. Only finite and non-negative
        values are retained. If 'None', the input noise is assumed to be white
        and homoscedastic over the valid channels.

    Returns
    -------
    fn_MM : float
        Relative variance propagation factor of the projected MM noise, defined
        as the ratio between the CCF noise variance after propagation through
        the masked high-pass operator and the reference variance obtained from a
        direct projection onto the same effective template.

    Raises
    ------
    ValueError
        If 'template' is not one-dimensional, if 'sigma' has an incompatible
        shape, if no valid channels remain after masking, if the masked template
        has zero norm, or if the reference projected variance is zero or
        non-finite.

    Notes
    -----
    The validity mask is defined from the finite entries of 'template', and
    additionally from the finite non-negative entries of 'sigma' when 'sigma'
    is provided.

    Outside the validity mask, the template and the optional standard deviation
    are set to zero. The quantity returned by this function therefore
    corresponds to the masked operator B = M H_HF M, not to an ideal high-pass
    operator acting on a complete spectral grid without rejected channels.

    This function describes the simplified masked MM noise model

    y = [n]_HF = B n = M H_HF M n,

    that is, the high-frequency filtered input noise restricted to the accepted
    spectral channels and does not include additional noise terms that may appear in a more
    complete MM residual model, such as low-frequency subtraction terms coupled
    to the stellar estimate.
    """
    # Convert inputs
    template = np.asarray(template, dtype=float)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)

    # Basic shape checks
    if template.ndim != 1:
        raise ValueError("template must be a 1D array")
    if sigma is not None and (sigma.ndim != 1 or sigma.shape != template.shape):
        raise ValueError("sigma must be a 1D array with the same shape as template")

    N = template.size

    # Valid channels:
    # - finite template values
    # - and, if provided, finite non-negative noise standard deviations
    valid = np.isfinite(template)
    if sigma is not None:
        valid &= np.isfinite(sigma) & (sigma >= 0)

    if not np.any(valid):
        raise ValueError("No valid spectral channels remain after masking")

    # Build the masked effective template on the full spectral grid
    t_eff        = np.zeros_like(template, dtype=float)
    t_eff[valid] = template[valid]

    norm_t = np.sqrt(np.sum(np.abs(t_eff)**2))
    if not np.isfinite(norm_t) or norm_t == 0:
        raise ValueError("Template has zero norm on valid channels")
    t_eff /= norm_t

    # Optional masked per-channel noise standard deviation
    if sigma is not None:
        sigma_eff        = np.zeros_like(sigma, dtype=float)
        sigma_eff[valid] = sigma[valid]

    # FFT-domain response of the high-pass operator H_HF
    H_HF, _ = _fft_filter_response(N=N, R=R, Rc=Rc, filter_type=filter_type)

    # Apply the adjoint of the masked operator B = M H_HF M:
    # since t_eff is already masked, we compute H_HF^* t_eff on the full grid,
    # then explicitly re-apply the mask to obtain q = M H_HF^* M t_eff = B^* t_eff.
    q         = np.fft.ifft(np.fft.fft(t_eff) * np.conj(H_HF))
    q[~valid] = 0.0

    # Relative variance propagation factor
    if sigma is None:
        # White, homoscedastic input noise on the valid channels
        fn_MM = float(np.vdot(q, q).real)
    else:
        # Independent, heteroscedastic input noise
        num = float(np.sum((sigma_eff**2) * np.abs(q)**2).real)
        den = float(np.sum((sigma_eff**2) * np.abs(t_eff)**2).real)

        if den <= 0 or not np.isfinite(den):
            raise ValueError("The reference CCF noise variance is zero or non-finite")

        fn_MM = num / den
    
    if fn_MM < 0 or fn_MM > 1:
        raise ValueError(f"This should not happen! (fn_MM = {fn_MM:.3f})")
    
    return fn_MM



def get_fn_MM_exact(trans, Ss, template, R, Rc, filter_type, sigma=None):
    """
    Compute the relative white-noise power factor induced by the approximate
    molecular mapping (MM) operator in the cross-correlation function (CCF).

    This function evaluates the factor

        fn_MM = sigma_CCF,MM^2 / sigma_CCF^2,

    where sigma_CCF,MM^2 is the variance of the noise after propagation through
    an approximate MM residual operator, and sigma_CCF^2 is the variance of the
    reference CCF obtained by projecting the input noise directly onto the
    normalized template.

    The approximate MM noise model is written as

        y = [n]_HF - [n]_LF * ([a]_HF / [a]_LF),

    where
        - n is the input spectral noise,
        - a = trans * Ss,
        - [.]_HF and [.]_LF denote the high-pass and low-pass branches of the
          spectral filter defined by '_fft_filter_response'.

    In operator form,

        y = B n
          = (H_HF - diag(r) @ H_LF) n,

    with

        r = [a]_HF / [a]_LF.

    If t_eff is the masked and L2-normalized template, the projected MM noise
    in the CCF can be written as

        CCF_MM,noise = <Bn, t_eff> = <n, B* t_eff>,

    where B* is the adjoint of B. Defining

        q = B* t_eff,

    the returned factor is

        fn_MM = sum_i sigma_i^2 |q_i|^2 / sum_i sigma_i^2 |t_eff,i|^2,

    for independent input noise with per-channel standard deviation sigma_i.

    If 'sigma' is not provided, the function assumes independent white noise
    with identical variance in all valid channels. In that case, because the
    template is normalized to unit L2 norm, the denominator is unity and

        fn_MM = ||q||^2.

    Parameters
    ----------
    trans : 1D ndarray
        Spectral transmission vector sampled on the same grid as 'Ss' and
        'template'.
    Ss : 1D ndarray
        Stellar spectrum sampled on the same spectral grid as 'trans'.
    template : 1D ndarray
        Template used to project the MM residual noise into the CCF. It is
        masked on invalid channels and normalized internally to unit L2 norm.
    R : float
        Spectral resolution of the data.
    Rc : float
        Cut-off resolution used for the high-pass / low-pass spectral split.
    filter_type : str
        Filter type passed to '_fft_filter_response'.
    sigma : 1D ndarray or None, optional
        Per-channel standard deviation of the input spectral noise. If provided,
        it must have the same shape as 'trans', 'Ss', and 'template', and is
        assumed to describe independent but not necessarily identically
        distributed noise. If None, the input noise is assumed to be white and
        homoscedastic across all valid channels.

    Returns
    -------
    fn_MM : float
        Relative MM white-noise power in the CCF, defined as the ratio between
        the projected CCF noise variance after the approximate MM operator and
        the projected CCF noise variance of the direct template projection.

    Raises
    ------
    ValueError
        If 'trans', 'Ss', or 'template' are not 1D arrays, if they do not have
        the same shape, if 'sigma' is provided with an incompatible shape, or
        if no valid spectral channels remain after masking.
    ValueError
        If the computed value of 'fn_MM' falls outside the expected range
        enforced by the implementation.

    Notes
    -----
    The validity mask is defined as

        valid = isfinite(trans * Ss) & isfinite(template),

    and additionally

        valid &= isfinite(sigma) & (sigma >= 0)

    when 'sigma' is provided.

    Outside this mask, 'a = trans * Ss', 'template', and 'sigma' (if provided)
    are zero-filled before applying the FFT-based filtering operators.

    The ratio

        r = [a]_HF / [a]_LF

    is evaluated only where both filtered quantities are finite and
    '[a]_LF != 0'; it is set to zero elsewhere.

    After applying the adjoint operator, the vector q = B* t_eff is explicitly
    set to zero outside the validity mask. Therefore, invalid channels do not
    contribute to the propagated noise variance.

    Caution
    -------
    This function is exact only for the surrogate linear model

        y = [n]_HF - [n]_LF * ([a]_HF / [a]_LF),

    not for the original MM expression

        y = a * [n / a]_HF.

    Its accuracy therefore depends on how well this approximation describes the
    actual MM noise in the regime of interest.
    """
    
    # --- Inputs ---
    trans     = np.asarray(trans,    dtype=float)
    Ss        = np.asarray(Ss,       dtype=float)
    template  = np.asarray(template, dtype=float)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)

    if trans.ndim != 1 or Ss.ndim != 1 or template.ndim != 1:
        raise ValueError("trans, Ss, and template must be 1D arrays")
    if trans.shape != Ss.shape or trans.shape != template.shape:
        raise ValueError("trans, Ss, and template must have the same shape")
    if sigma is not None:
        if sigma.ndim != 1 or sigma.shape != trans.shape:
            raise ValueError("sigma must be a 1D array with the same shape as trans")

    N = trans.size
    a = trans * Ss
    
    # Validity masks
    valid = np.isfinite(a) & np.isfinite(template)    
    if sigma is not None:
        valid &= np.isfinite(sigma) & (sigma >= 0)

    if not np.any(valid):
        raise ValueError("No valid spectral channels remain after masking")
    
    # Masked full-grid vectors
    a_eff         = np.zeros_like(a,        dtype=float)
    t_eff         = np.zeros_like(template, dtype=float)
    if sigma is not None:
        sigma_eff = np.zeros_like(sigma,    dtype=float)

    a_eff[valid]         = a[valid]
    t_eff[valid]         = template[valid]
    t_eff               /= np.sqrt(np.nansum(t_eff**2))
    if sigma is not None:
        sigma_eff[valid] = sigma[valid]
    
    # FFT-domain filter responses
    H_HF, H_LF = _fft_filter_response(N=N, R=R, Rc=Rc, filter_type=filter_type)

    # Build [a]_HF and [a]_LF on the full grid
    a_HF = np.fft.ifft(np.fft.fft(a_eff) * H_HF).real
    a_LF = np.fft.ifft(np.fft.fft(a_eff) * H_LF).real
    
    # Ratio r = [a]_HF / [a]_LF with zero safety
    r = np.zeros_like(a_eff, dtype=float)

    good_ratio    = np.isfinite(a_HF) & np.isfinite(a_LF) & (a_LF != 0)
    r[good_ratio] = a_HF[good_ratio] / a_LF[good_ratio]

    # Apply the adjoint operator:
    #
    # B^* t = H_HF^*(t) - H_LF^*(r * t)
    #
    q_hf      = np.fft.ifft(np.fft.fft(t_eff)     * np.conj(H_HF))
    q_lf      = np.fft.ifft(np.fft.fft(r * t_eff) * np.conj(H_LF))
    q         = q_hf - q_lf
    q[~valid] = 0.0
    
    if sigma is None:
        fn_MM = float(np.vdot(q, q).real)
    else:
        fn_MM = float(np.sum((sigma_eff**2) * np.abs(q)**2).real) / float(np.sum((sigma_eff**2) * np.abs(t_eff)**2))
    
    if fn_MM < 0 or fn_MM > 2:
        raise ValueError(f"This should not happen! (fn_MM = {fn_MM:.3f})")
    
    return fn_MM


#######################################################################################################################
##################################### SYSTEMATIC NOISE PROFILE CALCULATION: ###########################################
#######################################################################################################################

def get_systematics(config_data, band, tellurics, apodizer, strehl, coronagraph, R_band, Rc, filter_type, star_spectrum_instru, planet_spectrum_instru, wave_band, size_core, PCA=False, PCA_mask=False, N_PCA=20, mag_planet=None, separation_planet=None, mag_star=None, exposure_time=None, target_name=None, on_sky_data=False, sigma_outliers=3):
    """
    Estimate the systematic-noise radial profile (projected into the CCF), along with
    a few ancillary vectors used in performance predictions.

    The flow:
      1) Load a noiseless (or calibration) 3-D cube S_noiseless (Nchan, Ny, Nx).
      2) Renormalize each channel to the *current* stellar flux model [ph/mn].
      3) Apply the standard stellar high-pass filtering to obtain residuals S_res_wo_planet.
      4) Optionally simulate PCA on the filtered cube (with or without fake-planet injection)
         to estimate the signal loss factor M_pca and a PCA-cleaned CCF reference.
      5) Compute the CCF at each annulus and derive σ_syst'(ρ) ∝ Var(CCF)/F_star^2.
      6) Build Mp (planet modulation proxy).

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
    R_band : float
        Resolving power for the *band* being processed (used by filters).
    Rc : float or None
        Cut-off resolving power for the high-pass filter; None disables filtering.
    filter_type : str
        High-pass filter type ("gaussian", "smoothstep", "savitzky_golay", ...).
    star_spectrum_instru, planet_spectrum_instru : Spectrum
        Star, planet spectrum on the instrument-wide grid [ph/bin/mn] (total ph over the FoV).
    wave_band : (M,) ndarray
        Wavelength array for the *band* (for plotting and HF residual export).
    size_core : int
        Box size (in pixels) used to sum flux within the PSF core (~ FWHM box).
    PCA : bool, optional
        If True, attempt to reduce systematics via PCA on the high-pass filtered cube.
    PCA_mask : bool, optional
        Mask the planet location during PCA (only when a fake injection is done).
    N_PCA : int, optional
        Number of PCA components to subtract.
    mag_planet : optional
        Planet magnitude, required to be not None if you want a fake injection.
    separation_planet : float or None
        Planet separation in ['sep_unit'], required to be not None if you want a fake injection.
    mag_star, exposure_time, target_name : optional
        Metadata for PCA heuristics; exposure_time helps decide if PCA is beneficial.
    on_sky_data : bool, optional
        If True, use real calibration data (e.g. MAST) instead of simulated noiseless cubes.
    sigma_outliers : float, optional
        Sigma threshold for outlier rejection in stellar filtering steps if 'on_sky_data' is True.

    Returns
    -------
    sigma_syst_prime_2 : (Nr,) ndarray
        Estimated systematic noise power profile (variance of CCF per annulus / F_*^2).
    separation : (Nr,) ndarray
        Separation centers (in the instrument's separation unit).
    Mp : (M,) ndarray
        Proxy of the planet modulation function vs wavelength on 'wave_band'.
    M_pca : float
        Estimated multiplicative signal-loss factor due to PCA (≤ 1). Equals 1 if PCA not applied.
    wave_data : (Nchan,) ndarray
        Wavelength grid used by the cube and internal operations.
    pca : object or None
        The fitted PCA object (None if PCA not used or not computed).
    """
    PCA_verbose = None
    pca         = None
    PCA_calc    = False  # we’ll decide below if PCA = True
    instru      = config_data["name"]
    FOV         = config_data["FOV"]      # [arcsec]
    sep_unit    = config_data["sep_unit"] # 'mas' or 'arcsec'
    if sep_unit == "mas":
        FOV *= 1e3 # [mas]
    
    # --------------------------------------------------------------------------------
    # 1) Load stellar modulation function Ms (from noiseless or on-sky data) [no unit]
    # --------------------------------------------------------------------------------
    if instru == "MIRIMRS":
        #on_sky_data    = True
        correction     = "all_corrected" # correction = "with_fringes_straylight" # correction applied to the simulated MIRISim noiseless data
        T_star_sim_arr = np.array([4000, 6000, 8000]) # available values for the star temperature for MIRSim noiseless data
        T_star_sim     = T_star_sim_arr[np.abs(star_spectrum_instru.T - T_star_sim_arr).argmin()]
        if on_sky_data: # CALIBRATION DATA => High S/N per spectral channel => estimation of modulations: M_data 
            file = f"data/MIRIMRS/MAST/HD 159222_ch{band[0]}-shortmediumlong_s3d.fits" 
        else: # Using MIRISim (end-to-end simulation) data: in order to estimation modulations
            file = f"data/MIRIMRS/MIRISim/star_center/star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits"
    
    elif instru == "NIRSpec":
        file        = f"data/NIRSpec/MAST/HD 163466_nirspec_{band}_s3d.fits"
        on_sky_data = True
        T_star_sim  = None
        correction  = None
        warnings.simplefilter("ignore", category=RuntimeWarning) # Some slices are filled with NaN
    
    else:
        raise NotImplementedError("get_systematics not implemented for this instrument") # TODO: define for other instruments
        
    # Opening already existing Ms
    try: # [no unit], [µm] and [sep_unit/px] 
        Ms, wave_data, pxscale = _load_stellar_modulation_function(instru=instru, band=band, on_sky_data=on_sky_data, T_star_sim=T_star_sim, correction=correction)
    
    # Or create them (but the raw noiseless or on-sky data are needed)    
    except Exception as e:
        print(f"\n [get_systematics] Building modulation cube (first run): {e}")
        
        # Extracting 'fundamental-noiseless' data in [e-/bin/px] (S_noiseless = Ms * trans * Ss)
        if instru in {"MIRIMRS", "NIRSpec"}:
            S_noiseless, wave_data, pxscale, _, trans, exposure_time_data, _ = extract_jwst_data(instru=instru, target_name="sim", band=band, crop_band=True, outliers=on_sky_data, sigma_outliers=sigma_outliers, file=file, X0=None, Y0=None, R_crop=None, verbose=False)

        # For MIRISim data we know the injected star spectrum → re-derive Ss in [ph/bin]
        if not on_sky_data and instru == "MIRIMRS":
            Ss_data = np.loadtxt(f"{sim_data_path}/Systematics/{instru}/star_{T_star_sim}_mag7_J.txt", skiprows=1) # [J/s/m2/µm]
            Ss_data = Spectrum(wavelength=Ss_data[:, 0], flux=Ss_data[:, 1])                              # [J/s/m2/µm]
            Ss_data = Ss_data.evenly_spaced(lmin=0.98*wave_data[0], lmax=1.02*wave_data[-1])              # [J/s/m2/µm]
            Ss_data = Ss_data.degrade_resolution(wave_data, renorm=False, R_output=R_band, verbose=False) # [J/s/m2/µm]
            Ss_data = Ss_data.density_to_photons(config_data=config_data)                                 # [ph/bin/mn] (total ph over the FoV)
            Ss_data = Ss_data.flux * exposure_time_data                                                   # [ph/bin]    (total ph over the FoV)
        
        # Stellar flux in [ph/bin] from data cube (sum over spatial axes) and divide by trans
        else:
            Ss_data               = np.nansum(S_noiseless, axis=(1, 2)) / trans # [ph/bin]
            Ss_data[Ss_data == 0] = np.nan
        
        # Estimating the stellar modulation function
        Ms = S_noiseless / (trans*Ss_data)[:, None, None]
        
        # Saving
        hdr            = fits.Header()
        hdr['pxscale'] = pxscale
        if on_sky_data: # writing the data for systematics estimation purposes
            fits.writeto(f"{sim_data_path}/Systematics/{instru}/Ms_onsky_star_center_s3d_{band}.fits",   Ms,        overwrite=True, header=hdr)
            fits.writeto(f"{sim_data_path}/Systematics/{instru}/wave_onsky_star_center_s3d_{band}.fits", wave_data, overwrite=True)
        else:
            fits.writeto(f"{sim_data_path}/Systematics/{instru}/Ms_sim_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits",   Ms,        overwrite=True, header=hdr)
            fits.writeto(f"{sim_data_path}/Systematics/{instru}/wave_sim_star_center_T{T_star_sim}K_mag7_s3d_{band}_{correction}.fits", wave_data, overwrite=True)
    
    NbChannel, NbLine, NbColumn = Ms.shape
    y_center, x_center          = NbLine//2, NbColumn//2
    R_nyquist                   = get_resolution(wavelength=wave_data, func=np.nanmean)
    
    # Get transmission on the cube wavelength data grid in [e-/ph]
    trans = get_transmission(instru=instru, wave_band=wave_data, band=band, tellurics=tellurics, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, fill_value=np.nan)
    
    # Current stellar model on cube wavelength data grid grid in [ph/bin/mn] (total ph over the FoV)
    Ss = get_spectrum_band(spectrum_instru=star_spectrum_instru, wave_band=wave_data, R_output=R_band).flux # [ph/bin/mn] (total ph over the FoV)
    
    # Computing the noiseless data cube (without planet) in [e-/bin/px/mn]
    S_wo_planet = Ms * (trans*Ss)[:, None, None] # [e-/bin/px/mn]
    
    # Estimating the star flux from the data cube in [e-/bin/mn] (total e- over the FoV)
    trans_Ss                = np.nansum(S_wo_planet, (1, 2)) # [e-/bin/mn] (total e- over the FoV)
    trans_Ss[trans_Ss == 0] = np.nan                         # [e-/bin/mn] (total e- over the FoV)
        
    # --------------------
    # 2) Stellar filtering
    # --------------------
    S_res_wo_planet, _ = get_S_res(wave=wave_data, cube=S_wo_planet, Rc=Rc, filter_type=filter_type, outliers=on_sky_data, sigma_outliers=sigma_outliers, renorm_cube_res=False, only_high_pass=False, trans_Ss=trans_Ss, debug=False)

    # -----------------------------------
    # 3) Decide whether PCA is beneficial
    # -----------------------------------
    if PCA:
        if separation_planet is not None and Rc == 100: # For all FastYield calculations (with Rc = 100)
            T_star_t_syst_arr   = np.array([3000, 6000, 9000])
            T_star_t_syst       = T_star_t_syst_arr[np.abs(star_spectrum_instru.T-T_star_t_syst_arr).argmin()] 
            T_planet_t_syst_arr = np.arange(500, 3000+100, 100)
            T_planet_t_syst     = T_planet_t_syst_arr[np.abs(planet_spectrum_instru.T-T_planet_t_syst_arr).argmin()] 
            t_syst              = fits.getdata(f"{sim_data_path}/Systematics/{instru}/t_syst/t_syst_{instru}_{band}_Tp{T_planet_t_syst}K_Ts{T_star_t_syst}K_Rc{Rc}.fits")
            separation_t_syst   = fits.getdata(f"{sim_data_path}/Systematics/{instru}/t_syst/separation_{instru}_{band}_Tp{T_planet_t_syst}K_Ts{T_star_t_syst}K_Rc{Rc}.fits")
            mag_star_t_syst     = fits.getdata(f"{sim_data_path}/Systematics/{instru}/t_syst/mag_star_{instru}_{band}_Tp{T_planet_t_syst}K_Ts{T_star_t_syst}K_Rc{Rc}.fits")
            mag_star            = np.clip(mag_star, np.nanmin(mag_star_t_syst), np.nanmax(mag_star_t_syst))
            separation_planet   = np.clip(separation_planet, np.nanmin(separation_t_syst), np.nanmax(separation_t_syst))
            interp_func = RegularGridInterpolator((mag_star_t_syst, separation_t_syst), t_syst, method='linear')
            point       = np.array([[mag_star, separation_planet]])
            t_syst      = interp_func(point)[0]
            # If the systematics are not dominating for the given exoposure time, PCA is not necessary (EXCEPTION IS MADE FOR SIMUMATIONS)
            if 120 > t_syst or (target_name is not None and "sim" in target_name): # 120 mn (typical exposure_time) > t_syst => systematics are dominating
                PCA_calc = True
            if exposure_time < 120 and exposure_time < t_syst and 120 > t_syst: # 2 hours (~ order of magnitude of the observations generally made) is the exposure time considered in FastYield calculations
                PCA_calc    = False    
                PCA_verbose = f" PCA is not considered with t_exp = {exposure_time}mn but was considered in FastYield calculations with t_exp = 120mn."
        # Default to True if Rc!=100 (table is missing) or if separation_planet is unknown
        else:
            PCA_calc = True
    
    # ---------------------------------------------------
    # 4) PCA branch (with optional fake-planet injection)
    # ---------------------------------------------------
    if PCA_calc: 
        PCA_verbose = f" PCA, with {N_PCA} principal components subtracted, is included in the FastCurves estimations as a technique for systematic noise removal"
                
        Sp           = get_spectrum_band(spectrum_instru=planet_spectrum_instru, wave_band=wave_data, R_output=R_band).flux # [ph/bin/mn] (total ph over the FoV)
        Sp_HF, Sp_LF = filtered_flux(Sp, R=R_nyquist, Rc=Rc, filter_type=filter_type) # [ph/bin/mn] (total ph over the FoV)
        Ss_HF, Ss_LF = filtered_flux(Ss, R=R_nyquist, Rc=Rc, filter_type=filter_type) # [ph/bin/mn] (total ph over the FoV)
        
        # Planet fake injection (if the sep and the mag are known and the planet is inside the FoV) in order to estimate components that would be estimated on real data and thus estimating the systematic noise and signal reduction 
        if (mag_planet is not None) and (separation_planet is not None) and (separation_planet < FOV / 2):
            
            # Planet fake injection
            S_planet            = S_wo_planet * Sp[:, None, None] / Ss[:, None, None]
            dy                  = int(round( separation_planet/pxscale ))
            y0                  = int(round( (NbLine-1)/2 + dy ))
            x0                  = int(round( (NbColumn-1)/2 ))
            S_planet            = np.roll(S_planet, dy, 1)
            S_planet[:, :dy, :] = np.nan  
            S                   = S_wo_planet + np.nan_to_num(S_planet)
            
            # Stellar filtering + PCA in [e-/bin/px/mn]
            S_res, _       = get_S_res(wave=wave_data, cube=S, Rc=Rc, filter_type=filter_type, outliers=on_sky_data, sigma_outliers=sigma_outliers, renorm_cube_res=False, only_high_pass=False, trans_Ss=trans_Ss, debug=False) # stellar subtracted data with the fake planet injected
            S_res_pca, pca = PCA_subtraction(S_res=S_res, N_PCA=N_PCA, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=PCA_mask, PCA_plots=False, wave=wave_data, R=R_band) # apply PCA to it
            
            # Masking the planet (if beyond FWHM)
            S_res_wo_planet     = np.copy(S_res)
            S_res_wo_planet_pca = np.copy(S_res_pca)
            if separation_planet > size_core*pxscale: # if the planet is further than a FWHM from the star (otherwise hiding the planet will also hide the region used to estimate the noise)
                planet_mask                            = circular_mask(y0, x0, r=size_core, size=(NbLine, NbColumn))
                S_res_wo_planet[:, planet_mask==1]     = np.nan
                S_res_wo_planet_pca[:, planet_mask==1] = np.nan
            
            # BOX convolutions [e-/bin/FWHM/mn]
            S_res               = box_convolution(data=S_res,               size_core=size_core)
            S_res_pca           = box_convolution(data=S_res_pca,           size_core=size_core)
            S_res_wo_planet     = box_convolution(data=S_res_wo_planet,     size_core=size_core)
            S_res_wo_planet_pca = box_convolution(data=S_res_wo_planet_pca, size_core=size_core)

            # CCF computations [e-/FWHM/mn]
            CCF, _               = get_CCF_2D_rv(instru=instru, S_res=S_res,               trans_Ss=trans_Ss, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave_data, trans=trans, R=R_band, Rc=Rc, filter_type=filter_type, template_wo_shift=planet_spectrum_instru, pca=None)
            CCF_wo_planet, _     = get_CCF_2D_rv(instru=instru, S_res=S_res_wo_planet,     trans_Ss=trans_Ss, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave_data, trans=trans, R=R_band, Rc=Rc, filter_type=filter_type, template_wo_shift=planet_spectrum_instru, pca=None)
            CCF_pca, _           = get_CCF_2D_rv(instru=instru, S_res=S_res_pca,           trans_Ss=trans_Ss, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave_data, trans=trans, R=R_band, Rc=Rc, filter_type=filter_type, template_wo_shift=planet_spectrum_instru, pca=pca)
            CCF_wo_planet_pca, _ = get_CCF_2D_rv(instru=instru, S_res=S_res_wo_planet_pca, trans_Ss=trans_Ss, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave_data, trans=trans, R=R_band, Rc=Rc, filter_type=filter_type, template_wo_shift=planet_spectrum_instru, pca=pca)
            
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
            
            # PCA in [e-/bin/px/mn]
            S_res_wo_planet_pca, pca = PCA_subtraction(S_res=S_res_wo_planet, N_PCA=N_PCA, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=False, PCA_plots=False, wave=wave_data, R=R_band)
            
            # BOX convolutions in [e-/bin/FWHM/mn]
            S_res_wo_planet_pca = box_convolution(data=S_res_wo_planet_pca, size_core=size_core)

            # CCF computations in [e-/FWHM/mn]
            CCF_wo_planet_pca, _ = get_CCF_2D_rv(instru=instru, S_res=S_res_wo_planet_pca, trans_Ss=trans_Ss, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave_data, trans=trans, R=R_band, Rc=Rc, filter_type=filter_type, template_wo_shift=planet_spectrum_instru, pca=pca)
            
            # No signal loss due to PCA
            M_pca = 1
        
        # Analytical PCA signal-loss proxy: another way to estimate the signal loss due to the PCA: substract the PCA components to the planetary spectrum
        if Rc is None or Rc == 0:
            d = trans*Sp_HF
        else:
            d = trans*Sp_HF - trans*Ss_HF*Sp_LF/Ss_LF # Spectrum at the planet's location: see Eq.(18) of Martos et al. 2025
        template     = trans*Sp_HF
        template    /= np.sqrt(np.nansum(template**2))
        d_sub        = np.copy(d)
        template_sub = np.copy(template)
        for nk in range(N_PCA): # Subtracting the components 
            d_sub        -= np.nan_to_num(np.nansum(d*pca.components_[nk])*pca.components_[nk])
            template_sub -= np.nan_to_num(np.nansum(template*pca.components_[nk])*pca.components_[nk])
        m_pca = abs(np.nansum(d_sub*template_sub) / np.nansum(d*template)) # Analytical signal loss
        M_pca = min(M_pca, m_pca, 1) # Taking the minimal value between the two methods (and knowing that the signal loss ratio must be lower than 1)
        
        # Assigning the CCF that will be used for estimation of the systematic noise level
        CCF_wo_planet = CCF_wo_planet_pca # [e-/FWHM/mn]
    
    # -----------------------
    # 5) No-PCA branch
    # -----------------------
    else:
        
        # No signal loss
        if PCA:
            PCA_verbose = "PCA requested but skipped (systematics are not expected to dominate for this case)."
        M_pca = 1.
        
        # BOX convolutions
        S_res_wo_planet = box_convolution(data=S_res_wo_planet, size_core=size_core)
                
        # CCF computations
        CCF_wo_planet, template = get_CCF_2D_rv(instru=instru, S_res=S_res_wo_planet, trans_Ss=trans_Ss, T=None, lg=None, rv_arr=0, vsini=None, model=None, wave=wave_data, trans=trans, R=R_band, Rc=Rc, filter_type=filter_type, template_wo_shift=planet_spectrum_instru, pca=pca)

    # --------------------------------------------------------
    # 6) Build radial profile of systematic noise from the CCF
    # --------------------------------------------------------
    max_r              = int(round((FOV / 2) / pxscale)) + 1
    separation         = np.zeros((max_r)) + np.nan # [sep_unit]
    sigma_syst_prime_2 = np.zeros((max_r)) + np.nan # [e-/FWHM/mn]**2
    for r in range(len(separation)):
        r_int         = max(1, r - 1) if r > 1 else r
        r_ext         = r + 1 if r==0 else r
        separation[r] = (r_int + r_ext)/2 * pxscale
        amask         = annular_mask(r_int, r_ext, size=(NbLine, NbColumn)) == 1 # ring at separation r
        ccf           = CCF_wo_planet[amask] # [e-/FWHM/mn]
        if np.isfinite(ccf).sum() > 1:
            sigma_syst_prime_2[r] = np.nanvar(ccf) # Systematic noise at separation r in [e-/FWHM/mn]**2

    # -----------------------
    # 7) Mp (planet modulation proxy on FWHM box): mean within FWHM box around the star core, normalized by star flux
    # -----------------------
    core  = np.nan_to_num(S_wo_planet[:, y_center - size_core//2 : y_center + size_core//2 + 1, x_center - size_core//2 : x_center + size_core//2 + 1])
    Mp_0  = np.nanmean(core, axis=(1, 2)) / (trans*Ss)
    Mp_0 /= np.nanmean(Mp_0)
    Mp    = np.interp(wave_band, wave_data, Mp_0, left=np.nan, right=np.nan)
    Mp    = fill_nan_linear(x=wave_band, y=Mp) # NaN gaps are filled with linear interpolation (without extrapolation)
    
    return sigma_syst_prime_2, separation, Mp, M_pca, wave_data, pca, PCA_verbose






#######################################################################################################################
####################### Modeling and parametrizing speckle and systematic residual modulations: #######################
#######################################################################################################################

def get_gaussian_C(wave, Lcorr):
    """
    Build a Gaussian spectral covariance matrix.

    The covariance kernel is modeled as

        C_ij = exp[-0.5 * ((lambda_i - lambda_j) / L_ij)^2],

    where the effective correlation length L_ij is defined from 'Lcorr'.

    Parameters
    ----------
    wave : array_like
        One-dimensional wavelength grid.
    Lcorr : float or array_like
        Spectral correlation length. If scalar, a single constant correlation
        length is used for all wavelength pairs. If one-dimensional, it is
        interpreted as a wavelength-dependent correlation length Lcorr(lambda),
        and the pairwise correlation length is taken as the geometric mean

            L_ij = sqrt(Lcorr_i * Lcorr_j).

    Returns
    -------
    C : ndarray
        Two-dimensional Gaussian covariance matrix of shape (n_wave, n_wave).

    Raises
    ------
    ValueError
        If 'Lcorr' is neither a scalar nor a one-dimensional array.

    Notes
    -----
    A scalar 'Lcorr' corresponds to a stationary kernel, while a one-dimensional
    'Lcorr' allows for a non-stationary covariance model whose width varies
    across the spectral domain.
    """
    dw = wave[:, None] - wave[None, :]

    if np.ndim(Lcorr) == 0:
        return np.exp(-0.5 * (dw / Lcorr) ** 2)

    if np.ndim(Lcorr) == 1:
        Lcorr = np.asarray(Lcorr)
        Lij   = np.sqrt(Lcorr[:, None] * Lcorr[None, :])
        return np.exp(-0.5 * (dw / Lij) ** 2)

    raise ValueError("Lcorr must be either a scalar or a 1D array.")



def get_Pform(h, C):
    """
    Evaluate the quadratic form P = h^T C h (projected variance in the CCF for unitary modulation) while ignoring invalid entries.

    This function applies a common validity mask to the vector 'h' and to the
    covariance matrix 'C', keeping only indices for which:
      - h is finite,
      - the corresponding row of C is fully finite,
      - the corresponding column of C is fully finite.

    Parameters
    ----------
    h : array_like
        One-dimensional weight vector.
    C : array_like
        Two-dimensional covariance matrix.

    Returns
    -------
    P : float
        Value of the quadratic form computed on the valid subset.

    Notes
    -----
    This masking strategy is preferable to a simple 'nansum' approach because
    it removes invalid spectral channels consistently from both the vector and
    the covariance matrix.
    """
    h = np.asarray(h)
    C = np.asarray(C)

    valid = np.isfinite(h)
    valid &= np.all(np.isfinite(C), axis=0)
    valid &= np.all(np.isfinite(C), axis=1)

    wv = h[valid]
    Kv = C[np.ix_(valid, valid)]
    P  = np.einsum("i,ij,j->", wv, Kv, wv, optimize=True)
    
    return P



def get_Lcorr_aliasing_spectral(wave):
    """
    Estimate the spectral correlation length associated with spectral aliasing.

    The model assumes that the characteristic spectral width of the residual
    modulation is set by the native sampling step of the wavelength grid.
    This characteristic width is then converted into the Gaussian covariance
    length used in

        C_ij = exp[-0.5 * ((lambda_i - lambda_j) / Lcorr)^2].

    Parameters
    ----------
    wave : array_like
        One-dimensional wavelength grid.

    Returns
    -------
    L_al_spec : ndarray
        Wavelength-dependent spectral aliasing correlation length.
    """
    L_al_spec = np.gradient(wave) / (2.0 * np.sqrt(np.log(2.0)))
    return L_al_spec



def get_Lcorr_speckle(wave, separation, D, sep_unit, eps=1e-15):
    """
    Estimate the spectral correlation length associated with chromatic speckles.

    This function uses the standard scaling derived from the radial chromatic
    motion of speckles in a diffraction-limited PSF. The adopted expression is

        L_speck = [1.029 / (2 * sqrt(ln 2))] * wave**2 / (D * rho),

    where:
      - 'wave' is the wavelength,
      - 'D' is the telescope diameter,
      - 'rho' is the angular separation in radians,
      - the factor 1.029 corresponds to the Airy-pattern FWHM in units of
        lambda / D.

    Parameters
    ----------
    wave : array_like or float
        Wavelength grid. Must be expressed in the same length units as 'D'.
    separation : array_like or float
        Angular separation from the star. Units are set by 'sep_unit'.
    D : float
        Telescope diameter, in the same length units as 'wave'.
    sep_unit : {"mas", "arcsec"}
        Unit of 'separation'.
    eps : float, optional
        Minimum separation value used to avoid division by zero. Default is
        1e-15.

    Returns
    -------
    L_speck : ndarray or float
        Spectral correlation length of chromatic speckles, with the broadcasted
        shape of 'wave' and 'separation'.

    Notes
    -----
    The returned quantity has the same length unit as 'wave'.
    """
    if sep_unit == "mas":
        separation_rad = separation / (rad2arcsec * 1000.0)  # [mas] -> [rad]
    else:
        separation_rad = separation / rad2arcsec  # [arcsec] -> [rad]

    rho = np.maximum(separation_rad, eps)
    L_speck = 1.029 / (2.0 * np.sqrt(np.log(2.0))) * wave**2 / (D * rho)
    return L_speck



def get_Lcorr_aliasing_spatial(wave, pxscale, separation, sep_unit, eps=1e-15):
    """
    Estimate the spectral correlation length associated with spatial aliasing.

    This function models the chromatic residuals induced by spatial sampling
    and cube reconstruction. The adopted scaling is

        L_al_spat = pxscale_rad * wave / rho / (2 * sqrt(ln 2)),

    where:
      - 'pxscale_rad' is the spatial sampling in radians,
      - 'wave' is the wavelength,
      - 'rho' is the angular separation in radians.

    Parameters
    ----------
    wave : array_like or float
        Wavelength grid.
    pxscale : float
        Spatial sampling element (pixel scale or effective spaxel scale).
        Units are set by 'sep_unit'.
    separation : array_like or float
        Angular separation from the star. Units are set by 'sep_unit'.
    sep_unit : {"mas", "arcsec"}
        Unit of both 'pxscale' and 'separation'.
    eps : float, optional
        Minimum separation value used to avoid division by zero. Default is
        1e-15.

    Returns
    -------
    L_al_spat : ndarray or float
        Spectral correlation length associated with spatial aliasing, with the
        broadcasted shape of 'wave' and 'separation'.

    Notes
    -----
    The returned quantity has the same length unit as 'wave'.
    """
    if sep_unit == "mas":
        separation_rad = separation / (rad2arcsec * 1000) # [mas] => [rad]
        pxscale_rad = pxscale       / (rad2arcsec * 1000) # [mas] => [rad]
    else:
        separation_rad = separation / rad2arcsec # [arcsec] => [rad]
        pxscale_rad = pxscale       / rad2arcsec # [arcsec] => [rad]

    rho = np.maximum(separation_rad, eps)
    L_al_spat = pxscale_rad * wave / rho / (2.0 * np.sqrt(np.log(2.0)))
    return L_al_spat



def compress_h_for_P(wave, h_DI, h_MM, D, separation_ref, sep_unit, oversample=3.0, max_merge=64):
    """
    Compress the spectral grid used for P by merging neighboring channels.

    The compression is chosen such that the coarse spectral step remains smaller
    than the speckle correlation length divided by 'oversample', evaluated at a
    reference separation. The weights are summed (not averaged), because P is
    a quadratic form over the spectral channels.

    Parameters
    ----------
    wave : array_like
        Native wavelength grid, shape (n_wave,).
    h_DI : array_like
        Spectral weights entering P_DI, shape (n_wave, n_sep).
    h_MM : array_like
        Spectral weights entering P_MM, shape (n_wave, n_sep).
    D : float
        Telescope diameter, in the same length units as 'wave'.
    separation_ref : float
        Reference separation at which the compression is defined. A conservative
        choice is the largest separation considered, or the planet separation if
        only one value is needed.
    sep_unit : {"mas", "arcsec"}
        Unit of 'separation_ref'.
    oversample : float, optional
        Number of coarse bins per minimum speckle correlation length. Typical
        values are 2 to 4. Default is 3.
    max_merge : int, optional
        Maximum number of native channels merged into one coarse channel.

    Returns
    -------
    wave_c : ndarray
        Compressed wavelength grid, shape (n_wave_c,).
    h_DI_c : ndarray
        Compressed DI weights, shape (n_wave_c, n_sep).
    h_MM_c : ndarray
        Compressed DI weights, shape (n_wave_c, n_sep).
    n_merge : int
        Number of native channels merged into one coarse channel.
    """
    wave = np.asarray(wave, dtype=np.float32)
    h_DI = np.asarray(h_DI, dtype=np.float32)
    h_MM = np.asarray(h_MM, dtype=np.float32)

    L_ref = get_Lcorr_speckle(wave=wave, separation=separation_ref, D=D, sep_unit=sep_unit)

    dlam_native = np.nanmedian(np.diff(wave))
    dlam_target = np.nanmin(L_ref) / oversample

    n_merge = int(np.floor(dlam_target / dlam_native))
    n_merge = max(1, min(max_merge, n_merge))

    if n_merge == 1:
        return (np.ascontiguousarray(wave, dtype=np.float32), np.ascontiguousarray(h_DI, dtype=np.float32), np.ascontiguousarray(h_MM, dtype=np.float32), 1)

    idx0   = np.arange(0, len(wave), n_merge)
    counts = np.diff(np.append(idx0, len(wave)))

    # mean wavelength in each macro-bin
    wave_c = np.add.reduceat(wave, idx0) / counts

    # IMPORTANT: sum of weights, not mean
    h_DI_c = np.add.reduceat(h_DI.astype(np.float32), idx0, axis=0)
    h_MM_c = np.add.reduceat(h_MM.astype(np.float32), idx0, axis=0)

    return (np.ascontiguousarray(wave_c, dtype=np.float32), np.ascontiguousarray(h_DI_c, dtype=np.float32), np.ascontiguousarray(h_MM_c, dtype=np.float32), n_merge)



@njit(cache=True, fastmath=True, parallel=True)
def compute_P_speck_numba(h, wave, separation_rad, D, n_sigma_cut=4.0, eps=1e-15):
    """
    Compute the speckle covariance quadratic form for all separations.

    This function evaluates, for each angular separation, the quadratic form

        P = h.T @ C_speck @ h,

    where C_speck is the Gaussian spectral covariance matrix associated with
    chromatic speckles.

    Parameters
    ----------
    h : ndarray
        Two-dimensional array of shape (n_lambda, n_sep) containing the
        spectral weights for each separation.
    wave : ndarray
        One-dimensional wavelength grid of shape (n_lambda,).
    separation_rad : ndarray
        One-dimensional array of angular separations in radians, shape (n_sep,).
    D : float
        Telescope diameter, in the same length unit as `wave`.
    n_sigma_cut : float, optional
        Truncation radius of the Gaussian kernel, expressed in units of the
        local correlation length. Pairwise contributions beyond this threshold
        are neglected. Default is 4.0.
    eps : float, optional
        Minimum allowed separation in radians, used to avoid division by zero.
        Default is 1e-15.

    Returns
    -------
    out : ndarray
        One-dimensional array of shape (n_sep,) containing the quadratic form
        evaluated at each separation.

    Notes
    -----
    The local speckle correlation length is modeled as

        L_speck = [1.029 / (2 * sqrt(ln 2))] * wave^2 / (D * rho),

    where rho is the angular separation in radians.

    The implementation exploits the symmetry of the covariance matrix and only
    computes the upper triangle. The diagonal contribution is added explicitly.
    """
    nlam, nsep = h.shape
    out        = np.empty(nsep, dtype=np.float32)

    coeff       = np.float32(1.029 / (2.0 * math.sqrt(math.log(2.0))) / D)
    eps         = np.float32(eps)
    n_sigma_cut = np.float32(n_sigma_cut)

    for k in prange(nsep):
        rho = separation_rad[k]
        if rho < eps:
            rho = eps

        q = np.float32(0.0)

        for i in range(nlam):
            wi = h[i, k]
            if not np.isfinite(wi) or wi == 0.0:
                continue

            Li = coeff * wave[i] * wave[i] / rho

            # Diagonal term: C_ii = 1.
            q += wi * wi

            for j in range(i + 1, nlam):
                wj = h[j, k]
                if not np.isfinite(wj) or wj == 0.0:
                    continue

                Lj  = coeff * wave[j] * wave[j] / rho
                Lij = math.sqrt(Li * Lj)
                dw  = wave[j] - wave[i]
                x   = dw / Lij

                if x > n_sigma_cut:
                    break

                kij = math.exp(-0.5 * x * x)
                q += 2.0 * wi * wj * kij

        out[k] = q

    return out



@njit(cache=True, fastmath=True, parallel=True)
def compute_P_al_spec_numba(h, wave, n_sigma_cut=4.0):
    """
    Compute the spectral-aliasing quadratic form for all separations.

    This function evaluates, for each separation, the quadratic form

        P = h.T @ C @ h,

    where C is the Gaussian covariance matrix associated with spectral
    aliasing. The spectral correlation length is estimated locally from the
    native wavelength sampling.

    Parameters
    ----------
    h : ndarray
        Two-dimensional array of shape (n_lambda, n_sep) containing the
        spectral weights for each separation.
    wave : ndarray
        One-dimensional wavelength grid of shape (n_lambda,).
    n_sigma_cut : float, optional
        Truncation radius of the Gaussian kernel, expressed in units of the
        local correlation length. Pairwise contributions beyond this threshold
        are neglected. Default is 4.0.

    Returns
    -------
    out : ndarray
        One-dimensional array of shape (n_sep,) containing the quadratic form
        evaluated at each separation.

    Notes
    -----
    The local spectral correlation length is approximated as

        L_i = d_lambda_i / (2 * sqrt(ln 2)),

    where d_lambda_i is the local wavelength sampling step.

    The implementation exploits the symmetry of the covariance matrix and only
    computes the upper triangle. The diagonal contribution is added explicitly.
    """
    nlam, nsep = h.shape
    out        = np.empty(nsep, dtype=np.float32)

    # Local spectral correlation length.
    L = np.empty(nlam, dtype=np.float32)
    for i in range(nlam):
        if i == 0:
            dlam = wave[1] - wave[0]
        elif i == nlam - 1:
            dlam = wave[nlam - 1] - wave[nlam - 2]
        else:
            dlam = 0.5 * (wave[i + 1] - wave[i - 1])

        L[i] = dlam / (2.0 * math.sqrt(math.log(2.0)))

    n_sigma_cut = np.float32(n_sigma_cut)

    for k in prange(nsep):
        q = np.float32(0.0)

        for i in range(nlam):
            wi = h[i, k]
            if not np.isfinite(wi) or wi == 0.0:
                continue

            # Diagonal term: C_ii = 1.
            q += wi * wi

            for j in range(i + 1, nlam):
                wj = h[j, k]
                if not np.isfinite(wj) or wj == 0.0:
                    continue

                Lij = math.sqrt(L[i] * L[j])
                dw  = wave[j] - wave[i]
                x   = dw / Lij

                if x > n_sigma_cut:
                    break

                kij = math.exp(-0.5 * x * x)
                q  += 2.0 * wi * wj * kij

        out[k] = q

    return out



@njit(cache=True, fastmath=True, parallel=True)
def compute_P_al_spat_numba(h, wave, separation_rad, pxscale_rad, n_sigma_cut=4.0, eps=1e-15):
    """
    Compute the spatial-aliasing covariance quadratic form for all separations.

    This function evaluates, for each angular separation, the quadratic form

        P = h.T @ C_al_spat @ h,

    where C_al_spat is the Gaussian spectral covariance matrix associated with
    spatial aliasing.

    Parameters
    ----------
    h : ndarray
        Two-dimensional array of shape (n_lambda, n_sep) containing the
        spectral weights for each separation.
    wave : ndarray
        One-dimensional wavelength grid of shape (n_lambda,).
    separation_rad : ndarray
        One-dimensional array of angular separations in radians, shape (n_sep,).
    pxscale_rad : float
        Spatial sampling scale in radians.
    n_sigma_cut : float, optional
        Truncation radius of the Gaussian kernel, expressed in units of the
        local correlation length. Pairwise contributions beyond this threshold
        are neglected. Default is 4.0.
    eps : float, optional
        Minimum allowed separation in radians, used to avoid division by zero.
        Default is 1e-15.

    Returns
    -------
    out : ndarray
        One-dimensional array of shape (n_sep,) containing the quadratic form
        evaluated at each separation.

    Notes
    -----
    The local spatial-aliasing correlation length is modeled as

        L_al_spat = pxscale_rad * wave / [2 * sqrt(ln 2) * rho],

    where rho is the angular separation in radians.

    The implementation exploits the symmetry of the covariance matrix and only
    computes the upper triangle. The diagonal contribution is added explicitly.
    """
    nlam, nsep = h.shape
    out = np.empty(nsep, dtype=np.float32)

    coeff = np.float32(pxscale_rad / (2.0 * math.sqrt(math.log(2.0))))
    eps = np.float32(eps)
    n_sigma_cut = np.float32(n_sigma_cut)

    for k in prange(nsep):
        rho = separation_rad[k]
        if rho < eps:
            rho = eps

        q = np.float32(0.0)

        for i in range(nlam):
            wi = h[i, k]
            if not np.isfinite(wi) or wi == 0.0:
                continue

            Li = coeff * wave[i] / rho

            # Diagonal term: C_ii = 1.
            q += wi * wi

            for j in range(i + 1, nlam):
                wj = h[j, k]
                if not np.isfinite(wj) or wj == 0.0:
                    continue

                Lj  = coeff * wave[j] / rho
                Lij = math.sqrt(Li * Lj)
                dw  = wave[j] - wave[i]
                x   = dw / Lij

                if x > n_sigma_cut:
                    break

                kij = math.exp(-0.5 * x * x)
                q  += 2.0 * wi * wj * kij

        out[k] = q

    return out








#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_dark_noise_budget(instru, noise_level=None):
    RON_min = 0.5 # e-/px/DIT (achieved laboratory limit)
    config_data  = get_config_data(instru)
    min_DIT      = config_data["detector"]["minDIT"]       # minimal integration time (in mn)
    max_DIT      = config_data["detector"]["maxDIT"]       # maximal integration time (in mn)
    RON          = config_data["detector"]["RON"]          # read out noise (in e-/px/DIT)
    dark_current = config_data["detector"]["dark_current"] # dark current (in e-/px/s)
    DIT       = np.logspace(np.log10(min_DIT), np.log10(200*max_DIT), 100)
    sigma_dc  = np.zeros_like(DIT)
    sigma_ron = np.zeros_like(DIT)
    for i in range(len(DIT)):
        dit = DIT[i]
        sigma_dc[i] = np.sqrt(dark_current * dit * 60) # e-/px/DOT
        nb_min_DIT = 1 # "Up the ramp" reading mode: the pose is sequenced in several non-destructive readings to reduce reading noise (see https://en.wikipedia.org/wiki/Signal_averaging).
        if dit > nb_min_DIT*min_DIT: # choose 4 min_DIT because if intermittent readings are too short, the detector will heat up too quickly => + dark current
            N_i          = dit / (nb_min_DIT*min_DIT) # number of intermittent readings
            sigma_ron[i] = RON / np.sqrt(N_i)         # effective read out noise (in e-/px/DIT)
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
                x0, x1  = DIT[i], DIT[i+1]
                frac    = (noise_level - y0) / (y1 - y0)  # interpolation linéaire
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