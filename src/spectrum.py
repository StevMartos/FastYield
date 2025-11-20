from src.utils import *
from src.utils import _load_tell_trans

path_file = os.path.dirname(__file__)
load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")
vega_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/star_spectrum/VEGA_Fnu.fits")



#######################################################################################################################
############################################# Utils: #########################################################
#######################################################################################################################

def estimate_resolution(wavelength):
    """
    Estimate the spectral resolution of a spectrum, assuming Nyquist sampling.

    Parameters
    ----------
    wavelength : array_like
        Array of wavelength values [µm].

    Returns
    -------
    float
        Estimated resolving power R (dimensionless).
    """
    dl          = np.gradient(wavelength)           # Wavelength spacing Δλ
    dl[dl == 0] = np.nan
    R           = np.nanmedian(wavelength / (2*dl)) # Resolving power, factor 2 for Nyquist sampling (Shannon)
    return R



def downbin_spec(specHR, sigmaHR, weightHR, lamHR, lamLR, dlam):
    """
    Re-bin a high-resolution spectrum to a lower-resolution grid,
    using top-hat integration (mean within wavelength bins),
    and propagate the associated 1-sigma uncertainties.

    This is equivalent to a rectangular (top-hat) convolution, and assumes
    that the high-resolution values represent flux per bin (not per unit wavelength).
    The resulting spectrum is averaged within each low-res bin, and errors are
    propagated assuming independent Gaussian noise.

    Parameters
    ----------
    specHR : array-like
        High-resolution spectrum (flux per bin).
    sigmaHR : array-like
        1-sigma uncertainty on 'specHR' (same length and units).
    weightHR : array-like
        Weight function on 'specHR' (same length and units). NEEDS TO BE A QUALITY FUNCTION
    lamHR : array-like
        Wavelength grid of the high-resolution spectrum.
    lamLR : array-like
        Central wavelength grid of the target low-resolution spectrum.
    dlam : array-like
        Bin widths for the low-resolution grid (same length as lamLR).

    Returns
    -------
    specLR : numpy.ndarray
        Binned low-resolution spectrum (mean value in each bin).
    sigmaLR : numpy.ndarray
        Propagated 1-sigma uncertainties in each bin.
    """
    
    # If input wavelength grid is decreasing, reverse all relevant arrays
    if lamHR[0] > lamHR[1]:
        lamHR   = lamHR[::-1]
        specHR  = specHR[::-1]
        if sigmaHR is not None:
            sigmaHR = sigmaHR[::-1]
        if weightHR is not None:
            weightHR = weightHR[::-1]
    if lamLR[0] > lamLR[1]:
        lamLR = lamLR[::-1]
        dlam  = dlam[::-1]

    # Compute edges of the low-resolution bins
    # Bin_i = [lamLR[i] - dlam[i]/2, lamLR[i] + dlam[i]/2]
    LRedges = np.hstack([lamLR - 0.5 * dlam, lamLR[-1] + 0.5 * dlam[-1]])

    # --- NaN-aware masks
    mF = np.isfinite(specHR)

    # Re-bin the high-res spectrum using the mean in each bin:
    #     F_i = (1 / N_i) * sum_j f_j   for j in bin_i
    specLR = binned_statistic(lamHR[mF], specHR[mF], statistic="mean", bins=LRedges)[0]
    
    if sigmaHR is not None:
        # Propagate uncertainties assuming Gaussian independent errors:
        #     Var(F_i) = (1 / N_i^2) * sum_j sigma_j^2   for j in bin_i
        #     σ(F_i)   = sqrt( Var(F_i) ) = sqrt( sum_j sigma_j^2 ) / N_i
        # NaN-aware variance sum and counts (only finite contributors)
        mS = mF & np.isfinite(sigmaHR)
        var_sum = binned_statistic(lamHR[mS], (sigmaHR[mS]**2), statistic="sum", bins=LRedges)[0]
        count   = binned_statistic(lamHR[mF],  np.ones(mF.sum()), statistic="sum", bins=LRedges)[0]
        sigmaLR = np.divide(np.sqrt(var_sum), count, out=np.full_like(var_sum, np.nan), where=(count > 0))
    else:
        sigmaLR = None
        
    if weightHR is not None:
        mW       = np.isfinite(weightHR)
        weightLR = binned_statistic(lamHR[mW], weightHR[mW], statistic="mean", bins=LRedges)[0]
    else:
        weightLR = None
    
    specLR[specLR==0] = np.nan
    if sigmaHR is not None:
        sigmaLR[sigmaLR==0] = np.nan
    if weightHR is not None:
        weightLR[weightLR==0] = np.nan
    
    return specLR, sigmaLR, weightLR



def rebin_spec(specHR, sigmaHR, weightHR, lamHR, lamLR, dlam):
    """
    Flux-conserving rebinning from an irregular high-resolution grid to a target grid,
    with *measurement* uncertainty propagation via the α² rule.

    Overview
    --------
    Each low-resolution (LR) bin i gathers flux from high-resolution (HR) bins j
    in proportion to their geometric overlap. Let α_ij be the overlap fraction
    of HR bin j contributing to LR bin i. Then
        F_i     = Σ_j α_ij · F_j                          (flux-conserving)
        Var_i   = Σ_j α_ij² · Var_j                       (linear estimator propagation)
    where (F_j, Var_j) are the HR flux and its measurement variance per bin.
    A length-weighted quality/coverage indicator can also be propagated.

    Parameters
    ----------
    specHR : (N_hr,) array_like of float
        HR flux per native bin [e⁻/bin].
    sigmaHR : (N_hr,) array_like of float or None
        HR 1σ measurement uncertainties [e⁻/bin]. If None, LR uncertainties are not returned.
    weightHR : (N_hr,) array_like of float in [0, 1] or None
        HR per-bin quality/coverage. If provided, LR quality is the length-weighted mean
        of contributing HR values in each LR bin.
    lamHR : (N_hr,) array_like of float
        HR wavelength bin *centers* (monotonic, not necessarily uniform) [nm].
    lamLR : (N_lr,) array_like of float
        LR target wavelength bin centers (monotonic) [nm].
    dlam : (N_lr,) array_like of float
        LR target bin widths (same length as 'lamLR') [nm].

    Returns
    -------
    specLR : (N_lr,) ndarray of float
        Rebinned LR flux per bin [e⁻/bin].
    sigmaLR : (N_lr,) ndarray of float or None
        Rebinned LR 1σ measurement uncertainties [e⁻/bin], or None if 'sigmaHR' is None.
    weightLR : (N_lr,) ndarray of float or None
        LR length-weighted quality/coverage in [0, 1], or None if 'weightHR' is None.
    """
    # -- Convert to contiguous 1D float arrays
    specHR  = np.asarray(specHR,  dtype=float).ravel()
    lamHR   = np.asarray(lamHR,   dtype=float).ravel()
    lamLR   = np.asarray(lamLR,   dtype=float).ravel()
    dlam    = np.asarray(dlam,    dtype=float).ravel()
    sigmaHR = None if sigmaHR is None else np.asarray(sigmaHR, dtype=float).ravel()
    weightHR= None if weightHR is None else np.asarray(weightHR, dtype=float).ravel()

    # -- Basic shape checks
    n_hr = lamHR.size
    n_lr = lamLR.size
    if specHR.size != n_hr:
        raise ValueError("specHR and lamHR must have the same length.")
    if sigmaHR is not None and sigmaHR.size != n_hr:
        raise ValueError("sigmaHR and lamHR must have the same length.")
    if weightHR is not None and weightHR.size != n_hr:
        raise ValueError("weightHR and lamHR must have the same length.")
    if dlam.size != n_lr:
        raise ValueError("dlam and lamLR must have the same length.")
    if n_hr == 0 or n_lr == 0:
        # Trivial case
        return (np.zeros(n_lr, float),
                None if sigmaHR is None else np.full(n_lr, np.nan, float),
                None if weightHR is None else np.full(n_lr, np.nan, float))

    # -- Ensure increasing wavelength order (HR and LR)
    def ensure_increasing(x, *ys):
        if x.size >= 2 and x[0] > x[1]:
            x = x[::-1]
            ys = tuple(None if y is None else y[::-1] for y in ys)
        return (x,) + ys

    lamHR, specHR, sigmaHR, weightHR = ensure_increasing(lamHR, specHR, sigmaHR, weightHR)
    lamLR, dlam                      = ensure_increasing(lamLR, dlam)

    # -- Build edges from centers:
    # HR edges from midpoints (robust for irregular grids)
    hr_edges = np.empty(n_hr + 1, dtype=float)
    if n_hr == 1:
        # Single-bin fallback: assume width from LR scale if available, else zero
        est = dlam.mean() if n_lr > 0 else 0.0
        hr_edges[0] = lamHR[0] - 0.5 * est
        hr_edges[1] = lamHR[0] + 0.5 * est
    else:
        mid = 0.5 * (lamHR[1:] + lamHR[:-1])
        hr_edges[1:-1] = mid
        hr_edges[0]    = lamHR[0]  - 0.5 * (lamHR[1] - lamHR[0])
        hr_edges[-1]   = lamHR[-1] + 0.5 * (lamHR[-1] - lamHR[-2])

    lr_edges = np.empty(n_lr + 1, dtype=float)
    lr_edges[:-1] = lamLR - 0.5 * dlam
    lr_edges[-1]  = lamLR[-1] + 0.5 * dlam[-1]

    # -- Outputs
    specLR        = np.zeros(n_lr, dtype=float)
    sigmaLR       = None if sigmaHR is None else np.full(n_lr, np.nan, dtype=float)
    weightLR      = None if weightHR is None else np.full(n_lr, np.nan, dtype=float)
    scale_poisson = np.full(n_lr, np.nan, dtype=float)

    # -- Sliding pointer over HR bins
    j = 0
    for i in range(n_lr):
        A, B = lr_edges[i], lr_edges[i + 1]
        if not np.isfinite(A) or not np.isfinite(B) or B <= A:
            continue
        # Advance j past HR bins that end before A
        while j < n_hr and hr_edges[j + 1] <= A:
            j += 1

        Fi = 0.0      # Σ α F
        Vi = 0.0      # Σ α^2 σ^2
        Wi = 0.0      # length-weighted quality
        A1 = 0.0      # Σ α   (for scale_poisson) -- NaN-aware (only finite specHR)
        A2 = 0.0      # Σ α^2 (for scale_poisson) -- NaN-aware
        
        jj = j
        while jj < n_hr and hr_edges[jj] < B:
            left  = max(A, hr_edges[jj])
            right = min(B, hr_edges[jj + 1])
            overlap = right - left
            if overlap > 0.0:
                width_hr = hr_edges[jj + 1] - hr_edges[jj]
                if width_hr > 0 and np.isfinite(width_hr):
                    alpha = overlap / width_hr

                    # Flux (α) -- NaN-aware
                    if np.isfinite(specHR[jj]):
                        Fi += alpha * specHR[jj]
                        # Diagnostics for R and R_flat (valid-only)
                        A1  += alpha
                        A2  += alpha**2

                    # Variance (α^2) -- NaN-aware
                    if sigmaHR is not None and np.isfinite(sigmaHR[jj]) and np.isfinite(specHR[jj]):
                        Vi += alpha**2 * sigmaHR[jj]**2

                    # Quality (length-weighted) -- NaN-aware on weight only
                    if weightHR is not None and np.isfinite(weightHR[jj]):
                        Wi += overlap * weightHR[jj]
            jj += 1

        specLR[i] = Fi
        if sigmaHR is not None:
            sigmaLR[i] = np.sqrt(Vi) if Vi > 0.0 else (0.0 if Fi == 0.0 else np.nan)

        if weightHR is not None:
            width_lr    = B - A
            weightLR[i] = (Wi / width_lr) if width_lr > 0.0 else np.nan

        # --- scale_poisson_i diagnostics (photon noise gain scale factor) -- NaN-aware
        scale_poisson[i] = np.sqrt(A2 / A1) if A1 > 0 else np.nan  # see original comment
        
    specLR[specLR==0] = np.nan
    if sigmaHR is not None:
        sigmaLR[sigmaLR==0] = np.nan
    if weightHR is not None:
        weightLR[weightLR==0] = np.nan
    
    return specLR, sigmaLR, weightLR, scale_poisson



def interpolate_flux_with_error(wave, flux, sigma, weight, wave_new):
    """
    Vectorized linear interpolation of a spectrum with proper error propagation.

    Parameters
    ----------
    wave : np.ndarray
        Original wavelength axis (must be strictly increasing).
    flux : np.ndarray
        Flux values at each wavelength (e.g., in electrons).
    sigma : np.ndarray
        1-sigma uncertainty on the flux at each wavelength.
    weight : np.ndarray
        weight function at each wavelength.
    wave_new : np.ndarray
        New wavelength axis to interpolate onto.

    Returns
    -------
    flux_new : np.ndarray
        Interpolated flux at wave_new.
    sigma_new : np.ndarray
        Properly propagated uncertainty at wave_new.
    weight_new : np.ndarray
        Properly propagated weight function at wave_new.
    """
    # Mask for valid interpolation range
    valid = (wave_new >= wave[0]) & (wave_new <= wave[-1])

    # Allocate output arrays
    flux_new          = np.full_like(wave_new, np.nan, dtype=float)
    sigma_new         = np.full_like(wave_new, np.nan, dtype=float)
    weight_new        = np.full_like(wave_new, np.nan, dtype=float)
    scale_poisson_new = np.full_like(wave_new, np.nan, dtype=float)

    # For valid points only
    wave_valid = wave_new[valid]

    # Find indices k such that wave[k] <= wave_new[i] < wave[k+1]
    indices = np.searchsorted(wave, wave_valid) - 1
    indices = np.clip(indices, 0, len(wave) - 2)

    # Compute weights
    denom = wave[indices+1]  - wave[indices]
    w0    = (wave[indices+1] - wave_valid)    / denom
    w1    = (wave_valid      - wave[indices]) / denom

    # Interpolate flux and propagate error
    if flux is not None:
        flux_new[valid] = w0 * flux[indices] + w1 * flux[indices+1]
    if sigma is not None:
        sigma_new[valid] = np.sqrt(w0**2 * sigma[indices]**2 + w1**2 * sigma[indices+1]**2)
    if weight is not None:
        weight_new[valid] = w0 * weight[indices] + w1 * weight[indices+1]
    
    # --- scale_poisson_i diagnostics (photon noise gain scale factor ---
    scale_poisson_new[valid] = np.sqrt(w0**2 + w1**2) # Constant-flux approximation and assuming photon noise: scale_poisson_new = sigma_new / sqrt(flux_new) = sqrt( w_0**2 * sigma_0**2 + w_1**2 * sigma_1**2 /  w_0 * flux_0 + w_1 * flux_1 ) = sqrt( w_0**2 * flux_0 + w_1**2 * flux_1 /  w_0 * flux_0 + w_1 * flux_1 ) = sqrt( w_0**2 + w_1**2 ) (w_0 + w_1 = 1 by definition)
    
    return flux_new, sigma_new, weight_new, scale_poisson_new



@lru_cache(maxsize=50)
def _fft_filter_response(N, R, Rc, filter_type):
    """
    Cached frequency response of the (1D) low-pass/high-pass pair on the full FFT grid.

    Parameters
    ----------
    N : int
        Number of samples.
    R : float
        Effective resolving power (Nyquist def.).
    Rc : float or None
        Cut-off resolution. If None, LF=1 and HF=0.
    filter_type : {'gaussian_fast', 'gaussian_true', 'step', 'smoothstep'}

    Returns
    -------
    H_HF, H_LF : (N,) complex128 arrays
        High-pass and low-pass transfer functions (complex). For zero-phase
        filters, imag parts will be ~0 (up to round-off).
    """
    if N < 2:
        raise ValueError("N must be >= 2.")
    if R <= 0:
        raise ValueError("R must be > 0.")
    if Rc is None:
        H_LF = np.ones(N, dtype=np.complex128)
        H_HF = np.zeros(N, dtype=np.complex128)
        return H_HF, H_LF
    if Rc <= 0:
        raise ValueError("Rc must be > 0 when filtering is requested.")

    ffreq = np.fft.fftfreq(N) # cycles/sample
    res   = ffreq * 2*R       # "resolution" axis

    if filter_type == "gaussian_fast":
        # Discrete, truncated Gaussian kernel → circular conv (mode='wrap')
        sigma    = 2*R / (np.pi * Rc) * np.sqrt(np.log(2)/2.)  # samples
        truncate = 4.
        radius   = int(truncate * max(sigma, 1e-12) + 0.5)
        radius   = min(radius, max(0, (N//2) - 1))

        if radius == 0:
            H_LF = np.ones(N, dtype=np.complex128)
        else:
            x       = np.arange(-radius, radius+1, dtype=float)
            kernel  = np.exp(-0.5*(x/sigma)**2)
            kernel /= kernel.sum()

            # Pack kernel for circular conv: lags >= 0 at start, lags < 0 at end
            ker_pad = np.zeros(N, dtype=float)
            ker_pad[:radius+1] = kernel[radius:]   # 0..+radius
            ker_pad[-radius:]  = kernel[:radius]   # -radius..-1

            H_LF = np.fft.fft(ker_pad).astype(np.complex128)

    elif filter_type == "gaussian_true":
        sigma = 2*R / (np.pi * Rc) * np.sqrt(np.log(2)/2)
        H_LF  = np.exp(-2*np.pi**2*sigma**2*(res/(2*R))**2).astype(np.complex128)

    elif filter_type == "step":
        H_LF                   = np.ones_like(res, dtype=float)
        H_LF[np.abs(res) > Rc] = 0.
        H_LF                   = H_LF.astype(np.complex128)

    elif filter_type == "smoothstep":
        H_LF = smoothstep(res, Rc).astype(np.complex128)

    else:
        raise ValueError("Invalid filter_type. Use 'gaussian_fast','gaussian_true','step','smoothstep'.")

    H_HF = (1. - H_LF).astype(np.complex128)
    return H_HF, H_LF




def filtered_flux(flux, R, Rc, filter_type="gaussian", show=False):
    """
    Split an input 1D flux into high-pass and low-pass components using a cut-off
    spectral resolution 'Rc'.

    The mapping between FFT frequency 'f' and "resolution" follows:
        resolution ~ 2 * R * f
    so that applying a frequency-domain mask at |resolution| > Rc behaves like a
    high/low-pass in "lines-per-resolution-element" space.

    Parameters
    ----------
    flux : ndarray, shape (N,)
        Input flux samples on an evenly spaced wavelength grid (in practice,
        any uniform sampling along the spectral axis).
    R : float
        Effective spectral resolution of the input array (assuming Nyquist sampling).
    Rc : float or None
        Cut-off resolution for the filter. If None, no filtering is applied and
        the low-pass component is identically zero.
    filter_type : {'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep', 'savitzky_golay'}, optional
        Filtering method:
        - 'gaussian'      : real-space Gaussian blur with sigma derived from Rc (Appendix A, Martos+2024).
        - 'gaussian_fast' : Like 'gaussian' but without using gaussian_filter1d(): Faster but less accurate.
        - 'gaussian_true' : Analytic Gaussian convolution, as mathematically defined.
        - 'step'          : ideal top-hat in Fourier space (sharp cutoff at Rc).
        - 'smoothstep'    : smooth window in Fourier space around Rc.
        - 'savitzky_golay': polynomial smoothing with window matched to Rc.
    show : bool, optional
        If True, plot original, low-pass, and high-pass components.

    Returns
    -------
    flux_HF : ndarray, shape (N,)
        High-pass component (original - low-pass).
    flux_LF : ndarray, shape (N,)
        Low-pass component.

    Raises
    ------
    ValueError
        If inputs are inconsistent (non-finite flux, non-positive R, or invalid Rc).
    KeyError
        If 'filter_type' is unsupported.
    """
    # No filter applied
    if Rc is None:
        return flux, np.zeros_like(flux)
    
    valid        = np.isfinite(flux)
    flux_filled  = fill_nan_linear(x=None, y=flux) # NaN gaps are filled with linear interpolation
    valid_filled = np.isfinite(flux_filled)        # NaN edges can remain
    flux_valid   = flux_filled[valid_filled]
    
    # Real-space Gaussian sigma that approximates a cut at Rc in resolution space: sigma derived from Martos+2025 Appendix A
    if filter_type == "gaussian":
        sigma         = 2*R / (np.pi * Rc) * np.sqrt(np.log(2) / 2) 
        flux_valid_LF = gaussian_filter1d(flux_valid, sigma=sigma)
        
    # Savitzky-Golay filter
    elif filter_type == "savitzky_golay":
        # Map Rc to a window length (odd integer >= 5)
        sigma = 2*R / (np.pi * Rc) * np.sqrt(np.log(2) / 2)         # see Appendix A of Martos et al. (2025)
        N     = max(int(round(4*sigma)) | 1, 5)                     #  "| 1" :ensure odd and >=5 (gaussian approximation (sigma = N/4 is an empirical approximation) )
        N     = min(N, len(flux_valid) - (1 - len(flux_valid) % 2)) # keep <= size and odd (set window length based on Rc)
        # polyorder=3 safe default (must be < window_length)
        poly          = min(3, N - 2)
        flux_valid_LF = savgol_filter(flux_valid, window_length=N, polyorder=poly)
    
    # Cached frequency responses
    elif filter_type in {"gaussian_fast", "gaussian_true", "step", "smoothstep"}:
        fft           = np.fft.fft(np.asarray(flux_valid, dtype=np.complex128))  # FFT pleine
        _, H_LF       = _fft_filter_response(N=len(flux_valid), R=R, Rc=Rc, filter_type=filter_type)
        flux_valid_LF = np.real(np.fft.ifft(fft * H_LF))

    else:
        raise KeyError("Invalid 'filter_type'. Use one of: 'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep', 'savitzky_golay'." )

    # Reinsert into an array with original NaNs preserved
    flux_LF               = np.full_like(flux, np.nan)
    flux_LF[valid_filled] = flux_valid_LF
    flux_LF[~valid]       = np.nan # Retrieving original NaN 
    flux_HF               = flux - flux_LF

    if show:
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(flux,    'crimson',   label="Original")
        plt.plot(flux_LF, 'seagreen',  label="Low-Pass")
        plt.plot(flux_HF, 'royalblue', label="High-Pass")
        plt.xlabel("Wavelength axis", fontsize=14)
        plt.ylabel("Flux axis", fontsize=14)
        plt.xlim(0, len(flux))
        plt.legend(fontsize=14, loc="upper right", frameon=True, edgecolor="gray", facecolor="whitesmoke")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()        
        plt.show()
        
    return flux_HF, flux_LF


@lru_cache(maxsize=50)
def get_fraction_noise_filtered(N, R, Rc, filter_type, empirical=False):
    """
    Return the fractions of white-noise variance that pass through the
    high-pass (HF) and low-pass (LF) branches of a filter defined in the
    “resolution” domain.

    Parameters
    ----------
    N : int
        Number of uniformly sampled points (only the length matters).
    R : float
        Sampling resolving power of the HR grid (Nyquist convention). Used to map
        FFT frequency 'f' to resolution via 'res = 2 * R * f'.
    Rc : float or None
        Cutoff resolution for the filter. If None, no filtering is applied and
        the function returns '(1.0, 1.0)'.
    filter_type : {'gaussian', 'gaussian_fast', 'gaussian_true', 'step', 'smoothstep'}
        Shape of the low-pass transfer function used to build the complementary
        LF/HF pair. The value ''gaussian'' is implemented with the fast variant
        (''gaussian_fast'').
    empirical : bool, optional
        If 'True', estimate the fractions by Monte Carlo using standard normal
        noise (slow; for debugging/validation). If 'False' (default), compute
        the fractions analytically from the transfer functions (fast; cached).

    Returns
    -------
    fn_HF : float
        Fraction of the input white-noise variance that remains in the HF output.
    fn_LF : float
        Fraction of the input white-noise variance that remains in the LF output.

    Notes
    -----
    Analytical path: obtain the HF/LF transfer functions 'H_HF' and 'H_LF'
    on the discrete frequency grid (via a cached call to '_fft_filter_response'),
    then, for unit-variance white noise, the output variances are proportional to
    'mean(|H|^2)' over frequencies. The returned fractions are therefore
    'fn_HF = mean(|H_HF|^2)' and 'fn_LF = mean(|H_LF|^2)'.
    For complementary, energy-preserving splits one typically has
    'fn_HF + fn_LF ≈ 1' (up to discretization/edge effects).

    Empirical path: draw many (length 'N') standard-normal sequences, filter
    them, and return 'var(HF)/var(x)' and 'var(LF)/var(x)' averaged over the
    realizations. Much slower; intended for checks rather than production.
    """

    if Rc is None:
        return 1., 0.
    
    # --- Analytical: white-noise power fractions for the LF/HF split, cached by (N, R, Rc, type).
    if not empirical:
        if filter_type == "gaussian":
            filter_type = "gaussian_fast"
        H_HF, H_LF = _fft_filter_response(N=N, R=R, Rc=Rc, filter_type=filter_type)
        # For white noise, output variance ∝ mean(|H|^2) over frequencies.
        fn_LF = np.nanmean( np.abs(H_LF)**2 )
        fn_HF = np.nanmean( np.abs(H_HF)**2 )
        return fn_HF, fn_LF

    # --- Empirical (debug) path: slower, but straightforward ---
    # Keep sample count modest; analytic path is preferred.
    n     = 1000
    fn_HF = 0.
    fn_LF = 0.
    var   = 1
    sig   = np.sqrt(var)
    for i in range(n):
        noise              = np.random.normal(0, sig, N)
        noise_HF, noise_LF = filtered_flux(flux=noise, R=R, Rc=Rc, filter_type=filter_type)
        fn_HF += np.nanvar(noise_HF) / var / n
        fn_LF += np.nanvar(noise_LF) / var / n
    
    return fn_HF, fn_LF



def get_mag(flux_obs, flux_ref):
    """
    Compute the apparent magnitude of an observed flux relative to a reference flux.

    Parameters
    ----------
    flux_obs : array_like
        Observed flux values.
    flux_ref : array_like
        Reference flux values (e.g., Vega spectrum in the same band).

    Returns
    -------
    mag : float
        Apparent magnitude.
    """
    mean_obs = np.nanmean(flux_obs)
    mean_ref = np.nanmean(flux_ref)

    if mean_obs <= 0 or mean_ref <= 0:
        raise ValueError("Both observed and reference mean fluxes must be strictly positive.")

    return -2.5 * np.log10(mean_obs / mean_ref)



def get_mag_from_flux(flux, units, band0):
    """
    Compute the Vega-based magnitude for a given flux in a photometric band.

    Parameters
    ----------
    flux 
        Mean flux of the object in the chosen band (in 'units').
    units : str
        Flux units. Options:
            - "J/s/m2/um"   : SI
            - "W/m2/um"     : identical to above
            - "Jy"          : Jansky
            - "mJy"         : milliJansky
            - "erg/s/cm2/A" : CGS
    band0 : str
        Photometric band identifier (e.g., "J", "H", "K").
        The globals() must contain 'lmin_band0' and 'lmax_band0'.

    Returns
    -------
    magnitude 
        Vega-based magnitude in the chosen band.
    """

    # ---- Get wavelength bounds from globals
    try:
        lmin_band0 = globals()[f"lmin_{band0}"]
        lmax_band0 = globals()[f"lmax_{band0}"]
    except KeyError:
        raise KeyError(f"{band0} is not a considered band to define the magnitude. "f"Please choose among: {bands}, {instrus}")

    # ---- Load Vega spectrum in SI units (J/s/m2/µm)
    vega_spectrum = load_vega_spectrum().copy()
    vega_spectrum.crop(lmin_band0, lmax_band0)

    lam      = vega_spectrum.wavelength * 1e-6  # µm -> m
    F_lambda = vega_spectrum.flux               # J/s/m2/µm

    # ---- Convert Vega flux depending on units
    if units in ["J/s/m2/um", "W/m2/um", "J/s/m2/µm", "W/m2/µm"]:
        vega_flux = F_lambda

    elif units == "Jy":
        # Convert F_lambda [J/s/m2/µm] → F_nu [Jy]
        F_lambda_m = F_lambda * 1e6          # J/s/m2/m
        F_nu       = F_lambda_m * lam**2 / c # J/s/m2/Hz
        vega_flux  = F_nu / 1e-26            # Jy

    elif units == "mJy":
        F_lambda_m = F_lambda * 1e6
        F_nu       = F_lambda_m * lam**2 / c
        vega_flux  = F_nu / 1e-29            # mJy
    
    elif units == "muJy":
        F_lambda_m = F_lambda * 1e6
        F_nu       = F_lambda_m * lam**2 / c
        vega_flux  = F_nu / 1e-32            # muJy

    elif units == "erg/s/cm2/A":
        # Convert F_lambda [J/s/m2/µm] → erg/s/cm2/A
        # 1 J = 1e7 erg ; 1 m2 = 1e4 cm2 ; 1 µm = 1e4 A
        vega_flux = F_lambda * 1e7 / 1e4 / 1e4  # erg/s/cm2/A

    else:
        raise ValueError(f"Unit '{units}' not implemented.")

    # ---- Compute magnitude
    magnitude = get_mag(flux_obs=flux, flux_ref=vega_flux)
    print(f"{flux} {units} implies a {round(magnitude, 2):.2f} magnitude in {band0}-band")
    
    return magnitude



def get_mag_from_mag(T, lg, model, mag_input, band0_input, band0_output):
    
    """
    Convert a magnitude from 'band_in' to 'band_out' for a given object SED by
    *scaling the spectrum* so that its synthetic magnitude matches 'mag_input'
    in 'band_in', then computing the synthetic magnitude in 'band_out'.

    Magnitudes are in the Vega system:
        m_band = -2.5 * log10( ∫ F_obj(λ) S_band(λ) dλ / ∫ F_vega(λ) S_band(λ) dλ )

    Parameters
    ----------
    T : float
        Effective temperature for the spectrum loader (K).
    logg : float
        Surface gravity (dex, cgs) for the spectrum loader.
    model : str
        Spectrum model key. If in {"BT-NextGen", "Husser"} the object is treated
        as a star, otherwise as a planet (calls the corresponding loader).
    mag_input : float
        Known magnitude of the object in 'band_in' (Vega system).
    band0_input : str
        Input photometric band name (e.g., "K", "Ks", "W1", "W2").
    band0_output : str
        Output photometric band name.

    Returns
    -------
    float
        The synthetic Vega magnitude in 'band_out'.
    """
    
    lmin_band0_input = globals()[f"lmin_{band0_input}"] # [µm]
    lmax_band0_input = globals()[f"lmax_{band0_input}"] # [µm]
    
    lmin_band0_output = globals()[f"lmin_{band0_output}"] # [µm]
    lmax_band0_output = globals()[f"lmax_{band0_output}"] # [µm]

    lmin = min(lmin_band0_input, lmin_band0_output)
    lmax = max(lmax_band0_input, lmax_band0_output)
    R    = R0_max
    dl   = (lmin+lmax)/2 / (2*R)
    wave = np.arange(lmin, lmax, dl)
    
    mask_input  = (wave >= lmin_band0_input)  & (wave <= lmax_band0_input)
    mask_output = (wave >= lmin_band0_output) & (wave <= lmax_band0_output)

    vega = load_vega_spectrum() # [J/s/m2/µm]
    vega = vega.interpolate_wavelength(wave_output=wave, renorm=False).flux
    
    if model in {"BT-NextGen", "Husser"}:
        flux = load_star_spectrum(T_star=T, lg_star=lg, model=model) # [J/s/m2/µm]
    else:
        flux = load_planet_spectrum(T_planet=T, lg_planet=lg, model=model)  # [J/s/m2/µm]
    flux = flux.interpolate_wavelength(wave_output=wave, renorm=False).flux # [J/s/m2/µm]
    
    flux *= np.nanmean(vega[mask_input])*10**(-0.4*mag_input) / np.nanmean(flux[mask_input]) # Ratio by which to adjust the spectrum flux in order to have the input magnitude

    mag_output = get_mag(flux_obs=flux[mask_output], flux_ref=vega[mask_output])
    
    return mag_output



#######################################################################################################################
############################################# class Spectrum: #########################################################
#######################################################################################################################

class Spectrum:

    def __init__(self, wavelength, flux, R=None, T=None, lg=None, model=None, rv=0, vsini=0):
        """
        Container and utilities for 1D spectra.
    
        Attributes
        ----------
        wavelength : (N,) array_like
            Wavelength samples (usually in µm).
        flux : (N,) array_like
            Flux samples (arbitrary units unless specified).
        R : float or None
            (Average) resolving power of the spectrum (dimensionless).
        T : float or None
            Characteristic temperature [K].
        lg : float or None
            Surface gravity [dex(cm/s²)].
        model : str or None
            Model family/name.
        rv : float
            Radial velocity of the spectrum [km/s].
        vsini : float
            Projected rotational velocity [km/s].
        """
        # Sanity checks
        if wavelength.ndim != 1 or flux.ndim != 1:
            raise ValueError("'wavelength' and 'flux' must be 1D arrays.")
        if wavelength.size != flux.size:
            raise ValueError("'wavelength' and 'flux' must have the same length.")
        
        self.wavelength = wavelength                     # Wavelength axis of the spectrum
        self.flux       = flux                           # Flux of the spectrum
        self.R          = R if R is not None else None   # Resolution of the spectrum
        self.T          = T if T is not None else None   # Temperature of the spectrum
        self.lg         = lg if lg is not None else None # Surface gravity of the spectrum
        self.model      = model                          # Model of the spectrum
        self.rv         = rv                             # Radial velocity 
        self.vsini      = vsini                          # Rotationnal velocity

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def copy(self):
        """
        Return a deep copy of the spectrum.
        """
        return copy.deepcopy(self)
        
    def crop(self, lmin, lmax):
        """
        Crop the spectrum to the interval [lmin, lmax] and recompute R.

        Parameters
        ----------
        lmin, lmax : float
            Wavelength bounds (same unit as 'self.wavelength').
        """
        if lmax <= lmin:
            raise ValueError("'lmax' must be greater than 'lmin'.")
        mask_wavelength = (self.wavelength >= lmin) & (self.wavelength <= lmax)
        self.flux       = self.flux[mask_wavelength]
        self.wavelength = self.wavelength[mask_wavelength]
        self.R          = estimate_resolution(wavelength=self.wavelength)
        
    def crop_nan(self):
        """
        Remove samples where flux is not finite; recompute R.
        """
        mask_valid      = np.isfinite(self.flux)
        self.wavelength = self.wavelength[mask_valid]
        self.flux       = self.flux[mask_valid]
        self.R          = estimate_resolution(wavelength=self.wavelength)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def set_flux(self, flux_tot):
        """
        Renormalize flux to a specified total (e.g., photon count, J/s/m2/µm, etc.).

        Notes
        -----
        Assumes 'self.flux' is on the same grid and represents quantities
        that should sum linearly over bins (i.e., not a spectral density
        unless the wavelength spacing is constant).

        Parameters
        ----------
        flux_tot : float
            Target total sum of the flux array.
        """
        denom = np.nansum(self.flux)
        if not np.isfinite(denom) or denom == 0:
            raise ValueError("Cannot renormalize: current flux sum is zero or non-finite.")
        self.flux = (flux_tot * self.flux) / denom

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_nbphotons_min(self, config_data, wave_output):
        """
        Convert a spectral *density* [J/s/m2/µm] into photons per minute
        collected by the telescope over a target wavelength grid.

        Parameters
        ----------
        config_data : dict-like
            Instrument configuration; must contain 'telescope['area']' [m2].
        wave_output : (M,) array_like
            Target wavelength grid [µm], evenly spaced.

        Returns
        -------
        Spectrum
            New Spectrum instance on 'wave_output', flux in [photons/min].
        """
        area     = config_data["telescope"]["area"]                                   # Effective collecting area [m2], accounting for central hole, secondary mirror, and spider obscuration
        dl       = np.gradient(wave_output)                                           # Wavelength spacing Δλ [µm]
        spectrum = self.interpolate_wavelength(wave_output=wave_output, renorm=False) # Reinterpolating the flux (in density) on wave_output
        # [J/s/m2/µm] -> [photons/s/m2/µm] using λ[m]/(h*c)
        spectrum.flux = spectrum.flux * wave_output*1e-6 / (h*c)
        # Collecting area and bin width (µm) and minutes:
        spectrum.flux = spectrum.flux * area * dl * 60
        return spectrum # [photons/mn]

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def degrade_resolution(self, wave_output, renorm, gaussian_filtering=True, R_output=None, sigma_kernel=None, sigma=None, verbose=False):
        """
        Degrade spectral resolution to match the target resolving power and then
        down-bin onto 'wave_output'.
    
        Notes
        -----
        - If 'R_output' is provided, it is used as the target resolving power; otherwise
          the Nyquist resolving power of 'wave_output' is used.
        - The Gaussian kernel width comes from variance addition of Gaussians and the
          Nyquist FWHM=2 px assumption at R_old: 
            sigma_kernel(px) = (2/2.355)*sqrt( (R_old/R_new)**2 - 1 ).
        - 'gaussian_filtering=False' skips the convolution entirely (and ignores R_output).
    
        Parameters
        ----------
        wave_output : (M,) array_like
            Target wavelength grid (monotonic).
        renorm : bool
            Conserve total flux over overlap after rebinning (use if flux is not a density).
        gaussian_filtering : bool, optional
            Apply Gaussian convolution before down-binning. Default True.
        R_output : float or None, optional
            Target resolving power. If None, uses Nyquist resolution of 'wave_output'.
        sigma_kernel : float or None, optional
            Standard deviation for Gaussian kernel.
        sigma : (N,) array_like or None
            Optional 1-sigma flux uncertainty on the *current* grid.
    
        Returns
        -------
        Spectrum
            Degraded spectrum sampled on 'wave_output'.
        """
        
        # Interpolate current spectrum on a fine *regular* linear grid with the same range as wave_output: needed for the binning 
        spectrum_interp = self.evenly_spaced(renorm=renorm, wave_output=wave_output, sigma=sigma)
        flux_interp     = spectrum_interp.flux
        wave_interp     = spectrum_interp.wavelength
        sigma_interp    = getattr(spectrum_interp, "sigma", None)

        R_old      = round(spectrum_interp.R)                           # Old max resolution (assuming Nyquist sampling)
        R_sampling = round(estimate_resolution(wavelength=wave_output)) # New resolution     (assuming Nyquist sampling)
        
        # Only forbid up-resolution if we do NOT provide a stricter R_output
        if R_sampling > R_old:
            raise ValueError("Target grid is finer than the current max resolving power. This function only supports down-binning (otherwise NaN would be injected). Provide a coarser 'wave_output'.")
        if (R_output is not None) and (R_output > R_old):
            raise ValueError(f"'R_output' ({R_output:.0f}) must be ≤ current max resolving power ({R_old:.0f}).")
        if verbose and (R_output is not None) and (R_output > R_sampling):
            print(f"WARNING: 'R_output' ({R_output:.0f}) must be ≤ the sampling resolving power of the output grid ({R_sampling:.0f}) to satisfy Nyquist.")
        
        R_new = R_sampling if R_output is None else R_output

        # LSF Gaussian convolution (optional) 
        if gaussian_filtering:
            if sigma_kernel is None:
                # Nyquist assumption: FWHM_old = 2 px → sigma_old = 2/2.355 px: sigma_kernel = sqrt(sigma_new^2 - sigma_old^2) = (2/2.355)*sqrt(ratio^2 - 1)
                ratio_R      = R_old / R_new
                sigma_kernel = np.sqrt( (ratio_R**2 - 1) / (2*np.log(2)) )
            
            # Fill internal NaNs (linear) and crop to avoid edges inside the convolution window
            #flux_raw                  = np.copy(flux_interp)
            valid                     = np.isfinite(flux_interp)
            flux_filled               = fill_nan_linear(wave_interp, flux_interp) # NaN gaps are filled with linear interpolation
            valid_filled              = np.isfinite(flux_filled)                  # NaN edges can remain
            flux_interp[valid_filled] = gaussian_filter1d(flux_filled[valid_filled], sigma=sigma_kernel)
            flux_interp[~valid]       = np.nan # Retrieving original NaN

        # Down-bin to target grid
        dl_new               = np.gradient(wave_output) # New wavelength spacing Δλ
        flux_lr, sigma_lr, _ = downbin_spec(specHR=flux_interp, sigmaHR=sigma_interp, weightHR=None, lamHR=wave_interp, lamLR=wave_output, dlam=dl_new) # down binned flux
        
        spectrum = Spectrum(wave_output, flux_lr, R=R_sampling, T=self.T, lg=self.lg, model=self.model, rv=self.rv, vsini=self.vsini)
        if renorm:
            # Conserve the *total* flux in the overlapping domain
            mask_overlap  = (wave_interp >= wave_output[0]) & (wave_interp <= wave_output[-1]) # Inside wavelength_output
            flux_tot      = np.nansum(flux_interp[mask_overlap])
            spectrum.flux = flux_lr * flux_tot / np.nansum(flux_lr)
            if sigma is not None:
                spectrum.sigma = sigma_lr * flux_tot / np.nansum(flux_lr)
        else:
            # Not convserving the flux (e.g. for spectra in density or for transmissions)
            spectrum.flux = flux_lr
            if sigma is not None:
                spectrum.sigma = sigma_lr
        return spectrum
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
    def interpolate_wavelength(self, wave_output, renorm, fill_value=np.nan, wave_input=None, sigma=None):
        """
        Interpolate the spectrum onto a new wavelength grid.

        Parameters
        ----------
        wave_output : (M,) array_like
            Target wavelength grid.
        renorm : bool, optional
            Conserve *total* flux over the overlap domain. Default False.
        fill_value : float, optional
            Value used for extrapolation. Default np.nan.
        wave_input : (N,) array_like or None
            Custom input wavelength grid (useful for Doppler-shifted interpolation only).
            If None, uses 'self.wavelength'.
        sigma : (N,) array_like or None
            Optional per-pixel uncertainties on the input grid.

        Returns
        -------
        Spectrum
            Interpolated spectrum on 'wave_output'.
        """
        if wave_input is None: # wave_input is only usefull for the doppler_shift() function
            wave_input = self.wavelength
        
        # Interpolates flux values on the new axis (wave_output)
        flux_interp = interp1d(wave_input, self.flux, bounds_error=False, fill_value=fill_value)(wave_output) 
        
        if sigma is not None:
            sigma_interp = interpolate_flux_with_error(wave=wave_input, flux=None, sigma=sigma, weight=None, wave_new=wave_output)[1]
        else:
            sigma_interp = None
        
        spectrum = Spectrum(wave_output, flux_interp, R=estimate_resolution(wavelength=wave_output), T=self.T, lg=self.lg, model=self.model, rv=self.rv, vsini=self.vsini)
        if renorm:
            # Conserve the *total* flux in the overlapping domain
            mask_overlap  = (wave_input >= wave_output[0]) & (wave_input <= wave_output[-1])
            flux_tot      = np.nansum(self.flux[mask_overlap])
            spectrum.flux = flux_interp * flux_tot / np.nansum(flux_interp)
            if sigma is not None:
                spectrum.sigma = sigma_interp * flux_tot / np.nansum(flux_interp)
        else:
            # Not convserving the flux (e.g. for spectra in density or for transmissions)
            spectrum.flux = flux_interp
            if sigma is not None:
                spectrum.sigma = sigma_interp
        return spectrum
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def evenly_spaced(self, R_interp=None, renorm=False, fill_value=np.nan, wave_output=None, sigma=None):
        """
        Resample onto a *linear, constant-step* wavelength grid spanning
        either the spectrum range or the (slightly padded) 'wave_output' range.

        Parameters
        ----------
        renorm : bool, optional
            Conserve *total* flux after resampling. Default False.
        fill_value : float, optional
            Extrapolation fill value. Default np.nan.
        wave_output : (M,) array_like or None
            If provided, defines the target limits. A 2% padding is applied.
        sigma : (N,) array_like or None
            Optional per-pixel uncertainties to be propagated.

        Returns
        -------
        Spectrum
            Resampled spectrum on a fine, linear, evenly spaced grid.
        """
        if wave_output is not None:
            mask_wavelength = (self.wavelength >= wave_output[0]) & (self.wavelength <= wave_output[-1])
            lmin = wave_output[0]
            lmax = wave_output[-1]
        else:
            mask_wavelength = np.full(len(self.wavelength), True)
            lmin = self.wavelength[0]
            lmax = self.wavelength[-1]
        
        # Interpolation resolution: use max(R) (Nyquist-based) but cap at R0_max
        if R_interp is None:
            dl_old   = np.gradient(self.wavelength[mask_wavelength])              # Wavelength spacing Δλ
            R_interp = np.nanmax( self.wavelength[mask_wavelength] / (2*dl_old) ) # interpolation Resolution (need to be the max res to avoid nan with cg.downbin)        
        R_interp    = min(R_interp, R0_max)          # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
        dl          = (lmin + lmax)/2 / (2*R_interp) # Delta lambda
        wave_interp = np.arange(lmin, lmax, dl)      # Constant and linear input wavelength array
        return self.interpolate_wavelength(wave_output=wave_interp, renorm=renorm, fill_value=fill_value, sigma=sigma) 
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def doppler_shift(self, rv, renorm=False, fill_value=np.nan):
        """
        Apply a Doppler shift for a given radial velocity.

        Parameters
        ----------
        rv : float
            Radial velocity [km/s]. Positive = redshift, negative = blueshift.
        renorm : bool, optional
            Conserve *total* flux after interpolation. Default False.
        fill_value : float, optional
            Extrapolation fill value. Default np.nan.

        Returns
        -------
        Spectrum
            Doppler-shifted spectrum (same sampling as input).
        """
        if rv == 0:
            return self.copy()
        else: # λ' = λ * (1 + v/c) with v in m/s and c in m/s
            wshift          = self.wavelength * (1 + (1000*rv / c)) # offset wavelength axis
            spectrum_rv     = self.interpolate_wavelength(wave_input=wshift, wave_output=self.wavelength, renorm=renorm, fill_value=fill_value)
            spectrum_rv.rv += rv
            return spectrum_rv

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def broad(self, vsini, epsilon=0.8, fastbroad=True):
        """
        Convolve the spectrum with a rotational broadening profile.

        Parameters
        ----------
        vsini : float
            Projected rotational velocity [km/s]. Must be >= 0.
        epsilon : float, optional
            Linear limb-darkening coefficient in [0, 1]. Default 0.8.
        fastbroad : bool, optional
            Use 'pyasl.fastRotBroad' (faster, approximate) if True,
            else 'pyasl.rotBroad' (slower, more accurate).

        Returns
        -------
        Spectrum
            Rotationally broadened spectrum.
        """
        if vsini < 0:
            raise ValueError("'vsini' must be non-negative.")
        elif vsini == 0:
            return self.copy()
        else:
            if fastbroad: # fast spectral broadening (but less accurate)
                flux = pyasl.fastRotBroad(self.wavelength*1e4, self.flux, epsilon=epsilon, vsini=vsini)
            else: # slow spectral broadening (but more accurate)
                flux = pyasl.rotBroad(self.wavelength[np.isfinite(self.flux)]*1e4, self.flux[np.isfinite(self.flux)], epsilon=epsilon, vsini=vsini) # ignoring NaN values at the same time
                f    = interp1d(self.wavelength[np.isfinite(self.flux)], flux, bounds_error=False, fill_value=np.nan) 
                flux = f(self.wavelength) # pas besoin de "conserver le nb de photons" car c'est un effet intrinsèque (on ne change pas largeur des bins)
            spectrum_vsini       = self.copy()
            spectrum_vsini.flux  = flux
            spectrum_vsini.vsini += vsini
            return spectrum_vsini 

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def get_psd(self, smooth=0, one_sided=True):
        """
        Compute the power spectral density (PSD) of the flux array.
    
        Parameters
        ----------
        smooth : float, optional
            Standard deviation (in bins) of a Gaussian smoothing applied to the PSD.
            Use 0 to disable smoothing. Default is 0.
        one_sided : bool, optional
            If True, return a one-sided PSD (positive frequencies) using an rFFT
            with proper energy folding (doubling positive-frequency bins except DC
            and, if applicable, Nyquist). If False, return a two-sided PSD.
            Default is True.
    
        Returns
        -------
        res : ndarray
            “Resolution-like” frequency axis scaled by ~ 2*R (heuristic).
        PSD : ndarray
            Power spectral density (arbitrary units).
        """
        
        # Copy and handle NaNs robustly
        signal = np.asarray(self.flux, dtype=float)
        if np.isnan(signal).any():
            signal = signal[np.isfinite[signal]]
            print("WARNING (get_psd): NaN values inside self.flux...")
        
        N = signal.size
        if N < 2:
            raise ValueError("Signal too short for PSD computation.")
            
        # “Resolution-like” axis    
        ffreq = np.fft.rfftfreq(N) # cycles per sample
        res   = ffreq * 2*self.R   # 2*R = sampling resolution (heuristic)

        if one_sided:
            # One-sided PSD via rFFT with correct energy folding
            TF  = np.fft.rfft(signal)
            PSD = np.abs(TF)**2 / N
            if N % 2 == 0:
                # Even N: double bins 1..-2 (exclude DC and Nyquist)
                if PSD.size > 2:
                    PSD[1:-1] *= 2
            else:
                # Odd N: double bins 1..end (exclude DC only)
                if PSD.size > 1:
                    PSD[1:] *= 2
        else:
            # Two-sided PSD
            TF  = np.fft.fft(signal)
            PSD = np.abs(TF)**2 / N
    
        # Optional Gaussian smoothing
        if smooth > 0:
            PSD = gaussian_filter1d(PSD, sigma=smooth)
    
        return res, PSD



def get_psd(wave, flux, R=None, smooth=0):
    """
    Calculate the psd of the inpux flux.

    Parameters
    ----------
    wave : array 1d
        wavelength axis.
    flux : array 1d
        Flux axis.
    R : float
        Sampling resolution.
    smooth : float, optional
        Smoothing parameters of the PSF. The default is 0.

    Returns
    -------
    res : array 1d
        resolution axis.
    psd : array 1d
        PSD axis.
    """
    if wave is None and R is None:
        raise KeyError("'wave' and 'R' are None...")
    valid = np.isfinite(flux)
    if wave is not None:
        wave = wave[valid]
        R    = estimate_resolution(wave)
    flux     = flux[valid]
    res, psd = Spectrum(wavelength=wave, flux=flux, R=R).get_psd(smooth=smooth)
    return res, psd



#######################################################################################################################
##################################################### Loading functions: ##############################################
#######################################################################################################################

def get_model_grid(model, instru=None):
    """
    Return (T_grid, lg_grid) for a given model (and optional instrument).

    Parameters
    ----------
    model : str
        Model family (e.g., 'BT-Settl', 'Exo-REM', 'BT-NextGen', 'mol_CO', ...).
    instru : str, optional
        Instrument name; used to decide which subgrid to return for some models.

    Returns
    -------
    T_grid : ndarray
        Supported temperatures (K).
    lg_grid : ndarray
        For thermal/stellar models: supported log10(cm/s^2).
        For molecular models ('mol_*'): array of molecule names (dtype=object).

    Raises
    ------
    KeyError
        If the model name is unsupported.
    """
    
    # --------------------
    # Planets / substellar
    # ---------------------
    if model == "BT-Settl" or model == "PICASO":
        T_grid  = np.concatenate([[200, 220, 240, 250, 260, 280, 300, 320, 340, 360, 380, 400, 450], np.arange(500, 900, 50), np.arange(900, 3100, 100)])
        lg_grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

    elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
        T_grid  = np.arange(1400, 3100, 100)
        lg_grid = np.array([4.5, 5.0])

    elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
        if instru is None: # Default: low resolution
            T_grid = np.arange(400, 2050, 50)
            lg_grid = np.arange(3.0, 5.5, 0.5)
        else:
            lmin = globals()[f"lmin_{instru}"]
            lmax = globals()[f"lmax_{instru}"]
            if lmin >= 1.0 and lmax <= 5.3: # Very high resolution
                T_grid  = np.arange(200, 1950, 50)
                lg_grid = np.arange(3.0, 5.5, 0.5)
            elif lmin <= 4.0: # Low resolution
                T_grid  = np.arange(400, 2050, 50)
                lg_grid = np.arange(3.0, 5.5, 0.5)
            else: # High res (starts at 4 µm)
                T_grid  = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
                lg_grid = np.array([3.5, 4.0])

    elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
        T_grid  = np.array([200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        g_ms2   = np.array([100, 300, 1000, 3000])
        lg_grid = np.round(np.log10(g_ms2 * 1e2), 4)

    elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
        T_grid  = np.arange(400, 1250, 50)
        g_ms2   = np.array([10, 30, 100, 300, 1000])
        lg_grid = np.round(np.log10(g_ms2 * 1e2), 4)

    elif model == "SONORA": # https://zenodo.org/records/5063476
        T_grid  = np.concatenate([np.arange(200, 1050, 50), np.arange(1100, 2500, 100)])
        g_ms2   = np.array([10, 31, 100, 316, 1000, 3160])
        lg_grid = np.round(np.log10(g_ms2 * 1e2), 4)

    elif "mol_" in model: # https://hitran.org/lbl/
        T_grid = np.concatenate([np.arange(200, 1000, 50), np.arange(1000, 3100, 100)])
        lg_grid = np.array(["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "NO", "SO2", "NO2", "NH3"], dtype=object)
        
    elif model in ["Jupiter", "Saturn", "Uranus", "Neptune"]:
        if model == "Jupiter":
            T_grid  = np.array([88])
            lg_grid = np.array([3.4]) # np.log10(24.79*100) # https://en.wikipedia.org/wiki/Jupiter
        elif model == "Saturn":
            T_grid  = np.array([81])
            lg_grid = np.array([3.0]) # np.log10(10.44*100) # https://en.wikipedia.org/wiki/Saturn
        elif model == "Uranus":
            T_grid  = np.array([49])
            lg_grid = np.array([2.9]) # np.log10(8.69*100) # https://en.wikipedia.org/wiki/Uranus
        elif model == "Neptune":
            T_grid  = np.array([47])
            lg_grid = np.array([3.0]) # np.log10(11.15*100) # https://en.wikipedia.org/wiki/Neptune
    
    # ----------------
    # Stars
    # ----------------
    elif model == "BT-NextGen":
        T_grid = np.concatenate([np.arange(3000, 10000, 200), np.arange(10000, 41000, 1000)])
        lg_grid = np.array([3.0, 3.5, 4.0, 4.5])
    
    elif model == "Husser":
        T_grid = np.concatenate([np.arange(2300, 7100, 100), np.arange(7200, 12200, 200)])
        lg_grid = np.arange(0.0, 6.5, 0.5)
    
    else:
        raise KeyError(f"{model} is not a valid model. Supported: BT-Settl, PICASO, BT-Dusty, Exo-REM, Morley, Saumon, SONORA, BT-NextGen, Husser, Solar-system planets, or mol_*.")
    
    return T_grid, lg_grid



def get_T_lg_valid(T, lg, model, instru=None, T_grid=None, lg_grid=None):
    """
    Return the nearest valid (T, lg) pair on the requested model grid.

    Parameters
    ----------
    T : float or int
        Requested temperature (K).
    lg : float or int or str
        Requested surface gravity log10(cm/s^2) for thermal/stellar models.
        For molecular models ('mol_*'), this is the molecule name (str).
    model : str
        Model family (e.g., 'BT-Settl', 'Exo-REM', 'mol_CO', ...).
    instru : str, optional
        Instrument name (some grids depend on wavelength coverage).
    T_grid : ndarray, optional
        Pre-fetched temperature grid. If None, retrieved via 'get_model_grid'.
    lg_grid : ndarray or list[str], optional
        Pre-fetched gravity grid (or molecule list for 'mol_*'). If None, retrieved.

    Returns
    -------
    T_valid : float
        Closest temperature available in the model grid.
    lg_valid : float or str
        Closest gravity value (thermal/stellar models) or unchanged molecule (molecular models).
    """
    if T_grid is None or lg_grid is None:
        T_grid, lg_grid = get_model_grid(model, instru=instru)
    T_valid = T_grid[(np.abs(T_grid-T)).argmin()] # Closest available value
    if "mol_" in model:
        if model[4:] not in lg_grid:
            raise KeyError(f"{lg} is not a valid molecule, please choose among: {lg_grid}")
        lg_valid = lg
    else:
        lg_valid = lg_grid[(np.abs(lg_grid-lg)).argmin()] # Closest lg value
    return T_valid, lg_valid


@lru_cache(maxsize=100)
def load_spectrum(T, lg, model, albedo=False, instru=None, load_path=load_path):
    """
    Load a spectrum from disk for a given model and parameters.

    Parameters
    ----------
    T : float
        Temperature (K).
    lg : float or str
        Surface gravity log10(cm/s^2). For molecular models ('mol_*'), the molecule name.
    model : str
        Model family (e.g., 'BT-Settl', 'Exo-REM', 'PICASO', 'mol_CO', 'BT-NextGen', ...).
    albedo : bool, optional
        If True, loads a reflected-light (geometric albedo) spectrum when supported.
    instru : str, optional
        Instrument name; may select a different resolution grid for some models.
    load_path : str, optional
        Base directory where spectra are stored.

    Returns
    -------
    spectrum : Spectrum
        Spectrum with wavelength in µm and flux in J/s/m2/µm (unitless for albedos).
        The 'R' field is estimated from the wavelength grid.

    Raises
    ------
    KeyError
        If inputs are inconsistent with available files/grids.
    """
    try:
        if albedo:
            if model == "PICASO": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/albedo/PICASO/albedo_gas_giant_{T:.0f}K_lg{lg:.1f}.fits")
            else:
                raise KeyError(f"{model} IS NOT A VALID ALBEDO MODEL: PICASO.")
                
        elif not albedo:
            if model == "BT-Settl": # https://articles.adsabs.harvard.edu/pdf/2013MSAIS..24..128A              
                wave, flux = fits.getdata(load_path+f"/planet_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
            
            elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
                wave, flux = fits.getdata(load_path+f"/planet_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
                
            elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
                if instru is None: # Default: low resolution
                    load_path += f"/planet_spectrum/{model}/low_res/spectra_YGP_{T:.0f}K_logg{lg:.1f}_met1.00_CO0.50.fits"
                else:
                    lmin = globals()[f"lmin_{instru}"]
                    lmax = globals()[f"lmax_{instru}"]
                    if lmin >= 1.0 and lmax <= 5.3: # Very high resolution
                        FeH = 0.0  # Métallicité
                        CO  = 0.65 # Ratio C/O
                        load_path += f"/planet_spectrum/{model}/very_high_res/spect_Teff={T:04.0f}K_logg={lg:.1f}_FeH={FeH:+.1f}_CO={CO:.2f}.fits"
                    elif lmin <= 4.0: # Low resolution
                        load_path += f"/planet_spectrum/{model}/low_res/spectra_YGP_{T:.0f}K_logg{lg}_met1.00_CO0.50.fits"
                    else: # High res (starts at 4 µm)
                        load_path += f"/planet_spectrum/{model}/high_res/spectra_YGP_{T:.0f}K_logg{lg}_met1.00_CO0.50.fits"
                wave, flux = fits.getdata(load_path)
                
            elif model == "PICASO": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{T:.0f}K_lg{lg:.1f}.fits")
        
            elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
                g_planet = round(10**lg*1e-2) # m/s2
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/Morley/sp_t{T:.0f}g{g_planet}.fits")
                
            elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
                g_planet = round(10**lg*1e-2) # m/s2
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/Saumon/sp_t{T:.0f}g{g_planet}nc.fits")
            
            elif model == "SONORA": # https://zenodo.org/records/5063476
                g_planet = round(10**lg*1e-2) # m/s2
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/SONORA/sp_t{T:.0f}g{g_planet}nc_m0.0.fits")
                
            elif model[:4] == "mol_": # https://hitran.org/lbl/
                molecule = model[4:]
                wave, flux = fits.getdata(load_path + f"/planet_spectrum/molecular/{molecule}_T{T:.0f}K.fits")
            
            elif model in ["Jupiter", "Saturn", "Uranus", "Neptune"]:
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/solar system/psg_{model}_rad.fits")
                
            elif model == "BT-NextGen":
                wave, flux = fits.getdata(load_path+f"/star_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
            
            elif model == "Husser":
                wave = fits.getdata(load_path+f"/star_spectrum/{model}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                flux = fits.getdata(load_path+f"/star_spectrum/{model}/lte{T:05.0f}-{lg:4.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
            
            else:
                raise KeyError(model+" IS NOT A VALID THERMAL MODEL: BT-NextGen, Husser, BT-Settl, BT-Dusty, Exo-REM, PICASO, Morley, Saumon or SONORA.")
                
        spectrum = Spectrum(wavelength=wave, flux=flux, R=estimate_resolution(wavelength=wave), T=T, lg=lg, model=model, rv=0, vsini=0)
        return spectrum # [J/s/m2/µm] or [no unit] for albedos
    
    except Exception as e:
        raise KeyError(f"{T}K or {lg} are not valid parameters of the {model} grid: {e}")



def interpolate_T_lg_spectrum(T_valid, lg_valid, T, lg, model, albedo=False, instru=None, load_path=load_path, T_grid=None, lg_grid=None):
    """
    Interpolate a spectrum at (T, lg) using surrounding grid spectra.

    Parameters
    ----------
    T_valid : float
        Nearest grid temperature to the request.
    lg_valid : float or str
        Nearest grid gravity (or molecule name for 'mol_*').
    T : float
        Target temperature (K).
    lg : float
        Target gravity log10(cm/s^2).
    model : str
        Model family.
    albedo : bool, optional
        If True, interpolates in an albedo grid when supported.
    instru : str, optional
        Instrument name; can affect grid choice.
    load_path : str, optional
        Base directory of spectra on disk.
    T_grid : ndarray, optional
        Temperature grid; if None, fetched via 'get_model_grid'.
    lg_grid : ndarray or list[str], optional
        Gravity (or molecule-name) grid; if None, fetched via 'get_model_grid'.

    Returns
    -------
    spectrum : Spectrum
        Interpolated spectrum at (T, lg). Wavelength in µm, flux in J/s/m2/µm (unitless for albedos).
    """

    # Retrieve the model grid
    if T_grid is None or lg_grid is None:
        T_grid, lg_grid = get_model_grid(model, instru=instru)
    
    # Molecular models: only interpolate T; lg is a molecule label
    if "mol_" in model:
        T_lo, T_hi = get_bracket_values(T, T_grid)
        if T_lo == T_hi:
            return load_spectrum(T_lo, lg_valid, model, albedo=albedo, instru=instru, load_path=load_path)
        else:
            s_lo = load_spectrum(T_lo, lg_valid, model, albedo=albedo, instru=instru, load_path=load_path)
            s_hi = load_spectrum(T_hi, lg_valid, model, albedo=albedo, instru=instru, load_path=load_path)
            wave = s_lo.wavelength
            if len(s_hi.wavelength) != len(wave):
                s_hi = s_hi.interpolate_wavelength(wave, renorm=False)
            flux = linear_interpolate(s_lo.flux, s_hi.flux, T_lo, T_hi, T)
            R    = estimate_resolution(wavelength=wave)
            return Spectrum(wave, flux, R, T, lg, model)

    # Regular thermal/stellar models: bilinear in (T, lg)
    T_lo, T_hi   = get_bracket_values(T,  T_grid)
    lg_lo, lg_hi = get_bracket_values(lg, lg_grid)

    # No interpolation along T and lg
    if T_lo == T_hi and lg_lo == lg_hi:
        return load_spectrum(T_lo, lg_lo, model, albedo=albedo, instru=instru, load_path=load_path)

    # Interpolation along T
    elif lg_lo == lg_hi:
        s_lo = load_spectrum(T_lo, lg_lo, model, albedo=albedo, instru=instru, load_path=load_path)
        s_hi = load_spectrum(T_hi, lg_lo, model, albedo=albedo, instru=instru, load_path=load_path)
        wave = s_lo.wavelength
        if len(s_hi.wavelength) != len(wave):
            s_hi = s_hi.interpolate_wavelength(wave, renorm=False)
        flux = linear_interpolate(s_lo.flux, s_hi.flux, T_lo, T_hi, T)
        R    = estimate_resolution(wavelength=wave)
        return Spectrum(wave, flux, R, T, lg, model)
    
    # Interpolation along lg
    elif T_lo == T_hi:
        s_lo = load_spectrum(T_lo, lg_lo, model, albedo=albedo, instru=instru, load_path=load_path)
        s_hi = load_spectrum(T_lo, lg_hi, model, albedo=albedo, instru=instru, load_path=load_path)
        wave = s_lo.wavelength
        if len(s_hi.wavelength) != len(wave):
            s_hi = s_hi.interpolate_wavelength(wave, renorm=False)
        flux = linear_interpolate(s_lo.flux, s_hi.flux, lg_lo, lg_hi, lg)
        R    = estimate_resolution(wavelength=wave)
        return Spectrum(wave, flux, R, T, lg, model)

    # Interpolation along T and lg (Full bilinear interpolation)
    else:
        s_ll = load_spectrum(T_lo, lg_lo, model, albedo=albedo, instru=instru, load_path=load_path)
        s_lh = load_spectrum(T_lo, lg_hi, model, albedo=albedo, instru=instru, load_path=load_path)
        s_hl = load_spectrum(T_hi, lg_lo, model, albedo=albedo, instru=instru, load_path=load_path)
        s_hh = load_spectrum(T_hi, lg_hi, model, albedo=albedo, instru=instru, load_path=load_path)
        wave = s_ll.wavelength
        if len(s_lh.wavelength) != len(wave):
            s_lh = s_lh.interpolate_wavelength(wave, renorm=False)
        if len(s_hl.wavelength) != len(wave):
            s_hl = s_hl.interpolate_wavelength(wave, renorm=False)
        if len(s_hh.wavelength) != len(wave):
            s_hh = s_hh.interpolate_wavelength(wave, renorm=False)
        f_Tlo = linear_interpolate(s_ll.flux, s_lh.flux, lg_lo, lg_hi, lg)
        f_Thi = linear_interpolate(s_hl.flux, s_hh.flux, lg_lo, lg_hi, lg)
        flux  = linear_interpolate(f_Tlo,     f_Thi,     T_lo,  T_hi,  T)
        R     = estimate_resolution(wavelength=wave)
        return Spectrum(wave, flux, R, T, lg, model)



def load_planet_spectrum(T_planet=1000, lg_planet=4.0, model="BT-Settl", albedo=False, interpolated_spectrum=True, instru=None, load_path=load_path, T_grid=None, lg_grid=None):
    """
    Load a planet spectrum at the closest grid point or bilinearly interpolate to (T_planet, lg_planet).

    Parameters
    ----------
    T_planet : float, optional
        Planet temperature (K).
    lg_planet : float, optional
        Planet surface gravity log10(cm/s^2).
    model : str, optional
        Planet model family.
    albedo : bool, optional
        If True, uses an albedo grid (when supported).
    interpolated_spectrum : bool, optional
        If True, interpolates to the exact (T, lg). If False, returns the nearest grid spectrum.
    instru : str, optional
        Instrument name for model sub-grids.
    load_path : str, optional
        Base directory for spectra.
    T_grid : ndarray, optional
        Temperature grid (bypass fetching).
    lg_grid : ndarray, optional
        Gravity grid (bypass fetching).

    Returns
    -------
    spectrum : Spectrum
        Planet spectrum (wavelength in µm, flux in J/s/m2/µm) or unitless for albedos.
    """
    # Closest valid values parameters in the model grid
    T_valid, lg_valid = get_T_lg_valid(T=T_planet, lg=lg_planet, model=model, instru=instru, T_grid=T_grid, lg_grid=lg_grid)

    # Interpolates the grid in order to have the precise T_planet and lg_planet values
    if interpolated_spectrum and (T_valid != T_planet or lg_valid != lg_planet):
        return interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_planet, lg=lg_planet, model=model, albedo=albedo, instru=instru, load_path=load_path, T_grid=T_grid, lg_grid=lg_grid)

    # Load the spectrum with the closest parameters values in the model grid
    else:
        return load_spectrum(T_valid, lg_valid, model, albedo=albedo, instru=instru, load_path=load_path)



def load_albedo(T_planet, lg_planet, model="PICASO", airmass=2.5, interpolated_spectrum=True, instru=None, load_path=load_path):
    """
    Load an albedo spectrum and optionally transform it ('flat' or 'tellurics').
    see Eq.(1) of Lovis et al. (2017): https://arxiv.org/pdf/1609.03082

    Parameters
    ----------
    T_planet : float
        Planet temperature (K).
    lg_planet : float
        Planet surface gravity log10(cm/s^2).
    model : {'PICASO', 'flat', 'tellurics'}, optional
        Albedo model to return. 'PICASO' uses the geometric albedo; 'flat' sets a constant
        equal to the mean geometric albedo; 'tellurics' scales a sky-transmission curve
        to match the mean geometric albedo over its wavelength range.
    airmass : float, optional
        Airmass used for 'tellurics' transmission file name.
    interpolated_spectrum : bool, optional
        If True, interpolate the underlying albedo grid to (T, lg).
    instru : str, optional
        Instrument name (not used by 'PICASO' albedo files; included for symmetry).
    load_path : str, optional
        Base directory for spectra.

    Returns
    -------
    spectrum : Spectrum
        Unitless albedo spectrum (wavelength in µm).
    """
    # Always load the base Picaso albedo, then transform per 'model'
    albedo = load_planet_spectrum(T_planet=T_planet, lg_planet=lg_planet, model="PICASO", albedo=True, interpolated_spectrum=interpolated_spectrum, instru=instru, load_path=load_path)

    if model == "PICASO":
        return albedo

    elif model == "flat":
        albedo_geo  = np.nanmean(albedo.flux)
        albedo.flux = np.zeros_like(albedo.flux) + albedo_geo
        return albedo

    elif model == "tellurics":
        wave_tell, tell   = _load_tell_trans(airmass=airmass)
        mask_wavelength   = (albedo.wavelength >= wave_tell[0]) & (albedo.wavelength <= wave_tell[-1])
        albedo_geo        = np.nanmean(albedo.flux[mask_wavelength])
        albedo.flux       = albedo_geo / np.nanmean(tell) * tell
        albedo.wavelength = wave_tell
        albedo.R          = estimate_resolution(wavelength=albedo.wavelength)
        return albedo
    
    else:
        raise KeyError("Invalid reflected model: use 'tellurics', 'flat', or 'PICASO'.")
    


def load_star_spectrum(T_star, lg_star, model="BT-NextGen", interpolated_spectrum=True, load_path=load_path):
    """
    Load a stellar spectrum at the closest grid point or interpolate to (T_star, lg_star).

    Parameters
    ----------
    T_star : float
        Stellar effective temperature (K).
    lg_star : float
        Surface gravity log10(cm/s^2).
    model : {'BT-NextGen', 'Husser'}, optional
        Stellar model family.
    interpolated_spectrum : bool, optional
        If True, interpolates to the exact (T, lg). If False, returns nearest grid spectrum.
    load_path : str, optional
        Base directory.

    Returns
    -------
    spectrum : Spectrum
        Stellar spectrum with wavelength in µm and flux in J/s/m2/µm.
    """
    # Closest valid values parameters in the model grid
    T_valid, lg_valid = get_T_lg_valid(T=T_star, lg=lg_star, model=model, instru=None, T_grid=None, lg_grid=None)

    # Interpolates the grid in order to have the precise T_star and lg_star values
    if interpolated_spectrum and (T_valid != T_star or lg_valid != lg_star):
        return interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_star, lg=lg_star, model=model, albedo=False, instru=None, load_path=load_path, T_grid=None, lg_grid=None)

    # Load the spectrum with the closest parameters values in the model grid
    else:
        return load_spectrum(T_valid, lg_valid, model, albedo=False, instru=None, load_path=load_path)



@lru_cache(maxsize=3)
def load_vega_spectrum(vega_path=vega_path):
    """
    Load the Vega reference spectrum (for photometric calibration).

    Parameters
    ----------
    vega_path : str, optional
        Path to a 2-column table containing wavelength [nm] and flux [erg/s/cm^2/A].

    Returns
    -------
    vega_spectrum : Spectrum
        Vega spectrum with wavelength in µm and flux in J/s/m2/µm.
    """
    f             = fits.getdata(os.path.join(vega_path))
    wave          = f[:, 0]*1e-3 # nm => µm
    flux          = f[:, 1]*10   # 10 = 1e4 * 1e4 * 1e-7: erg/s/cm2/A -> erg/s/cm2/µm -> erg/s/m2/µm -> J/s/m2/µm
    vega_spectrum = Spectrum(wave, flux)
    return vega_spectrum
        


#######################################################################################################################
############################################# Spectra on instru and band: #############################################
#######################################################################################################################

def get_spectrum_instru(band0, R, config_data, mag, spectrum):
    """
    Restrict a spectrum to the instrument wavelength range and scale it to match
    the given Vega magnitude in 'band0'. Return both the photon-rate spectrum
    (photons/min) over the instrument range and the corresponding energy-density
    spectrum (J/s/m2/µm) sampled on the same grid.

    Notes
    -----
    - The input 'spectrum' **must** be in energy density units (J/s/m2/µm).
    - Magnitude is assumed to be Vega-based; the zero point comes from 'load_vega_spectrum()'.
    - No in-place modification is performed on the input 'spectrum'.

    Parameters
    ----------
    band0 : str
        Band in which the input magnitude is defined. Use '"instru"' to
        use the full instrument range; otherwise provide a band key like "J", "H", etc.
    R : float
        Reference resolution used to construct intermediate wavelength grids.
        Should be comfortably higher than the instrument resolving power.
    config_data : dict
        Instrument configuration dict containing:
        - "name"
        - "lambda_range": {"lambda_min", "lambda_max"}
        - "telescope": {"area": ...}
        (and other fields used downstream by 'set_nbphotons_min').
    mag : float
        Vega magnitude to impose in 'band0'.
    spectrum : Spectrum
        Input spectrum in J/s/m2/µm.

    Returns
    -------
    spectrum_instru : Spectrum
        Spectrum restricted to the instrument range and converted to photons/min.
    spectrum_density : Spectrum
        Same wavelength grid as 'spectrum_instru', but in J/s/m2/µm (energy density).

    Raises
    ------
    KeyError
        If 'band0' cannot be resolved to a wavelength range via globals.
    """
    # Resolve the reference band limits for magnitude scaling
    try:
        if band0 == "instru":
            lmin_band0 = globals()[f"lmin_{config_data['name']}"]
            lmax_band0 = globals()[f"lmax_{config_data['name']}"]
        else:
            lmin_band0 = globals()[f"lmin_{band0}"]
            lmax_band0 = globals()[f"lmax_{band0}"]
    except Exception:
        raise KeyError(f"{band0} is not a recognized band. Choose among: {bands} or 'instru' for full instrument range.")
    
    # Build an intermediate grid on band0 to compute the Vega scaling ratio: F_obj = F_vega * 10^{-0.4*mag}
    dl_band0       = (lmax_band0 + lmin_band0)/2 / (2*R)                                   # Delta lambda [µm]
    wave_band0     = np.arange(lmin_band0, lmax_band0, dl_band0)                           # Wavelength array on band0 [µm]
    spectrum_band0 = spectrum.interpolate_wavelength(wave_band0, renorm=False)             # Interpolating the input spectrum on band0
    vega_band0     = load_vega_spectrum().interpolate_wavelength(wave_band0, renorm=False) # Getting the vega spectrum
    mean_spectrum  = np.nanmean(spectrum_band0.flux)
    mean_vega      = np.nanmean(vega_band0.flux)
    if not np.isfinite(mean_spectrum) or mean_spectrum <= 0.0:
        raise ValueError("Mean flux of the input spectrum over the magnitude band is invalid (NaN/<=0).")
    if not np.isfinite(mean_vega) or mean_vega <= 0.0:
        raise ValueError("Mean flux of Vega over the magnitude band is invalid (NaN/<=0).")
    ratio = mean_vega*10**(-0.4*mag) / mean_spectrum # Ratio by which to adjust the spectrum flux in order to have the input magnitude
    
    # Conversion to photons/mn + restriction of spectra to instrumental range + adjustment of spectra to the input magnitude
    lmin_instru           = 0.98*config_data["lambda_range"]["lambda_min"] # Lambda min [µm]
    lmax_instru           = 1.02*config_data["lambda_range"]["lambda_max"] # Lambda max [µm]
    dl_instru             = (lmax_instru + lmin_instru)/2 / (2*R)          # Delta lambda [µm]
    wave_instru           = np.arange(lmin_instru, lmax_instru, dl_instru) # Constant and linear wavelength array on the instrumental bandwidth with equivalent resolution than the raw one
    spectrum_scaled       = spectrum.copy()
    spectrum_scaled.flux *= ratio                                                      # Adjusting the spectrum to the input magnitude
    spectrum_density      = spectrum_scaled.interpolate_wavelength(wave_instru, renorm=False) # In order to have a spectrum in density (i.e. J/s/m2/µm)     
    spectrum_instru       = spectrum_scaled.set_nbphotons_min(config_data, wave_instru)       # J/s/m2/µm => photons/mn on the instrumental bandwidth
    
    return spectrum_instru, spectrum_density # in ph/mn and J/s/m2/µm respectively



def get_spectrum_band(config_data, band, spectrum_instru):
    """
    Restrict a (photons/min) instrument-spectrum to a given instrument band
    and degrade its resolution to the band's resolving power (if any).

    Parameters
    ----------
    config_data : dict
        Instrument configuration dict with a 'gratings' entry mapping band names
        to GratingInfo(lmin, lmax, R).
    band : str
        Instrument band key (e.g., 'H', 'G235H_F170LP', etc.).
    spectrum_instru : Spectrum
        Spectrum already on the instrument wavelength range and in photons/min.

    Returns
    -------
    spectrum_band : Spectrum
        Band-restricted and resolution-degraded spectrum in photons/min.

    Raises
    ------
    KeyError
        If 'band' is not present in 'config_data['gratings']'.
    ValueError
        If the computed sampling step is invalid.
    """
    if band not in config_data["gratings"]:
        raise KeyError(f"Band '{band}' is not defined in config_data['gratings']: {[band for band in config_data['gratings']]}.")
    
    lmin_band = config_data['gratings'][band].lmin       # Lambda_min of the considered band [µm]
    lmax_band = config_data['gratings'][band].lmax       # Lambda_max of the considered band [µm]
    R_band    = config_data['gratings'][band].R          # Spectral resolution of the band
    dl_band   = (lmin_band+lmax_band)/2 / (2*R_band)     # Delta lambda [µm]
    wave_band = np.arange(lmin_band, lmax_band, dl_band) # Constant and linear wavelength array on the considered band
    
    return spectrum_instru.degrade_resolution(wave_band, renorm=True) # Degrated spectrum in ph/mn



#######################################################################################################################
############################################# FastYield part: #########################################################
#######################################################################################################################

def make_pretty_table(ax, params, bbox):
    tbl = ax.table(cellText=list(params.items()), colLabels=["Parameter", "Value"], cellLoc='left', colLoc='left', bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#bbbbbb')
        cell.set_linewidth(0.5)
    return tbl



def get_blackbody(wave, Teff):
    """
    Planck spectral radiance B_λ(λ, T) evaluated at wavelengths 'wave' (in µm).

    This returns the monochromatic *radiance* per unit wavelength (not multiplied
    by π). If you need the flux from a uniform disk, multiply by π and any
    dilution factor (e.g., (R / d)^2) downstream.

    Parameters
    ----------
    wave : array-like or float
        Wavelength(s) in microns (µm). Must be > 0.
    Teff : float
        Effective temperature in Kelvin. Must be > 0.

    Returns
    -------
    B_lambda : np.ndarray
        Spectral radiance in J / s / m2 / µm / sr (after unit conversion from SI),
        returned as a NumPy array with the same shape as 'wave'.

    Notes
    -----
    - Uses a numerically stable formulation with ''np.expm1''.
    - Invalid inputs (non-positive wavelength or temperature) yield NaNs.
    """
    w = wave * u.micron
    BB = (2 * const.h * const.c ** 2 / w ** 5) / np.expm1(const.h * const.c / (w * const.k_B * Teff * u.K))
    return BB.to(u.J / u.s / u.m**2 / u.micron).value



def get_thermal_reflected_spectrum(planet, thermal_model="BT-Settl", reflected_model="PICASO", instru=None, wave_instru=None, wave_K=None, vega_spectrum_K=None, show=True, in_im_mag=True):
    """
    Build the planet's *thermal* and *reflected* spectra (plus the host-star spectrum)
    over an instrument-like wavelength grid, optionally visualize the components,
    and return:
        (planet_total, planet_thermal, planet_reflected, star)

    Notes
    -----
    - All spectra are returned in *energy density* units (J/s/m2/µm), not photons.
    - Vega-based magnitudes in K band are used for renormalization.
    - If both 'thermal_model' and 'reflected_model' are "None", a clear error is raised.
    - No in-place modification is performed on input arguments.

    Parameters
    ----------
    planet : dict
        Planet/host properties (FastYield-style). Must contain the following keys
        (with '.value' where noted), and quantities must be convertible to floats:
        - "StarTeff".value, "StarLogg".value, "StarVsini".value,
          "StarKmag", "StarRadialVelocity".value,
          "StarRadius", "Distance"
        - "PlanetTeq".value, "PlanetLogg".value, "PlanetVsini".value,
          "PlanetRadialVelocity".value, "PlanetName", "PlanetType",
          "PlanetRadius", "SMA", "g_alpha"
        - Optional: "DiscoveryMethod" (string), "PlanetKmag(thermal+reflected)"
    thermal_model : str, optional
        Name of the thermal-emission model ("BT-Settl", "Exo-REM", ... or "None").
    reflected_model : str, optional
        Name of the reflected-light model ("PICASO", "flat", "tellurics", ... or "None").
    instru : str, optional
        Instrument name. If 'wave_instru' is not provided, it will be derived
        from this instrument's 'lambda_range' with a very high sampling ('R0_max').
    wave_instru : ndarray, optional
        Custom wavelength grid [µm] for the output spectra. If None, it is built from
        'instru' using high sampling between 0.98×λ_min and 1.02×λ_max.
    wave_K : ndarray, optional
        K-band wavelength grid [µm] for magnitude computations. If None, a dense grid
        is created using 'lmin_K'–'lmax_K' at R=10,000.
    vega_spectrum_K : Spectrum, optional
        Vega spectrum resampled on 'wave_K'. If None, it is loaded and resampled.
    show : bool, optional
        If True, display diagnostic plots (components, blackbodies, contrast).
    in_im_mag : bool, optional
        If True and the discovery method is "Imaging" and a K magnitude for the planet
        is known, rescale the planet components to match that observed K magnitude.

    Returns
    -------
    planet_spectrum : Spectrum
        Planet total spectrum (thermal + reflected) on 'wave_instru'.
    planet_thermal : Spectrum
        Thermal component on 'wave_instru' (zeros if thermal_model == "None").
    planet_reflected : Spectrum
        Reflected component on 'wave_instru' (zeros if reflected_model == "None").
    star_spectrum : Spectrum
        Star spectrum on 'wave_instru' (renormalized to the input K magnitude,
        rotationally broadened, and Doppler-shifted).

    Raises
    ------
    KeyError
        If both 'thermal_model' and 'reflected_model' are "None", or if 'instru'
        is missing while 'wave_instru' is None.
    ValueError
        If required inputs cannot be converted to finite floats.
    """
    
    # Sanity checks
    if thermal_model == "None" and reflected_model == "None":
        raise KeyError("Define at least one component: thermal_model or reflected_model must not be 'None'.")
        
    # Build K-band grid if needed (photometric only)
    if wave_K is None:
        R_K    = R0_min
        dl_K   = (lmin_K + lmax_K)/2 / (2*R_K)
        wave_K = np.arange(lmin_K, lmax_K, dl_K)
        
    # Build instrument grid if needed
    if wave_instru is None:
        if instru is None:
            raise KeyError("Either provide 'wave_instru' or an 'instru' name to derive it.")
        config_data = get_config_data(instru)
        lmin_instru = config_data["lambda_range"]["lambda_min"] # [µm]
        lmax_instru = config_data["lambda_range"]["lambda_max"] # [µm]
        R_instru    = R0_max
        dl_instru   = ((lmin_instru+lmax_instru)/2) / (2*R_instru)
        wave_instru = np.arange(0.98*lmin_instru, 1.02*lmax_instru, dl_instru)
    
    # Vega on K band
    if vega_spectrum_K is None:
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)

                
    # Host star spectrum (load → resample → broaden → renormalize to K mag)
    T_star     = planet["StarTeff"].value  # [K]
    lg_star    = planet["StarLogg"].value  # [dex(cm/s2)]
    Vsini_star = planet["StarVsini"].value # [km/s]
    mag_star_K = planet["StarKmag"].value  # [no unit]
    star_spectrum   = load_star_spectrum(T_star=T_star, lg_star=lg_star)              # [J/s/m2/µm]
    star_spectrum_K = star_spectrum.interpolate_wavelength(wave_K, renorm=False)      # [J/s/m2/µm]
    star_spectrum   = star_spectrum.interpolate_wavelength(wave_instru, renorm=False) # [J/s/m2/µm]
    star_spectrum   = star_spectrum.broad(Vsini_star)                                 # [J/s/m2/µm]

    # Vega-based scaling in K:
    mean_vega_K = np.nanmean(vega_spectrum_K.flux) # [J/s/m2/µm]
    mean_star_K = np.nanmean(star_spectrum_K.flux) # [J/s/m2/µm]
    if not np.isfinite(mean_vega_K) or mean_vega_K <= 0 or not np.isfinite(mean_star_K) or mean_star_K <= 0:
        raise ValueError("Invalid K-band means for Vega or star spectrum.")
    ratio_star_K = mean_vega_K * 10 ** (-0.4 * mag_star_K) / mean_star_K
    star_spectrum.flux   *= ratio_star_K # [J/s/m2/µm]
    star_spectrum_K.flux *= ratio_star_K # [J/s/m2/µm]
    
    # Thermal emission (energy density, dilution by (R/d)^2)
    T_planet        = planet["PlanetTeq"].value                # [K]
    lg_planet       = planet["PlanetLogg"].value               # [dex(cm/s2)]
    R_planet        = planet["PlanetRadius"]                   # [R_earth]
    distance        = planet["Distance"]                       # [pc]
    dilution_planet = (R_planet/distance).decompose().value**2 # [no unit]
    if thermal_model != "None":
        planet_thermal   = load_planet_spectrum(T_planet=T_planet, lg_planet=lg_planet, model=thermal_model, interpolated_spectrum=True) # [J/s/m2/µm]
        planet_thermal_K = planet_thermal.interpolate_wavelength(wave_K,      renorm=False) # [J/s/m2/µm]
        planet_thermal   = planet_thermal.interpolate_wavelength(wave_instru, renorm=False) # [J/s/m2/µm]
        planet_thermal_K.flux *= dilution_planet # [J/s/m2/µm]
        planet_thermal.flux   *= dilution_planet # [J/s/m2/µm]
    else:
        planet_thermal_K = Spectrum(wave_K,      np.zeros_like(wave_K),      R0_min,          T_planet, lg_planet, thermal_model)
        planet_thermal   = Spectrum(wave_instru, np.zeros_like(wave_instru), star_spectrum.R, T_planet, lg_planet, thermal_model)
    
    # Reflected light (energy density): F_planet_reflected(λ) = F_star(λ) * A(λ) * g(α) * (Rp/SMA)^2
    g_alpha        = planet["g_alpha"].value             # [no unit] Lambert phase function
    SMA            = planet["SMA"]                       # [AU]
    scaling_planet = (R_planet/SMA).decompose().value**2 # [no unit]
    if reflected_model != "None":
        albedo   = load_albedo(T_planet=T_planet, lg_planet=lg_planet, model=reflected_model, interpolated_spectrum=True) # [no unit]
        albedo_K = albedo.interpolate_wavelength(wave_K,      renorm=False) # [no unit]
        albedo   = albedo.interpolate_wavelength(wave_instru, renorm=False) # [no unit]
        planet_reflected_K = star_spectrum_K.flux * albedo_K.flux * g_alpha * scaling_planet # [J/s/m2/µm]
        planet_reflected   = star_spectrum.flux   * albedo.flux   * g_alpha * scaling_planet # [J/s/m2/µm]
    else:
        planet_reflected_K = np.zeros_like(wave_K)
        planet_reflected   = np.zeros_like(wave_instru)
    planet_reflected_K = Spectrum(wave_K,      np.nan_to_num(planet_reflected_K), star_spectrum_K.R, T_planet, lg_planet, reflected_model)
    planet_reflected   = Spectrum(wave_instru, np.nan_to_num(planet_reflected),   star_spectrum.R,   T_planet, lg_planet, reflected_model)
    
    # Total planet spectrum (thermal + reflected)
    planet_spectrum_K = Spectrum(wave_K,      planet_thermal_K.flux + planet_reflected_K.flux, R0_min,                                    T_planet, lg_planet, thermal_model+"+"+reflected_model)
    planet_spectrum   = Spectrum(wave_instru, planet_thermal.flux   + planet_reflected.flux,   max(planet_thermal.R, planet_reflected.R), T_planet, lg_planet, thermal_model+"+"+reflected_model)
    
    # Rotational broadening of planet spectra
    Vsini_planet = planet["PlanetVsini"].value # [km/s]
    if thermal_model != "None":
        planet_thermal = planet_thermal.broad(Vsini_planet) # [J/s/m2/µm]
    if reflected_model != "None":
        planet_reflected = planet_reflected.broad(Vsini_planet) # [J/s/m2/µm]
    planet_spectrum = planet_spectrum.broad(Vsini_planet) # [J/s/m2/µm]
    
    # Doppler shifts
    rv_star       = planet["StarRadialVelocity"].value   # [km/s]
    rv_planet     = planet["PlanetRadialVelocity"].value # [km/s]
    star_spectrum = star_spectrum.doppler_shift(rv_star) # [J/s/m2/µm]
    if thermal_model != "None":
        planet_thermal = planet_thermal.doppler_shift(rv_planet) # [J/s/m2/µm]
    if reflected_model != "None":
        planet_reflected = planet_reflected.doppler_shift(rv_planet) # [J/s/m2/µm]
    planet_spectrum = planet_spectrum.doppler_shift(rv_planet) # [J/s/m2/µm]
    
    # Enforce observed K magnitude for directly imaged planets (if known and available)
    if in_im_mag and (planet["DiscoveryMethod"] == "Imaging") and (thermal_model != "None"):
        mag_planet_K = planet["PlanetKmag(thermal+reflected)"].value
        if np.isfinite(mag_planet_K):
            mean_planet_K = np.nanmean(planet_spectrum_K.flux)
            if not np.isfinite(mean_planet_K) or mean_planet_K <= 0:
                raise ValueError("Invalid K-band mean for planet spectrum.")
            ratio_planet_K = mean_vega_K * 10 ** (-0.4 * mag_planet_K) / mean_planet_K
            planet_spectrum.flux    *= ratio_planet_K
            planet_thermal.flux     *= ratio_planet_K
            planet_reflected.flux   *= ratio_planet_K
            planet_spectrum_K.flux  *= ratio_planet_K
            planet_thermal_K.flux   *= ratio_planet_K
            planet_reflected_K.flux *= ratio_planet_K
    
    # Visualization (optional)
    if show:
        
        # K-band magnitudes
        mag_planet_total_K = get_mag(flux_obs=planet_spectrum_K.flux, flux_ref=mean_vega_K)
        if thermal_model != "None":
            mag_planet_thermal_K = get_mag(flux_obs=planet_thermal_K.flux, flux_ref=mean_vega_K)
        if reflected_model != "None":
            mag_planet_reflected_K = get_mag(flux_obs=planet_reflected_K.flux, flux_ref=mean_vega_K)
        
        # Blackbodies (for reference only)
        bb_star  = get_blackbody(wave_instru, T_star) # [J/s/m2/µm]
        bb_star *= np.nanmean(star_spectrum.flux) / np.nanmean(bb_star)
        if thermal_model != "None":
            bb_planet_thermal  = get_blackbody(wave_instru, T_planet) # [J/s/m2/µm]
            bb_planet_thermal *= np.nanmean(planet_thermal.flux) / np.nanmean(bb_planet_thermal)
        else:
            bb_planet_thermal = np.zeros_like(wave_instru)
        if reflected_model != "None":
            bb_planet_reflected = bb_star * np.nanmean(planet_reflected.flux) / np.nanmean(bb_star) # [J/s/m2/µm]
        else:
            bb_planet_reflected = np.zeros_like(wave_instru)
        bb_planet = bb_planet_thermal + bb_planet_reflected
        
        # Plot
        fig       = plt.figure(figsize=(13.5, 9.5), dpi=300, constrained_layout=False)
        gs        = gridspec.GridSpec(2, 2, width_ratios=[3.7, 1.3], height_ratios=[1, 1], wspace=0.1, hspace=0.2)
        ax_star   = fig.add_subplot(gs[0, 0])
        ax_planet = fig.add_subplot(gs[1, 0], sharex=ax_star)
        ax_table  = fig.add_subplot(gs[:, 1])
        ax_table.axis("off")
        ax_star.plot(wave_instru, star_spectrum.flux, c="#d62728", label=f"Star ({star_spectrum.model}), K = {round(mag_star_K, 1):.1f}")
        ax_star.plot(wave_instru, bb_star, c="black", ls="--", label="Blackbody (thermal)")
        if np.nanmax(wave_instru) > lmin_K and np.nanmin(wave_instru) < lmax_K: # Highlight K-band range if covered
            ax_star.axvspan(lmin_K, lmax_K, color="gray", alpha=0.2, lw=0)
        ax_star.set_yscale("log")
        ax_star.set_xlim(wave_instru[0], wave_instru[-1])
        ax_star.set_ylabel("Flux [J/s/µm/m2]", fontsize=14)
        ax_star.set_title(f"Star spectrum", fontsize=16, fontweight='bold', pad=10)
        ax_star.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_star.minorticks_on()
        ax_star.legend(loc="upper right", fontsize=12, frameon=True, edgecolor='0.3')
        if thermal_model != "None":
            ax_planet.plot(wave_instru, planet_thermal.flux, c="#d62728", alpha=0.7, label=f"Thermal ({thermal_model}), K = {round(mag_planet_thermal_K, 1):.1f}")
        if reflected_model != "None":
            ax_planet.plot(wave_instru, planet_reflected.flux, c="#1f77b4", alpha=0.7, label=f"Reflected ({star_spectrum.model}+{reflected_model}), K = {round(mag_planet_reflected_K, 1):.1f}")
        if thermal_model != "None":
            ax_planet.plot(wave_instru, bb_planet_thermal, c="black", ls="--", label="Blackbody (thermal)")
        if reflected_model != "None":
            ax_planet.plot(wave_instru, bb_planet_reflected, c="black", ls="-.", label="Blackbody (reflected)")
        ax_planet.axvspan(lmin_K, lmax_K, color="gray", alpha=0.2, lw=0)
        ax_planet.set_yscale("log")
        ax_planet.set_xlabel("Wavelength [µm]", fontsize=14)
        ax_planet.set_ylabel("Flux [J/s/µm/m2]", fontsize=14)
        ax_planet.set_title(f"Planet spectrum", fontsize=16, fontweight='bold', pad=10)
        ax_planet.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax_planet.minorticks_on()
        ax_planet.legend(loc="upper right", fontsize=12, frameon=True, edgecolor='0.3')
        ymin = max(np.nanmin(planet_spectrum.flux) / 10, np.nanmin(star_spectrum.flux) * 1e-12)
        ymax = np.nanmax(planet_spectrum.flux) * 10
        ax_planet.set_ylim(ymin, ymax)
        star_params = OrderedDict([
            (r"$R$",        f"{round(planet['StarRadius'].value, 1):.1f} $R_\\odot$"),
            (r"$M$",        f"{round(planet['StarMass'].value, 1):.1f} $M_\\odot$"),
            (r"$T_{eff}$",  f"{round(planet['StarTeff'].value, 0):.0f} K"),
            (r"$\log g$",   f"{round(planet['StarLogg'].value, 2):.2f} dex"),
            (r"$RV$",       f"{round(planet['StarRadialVelocity'].value, 1):.1f} km/s"),
            (r"$V\sin i$",  f"{round(planet['StarVsini'].value, 1):.1f} km/s"),
            ("Distance",    f"{round(planet['Distance'].value, 1):.1f} pc"),
        ])
        planet_params = OrderedDict([
            (r"$R$",            f"{round(planet['PlanetRadius'].value, 1):.1f} $R_\\oplus$"),
            (r"$M$",            f"{round(planet['PlanetMass'].value, 1):.1f} $M_\\oplus$"),
            (r"$T_{eff}$",      f"{round(planet['PlanetTeq'].value, 0):.0f} K"),
            (r"$\log g$",       f"{round(planet['PlanetLogg'].value, 2):.2f} dex"),
            (r"$RV$",           f"{round(planet['PlanetRadialVelocity'].value, 1):.1f} km/s"),
            (r"$V\sin i$",      f"{round(planet['PlanetVsini'].value, 1):.1f} km/s"),
            ("$SMA$",           f"{round(planet['SMA'].value, 2):.2f} AU"),
            (r"$sep$",          f"{round(planet['AngSep'].value, 0):.0f} mas"),
        ])
        make_pretty_table(ax_table, star_params,   bbox=[0.05, 0.55, 0.9, 0.44])
        make_pretty_table(ax_table, planet_params, bbox=[0.05, 0.0, 0.9, 0.44])
        fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.08, hspace=0.28, wspace=0.15)
        fig.suptitle(f"{planet['PlanetName']} ({planet['PlanetType']}) modelisation", fontsize=20, fontweight='bold', y=1.0)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)        
        ax.set_xlabel("Wavelength [µm]", fontsize=16, labelpad=10)
        ax.set_ylabel("Flux (contrast unit)", fontsize=16, labelpad=10)
        ax.set_title(f"Final {planet['PlanetName']} modelized spectrum", fontsize=18, pad=15, fontweight="bold")        
        ax.set_yscale('log')
        ax.set_xlim(wave_instru[0], wave_instru[-1])
        ax.plot(wave_instru, planet_spectrum.flux / star_spectrum.flux, color='forestgreen', lw=2, linestyle='-', alpha=0.7, label=f"Thermal+Reflected ({planet_spectrum.model}), K = {round(mag_planet_total_K, 1):.1f}")        
        if np.nanmax(wave_instru) > lmin_K and np.nanmin(wave_instru) < lmax_K: # Highlight K-band range if covered
            ax.axvspan(lmin_K, lmax_K, color='gray', alpha=0.2, lw=0, label="K-band")        
        min_flux = np.nanmin(planet_spectrum.flux / star_spectrum.flux)
        max_flux = np.nanmax(planet_spectrum.flux / star_spectrum.flux)
        ax.set_ylim(max(1e-12, min_flux / 10), max_flux * 10)        
        ax.legend(fontsize=13, loc='upper right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='gray')        
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.minorticks_on()
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        ax2.set_ylabel(r'$\Delta$mag', fontsize=16, labelpad=20, rotation=270)
        ax2.tick_params(axis='y')
        ax2.minorticks_on() 
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))  
        plt.tight_layout()
        plt.show()

    return planet_spectrum, planet_thermal, planet_reflected, star_spectrum



#######################################################################################################################
###################################################### PICASO PART: ###################################################
#######################################################################################################################

def import_picaso():
    """
    Imports the picaso packages
    """
    try: 
        os.environ['picaso_refdata'] = '/home/martoss/picaso/reference/'
        os.environ['PYSYN_CDBS']     = '/home/martoss/picaso/grp/redcat/trds/'
        import picaso
        from picaso import justdoit as jdi
    except ImportError:
        print("Tried importing picaso, but couldn't do it")
    return picaso, jdi



def simulate_picaso_spectrum(planet, spectrum_contributions='thermal+reflected', planet_type="gas", clouds=True, stellar_mh=0.0122, opacity=None):
    '''
    TAKEN FROM: https://github.com/planetarysystemsimager/psisim/tree/main
    A function that returns the required inputs for picaso, given a row from a universe planet table. 
    
    Inputs:
    planet - a single row, corresponding to a single planet from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice", or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    stellar_mh - stellar metalicity
    Opacity class from justdoit.opannection
    NOTE: this assumes a planet phase of 0. You can change the phase in the resulting params object afterwards.
    '''
    
    picaso, jdi = import_picaso()
    
    # Retrieve star-planet configuration
    phase =  0. # float(planet['Phase'].value) # in order to have the geometric Albedo (by definition)
    SMA = float(planet['SMA'].value)
    R_star = float(planet['StarRadius'].value)
    host_temp_list = np.hstack([np.arange(3500, 13000, 250), np.arange(13000, 50000, 1000)])
    host_logg_list = [5.00, 4.50, 4.00, 3.50, 3.00, 2.50, 2.00, 1.50, 1.00, 0.50, 0.0] # Define the grids that phoenix / ckmodel models like
    f_teff_grid    = interp1d(host_temp_list, host_temp_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    f_logg_grid    = interp1d(host_logg_list, host_logg_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    T_star         = f_teff_grid(float(planet['StarTeff'].to(u.K).value))
    lg_star        = f_logg_grid(float(planet['StarLogg'].to(u.dex(u.cm/ u.s**2)).value))
    T_planet  = float(planet['PlanetTeq'].value)
    lg_planet = float(planet['PlanetLogg'].value)
    
    params = jdi.inputs()
    params.approx(raman='none') # see justdoit.py => class inputs():
    params.phase_angle(phase)
    params.gravity(gravity=float(planet['PlanetLogg'].value), gravity_unit=planet['PlanetLogg'].physical.unit) # NOTE: picaso gravity() won't use the "gravity" input if mass and radius are provided
    if lg_star > 5.0: # The current stellar models do not like log g > 5, so we'll force it here for now. 
        lg_star = 5.0
    if T_star < 3000: # The current stellar models do not like Teff < 3000, so we'll force it here for now. 
        T_star = 3000
    params.star(opacity, T_star, stellar_mh, lg_star, radius=R_star, semi_major=SMA, semi_major_unit=planet['SMA'].unit, radius_unit=planet['StarRadius'].unit) 
    if planet_type == 'gas': #-- Define atmosphere PT profile, mixing ratios, and clouds
        params.guillot_pt(T_planet, T_int=150, logg1=-0.5, logKir=-1) # T_int = Internal temperature / logg1, logKir = see parameterization Guillot 2010
        params.channon_grid_high() # get chemistry via chemical equillibrium
        if clouds: # may need to consider tweaking these for reflected light
            params.clouds(g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5]) # g0 = Asymmetry factor / w0 = Single Scattering Albedo / opd = Total Extinction in 'dp' / p = Bottom location of cloud deck (LOG10 bars) / dp = Total thickness cloud deck above p (LOG10 bars)
    elif planet_type == 'terrestrial':
        pass # TODO: add Terrestrial type
    elif planet_type == 'ice':
        pass # TODO: add ice type
    
    if phase == 0: # non-0 phases require special geometry which takes longer to run.
        df = params.spectrum(opacity, full_output=True, calculation=spectrum_contributions, plot_opacity=False) # Perform the simple simulation since 0-phase allows simple geometry
    else:
        df1 = params.spectrum(opacity, full_output=True, calculation='thermal') # Perform the thermal simulation as usual with simple geometry
        params.phase_angle(phase, num_tangle=8, num_gangle=8) # Apply the true phase and change geometry for the reflected simulation
        df2 = params.spectrum(opacity, full_output=True, calculation='reflected')
        df = df1.copy() ; df.update(df2) # Combine the output dfs into one df to be returned
        df['full_output_therm'] = df1.pop('full_output')
        df['full_output_ref']   = df2.pop('full_output')
    
    model_wvs  = 1./df['wavenumber'] * 1e4 *u.micron
    argsort    = np.argsort(model_wvs)
    model_wvs  = model_wvs[argsort]
    model_R    = estimate_resolution(wavelength=model_wvs).value # = 30000
    
    if spectrum_contributions == "thermal":
        planet_thermal    = np.zeros((2, len(model_wvs)))
        planet_thermal[0] = model_wvs
        thermal_flux      = df["thermal"][argsort] * u.erg/u.s/u.cm**2/u.cm
        thermal_flux      = thermal_flux.to(u.J/u.s/u.m**2/u.micron)
        planet_thermal[1] = np.array(thermal_flux.value)
        fits.writeto(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{round(float(planet['PlanetTeq'].value))}K_lg{round(float(planet['PlanetLogg'].value), 1)}.fits", planet_thermal, overwrite=True)
        plt.figure(dpi=300) ; plt.plot(planet_thermal[0], planet_thermal[1]) ; plt.title(f'Thermal: T = {round(float(planet["PlanetTeq"].value))}K and lg = {round(float(planet["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength [µm]') ; plt.ylabel("flux (in J/s/µm/m2)") ; plt.yscale('log') ; plt.show()
    
    elif spectrum_contributions == "reflected":
        albedo    = np.zeros((2, len(model_wvs)))
        albedo[0] = model_wvs
        albedo[1] = df['albedo'][argsort]
        fits.writeto(f"sim_data/Spectra/planet_spectrum/albedo/PICASO/albedo_gas_giant_{round(float(planet['PlanetTeq'].value))}K_lg{round(float(planet['PlanetLogg'].value), 1)}.fits", albedo, overwrite=True)
        plt.figure(dpi=300) ; plt.plot(albedo[0], albedo[1]) ; plt.title(f'Albedo: T = {round(float(planet["PlanetTeq"].value))}K and lg = {round(float(planet["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength [µm]') ; plt.ylabel("albedo") ; plt.yscale('log') ; plt.show()

    

def get_picasso_thermal():
    from src.FastYield import load_planet_table, get_planet_index
    picaso, jdi = import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname         = 'all_opacities_0.6_6_R60000.db'
    dbname         = os.path.join(opacity_folder, dbname)
    opacity        = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx          = get_planet_index(planet_table, "HR 8799 b") # "HR 8799 b" => does not change anything
    planet       = planet_table[idx]
    T0, lg0 = get_model_grid("PICASO")
    for i in tqdm(range(len(T0))):
        T_planet = T0[i]
        for lg_planet in lg0:
            planet["PlanetTeq"]  = T_planet * planet["PlanetTeq"].unit # 
            planet["PlanetLogg"] = lg_planet * planet["PlanetLogg"].unit # 
            simulate_picaso_spectrum(planet, spectrum_contributions="thermal", opacity=opacity)


            
def get_picasso_albedo():
    from src.FastYield import load_planet_table, get_planet_index
    picaso, jdi = import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname         = 'all_opacities_0.6_6_R60000.db'
    dbname         = os.path.join(opacity_folder, dbname)
    opacity        = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx          = get_planet_index(planet_table, "HR 8799 b") # "HR 8799 b" => does not change anything
    planet       = planet_table[idx]
    T0, lg0 = get_model_grid("PICASO")
    for i in tqdm(range(len(T0))):
        T_planet = T0[i]
        for lg_planet in lg0:
            planet["PlanetTeq"]  = T_planet * planet["PlanetTeq"].unit # 
            planet["PlanetLogg"] = lg_planet * planet["PlanetLogg"].unit # 
            simulate_picaso_spectrum(planet, spectrum_contributions="reflected", opacity=opacity)
            









