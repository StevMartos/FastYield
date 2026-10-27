# import FastYield modules
from fastyield.config import rad2arcsec
from fastyield.signal_noise import compute_sigma_base_2_speck_numba, compute_sigma_base_2_al_spat_numba, compute_sigma_base_2_al_spec_fast


# import numpy modules
import numpy as np
from numpy.lib.format import open_memmap

# import other modules
from pathlib import Path
import json, hashlib



# -------------------------------------
# Helpers: meta -> stable JSON -> hash
# -------------------------------------

def _jsonable(x):
    """Convert x to a JSON-serializable object with stable float representation."""
    # numpy scalars
    if isinstance(x, np.generic):
        return x.item()
    # numpy arrays
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    # pathlib
    if isinstance(x, Path):
        return str(x)
    # floats: round to make hashing stable across minor repr differences
    if isinstance(x, float):
        return float(f"{x:.12g}")
    # containers
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x

def make_suffix(meta: dict, n=16):
    """Return (suffix, meta_clean, json_payload)."""
    meta_clean = _jsonable(meta)
    payload    = json.dumps(meta_clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    suffix     = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n]
    return suffix, meta_clean, payload

def write_meta(sim_dir: Path, suffix: str, meta_clean: dict, suffix_as_prefix: bool = False):
    if suffix_as_prefix:
        meta_path = sim_dir / f"{suffix}_meta.json"
    else:
        meta_path = sim_dir / f"meta_{suffix}.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta_clean, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != meta_clean:
            raise RuntimeError(f"Meta mismatch for existing {meta_path} (hash collision or changed meta).")
    return meta_path


# -------------------------------------
# Helpers: simulation parameters prints
# -------------------------------------

def _fmt_scalar(x):
    """Pretty formatter for scalar values."""
    if isinstance(x, (np.floating, float)):
        x = float(x)
        ax = abs(x)
        if ax == 0:
            return "0"
        if ax < 1e-3 or ax >= 1e4:
            return f"{x:.3e}"
        return f"{x:.4g}"
    if isinstance(x, (np.integer, int)):
        return f"{int(x)}"
    return str(x)

def _fmt_axis(arr, unit="", log_hint=False):
    """Pretty summary of a 1D parameter axis."""
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return "empty"
    if n == 1:
        return f"fixed at {_fmt_scalar(arr[0])}{unit}"
    spacing = "log" if log_hint else "lin"
    return f"{n:>3d} values | {spacing:>3s} | [{_fmt_scalar(arr[0])}, {_fmt_scalar(arr[-1])}]{unit}"

def _print_section(title, char="="):
    line = char * 72
    print(f"\n{line}\n{title}\n{line}")

def _print_kv(key, value, width=38):
    print(f"{key:<{width}} : {value}")

def print_simulation_summary(instru, instru_type, detector, post_processing, D, S, N_mirror, trans_dust, RON0, RON_lim, DC0, saturation_e, min_DIT, max_DIT, coronagraph, apodizer, strehl, thermal_model, reflected_model, SNR_thr, exposure_time, spatial_sampling, A_FWHM, Rc, filter_type, table_type, sep_min, sep_max, N_PT_raw, N_PT, force_new_calc, N_px=None, l0=None, R=None, Nl=None, Dl=None, WFE=None, r_WFE=None, C_raw=None, IWA=None, IWA_unit="mas", trans_instru=None, sigma_m=None, FoV=None):
    _print_section("FastYield simulation summary")

    # General configuration
    _print_kv("Instrument", f"ELT/{instru}")
    _print_kv("Instrument type", instru_type)
    _print_kv("Detector", detector)
    _print_kv("Post-processing", post_processing)
    _print_kv("Force new calculation", force_new_calc)

    # Telescope / detector
    _print_section("Telescope and detector", "-")
    _print_kv("Diameter D", f"{_fmt_scalar(D)} m")
    _print_kv("Collecting area S", f"{_fmt_scalar(S)} m²")
    _print_kv("ELT mirrors", _fmt_scalar(N_mirror))
    _print_kv("Dust transmission", _fmt_scalar(trans_dust))
    if N_px is not None:
        _print_kv("Detector linear size", f"{_fmt_scalar(N_px)} px")
    _print_kv("RON0", f"{_fmt_scalar(RON0)} e-/px/read")
    _print_kv("RON floor", f"{_fmt_scalar(RON_lim)} e-/px")
    _print_kv("Dark current", f"{_fmt_scalar(DC0)} e-/px/mn")
    _print_kv("Saturation", f"{_fmt_scalar(saturation_e)} e-/px")
    _print_kv("Minimum DIT", f"{_fmt_scalar(min_DIT)} mn")
    _print_kv("Maximum DIT", f"{_fmt_scalar(max_DIT)} mn")

    # Science setup
    _print_section("Science setup", "-")
    _print_kv("Coronagraph", coronagraph)
    _print_kv("Apodizer", apodizer)
    _print_kv("Strehl condition", strehl)
    _print_kv("Thermal model", thermal_model)
    _print_kv("Reflected model", reflected_model)
    _print_kv("Detection threshold", f"S/N >= {_fmt_scalar(SNR_thr)}")
    _print_kv("Exposure time per target", f"{_fmt_scalar(exposure_time)} mn")
    _print_kv("Spatial sampling", f"{_fmt_scalar(spatial_sampling)} px/FWHM")
    _print_kv("FWHM box area", f"{_fmt_scalar(A_FWHM)} px")

    if instru_type == "IFU":
        _print_kv("High-pass cut-off Rc", _fmt_scalar(Rc))
        _print_kv("MM filter type", filter_type)

    # Catalog parameters
    _print_section("Catalog parameters", "-")
    _print_kv("Catalog type", table_type)
    _print_kv("Separation range", f"[{_fmt_scalar(sep_min)}, {_fmt_scalar(sep_max)}] mas")
    _print_kv("Initial catalog size", f"{_fmt_scalar(N_PT_raw)} planets")
    _print_kv("Filtered catalog size", f"{_fmt_scalar(N_PT)} planets")

    # Parameter space
    _print_section("Explored parameter space", "-")
    if R is not None:
        _print_kv("R", _fmt_axis(R, log_hint=True))
    if l0 is not None:
        _print_kv("lambda_0", _fmt_axis(l0, unit=" µm", log_hint=False))
    if Nl is not None:
        _print_kv("N_lambda", _fmt_axis(Nl, log_hint=True))
    if Dl is not None:
        _print_kv("Delta_lambda", _fmt_axis(Dl, unit=" µm", log_hint=False))
    if WFE is not None:
        _print_kv("WFE", _fmt_axis(WFE, unit=" nm RMS", log_hint=False))
    if r_WFE is not None:
        _print_kv("r_WFE", _fmt_axis(r_WFE, unit="", log_hint=False))
    if C_raw is not None:
        _print_kv("C_raw", _fmt_axis(C_raw, log_hint=True))
    if IWA is not None:
        _print_kv("IWA", _fmt_axis(IWA, unit=f" {IWA_unit}", log_hint=True))
    if trans_instru is not None:
        _print_kv("trans_instru", _fmt_axis(trans_instru, unit="e-/ph", log_hint=False))
    if sigma_m is not None:
        _print_kv("sigma_m", _fmt_axis(100*np.asarray(sigma_m), unit=" %", log_hint=True))
    if FoV is not None:
        _print_kv("FoV", _fmt_axis(FoV, unit=" mas", log_hint=True))

    axes_lengths = []
    for arr in (R, l0, Nl, Dl, WFE, C_raw, IWA, trans_instru, sigma_m, FoV):
        if arr is not None:
            axes_lengths.append(len(arr))
    if axes_lengths:
        total_grid = int(np.prod(axes_lengths, dtype=np.int64))
        _print_kv("Total grid points", f"{total_grid:,}".replace(",", " "))



# ------------------------------
# Helpers: creating memmap files
# ------------------------------

def format_nbytes(nbytes):
    """Human-readable size using binary units."""
    units = ["o", "Ko", "Mo", "Go", "To"]
    size = float(nbytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

def memmap_nbytes(shape, dtype=np.float32):
    """Return theoretical file size in bytes for a memmap array."""
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)

def create_memmap_with_log(path, shape, dtype=np.float32, mode="w+"):
    """Create a memmap file and print its expected size."""
    nbytes = memmap_nbytes(shape, dtype=dtype)
    print(f"Creating {path.name:<35} | shape={shape} | dtype={np.dtype(dtype)} | size={format_nbytes(nbytes)}")
    return open_memmap(path, mode=mode, dtype=dtype, shape=shape)



# --------------
# General helper
# --------------

def interp_flat_last_axis(x, y, x0, dtype=None):
    """
    Fast linear interpolation along the last axis, with flat extrapolation.
    This is optimized for quantities sampled as (..., Nx), for example
    PSF_profile_5D[..., separation].
    """
    x0  = np.clip(x0, x[0], x[-1])
    i1  = int(np.searchsorted(x, x0, side="right"))
    i1  = int(np.clip(i1, 1, x.size - 1))
    i0  = i1 - 1
    w   = (x0 - x[i0]) / (x[i1] - x[i0])
    out = (1.0 - w) * y[..., i0] + w * y[..., i1]
    return out.astype(dtype, copy=False) if dtype is not None else out



def normalize_template_or_none(template):
    """
    Normalize a matched-filter template.

    Returns
    -------
    template_norm : ndarray or None
        Normalized template if its norm is finite and strictly positive.
        None otherwise.
    """
    template = np.asarray(template, dtype=float)
    norm2    = np.nansum(template**2)
    if (not np.isfinite(norm2)) or (norm2 <= 0):
        return None
    return template / np.sqrt(norm2)



def window_sums_1d(values, idx_lo, idx_hi):
    """
    Fast sums of a 1D array over many [idx_lo, idx_hi) windows.

    Parameters
    ----------
    values : ndarray, shape (Nwave,)
        1D array to integrate.
    idx_lo, idx_hi : ndarray
        Integer arrays with the same shape, defining integration windows (l0, Dl) for IFU or (l0, Dl) for imager.

    Returns
    -------
    sums : ndarray
        Sum of values[idx_lo:idx_hi] for each window.
    """
    values = np.asarray(values)
    if np.all(np.isfinite(values)):
        v = values
    else:
        v = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    cs    = np.empty(v.size + 1, dtype=v.dtype if np.issubdtype(v.dtype, np.floating) else float)
    cs[0] = 0.0
    np.cumsum(v, out=cs[1:])
    return cs[idx_hi] - cs[idx_lo]



def window_nanmax_1d(values, idx_lo, idx_hi):
    """
    Fast maximum of a 1D array over many [idx_lo, idx_hi) windows.
    """
    values = np.asarray(values, dtype=float)
    out    = np.zeros(np.shape(idx_lo), dtype=float)
    for ind in np.ndindex(out.shape):
        lo = int(idx_lo[ind])
        hi = int(idx_hi[ind])
        if hi <= lo:
            out[ind] = 0.0
            continue
        v = values[lo:hi]
        m = np.isfinite(v)
        if np.any(m):
            out[ind] = np.nanmax(v[m])
        else:
            out[ind] = 0.0
    return out



def window_sigma_al_spec_1d(g, norm2, idx_lo, idx_hi):
    """
    Compute the spectral-aliasing covariance term for all spectral windows.

    Equivalent to compute_sigma_base_2_al_spec_fast(g[lo:hi] / sqrt(norm2))
    for each valid window, up to floating-point rounding.
    """
    g         = np.nan_to_num(np.asarray(g), nan=0.0, posinf=0.0, neginf=0.0)
    s0        = window_sums_1d(g**2, idx_lo, idx_hi)
    s1        = window_sums_1d(g[:-1] * g[1:], idx_lo, np.maximum(idx_lo, idx_hi - 1))
    s2        = window_sums_1d(g[:-2] * g[2:], idx_lo, np.maximum(idx_lo, idx_hi - 2))
    numerator = s0 + 0.5 * s1 + (2.0 / 256.0) * s2
    return np.divide(np.maximum(numerator, 0.0), norm2, out=np.zeros_like(norm2, dtype=float), where=np.isfinite(norm2) & (norm2 > 0))



def get_valid_IFU_windows(R, l0, Dl, Nl_max, Nl_min=3):
    """
    Return the validity mask for the (R, l0, Dl) parameter grid.

    The spectral sampling is the same as in process_IFU:
    dln(lambda) = 1 / (2*R), corresponding to 2 pixels per
    spectral resolution element.

    Returns
    -------
    valid_windows_3D : bool array, shape (R, l0, Dl)
    """
    lmin = l0[:, None] - Dl[None, :] / 2
    lmax = l0[:, None] + Dl[None, :] / 2
    if np.any(lmin <= 0):
        raise ValueError("IFU spectral windows must have positive wavelengths.")
    # Same wavelength-grid origin as in the main IFU calculations.
    lmin_model       = 0.98 * (l0.min() - Dl.max()/2)
    lmax_model       = 1.02 * (l0.max() + Dl.max()/2)
    valid_windows_3D = np.zeros((len(R), len(l0), len(Dl)), dtype=bool)
    print(f"Valid windows: {Nl_min} <= Nl <= {Nl_max if Nl_max is not None else 'unlimited'}")
    for i, res in enumerate(R):
        dln        = 1.0 / (2.0 * res)
        n          = int(np.floor(np.log(lmax_model/lmin_model)/dln)) + 1
        wave_res   = lmin_model * np.exp(np.arange(n)*dln)
        idx_lo     = np.searchsorted(wave_res, lmin, side="left")
        idx_hi     = np.searchsorted(wave_res, lmax, side="right")
        n_channels = idx_hi - idx_lo
        valid      = n_channels >= Nl_min
        if Nl_max is not None:
            valid &= n_channels <= Nl_max
        valid_windows_3D[i] = valid
        print(f"R={res:.0f}: {np.count_nonzero(valid)}/{valid.size} valid configurations")
    return valid_windows_3D



def sigma_RON_2_from_DIT(DIT, min_DIT, RON0, RON_lim, A_FWHM, dtype, NbPixel=None, is_UTR=True):
    """
    Effective readout-noise variance in [e-/FWHM/DIT]**2.

    UTR detectors     : up-the-ramp read-noise reduction.
    Non-UTR detectors : one effective read-noise contribution per DIT.

    Detector parameters and is_UTR can be scalars or arrays
    broadcastable to DIT.shape.
    """
    DIT     = np.asarray(DIT, dtype=dtype)
    min_DIT = np.broadcast_to(np.asarray(min_DIT, dtype=dtype), DIT.shape)
    RON0    = np.broadcast_to(np.asarray(RON0,    dtype=dtype), DIT.shape)
    RON_lim = np.broadcast_to(np.asarray(RON_lim, dtype=dtype), DIT.shape)
    is_UTR  = np.broadcast_to(np.asarray(is_UTR,  dtype=bool),  DIT.shape)

    valid              = np.isfinite(DIT) & (DIT > 0)
    sigma_RON_2        = np.full(DIT.shape, np.nan, dtype=dtype)
    sigma_RON_2[valid] = RON0[valid]**2
    mask_UTR           = valid & is_UTR & (DIT >= 2 * min_DIT)
    if np.any(mask_UTR):
        N_read                = np.floor(DIT[mask_UTR] / min_DIT[mask_UTR])
        sigma_RON_2[mask_UTR] = RON0[mask_UTR]**2 * 12 * (N_read - 1) / (N_read * (N_read + 1)) + RON_lim[mask_UTR]**2

    sigma_RON_2 *= A_FWHM
    if NbPixel is not None:
        sigma_RON_2 *= NbPixel

    return sigma_RON_2



def effective_number_of_DIT(DIT, exposure_time, dtype):
    """Return floor(exposure_time / DIT) for valid DIT, preserving NaNs."""
    DIT      = np.asarray(DIT)
    n        = np.full(DIT.shape, np.nan, dtype=dtype)
    valid    = np.isfinite(DIT) & (DIT > 0)
    n[valid] = np.maximum(np.floor(exposure_time / DIT[valid]), 1.0)
    return n



def warmup_signal_noise_numba(dtype):
    """
    Compile/load the Numba covariance kernels before multiprocessing.
    On Linux, the compiled functions are inherited by forked workers, which avoids
    paying the compilation/loading overhead independently in every worker.
    """
    wave = np.linspace(1.0, 1.1, 32, dtype=dtype)
    h    = np.ones((32, 1), dtype=dtype)
    sep  = np.array([50.0 / (rad2arcsec * 1000.0)], dtype=dtype)
    _    = compute_sigma_base_2_speck_numba(h=h, wave=wave, separation_rad=sep, D=38.54)
    _    = compute_sigma_base_2_al_spat_numba(h=h, wave=wave, separation_rad=sep, pxscale_rad=1.0 / (rad2arcsec * 1000.0))
    _    = compute_sigma_base_2_al_spec_fast(h=h)


