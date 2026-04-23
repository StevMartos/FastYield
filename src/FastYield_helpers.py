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

def write_meta(sim_dir: Path, suffix: str, meta_clean: dict):
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

def print_simulation_summary(instru, instru_type, detector, post_processing, D, S, N_mirror, trans_dust, RON0, RON_lim, DC0, saturation_e, min_DIT, max_DIT, N_px, coronagraph, apodizer, strehl, thermal_model, reflected_model, SNR_thr, exposure_time, size_core, A_FWHM, Rc, filter_type, rho_m, table_type, band_regime, light_regime, sep_min, sep_max, N_PT_raw, N_PT, force_new_calc, l0=None, R=None, Nl=None, Dl=None, WFE=None, IWA=None, trans_instru=None, sigma_m=None, FoV=None):
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
    _print_kv("Spatial sampling", f"{_fmt_scalar(size_core)} px/FWHM")
    _print_kv("FWHM box area", f"{_fmt_scalar(A_FWHM)} px")

    if instru_type == "IFU":
        _print_kv("High-pass cut-off Rc", _fmt_scalar(Rc))
        _print_kv("MM filter type", filter_type)
        if post_processing == "MM":
            _print_kv("Systematics spectral correlation rho_m", _fmt_scalar(rho_m))
        elif post_processing == "DI":
            _print_kv("Speckles spectral correlation rho_m", _fmt_scalar(rho_m))

    # Catalog parameters
    _print_section("Catalog parameters", "-")
    _print_kv("Catalog type", table_type)
    _print_kv("Regime selection", light_regime)
    _print_kv("Regime reference band", band_regime)
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
    if IWA is not None:
        _print_kv("IWA", _fmt_axis(IWA, unit=" mas", log_hint=True))
    if trans_instru is not None:
        _print_kv("gamma_instru", _fmt_axis(trans_instru, unit="e-/ph", log_hint=False))
    if sigma_m is not None:
        _print_kv("sigma_m", _fmt_axis(100*np.asarray(sigma_m), unit=" %", log_hint=True))
    if FoV is not None:
        _print_kv("FoV", _fmt_axis(FoV, unit=" mas", log_hint=True))

    axes_lengths = []
    for arr in (R, l0, Nl, Dl, WFE, IWA, trans_instru, sigma_m, FoV):
        if arr is not None:
            axes_lengths.append(len(arr))
    if axes_lengths:
        total_grid = int(np.prod(axes_lengths, dtype=np.int64))
        _print_kv("Total grid points", f"{total_grid:,}".replace(",", " "))



# -------------------------------------
# Helpers: creating memmap files
# -------------------------------------

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

