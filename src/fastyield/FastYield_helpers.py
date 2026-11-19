# import FastYield modules
from .config import rad2arcsec, dpi_fig, SNR_thresh, planet_types, get_planet_table_path, planet_table_path
from .signal_noise import compute_sigma_base_2_speck_numba, compute_sigma_base_2_al_spat_numba, compute_sigma_base_2_al_spec_fast
from .get_specs import get_detector_specs

# import numpy modules
import numpy as np
from numpy.lib.format import open_memmap

# import astropy modules
from astropy import constants as const
from astropy.table import QTable
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, SkyCoord

# import scipy modules
from scipy.interpolate import RegularGridInterpolator

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl

# import other modules
from pathlib import Path
import json, hashlib
import pandas as pd



#######################################################################################################################
#################################################### Utils function: #################################################
#######################################################################################################################

class PlainLogFormatter(mpl.ticker.LogFormatter):
    """Log-axis tick formatter labelling the same ticks as matplotlib's default one, but in plain number notation"""
    def __call__(self, x, pos=None):
        return np.format_float_positional(x, precision=6, unique=False, fractional=False, trim="-") if super().__call__(x, pos) else ""



def mass_to_size(mass, s0, ds, mass_min=None, mass_max=None):
    if mass_min is None:
        mass_min = np.nanmin(mass[np.isfinite(mass) & (mass > 0)])
    if mass_max is None:
        mass_max = np.nanmax(mass[np.isfinite(mass) & (mass > 0)])
    t = (np.log10(np.clip(mass, mass_min, mass_max)) - np.log10(mass_min)) / (np.log10(mass_max) - np.log10(mass_min))
    t = np.clip(t, 0, 1)
    return s0 + ds * t



def get_filename_table(table, instru, apodizer, strehl, coronagraph, systematics, PCA, name_model):
    """Return the <table>_Pull_*.ecsv filename (without its directory)."""
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"
    return filename



def load_planet_table(table_name, table="Archive"):
    """Read sim_data/planet_table/<table>/<table_name> as an ECSV QTable."""
    return QTable.read(get_planet_table_path(table) / table_name, format="ascii.ecsv")



def get_valid_mask(col):
    """Get a mask with the valid values of the planet_table's column"""
    return np.isfinite(col)



def get_invalid_mask(col):
    """Get a mask with the invalid values of the planet_table's column"""
    return np.logical_not(get_valid_mask(col))



def get_mask_planet_type(planet_table, planet_type):
    """Get a mask with the 'planet_type' planets"""
    planets_type = np.array(planet_table["PlanetType"], dtype=str)
    mask_ptype   = (np.char.find(np.char.lower(planets_type), planet_type.lower()) >= 0) # substring match
    return mask_ptype



def get_mask_earth(planet_table):
    """Get a mask with the Earth-like planets"""

    # --- Ranges "Earth-like"
    R_min,   R_max     = 0, 1.6 # [R_earth]
    M_min,   M_max     = 0, 10  # [M_earth]
    Teff_min, Teff_max = 0, 500 # [K]

    R          = planet_table["PlanetRadius"].value        # R_earth
    M          = planet_table["PlanetMass"].value          # M_earth
    Teff       = planet_table["PlanetTeff"].value          # K
    n          = len(planet_table)
    mask_earth = np.ones(n, dtype=bool)
    print("Filtering Earth-Like planets")

    # Radius & mass & Teff
    before      = int(mask_earth.sum())
    mask_earth &= np.isfinite(R) & (R >= R_min) & (R <= R_max)
    after       = int(mask_earth.sum())
    print(f" After radius filtering:      {after} / {n} (-{before - after})")

    before      = int(mask_earth.sum())
    mask_earth &= np.isfinite(M) & (M >= M_min) & (M <= M_max)
    after       = int(mask_earth.sum())
    print(f" After mass filtering:        {after} / {n} (-{before - after})")

    before      = int(mask_earth.sum())
    mask_earth &= np.isfinite(Teff) & (Teff >= Teff_min) & (Teff <= Teff_max)
    after       = int(mask_earth.sum())
    print(f" After temperature filtering: {after} / {n} (-{before - after})")

    return mask_earth



def get_planet_index(planet_table, planet_name):
    """Row index for a given planet name (exact match)."""
    return np.where(planet_table["PlanetName"] == planet_name)[0][0]



def get_evolutionary_model(planet_table):
    """
    Estimate (Teff, Radius, Lbol) from an evolutionary grid for each row of 'planet_table'.

    Sonora Bobcat: Marley et al. (2021, Astrophysical Journal, Volume 920, Issue 2, id.85.)
    https://zenodo.org/records/5063476

    Required columns in 'planet_table':
      - PlanetMass  [unit convertible to u.M_jup]
      - StarAge     [unit convertible to u.Gyr]

    Returns
    -------
    Teff : Quantity [K]
    R    : Quantity [R_earth]
    Lbol : Quantity [L_sun]
    """

    # ---- Inputs (mass [Mjup], age [Gyr])
    M_mj  = planet_table["PlanetMass"].to_value(u.M_jup)
    A_gyr = planet_table["StarAge"].to_value(u.Gyr)

    # ---- Load grid (cached)
    EVOL_GRID_NPZ = planet_table_path / "sonora_bobcat_evol_grid.npz"
    data          = np.load(EVOL_GRID_NPZ, allow_pickle=True, mmap_mode="r")
    m_axis        = np.asarray(data["mass_axis_mjup"], dtype=float) # (Nm,)
    a_axis        = np.asarray(data["age_axis_gyr"],   dtype=float) # (Na,)
    Tgrid         = np.asarray(data["teff_K"],         dtype=float) # (Nm, Na)
    Rgrid         = np.asarray(data["radius_Rjup"],    dtype=float) # (Nm, Na)
    Lgrid         = np.asarray(data["lbol_Lsun"],      dtype=float) # (Nm, Na)
    it_T          = RegularGridInterpolator((m_axis, a_axis), Tgrid, bounds_error=False, fill_value=np.nan)
    it_R          = RegularGridInterpolator((m_axis, a_axis), Rgrid, bounds_error=False, fill_value=np.nan)
    it_L          = RegularGridInterpolator((m_axis, a_axis), Lgrid, bounds_error=False, fill_value=np.nan)
    domain        = dict(mmin=m_axis.min(), mmax=m_axis.max(), amin=a_axis.min(), amax=a_axis.max())

    # ---- Prepare query points
    m = M_mj.copy()
    a = A_gyr.copy()
    a[(a < domain["amin"]) | (a > domain["amax"])] = np.nan
    m[(m < domain["mmin"]) | (m > domain["mmax"])] = np.nan

    # --- Valid points
    valid = np.isfinite(m) & np.isfinite(a) & (M_mj > 0) & (A_gyr > 0)

    Teff = np.full_like(M_mj, np.nan, dtype=float)
    Rjup = np.full_like(M_mj, np.nan, dtype=float)
    Lsun = np.full_like(M_mj, np.nan, dtype=float)

    if not np.any(valid):
        # Nothing valid
        return Teff * u.K, (Rjup * u.R_jup).to(u.R_earth), Lsun * u.L_sun

    pts = np.column_stack([m[valid], a[valid]])  # (N_valid, 2) with (mass, age)

    # ---- Interpolate
    Teff[valid] = it_T(pts)
    Rjup[valid] = it_R(pts)
    if it_L is not None:
        Lsun[valid] = it_L(pts)

    # ---- If Lbol not provided (or partially NaN), compute via Stefan–Boltzmann
    need_sb = ~np.isfinite(Lsun)
    if np.any(need_sb & valid):
        Tq = (Teff[need_sb & valid] * u.K)
        Rq = (Rjup[need_sb & valid] * u.R_jup)
        Lq = (4*np.pi * (Rq.to(u.m))**2 * const.sigma_sb * Tq**4).to_value(u.L_sun)
        Lsun[need_sb & valid] = Lq

    # ---- Wrap with units
    Teff_q = Teff * u.K
    R_q    = (Rjup * u.R_jup).to(u.R_earth)
    L_q    = Lsun * u.L_sun
    return Teff_q, R_q, L_q



def chen_kipping_radius_from_mass(Mp):
    """
    Chen & Kipping (2017) mass-radius relation, using the same deterministic
    piecewise prescription as NASA Exoplanet Archive PSCompPars.

    Input
    -----
    Mp : Quantity
        Planet mass, convertible to Earth masses.

    Output
    ------
    Rp : Quantity
        Planet radius, in Earth radii.
    """
    m           = np.asarray(u.Quantity(Mp).to_value(u.M_earth), dtype=float)
    r           = np.full(m.shape, np.nan, dtype=float)
    valid       = np.isfinite(m) & (m > 0)
    m1          = valid & (m < 2.04)
    m2          = valid & (m >= 2.04) & (m < 132.0)
    m3          = valid & (m >= 132.0) & (m < 26600.0)
    m4          = valid & (m >= 26600.0)
    logm        = np.full(m.shape, np.nan, dtype=float)
    logm[valid] = np.log10(m[valid])
    logr        = np.full(m.shape, np.nan, dtype=float)
    logr[m1]    =  0.00346 + 0.2790 * logm[m1]
    logr[m2]    = -0.09250 + 0.5890 * logm[m2]
    logr[m3]    =  1.25000 - 0.0440 * logm[m3]
    logr[m4]    = -2.85000 + 0.8810 * logm[m4]
    ok          = np.isfinite(logr)
    r[ok]       = 10**logr[ok]
    return r * u.R_earth


def chen_kipping_mass_from_radius(Rp):
    """
    Inverse Chen & Kipping (2017) radius-mass relation, using the same
    deterministic piecewise prescription as NASA Exoplanet Archive PSCompPars.

    Input
    -----
    Rp : Quantity
        Planet radius, convertible to Earth radii.

    Output
    ------
    Mp : Quantity
        Planet mass, in Earth masses.

    Notes
    -----
    The NASA PSCompPars prescription deliberately does not infer a mass in the
    degenerate interval 11.1 <= Rp/Rearth < 14.3.
    """
    r           = np.asarray(u.Quantity(Rp).to_value(u.R_earth), dtype=float)
    m           = np.full(r.shape, np.nan, dtype=float)
    valid       = np.isfinite(r) & (r > 0)
    r1          = valid & (r < 1.23)
    r2          = valid & (r >= 1.23) & (r < 11.1)
    r4          = valid & (r >= 14.3)
    logr        = np.full(r.shape, np.nan, dtype=float)
    logr[valid] = np.log10(r[valid])
    logm        = np.full(r.shape, np.nan, dtype=float)
    logm[r1]    = (logr[r1] - 0.00346) / 0.2790
    logm[r2]    = (logr[r2] + 0.09250) / 0.5890
    logm[r4]    = (logr[r4] + 2.85000) / 0.8810
    ok          = np.isfinite(logm)
    m[ok]       = 10**logm[ok]
    return m * u.M_earth


def _get_value(x):
    """
    Return the numerical value of an input quantity.

    This helper makes the code compatible with both plain floats and
    astropy Quantity-like objects.
    """
    return x.value if hasattr(x, "value") else x



def _distance_to_interval(x, xmin, xmax, log=False, scale=1.0):
    """
    Compute the distance of a value to an interval.

    The distance is zero if the value is inside the interval. If the value is
    outside, the distance is measured relative to the closest boundary.

    Parameters
    ----------
    x : float
        Value to test.

    xmin, xmax : float
        Lower and upper bounds of the interval.

    log : bool, optional
        If True, compute the distance in log-space. This is useful for mass
        and radius, which span several orders of magnitude.

    scale : float, optional
        Normalization scale used for linear distances.
    """
    # Inside the interval: perfect match.
    if xmin <= x <= xmax:
        return 0.0
    # Logarithmic distance, useful for mass and radius.
    if log:
        if x <= 0:
            return np.inf
        if x < xmin:
            return np.log10(xmin / x) if xmin > 0 else 0.0
        return np.log10(x / xmax) if np.isfinite(xmax) else 0.0
    # Linear distance below the interval.
    if x < xmin:
        return (xmin - x) / scale
    # Linear distance above the interval.
    return (x - xmax) / scale if np.isfinite(xmax) else 0.0



def _closest_planet_type(mass, radius, teff, planet_types_ref=planet_types):
    """
    Return the closest planet type based on mass, radius, and temperature.

    A planet inside a category box has a distance of zero to that category.
    A planet outside all boxes is attached to the closest category.
    """
    best_ptype = None
    best_score = np.inf
    for ptype, criteria in planet_types_ref.items():
        # Distance to the allowed mass range.
        dm = _distance_to_interval(mass, criteria["mass_min"], criteria["mass_max"], log=True)
        # Distance to the allowed radius range.
        dr = _distance_to_interval(radius, criteria["radius_min"], criteria["radius_max"], log=True)
        # Distance to the allowed temperature range.
        # The scale=500.0 keeps temperature differences reasonably normalized.
        dt = _distance_to_interval(teff, criteria["teff_min"], criteria["teff_max"], log=False, scale=500.0)
        # Combined distance in the normalized (mass, radius, temperature) space.
        score = dm**2 + dr**2 + dt**2
        # Keep the first best match. This preserves dictionary order in case of ties.
        if score < best_score:
            best_ptype = ptype
            best_score = score
    return best_ptype



def get_planet_type(planet, planet_types_ref=planet_types):
    """
    Assign a planet type from mass, radius, and temperature.

    If mass, radius, and temperature are finite and physically positive where
    required, a planet type is returned. Planets outside the nominal category
    boxes are attached to the closest category. Missing or non-physical values
    remain unidentified.
    """
    mass   = float(_get_value(planet["PlanetMass"]))
    radius = float(_get_value(planet["PlanetRadius"]))
    teff   = float(_get_value(planet["PlanetTeff"]))
    if not np.isfinite(mass) or mass <= 0:
        return "Unidentified - missing mass"
    if not np.isfinite(radius) or radius <= 0:
        return "Unidentified - missing radius"
    if not np.isfinite(teff):
        return "Unidentified - missing temperature"
    return _closest_planet_type(mass, radius, teff, planet_types_ref=planet_types_ref)



def find_matching_planets(ptype, planet_table, mode, selected_planets=None, Nmax=None):
    """
    Select planets assigned to a given planet type.

    This function assumes that planet_table is a pandas DataFrame containing
    the temporary column '_AssignedPlanetType', usually created by
    build_match_dict(). In 'multi' mode, all planets assigned to the requested
    type are returned. In 'unique' mode, only the planet with the highest SNR is
    returned, while avoiding planets already selected for another type.
    """
    # Initialize the set of already selected planets for unique-mode selection.
    if selected_planets is None:
        selected_planets = set()

    # Basic safety checks.
    if mode not in ("unique", "multi"):
        raise ValueError("mode must be either 'unique' or 'multi'.")
    if "_AssignedPlanetType" not in planet_table.columns:
        raise KeyError("planet_table must contain the '_AssignedPlanetType' column. Use build_match_dict() to create it.")

    # Select planets assigned to the requested planet type.
    filtered_planets = planet_table.loc[planet_table["_AssignedPlanetType"] == ptype].copy()

    # Unique mode: keep only the best-SNR planet not already selected.
    if mode == "unique":
        filtered_planets = filtered_planets.loc[~filtered_planets["PlanetName"].isin(selected_planets)]
        if filtered_planets.empty:
            return []
        if "SNR" not in filtered_planets.columns:
            raise KeyError("planet_table must contain an 'SNR' column when mode='unique'.")
        snr       = pd.to_numeric(filtered_planets["SNR"], errors="coerce")
        valid_snr = np.isfinite(snr.to_numpy())
        if not np.any(valid_snr):
            return []
        chosen_idx    = snr.loc[valid_snr].idxmax()
        chosen_planet = filtered_planets.loc[chosen_idx]
        selected_planets.add(chosen_planet["PlanetName"])
        return [chosen_planet]

    # Multi mode: return all matching planets, optionally limited to Nmax.
    if Nmax is not None:
        filtered_planets = filtered_planets.head(Nmax)

    return filtered_planets.to_dict(orient="records") if not filtered_planets.empty else []



def build_match_dict(planet_table, planet_types, mode="multi", Nmax=None):
    """
    Build a dictionary of planets matched to each planet type.

    Each planet is assigned once with get_planet_type(), including the
    nearest-category fallback for planets outside the nominal boxes. The
    resulting temporary '_AssignedPlanetType' column is then used to build:

        {ptype: list_of_matching_planets}

    This avoids recomputing the planet classification for every planet type.
    """
    # Convert to pandas if needed, and copy to avoid modifying the input planet_table.
    df = planet_table.to_pandas().copy() if hasattr(planet_table, "to_pandas") else planet_table.copy()

    # Assign each planet to one planet type only once.
    df["_AssignedPlanetType"] = df.apply(lambda row: get_planet_type(row, planet_types_ref=planet_types), axis=1)

    # Shared set used only in unique mode to prevent the same planet from being selected several times.
    selected_planets = set()

    # Build the output dictionary.
    return {ptype: find_matching_planets(ptype=ptype, planet_table=df, mode=mode, selected_planets=selected_planets, Nmax=Nmax) for ptype in planet_types}



def plot_matching_planets(matching_planets, exposure_time, mode, planet_types=planet_types, instru=None, dpi=dpi_fig):
    """
    Render a simple table showing either:
      - (unique) the best planet per type and its SNR, or
      - (multi) the numeric cuts + counts and detections (> SNR_thresh).

    Parameters
    ----------
    matching_planets : dict
        Output of 'build_match_dict' — {ptype: list_of_rows_or_dicts}
    exposure_time : float
        Total exposure time in [mn] (used for the title text only here).
    mode : {"unique","multi"}
    instru : str or None
        Optional instrument label for the figure title.
    """

    def _format_range(criteria, key):
        min_key = f"{key}_min"
        max_key = f"{key}_max"
        if criteria[min_key] != 0 and criteria[max_key] != np.inf:
            return f"{criteria[min_key]} - {criteria[max_key]}"
        elif criteria[min_key] != 0:
            return f">{criteria[min_key]}"
        elif criteria[max_key] != np.inf :
            return f"<{criteria[max_key]}"
        return "N/A"

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    if instru is not None:
        fig.suptitle(f"ELT/{instru}", fontsize=18, weight="bold", y=1.13)
        #fig.suptitle("Predicted ELT/{instru} performance for the best targets in each planetary class", fontsize=16, y=1.15)
    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Generate the table based on the mode
    if instru is not None:
        config_str = "Coronagraph" if "ANDES" in instru else "Apodizer"
    else:
        config_str = "Config"
    if mode == "unique":
        matching_planets_df = pd.DataFrame([
            {"Type":              ptype,
             "Name":              planet["PlanetName"],
             "Mass [M⊕]":        round(planet["PlanetMass"], 2),
             "Radius [R⊕]":      round(planet["PlanetRadius"], 2),
             "Temperature [K]":   int(round(planet["PlanetTeff"])),
             "Band":              str(planet["Band"]).replace("_", "-"),
             config_str:          str(planet["Config"]).replace("_", " "),
             f"SNR (in {int(round(exposure_time/60))} h)": round(planet["SNR"], 1)}
            for ptype, planets in matching_planets.items() for planet in planets])
        table = ax.table(cellText=matching_planets_df.values, colLabels=matching_planets_df.columns, cellLoc="center", loc="center")

    elif mode == 'multi':
        conditions_df = pd.DataFrame([
            {"Type":            ptype,
             "Mass [M⊕]":      _format_range(criteria, "mass"),
             "Radius [R⊕]":    _format_range(criteria, "radius"),
             "Temperature [K]": _format_range(criteria, "teff"),
             "Number of planets\nconsidered": len(matching_planets[ptype]),
             f"Number of planets\ndetected (in {int(round(exposure_time/60))} h)": sum(planet["SNR"] > SNR_thresh for planet in matching_planets[ptype])}
            for ptype, criteria in planet_types.items()])
        table = ax.table(cellText=conditions_df.values, colLabels=conditions_df.columns, cellLoc='center', loc='center')

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2f4f6f')
            cell.set_height(0.07)
        else:
            cell.set_fontsize(10)
            cell.set_height(0.06)
            if i % 2 == 0:
                cell.set_facecolor('#e6e6e6')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    if mode == 'unique':
        table.auto_set_column_width([i for i in range(len(matching_planets_df.columns))])
    elif mode == 'multi':
        table.auto_set_column_width([i for i in range(len(conditions_df.columns))])
    plt.show(block=False)



def get_SNR_from_table(planet_table, exposure_time, band):
    """
    Compute SNR from a table with per-band signal/noise/DIT columns.

    Invalid rows are returned as NaN. This avoids divisions by zero and
    prevents invalid DIT/noise values from producing misleading infinite SNRs.
    """
    DIT        = np.asarray(planet_table[f"DIT_{band}"],        dtype=float)
    signal     = np.asarray(planet_table[f"signal_{band}"],     dtype=float)
    sigma_fund = np.asarray(planet_table[f"sigma_fund_{band}"], dtype=float)
    sigma_syst = np.asarray(planet_table[f"sigma_syst_{band}"], dtype=float)
    SNR        = np.full(len(planet_table), np.nan, dtype=float)
    valid = (np.isfinite(DIT) & np.isfinite(signal) & np.isfinite(sigma_fund) & np.isfinite(sigma_syst) & (DIT > 0))
    N_DIT = exposure_time / DIT[valid]
    S     = N_DIT * signal[valid]
    N     = np.sqrt(N_DIT * sigma_fund[valid]**2 + N_DIT**2 * sigma_syst[valid]**2)
    valid_noise                 = np.isfinite(N) & (N > 0)
    idx_valid                   = np.where(valid)[0]
    SNR[idx_valid[valid_noise]] = S[valid_noise] / N[valid_noise]
    return SNR



def get_contrast_from_table(planet_table, exposure_time, band_quantity, band_contrast, instru, spectrum_contributions):
    """
    Compute the 5-sigma contrast limit from a table with per-band
    signal/noise/DIT columns.

    band_quantity defines the band in which signal and noise are evaluated.
    band_contrast defines the photometric band in which the returned contrast
    is expressed.

    Invalid rows are returned as NaN.
    """
    DIT        = np.asarray(planet_table[f"DIT_{band_quantity}"], dtype=float)
    signal     = np.asarray(planet_table[f"signal_{band_quantity}"], dtype=float)
    sigma_fund = np.asarray(planet_table[f"sigma_fund_{band_quantity}"], dtype=float)
    sigma_syst = np.asarray(planet_table[f"sigma_syst_{band_quantity}"], dtype=float)
    contrast   = np.full(len(planet_table), np.nan, dtype=float)
    if band_contrast == "INSTRU":
        star_mag_col   = f"StarINSTRUmag({instru})"
        planet_mag_col = f"PlanetINSTRUmag({instru})({spectrum_contributions})"
    else:
        star_mag_col   = f"Star{band_contrast}mag"
        planet_mag_col = f"Planet{band_contrast}mag({spectrum_contributions})"
    missing_cols = [col for col in [star_mag_col, planet_mag_col] if col not in planet_table.colnames]
    if missing_cols:
        raise KeyError(f"Missing magnitude columns: {missing_cols}")
    mag_star   = np.asarray(planet_table[star_mag_col], dtype=float)
    mag_planet = np.asarray(planet_table[planet_mag_col], dtype=float)
    valid = np.isfinite(DIT) & np.isfinite(signal) & np.isfinite(sigma_fund) & np.isfinite(sigma_syst) & np.isfinite(mag_star) & np.isfinite(mag_planet) & (DIT > 0) & (signal > 0) & (sigma_fund >= 0) & (sigma_syst >= 0)
    N_DIT       = exposure_time / DIT[valid]
    flux_ratio  = 10**(0.4*(mag_planet[valid] - mag_star[valid]))  # F_star/F_planet
    signal_0    = signal[valid] * flux_ratio                       # Signal per DIT at unit contrast
    S_0         = N_DIT * signal_0
    N           = np.sqrt(N_DIT*sigma_fund[valid]**2 + N_DIT**2*sigma_syst[valid]**2)
    valid_value = np.isfinite(S_0) & np.isfinite(N) & (S_0 > 0) & (N > 0)
    idx_valid   = np.where(valid)[0]
    contrast[idx_valid[valid_value]] = 5*N[valid_value] / S_0[valid_value]
    return contrast



def get_contrast_from_saved_SNR(planet_table, band_contrast, instru, spectrum_contributions, SNR_col):
    """
    Compute the 5-sigma contrast limit directly from a precomputed SNR.

    The saved SNR must correspond to the actual planet flux represented
    by the magnitude columns.
    """
    if band_contrast == "INSTRU":
        star_mag_col   = f"StarINSTRUmag({instru})"
        planet_mag_col = f"PlanetINSTRUmag({instru})({spectrum_contributions})"
    else:
        star_mag_col   = f"Star{band_contrast}mag"
        planet_mag_col = f"Planet{band_contrast}mag({spectrum_contributions})"
    missing_cols = [col for col in [star_mag_col, planet_mag_col, SNR_col] if col not in planet_table.colnames]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")
    mag_star   = np.asarray(planet_table[star_mag_col], dtype=float)
    mag_planet = np.asarray(planet_table[planet_mag_col], dtype=float)
    SNR        = np.asarray(planet_table[SNR_col], dtype=float)
    contrast   = np.full(len(planet_table), np.nan, dtype=float)
    valid = np.isfinite(mag_star) & np.isfinite(mag_planet) & np.isfinite(SNR) & (SNR > 0)
    planet_contrast = 10**(-0.4*(mag_planet[valid] - mag_star[valid]))
    contrast[valid] = 5*planet_contrast/SNR[valid]
    return contrast



def get_size_from_SNR(SNR, s0=50, ds=200, SNR_min=SNR_thresh, SNR_max=1_000):
    """
    Map SNR to a marker size for plotting using a log stretch.

    Parameters
    ----------
    SNR : float or array-like
    s0 : float
        Base marker size.
    ds : float
        Extra size added when SNR reaches SNR_max.
    SNR_min, SNR_max : float
        Lower/upper clamps for the scaling.

    Returns
    -------
    np.ndarray
        Marker sizes (same shape as input SNR).
    """
    SNR_min = max(1e-30, SNR_min)
    SNR     = np.asarray(SNR, dtype=float)
    SNR     = np.clip(SNR, SNR_min, SNR_max)
    t       = np.log10(SNR) / np.log10(SNR_max)
    t       = np.maximum(t, 0)
    return s0 + ds * t



def are_planets_observable(latitude, longitude, altitude, planet_table, date_obs, min_elevation_deg=30.0, hours_span=6, ntimes=24):
    """
    Boolean mask of planets observable at least once around local midnight.

    A planet is “observable” if, for any time in a grid from -hours_span to
    +hours_span around local midnight, it has:
      - altitude > min_elevation_deg, AND
      - the Sun is below the horizon (civil night simplification).

    Parameters
    ----------
    latitude, longitude : float
        Site geodetic coordinates (deg).
    altitude : float
        Site altitude (meters).
    planet_table : astropy.table.Table/QTable
        Must provide 'RA' and 'Dec' in degrees.
    date_obs : str
        "DD/MM/YYYY" local calendar date.
    min_elevation_deg : float, optional
        Minimum altitude for observability (deg).
    hours_span : float, optional
        Half-width time window around local midnight (hours).
    ntimes : int, optional
        Number of time samples within the window.

    Returns
    -------
    np.ndarray (bool), shape (N_planets,)
        True if observable at least once.
    """

    from timezonefinder import TimezoneFinder
    import pytz
    from datetime import datetime

    # Site & local midnight
    site           = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
    tzname         = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)
    local_tz       = pytz.timezone(tzname if tzname else "UTC")
    d0             = datetime.strptime(date_obs, "%d/%m/%Y")
    local_midnight = local_tz.localize(d0.replace(hour=0, minute=0, second=0, microsecond=0))

    # Time grid around local midnight
    delta = np.linspace(-hours_span, +hours_span, ntimes) * u.hour
    times = Time(local_midnight) + delta

    # Night mask (Sun below horizon)
    frame_all = AltAz(obstime=times, location=site)
    sun_alt   = get_sun(times).transform_to(frame_all).alt
    is_night  = (sun_alt < 0 * u.deg)  # (Nt,)

    # Planet coordinates (vectorized over planets)
    ra_deg  = np.asarray(planet_table["RA"].to_value(u.deg),  float)
    dec_deg = np.asarray(planet_table["Dec"].to_value(u.deg), float)
    coords  = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)

    # Altitude vs time (loop on times, vectorized on planets)
    alts = []
    for t in times:
        frame_t = AltAz(obstime=t, location=site)
        alts.append(coords.transform_to(frame_t).alt)
    alt_stack = u.Quantity(alts)  # (Nt, Np)

    # Observable if any time satisfies both constraints
    ok_elev = alt_stack > (min_elevation_deg * u.deg)
    obs = ok_elev & is_night[:, None]
    return np.any(obs, axis=0)



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



def print_simulation_summary(instru, instru_type, detector, post_processing, D, S, N_mirror, trans_dust, RON0, RON_lim, DC0, saturation_e, min_DIT, max_DIT, coronagraph, apodizer, strehl, thermal_model, reflected_model, SNR_thr, exposure_time, spatial_sampling, size_core, A_FWHM, Rc, filter_type, table_type, sep_min, sep_max, N_PT_raw, N_PT, force_new_calc, N_px=None, l0=None, R=None, Nl=None, Dl=None, WFE=None, r_WFE=None, C_raw=None, IWA=None, IWA_unit="mas", trans_instru=None, sigma_m=None, FoV=None):
    _print_section("FastYield simulation summary")

    # General configuration
    _print_kv("Instrument",            f"ELT/{instru}")
    _print_kv("Instrument type",       instru_type)
    _print_kv("Detector",              detector)
    _print_kv("Post-processing",       post_processing)
    _print_kv("Force new calculation", force_new_calc)

    # Telescope / detector
    _print_section("Telescope and detector", "-")
    _print_kv("Diameter D",        f"{_fmt_scalar(D)} m")
    _print_kv("Collecting area S", f"{_fmt_scalar(S)} m²")
    _print_kv("ELT mirrors",       _fmt_scalar(N_mirror))
    _print_kv("Dust transmission", _fmt_scalar(trans_dust))
    if N_px is not None:
        _print_kv("Detector linear size", f"{_fmt_scalar(N_px)} px")
    _print_kv("RON0",         f"{_fmt_scalar(np.unique(RON0))} e-/px/read")
    _print_kv("RON floor",    f"{_fmt_scalar(np.unique(RON_lim))} e-/px")
    _print_kv("Dark current", f"{_fmt_scalar(np.unique(DC0))} e-/px/mn")
    _print_kv("Saturation",   f"{_fmt_scalar(np.unique(saturation_e))} e-/px")
    _print_kv("Minimum DIT",  f"{_fmt_scalar(np.unique(min_DIT))} mn")
    _print_kv("Maximum DIT",  f"{_fmt_scalar(np.unique(max_DIT))} mn")

    # Science setup
    _print_section("Science setup", "-")
    _print_kv("Coronagraph",              coronagraph)
    _print_kv("Apodizer",                 apodizer)
    _print_kv("Strehl condition",         strehl)
    _print_kv("Thermal model",            thermal_model)
    _print_kv("Reflected model",          reflected_model)
    _print_kv("Detection threshold",      f"S/N >= {_fmt_scalar(SNR_thr)}")
    _print_kv("Exposure time per target", f"{_fmt_scalar(exposure_time)} mn")
    if spatial_sampling is not None:
        _print_kv("Spatial sampling", f"{_fmt_scalar(spatial_sampling)} px/FWHM")
    if size_core is not None:
        _print_kv("Core diameter", f"{_fmt_scalar(size_core)} px")
    _print_kv("Core area", f"{_fmt_scalar(A_FWHM)} px")

    if instru_type == "IFU":
        _print_kv("High-pass cut-off Rc", _fmt_scalar(Rc))
        _print_kv("MM filter type",       filter_type)

    # Catalog parameters
    _print_section("Catalog parameters", "-")
    _print_kv("Catalog type",          table_type)
    _print_kv("Separation range",      f"[{_fmt_scalar(sep_min)}, {_fmt_scalar(sep_max)}] mas")
    _print_kv("Initial catalog size",  f"{_fmt_scalar(N_PT_raw)} planets")
    _print_kv("Filtered catalog size", f"{_fmt_scalar(N_PT)} planets")

    # Parameter space
    _print_section("Explored parameter space", "-")
    if R is not None:
        _print_kv("R",            _fmt_axis(R, log_hint=True))
    if l0 is not None:
        _print_kv("lambda_0",     _fmt_axis(l0, unit=" µm", log_hint=False))
    if Nl is not None:
        _print_kv("N_lambda",     _fmt_axis(Nl, log_hint=True))
    if Dl is not None:
        _print_kv("Delta_lambda", _fmt_axis(Dl, unit=" µm", log_hint=False))
    if WFE is not None:
        _print_kv("WFE",          _fmt_axis(WFE, unit=" nm RMS", log_hint=False))
    if r_WFE is not None:
        _print_kv("r_WFE",        _fmt_axis(r_WFE, unit="", log_hint=False))
    if C_raw is not None:
        _print_kv("C_raw",        _fmt_axis(C_raw, log_hint=True))
    if IWA is not None:
        _print_kv("IWA",          _fmt_axis(IWA, unit=f" {IWA_unit}", log_hint=True))
    if trans_instru is not None:
        _print_kv("trans_instru", _fmt_axis(trans_instru, unit="e-/ph", log_hint=False))
    if sigma_m is not None:
        _print_kv("sigma_m",      _fmt_axis(100*np.asarray(sigma_m), unit=" %", log_hint=True))
    if FoV is not None:
        _print_kv("FoV",          _fmt_axis(FoV, unit=" mas", log_hint=True))

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

def spectral_window_quadrature(l0, Dl, n_nodes=12):
    """Gauss--Legendre nodes and normalized weights for rectangular spectral windows."""
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    wavelengths    = np.asarray(l0)[:, None, None] + 0.5*np.asarray(Dl)[None, :, None]*nodes
    return wavelengths, np.broadcast_to(weights / 2, wavelengths.shape)



def average_detector_specs(l0, Dl, detector, dtype=np.float32):
    """Window-integrated detector assumptions, without averaging discrete readout modes.

    Continuous noise variances are averaged by fractional spectral coverage. For
    mixed-family windows, the non-UTR readout is a conservative approximation;
    saturation is the lowest full well, and DIT limits use their intersection.
    This is an effective-detector model, not a physical multi-arm instrument.
    Supported presets: HWO and PCS families, plus constant_<detector_name>.
    The detector_l0 output is descriptive only; all noise fields use (l0, Dl).
    """
    # Detector families are assigned over physical wavelength intervals. The
    # same window-averaging rules apply to HWO and PCS; only the band edges and
    # detector-family names depend on the chosen instrument prescription.
    l0 = np.asarray(l0, dtype=float)
    Dl = np.asarray(Dl, dtype=float)
    lo = l0[:, None] - Dl[None, :] / 2
    hi = l0[:, None] + Dl[None, :] / 2
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
        raise ValueError("Spectral windows must be finite and have positive widths.")

    if detector.startswith("constant_"):
        names       = [detector[len("constant_"):]]
        edges       = [-np.inf, np.inf]
    else:
        prescriptions = {
            "HWO_conservative":         (["UV_MCP",  "EMCCD",       "H4RG"], [-np.inf, 0.30, 0.95, np.inf]),
            "HWO_photon_counting":      (["UV_MCP",  "SKIPPER_CCD", "MKID"], [-np.inf, 0.30, 0.95, np.inf]),
            "PCS_visNIR_conservative":  (["VIS_CCD", "H4RG"],                [-np.inf, 0.80, np.inf]),
            "PCS_low_noise_visNIR":     (["EMCCD",   "H4RG"],                [-np.inf, 0.80, np.inf]),
            "PCS_NIR_APD_optimistic":   (["EMCCD",   "SAPHIRA"],             [-np.inf, 0.80, np.inf]),
        }
        if detector not in prescriptions:
            raise ValueError(f"Unknown detector mode: {detector}")
        names, edges = prescriptions[detector]

    fractions = np.stack([np.maximum(0.0, np.minimum(hi, b) - np.maximum(lo, a)) / (hi - lo) for a, b in zip(edges[:-1], edges[1:])])
    # Metadata for identifying the detector at the window central wavelength;
    # noise calculations always use the window-integrated (l0, Dl) arrays.
    detector_l0 = np.asarray(names)[np.searchsorted(edges[1:-1], l0, side="right")]
    specs  = [get_detector_specs(n) for n in names]
    active = fractions > 1e-12
    mixed  = np.count_nonzero(active, axis=0) > 1
    def weighted(key, squared=False):
        values = np.asarray([x[key] for x in specs], dtype=float)
        out    = np.tensordot(values**2 if squared else values, fractions, axes=(0, 0))
        return np.sqrt(out) if squared else out
    def conservative(key, func):
        vals = np.asarray([x[key] for x in specs], dtype=float)
        return func(np.where(active, vals[:, None, None], np.nan), axis=0)
    RON0         = weighted("RON0", squared=True)
    RON_lim      = weighted("RON_lim", squared=True)
    DC0          = weighted("DC0")
    saturation_e = conservative("saturation_e", np.nanmin)
    min_DIT      = conservative("min_DIT",      np.nanmax)
    max_DIT      = conservative("max_DIT",      np.nanmin)
    if np.any(min_DIT > max_DIT):
        raise ValueError("Detector families have incompatible DIT limits in a spectral window")
    utr    = np.asarray([x["family"] == "HgCdTe_HxRG" for x in specs], dtype=bool)
    is_UTR = np.all(~active | utr[:, None, None], axis=0) & ~mixed
    return dict(RON0=RON0.astype(dtype), RON_lim=RON_lim.astype(dtype), DC0=DC0.astype(dtype), saturation_e=saturation_e.astype(dtype), min_DIT=min_DIT.astype(dtype), max_DIT=max_DIT.astype(dtype), is_UTR=is_UTR, mixed=mixed, detector_l0=detector_l0)



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


