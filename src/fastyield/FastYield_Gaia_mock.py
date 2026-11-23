"""
Build a FastYield planet table from the Gaia DR4/DR5 mock catalogs of Lammers & Winn (2026).

Each release is read and saved independently in sim_data/planet_table/:
    Gaia_DR4_mock/
        DR4_mock_exoplanet_catalog.csv
        DR4_mock_planet_impostor_catalog.csv
        Gaia_DR4_mock_Pull_For_FastYield.ecsv        (output)
    Gaia_DR5_mock/
        DR5_mock_exoplanet_catalog.csv
        DR5_mock_planet_impostor_catalog.csv
        Gaia_DR5_mock_Pull_For_FastYield.ecsv        (output)

Usage:
    from fastyield.FastYield_Gaia import get_gaia_mock_table
    get_gaia_mock_table(release="DR4")
    get_gaia_mock_table(release="DR5")

    # Then, for example:
    from fastyield.FastYield import get_planet_table_SNR
    get_planet_table_SNR(instru="ANDES", table="Gaia_DR4_mock")

Only the exoplanet CSV is converted. The planet-impostor CSV contains stellar
binaries: these are counted but excluded from the FastYield planet table.

IMPORTANT: These catalogs are forecasts, not actual Gaia detections. Stellar
Teff, radius, age, rotation, and Ks magnitude are estimated rather than measured;
the inferred Ks magnitude is particularly approximate. For precise yields,
replace these assumptions with observed stellar parameters and photometry.
FastYield's shared Sonora Bobcat grid is expected in planet_table/Archive/.
"""

# import FastYield modules
from .config import get_planet_table_path, instrus, bands, T_sun
from .FastYield_helpers import chen_kipping_radius_from_mass, get_evolutionary_model, get_planet_type, write_planet_table
from .FastYield import get_planet_table_magnitudes

# import astropy modules
from astropy import constants as const
from astropy import units as u
from astropy.table import QTable

# import numpy modules
import numpy as np

# import other modules
import pandas as pd
import time



def _stellar_main_sequence(mass):
    """Approximate main-sequence luminosity, radius, and Teff from mass in Msun."""
    luminosity = np.where(mass < 0.43, 0.23*mass**2.3, np.where(mass < 2., mass**4., 1.5*mass**3.5))  # [Lsun]
    radius     = np.where(mass < 1., mass**0.8, mass**0.57)                                         # [Rsun]
    teff       = T_sun * (luminosity/radius**2)**0.25                                               # [K]
    return luminosity, radius, teff



def _gaia_g_to_k(gmag, teff):
    """Approximate 2MASS Ks from Gaia G via solar-calibrated monochromatic blackbody color.

    This is a rough color proxy, NOT a calibrated Gaia-to-2MASS transformation.
    It neglects extinction, metallicity and molecular bands (especially for M stars).
    Uses approximate effective wavelengths G=0.673 um and Ks=2.159 um, and
    (G-Ks)_Sun ~= 1.39 mag. Prefer observed 2MASS Ks whenever available.
    """
    hc_over_k = (const.h * const.c / const.k_B).to_value(u.um * u.K)
    def flux_ratio(T):
        return (2.159/0.673)**5 * np.expm1(hc_over_k/(2.159*T)) / np.expm1(hc_over_k/(0.673*T))
    g_minus_k = 1.39 - 2.5*np.log10(flux_ratio(teff)/flux_ratio(T_sun))
    return gmag - g_minus_k



def _draw_positive_normal(rng, mean, std, n):
    """Draw positive rotation velocities using the Archive truncated-normal prior."""
    values = rng.normal(mean, std, n)
    invalid = values < 0.
    while invalid.any():
        values[invalid] = rng.normal(mean, std, invalid.sum())
        invalid = values < 0.
    return values



def get_gaia_mock_table(release="DR4", seed=42, age_gyr=5.):
    """
    Convert a Gaia mock planet catalog into a FastYield-compatible QTable.

    Parameters
    ----------
    release : str, optional
        "DR4" or "DR5". Reads and writes in Gaia_<release>_mock/.
        The two releases are kept separate.
    seed : int, optional
        Random seed for the explicitly assumed stellar and planetary rotations
        and systemic stellar RVs.
    age_gyr : float, optional
        Assumed stellar age in Gyr, capped at half the approximate main-sequence
        lifetime (5 * Mstar**-2.5 Gyr). Used for Sonora cooling estimates.

    Returns
    -------
    planet_table : astropy.table.QTable
        Same key column names and units required by FastYield as the Archive
        catalog, plus Gaia input/provenance columns.
    """
    if release not in ("DR4", "DR5"):
        raise ValueError("release must be 'DR4' or 'DR5'.")
    if not np.isfinite(age_gyr) or age_gyr <= 0.:
        raise ValueError("age_gyr must be finite and strictly positive.")

    time1 = time.time()
    rng   = np.random.default_rng(seed)

    def report_step(number, title):
        print(f"\n[{number:02d}] {title}")

    def report_stat(label, value):
        print(f"         {label:<45} {value}")

    # -----------------------------------------------------------------------------
    # 1) Load the chosen Gaia mock catalog (planet impostors are excluded)
    # -----------------------------------------------------------------------------
    report_step(1, "Lammers & Winn (2026) mock Gaia catalogs")
    table          = f"Gaia_{release}_mock"
    path           = get_planet_table_path(table)
    planets_file   = path / f"{release}_mock_exoplanet_catalog.csv"
    impostors_file = path / f"{release}_mock_planet_impostor_catalog.csv"
    raw            = pd.read_csv(planets_file)
    with impostors_file.open(encoding="utf-8") as f:
        n_impostors = sum(1 for _ in f) - 1

    required = ["Gaia source ID", "True RA [deg]", "True Dec [deg]", "True planet mass [M_J]", "True period [days]", "Stellar mass [M_\\odot]", "G-band mag", "MCMC distance 50th [pc]", "Best-fit planet mass [M_J]", "Best-fit period [days]", "Best-fit inclination [deg]", "Best-fit eccentricity", "Best-fit omega [deg]"]
    missing  = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing Gaia columns in {planets_file}: {missing}")

    valid = (np.isfinite(raw["Stellar mass [M_\\odot]"]) & (raw["Stellar mass [M_\\odot]"] > 0.)
             & np.isfinite(raw["MCMC distance 50th [pc]"]) & (raw["MCMC distance 50th [pc]"] > 0.)
             & np.isfinite(raw["G-band mag"])
             & np.isfinite(raw["Best-fit planet mass [M_J]"]) & (raw["Best-fit planet mass [M_J]"] > 0.)
             & np.isfinite(raw["Best-fit period [days]"]) & (raw["Best-fit period [days]"] > 0.)
             & np.isfinite(raw["Best-fit inclination [deg]"]) & raw["Best-fit inclination [deg]"].between(0., 180.)
             & raw["Best-fit eccentricity"].between(0., 1., inclusive="left")
             & np.isfinite(raw["Best-fit omega [deg]"]))
    n_raw = len(raw)
    raw   = raw.loc[valid].reset_index(drop=True)
    n     = len(raw)
    if n == 0:
        raise ValueError("No valid planets remain after the basic Gaia input checks.")
    report_stat("Mock planet candidates",               n_raw)
    report_stat("Rejected incomplete/nonphysical rows", n_raw-n)
    report_stat("Binary impostors (excluded)",          n_impostors)

    # -----------------------------------------------------------------------------
    # 2) Rename and unit-convert planetary and stellar parameters
    # -----------------------------------------------------------------------------
    report_step(2, "Catalog identifiers, fitted orbits, and units")
    q                    = QTable()
    ids                  = raw["Gaia source ID"].to_numpy(dtype=np.int64)
    prefix               = f"Gaia {release} "
    names                = ids.astype(str)
    q["PlanetName"]      = np.char.add(np.char.add(prefix, names), " b")
    q["StarName"]        = np.char.add("Gaia DR3 ", names)
    q["GaiaSourceID"]    = ids
    q["GaiaRelease"]     = np.full(n, release)
    q["DiscoveryMethod"] = np.full(n, "Astrometry")
    q["DiscoveryRef"]    = np.full(n, "Lammers & Winn (2026), mock")
    q["RA"]              = raw["True RA [deg]"].to_numpy(float) * u.deg
    q["Dec"]             = raw["True Dec [deg]"].to_numpy(float) * u.deg
    q["Distance"]        = raw["MCMC distance 50th [pc]"].to_numpy(float) * u.pc
    q["StarMass"]        = raw["Stellar mass [M_\\odot]"].to_numpy(float) * u.M_sun
    q["StarGmag"]        = raw["G-band mag"].to_numpy(float) * u.dimensionless_unscaled
    q["PlanetMass"]      = (raw["Best-fit planet mass [M_J]"].to_numpy(float) * u.M_jup).to(u.M_earth)
    q["Period"]          = raw["Best-fit period [days]"].to_numpy(float) * u.day
    q["Inc"]             = raw["Best-fit inclination [deg]"].to_numpy(float) * u.deg
    q["Ecc"]             = raw["Best-fit eccentricity"].to_numpy(float) * u.dimensionless_unscaled
    q["ArgPeri"]         = raw["Best-fit omega [deg]"].to_numpy(float) * u.deg
    q["LongAscNode"]     = raw["Best-fit Omega [deg]"].to_numpy(float) * u.deg
    q["TPeri"]           = raw["Best-fit T_peri [days]"].to_numpy(float) * u.day
    q["GaiaDeltaChi2"]   = raw[r"\Delta \chi^2"].to_numpy(float) * u.dimensionless_unscaled
    q["PlanetMassRef"]   = np.full(n, "Gaia mock best fit")
    q["PeriodRef"]       = np.full(n, "Gaia mock best fit")
    q["IncRef"]          = np.full(n, "Gaia mock best fit")
    q["EccRef"]          = np.full(n, "Gaia mock best fit")
    q["ArgPeriRef"]      = np.full(n, "Gaia mock best fit")

    # Preserve selected injected values and marginal posterior quantiles.
    # Quoted +/- uncertainties below are MCMC widths around the MEDIAN,
    # not formal error bars around the best-fit orbital solution.
    for key, source, unit in [("PlanetMass", "planet mass", u.M_jup), ("Period", "period", u.day),
                              ("Inc", "inclination", u.deg), ("Ecc", "eccentricity", u.dimensionless_unscaled),
                              ("Distance", "distance", u.pc)]:
        if key != "Distance":
            true_col = {"PlanetMass": "True planet mass [M_J]", "Period": "True period [days]",
                        "Inc": "True inclination [deg]", "Ecc": "True eccentricity"}[key]
            q[f"GaiaTrue{key}"] = (raw[true_col].to_numpy(float) * unit).to(q[key].unit)
        for percentile in (16, 50, 84):
            suffix = {"PlanetMass": " [M_J]", "Period": " [days]", "Inc": " [deg]", "Ecc": "", "Distance": " [pc]"}[key]
            q[f"Gaia{key}P{percentile}"] = (raw[f"MCMC {source} {percentile}th{suffix}"].to_numpy(float) * unit).to(q[key].unit)

    q["GaiaTrueDistance"]    = raw["True distance [pc]"].to_numpy(float) * u.pc
    q["GaiaTrueArgPeri"]     = raw["True omega [deg]"].to_numpy(float) * u.deg
    q["GaiaTrueLongAscNode"] = raw["True Omega [deg]"].to_numpy(float) * u.deg
    q["GaiaTrueTPeri"]       = raw["True T_peri [days]"].to_numpy(float) * u.day
    q["+DeltaPlanetMass"]    = q["GaiaPlanetMassP84"] - q["GaiaPlanetMassP50"]
    q["-DeltaPlanetMass"]        = q["GaiaPlanetMassP16"] - q["GaiaPlanetMassP50"]
    q["+DeltaDistance"]          = q["GaiaDistanceP84"] - q["GaiaDistanceP50"]
    q["-DeltaDistance"]          = q["GaiaDistanceP16"] - q["GaiaDistanceP50"]
    q.meta["source"]             = "Lammers & Winn 2026 Gaia forecast (mock, not observed)"
    q.meta["gaia_release"]       = release
    q.meta["age_prior_gyr"]      = float(age_gyr)
    q.meta["orbit_solution"]     = "Best fit; observed distance = MCMC median"
    q.meta["impostors_included"] = False
    report_stat("Stellar systems",  len(np.unique(ids)))
    report_stat("Orbital solution", "best fit (not true injected orbit)")

    # -----------------------------------------------------------------------------
    # 3) Estimate missing stellar parameters from mass and Gaia G magnitude
    # -----------------------------------------------------------------------------
    report_step(3, "Stellar priors (not measured by these CSVs)")
    star_mass       = q["StarMass"].to_value(u.M_sun)
    star_lum, star_radius, star_teff = _stellar_main_sequence(star_mass)
    star_age        = np.minimum(age_gyr, 5.*star_mass**-2.5) # [Gyr], <= 1/2 MS lifetime
    q["StarRadius"] = star_radius * u.R_sun
    q["StarTeff"]   = star_teff * u.K
    q["StarLum"]    = np.log10(star_lum) * u.dex(u.L_sun)
    q["StarAge"]    = star_age * u.Gyr
    q["StarKmag"]   = _gaia_g_to_k(q["StarGmag"].value, star_teff) * u.dimensionless_unscaled
    q["StarLogg"]   = np.log10((const.G*q["StarMass"]/q["StarRadius"]**2).to_value(u.cm/u.s**2)) * u.dex(u.cm/u.s**2)
    q["StarFeH"]    = np.zeros(n) * u.dimensionless_unscaled
    q["StarVmag"]   = np.full(n, np.nan) * u.dimensionless_unscaled
    q["StarSpT"]    = np.full(n, "", dtype="U1")
    q["StarTeffImputed"]   = np.ones(n, dtype=bool)
    q["StarRadiusImputed"] = np.ones(n, dtype=bool)
    q["StarKmagImputed"]   = np.ones(n, dtype=bool)
    q["StarAgeImputed"]    = np.ones(n, dtype=bool)
    q["StarKmagRef"]       = np.full(n, "Approx. G-Ks (solar-calibrated blackbody)")
    report_stat("Teff / radius / age", "mass-based main-sequence priors")
    report_stat("Stellar K magnitudes", "approximate from G; NOT observed Ks")
    report_stat("Adopted age", f"{age_gyr:g} Gyr, capped by stellar lifetime")

    # -----------------------------------------------------------------------------
    # 4) Orbital semi-major axes and geometry at quadrature, as in Archive
    # -----------------------------------------------------------------------------
    report_step(4, "Keplerian orbits and quadrature geometry")
    total_mass      = q["StarMass"] + q["PlanetMass"]
    q["SMA"]        = (const.G*total_mass * (q["Period"]/(2*np.pi))**2)**(1/3)
    q["SMA"]        = q["SMA"].to(u.AU)
    q["SMARef"]     = np.full(n, "Kepler (Gaia best-fit period and masses)")
    q["Phase"]      = np.full(n, np.pi/2) * u.rad
    q["sini"]       = np.sin(q["Inc"].to_value(u.rad)) * u.dimensionless_unscaled
    omega          = q["ArgPeri"].to(u.rad)
    inc            = q["Inc"].to(u.rad)
    nu             = q["Phase"] - omega - (np.pi/2)*u.rad
    eccentricity   = q["Ecc"].value
    r_orbit        = q["SMA"]*(1.-eccentricity**2)/(1.+eccentricity*np.cos(nu))
    projection     = np.sqrt(np.cos(nu+omega)**2 + np.sin(nu+omega)**2*np.cos(inc)**2)
    q["AngSep"]    = (r_orbit/q["Distance"]*projection).to(u.mas, equivalencies=u.dimensionless_angles())
    q["alpha"]     = np.arccos(np.clip(-np.sin(inc)*np.cos(q["Phase"]), -1., 1.)).to(u.rad)
    q["g_alpha"]   = ((np.sin(q["alpha"]) + (np.pi-q["alpha"].value)*np.cos(q["alpha"]))/np.pi).to(u.dimensionless_unscaled)
    Kp             = (2*np.pi*q["SMA"]/q["Period"]*np.sin(inc)/np.sqrt(1.-eccentricity**2)).to(u.km/u.s)
    q["DeltaRadialVelocity"] = Kp*(np.cos(nu+omega) + eccentricity*np.cos(omega))
    report_stat("Orbital phase", "pi/2 rad (Archive quadrature convention)")
    report_stat("SMA",           "from Kepler's law; true orbital a, not projected a")

    # -----------------------------------------------------------------------------
    # 5) Stellar/planet rotation and stellar systemic velocities (assumed)
    # -----------------------------------------------------------------------------
    report_step(5, "Rotations and radial-velocity priors")
    q["StarRadialVelocity"]        = rng.normal(0., 30., n) * u.km/u.s # Galactic RV prior
    q["StarRadialVelocityImputed"] = np.ones(n, dtype=bool)
    q["PlanetRadialVelocity"]      = q["StarRadialVelocity"] + q["DeltaRadialVelocity"]
    vrot_star = np.empty(n, dtype=float)
    for low, high, mean, std in [(0., 3500., 1., 0.5), (3500., 6000., 3., 1.), (6000., 7500., 10., 5.), (7500., 10000., 120., 60.), (10000., np.inf, 200., 80.)]:
        mask            = (star_teff >= low) & (star_teff < high)
        vrot_star[mask] = _draw_positive_normal(rng, mean, std, mask.sum())
    q["StarVrot"]           = vrot_star * u.km/u.s
    q["StarVsini"]          = q["StarVrot"] * q["sini"]
    q["StarVrotImputed"]    = np.ones(n, dtype=bool)
    orbital_omega           = (2*np.pi/q["Period"])*(1.+eccentricity*np.cos(nu))**2/(1.-eccentricity**2)**1.5
    q["StarVrotFromPlanet"] = np.abs(q["StarVrot"] - (q["StarRadius"]*orbital_omega).to(u.km/u.s))

    vrot_planet = np.empty(n, dtype=float)
    mass_earth  = q["PlanetMass"].to_value(u.M_earth)
    for low, high, mean, std in [(0., 5., 0.5, 0.5), (5., 20., 2., 1.), (20., 100., 3., 1.5), (100., 300., 12., 4.), (300., np.inf, 20., 8.)]:
        mask              = (mass_earth > low) & (mass_earth <= high)
        vrot_planet[mask] = _draw_positive_normal(rng, mean, std, mask.sum())
    q["PlanetVrot"]        = vrot_planet * u.km/u.s
    q["PlanetVsini"]       = q["PlanetVrot"] * q["sini"]
    q["PlanetVrotImputed"] = np.ones(n, dtype=bool)
    report_stat("Stellar systemic RV", "N(0, 30 km/s); not measured")
    report_stat("Spin axes",           "assumed aligned with orbital inclinations")

    # -----------------------------------------------------------------------------
    # 6) Planet radius and temperatures: same FastYield prescriptions as Archive
    # -----------------------------------------------------------------------------
    report_step(6, "Planet radii, Sonora cooling, and temperatures")
    q["PlanetRadius"]        = chen_kipping_radius_from_mass(q["PlanetMass"])
    q["PlanetRadiusRef"]     = np.full(n, "Chen & Kipping (2017), deterministic relation")
    q["PlanetRadiusImputed"] = np.ones(n, dtype=bool)
    bond_albedo              = 0.3
    q["PlanetTeq"]           = q["StarTeff"] * np.sqrt((q["StarRadius"]/(2*q["SMA"])).decompose()) * (1.-bond_albedo)**0.25
    # This grid is shared with Archive via get_evolutionary_model(); missing
    # age measurements are replaced by the explicit age prior in step 3.
    tint, _, _                     = get_evolutionary_model(q)
    q["PlanetTint"]                = tint
    tint_value                     = tint.to_value(u.K)
    teq_value                      = q["PlanetTeq"].to_value(u.K)
    q["PlanetTeff"]                = (teq_value**4 + np.where(np.isfinite(tint_value), tint_value, 0.)**4)**0.25 * u.K
    q["PlanetTeffRef"]             = np.full(n, f"Teq (A_B={bond_albedo}, 4pi) + Sonora Tint if available")
    q["PlanetTintAvailable"]       = np.isfinite(tint_value) & (tint_value > 0.)
    planet_gravity                 = (const.G*q["PlanetMass"]/q["PlanetRadius"]**2).to(u.m/u.s**2)
    q["PlanetLogg"]                = np.log10(planet_gravity.to_value(u.cm/u.s**2)) * u.dex(u.cm/u.s**2)
    q["HasAtmosphere"]             = np.ones(n, dtype=bool) # Giant planets; no rocky-planet heuristic needed
    earth_gravity                  = (const.G*const.M_earth/const.R_earth**2).to(u.m/u.s**2)
    q["TelluricEquivalentAirmass"] = np.clip(2.*(earth_gravity/planet_gravity).value, 0.5, 8.) * u.dimensionless_unscaled
    report_stat("Radius",                        "Chen & Kipping (2017)")
    report_stat("Sonora intrinsic temperatures", f"{q['PlanetTintAvailable'].sum()}/{n} planets")

    # -----------------------------------------------------------------------------
    # 7) Classify planets and prepare the standard magnitude columns
    # -----------------------------------------------------------------------------
    report_step(7, "Planet classification and photometry columns")
    q["PlanetType"]                          = np.asarray([get_planet_type(planet) for planet in q], dtype="U48")
    q["PlanetKmag(thermal+reflected)"]       = np.full(n, np.nan) * u.dimensionless_unscaled
    q["+DeltaPlanetKmag(thermal+reflected)"] = np.full(n, np.nan) * u.dimensionless_unscaled
    q["-DeltaPlanetKmag(thermal+reflected)"] = np.full(n, np.nan) * u.dimensionless_unscaled
    q["PlanetKmag(thermal+reflected)Ref"]    = np.full(n, "", dtype="U32")
    for instru in instrus:
        q[f"StarINSTRUmag({instru})"] = np.full(n, np.nan) * u.dimensionless_unscaled
        for contribution in ("thermal+reflected", "thermal", "reflected"):
            q[f"PlanetINSTRUmag({instru})({contribution})"] = np.full(n, np.nan) * u.dimensionless_unscaled
    for band in bands:
        if band != "K":
            q[f"Star{band}mag"] = np.full(n, np.nan) * u.dimensionless_unscaled
        for contribution in ("thermal+reflected", "thermal", "reflected"):
            q[f"Planet{band}mag({contribution})"] = np.full(n, np.nan) * u.dimensionless_unscaled
    q = get_planet_table_magnitudes(q)
    report_stat("Full synthetic photometry", "computed")

    # -----------------------------------------------------------------------------
    # 8) Filter unusable rows and save the FastYield catalog
    # -----------------------------------------------------------------------------
    report_step(8, "SNR usability checks and final catalog")
    required_valid = ["StarTeff", "StarKmag", "PlanetTeff", "PlanetMass", "PlanetRadius", "PlanetLogg", "StarLogg", "StarVsini", "StarVrotFromPlanet", "Distance", "SMA", "AngSep", "PlanetRadialVelocity", "PlanetVsini"]
    valid_snr      = np.ones(n, dtype=bool)
    for col in required_valid:
        valid_snr &= np.isfinite(q[col])
    valid_snr &= (q["PlanetTeff"] > 0.*u.K) & (q["PlanetRadius"] > 0.*u.R_earth) & (q["AngSep"] > 0.*u.mas)
    n_usable   = valid_snr.sum()
    q          = q[valid_snr]
    output     = path / f"{table}_Pull_For_FastYield.ecsv"
    write_planet_table(q, output)
    report_stat("Planets usable for SNR", f"{n_usable}/{n}")
    report_stat("Saved catalog",          output)
    report_stat("Elapsed time",          f"{(time.time()-time1)/60:.1f} mn")
    return q


