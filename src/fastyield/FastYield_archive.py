# import FastYield modules
from .config import m_u, kB, G, vesc_earth, T_sun, get_planet_table_path, instrus, bands
from .prints_helpers import print_warning
from .FastYield_helpers import get_valid_mask, get_invalid_mask, load_planet_table, chen_kipping_mass_from_radius, chen_kipping_radius_from_mass, get_evolutionary_model, get_planet_type, get_mask_planet_type

# import astropy modules
from astropy import constants as const
from astropy import units as u
from astropy.table import QTable, Column

# import numpy modules
import numpy as np

# import other modules
import time
import pyvo as vo



def inject_dace_values(planet_table):
    """
    Merge DACE values into planet_table in a vectorized way.
    https://dace.unige.ch/exoplanets/
    https://dace-query.readthedocs.io/en/latest/output_format.html

    Parameters
    ----------
    planet_table : QTable
        Archive table (columns like 'PlanetName', 'Distance', etc.)

    Returns
    -------
    planet_table : QTable
        Updated table (modified in place and also returned).
    """
    print("\nRetrieving DACE archive table from https://dace.unige.ch/exoplanets/ ...")
    from dace_query.exoplanet import Exoplanet
    planet_table_dace = Exoplanet.query_database(output_format='astropy_table')

    pt_names               = np.asarray(planet_table["PlanetName"], dtype=object)
    dc_names               = np.asarray(planet_table_dace["planet_name"], dtype=object)
    common, pt_idx, dc_idx = np.intersect1d(pt_names, dc_names, return_indices=True)
    missing                = np.setdiff1d(dc_names, common)
    print(f"\n         Injecting DACE (https://dace.unige.ch/exoplanets/) values for {len(common)} planets:")
    print(f"            {len(dc_names) - len(common)} DACE planets are not present in the NASA exo-archive: {missing}")

    def _assign(pt_colname, dc_colname, dc_unit, postproc=None):
        # DACE values as Quantity
        q = planet_table_dace[dc_colname][dc_idx] * dc_unit
        # Convert to target units if any
        pt_unit = getattr(planet_table[pt_colname], "unit", None)
        if pt_unit is not None:
            q = q.to(pt_unit)
        # Negative convention
        if pt_colname[0] == "-":
            q = q*(-1)
        # Decide where to write
        dc_valid = get_valid_mask(q)
        # Perform assignment
        planet_table[pt_colname][pt_idx[dc_valid]] = q[dc_valid]
        # Optional provenance column
        pt_colname_ref = f"{pt_colname}Ref"
        if pt_colname_ref in planet_table.colnames:
            planet_table[pt_colname_ref][pt_idx[dc_valid]]  = planet_table_dace['reference'][dc_idx][dc_valid]
            planet_table[pt_colname_ref][pt_idx[dc_valid]] += " (DACE)"

    # --- Assign fields (vectorized) ---------------------------------------
    # Period (days)
    _assign("Period", "period", u.day)
    # Semi-major axis (AU)
    _assign("SMA", "semi_major_axis", u.AU)
    # Eccentricity (dimensionless)
    _assign("Ecc", "ecc", u.dimensionless_unscaled)
    # Inclination (deg)
    _assign("Inc", "inclination", u.deg)
    # Planet mass (DACE in Mjup) and +- errors
    _assign("PlanetMass",        "planet_mass"      , u.Mjup)
    _assign("+DeltaPlanetMass",  "planet_mass_upper", u.Mjup)
    _assign("-DeltaPlanetMass",  "planet_mass_lower", u.Mjup)
    # Planet radius (DACE in Rjup) and +- errors
    _assign("PlanetRadius",       "planet_radius"      , u.Rjup)
    _assign("+DeltaPlanetRadius", "planet_radius_upper", u.Rjup)
    _assign("-DeltaPlanetRadius", "planet_radius_lower", u.Rjup)
    # Planet Teff (DACE in K) and +- errors
    _assign("PlanetTeff",       "equilibrium_temp"      , u.K)
    _assign("+DeltaPlanetTeff", "equilibrium_temp_upper", u.K)
    _assign("-DeltaPlanetTeff", "equilibrium_temp_lower", u.K)
    # RA and DEC (DACE in °)
    _assign("RA",  "right_ascension", u.degree)
    _assign("Dec", "declination",     u.degree)
    # Distance (DACE in pc) and +- errors
    _assign("Distance",       "distance",       u.pc)
    _assign("+DeltaDistance", "distance_upper", u.pc)
    _assign("-DeltaDistance", "distance_lower", u.pc)
    # Stellar RV, vsini (km/s), age (Gyr), Teff (K), M (Msun), R (Rsun), K-mag and metallicity
    _assign("StarRadialVelocity", "radial_velocity",             u.km/u.s)
    _assign("StarVsini",          "stellar_rotational_velocity", u.km/u.s)
    _assign("StarAge",            "stellar_age",                 u.Gyr)
    _assign("StarTeff",           "stellar_eff_temp",            u.K)
    _assign("StarMass",           "stellar_mass",                u.Msun)
    _assign("StarRadius",         "stellar_radius",              u.Rsun)
    _assign("StarLogg",           "stellar_surface_gravity",     u.dex(u.cm/(u.s**2)))
    _assign("StarKmag",           "k_mag",                       u.dimensionless_unscaled)
    _assign("StarFeH",            "stellar_metallicity",         u.dimensionless_unscaled)

    return planet_table




def _IKV(planet_table, planet_name, value, unit, quantity, obj, uncertainty=None, reference="inject_known_values()"):
    """
    Inject a manually curated value into a planet catalogue.

    If the planet is absent from the catalogue, a warning is printed and the
    injection is skipped. If a valid value already exists, a warning is printed
    before overwriting it.

    Values and symmetric uncertainties are converted to the target column units
    when necessary. The corresponding reference column is also updated.

    Parameters
    ----------
    planet_table : astropy.table.Table or astropy.table.QTable
        Planet catalogue containing the columns to update.

    planet_name : str
        Exact name of the planet in the "PlanetName" column.

    value : float
        Value to inject, expressed in the specified unit.

    unit : astropy.units.Unit
        Unit of the injected value.

    quantity : str
        Physical quantity to inject. Supported values are:
        "Kmag", "radius", "teff", "distance", and "sma".

    obj : str
        Object associated with the quantity: "planet", "star", or "system".
        Ignored for system-level quantities ("distance" and "sma").

    uncertainty : float, optional
        Symmetric uncertainty expressed in the same unit as value.
        If provided, both positive and negative uncertainty columns are updated.

    reference : str, optional
        Reference associated with the injected value.

    Returns
    -------
    planet_table : astropy.table.Table or astropy.table.QTable
        Original catalogue, modified in place if the planet exists.

    Raises
    ------
    ValueError
        If the quantity/object combination is unsupported, if a required
        column is missing, or if unit conversion is incompatible.
    """

    # ------------------------------------------------------------------
    # 1) Identify the target planet
    # ------------------------------------------------------------------
    indices = np.where(planet_table["PlanetName"] == planet_name)[0]
    if len(indices) == 0:
        print_warning(f"WARNING: Planet {planet_name!r} not found in the archive table. Skipping injection of {quantity} ({obj}).")
        return planet_table
    if len(indices) > 1:
        print_warning(f"WARNING: Multiple entries found for {planet_name!r} ({len(indices)} matches). Using the first occurrence.")
    idx = indices[0]

    # ------------------------------------------------------------------
    # 2) Identify the target column
    # ------------------------------------------------------------------
    quantity_lower = quantity.lower()
    obj_lower      = obj.lower()
    colname        = None
    if "kmag" in quantity_lower:
        if "planet" in obj_lower:
            colname = "PlanetKmag(thermal+reflected)"
        elif "star" in obj_lower:
            colname = "StarKmag"
    elif "radius" in quantity_lower:
        if "planet" in obj_lower:
            colname = "PlanetRadius"
        elif "star" in obj_lower:
            colname = "StarRadius"
    elif "teff" in quantity_lower:
        if "planet" in obj_lower:
            colname = "PlanetTeff"
        elif "star" in obj_lower:
            colname = "StarTeff"
    elif "distance" in quantity_lower:
        colname = "Distance"
    elif "sma" in quantity_lower:
        colname = "SMA"
    if colname is None:
        raise ValueError(f"Unsupported quantity/object combination: quantity={quantity!r}, obj={obj!r}")
    ref_colname   = f"{colname}Ref"
    plus_colname  = f"+Delta{colname}"
    minus_colname = f"-Delta{colname}"

    # ------------------------------------------------------------------
    # 3) Validate the required columns
    # ------------------------------------------------------------------
    required_columns = [colname, ref_colname]
    if uncertainty is not None:
        required_columns.extend([plus_colname, minus_colname])
    missing_columns = [name for name in required_columns if name not in planet_table.colnames]
    if missing_columns:
        raise ValueError(f"Cannot inject {quantity} for {planet_name!r}: missing columns {missing_columns}.")

    # ------------------------------------------------------------------
    # 4) Convert the value and uncertainty to the target units
    # ------------------------------------------------------------------
    target_unit = planet_table[colname].unit
    if target_unit is None:
        converted_value = value
        if uncertainty is not None:
            converted_uncertainty = np.abs(uncertainty)
    else:
        converted_value = (value * unit).to(target_unit)
        if uncertainty is not None:
            converted_uncertainty = (np.abs(uncertainty) * unit).to(target_unit)

    # ------------------------------------------------------------------
    # 5) Warn if an existing valid value will be overwritten
    # ------------------------------------------------------------------
    current_value = planet_table[colname][idx]
    if np.isfinite(current_value):
        print_warning(f"WARNING: A valid {colname} value already exists for {planet_name}: {current_value} (new value: {converted_value}). Overwriting.")

    # ------------------------------------------------------------------
    # 6) Inject the value and reference
    # ------------------------------------------------------------------
    planet_table[colname][idx]     = converted_value
    planet_table[ref_colname][idx] = reference

    # ------------------------------------------------------------------
    # 7) Inject symmetric uncertainties, if provided
    # ------------------------------------------------------------------
    if uncertainty is not None:
        planet_table[plus_colname][idx]  = +converted_uncertainty
        planet_table[minus_colname][idx] = -converted_uncertainty

    return planet_table



def inject_known_values(planet_table):
    """
    Inject manually curated values for directly imaged planets and companions.

    This function fills missing catalogue entries using literature values or
    explicitly documented model-based estimates. Values are injected through
    '_IKV', which prevents overwriting already valid entries and writes
    the corresponding reference flag.

    Notes
    -----
    - K-band magnitudes are assumed to be apparent magnitudes in a K-like band
      unless otherwise stated in the inline comments.
    - Some values are not direct K/Ks measurements but model-based conversions
      from other filters. These cases are explicitly documented in comments.
    - This manual list may need to be updated as new photometry becomes available.

    Usefull reference database
    -----------------------
    https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json

    Parameters
    ----------
    planet_table : astropy.table.Table or astropy.table.QTable
        Planet catalogue to update.

    Returns
    -------
    planet_table : astropy.table.Table or astropy.table.QTable
        Updated planet catalogue.
    """

    mag_unit = u.dimensionless_unscaled

    # ======================================================================
    # Planets with already filled temperatures
    # ======================================================================

    _IKV(planet_table, planet_name="GJ 504 b",                     value=19.38, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.11)  # https://arxiv.org/pdf/1807.00657.pdf (page 6)
    _IKV(planet_table, planet_name="2MASS J01225093-2439505 b",    value=14.53, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # https://iopscience.iop.org/article/10.1088/0004-637X/774/1/55/pdf (page 4)
    _IKV(planet_table, planet_name="HD 206893 b",                  value=15.05, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.07)  # https://iopscience.iop.org/article/10.3847/1538-3881/abc263/pdf (page 5)
    _IKV(planet_table, planet_name="HD 206893 c",                  value=16.11, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://www.aanda.org/articles/aa/pdf/2021/08/aa40749-21.pdf
    _IKV(planet_table, planet_name="GSC 06214-00210 b",            value=14.95, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.06)  # https://arxiv.org/pdf/1311.7664 (table 4)
    _IKV(planet_table, planet_name="1RXS J160929.1-210524 b",      value=16.23, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.06)  # https://arxiv.org/pdf/1311.7664 (table 4)
    _IKV(planet_table, planet_name="HIP 78530 b",                  value=14.18, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.04)  # https://arxiv.org/pdf/1503.07586.pdf (page 6)
    _IKV(planet_table, planet_name="HR 2562 b",                    value=15.52, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.12)  # https://arxiv.org/pdf/1608.06660.pdf (page 4)
    _IKV(planet_table, planet_name="HIP 65426 b",                  value=16.90, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.06)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="PDS 70 b",                     value=16.54, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)
    _IKV(planet_table, planet_name="PDS 70 c",                     value=17.70, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.20)  # https://iopscience.iop.org/article/10.3847/1538-3881/abdb2d/pdf
    _IKV(planet_table, planet_name="HIP 21152 b",                  value=16.55, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.12)  # https://iopscience.iop.org/article/10.3847/2041-8213/ac772f/pdf (page 3 bottom)
    _IKV(planet_table, planet_name="bet Pic b",                    value=12.64, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.11)  # https://www.aanda.org/articles/aa/pdf/2011/04/aa16224-10.pdf (page 3)
    _IKV(planet_table, planet_name="bet Pic c",                    value=14.30, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/pdf/2010.04442
    _IKV(planet_table, planet_name="bet Pic d",                    value=17.35, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/pdf/2606.23801 (Table 1)
    _IKV(planet_table, planet_name="HR 8799 b",                    value=17.03, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="HR 8799 c",                    value=16.11, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="HR 8799 d",                    value=16.09, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.12)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="HR 8799 e",                    value=15.91, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.22)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="HD 95086 b",                   value=18.99, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.20)  # https://www.aanda.org/articles/aa/pdf/2022/08/aa43097-22.pdf (page 5)
    _IKV(planet_table, planet_name="USco1621 b",                   value=14.67, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)
    _IKV(planet_table, planet_name="USco1556 b",                   value=14.85, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)
    _IKV(planet_table, planet_name="Oph 11 b",                     value=14.44, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.44)  # https://arxiv.org/pdf/astro-ph/0608574.pdf (page 27)
    _IKV(planet_table, planet_name="FU Tau b",                     value=13.33, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # http://cdsportal.u-strasbg.fr/?target=FU%20Tau%20b
    _IKV(planet_table, planet_name="2MASS J12073346-3932539 b",    value=16.93, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.11)  # https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (page 3)
    _IKV(planet_table, planet_name="AF Lep b",                     value=16.78, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="DH Tau b",                     value=14.19, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.07)  # https://iopscience.iop.org/article/10.1086/427086/pdf (page 3 bottom)
    _IKV(planet_table, planet_name="WD 0806-661 b",                value=27.00, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.75)  # https://arxiv.org/pdf/1605.06655.pdf (page 10)
    _IKV(planet_table, planet_name="HIP 79098 AB b",               value=14.15, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/pdf/1906.02787.pdf (page 5)
    _IKV(planet_table, planet_name="VHS J125601.92-125723.9 b",    value=14.57, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.12)  # https://www.aanda.org/articles/aa/pdf/2023/02/aa44494-22.pdf (page 2)
    _IKV(planet_table, planet_name="ROXs 42 B b",                  value=15.01, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # https://arxiv.org/pdf/1311.7664 (table 4)
    _IKV(planet_table, planet_name="2M0437 b",                     value=17.21, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.15)  # https://arxiv.org/pdf/2110.08655.pdf (page 1)
    _IKV(planet_table, planet_name="kap And b",                    value=14.32, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.09)  # https://www.aanda.org/articles/aa/pdf/2014/02/aa22119-13.pdf (page 14)
    _IKV(planet_table, planet_name="GU Psc b",                     value=17.40, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://iopscience.iop.org/article/10.1088/0004-637X/787/1/5/pdf (page 16)
    _IKV(planet_table, planet_name="GQ Lup b",                     value=13.39, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.06)  # https://arxiv.org/pdf/0801.2287 (page 9)
    _IKV(planet_table, planet_name="CT Cha b",                     value=14.95, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.30)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="COCONUTS-2 b",                 value=20.03, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://iopscience.iop.org/article/10.3847/2041-8213/ac1123/pdf (table 1)
    _IKV(planet_table, planet_name="HD 106906 b",                  value=15.46, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://iopscience.iop.org/article/10.1088/2041-8205/780/1/L4/pdf (page 4)
    _IKV(planet_table, planet_name="HD 203030 b",                  value=16.21, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.11)  # https://iopscience.iop.org/article/10.3847/1538-3881/aa9711/pdf (page 5)
    _IKV(planet_table, planet_name="BD+60 1417 b",                 value=15.65, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # https://iopscience.iop.org/article/10.3847/1538-4357/ac2499/pdf (page 5)
    _IKV(planet_table, planet_name="MWC 758 c",                    value=20.00, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.30)  # https://iopscience.iop.org/article/10.3847/1538-3881/ad11d5/pdf
    _IKV(planet_table, planet_name="HD 284149 AB b",               value=14.33, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.06)  # https://iopscience.iop.org/article/10.3847/1538-3881/ace442/pdf (page 7)
    _IKV(planet_table, planet_name="HD 135344 A b",                value=16.66, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://www.aanda.org/articles/aa/pdf/2025/08/aa55064-25.pdf (page 6); get_mag_from_flux((8.49e-17 + 9.65e-17) / 2, "J/s/m2/um", "K")
    _IKV(planet_table, planet_name="LP 261-75 b",                  value=15.14, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://arxiv.org/pdf/1710.08433 (section 2)
    #_IKV(planet_table, planet_name="CD-35 2722 b",                 value=10.37, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.09)  # https://arxiv.org/pdf/1201.3537
    _IKV(planet_table, planet_name="WISPIT 2 c",                   value=15.84, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.25)  # https://www.eso.org/public/archives/releases/sciencepapers/eso2604/eso2604a.pdf; get_mag_from_mag(T=2050, lg=3.986, model="BT-Settl", mag_input=16.40, band0_input="H", band0_output="K")
    _IKV(planet_table, planet_name="TYC 8998-760-1 c",             value=18.10, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)  # https://iopscience.iop.org/article/10.3847/2041-8213/aba27e/pdf (page 4)
    _IKV(planet_table, planet_name="2MASS J22501512+2325342 b",    value=14.89, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/html/2402.09067v1
    _IKV(planet_table, planet_name="G 196-3 b",                    value=14.86, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/pdf/1103.1880
    _IKV(planet_table, planet_name="TWA 7 b",                      value=32.91, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.19)  # https://www.nature.com/articles/s41586-025-09150-4 get_mag_from_mag(T=316, lg=2.817, model="BT-Settl", mag_input=20.54, band0_input="F1140C", band0_output="K")
    _IKV(planet_table, planet_name="HD 143811 AB b",               value=19.27, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.26)  # https://arxiv.org/html/2509.06729v1?utm.com (get_mag_from_mag(T=1042, lg=3.8873, model="BT-Settl", mag_input=19.82, band0_input="H", band0_output="K") + get_mag_from_mag(T=1042, lg=3.8873, model="BT-Settl", mag_input=16.56, band0_input="L", band0_output="K"))/2
    _IKV(planet_table, planet_name="2MASS J03590986+2009361 b",    value=14.45, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.35)  # https://arxiv.org/pdf/1907.06767 get_mag_from_mag(T=2369, lg=4.576, model="BT-Settl", mag_input=13.89, band0_input="[3.6]", band0_output="K")
    _IKV(planet_table, planet_name="2MASS J21252752-8138278 b",    value=13.00, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.25)  # https://arxiv.org/pdf/2501.07976 get_mag_from_mag(T=1616, lg=4.3857, model="BT-Settl", mag_input=15.5, band0_input="J", band0_output="K")
    _IKV(planet_table, planet_name="HIP 77900 b",                  value=15.58, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.75)  # https://arxiv.org/pdf/2102.04385 get_mag_from_mag(T=2507, lg=4.65, model="BT-Settl", mag_input=19.6, band0_input="G", band0_output="K")
    _IKV(planet_table, planet_name="eps Ind A b",                  value=24.19, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.75)  # https://arxiv.org/pdf/2603.08780 get_mag_from_mag(T=275, lg=4.08577, model="BT-Settl", mag_input=11.16, band0_input="F1550C", band0_output="K")

    # ======================================================================
    # Planets without already filled temperatures and/or additional metadata
    # ======================================================================

    _IKV(planet_table, planet_name="51 Eri b", value=18.67, unit=mag_unit,  quantity="Kmag",   obj="planet", uncertainty=0.19)  # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="51 Eri b", value=26.0,  unit=u.R_earth, quantity="radius", obj="planet", uncertainty=3.0)  # https://arxiv.org/pdf/2602.10260; logg = 3.7 and M = 3464.33 M_earth => 26 R_earth

    _IKV(planet_table, planet_name="CFHTWIR-Oph 98 b", value=16.408, unit=mag_unit, quantity="Kmag",     obj="planet", uncertainty=0.05)  # https://arxiv.org/pdf/2011.08871.pdf (page 6)
    _IKV(planet_table, planet_name="CFHTWIR-Oph 98 b", value=137.0,  unit=u.pc,     quantity="distance", obj="system", uncertainty=5.0)   # https://www.openexoplanetcatalogue.com/planet/CFHTWIR-Oph%2098%20A%20b/

    _IKV(planet_table, planet_name="CFBDSIR J145829+101343 b", value=22.83, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.30)  # https://iopscience.iop.org/article/10.1088/0004-637X/758/1/57/pdf (table 4)
    _IKV(planet_table, planet_name="CFBDSIR J145829+101343 b", value=20.6,  unit=mag_unit, quantity="Kmag", obj="star",   uncertainty=0.20)  # http://cdsportal.u-strasbg.fr/?target=CFBDSIR%20J145829%2B101343

    _IKV(planet_table, planet_name="WISEP J121756.91+162640.2 A b", value=21.10, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.25)  # https://iopscience.iop.org/article/10.1088/0004-637X/758/1/57/pdf (table 2)
    _IKV(planet_table, planet_name="WISEP J121756.91+162640.2 A b", value=18.94, unit=mag_unit, quantity="Kmag", obj="star",   uncertainty=0.15)  # https://arxiv.org/pdf/1311.2108 (table 1)

    _IKV(planet_table, planet_name="WISE J033605.05-014350.4 b", value=27.315, unit=mag_unit, quantity="Kmag",     obj="planet", uncertainty=0.60)  # https://arxiv.org/pdf/2303.16923 (table 1); mean of get_mag_from_mag(T=325, lg=4.21, model="BT-Settl", mag_input=24.87, band0_input="F150W", band0_output="K") and get_mag_from_mag(T=325, lg=4.21, model="BT-Settl", mag_input=16.51, band0_input="F480M", band0_output="K")
    _IKV(planet_table, planet_name="WISE J033605.05-014350.4 b", value=21.8,   unit=mag_unit, quantity="Kmag",     obj="star",   uncertainty=0.30)  # https://en.wikipedia.org/wiki/WISE_J0336%E2%88%920143
    _IKV(planet_table, planet_name="WISE J033605.05-014350.4 b", value=10.0,   unit=u.pc,     quantity="distance", obj="system", uncertainty=1.0)   # https://en.wikipedia.org/wiki/WISE_J0336%E2%88%920143

    _IKV(planet_table, planet_name="HIP 81208 C b", value=15.758, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://www.research-collection.ethz.ch/server/api/core/bitstreams/f5e17601-aedc-4f81-8bec-44c64a3cfa6f/content (table 2)
    _IKV(planet_table, planet_name="HIP 81208 C b", value=12.6,   unit=mag_unit, quantity="Kmag", obj="star",   uncertainty=0.03)  # https://arxiv.org/pdf/2305.19122
    _IKV(planet_table, planet_name="HIP 81208 C b", value=3165.0, unit=u.K,      quantity="teff", obj="star",   uncertainty=100)   # https://arxiv.org/pdf/2305.19122

    _IKV(planet_table, planet_name="HD 100546 b", value=15.62, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.35)  # https://home.strw.leidenuniv.nl/~kenworthy/papers/2015ApJ...807...64Q.pdf (table 2); mean of get_mag_from_mag(T=932, lg=3.75, model="BT-Settl", mag_input=13.92, band0_input="L", band0_output="K") and get_mag_from_mag(T=932, lg=3.75, model="BT-Settl", mag_input=13.33, band0_input="M", band0_output="K")
    _IKV(planet_table, planet_name="HD 100546 b", value=932.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=200)   # https://home.strw.leidenuniv.nl/~kenworthy/papers/2015ApJ...807...64Q.pdf

    _IKV(planet_table, planet_name="USco CTIO 108 b", value=15.11,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)   # https://arxiv.org/pdf/0712.3482 (table 2)
    _IKV(planet_table, planet_name="USco CTIO 108 b", value=2350.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=150.0)  # https://arxiv.org/pdf/0712.3482 (table 2)

    _IKV(planet_table, planet_name="2MASS J22362452+4751425 b", value=17.34,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)   # https://arxiv.org/pdf/1611.00364 (table 4)
    _IKV(planet_table, planet_name="2MASS J22362452+4751425 b", value=1070.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/1611.00364

    _IKV(planet_table, planet_name="TYC 8998-760-1 b", value=14.70,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.14)   # https://home.strw.leidenuniv.nl/~kenworthy/papers/2020MNRAS.492..431B.pdf (table 4)

    _IKV(planet_table, planet_name="PZ Tel b", value=11.92,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)   # https://arxiv.org/pdf/1404.2870 (table 2)
    _IKV(planet_table, planet_name="PZ Tel b", value=2500.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/1404.2870

    _IKV(planet_table, planet_name="UCAC3 113-933 b", value=18.5,   unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.35)   # https://arxiv.org/pdf/2403.04592 (table 2); mean of get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=19.94, band0_input="J", band0_output="K"), get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=19.63, band0_input="H", band0_output="K"), get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=17.91, band0_input="W1", band0_output="K"), and get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=15.56, band0_input="W2", band0_output="K")
    _IKV(planet_table, planet_name="UCAC3 113-933 b", value=1150.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/2403.04592 (table 1)

    _IKV(planet_table, planet_name="HIP 39017 b", value=17.02,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)   # https://arxiv.org/pdf/2403.04000 (section 3)
    _IKV(planet_table, planet_name="HIP 39017 b", value=1300.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/2403.04000 (section 5)

    _IKV(planet_table, planet_name="2MASS J04414489+2301513 b", value=14.94,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/pdf/1004.0539 (section 4.2)
    _IKV(planet_table, planet_name="2MASS J04414489+2301513 b", value=1800.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=150)   # https://arxiv.org/pdf/1509.01658

    _IKV(planet_table, planet_name="AB Aur b", value=14.9,   unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.40)  # https://arxiv.org/pdf/2406.00107 (section 2); get_mag_from_mag(T=2200, lg=4.25, model="BT-Settl", mag_input=15.436, band0_input="PaB", band0_output="K")
    _IKV(planet_table, planet_name="AB Aur b", value=2200.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=300)   # https://www.nature.com/articles/s41550-022-01634-x

    _IKV(planet_table, planet_name="2MASS J0249-0557 c", value=14.78,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)   # https://www.pure.ed.ac.uk/ws/portalfiles/portal/76315059/Dupuy_2018_AJ_156_57.pdf (table 4)

    _IKV(planet_table, planet_name="HD 169142 b", value=16.13,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.25)   # https://www.aanda.org/articles/aa/pdf/2019/03/aa34760-18.pdf (table 5)
    _IKV(planet_table, planet_name="HD 169142 b", value=1260.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=150.0)  # https://www.aanda.org/articles/aa/pdf/2019/03/aa34760-18.pdf

    _IKV(planet_table, planet_name="CHXR 73 b", value=15.5,   unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)   # https://arxiv.org/pdf/astro-ph/0609187
    _IKV(planet_table, planet_name="CHXR 73 b", value=2600.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=200.0)  # https://arxiv.org/pdf/0809.2812 (section 3.5)

    _IKV(planet_table, planet_name="HN Peg b", value=15.12,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)   # https://lweb.cfa.harvard.edu/~mmarengo/pub/2007ApJ...654..570L.pdf (table 3)
    _IKV(planet_table, planet_name="HN Peg b", value=1130.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://lweb.cfa.harvard.edu/~mmarengo/pub/2007ApJ...654..570L.pdf (section 3.3.3)

    _IKV(planet_table, planet_name="Ross 458 c", value=16.90, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)  # https://arxiv.org/pdf/1002.2637 (table 3)
    _IKV(planet_table, planet_name="Ross 458 c", value=695.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=75.0)  # https://arxiv.org/pdf/1103.1617

    _IKV(planet_table, planet_name="AB Pic b", value=14.14,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)   # https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (table 1)
    _IKV(planet_table, planet_name="AB Pic b", value=1700.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/2211.01474

    _IKV(planet_table, planet_name="2MASS J02192210-3925225 b", value=13.82,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.05)   # https://arxiv.org/pdf/1505.01747 (table 2)
    _IKV(planet_table, planet_name="2MASS J02192210-3925225 b", value=1700.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/1505.01747 (section 4.5)

    _IKV(planet_table, planet_name="UCAC4 328-061594 b", value=18.37,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.40)  # https://arxiv.org/pdf/2403.04592 (table 2); mean of get_mag_from_mag(T=1000, lg=4.65, model="BT-Settl", mag_input=18.07, band0_input="W1", band0_output="K") and get_mag_from_mag(T=1000, lg=4.65, model="BT-Settl", mag_input=15.82, band0_output="K")
    _IKV(planet_table, planet_name="UCAC4 328-061594 b", value=1000.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/2403.04592 (table 1)
    _IKV(planet_table, planet_name="UCAC4 328-061594 b", value=5500.0, unit=u.K,      quantity="teff", obj="star",   uncertainty=150.0)  # https://www.exoplanetkyoto.org/exohtml/UCAC4_328-061594.html

    _IKV(planet_table, planet_name="GJ 900 b", value=21.69, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.40)  # https://arxiv.org/pdf/2403.04592 (table 2); mean of get_mag_from_mag(T=500, lg=4.35, model="BT-Settl", mag_input=18.83, band0_input="W1", band0_output="K") and get_mag_from_mag(T=500, lg=4.35, model="BT-Settl", mag_input=15.90, band0_input="W2", band0_output="K")
    _IKV(planet_table, planet_name="GJ 900 b", value=500.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=75.0)  # https://arxiv.org/pdf/2403.04592 (table 1)

    _IKV(planet_table, planet_name="HIP 99770 b", value=15.59,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (section 4.1)
    _IKV(planet_table, planet_name="HIP 99770 b", value=1300.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (table 4)

    _IKV(planet_table, planet_name="2MASS J01033563-5515561 AB b", value=13.690, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.08)   # https://www.aanda.org/articles/aa/pdf/2025/09/aa54894-25.pdf (section A.3)
    _IKV(planet_table, planet_name="2MASS J01033563-5515561 AB b", value=1731.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://www.aanda.org/articles/aa/pdf/2025/09/aa54894-25.pdf (table A.2)
    _IKV(planet_table, planet_name="2MASS J01033563-5515561 AB b", value=3000.0, unit=u.K,      quantity="teff", obj="star",   uncertainty=150.0)  # https://www.openexoplanetcatalogue.com/planet/2MASS%20J01033563-5515561%20A%20b/

    _IKV(planet_table, planet_name="ROXs 12 b", value=14.32,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.06)   # https://arxiv.org/pdf/1311.7664 (table 4)
    _IKV(planet_table, planet_name="ROXs 12 b", value=2600.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/1311.7664

    _IKV(planet_table, planet_name="WISPIT 1 b", value=17.78,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.15)   # https://arxiv.org/pdf/2508.18456 (table 7)
    _IKV(planet_table, planet_name="WISPIT 1 b", value=1470.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/2508.18456

    _IKV(planet_table, planet_name="WISPIT 1 c", value=20.49,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.15)   # https://arxiv.org/pdf/2508.18456 (table 7)
    _IKV(planet_table, planet_name="WISPIT 1 c", value=1030.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (table 4)

    _IKV(planet_table, planet_name="WISPIT 2 b", value=17.57,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.12)   # https://github.com/tomasstolker/species/blob/main/species/data/companion_data/companion_data.json
    _IKV(planet_table, planet_name="WISPIT 2 b", value=1500.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=100.0)  # https://arxiv.org/pdf/2508.19046

    _IKV(planet_table, planet_name="HIP 54515 b", value=15.08,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.10)  # https://arxiv.org/pdf/2512.02159
    _IKV(planet_table, planet_name="HIP 54515 b", value=2348.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=150.0)  # https://arxiv.org/pdf/2512.02159

    _IKV(planet_table, planet_name="mu2 Sco b", value=16.01,  unit=mag_unit, quantity="Kmag",     obj="planet", uncertainty=0.15)   # https://pure-oai.bham.ac.uk/ws/portalfiles/portal/173547674/aa43675_22.pdf (table 7)
    _IKV(planet_table, planet_name="mu2 Sco b", value=2050.0, unit=u.K,      quantity="teff",     obj="planet", uncertainty=100.0)  # https://pure-oai.bham.ac.uk/ws/portalfiles/portal/173547674/aa43675_22.pdf
    _IKV(planet_table, planet_name="mu2 Sco b", value=145.0,  unit=u.pc,     quantity="distance", obj="system", uncertainty=5.0)    # https://en.wikipedia.org/wiki/Mu2_Scorpii

    _IKV(planet_table, planet_name="b Cen AB b", value=16.25,   unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.04)   # https://www.eso.org/public/archives/releases/sciencepapers/eso2118/eso2118a.pdf (table 2)
    _IKV(planet_table, planet_name="b Cen AB b", value=1600.0,  unit=u.K,      quantity="teff", obj="planet", uncertainty=150.0)  # get_evolutionary_model(planet_table[planet_table["PlanetName"] == "b Cen AB b"])
    _IKV(planet_table, planet_name="b Cen AB b", value=18445.0, unit=u.K,      quantity="teff", obj="star",   uncertainty=500.0)  # https://en.wikipedia.org/wiki/HD_129116

    _IKV(planet_table, planet_name="SR 12 AB c", value=15.05,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.25)   # https://www.nao.ac.jp/contents/about-naoj/reports/annual-report/en/2011/e_web_045.pdf; get_mag_from_mag(T=2400, lg=4.43, model="BT-Settl", mag_input=16.0, band0_input="J", band0_output="K")
    _IKV(planet_table, planet_name="SR 12 AB c", value=2600.0, unit=u.K,      quantity="teff", obj="planet", uncertainty=200.0)  # https://academic.oup.com/mnras/article/475/3/2994/4781312
    _IKV(planet_table, planet_name="SR 12 AB c", value=1083.0, unit=u.AU,     quantity="sma",  obj="system", uncertainty=50.0)   # https://www.exoplanetkyoto.org/exohtml/SR_12_AB_c.html

    _IKV(planet_table, planet_name="Luhman 16 b", value=9.73,   unit=mag_unit, quantity="Kmag",     obj="planet", uncertainty=0.05)  # https://www.eso.org/public/archives/releases/sciencepapers/eso1404/eso1404a.pdf
    _IKV(planet_table, planet_name="Luhman 16 b", value=1320.0, unit=u.K,      quantity="teff",     obj="planet", uncertainty=50.0)  # https://arxiv.org/pdf/1506.08848; https://simbad.cds.unistra.fr/simbad/sim-id?Ident=NAME+WISE+J1049-5319B
    _IKV(planet_table, planet_name="Luhman 16 b", value=1310.0, unit=u.K,      quantity="teff",     obj="star",   uncertainty=50.0)  # https://arxiv.org/pdf/1406.1518
    _IKV(planet_table, planet_name="Luhman 16 b", value=2.0,    unit=u.pc,     quantity="distance", obj="system", uncertainty=0.02)  # https://en.wikipedia.org/wiki/Luhman_16

    _IKV(planet_table, planet_name="KOINTREAU-1 b", value=16.7, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.1)  # https://iopscience.iop.org/article/10.3847/1538-3881/ae267f/pdf
    _IKV(planet_table, planet_name="KOINTREAU-1 b", value=2450, unit=u.K,      quantity="teff", obj="planet", uncertainty=300)  # https://arxiv.org/html/2512.05191v1

    _IKV(planet_table, planet_name="KOINTREAU-3 b", value=17.9, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.15)   # https://arxiv.org/pdf/2602.10198
    _IKV(planet_table, planet_name="KOINTREAU-3 b", value=1800, unit=u.K,      quantity="teff", obj="planet", uncertainty=150.0)  # https://arxiv.org/pdf/2602.10198

    _IKV(planet_table, planet_name="KOINTREAU-4 b", value=15.6, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.15)   # https://arxiv.org/pdf/2602.10198
    _IKV(planet_table, planet_name="KOINTREAU-4 b", value=1900, unit=u.K,      quantity="teff", obj="planet", uncertainty=150.0)  # https://arxiv.org/pdf/2602.10198

    _IKV(planet_table, planet_name="CWISEP J193518.59-154620.3 b", value=23.84, unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.60)  # https://arxiv.org/html/2508.17176v1 (get_mag_from_mag(T=390, lg=4.511, model="BT-Settl", mag_input=15.420, band0_input="F1000W", band0_output="K") + get_mag_from_mag(T=390, lg=4.511, model="BT-Settl", mag_input=14.211, band0_input="F1280W", band0_output="K"))/2
    _IKV(planet_table, planet_name="CWISEP J193518.59-154620.3 b", value=21.87, unit=mag_unit, quantity="Kmag", obj="star",   uncertainty=0.30)  # https://arxiv.org/pdf/2508.17176 (get_mag_from_mag(T=482, lg=4.457, model="BT-NextGen", mag_input=14.000, band0_input="F1000W", band0_output="K") + get_mag_from_mag(T=482, lg=4.457, model="BT-NextGen", mag_input=13.631, band0_input="F1280W", band0_output="K"))/2
    _IKV(planet_table, planet_name="CWISEP J193518.59-154620.3 b", value=482,   unit=u.K,      quantity="teff", obj="star",   uncertainty=60.0)  # https://arxiv.org/pdf/2508.17176

    _IKV(planet_table, planet_name="2MASS J11011926-7732383 b", value=12.81, unit=mag_unit, quantity="Kmag",     obj="planet", uncertainty=0.25)   # https://arxiv.org/pdf/astro-ph/0407344 get_mag_from_mag(T=1616, lg=4.3857, model="BT-Settl", mag_input=15.5, band0_input="J", band0_output="K")
    _IKV(planet_table, planet_name="2MASS J11011926-7732383 b", value=168.0, unit=u.pc,     quantity="distance", obj="system", uncertainty=10.0)   # https://www.aanda.org/articles/aa/full/2007/40/aa7564-07/aa7564-07.html.com
    _IKV(planet_table, planet_name="2MASS J11011926-7732383 b", value=2838,  unit=u.K,      quantity="teff",     obj="star",   uncertainty=100.0)  # https://www.aanda.org/articles/aa/full/2007/40/aa7564-07/aa7564-07.html.com

    # # Accretion detections, no Teff...
    # _IKV(planet_table, planet_name="LkCa 15 b", value=14.19,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.54)  # https://arxiv.org/pdf/1511.07456
    # _IKV(planet_table, planet_name="LkCa 15 b", value=np.nan, unit=u.K,      quantity="teff", obj="planet", uncertainty=None)  # https://arxiv.org/pdf/1511.07456
    # _IKV(planet_table, planet_name="LkCa 15 c", value=13.66,  unit=mag_unit, quantity="Kmag", obj="planet", uncertainty=0.52)  # https://arxiv.org/pdf/1511.07456
    # _IKV(planet_table, planet_name="LkCa 15 c", value=np.nan, unit=u.K,      quantity="teff", obj="planet", uncertainty=None)  # https://arxiv.org/pdf/1511.07456

    return planet_table



#def print_missing_known_values(planet_table, exclude_refs=("MICHEL__AMP__MUGRAUER_2024",)):
def print_missing_known_values(planet_table, exclude_refs=()):
    """
    Print which directly-imaged planets are still missing K-band magnitude
    and/or Teff after injecting your hand-curated “known” values.
    """
    planet_table  = planet_table[planet_table["DiscoveryMethod"]=="Imaging"].copy()
    pkm_invalid   = get_invalid_mask(planet_table["PlanetKmag(thermal+reflected)"])
    pteff_invalid = get_invalid_mask(planet_table["PlanetTeff"])

    pkm_to_print = []
    for row in planet_table[pkm_invalid]:
        ref = str(row.get("DiscoveryRef", ""))
        if not any(tag in ref for tag in exclude_refs):
            pkm_to_print.append(row["PlanetName"])
    print(f"\n {pkm_invalid.sum()}/{len(planet_table)} directly-imaged planets still missing K-band mag:")
    if pkm_to_print:
        print(pkm_to_print)
    else:
        print("  — none —")

    pteff_to_print = []
    for row in planet_table[pteff_invalid]:
        ref = str(row.get("DiscoveryRef", ""))
        if not any(tag in ref for tag in exclude_refs):
            pteff_to_print.append(row["PlanetName"])
    print(f"\n {pteff_invalid.sum()}/{len(planet_table)} directly-imaged planets still missing Teff:")
    if pteff_to_print:
        print(pteff_to_print)
    else:
        print("  — none —")



def get_archive_table(seed=None):
    """
    Build the FastYield working exoplanet catalog from the NASA Exoplanet Archive.

    Query the 'pscomppars' composite table, normalize units, inject external and
    manually curated values, complete selected missing parameters, derive orbital
    and spectroscopic quantities, filter unusable objects, compute magnitudes,
    and save the resulting ECSV catalog.

    Parameters
    ----------
    seed : int or None, optional
        Seed for 'numpy.random.default_rng'. A fixed seed reproduces random
        imputations for a fixed archive snapshot and unchanged input ordering.
        If None, the random generator is unseeded.

    Returns
    -------
    planet_table : astropy.table.QTable
        Filtered FastYield catalog with physical quantities stored as Astropy
        quantities and supplementary provenance/imputation flags.

    Data sources and files
    ----------------------
    NASA Exoplanet Archive TAP: https://exoplanetarchive.ipac.caltech.edu/TAP
    Table: 'pscomppars'. Column definitions:
    https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

    The routine writes 'Archive_Pull_Raw.ecsv' (after column normalization)
    and 'Archive_Pull_For_FastYield.ecsv' (final catalog) into
    sim_data/planet_table/Archive/.
    It also generates diagnostic plots through the existing FastYield helpers.

    Adopted conventions and approximations
    --------------------------------------
    - All planets are evaluated at the adopted quadrature phase phi = pi/2.
      The catalog is not a time-resolved ephemeris or a random-epoch survey.
    - Missing orbital inclinations are completed assuming coplanarity within
      each host system. Known inclinations are preserved and used to impute
      missing values after folding i -> min(i, 180-i). Systems without
      measured inclinations are assigned i = 90 deg if they contain transiting
      planets; otherwise one isotropic draw is used per system, with
      cos(i) uniform on [0, 1].
    - NASA masses marked 'M*sini' are converted to conditional true-mass
      estimates when the adopted inclination satisfies sin(i) >= 0.1 and the
      NASA mass is still identifiable after external injections. Such masses
      are scenario assumptions, not additional measurements. Other minimum
      masses are left unchanged and flagged.
    - Missing eccentricities are set to zero; missing arguments of periastron
      are drawn uniformly in [0, 360) degrees. NASA arguments of periastron
      are not guaranteed to share a planet-versus-star convention.
    - 'SMA' may be a projected separation for directly imaged or microlensed
      planets. For these classes, available archive angular separations are
      preferred as observational proxies; otherwise SMA/d is used. Neither
      necessarily describes a simultaneous quadrature observation. Keplerian
      quantities derived from these SMA values remain uncertain.
    - Stellar spin inclinations are approximated by orbital inclinations
      (aligned-spin assumption). Measured stellar rotation periods are used
      before observed v*sin(i), then temperature-dependent rotation priors.
      The stellar rotation seen by the planet uses an aligned, prograde,
      instantaneous-angular-speed approximation.
    - Planetary equatorial rotations use positive mass-dependent Gaussian
      priors, with zero reserved as a compatibility fallback if mass is absent.
      Planetary spin axes are assumed aligned with the orbital axes.
    - Archive 'pl_eqt' is kept as the operational 'PlanetTeff': it may represent
      an equilibrium temperature or, for directly imaged companions, an
      effective temperature. Missing temperatures are approximated by
      (Teq**4 + Tint**4)**0.25, with Bond albedo 0.3, full redistribution,
      and a Sonora Bobcat intrinsic-temperature estimate when available.
    - Rocky-planet atmospheric retention and telluric-equivalent airmass are
      heuristic proxies, not physical atmosphere detections or measurements.

    Caveats
    -------
    'pscomppars' merges parameters from different publications. Orbital
    solutions, measurement epochs, and provenance may therefore differ.
    Synthetic orientations, stellar/planetary rotations, orbital RV fallbacks,
    atmospheric flags, and evolutionary-model values are useful for comparative
    FastYield studies but must not be interpreted as precise measurements.
    """

    # Import plotting/magnitude routines only for the full catalog workflow.
    from .FastYield import get_planet_table_magnitudes, planet_table_classification, planet_table_classification_histogram, planet_table_statistics

    time1        = time.time()
    rng          = np.random.default_rng(seed=seed)
    archive_path = get_planet_table_path("Archive", create=True)

    # Consistent, compact progress reports.
    def report_step(number, title):
        print(f"\n[{number:02d}] {title}")

    def report_stat(label, value):
        print(f"         {label:<45} {value}")

    # -----------------------------------------------------------------------------
    # 1) Retrieve the composite archive table and normalize units
    # -----------------------------------------------------------------------------
    COLS_TO_PULL = (
        "pl_name, hostname, "
        "pl_orbper, pl_orbper_reflink, pl_orbsmax, pl_orbsmaxerr1, pl_orbsmaxerr2, pl_orbsmax_reflink, pl_angsep, pl_angsep_reflink, pl_orbeccen, pl_orbeccen_reflink, pl_orbincl, pl_orbincl_reflink, pl_orblper, pl_orbtper_reflink, "
        "pl_bmasse, pl_bmassprov, pl_bmasseerr1, pl_bmasseerr2, pl_bmasse_reflink, "
        "pl_rade, pl_radeerr1, pl_radeerr2, pl_rade_reflink, "
        "pl_eqt, pl_eqterr1, pl_eqterr2, pl_eqt_reflink, "
        "ra, dec, sy_dist, sy_disterr1, sy_disterr2, sy_dist_reflink, "
        "st_spectype, st_mass, st_teff, st_tefferr1, st_tefferr2, st_teff_reflink, st_rad, st_logg, st_lum, st_age, st_vsin, st_rotp, st_rotp_reflink, st_radv, st_met, "
        "sy_kmag, sy_kmagerr1, sy_kmagerr2, sy_kmag_reflink, sy_vmag, "
        "discoverymethod, disc_refname"
    )

    NEW_NAMES = [
        "PlanetName", "StarName",
        "Period", "PeriodRef", "SMA", "+DeltaSMA", "-DeltaSMA", "SMARef", "ArchiveAngSep", "ArchiveAngSepRef", "Ecc", "EccRef", "Inc", "IncRef", "ArgPeri", "ArgPeriRef",
        "PlanetMass", "PlanetMassNASAProvenance", "+DeltaPlanetMass", "-DeltaPlanetMass", "PlanetMassRef",
        "PlanetRadius", "+DeltaPlanetRadius", "-DeltaPlanetRadius", "PlanetRadiusRef",
        "PlanetTeff", "+DeltaPlanetTeff", "-DeltaPlanetTeff", "PlanetTeffRef",
        "RA", "Dec", "Distance", "+DeltaDistance", "-DeltaDistance", "DistanceRef",
        "StarSpT", "StarMass", "StarTeff", "+DeltaStarTeff", "-DeltaStarTeff", "StarTeffRef", "StarRadius", "StarLogg", "StarLum", "StarAge",
        "StarVsini", "StarRotationPeriod", "StarRotationPeriodRef", "StarRadialVelocity", "StarFeH",
        "StarKmag", "+DeltaStarKmag", "-DeltaStarKmag", "StarKmagRef", "StarVmag",
        "DiscoveryMethod", "DiscoveryRef"
    ]

    # 'pl_orblper_reflink' does not exist in pscomppars; this is only a proxy
    # for the publication describing the periastron solution, not a direct ref.
    report_step(1, "NASA Exoplanet Archive")
    svc   = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    query = f"SELECT {COLS_TO_PULL} FROM pscomppars"
    for attempt in range(3):
        try:
            table = svc.search(query, maxrec=100000).to_table()
            if len(table) == 0:
                raise RuntimeError("NASA TAP returned an empty archive table.")
            break
        except Exception as exc:
            if "502" not in str(exc) or attempt == 2:
                raise
            print(f"    TAP HTTP 502; retry {attempt + 1}/2")
            time.sleep(2**attempt)

    planet_table                      = QTable(table)
    planet_table.meta["isPSCOMPPARS"] = True
    report_stat("Archive entries", len(planet_table))
    report_stat("Selected columns", len(planet_table.colnames))

    UNIT_MAP = {
        "Earth Mass": u.M_earth, "Earth Radius": u.R_earth,
        "Solar mass": u.M_sun, "Solar Radius": u.R_sun,
        "days": u.day, "AU": u.AU, "log(Solar)": u.dex(u.solLum),
        "log10(cm/s**2)": u.dex(u.cm/u.s**2), "dex": u.dimensionless_unscaled,
        "pl_bmasseerr1": u.M_earth, "pl_bmasseerr2": u.M_earth,
        "sy_disterr1": u.pc, "sy_disterr2": u.pc,
        "pl_orbsmax": u.AU, "pl_angsep": u.mas, "st_rotp": u.day,
        "pl_orbincl": u.deg, "pl_orblper": u.deg,
        "sy_dist": u.pc, "st_teff": u.K, "pl_eqt": u.K,
        "sy_kmag": u.dimensionless_unscaled, "sy_vmag": u.dimensionless_unscaled,
    }

    for colname in planet_table.colnames:
        col          = planet_table[colname]
        current_unit = getattr(col, "unit", None)
        if current_unit is not None and str(current_unit) in UNIT_MAP:
            planet_table[colname] = col.value * UNIT_MAP[str(current_unit)]
        elif current_unit is None and colname in UNIT_MAP:
            planet_table[colname] = col.value * UNIT_MAP[colname]

    planet_table.rename_columns(COLS_TO_PULL.split(", "), NEW_NAMES)

    # Replace masked float-like columns with unmasked quantities containing NaN.
    for colname in list(planet_table.colnames):
        col = planet_table[colname]
        if col.dtype.kind in ("i", "O", "U", "S", "b"):
            if not isinstance(col, Column):
                raise KeyError(f"Expected an Astropy Column for {colname!r}.")
            continue
        if col.dtype.kind != "f":
            raise KeyError(f"Unsupported dtype {col.dtype.kind!r} for {colname!r}.")
        unit = getattr(col, "unit", None)
        mask = getattr(col, "mask", None)
        vals = col.to_value(unit) if unit is not None and hasattr(col, "to_value") else col.value
        vals = np.asarray(vals, dtype=float).copy()
        if mask is not None:
            vals[np.asarray(mask, dtype=bool)] = np.nan
        planet_table.replace_column(colname, vals * (unit if unit is not None else u.dimensionless_unscaled))

    # Keep the raw ECSV/reload convention expected by existing FastYield tools.
    planet_table.write(archive_path / "Archive_Pull_Raw.ecsv", format="ascii.ecsv", overwrite=True)
    planet_table = load_planet_table("Archive_Pull_Raw.ecsv")
    planet_table_classification_histogram(planet_table)
    report_stat("Raw snapshot", "Archive_Pull_Raw.ecsv")

    # Preserve the NASA mass and its reference before external injections.
    planet_table["PlanetMassNASAOriginal"]       = planet_table["PlanetMass"].copy()
    nasa_mass_ref                                = np.asarray(planet_table["PlanetMassRef"], dtype=str).copy()
    planet_table["PlanetMassNASADeprojected"]    = np.zeros(len(planet_table), dtype=bool)
    planet_table["PlanetMassInclinationImputed"] = np.zeros(len(planet_table), dtype=bool)
    planet_table["PlanetMassMinimumUncorrected"] = np.zeros(len(planet_table), dtype=bool)


    # -----------------------------------------------------------------------------
    # 2) Inject DACE/literature data and initialize magnitude columns
    # -----------------------------------------------------------------------------
    report_step(2, "External and literature values")
    try:
        planet_table = inject_dace_values(planet_table.copy(copy_data=True))
        report_stat("DACE injection", "completed")
    except Exception as exc:
        report_stat("DACE injection", f"skipped ({exc})")
    for instru in instrus:
        planet_table[f"StarINSTRUmag({instru})"]                      = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f"PlanetINSTRUmag({instru})(thermal+reflected)"] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f"PlanetINSTRUmag({instru})(thermal)"]           = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f"PlanetINSTRUmag({instru})(reflected)"]         = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    for band in bands:
        if band != "K":
            planet_table[f"Star{band}mag"] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f"Planet{band}mag(thermal+reflected)"] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f"Planet{band}mag(thermal)"]           = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f"Planet{band}mag(reflected)"]         = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    planet_table["+DeltaPlanetKmag(thermal+reflected)"] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    planet_table["-DeltaPlanetKmag(thermal+reflected)"] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    planet_table["PlanetKmag(thermal+reflected)Ref"]    = np.full(len(planet_table), "", dtype="<U32")

    planet_table = inject_known_values(planet_table)
    print_missing_known_values(planet_table)
    report_stat("Known-value injection", "completed")
    
    # Refresh discovery-method flags after external and manually curated injections.
    discovery_methods                          = np.asarray(planet_table["DiscoveryMethod"], dtype=str)
    tr_mask                                    = discovery_methods == "Transit"
    planet_table["SMAInterpretationUncertain"] = np.isin(discovery_methods, ["Imaging", "Microlensing"])


    # -----------------------------------------------------------------------------
    # 3) Adopt orbital inclinations and correct identifiable NASA M*sin(i) masses
    # -----------------------------------------------------------------------------
    report_step(3, "Orbital inclinations and masses")
    i_invalid                          = get_invalid_mask(planet_table["Inc"])
    i_outlier                          = get_valid_mask(planet_table["Inc"]) & ((planet_table["Inc"] < 0*u.deg) | (planet_table["Inc"] > 180*u.deg))
    i_filling                          = i_invalid | i_outlier
    planet_table["InclinationImputed"] = i_filling.copy()
    star_names                         = np.asarray(planet_table["StarName"], dtype=str)
    planet_names                       = np.asarray(planet_table["PlanetName"], dtype=str)
    star_names                         = np.where(np.char.str_len(np.char.strip(star_names)) > 0, star_names, planet_names)
    systems_without_inclination        = 0
    for star_name in np.unique(star_names):
        idx     = np.where(star_names == star_name)[0]
        missing = idx[i_filling[idx]]
        if missing.size == 0:
            continue
        known = idx[~i_filling[idx]]
        if known.size:
            inc_known = planet_table["Inc"][known].to_value(u.deg)
            inc       = np.median(np.minimum(inc_known, 180. - inc_known))
        elif np.any(tr_mask[idx]):
            inc = 90.
        else:
            inc                          = np.rad2deg(np.arccos(rng.uniform(0., 1.)))
            systems_without_inclination += 1
        planet_table["Inc"][missing] = inc * u.deg
    planet_table["sini"] = np.sin(planet_table["Inc"].to(u.rad))

    # Do not deproject externally injected masses: require the NASA value AND
    # the original mass-reference string to have survived the injections.
    mass_provenance = np.char.lower(np.char.replace(np.asarray(planet_table["PlanetMassNASAProvenance"], dtype=str), " ", ""))
    msini_mask      = np.isin(mass_provenance, ["m*sini", "m*sin(i)"])
    nasa_values     = planet_table["PlanetMassNASAOriginal"].to_value(u.M_earth)
    current_values  = planet_table["PlanetMass"].to_value(u.M_earth)
    reference_same  = np.asarray(planet_table["PlanetMassRef"], dtype=str) == nasa_mass_ref
    mass_same       = np.isclose(current_values, nasa_values, rtol=1e-12, atol=0., equal_nan=False)
    nasa_unchanged  = msini_mask & reference_same & mass_same & np.isfinite(nasa_values) & (nasa_values > 0.)
    sini_values     = planet_table["sini"].to_value(u.dimensionless_unscaled)
    deproject       = nasa_unchanged & np.isfinite(sini_values) & (sini_values >= 0.1)

    planet_table["PlanetMass"][deproject]                   = planet_table["PlanetMass"][deproject] / sini_values[deproject]
    planet_table["PlanetMassNASADeprojected"][deproject]    = True
    planet_table["PlanetMassInclinationImputed"][deproject] = i_filling[deproject]
    planet_table["PlanetMassMinimumUncorrected"]            = nasa_unchanged & ~deproject
    for colname in ["+DeltaPlanetMass", "-DeltaPlanetMass"]:
        planet_table[colname][deproject] = planet_table[colname][deproject] / sini_values[deproject]

    n_stars = len(np.unique(star_names))
    report_stat("Orbital inclinations imputed",          f"{i_filling.sum()}/{len(planet_table)} planets")
    report_stat("Isotropic system draws",                f"{systems_without_inclination}/{n_stars} systems")
    report_stat("NASA M*sin(i) deprojected",             f"{deproject.sum()}/{len(planet_table)} planets")
    report_stat("Deprojected with assumed inclinations", f"{(deproject & i_filling).sum()}/{len(planet_table)} planets")
    report_stat("Minimum masses not deprojected",        f"{planet_table['PlanetMassMinimumUncorrected'].sum()}/{len(planet_table)} planets")    


    # -----------------------------------------------------------------------------
    # 4) Complete missing planetary masses and radii (Chen & Kipping 2017)
    # -----------------------------------------------------------------------------
    
    # Missing radius from mass
    report_step(4, "Mass-radius completion (Chen & Kipping)")
    pr_invalid                                     = get_invalid_mask(planet_table["PlanetRadius"])
    pm_valid                                       = get_valid_mask(planet_table["PlanetMass"]) & (planet_table["PlanetMass"] > 0*u.M_earth)
    pr_ck                                          = chen_kipping_radius_from_mass(planet_table["PlanetMass"])
    pr_ck_valid                                    = get_valid_mask(pr_ck) & (pr_ck > 0*u.R_earth)
    pr_ck_filling                                  = pr_invalid & pm_valid & pr_ck_valid
    planet_table["PlanetRadius"][pr_ck_filling]    = pr_ck[pr_ck_filling]
    planet_table["PlanetRadiusRef"][pr_ck_filling] = "Calculated Value (Chen & Kipping 2017)"

    # Missing mass from radius
    pm_invalid                                   = get_invalid_mask(planet_table["PlanetMass"])
    pr_valid                                     = get_valid_mask(planet_table["PlanetRadius"]) & (planet_table["PlanetRadius"] > 0*u.R_earth)
    pm_ck                                        = chen_kipping_mass_from_radius(planet_table["PlanetRadius"])
    pm_ck_valid                                  = get_valid_mask(pm_ck) & (pm_ck > 0*u.M_earth)
    pm_ck_filling                                = pm_invalid & pr_valid & pm_ck_valid
    planet_table["PlanetMass"][pm_ck_filling]    = pm_ck[pm_ck_filling]
    planet_table["PlanetMassRef"][pm_ck_filling] = "Calculated Value (Chen & Kipping 2017)"

    rp            = planet_table["PlanetRadius"].to_value(u.R_earth)
    ck_degenerate = pm_invalid & pr_valid & np.isfinite(rp) & (rp >= 11.1) & (rp < 14.3)
    
    report_stat("Planetary radius from mass", f"{pr_ck_filling.sum()}/{len(planet_table)} planets")
    report_stat("Planetary mass from radius", f"{pm_ck_filling.sum()}/{len(planet_table)} planets")
    report_stat("Degenerate radii skipped",   f"{ck_degenerate.sum()}/{len(planet_table)} planets")


    # -----------------------------------------------------------------------------
    # 5) Complete stellar radii and luminosities
    # -----------------------------------------------------------------------------
    report_step(5, "Stellar radii and luminosities")
    
    lum_ratio                                 = planet_table["StarLum"].to_value(u.dex(u.solLum))
    teff                                      = planet_table["StarTeff"].to_value(u.K)
    valid_lum                                 = np.isfinite(lum_ratio) & np.isfinite(teff) & (teff > 0.)
    sr_invalid                                = get_invalid_mask(planet_table["StarRadius"])
    R_over_Rsun                               = np.full(len(planet_table), np.nan)
    R_over_Rsun[valid_lum]                    = np.sqrt(10**lum_ratio[valid_lum]) * (T_sun / teff[valid_lum])**2
    sr_filled_lum                             = sr_invalid & np.isfinite(R_over_Rsun) & (R_over_Rsun > 0.)
    planet_table["StarRadius"][sr_filled_lum] = R_over_Rsun[sr_filled_lum] * u.R_sun

    # R = sqrt(G*M/g), with StarLogg stored as a logarithmic Astropy unit.
    g_star                                     = planet_table["StarLogg"].to(u.cm/u.s**2)
    R_from_M_logg                              = np.sqrt(const.G * planet_table["StarMass"] / g_star).to(u.R_sun)
    sr_invalid                                 = get_invalid_mask(planet_table["StarRadius"])
    sr_filled_logg                             = sr_invalid & get_valid_mask(R_from_M_logg) & (R_from_M_logg > 0*u.R_sun)
    planet_table["StarRadius"][sr_filled_logg] = R_from_M_logg[sr_filled_logg]

    R_ratio                            = planet_table["StarRadius"].to_value(u.R_sun)
    valid_SB                           = np.isfinite(R_ratio) & (R_ratio > 0.) & np.isfinite(teff) & (teff > 0.)
    L_ratio                            = np.full(len(planet_table), np.nan)
    L_ratio[valid_SB]                  = R_ratio[valid_SB]**2 * (teff[valid_SB] / T_sun)**4
    sl_invalid                         = get_invalid_mask(planet_table["StarLum"])
    sl_filled                          = sl_invalid & np.isfinite(L_ratio) & (L_ratio > 0.)
    planet_table["StarLum"][sl_filled] = np.log10(L_ratio[sl_filled]) * u.dex(u.solLum)
    
    report_stat("Stellar radius from L and Teff",     f"{sr_filled_lum.sum()}/{len(planet_table)} planet entries")
    report_stat("Stellar radius from M and logg",     f"{sr_filled_logg.sum()}/{len(planet_table)} planet entries")
    report_stat("Stellar luminosity from R and Teff", f"{sl_filled.sum()}/{len(planet_table)} planet entries")
    report_stat("Stellar radius still missing",       f"{get_invalid_mask(planet_table['StarRadius']).sum()}/{len(planet_table)} planet entries")
    report_stat("Stellar luminosity still missing",   f"{get_invalid_mask(planet_table['StarLum']).sum()}/{len(planet_table)} planet entries")


    # -----------------------------------------------------------------------------
    # 6) Orbital geometry and line-of-sight velocities at adopted quadrature
    # -----------------------------------------------------------------------------
    report_step(6, "Orbital geometry and radial velocities")
    phi0                  = np.pi/2
    planet_table["Phase"] = u.Quantity(np.full(len(planet_table), phi0), u.rad, copy=False)

    e_invalid                      = get_invalid_mask(planet_table["Ecc"])
    e_outlier                      = get_valid_mask(planet_table["Ecc"]) & ((planet_table["Ecc"] < 0.) | (planet_table["Ecc"] >= 1.))
    e_filling                      = e_invalid | e_outlier
    planet_table["EccImputed"]     = e_filling.copy()
    planet_table["Ecc"][e_filling] = 0. * u.dimensionless_unscaled

    omega_invalid                          = get_invalid_mask(planet_table["ArgPeri"])
    planet_table["ArgPeriImputed"]         = omega_invalid.copy()
    planet_table["ArgPeri"][omega_invalid] = rng.uniform(0., 360., omega_invalid.sum()) * u.deg
    
    # Treat non-finite or non-positive orbital periods and semi-major axes as missing.
    P_values                                  = planet_table["Period"].to_value(u.d)
    a_values                                  = planet_table["SMA"].to_value(u.AU)
    P_invalid_initial                         = ~np.isfinite(P_values) | (P_values <= 0.)
    a_invalid_initial                         = ~np.isfinite(a_values) | (a_values <= 0.)
    planet_table["Period"][P_invalid_initial] = np.nan * u.d
    planet_table["SMA"][a_invalid_initial]    = np.nan * u.AU

    # Orbital parameters
    a     = planet_table["SMA"]
    d     = planet_table["Distance"]
    phi   = planet_table["Phase"]
    i     = planet_table["Inc"].to(u.rad)
    e     = planet_table["Ecc"]
    omega = planet_table["ArgPeri"].to(u.rad)
    
    # Use the planetary mass when available; otherwise neglect it in Keplerian calculations.
    # The fallback Mtot = Mstar remains approximate for massive companions.
    Ms        = planet_table["StarMass"]
    Mp        = planet_table["PlanetMass"]
    Ms_values = Ms.to_value(u.M_sun)
    Mp_values = Mp.to_value(u.M_earth)
    Ms_orbit  = np.where(np.isfinite(Ms_values) & (Ms_values > 0.), Ms_values, np.nan) * u.M_sun
    Mp_orbit  = np.where(np.isfinite(Mp_values) & (Mp_values > 0.), Mp_values, 0.) * u.M_earth
    Mtot      = Ms_orbit + Mp_orbit

    # Kepler's third law. For imaging/microlensing, SMA may instead be projected.
    P_from_a                               = (2*np.pi*np.sqrt(a**3/(const.G*Mtot))).to(u.d)
    P_invalid                              = get_invalid_mask(planet_table["Period"])
    P_filling                              = P_invalid & get_valid_mask(P_from_a) & (P_from_a > 0*u.d)
    planet_table["Period"][P_filling]      = P_from_a[P_filling]
    planet_table["PeriodImputed"]          = P_filling.copy()
    planet_table["PeriodFromProjectedSMA"] = P_filling & planet_table["SMAInterpretationUncertain"]

    P                              = planet_table["Period"]
    a_from_period                  = (const.G*Mtot*(P/(2*np.pi))**2)**(1/3)
    a_from_period                  = a_from_period.to(u.AU)
    a_invalid                      = get_invalid_mask(planet_table["SMA"])
    a_filling                      = a_invalid & get_valid_mask(a_from_period) & (a_from_period > 0*u.AU) & (P > 0*u.d) & (Mtot > 0*u.M_sun)
    planet_table["SMA"][a_filling] = a_from_period[a_filling]
    planet_table["SMAImputed"]     = a_filling.copy()
    
    # A semi-major axis inferred from a genuine orbital period is not a sky-projection measurement.
    planet_table["SMAInterpretationUncertain"][a_filling] = False
    a                                                     = planet_table["SMA"]

    # Check Keplerian consistency only for independently available orbital parameters.
    # Exclude imputed periods/SMA and uncertain projected separations.
    # Flag discrepancies exceeding 20%, which may reflect inconsistent NASA references.
    P_kepler                           = (2*np.pi*np.sqrt(a**3/(const.G*Mtot))).to(u.d)
    P_consistent_mask                  = get_valid_mask(P) & get_valid_mask(P_kepler) & (P > 0*u.d) & (P_kepler > 0*u.d) & ~P_filling & ~a_filling & ~planet_table["SMAInterpretationUncertain"]
    P_ratio                            = np.full(len(planet_table), np.nan)
    P_ratio[P_consistent_mask]         = (P[P_consistent_mask] / P_kepler[P_consistent_mask]).to_value(u.dimensionless_unscaled)
    planet_table["KeplerInconsistent"] = P_consistent_mask & (np.abs(P_ratio - 1.) > 0.2)

    # nu is the true anomaly for the adopted photometric-phase convention.
    # The NASA omega convention can differ by 180 degrees between references.
    nu = phi - omega - (np.pi/2)*u.rad
    r  = a * (1. - e**2) / (1. + e*np.cos(nu))

    # For genuine orbital semi-major axes, project the instantaneous separation.
    # For ambiguous SMA (imaging/microlensing), use archive sky separation when
    # available, otherwise the projected-separation proxy SMA/d.
    planet_table["AngSep"]                    = (r/d * np.sqrt(np.cos(nu+omega)**2 + np.sin(nu+omega)**2 * np.cos(i)**2)).to(u.mas, equivalencies=u.dimensionless_angles())
    uncertain_sma                             = np.asarray(planet_table["SMAInterpretationUncertain"], dtype=bool)
    archive_sep_valid                         = get_valid_mask(planet_table["ArchiveAngSep"]) & (planet_table["ArchiveAngSep"] > 0*u.mas)
    use_archive_sep                           = uncertain_sma & archive_sep_valid
    use_projected_sma                         = uncertain_sma & ~archive_sep_valid & get_valid_mask(a) & (a > 0*u.AU) & get_valid_mask(d) & (d > 0*u.pc)
    planet_table["AngSep"][use_archive_sep]   = planet_table["ArchiveAngSep"][use_archive_sep].to(u.mas)
    planet_table["AngSep"][use_projected_sma] = (a[use_projected_sma] / d[use_projected_sma]).to(u.mas, equivalencies=u.dimensionless_angles())
    planet_table["AngSepFromArchive"]         = use_archive_sep.copy()
    planet_table["AngSepProjectedProxy"]      = use_projected_sma.copy()

    # Velocity relative to the star (not the planet's barycentric RV amplitude).
    Kp_from_period                      = (2*np.pi*a/P * np.sin(i)/np.sqrt(1.-e**2)).to(u.km/u.s)
    Kp_from_mass                        = (np.sqrt(const.G*Mtot/a) * np.sin(i)/np.sqrt(1.-e**2)).to(u.km/u.s)
    Kp                                  = Kp_from_period.copy()
    Kp_filling                          = get_invalid_mask(Kp) & get_valid_mask(Kp_from_mass)
    Kp[Kp_filling]                      = Kp_from_mass[Kp_filling]
    planet_table["DeltaRadialVelocity"] = Kp * (np.cos(nu+omega) + e*np.cos(omega))

    # Stellar systemic velocity: use a measured value within the same host,
    # otherwise draw once per host from the measured Galactic RV distribution.
    srv_values  = planet_table["StarRadialVelocity"].to_value(u.km/u.s)
    srv_known   = np.isfinite(srv_values)
    srv_mean    = np.nanmedian(srv_values) if srv_known.any() else 0.
    srv_std     = np.nanstd(srv_values) if srv_known.sum() > 1 else 0.
    srv_imputed = ~srv_known.copy()
    for star_name in np.unique(star_names):
        idx     = np.where(star_names == star_name)[0]
        known   = idx[np.isfinite(srv_values[idx])]
        missing = idx[~np.isfinite(srv_values[idx])]
        if missing.size == 0:
            continue
        rv = np.nanmedian(srv_values[known]) if known.size else rng.normal(srv_mean, srv_std)
        planet_table["StarRadialVelocity"][missing] = rv * u.km/u.s
        srv_values[missing] = rv
    planet_table["StarRadialVelocityImputed"] = srv_imputed

    # Last-resort statistical DeltaRV completion. Values are explicitly flagged;
    # they are not equivalent to orbital solutions and can bias HRS estimates.
    drv_values                                       = planet_table["DeltaRadialVelocity"].to_value(u.km/u.s)
    drv_known                                        = np.isfinite(drv_values)
    drv_missing                                      = ~drv_known
    drv_mean                                         = np.nanmedian(drv_values[drv_known]) if drv_known.any() else 0.
    drv_std                                          = np.nanstd(drv_values[drv_known]) if drv_known.sum() > 1 else 0.
    planet_table["DeltaRadialVelocity"][drv_missing] = rng.normal(drv_mean, drv_std, drv_missing.sum()) * u.km/u.s
    planet_table["DeltaRadialVelocityImputed"]       = drv_missing.copy()
    planet_table["PlanetRadialVelocity"]             = planet_table["StarRadialVelocity"] + planet_table["DeltaRadialVelocity"]
    
    # Track orbital RV estimates relying on uncertain or assumed orbital parameters.
    planet_table["DeltaRVFromOrbit"]      = ~drv_missing
    planet_table["DeltaRVOrbitUncertain"] = (~drv_missing) & (i_filling | uncertain_sma | planet_table["KeplerInconsistent"] | (omega_invalid & (e > 0.)))
        
    # Lambertian phase function. At quadrature alpha = pi/2 and g = 1/pi.
    alpha                   = np.arccos(np.clip(-np.sin(i)*np.cos(phi), -1., 1.))
    planet_table["alpha"]   = alpha.to(u.rad)
    planet_table["g_alpha"] = ((np.sin(alpha) + (np.pi-alpha.to_value(u.rad))*np.cos(alpha))/np.pi) * u.dimensionless_unscaled

    sep_from_geometry = ~uncertain_sma & get_valid_mask(planet_table["AngSep"])
    report_stat("Orbital phase",                "pi/2 rad (quadrature, all planets)")
    report_stat("Eccentricities set to zero",   f"{e_filling.sum()}/{len(planet_table)} planets")
    report_stat("Periastron angles drawn",      f"{omega_invalid.sum()}/{len(planet_table)} planets")
    report_stat("Periods inferred from SMA",    f"{P_filling.sum()}/{len(planet_table)} planets")
    report_stat("SMA inferred from periods",    f"{a_filling.sum()}/{len(planet_table)} planets")
    report_stat("Periods from uncertain SMA",   f"{planet_table['PeriodFromProjectedSMA'].sum()}/{len(planet_table)} planets")
    report_stat("Kepler mismatch (>20%)",       f"{planet_table['KeplerInconsistent'].sum()}/{P_consistent_mask.sum()} tested orbits")
    report_stat("AngSep from orbital geometry", f"{sep_from_geometry.sum()}/{len(planet_table)} planets")
    report_stat("AngSep from NASA archive",     f"{use_archive_sep.sum()}/{len(planet_table)} planets")
    report_stat("AngSep from SMA/d proxy",      f"{use_projected_sma.sum()}/{len(planet_table)} planets")
    report_stat("Systemic RV imputed",          f"{srv_imputed.sum()}/{len(planet_table)} planets")
    report_stat("DeltaRV statistically drawn",  f"{drv_missing.sum()}/{len(planet_table)} planets")


    # -----------------------------------------------------------------------------
    # 7) Stellar rotation and apparent stellar rotation
    # -----------------------------------------------------------------------------
    report_step(7, "Stellar rotation")
    planet_table["StarVsiniObserved"]    = planet_table["StarVsini"].copy()
    planet_table["StarVrot"]             = np.full(len(planet_table), np.nan) * u.km/u.s
    planet_table["StarSpinSini"]         = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    planet_table["StarVrotImputed"]      = np.zeros(len(planet_table), dtype=bool)
    planet_table["StarVrotFromPeriod"]   = np.zeros(len(planet_table), dtype=bool)
    planet_table["StarVrotFromVsini"]    = np.zeros(len(planet_table), dtype=bool)
    planet_table["StarSpinSiniAssumed"]  = np.ones(len(planet_table), dtype=bool)
    for star_name in np.unique(star_names):
        idx = np.where(star_names == star_name)[0]

        # Under spin-orbit alignment, the stellar inclination follows the
        # orbital inclination (measured or imputed) adopted for this system.
        orbital_sini   = planet_table["sini"][idx].to_value(u.dimensionless_unscaled)
        valid_inc      = np.isfinite(orbital_sini) & (orbital_sini >= 0.)
        spin_sini      = np.median(orbital_sini[valid_inc]) if valid_inc.any() else np.nan
        observed_vsini = planet_table["StarVsiniObserved"][idx].to_value(u.km/u.s)
        observed_vsini = observed_vsini[np.isfinite(observed_vsini) & (observed_vsini >= 0.)]
        periods        = planet_table["StarRotationPeriod"][idx].to_value(u.d)
        radii          = planet_table["StarRadius"][idx].to_value(u.R_sun)
        valid_rotation = np.isfinite(periods) & (periods > 0.) & np.isfinite(radii) & (radii > 0.)
        if valid_rotation.any():
            vrot    = np.median((2*np.pi*planet_table["StarRadius"][idx][valid_rotation] / planet_table["StarRotationPeriod"][idx][valid_rotation]).to_value(u.km/u.s))
            source  = "period"
        elif observed_vsini.size and np.isfinite(spin_sini) and spin_sini >= 0.1:
            vrot    = np.median(observed_vsini) / spin_sini
            source  = "vsini"
        else:
            teff_values = planet_table["StarTeff"][idx].to_value(u.K)
            teff_values = teff_values[np.isfinite(teff_values) & (teff_values > 0.)]
            teff_star   = np.median(teff_values) if teff_values.size else np.nan
            if np.isfinite(teff_star):
                if teff_star <= 3500.:
                    mean, std = 1., 0.5
                elif teff_star <= 6000.:
                    mean, std = 3., 1.
                elif teff_star <= 7500.:
                    mean, std = 10., 5.
                elif teff_star <= 10000.:
                    mean, std = 120., 60.
                else:
                    mean, std = 200., 80.
                vrot = rng.normal(mean, std)
                while vrot < 0.:
                    vrot = rng.normal(mean, std)
                source = "prior"
            else:
                vrot   = np.nan
                source = "missing"
        vsini                                   = np.median(observed_vsini) if observed_vsini.size else vrot*spin_sini
        planet_table["StarSpinSini"][idx]       = spin_sini * u.dimensionless_unscaled
        planet_table["StarVrot"][idx]           = vrot * u.km/u.s
        planet_table["StarVsini"][idx]          = vsini * u.km/u.s
        planet_table["StarVrotImputed"][idx]    = source == "prior"
        planet_table["StarVrotFromPeriod"][idx] = source == "period"
        planet_table["StarVrotFromVsini"][idx]  = source == "vsini"
    planet_table["StarVsiniModel"] = planet_table["StarVrot"] * planet_table["StarSpinSini"]
    
    # Apparent stellar rotation as seen from the planet: aligned prograde model.
    # d(nu)/dt is the instantaneous Keplerian angular velocity, not 2*pi/P
    # for an eccentric orbit. Missing orbital inputs fall back to StarVrot.
    planet_table["StarVrotFromPlanet"] = planet_table["StarVrot"].copy()
    svfp_valid                         = (get_valid_mask(planet_table["StarVrot"]) & get_valid_mask(planet_table["StarRadius"]) & get_valid_mask(P) & (planet_table["StarRadius"] > 0*u.R_sun) & (P > 0*u.d) & ~uncertain_sma)
    with np.errstate(invalid="ignore", divide="ignore"):
        orbital_angular_velocity = (2*np.pi/P) * (1.+e*np.cos(nu))**2 / (1.-e**2)**1.5
    orbital_rotation_term                          = np.full(len(planet_table), np.nan) * u.km/u.s
    orbital_rotation_term[svfp_valid]              = (planet_table["StarRadius"][svfp_valid] * orbital_angular_velocity[svfp_valid]).to(u.km/u.s)
    planet_table["StarVrotFromPlanet"][svfp_valid] = np.abs(planet_table["StarVrot"][svfp_valid] - orbital_rotation_term[svfp_valid])
    planet_table["StarVrotFromPlanetFallback"]     = get_valid_mask(planet_table["StarVrot"]) & ~svfp_valid

    # Stellar diagnostics count unique hosts, not the associated planet rows.
    def count_stars(mask):
        return np.unique(star_names[np.asarray(mask, dtype=bool)]).size
    n_stars = len(np.unique(star_names))
    report_stat("Stellar rotation from period",    f"{count_stars(planet_table['StarVrotFromPeriod'])}/{n_stars} stars")
    report_stat("Stellar rotation from vsini",     f"{count_stars(planet_table['StarVrotFromVsini'])}/{n_stars} stars")
    report_stat("Stellar rotation from prior",     f"{count_stars(planet_table['StarVrotImputed'])}/{n_stars} stars")
    report_stat("Stellar Vrot seen from planet",   f"{svfp_valid.sum()}/{len(planet_table)} planets")


    # -----------------------------------------------------------------------------
    # 8) Complete missing planetary temperatures
    # -----------------------------------------------------------------------------
    report_step(8, "Planetary temperatures")
    bond_albedo = 0.3
    with np.errstate(invalid="ignore", divide="ignore"):
        pteq = planet_table["StarTeff"] * np.sqrt((planet_table["StarRadius"]/(2*planet_table["SMA"])).decompose()) * (1.-bond_albedo)**0.25
    planet_table["PlanetTeq"] = pteq.copy()

    # Sonora Bobcat: intrinsic non-irradiated cooling temperature and radius.
    ptint_em, pr_em, _                                                             = get_evolutionary_model(planet_table)
    planet_table["PlanetTint"]                                                     = ptint_em.copy()
    pteq_values                                                                    = pteq.to_value(u.K)
    ptint_values                                                                   = ptint_em.to_value(u.K)
    pteq_valid                                                                     = np.isfinite(pteq_values) & (pteq_values > 0.)
    ptint_em_valid                                                                 = np.isfinite(ptint_values) & (ptint_values > 0.)
    pteff_invalid                                                                  = get_invalid_mask(planet_table["PlanetTeff"])
    pteff_filling                                                                  = pteff_invalid & (pteq_valid | ptint_em_valid)
    pteq_safe                                                                      = np.where(pteq_valid, pteq_values, 0.)
    ptint_safe                                                                     = np.where(ptint_em_valid, ptint_values, 0.)
    pteff_values                                                                   = (pteq_safe**4 + ptint_safe**4)**0.25
    planet_table["PlanetTeff"][pteff_filling]                                      = pteff_values[pteff_filling] * u.K
    planet_table["PlanetTeffRef"][pteff_filling]                                   = f"Equilibrium (A_B={bond_albedo}, 4pi) + intrinsic Sonora temperature where available"
    planet_table["PlanetTeffOrigin"]                                               = np.full(len(planet_table), "Archive/injected", dtype="<U32")
    planet_table["PlanetTeffOrigin"][pteff_filling & pteq_valid & ptint_em_valid]  = "Teq+Tint model"
    planet_table["PlanetTeffOrigin"][pteff_filling & pteq_valid & ~ptint_em_valid] = "Teq model"
    planet_table["PlanetTeffOrigin"][pteff_filling & ~pteq_valid & ptint_em_valid] = "Tint model"
    planet_table["PlanetTeffOrigin"][pteff_invalid & ~pteff_filling]               = "Unavailable"
    
    report_stat("Planetary Teff completed",        f"{pteff_filling.sum()}/{len(planet_table)} planets")
    report_stat("Including intrinsic temperature", f"{(pteff_filling & ptint_em_valid).sum()}/{len(planet_table)} planets")
    report_stat("Planetary Teff still missing",    f"{get_invalid_mask(planet_table['PlanetTeff']).sum()}/{len(planet_table)} planets")


    # -----------------------------------------------------------------------------
    # 9) Complete radii, masses, and planetary rotation
    # -----------------------------------------------------------------------------
    report_step(9, "Evolutionary radii, Chen & Kipping masses, and planetary rotation")
    pr_invalid                                   = get_invalid_mask(planet_table["PlanetRadius"])
    pr_em_valid                                  = get_valid_mask(pr_em) & (pr_em > 0*u.R_earth)
    pr_filling                                   = pr_invalid & pr_em_valid
    planet_table["PlanetRadius"][pr_filling]     = pr_em[pr_filling]
    planet_table["PlanetRadiusRef"][pr_filling]  = "Evolutionary model"
    pm_invalid                                   = get_invalid_mask(planet_table["PlanetMass"])
    pr_valid                                     = get_valid_mask(planet_table["PlanetRadius"]) & (planet_table["PlanetRadius"] > 0*u.R_earth)
    pm_ck                                        = chen_kipping_mass_from_radius(planet_table["PlanetRadius"])
    pm_ck_valid                                  = get_valid_mask(pm_ck) & (pm_ck > 0*u.M_earth)
    pm_ck_filling                                = pm_invalid & pr_valid & pm_ck_valid
    planet_table["PlanetMass"][pm_ck_filling]    = pm_ck[pm_ck_filling]
    planet_table["PlanetMassRef"][pm_ck_filling] = "Calculated Value (Chen & Kipping 2017)"
    rp                                           = planet_table["PlanetRadius"].to_value(u.R_earth)
    ck_degenerate                                = pm_invalid & pr_valid & np.isfinite(rp) & (rp >= 11.1) & (rp < 14.3)
    
    # Check whether late mass completion significantly changes the Keplerian mass budget.
    late_mass_ratio       = (planet_table["PlanetMass"][pm_ck_filling] / planet_table["StarMass"][pm_ck_filling]).to_value(u.dimensionless_unscaled)
    late_mass_significant = np.isfinite(late_mass_ratio) & (late_mass_ratio > 0.01)
        
    # Positive mass-dependent Gaussian priors for planetary equatorial rotation.
    # A zero fallback is retained for missing masses for backward compatibility.
    Mp                         = planet_table["PlanetMass"]
    planet_table["PlanetVrot"] = np.zeros(len(planet_table)) * u.km/u.s
    p1                         = (Mp            > 0*u.M_earth) & (Mp <= 5*u.M_earth)
    p2                         = (5*u.M_earth   < Mp)          & (Mp <= 20*u.M_earth)
    p3                         = (20*u.M_earth  < Mp)          & (Mp <= 100*u.M_earth)
    p4                         = (100*u.M_earth < Mp)          & (Mp <= 300*u.M_earth)
    p5                         = Mp > 300*u.M_earth
    priors                     = [(p1, 0.5, 0.5), (p2, 2., 1.), (p3, 3., 1.5), (p4, 12., 4.), (p5, 20., 8.)]
    vrot                       = planet_table["PlanetVrot"].to_value(u.km/u.s)
    for mask, mean, std in priors:
        vrot[mask] = rng.normal(mean, std, mask.sum())
        negative   = mask & (vrot < 0.)
        while negative.any():
            vrot[negative] = rng.normal(mean, std, negative.sum())
            negative       = mask & (vrot < 0.)
    planet_table["PlanetVrot"]                 = vrot * u.km/u.s
    planet_table["PlanetVsini"]                = planet_table["PlanetVrot"] * planet_table["sini"]
    planet_table["PlanetVrotMassUnavailable"] = ~(p1 | p2 | p3 | p4 | p5)
            
    report_stat("Planetary radius from evolution",  f"{pr_filling.sum()}/{len(planet_table)} planets")
    report_stat("Planetary mass from new radius",   f"{pm_ck_filling.sum()}/{len(planet_table)} planets")
    report_stat("Degenerate radii skipped",         f"{ck_degenerate.sum()}/{len(planet_table)} planets")
    report_stat("Planetary rotation without mass",  f"{planet_table['PlanetVrotMassUnavailable'].sum()}/{len(planet_table)} planets")
    report_stat("Late masses > 1% of stellar mass", f"{late_mass_significant.sum()}/{pm_ck_filling.sum()} newly completed planets")

    
    # -----------------------------------------------------------------------------
    # 10) Compute surface gravities
    # -----------------------------------------------------------------------------
    report_step(10, "Surface gravities")
    g_planet                             = const.G * planet_table["PlanetMass"] / planet_table["PlanetRadius"]**2
    planet_table["PlanetLogg"]            = np.log10(g_planet.to_value(u.cm/u.s**2)) * u.dex(u.cm/u.s**2)
    g_star                                = const.G * planet_table["StarMass"] / planet_table["StarRadius"]**2
    g_star_values                         = np.log10(g_star.to_value(u.cm/u.s**2))
    slg_invalid                           = get_invalid_mask(planet_table["StarLogg"])
    slg_filling                           = slg_invalid & np.isfinite(g_star_values)
    planet_table["StarLogg"][slg_filling] = g_star_values[slg_filling] * u.dex(u.cm/u.s**2)
    
    report_stat("Stellar logg completed",   f"{slg_filling.sum()}/{len(planet_table)} planet entries")
    report_stat("Planetary logg available", f"{get_valid_mask(planet_table['PlanetLogg']).sum()}/{len(planet_table)} planets")


    # -----------------------------------------------------------------------------
    # 11) Filter planets outside the adopted physical/model parameter ranges
    # -----------------------------------------------------------------------------
    report_step(11, "Planetary parameter-range filtering")
    M_min, M_max       = 0.*u.M_earth, 80.*u.M_jup
    R_min, R_max       = 0.*u.R_earth, 40.*u.R_earth
    Teff_min, Teff_max = 0.*u.K, 3000.*u.K

    Mp, Rp, Teff         = planet_table["PlanetMass"], planet_table["PlanetRadius"], planet_table["PlanetTeff"]
    Mp_valid             = get_valid_mask(Mp)
    Rp_valid             = get_valid_mask(Rp)
    Teff_valid           = get_valid_mask(Teff)
    Mp_outlier           = Mp_valid & ((Mp <= M_min) | (Mp > M_max))
    Rp_outlier           = Rp_valid & ((Rp <= R_min) | (Rp > R_max))
    Teff_outlier         = Teff_valid & ((Teff <= Teff_min) | (Teff > Teff_max))
    planet_param_outlier = Mp_outlier | Rp_outlier | Teff_outlier

    report_stat("Allowed mass",            "(0, 80] Mjup")
    report_stat("Allowed radius",          "(0, 40] Rearth")
    report_stat("Allowed temperature",     "(0, 3000] K")
    report_stat("Rejected by mass",        f"{Mp_outlier.sum()}/{len(planet_table)} planets")
    report_stat("Rejected by radius",      f"{Rp_outlier.sum()}/{len(planet_table)} planets")
    report_stat("Rejected by temperature", f"{Teff_outlier.sum()}/{len(planet_table)} planets")
    report_stat("Total planets rejected",  f"{planet_param_outlier.sum()}/{len(planet_table)} planets")
    report_stat("Planets retained",        f"{(~planet_param_outlier).sum()}/{len(planet_table)} planets")
    if planet_param_outlier.any():
        rejected = np.asarray(planet_table["PlanetName"][planet_param_outlier], dtype=str)
        report_stat("Rejected examples", ", ".join(rejected[:3]) + (" ..." if rejected.size > 3 else ""))
    planet_table = planet_table[~planet_param_outlier]

    # Refresh discovery-method masks after row selection.
    im_mask = planet_table["DiscoveryMethod"] == "Imaging"
    rv_mask = planet_table["DiscoveryMethod"] == "Radial Velocity"
    tr_mask = planet_table["DiscoveryMethod"] == "Transit"
    ot_mask = (~im_mask) & (~rv_mask) & (~tr_mask)


    # -----------------------------------------------------------------------------
    # 12) Filter rows required by FastYield S/N calculations
    # -----------------------------------------------------------------------------
    report_step(12, "S/N usability filtering")
    steff_valid = get_valid_mask(planet_table["StarTeff"]) & (planet_table["StarTeff"] > 0*u.K)
    pteff_valid = get_valid_mask(planet_table["PlanetTeff"]) & (planet_table["PlanetTeff"] > 0*u.K)
    skm_valid   = get_valid_mask(planet_table["StarKmag"])
    pr_valid    = (get_valid_mask(planet_table["PlanetRadius"]) & (planet_table["PlanetRadius"] > 0*u.R_earth)) | get_valid_mask(planet_table["PlanetKmag(thermal+reflected)"])
    d_valid     = get_valid_mask(planet_table["Distance"]) & (planet_table["Distance"] > 0*u.pc)
    sma_valid   = get_valid_mask(planet_table["SMA"]) & (planet_table["SMA"] > 0*u.AU)
    snr_valid   = steff_valid & pteff_valid & skm_valid & pr_valid & d_valid & sma_valid

    def missing_percentage(valid, mask):
        return f"{100*np.count_nonzero(~valid & mask)/np.count_nonzero(mask):.0f}%" if np.any(mask) else "--"
    for label, valid in [("StarTeff", steff_valid), ("PlanetTeff", pteff_valid), ("StarKmag", skm_valid), ("Radius / planet Kmag", pr_valid), ("Distance", d_valid), ("SMA", sma_valid)]:
        report_stat(f"Missing {label}", f"IM {missing_percentage(valid, im_mask)} | RV {missing_percentage(valid, rv_mask)} | TR {missing_percentage(valid, tr_mask)} | OT {missing_percentage(valid, ot_mask)}")
    report_stat("Planets usable for S/N", f"{snr_valid.sum()}/{len(planet_table)} planets")
    planet_table = planet_table[snr_valid]


    # -----------------------------------------------------------------------------
    # 13) Classify planets and complete missing surface gravities
    # -----------------------------------------------------------------------------
    report_step(13, "Planet classification and logg priors")
    planet_table["PlanetType"] = np.full(len(planet_table), "Unidentified", dtype="<U48")
    for idx in range(len(planet_table)):
        planet_table[idx]["PlanetType"] = get_planet_type(planet_table[idx])

    slg_missing_initial = get_invalid_mask(planet_table["StarLogg"]).copy()
    plg_missing_initial = get_invalid_mask(planet_table["PlanetLogg"]).copy()
    star_teff           = planet_table["StarTeff"].to_value(u.K)
    for tmin, tmax in [(0, 3500), (3500, 6000), (6000, 7500), (7500, np.inf)]:
        group   = (star_teff >= tmin) & (star_teff < tmax)
        valid   = group & get_valid_mask(planet_table["StarLogg"])
        missing = group & get_invalid_mask(planet_table["StarLogg"])
        if valid.any():
            planet_table["StarLogg"][missing] = np.nanmedian(planet_table["StarLogg"][valid])

    planet_types_available = np.unique(np.asarray(planet_table["PlanetType"], dtype=str))
    for ptype in planet_types_available:
        group   = planet_table["PlanetType"] == ptype
        valid   = group & get_valid_mask(planet_table["PlanetLogg"])
        missing = group & get_invalid_mask(planet_table["PlanetLogg"])
        if valid.any():
            planet_table["PlanetLogg"][missing] = np.nanmedian(planet_table["PlanetLogg"][valid])

    planet_radii = planet_table["PlanetRadius"].to_value(u.R_earth)
    for idx in np.where(get_invalid_mask(planet_table["PlanetLogg"]))[0]:
        radius = planet_radii[idx]
        if not np.isfinite(radius) or radius <= 0.:
            continue
        comparable = get_valid_mask(planet_table["PlanetLogg"]) & np.isfinite(planet_radii) & (planet_radii >= radius/1.4) & (planet_radii <= radius*1.4)
        if comparable.sum() >= 10:
            planet_table["PlanetLogg"][idx] = np.nanmedian(planet_table["PlanetLogg"][comparable])

    discovery_methods = np.asarray(planet_table["DiscoveryMethod"], dtype=str)
    for method in np.unique(discovery_methods):
        group   = discovery_methods == method
        valid   = group & get_valid_mask(planet_table["PlanetLogg"])
        missing = group & get_invalid_mask(planet_table["PlanetLogg"])
        if valid.any() and missing.any():
            planet_table["PlanetLogg"][missing] = np.nanmedian(planet_table["PlanetLogg"][valid])

    plg_missing_after                 = get_invalid_mask(planet_table["PlanetLogg"])
    planet_table["StarLoggImputed"]   = slg_missing_initial & get_valid_mask(planet_table["StarLogg"])
    planet_table["PlanetLoggImputed"] = plg_missing_initial & get_valid_mask(planet_table["PlanetLogg"])
    
    report_stat("Stellar logg imputed",         f"{planet_table['StarLoggImputed'].sum()}/{len(planet_table)} planet entries")
    report_stat("Planetary logg imputed",       f"{planet_table['PlanetLoggImputed'].sum()}/{len(planet_table)} planets")
    report_stat("Planetary logg still missing", f"{plg_missing_after.sum()}/{len(planet_table)} planets")
    if plg_missing_after.any():
        raise ValueError(f"Still missing PlanetLogg for: {list(planet_table['PlanetName'][plg_missing_after])}")


    # -----------------------------------------------------------------------------
    # 14) Atmospheric-retention flags for rocky planets (heuristics)
    # -----------------------------------------------------------------------------
    report_step(14, "Rocky-planet atmospheric proxies")
    planet_table["HasAtmosphere"] = np.ones(len(planet_table), dtype=bool)
    mu                            = 28.
    lambda_jeans_thr              = 100.
    m                             = mu * m_u
    M                             = planet_table["PlanetMass"].to_value(u.kg)
    R                             = planet_table["PlanetRadius"].to_value(u.m)
    T                             = planet_table["PlanetTeff"].to_value(u.K)
    with np.errstate(invalid="ignore", divide="ignore"):
        lambda_jeans = G*M*m/(kB*T*R)
        I_rel        = planet_table["StarLum"].to_value(u.L_sun) / planet_table["SMA"].to_value(u.AU)**2
        vesc         = 1e-3*np.sqrt(2*G*M/R)
        I_crit       = 320.*(vesc/vesc_earth)**4
    no_retention                                 = lambda_jeans < lambda_jeans_thr
    blown_atmosphere                             = I_rel > I_crit
    rocky_planets                                = get_mask_planet_type(planet_table, planet_type="earth")
    no_atmosphere                                = rocky_planets & (no_retention | blown_atmosphere)
    planet_table["HasAtmosphere"][no_atmosphere] = False
    jeans_available                              = np.isfinite(lambda_jeans) & (lambda_jeans > 0.)
    shore_available                              = np.isfinite(I_rel) & np.isfinite(vesc) & np.isfinite(I_crit) & (I_crit > 0.)
    planet_table["AtmosphereJeansAvailable"]     = jeans_available
    planet_table["AtmosphereShorelineAvailable"] = shore_available
    planet_table["AtmosphereCriteriaAvailable"]  = jeans_available | shore_available
    planet_table["AtmosphereCriteriaComplete"]   = jeans_available & shore_available
    planet_table["AtmosphereStatusUncertain"]    = rocky_planets & ~(jeans_available & shore_available) & ~no_atmosphere
    
    report_stat("Rocky planets",                 f"{rocky_planets.sum()}/{len(planet_table)} planets")
    report_stat("Failed Jeans criterion",        f"{(rocky_planets & no_retention).sum()}/{rocky_planets.sum()} rocky planets")
    report_stat("Failed shoreline criterion",    f"{(rocky_planets & blown_atmosphere).sum()}/{rocky_planets.sum()} rocky planets")
    report_stat("Classified without atmosphere", f"{no_atmosphere.sum()}/{rocky_planets.sum()} rocky planets")
    report_stat("Atmosphere undetermined",       f"{planet_table['AtmosphereStatusUncertain'].sum()}/{rocky_planets.sum()} rocky planets")


    # -----------------------------------------------------------------------------
    # 15) Telluric-equivalent airmass (gravity-based reflected-light proxy)
    # -----------------------------------------------------------------------------
    report_step(15, "Telluric-equivalent airmass")
    P_ref, P_ref_earth                        = 1.*u.bar, 1.*u.bar
    eta, X_min, X_max                         = 1., 0.5, 8.
    g_earth                                   = (const.G*const.M_earth/const.R_earth**2).to(u.m/u.s**2)
    g_planet                                  = planet_table["PlanetLogg"].to(u.m/u.s**2)
    X_eq                                      = 2.*(P_ref/P_ref_earth)*(g_earth/g_planet)**eta
    planet_table["TelluricEquivalentAirmass"] = np.clip(X_eq.to_value(u.dimensionless_unscaled), X_min, X_max) * u.dimensionless_unscaled
    report_stat("Proxy range", f"[{X_min}, {X_max}]")
    report_stat("Reference",   "Earth-like gravity -> X_eq = 2")


    # -----------------------------------------------------------------------------
    # 16) Compute synthetic magnitudes, save, and plot diagnostics
    # -----------------------------------------------------------------------------
    report_step(16, "Synthetic magnitudes and final catalog")
    planet_table = get_planet_table_magnitudes(planet_table)
    planet_table.write(archive_path / "Archive_Pull_For_FastYield.ecsv", format="ascii.ecsv", overwrite=True)
    final_stars = np.asarray(planet_table["StarName"], dtype=str)
    final_names = np.asarray(planet_table["PlanetName"], dtype=str)
    final_stars = np.where(np.char.str_len(np.char.strip(final_stars)) > 0, final_stars, final_names)
    report_stat("Final planets",      len(planet_table))
    report_stat("Final host systems", len(np.unique(final_stars)))
    report_stat("Saved catalog",      "Archive_Pull_For_FastYield.ecsv")
    report_stat("Elapsed time",       f"{(time.time()-time1)/60:.1f} mn")
    planet_table_classification(planet_table)
    planet_table_classification_histogram(planet_table)
    planet_table_statistics(planet_table)
    return planet_table