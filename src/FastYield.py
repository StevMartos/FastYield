from src.FastCurves import *



# Models list
thermal_models   = ["None", "BT-Settl", "Exo-REM", "PICASO"]
reflected_models = ["None", "tellurics", "flat", "PICASO"]

# Ignore reflected contribution if lmin > 6 µm
ignore_reflected_thresh_um = 6 # [µm]

# Planet types list [M_earth, R_earth, K]
planet_types = {
    # ---- Cold (Teq < 500 K)
    "Cold Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teq_min": 0, "teq_max": 500},
    "Cold Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teq_min": 0, "teq_max": 500},
    "Cold Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teq_min": 0, "teq_max": 500},
    "Cold Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teq_min": 0, "teq_max": 500},
    "Cold Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teq_min": 0, "teq_max": 500},
    "Cold Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": 500},
    "Cold Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": 500},
    "Cold Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": 500},
    # ---- Warm (500 ≤ Teq < 1000 K)
    "Warm Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teq_min": 500, "teq_max": 1000},
    "Warm Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teq_min": 500, "teq_max": 1000},
    "Warm Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teq_min": 500, "teq_max": 1000},
    "Warm Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teq_min": 500, "teq_max": 1000},
    "Warm Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teq_min": 500, "teq_max": 1000},
    "Warm Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 500, "teq_max": 1000},
    "Warm Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 500, "teq_max": 1000},
    "Warm Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teq_min": 500, "teq_max": 1000},
    # ---- Hot (Teq ≥ 1000 K)
    "Hot Sub-Earth":       {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teq_min": 1000, "teq_max": np.inf},
    "Hot Earth-like":      {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teq_min": 1000, "teq_max": np.inf},
    "Hot Super-Earth":     {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teq_min": 1000, "teq_max": np.inf},
    "Hot Sub-Neptune":     {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teq_min": 1000, "teq_max": np.inf},
    "Hot Neptune-like":    {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teq_min": 1000, "teq_max": np.inf},
    "Hot Saturn-like":     {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 1000, "teq_max": np.inf},
    "Hot Jupiter-like":    {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 1000, "teq_max": np.inf},
    "Hot Super-Jupiter":   {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teq_min": 1000, "teq_max": np.inf},
}

# Planet types list [M_earth, R_earth, K]
planet_types_semireduced = {
    # ---- Cold (Teq < 500 K)
    "Cold Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teq_min": 0, "teq_max": 500},
    "Cold Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teq_min": 0, "teq_max": 500},
    "Cold Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teq_min": 0, "teq_max": 500},
    "Cold Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teq_min": 0, "teq_max": 500},
    "Cold Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teq_min": 0, "teq_max": 500},
    "Cold Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": 500},
    "Cold Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": 500},
    "Cold Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": 500},
    # ---- Hot (Teq ≥ 1000 K)
    "Hot Sub-Earth":       {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teq_min": 500, "teq_max": np.inf},
    "Hot Earth-like":      {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teq_min": 500, "teq_max": np.inf},
    "Hot Super-Earth":     {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teq_min": 500, "teq_max": np.inf},
    "Hot Sub-Neptune":     {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teq_min": 500, "teq_max": np.inf},
    "Hot Neptune-like":    {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teq_min": 500, "teq_max": np.inf},
    "Hot Saturn-like":     {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 500, "teq_max": np.inf},
    "Hot Jupiter-like":    {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 500, "teq_max": np.inf},
    "Hot Super-Jupiter":   {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teq_min": 500, "teq_max": np.inf},
}

# Planet types list  [M_earth, R_earth]
planet_types_reduced = {
    "Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teq_min": 0, "teq_max": np.inf},
    "Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teq_min": 0, "teq_max": np.inf},
    "Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teq_min": 0, "teq_max": np.inf},
    "Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teq_min": 0, "teq_max": np.inf},
    "Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teq_min": 0, "teq_max": np.inf},
    "Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": np.inf},
    "Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": np.inf},
    "Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teq_min": 0, "teq_max": np.inf},
}

# Paths
path_file      = os.path.dirname(__file__)
archive_path   = os.path.join(os.path.dirname(path_file), "sim_data/Archive_table/")
simulated_path = os.path.join(os.path.dirname(path_file), "sim_data/Simulated_table/")

SNR_threshold = 5 # Detection threshold for yields



#######################################################################################################################
#################################################### Utils function: #################################################
#######################################################################################################################

def mass_to_size(mass, s0, ds, mass_min=None, mass_max=None):
    # éviter log10(0)
    if mass_min is None:
        mass_min = np.nanmin(mass[np.isfinite(mass) & (mass > 0)])
    if mass_max is None:
        mass_max = np.nanmax(mass[np.isfinite(mass) & (mass > 0)])

    t = (np.log10(np.clip(mass, mass_min, mass_max)) - np.log10(mass_min)) / \
        (np.log10(mass_max) - np.log10(mass_min))
    t = np.clip(t, 0, 1)
    return s0 + ds * t



def load_planet_table(table_name):
    """Load table_name from the appropriate path."""
    path = archive_path if "Archive" in table_name else simulated_path
    return QTable.read(path + table_name, format="ascii.ecsv")



def get_valid_mask(col):
    """Get a mask with the valid values of the planet_table's column"""
    return np.isfinite(col)



def get_invalid_mask(col):
    """Get a mask with the invalid values of the planet_table's column"""
    return np.logical_not(get_valid_mask(col))



def get_planet_index(planet_table, planet_name):
    """Row index for a given planet name (exact match)."""
    return np.where(planet_table["PlanetName"] == planet_name)[0][0]



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
    print(f"\nInjecting DACE (https://dace.unige.ch/exoplanets/) values for {len(common)} planets:")
    print(f"   {len(dc_names) - len(common)} DACE planets are not present in the NASA exo-archive: {missing}")

    def _assign(pt_colname, dc_colname, dc_unit, postproc=None):
        # DACE values as Quantity
        q = planet_table_dace[dc_colname][dc_idx] * dc_unit
        # Convert to target units if any
        pt_unit = getattr(planet_table[pt_colname], "unit", None)
        if pt_unit is not None:
            q = q.to(pt_unit)
        # Negative convention
        if pt_colname[0] == "-":
            q *= -1 
        # Decide where to write
        dc_valid = get_valid_mask(q)
        # Perform assignment
        planet_table[pt_colname][pt_idx[dc_valid]] = q[dc_valid]
        # Optional provenance column
        pt_colname_ref = f"{pt_colname}Ref"
        if pt_colname_ref in planet_table.colnames:
            planet_table[pt_colname_ref][pt_idx[dc_valid]] = planet_table_dace['reference'][dc_idx][dc_valid]
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
    # Planet Teq (DACE in K) and +- errors
    _assign("PlanetTeq",       "equilibrium_temp"      , u.K)
    _assign("+DeltaPlanetTeq", "equilibrium_temp_upper", u.K)
    _assign("-DeltaPlanetTeq", "equilibrium_temp_lower", u.K)
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
    _assign("StarZ",              "stellar_metallicity",         u.dimensionless_unscaled)
    
    return planet_table



def inject_known_values(planet_table):
    """
    Injects known values of planets detected by direct imaging from a variety of references, (may need to be updated as required)
    """
    # Planets with already filled temperatures
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "GJ 504 b"]                      = 19.4         # https://arxiv.org/pdf/1807.00657.pdf (page 6)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J01225093-2439505 b"]     = 14.53        # https://iopscience.iop.org/article/10.1088/0004-637X/774/1/55/pdf (page 4)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 206893 b"]                   = 15.05        # https://iopscience.iop.org/article/10.3847/1538-3881/abc263/pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "GSC 06214-00210 b"]             = 14.95        # https://arxiv.org/pdf/1311.7664 (table 4)
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "1RXS J160929.1-210524 b"]       = 16.23        # https://arxiv.org/pdf/1311.7664 (table 4)        
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 78530 b"]                   = 14.18        # https://arxiv.org/pdf/1503.07586.pdf (page 6)              
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HR 2562 b"]                     = 5.02+10.5    # https://arxiv.org/pdf/1608.06660.pdf (page 4)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 65426 b"]                   = 6.771+9.85   # https://arxiv.org/pdf/1707.01413.pdf (page 8)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "PDS 70 c"]                      = 8.542+8.8    # https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "PDS 70 b"]                      = 8.542+8.0    # https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 21152 b"]                   = 16.55        # https://iopscience.iop.org/article/10.3847/2041-8213/ac772f/pdf (page 3 bas)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "bet Pic b"]                     = 3.48+9.2     # https://www.aanda.org/articles/aa/pdf/2011/04/aa16224-10.pdf (page 3)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HR 8799 b"]                     = 14.05        # https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HR 8799 c"]                     = 13.13        # https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HR 8799 d"]                     = 13.11        # https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HR 8799 e"]                     = 5.24+10.67   # https://arxiv.org/ftp/arxiv/papers/1011/1011.4918.pdf (page 10)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 95086 b"]                    = 6.789+12.2   # https://www.aanda.org/articles/aa/pdf/2022/08/aa43097-22.pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "USco1621 b"]                    = 14.67        # https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "USco1556 b"]                    = 14.85        # https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "Oph 11 b"]                      = 14.44        # https://arxiv.org/pdf/astro-ph/0608574.pdf (page 27)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "FU Tau b"]                      = 13.329       # http://cdsportal.u-strasbg.fr/?target=FU%20Tau%20b            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J12073346-3932539 b"]     = 16.93        # https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (page 3)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "AF Lep b"]                      = 4.926+11.7   # https://arxiv.org/pdf/2302.06213.pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "DH Tau b"]                      = 14.19        # https://iopscience.iop.org/article/10.1086/427086/pdf (page 3 bas)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WD 0806-661 b"]                 = 27           # https://arxiv.org/pdf/1605.06655.pdf (page 10)          
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 79098 AB b"]                = 14.15        # https://arxiv.org/pdf/1906.02787.pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "VHS J125601.92-125723.9 b"]     = 14.57        # https://www.aanda.org/articles/aa/pdf/2023/02/aa44494-22.pdf (page 2)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "ROXs 42 B b"]                   = 15.01        # https://arxiv.org/pdf/1311.7664 (table 4)  
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2M0437 b"]                      = 17.21        # https://arxiv.org/pdf/2110.08655.pdf (page 1)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "kap And b"]                     = 14.32        # https://www.aanda.org/articles/aa/pdf/2014/02/aa22119-13.pdf (page 14)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "GU Psc b"]                      = 17.40        # https://iopscience.iop.org/article/10.1088/0004-637X/787/1/5/pdf (page 16)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "GQ Lup b"]                      = 13.33        # https://arxiv.org/pdf/1311.7664 (table 4)
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "CT Cha b"]                      = 14.9         # https://www.aanda.org/articles/aa/pdf/2008/43/aa8840-07.pdf (page 4)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 206893 c"]                   = 15.2         # https://www.aanda.org/articles/aa/pdf/2021/08/aa40749-21.pdf     
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "TYC 8998-760-1 c"]              = 8.3+9.8      # https://iopscience.iop.org/article/10.3847/2041-8213/aba27e/pdf (page 4)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "COCONUTS-2 b"]                  = 20.030       # https://iopscience.iop.org/article/10.3847/2041-8213/ac1123/pdf (table 1) 
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "51 Eri b"]                      = 15.8         # https://www.aanda.org/articles/aa/pdf/2023/05/aa44826-22.pdf
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 106906 b"]                   = 15.46        # https://iopscience.iop.org/article/10.1088/2041-8205/780/1/L4/pdf (page 4)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 203030 b"]                   = 16.21        # https://iopscience.iop.org/article/10.3847/1538-3881/aa9711/pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "BD+60 1417 b"]                  = 15.645       # https://iopscience.iop.org/article/10.3847/1538-4357/ac2499/pdf (page 5)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 75056 A b"]                 = 7.3+6.8      # https://arxiv.org/pdf/2009.08537.pdf (page 3)            
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "MWC 758 c"]                     = 20           # https://iopscience.iop.org/article/10.3847/1538-3881/ad11d5/pdf        
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 284149 AB b"]                = 14.332       # https://iopscience.iop.org/article/10.3847/1538-3881/ace442/pdf (page 7) 
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 135344 A b"]                 = 16.66        # https://www.aanda.org/articles/aa/pdf/2025/08/aa55064-25.pdf (page 6) : get_mag_from_flux((8.49e-17+9.65e-17)/2, "J/s/m2/um", "K")
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "LP 261-75 b"]                   = 15.14        # https://arxiv.org/pdf/1710.08433 (section 2)

    # Planets without already filled temperatures et al.
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "CFHTWIR-Oph 98 b"]              = 16.408       # https://arxiv.org/pdf/2011.08871.pdf (page 6)            
    planet_table["Distance"][planet_table["PlanetName"]                      == "CFHTWIR-Oph 98 b"]              = 137 * u.pc   # https://www.openexoplanetcatalogue.com/planet/CFHTWIR-Oph%2098%20A%20b/

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "CFBDSIR J145829+101343 b"]      = 22.83        # https://iopscience.iop.org/article/10.1088/0004-637X/758/1/57/pdf (table 4)
    planet_table["StarKmag"][planet_table["PlanetName"]                      == "CFBDSIR J145829+101343 b"]      = 20.6         # http://cdsportal.u-strasbg.fr/?target=CFBDSIR%20J145829%2B101343
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISEP J121756.91+162640.2 A b"] = 21.10        # https://iopscience.iop.org/article/10.1088/0004-637X/758/1/57/pdf (table 2)
    planet_table["StarKmag"][planet_table["PlanetName"]                      == "WISEP J121756.91+162640.2 A b"] = 18.94        # https://arxiv.org/pdf/1311.2108 (table 1)

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISE J033605.05-014350.4 b"]    = 27.315       # https://arxiv.org/pdf/2303.16923 (table 1): (get_mag_from_mag(T=325, lg=4.21, model="BT-Settl", mag_input=24.87, band0_input="F150W", band0_output="K") + get_mag_from_mag(T=325, lg=4.21, model="BT-Settl", mag_input=16.51, band0_input="F480M", band0_output="K")) / 2
    planet_table["StarKmag"][planet_table["PlanetName"]                      == "WISE J033605.05-014350.4 b"]    = 21.8         # https://en.wikipedia.org/wiki/WISE_J0336%E2%88%920143
    planet_table["Distance"][planet_table["PlanetName"]                      == "WISE J033605.05-014350.4 b"]    = 10 * u.pc    # https://en.wikipedia.org/wiki/WISE_J0336%E2%88%920143

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 81208 C b"]                 = 6.768+8.99   # https://www.research-collection.ethz.ch/server/api/core/bitstreams/f5e17601-aedc-4f81-8bec-44c64a3cfa6f/content (table 2)
    planet_table["StarKmag"][planet_table["PlanetName"]                      == "HIP 81208 C b"]                 = 12.6         # https://arxiv.org/pdf/2305.19122
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "HIP 81208 C b"]                 = 3165 * u.K   # https://arxiv.org/pdf/2305.19122
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 100546 b"]                   = 15.62        # https://home.strw.leidenuniv.nl/~kenworthy/papers/2015ApJ...807...64Q.pdf (table 2): (get_mag_from_mag(T=932, lg=3.75, model="BT-Settl", mag_input=13.92, band0_input="L", band0_output="K") + get_mag_from_mag(T=932, lg=3.75, model="BT-Settl", mag_input=13.33, band0_input="M", band0_output="K")) / 2
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "HD 100546 b"]                   = 932 * u.K    # https://home.strw.leidenuniv.nl/~kenworthy/papers/2015ApJ...807...64Q.pdf
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "HD 100546 b"]                   = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "USco CTIO 108 b"]               = 15.11        # https://arxiv.org/pdf/0712.3482 (table 2)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "USco CTIO 108 b"]               = 2350 * u.K   # https://arxiv.org/pdf/0712.3482 (table 2)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "USco CTIO 108 b"]               = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J22362452+4751425 b"]     = 17.34        # https://arxiv.org/pdf/1611.00364 (table 4)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "2MASS J22362452+4751425 b"]     = 1070 * u.K   # https://arxiv.org/pdf/1611.00364
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "2MASS J22362452+4751425 b"]     = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "TYC 8998-760-1 b"]              = 14.70        # https://home.strw.leidenuniv.nl/~kenworthy/papers/2020MNRAS.492..431B.pdf (table 4)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "TYC 8998-760-1 b"]              = 1727 * u.K   # https://home.strw.leidenuniv.nl/~kenworthy/papers/2020MNRAS.492..431B.pdf (section 4.2.2)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "TYC 8998-760-1 b"]              = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISPIT 2 b"]                    = 16.84        # https://arxiv.org/pdf/2508.19046 (table 3): get_mag_from_mag(T=1500, lg=3.67, model="BT-Settl", mag_input=15.3, band0_input="L", band0_output="K")
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "WISPIT 2 b"]                    = 1500 * u.K   # https://arxiv.org/pdf/2508.19046
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "WISPIT 2 b"]                    = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "PZ Tel b"]                      = 6.366+5.55   # https://arxiv.org/pdf/1404.2870 (table 2)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "PZ Tel b"]                      = 2500 * u.K   # https://arxiv.org/pdf/1404.2870
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "PZ Tel b"]                      = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "UCAC3 113-933 b"]               = 18.5         # https://arxiv.org/pdf/2403.04592 (table 2): (get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=19.94, band0_input="J", band0_output="K") + get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=19.63, band0_input="H", band0_output="K") + get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=17.91, band0_input="W1", band0_output="K") + get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=15.56, band0_input="W2", band0_output="K")) / 4
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "UCAC3 113-933 b"]               = 1150 * u.K   # https://arxiv.org/pdf/2403.04592 (table 1)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "UCAC3 113-933 b"]               = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 39017 b"]                   = 17.02        # https://arxiv.org/pdf/2403.04000 (section 3)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "HIP 39017 b"]                   = 1300 * u.K   # https://arxiv.org/pdf/2403.04000 (section 5)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "HIP 39017 b"]                   = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J04414489+2301513 b"]     = 14.94        # https://arxiv.org/pdf/1004.0539 (section 4.2)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "2MASS J04414489+2301513 b"]     = 1800 * u.K   # https://arxiv.org/pdf/1509.01658
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "2MASS J04414489+2301513 b"]     = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "AB Aur b"]                      = 14.9         # https://arxiv.org/pdf/2406.00107 (section 2): get_mag_from_mag(T=2200, lg=4.25, model="BT-Settl", mag_input=15.436, band0_input="PaB", band0_output="K")
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "AB Aur b"]                      = 2200 * u.K   # https://www.nature.com/articles/s41550-022-01634-x
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "AB Aur b"]                      = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J0249-0557 c"]            = 14.78        # https://www.pure.ed.ac.uk/ws/portalfiles/portal/76315059/Dupuy_2018_AJ_156_57.pdf (table 4)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "2MASS J0249-0557 c"]            = 1700 * u.K   # https://www.pure.ed.ac.uk/ws/portalfiles/portal/76315059/Dupuy_2018_AJ_156_57.pdf
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "2MASS J0249-0557 c"]            = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 169142 b"]                   = 6.41+9.72    # https://www.aanda.org/articles/aa/pdf/2019/03/aa34760-18.pdf (table 5)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "HD 169142 b"]                   = 1260 * u.K   # https://www.aanda.org/articles/aa/pdf/2019/03/aa34760-18.pdf
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "HD 169142 b"]                   = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "CHXR 73 b"]                     = 15.5         # https://arxiv.org/pdf/astro-ph/0609187
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "CHXR 73 b"]                     = 2600 * u.K   # https://arxiv.org/pdf/0809.2812 (section 3.5)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "CHXR 73 b"]                     = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HN Peg b"]                      = 15.12        # https://lweb.cfa.harvard.edu/~mmarengo/pub/2007ApJ...654..570L.pdf (table 3)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "HN Peg b"]                      = 1130 * u.K   # https://lweb.cfa.harvard.edu/~mmarengo/pub/2007ApJ...654..570L.pdf (section 3.3.3)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "HN Peg b"]                      = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "Ross 458 c"]                    = 16.90        # https://arxiv.org/pdf/1002.2637 (table 3)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "Ross 458 c"]                    = 695 * u.K    # https://arxiv.org/pdf/1103.1617
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "Ross 458 c"]                    = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "AB Pic b"]                      = 14.14        # https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (table 1)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "AB Pic b"]                      = 1700 * u.K   # https://arxiv.org/pdf/2211.01474
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "AB Pic b"]                      = "inject_known_values"
 
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J02192210-3925225 b"]     = 13.82        # https://arxiv.org/pdf/1505.01747 (table 2)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "2MASS J02192210-3925225 b"]     = 1700 * u.K   # https://arxiv.org/pdf/1505.01747 (section 4.5)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "2MASS J02192210-3925225 b"]     = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "UCAC4 328-061594 b"]            = 18.37        # https://arxiv.org/pdf/2403.04592 (table 2): (get_mag_from_mag(T=1000, lg=4.65, model="BT-Settl", mag_input=18.07, band0_input="W1", band0_output="K") + get_mag_from_mag(T=1000, lg=4.65, model="BT-Settl", mag_input=15.82, band0_input="W2", band0_output="K") ) / 2
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "UCAC4 328-061594 b"]            = 1000 * u.K   # https://arxiv.org/pdf/2403.04592 (table 1)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "UCAC4 328-061594 b"]            = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "UCAC4 328-061594 b"]            = 5500 * u.K   # https://www.exoplanetkyoto.org/exohtml/UCAC4_328-061594.html

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "GJ 900 b"]                      = 21.69        # https://arxiv.org/pdf/2403.04592 (table 2): (get_mag_from_mag(T=500, lg=4.35, model="BT-Settl", mag_input=18.83, band0_input="W1", band0_output="K") + get_mag_from_mag(T=500, lg=4.35, model="BT-Settl", mag_input=15.90, band0_input="W2", band0_output="K") ) / 2
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "GJ 900 b"]                      = 500 * u.K    # https://arxiv.org/pdf/2403.04592 (table 1)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "GJ 900 b"]                      = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 99770 b"]                   = 12.460       # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (section 4.1)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "HIP 99770 b"]                   = 1300 * u.K   # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (table 4)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "HIP 99770 b"]                   = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J01033563-5515561 AB b"]  = 13.690       # https://www.aanda.org/articles/aa/pdf/2025/09/aa54894-25.pdf (section A.3)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "2MASS J01033563-5515561 AB b"]  = 1731 * u.K   # https://www.aanda.org/articles/aa/pdf/2025/09/aa54894-25.pdf (table A.2)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "2MASS J01033563-5515561 AB b"]  = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "2MASS J01033563-5515561 AB b"]  = 3000 * u.K   # https://www.openexoplanetcatalogue.com/planet/2MASS%20J01033563-5515561%20A%20b/

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "ROXs 12 b"]                     = 14.32        # https://arxiv.org/pdf/1311.7664 (table 4)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "ROXs 12 b"]                     = 2600 * u.K   # https://arxiv.org/pdf/1311.7664
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "ROXs 12 b"]                     = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISPIT 1 b"]                    = 17.78        # https://arxiv.org/pdf/2508.18456 (table 7)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "WISPIT 1 b"]                    = 1470 * u.K   # https://arxiv.org/pdf/2508.18456
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "WISPIT 1 b"]                    = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISPIT 1 c"]                    = 20.49        # https://arxiv.org/pdf/2508.18456 (table 7)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "WISPIT 1 c"]                    = 1030 * u.K   # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (table 4)
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "WISPIT 1 c"]                    = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "mu2 Sco b"]                     = 16.01        # https://pure-oai.bham.ac.uk/ws/portalfiles/portal/173547674/aa43675_22.pdf (table 7)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "mu2 Sco b"]                     = 2050 * u.K   # https://pure-oai.bham.ac.uk/ws/portalfiles/portal/173547674/aa43675_22.pdf
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "mu2 Sco b"]                     = "inject_known_values"
    planet_table["Distance"][planet_table["PlanetName"]                      == "mu2 Sco b"]                     = 145 * u.pc   # https://en.wikipedia.org/wiki/Mu2_Scorpii

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "b Cen AB b"]                    = 16.37        # https://www.eso.org/public/archives/releases/sciencepapers/eso2118/eso2118a.pdf (table 2)
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "b Cen AB b"]                    = 1600 * u.K   # get_evolutionary_model(planet_table[planet_table["PlanetName"] == "b Cen AB b"])
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "b Cen AB b"]                    = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "b Cen AB b"]                    = 18445 * u.K  # https://en.wikipedia.org/wiki/HD_129116

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "SR 12 AB c"]                    = 15.05        # https://www.nao.ac.jp/contents/about-naoj/reports/annual-report/en/2011/e_web_045.pdf: get_mag_from_mag(T=2400, lg=4.43, model="BT-Settl", mag_input=16.0, band0_input="J", band0_output="K")
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "SR 12 AB c"]                    = 2600 * u.K   # https://academic.oup.com/mnras/article/475/3/2994/4781312
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "SR 12 AB c"]                    = "inject_known_values"
    planet_table["SMA"][planet_table["PlanetName"]                           == "SR 12 AB c"]                    = 1083 * u.AU  # https://www.exoplanetkyoto.org/exohtml/SR_12_AB_c.html

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "Luhman 16 b"]                   = 9.73         # https://www.eso.org/public/archives/releases/sciencepapers/eso1404/eso1404a.pdf
    planet_table["PlanetTeq"][planet_table["PlanetName"]                     == "Luhman 16 b"]                   = 1320 * u.K   # https://arxiv.org/pdf/1506.08848 / https://simbad.cds.unistra.fr/simbad/sim-id?Ident=NAME+WISE+J1049-5319B
    planet_table["PlanetTeqRef"][planet_table["PlanetName"]                  == "Luhman 16 b"]                   = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "Luhman 16 b"]                   = 1310 * u.K   # https://arxiv.org/pdf/1406.1518
    planet_table["Distance"][planet_table["PlanetName"]                      == "Luhman 16 b"]                   = 2.0 * u.pc   # https://en.wikipedia.org/wiki/Luhman_16
    
    return planet_table



def print_missing_known_values(planet_table, exclude_refs=("MICHEL__AMP__MUGRAUER_2024",)):
    """
    Print which directly-imaged planets are still missing K-band magnitude
    and/or Teq after injecting your hand-curated “known” values.
    """
    planet_table = planet_table[planet_table["DiscoveryMethod"]=="Imaging"].copy()
    planet_table = inject_known_values(planet_table)
    
    pkm_valid = get_valid_mask(planet_table["PlanetKmag(thermal+reflected)"])
    pte_valid = get_valid_mask(planet_table["PlanetTeq"])
    valid     = pkm_valid & pte_valid

    still_missing = ~valid
    to_print      = []
    for row in planet_table[still_missing]:
        ref = str(row.get("DiscoveryRef", ""))
        if not any(tag in ref for tag in exclude_refs):
            to_print.append(row["PlanetName"])
            
    print(f" {still_missing.sum()}/{len(planet_table)} directly-imaged planets still missing K-band mag or Teq:")
    if to_print:
        print("\n", to_print)
    else:
        print("\n  — none —")



def get_evolutionary_model(planet_table):
    """
    Estimate (Teff, Radius, Lbol) from an evolutionary grid for each row of 'planet_table'.
    
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
    EVOL_GRID_NPZ = "sim_data/Archive_table/sonora_bobcat_evol_grid.npz"
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
    a[a < domain["amin"]] = np.nan * u.Gyr
    a[a > domain["amax"]] = np.nan * u.Gyr
    m[m < domain["mmin"]] = np.nan * u.M_jup
    m[m > domain["mmax"]] = np.nan * u.M_jup
        
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



def get_planet_type(planet):
    """
    Classify a single planet row using the global 'planet_types' rules.

    Expects
    -------
    planet : astropy.table.Row (from your main table)
        Must provide 'PlanetMass', 'PlanetRadius', 'PlanetTeq' as quantities
        convertible to M_earth, R_earth, and K, respectively.

    Returns
    -------
    str
        A human-readable planet type label (first matching rule),
        or an informative 'Unidentified' string if no match / missing data.
    """
    # Extract properties from the planet
    mass   = planet["PlanetMass"].value
    radius = planet["PlanetRadius"].value
    teq    = planet["PlanetTeq"].value
    
    # If values are missing, return "Unidentified"
    if not np.isfinite(mass):
        return "Unidentified - missing mass"
    if not np.isfinite(radius):
        return "Unidentified - missing radius"
    if not np.isfinite(teq):
        return "Unidentified - missing temperature"
    
    # Loop through planet types to find a match
    for ptype, criteria in planet_types.items():
        mass_match   = (mass   >= criteria["mass_min"])   and (mass   <= criteria["mass_max"])
        radius_match = (radius >= criteria["radius_min"]) and (radius <= criteria["radius_max"])
        teq_match    = (teq    >= criteria["teq_min"])    and (teq    <= criteria["teq_max"])
        if mass_match and radius_match and teq_match:
            return ptype  # Return the first matching type
    
    return "Unidentified - outside class"



def find_matching_planets(criteria, planet_table, mode, selected_planets=None, Nmax=None):
    """
    Filter a pandas DataFrame of planets against numeric box constraints.

    Parameters
    ----------
    criteria : dict
        Keys: 'mass_min', 'mass_max', 'radius_min', 'radius_max', 'teq_min', 'teq_max'
        Units are expected to match the DataFrame columns (already scalar floats).
    planet_table : astropy.table.QTable
    mode : {"unique", "multi"}
        - "unique": Return a *single* highest-SNR row not already in 'selected_planets'.
        - "multi" : Return all rows (optionally truncated by 'Nmax').
    selected_planets : set[str], optional
        Names already chosen (used for unique mode).
    Nmax : int or None
        Limit on count when mode="multi".

    Returns
    -------
    list[dict] or list[pandas.Series]
        - mode="unique" : [row] or [] if none
        - mode="multi"  : list of dict rows (<= Nmax) or []
    """
    # Build the query string dynamically
    query = " & ".join([
        f"PlanetMass >= {criteria['mass_min']}",
        f"PlanetMass <= {criteria['mass_max']}",
        f"PlanetRadius >= {criteria['radius_min']}",
        f"PlanetRadius <= {criteria['radius_max']}",
        f"PlanetTeq >= {criteria['teq_min']}",
        f"PlanetTeq <= {criteria['teq_max']}"
    ])
    
    # Remove empty conditions from query
    query            = " & ".join(filter(None, query.split(" & ")))
    filtered_planets = planet_table.query(query) if query else planet_table
    
    # Mode 'unique': Pick only ONE planet with the highest SNR
    if mode == 'unique':
        filtered_planets = filtered_planets[~filtered_planets["PlanetName"].isin(selected_planets)]
        if not filtered_planets.empty:
            chosen_planet = filtered_planets.loc[filtered_planets["SNR"].idxmax()]
            selected_planets.add(chosen_planet["PlanetName"])
            return [chosen_planet]
        return []
    
    # Mode 'multi': Return all filtered planets, limited to Nmax if specified
    elif mode == 'multi':
        # Limit the number of results if Nmax is set
        if Nmax is not None:
            filtered_planets = filtered_planets.iloc[:Nmax]
        return filtered_planets.head(Nmax).to_dict(orient='records') if not filtered_planets.empty else []



def build_match_dict(table, planet_types, mode="multi", Nmax=None):
    """
    Build {ptype: list_of_matches} by applying 'find_matching_planets' for each rule.

    Parameters
    ----------
    table : astropy.table.Table or QTable
        Converted internally to pandas for filtering.
    planet_types : dict
        Your rule dictionary for typing (same shape used by get_planet_type()).
    mode : {"unique","multi"}
        See 'find_matching_planets'.
    Nmax : int or None
        Forwarded to 'find_matching_planets' when mode="multi".
    """
    df = table.to_pandas()
    return {ptype: find_matching_planets(criteria=criteria, planet_table=df, mode=mode, Nmax=Nmax) for ptype, criteria in planet_types.items()}



def plot_matching_planets(matching_planets, exposure_time, mode, planet_types=planet_types, instru=None):
    """
    Render a simple table showing either:
      - (unique) the best planet per type and its SNR, or
      - (multi) the numeric cuts + counts and detections (> SNR_threshold).

    Parameters
    ----------
    matching_planets : dict
        Output of 'build_match_dict' — {ptype: list_of_rows_or_dicts}
    exposure_time : float
        Total exposure time in seconds (used for the title text only here).
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
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    if instru is not None:
        fig.suptitle(f"{instru}", fontsize=16, y=0.88)
    ax.set_frame_on(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Generate the table based on the mode
    if mode == 'unique':
        matching_planets_df = pd.DataFrame([
            {"Type":            ptype,
             "Name":            planet["PlanetName"], 
             "Mass [M⊕]":      round(planet["PlanetMass"], 2), 
             "Radius [R⊕]":    round(planet["PlanetRadius"], 2), 
             "Temperature [K]": int(round(planet["PlanetTeq"])), 
             f"SNR (in {int(round(exposure_time/60))} h)": round(planet["SNR"], 1)}
            for ptype, planets in matching_planets.items() for planet in planets])
        table = ax.table(cellText=matching_planets_df.values, colLabels=matching_planets_df.columns, cellLoc='center', loc='center')
    
    elif mode == 'multi':    
        conditions_df = pd.DataFrame([
            {"Type":            ptype,
             "Mass [M⊕]":      _format_range(criteria, "mass"),
             "Radius [R⊕]":    _format_range(criteria, "radius"),
             "Temperature [K]": _format_range(criteria, "teq"),
             "Number of Planets\nconsidered": len(matching_planets[ptype]),
             f"Number of Planets\ndetected (in {int(round(exposure_time/60))} h)": sum(planet["SNR"] > SNR_threshold for planet in matching_planets[ptype])}
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
    plt.show()



def SNR_from_table(table, exposure_time, band):
    """
    Compute SNR from a table with per-band signal/noise/DIT columns.

    Parameters
    ----------
    table : astropy.table.Table/QTable (row-wise or column-wise access)
        Must contain the columns:
          f'signal_{band}', f'sigma_fund_{band}', f'sigma_syst_{band}', f'DIT_{band}'
        Units are ignored; we operate on floats.
    exposure_time : float
        Total exposure time in the same units as DIT_{band}. If DIT is in seconds,
        pass seconds; if minutes, pass minutes.
    band : str
        Name of the internal/instrument band suffix used in your columns.

    Returns
    -------
    np.ndarray
        SNR values (shape = len(table)).
    """
    dit    = np.asarray(table[f'DIT_{band}'], dtype=float)        # mn/DIT
    ndit   = exposure_time / dit                                  # DIT
    signal = np.asarray(table[f'signal_{band}'], dtype=float)     # signal/DIT
    sfund  = np.asarray(table[f'sigma_fund_{band}'], dtype=float) # noise/DIT
    ssyst  = np.asarray(table[f'sigma_syst_{band}'], dtype=float) # noise/DIT
    S      = ndit * signal
    N      = np.sqrt(ndit*sfund**2 + ndit**2*ssyst**2)
    SNR    = S / N
    return SNR



def SNR_to_size(SNR, s0=50, ds=200, SNR_min=SNR_threshold, SNR_max=1_000):
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
    SNR = np.clip(SNR, SNR_min, SNR_max)
    if SNR_min != 0:
        t = (np.log10(SNR) - np.log10(SNR_min)) / (np.log10(SNR_max) - np.log10(SNR_min))
    else:
        t = (np.log10(SNR)) / (np.log10(SNR_max))
    t[t<0] = 0
    return s0 + ds * t



def get_spectrum_contribution_name_model(thermal_model, reflected_model):
    """
    Decide (string) labels for spectrum bookkeeping, given model choices.

    Returns
    -------
    spectrum_contributions : {"thermal", "reflected", "thermal+reflected"}
    name_model : str
        If only one model is active, it is returned (with a suffix for PICASO).
        If both are active, the two are joined with '+'.
    """
    if thermal_model == "None":
        spectrum_contributions = "reflected"
        name_model = reflected_model
        if name_model == "PICASO":
            name_model += "_reflected_only"
    elif reflected_model == "None":
        spectrum_contributions = "thermal"
        name_model = thermal_model
        if name_model == "PICASO":
            name_model += "_thermal_only"
    elif thermal_model == "None" and reflected_model == "None":
        raise KeyError("PLEASE DEFINE A MODEL FOR THE THERMAL OR THE REFLECTED COMPONENT !")
    elif thermal_model != "None" and reflected_model != "None":
        spectrum_contributions = "thermal+reflected"
        name_model = thermal_model+"+"+reflected_model
    return spectrum_contributions, name_model



#######################################################################################################################
#################################################### Magnitudes computations: #########################################
#######################################################################################################################

def process_magnitudes(args):
    """
    Worker that computes all magnitudes for a single planet.

    Parameters
    ----------
    args : tuple
        (idx, planet_row), where 'planet_row' is a single row of the QTable.
        All shared, read-only resources (wavelength grid, Vega spectrum,
        bandpass masks, model names, etc.) are provided through the global
        _MAG_CTX set by the pool initializer.

    Returns
    -------
    idx : int
        Row index in the original table.
    ptype : str
        Planet type label for this target.
    mags : dict[str, float or Quantity]
        Magnitudes keyed by the final column names to be written.
    """
    
    idx, planet     = args
    wave_instru     = _MAG_CTX["wave_instru"]
    wave_K          = _MAG_CTX["wave_K"]
    vega_flux       = _MAG_CTX["vega_flux"]
    vega_spectrum_K = _MAG_CTX["vega_spectrum_K"]
    masks           = _MAG_CTX["masks"]
    use_reflected   = _MAG_CTX["use_reflected"]
    
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model="BT-Settl", reflected_model="PICASO", instru=None, wave_instru=wave_instru, wave_K=wave_K, vega_spectrum_K=vega_spectrum_K, show=False, in_im_mag=True)
    
    ptype = get_planet_type(planet)
    
    mags = {}
    
    for instru in instru_name_list:
        mask_instru = masks[instru]
        mags[f"StarINSTRUmag({instru})"]                      = get_mag(flux_obs=star_spectrum.flux[mask_instru], flux_ref=vega_flux[instru])
        mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"] = get_mag(flux_obs=planet_spectrum.flux[mask_instru], flux_ref=vega_flux[instru])
        mags[f"PlanetINSTRUmag({instru})(thermal)"]           = get_mag(flux_obs=planet_thermal.flux[mask_instru], flux_ref=vega_flux[instru])
        if use_reflected[instru]:
            mags[f"PlanetINSTRUmag({instru})(reflected)"] = get_mag(flux_obs=planet_reflected.flux[mask_instru], flux_ref=vega_flux[instru])
   
    for band in bands:
        mask_band = masks[band]
        if band != "K":
            mags[f'Star{band}mag']                  = get_mag(flux_obs=star_spectrum.flux[mask_band], flux_ref=vega_flux[band])
        mags[f"Planet{band}mag(thermal+reflected)"] = get_mag(flux_obs=planet_spectrum.flux[mask_band], flux_ref=vega_flux[band])
        mags[f"Planet{band}mag(thermal)"]           = get_mag(flux_obs=planet_thermal.flux[mask_band], flux_ref=vega_flux[band])
        if use_reflected[band]:
            mags[f"Planet{band}mag(reflected)"] = get_mag(flux_obs=planet_reflected.flux[mask_band], flux_ref=vega_flux[band])
    
    return idx, ptype, mags



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
    # --- 1) Wavelength grids and Vega ---
    wave_instru     = np.arange(LMIN, LMAX, 1e-2)
    wave_K          = wave_instru[(wave_instru >= lmin_K) & (wave_instru <= lmax_K)]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_instru, renorm=False)
    
    # --- 2) Bandpass masks, Vega flux on bands and “use_reflected” flags ---
    vega_flux     = {}
    masks         = {}
    use_reflected = {}
    for instru in instru_name_list:
        lmin, lmax            = get_band_lims(instru)
        masks[instru]         = (wave_instru >= lmin) & (wave_instru <= lmax)
        use_reflected[instru] = (lmin < ignore_reflected_thresh_um)
        vega_flux[instru]     = np.nanmean(vega_spectrum.flux[masks[instru]])
    for band in bands:
        lmin, lmax          = get_band_lims(band)
        masks[band]         = (wave_instru >= lmin) & (wave_instru <= lmax)
        use_reflected[band] = (lmin < ignore_reflected_thresh_um)
        vega_flux[band]     = np.nanmean(vega_spectrum.flux[masks[band]])

    # --- 3) Init global context for workers ---
    global _MAG_CTX
    _MAG_CTX = dict(wave_instru=wave_instru, wave_K=wave_K, vega_flux=vega_flux, vega_spectrum_K=vega_spectrum_K, masks=masks, use_reflected=use_reflected)
    
    # Magnitude estimations
    with Pool(processes=cpu_count()-1) as pool: 
        results = list(tqdm(pool.imap(process_magnitudes, [(idx, planet_table[idx]) for idx in range(len(planet_table))]), total=len(planet_table), desc="Estimating magnitudes..."))
        for (idx, ptype, mags) in results:
            planet_table[idx]['PlanetType'] = ptype
            for instru in instru_name_list:
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
#################################################### Creating tables: #################################################
#######################################################################################################################

# # TODO:
# def create_simulated_planet_table():
#     """
#     To create an exoplanet simulated table
#     """
#     import EXOSIMS.MissionSim, EXOSIMS.SimulatedUniverse.SAG13Universe
#     filename = "sim_data/Simulated_table/FastYield_sim_EXOCAT1.json"
#     with open(filename) as ff:
#         specs = json.loads(ff.read())
#     su = EXOSIMS.SimulatedUniverse.SAG13Universe.SAG13Universe(**specs)
#     flux_ratios = 10**(su.dMag/-2.5)  # grab for now from EXOSIMS
#     angseps = su.WA.value * 1000 *u.mas # mas
#     projaus = su.d.value * u.AU # au
#     phase = np.arccos(su.r[:, 2]/su.d)# planet phase  [0, pi]
#     smas = su.a.value*u.AU # au
#     eccs = su.e # eccentricity
#     incs = su.I.value*u.deg # degrees
#     masses = su.Mp  # earth masses
#     radii = su.Rp # earth radii
#     grav = const.G * (masses)/(radii)**2
#     logg = np.log10(grav.to(u.cm/u.s**2).value) * u.dex(u.cm/u.s**2) # logg cgs
#     ras = [] # deg
#     decs = [] # deg
#     distances = [] # pc
#     for index in su.plan2star:
#         coord = su.TargetList.coords[index]
#         ras.append(coord.ra.value)
#         decs.append(coord.dec.value)
#         distances.append(coord.distance.value)
#     ras = np.array(ras)
#     decs = np.array(decs)
#     distances = np.array(distances) * u.pc
#     star_names =  np.array([su.TargetList.Name[i] for i in su.plan2star])
#     planet_names = np.copy(star_names)
#     planet_types = np.copy(planet_names)
#     for i in range(len(star_names)):
#         k = 1
#         pname = np.char.add(star_names[i], f" {k}")
#         while pname in planet_names:
#             pname = np.char.add(star_names[i], f" {k}")
#             k+=1
#         planet_names[i] = pname #np.append(planet_names, pname)
#         if masses[i] < 2.0 * u.earthMass:
#             planet_types[i] = "Terran"
#         elif 2.0 * u.earthMass < masses[i] < 0.41 * u.jupiterMass:
#             planet_types[i] = "Neptunian"
#         elif 0.41 * u.jupiterMass < masses[i] < 0.80 * u.solMass:
#             planet_types[i] = "Jovian"
#         elif 0.80 * u.solMass < masses[i]:
#             planet_types[i] = "Stellar"
#         print("giving name and type: ", planet_names[i], " & ", planet_types[i], ": ", round(100*(i+1)/len(star_names), 3), " %")
#     spts = np.array([su.TargetList.Spec[i] for i in su.plan2star])
#     su.TargetList.stellar_mass() # generate masses if haven't
#     host_mass = np.array([su.TargetList.MsTrue[i].value for i in su.plan2star]) * u.solMass
#     su.TargetList.stellar_Teff()
#     host_teff = np.array([su.TargetList.Teff[i].value for i in su.plan2star]) * u.K
#     host_Vmags = np.array([su.TargetList.Vmag[i] for i in su.plan2star])
#     host_Kmags = np.array([su.TargetList.Kmag[i] for i in su.plan2star]) * u.mag
#     # guess the radius and gravity from Vmag and Teff. This is of questionable reliability
#     host_MVs   = host_Vmags - 5 * np.log10(distances.value/10) # absolute V mag
#     host_lums  = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
#     host_radii = (5800/host_teff.value)**2 * np.sqrt(host_lums) * u.solRad# Rsun
#     host_gravs = const.G * host_mass/(host_radii**2)
#     host_logg  = np.log10(host_gravs.to(u.cm/u.s**2).value) * u.dex(u.cm/(u.s**2))# logg cgs
#     teq      = su.PlanetPhysicalModel.calc_Teff(host_lums, smas, su.p)
#     all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, planet_names, planet_types, masses, radii, teq, logg, spts, host_mass, host_teff, host_radii, host_logg, host_Kmags]
#     labels   = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetName", "PlanetType", "PlanetMass", "PlanetRadius", "PlanetTeq", "PlanetLogg", "StarSpT", "StarMass", "StarTeff", "StarRadius", "StarLogg", "StarKmag"]
#     planet_table = QTable(all_data, names=labels)
#     slg_mask = np.logical_not(np.isnan(planet_table['StarLogg']))
#     stq_mask = np.logical_not(np.isnan(planet_table['StarTeff']))
#     plg_mask = np.logical_not(np.isnan(planet_table['PlanetLogg'])) 
#     ptq_mask = np.logical_not(np.isnan(planet_table['PlanetTeq']))
#     stk_mask = np.logical_not(np.isnan(planet_table['StarKmag']))
#     prd_mask = np.logical_not(np.isnan(planet_table['PlanetRadius']))
#     dis_mask = np.logical_not(np.isnan(planet_table['Distance']))
#     sma_mask = np.logical_not(np.isnan(planet_table['SMA']))
#     planet_table = planet_table[slg_mask & stq_mask & plg_mask & ptq_mask & stk_mask & prd_mask & dis_mask & sma_mask]
#     planet_table["PlanetTeq"] = (planet_table['StarRadius']/(planet_table['SMA'])).decompose()**(1/2) * planet_table['StarTeff'].value *u.K
#     planet_table["DiscoveryMethod"] = np.full((len(planet_table), ), "None")
#     planet_table["StarRadialVelocity"] = np.full((len(planet_table), ), 0) * u.km / u.s
#     planet_table["StarVsini"] = np.full((len(planet_table), ), 0) * u.km / u.s
#     planet_table.write(simulated_path+"Simulated_Pull_raw.ecsv", format='ascii.ecsv', overwrite=True)



def get_archive_table():
    """
    Build the working planet catalog used by FastYield from the NASA Exoplanet Archive.

    This routine downloads a curated subset of the 'pscomppars' table via the
    TAP service, normalizes units, renames columns to the local “house” schema,
    enriches/repairs the data (DACE values, literature fixes, simple physical
    inferences), computes additional derived quantities (e.g., log g, geometry,
    velocities), estimates missing temperatures/radii (including a fallback
    evolutionary-model pathway), filters rows to those usable for S/N
    calculations, computes synthetic magnitudes, and writes both an
    intermediate and a final ECSV file. It returns the final 'QTable'.

    Data source
    ----------
    NASA Exoplanet Archive TAP endpoint:
      https://exoplanetarchive.ipac.caltech.edu/TAP
    Table queried: 'pscomppars' (columns subset defined in-code).

    Side effects
    ------------
    Writes two ECSV files under 'archive_path' (a global/path variable):
      - 'Archive_Pull_Raw.ecsv'          : raw pull after unit normalization & renaming
      - 'Archive_Pull_For_FastYield.ecsv': final table used by FastYield

    Returns
    -------
    planet_table : astropy.table.QTable
        The filtered and augmented catalog. Columns are Quantities where
        applicable; strings remain as plain Columns.

    Main steps (high level)
    -----------------------
    1) Query 'pscomppars' over TAP and load as an Astropy table.
    2) Normalize reported units to 'astropy.units' using a unit map.
    3) Rename archive columns to local names (e.g., 'pl_bmasse' → 'PlanetMass').
    4) Ensure float-like columns are **non-masked** 'Quantity' objects with NaNs
       marking missing values (integers/strings left as 'Column').
    5) Save a raw snapshot ('Archive_Pull_Raw.ecsv') then re-load via the local
       loader to keep formatting consistent.
    6) Inject DACE and other known literature values; create convenience columns
       (e.g., 'PlanetType'), and mag placeholders for instruments/bands.
    7) Build discovery-method masks (Imaging / RV / Transit / Other).
    8) Compute log g for planets and stars; if stellar luminosity/radius/log g
       are missing, infer them from V-band magnitude, distance, and Teff using
       standard relations (no extinction correction).
    9) Adopt simple geometry/kinematics assumptions: maximum elongation
       ('phase = π/2'), fill missing inclinations with 90°, compute angular
       separation, Lambert phase function 'g(alpha)', and radial-velocity terms.
       Draw missing stellar RV and projected rotation 'v sin i' from
       temperature-type-dependent priors; same for planetary 'v sin i' using
       mass-dependent priors. Negative draws are clipped to zero.
    10) Estimate missing planet temperatures:
        - Equilibrium temperature from insolation (Bond albedo A_B=0.3, full
          4π redistribution).
        - Internal temperature from an evolutionary grid
          ('get_evolutionary_model').
        - Combine in quadrature of fluxes: 'T_eff = (T_eq^4 + T_int^4)^{1/4}'.
        - For directly imaged planets, also fill missing radii from the
          evolutionary model; if any still missing, fill with the sample median
          (unit-safe).
    11) Filter to rows suitable for S/N computations (require StarTeff,
        PlanetTeq, StarKmag, PlanetRadius, Distance, and SMA).
    12) Compute synthetic magnitudes for all configured bands/instruments.
    13) Save the final table ('Archive_Pull_For_FastYield.ecsv') and return it.

    Important columns (examples)
    ----------------------------
    Inputs/renamed: PlanetName, StarName, Period, SMA, Ecc, Inc, PlanetMass,
    PlanetRadius, PlanetTeq, RA, Dec, Distance, StarSpT, StarMass, StarTeff,
    StarRadius, StarLogg, StarLum, StarAge, StarVsini, StarRadialVelocity,
    StarZ, StarKmag, StarVmag, DiscoveryMethod, DiscoveryRef.

    Derived/added: PlanetLogg, Phase, AngSep, alpha, g_alpha,
    DeltaRadialVelocity, PlanetRadialVelocity, PlanetVsini, PlanetType,
    and magnitude columns per instrument/band (thermal / reflected / total).

    Assumptions & caveats
    ---------------------
    * No extinction correction is applied when deriving stellar luminosities
      from V-band magnitudes.
    * RV and rotation priors are simple Gaussians conditioned on spectral type
      (stars) or mass regime (planets); set a NumPy seed externally for
      reproducibility if desired.
    * The evolutionary-model path requires a prebuilt grid (see
      'get_evolutionary_model'). Temperatures are combined as fluxes.
    * Directly imaged radii that remain missing after the model are filled with
      the sample median—adequate for magnitude predictions but not for precise
      structure work.
    """
    time1 = time.time()
    
    # -----------------------------------------------------------------------------
    # 1) Pull PSCOMPPARS subset and normalize units
    # -----------------------------------------------------------------------------
    COLS_TO_PULL = (
                    "pl_name, hostname, "
                    "pl_orbper, pl_orbsmax, pl_orbeccen, pl_orbincl, "
                    "pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmasse_reflink, "
                    "pl_rade, pl_radeerr1, pl_radeerr2, pl_rade_reflink, "
                    "pl_eqt, pl_eqterr1, pl_eqterr2, pl_eqt_reflink, "
                    "ra, dec, sy_dist, sy_disterr1, sy_disterr2, "
                    "st_spectype, st_mass, st_teff, st_rad, st_logg, st_lum, st_age, st_vsin, st_radv, st_met, "
                    "sy_kmag, sy_vmag, "
                    "discoverymethod, disc_refname"
                   )
    
    NEW_NAMES = [
                 "PlanetName", "StarName",
                 "Period", "SMA", "Ecc", "Inc",
                 "PlanetMass", "+DeltaPlanetMass", "-DeltaPlanetMass", "PlanetMassRef",
                 "PlanetRadius", "+DeltaPlanetRadius", "-DeltaPlanetRadius", "PlanetRadiusRef",
                 "PlanetTeq", "+DeltaPlanetTeq", "-DeltaPlanetTeq", "PlanetTeqRef",
                 "RA", "Dec", "Distance", "+DeltaDistance", "-DeltaDistance",
                 "StarSpT", "StarMass", "StarTeff", "StarRadius", "StarLogg", "StarLum", "StarAge",
                 "StarVsini", "StarRadialVelocity", "StarZ",
                 "StarKmag", "StarVmag",
                 "DiscoveryMethod", "DiscoveryRef"
                ]
    
    print("\nRetrieving NASA archive table from https://exoplanetarchive.ipac.caltech.edu/TAP ...")
    svc   = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    table = svc.search(f"SELECT {COLS_TO_PULL} FROM pscomppars").to_table()
    planet_table = QTable(table)
    planet_table.meta["isPSCOMPPARS"] = True
    
    # --- Unit normalization -------------------------------------------------------
    # Map archive-reported units (by string) to astropy units
    UNIT_MAP = {
                "Earth Mass":     u.M_earth,
                "Earth Radius":   u.R_earth,
                "Solar mass":     u.M_sun,
                "Solar Radius":   u.R_sun,
                "days":           u.day,
                "AU":             u.AU,
                "log(Solar)":     u.dex(u.solLum),
                "log10(cm/s**2)": u.dex(u.cm/(u.s**2)),
                "dex":            u.dimensionless_unscaled,
                "sy_kmag":        u.dimensionless_unscaled,
                "sy_vmag":        u.dimensionless_unscaled,
                "pl_bmasseerr1":  u.M_earth,
                "pl_bmasseerr2":  u.M_earth,
                "sy_disterr1":    u.pc,
                "sy_disterr2":    u.pc,
                "pl_orbsmax":     u.AU,
               }
    
    print("\nNormalizing units...")
    for colname in planet_table.colnames:
        current_unit = planet_table[colname].unit
        if current_unit is not None and str(current_unit) in UNIT_MAP:
            new_unit              = UNIT_MAP[str(current_unit)]
            planet_table[colname] = planet_table[colname].value * new_unit
        elif current_unit is None and str(colname) in UNIT_MAP:
            new_unit              = UNIT_MAP[str(colname)]
            planet_table[colname] = planet_table[colname].value * new_unit
    
    # Rename to our house schema
    print("\nRenaming columns...")
    planet_table.rename_columns(COLS_TO_PULL.split(", "), NEW_NAMES)
    
    # -----------------------------------------------------------------------------
    # 2) Make float-like columns plain (non-masked) Quantities; keep strings as Column
    # -----------------------------------------------------------------------------
    for colname in list(planet_table.colnames):
        col = planet_table[colname]
        # Keep integer and object/string columns as plain Columns
        if col.dtype.kind in ("i", "O"): # Integer / str
            if not isinstance(col, Column):
                raise KeyError(f"Column {colname!r} should be an astropy Column at this point.")
            continue
        # Convert float-like columns (MaskedQuantity/MaskedColumn/Quantity/Column) to non-masked Quantity
        if col.dtype.kind == "f":
            unit = getattr(col, "unit", None)
            # Extract underlying numeric values (may be a masked array)
            if hasattr(col, "to_value"):  # Quantity / MaskedQuantity / MaskedDex
                vals = col.to_value(unit) if unit is not None else np.array(col.value, copy=False)
            else:  # Column / MaskedColumn
                vals = np.asanyarray(col)
            # Drop mask -> fill masked entries with NaN
            vals = np.ma.filled(vals, np.nan)
            # Build a non-masked Quantity (multiplication also supports logarithmic units like dex)
            q = vals * (unit or u.dimensionless_unscaled)
            # Replace the current column with the Quantity
            planet_table.replace_column(colname, q)
            continue
        # Anything else is unexpected here
        raise KeyError(f"Unhandled dtype kind {col.dtype.kind!r} for column {colname!r}.")
    # Verification: no float-like column should remain masked
    for colname in planet_table.colnames:
        col = planet_table[colname]
        if col.dtype.kind == "f":
            assert not hasattr(col, "mask"), f"{colname} is still masked!"
    
    # -----------------------------------------------------------------------------
    # 3) Saving raw pull and re-load via the loader
    # -----------------------------------------------------------------------------
    print(f"\nSaving the raw archive planet_table ({len(planet_table)} planets) to {archive_path}Archive_Pull_Raw.ecsv ...\n")
    planet_table.write(archive_path + "Archive_Pull_Raw.ecsv", format='ascii.ecsv', overwrite=True)
    planet_table = load_planet_table("Archive_Pull_Raw.ecsv")
    
    # -----------------------------------------------------------------------------
    # 4) Inject DACE values (vectorized, unit-aware) and known missing values for papers
    # -----------------------------------------------------------------------------
    planet_table = inject_dace_values(planet_table)
    
    # Creating planet type column
    planet_table["PlanetType"] = np.full(len(planet_table), "Unidentified", dtype="<U32")
    
    # Creating magnitudes columns
    for instru in instru_name_list:
        planet_table[f'StarINSTRUmag({instru})']                      = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f'PlanetINSTRUmag({instru})(thermal+reflected)'] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f'PlanetINSTRUmag({instru})(thermal)']           = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f'PlanetINSTRUmag({instru})(reflected)']         = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    for band in bands:
        if band != "K":
            planet_table[f'Star{band}mag']                  = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f'Planet{band}mag(thermal+reflected)'] = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f'Planet{band}mag(thermal)']           = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
        planet_table[f'Planet{band}mag(reflected)']         = np.full(len(planet_table), np.nan) * u.dimensionless_unscaled
    
    # Also injecting known missing values
    print("\nInjecting known K-band magnitudes and temperatures for directly-imaged planets...")
    planet_table = inject_known_values(planet_table)
    print_missing_known_values(planet_table)
    # # TODO: Deleting directly-imaged planets with missing magnitudes
    # pkm_invalid  = get_invalid_mask(planet_table["PlanetKmag(thermal+reflected)"])
    # planet_table = planet_table[~(im_mask & pkm_invalid)]
    
    # -----------------------------------------------------------------------------
    # 5) Discovery-method masks
    # -----------------------------------------------------------------------------    
    im_mask = planet_table["DiscoveryMethod"] == "Imaging"
    rv_mask = planet_table["DiscoveryMethod"] == "Radial Velocity"
    tr_mask = planet_table["DiscoveryMethod"] == "Transit"
    ot_mask = (~im_mask) & (~rv_mask) & (~tr_mask) # Other: “Microlensing”, “Astrometry”, “Transit Timing Variations”...
    
    # -----------------------------------------------------------------------------
    # 6) Compute Planet log g (from M and R) where possible
    # -----------------------------------------------------------------------------
    g_planet                   = const.G * planet_table["PlanetMass"] / planet_table["PlanetRadius"]**2
    planet_table["PlanetLogg"] = np.log10(g_planet.to(u.cm / u.s**2).value) * u.dex(u.cm / u.s**2)
    
    # -----------------------------------------------------------------------------
    # 7) Compute Star log g via (StarLum -> StarRadius) if missing
    #     Using L/Lsun = (R/Rsun)^2 (T/Tsun)^4 => R/Rsun = sqrt(L/Lsun) * (Tsun/T)^2
    # -----------------------------------------------------------------------------
    # 7a) StarLum from V and Distance (no extinction correction here)
    abs_V = planet_table["StarVmag"].value - 5 * np.log10((planet_table["Distance"] / (10 * u.pc)).to_value())
    # L/Lsun ≈ 10^{-0.4 (M_V - M_V,⊙)} with M_V,⊙ ≈ 4.83
    star_lum_from_V                     = (- (abs_V - 4.83) / 2.5) * u.dex(u.solLum)
    sl_invalid                          = get_invalid_mask(planet_table["StarLum"])
    planet_table["StarLum"][sl_invalid] = star_lum_from_V[sl_invalid]
    
    # 7b) StarRadius from StarLum and StarTeff
    R_over_Rsun                            = np.sqrt(10**planet_table["StarLum"].to_value(u.dex(u.solLum))) * (5772 * u.K / planet_table["StarTeff"])**2
    sr_invalid                             = get_invalid_mask(planet_table["StarRadius"])
    planet_table["StarRadius"][sr_invalid] = R_over_Rsun[sr_invalid] * u.R_sun
    
    # 7c) StarLogg from StarMass and StarRadius
    g_star                                = const.G * planet_table["StarMass"] / planet_table["StarRadius"]**2
    slg_invalid                           = get_invalid_mask(planet_table["StarLogg"])
    planet_table["StarLogg"][slg_invalid] = np.log10(g_star.to(u.cm / u.s**2).value)[slg_invalid] * u.dex(u.cm / u.s**2)
    
    # 7d) Final fallback: median by discovery method (log g is not critical here)
    slg_invalid = get_invalid_mask(planet_table["StarLogg"])
    plg_invalid = get_invalid_mask(planet_table["PlanetLogg"])
    for det_mask in (im_mask, rv_mask, tr_mask, ot_mask):
        planet_table["StarLogg"][slg_invalid & det_mask]   = np.nanmedian(planet_table["StarLogg"][det_mask])
        planet_table["PlanetLogg"][plg_invalid & det_mask] = np.nanmedian(planet_table["PlanetLogg"][det_mask])
    
    # -----------------------------------------------------------------------------
    # 8) Assumptions for geometry and kinematics
    # -----------------------------------------------------------------------------    
    print(f"\n(1) Hypothesis (mostly optimistic): Planets are assumed to be at their maximum elongation and missing inclinations are set to 90°")
    phase                  = np.pi / 2
    planet_table["Phase"]  = u.Quantity(np.full(len(planet_table), phase, dtype=float), u.rad, copy=False)
    print(f"                  => Phase  = pi/2 (for all planets)")
    planet_table['AngSep'] = (planet_table['SMA']/planet_table['Distance']).value * 1e3 * u.mas
    print(f"                  => AngSep = SMA / Distance (for all planets)")
    inc_invalid                      = get_invalid_mask(planet_table["Inc"])
    planet_table["Inc"][inc_invalid] = 90 * planet_table["Inc"].unit
    i_rad                            = np.array(planet_table["Inc"].value) * np.pi/180 # inc in rad
    print(f"                  => Inc    = 90° (for {len(planet_table[inc_invalid])}/{len(planet_table)} planets)")
    planet_table['DeltaRadialVelocity'] = np.sqrt( const.G * ( planet_table["StarMass"] + planet_table["PlanetMass"] ) / planet_table["SMA"] ).decompose().to(u.km/u.s) * np.sin(i_rad)
    print(f"                  => DelaRadialVelocity = sin(i)·sqrt(G·(StarMass+PlanetMass)/SMA) (for all planets)")
    
    # Lambert phase function: g(alpha)
    alpha                   = np.arccos(-np.sin(i_rad) * np.cos(planet_table["Phase"].to_value(u.rad)))
    planet_table["alpha"]   = alpha * u.rad
    planet_table["g_alpha"] = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi * u.dimensionless_unscaled
    
    # Randomly draw radial velocities of the stars and Drv according to a normal distribution when they are missing
    srv_invalid                                     = get_invalid_mask(planet_table["StarRadialVelocity"])
    srv_mean                                        = np.nanmean(planet_table["StarRadialVelocity"]).value
    srv_std                                         = np.nanstd(planet_table["StarRadialVelocity"]).value
    planet_table["StarRadialVelocity"][srv_invalid] = np.random.normal(srv_mean, srv_std, len(planet_table[srv_invalid])) * planet_table["StarRadialVelocity"].unit
    drv_invalid                                      = get_invalid_mask(planet_table["DeltaRadialVelocity"])
    drv_mean                                         = np.nanmean(planet_table["DeltaRadialVelocity"]).value
    drv_std                                          = np.nanstd(planet_table["DeltaRadialVelocity"]).value
    planet_table["DeltaRadialVelocity"][drv_invalid] = np.random.normal(drv_mean, drv_std, len(planet_table[drv_invalid])) * planet_table["DeltaRadialVelocity"].unit
    
    # Computing planet radial velocities
    planet_table["PlanetRadialVelocity"] = planet_table["StarRadialVelocity"] + planet_table["DeltaRadialVelocity"]
    
    # -----------------------------------------------------------------------------
    # 9) v sin i priors for stars and planets (fills only where missing)
    # -----------------------------------------------------------------------------
    # Randomly draw rotationnal velocities of the stars (depending on the type) to a normal distribution when they are missing
    svs_invalid = get_invalid_mask(planet_table["StarVsini"])
    planet_table["StarVsini"][svs_invalid &                                          (planet_table["StarTeff"] <= 3500*u.K) ] = np.random.normal(1, 0.5, size=len(planet_table[svs_invalid &                                          (planet_table["StarTeff"] <= 3500*u.K) ])) * u.km / u.s # Cool stars (Teff <= 3500 K):                mu = 1 km/s,  sigma = 0.5 km/s (M-dwarfs, Newton et al. 2017)
    planet_table["StarVsini"][svs_invalid & (3500*u.K  < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000*u.K) ] = np.random.normal(3, 1,   size=len(planet_table[svs_invalid & (3500*u.K  < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000*u.K) ])) * u.km / u.s # Solar-type stars (3500 K < Teff <= 6000 K): mu = 3 km/s,  sigma = 1 km/s   (G and K-type stars, Nielsen et al. 2013)
    planet_table["StarVsini"][svs_invalid & (6000*u.K  < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000*u.K)] = np.random.normal(10, 4,  size=len(planet_table[svs_invalid & (6000*u.K  < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000*u.K)])) * u.km / u.s # Hot stars (6000 K < Teff <= 10000 K):       mu = 10 km/s, sigma = 4 km/s   (F-type stars, Royer et al. 2007)
    planet_table["StarVsini"][svs_invalid & (10000*u.K < planet_table["StarTeff"])                                          ] = np.random.normal(50, 15, size=len(planet_table[svs_invalid & (10000*u.K < planet_table["StarTeff"])                                          ])) * u.km / u.s # Very hot stars (Teff > 10000 K):            mu = 50 km/s, sigma = 15 km/s  (O, B, and A-type stars, Zorec & Royer 2012)
    planet_table["StarVsini"][planet_table["StarVsini"] < 0] = 0.
    
    # Randomly draw rotationnal velocities of all the planets (depending on the type) to a normal distribution
    planet_table["PlanetVsini"] = 0 * planet_table["StarVsini"]
    pm_valid                    = get_valid_mask(planet_table["PlanetMass"])
    planet_table["PlanetVsini"][                                              (planet_table["PlanetMass"] <= 5*u.Mearth)  ] = np.random.normal(2,  1,  size=len(planet_table[                                              (planet_table["PlanetMass"] <= 5*u.Mearth)  ])) * u.km / u.s # Earths-like:    mu = 2 km/s,  sigma = 1 km/s  (McQuillan, A., Mazeh, T., & Aigrain, S. (2013). MNRAS, 432, 1203.)
    planet_table["PlanetVsini"][(5*u.Mearth   < planet_table["PlanetMass"]) & (planet_table["PlanetMass"] <= 20*u.Mearth) ] = np.random.normal(5,  2,  size=len(planet_table[(5*u.Mearth   < planet_table["PlanetMass"]) & (planet_table["PlanetMass"] <= 20*u.Mearth) ])) * u.km / u.s # Super-Earths:   mu = 5 km/s,  sigma = 2 km/s  (Dai, F., et al. (2016). ApJ, 823, 115.)
    planet_table["PlanetVsini"][(20*u.Mearth  < planet_table["PlanetMass"]) & (planet_table["PlanetMass"] <= 100*u.Mearth)] = np.random.normal(12, 5,  size=len(planet_table[(20*u.Mearth  < planet_table["PlanetMass"]) & (planet_table["PlanetMass"] <= 100*u.Mearth)])) * u.km / u.s # Neptunes-like:  mu = 12 km/s, sigma = 5 km/s  (Snellen, I.A.G., et al. (2014). Nature, 509, 63-65.)
    planet_table["PlanetVsini"][(100*u.Mearth < planet_table["PlanetMass"]) & (planet_table["PlanetMass"] <= 300*u.Mearth)] = np.random.normal(25, 10, size=len(planet_table[(100*u.Mearth < planet_table["PlanetMass"]) & (planet_table["PlanetMass"] <= 300*u.Mearth)])) * u.km / u.s # Jupiters-like:  mu = 25 km/s, sigma = 10 km/s (Bryan, M.L., et al. (2018). AJ, 156, 142.)
    planet_table["PlanetVsini"][(300*u.Mearth < planet_table["PlanetMass"])                                               ] = np.random.normal(40, 15, size=len(planet_table[(300*u.Mearth < planet_table["PlanetMass"])                                               ])) * u.km / u.s # Super-Jupiters: mu = 40 km/s, sigma = 15 km/s (Snellen, I.A.G., et al. (2010). Nature, 465, 1049-1051.)
    planet_table["PlanetVsini"][planet_table["PlanetVsini"] < 0] = 0.
    
    # -----------------------------------------------------------------------------
    # 10) Estimating missing planet temperatures (skip directly-imaged planets to avoid biasing hot young giants)
    # -----------------------------------------------------------------------------
    # Equilibrium temperature (from the received flux of the star)
    bond_albedo = 0.3  # Default value (mean value for exoplanets)
    pte         = planet_table['StarTeff'] * np.sqrt((planet_table['StarRadius']) / (2*planet_table['SMA'])).decompose() * (1 - bond_albedo)**0.25
    
    # Temperature from internal and initial energy (from evolutionary model)
    pte_int, pr, _ = get_evolutionary_model(planet_table)
    
    print(f"\n(2) Hypothesis (possibly pessimistic for young planets): Non-imaged planets are assumed to be at their equilibrium temperature with insolation when missing (A_B={bond_albedo}, 4π redistribution) + Internal energy when evolutionary model is possible")
    pte_invalid                               = get_invalid_mask(planet_table["PlanetTeq"])
    pte_missing                               = pte_invalid & (~im_mask)
    planet_table["PlanetTeq"][pte_missing]    = ( np.nan_to_num(pte[pte_missing])**4 + np.nan_to_num(pte_int[pte_missing])**4 )**(1/4)
    planet_table["PlanetTeq"][planet_table["PlanetTeq"] == 0*u.K] = np.nan 
    planet_table["PlanetTeqRef"][pte_missing] = "Equilibrium Teq (A_B=0.3, 4π redistribution) + Internal energy (when possible)"
    print(f"                  => PlanetTeq = (1 - A_B)·StarTeff·sqrt(StarRadius/2·SMA) (for {len(planet_table[pte_missing])}/{len(planet_table)} planets)")
    
    # -----------------------------------------------------------------------------
    # 11) Filling missing radius for directly-imaged planets from evolutionary model if possible
    # -----------------------------------------------------------------------------
    pr_invalid                                            = get_invalid_mask(planet_table["PlanetRadius"])
    planet_table["PlanetRadius"][pr_invalid & im_mask]    = pr[pr_invalid & im_mask]
    planet_table["PlanetRadiusRef"][pr_invalid & im_mask] = "Evolutionary model"
    
    # For directly-imaged planets, radius is not critical/usefull since the K-band magnitudes is known → fill invalid with sample median
    pr_invalid                                            = get_invalid_mask(planet_table["PlanetRadius"])
    planet_table["PlanetRadius"][pr_invalid & im_mask]    = np.nanmedian(planet_table['PlanetRadius'][im_mask])
    planet_table["PlanetRadiusRef"][pr_invalid & im_mask] = "Filled with median value"
    
    # -----------------------------------------------------------------------------
    # 12) Filter down to rows usable for S/N calculations
    # -----------------------------------------------------------------------------
    
    # Clipping outlier temperatures
    planet_table["PlanetTeq"][planet_table["PlanetTeq"] > 3000*u.K] = 3000*u.K
    
    # Clipping outlier radius
    planet_table["PlanetRadius"][planet_table["PlanetRadius"] > 40*u.R_earth] = 40*u.R_earth

    ste_valid = get_valid_mask(planet_table["StarTeff"])
    pte_valid = get_valid_mask(planet_table["PlanetTeq"])
    skm_valid = get_valid_mask(planet_table["StarKmag"])
    pr_valid  = get_valid_mask(planet_table["PlanetRadius"])
    d_valid   = get_valid_mask(planet_table["Distance"])
    sma_valid = get_valid_mask(planet_table["SMA"])
    snr_valid = ste_valid & pte_valid & skm_valid & pr_valid & d_valid & sma_valid
    print(f"\n(3) Filtering planets for S/N computations: keeping {len(planet_table[snr_valid])}/{len(planet_table)} planets")
    print(f"                  => Missing StarTeff:     IM: {round(100*len(planet_table[~ste_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~ste_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~ste_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~ste_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing PlanetTeq:    IM: {round(100*len(planet_table[~pte_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~pte_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~pte_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~pte_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing StarKmag:     IM: {round(100*len(planet_table[~skm_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~skm_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~skm_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~skm_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing PlanetRadius: IM: {round(100*len(planet_table[~pr_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~pr_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~pr_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~pr_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing Distance:     IM: {round(100*len(planet_table[~d_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~d_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~d_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~d_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing SMA:          IM: {round(100*len(planet_table[~sma_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~sma_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~sma_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~sma_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"\nKeeping: TOTAL: {len(planet_table[snr_valid])}/{len(planet_table)} | IM: {len(planet_table[snr_valid & im_mask])}/{len(planet_table[im_mask])} | RV: {len(planet_table[snr_valid & rv_mask])}/{len(planet_table[rv_mask])} | TR: {len(planet_table[snr_valid & tr_mask])}/{len(planet_table[tr_mask])} | OT: {len(planet_table[snr_valid & ot_mask])}/{len(planet_table[ot_mask])}")
    planet_table = planet_table[snr_valid]
    
    # -----------------------------------------------------------------------------
    # 13) Computing magnitudes for all bands
    # -----------------------------------------------------------------------------
    planet_table = get_planet_table_magnitudes(planet_table)
    
    # -----------------------------------------------------------------------------
    # 14) Saving final table
    # -----------------------------------------------------------------------------
    print(f"\nTotal number of planets for FastYield calculations: {len(planet_table)}")
    print(f"\nGenerating the table took {round((time.time()-time1)/60, 1)} mn")
    planet_table.write(archive_path + "Archive_Pull_For_FastYield.ecsv", format='ascii.ecsv', overwrite=True)
    
    # Few sanity check plots
    planet_table_classification()
    planet_table_statistics()

#######################################################################################################################
########################################### SNR computations: #########################################################
#######################################################################################################################

def process_SNR(args):
    """
    Worker: compute SNR and band magnitudes for a single planet.

    Parameters
    ----------
    args : tuple
        (idx, planet_row)

    Returns
    -------
    result : tuple
         - idx : int
         - mags_p : dict {'Star{instru}mag' : float, 'Planet{instru}mag' : float, 'Star{band}mag' : float, ... }
         - name_band : list[str]
         - signal    : np.ndarray [e-/DIT]  (per name_band entry)
         - sigma_f   : np.ndarray [e-/DIT]
         - sigma_s   : np.ndarray [e-/DIT]
         - DIT       : np.ndarray [s]
    """
    # idx, planet row
    idx, planet = args
    
    # Context
    instru          = _SNR_CTX["instru"]
    thermal_model   = _SNR_CTX["thermal_model"]
    reflected_model = _SNR_CTX["reflected_model"]
    wave_instru     = _SNR_CTX["wave_instru"]
    wave_K          = _SNR_CTX["wave_K"]
    vega_flux       = _SNR_CTX["vega_flux"]
    vega_spectrum_K = _SNR_CTX["vega_spectrum_K"]
    band0           = _SNR_CTX["band0"]
    exposure_time   = _SNR_CTX["exposure_time"]
    apodizer        = _SNR_CTX["apodizer"]
    strehl          = _SNR_CTX["strehl"]
    coronagraph     = _SNR_CTX["coronagraph"]
    Rc              = _SNR_CTX["Rc"]
    filter_type     = _SNR_CTX["filter_type"]
    systematic      = _SNR_CTX["systematic"]
    PCA             = _SNR_CTX["PCA"]
    N_PCA           = _SNR_CTX["N_PCA"]
    masks           = _SNR_CTX["masks"]
    bands_valid     = _SNR_CTX["bands_valid"]
    
    # Computing models on wave_instru    
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=instru, wave_instru=wave_instru, wave_K=wave_K, vega_spectrum_K=vega_spectrum_K, show=False)
    
    # Recalculates the magnitude in case the thermal model is no longer BT-Settl or the reflected model is no longer PICASO (the mag changes with regards to the raw archive table with the estimated magnitudes)
    mask_instru                = masks[instru]
    mag_s                      = get_mag(flux_obs=star_spectrum.flux[mask_instru],   flux_ref=vega_flux[instru])
    mag_p                      = get_mag(flux_obs=planet_spectrum.flux[mask_instru], flux_ref=vega_flux[instru])
    mags                       = {}
    mags[f"Star{instru}mag"]   = mag_s
    mags[f"Planet{instru}mag"] = mag_p
    for band in bands_valid:
        mask_band                = masks[band]
        mags[f"Star{band}mag"]   = get_mag(flux_obs=star_spectrum.flux[mask_band],   flux_ref=vega_flux[band])
        mags[f"Planet{band}mag"] = get_mag(flux_obs=planet_spectrum.flux[mask_band], flux_ref=vega_flux[band])
        
    # Computing the SNR for the planet
    name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = FastCurves(instru=instru, calculation="SNR", mag_star=mag_s, band0=band0, exposure_time=exposure_time, mag_planet=mag_p, separation_planet=planet["AngSep"].value/1000, return_SNR_planet=True, show_plot=False, verbose=False, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematic=systematic, PCA=PCA, N_PCA=N_PCA)
    
    return idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band



def get_planet_table_SNR(instru, table="Archive", thermal_model="None", reflected_model="None", apodizer="NO_SP", strehl="NO_JQ", coronagraph=None, Rc=100, filter_type="gaussian", systematic=False, PCA=False, N_PCA=20):
    """"
    Compute per-planet SNRs for a given instrument and write results to the table.

    Steps
    -----
    1) Load the working table (“Archive” or “Simulated”).
    2) Filter out targets inside the IWA for the instrument.
    3) If requested, restrict to Exo-REM Teq validity.
    4) Precompute wavelength grids, Vega on those grids, and masks.
    5) Run per-planet SNR using multiprocessing (unless PCA).
    6) Save internal-band metrics returned by FastCurves.

    Parameters
    ----------
    instru : str
        Instrument name (must be known by your config & globals).
    table : {"Archive", "Simulated"}, optional
        Which table file to load.
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
    systematic : bool, optional
        Whether to include systematic noise in FastCurves.
    PCA : bool, optional
        Whether to include PCA post-processing in FastCurves.
    N_PCA : int, optional
        Number of PCA components if PCA=True.
    """
    config_data   = get_config_data(instru)
    exposure_time = 120 # [mn]
    time1         = time.time()
    if systematic and filter_type == "gaussian_fast":
        filter_type = "gaussian" # "gaussian_fast" is bad for handling systematic estimations
    
    # --- 1) Loading table ---
    if table == "Archive":
        planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
        path = archive_path
    elif table == "Simulated":
        planet_table = load_planet_table("Simulated_Pull_For_FastYield.ecsv")
        path = simulated_path
    else:
        raise ValueError("table must be 'Archive' or 'Simulated'")

    # --- 2) IWA/OWA filtering (removing planets inside iwa) ---
    iwa, owa     = get_wa(config_data=config_data, sep_unit="mas")
    planet_table = planet_table[planet_table["AngSep"] > iwa * u.mas]
    
    # --- 3) Optional Exo-REM Teq range filtering ---
    if thermal_model == "Exo-REM":
        if globals()[f"lmin_{instru}"] >= 1 and globals()[f"lmax_{instru}"] <= 5.3:
            planet_table = planet_table[(planet_table['PlanetTeq'] < 2000*u.K)]
        else:
            planet_table = planet_table[(planet_table['PlanetTeq'] > 400*u.K) & (planet_table['PlanetTeq'] < 2000*u.K)]
            
    # --- 4) Wavelength grids and Vega ---
    # K-band for photometry
    R_K    = R0_min # Only for photometric purposes (does not need more resolution)
    dl_K   = (lmin_K+lmax_K)/2 / (2*R_K)
    wave_K = np.arange(lmin_K, lmax_K, dl_K)
    
    # Instrument band (high R for spectrophotometry)
    lmin_instru = config_data["lambda_range"]["lambda_min"] # in µm
    lmax_instru = config_data["lambda_range"]["lambda_max"] # in µm
    R_instru    = R0_max # Abritrary resolution (needs to be high enough)
    dl_instru   = (lmin_instru+lmax_instru)/2 / (2*R_instru)
    wave_instru = np.arange(0.98*lmin_instru, 1.02*lmax_instru, dl_instru)
    
    # Vega spectrum on K-band and instru-band [J/s/m2/µm]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_instru, renorm=False)
    
    # --- 5) Create columns for signal, noise and DIT length ---
    planet_table["signal_INSTRU"]     = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["sigma_fund_INSTRU"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["sigma_syst_INSTRU"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["DIT_INSTRU"]        = np.full(len(planet_table), np.nan) # [mn]
    for band in bands:
        planet_table[f"signal_{band}"]     = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"sigma_fund_{band}"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"sigma_syst_{band}"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"DIT_{band}"]        = np.full(len(planet_table), np.nan) # [nm]
    
    # --- 6) Bandpass masks and Vega flux on bands ---
    vega_flux         = {}
    masks             = {}
    bands_valid       = []
    masks[instru]     = (wave_instru >= lmin_instru) & (wave_instru <= lmax_instru)
    vega_flux[instru] = np.nanmean(vega_spectrum.flux[masks[instru]])
    for band in bands:
        lmin_band, lmax_band = get_band_lims(band)
        if lmin_instru <= lmin_band and lmax_instru >= lmax_band:
            bands_valid.append(band)
            masks[band]     = (wave_instru >= lmin_band) & (wave_instru <= lmax_band)
            vega_flux[band] = np.nanmean(vega_spectrum.flux[masks[band]])
            
    # Band where magnitudes are defined for the FastCurves computations
    band0 = "instru"
    
    # Contribution and model labels
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # Suffix
    suffix = "with systematics+PCA" if (systematic and PCA) else ("with systematics" if systematic else "without systematics")
    
    # Print
    print(f"\n {instru} ({apodizer} & {strehl} & {coronagraph}) {suffix} ({thermal_model} & {reflected_model})")

    # --- 7) Init global context for workers ---
    global _SNR_CTX
    _SNR_CTX = dict(instru=instru, thermal_model=thermal_model, reflected_model=reflected_model, wave_instru=wave_instru, wave_K=wave_K, vega_flux=vega_flux, vega_spectrum_K=vega_spectrum_K, band0=band0, exposure_time=exposure_time, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematic=systematic, PCA=PCA, N_PCA=N_PCA, masks=masks, bands_valid=bands_valid)    

    # --- 8) Run SNR for each planet (parallel if not PCA or serial if PCA) ---
    if PCA: # if PCA, no multiprocessing (otherwise it crashes: TODO ?)
        for idx in tqdm(range(len(planet_table)), desc="Serial"):
            args = (idx, planet_table[idx])
            idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = process_SNR(args)
            planet_table[idx][f"StarINSTRUmag({instru})"]                             = mags[f"Star{instru}mag"]
            planet_table[idx][f"PlanetINSTRUmag({instru})({spectrum_contributions})"] = mags[f"Planet{instru}mag"]
            for band in bands_valid:
                planet_table[idx][f"Star{band}mag"]                             = mags[f"Star{band}mag"]
                planet_table[idx][f"Planet{band}mag({spectrum_contributions})"] = mags[f"Planet{band}mag"]
            SNR     = np.sqrt(exposure_time/DIT_band) * signal_planet / np.sqrt( sigma_fund_planet**2 + (exposure_time/DIT_band)*sigma_syst_planet**2 )
            idx_max = SNR.argmax() # Best band idx
            planet_table[idx]["signal_INSTRU"]     = signal_planet[idx_max]
            planet_table[idx]["sigma_fund_INSTRU"] = sigma_fund_planet[idx_max]
            planet_table[idx]["sigma_syst_INSTRU"] = sigma_syst_planet[idx_max]
            planet_table[idx]["DIT_INSTRU"]        = DIT_band[idx_max]
            for nb, band in enumerate(name_band):
                planet_table[idx][f"signal_{band}"]     = signal_planet[nb]
                planet_table[idx][f"sigma_fund_{band}"] = sigma_fund_planet[nb]
                planet_table[idx][f"sigma_syst_{band}"] = sigma_syst_planet[nb]
                planet_table[idx][f"DIT_{band}"]        = DIT_band[nb]
    
    else: # if no PCA, uses multiprocessing
        with Pool(processes=cpu_count()-1) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            results = list(tqdm(pool.imap(process_SNR, [(idx, planet_table[idx]) for idx in range(len(planet_table))]), total=len(planet_table), desc="Multiprocessing"))
            for (idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band) in results:
                planet_table[idx][f"StarINSTRUmag({instru})"]                             = mags[f"Star{instru}mag"]
                planet_table[idx][f"PlanetINSTRUmag({instru})({spectrum_contributions})"] = mags[f"Planet{instru}mag"]
                for band in bands_valid:
                    planet_table[idx][f"Star{band}mag"]                             = mags[f"Star{band}mag"]
                    planet_table[idx][f"Planet{band}mag({spectrum_contributions})"] = mags[f"Planet{band}mag"]
                SNR     = np.sqrt(exposure_time/DIT_band) * signal_planet / np.sqrt( sigma_fund_planet**2 + (exposure_time/DIT_band)*sigma_syst_planet**2 )
                idx_max = SNR.argmax() # Best band idx
                planet_table[idx]["signal_INSTRU"]     = signal_planet[idx_max]
                planet_table[idx]["sigma_fund_INSTRU"] = sigma_fund_planet[idx_max]
                planet_table[idx]["sigma_syst_INSTRU"] = sigma_syst_planet[idx_max]
                planet_table[idx]["DIT_INSTRU"]        = DIT_band[idx_max]
                for nb, band in enumerate(name_band):
                    planet_table[idx][f"signal_{band}"]     = signal_planet[nb]
                    planet_table[idx][f"sigma_fund_{band}"] = sigma_fund_planet[nb]
                    planet_table[idx][f"sigma_syst_{band}"] = sigma_syst_planet[nb]
                    planet_table[idx][f"DIT_{band}"]        = DIT_band[nb]
        
    print(f"\n Calculating SNR took {(time.time()-time1)/60:.1f} mn")
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename = f"{path}{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"
    planet_table.write(filename, format='ascii.ecsv', overwrite=True)
    


def all_SNR_table(table="Archive", instru_name_list=instru_name_list): # takes ~ 13 hours
    """
    To compute SNR for every instruments with different model spectra
    """
    time0 = time.time()
    for instru in instru_name_list:
        config_data = get_config_data(instru)
        if config_data["lambda_range"]["lambda_max"] > 6:
            thermal_models   = ["None", "BT-Settl", "Exo-REM"]
            reflected_models = ["None"]
        else:
            thermal_models   = ["None", "BT-Settl", "Exo-REM", "PICASO"]
            reflected_models = ["None", "tellurics", "flat", "PICASO"]
        for apodizer in config_data["apodizers"]:
            for strehl in config_data["strehls"]:
                for coronagraph in config_data["coronagraphs"]:
                    for thermal_model in thermal_models:
                        for reflected_model in reflected_models:
                            if thermal_model == "None" and reflected_model == "None":
                                continue
                            else:
                                get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=False)
                                if instru in instru_with_systematics:
                                    get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=True)
                                    get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematic=True, PCA=True, N_PCA=20)
    print('\n Calculating all SNR took {0:.3f} s'.format(time.time()-time0))



#######################################################################################################################
################################################# PLOTS: ##############################################################
#######################################################################################################################

########################################
# Classifications and statistics plots #
########################################

def planet_table_classification():
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    
    radius = np.array(planet_table["PlanetRadius"].value) # R_earth
    mass   = np.array(planet_table["PlanetMass"].value)   # M_earth
    
    mask_im = planet_table["DiscoveryMethod"]=="Imaging"
    mask_rv = planet_table["DiscoveryMethod"]=="Radial Velocity"
    mask_tr = planet_table["DiscoveryMethod"]=="Transit"
    mask_ot = (~mask_im) & (~mask_rv) & (~mask_tr)
    
    mask_cold = np.array(planet_table["PlanetTeq"].value < 250)
    mask_temp = np.array(250 <= planet_table["PlanetTeq"].value) & np.array(planet_table["PlanetTeq"].value < 500) 
    mask_warm = np.array(500 <= planet_table["PlanetTeq"].value) & np.array(planet_table["PlanetTeq"].value < 1000) 
    mask_hot  = np.array(1000 <= planet_table["PlanetTeq"].value) & np.array(planet_table["PlanetTeq"].value < 1500) 
    mask_vhot = np.array(1500 <= planet_table["PlanetTeq"].value) & np.array(planet_table["PlanetTeq"].value < 2000) 
    mask_uhot = np.array(2000 <= planet_table["PlanetTeq"].value)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$M_\oplus$", fontsize=14)
    plt.ylabel(r"$R_\oplus$", fontsize=14)
    plt.title(f"FastYield classification: {len(planet_table)} known exoplanets", fontsize=16)
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
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    
    # LEGEND
    ax = plt.gca()
    temp_colors_merged = {"Cold (<250 K)":          "#6fb0c9",
                          "Temperate (250–500 K)":  "#7ac87a",
                          "Warm (500–1000 K)":      "#c8c26f",
                          "Hot (1000–1500 K)":      "#f0a44f",
                          "Very-hot (1500–2000 K)": "#e36c4a",
                          "Ultra-hot (≥2000 K)":    "#c23a3a"}
    legend_temp = [mlines.Line2D([0], [0], marker='o', linestyle='', markersize=8, label=lbl, markerfacecolor=col, markeredgecolor='none') for lbl, col in temp_colors_merged.items()]
    leg1 = ax.legend(handles=legend_temp, title="Temperature bands", loc='lower right', frameon=True)
    ax.add_artist(leg1)
    legend_methods = [mlines.Line2D([], [], marker='s', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Imaging'), 
                      mlines.Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Radial Velocity'),
                      mlines.Line2D([], [], marker='v', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Transit'),
                      mlines.Line2D([], [], marker='P', linestyle='', markersize=8, markerfacecolor='black', markeredgecolor='black', label='Other')]
    ax.legend(handles=legend_methods, title="Discovery method", loc='upper left', frameon=True)
    
    # TYPES ZONES
    def build_type_bounds(planet_types, ax):
        """Retourne une liste {name, m1,m2,r1,r2} en agrégeant toutes les bandes Teq."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        by_type = {}
        for key, c in planet_types.items():
            tname = key.split(" ", 1)[1]  # ex: "Sub-Earth"
            d = by_type.setdefault(tname, {
                "m_min": np.inf, "m_max": -np.inf,
                "r_min": np.inf, "r_max": -np.inf
            })
            d["m_min"] = min(d["m_min"], c["mass_min"])
            d["m_max"] = max(d["m_max"], c["mass_max"])
            d["r_min"] = min(d["r_min"], c["radius_min"])
            d["r_max"] = max(d["r_max"], c["radius_max"])
    
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
    
    def draw_grey_highlight(ax, bounds,
                            face_alpha=0.10, edge_alpha=0.9,
                            edge_lw=1.5, label_fs=10):
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
            cx = np.sqrt(b["m1"]*b["m2"])
            cy = np.sqrt(b["r1"]*b["r2"])
            ax.text(cx, cy, b["name"],
                    ha="center", va="center",
                    fontsize=label_fs, color="black",
                    zorder=1e6+1, alpha=0.9,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.8)])
    
    bounds = build_type_bounds(planet_types, ax)
    draw_grey_highlight(ax, bounds, face_alpha=0.08, edge_alpha=0.85, edge_lw=1.2, label_fs=10)
    plt.draw()
    plt.tight_layout()
    plt.show()



def planet_table_statistics():
    smooth_corner = 1
    planet_table  = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table  = planet_table[get_valid_mask(planet_table["PlanetMass"])]

    ndim       = 5 # Mp, Rp, Tp, d, a
    data       = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeq"].value)
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
        contour_kwargs = {"colors": ["black"], "alpha": 0.85, "linewidth": 1.3},
        hist_kwargs    = {"color": "black", "alpha": 0.85, "linewidth": 1.3},
        label_kwargs   = {"fontsize": 16})
    figure.suptitle(f"Archive table statistics (with {len(planet_table)} known exoplanets)", fontsize=18, y=1.05, fontweight="bold")
    plt.gcf().set_dpi(300)
    plt.show()



###############
# Yield plots #
###############

def yield_plot_instrus_texp(thermal_model="BT-Settl", reflected_model="PICASO", fraction=False):
        
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table_harmoni          = load_planet_table(f"Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_andes            = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_{name_model}.ecsv")
    planet_table_eris             = load_planet_table(f"Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_{name_model}.ecsv")
    planet_table_mirimrs_non_syst = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_{thermal_model}.ecsv")
    planet_table_mirimrs_syst     = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_{thermal_model}.ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}.ecsv")
    planet_table_nircam           = load_planet_table(f"Archive_Pull_NIRCam_NO_SP_NO_JQ_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_non_syst = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_syst     = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{name_model}.ecsv")
    planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_{name_model}.ecsv")
        
    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    yield_harmoni          = np.zeros(len(exposure_time))
    yield_andes            = np.zeros(len(exposure_time))
    yield_eris             = np.zeros(len(exposure_time)) 
    yield_mirimrs_non_syst = np.zeros(len(exposure_time))
    yield_mirimrs_syst     = np.zeros(len(exposure_time))
    yield_mirimrs_syst_pca = np.zeros(len(exposure_time))
    yield_nircam           = np.zeros(len(exposure_time))
    yield_nirspec_non_syst = np.zeros(len(exposure_time))
    yield_nirspec_syst     = np.zeros(len(exposure_time))
    yield_nirspec_syst_pca = np.zeros(len(exposure_time))
    
    if fraction:
        ratio = 100
        norm_harmoni = len(planet_table_harmoni)
        norm_andes   = len(planet_table_andes)
        norm_eris    = len(planet_table_eris)
        norm_mirimrs = len(planet_table_mirimrs_non_syst)
        norm_nircam  = len(planet_table_nircam)
        norm_nirspec = len(planet_table_nirspec_non_syst)
    else:
        ratio = 1
        norm_harmoni = norm_andes = norm_eris = norm_mirimrs = norm_nircam = norm_nirspec = 1
    
    for i in range(len(exposure_time)):
        SNR_harmoni          = SNR_from_table(table=planet_table_harmoni,          exposure_time=exposure_time[i], band="INSTRU")
        SNR_andes            = SNR_from_table(table=planet_table_andes,            exposure_time=exposure_time[i], band="INSTRU")
        SNR_eris             = SNR_from_table(table=planet_table_eris,             exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_non_syst = SNR_from_table(table=planet_table_mirimrs_non_syst, exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_syst     = SNR_from_table(table=planet_table_mirimrs_syst,     exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_syst_pca = SNR_from_table(table=planet_table_mirimrs_syst_pca, exposure_time=exposure_time[i], band="INSTRU")
        SNR_nircam           = SNR_from_table(table=planet_table_nircam,           exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_non_syst = SNR_from_table(table=planet_table_nirspec_non_syst, exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_syst     = SNR_from_table(table=planet_table_nirspec_syst,     exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_syst_pca = SNR_from_table(table=planet_table_nirspec_syst_pca, exposure_time=exposure_time[i], band="INSTRU")

        yield_harmoni[i]          = ratio * len(planet_table_harmoni[SNR_harmoni>SNR_threshold])                   / norm_harmoni
        yield_andes[i]            = ratio * len(planet_table_andes[SNR_andes>SNR_threshold])                       / norm_andes
        yield_eris[i]             = ratio * len(planet_table_eris[SNR_eris>SNR_threshold])                         / norm_eris
        yield_mirimrs_non_syst[i] = ratio * len(planet_table_mirimrs_non_syst[SNR_mirimrs_non_syst>SNR_threshold]) / norm_mirimrs
        yield_mirimrs_syst[i]     = ratio * len(planet_table_mirimrs_syst[SNR_mirimrs_syst>SNR_threshold])         / norm_mirimrs
        yield_mirimrs_syst_pca[i] = ratio * len(planet_table_mirimrs_syst_pca[SNR_mirimrs_syst_pca>SNR_threshold]) / norm_mirimrs
        yield_nircam[i]           = ratio * len(planet_table_nircam[SNR_nircam>SNR_threshold])                     / norm_nircam
        yield_nirspec_non_syst[i] = ratio * len(planet_table_nirspec_non_syst[SNR_nirspec_non_syst>SNR_threshold]) / norm_nirspec
        yield_nirspec_syst[i]     = ratio * len(planet_table_nirspec_syst[SNR_nirspec_syst>SNR_threshold])         / norm_nirspec
        yield_nirspec_syst_pca[i] = ratio * len(planet_table_nirspec_syst_pca[SNR_nirspec_syst_pca>SNR_threshold]) / norm_nirspec
    
    lw = 2
    plt.figure(dpi=300, figsize=(14, 8))
    plt.plot(exposure_time, yield_harmoni,          lw=lw, c=colors_instru["HARMONI"], label="ELT/HARMONI")
    plt.plot(exposure_time, yield_andes,            lw=lw, c=colors_instru["ANDES"],   label="ELT/ANDES")
    plt.plot(exposure_time, yield_eris,             lw=lw, c=colors_instru["ERIS"],    label="VLT/ERIS")
    plt.plot(exposure_time, yield_mirimrs_non_syst, lw=lw, c=colors_instru["MIRIMRS"], label="JWST/MIRI/MRS")
    plt.plot(exposure_time, yield_mirimrs_syst,     lw=lw, c=colors_instru["MIRIMRS"], ls='--')
    plt.plot(exposure_time, yield_mirimrs_syst_pca, lw=lw, c=colors_instru["MIRIMRS"], ls=':')
    plt.plot(exposure_time, yield_nircam,           lw=lw, c=colors_instru["NIRCam"],  label="JWST/NIRCam")
    plt.plot(exposure_time, yield_nirspec_non_syst, lw=lw, c=colors_instru["NIRSpec"], label="JWST/NIRSpec/IFU")
    plt.plot(exposure_time, yield_nirspec_syst,     lw=lw, c=colors_instru["NIRSpec"], ls='--')
    plt.plot(exposure_time, yield_nirspec_syst_pca, lw=lw, c=colors_instru["NIRSpec"], ls=':')    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure time per target [mn]', fontsize=16)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=16)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=16)
        plt.yscale('log')
    plt.xlim(exposure_time[0], exposure_time[-1])
    plt.title('Known exoplanets detection yield', fontsize=18, weight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=13, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")    
    ax = plt.gca()
    ax_legend = ax.twinx()
    ax_legend.plot([], [], 'k-', label='Without Systematics', linewidth=3)
    ax_legend.plot([], [], 'k--', label='With Systematics', linewidth=3)
    ax_legend.plot([], [], 'k:', label='With Systematics + PCA', linewidth=3)
    ax_legend.legend(fontsize=13, loc="lower right", frameon=True, edgecolor="gray", facecolor="whitesmoke")
    ax_legend.tick_params(axis='y', colors='w')  # Masking ticks
    plt.tight_layout()
    plt.show()



def yield_plot_bands_texp(instru="HARMONI", thermal_model="BT-Settl", reflected_model="PICASO", systematic=False, PCA=False, fraction=False):
        
    ANDES_R = False
    
    ls_modes     = ["-", "--", ":"]

    config_data = get_config_data(instru)
    apodizer    = "NO_SP"
    strehl      = config_data["strehls"][0]
    coronagraph = None
    iwa, owa    = get_wa(config_data=config_data, sep_unit="mas")    

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"  
    if instru == "HARMONI":
        modes         = ["NO_SP", "SP1", "SP_Prox"]
        planet_tables = []
        NbPlanet      = np.zeros(len(modes), dtype=int)
        iwa           = 30  # mas
        for nm, mode in enumerate(modes):
            pt           = load_planet_table(f"Archive_Pull_{instru}_{mode}_{strehl}_{suffix}_{name_model}.ecsv")
            pt           = pt[pt["AngSep"] > iwa * u.mas]
            NbPlanet[nm] = len(pt)
            planet_tables.append(pt)
    elif instru == "ANDES":
        modes         = [None, "LYOT"]
        planet_tables = []
        NbPlanet      = np.zeros(len(modes), dtype=int)
        for nm, mode in enumerate(modes):
            mode_str     = "_" + str(mode) if mode is not None else ""
            pt           = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}{mode_str}_{suffix}_{name_model}.ecsv")
            pt           = pt[pt["AngSep"] > iwa * u.mas]
            NbPlanet[nm] = len(pt)
            planet_tables.append(pt)
    else:
        pt            = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}_{suffix}_{name_model}.ecsv")
        pt            = pt[pt["AngSep"] > iwa * u.mas]
        planet_tables = [pt]
        NbPlanet      = np.array([len(pt)])
        modes         = [""]

    name_bands      = config_data["gratings"]
    if instru == "ANDES" and ANDES_R:
        pxscale_to_keep = "10mas" # for ANDES
        mask       = [pxscale_to_keep in band for band in name_bands]
        name_bands = [band for band, keep in zip(name_bands, mask) if keep]
    n_gratings = len(name_bands)

    cmap = plt.get_cmap("Spectral", n_gratings+1) if (n_gratings%2 != 0) else plt.get_cmap("Spectral", n_gratings)

    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    yields        = np.zeros((n_gratings, len(modes), len(exposure_time)))

    for i, band in enumerate(name_bands):
        for nm in range(len(modes)):
            for j in range(len(exposure_time)):
                SNR = SNR_from_table(table=planet_tables[nm], exposure_time=exposure_time[j], band=band)
                if fraction :
                    yields[i, nm, j] = len(planet_tables[nm][SNR>SNR_threshold]) * 100/NbPlanet[nm]
                else:
                    yields[i, nm, j] = len(planet_tables[nm][SNR>SNR_threshold])

    plt.figure(dpi=300, figsize=(14, 8))
    for i, band in enumerate(name_bands):
        for nm in range(len(modes)):
            if not (yields[i, nm]==0).all():
                if nm == 0:
                    if instru == "ANDES" and ANDES_R:
                        plt.plot(exposure_time, yields[i, nm], color=cmap(i), label=band.replace('_', ' ').replace(' '+pxscale_to_keep, '').replace('YJH', 'R =')+' 000', ls=ls_modes[nm], lw=3)
                    else:
                        plt.plot(exposure_time, yields[i, nm], color=cmap(i), label=band.replace("_", " "), ls=ls_modes[nm], lw=3)
                else:
                    plt.plot(exposure_time, yields[i, nm], color=cmap(i), ls=ls_modes[nm], lw=3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure time per target [mn]', fontsize=16)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=16)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=16)
        plt.yscale('log')
    plt.xlim(exposure_time[0], exposure_time[-1])
    if strehl == "NO_JQ":
        plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above {iwa:.0f} mas\n (with {name_model} model in {spectrum_contributions} light)", fontsize=18, weight='bold')
    else:
        plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above {iwa:.0f} mas for {strehl} strehl\n (with {name_model} model in {spectrum_contributions} light)", fontsize=18, weight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(title="Bands", title_fontsize=16, fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")    
    if len(modes) > 1: 
        ax = plt.gca()
        ax_legend = ax.twinx()
        for nm, mode in enumerate(modes):
            ax_legend.plot([], [], c="k", ls=ls_modes[nm], label=str(mode).replace("_", " ").replace("None", "w/o coronagraph"), lw=3)
        if instru=="HARMONI":
            ax_legend.legend(title="Apodizers", title_fontsize=16, fontsize=14, loc="lower right", frameon=True, edgecolor="gray", facecolor="whitesmoke")
        elif instru=="ANDES":
            ax_legend.legend(title="Coronagraphs", title_fontsize=16, fontsize=14, loc="lower right", frameon=True, edgecolor="gray", facecolor="whitesmoke")
        ax_legend.tick_params(axis='y', colors='w')  # Masking ticks
    plt.tight_layout()
    plt.show()



####################
# Yield histograms #
####################

def yield_hist_instrus_ptypes(exposure_time=120, thermal_model="BT-Settl", reflected_model="tellurics", planet_types=planet_types_reduced, fraction=False):
        
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    planet_table_harmoni          = load_planet_table(f"Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_andes            = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_{name_model}.ecsv")
    planet_table_eris             = load_planet_table(f"Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_{name_model}.ecsv")
    planet_table_mirimrs_non_syst = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_{thermal_model}.ecsv")
    planet_table_mirimrs_syst     = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_{thermal_model}.ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}.ecsv")
    planet_table_nircam           = load_planet_table(f"Archive_Pull_NIRCam_NO_SP_NO_JQ_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_non_syst = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_syst     = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{name_model}.ecsv")
    planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_{name_model}.ecsv") # à modifier

    planet_table_harmoni["SNR"]          = SNR_from_table(table=planet_table_harmoni,          exposure_time=exposure_time, band="INSTRU")
    planet_table_andes["SNR"]            = SNR_from_table(table=planet_table_andes,            exposure_time=exposure_time, band="INSTRU")
    planet_table_eris["SNR"]             = SNR_from_table(table=planet_table_eris,             exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_non_syst["SNR"] = SNR_from_table(table=planet_table_mirimrs_non_syst, exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_syst["SNR"]     = SNR_from_table(table=planet_table_mirimrs_syst,     exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_syst_pca["SNR"] = SNR_from_table(table=planet_table_mirimrs_syst_pca, exposure_time=exposure_time, band="INSTRU")
    planet_table_nircam["SNR"]           = SNR_from_table(table=planet_table_nircam,           exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_non_syst["SNR"] = SNR_from_table(table=planet_table_nirspec_non_syst, exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_syst["SNR"]     = SNR_from_table(table=planet_table_nirspec_syst,     exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_syst_pca["SNR"] = SNR_from_table(table=planet_table_nirspec_syst_pca, exposure_time=exposure_time, band="INSTRU")

    # Dictionnaries giving the list of planets for mp[type]
    mp_harmoni          = build_match_dict(planet_table_harmoni, planet_types=planet_types)
    mp_andes            = build_match_dict(planet_table_andes, planet_types=planet_types)
    mp_eris             = build_match_dict(planet_table_eris, planet_types=planet_types)
    mp_mirimrs_non_syst = build_match_dict(planet_table_mirimrs_non_syst, planet_types=planet_types)
    mp_mirimrs_syst     = build_match_dict(planet_table_mirimrs_syst, planet_types=planet_types)
    mp_mirimrs_syst_pca = build_match_dict(planet_table_mirimrs_syst_pca, planet_types=planet_types)
    mp_nircam           = build_match_dict(planet_table_nircam, planet_types=planet_types)
    mp_nirspec_non_syst = build_match_dict(planet_table_nirspec_non_syst, planet_types=planet_types)
    mp_nirspec_syst     = build_match_dict(planet_table_nirspec_syst, planet_types=planet_types)
    mp_nirspec_syst_pca = build_match_dict(planet_table_nirspec_syst_pca, planet_types=planet_types)

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
            ratio = 1
            norm_harmoni = norm_andes = norm_eris = norm_mirimrs = norm_nircam = norm_nirspec = 1
        
        yield_harmoni[i]          = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_harmoni[ptype])          / norm_harmoni if norm_harmoni > 0 else 0
        yield_andes[i]            = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_andes[ptype])            / norm_andes   if norm_andes > 0   else 0
        yield_eris[i]             = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_eris[ptype])             / norm_eris    if norm_eris > 0    else 0
        yield_mirimrs_non_syst[i] = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_mirimrs_non_syst[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst[i]     = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_mirimrs_syst[ptype])     / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst_pca[i] = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_mirimrs_syst_pca[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_nircam[i]           = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_nircam[ptype])           / norm_nircam  if norm_nircam > 0  else 0
        yield_nirspec_non_syst[i] = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_nirspec_non_syst[ptype]) / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst[i]     = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_nirspec_syst[ptype])     / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst_pca[i] = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_nirspec_syst_pca[ptype]) / norm_nirspec if norm_nirspec > 0 else 0

    N_table   = 10
    bar_width = 1 / (N_table+2)
    linewidth = 1
    indices   = np.arange(len(planet_types_array))

    plt.figure(figsize=(14, 8), dpi=300)

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
    plt.bar(indices - 2.5*bar_width, yield_eris,             bar_width, edgecolor="black", color=colors_instru["ERIS"],    linewidth=linewidth, label="VLT/ERIS")
    plt.bar(indices - 1.5*bar_width, yield_mirimrs_non_syst, bar_width, edgecolor="black", color=colors_instru["MIRIMRS"], linewidth=linewidth, label="JWST/MIRI/MRS")
    plt.bar(indices - 0.5*bar_width, yield_mirimrs_syst,     bar_width, edgecolor="black", color=colors_instru["MIRIMRS"], linewidth=linewidth, label="JWST/MIRI/MRS (with syst)",        hatch='//')
    plt.bar(indices + 0.5*bar_width, yield_mirimrs_syst_pca, bar_width, edgecolor="black", color=colors_instru["MIRIMRS"], linewidth=linewidth, label="JWST/MIRI/MRS (with syst+PCA)",    hatch='xx')
    plt.bar(indices + 1.5*bar_width, yield_nircam,           bar_width, edgecolor="black", color=colors_instru["NIRCam"],  linewidth=linewidth, label="JWST/NIRCam")
    plt.bar(indices + 2.5*bar_width, yield_nirspec_non_syst, bar_width, edgecolor="black", color=colors_instru["NIRSpec"], linewidth=linewidth, label="JWST/NIRSpec/IFU")
    plt.bar(indices + 3.5*bar_width, yield_nirspec_syst,     bar_width, edgecolor="black", color=colors_instru["NIRSpec"], linewidth=linewidth, label="JWST/NIRSpec/IFU (with syst)",     hatch='//')
    plt.bar(indices + 4.5*bar_width, yield_nirspec_syst_pca, bar_width, edgecolor="black", color=colors_instru["NIRSpec"], linewidth=linewidth, label="JWST/NIRSpec/IFU (with syst+PCA)", hatch='xx')

    plt.xticks(indices, planet_types_array, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.5, len(planet_types_array) - 0.5)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=16)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=16)
        plt.yscale('log')
    plt.title(f'Known Exoplanets Detection Yield for {int(round(exposure_time/60))} h per target', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=-10)
    plt.legend(title="Instruments", title_fontsize=14, fontsize=12, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke", ncol=2)
    plt.tight_layout()
    plt.show()



def yield_hist_instrus_ptypes_ELT(exposure_time=120, thermal_model="BT-Settl", reflected_model="tellurics", planet_types=planet_types, fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox", "ANDES", "ANDES+LYOT"]):
    
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table_harmoni      = load_planet_table("Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_harmoni_prox = load_planet_table("Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_andes        = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_without_systematics_"+name_model+".ecsv")
    planet_table_andes_lyot   = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_"+name_model+".ecsv")
    
    planet_table_harmoni["SNR"]      = SNR_from_table(table=planet_table_harmoni,      exposure_time=exposure_time, band="INSTRU")
    planet_table_harmoni_prox["SNR"] = SNR_from_table(table=planet_table_harmoni_prox, exposure_time=exposure_time, band="INSTRU")
    planet_table_andes["SNR"]        = SNR_from_table(table=planet_table_andes,        exposure_time=exposure_time, band="INSTRU")
    planet_table_andes_lyot["SNR"]   = SNR_from_table(table=planet_table_andes_lyot,   exposure_time=exposure_time, band="INSTRU")
    
    mp_harmoni      = build_match_dict(planet_table_harmoni,      planet_types=planet_types)
    mp_harmoni_prox = build_match_dict(planet_table_harmoni_prox, planet_types=planet_types)
    mp_andes        = build_match_dict(planet_table_andes,        planet_types=planet_types)
    mp_andes_lyot   = build_match_dict(planet_table_andes_lyot,   planet_types=planet_types)
    
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
        
        yield_harmoni[i]      = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_harmoni[ptype])      / norm_harmoni      if norm_harmoni > 0      else 0
        yield_harmoni_prox[i] = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_harmoni_prox[ptype]) / norm_harmoni_prox if norm_harmoni_prox > 0 else 0
        yield_andes[i]        = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_andes[ptype])        / norm_andes        if norm_andes > 0        else 0
        yield_andes_lyot[i]   = ratio * sum(planet["SNR"] > SNR_threshold for planet in mp_andes_lyot[ptype])   / norm_andes_lyot   if norm_andes_lyot > 0   else 0
    
    N_table   = len(instrus)
    bar_width = 1 / (N_table+2)
    linewidth = 1
    indices   = np.arange(len(planet_types_array))
    
    plt.figure(figsize=(14, 8), dpi=300)
    
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
        plt.text(xmid, 1.01, tag.upper(), ha='center', va='bottom', transform=plt.gca().get_xaxis_transform(), fontsize=14, color=sep.get(tag, '0.4'), fontweight='bold')
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
    plt.yticks(fontsize=14)
    plt.xlim(-0.5, len(planet_types_array) - 0.5)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=16)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=16)
        plt.yscale('log')
    plt.title(f'Known Exoplanets Detection Yield for {int(round(exposure_time/60))} h per target', fontsize=18, fontweight='bold', pad=30)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=-10)
    plt.legend(title="Instruments", title_fontsize=14, fontsize=12, loc="upper center", frameon=True, edgecolor="gray", facecolor="whitesmoke", ncol=2)
    plt.tight_layout()
    plt.show()



#################
# Yield corners #
#################

def yield_corner_instru(instru="HARMONI", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", coronagraph=None, band="INSTRU", systematic=False, PCA=False):
    smooth_corner = 1
    ndim          = 6 # Mp, Rp, Tp, a, d, sep
    config_data = get_config_data(instru)
    
    # WORKING ANGLE
    iwa, owa = get_wa(config_data=config_data, sep_unit="mas")    

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeq"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    
    # SETTING DATA
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
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
        contour_kwargs = {"colors": ["black"], "alpha": 0.85, "linewidth": 1.3},
        hist_kwargs    = {"color": "black", "alpha": 0.85, "linewidth": 1.3},
        label_kwargs  = {"fontsize": 16})
    
    # DETECTIONS TABLE
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv")
    planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeq"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
    planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
    SNR          = SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band)
    planet_table = planet_table[SNR>SNR_threshold]
    data       = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeq"].value)
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
        contour_kwargs = {"colors": ["crimson"], "alpha": 0.85, "linewidth": 1.3},
        hist_kwargs    = {"color": "crimson", "alpha": 0.85, "linewidth": 1.3},
        smooth         = smooth_corner,
        smooth1d       = smooth_corner)

    figure.suptitle(f"{instru} re-detections statistics with {len(planet_table)} / {len(planet_table_raw)} detections between {round(iwa)} and {round(owa)} mas\nfor {round(exposure_time/60)} hours per target (with {spectrum_contributions} light with {name_model})", fontsize=18, y=1.05, fontweight="bold")
    plt.gcf().set_dpi(300)
    plt.show()
    


def yield_corner_instrus(instru1="HARMONI", instru2="ANDES", apodizer1="NO_SP", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", coronagraph1=None, coronagraph2=None, exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", systematic=False, PCA=False):
    instrus       = [instru1,      instru2]
    apodizers     = [apodizer1,    apodizer2]
    strehls       = [strehl1,      strehl2]
    coronagraphs  = [coronagraph1, coronagraph2]
    colors_instru = ["crimson", "royalblue"]
    smooth_corner = 1
    ndim          = 6 # Mp, Rp, Tp, a, d, sep
    alpha_earth   = 0.5
    color_earth   = "seagreen"
    earth_values  = np.array([np.log10(1), np.log10(1), 300, np.log10(1), None, None]) # M_earth, R_earth, T_earth, 1 AU, None, None

    # WORKING ANGLE
    IWA = np.zeros((len(instrus)))
    OWA = np.zeros((len(instrus)))
    for ni, instru in enumerate(instrus):
        IWA[ni], OWA[ni] = get_wa(config_data=get_config_data(instru), sep_unit="mas")    
    iwa = np.nanmin(IWA)
    owa = np.nanmax(OWA)
    
    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeq"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    #planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)] # TODO : crop upper WA (with owa) ?
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data           = data_raw,
        bins           = 20,
        labels         = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles      = [0.16, 0.5, 0.84],   # below -+1 sigma 
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
        contour_kwargs = {"colors": ["black"], "alpha": 0.85, "linewidth": 1.3},
        hist_kwargs    = {"color": "black", "alpha": 0.85, "linewidth": 1.3},
        label_kwargs   = {"fontsize": 16})
    
    # DETECTIONS TABLES
    yields = np.zeros((len(instrus)))
    for ni, instru in enumerate(instrus):
        apodizer        = apodizers[ni]
        strehl          = strehls[ni]
        coronagraph_str = "_"+str(coronagraphs[ni]) if coronagraphs[ni] is not None else ""
        if systematic:
            suffix = "with_systematics+PCA" if PCA else "with_systematics"
        else:
            suffix = "without_systematics"
        planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv")
        planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeq"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
        planet_table = planet_table[(planet_table["AngSep"]>IWA[ni]*u.mas)&(planet_table["AngSep"]<OWA[ni]*u.mas)]
        #planet_table = planet_table[(planet_table["AngSep"]>IWA[ni]*u.mas)] # TODO : crop upper WA (with owa) ?
        SNR          = SNR_from_table(table=planet_table, exposure_time=exposure_time, band="INSTRU")
        planet_table = planet_table[SNR>SNR_threshold]
        yields[ni]   = len(planet_table)
        data       = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeq"].value)
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
            color          = colors_instru[ni],
            contour_kwargs = {"colors": [colors_instru[ni]], "alpha": 0.85, "linewidth": 1.3},
            hist_kwargs    = {"color": colors_instru[ni], "alpha": 0.85, "linewidth": 1.3},
            smooth         = smooth_corner,
            smooth1d       = smooth_corner)
            
    # Earth points
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(ndim):
        if earth_values[i] is not None:
            ax = axes[i, i]
            ax.axvline(earth_values[i], color=color_earth, alpha=alpha_earth)
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if earth_values[xi] is not None:
                ax.axvline(earth_values[xi], color=color_earth, alpha=alpha_earth, lw=3)
            if earth_values[yi] is not None:
                ax.axhline(earth_values[yi], color=color_earth, alpha=alpha_earth, lw=3)
            if earth_values[xi] is not None and earth_values[yi] is not None:
                ax.plot(earth_values[xi], earth_values[yi], marker="s", c=color_earth, ms=10)
               
    handles = []
    for ni, instru in enumerate(instrus):
        handles.append(mlines.Line2D([], [], color="w", label=""))
        if instru == "HARMONI":
            if apodizers[ni] == "NO_SP":
                instru_label = "HARMONI w/o apodizer"
            else:
                instru_label = f"HARMONI w/ {apodizers[ni].replace('_', ' ')}"
        elif instru == "ANDES":
            if coronagraphs[ni] is None:
                instru_label = "ANDES w/o coronagraph"
            else:
                instru_label = f"ANDES w/ {coronagraphs[ni]}"
        else:
            instru_label = instru
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=colors_instru[ni], label=instru_label+f" ({round(yields[ni])} / {len(planet_table_raw)})"))
    plt.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=14, title="Instruments:", title_fontsize=16)
    handles = [mlines.Line2D([], [], linestyle="-", marker="s", color=color_earth, label="Earth")]
    ax_legend = plt.gca().twinx()
    ax_legend.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="lower right", fontsize=14)
    ax_legend.tick_params(axis='y', colors='w')
    if instru1 == instru2 == "HARMONI" or instru1 == instru2 == "ANDES":
        figure.suptitle(f"{instru1} re-detections statistics with {len(planet_table_raw)} known planets between {round(iwa)} and {round(owa)} mas\nfor {round(exposure_time/60)} hours per target (with {name_model} model in {spectrum_contributions} light)", fontsize=18, y=1.05, fontweight="bold")
    else:
        figure.suptitle(f"{instru1} VS {instru2} re-detections statistics with {len(planet_table_raw)} known planets between {round(iwa)} and {round(owa)} mas\nfor {round(exposure_time/60)} hours per target (with {name_model} model in {spectrum_contributions} light)", fontsize=18, y=1.05, fontweight="bold")
    plt.gcf().set_dpi(300)
    plt.tight_layout()
    plt.show()



def yield_corner_models(model1="tellurics", model2="flat", instru="ANDES", apodizer="NO_SP", strehl="MED", exposure_time=6*60, band="INSTRU"):
    models        = [model1, model2]
    color_models  = ["crimson", "royalblue"]
    smooth_corner = 1
    ndim          = 6 # Mp, Rp, Tp, d, a, sep
    alpha_earth   = 0.5
    color_earth   = "g"
    earth_values  = np.array([np.log10(1), np.log10(1), 300, np.log10(1), None, None])
    config_data   = get_config_data(instru)
    
    # WORKING ANGLE
    iwa, owa = get_wa(config_data=config_data, sep_unit="mas")    

    # MODELS NAME
    if model1 in thermal_models and model2 in thermal_models:
        spectrum_contributions = "thermal"
    elif model1 in reflected_models and model2 in reflected_models:
        spectrum_contributions = "reflected"
    else:
        raise KeyError("WRONG MODEL1 OR MODEL2")

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeq"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    #planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)] # TODO : crop upper WA (with owa) ?
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data           = data_raw,
        bins           = 20,
        labels         = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles      = [0.16, 0.5, 0.84],   # below -+1 sigma 
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
        contour_kwargs = {"colors": ["black"], "alpha": 0.85, "linewidth": 1.3},
        hist_kwargs    = {"color": "black", "alpha": 0.85, "linewidth": 1.3},
        label_kwargs   = {"fontsize": 16})
    
    # DETECTIONS TABLES
    yields = np.zeros((len(models)))
    for nm, model in enumerate(models):
        if model == "PICASO":
            name_model = "PICASO_"+spectrum_contributions+"_only"
        else:
            name_model = model
        planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}_without_systematics_{name_model}.ecsv")
        planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeq"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
        planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
        #planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)] # TODO : crop upper WA (with owa) ?
        SNR          = SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band)
        planet_table = planet_table[SNR>SNR_threshold]
        yields[nm]   = len(planet_table)
        data       = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeq"].value)
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
            color          = color_models[nm],
            contour_kwargs = {"colors": [color_models[nm]], "alpha": 0.85, "linewidth": 1.3},
            hist_kwargs    = {"color": color_models[nm], "alpha": 0.85, "linewidth": 1.3},
            smooth         = smooth_corner,
            smooth1d       = smooth_corner)
    
    # Earth points
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for i in range(ndim):
        if earth_values[i] is not None:
            ax = axes[i, i]
            ax.axvline(earth_values[i], color=color_earth, alpha=alpha_earth)
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if earth_values[xi] is not None:
                ax.axvline(earth_values[xi], color=color_earth, alpha=alpha_earth, lw=5)
            if earth_values[yi] is not None:
                ax.axhline(earth_values[yi], color=color_earth, alpha=alpha_earth, lw=5)
            if earth_values[xi] is not None and earth_values[yi] is not None:
                ax.plot(earth_values[xi], earth_values[yi], "s"+color_earth, ms=10)
                
    handles = []
    for nm, model in enumerate(models):
        handles.append(mlines.Line2D([], [], color="w", label=""))
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=color_models[nm], label=model+f" ({round(yields[nm])} / {len(planet_table_raw)})"))
    plt.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=14, title="Models:", title_fontsize=16)
    handles = [mlines.Line2D([], [], linestyle="-", marker="s", color=color_earth, label="Earth")]
    ax_legend = plt.gca().twinx()
    ax_legend.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="lower right", fontsize=14)
    figure.suptitle(f"{instru} re-detections statistics with {len(planet_table_raw)} known planets between {round(iwa)} and {round(owa)} mas\nfor {round(exposure_time/60)} hours per target (with {spectrum_contributions} light)", fontsize=18, y=1.05, fontweight="bold")
    plt.gcf().set_dpi(300)
    plt.show()



##################
# Yield contrast #
##################

def yield_contrast_instru(instru="ANDES", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", coronagraph=None, systematic=False, PCA=False, table="Archive", band="INSTRU"):
    config_data = get_config_data(instru)

    # WORKING ANGLE
    iwa, owa = get_wa(config_data=config_data)    

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
    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    tablename = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"

    planet_table = load_planet_table(tablename)
    SNR          = SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band)
    planet_table = planet_table[SNR>SNR_threshold]
    SNR          = SNR[SNR>SNR_threshold]
    
    x = np.array(planet_table["AngSep"].value) # sep axis [mas]
    if config_data["sep_unit"]=="arcsec":
        x /= 1000
    mag_p = planet_table[f"PlanetINSTRUmag({instru})({spectrum_contributions})"]
    mag_s = planet_table[f"StarINSTRUmag({instru})"]
    y     = 10**(-(mag_p-mag_s)/2.5)
    z     = np.array(planet_table["PlanetTeq"].value) # color axis [K]

    im_mask = planet_table["DiscoveryMethod"]=="Imaging"
    tr_mask = planet_table["DiscoveryMethod"]=="Transit"
    rv_mask = planet_table["DiscoveryMethod"]=="Radial Velocity"
    ot_mask = (planet_table["DiscoveryMethod"]!="Imaging") & (planet_table["DiscoveryMethod"]!="Radial Velocity") & (planet_table["DiscoveryMethod"]!="Transit")

    # Taille des points (capés) en fonction du SNR
    s = SNR_to_size(SNR=SNR, s0=50, ds=200, SNR_min=SNR_threshold, SNR_max=1_000)

    # Colormap + normalisation (Teff en log)
    cmap = plt.get_cmap("rainbow")
    znanmin = np.nanmin(z[np.isfinite(z)])
    znanmax = np.nanmax(z[np.isfinite(z)])
    norm = LogNorm(vmin=znanmin, vmax=znanmax)
        
    # Figure/axes
    fig = plt.figure(figsize=(13.5, 6.5), dpi=300)
    ax1 = plt.gca()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(np.nanmin(x), np.nanmax(x))
    ax1.set_ylim(np.nanmin(y), np.nanmax(y))
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
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
    plt.show()



##########
# Others #
##########   

def Vsini_plots():
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    
    nbins = 50
    pm_mask = np.logical_not(get_invalid_mask(planet_table["PlanetMass"]))
    plt.figure(dpi=300)
    plt.hist(np.array(planet_table["PlanetVsini"][pm_mask&(300*u.Mearth<planet_table["PlanetMass"])]), bins=nbins, edgecolor='black', alpha=0.666, label="Super-Jupiters", zorder=3)
    plt.hist(np.array(planet_table["PlanetVsini"][pm_mask&(100*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=300*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Jupiters-like", zorder=3)
    plt.hist(np.array(planet_table["PlanetVsini"][pm_mask&(20*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=100*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Neptunes-like", zorder=3)
    plt.hist(np.array(planet_table["PlanetVsini"][pm_mask&(5*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=20*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Super-Earths", zorder=3)
    plt.hist(np.array(planet_table["PlanetVsini"][pm_mask&(planet_table["PlanetMass"]<=5*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Earths-like", zorder=3)
    plt.ylabel("Occurences", fontsize=14)
    plt.xlabel("Vsin(i) [km/s]", fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    
    st_mask = np.logical_not(get_invalid_mask(planet_table["StarTeff"]))
    plt.figure(dpi=300)
    plt.hist(np.array(planet_table["StarVsini"][st_mask & (10000 * u.K < planet_table["StarTeff"])]), bins=nbins, edgecolor='black', alpha=0.666, label="Very hot stars", zorder=3)
    plt.hist(np.array(planet_table["StarVsini"][st_mask & (6000 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Hot stars", zorder=3)
    plt.hist(np.array(planet_table["StarVsini"][st_mask & (3500 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Solar-type stars", zorder=3)
    plt.hist(np.array(planet_table["StarVsini"][st_mask & (planet_table["StarTeff"] <= 3500 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Cool stars", zorder=3)
    plt.ylabel("Occurrences", fontsize=14)
    plt.xlabel("Vsin(i) [km/s]", fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()



################################
# ELT exo-earth yield contrast #
################################

def process_contrast(args):
    """
    Worker: compute SNR and band magnitudes for a single planet.

    Parameters
    ----------
    args : tuple
        (idx, planet_row)

    Returns
    -------
    result : tuple
         - idx : int
         - mags_p : dict {'Star{instru}mag' : float, 'Planet{instru}mag' : float, 'Star{band}mag' : float, ... }
         - name_band : list[str]
         - signal    : np.ndarray [e-/DIT]  (per name_band entry)
         - sigma_f   : np.ndarray [e-/DIT]
         - sigma_s   : np.ndarray [e-/DIT]
         - DIT       : np.ndarray [s]
    """
    # idx, planet row
    idx, planet = args
    
    # Context
    instru                 = _CONTRAST_CTX["instru"]
    config_data            = _CONTRAST_CTX["config_data"]
    thermal_model          = _CONTRAST_CTX["thermal_model"]
    reflected_model        = _CONTRAST_CTX["reflected_model"]
    wave_instru            = _CONTRAST_CTX["wave_instru"]
    wave_K                 = _CONTRAST_CTX["wave_K"]
    vega_flux              = _CONTRAST_CTX["vega_flux"]
    vega_spectrum_K        = _CONTRAST_CTX["vega_spectrum_K"]
    band0                  = _CONTRAST_CTX["band0"]
    exposure_time          = _CONTRAST_CTX["exposure_time"]
    Rc                     = _CONTRAST_CTX["Rc"]
    filter_type            = _CONTRAST_CTX["filter_type"]
    systematic             = _CONTRAST_CTX["systematic"]
    PCA                    = _CONTRAST_CTX["PCA"]
    N_PCA                  = _CONTRAST_CTX["N_PCA"]
    masks                  = _CONTRAST_CTX["masks"]
    sep_max                = _CONTRAST_CTX["sep_max"]
    spectrum_contributions = _CONTRAST_CTX["spectrum_contributions"]

    # Computing models on wave_instru    
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=instru, wave_instru=wave_instru, wave_K=wave_K, vega_spectrum_K=vega_spectrum_K, show=False)
    
    # Recalculates the magnitude in case the thermal model is no longer BT-Settl or the reflected model is no longer PICASO (the mag changes with regards to the raw archive table with the estimated magnitudes)
    mask_instru     = masks[instru]
    mag_s           = get_mag(flux_obs=star_spectrum.flux[mask_instru],    flux_ref=vega_flux[instru])
    mag_p_thermal   = get_mag(flux_obs=planet_thermal.flux[mask_instru],   flux_ref=vega_flux[instru])
    mag_p_reflected = get_mag(flux_obs=planet_reflected.flux[mask_instru], flux_ref=vega_flux[instru])
    mag_p_total     = get_mag(flux_obs=planet_spectrum.flux[mask_instru],  flux_ref=vega_flux[instru])
    
    if spectrum_contributions=="thermal":
        planet_spectrum = planet_thermal.copy()
        mag_p           = mag_p_thermal
    elif spectrum_contributions=="reflected":
        planet_spectrum = planet_reflected.copy()
        mag_p           = mag_p_reflected
    elif spectrum_contributions=="thermal+reflected":
        planet_spectrum = planet_spectrum.copy()
        mag_p           = mag_p_total

    # Computing the SNR for the planet
    contrasts_min = []
    for apodizer in config_data["apodizers"]:
        for strehl in config_data["strehls"]:
            # strehl = "MED"
            # if apodizer not in ["NO_SP", "SP1"]:
            #     continue
            for coronagraph in config_data["coronagraphs"]:
                name_bands, separation, curves = FastCurves(instru=instru, calculation="contrast", mag_star=mag_s, band0=band0, exposure_time=exposure_time, mag_planet=mag_p, separation_planet=sep_max/1000, show_plot=False, verbose=False, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematic=systematic, PCA=PCA, N_PCA=N_PCA)
                contrasts_min.append(np.nanmin(np.stack(curves, axis=0), axis=0))

    contrast_min = np.nanmin(np.stack(contrasts_min, axis=0), axis=0)

    return idx, mag_s, mag_p_thermal, mag_p_reflected, mag_p_total, separation[0], contrast_min



def get_planet_table_contrast(instru, planet_table, exposure_time, thermal_model="None", reflected_model="None", spectrum_contributions=None, Rc=100, filter_type="gaussian", systematic=False, PCA=False, N_PCA=20, force_table_calc=False, sep_max=None):
   
    # Contribution and model labels
    _, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # Suffix
    suffix = "with systematics+PCA" if (systematic and PCA) else ("with systematics" if systematic else "without systematics")
    
    # Filename
    if systematic:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename = archive_path + f"ELT_Contrast_Earth_Like_Archive_Pull_{instru}_{suffix}_{name_model}_{spectrum_contributions}_t{exposure_time}mn_Rc{Rc}.ecsv"

    # Retrieving or computing the table
    try:
        print(f"\nOpening existing table: {filename}")
        if force_table_calc:
            raise ValueError("Forcing table calculation: force_table_calc=True")
        planet_table = QTable.read(filename, format="ascii.ecsv")
        separation   = fits.getdata(filename.replace(".ecsv", "_separation.fits"))

    except Exception as e: # CREATING NOISELESS CUBE
    
        print(f"\nComputing the table: {e}...")
        
        planet_table = planet_table.copy()
        config_data  = get_config_data(instru)
        time1        = time.time()
        if systematic and filter_type == "gaussian_fast":
            filter_type = "gaussian" # "gaussian_fast" is bad for handling systematic estimations
        
        # --- 4) Wavelength grids and Vega ---
        # K-band for photometry
        R_K    = R0_min # Only for photometric purposes (does not need more resolution)
        dl_K   = (lmin_K+lmax_K)/2 / (2*R_K)
        wave_K = np.arange(lmin_K, lmax_K, dl_K)
        
        # Instrument band (high R for spectrophotometry)
        lmin_instru = config_data["lambda_range"]["lambda_min"] # in µm
        lmax_instru = config_data["lambda_range"]["lambda_max"] # in µm
        R_instru    = R0_max # Abritrary resolution (needs to be high enough)
        dl_instru   = (lmin_instru+lmax_instru)/2 / (2*R_instru)
        wave_instru = np.arange(0.98*lmin_instru, 1.02*lmax_instru, dl_instru)
        
        # Vega spectrum on K-band and instru-band [J/s/m2/µm]
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm=False)
        vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_instru, renorm=False)
        
        # --- 5) Create columns for signal, noise and DIT length ---
        planet_table["5sigma-contrast"]  = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    
        # --- 6) Bandpass masks and Vega flux on bands ---
        vega_flux         = {}
        masks             = {}
        masks[instru]     = (wave_instru >= lmin_instru) & (wave_instru <= lmax_instru)
        vega_flux[instru] = np.nanmean(vega_spectrum.flux[masks[instru]])
    
        # Band where magnitudes are defined for the FastCurves computations
        band0 = "instru"
        
        # Print
        print(f"\n {instru} {suffix} ({thermal_model} & {reflected_model})")
    
        # --- 7) Init global context for workers ---
        global _CONTRAST_CTX
        _CONTRAST_CTX = dict(instru=instru, config_data=config_data, thermal_model=thermal_model, reflected_model=reflected_model, wave_instru=wave_instru, wave_K=wave_K, vega_flux=vega_flux, vega_spectrum_K=vega_spectrum_K, band0=band0, exposure_time=exposure_time, Rc=Rc, filter_type=filter_type, systematic=systematic, PCA=PCA, N_PCA=N_PCA, masks=masks, sep_max=sep_max, spectrum_contributions=spectrum_contributions)    
    
        # --- 8) Run contrast---
        with Pool(processes=cpu_count()-1) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            results = list(tqdm(pool.imap(process_contrast, [(idx, planet_table[idx]) for idx in range(len(planet_table))]), total=len(planet_table), desc="Multiprocessing"))
            for (idx, mag_s, mag_p_thermal, mag_p_reflected, mag_p_total, separation, contrast_5sigma) in results:
                if idx==0:
                    planet_table["contrast_5sigma"] = np.full((len(planet_table), len(separation)), np.nan)
                planet_table[idx][f"StarINSTRUmag({instru})"]                      = mag_s
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal)"]           = mag_p_thermal
                planet_table[idx][f"PlanetINSTRUmag({instru})(reflected)"]         = mag_p_reflected
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal+reflected)"] = mag_p_total
                planet_table[idx]["contrast_5sigma"]                               = contrast_5sigma
    
        print(f"\n Calculating SNR took {(time.time()-time1)/60:.1f} mn")
        planet_table.write(filename, format='ascii.ecsv', overwrite=True)
        fits.writeto(filename.replace(".ecsv", "_separation.fits"), separation, overwrite=True)
        print(f"\nSaving table: {filename}")

    return separation, planet_table



def yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", spectrum_contributions="reflected", force_table_calc=False, exposure_time=10*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3):
    # --- Archive table of known exoplanets
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    
    # --- Ranges "Earth-like" 
    R_min, R_max     = 0, 2     # en R_earth
    M_min, M_max     = 0, 10    # en M_earth
    Teq_min, Teq_max = 0, 500   # en K
    
    def col(name):
        return planet_table[name].value if name in planet_table.colnames else None
    
    R    = col("PlanetRadius")        # R_earth
    M    = col("PlanetMass")          # M_earth
    Teq  = col("PlanetTeq")           # K
    a_AU = col("SMA")                 # AU
    n    = len(planet_table)
    mask_earth = np.ones(n, dtype=bool)
    print(f"Filtering Earth-Like planets")
    
    # Rayon & masse & Teq
    if R is not None:
        before = int(mask_earth.sum())
        mask_earth &= np.isfinite(R) & (R >= R_min) & (R <= R_max)
        after = int(mask_earth.sum())
        print(f" After radius filtering:      {after} / {n} (-{before - after})")
    if M is not None:
        before = int(mask_earth.sum())
        mask_earth &= np.isfinite(M) & (M >= M_min) & (M <= M_max)
        after = int(mask_earth.sum())
        print(f" After mass filtering:        {after} / {n} (-{before - after})")
    if Teq is not None:
        before = int(mask_earth.sum())
        mask_earth &= np.isfinite(Teq) & (Teq >= Teq_min) & (Teq <= Teq_max)
        after = int(mask_earth.sum())
        print(f" After temperature filtering: {after} / {n} (-{before - after})")
        
    planet_table = planet_table[mask_earth]
    
    # ---Retrieving tables
    separation_HARMONI, planet_table_HARMONI = get_planet_table_contrast(instru="HARMONI", planet_table=planet_table, exposure_time=exposure_time, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, Rc=Rc, filter_type="gaussian", systematic=False, PCA=False, N_PCA=20, force_table_calc=force_table_calc, sep_max=sep_max)
    separation_ANDES, planet_table_ANDES     = get_planet_table_contrast(instru="ANDES",   planet_table=planet_table, exposure_time=exposure_time, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, Rc=Rc, filter_type="gaussian", systematic=False, PCA=False, N_PCA=20, force_table_calc=force_table_calc, sep_max=sep_max)
    
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
    planet_magnitude_total     = planet_table_HARMONI["PlanetINSTRUmag(HARMONI)(thermal+reflected)"].value # thermal + reflected contribution
    star_magnitude             = planet_table_HARMONI["StarINSTRUmag(HARMONI)"].value
    contrast_thermal_HARMONI   = 10**(-(planet_magnitude_thermal-star_magnitude)/2.5)
    contrast_reflected_HARMONI = 10**(-(planet_magnitude_reflected-star_magnitude)/2.5)
    
    # ANDES
    planet_magnitude_thermal   = planet_table_ANDES["PlanetINSTRUmag(ANDES)(thermal)"].value           # thermal contribution
    planet_magnitude_reflected = planet_table_ANDES["PlanetINSTRUmag(ANDES)(reflected)"].value         # reflected contribution
    planet_magnitude_total     = planet_table_ANDES["PlanetINSTRUmag(ANDES)(thermal+reflected)"].value # thermal + reflected contribution
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
    
    from math import erf
    from matplotlib.patches import Patch
    
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
    fig = plt.figure(figsize=(13.5, 6.5), dpi=300)
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
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=z0-1)
    
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
    plt.show()
    
    
    
    
    
    
    
    #######################################################################
    # V2
    #####################################################################
    
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch  # si tu utilises ailleurs
    
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
    
    # ---------------------- Ta taille de points ----------------------
    pm_valid  = planet_mass[np.isfinite(planet_mass) & (planet_mass > 0)]
    mass_minG = pm_valid.min()
    mass_maxG = pm_valid.max()
    s         = mass_to_size(planet_mass, s0=s0, ds=ds,
                             mass_min=mass_minG, mass_max=mass_maxG)
    
    # ---------------------- Figure / Axes ----------------------
    fig = plt.figure(figsize=(13.5, 6.5), dpi=300)
    ax1 = plt.gca()
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(np.nanmin(separation)/2, sep_max)
    ax1.set_ylim(1e-9, 1e-5)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.minorticks_on()
    ax1.set_xlabel("Separation [mas]", fontsize=16)
    ax1.set_ylabel("Contrast", fontsize=16)
    
    # Grille sous les contours
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0.2)
    
    # ---------------------- Probabilité de détection ----------------------
    ymin, ymax = ax1.get_ylim()
    y_grid = np.logspace(np.log10(ymin), np.log10(ymax), 400)
    
    # Champs P_det
    P_H = pdet_grid(separation_HARMONI, contrast_5sigma_HARMONI, y_grid)
    P_A = pdet_grid(separation_ANDES,   contrast_5sigma_ANDES,   y_grid)
    
    # Contours
    CS_H = draw_prob_contours(ax1, separation_HARMONI, P_H, y_grid,
                              colors_instru["HARMONI"], levels=levels, lw=lw, z=0.4)
    CS_A = draw_prob_contours(ax1, separation_ANDES,   P_A, y_grid,
                              colors_instru["ANDES"],   levels=levels, lw=lw, z=0.5)
    
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
    leg_planets = ax1.legend(fontsize=14, loc="upper left", frameon=True,
                             edgecolor="gray", facecolor="white",
                             title="Planet-light regime", title_fontsize=16)
    ax1.add_artist(leg_planets)
    
    leg_inst = ax1.legend(handles=[
            Line2D([0],[0], color=colors_instru["HARMONI"], lw=lw, ls='-',  label="HARMONI"),
            Line2D([0],[0], color=colors_instru["ANDES"],   lw=lw, ls='-',  label="ANDES")
        ],
        loc="lower left", fontsize=14, frameon=True, edgecolor="gray",
        facecolor="white", title="Detection probability", title_fontsize=16)
    ax1.add_artist(leg_inst)
    
    for lg in (leg_planets, leg_inst):
        lg.set_zorder(100); lg.get_frame().set_alpha(1.0)
    
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
    plt.show()

    






