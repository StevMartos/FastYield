# import FastYield modules
from src.config import R0_max, LMIN, LMAX, bands, instrus, instrus_with_systematics, colors_instru, simulated_path, archive_path, thermal_models, reflected_models, ignore_reflected_thresh_um, SNR_thresh, planet_types, planet_types_reduced, m_u, kB, G, vesc_earth, sim_data_path
from src.get_specs import get_config_data, get_band_lims, get_wa, get_R_instru
from src.utils import faded
from src.spectrum import load_vega_spectrum, get_mag, get_spectrum_contribution_name_model, get_thermal_reflected_spectrum, get_counts_from_density, get_wave_K
from src.FastCurves import FastCurves

# import astropy modules
from astropy import constants as const
from astropy.table import QTable, Column
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, SkyCoord

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import RegularGridInterpolator

# import other modules
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import corner

# For fits warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter("ignore", category=VerifyWarning)
warnings.filterwarnings("ignore", message="Header block contains null bytes*")



#######################################################################################################################
#################################################### Utils function: #################################################
#######################################################################################################################

def mass_to_size(mass, s0, ds, mass_min=None, mass_max=None):
    if mass_min is None:
        mass_min = np.nanmin(mass[np.isfinite(mass) & (mass > 0)])
    if mass_max is None:
        mass_max = np.nanmax(mass[np.isfinite(mass) & (mass > 0)])
    t = (np.log10(np.clip(mass, mass_min, mass_max)) - np.log10(mass_min)) / (np.log10(mass_max) - np.log10(mass_min))
    t = np.clip(t, 0, 1)
    return s0 + ds * t


def get_filename_table(table, instru, apodizer, strehl, coronagraph, systematics, PCA, name_model):
    coronagraph_str = "_"+str(coronagraph) if coronagraph is not None else ""
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    filename = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"
    return filename



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



def get_mask_planet_type(planet_table, planet_type):
    """Get a mask with the 'planet_type' planets"""
    planets_type = np.array(planet_table["PlanetType"], dtype=str)
    mask_ptype   = (np.char.find(np.char.lower(planets_type), planet_type.lower()) >= 0) # substring match
    return mask_ptype



def get_mask_earth(planet_table):
    """Get a mask with the Earth-like planets"""

    # --- Ranges "Earth-like" 
    R_min,   R_max     = 0, 2   # [R_earth]
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
            q = q*(-1) 
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
    _assign("StarVrot",          "stellar_rotational_velocity", u.km/u.s)
    _assign("StarAge",            "stellar_age",                 u.Gyr)
    _assign("StarTeff",           "stellar_eff_temp",            u.K)
    _assign("StarMass",           "stellar_mass",                u.Msun)
    _assign("StarRadius",         "stellar_radius",              u.Rsun)
    _assign("StarLogg",           "stellar_surface_gravity",     u.dex(u.cm/(u.s**2)))
    _assign("StarKmag",           "k_mag",                       u.dimensionless_unscaled)
    _assign("StarFeH",            "stellar_metallicity",         u.dimensionless_unscaled)
    
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
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "CD-35 2722 b"]                  = 10.37        # https://arxiv.org/pdf/1201.3537

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
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "HD 100546 b"]                   = 932 * u.K    # https://home.strw.leidenuniv.nl/~kenworthy/papers/2015ApJ...807...64Q.pdf
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "HD 100546 b"]                   = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "USco CTIO 108 b"]               = 15.11        # https://arxiv.org/pdf/0712.3482 (table 2)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "USco CTIO 108 b"]               = 2350 * u.K   # https://arxiv.org/pdf/0712.3482 (table 2)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "USco CTIO 108 b"]               = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J22362452+4751425 b"]     = 17.34        # https://arxiv.org/pdf/1611.00364 (table 4)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "2MASS J22362452+4751425 b"]     = 1070 * u.K   # https://arxiv.org/pdf/1611.00364
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "2MASS J22362452+4751425 b"]     = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "TYC 8998-760-1 b"]              = 14.70        # https://home.strw.leidenuniv.nl/~kenworthy/papers/2020MNRAS.492..431B.pdf (table 4)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "TYC 8998-760-1 b"]              = 1727 * u.K   # https://home.strw.leidenuniv.nl/~kenworthy/papers/2020MNRAS.492..431B.pdf (section 4.2.2)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "TYC 8998-760-1 b"]              = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISPIT 2 b"]                    = 16.84        # https://arxiv.org/pdf/2508.19046 (table 3): get_mag_from_mag(T=1500, lg=3.67, model="BT-Settl", mag_input=15.3, band0_input="L", band0_output="K")
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "WISPIT 2 b"]                    = 1500 * u.K   # https://arxiv.org/pdf/2508.19046
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "WISPIT 2 b"]                    = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "PZ Tel b"]                      = 6.366+5.55   # https://arxiv.org/pdf/1404.2870 (table 2)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "PZ Tel b"]                      = 2500 * u.K   # https://arxiv.org/pdf/1404.2870
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "PZ Tel b"]                      = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "UCAC3 113-933 b"]               = 18.5         # https://arxiv.org/pdf/2403.04592 (table 2): (get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=19.94, band0_input="J", band0_output="K") + get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=19.63, band0_input="H", band0_output="K") + get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=17.91, band0_input="W1", band0_output="K") + get_mag_from_mag(T=1150, lg=4.65, model="BT-Settl", mag_input=15.56, band0_input="W2", band0_output="K")) / 4
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "UCAC3 113-933 b"]               = 1150 * u.K   # https://arxiv.org/pdf/2403.04592 (table 1)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "UCAC3 113-933 b"]               = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 39017 b"]                   = 17.02        # https://arxiv.org/pdf/2403.04000 (section 3)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "HIP 39017 b"]                   = 1300 * u.K   # https://arxiv.org/pdf/2403.04000 (section 5)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "HIP 39017 b"]                   = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J04414489+2301513 b"]     = 14.94        # https://arxiv.org/pdf/1004.0539 (section 4.2)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "2MASS J04414489+2301513 b"]     = 1800 * u.K   # https://arxiv.org/pdf/1509.01658
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "2MASS J04414489+2301513 b"]     = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "AB Aur b"]                      = 14.9         # https://arxiv.org/pdf/2406.00107 (section 2): get_mag_from_mag(T=2200, lg=4.25, model="BT-Settl", mag_input=15.436, band0_input="PaB", band0_output="K")
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "AB Aur b"]                      = 2200 * u.K   # https://www.nature.com/articles/s41550-022-01634-x
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "AB Aur b"]                      = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J0249-0557 c"]            = 14.78        # https://www.pure.ed.ac.uk/ws/portalfiles/portal/76315059/Dupuy_2018_AJ_156_57.pdf (table 4)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "2MASS J0249-0557 c"]            = 1700 * u.K   # https://www.pure.ed.ac.uk/ws/portalfiles/portal/76315059/Dupuy_2018_AJ_156_57.pdf
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "2MASS J0249-0557 c"]            = "inject_known_values"

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HD 169142 b"]                   = 6.41+9.72    # https://www.aanda.org/articles/aa/pdf/2019/03/aa34760-18.pdf (table 5)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "HD 169142 b"]                   = 1260 * u.K   # https://www.aanda.org/articles/aa/pdf/2019/03/aa34760-18.pdf
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "HD 169142 b"]                   = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "CHXR 73 b"]                     = 15.5         # https://arxiv.org/pdf/astro-ph/0609187
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "CHXR 73 b"]                     = 2600 * u.K   # https://arxiv.org/pdf/0809.2812 (section 3.5)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "CHXR 73 b"]                     = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HN Peg b"]                      = 15.12        # https://lweb.cfa.harvard.edu/~mmarengo/pub/2007ApJ...654..570L.pdf (table 3)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "HN Peg b"]                      = 1130 * u.K   # https://lweb.cfa.harvard.edu/~mmarengo/pub/2007ApJ...654..570L.pdf (section 3.3.3)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "HN Peg b"]                      = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "Ross 458 c"]                    = 16.90        # https://arxiv.org/pdf/1002.2637 (table 3)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "Ross 458 c"]                    = 695 * u.K    # https://arxiv.org/pdf/1103.1617
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "Ross 458 c"]                    = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "AB Pic b"]                      = 14.14        # https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (table 1)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "AB Pic b"]                      = 1700 * u.K   # https://arxiv.org/pdf/2211.01474
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "AB Pic b"]                      = "inject_known_values"
 
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J02192210-3925225 b"]     = 13.82        # https://arxiv.org/pdf/1505.01747 (table 2)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "2MASS J02192210-3925225 b"]     = 1700 * u.K   # https://arxiv.org/pdf/1505.01747 (section 4.5)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "2MASS J02192210-3925225 b"]     = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "UCAC4 328-061594 b"]            = 18.37        # https://arxiv.org/pdf/2403.04592 (table 2): (get_mag_from_mag(T=1000, lg=4.65, model="BT-Settl", mag_input=18.07, band0_input="W1", band0_output="K") + get_mag_from_mag(T=1000, lg=4.65, model="BT-Settl", mag_input=15.82, band0_input="W2", band0_output="K") ) / 2
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "UCAC4 328-061594 b"]            = 1000 * u.K   # https://arxiv.org/pdf/2403.04592 (table 1)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "UCAC4 328-061594 b"]            = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "UCAC4 328-061594 b"]            = 5500 * u.K   # https://www.exoplanetkyoto.org/exohtml/UCAC4_328-061594.html

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "GJ 900 b"]                      = 21.69        # https://arxiv.org/pdf/2403.04592 (table 2): (get_mag_from_mag(T=500, lg=4.35, model="BT-Settl", mag_input=18.83, band0_input="W1", band0_output="K") + get_mag_from_mag(T=500, lg=4.35, model="BT-Settl", mag_input=15.90, band0_input="W2", band0_output="K") ) / 2
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "GJ 900 b"]                      = 500 * u.K    # https://arxiv.org/pdf/2403.04592 (table 1)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "GJ 900 b"]                      = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 99770 b"]                   = 12.460       # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (section 4.1)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "HIP 99770 b"]                   = 1300 * u.K   # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (table 4)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "HIP 99770 b"]                   = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "2MASS J01033563-5515561 AB b"]  = 13.690       # https://www.aanda.org/articles/aa/pdf/2025/09/aa54894-25.pdf (section A.3)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "2MASS J01033563-5515561 AB b"]  = 1731 * u.K   # https://www.aanda.org/articles/aa/pdf/2025/09/aa54894-25.pdf (table A.2)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "2MASS J01033563-5515561 AB b"]  = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "2MASS J01033563-5515561 AB b"]  = 3000 * u.K   # https://www.openexoplanetcatalogue.com/planet/2MASS%20J01033563-5515561%20A%20b/

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "ROXs 12 b"]                     = 14.32        # https://arxiv.org/pdf/1311.7664 (table 4)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "ROXs 12 b"]                     = 2600 * u.K   # https://arxiv.org/pdf/1311.7664
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "ROXs 12 b"]                     = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISPIT 1 b"]                    = 17.78        # https://arxiv.org/pdf/2508.18456 (table 7)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "WISPIT 1 b"]                    = 1470 * u.K   # https://arxiv.org/pdf/2508.18456
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "WISPIT 1 b"]                    = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "WISPIT 1 c"]                    = 20.49        # https://arxiv.org/pdf/2508.18456 (table 7)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "WISPIT 1 c"]                    = 1030 * u.K   # https://www.aanda.org/articles/aa/pdf/2025/08/aa54766-25.pdf (table 4)
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "WISPIT 1 c"]                    = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "HIP 54515 b"]                    = 15.08        # https://arxiv.org/pdf/2512.02159
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "HIP 54515 b"]                    = 2348 * u.K   # https://arxiv.org/pdf/2512.02159
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "HIP 54515 b"]                    = "inject_known_values"
    
    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "mu2 Sco b"]                     = 16.01        # https://pure-oai.bham.ac.uk/ws/portalfiles/portal/173547674/aa43675_22.pdf (table 7)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "mu2 Sco b"]                     = 2050 * u.K   # https://pure-oai.bham.ac.uk/ws/portalfiles/portal/173547674/aa43675_22.pdf
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "mu2 Sco b"]                     = "inject_known_values"
    planet_table["Distance"][planet_table["PlanetName"]                      == "mu2 Sco b"]                     = 145 * u.pc   # https://en.wikipedia.org/wiki/Mu2_Scorpii

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "b Cen AB b"]                    = 16.37        # https://www.eso.org/public/archives/releases/sciencepapers/eso2118/eso2118a.pdf (table 2)
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "b Cen AB b"]                    = 1600 * u.K   # get_evolutionary_model(planet_table[planet_table["PlanetName"] == "b Cen AB b"])
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "b Cen AB b"]                    = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "b Cen AB b"]                    = 18445 * u.K  # https://en.wikipedia.org/wiki/HD_129116

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "SR 12 AB c"]                    = 15.05        # https://www.nao.ac.jp/contents/about-naoj/reports/annual-report/en/2011/e_web_045.pdf: get_mag_from_mag(T=2400, lg=4.43, model="BT-Settl", mag_input=16.0, band0_input="J", band0_output="K")
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "SR 12 AB c"]                    = 2600 * u.K   # https://academic.oup.com/mnras/article/475/3/2994/4781312
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "SR 12 AB c"]                    = "inject_known_values"
    planet_table["SMA"][planet_table["PlanetName"]                           == "SR 12 AB c"]                    = 1083 * u.AU  # https://www.exoplanetkyoto.org/exohtml/SR_12_AB_c.html

    planet_table["PlanetKmag(thermal+reflected)"][planet_table["PlanetName"] == "Luhman 16 b"]                   = 9.73         # https://www.eso.org/public/archives/releases/sciencepapers/eso1404/eso1404a.pdf
    planet_table["PlanetTeff"][planet_table["PlanetName"]                    == "Luhman 16 b"]                   = 1320 * u.K   # https://arxiv.org/pdf/1506.08848 / https://simbad.cds.unistra.fr/simbad/sim-id?Ident=NAME+WISE+J1049-5319B
    planet_table["PlanetTeffRef"][planet_table["PlanetName"]                 == "Luhman 16 b"]                   = "inject_known_values"
    planet_table["StarTeff"][planet_table["PlanetName"]                      == "Luhman 16 b"]                   = 1310 * u.K   # https://arxiv.org/pdf/1406.1518
    planet_table["Distance"][planet_table["PlanetName"]                      == "Luhman 16 b"]                   = 2.0 * u.pc   # https://en.wikipedia.org/wiki/Luhman_16
    
    return planet_table



def print_missing_known_values(planet_table, exclude_refs=("MICHEL__AMP__MUGRAUER_2024",)):
    """
    Print which directly-imaged planets are still missing K-band magnitude
    and/or Teff after injecting your hand-curated “known” values.
    """
    planet_table = planet_table[planet_table["DiscoveryMethod"]=="Imaging"].copy()
    planet_table = inject_known_values(planet_table)
    
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
    EVOL_GRID_NPZ = f"{sim_data_path}/Archive_table/sonora_bobcat_evol_grid.npz"
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



def get_planet_type(planet):
    """
    Classify a single planet row using the global 'planet_types' rules.

    Expects
    -------
    planet : astropy.table.Row (from your main table)
        Must provide 'PlanetMass', 'PlanetRadius', 'PlanetTeff' as quantities
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
    teff   = planet["PlanetTeff"].value
    
    # If values are missing, return "Unidentified"
    if not np.isfinite(mass):
        return "Unidentified - missing mass"
    if not np.isfinite(radius):
        return "Unidentified - missing radius"
    if not np.isfinite(teff):
        return "Unidentified - missing temperature"
    
    # Loop through planet types to find a match
    for ptype, criteria in planet_types.items():
        mass_match   = (mass   >= criteria["mass_min"])   and (mass   <= criteria["mass_max"])
        radius_match = (radius >= criteria["radius_min"]) and (radius <= criteria["radius_max"])
        teff_match   = (teff   >= criteria["teff_min"])   and (teff   <= criteria["teff_max"])
        if mass_match and radius_match and teff_match:
            return ptype  # Return the first matching type
    
    return "Unidentified - outside class"



def find_matching_planets(criteria, planet_table, mode, selected_planets=None, Nmax=None):
    """
    Filter a pandas DataFrame of planets against numeric box constraints.

    Parameters
    ----------
    criteria : dict
        Keys: 'mass_min', 'mass_max', 'radius_min', 'radius_max', 'teff_min', 'teff_max'
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
        f"PlanetTeff >= {criteria['teff_min']}",
        f"PlanetTeff <= {criteria['teff_max']}"
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
    
    import pandas as pd
    
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
             "Temperature [K]": int(round(planet["PlanetTeff"])), 
             f"SNR (in {int(round(exposure_time/60))} h)": round(planet["SNR"], 1)}
            for ptype, planets in matching_planets.items() for planet in planets])
        table = ax.table(cellText=matching_planets_df.values, colLabels=matching_planets_df.columns, cellLoc='center', loc='center')
    
    elif mode == 'multi':    
        conditions_df = pd.DataFrame([
            {"Type":            ptype,
             "Mass [M⊕]":      _format_range(criteria, "mass"),
             "Radius [R⊕]":    _format_range(criteria, "radius"),
             "Temperature [K]": _format_range(criteria, "teff"),
             "Number of Planets\nconsidered": len(matching_planets[ptype]),
             f"Number of Planets\ndetected (in {int(round(exposure_time/60))} h)": sum(planet["SNR"] > SNR_thresh for planet in matching_planets[ptype])}
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



def get_SNR_from_table(planet_table, exposure_time, band):
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
    DIT        = np.asarray(planet_table[f'DIT_{band}'],        dtype=float) # mn/DIT
    signal     = np.asarray(planet_table[f'signal_{band}'],     dtype=float) # signal/DIT
    sigma_fund = np.asarray(planet_table[f'sigma_fund_{band}'], dtype=float) # noise/DIT
    sigma_syst = np.asarray(planet_table[f'sigma_syst_{band}'], dtype=float) # noise/DIT
    N_DIT      = exposure_time / DIT
    S          = N_DIT * signal
    N          = np.sqrt(N_DIT*sigma_fund**2 + N_DIT**2*sigma_syst**2)
    SNR        = S / N
    return SNR



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
    SNR     = np.clip(SNR, SNR_min, SNR_max)
    t       = (np.log10(SNR)) / (np.log10(SNR_max))
    t[t<0]  = 0
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



#######################################################################################################################
#################################################### Magnitudes computations: #########################################
#######################################################################################################################

def process_magnitudes(idx):
    """
    Worker that computes all magnitudes for a single planet.

    Parameters
    ----------
    idx: int
        Row index in the original table.

    Returns
    -------
    idx : int
        Row index in the original table.
    mags : dict[str, float or Quantity]
        Magnitudes keyed by the final column names to be written.
    """
    
    planet_table    = _MAG_CTX["planet_table"]
    wave_model      = _MAG_CTX["wave_model"]
    wave_K          = _MAG_CTX["wave_K"]
    counts_vega     = _MAG_CTX["counts_vega"]
    counts_vega_K   = _MAG_CTX["counts_vega_K"]
    masks           = _MAG_CTX["masks"]
    use_reflected   = _MAG_CTX["use_reflected"]
    
    # Planet row
    planet = planet_table[idx]
    
    # Computing models on wave_model    
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model="auto", reflected_model="auto", instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_im_mag=True)
    
    # Computing the magnitudes
    mags = {}
    
    # Instrus mags
    for instru in instrus:
        mask_instru = masks[instru]
        mags[f"StarINSTRUmag({instru})"]                      = get_mag(wave=wave_model[mask_instru], density_obs=star_spectrum.flux[mask_instru],   density_vega=None, counts_vega=counts_vega[instru])
        mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_spectrum.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
        mags[f"PlanetINSTRUmag({instru})(thermal)"]           = get_mag(wave=wave_model[mask_instru], density_obs=planet_thermal.flux[mask_instru],  density_vega=None, counts_vega=counts_vega[instru])
        if use_reflected[instru]:
            mags[f"PlanetINSTRUmag({instru})(reflected)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_reflected.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
   
    # Bands mags
    for band in bands:
        mask_band = masks[band]
        if band != "K":
            mags[f'Star{band}mag']                  = get_mag(wave=wave_model[mask_band], density_obs=star_spectrum.flux[mask_band],   density_vega=None, counts_vega=counts_vega[band])
        mags[f"Planet{band}mag(thermal+reflected)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_spectrum.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])
        mags[f"Planet{band}mag(thermal)"]           = get_mag(wave=wave_model[mask_band], density_obs=planet_thermal.flux[mask_band],  density_vega=None, counts_vega=counts_vega[band])
        if use_reflected[band]:
            mags[f"Planet{band}mag(reflected)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_reflected.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])
    
    return idx, mags



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
        
    # --- 1) Wavelength grids and Vega spectrum ---
    
    # Model bandwidth
    wave_model = np.arange(LMIN, LMAX, 1e-4)
    
    # K-band for photometry
    wave_K = get_wave_K()

    # Vega spectrum on K-band and model-band [J/s/m2/µm]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K,     renorm=False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_model, renorm=False)
    
    # --- 2) Bandpass masks, Vega flux on bands and “use_reflected” flags ---
    counts_vega_K = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
    counts_vega   = {}
    masks         = {}
    use_reflected = {}
    for instru in instrus:
        lmin, lmax            = get_band_lims(instru)
        masks[instru]         = (wave_model >= lmin) & (wave_model <= lmax)
        use_reflected[instru] = (lmin < ignore_reflected_thresh_um)
        counts_vega[instru]   = get_counts_from_density(wave=wave_model[masks[instru]], density=vega_spectrum.flux[masks[instru]])
    for band in bands:
        lmin, lmax          = get_band_lims(band)
        masks[band]         = (wave_model >= lmin) & (wave_model <= lmax)
        use_reflected[band] = (lmin < ignore_reflected_thresh_um)
        counts_vega[band]   = get_counts_from_density(wave=wave_model[masks[band]], density=vega_spectrum.flux[masks[band]])
    
    # --- 3) Init global context for workers ---
    global _MAG_CTX
    _MAG_CTX = dict(planet_table=planet_table, wave_model=wave_model, wave_K=wave_K, counts_vega=counts_vega, counts_vega_K=counts_vega_K, masks=masks, use_reflected=use_reflected)
    
    # Magnitude estimations
    print()
    with Pool(processes=cpu_count()//2) as pool: 
        for (idx, mags) in tqdm(pool.imap(process_magnitudes, [(idx) for idx in range(len(planet_table))]), total=len(planet_table), desc="(6) Estimating all magnitudes..."):
            for instru in instrus:
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

def get_archive_table(seed=None):
    """
    Build the working exoplanet catalog used by FastYield from the NASA Exoplanet Archive.
    
    This routine queries a curated subset of the 'pscomppars' table through the
    NASA Exoplanet Archive TAP service, converts archive-reported units to
    'astropy.units', renames the columns to the local FastYield schema, fills or
    repairs missing values using external sources and simple physical prescriptions,
    derives additional quantities required by the pipeline, filters the sample to
    planets usable for signal-to-noise calculations, computes synthetic magnitudes,
    and writes both an intermediate and a final ECSV catalog to disk.
    
    The returned table is the final FastYield working catalog, with physical
    quantities stored as 'astropy.units.Quantity' objects whenever applicable.
    
    Parameters
    ----------
    seed : int or None, optional
        Random seed passed to 'numpy.random.default_rng' for reproducible draws
        of missing stellar and planetary properties inferred from adopted priors
        (e.g., radial velocities, projected rotation velocities, argument of
        periastron). If 'None', the random generator is left unseeded.
    
    Returns
    -------
    planet_table : astropy.table.QTable
        Final filtered and augmented catalog used by FastYield. Numerical columns
        are stored as non-masked 'Quantity' objects when units are relevant, while
        string-like metadata remain plain Astropy columns.
    
    Data source
    -----------
    NASA Exoplanet Archive TAP service:
    https://exoplanetarchive.ipac.caltech.edu/TAP
    
    Queried table:
        'pscomppars'
    
    Side Effects
    ------------
    This function writes two ECSV files under the global 'archive_path' directory:
    
    - 'Archive_Pull_Raw.ecsv'
        Raw archive pull after unit normalization and column renaming.
    - 'Archive_Pull_For_FastYield.ecsv'
        Final FastYield catalog after augmentation, filtering, and magnitude
        calculations.
    
    Workflow Summary
    ----------------
    The main processing steps are:
    
    1. Query a selected subset of the NASA Exoplanet Archive 'pscomppars' table.
    2. Normalize archive-reported units to Astropy units.
    3. Rename archive columns to the internal FastYield naming convention.
    4. Convert float-like columns to non-masked 'Quantity' objects, with missing
       values stored as 'NaN'.
    5. Save a raw snapshot of the pull and reload it through the local table loader.
    6. Inject additional values from DACE and other known literature fixes.
    7. Initialize placeholder magnitude columns for all configured instruments and bands.
    8. Build discovery-method masks (Imaging, Radial Velocity, Transit, Other).
    9. Derive planetary and stellar surface gravities, and infer missing stellar
       luminosities, radii, and 'log g' values when possible.
    10. Adopt simplified geometric and orbital assumptions to derive projected
        angular separations, Lambert phase factors, and line-of-sight velocities.
    11. Fill missing stellar and planetary projected rotational velocities using
        simple Gaussian priors conditioned on stellar effective temperature or
        planetary mass.
    12. Estimate missing planetary effective temperatures from equilibrium heating
        and, when available, evolutionary-model predictions.
    13. Fill missing planetary radii from the evolutionary model, with a median-value
        fallback for directly imaged planets.
    14. Filter the sample to planets with sufficient information for FastYield
        signal-to-noise calculations.
    15. Classify planets by type.
    16. Flag rocky planets as atmosphere-retaining or atmosphere-free using simple
        escape and irradiation criteria.
    17. Compute a telluric-equivalent airmass proxy used to scale the Earth-like
        reflected-light albedo template.
    18. Compute synthetic magnitudes for all configured instruments and bands.
    19. Save the final catalog and generate a few diagnostic plots.
    
    Important Derived Quantities
    ----------------------------
    In addition to archive quantities, the returned table includes several derived
    columns used by FastYield, including for example:
    
    - 'PlanetLogg'
    - 'Phase'
    - 'AngSep'
    - 'alpha'
    - 'g_alpha'
    - 'DeltaRadialVelocity'
    - 'PlanetRadialVelocity'
    - 'PlanetVrot'
    - 'PlanetType'
    - 'HasAtmosphere'
    - 'TelluricEquivalentAirmass'
    
    It also includes synthetic magnitude columns for the configured bands and
    instruments, separately tracking thermal, reflected, and combined contributions.
    
    Physical Assumptions
    --------------------
    Several simplified assumptions are adopted to build a homogeneous working
    catalog:
    
    - Missing orbital phases are set to quadrature ('phase = pi/2').
    - Missing inclinations default to '90 deg'.
    - Missing eccentricities default to zero.
    - Missing arguments of periastron are drawn uniformly over '[0, 360 deg]'.
    - Missing stellar and planetary projected rotation velocities are drawn from
      simple empirical Gaussian priors.
    - Missing stellar luminosities inferred from 'V' magnitude and distance do not
      include extinction corrections.
    - Missing planetary temperatures are estimated from equilibrium heating
      ('A_B = 0.3', full '4pi' redistribution) and, when possible, an evolutionary
      model.
    - Rocky planets are flagged as atmosphere-free when they fail a simple Jeans
      escape criterion and/or a coarse irradiation-based atmospheric-loss criterion.
    - The telluric-equivalent airmass is a heuristic reflected-light proxy used to
      scale an Earth-like telluric albedo template; it is not a true observational
      airmass.
    
    Caveats
    -------
    This function is designed to build a practical catalog for yield and
    signal-to-noise calculations rather than a rigorously homogeneous physical
    population model. Several inferred quantities rely on simplified assumptions,
    empirical priors, or approximate fallback prescriptions. The resulting catalog
    is therefore suitable for FastYield performance studies, but some derived values
    should not be interpreted as precise planetary or stellar measurements.
    """
    
    import pyvo as vo
    
    time1 = time.time()
    
    rng = np.random.default_rng(seed=seed) # Fixing the seed (if required, i.e. seed is not None)

    # -----------------------------------------------------------------------------
    # 1) Pull PSCOMPPARS subset and normalize units
    # -----------------------------------------------------------------------------
    COLS_TO_PULL = (
                    "pl_name, hostname, "
                    "pl_orbper, pl_orbper_reflink, pl_orbsmax, pl_orbsmax_reflink, pl_orbeccen, pl_orbeccen_reflink, pl_orbincl, pl_orbincl_reflink, pl_orblper, pl_orbtper_reflink, "
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
                 "Period", "PeriodRef", "SMA", "SMARef", "Ecc", "EccRef", "Inc", "IncRef", "ArgPeri", "ArgPeriRef",
                 "PlanetMass", "+DeltaPlanetMass", "-DeltaPlanetMass", "PlanetMassRef",
                 "PlanetRadius", "+DeltaPlanetRadius", "-DeltaPlanetRadius", "PlanetRadiusRef",
                 "PlanetTeff", "+DeltaPlanetTeff", "-DeltaPlanetTeff", "PlanetTeffRef",
                 "RA", "Dec", "Distance", "+DeltaDistance", "-DeltaDistance",
                 "StarSpT", "StarMass", "StarTeff", "StarRadius", "StarLogg", "StarLum", "StarAge",
                 "StarVrot", "StarRadialVelocity", "StarFeH",
                 "StarKmag", "StarVmag",
                 "DiscoveryMethod", "DiscoveryRef"
                ]
    
    print("\nRetrieving NASA archive table from https://exoplanetarchive.ipac.caltech.edu/TAP")
    svc          = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    table        = svc.search(f"SELECT {COLS_TO_PULL} FROM pscomppars").to_table()
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
        if col.dtype.kind in ("i", "O", "U", "S", "b"): # Integer / str
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
    try:
        planet_table = inject_dace_values(planet_table)
    except Exception as e:
        print(f" Could not inject DACE values in the table: {e}")
    
    # Creating magnitudes columns (in order to inject known K-band magnitudes)
    for instru in instrus:
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
    # # TODO: Deleting directly-imaged planets with missing known magnitudes ?
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
    abs_V = planet_table["StarVmag"].value - 5 * np.log10((planet_table["Distance"] / (10 * u.pc)).value)
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
    # 8) Assumptions and computations for geometry and kinematics
    # -----------------------------------------------------------------------------    
    
    # Phase [rad] (Photometric/geometric convention: phi=0 => inferior conjunction, phi=pi/2 => quadrature 'redshift', phi=pi => superior conjunction and phi=3pi/2 => quadrature 'blueshift')
    phi0                  = np.pi / 2
    planet_table["Phase"] = u.Quantity(np.full(len(planet_table), phi0, dtype=float), u.rad, copy=False)
    
    # Inclination [°]
    i0                             = 90
    i_invalid                      = get_invalid_mask(planet_table["Inc"])
    planet_table["Inc"][i_invalid] = i0 * u.deg
    
    # Eccentricity [dimensionless]
    e0                             = 0.
    e_invalid                      = get_invalid_mask(planet_table["Ecc"])
    planet_table["Ecc"][e_invalid] = e0 * u.dimensionless_unscaled

    # Argument of periastron [°]    
    omega_invalid                          = get_invalid_mask(planet_table["ArgPeri"])
    planet_table["ArgPeri"][omega_invalid] = rng.uniform(0, 360, omega_invalid.sum()) * u.deg    

    # Retrieving usefull parameters
    a     = planet_table["SMA"]                                        # [AU]
    d     = planet_table["Distance"]                                   # [pc]
    phi   = planet_table["Phase"]                                      # [rad]
    i     = np.clip(planet_table["Inc"], 0*u.deg, 180*u.deg).to(u.rad) # [rad]
    e     = np.clip(planet_table["Ecc"], 0, 0.999)                     # dimensionless
    omega = planet_table["ArgPeri"].to(u.rad)                          # [rad]
    Ms    = planet_table["StarMass"]                                   # [M_sun]
    Mp    = planet_table["PlanetMass"]                                 # [M_earth]
    
    # Fillng missing period [d]
    P                                 = ( 2*np.pi*np.sqrt( a**3 / (const.G*(Ms+Mp)) ) ).to(u.d)
    P_invalid                         = get_invalid_mask(planet_table["Period"])
    planet_table["Period"][P_invalid] = P[P_invalid]
    
    # True anomaly [rad] (cos(nu+omega) = sin(phi) and sin(nu+omega) = -cos(phi))
    nu = phi - omega - np.pi/2 * u.rad
    
    # Instantaneous separation [AU]
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Computing projected angular separation
    planet_table['AngSep'] = ( r/d * np.sqrt( np.cos(nu+omega)**2 + np.sin(nu+omega)**2 * np.cos(i)**2 ) ).to(u.mas, equivalencies=u.dimensionless_angles())
    
    # Computing projected orbital radial velocity (=> Delta RV) [km/s]
    Kp                                  = ( np.sqrt( const.G*(Ms+Mp) / a ) * np.sin(i) / np.sqrt( 1 - e**2 ) ).to(u.km/u.s)
    planet_table['DeltaRadialVelocity'] = Kp * (np.cos(nu+omega) + e*np.cos(omega))
    
    print("\n(1) Hypothesis (mostly optimistic): Planets are assumed to be at their quadrature and:")
    print(f"                  => Phase   = {phi0:.2f} rad   (for all planets)")
    print(f"                  => Inc     = {i0:.0f}°        (for {len(planet_table[i_invalid])}/{len(planet_table)} missing values)")
    print(f"                  => Ecc     = {e0:.1f}        (for {len(planet_table[e_invalid])}/{len(planet_table)} missing values)")
    print(f"                  => ArgPeri = U(0, 360°) (for {len(planet_table[omega_invalid])}/{len(planet_table)} missing values)")
        
    # Lambert phase function: g(alpha)
    alpha                   = np.arccos(-np.sin(i) * np.cos(phi))
    planet_table["alpha"]   = alpha.value * u.rad
    planet_table["g_alpha"] = ( (np.sin(alpha) + (np.pi - alpha.value) * np.cos(alpha)) / np.pi ) * u.dimensionless_unscaled
    
    # Randomly draw radial velocities of the stars and Drv according to a normal distribution when they are missing
    srv_invalid                                     = get_invalid_mask(planet_table["StarRadialVelocity"])
    srv_mean                                        = np.nanmedian(planet_table["StarRadialVelocity"]).value
    srv_std                                         = np.nanstd(planet_table["StarRadialVelocity"]).value
    planet_table["StarRadialVelocity"][srv_invalid] = rng.normal(srv_mean, srv_std, srv_invalid.sum()) * u.km/u.s
    drv_invalid                                      = get_invalid_mask(planet_table["DeltaRadialVelocity"])
    drv_mean                                         = np.nanmedian(planet_table["DeltaRadialVelocity"]).value
    drv_std                                          = np.nanstd(planet_table["DeltaRadialVelocity"]).value
    planet_table["DeltaRadialVelocity"][drv_invalid] = rng.normal(drv_mean, drv_std, drv_invalid.sum()) * u.km/u.s
    
    # Computing planet radial velocities (by definiton)
    planet_table["PlanetRadialVelocity"] = planet_table["StarRadialVelocity"] + planet_table["DeltaRadialVelocity"]
    
    # -----------------------------------------------------------------------------
    # 9) v sin i priors for stars and planets (fills only where missing)
    # -----------------------------------------------------------------------------
    
    # Stellar v sin i
    svs_invalid = get_invalid_mask(planet_table["StarVrot"])
    Teff = planet_table["StarTeff"]
    
    m1 = svs_invalid & (Teff <= 3500 * u.K)
    m2 = svs_invalid & (3500 * u.K < Teff) & (Teff <= 6000 * u.K)
    m3 = svs_invalid & (6000 * u.K < Teff) & (Teff <= 7500 * u.K)
    m4 = svs_invalid & (7500 * u.K < Teff) & (Teff <= 10000 * u.K)
    m5 = svs_invalid & (10000 * u.K < Teff)
    
    planet_table["StarVrot"][m1] = rng.normal(1,   0.5, size=m1.sum()) * u.km / u.s
    planet_table["StarVrot"][m2] = rng.normal(3,   1.0, size=m2.sum()) * u.km / u.s
    planet_table["StarVrot"][m3] = rng.normal(10,  5.0, size=m3.sum()) * u.km / u.s
    planet_table["StarVrot"][m4] = rng.normal(120, 60., size=m4.sum()) * u.km / u.s
    planet_table["StarVrot"][m5] = rng.normal(200, 80., size=m5.sum()) * u.km / u.s
    
    planet_table["StarVrot"][planet_table["StarVrot"] < 0 * u.km / u.s] = 0 * u.km / u.s
    
    print(f"\nFilling star Vrot according to adopted priors for {svs_invalid.sum()}/{len(planet_table)} planets")
    
    # Planetary v sin i
    planet_table["PlanetVrot"] = np.zeros(len(planet_table)) * u.km / u.s
    Mp = planet_table["PlanetMass"]
    
    p1 = (Mp <= 5 * u.Mearth)
    p2 = (5 * u.Mearth < Mp) & (Mp <= 20 * u.Mearth)
    p3 = (20 * u.Mearth < Mp) & (Mp <= 100 * u.Mearth)
    p4 = (100 * u.Mearth < Mp) & (Mp <= 300 * u.Mearth)
    p5 = (300 * u.Mearth < Mp)
    
    planet_table["PlanetVrot"][p1] = rng.normal(0.5, 0.5, size=p1.sum()) * u.km / u.s
    planet_table["PlanetVrot"][p2] = rng.normal(2.0, 1.0, size=p2.sum()) * u.km / u.s
    planet_table["PlanetVrot"][p3] = rng.normal(3.0, 1.5, size=p3.sum()) * u.km / u.s
    planet_table["PlanetVrot"][p4] = rng.normal(12., 4.0, size=p4.sum()) * u.km / u.s
    planet_table["PlanetVrot"][p5] = rng.normal(20., 8.0, size=p5.sum()) * u.km / u.s
    
    planet_table["PlanetVrot"][planet_table["PlanetVrot"] < 0 * u.km / u.s] = 0 * u.km / u.s
    
    
    planet_table["sini"]        = np.sin(planet_table["Inc"].to(u.rad))
    planet_table["StarVsini"]   = planet_table["StarVrot"]   * planet_table["sini"]
    planet_table["PlanetVsini"] = planet_table["PlanetVrot"] * planet_table["sini"]

    # -----------------------------------------------------------------------------
    # 10) Estimating missing planet temperatures
    # -----------------------------------------------------------------------------
    # Equilibrium temperature (from the received flux of the star)
    bond_albedo = 0.3  # Default value (mean value for exoplanets)
    pteq        = planet_table['StarTeff'] * np.sqrt((planet_table['StarRadius']) / (2*planet_table['SMA'])).decompose() * (1 - bond_albedo)**(1/4)
    
    # Temperature from internal and initial energy (from evolutionary model) and radii
    ptint_em, pr_em, _ = get_evolutionary_model(planet_table)
    
    print(f"\n(2) Hypothesis (possibly pessimistic for young planets): Planets are assumed to be at their equilibrium temperature with insolation when missing (A_B={bond_albedo}, 4π redistribution) + Internal energy when evolutionary model is possible")
    pteff_invalid  = get_invalid_mask(planet_table["PlanetTeff"])
    pteq_valid     = get_valid_mask(pteq)
    ptint_em_valid = get_valid_mask(ptint_em)
    pteff_filling  = (pteff_invalid) & (pteq_valid | ptint_em_valid)
    planet_table["PlanetTeff"][pteff_filling]    = ( np.nan_to_num(pteq[pteff_filling])**4 + np.nan_to_num(ptint_em[pteff_filling])**4 )**(1/4)
    planet_table["PlanetTeffRef"][pteff_filling] = f"Equilibrium temperature (A_B={bond_albedo}, 4π redistribution) + Internal energy (when possible)"
    print(f"                  => PlanetTeff = (1 - A_B)**(1/4)·StarTeff·sqrt(StarRadius/2·SMA) (for {len(planet_table[pteff_filling])}/{len(planet_table)} planets)")
    
    # -----------------------------------------------------------------------------
    # 11) Filling missing radius from evolutionary model if possible
    # -----------------------------------------------------------------------------
    pr_invalid                                  = get_invalid_mask(planet_table["PlanetRadius"])
    pr_em_valid                                 = get_valid_mask(pr_em)
    pr_filling                                  = pr_invalid & pr_em_valid
    planet_table["PlanetRadius"][pr_filling]    = pr_em[pr_filling]
    planet_table["PlanetRadiusRef"][pr_filling] = "Evolutionary model"
    print(f"\n Filling planet radius from evolutionnary model for {pr_filling.sum()}/{len(planet_table)}")
    
    # For directly-imaged planets, radius is not critical/usefull since the K-band magnitudes is known => fill invalid with sample median
    pr_invalid                                            = get_invalid_mask(planet_table["PlanetRadius"])
    planet_table["PlanetRadius"][pr_invalid & im_mask]    = np.nanmedian(planet_table['PlanetRadius'][im_mask])
    planet_table["PlanetRadiusRef"][pr_invalid & im_mask] = "Filled with median value"
    
    # -----------------------------------------------------------------------------
    # 12) Filter down to rows usable for S/N calculations
    # -----------------------------------------------------------------------------
    # Clipping outlier temperatures
    planet_table["PlanetTeff"][planet_table["PlanetTeff"] > 3000*u.K] = 3000*u.K
    
    # Clipping outlier radius
    planet_table["PlanetRadius"][planet_table["PlanetRadius"] > 40*u.R_earth] = 40*u.R_earth

    ste_valid   = get_valid_mask(planet_table["StarTeff"])
    pteff_valid = get_valid_mask(planet_table["PlanetTeff"])
    skm_valid   = get_valid_mask(planet_table["StarKmag"])
    pr_valid    = get_valid_mask(planet_table["PlanetRadius"])
    d_valid     = get_valid_mask(planet_table["Distance"])
    sma_valid   = get_valid_mask(planet_table["SMA"])
    snr_valid   = ste_valid & pteff_valid & skm_valid & pr_valid & d_valid & sma_valid
    print(f"\n(3) Filtering planets for S/N computations: keeping {len(planet_table[snr_valid])}/{len(planet_table)} planets")
    print(f"                  => Missing StarTeff:     IM: {round(100*len(planet_table[~ste_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~ste_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~ste_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~ste_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing PlanetTeff:   IM: {round(100*len(planet_table[~pteff_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~pteff_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~pteff_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~pteff_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing StarKmag:     IM: {round(100*len(planet_table[~skm_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~skm_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~skm_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~skm_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing PlanetRadius: IM: {round(100*len(planet_table[~pr_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~pr_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~pr_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~pr_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing Distance:     IM: {round(100*len(planet_table[~d_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~d_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~d_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~d_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"                  => Missing SMA:          IM: {round(100*len(planet_table[~sma_valid & im_mask])/len(planet_table[im_mask]))}% | RV: {round(100*len(planet_table[~sma_valid & rv_mask])/len(planet_table[rv_mask]))}% | TR: {round(100*len(planet_table[~sma_valid & tr_mask])/len(planet_table[tr_mask]))}% | OT: {round(100*len(planet_table[~sma_valid & ot_mask])/len(planet_table[ot_mask]))}%")
    print(f"\nKeeping: TOTAL: {len(planet_table[snr_valid])}/{len(planet_table)} | IM: {len(planet_table[snr_valid & im_mask])}/{len(planet_table[im_mask])} | RV: {len(planet_table[snr_valid & rv_mask])}/{len(planet_table[rv_mask])} | TR: {len(planet_table[snr_valid & tr_mask])}/{len(planet_table[tr_mask])} | OT: {len(planet_table[snr_valid & ot_mask])}/{len(planet_table[ot_mask])}")
    planet_table = planet_table[snr_valid]
    
    # -----------------------------------------------------------------------------
    # 13) Classifying planet types
    # -----------------------------------------------------------------------------
    # Creating planet type column
    planet_table["PlanetType"] = np.full(len(planet_table), "Unidentified", dtype="<U32")
    
    # Classifying (from 'planet_types')
    for idx in range(len(planet_table)):
        planet_table[idx]["PlanetType"] = get_planet_type(planet_table[idx])
    
    # -----------------------------------------------------------------------------
    # 14) Determining whether the rocky ("Earth-Like") planets have atmosphere
    # -----------------------------------------------------------------------------
    # Creating planet atmosphere column
    planet_table["HasAtmosphere"] = np.full(len(planet_table), True, dtype=bool)
    
    # Jeans escape parameter λJ for a heavy (N2-like) atmosphere
    mu               = 28   # mean molecular weight [amu], N2-like
    lambda_jeans_thr = 100  # heuristic threshold (dimensionless), calibrated with the SAME T definition:
                            # using global Teff, λJ(N2) ~ 38 for the Moon (T~270 K) and ~69 for Mercury (T~440 K).
                            # Therefore λJ ≲ O(1e2) corresponds to Moon/Mercury-like weak binding => likely airless/featureless rocky planets.
    m            = mu * m_u                                   # [kg]
    M            = planet_table["PlanetMass"].to(u.kg).value  # [kg]
    R            = planet_table["PlanetRadius"].to(u.m).value # [m]
    T            = planet_table["PlanetTeff"].to(u.K).value   # [K]
    lambda_jeans = G * M * m / (kB * T * R)                   # [no unit]
    no_retention = lambda_jeans < lambda_jeans_thr
    
    # Blowned atmosphere
    C_shore          = 320  # ad hoc normalization of the cosmic-shoreline scaling (I ~ v_esc^4), set so that Mercury lies near the boundary (Sun-like irradiation): C_mercury = I_rel,mercury / (v_esc,mercury / v_esc,earth )**4
    I_rel            = (planet_table["StarLum"].to(u.L_sun).value) / (planet_table["SMA"].to(u.AU).value**2) # Insolation relative to Earth (assuming StarLum in Lsun and SMA in AU)
    vesc             = 1e-3*np.sqrt(2 * G * M / R) # [km/s] escape velocity
    I_crit           = C_shore * (vesc / vesc_earth)**4
    blown_atmosphere = I_rel > I_crit
    
    # Flagging planets without atmosphere
    rocky_planets = get_mask_planet_type(planet_table, planet_type="earth")
    no_atmosphere = rocky_planets & (no_retention | blown_atmosphere) 
    planet_table["HasAtmosphere"][no_atmosphere] = False
    
    print("\n(4) Flagging rocky planets without atmosphere:")
    print(f"Rocky planets without retention:           {(rocky_planets & no_retention).sum()} / {rocky_planets.sum()}")
    print(f"Rocky planets with blown atmosphere:       {(rocky_planets & blown_atmosphere).sum()} / {rocky_planets.sum()}")
    print(f"Rocky planets with either (no atmosphere): {no_atmosphere.sum()} / {rocky_planets.sum()}")
    
    # --------------------------------------------------------------------------------------------------------
    # 15) Telluric-equivalent airmass for rocky planets with atmosphere (X_eq = airmass = 2.0 at Earth values)
    # --------------------------------------------------------------------------------------------------------
    # Heuristic proxy used only to scale the Earth-like telluric albedo template.
    # By construction, X_eq = 2 for Earth-like gravity when P_ref = 1 bar and eta = 1.
    # This is not a true observational airmass.
    # In practice, we compute this quantity for all planets, in case we later choose to apply or test reflected_model = "tellurics" across the full sample.

    # Models parameters
    P_ref = 1.0 * u.bar # simple first-order assumption
    eta   = 1.0         # g^{-1} scaling
    X_min = 0.5
    X_max = 8.0
    
    P_ref_earth = 1.0 * u.bar
    g_earth     = (const.G * const.M_earth / const.R_earth**2).to(u.m / u.s**2)
    g_planet    = (const.G * planet_table["PlanetMass"] / planet_table["PlanetRadius"]**2).to(u.m / u.s**2)
    X_eq        = 2.0 * (P_ref / P_ref_earth) * (g_earth / g_planet)**eta
    X_eq        = np.clip(X_eq.to_value(u.dimensionless_unscaled), X_min, X_max) * u.dimensionless_unscaled
    
    planet_table["TelluricEquivalentAirmass"] = X_eq
    print("\n(5) Assigning telluric-equivalent airmass values")  
    
    # -------------------------------------------------------------------
    # 16) Computing magnitudes for all bands and classifying planet types
    # -------------------------------------------------------------------
    planet_table = get_planet_table_magnitudes(planet_table)
    
    # ----------------------
    # 17) Saving final table
    # ----------------------
    print(f"\nTotal number of planets for FastYield calculations: {len(planet_table)}")
    print(f"\nGenerating the table took {round((time.time()-time1)/60, 1)} mn")
    planet_table.write(archive_path + "Archive_Pull_For_FastYield.ecsv", format='ascii.ecsv', overwrite=True)
    
    # Few sanity check plots
    planet_table_classification()
    planet_table_statistics()



#######################################################################################################################
########################################### SNR computations: #########################################################
#######################################################################################################################

def process_SNR(idx):
    """
    Worker: compute SNR and band magnitudes for a single planet.

    Parameters
    ----------
    idx: int

    Returns
    -------
    result : tuple
         - idx       : int
         - mags_p    : dict {'Star{instru}mag' : float, 'Planet{instru}mag' : float, 'Star{band}mag' : float, ... }
         - name_band : list[str]
         - signal    : np.ndarray [e-/DIT]  (per name_band entry)
         - sigma_f   : np.ndarray [e-/DIT]
         - sigma_s   : np.ndarray [e-/DIT]
         - DIT       : np.ndarray [mn]
    """
    
    # Context
    planet_table           = _SNR_CTX["planet_table"]
    instru                 = _SNR_CTX["instru"]
    thermal_model          = _SNR_CTX["thermal_model"]
    reflected_model        = _SNR_CTX["reflected_model"]
    spectrum_contributions = _SNR_CTX["spectrum_contributions"]
    wave_model             = _SNR_CTX["wave_model"]
    wave_K                 = _SNR_CTX["wave_K"]
    counts_vega            = _SNR_CTX["counts_vega"]
    counts_vega_K          = _SNR_CTX["counts_vega_K"]
    band0                  = _SNR_CTX["band0"]
    exposure_time          = _SNR_CTX["exposure_time"]
    apodizer               = _SNR_CTX["apodizer"]
    strehl                 = _SNR_CTX["strehl"]
    coronagraph            = _SNR_CTX["coronagraph"]
    Rc                     = _SNR_CTX["Rc"]
    filter_type            = _SNR_CTX["filter_type"]
    systematics            = _SNR_CTX["systematics"]
    PCA                    = _SNR_CTX["PCA"]
    N_PCA                  = _SNR_CTX["N_PCA"]
    masks                  = _SNR_CTX["masks"]
    bands_valid            = _SNR_CTX["bands_valid"]
    
    # Planet row
    planet = planet_table[idx]
    
    # Computing models on wave_model    
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=None, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False, in_im_mag=True)
    
    # Re-computing the magnitudes (in case the models is different from the original table)
    mags = {}
    
    # Instru mags
    mask_instru = masks[instru]
    mags[f"StarINSTRUmag({instru})"] = get_mag(wave=wave_model[mask_instru], density_obs=star_spectrum.flux[mask_instru],   density_vega=None, counts_vega=counts_vega[instru])
    if "thermal" in spectrum_contributions:
        mags[f"PlanetINSTRUmag({instru})(thermal)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_thermal.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
    if "reflected" in spectrum_contributions:
        mags[f"PlanetINSTRUmag({instru})(reflected)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_reflected.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
    if spectrum_contributions == "thermal+reflected":
        mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"] = get_mag(wave=wave_model[mask_instru], density_obs=planet_spectrum.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
        
    # Bands mags
    for band in bands_valid:
        mask_band = masks[band]
        if band != "K":
            mags[f"Star{band}mag"] = get_mag(wave=wave_model[mask_band], density_obs=star_spectrum.flux[mask_band],   density_vega=None, counts_vega=counts_vega[band])
        if "thermal" in spectrum_contributions:
            mags[f"Planet{band}mag(thermal)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_thermal.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])
        if "reflected" in spectrum_contributions:
            mags[f"Planet{band}mag(reflected)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_reflected.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])
        if spectrum_contributions == "thermal+reflected":
            mags[f"Planet{band}mag(thermal+reflected)"] = get_mag(wave=wave_model[mask_band], density_obs=planet_spectrum.flux[mask_band], density_vega=None, counts_vega=counts_vega[band])

    # Computing the SNR for the planet
    mag_s = mags[f"StarINSTRUmag({instru})"]
    mag_p = mags[f"PlanetINSTRUmag({instru})({spectrum_contributions})"]
    name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = FastCurves(instru=instru, band_only=None, calculation="SNR", mag_star=mag_s, band0=band0, exposure_time=exposure_time, mag_planet=mag_p, separation_planet=planet["AngSep"].value/1000, return_FastYield=True, show_plot=False, verbose=False, planet_name=None, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematics=systematics, PCA=PCA, N_PCA=N_PCA)
    
    return idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band



def get_planet_table_SNR(instru, table="Archive", thermal_model="None", reflected_model="None", apodizer="NO_SP", strehl="NO_JQ", coronagraph=None, Rc=100, filter_type="gaussian", systematics=False, PCA=False, N_PCA=20):
    """"
    Compute per-planet SNRs for a given instrument and write results to the table.
    
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
    systematics : bool, optional
        Whether to include systematics noise in FastCurves.
    PCA : bool, optional
        Whether to include PCA post-processing in FastCurves.
    N_PCA : int, optional
        Number of PCA components if PCA=True.
    """
    config_data   = get_config_data(instru)
    exposure_time = 120 # [mn]
    time1         = time.time()
    if systematics and filter_type == "gaussian_fast":
        filter_type = "gaussian" # "gaussian_fast" is bad for handling systematics estimations
    
    # --- 1) Loading table ---
    if table == "Archive":
        planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
        path = archive_path
    elif table == "Simulated":
        planet_table = load_planet_table("Simulated_Pull_For_FastYield.ecsv")
        path = simulated_path
    else:
        raise ValueError("table must be 'Archive' or 'Simulated'")
    
    # --- 2) Optional Exo-REM Teff range filtering ---
    if thermal_model == "Exo-REM":
        planet_table = planet_table[(planet_table['PlanetTeff'] > 400*u.K) & (planet_table['PlanetTeff'] < 2000*u.K)]
            
    # --- 3) Wavelength grids and Vega ---
    # K-band for photometry
    wave_K = get_wave_K()
    
    # Model bandwidth
    lmin_instru = config_data["lambda_range"]["lambda_min"]   # [µm]
    lmax_instru = config_data["lambda_range"]["lambda_max"]   # [µm]
    lmin_model  = 0.98*lmin_instru                            # [µm]
    lmax_model  = 1.02*lmax_instru                            # [µm] (a bit larger than the instrumental bandwidth to avoid edge effects)
    R_instru    = get_R_instru(config_data=config_data)       # Max instrument resolution (factor 2 to be sure to not loose spectral information)
    R_model     = min(R_instru, R0_max)                       # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
    dl_model    = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
    wave_model  = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
    
    # Vega spectrum on K-band and instru-band [J/s/m2/µm]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K,     renorm=False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_model, renorm=False)
    
    # --- 4) Create columns for signal, noise and DIT length ---
    planet_table["signal_INSTRU"]     = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["sigma_fund_INSTRU"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["sigma_syst_INSTRU"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
    planet_table["DIT_INSTRU"]        = np.full(len(planet_table), np.nan) # [mn]
    for band in bands:
        planet_table[f"signal_{band}"]     = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"sigma_fund_{band}"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"sigma_syst_{band}"] = np.full(len(planet_table), np.nan) # [e-/FWHM/DIT]
        planet_table[f"DIT_{band}"]        = np.full(len(planet_table), np.nan) # [nm]
    
    # --- 5) Bandpass masks and Vega flux on bands ---
    counts_vega_K       = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
    counts_vega         = {}
    masks               = {}
    bands_valid         = []
    masks[instru]       = (wave_model >= lmin_instru) & (wave_model <= lmax_instru)
    counts_vega[instru] = get_counts_from_density(wave=wave_model[masks[instru]], density=vega_spectrum.flux[masks[instru]])
    for band in bands:
        lmin_band, lmax_band = get_band_lims(band)
        if lmin_instru <= lmin_band and lmax_instru >= lmax_band: # if this band is inside the instrumental range
            bands_valid.append(band)
            masks[band]       = (wave_model >= lmin_band) & (wave_model <= lmax_band)
            counts_vega[band] = get_counts_from_density(wave=wave_model[masks[band]], density=vega_spectrum.flux[masks[band]])
    
    # Band where magnitudes are defined for the FastCurves computations
    band0 = "instru"
    
    # Contribution and model labels
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # Suffix
    suffix = "with systematics+PCA" if (systematics and PCA) else ("with systematics" if systematics else "without systematics")
    
    # Print
    print(f"\n {instru} ({apodizer} & {strehl} & {coronagraph}) {suffix} ({thermal_model} & {reflected_model})")

    # --- 6) Init global context for workers ---
    global _SNR_CTX
    _SNR_CTX = dict(planet_table=planet_table, instru=instru, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, wave_model=wave_model, wave_K=wave_K, counts_vega=counts_vega, counts_vega_K=counts_vega_K, band0=band0, exposure_time=exposure_time, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematics=systematics, PCA=PCA, N_PCA=N_PCA, masks=masks, bands_valid=bands_valid)    
    
    # Function to enter the estimations in the planet_table
    def set_planet_table_values(planet_table, idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band):
        planet_table[idx][f"StarINSTRUmag({instru})"] = mags[f"StarINSTRUmag({instru})"]
        if "thermal" in spectrum_contributions:
            planet_table[idx][f"PlanetINSTRUmag({instru})(thermal)"] = mags[f"PlanetINSTRUmag({instru})(thermal)"]
        if "reflected" in spectrum_contributions:
            planet_table[idx][f"PlanetINSTRUmag({instru})(reflected)"] = mags[f"PlanetINSTRUmag({instru})(reflected)"]
        if spectrum_contributions == "thermal+reflected":
            planet_table[idx][f"PlanetINSTRUmag({instru})(thermal+reflected)"] = mags[f"PlanetINSTRUmag({instru})(thermal+reflected)"]
        for band in bands_valid:
            if band != "K":
                planet_table[idx][f"Star{band}mag"] = mags[f"Star{band}mag"]
            if "thermal" in spectrum_contributions:
                planet_table[idx][f"Planet{band}mag(thermal)"] = mags[f"Planet{band}mag(thermal)"]
            if "reflected" in spectrum_contributions:
                planet_table[idx][f"Planet{band}mag(reflected)"] = mags[f"Planet{band}mag(reflected)"]
            if spectrum_contributions == "thermal+reflected":
                planet_table[idx][f"Planet{band}mag(thermal+reflected)"] = mags[f"Planet{band}mag(thermal+reflected)"]
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
    
    # --- 7) Run SNR for each planet (parallel if not PCA or serial if PCA) ---
    if PCA: # if PCA, no multiprocessing (otherwise it crashes: TODO ?)
        for idx in tqdm(range(len(planet_table)), desc="Serial"):
            idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = process_SNR(idx)
            set_planet_table_values(planet_table, idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band)
    
    else: # if no PCA, uses multiprocessing
        with Pool(processes=cpu_count()//2) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            for (idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band) in tqdm(pool.imap(process_SNR, [(idx) for idx in range(len(planet_table))]), total=len(planet_table), desc="Multiprocessing"):
                set_planet_table_values(planet_table, idx, mags, name_band, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band)
                    
    print(f"\n Calculating SNR took {(time.time()-time1)/60:.1f} mn")
    filename = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=name_model)
    planet_table.write(path+filename, format='ascii.ecsv', overwrite=True)
    


def all_SNR_table(table="Archive", instrus=instrus): # takes ~ 13 hours
    """
    To compute SNR for every instruments with different model spectra
    """
    time0 = time.time()
    for instru in instrus:
        config_data = get_config_data(instru)
        if config_data["lambda_range"]["lambda_max"] > ignore_reflected_thresh_um:
            thermal_models   = ["None", "auto"]
            reflected_models = ["None"]
        else:
            thermal_models   = ["None", "auto"]
            reflected_models = ["None", "auto"]
        for apodizer in config_data["apodizers"]:
            for strehl in config_data["strehls"]:
                for coronagraph in config_data["coronagraphs"]:
                    for thermal_model in thermal_models:
                        for reflected_model in reflected_models:
                            if thermal_model == "None" and reflected_model == "None":
                                continue
                            else:
                                get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=False)
                                if instru in instrus_with_systematics:
                                    get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=True)
                                    get_planet_table_SNR(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=True, PCA=True, N_PCA=20)
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
    
    mask_cold = np.array(planet_table["PlanetTeff"].value < 250)
    mask_temp = np.array(250 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 500) 
    mask_warm = np.array(500 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 1000) 
    mask_hot  = np.array(1000 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 1500) 
    mask_vhot = np.array(1500 <= planet_table["PlanetTeff"].value) & np.array(planet_table["PlanetTeff"].value < 2000) 
    mask_uhot = np.array(2000 <= planet_table["PlanetTeff"].value)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$M_\oplus$", fontsize=14)
    plt.ylabel(r"$R_\oplus$", fontsize=14)
    #plt.title(f"FastYield classification: {len(planet_table)} known exoplanets", fontsize=16)
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
    #plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    
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
        """Retourne une liste {name, m1,m2,r1,r2} en agrégeant toutes les bandes Teff."""
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
    data[:, 2] = np.array(planet_table["PlanetTeff"].value)
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
    #figure.suptitle(f"Archive table statistics (with {len(planet_table)} known exoplanets)", fontsize=18, y=1.05, fontweight="bold")
    plt.gcf().set_dpi(300)
    plt.show()



###############
# Yield plots #
###############

def yield_plot_instrus_texp(thermal_model="auto", reflected_model="auto", fraction=False):
        
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table_harmoni          = load_planet_table(f"Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_andes            = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_{name_model}.ecsv")
    planet_table_eris             = load_planet_table(f"Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_{name_model}.ecsv")
    planet_table_hirise           = load_planet_table(f"Archive_Pull_HiRISE_NO_SP_MED_without_systematics_{name_model}.ecsv")
    planet_table_mirimrs_non_syst = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst     = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nircam           = load_planet_table(f"Archive_Pull_NIRCam_NO_SP_NO_JQ_MASK335R_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_non_syst = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst     = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    #planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")

    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    yield_harmoni          = np.zeros(len(exposure_time))
    yield_andes            = np.zeros(len(exposure_time))
    yield_eris             = np.zeros(len(exposure_time)) 
    yield_hirise           = np.zeros(len(exposure_time)) 
    yield_mirimrs_non_syst = np.zeros(len(exposure_time))
    yield_mirimrs_syst     = np.zeros(len(exposure_time))
    yield_mirimrs_syst_pca = np.zeros(len(exposure_time))
    yield_nircam           = np.zeros(len(exposure_time))
    yield_nirspec_non_syst = np.zeros(len(exposure_time))
    yield_nirspec_syst     = np.zeros(len(exposure_time))
    yield_nirspec_syst_pca = np.zeros(len(exposure_time))

    if fraction:
        ratio        = 100
        norm_harmoni = len(planet_table_harmoni)
        norm_andes   = len(planet_table_andes)
        norm_eris    = len(planet_table_eris)
        norm_hirise  = len(planet_table_hirise)
        norm_mirimrs = len(planet_table_mirimrs_non_syst)
        norm_nircam  = len(planet_table_nircam)
        norm_nirspec = len(planet_table_nirspec_non_syst)
    else:
        ratio = norm_harmoni = norm_andes = norm_eris = norm_hirise = norm_mirimrs = norm_nircam = norm_nirspec = 1
    
    for i in range(len(exposure_time)):
        SNR_harmoni          = get_SNR_from_table(table=planet_table_harmoni,          exposure_time=exposure_time[i], band="INSTRU")
        SNR_andes            = get_SNR_from_table(table=planet_table_andes,            exposure_time=exposure_time[i], band="INSTRU")
        SNR_eris             = get_SNR_from_table(table=planet_table_eris,             exposure_time=exposure_time[i], band="INSTRU")
        SNR_hirise           = get_SNR_from_table(table=planet_table_hirise,           exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_non_syst = get_SNR_from_table(table=planet_table_mirimrs_non_syst, exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_syst     = get_SNR_from_table(table=planet_table_mirimrs_syst,     exposure_time=exposure_time[i], band="INSTRU")
        SNR_mirimrs_syst_pca = get_SNR_from_table(table=planet_table_mirimrs_syst_pca, exposure_time=exposure_time[i], band="INSTRU")
        SNR_nircam           = get_SNR_from_table(table=planet_table_nircam,           exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_non_syst = get_SNR_from_table(table=planet_table_nirspec_non_syst, exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_syst     = get_SNR_from_table(table=planet_table_nirspec_syst,     exposure_time=exposure_time[i], band="INSTRU")
        SNR_nirspec_syst_pca = get_SNR_from_table(table=planet_table_nirspec_syst_pca, exposure_time=exposure_time[i], band="INSTRU")

        yield_harmoni[i]          = ratio * len(planet_table_harmoni[SNR_harmoni > SNR_thresh])                   / norm_harmoni
        yield_andes[i]            = ratio * len(planet_table_andes[SNR_andes > SNR_thresh])                       / norm_andes
        yield_eris[i]             = ratio * len(planet_table_eris[SNR_eris > SNR_thresh])                         / norm_eris
        yield_hirise[i]           = ratio * len(planet_table_hirise[SNR_hirise > SNR_thresh])                     / norm_hirise
        yield_mirimrs_non_syst[i] = ratio * len(planet_table_mirimrs_non_syst[SNR_mirimrs_non_syst > SNR_thresh]) / norm_mirimrs
        yield_mirimrs_syst[i]     = ratio * len(planet_table_mirimrs_syst[SNR_mirimrs_syst > SNR_thresh])         / norm_mirimrs
        yield_mirimrs_syst_pca[i] = ratio * len(planet_table_mirimrs_syst_pca[SNR_mirimrs_syst_pca > SNR_thresh]) / norm_mirimrs
        yield_nircam[i]           = ratio * len(planet_table_nircam[SNR_nircam > SNR_thresh])                     / norm_nircam
        yield_nirspec_non_syst[i] = ratio * len(planet_table_nirspec_non_syst[SNR_nirspec_non_syst > SNR_thresh]) / norm_nirspec
        yield_nirspec_syst[i]     = ratio * len(planet_table_nirspec_syst[SNR_nirspec_syst > SNR_thresh])         / norm_nirspec
        yield_nirspec_syst_pca[i] = ratio * len(planet_table_nirspec_syst_pca[SNR_nirspec_syst_pca > SNR_thresh]) / norm_nirspec

    lw = 2
    plt.figure(dpi=300, figsize=(14, 8))
    plt.plot(exposure_time, yield_harmoni,          lw=lw, c=colors_instru["HARMONI"], label="ELT/HARMONI")
    plt.plot(exposure_time, yield_andes,            lw=lw, c=colors_instru["ANDES"],   label="ELT/ANDES")
    plt.plot(exposure_time, yield_eris,             lw=lw, c=colors_instru["ERIS"],    label="VLT/ERIS")
    plt.plot(exposure_time, yield_hirise,           lw=lw, c=colors_instru["HiRISE"],  label="VLT/HiRISE")    
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
    plt.xlabel('Exposure time per target [mn]', fontsize=18)
    if fraction:
        plt.ylabel('Fraction of planets re-detected [%]', fontsize=18)
    else:
        plt.ylabel('Number of planets re-detected', fontsize=18)
        plt.yscale('log')
    plt.xlim(exposure_time[0], exposure_time[-1])
    plt.title('Known exoplanets detection yield', fontsize=20, weight='bold')
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=16, loc="upper left", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Instruments", title_fontsize=18)    
    ax        = plt.gca()
    ax_legend = ax.twinx()
    ax_legend.plot([], [], 'k-',  label='Without Systematics',    linewidth=lw)
    ax_legend.plot([], [], 'k--', label='With Systematics',       linewidth=lw)
    ax_legend.plot([], [], 'k:',  label='With Systematics + PCA', linewidth=lw)
    ax_legend.legend(fontsize=16, loc="upper center", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Systematic assumption", title_fontsize=18)
    ax_legend.tick_params(axis='y', colors='w')  # Masking ticks
    plt.tight_layout()
    plt.show()



def yield_plot_bands_texp(table="Archive", instru="HARMONI", thermal_model="auto", reflected_model="auto", systematics=False, PCA=False, fraction=False):
            
    ls_modes    = ["-", "--", ":"]
    config_data = get_config_data(instru)
    apodizer    = "NO_SP"
    coronagraph = None
    strehl      = config_data["strehls"][0]
    bands       = config_data["gratings"]
    NbBand      = len(bands)
    cmap        = plt.get_cmap("rainbow", NbBand)

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    if instru == "HARMONI":
        modes         = ["NO_SP", "SP1", "SP_Prox"]
        planet_tables = []
        NbPlanet      = np.zeros(len(modes), dtype=int)
        for im, mode in enumerate(modes):
            filename_pt  = get_filename_table(table=table, instru=instru, apodizer=mode, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=name_model)
            pt           = load_planet_table(filename_pt)
            NbPlanet[im] = len(pt)
            planet_tables.append(pt)

    elif instru == "ANDES":
        modes         = [None, "LYOT"]
        planet_tables = []
        NbPlanet      = np.zeros(len(modes), dtype=int)
        for im, mode in enumerate(modes):
            filename_pt  = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=mode, systematics=systematics, PCA=PCA, name_model=name_model)
            pt           = load_planet_table(filename_pt)
            NbPlanet[im] = len(pt)
            planet_tables.append(pt)
    
    else:
        filename_pt   = get_filename_table(table=table, instru=instru, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, systematics=systematics, PCA=PCA, name_model=name_model)
        pt            = load_planet_table(filename_pt)
        planet_tables = [pt]
        NbPlanet      = np.array([len(pt)])
        modes         = [""]
    
    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    yields        = np.zeros((NbBand, len(modes), len(exposure_time)))
    for ib, band in enumerate(bands):
        for im in range(len(modes)):
            for it in range(len(exposure_time)):
                SNR = get_SNR_from_table(table=planet_tables[im], exposure_time=exposure_time[it], band=band)
                if fraction :
                    yields[ib, im, it] = len(planet_tables[im][SNR > SNR_thresh]) * 100/NbPlanet[im]
                else:
                    yields[ib, im, it] = len(planet_tables[im][SNR > SNR_thresh])

    plt.figure(dpi=300, figsize=(14, 8))
    for ib, band in enumerate(bands):
        for im in range(len(modes)):
            if not (yields[ib, im]==0).all():
                if im == 0:
                    plt.plot(exposure_time, yields[ib, im], color=cmap(ib), label=band.replace("_", " "), ls=ls_modes[im], lw=3)
                else:
                    plt.plot(exposure_time, yields[ib, im], color=cmap(ib), ls=ls_modes[im], lw=3)
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
    if config_data["base"] == "space":
        plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above\n (with {name_model} model in {spectrum_contributions} light)", fontsize=18, weight='bold')
    else:
        plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above for {strehl} strehl\n (with {name_model} model in {spectrum_contributions} light)", fontsize=18, weight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=16, loc="upper left", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Bands", title_fontsize=18)    
    if len(modes) > 1: 
        ax = plt.gca()
        ax_legend = ax.twinx()
        for im, mode in enumerate(modes):
            ax_legend.plot([], [], c="k", ls=ls_modes[im], label=str(mode).replace("_", " ").replace("None", "w/o coronagraph"), lw=3)
        if instru=="HARMONI":
            ax_legend.legend(fontsize=16, loc="upper center", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Apodizers", title_fontsize=18)    
        elif instru=="ANDES":
            ax_legend.legend(fontsize=16, loc="upper center", frameon=True, fancybox=True, edgecolor="gray", facecolor="whitesmoke", title="Coronagraphs", title_fontsize=18)    
        ax_legend.tick_params(axis='y', colors='w') # Masking ticks
    plt.tight_layout()
    plt.show()



####################
# Yield histograms #
####################

def yield_hist_instrus_ptypes(exposure_time=10*60, thermal_model="auto", reflected_model="auto", planet_types=planet_types_reduced, fraction=False):
        
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    planet_table_harmoni          = load_planet_table(f"Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_harmoni_sp_prox  = load_planet_table(f"Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_{name_model}.ecsv")
    planet_table_andes            = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_without_systematics_{name_model}.ecsv")
    planet_table_andes_lyot       = load_planet_table(f"Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_{name_model}.ecsv")
    planet_table_eris             = load_planet_table(f"Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_{name_model}.ecsv")
    planet_table_mirimrs_non_syst = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst     = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table(f"Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nircam           = load_planet_table(f"Archive_Pull_NIRCam_NO_SP_NO_JQ_MASK335R_without_systematics_{name_model}.ecsv")
    planet_table_nirspec_non_syst = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst     = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    #planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_{thermal_model}+None.ecsv")
    planet_table_nirspec_syst_pca = load_planet_table(f"Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_{thermal_model}+None.ecsv")
    
    planet_table_harmoni["SNR"]          = np.fmax(get_SNR_from_table(table=planet_table_harmoni,  exposure_time=exposure_time, band="INSTRU"), get_SNR_from_table(table=planet_table_harmoni_sp_prox, exposure_time=exposure_time, band="INSTRU"))
    planet_table_andes["SNR" ]           = np.fmax(get_SNR_from_table(table=planet_table_andes,    exposure_time=exposure_time, band="INSTRU"), get_SNR_from_table(table=planet_table_andes_lyot,      exposure_time=exposure_time, band="INSTRU"))
    planet_table_eris["SNR"]             = get_SNR_from_table(table=planet_table_eris,             exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_non_syst["SNR"] = get_SNR_from_table(table=planet_table_mirimrs_non_syst, exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_syst["SNR"]     = get_SNR_from_table(table=planet_table_mirimrs_syst,     exposure_time=exposure_time, band="INSTRU")
    planet_table_mirimrs_syst_pca["SNR"] = get_SNR_from_table(table=planet_table_mirimrs_syst_pca, exposure_time=exposure_time, band="INSTRU")
    planet_table_nircam["SNR"]           = get_SNR_from_table(table=planet_table_nircam,           exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_non_syst["SNR"] = get_SNR_from_table(table=planet_table_nirspec_non_syst, exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_syst["SNR"]     = get_SNR_from_table(table=planet_table_nirspec_syst,     exposure_time=exposure_time, band="INSTRU")
    planet_table_nirspec_syst_pca["SNR"] = get_SNR_from_table(table=planet_table_nirspec_syst_pca, exposure_time=exposure_time, band="INSTRU")

    
    # # TODO: only keeping reflected light planets for ELT instruments
    # band_regime    = "H"
    # mask_thermal   = (planet_table_harmoni[f"Planet{band_regime}mag(thermal)"] < planet_table_harmoni[f"Planet{band_regime}mag(reflected)"])
    # mask_reflected = ~mask_thermal
    # planet_table_harmoni = planet_table_harmoni[mask_reflected]
    
    # mask_thermal   = (planet_table_andes[f"Planet{band_regime}mag(thermal)"] < planet_table_andes[f"Planet{band_regime}mag(reflected)"])
    # mask_reflected = ~mask_thermal
    # planet_table_andes = planet_table_andes[mask_reflected]
    

    # Dictionnaries giving the list of planets for mp[type]
    mp_harmoni          = build_match_dict(planet_table_harmoni,          planet_types=planet_types)
    mp_andes            = build_match_dict(planet_table_andes,            planet_types=planet_types)
    mp_eris             = build_match_dict(planet_table_eris,             planet_types=planet_types)
    mp_mirimrs_non_syst = build_match_dict(planet_table_mirimrs_non_syst, planet_types=planet_types)
    mp_mirimrs_syst     = build_match_dict(planet_table_mirimrs_syst,     planet_types=planet_types)
    mp_mirimrs_syst_pca = build_match_dict(planet_table_mirimrs_syst_pca, planet_types=planet_types)
    mp_nircam           = build_match_dict(planet_table_nircam,           planet_types=planet_types)
    mp_nirspec_non_syst = build_match_dict(planet_table_nirspec_non_syst, planet_types=planet_types)
    mp_nirspec_syst     = build_match_dict(planet_table_nirspec_syst,     planet_types=planet_types)
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
            ratio = norm_harmoni = norm_andes = norm_eris = norm_mirimrs = norm_nircam = norm_nirspec = 1
        
        yield_harmoni[i]          = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_harmoni[ptype])          / norm_harmoni if norm_harmoni > 0 else 0
        yield_andes[i]            = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_andes[ptype])            / norm_andes   if norm_andes > 0   else 0
        yield_eris[i]             = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_eris[ptype])             / norm_eris    if norm_eris > 0    else 0
        yield_mirimrs_non_syst[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_mirimrs_non_syst[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst[i]     = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_mirimrs_syst[ptype])     / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst_pca[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_mirimrs_syst_pca[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_nircam[i]           = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nircam[ptype])           / norm_nircam  if norm_nircam > 0  else 0
        yield_nirspec_non_syst[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nirspec_non_syst[ptype]) / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst[i]     = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nirspec_syst[ptype])     / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst_pca[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_nirspec_syst_pca[ptype]) / norm_nirspec if norm_nirspec > 0 else 0

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



def yield_hist_instrus_ptypes_ELT(exposure_time=10*60, thermal_model="auto", reflected_model="auto", planet_types=planet_types, fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox", "ANDES", "ANDES+LYOT"]):
    
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table_harmoni      = load_planet_table("Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_harmoni_prox = load_planet_table("Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_andes        = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_without_systematics_"+name_model+".ecsv")
    planet_table_andes_lyot   = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_LYOT_without_systematics_"+name_model+".ecsv")
    
    planet_table_harmoni["SNR"]      = get_SNR_from_table(planet_table=planet_table_harmoni,      exposure_time=exposure_time, band="INSTRU")
    planet_table_harmoni_prox["SNR"] = get_SNR_from_table(planet_table=planet_table_harmoni_prox, exposure_time=exposure_time, band="INSTRU")
    planet_table_andes["SNR"]        = get_SNR_from_table(planet_table=planet_table_andes,        exposure_time=exposure_time, band="INSTRU")
    planet_table_andes_lyot["SNR"]   = get_SNR_from_table(planet_table=planet_table_andes_lyot,   exposure_time=exposure_time, band="INSTRU")
    
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
        
        yield_harmoni[i]      = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_harmoni[ptype])      / norm_harmoni      if norm_harmoni > 0      else 0
        yield_harmoni_prox[i] = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_harmoni_prox[ptype]) / norm_harmoni_prox if norm_harmoni_prox > 0 else 0
        yield_andes[i]        = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_andes[ptype])        / norm_andes        if norm_andes > 0        else 0
        yield_andes_lyot[i]   = ratio * sum(planet["SNR"] > SNR_thresh for planet in mp_andes_lyot[ptype])   / norm_andes_lyot   if norm_andes_lyot > 0   else 0
    
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

def yield_corner_instru(instru="HARMONI", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", coronagraph=None, band="INSTRU", systematics=False, PCA=False):
    smooth_corner = 1
    ndim          = 6 # Mp, Rp, Tp, a, d, sep
    config_data = get_config_data(instru)
    
    # WORKING ANGLE
    iwa, owa = get_wa(config_data=config_data, sep_unit="mas")    

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeff"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    
    # SETTING DATA
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeff"].value)
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
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv")
    planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeff"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
    planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
    SNR          = get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band)
    planet_table = planet_table[SNR > SNR_thresh]
    data       = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeff"].value)
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
    


def yield_corner_instrus(instru1="HARMONI", instru2="ANDES", apodizer1="NO_SP", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", coronagraph1=None, coronagraph2=None, exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", systematics=False, PCA=False):
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
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeff"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    #planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)] # TODO : crop upper WA (with owa) ?
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeff"].value)
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
        if systematics:
            suffix = "with_systematics+PCA" if PCA else "with_systematics"
        else:
            suffix = "without_systematics"
        planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv")
        planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeff"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
        planet_table = planet_table[(planet_table["AngSep"]>IWA[ni]*u.mas)&(planet_table["AngSep"]<OWA[ni]*u.mas)]
        #planet_table = planet_table[(planet_table["AngSep"]>IWA[ni]*u.mas)] # TODO : crop upper WA (with owa) ?
        SNR          = get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band="INSTRU")
        planet_table = planet_table[SNR > SNR_thresh]
        yields[ni]   = len(planet_table)
        data       = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeff"].value)
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
    planet_table_raw = planet_table_raw[ (~get_invalid_mask(planet_table_raw["PlanetMass"])) & (~get_invalid_mask(planet_table_raw["PlanetRadius"])) & (~get_invalid_mask(planet_table_raw["PlanetTeff"])) & (~get_invalid_mask(planet_table_raw["SMA"])) & (~get_invalid_mask(planet_table_raw["Distance"])) & (~get_invalid_mask(planet_table_raw["AngSep"]))]
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    #planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)] # TODO : crop upper WA (with owa) ?
    data_raw       = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeff"].value)
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
    for im, model in enumerate(models):
        if model == "PICASO":
            name_model = "PICASO_"+spectrum_contributions+"_only"
        else:
            name_model = model
        planet_table = load_planet_table(f"Archive_Pull_{instru}_{apodizer}_{strehl}_without_systematics_{name_model}.ecsv")
        planet_table = planet_table[ (~get_invalid_mask(planet_table["PlanetMass"])) & (~get_invalid_mask(planet_table["PlanetRadius"])) & (~get_invalid_mask(planet_table["PlanetTeff"])) & (~get_invalid_mask(planet_table["SMA"])) & (~get_invalid_mask(planet_table["Distance"])) & (~get_invalid_mask(planet_table["AngSep"]))]
        planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
        #planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)] # TODO : crop upper WA (with owa) ?
        SNR          = get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band)
        planet_table = planet_table[SNR > SNR_thresh]
        yields[im]   = len(planet_table)
        data       = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeff"].value)
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
            color          = color_models[im],
            contour_kwargs = {"colors": [color_models[im]], "alpha": 0.85, "linewidth": 1.3},
            hist_kwargs    = {"color": color_models[im], "alpha": 0.85, "linewidth": 1.3},
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
    for im, model in enumerate(models):
        handles.append(mlines.Line2D([], [], color="w", label=""))
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=color_models[im], label=model+f" ({round(yields[im])} / {len(planet_table_raw)})"))
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

def yield_contrast_instru(instru="ANDES", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", coronagraph=None, systematics=False, PCA=False, table="Archive", band="INSTRU"):
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
    if systematics:
        suffix = "with_systematics+PCA" if PCA else "with_systematics"
    else:
        suffix = "without_systematics"
    tablename = f"{table}_Pull_{instru}_{apodizer}_{strehl}{coronagraph_str}_{suffix}_{name_model}.ecsv"

    planet_table = load_planet_table(tablename)
    SNR          = get_SNR_from_table(table=planet_table, exposure_time=exposure_time, band=band)
    planet_table = planet_table[SNR > SNR_thresh]
    SNR          = SNR[SNR > SNR_thresh]
    
    x = np.array(planet_table["AngSep"].value) # sep axis [mas]
    if config_data["sep_unit"]=="arcsec":
        x = x/1000
    mag_p = planet_table[f"PlanetINSTRUmag({instru})({spectrum_contributions})"]
    mag_s = planet_table[f"StarINSTRUmag({instru})"]
    y     = 10**(-(mag_p-mag_s)/2.5)
    z     = np.array(planet_table["PlanetTeff"].value) # color axis [K]

    im_mask = planet_table["DiscoveryMethod"]=="Imaging"
    tr_mask = planet_table["DiscoveryMethod"]=="Transit"
    rv_mask = planet_table["DiscoveryMethod"]=="Radial Velocity"
    ot_mask = (planet_table["DiscoveryMethod"]!="Imaging") & (planet_table["DiscoveryMethod"]!="Radial Velocity") & (planet_table["DiscoveryMethod"]!="Transit")

    # Taille des points (capés) en fonction du SNR
    s = get_size_from_SNR(SNR=SNR, s0=50, ds=200, SNR_min=SNR_thresh, SNR_max=1_000)

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

def Vrot_plots():
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    
    nbins = 50
    pm_mask = np.logical_not(get_invalid_mask(planet_table["PlanetMass"]))
    plt.figure(dpi=300)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(300*u.Mearth<planet_table["PlanetMass"])]), bins=nbins, edgecolor='black', alpha=0.666, label="Super-Jupiters", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(100*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=300*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Jupiters-like", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(20*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=100*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Neptunes-like", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(5*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=20*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Super-Earths", zorder=3)
    plt.hist(np.array(planet_table["PlanetVrot"][pm_mask&(planet_table["PlanetMass"]<=5*u.Mearth)]), bins=nbins, edgecolor='black', alpha=0.666, label="Earths-like", zorder=3)
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
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (10000 * u.K < planet_table["StarTeff"])]), bins=nbins, edgecolor='black', alpha=0.666, label="Very hot stars", zorder=3)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (6000 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Hot stars", zorder=3)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (3500 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Solar-type stars", zorder=3)
    plt.hist(np.array(planet_table["StarVrot"][st_mask & (planet_table["StarTeff"] <= 3500 * u.K)]), bins=nbins, edgecolor='black', alpha=0.666, label="Cool stars", zorder=3)
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
    Worker: compute contrast and band magnitudes for a single planet.
    """
    # idx, planet row
    idx, planet = args
    
    # Context
    instru                 = _CONTRAST_CTX["instru"]
    config_data            = _CONTRAST_CTX["config_data"]
    thermal_model          = _CONTRAST_CTX["thermal_model"]
    reflected_model        = _CONTRAST_CTX["reflected_model"]
    wave_model             = _CONTRAST_CTX["wave_model"]
    wave_K                 = _CONTRAST_CTX["wave_K"]
    counts_vega            = _CONTRAST_CTX["counts_vega"]
    counts_vega_K          = _CONTRAST_CTX["counts_vega_K"]
    band0                  = _CONTRAST_CTX["band0"]
    exposure_time          = _CONTRAST_CTX["exposure_time"]
    Rc                     = _CONTRAST_CTX["Rc"]
    filter_type            = _CONTRAST_CTX["filter_type"]
    systematics            = _CONTRAST_CTX["systematics"]
    PCA                    = _CONTRAST_CTX["PCA"]
    N_PCA                  = _CONTRAST_CTX["N_PCA"]
    masks                  = _CONTRAST_CTX["masks"]
    sep_max                = _CONTRAST_CTX["sep_max"]
    spectrum_contributions = _CONTRAST_CTX["spectrum_contributions"]

    # Computing models on wave_model    
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=planet, thermal_model=thermal_model, reflected_model=reflected_model, instru=instru, wave_model=wave_model, wave_K=wave_K, counts_vega_K=counts_vega_K, show=False)
    
    # Recalculates the magnitude in case the thermal model is no longer BT-Settl or the reflected model is no longer PICASO (the mag changes with regards to the raw archive table with the estimated magnitudes)
    mask_instru     = masks[instru]
    mag_s           = get_mag(wave=wave_model[mask_instru], density_obs=star_spectrum.flux[mask_instru],    density_vega=None, counts_vega=counts_vega[instru])
    mag_p_thermal   = get_mag(wave=wave_model[mask_instru], density_obs=planet_thermal.flux[mask_instru],   density_vega=None, counts_vega=counts_vega[instru])
    mag_p_reflected = get_mag(wave=wave_model[mask_instru], density_obs=planet_reflected.flux[mask_instru], density_vega=None, counts_vega=counts_vega[instru])
    mag_p_total     = get_mag(wave=wave_model[mask_instru], density_obs=planet_spectrum.flux[mask_instru],  density_vega=None, counts_vega=counts_vega[instru])
    
    if spectrum_contributions=="thermal":
        planet_spectrum = planet_thermal.copy()
        mag_p           = mag_p_thermal
    elif spectrum_contributions=="reflected":
        planet_spectrum = planet_reflected.copy()
        mag_p           = mag_p_reflected
    elif spectrum_contributions=="thermal+reflected":
        planet_spectrum = planet_spectrum.copy()
        mag_p           = mag_p_total

    # Computing the contrast for the planet
    contrasts_min = []
    for apodizer in config_data["apodizers"]:
        for strehl in config_data["strehls"]:
            # strehl = "MED"
            # if apodizer not in ["NO_SP", "SP1"]:
            #     continue
            for coronagraph in config_data["coronagraphs"]:
                _, separation, curves = FastCurves(instru=instru, calculation="contrast", mag_star=mag_s, band0=band0, exposure_time=exposure_time, mag_planet=mag_p, separation_planet=sep_max/1000, show_plot=False, verbose=False, planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=apodizer, strehl=strehl, coronagraph=coronagraph, Rc=Rc, filter_type=filter_type, systematics=systematics, PCA=PCA, N_PCA=N_PCA)
                contrasts_min.append(np.nanmin(np.stack(curves, axis=0), axis=0))

    contrast_min = np.nanmin(np.stack(contrasts_min, axis=0), axis=0)

    return idx, mag_s, mag_p_thermal, mag_p_reflected, mag_p_total, separation[0], contrast_min



def get_planet_table_contrast(instru, planet_table, exposure_time, thermal_model="None", reflected_model="None", spectrum_contributions=None, Rc=100, filter_type="gaussian", systematics=False, PCA=False, N_PCA=20, force_table_calc=False, sep_max=None):
   
    # Contribution and model labels
    _, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # Suffix
    suffix = "with systematics+PCA" if (systematics and PCA) else ("with systematics" if systematics else "without systematics")
    
    # Filename
    if systematics:
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

    except Exception as e:
    
        print(f"\nComputing the table: {e}...")
        
        planet_table = planet_table.copy()
        config_data  = get_config_data(instru)
        time1        = time.time()
        if systematics and filter_type == "gaussian_fast":
            filter_type = "gaussian" # "gaussian_fast" is bad for handling systematics estimations
        
        # --- 4) Wavelength grids and Vega ---
        # K-band for photometry
        wave_K = get_wave_K()
        
        # Model bandwidth
        lmin_instru = config_data["lambda_range"]["lambda_min"]   # [µm]
        lmax_instru = config_data["lambda_range"]["lambda_max"]   # [µm]
        lmin_model  = 0.98*lmin_instru                            # [µm]
        lmax_model  = 1.02*lmax_instru                            # [µm] (a bit larger than the instrumental bandwidth to avoid edge effects)
        R_instru    = get_R_instru(config_data=config_data)       # Max instrument resolution (factor 2 to be sure to not loose spectral information)
        R_model     = min(R_instru, R0_max)                       # Fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
        dl_model    = lmin_model / (2*R_model)                    # [µm/bin] Nyquist sampling of a spectrum with max resolving power R_model: 2 samples per resolution element at lmin_model
        wave_model  = np.arange(lmin_model, lmax_model, dl_model) # [µm] Model wavelength axis (with constant dl step)
        
        # Vega spectrum on K-band and instru-band [J/s/m2/µm]
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K,      renorm=False)
        vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_model, renorm=False)
        
        # --- 5) Bandpass masks and Vega flux on bands ---
        counts_vega_K       = get_counts_from_density(wave=wave_K, density=vega_spectrum_K.flux)
        counts_vega         = {}
        masks               = {}
        masks[instru]       = (wave_model >= lmin_instru) & (wave_model <= lmax_instru)
        counts_vega[instru] = get_counts_from_density(wave=wave_model[masks[instru]], density=vega_spectrum.flux[masks[instru]])
        
        # Band where magnitudes are defined for the FastCurves computations
        band0 = "instru"
        
        # Print
        print(f"\n {instru} {suffix} ({thermal_model} & {reflected_model})")
    
        # --- 6) Init global context for workers ---
        global _CONTRAST_CTX
        _CONTRAST_CTX = dict(instru=instru, config_data=config_data, thermal_model=thermal_model, reflected_model=reflected_model, wave_model=wave_model, wave_K=wave_K, counts_vega=counts_vega, counts_vega_K=counts_vega_K, band0=band0, exposure_time=exposure_time, Rc=Rc, filter_type=filter_type, systematics=systematics, PCA=PCA, N_PCA=N_PCA, masks=masks, sep_max=sep_max, spectrum_contributions=spectrum_contributions)    
    
        # --- 7) Run contrast---
        with Pool(processes=cpu_count()//2) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            for (idx, mag_s, mag_p_thermal, mag_p_reflected, mag_p_total, separation, contrast_5sigma) in tqdm(pool.imap(process_contrast, [(idx, planet_table[idx]) for idx in range(len(planet_table))]), total=len(planet_table), desc="Multiprocessing"):
                if idx==0:
                    planet_table["contrast_5sigma"] = np.full((len(planet_table), len(separation)), np.nan)
                planet_table[idx][f"StarINSTRUmag({instru})"]                      = mag_s
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal)"]           = mag_p_thermal
                planet_table[idx][f"PlanetINSTRUmag({instru})(reflected)"]         = mag_p_reflected
                planet_table[idx][f"PlanetINSTRUmag({instru})(thermal+reflected)"] = mag_p_total
                planet_table[idx]["contrast_5sigma"]                               = contrast_5sigma
    
        print(f"\n Calculating 5sigma-contrasts took {(time.time()-time1)/60:.1f} mn")
        planet_table.write(filename, format='ascii.ecsv', overwrite=True)
        fits.writeto(filename.replace(".ecsv", "_separation.fits"), separation, overwrite=True)
        print(f"\nSaving table: {filename}")

    return separation, planet_table



def yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", spectrum_contributions="reflected", force_table_calc=False, exposure_time=10*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3):
    
    from math import erf
    
    # --- Archive table of known exoplanets
    planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv")
    
    # --- Ranges "Earth-like" 
    R_min, R_max     = 0, 2     # [R_earth]
    M_min, M_max     = 0, 10    # [M_earth]
    Teff_min, Teff_max = 0, 500   # [K]
    
    def col(name):
        return planet_table[name].value if name in planet_table.colnames else None
    
    R    = col("PlanetRadius")        # R_earth
    M    = col("PlanetMass")          # M_earth
    Teff = col("PlanetTeff")           # K
    #a_AU = col("SMA")                 # AU
    n    = len(planet_table)
    mask_earth = np.ones(n, dtype=bool)
    print("Filtering Earth-Like planets")
    
    # Rayon & masse & Teff
    if R is not None:
        before      = int(mask_earth.sum())
        mask_earth &= np.isfinite(R) & (R >= R_min) & (R <= R_max)
        after       = int(mask_earth.sum())
        print(f" After radius filtering:      {after} / {n} (-{before - after})")
    if M is not None:
        before      = int(mask_earth.sum())
        mask_earth &= np.isfinite(M) & (M >= M_min) & (M <= M_max)
        after       = int(mask_earth.sum())
        print(f" After mass filtering:        {after} / {n} (-{before - after})")
    if Teff is not None:
        before      = int(mask_earth.sum())
        mask_earth &= np.isfinite(Teff) & (Teff >= Teff_min) & (Teff <= Teff_max)
        after       = int(mask_earth.sum())
        print(f" After temperature filtering: {after} / {n} (-{before - after})")
        
    planet_table = planet_table[mask_earth]
    
    # ---Retrieving tables
    separation_HARMONI, planet_table_HARMONI = get_planet_table_contrast(instru="HARMONI", planet_table=planet_table, exposure_time=exposure_time, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, Rc=Rc, filter_type="gaussian", systematics=False, PCA=False, N_PCA=20, force_table_calc=force_table_calc, sep_max=sep_max)
    separation_ANDES, planet_table_ANDES     = get_planet_table_contrast(instru="ANDES",   planet_table=planet_table, exposure_time=exposure_time, thermal_model=thermal_model, reflected_model=reflected_model, spectrum_contributions=spectrum_contributions, Rc=Rc, filter_type="gaussian", systematics=False, PCA=False, N_PCA=20, force_table_calc=force_table_calc, sep_max=sep_max)
    
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
    #planet_magnitude_total     = planet_table_HARMONI["PlanetINSTRUmag(HARMONI)(thermal+reflected)"].value # thermal + reflected contribution
    star_magnitude             = planet_table_HARMONI["StarINSTRUmag(HARMONI)"].value
    contrast_thermal_HARMONI   = 10**(-(planet_magnitude_thermal-star_magnitude)/2.5)
    contrast_reflected_HARMONI = 10**(-(planet_magnitude_reflected-star_magnitude)/2.5)
    
    # ANDES
    planet_magnitude_thermal   = planet_table_ANDES["PlanetINSTRUmag(ANDES)(thermal)"].value           # thermal contribution
    planet_magnitude_reflected = planet_table_ANDES["PlanetINSTRUmag(ANDES)(reflected)"].value         # reflected contribution
    #planet_magnitude_total     = planet_table_ANDES["PlanetINSTRUmag(ANDES)(thermal+reflected)"].value # thermal + reflected contribution
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
    plt.figure(figsize=(13.5, 6.5), dpi=300)
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
    
    import matplotlib.patheffects as pe
    
    # ---------------------- Ta taille de points ----------------------
    pm_valid  = planet_mass[np.isfinite(planet_mass) & (planet_mass > 0)]
    mass_minG = pm_valid.min()
    mass_maxG = pm_valid.max()
    s         = mass_to_size(planet_mass, s0=s0, ds=ds,
                             mass_min=mass_minG, mass_max=mass_maxG)
    
    # ---------------------- Figure / Axes ----------------------
    plt.figure(figsize=(13.5, 6.5), dpi=300)
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
    CS_H = draw_prob_contours(ax1, separation_HARMONI, P_H, y_grid, colors_instru["HARMONI"], levels=levels, lw=lw, z=0.4)
    CS_A = draw_prob_contours(ax1, separation_ANDES,   P_A, y_grid, colors_instru["ANDES"],   levels=levels, lw=lw, z=0.5)
    
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
    leg_planets = ax1.legend(fontsize=14, loc="upper left", frameon=True, edgecolor="gray", facecolor="white", title="Planet-light regime", title_fontsize=16)
    ax1.add_artist(leg_planets)
    
    leg_inst = ax1.legend(handles=[Line2D([0],[0], color=colors_instru["HARMONI"], lw=lw, ls='-',  label="HARMONI"), Line2D([0],[0], color=colors_instru["ANDES"],   lw=lw, ls='-',  label="ANDES")],
        loc="lower left", fontsize=14, frameon=True, edgecolor="gray",
        facecolor="white", title="Detection probability", title_fontsize=16)
    ax1.add_artist(leg_inst)
    
    for lg in (leg_planets, leg_inst):
        lg.set_zorder(100)
        lg.get_frame().set_alpha(1.0)
    
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

    






