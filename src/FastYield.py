"""
Some parts of this script are taken from PSISIM (see : https://github.com/planetarysystemsimager/psisim)
"""

import numpy as np
import astropy.units as u
import astropy.constants as constants
import scipy.interpolate as si
import pyvo
import json
import time
import os
import copy
from astropy.table import QTable, Table, Column, MaskedColumn
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import importlib
import warnings
#import picaso.opacity_factory as opa
import astropy.constants as consts
from src.spectrum import *
from src.FastCurves import *





bands = []
for instru in config_data_list :
    globals()["lmin_"+instru["name"]] = get_config_data(instru["name"])["lambda_range"]["lambda_min"]
    globals()["lmax_"+instru["name"]] = get_config_data(instru["name"])["lambda_range"]["lambda_max"]
    config_data = get_config_data(instru["name"])
    for name_band in config_data["gratings"]:
        if name_band not in bands :
            bands.append(name_band)
            globals()["lmin_"+name_band] = config_data['gratings'][name_band].lmin
            globals()["lmax_"+name_band] = config_data['gratings'][name_band].lmax

instru_with_systematics = ["MIRIMRS"]

path_file = os.path.dirname(__file__)
archive_path = os.path.join(os.path.dirname(path_file), "sim_data/Archive_table/")
simulated_path = os.path.join(os.path.dirname(path_file), "sim_data/Simulated_table/")


class Universe():
    '''
    A universe class that includes
    Inherited from EXOSIMS? TBD
    Properties:
    planets    - A planet table that holds all the planet properties [Astropy table]. It has the following columns:
    '''
    def __init__(self):
        '''
        '''
        pass 
class ExoArchive_Universe(Universe):
    '''
    A child class of Universe that is adapted to create a universe from known NASA Exoplanet Archive Data
    Uses the pyVO package to read in known exoplanets
    '''
    def __init__(self,table_filename):
        super(ExoArchive_Universe, self).__init__()
        self.filename = table_filename
        self.planets = None
        self.MJUP2EARTH = 317.82838    # conversion from Jupiter to Earth masses
        self.MSOL2EARTH = 332946.05    # conversion from Solar to Earth masses
        self.RJUP2EARTH = 11.209       # conversion from Jupiter to Earth radii
        #-- Chen & Kipping 2016 constants
            # ref.: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
        # Exponent terms from paper (Table 1)
        self._CKe0  = 0.279    # Terran 
        self._CKe1  = 0.589    # Neptunian 
        self._CKe2  =-0.044    # Jovian 
        self._CKe3  = 0.881    # Stellar 
        # Object-type transition points from paper (Table 1) - Earth-mass units
        self._CKMc0 = 2.04                   # terran-netpunian transition
        self._CKMc1 = 0.414*self.MJUP2EARTH  # neptunian-jovian transition 
        self._CKMc2 = 0.080*self.MSOL2EARTH  # jovian-stellar transition
        # Coefficient terms
        self._CKC0  = 1.008    # Terran - from paper (Table 1)
        self._CKC1  = 0.808    # Neptunian - computed from intercept with terran domain
        self._CKC2  = 17.74    # Jovian - computed from intercept neptunian domain
        self._CKC3  = 0.00143  # Stellar - computed from intercept with jovian domain
        #-- Thorngren 2019 Constants
            # ref.: https://doi.org/10.3847/2515-5172/ab4353
        # Coefficient terms from paper
        self._ThC0  = 0.96
        self._ThC1  = 0.21
        self._ThC2  =-0.20
        # Define Constraints
        self._ThMlow = 15            # [M_earth] Lower bound of applicability
        self._ThMhi  = 12*self.MJUP2EARTH # [M_earth] Upper bound of applicability
        self._ThThi  = 1000          # [K] Temperature bound of applicability
        
    def Load_ExoArchive_Universe(self, composite_table=True, force_new_pull=False, fill_empties=True):
        '''
        A function that reads the Exoplanet Archive data to populate the planet table
        Unless force_new_pull=True:
        If the filename provided in constructor is new, new data is pulled from the archive
        If the filename already exists, we try to load that file as an astroquery QTable
        Kwargs:
        composite_table  - Bool. True [default]: pull "Planetary Systems Composite
                           Parameters Table". False: pull simple "Planetary Systems" Table
                           NOTE: see Archive website for difference between these tables
        force_new_pull   - Bool. False [default]: loads table from filename if filename
                           file exists. True: pull new archive data and overwrite filename
        fill_empties     - Bool. True [default]: approximate empty table values using
                           other values present in data. Ex: radius, mass, logg, angsep, etc.
                           NOTE: When composite_table=True we do not approximate the planet 
                             radius or mass; we keep the archive-computed approx.
        Approximation methods:
        - AngSep     - theta[mas] = SMA[au]/distance[pc] * 1e3
        - logg       - logg [log(cgs)] = log10(G*mass/radius**2)
        - StarLum    - absVmag = Vmag - 5*log10(distance[pc]/10)
                       starlum[L/Lsun] = 10**-(absVmag-4.83)/2.5
        - StarRad    - rad[Rsun] = (5800/Teff[K])**2 *sqrt(starlum)
        - PlanetRad  - ** when composite_table=True, keep archive-computed approx
                       Based on Thorngren 2019 and Chen&Kipping 2016
        - PlanetMass - ^^ Inverse of PlanetRad
        *** Note: the resulting planet table will have nan's where data is missing/unknown. 
            Ex. if a planet lacks a radius val, the 'PlanetRadius' for will be np.nan        
        '''
        #-- Define columns to read. NOTE: add columns here if needed. 
          # col2pull entries should be matched with colNewNames entries
        if composite_table :
            col2pull =  "pl_name,hostname,pl_orbper,pl_orbsmax,pl_orbeccen,pl_orbincl,pl_bmasse,pl_bmasseerr1,pl_bmasseerr2,pl_bmasse_reflink,pl_rade,pl_radeerr1,pl_radeerr2,pl_rade_reflink," + \
                        "pl_eqt,pl_eqterr1,pl_eqterr2,pl_eqt_reflink,ra,dec,sy_dist,st_spectype,st_mass,st_teff," + \
                        "st_rad,st_logg,st_lum,st_age,st_vsin,st_radv," + \
                        "st_met,sy_plx,sy_bmag,sy_vmag,sy_rmag,sy_icmag," + \
                        "sy_jmag,sy_hmag,sy_kmag,discoverymethod,disc_year,disc_refname"
            colNewNames = ["PlanetName","StarName","Period","SMA","Ecc","Inc","PlanetMass","+DeltaPlanetMass","-DeltaPlanetMass","PlanetMassRef","PlanetRadius","+DeltaPlanetRadius","-DeltaPlanetRadius","PlanetRadiusRef",
                           "PlanetTeq","+DeltaPlanetTeq","-DeltaPlanetTeq","PlanetTeqRef","RA","Dec","Distance","StarSpT","StarMass","StarTeff",
                           "StarRad","StarLogg","StarLum","StarAge","StarVsini","StarRadialVelocity",
                           "StarZ","StarParallax","StarBMag","StarVmag","StarRmag","StarImag",
                           "StarJmag","StarHmag","StarKmag","DiscoveryMethod","DiscoveryYear","DiscoveryRef"]
        else :
            col2pull =  "pl_name,hostname,pl_orbsmax,pl_orbeccen,pl_orbincl,pl_bmasse,pl_rade," + \
                        "pl_eqt,ra,dec,sy_dist,st_spectype,st_mass,st_teff," + \
                        "st_rad,st_logg,st_lum,st_age,st_vsin,st_radv," + \
                        "st_met,sy_plx,sy_bmag,sy_vmag,sy_rmag,sy_icmag," + \
                        "sy_jmag,sy_hmag,sy_kmag,discoverymethod"
            colNewNames = ["PlanetName","StarName","SMA","Ecc","Inc","PlanetMass","PlanetRadius",
                           "PlanetTeq","RA","Dec","Distance","StarSpT","StarMass","StarTeff",
                           "StarRad","StarLogg","StarLum","StarAge","StarVsini","StarRadialVelocity",
                           "StarZ","StarParallax","StarBMag","StarVmag","StarRmag","StarImag",
                           "StarJmag","StarHmag","StarKmag","DiscoveryMethod"]
        #-- Load/Pull data depending on provided filename
        import os
        if os.path.isfile(self.filename) and not force_new_pull:
            # Existing filename was provided so let's try use that
            print("%s already exists:\n    we'll attempt to read this file as an astropy QTable"%self.filename)
            NArx_table = QTable.read(self.filename, format='ascii.ecsv')
            # Check that the provided table file matches the requested table type
            if NArx_table.meta['isPSCOMPPARS'] != composite_table:
                err0 = '%s contained the wrong table-type:'%self.filename
                err1 = 'pscomppars' if composite_table else 'ps'
                err2 = 'pscomppars' if NArx_table.meta['isPSCOMPPARS'] else 'ps'
                err3 = " Expected '{}' table but found '{}' table.".format(err1,err2)
                err4 = ' Consider setting force_new_pull=True.'
                raise ValueError(err0+err3+err4)
        else:
            # New filename was provided or a new pull was explicitly requested. Pull new data
            if not force_new_pull:
                print("%s does not exist:\n    we'll pull new data from the archive and save it to this filename"%self.filename)
            else:
                print("%s may or may not exist:\n    force_new_pull=True so we'll pull new data regardless and overwrite as needed"%self.filename) 
            # Import pyVO package used to query the Exoplanet Archive
            import pyvo as vo
            # Create a "service" which can be used to access the archive TAP server
            NArx_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
            # Create a "query" string formatted per the TAP specifications
              # 'select': specify which columns to pull
              # 'from': specify which table to pull 
              # 'where': (optional) specify parameters to be met when choosing what to pull
                # Add where flag for ps to only pull the best row for each planet
            tab2pull = "pscomppars" if composite_table else "ps where default_flag=1"
            query = "select "+col2pull+" from "+tab2pull
            # Pull the data and convert to astropy masked QTable
            NArx_res = NArx_service.search(query) 
            NArx_table = QTable(NArx_res.to_table())
            # Add a flag to the table metadata to denote what kind of table it was
              # This'll prevent trying to read the table as the wrong type later
            NArx_table.meta['isPSCOMPPARS'] = composite_table
            # Save raw table for future use 
            NArx_table.write(self.filename,format='ascii.ecsv',overwrite=force_new_pull)
            # Read table back in to ensure that formatting from a fresh pull matches
              # the formatting from an old pull (as done when filename exists)
            NArx_table = QTable.read(self.filename, format='ascii.ecsv')
        #-- Rename columns to psisim-expected names
        NArx_table.rename_columns(col2pull.split(','),colNewNames)
        #-- Change fill value from default 1e20 to np.nan
        for col in NArx_table.colnames:
            if isinstance(NArx_table[col],MaskedColumn) and isinstance(NArx_table[col].fill_value,(int,float)):
                # Only change numeric fill values to nan
                NArx_table[col].fill_value = np.nan
        #-- Add new columns for values not easily available or computable from table
          # TODO: for now, these are masked but we should find a good way to populate them
        NArx_table.add_columns([MaskedColumn(length=len(NArx_table),mask=True,fill_value=np.nan)]*3,names=['Flux Ratio','ProjAU','Phase'])
        if fill_empties:
            #-- Compute missing planet columns
            # Compute missing masses and radii using mass-radius relations
            if not composite_table:
                # NOTE: composite table already has radius-mass approximation so we'll
                  # only repeat them if we don't pull that table
                # Convert masked columns to ndarrays with 0's instead of mask
                  # as needed by the approximate_... functions
                #print("HERE1 =",len(NArx_table["PlanetRadius"][NArx_table["PlanetRadius"].mask]))
                masses   = np.array(NArx_table['PlanetMass'].filled(fill_value=0.0))
                radii    = np.array(NArx_table['PlanetRadius'].filled(fill_value=0.0))
                eqtemps  = np.array(NArx_table['PlanetTeq'].filled(fill_value=0.0))
                # Perform approximations
                radii = self.approximate_radii(masses,radii,eqtemps)
                masses = self.approximate_masses(masses,radii,eqtemps)
                # Create masks for non-zero values (0's are values where data was missing)
                rad_mask = (radii != 0.)
                mss_mask = (masses != 0.)
                # Create mask to only missing values in NArx_table with valid values
                rad_mask = NArx_table['PlanetRadius'].mask & rad_mask
                mss_mask = NArx_table['PlanetMass'].mask & mss_mask
                # Place results back in the table
                NArx_table['PlanetRadius'][rad_mask] = radii[rad_mask]*NArx_table['PlanetRadius'].unit
                NArx_table['PlanetMass'][mss_mask] = masses[mss_mask]*NArx_table['PlanetMass'].unit
                #print("HERE2 =",len(NArx_table["PlanetRadius"][NArx_table["PlanetRadius"].mask]))
            # Angular separation
            NArx_table['AngSep'] = NArx_table['SMA']/NArx_table['Distance'] * 1e3
            # Planet logg
            grav = constants.G * (NArx_table['PlanetMass'].filled(fill_value=np.nan)) / (NArx_table['PlanetRadius'].filled(fill_value=np.nan))**2
            NArx_table['PlanetLogg'] = np.ma.log10(MaskedColumn(np.ma.masked_invalid(grav.cgs.value),fill_value=np.nan))  # logg cgs
            #-- Guess star luminosity, radius, and gravity for missing (masked) values only
              # The guesses will be questionably reliable
            # Star Luminosity
            host_MVs = NArx_table['StarVmag'].value - 5*np.ma.log10(NArx_table['Distance'].value/10)  # absolute v mag
            host_lum = -(host_MVs-4.83)/2.5    #log10(L/Lsun)
            NArx_table['StarLum']=NArx_table['StarLum'].value
            NArx_table['StarLum'][NArx_table['StarLum'].mask] = host_lum[NArx_table['StarLum'].mask]
            # Star radius
            host_rad = (5800/NArx_table['StarTeff'])**2 *np.ma.sqrt(10**NArx_table['StarLum'])   # Rsun
            NArx_table['StarRad']=NArx_table['StarRad'].value
            NArx_table['StarRad'][NArx_table['StarRad'].mask] = host_rad[NArx_table['StarRad'].mask].value
            # Star logg
            host_grav = constants.G * (NArx_table['StarMass'].filled(fill_value=np.nan)*u.solMass) / (NArx_table['StarRad'].filled(fill_value=np.nan)*u.solRad)**2
            host_logg = np.ma.log10(np.ma.masked_invalid(host_grav.cgs.value))  # logg cgs
            NArx_table['StarLogg']=NArx_table['StarLogg'].value
            NArx_table['StarLogg'][NArx_table['StarLogg'].mask] = host_logg[NArx_table['StarLogg'].mask]
        else:
            # Create fully masked columns for AngSep and PlanetLogg
            NArx_table.add_columns([MaskedColumn(length=len(NArx_table),mask=True,fill_value=np.nan)]*2,names=['AngSep','PlanetLogg'])
        #-- Deal with units (conversions and Quantity multiplications)
        # Set host luminosity to L/Lsun from log10(L/Lsun)
        if fill_empties :
            NArx_table['StarLum'] = 10**NArx_table['StarLum']   # L/Lsun
        else :
            NArx_table['StarLum'] = 10**NArx_table['StarLum'].value    # L/Lsun
        # Make sure all number fill_values are np.nan after the column manipulations
        for col in NArx_table.colnames:
            if isinstance(NArx_table[col],MaskedColumn) and isinstance(NArx_table[col].fill_value,(int,float)):
                # Only change numeric fill values to nan
                NArx_table[col].fill_value = np.nan
        # Fill in masked values 
        #NArx_table = NArx_table.filled()
        # Apply units
        NArx_table['SMA'] = NArx_table['SMA'].value ; NArx_table['SMA'] *= u.AU # semi major axis
        NArx_table['Inc'] = NArx_table['Inc'].value ; NArx_table['Inc'] *= u.deg
        NArx_table['PlanetMass'] = NArx_table['PlanetMass'].value ; NArx_table['PlanetMass'] *= u.earthMass
        NArx_table['PlanetRadius'] = NArx_table['PlanetRadius'].value ; NArx_table['PlanetRadius'] *= u.earthRad
        NArx_table['PlanetTeq'] = NArx_table['PlanetTeq'].value ; NArx_table['PlanetTeq'] *= u.K
        NArx_table['RA'] = NArx_table['RA'].value ; NArx_table['RA'] *= u.deg
        NArx_table['Dec'] = NArx_table['Dec'].value ; NArx_table['Dec'] *= u.deg
        NArx_table['Distance'] = NArx_table['Distance'].value ; NArx_table['Distance'] *= u.pc
        NArx_table['StarMass'] = NArx_table['StarMass'].value ; NArx_table['StarMass'] *= u.solMass
        NArx_table['StarTeff'] = NArx_table['StarTeff'].value ; NArx_table['StarTeff'] *= u.K
        NArx_table['StarRad'] *= u.solRad
        if fill_empties :
            NArx_table['StarLogg'] = NArx_table['StarLogg'] * u.dex(u.cm/(u.s**2))
        else :
            NArx_table['StarLogg'] = NArx_table['StarLogg'].value * u.dex(u.cm/(u.s**2))
        NArx_table['StarLum'] *= u.solLum
        NArx_table['StarAge'] = NArx_table['StarAge'].value ; NArx_table['StarAge'] *= u.Gyr
        NArx_table['StarVsini'] = NArx_table['StarVsini'].value ; NArx_table['StarVsini'] *= u.km/u.s
        NArx_table['StarRadialVelocity'] = NArx_table['StarRadialVelocity'].value ; NArx_table['StarRadialVelocity'] *= u.km/u.s
        #NArx_table['StarZ']  *= u.dex
        NArx_table['StarParallax'] = NArx_table['StarParallax'].value ; NArx_table['StarParallax'] *= u.mas
        #print(NArx_table['ProjAU'].value)
        NArx_table['ProjAU'] = NArx_table['ProjAU'].value ; NArx_table['ProjAU'] = NArx_table['ProjAU']*u.AU
        NArx_table['Phase'] = NArx_table['Phase'].value ; NArx_table['Phase'] = NArx_table['Phase']*u.rad
        NArx_table['AngSep'] = NArx_table['AngSep'].value ; NArx_table['AngSep'] = NArx_table['AngSep']*u.mas
        NArx_table['PlanetLogg'] = NArx_table['PlanetLogg'].value ; NArx_table['PlanetLogg'] = NArx_table['PlanetLogg']*u.dex(u.cm/(u.s**2))
        self.planets = NArx_table
    
    def approximate_radii(self,masses,radii,eqtemps):
        '''
        Approximate planet radii given the planet masses
        Arguments:
        masses    - ndarray of planet masses
        radii     - ndarray of planet radii. 0-values will be replaced with approximation.
        eqtemps   - ndarray of planet equilibrium temperatures (needed for Thorngren constraints)
        Returns:
        radii     - ndarray of planet radii after approximation.
        Methodology:
        - Uses Thorngren 2019 for targets with 15M_E < M < 12M_J and T_eq < 1000 K.
            ref.: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
        - Uses Chen and Kipping 2016 for all other targets.
            ref.: https://doi.org/10.3847/2515-5172/ab4353
            
        * Only operates on 0-valued elementes in radii vector (ie. prioritizes Archive-provided radii).
        '''
        ##-- Find indices for missing radii so we don't replace Archive-provided values
        rad_mask = (radii == 0.0)        
        ##-- Compute radii assuming Chen&Kipping 2016 (for hot giants)
        # Compute radii for "Terran"-like planets
        ter_mask = (masses < self._CKMc0) # filter for terran-mass objects
        com_mask = rad_mask & ter_mask # planets in terran range and missing radius value
        radii[com_mask] = self._CKC0*(masses[com_mask]**self._CKe0)
        # Compute radii for "Neptune"-like planets
        nep_mask = (masses < self._CKMc1) # filter for neptune-mass objects
        com_mask = rad_mask & np.logical_not(ter_mask) & nep_mask # planets in neptune range and missing radius value
        radii[com_mask] = self._CKC1*(masses[com_mask]**self._CKe1)
        # Compute radii for "Jovian"-like planets
        jov_mask = (masses < self._CKMc2) # filter for jovian-mass objects
        com_mask = rad_mask & np.logical_not(nep_mask) & jov_mask # planets in jovian range and missing radius value
        radii[com_mask] = self._CKC2*(masses[com_mask]**self._CKe2)
        # Compute radii for "stellar" objects
        ste_mask = (masses > self._CKMc2) # filter for stellar-mass objects
        com_mask = rad_mask & ste_mask # planets in stellar range and missing radius value
        radii[com_mask] = self._CKC3*(masses[com_mask]**self._CKe3)
        ##-- Compute radii assuming Thorngren 2019 (for cool giants)
        # Create mask to find planets that meet the constraints
        Mlow_mask = (masses  > self._ThMlow)
        Mhi_mask  = (masses  < self._ThMhi)
        tmp_mask  = (eqtemps < self._ThThi) & (eqtemps != 0.0)  # omit temp=0 since those are actually empties
        com_mask  = rad_mask & Mlow_mask & Mhi_mask & tmp_mask
        # Convert planet mass vector to M_jup for equation
        logmass_com = np.log10(masses[com_mask]/self.MJUP2EARTH)
        # Apply equation to said planets (including conversion back to Rad_earth)
        radii[com_mask] = (self._ThC0 + self._ThC1*logmass_com + self._ThC2*(logmass_com**2))*self.RJUP2EARTH
        return radii
    
    def approximate_masses(self,masses,radii,eqtemps):
        '''
        Approximate planet masses given the planet radii
        Arguments:
        masses    - ndarray of planet masses. 0-values will be replaced with approximation.
        radii     - ndarray of planet radii
        eqtemps   - ndarray of planet equilibrium temperatures (needed for Thorngren constraints)
        Returns:
        masses    - ndarray of planet masses after approximation.
        Methodology:
        - Uses Thorngren 2019 for targets with ~ 3.7R_E < R < 10.7R_E and T_eq < 1000 K.
            ref.: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
        - Uses Chen and Kipping 2016 for all other targets.
            ref.: https://doi.org/10.3847/2515-5172/ab4353
            
        * Only operates on 0-valued elementes in masses vector (ie. prioritizes Archive-provided masses).
        '''
        ##-- Find indices for missing masses so we don't replace Archive-provided values
        mss_mask = (masses == 0.0)
        ##-- Compute masses assuming Chen&Kipping 2016 (for hot giants)
        # Transition points (in radii) - computed by solving at critical mass points
        R_TN = self._CKC1*(self._CKMc0**self._CKe1)
        R_NJ = self._CKC2*(self._CKMc1**self._CKe2)
        R_JS = self._CKC3*(self._CKMc2**self._CKe3)
        # Compute masses for Terran objects
            # These are far below Jovian range so no concern about invertibility
        ter_mask = (radii < R_TN) # filter for terran-size objects
        com_mask = mss_mask & ter_mask # planets in terran range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC0)**(1/self._CKe0)
        # Compute masses for Neptunian objects
            # Cut off computation at lower non-invertible radius limit (Jovian-stellar crit point)
        nep_mask = (radii < R_JS) # filter for neptune-size objects in invertible range
        com_mask = mss_mask & np.logical_not(ter_mask) & nep_mask # planets in invertible neptune range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC1)**(1/self._CKe1)
        # Ignore Jovian objects since in non-invertible range
        # Compute masses for Stellar objects
            # Cut off computation at upper non-invertible radius limit (Neptune-Jovian crit point)
        ste_mask = (radii > R_NJ) # filter for stellar-size objects in invertible range
        com_mask = mss_mask & ste_mask # planets in invertible stellar range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC3)**(1/self._CKe3)
        ##-- Compute masses assuming Thorngren 2019 (for cool giants)
        #- Use mass constraints to determine applicabile domain in radii
        # Convert constraint masses to M_jup for equation and compute log10 for simplicity in eq.
        log_M = np.log10(np.array([self._ThMlow,self._ThMhi])/self.MJUP2EARTH)
        # Apply equation (including conversion back to Rad_earth)
        cool_Rbd = (self._ThC0 + self._ThC1*log_M + self._ThC2*(log_M**2))*self.RJUP2EARTH
        # Extract bounds (in Earth radii) where Thorngren is applicable
        cool_Rlow = cool_Rbd[0]; cool_Rhi = cool_Rbd[1]; 
        # Create mask to find planets that meet the bounds
        Rlow_mask = (radii   > cool_Rlow)
        Rhi_mask  = (radii   < cool_Rhi)
        tmp_mask  = (eqtemps < self._ThThi) & (eqtemps != 0.0)  # omit temp=0 since those are actually empties
        com_mask  = mss_mask & Rlow_mask & Rhi_mask & tmp_mask
        # Convert planet radius vector to R_jup for equation
        rad_com = radii[com_mask]/self.RJUP2EARTH
        # Apply equation to said planets
            # Use neg. side of quad. eq. so we get the mass values on the left side of the curve
        logM    = (-1*self._ThC1 - np.sqrt(self._ThC1**2 - 4*self._ThC2*(self._ThC0-rad_com)))/(2*self._ThC2)
        masses[com_mask] = (10**logM)/self.MJUP2EARTH    # convert back to Earth mass
        return masses


def input_mag_imaging_planets(planet_table):
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="GJ 504 b"] = 19.4 #  https://arxiv.org/pdf/1807.00657.pdf (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="2MASS J01225093-2439505 b"] = 14.53 #  https://iopscience.iop.org/article/10.1088/0004-637X/774/1/55/pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HD 206893 b"] = 15.05 #  https://iopscience.iop.org/article/10.3847/1538-3881/abc263/pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="GSC 06214-00210 b"] = 14.87 #  https://arxiv.org/pdf/1503.07586.pdf (page 6)
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="1RXS J160929.1-210524 b"] = 16.15 #  https://arxiv.org/pdf/1503.07586.pdf (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HIP 78530 b"] = 14.18 #  https://arxiv.org/pdf/1503.07586.pdf (page 6)              
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HR 2562 b"] = 5.02+10.5 #  https://arxiv.org/pdf/1608.06660.pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HIP 65426 b"] = 6.771+9.85 #  https://arxiv.org/pdf/1707.01413.pdf (page 8)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="PDS 70 c"] = 8.542+8.8 #  https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="PDS 70 b"] = 8.542+8.0 #  https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HIP 21152 b"] = 16.55 #  https://iopscience.iop.org/article/10.3847/2041-8213/ac772f/pdf (page 3 bas)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="bet Pic b"] = 3.48+9.2 #  https://www.aanda.org/articles/aa/pdf/2011/04/aa16224-10.pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HR 8799 b"] = 14.05 #  https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HR 8799 c"] = 13.13 #  https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HR 8799 d"] = 13.11 #  https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HR 8799 e"] = 5.24+10.67  #  https://arxiv.org/ftp/arxiv/papers/1011/1011.4918.pdf (page 10)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HD 95086 b"] = 6.789+12.2  #  https://www.aanda.org/articles/aa/pdf/2022/08/aa43097-22.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="USco1621 b"] = 14.67  #  https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="USco1556 b"] =  14.85  #  https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="Oph 11 b"] =  14.44  #  https://arxiv.org/pdf/astro-ph/0608574.pdf (page 27)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="FU Tau b"] =  13.329  #  http://cdsportal.u-strasbg.fr/?target=FU%20Tau%20b            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="2MASS J12073346-3932539 b"] =  16.93  #  https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="AF Lep b"] =  4.926+11.7  #  https://arxiv.org/pdf/2302.06213.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="DH Tau b"] =  14.19  #  https://iopscience.iop.org/article/10.1086/427086/pdf (page 3 bas)            
    #planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="WD 0806-661 b"] =  k_band_mag("J",25,planet_table[planet_table["PlanetName"]=="WD 0806-661 b"])  # https://arxiv.org/pdf/1605.06655.pdf (page 10)          
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HIP 79098 AB b"] =  14.15  #  https://arxiv.org/pdf/1906.02787.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="VHS J125601.92-125723.9 b"] =  14.57  #  https://www.aanda.org/articles/aa/pdf/2023/02/aa44494-22.pdf (page 2)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="ROXs 42 B b"] =  15.01  #  https://iopscience.iop.org/article/10.1088/0004-637X/787/2/104/pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="2M0437 b"] =  17.21  #  https://arxiv.org/pdf/2110.08655.pdf (page 1)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="kap And b"] =  14.32  #  https://www.aanda.org/articles/aa/pdf/2014/02/aa22119-13.pdf (page 14)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="GU Psc b"] =  17.40  #  https://iopscience.iop.org/article/10.1088/0004-637X/787/1/5/pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="GQ Lup b"] =  13.5  #  https://watermark.silverchair.com/stu1586.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA4IwggN-BgkqhkiG9w0BBwagggNvMIIDawIBADCCA2QGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMe5lHC4E6oNfCKjMnAgEQgIIDNRBTyFAcCWw3I5edmpWsquUEeYqTdh9wSPyRjFSX8zAixWA69s-k7R2eYl4nU2vHPc3e6fOztwAJo0-QCF5tTT93oOxjfr7Ta533fcPZpjUiVYqKJttEITYHUEq2dKrhTepyhTRI08y-k09vTdzHLx-P61HYl12Xu5WB01fVn0_Ch21J-vy2C0mcMJt_wIAsBLvA6IUqf4dyiRljVQmL74dhgJhpSUOJnL3g2xMd0G-YN1JyWOtjTpjuCsczmncHC0vDIJmVuvgamYfR5E2BNeyY5QMjnb584CExKWTGOl4ON_CIrKNlvJI0gaInIvfLc_N_0ylQE5bPFlqe_j6bT7UuqzT7deIS4E-xnRV7t3PTA7JVo6WNKyhIs0n158LGaZMBdnxopxwHXhfmIlmkwr5mKfMYVPItmZXWJK3yGLPirvXard_TY0u34glhbsYXszUjvlQSTji58elFaTZBb9-eban2Jiz8hVsJ9JKnxE4tAKFNTWYXbEf-mxlFTQ0kh0X9sGIhMkunV5eW0VCiFtN_7DDYPROCCEKwTSWQSTz0JUH4eIzSGc7UN1gKbxjcbz5ZKpXc9oqh78ny1MigT9dX8qNpiP5AAzFohNGc1hYI_pjcF-XoJsudCR1Ig6YOQhRfxJ-EFFeAwdDOWpff2ffRp_A5scAVlSwTRqW6BPX22m99ocwwFJu4Nso4UxFzJ0d100Eszm5W792c3ZKwUcnKw2bccZz0sCk_VXCZLGVQG5vcPQY1KD1xDng9LFOkxIhIkhkRuR-o0TYZM7eIf039l8WzWsncCpQ8aV_oeMaA5Cq_qZ96H9fxSO6d3XPk9aV8KrUHsC77Ox2REXCjbusBy7wlgBgTjk01csXRuh5CDSHXLLy5GTMHAsrzh7XDfpbPhVIkVn9oOT7-KjuIAwZPErtct0tu2tBmsKO9DxQa8hCBwv0Zw2I3-yGY7IqqK4w5owijpuOo-Fk5hGCf8RMqI9eR7SWRl6FgA5CgP3nv_ZcgpJUAhMDBjwHKWds9F0rmrIU7y4dIb5pxM_JkOx8tYdM2bqwzSnJrcT2umGMCtHYoua3K0qLMCoJOUbSKfqjzpeYW (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="CT Cha b"] =  14.9  #  https://www.aanda.org/articles/aa/pdf/2008/43/aa8840-07.pdf (page 4)            
    #planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HD 206893 c"] =      
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="TYC 8998-760-1 c"] =  8.3+9.8  #  https://iopscience.iop.org/article/10.3847/2041-8213/aba27e/pdf (page 4)            
    #planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="COCONUTS-2 b"] =      
    #planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="51 Eri b"] =      
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HD 106906 b"] = 15.46  #  https://iopscience.iop.org/article/10.1088/2041-8205/780/1/L4/pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="CFHTWIR-Oph 98 b"] = 16.408  #  https://arxiv.org/pdf/2011.08871.pdf (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HD 203030 b"] = 16.21  #  https://iopscience.iop.org/article/10.3847/1538-3881/aa9711/pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="BD+60 1417 b"] = 15.645  #  https://iopscience.iop.org/article/10.3847/1538-4357/ac2499/pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"]=="HIP 75056 A b"] = 7.3+6.8  #  https://arxiv.org/pdf/2009.08537.pdf (page 3)            
    return planet_table


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
########################################### Création de la table d'archive: #########################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



def planet_index(planet_table,planet_name,TAP=True):
    for i in range(len(planet_table)):
        if TAP :
            if planet_table["PlanetName"][i]==planet_name:
                idx=i
        else :
            if planet_table["name"][i]==planet_name:
                idx=i
    return idx



def load_planet_table(table_name):
    if table_name[:7]=="Archive":
        planet_table = QTable.read(archive_path+table_name, format='ascii.ecsv')
    else :
        planet_table = QTable.read(simulated_path+table_name, format='ascii.ecsv')
    return planet_table



def create_archive_planet_table_raw():
    #-----------------------------------------------------------------------------------------------------------
    # Create universe from Exoplanet Archive data :
    #-------------------------------------------------------------------------------------
    # Filename in which to save raw exoplanet archive table
    archive_table_filename = archive_path+"Archive_Pull_raw.ecsv" 
    # Instantiate universe object
    uni = ExoArchive_Universe(archive_table_filename)
    # Pull and populate the planet table
    uni.Load_ExoArchive_Universe(force_new_pull=True)
    planet_table_raw = uni.planets
    #-- Estimate planet Teq when missing (except for direct imaged planets)
    ptq_mask = np.logical_not(planet_table_raw['PlanetTeq'].mask) # donne que les valeurs valides/existantes de Tp
    im_mask = planet_table_raw["DiscoveryMethod"]=="Imaging"
    nb_missing_Tp = len(planet_table_raw["PlanetTeq"][(~ptq_mask) & (~im_mask)]) # nb de températures manquantes (hors planètes en imagerie)
    planet_teq = (planet_table_raw['StarRad']/(planet_table_raw['SMA'])).decompose()**(1/2) * planet_table_raw['StarTeff'].value *u.K
    planet_table_raw["PlanetTeq"][(~ptq_mask) & (~im_mask)] = planet_teq[(~ptq_mask) & (~im_mask)] # on estime les températures manquantes (sauf pour les planètes détéctées en imagerie directe car la température serait grandement sous-estimée)
    prd_mask = np.logical_not(planet_table_raw['PlanetRadius'].mask) # donne que les valeurs valides/existantes 
    planet_table_raw["PlanetRadius"][(~prd_mask) & im_mask] = 1.111111111 * planet_table_raw["PlanetRadius"].unit # on se moque du rayon planétaire pour les planètes en imagerie directe (car on renormalise le spectre avec la magnitude mesurée en bande K) => il faut juste éviter de mettre 0
    # Create masks for missing entries for contrast calculation
    slg_mask = np.logical_not(np.isnan(planet_table_raw['StarLogg']))
    stq_mask = np.logical_not(planet_table_raw['StarTeff'].mask)
    plg_mask = np.logical_not(np.isnan(planet_table_raw['PlanetLogg'])) 
    ptq_mask = np.logical_not(planet_table_raw['PlanetTeq'].mask)
    stk_mask = np.logical_not(planet_table_raw['StarKmag'].mask)
    prd_mask = np.logical_not(planet_table_raw['PlanetRadius'].mask)
    dis_mask = np.logical_not(planet_table_raw['Distance'].mask)
    sma_mask = np.logical_not(planet_table_raw['SMA'].mask)
    planet_table_raw = planet_table_raw[slg_mask & stq_mask & plg_mask & ptq_mask & stk_mask & prd_mask & dis_mask & sma_mask]
    planet_table_raw["Phase"] = np.pi/2 * planet_table_raw["Phase"].unit # on suppose que les planètes sont à leur élongation max (phi = pi/2)
    print('\n Estimating the temperature for : ',nb_missing_Tp,"planets")
    print('\n Nb of planets for which the SNR value can be obtained : %d'%len(planet_table_raw[np.logical_not(planet_table_raw['AngSep'].mask)]))
    planet_table_raw.write(archive_path+"Archive_Pull_raw.ecsv",format='ascii.ecsv',overwrite=True)

def create_simulated_planet_table_raw():
    import EXOSIMS.MissionSim
    import EXOSIMS.SimulatedUniverse.SAG13Universe
    filename = "sim_data/Simulated_table/FastYield_sim_EXOCAT1.json"
    with open(filename) as ff:
        specs = json.loads(ff.read())
    su = EXOSIMS.SimulatedUniverse.SAG13Universe.SAG13Universe(**specs)
    flux_ratios = 10**(su.dMag/-2.5)  # grab for now from EXOSIMS
    angseps = su.WA.value * 1000 *u.mas # mas
    projaus = su.d.value * u.AU # au
    phase = np.arccos(su.r[:,2]/su.d)# planet phase  [0, pi]
    smas = su.a.value*u.AU # au
    eccs = su.e # eccentricity
    incs = su.I.value*u.deg # degrees
    masses = su.Mp  # earth masses
    radii = su.Rp # earth radii
    grav = constants.G * (masses)/(radii)**2
    logg = np.log10(grav.to(u.cm/u.s**2).value)*u.dex(u.cm/u.s**2) # logg cgs
    # stellar properties
    ras = [] # deg
    decs = [] # deg
    distances = [] # pc
    for index in su.plan2star:
        coord = su.TargetList.coords[index]
        ras.append(coord.ra.value)
        decs.append(coord.dec.value)
        distances.append(coord.distance.value)
    ras = np.array(ras)
    decs = np.array(decs)
    distances = np.array(distances) * u.pc
    star_names =  np.array([su.TargetList.Name[i] for i in su.plan2star])
    planet_names = np.copy(star_names)
    planet_types = np.copy(planet_names)
    for i in range(len(star_names)):
        k = 1
        pname = np.char.add(star_names[i],f" {k}")
        while pname in planet_names :
            pname = np.char.add(star_names[i],f" {k}")
            k+=1
        planet_names[i] = pname #np.append(planet_names,pname)
        if masses[i] < 2.0 * u.earthMass :
            planet_types[i] = "Terran"
        elif 2.0 * u.earthMass < masses[i] < 0.41 * u.jupiterMass :
            planet_types[i] = "Neptunian"
        elif 0.41 * u.jupiterMass < masses[i] < 0.80 * u.solMass :
            planet_types[i] = "Jovian"
        elif 0.80 * u.solMass < masses[i] :
            planet_types[i] = "Stellar"
        print("giving name and type : " , planet_names[i]," & ", planet_types[i], " : " , round(100*(i+1)/len(star_names),3)," %")
    spts = np.array([su.TargetList.Spec[i] for i in su.plan2star])
    su.TargetList.stellar_mass() # generate masses if haven't
    host_mass = np.array([su.TargetList.MsTrue[i].value for i in su.plan2star]) * u.solMass
    su.TargetList.stellar_Teff()
    host_teff = np.array([su.TargetList.Teff[i].value for i in su.plan2star]) * u.K
    host_Vmags = np.array([su.TargetList.Vmag[i] for i in su.plan2star])
    host_Kmags = np.array([su.TargetList.Kmag[i] for i in su.plan2star]) * u.mag
    # guess the radius and gravity from Vmag and Teff. This is of questionable reliability
    host_MVs = host_Vmags - 5 * np.log10(distances.value/10) # absolute V mag
    host_lums = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
    host_radii = (5800/host_teff.value)**2 * np.sqrt(host_lums) * u.solRad# Rsun
    host_gravs = constants.G * host_mass/(host_radii**2)
    host_logg = np.log10(host_gravs.to(u.cm/u.s**2).value) * u.dex(u.cm/(u.s**2))# logg cgs
    teq = su.PlanetPhysicalModel.calc_Teff(host_lums,smas,su.p)
    all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, planet_names, planet_types, masses, radii, teq, logg, spts, host_mass, host_teff, host_radii, host_logg, host_Kmags]
    labels = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetName", "PlanetType", "PlanetMass", "PlanetRadius", "PlanetTeq", "PlanetLogg", "StarSpT", "StarMass", "StarTeff", "StarRad", "StarLogg","StarKmag"]
    planet_table_raw = QTable(all_data, names=labels)
    slg_mask = np.logical_not(np.isnan(planet_table_raw['StarLogg']))
    stq_mask = np.logical_not(np.isnan(planet_table_raw['StarTeff']))
    plg_mask = np.logical_not(np.isnan(planet_table_raw['PlanetLogg'])) 
    ptq_mask = np.logical_not(np.isnan(planet_table_raw['PlanetTeq']))
    stk_mask = np.logical_not(np.isnan(planet_table_raw['StarKmag']))
    prd_mask = np.logical_not(np.isnan(planet_table_raw['PlanetRadius']))
    dis_mask = np.logical_not(np.isnan(planet_table_raw['Distance']))
    sma_mask = np.logical_not(np.isnan(planet_table_raw['SMA']))
    planet_table_raw = planet_table_raw[slg_mask & stq_mask & plg_mask & ptq_mask & stk_mask & prd_mask & dis_mask & sma_mask]
    planet_table_raw["PlanetTeq"] = (planet_table_raw['StarRad']/(planet_table_raw['SMA'])).decompose()**(1/2) * planet_table_raw['StarTeff'].value *u.K
    planet_table_raw["DiscoveryMethod"] = np.full((len(planet_table_raw),), "None")
    planet_table_raw["StarRadialVelocity"] = np.full((len(planet_table_raw),), 0) * u.km / u.s
    planet_table_raw["StarVsini"] = np.full((len(planet_table_raw),), 0) * u.km / u.s
    planet_table_raw.write(simulated_path+"Simulated_Pull_raw.ecsv",format='ascii.ecsv',overwrite=True)


def create_fastcurves_table(table="Archive"): # take ~ 5 minutes for Archive and more for Simulated
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    time1 = time.time()
    if table=="Archive":
        create_archive_planet_table_raw()
        planet_table = load_planet_table("Archive_Pull_raw.ecsv")
    elif table=="Simulated":
        create_simulated_planet_table_raw()
        planet_table_raw = load_planet_table("Simulated_Pull_raw.ecsv")        
        print("Raw number of planets = ",len(planet_table_raw))
        n_planets = len(planet_table_raw)
        n_planets_now = int(n_planets/10) 
        rand_planets = np.random.randint(0, n_planets, n_planets_now)
        planet_table = planet_table_raw[rand_planets]
        plt.figure() ; plt.grid(True) ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel(f"angular separation (in {planet_table['AngSep'].unit})") ; plt.ylabel(f"flux ratio")
        plt.scatter(planet_table_raw["AngSep"],planet_table_raw["Flux Ratio"],c='k',alpha=0.333,zorder=10)   
        plt.scatter(planet_table["AngSep"],planet_table["Flux Ratio"],c='r',alpha=0.333,zorder=10) 
        plt.show()
        plt.figure() ; plt.grid(True) ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel(f"planet mass (in {planet_table['PlanetMass'].unit})") ; plt.ylabel(f"planet radius (in {planet_table['PlanetRadius'].unit})")
        plt.scatter(planet_table_raw["PlanetMass"],planet_table_raw["PlanetRadius"],c='k',alpha=0.333,zorder=10)
        plt.scatter(planet_table["PlanetMass"],planet_table["PlanetRadius"],c='r',alpha=0.333,zorder=10)
        plt.show()
        plt.figure() ; plt.grid(True) ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel(f"semi major axis (in {planet_table['SMA'].unit})") ; plt.ylabel(f"planet radius (in {planet_table['PlanetRadius'].unit})")
        plt.scatter(planet_table_raw["SMA"],planet_table_raw["PlanetRadius"],c='k',alpha=0.333,zorder=10)
        plt.scatter(planet_table["SMA"],planet_table["PlanetRadius"],c='r',alpha=0.333,zorder=10)
        plt.show()
    print('\n Total nb of planets : %d'%len(planet_table))
    min_Tp = 200 * u.K # on fixe une température minimale (celle des spectres BT-Settl, Morley et SONORA)
    planet_table["PlanetTeq"][planet_table["PlanetTeq"] < min_Tp ] = min_Tp
    # on tire aléatoirement les vitesses radiales des étoiles (uniformément entre -30 et 30 km/s) lorsqu'elles sont manquantes
    planet_table["StarRadialVelocity"][np.array(planet_table["StarRadialVelocity"].value)==0 ] = (np.random.uniform(size = len(planet_table["StarRadialVelocity"][np.array(planet_table["StarRadialVelocity"].value)==0]))*60-30 ) * planet_table["StarRadialVelocity"].unit
    planet_table["StarVsini"][np.array(planet_table["StarVsini"].value)==0 ] = (np.random.uniform(size=len(planet_table["StarVsini"][np.array(planet_table["StarVsini"].value)==0]))*20 ) * planet_table["StarVsini"].unit # 20 = np.nanmean(planet_table["StarVsini"]) + np.std(planet_table["StarVsini"])
    # on tire aléatoirement les inclinaisons (uniformément entre 0 et 180°) lorsqu'elles sont manquantes
    planet_table["Inc"][np.array(planet_table["Inc"].value)==0 ] = np.random.uniform(size = len(planet_table["Inc"][np.array(planet_table["Inc"].value)==0])) * 180 * planet_table["Inc"].unit
    i = np.array(planet_table["Inc"].value) * np.pi/180 # on estime le décalage Doppler entre la planète et l'étoile en suposant qu'il s'agit de la vitesse orbitale (circulaire) * sin(i)
    planet_table['DeltaRadialVelocity'] = np.sqrt(const.G*(planet_table["StarMass"]+planet_table["PlanetMass"])/planet_table["SMA"]).decompose().to(u.km/u.s) * np.sin(i)
    planet_table["alpha"] = np.arccos(-np.sin(i)*np.cos(np.array(planet_table["Phase"].value)))
    planet_table["g_alpha"] = (np.sin(planet_table["alpha"])+(np.pi-planet_table["alpha"])*np.cos(planet_table["alpha"]))/np.pi # fonction de phase de Lambert
    for instru in config_data_list :
        planet_table['StarINSTRUmag('+instru["name"]+')'] = np.full((len(planet_table),), np.nan)
        planet_table['PlanetINSTRUmag('+instru["name"]+')(thermal+reflected)'] = np.full((len(planet_table),), np.nan)
        planet_table['PlanetINSTRUmag('+instru["name"]+')(thermal)'] = np.full((len(planet_table),), np.nan)
        planet_table['PlanetINSTRUmag('+instru["name"]+')(reflected)'] = np.full((len(planet_table),), np.nan)
    for band in bands :
        if band == "K":
            planet_table['StarKmag'] = planet_table['StarKmag'].value
        else :
            planet_table['Star'+band+'mag'] = np.full((len(planet_table),), np.nan)
        planet_table['Planet'+band+'mag(thermal+reflected)'] = np.full((len(planet_table),), np.nan)
        planet_table['Planet'+band+'mag(thermal)'] = np.full((len(planet_table),), np.nan)
        planet_table['Planet'+band+'mag(reflected)'] = np.full((len(planet_table),), np.nan)
    if table=="Archive":
        planet_table = input_mag_imaging_planets(planet_table) # on rentre les valeurs connues des magnitudes (en bande K) des planètes détectées par imagerie directe
    for instru in config_data_list :
        globals()["lmin_"+instru["name"]] = get_config_data(instru["name"])["lambda_range"]["lambda_min"]
        globals()["lmax_"+instru["name"]] = get_config_data(instru["name"])["lambda_range"]["lambda_max"]
    wave = np.arange(0.1,12,1e-4)
    vega_spectrum = load_vega_spectrum()
    vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave, renorm = False)
    for idx in range(len(planet_table)):
        print("\n",planet_table[idx]["PlanetName"]," : ",round(100*(idx+1)/len(planet_table),2),"%")
        planet_spectrum , planet_thermal , planet_reflected , star_spectrum = thermal_reflected_spectrum(planet_table[idx],instru=None,thermal_model="BT-Settl",reflected_model="PICASO",wave=wave,vega_spectrum=vega_spectrum,show=False)
        for instru in config_data_list :
            planet_table[idx]['StarINSTRUmag('+instru["name"]+')'] = -2.5*np.log10(np.nanmean(star_spectrum.flux[(wave>globals()["lmin_"+instru["name"]])&(wave<globals()["lmax_"+instru["name"]])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+instru["name"]]) & (wave<globals()["lmax_"+instru["name"]])]))
            planet_table[idx]['PlanetINSTRUmag('+instru["name"]+')(thermal+reflected)'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave>globals()["lmin_"+instru["name"]])&(wave<globals()["lmax_"+instru["name"]])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+instru["name"]]) & (wave<globals()["lmax_"+instru["name"]])]))
            planet_table[idx]['PlanetINSTRUmag('+instru["name"]+')(thermal)'] = -2.5*np.log10(np.nanmean(planet_thermal.flux[(wave>globals()["lmin_"+instru["name"]])&(wave<globals()["lmax_"+instru["name"]])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+instru["name"]]) & (wave<globals()["lmax_"+instru["name"]])]))
            planet_table[idx]['PlanetINSTRUmag('+instru["name"]+')(reflected)'] = -2.5*np.log10(np.nanmean(planet_reflected.flux[(wave>globals()["lmin_"+instru["name"]])&(wave<globals()["lmax_"+instru["name"]])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+instru["name"]]) & (wave<globals()["lmax_"+instru["name"]])]))
        for band in bands :
            if band != "K":
                planet_table[idx]['Star'+band+'mag'] = -2.5*np.log10(np.nanmean(star_spectrum.flux[(wave>globals()["lmin_"+band])&(wave<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+band]) & (wave<globals()["lmax_"+band])]))
            planet_table[idx]['Planet'+band+'mag(thermal+reflected)'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave>globals()["lmin_"+band])&(wave<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+band]) & (wave<globals()["lmax_"+band])]))
            planet_table[idx]['Planet'+band+'mag(thermal)'] = -2.5*np.log10(np.nanmean(planet_thermal.flux[(wave>globals()["lmin_"+band])&(wave<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+band]) & (wave<globals()["lmax_"+band])])) 
            planet_table[idx]['Planet'+band+'mag(reflected)'] = -2.5*np.log10(np.nanmean(planet_reflected.flux[(wave>globals()["lmin_"+band])&(wave<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave>globals()["lmin_"+band]) & (wave<globals()["lmax_"+band])])) 
        print(" mag(K)_p_total = ", round(planet_table[idx]['PlanetKmag(thermal+reflected)'],3))
        print(" mag(K)_p_thermal = ", round(planet_table[idx]['PlanetKmag(thermal)'],3))
        print(" mag(K)_p_reflected = ", round(planet_table[idx]['PlanetKmag(reflected)'],3))
    print('\n Nb of planets for FastCurves calculations : %d'%len(planet_table))
    print('\n Generating the table took {0:.3f} s'.format(time.time()-time1))
    if table == "Archive":
        planet_table.write(archive_path+"Archive_Pull_for_FastCurves.ecsv",format='ascii.ecsv',overwrite=True)
    else :
        planet_table.write(simulated_path+"Simulated_Pull_for_FastCurves.ecsv",format='ascii.ecsv',overwrite=True)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
########################################### CALCUL SNR : ##############################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



def calculate_SNR_table(instru,table="Archive",thermal_model="None",reflected_model="None",apodizer="NO_SP",strehl="NO_JQ",systematic=False):
    exposure_time=120
    time1 = time.time()
    if table=="Archive":
        planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
        planet_table = planet_table[np.logical_not(planet_table['AngSep'].mask)]
        path = archive_path
    elif table=="Simulated":
        planet_table = load_planet_table("Simulated_Pull_for_FastCurves.ecsv")
        if instru=="HARMONI" or instru=="ERIS":
            planet_table = planet_table[(90-np.abs(-24.6-planet_table["Dec"].value))>30]
        path = simulated_path
    config_data = get_config_data(instru)
    if instru == "HARMONI":
        iwa = config_data["apodizers"][apodizer].sep * u.mas
    else :
        lambda_c = config_data["lambda_range"]["lambda_min"] *1e-6*u.m #(config_data["lambda_range"]["lambda_max"]+config_data["lambda_range"]["lambda_min"])/2 *1e-6*u.m
        diameter = config_data['telescope']['diameter'] *u.m
        iwa = lambda_c/diameter*u.rad ; iwa = iwa.to(u.mas)
    planet_table = planet_table[planet_table['AngSep'] > iwa] # on filtre les planètes en dessous de l'iwa (fixé ~ par la FWHM de la PSF)
    
    if thermal_model == "Exo-REM": # on filtre à la range en température d'Exo-REM
        planet_table = planet_table[(planet_table['PlanetTeq'] > 400*u.K) & (planet_table['PlanetTeq'] < 2000*u.K)]
    
    if thermal_model == "None" :
        spectrum_contributions = "reflected"
        name_model = reflected_model
        if name_model == "PICASO":
            name_model += "_reflected_only"
    elif reflected_model == "None" :
        spectrum_contributions = "thermal"
        name_model = thermal_model
        if name_model == "PICASO":
            name_model += "_thermal_only"
    elif thermal_model == "None" and reflected_model == "None" :
        raise KeyError("PLEASE DEFINE A MODEL FOR THE THERMAL OR THE REFLECTED COMPONENT !")
    elif thermal_model != "None" and reflected_model != "None":
        spectrum_contributions = "thermal+reflected"
        name_model = thermal_model+"+"+reflected_model
    
    print('\n Nb of planets for which SNR is calculated for '+instru+': %d'%len(planet_table))

    planet_table['signal_INSTRU'] = np.full((len(planet_table),), np.nan)
    planet_table['sigma_non-syst_INSTRU'] = np.full((len(planet_table),), np.nan)
    planet_table['sigma_syst_INSTRU'] = np.full((len(planet_table),), np.nan)
    planet_table['DIT_INSTRU'] = np.full((len(planet_table),), np.nan)
    for band in bands :
        planet_table['signal_'+band] = np.full((len(planet_table),), np.nan)
        planet_table['sigma_non-syst_'+band] = np.full((len(planet_table),), np.nan)
        planet_table['sigma_syst_'+band] = np.full((len(planet_table),), np.nan)
        planet_table['DIT_'+band] = np.full((len(planet_table),), np.nan)

        
    lmin_instru = get_config_data(instru)["lambda_range"]["lambda_min"] ; lmax_instru = get_config_data(instru)["lambda_range"]["lambda_max"]
    R = 200000 # environ la résolution des spectres BT-Settl / BT-NextGen (pas nécessaire de mettre plus)
    delta_lamb_instru = ((max(lmax_K,lmax_instru)+min(lmin_K,lmin_instru))/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave = np.arange(0.98*min(lmin_K,lmin_instru), 1.02*max(lmax_K,lmax_instru), delta_lamb_instru)
    vega_spectrum = load_vega_spectrum()
    vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave, renorm = False)
    
    for idx in range(len(planet_table)):
        if systematic :
            print("\n "+instru+" with systematics ("+thermal_model+" & "+reflected_model+") : ",planet_table[idx]["PlanetName"]," / ", round(100*(idx+1)/len(planet_table),2),"%")
        else :
            print("\n "+instru+" without systematics ("+thermal_model+" & "+reflected_model+") : ",planet_table[idx]["PlanetName"]," / ", round(100*(idx+1)/len(planet_table),2),"%")
        planet_spectrum , planet_thermal , planet_reflected , star_spectrum = thermal_reflected_spectrum(planet_table[idx],instru,thermal_model=thermal_model,reflected_model=reflected_model,wave=wave,vega_spectrum=vega_spectrum,show=False)
        if spectrum_contributions=="thermal":
            planet_spectrum = planet_thermal
        elif spectrum_contributions=="reflected":
            planet_spectrum = planet_reflected
                
        # on recalcule la magnitude dans le cas où le modèle thermique n'est plus BT-Settl où que le modèle réfléchi n'est plus PICASO (la mag change)
        planet_table[idx]['PlanetINSTRUmag('+instru+')('+spectrum_contributions+')'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave>lmin_instru)&(wave<lmax_instru)])/np.nanmean(vega_spectrum.flux[(vega_spectrum.wavelength>lmin_instru) & (vega_spectrum.wavelength<lmax_instru)]))
        for band in bands :
            if lmin_instru < globals()["lmin_"+band] and globals()["lmax_"+band] < lmax_instru :
                planet_table[idx]['Planet'+band+'mag('+spectrum_contributions+')'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave>globals()["lmin_"+band])&(wave<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(vega_spectrum.wavelength>globals()["lmin_"+band]) & (vega_spectrum.wavelength<globals()["lmax_"+band])]))
    
        mag_p = planet_table[idx]["PlanetINSTRUmag("+instru+")("+spectrum_contributions+")"]
        mag_s = planet_table[idx]["StarINSTRUmag("+instru+")"]
        band0= "instru"
        
        if instru=="HARMONI":
            name_band,SNR_planet,signal_planet,sigma_ns_planet,sigma_s_planet,DIT_band=harmoni(calculation="SNR",systematic=systematic,T_planet=float(planet_table[idx]["PlanetTeq"].value),lg_planet=float(planet_table[idx]["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(planet_table[idx]["StarTeff"].value),lg_star=float(planet_table[idx]["StarLogg"].value),syst_radial_velocity=float(planet_table[idx]["StarRadialVelocity"].value),delta_radial_velocity=float(planet_table[idx]["DeltaRadialVelocity"].value),star_broadening=float(planet_table[idx]["StarVsini"].value),exposure_time=exposure_time,model=name_model,mag_planet=mag_p,separation_planet=float(planet_table[idx]["AngSep"].value/1000),name_planet=planet_table[idx]["PlanetName"],return_SNR_planet=True,show_plot=False,print_value=False,planet_spectrum=planet_spectrum,star_spectrum=star_spectrum,apodizer=apodizer,strehl=strehl)
        elif instru=="ERIS":
            name_band,SNR_planet,signal_planet,sigma_ns_planet,sigma_s_planet,DIT_band=eris(calculation="SNR",systematic=systematic,T_planet=float(planet_table[idx]["PlanetTeq"].value),lg_planet=float(planet_table[idx]["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(planet_table[idx]["StarTeff"].value),lg_star=float(planet_table[idx]["StarLogg"].value),syst_radial_velocity=float(planet_table[idx]["StarRadialVelocity"].value),delta_radial_velocity=float(planet_table[idx]["DeltaRadialVelocity"].value),star_broadening=float(planet_table[idx]["StarVsini"].value),exposure_time=exposure_time,model=name_model,mag_planet=mag_p,separation_planet=float(planet_table[idx]["AngSep"].value/1000),name_planet=planet_table[idx]["PlanetName"],return_SNR_planet=True,show_plot=False,print_value=False,planet_spectrum=planet_spectrum,star_spectrum=star_spectrum,strehl=strehl)
        elif instru=="MIRIMRS":
            name_band,SNR_planet,signal_planet,sigma_ns_planet,sigma_s_planet,DIT_band=mirimrs(calculation="SNR",systematic=systematic,T_planet=float(planet_table[idx]["PlanetTeq"].value),lg_planet=float(planet_table[idx]["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(planet_table[idx]["StarTeff"].value),lg_star=float(planet_table[idx]["StarLogg"].value),syst_radial_velocity=float(planet_table[idx]["StarRadialVelocity"].value),delta_radial_velocity=float(planet_table[idx]["DeltaRadialVelocity"].value),star_broadening=float(planet_table[idx]["StarVsini"].value),exposure_time=exposure_time,model=name_model,mag_planet=mag_p,separation_planet=float(planet_table[idx]["AngSep"].value/1000),name_planet=planet_table[idx]["PlanetName"],return_SNR_planet=True,show_plot=False,print_value=False,star_spectrum=star_spectrum) # pas besoin d'injecter le spectre planétaire pour MIRI car il n'y a pas de composante réfléchie           
        elif instru=="NIRCam":
            name_band,SNR_planet,signal_planet,sigma_ns_planet,sigma_s_planet,DIT_band=nircam(calculation="SNR",systematic=systematic,T_planet=float(planet_table[idx]["PlanetTeq"].value),lg_planet=float(planet_table[idx]["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(planet_table[idx]["StarTeff"].value),lg_star=float(planet_table[idx]["StarLogg"].value),syst_radial_velocity=float(planet_table[idx]["StarRadialVelocity"].value),delta_radial_velocity=float(planet_table[idx]["DeltaRadialVelocity"].value),star_broadening=float(planet_table[idx]["StarVsini"].value),exposure_time=exposure_time,model=name_model,mag_planet=mag_p,separation_planet=float(planet_table[idx]["AngSep"].value/1000),name_planet=planet_table[idx]["PlanetName"],return_SNR_planet=True,show_plot=False,print_value=False,planet_spectrum=planet_spectrum,star_spectrum=star_spectrum)
        elif instru=="NIRSpec":
            name_band,SNR_planet,signal_planet,sigma_ns_planet,sigma_s_planet,DIT_band=nirspec(calculation="SNR",systematic=systematic,T_planet=float(planet_table[idx]["PlanetTeq"].value),lg_planet=float(planet_table[idx]["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(planet_table[idx]["StarTeff"].value),lg_star=float(planet_table[idx]["StarLogg"].value),syst_radial_velocity=float(planet_table[idx]["StarRadialVelocity"].value),delta_radial_velocity=float(planet_table[idx]["DeltaRadialVelocity"].value),star_broadening=float(planet_table[idx]["StarVsini"].value),exposure_time=exposure_time,model=name_model,mag_planet=mag_p,separation_planet=float(planet_table[idx]["AngSep"].value/1000),name_planet=planet_table[idx]["PlanetName"],return_SNR_planet=True,show_plot=False,print_value=False,planet_spectrum=planet_spectrum,star_spectrum=star_spectrum)
        else :
            raise KeyError("PLEASE DEFINE INSTRUMENT CALCULATION FUNCTION !")
            
        if systematic :
            SNR = signal_planet / sigma_s_planet / np.sqrt(DIT_band) # 
        else :
            SNR = signal_planet / sigma_ns_planet / np.sqrt(DIT_band) # snr / sqrt(mn)
        idx_max = SNR.argmax()
        planet_table[idx]['signal_INSTRU'] = signal_planet[idx_max]
        planet_table[idx]['sigma_non-syst_INSTRU'] = sigma_ns_planet[idx_max]
        planet_table[idx]['sigma_syst_INSTRU'] = sigma_s_planet[idx_max]
        planet_table[idx]['DIT_INSTRU'] = DIT_band[idx_max]
        for nb,band in enumerate(name_band) :
            planet_table[idx]['signal_'+band] = signal_planet[nb]
            planet_table[idx]['sigma_non-syst_'+band] = sigma_ns_planet[nb]
            planet_table[idx]['sigma_syst_'+band] = sigma_s_planet[nb]
            planet_table[idx]['DIT_'+band] = DIT_band[nb]
    print('\n Calculating SNR took {0:.3f} s'.format(time.time()-time1))
    if systematic :
        planet_table.write(path+table+"_Pull_"+instru+"_"+apodizer+"_"+strehl+"_with_systematics_"+name_model+".ecsv",format='ascii.ecsv',overwrite=True)
    else :
        planet_table.write(path+table+"_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv",format='ascii.ecsv',overwrite=True)




def all_SNR_table(table="Archive"):
    for instru in config_data_list :
        instru = instru["name"] ; config_data = get_config_data(instru)
        if config_data["lambda_range"]["lambda_max"] > 6 :
            thermal_models = ["None","BT-Settl","Exo-REM"]
            reflected_models = ["None"]
        else : 
            thermal_models = ["None","BT-Settl","Exo-REM","PICASO"]
            reflected_models = ["None","flat","tellurics","PICASO"]
        apodizer = "NO_SP"
        if instru == "HARMONI" or instru == "ANDES" :
            strehl = "JQ1"
        elif instru == "ERIS":
            strehl = "JQ0"
        else :
            strehl = "NO_JQ"
        for thermal_model in thermal_models:
            for reflected_model in reflected_models :
                if thermal_model == "None" and reflected_model == "None" :
                    pass
                else :
                    calculate_SNR_table(instru=instru,table=table,thermal_model=thermal_model,reflected_model=reflected_model,apodizer=apodizer,strehl=strehl,systematic=False)
                    if instru in instru_with_systematics:
                        calculate_SNR_table(instru=instru,table=table,thermal_model=thermal_model,reflected_model=reflected_model,apodizer=apodizer,strehl=strehl,systematic=True)

                        


def all_simulated_SNR_table():
    calculate_SNR_table("HARMONI",table="Simulated",thermal_model="BT-Settl",reflected_model="PICASO",apodizer="NO_SP",strehl="JQ1",systematic=False)
    calculate_SNR_table("ERIS",table="Simulated",thermal_model="BT-Settl",reflected_model="PICASO",systematic=False)
    calculate_SNR_table("MIRIMRS",table="Simulated",thermal_model="BT-Settl",systematic=False) # without systematics
    calculate_SNR_table("MIRIMRS",table="Simulated",thermal_model="BT-Settl",systematic=True,exposure_time=1e9) # systematic limit
    calculate_SNR_table("NIRCam",table="Simulated",thermal_model="BT-Settl",reflected_model="PICASO",systematic=False)
    calculate_SNR_table("NIRSpec",table="Simulated",thermal_model="BT-Settl",reflected_model="PICASO",systematic=False)



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
########################################### PARTIE PLOT : #############################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



def archive_yield(exposure_time,contrast=False,save=False):
    t = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    t = t[np.logical_not(t['AngSep'].mask)]
    if contrast :
        t_x = t["AngSep"]
        t_y = 10**(-(t["PlanetKmag(thermal+reflected)"]-np.array(t["StarKmag"]))/2.5)*u.dimensionless_unscaled
    else :
        t_x = t['SMA']
        t_y = t["PlanetMass"]
    x_rv=np.array([]) ; y_rv=np.array([]) ; x_im=np.array([]) ; y_im=np.array([]) ; x_tr=np.array([]) ; y_tr=np.array([])
    t_harmoni = load_planet_table("Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_BT-Settl+PICASO.ecsv")
    t_harmoni = t_harmoni[t_harmoni['SNR_INSTRU']*np.sqrt(exposure_time)>5]
    x_harmoni_rv=np.array([]) ; y_harmoni_rv=np.array([]) ; x_harmoni_im=np.array([]) ; y_harmoni_im=np.array([]) ; x_harmoni_tr=np.array([]) ; y_harmoni_tr=np.array([])
    t_eris = load_planet_table("Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_BT-Settl+PICASO.ecsv")
    t_eris = t_eris[t_eris['SNR_INSTRU']*np.sqrt(exposure_time)>5]
    x_eris_rv=np.array([]) ; y_eris_rv=np.array([]) ; x_eris_im=np.array([]) ; y_eris_im=np.array([]) ; x_eris_tr=np.array([]) ; y_eris_tr=np.array([])
    t_mirimrs = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_BT-Settl.ecsv")
    t_mirimrs = t_mirimrs[t_mirimrs['SNR_INSTRU']*np.sqrt(exposure_time)>5]
    x_mirimrs_rv=np.array([]) ; y_mirimrs_rv=np.array([]) ; x_mirimrs_im=np.array([]) ; y_mirimrs_im=np.array([]) ; x_mirimrs_tr=np.array([]) ; y_mirimrs_tr=np.array([])
    for idx in range(len(t)):
        if t[idx]["PlanetName"] in t_harmoni["PlanetName"] :
            if t[idx]["DiscoveryMethod"]=="Radial Velocity":
                x_harmoni_rv=np.append(x_harmoni_rv,t_x[idx].value) ; y_harmoni_rv=np.append(y_harmoni_rv,t_y[idx].value)
            elif t[idx]["DiscoveryMethod"]=="Imaging":
                x_harmoni_im=np.append(x_harmoni_im,t_x[idx].value) ; y_harmoni_im=np.append(y_harmoni_im,t_y[idx].value)
            else :
                x_harmoni_tr=np.append(x_harmoni_tr,t_x[idx].value) ; y_harmoni_tr=np.append(y_harmoni_tr,t_y[idx].value)
        if t[idx]["PlanetName"] in t_eris["PlanetName"]:
            if t[idx]["DiscoveryMethod"]=="Radial Velocity":
                x_eris_rv=np.append(x_eris_rv,t_x[idx].value) ; y_eris_rv=np.append(y_eris_rv,t_y[idx].value)
            elif t[idx]["DiscoveryMethod"]=="Imaging":
                x_eris_im=np.append(x_eris_im,t_x[idx].value) ; y_eris_im=np.append(y_eris_im,t_y[idx].value)
            else :
                x_eris_tr=np.append(x_eris_tr,t_x[idx].value) ; y_eris_tr=np.append(y_eris_tr,t_y[idx].value)
        if t[idx]["PlanetName"] in t_mirimrs["PlanetName"]:
            if t[idx]["DiscoveryMethod"]=="Radial Velocity":
                x_mirimrs_rv=np.append(x_mirimrs_rv,t_x[idx].value) ; y_mirimrs_rv=np.append(y_mirimrs_rv,t_y[idx].value)
            elif t[idx]["DiscoveryMethod"]=="Imaging":
                x_mirimrs_im=np.append(x_mirimrs_im,t_x[idx].value) ; y_mirimrs_im=np.append(y_mirimrs_im,t_y[idx].value)
            else :
                x_mirimrs_tr=np.append(x_mirimrs_tr,t_x[idx].value) ; y_mirimrs_tr=np.append(y_mirimrs_tr,t_y[idx].value) 
        if t[idx]["PlanetName"] not in t_harmoni["PlanetName"] and t[idx]["PlanetName"] not in t_eris["PlanetName"] and t[idx]["PlanetName"] not in t_mirimrs["PlanetName"] :
            if t[idx]["DiscoveryMethod"]=="Radial Velocity":
                x_rv=np.append(x_rv,t_x[idx].value) ; y_rv=np.append(y_rv,t_y[idx].value)
            elif t[idx]["DiscoveryMethod"]=="Imaging":
                x_im=np.append(x_im,t_x[idx].value) ; y_im=np.append(y_im,t_y[idx].value)
            else :
                x_tr=np.append(x_tr,t_x[idx].value) ; y_tr=np.append(y_tr,t_y[idx].value) 
    plt.figure(figsize=(10,7)) ;
    plt.plot(x_tr,y_tr,'kv',alpha=0.5,ms=6) ; plt.plot(x_harmoni_tr,y_harmoni_tr,'bv',ms=12) ; plt.plot(x_eris_tr,y_eris_tr,'rv',ms=9) ; plt.plot(x_mirimrs_tr,y_mirimrs_tr,'gv',ms=6)
    plt.plot(x_rv,y_rv,'ko',alpha=0.5,ms=6) ; plt.plot(x_harmoni_rv,y_harmoni_rv,'bo',ms=12) ; plt.plot(x_eris_rv,y_eris_rv,'ro',ms=9) ; plt.plot(x_mirimrs_rv,y_mirimrs_rv,'go',ms=6)
    plt.plot(x_im,y_im,'ks',alpha=0.5,ms=6) ; plt.plot(x_harmoni_im,y_harmoni_im,'bs',ms=12) ; plt.plot(x_eris_im,y_eris_im,'rs',ms=9) ; plt.plot(x_mirimrs_im,y_mirimrs_im,'gs',ms=6)
    plt.legend(loc="lower right") ; plt.yscale('log') ; plt.xscale('log') ; plt.grid(True) ; plt.title('Known exoplanets detection yield with molecular mapping \n $t_{exp}$=' + str(round(exposure_time,1)) + 'mn', fontsize = 16) ; plt.tight_layout()
    if contrast :
        plt.xlabel(f'Angular separation (in {t_x.unit})', fontsize = 14) ; plt.ylabel(f'Contrast (in K band)', fontsize = 14)
    else :
        plt.xlabel(f'Semi Major Axis (in {t_x.unit})', fontsize = 14) ; plt.ylabel(f"Planet mass (in {t_y.unit})", fontsize = 14) ; plt.ylim(1e-1,5e4)
    ax = plt.gca() ; ax.plot([],[],'bs',label='HARMONI') ; ax.plot([],[],'rs',label='ERIS') ; ax.plot([],[],'gs',label='MIRI/MRS (without systematics)') ; ax.legend(loc='lower right',fontsize=14)
    ax_legend = ax.twinx() ; ax_legend.plot([],[],'ko',label='Radial Velocity') ; ax_legend.plot([],[],'ks',label='Imaging') ; ax_legend.plot([],[],'kv',label='Transit') ; ax_legend.legend(loc='lower left',fontsize=14) ; ax_legend.tick_params(axis='y', colors='w')
    if save :
        plt.savefig("plots/archive_yield/archive_yield_"+str(round(exposure_time,1))+".png", format='png', bbox_inches='tight') ; plt.close()




def archive_yield_plot(band="INSTRU"):
    exposure_time = np.logspace(np.log10(1), np.log10(1000), 100)
    t_harmoni = load_planet_table("Archive_Pull_HARMONI_NO_SP_JQ1_without_systematics_BT-Settl+PICASO.ecsv")
    t_eris = load_planet_table("Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_BT-Settl+PICASO.ecsv")
    t_mirimrs_non_syst = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_BT-Settl.ecsv")
    t_mirimrs_syst = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_BT-Settl.ecsv")
    t_nircam = load_planet_table("Archive_Pull_NIRCam_NO_SP_NO_JQ_without_systematics_BT-Settl+PICASO.ecsv")
    t_nirspec = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_BT-Settl+PICASO.ecsv")
    yield_harmoni = np.zeros(exposure_time.shape)
    yield_eris = np.zeros(exposure_time.shape) 
    yield_mirimrs_non_syst = np.zeros(exposure_time.shape)
    yield_mirimrs_syst = np.zeros(exposure_time.shape)
    yield_nircam = np.zeros(exposure_time.shape)
    yield_nirspec = np.zeros(exposure_time.shape)
    for i in range(len(exposure_time)):
        SNR_harmoni = np.sqrt(exposure_time[i]/t_harmoni['DIT_'+band]) * t_harmoni['signal_'+band] / np.sqrt(  t_harmoni['sigma_non-syst_'+band]**2 + (exposure_time[i]/t_harmoni['DIT_'+band])*t_harmoni['sigma_syst_'+band]**2 )
        yield_harmoni[i] = len(t_harmoni[SNR_harmoni>5])
        SNR_eris = np.sqrt(exposure_time[i]/t_eris['DIT_'+band]) * t_eris['signal_'+band] / np.sqrt(  t_eris['sigma_non-syst_'+band]**2 + (exposure_time[i]/t_eris['DIT_'+band])*t_eris['sigma_syst_'+band]**2 )
        yield_eris[i] = len(t_eris[SNR_eris>5])
        SNR_mirimrs_non_syst = np.sqrt(exposure_time[i]/t_mirimrs_non_syst['DIT_'+band]) * t_mirimrs_non_syst['signal_'+band] / np.sqrt(  t_mirimrs_non_syst['sigma_non-syst_'+band]**2 + (exposure_time[i]/t_mirimrs_non_syst['DIT_'+band])*t_mirimrs_non_syst['sigma_syst_'+band]**2 )
        yield_mirimrs_non_syst[i] = len(t_mirimrs_non_syst[SNR_mirimrs_non_syst>5])
        SNR_mirimrs_syst = np.sqrt(exposure_time[i]/t_mirimrs_syst['DIT_'+band]) * t_mirimrs_syst['signal_'+band] / np.sqrt(  t_mirimrs_syst['sigma_non-syst_'+band]**2 + (exposure_time[i]/t_mirimrs_syst['DIT_'+band])*t_mirimrs_syst['sigma_syst_'+band]**2 )
        yield_mirimrs_syst[i] = len(t_mirimrs_syst[SNR_mirimrs_syst>5])
        SNR_nircam = np.sqrt(exposure_time[i]/t_nircam['DIT_'+band]) * t_nircam['signal_'+band] / np.sqrt(  t_nircam['sigma_non-syst_'+band]**2 + (exposure_time[i]/t_nircam['DIT_'+band])*t_nircam['sigma_syst_'+band]**2 )
        yield_nircam[i] = len(t_nircam[SNR_nircam>5])
        SNR_nirspec = np.sqrt(exposure_time[i]/t_nirspec['DIT_'+band]) * t_nirspec['signal_'+band] / np.sqrt(  t_nirspec['sigma_non-syst_'+band]**2 + (exposure_time[i]/t_nirspec['DIT_'+band])*t_nirspec['sigma_syst_'+band]**2 )
        yield_nirspec[i] = len(t_nirspec[SNR_nirspec>5])
    plt.figure()
    plt.plot(exposure_time,yield_harmoni,'b',label="ELT/HARMONI")
    plt.plot(exposure_time,yield_eris,'r',label="VLT/ERIS")
    plt.plot(exposure_time,yield_mirimrs_non_syst,'g',label="JWST/MIRI/MRS")
    plt.plot(exposure_time,yield_mirimrs_syst,'g--')
    plt.plot(exposure_time,yield_nircam,'m',label="JWST/NIRCam")
    plt.plot(exposure_time,yield_nirspec,'c',label="JWST/NIRSpec/IFU")
    plt.grid(True) ; plt.xscale('log') ; plt.xlabel('exposure time per target (in mn)', fontsize = 14) ; plt.ylabel('number of planets re-detected', fontsize = 14)
    plt.title('Known exoplanets detection yield', fontsize = 16)
    plt.legend(loc="upper left",fontsize = 12)
    ax = plt.gca() ; ax_legend = ax.twinx() ; ax_legend.plot([], [],'k-',label='without systematics') ; ax_legend.plot([], [], 'k--',label='with systematics') ; ax_legend.legend(loc='lower right',fontsize=12) ; ax_legend.tick_params(axis='y', colors='w')
    plt.show()
    


