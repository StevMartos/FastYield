from src.FastCurves import *


# Models list
thermal_models   = ["None", "BT-Settl", "Exo-REM", "PICASO"]
reflected_models = ["None", "tellurics", "flat", "PICASO"]

# Planet types list
planet_types = {
    # 1️⃣ Giant Planets
    "Hot Jupiter": {"mass_min": 100, "radius_min": 10, "teq_min": 1000},  
    "Warm Jupiter": {"mass_min": 100, "radius_min": 10, "teq_min": 500, "teq_max": 1000},  
    "Cold Jupiter": {"mass_min": 100, "radius_min": 10, "teq_max": 500},  
    "Hot Neptune": {"mass_min": 10, "mass_max": 100, "radius_min": 3, "radius_max": 10, "teq_min": 800},  
    "Warm Neptune": {"mass_min": 10, "mass_max": 100, "radius_min": 3, "radius_max": 10, "teq_min": 500, "teq_max": 800},  
    "Cold Neptune": {"mass_min": 10, "mass_max": 100, "radius_min": 3, "radius_max": 10, "teq_max": 500},  

    # 2️⃣ Intermediate Planets
    "Sub-Neptune": {"mass_min": 2, "mass_max": 10, "radius_min": 2, "radius_max": 4, "teq_min": 200, "teq_max": 1000},  

    # 3️⃣ Rocky Planets
    "Super-Earth": {"mass_min": 2, "mass_max": 10, "radius_min": 1.2, "radius_max": 2, "teq_min": 200, "teq_max": 1000},  
    "Exo-Earth": {"mass_min": 0.5, "mass_max": 2, "radius_min": 0.8, "radius_max": 1.2, "teq_min": 200, "teq_max": 400},  
}

# Path 
path_file      = os.path.dirname(__file__)
archive_path   = os.path.join(os.path.dirname(path_file), "sim_data/Archive_table/")
simulated_path = os.path.join(os.path.dirname(path_file), "sim_data/Simulated_table/")


    
class ExoArchive_Universe():
    # Some parts of this class are taken from PSISIM (see: https://github.com/planetarysystemsimager/psisim)
    def __init__(self, table_filename):
        super(ExoArchive_Universe, self).__init__()
        self.filename = table_filename
        self.planets = None
        self.MJUP2EARTH = 317.82838    # conversion from Jupiter to Earth masses
        self.MSOL2EARTH = 332946.05    # conversion from Solar to Earth masses
        self.RJUP2EARTH = 11.209       # conversion from Jupiter to Earth radii
        #-- Chen & Kipping 2016 constants: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
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
        #-- Thorngren 2019 Constants: https://doi.org/10.3847/2515-5172/ab4353
        # Coefficient terms from paper
        self._ThC0  = 0.96
        self._ThC1  = 0.21
        self._ThC2  =-0.20
        # Define Constraints
        self._ThMlow = 15                 # [M_earth] Lower bound of applicability
        self._ThMhi  = 12*self.MJUP2EARTH # [M_earth] Upper bound of applicability
        self._ThThi  = 1000               # [K] Temperature bound of applicability
        
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
        - StarRadius    - rad[Rsun] = (5800/Teff[K])**2 *sqrt(starlum)
        - PlanetRad  - ** when composite_table=True, keep archive-computed approx
                       Based on Thorngren 2019 and Chen&Kipping 2016
        - PlanetMass - ^^ Inverse of PlanetRad
        *** Note: the resulting planet table will have nan's where data is missing/unknown. 
            Ex. if a planet lacks a radius val, the 'PlanetRadius' for will be np.nan        
        '''
        if composite_table: #-- Define columns to read. NOTE: add columns here if needed. # col2pull entries should be matched with colNewNames entries
            col2pull =  "pl_name, hostname, pl_orbper, pl_orbsmax, pl_orbeccen, pl_orbincl, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmasse_reflink, pl_rade, pl_radeerr1, pl_radeerr2, pl_rade_reflink, " + \
                        "pl_eqt, pl_eqterr1, pl_eqterr2, pl_eqt_reflink, ra, dec, sy_dist, sy_disterr1, sy_disterr2, st_spectype, st_mass, st_teff, " + \
                        "st_rad, st_logg, st_lum, st_age, st_vsin, st_radv, " + \
                        "st_met, sy_plx, sy_bmag, sy_vmag, sy_rmag, sy_icmag, " + \
                        "sy_jmag, sy_hmag, sy_kmag, discoverymethod, disc_year, disc_refname"
            colNewNames = ["PlanetName", "StarName", "Period", "SMA", "Ecc", "Inc", "PlanetMass", "+DeltaPlanetMass", "-DeltaPlanetMass", "PlanetMassRef", "PlanetRadius", "+DeltaPlanetRadius", "-DeltaPlanetRadius", "PlanetRadiusRef", 
                           "PlanetTeq", "+DeltaPlanetTeq", "-DeltaPlanetTeq", "PlanetTeqRef", "RA", "Dec", "Distance", "+DeltaDistance", "-DeltaDistance", "StarSpT", "StarMass", "StarTeff", 
                           "StarRadius", "StarLogg", "StarLum", "StarAge", "StarVsini", "StarRadialVelocity", 
                           "StarZ", "StarParallax", "StarBMag", "StarVmag", "StarRmag", "StarImag", 
                           "StarJmag", "StarHmag", "StarKmag", "DiscoveryMethod", "DiscoveryYear", "DiscoveryRef"]
        else:
            col2pull =  "pl_name, hostname, pl_orbsmax, pl_orbeccen, pl_orbincl, pl_bmasse, pl_rade, " + \
                        "pl_eqt, ra, dec, sy_dist, st_spectype, st_mass, st_teff, " + \
                        "st_rad, st_logg, st_lum, st_age, st_vsin, st_radv, " + \
                        "st_met, sy_plx, sy_bmag, sy_vmag, sy_rmag, sy_icmag, " + \
                        "sy_jmag, sy_hmag, sy_kmag, discoverymethod"
            colNewNames = ["PlanetName", "StarName", "SMA", "Ecc", "Inc", "PlanetMass", "PlanetRadius", 
                           "PlanetTeq", "RA", "Dec", "Distance", "StarSpT", "StarMass", "StarTeff", 
                           "StarRadius", "StarLogg", "StarLum", "StarAge", "StarVsini", "StarRadialVelocity", 
                           "StarZ", "StarParallax", "StarBMag", "StarVmag", "StarRmag", "StarImag", 
                           "StarJmag", "StarHmag", "StarKmag", "DiscoveryMethod"]
            
        if os.path.isfile(self.filename) and not force_new_pull: #-- Load/Pull data depending on provided filename
            print("%s already exists:\n    we'll attempt to read this file as an astropy QTable"%self.filename) # Existing filename was provided so let's try use that
            NArx_table = QTable.read(self.filename, format='ascii.ecsv')
            if NArx_table.meta['isPSCOMPPARS'] != composite_table: # Check that the provided table file matches the requested table type
                err0 = '%s contained the wrong table-type:'%self.filename
                err1 = 'pscomppars' if composite_table else 'ps'
                err2 = 'pscomppars' if NArx_table.meta['isPSCOMPPARS'] else 'ps'
                err3 = " Expected '{}' table but found '{}' table.".format(err1, err2)
                err4 = ' Consider setting force_new_pull=True.'
                raise ValueError(err0+err3+err4)
        
        else: # New filename was provided or a new pull was explicitly requested. Pull new data
            if not force_new_pull:
                print("%s does not exist:\n    we'll pull new data from the archive and save it to this filename"%self.filename)
            else:
                print("%s may or may not exist:\n    force_new_pull=True so we'll pull new data regardless and overwrite as needed"%self.filename) 
            NArx_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP") # Create a "service" which can be used to access the archive TAP server
            tab2pull     = "pscomppars" if composite_table else "ps where default_flag=1" # Create a "query" string formatted per the TAP specifications # 'select': specify which columns to pull # 'from': specify which table to pull # 'where': (optional) specify parameters to be met when choosing what to pull # Add where flag for ps to only pull the best row for each planet
            query        = "select "+col2pull+" from "+tab2pull
            NArx_res     = NArx_service.search(query) # Pull the data and convert to astropy masked QTable
            NArx_table   = QTable(NArx_res.to_table())
            NArx_table.meta['isPSCOMPPARS'] = composite_table # Add a flag to the table metadata to denote what kind of table it was: This'll prevent trying to read the table as the wrong type later
            
            # Correcting units
            unit_corrections = {
                "Earth Mass":     u.M_earth,
                "Earth Radius":   u.R_earth,
                "Solar mass":     u.M_sun,
                "Solar Radius":   u.R_sun,
                "days":           u.day,
                "log(Solar)":     u.dex(u.solLum),
                "log10(cm/s**2)": u.dex(u.cm/(u.s**2)),
                "pl_bmasseerr1":  u.M_earth,
                "pl_bmasseerr2":  u.M_earth,
                "sy_disterr1":    u.pc,
                "sy_disterr2":    u.pc,
                "pl_orbsmax":     u.AU,
            }
            for col in NArx_table.colnames:
                current_unit = NArx_table[col].unit
                if current_unit is not None and str(current_unit) in unit_corrections:
                    new_unit        = unit_corrections[str(current_unit)]
                    NArx_table[col] = NArx_table[col].value * new_unit
                elif current_unit is None and str(col) in unit_corrections:
                    new_unit        = unit_corrections[str(col)]
                    NArx_table[col] = NArx_table[col].value * new_unit
            
            NArx_table.write(self.filename, format='ascii.ecsv', overwrite=force_new_pull) # Save raw table for future use 
            NArx_table = QTable.read(self.filename, format='ascii.ecsv') # Read table back in to ensure that formatting from a fresh pull matches: the formatting from an old pull (as done when filename exists)
        
        NArx_table.rename_columns(col2pull.split(', '), colNewNames) #-- Rename columns to psisim-expected names
        for col in NArx_table.colnames: #-- Change fill value from default 1e20 to np.nan
            if isinstance(NArx_table[col], MaskedColumn) and isinstance(NArx_table[col].fill_value, (int, float)):
                NArx_table[col].fill_value = np.nan # Only change numeric fill values to nan
        NArx_table.add_columns([MaskedColumn(length=len(NArx_table), mask=True, fill_value=np.nan)]*2, names=['ProjAU', 'Phase']) #-- Add new columns for values not easily available or computable from table 
        NArx_table['ProjAU'] = NArx_table['ProjAU'].value * u.AU
        NArx_table['Phase']  = NArx_table['Phase'].value * u.rad
        NArx_table['SMA']    = np.array(NArx_table['SMA'].value) ; NArx_table['SMA'][NArx_table['SMA']==0] = np.nan
        NArx_table['SMA']    = Masked(Quantity(np.ma.masked_array(NArx_table['SMA'], mask=np.isnan(NArx_table['SMA'])), unit=u.AU)) ; NArx_table['SMA'].mask = np.isnan(NArx_table['SMA'])
        
        # Compute missing planet columns
        if fill_empties:
            if not composite_table: # Compute missing masses and radii using mass-radius relations
                masses   = np.array(NArx_table['PlanetMass'].filled(fill_value=0.0)) # NOTE: composite table already has radius-mass approximation so we'll # only repeat them if we don't pull that table
                radii    = np.array(NArx_table['PlanetRadius'].filled(fill_value=0.0)) # Convert masked columns to ndarrays with 0's instead of mask # as needed by the approximate_... functions
                eqtemps  = np.array(NArx_table['PlanetTeq'].filled(fill_value=0.0))
                radii    = self.approximate_radii(masses, radii, eqtemps) # Perform approximations
                masses   = self.approximate_masses(masses, radii, eqtemps)
                rad_mask = (radii != 0.)  # Create masks for non-zero values (0's are values where data was missing)
                mss_mask = (masses != 0.)
                rad_mask = NArx_table['PlanetRadius'].mask & rad_mask # Create mask to only missing values in NArx_table with valid values
                mss_mask = NArx_table['PlanetMass'].mask & mss_mask
                NArx_table['PlanetRadius'][rad_mask] = radii[rad_mask]*NArx_table['PlanetRadius'].unit # Place results back in the table
                NArx_table['PlanetMass'][mss_mask]   = masses[mss_mask]*NArx_table['PlanetMass'].unit
            
            # Angular separation in mas
            NArx_table['AngSep'] = (NArx_table['SMA']/NArx_table['Distance']).value * 1e3 * u.mas
            
            # Planet logg
            grav = const.G * (NArx_table['PlanetMass'].filled(fill_value=np.nan)) / (NArx_table['PlanetRadius'].filled(fill_value=np.nan))**2
            NArx_table['PlanetLogg'] = np.ma.log10(MaskedColumn(np.ma.masked_invalid(grav.cgs.value), fill_value=np.nan)) # logg cgs
            NArx_table['PlanetLogg'] = Masked(Quantity(np.ma.masked_array(np.array(NArx_table['PlanetLogg'].value), mask=NArx_table['PlanetLogg'].mask))) # logg cgs
            NArx_table['PlanetLogg'] = NArx_table['PlanetLogg'].value * u.dex(u.cm/(u.s**2))
            
            # Star Luminosity
            host_MVs = NArx_table['StarVmag'].value - 5*np.ma.log10(NArx_table['Distance'].value/10)  # absolute v mag
            host_lum = -(host_MVs-4.83)/2.5 * u.dex(u.solLum) # log10(L/Lsun)
            NArx_table['StarLum'][NArx_table['StarLum'].mask] = host_lum[NArx_table['StarLum'].mask]
            
            # Star Radius
            host_rad = (5800/NArx_table['StarTeff'])**2 *np.ma.sqrt(10**NArx_table['StarLum'].value)   # Rsun
            NArx_table['StarRadius'][NArx_table['StarRadius'].mask] = host_rad[NArx_table['StarRadius'].mask].value * u.solRad
            
            # Star logg
            host_grav = const.G * (NArx_table['StarMass'].filled(fill_value=np.nan)*u.solMass) / (NArx_table['StarRadius'].filled(fill_value=np.nan)*u.solRad)**2
            host_logg = np.ma.log10(np.ma.masked_invalid(host_grav.cgs.value))  # logg cgs
            NArx_table['StarLogg'][NArx_table['StarLogg'].mask] = host_logg[NArx_table['StarLogg'].mask] * u.dex(u.cm/(u.s**2))
        else:
            NArx_table.add_columns([MaskedColumn(length=len(NArx_table), mask=True, fill_value=np.nan)]*2, names=['AngSep', 'PlanetLogg']) # Create fully masked columns for AngSep and PlanetLogg
        NArx_table['StarLum'] = 10**NArx_table['StarLum'].value * u.solLum # L/Lsun
        for col in NArx_table.colnames: # Make sure all number fill_values are np.nan after the column manipulations
            if isinstance(NArx_table[col], MaskedColumn) and isinstance(NArx_table[col].fill_value, (int, float)):
                NArx_table[col].fill_value = np.nan # Only change numeric fill values to nan
        self.planets = NArx_table
    
    def approximate_radii(self, masses, radii, eqtemps):
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
        rad_mask = (radii == 0.0) ##-- Find indices for missing radii so we don't replace Archive-provided values
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
        Mlow_mask = (masses  > self._ThMlow) # Create mask to find planets that meet the constraints
        Mhi_mask  = (masses  < self._ThMhi)
        tmp_mask  = (eqtemps < self._ThThi) & (eqtemps != 0.0)  # omit temp=0 since those are actually empties
        com_mask  = rad_mask & Mlow_mask & Mhi_mask & tmp_mask
        logmass_com = np.log10(masses[com_mask]/self.MJUP2EARTH) # Convert planet mass vector to M_jup for equation
        radii[com_mask] = (self._ThC0 + self._ThC1*logmass_com + self._ThC2*(logmass_com**2))*self.RJUP2EARTH # Apply equation to said planets (including conversion back to Rad_earth)
        return radii
    
    def approximate_masses(self, masses, radii, eqtemps):
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
        mss_mask = (masses == 0.0) ##-- Find indices for missing masses so we don't replace Archive-provided values
        ##-- Compute masses assuming Chen&Kipping 2016 (for hot giants): Transition points (in radii) - computed by solving at critical mass points
        R_TN = self._CKC1*(self._CKMc0**self._CKe1)
        R_NJ = self._CKC2*(self._CKMc1**self._CKe2)
        R_JS = self._CKC3*(self._CKMc2**self._CKe3)
        # Compute masses for Terran objects: These are far below Jovian range so no concern about invertibility
        ter_mask = (radii < R_TN) # filter for terran-size objects
        com_mask = mss_mask & ter_mask # planets in terran range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC0)**(1/self._CKe0)
        # Compute masses for Neptunian objects: Cut off computation at lower non-invertible radius limit (Jovian-stellar crit point)
        nep_mask = (radii < R_JS) # filter for neptune-size objects in invertible range
        com_mask = mss_mask & np.logical_not(ter_mask) & nep_mask # planets in invertible neptune range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC1)**(1/self._CKe1)
        # Ignore Jovian objects since in non-invertible range: Compute masses for Stellar objects: Cut off computation at upper non-invertible radius limit (Neptune-Jovian crit point)
        ste_mask = (radii > R_NJ) # filter for stellar-size objects in invertible range
        com_mask = mss_mask & ste_mask # planets in invertible stellar range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC3)**(1/self._CKe3)
        ##-- Compute masses assuming Thorngren 2019 (for cool giants): Use mass constraints to determine applicabile domain in radii: Convert constraint masses to M_jup for equation and compute log10 for simplicity in eq.
        log_M = np.log10(np.array([self._ThMlow, self._ThMhi])/self.MJUP2EARTH)
        cool_Rbd = (self._ThC0 + self._ThC1*log_M + self._ThC2*(log_M**2))*self.RJUP2EARTH # Apply equation (including conversion back to Rad_earth)
        cool_Rlow = cool_Rbd[0] ; cool_Rhi = cool_Rbd[1] # Extract bounds (in Earth radii) where Thorngren is applicable
        Rlow_mask = (radii   > cool_Rlow) # Create mask to find planets that meet the bounds
        Rhi_mask  = (radii   < cool_Rhi)
        tmp_mask  = (eqtemps < self._ThThi) & (eqtemps != 0.0)  # omit temp=0 since those are actually empties
        com_mask  = mss_mask & Rlow_mask & Rhi_mask & tmp_mask
        rad_com = radii[com_mask]/self.RJUP2EARTH # Convert planet radius vector to R_jup for equation
        logM    = (-1*self._ThC1 - np.sqrt(self._ThC1**2 - 4*self._ThC2*(self._ThC0-rad_com)))/(2*self._ThC2) # Apply equation to said planets: Use neg. side of quad. eq. so we get the mass values on the left side of the curve
        masses[com_mask] = (10**logM)/self.MJUP2EARTH    # convert back to Earth mass
        return masses


def inject_known_values(planet_table):
    """
    Injects K-band magnitude values of planets detected by direct imaging from a variety of references, (may need to be updated as required)
    """
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "GJ 504 b"]                  = 19.4         # https://arxiv.org/pdf/1807.00657.pdf (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "2MASS J01225093-2439505 b"] = 14.53        # https://iopscience.iop.org/article/10.1088/0004-637X/774/1/55/pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HD 206893 b"]               = 15.05        # https://iopscience.iop.org/article/10.3847/1538-3881/abc263/pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "GSC 06214-00210 b"]         = 14.87        # https://arxiv.org/pdf/1503.07586.pdf (page 6)
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "1RXS J160929.1-210524 b"]   = 16.15        # https://arxiv.org/pdf/1503.07586.pdf (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HIP 78530 b"]               = 14.18        # https://arxiv.org/pdf/1503.07586.pdf (page 6)              
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HR 2562 b"]                 = (5.02+10.5)  # https://arxiv.org/pdf/1608.06660.pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HIP 65426 b"]               = (6.771+9.85) # https://arxiv.org/pdf/1707.01413.pdf (page 8)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "PDS 70 c"]                  = (8.542+8.8)  # https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "PDS 70 b"]                  = (8.542+8.0)  # https://arxiv.org/ftp/arxiv/papers/1906/1906.01486.pdf (page 10)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HIP 21152 b"]               = 16.55        # https://iopscience.iop.org/article/10.3847/2041-8213/ac772f/pdf (page 3 bas)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "bet Pic b"]                 = (3.48+9.2)   # https://www.aanda.org/articles/aa/pdf/2011/04/aa16224-10.pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HR 8799 b"]                 = 14.05        # https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HR 8799 c"]                 = 13.13        # https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HR 8799 d"]                 = 13.11        # https://arxiv.org/pdf/0811.2606.pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HR 8799 e"]                 = (5.24+10.67) # https://arxiv.org/ftp/arxiv/papers/1011/1011.4918.pdf (page 10)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HD 95086 b"]                = (6.789+12.2) # https://www.aanda.org/articles/aa/pdf/2022/08/aa43097-22.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "USco1621 b"]                = 14.67        # https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "USco1556 b"]                = 14.85        # https://www.aanda.org/articles/aa/pdf/2020/01/aa36130-19.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "Oph 11 b"]                  = 14.44        # https://arxiv.org/pdf/astro-ph/0608574.pdf (page 27)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "FU Tau b"]                  = 13.329       # http://cdsportal.u-strasbg.fr/?target=FU%20Tau%20b            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "2MASS J12073346-3932539 b"] = 16.93        # https://www.aanda.org/articles/aa/pdf/2004/38/aagg222.pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "AF Lep b"]                  = (4.926+11.7) # https://arxiv.org/pdf/2302.06213.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "DH Tau b"]                  = 14.19        # https://iopscience.iop.org/article/10.1086/427086/pdf (page 3 bas)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "WD 0806-661 b"]             = 27           # https://arxiv.org/pdf/1605.06655.pdf (page 10)          
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HIP 79098 AB b"]            = 14.15        # https://arxiv.org/pdf/1906.02787.pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "VHS J125601.92-125723.9 b"] = 14.57        # https://www.aanda.org/articles/aa/pdf/2023/02/aa44494-22.pdf (page 2)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "ROXs 42 B b"]               = 15.01        # https://iopscience.iop.org/article/10.1088/0004-637X/787/2/104/pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "2M0437 b"]                  = 17.21        # https://arxiv.org/pdf/2110.08655.pdf (page 1)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "kap And b"]                 = 14.32        # https://www.aanda.org/articles/aa/pdf/2014/02/aa22119-13.pdf (page 14)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "GU Psc b"]                  = 17.40        # https://iopscience.iop.org/article/10.1088/0004-637X/787/1/5/pdf (page 16)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "GQ Lup b"]                  = 13.5         # https://watermark.silverchair.com/stu1586.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA4IwggN-BgkqhkiG9w0BBwagggNvMIIDawIBADCCA2QGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMe5lHC4E6oNfCKjMnAgEQgIIDNRBTyFAcCWw3I5edmpWsquUEeYqTdh9wSPyRjFSX8zAixWA69s-k7R2eYl4nU2vHPc3e6fOztwAJo0-QCF5tTT93oOxjfr7Ta533fcPZpjUiVYqKJttEITYHUEq2dKrhTepyhTRI08y-k09vTdzHLx-P61HYl12Xu5WB01fVn0_Ch21J-vy2C0mcMJt_wIAsBLvA6IUqf4dyiRljVQmL74dhgJhpSUOJnL3g2xMd0G-YN1JyWOtjTpjuCsczmncHC0vDIJmVuvgamYfR5E2BNeyY5QMjnb584CExKWTGOl4ON_CIrKNlvJI0gaInIvfLc_N_0ylQE5bPFlqe_j6bT7UuqzT7deIS4E-xnRV7t3PTA7JVo6WNKyhIs0n158LGaZMBdnxopxwHXhfmIlmkwr5mKfMYVPItmZXWJK3yGLPirvXard_TY0u34glhbsYXszUjvlQSTji58elFaTZBb9-eban2Jiz8hVsJ9JKnxE4tAKFNTWYXbEf-mxlFTQ0kh0X9sGIhMkunV5eW0VCiFtN_7DDYPROCCEKwTSWQSTz0JUH4eIzSGc7UN1gKbxjcbz5ZKpXc9oqh78ny1MigT9dX8qNpiP5AAzFohNGc1hYI_pjcF-XoJsudCR1Ig6YOQhRfxJ-EFFeAwdDOWpff2ffRp_A5scAVlSwTRqW6BPX22m99ocwwFJu4Nso4UxFzJ0d100Eszm5W792c3ZKwUcnKw2bccZz0sCk_VXCZLGVQG5vcPQY1KD1xDng9LFOkxIhIkhkRuR-o0TYZM7eIf039l8WzWsncCpQ8aV_oeMaA5Cq_qZ96H9fxSO6d3XPk9aV8KrUHsC77Ox2REXCjbusBy7wlgBgTjk01csXRuh5CDSHXLLy5GTMHAsrzh7XDfpbPhVIkVn9oOT7-KjuIAwZPErtct0tu2tBmsKO9DxQa8hCBwv0Zw2I3-yGY7IqqK4w5owijpuOo-Fk5hGCf8RMqI9eR7SWRl6FgA5CgP3nv_ZcgpJUAhMDBjwHKWds9F0rmrIU7y4dIb5pxM_JkOx8tYdM2bqwzSnJrcT2umGMCtHYoua3K0qLMCoJOUbSKfqjzpeYW (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "CT Cha b"]                  = 14.9         # https://www.aanda.org/articles/aa/pdf/2008/43/aa8840-07.pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HD 206893 c"]               = 15.2         # https://www.aanda.org/articles/aa/pdf/2021/08/aa40749-21.pdf     
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "TYC 8998-760-1 c"]          = (8.3+9.8)    # https://iopscience.iop.org/article/10.3847/2041-8213/aba27e/pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "COCONUTS-2 b"]              = 20.030       # https://iopscience.iop.org/article/10.3847/2041-8213/ac1123/pdf (table 1) 
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "51 Eri b"]                  = 15.8         # https://www.aanda.org/articles/aa/pdf/2023/05/aa44826-22.pdf
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HD 106906 b"]               = 15.46        # https://iopscience.iop.org/article/10.1088/2041-8205/780/1/L4/pdf (page 4)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "CFHTWIR-Oph 98 b"]          = 16.408       # https://arxiv.org/pdf/2011.08871.pdf (page 6)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HD 203030 b"]               = 16.21        # https://iopscience.iop.org/article/10.3847/1538-3881/aa9711/pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "BD+60 1417 b"]              = 15.645       # https://iopscience.iop.org/article/10.3847/1538-4357/ac2499/pdf (page 5)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HIP 75056 A b"]             = (7.3+6.8)    # https://arxiv.org/pdf/2009.08537.pdf (page 3)            
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "MWC 758 c"]                 = 20           # https://iopscience.iop.org/article/10.3847/1538-3881/ad11d5/pdf        
    planet_table['PlanetKmag(thermal+reflected)'][planet_table["PlanetName"] == "HD 284149 AB b"]            = 14.332       # https://iopscience.iop.org/article/10.3847/1538-3881/ace442/pdf (page 7) 
    
    planet_table['PlanetTeq'][planet_table["PlanetName"] == "bet Pic c"] = 1250 * planet_table['PlanetTeq'].unit # https://www.aanda.org/articles/aa/pdf/2020/10/aa39039-20.pdf
    
    planet_table['StarRadialVelocity'][planet_table["PlanetName"] == "Proxima Cen b"] = - 22.2 * planet_table["StarRadialVelocity"].unit # https://arxiv.org/pdf/1611.03495
    
    return planet_table

def inject_dace_values(planet_table): # https://dace-query.readthedocs.io/en/latest/output_format.html
    from dace_query.exoplanet import Exoplanet
    from astropy.table import Table
    planet_table_dace = Exoplanet.query_database(output_format='astropy_table')
    for planet in planet_table_dace:
        if planet["planet_name"] in planet_table["PlanetName"]:
            idx = get_planet_index(planet_table, planet["planet_name"])
            planet_table[idx]["Distance"]           = planet["distance"] * planet_table[idx]["Distance"].unit
            planet_table[idx]["+DeltaDistance"]     = planet["distance_upper"] * planet_table[idx]["Distance"].unit
            planet_table[idx]["-DeltaDistance"]     = -planet["distance_lower"] * planet_table[idx]["Distance"].unit
            planet_table[idx]["Ecc"]                = -planet["ecc"]
            planet_table[idx]["Inc"]                = planet["inclination"] * planet_table[idx]["Inc"].unit
            planet_table[idx]["Period"]             = planet["period"] * planet_table[idx]["Period"].unit
            planet_table[idx]["PlanetMass"]         = (planet["planet_mass"] * u.Mjup).to(planet_table[idx]["PlanetMass"].unit)
            planet_table[idx]["+DeltaPlanetMass"]   = (planet["planet_mass_upper"] * u.Mjup).to(planet_table[idx]["PlanetMass"].unit)
            planet_table[idx]["-DeltaPlanetMass"]   = - (planet["planet_mass_lower"] * u.Mjup).to(planet_table[idx]["PlanetMass"].unit)
            planet_table[idx]["PlanetRadius"]       = (planet["planet_radius"] * u.Rjup).to(planet_table[idx]["PlanetRadius"].unit)
            planet_table[idx]["+DeltaPlanetRadius"] = (planet["planet_radius_upper"] * u.Rjup).to(planet_table[idx]["PlanetRadius"].unit)
            planet_table[idx]["-DeltaPlanetRadius"] = - (planet["planet_radius_lower"] * u.Rjup).to(planet_table[idx]["PlanetRadius"].unit)
            planet_table[idx]["StarRadialVelocity"] = planet["radial_velocity"] * planet_table[idx]["StarRadialVelocity"].unit        
            planet_table[idx]["SMA"]                = planet["semi_major_axis"] * planet_table[idx]["SMA"].unit
            planet_table[idx]["StarAge"]            = planet["stellar_age"] * planet_table[idx]["StarAge"].unit
            planet_table[idx]["StarTeff"]           = planet["stellar_eff_temp"] * planet_table[idx]["StarTeff"].unit
            planet_table[idx]["StarMass"]           = planet["stellar_mass"] * planet_table[idx]["StarMass"].unit
            planet_table[idx]["StarRadius"]         = planet["stellar_radius"] * planet_table[idx]["StarRadius"].unit
            planet_table[idx]["StarVsini"]          = planet["stellar_rotational_velocity"] * planet_table[idx]["StarVsini"].unit
            planet_table[idx]["StarLogg"]           = planet["stellar_surface_gravity"] * planet_table[idx]["StarLogg"].unit
    return planet_table

def get_missing_k_mags(): # Missing values for K-band mag of Direct Imaging planets
    planet_table = load_planet_table("Archive_Pull.ecsv")
    planet_table = planet_table[planet_table["DiscoveryMethod"]=="Imaging"]
    planet_table['PlanetKmag(thermal+reflected)'] = np.full(len(planet_table), np.nan)
    planet_table = inject_known_values(planet_table) # on rentre les valeurs connues des magnitudes (en bande K) des planètes détectées par imagerie directe
    print(f" {len(planet_table[np.isnan(planet_table['PlanetKmag(thermal+reflected)'])])}/{len(planet_table)} K-band value are missing for Imaging planets :")
    for i in range(len(planet_table)):
        if np.isnan(planet_table[i]['PlanetKmag(thermal+reflected)']) : 
            print(f"\n {planet_table[i]['PlanetName']} : {planet_table[i]['DiscoveryRef']}")
    


#######################################################################################################################
#################################################### Utils function: #################################################
#######################################################################################################################



def get_mask(planet_table, column):
    " get a mask with the invalid values of the planet_table's column"
    try:
        mask = (planet_table[column].mask)|(np.isnan(np.array(planet_table[column])))|(np.array(planet_table[column])==np.nan)|(np.array(planet_table[column])==0)
    except:
        mask = (np.isnan(np.array(planet_table[column])))|(np.array(planet_table[column])==np.nan)|(np.array(planet_table[column])==0)
    return mask

def get_planet_index(planet_table, planet_name, TAP=True):
    """
    Gives the index of "planet_name" in "planet_table"
    """
    if TAP:
        idx = np.where(planet_table["PlanetName"] == planet_name)[0][0]
    else:
        idx = np.where(planet_table["name"] == planet_name)[0][0]
    return idx

def get_closest_planet(planet_table, T_planet, lg_planet): # permet de trouver la planète ayant les paramètres les plus proches de T et lg
    diff_T   = (planet_table["PlanetTeq"].value-T)/T
    diff_lg  = (planet_table["PlanetLogg"].value-lg)/lg
    distance = np.sqrt(diff_T**2+diff_lg**2)
    idx      = np.argmin(distance)
    return idx

def get_planet_type(planet):
    """
    Determine the type of a planet based on predefined conditions in `planet_types`.

    Parameters:
    - planet: a row from `planet_table` containing PlanetMass, PlanetRadius, and PlanetTeq.

    Returns:
    - A string representing the type of the planet.
    """

    # Extract properties from the planet
    mass   = float(planet["PlanetMass"].value)
    radius = float(planet["PlanetRadius"].value)
    teq    = float(planet["PlanetTeq"].value)

    # If mass or radius is masked, return "Unidentified"
    if np.ma.is_masked(mass) or np.ma.is_masked(radius):
        return "Unidentified"

    # Loop through planet types to find a match
    for ptype, criteria in planet_types.items():
        mass_match   = ("mass_min" not in criteria or mass >= criteria["mass_min"]) and ("mass_max" not in criteria or mass <= criteria["mass_max"])
        radius_match = ("radius_min" not in criteria or radius >= criteria["radius_min"]) and ("radius_max" not in criteria or radius <= criteria["radius_max"])
        teq_match    = teq is None or (("teq_min" not in criteria or teq >= criteria["teq_min"]) and ("teq_max" not in criteria or teq <= criteria["teq_max"]))
        if mass_match and radius_match and teq_match:
            return ptype  # Return the first matching type

    return "Unidentified"  # If no match is found

def find_matching_planets(criteria, planet_table, mode, selected_planets=None, Nmax=None):  
    # Build the query string dynamically
    query = " & ".join([
        f"PlanetMass >= {criteria['mass_min']}" if "mass_min" in criteria else "",
        f"PlanetMass <= {criteria['mass_max']}" if "mass_max" in criteria else "",
        f"PlanetRadius >= {criteria['radius_min']}" if "radius_min" in criteria else "",
        f"PlanetRadius <= {criteria['radius_max']}" if "radius_max" in criteria else "",
        f"PlanetTeq >= {criteria['teq_min']}" if "teq_min" in criteria else "",
        f"PlanetTeq <= {criteria['teq_max']}" if "teq_max" in criteria else ""
    ])
    
    # Remove empty conditions from query
    query = " & ".join(filter(None, query.split(" & ")))
    filtered_planets = planet_table.query(query) if query else planet_table

    # Mode 'unique': Pick the planet with highest SNR
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

def plot_matching_planets(matching_planets, exposure_time, mode, instru=None):
    snr_threshold = 5
    
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
            {"Type": ptype, "Name": planet["PlanetName"], 
             "Mass [M⊕]":  round(planet["PlanetMass"], 2), 
             "Radius [R⊕]":  round(planet["PlanetRadius"], 2), 
             "Temperature [K]": int(round(planet["PlanetTeq"])), 
             f"SNR (in {int(round(exposure_time/60))} h)": round(planet["SNR"], 1)}
            for ptype, planets in matching_planets.items() for planet in planets])
        table = ax.table(cellText=matching_planets_df.values, colLabels=matching_planets_df.columns, cellLoc='center', loc='center')
    elif mode == 'multi':    
        snr_threshold = 5
        conditions_df = pd.DataFrame([
            {"Type": ptype,
             "Mass [M⊕]": format_range(criteria, "mass"),
             "Radius [R⊕]": format_range(criteria, "radius"),
             "Temperature [K]": format_range(criteria, "teq"),
             "Number of Planets\nconsidered": len(matching_planets[ptype]),
             f"Number of Planets\ndetected (in {int(round(exposure_time/60))} h)": sum(planet["SNR"] > snr_threshold for planet in matching_planets[ptype])}
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

def format_range(criteria, key):
    min_key = f"{key}_min"
    max_key = f"{key}_max"
    if min_key in criteria and max_key in criteria:
        return f"{criteria[min_key]} - {criteria[max_key]}"
    elif min_key in criteria:
        return f">{criteria[min_key]}"
    elif max_key in criteria:
        return f"<{criteria[max_key]}"
    return "N/A"

def get_spectrum_contribution_name_model(thermal_model, reflected_model):
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

def load_planet_table(table_name):
    """
    To load "table_name"
    """
    if "Archive" in table_name:
        planet_table = QTable.read(archive_path+table_name, format='ascii.ecsv')
    else:
        planet_table = QTable.read(simulated_path+table_name, format='ascii.ecsv')
    return planet_table



#######################################################################################################################
#################################################### Creating tables: #################################################
#######################################################################################################################



def create_archive_planet_table():
    """
    To create an exoplanet archive table
    """
    archive_table_filename = archive_path+"Archive_Pull_raw.ecsv" # Filename in which to save raw exoplanet archive table
    uni                    = ExoArchive_Universe(archive_table_filename) # Instantiate universe object
    uni.Load_ExoArchive_Universe(force_new_pull=True)  # Pull and populate the planet table
    planet_table_raw = uni.planets
    planet_table_raw.write(archive_path+"Archive_Pull_raw.ecsv", format='ascii.ecsv', overwrite=True)
    planet_table = load_planet_table("Archive_Pull_raw.ecsv")
    
    # injecting DACE values
    planet_table = inject_dace_values(planet_table)
    
    # Detection method masks
    im_mask = planet_table["DiscoveryMethod"] == "Imaging"
    rv_mask = planet_table["DiscoveryMethod"] == "Radial Velocity"
    tr_mask = planet_table["DiscoveryMethod"] == "Transit"
    ot_mask = (planet_table["DiscoveryMethod"]!="Imaging") & (planet_table["DiscoveryMethod"]!="Radial Velocity") & (planet_table["DiscoveryMethod"]!="Transit")
    
    # Estimate planet Teq when missing (except for direct imaged planets) => equilibrium temperature
    ptq_mask                                               = np.logical_not(get_mask(planet_table, "PlanetTeq")) # gives the valid (existing) values of the planet's temperature
    nb_missing_Tp                                          = len(planet_table["PlanetTeq"][(~ptq_mask) & (~im_mask)]) # nb of missing temperatures (except direct imaged planets)
    planet_teq                                             = (planet_table['StarRadius']/(planet_table['SMA'])).decompose()**(1/2) * planet_table['StarTeff'].value *u.K
    planet_table["PlanetTeq"][(~ptq_mask) & (~im_mask)]    = planet_teq[(~ptq_mask) & (~im_mask)] # estimates missing temperatures, except for planets detected by direct imaging, where the temperature would be greatly underestimated (hot young planets)
    planet_table["PlanetTeqRef"][(~ptq_mask) & (~im_mask)] = "thermal equilibrium estimation" # estimates missing temperatures, except for planets detected by direct imaging, where the temperature would be greatly underestimated (hot young planets)
    
    # Since the radius is not important for direct imaged planets (renormalization with K-band measured magnitude), zeros values are replaced by an arbitrary value
    prd_mask = np.logical_not(get_mask(planet_table, "PlanetRadius")) # gives the valid (existing) values of the planet's radius
    planet_table["PlanetRadius"][(~prd_mask) & im_mask] = np.nanmean(planet_table['PlanetRadius'][im_mask].value) * planet_table["PlanetRadius"].unit
    planet_table["PlanetRadiusRef"][(~prd_mask) & im_mask] = "mean value filled"
    
    # Fill missing lg values with mean values according to the detection method (since this is not a critical parameter, it is an acceptable approximation)
    slg_mask = np.logical_not(get_mask(planet_table, "StarLogg"))
    planet_table['StarLogg'][~slg_mask & im_mask] = np.nanmean(planet_table['StarLogg'][im_mask].value) * planet_table['StarLogg'].unit
    planet_table['StarLogg'][~slg_mask & rv_mask] = np.nanmean(planet_table['StarLogg'][rv_mask].value) * planet_table['StarLogg'].unit
    planet_table['StarLogg'][~slg_mask & tr_mask] = np.nanmean(planet_table['StarLogg'][tr_mask].value) * planet_table['StarLogg'].unit
    planet_table['StarLogg'][~slg_mask & ot_mask] = np.nanmean(planet_table['StarLogg'][ot_mask].value) * planet_table['StarLogg'].unit
    plg_mask = np.logical_not(get_mask(planet_table, "PlanetLogg"))
    planet_table['PlanetLogg'][~plg_mask & im_mask] = np.nanmean(planet_table['PlanetLogg'][im_mask].value) * planet_table['PlanetLogg'].unit
    planet_table['PlanetLogg'][~plg_mask & rv_mask] = np.nanmean(planet_table['PlanetLogg'][rv_mask].value) * planet_table['PlanetLogg'].unit
    planet_table['PlanetLogg'][~plg_mask & tr_mask] = np.nanmean(planet_table['PlanetLogg'][tr_mask].value) * planet_table['PlanetLogg'].unit
    planet_table['PlanetLogg'][~plg_mask & ot_mask] = np.nanmean(planet_table['PlanetLogg'][ot_mask].value) * planet_table['PlanetLogg'].unit
    
    # Create masks for missing entries for contrast calculation
    stq_mask = np.logical_not(get_mask(planet_table, "StarTeff"))
    ptq_mask = np.logical_not(get_mask(planet_table, "PlanetTeq"))
    stk_mask = np.logical_not(get_mask(planet_table, "StarKmag"))
    prd_mask = np.logical_not(get_mask(planet_table, "PlanetRadius"))
    dis_mask = np.logical_not(get_mask(planet_table, "Distance"))
    sma_mask = np.logical_not(get_mask(planet_table, "SMA"))
    planet_table = planet_table[stq_mask & ptq_mask & stk_mask & prd_mask & dis_mask & sma_mask]
    
    # Assumes planets are at their maximum elongation (phi = pi/2)
    planet_table["Phase"] = np.pi/2 * planet_table["Phase"].unit

    # Fixing the minimum planets temperature
    min_Tp = 200 * u.K # fixing minimum temperature for the planets (the one of the BT-Settl, Morley and SONORA spectra)
    planet_table["PlanetTeq"][planet_table["PlanetTeq"] < min_Tp ] = min_Tp
    
    # Fixing the maximum planets temperature
    max_Tp = 3000 * u.K # fixing maximum temperature for the planets
    planet_table["PlanetTeq"][planet_table["PlanetTeq"] > max_Tp ] = max_Tp
    
    # Fixing the maximum planets mass
    max_Mp = 10000 * u.Mearth # otherwise it can be considered as a star 
    planet_table["PlanetMass"][planet_table["PlanetMass"] > max_Mp ] = max_Mp
    
    # Randomly draw radial velocities of the stars according to a normal distribution when they are missing
    srv_mask = get_mask(planet_table, "StarRadialVelocity")
    planet_table["StarRadialVelocity"][srv_mask] = np.random.normal(np.nanmean(np.array(planet_table["StarRadialVelocity"][~srv_mask].value)), np.nanstd(np.array(planet_table["StarRadialVelocity"][~srv_mask].value)), len(planet_table["StarRadialVelocity"][srv_mask])) * planet_table["StarRadialVelocity"].unit
    
    # Simulating random star Vsini values (depending on the type of stars) when the values are missing
    st_mask   = np.logical_not(get_mask(planet_table, "StarTeff"))
    svsi_mask = get_mask(planet_table, "StarVsini")
    planet_table["StarVsini"][svsi_mask & st_mask & (planet_table["StarTeff"] <= 3500 * u.K)]                                            = np.random.normal(1, 0.5, size=len(planet_table["StarVsini"][svsi_mask & st_mask & (planet_table["StarTeff"] <= 3500 * u.K)])) * u.km / u.s # Cool stars (Teff <= 3500 K): mu = 1 km/s, sigma = 0.5 km/s (M-dwarfs, Newton et al. 2017)
    planet_table["StarVsini"][svsi_mask & st_mask & (3500 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000 * u.K)]  = np.random.normal(3, 1, size=len(planet_table["StarVsini"][svsi_mask & st_mask & (3500 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 6000 * u.K)])) * u.km / u.s # Solar-type stars (3500 K < Teff <= 6000 K): mu = 3 km/s, sigma = 1 km/s (G and K-type stars, Nielsen et al. 2013)
    planet_table["StarVsini"][svsi_mask & st_mask & (6000 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000 * u.K)] = np.random.normal(10, 4, size=len(planet_table["StarVsini"][svsi_mask & st_mask & (6000 * u.K < planet_table["StarTeff"]) & (planet_table["StarTeff"] <= 10000 * u.K)])) * u.km / u.s # Hot stars (6000 K < Teff <= 10000 K): mu = 10 km/s, sigma = 4 km/s (F-type stars, Royer et al. 2007)
    planet_table["StarVsini"][svsi_mask & st_mask & (10000 * u.K < planet_table["StarTeff"])]                                            = np.random.normal(50, 15, size=len(planet_table["StarVsini"][svsi_mask & st_mask & (10000 * u.K < planet_table["StarTeff"])])) * u.km / u.s # Very hot stars (Teff > 10000 K): mu = 50 km/s, sigma = 15 km/s (O, B, and A-type stars, Zorec & Royer 2012)
    planet_table["StarVsini"][planet_table["StarVsini"] < 0] = 0.
    
    # Simulating random planet Vsini values (depending on the type of planets)
    planet_table["PlanetVsini"] = 0 * planet_table["StarVsini"]
    pm_mask                     = np.logical_not(get_mask(planet_table, "PlanetMass"))
    planet_table["PlanetVsini"][pm_mask & (planet_table["PlanetMass"]<=5*u.Mearth)]                                               = np.random.normal(2, 1, size=len(planet_table["PlanetVsini"][pm_mask&(planet_table["PlanetMass"]<=5*u.Mearth)])) * u.km / u.s # Earths-like : mu = 2 km/s, sigma = 1 km/s (McQuillan, A., Mazeh, T., & Aigrain, S. (2013). MNRAS, 432, 1203.)
    planet_table["PlanetVsini"][pm_mask & (5*u.Mearth<planet_table["PlanetMass"]) & (planet_table["PlanetMass"]<=20*u.Mearth)]    = np.random.normal(5, 2, size=len(planet_table["PlanetVsini"][pm_mask&(5*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=20*u.Mearth)])) * u.km / u.s # Super-Earths : mu = 5 km/s, sigma = 2 km/s (Dai, F., et al. (2016). ApJ, 823, 115.)
    planet_table["PlanetVsini"][pm_mask & (20*u.Mearth<planet_table["PlanetMass"]) & (planet_table["PlanetMass"]<=100*u.Mearth)]  = np.random.normal(12, 5, size=len(planet_table["PlanetVsini"][pm_mask&(20*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=100*u.Mearth)])) * u.km / u.s # Neptunes-like : mu = 12 km/s, sigma = 5 km/s (Snellen, I.A.G., et al. (2014). Nature, 509, 63-65.)
    planet_table["PlanetVsini"][pm_mask & (100*u.Mearth<planet_table["PlanetMass"]) & (planet_table["PlanetMass"]<=300*u.Mearth)] = np.random.normal(25, 10, size=len(planet_table["PlanetVsini"][pm_mask&(100*u.Mearth<planet_table["PlanetMass"])&(planet_table["PlanetMass"]<=300*u.Mearth)])) * u.km / u.s # Jupiter-like : mu = 25 km/s, sigma = 10 km/s (Bryan, M.L., et al. (2018). AJ, 156, 142.)
    planet_table["PlanetVsini"][pm_mask & (300*u.Mearth<planet_table["PlanetMass"])]                                              = np.random.normal(40, 15, size=len(planet_table["PlanetVsini"][pm_mask&(300*u.Mearth<planet_table["PlanetMass"])])) * u.km / u.s # Super-Jupiters : mu = 40 km/s, sigma = 15 km/s (Snellen, I.A.G., et al. (2010). Nature, 465, 1049-1051.)
    planet_table["PlanetVsini"][planet_table["PlanetVsini"]<0] = 0.
    
    # Randomly drawing inclinations when they are missing (actually setting inc = 90°, in order to have the most optimistic cases when the Inc are missing)
    inc_mask                      = get_mask(planet_table, "Inc")
    planet_table["Inc"][inc_mask] = 90 * planet_table["Inc"].unit
    i                             = np.array(planet_table["Inc"].value) * np.pi/180
    
    # Estimates the Doppler shift between planet and star, assuming it is the orbital (circular) velocity * sin(i) (STILL WITH THE ASSUMPTION THAT ALL THE PLANETS ARE ON THEIR MAX ELONGATION)
    planet_table['DeltaRadialVelocity']           = np.sqrt(const.G*(planet_table["StarMass"]+planet_table["PlanetMass"])/planet_table["SMA"]).decompose().to(u.km/u.s) * np.sin(i)
    drv_mask                                      = get_mask(planet_table, "DeltaRadialVelocity")
    planet_table["DeltaRadialVelocity"][drv_mask] = np.random.normal(np.nanmean(np.array(planet_table["DeltaRadialVelocity"].value)), np.nanstd(np.array(planet_table["DeltaRadialVelocity"].value)), len(planet_table["DeltaRadialVelocity"][drv_mask])) * planet_table["DeltaRadialVelocity"].unit
    planet_table["PlanetRadialVelocity"]          = planet_table["StarRadialVelocity"] + planet_table["DeltaRadialVelocity"]
    
    # Calcuting phase functions
    planet_table["alpha"]   = np.arccos(-np.sin(i)*np.cos(np.array(planet_table["Phase"].value)))
    planet_table["g_alpha"] = (np.sin(planet_table["alpha"])+(np.pi-planet_table["alpha"])*np.cos(planet_table["alpha"]))/np.pi # fonction de phase de Lambert
    
    # Saving the table
    planet_table.write(archive_path+"Archive_Pull.ecsv", format='ascii.ecsv', overwrite=True)
    
    # prints
    print('\n Estimating the temperature for: ', nb_missing_Tp, "planets")
    print('\n Nb of planets for which the SNR value can be obtained: %d'%len(planet_table[np.logical_not(get_mask(planet_table, "AngSep"))]))
    



def create_simulated_planet_table():
    """
    To create an exoplanet simulated table
    """
    import EXOSIMS.MissionSim, EXOSIMS.SimulatedUniverse.SAG13Universe
    filename = "sim_data/Simulated_table/FastYield_sim_EXOCAT1.json"
    with open(filename) as ff:
        specs = json.loads(ff.read())
    su = EXOSIMS.SimulatedUniverse.SAG13Universe.SAG13Universe(**specs)
    flux_ratios = 10**(su.dMag/-2.5)  # grab for now from EXOSIMS
    angseps = su.WA.value * 1000 *u.mas # mas
    projaus = su.d.value * u.AU # au
    phase = np.arccos(su.r[:, 2]/su.d)# planet phase  [0, pi]
    smas = su.a.value*u.AU # au
    eccs = su.e # eccentricity
    incs = su.I.value*u.deg # degrees
    masses = su.Mp  # earth masses
    radii = su.Rp # earth radii
    grav = const.G * (masses)/(radii)**2
    logg = np.log10(grav.to(u.cm/u.s**2).value) * u.dex(u.cm/u.s**2) # logg cgs
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
        pname = np.char.add(star_names[i], f" {k}")
        while pname in planet_names:
            pname = np.char.add(star_names[i], f" {k}")
            k+=1
        planet_names[i] = pname #np.append(planet_names, pname)
        if masses[i] < 2.0 * u.earthMass:
            planet_types[i] = "Terran"
        elif 2.0 * u.earthMass < masses[i] < 0.41 * u.jupiterMass:
            planet_types[i] = "Neptunian"
        elif 0.41 * u.jupiterMass < masses[i] < 0.80 * u.solMass:
            planet_types[i] = "Jovian"
        elif 0.80 * u.solMass < masses[i]:
            planet_types[i] = "Stellar"
        print("giving name and type: ", planet_names[i], " & ", planet_types[i], ": ", round(100*(i+1)/len(star_names), 3), " %")
    spts = np.array([su.TargetList.Spec[i] for i in su.plan2star])
    su.TargetList.stellar_mass() # generate masses if haven't
    host_mass = np.array([su.TargetList.MsTrue[i].value for i in su.plan2star]) * u.solMass
    su.TargetList.stellar_Teff()
    host_teff = np.array([su.TargetList.Teff[i].value for i in su.plan2star]) * u.K
    host_Vmags = np.array([su.TargetList.Vmag[i] for i in su.plan2star])
    host_Kmags = np.array([su.TargetList.Kmag[i] for i in su.plan2star]) * u.mag
    # guess the radius and gravity from Vmag and Teff. This is of questionable reliability
    host_MVs   = host_Vmags - 5 * np.log10(distances.value/10) # absolute V mag
    host_lums  = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
    host_radii = (5800/host_teff.value)**2 * np.sqrt(host_lums) * u.solRad# Rsun
    host_gravs = const.G * host_mass/(host_radii**2)
    host_logg  = np.log10(host_gravs.to(u.cm/u.s**2).value) * u.dex(u.cm/(u.s**2))# logg cgs
    teq      = su.PlanetPhysicalModel.calc_Teff(host_lums, smas, su.p)
    all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, planet_names, planet_types, masses, radii, teq, logg, spts, host_mass, host_teff, host_radii, host_logg, host_Kmags]
    labels   = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetName", "PlanetType", "PlanetMass", "PlanetRadius", "PlanetTeq", "PlanetLogg", "StarSpT", "StarMass", "StarTeff", "StarRadius", "StarLogg", "StarKmag"]
    planet_table = QTable(all_data, names=labels)
    slg_mask = np.logical_not(np.isnan(planet_table['StarLogg']))
    stq_mask = np.logical_not(np.isnan(planet_table['StarTeff']))
    plg_mask = np.logical_not(np.isnan(planet_table['PlanetLogg'])) 
    ptq_mask = np.logical_not(np.isnan(planet_table['PlanetTeq']))
    stk_mask = np.logical_not(np.isnan(planet_table['StarKmag']))
    prd_mask = np.logical_not(np.isnan(planet_table['PlanetRadius']))
    dis_mask = np.logical_not(np.isnan(planet_table['Distance']))
    sma_mask = np.logical_not(np.isnan(planet_table['SMA']))
    planet_table = planet_table[slg_mask & stq_mask & plg_mask & ptq_mask & stk_mask & prd_mask & dis_mask & sma_mask]
    planet_table["PlanetTeq"] = (planet_table['StarRadius']/(planet_table['SMA'])).decompose()**(1/2) * planet_table['StarTeff'].value *u.K
    planet_table["DiscoveryMethod"] = np.full((len(planet_table), ), "None")
    planet_table["StarRadialVelocity"] = np.full((len(planet_table), ), 0) * u.km / u.s
    planet_table["StarVsini"] = np.full((len(planet_table), ), 0) * u.km / u.s
    planet_table.write(simulated_path+"Simulated_Pull_raw.ecsv", format='ascii.ecsv', overwrite=True)



def create_fastcurves_table(table="Archive"): # take ~ 3 minutes for Archive and even more for Simulated
    """
    To create FastCurves compatible planet tables from an archive or a simulated table
    """
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    time1 = time.time()
    
    if table == "Archive":
        create_archive_planet_table()
        planet_table = load_planet_table("Archive_Pull.ecsv")
        
    elif table == "Simulated":
        create_simulated_planet_table()
        planet_table = load_planet_table("Simulated_Pull_raw.ecsv")        
        print("Raw number of planets = ", len(planet_table))
        n_planets = len(planet_table)
        n_planets_now = int(n_planets/10) 
        rand_planets = np.random.randint(0, n_planets, n_planets_now)
        planet_table = planet_table[rand_planets]
        plt.figure() ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel(f"angular separation (in {planet_table['AngSep'].unit})") ; plt.ylabel(f"flux ratio")
        plt.scatter(planet_table["AngSep"], planet_table["Flux Ratio"], c='k', alpha=0.333, zorder=10)   
        plt.scatter(planet_table["AngSep"], planet_table["Flux Ratio"], c='r', alpha=0.333, zorder=10) 
        plt.show()
        plt.figure() ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel(f"planet mass (in {planet_table['PlanetMass'].unit})") ; plt.ylabel(f"planet radius (in {planet_table['PlanetRadius'].unit})")
        plt.scatter(planet_table["PlanetMass"], planet_table["PlanetRadius"], c='k', alpha=0.333, zorder=10)
        plt.scatter(planet_table["PlanetMass"], planet_table["PlanetRadius"], c='r', alpha=0.333, zorder=10)
        plt.show()
        plt.figure() ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on() ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel(f"semi major axis (in {planet_table['SMA'].unit})") ; plt.ylabel(f"planet radius (in {planet_table['PlanetRadius'].unit})")
        plt.scatter(planet_table["SMA"], planet_table["PlanetRadius"], c='k', alpha=0.333, zorder=10)
        plt.scatter(planet_table["SMA"], planet_table["PlanetRadius"], c='r', alpha=0.333, zorder=10)
        plt.show()
    print('\n Total nb of planets considered: %d'%len(planet_table))
    
    # creating planet type column
    planet_table["PlanetType"] = np.full(len(planet_table), "Unidentified", dtype="<U32")

    # creating magnitudes columns
    for instru in instru_name_list:
        planet_table['StarINSTRUmag('+instru+')']                      = np.full(len(planet_table), np.nan)
        planet_table['PlanetINSTRUmag('+instru+')(thermal+reflected)'] = np.full(len(planet_table), np.nan)
        planet_table['PlanetINSTRUmag('+instru+')(thermal)']           = np.full(len(planet_table), np.nan)
        planet_table['PlanetINSTRUmag('+instru+')(reflected)']         = np.full(len(planet_table), np.nan)
    for band in bands:
        if band == "K":
            planet_table['StarKmag'] = planet_table['StarKmag'].value
        else:
            planet_table['Star'+band+'mag']                  = np.full(len(planet_table), np.nan)
        planet_table['Planet'+band+'mag(thermal+reflected)'] = np.full(len(planet_table), np.nan)
        planet_table['Planet'+band+'mag(thermal)']           = np.full(len(planet_table), np.nan)
        planet_table['Planet'+band+'mag(reflected)']         = np.full(len(planet_table), np.nan)
    
    # enters the known K-band magnitudes of planets detected by direct imaging
    if table == "Archive":
        planet_table = inject_known_values(planet_table)
        
    # defining bands and vega spectrum
    wave_instru     = np.arange(LMIN, LMAX, 1e-2)
    wave_K          = wave_instru[(wave_instru>=lmin_K)&(wave_instru<=lmax_K)]
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm = False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_instru, renorm = False)
    
    # spectra estimations
    with Pool(processes=cpu_count()//2) as pool: 
        results = list(tqdm(pool.imap(process_fastcurves_table, [(idx, make_table_serializable(planet_table[idx]), wave_instru, wave_K, vega_spectrum_K) for idx in range(len(planet_table))]), total=len(planet_table), desc="Estimating magnitudes..."))
        for (idx, ptype, planet_spectrum, planet_thermal, planet_reflected, star_spectrum) in results:
            planet_table[idx]['PlanetType'] = ptype
            for instru in instru_name_list:
                planet_table[idx]['StarINSTRUmag('+instru+')']                      = -2.5*np.log10(np.nanmean(star_spectrum.flux[(wave_instru>globals()["lmin_"+instru])&(wave_instru<globals()["lmax_"+instru])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+instru]) & (wave_instru<globals()["lmax_"+instru])]))
                planet_table[idx]['PlanetINSTRUmag('+instru+')(thermal+reflected)'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave_instru>globals()["lmin_"+instru])&(wave_instru<globals()["lmax_"+instru])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+instru]) & (wave_instru<globals()["lmax_"+instru])]))
                planet_table[idx]['PlanetINSTRUmag('+instru+')(thermal)']           = -2.5*np.log10(np.nanmean(planet_thermal.flux[(wave_instru>globals()["lmin_"+instru])&(wave_instru<globals()["lmax_"+instru])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+instru]) & (wave_instru<globals()["lmax_"+instru])]))
                if globals()["lmin_"+instru] < 6: # above 6 µm, we neglect the reflected contribution
                    planet_table[idx]['PlanetINSTRUmag('+instru+')(reflected)'] = -2.5*np.log10(np.nanmean(planet_reflected.flux[(wave_instru>globals()["lmin_"+instru])&(wave_instru<globals()["lmax_"+instru])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+instru]) & (wave_instru<globals()["lmax_"+instru])]))
            for band in bands:
                if band != "K":
                    planet_table[idx]['Star'+band+'mag']                  = -2.5*np.log10(np.nanmean(star_spectrum.flux[(wave_instru>globals()["lmin_"+band])&(wave_instru<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+band]) & (wave_instru<globals()["lmax_"+band])]))
                planet_table[idx]['Planet'+band+'mag(thermal+reflected)'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave_instru>globals()["lmin_"+band])&(wave_instru<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+band]) & (wave_instru<globals()["lmax_"+band])]))
                planet_table[idx]['Planet'+band+'mag(thermal)']           = -2.5*np.log10(np.nanmean(planet_thermal.flux[(wave_instru>globals()["lmin_"+band])&(wave_instru<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+band]) & (wave_instru<globals()["lmax_"+band])])) 
                if globals()["lmin_"+band] < 6:
                    planet_table[idx]['Planet'+band+'mag(reflected)'] = -2.5*np.log10(np.nanmean(planet_reflected.flux[(wave_instru>globals()["lmin_"+band])&(wave_instru<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+band]) & (wave_instru<globals()["lmax_"+band])])) 
            # print("\n", planet_table[idx]["PlanetName"], ": ")
            # print(" mag(K)_p_total = ", round(planet_table[idx]['PlanetKmag(thermal+reflected)'], 1))
            # print(" mag(K)_p_thermal = ", round(planet_table[idx]['PlanetKmag(thermal)'], 1))
            # print(" mag(K)_p_reflected = ", round(planet_table[idx]['PlanetKmag(reflected)'], 1))
            
    print('\n Nb of planets for FastCurves calculations: %d'%len(planet_table))
    print('\n Generating the table took {0:.3f} s'.format(time.time()-time1))
    if table == "Archive":
        planet_table.write(archive_path+"Archive_Pull_for_FastCurves.ecsv", format='ascii.ecsv', overwrite=True)
    else:
        planet_table.write(simulated_path+"Simulated_Pull_for_FastCurves.ecsv", format='ascii.ecsv', overwrite=True)

def process_fastcurves_table(args):
    idx, planet, wave_instru, wave_K, vega_spectrum_K = args
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = thermal_reflected_spectrum(planet, instru=None, thermal_model="BT-Settl", reflected_model="PICASO", wave_instru=wave_instru, wave_K=wave_K, vega_spectrum_K=vega_spectrum_K, show=False, in_im_mag=True)
    ptype = get_planet_type(planet)
    return idx, ptype, planet_spectrum, planet_thermal, planet_reflected, star_spectrum


def make_table_serializable(table):
    serializable_data = {}
    for col_name in table.colnames:
        column_data = table[col_name]        
        if isinstance(column_data, (np.ma.MaskedArray, np.ndarray)) and hasattr(column_data, 'mask'):
            serializable_data[col_name] = column_data.filled(np.nan)  # Remplacer les valeurs masquées par NaN
        else:
            try:
                serializable_data[col_name] = np.array(column_data)  # Convertir en tableau NumPy
            except Exception as e:
                print(f"Error when converting {col_name}: {e}")
                serializable_data[col_name] = None  # Ou gérer comme vous le souhaitez
    return serializable_data



#######################################################################################################################
########################################### CALCUL SNR: ##############################################################
#######################################################################################################################



def calculate_SNR_table(instru, table="Archive", thermal_model="None", reflected_model="None", apodizer="NO_SP", strehl="NO_JQ", systematic=False, PCA=False, Nc=20):
    """
    Compute every SNR for every valid planet inside "table" for "instru"
    """
    exposure_time = 120
    time1         = time.time()
    if table == "Archive":
        planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
        planet_table = planet_table[~get_mask(planet_table, "AngSep")]
        path         = archive_path
    elif table == "Simulated":
        planet_table = load_planet_table("Simulated_Pull_for_FastCurves.ecsv")
        if instru == "HARMONI" or instru == "ERIS":
            planet_table = planet_table[(90-np.abs(-24.6-planet_table["Dec"].value))>30]
        path = simulated_path
    config_data = get_config_data(instru)
    
    # WORKING ANGLE
    iwa, owa = get_wa(config_data=config_data, band="INSTRU", apodizer=apodizer, sep_unit="mas")
    planet_table = planet_table[planet_table['AngSep'] > iwa * u.mas] # filtering planets with SMA below IWA
    
    if thermal_model == "Exo-REM": # filtering to the Exo-REM temperature range (if needed)
        if globals()["lmin_"+instru] >= 1 and globals()["lmax_"+instru] <= 5.3: # very high res
            planet_table = planet_table[(planet_table['PlanetTeq'] < 2000*u.K)]
        else:
            planet_table = planet_table[(planet_table['PlanetTeq'] > 400*u.K) & (planet_table['PlanetTeq'] < 2000*u.K)]
    
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table['signal_INSTRU']     = np.full(len(planet_table), np.nan)
    planet_table['sigma_fund_INSTRU'] = np.full(len(planet_table), np.nan)
    planet_table['sigma_syst_INSTRU'] = np.full(len(planet_table), np.nan)
    planet_table['DIT_INSTRU']        = np.full(len(planet_table), np.nan)
    for band in bands:
        planet_table['signal_'+band]     = np.full(len(planet_table), np.nan)
        planet_table['sigma_fund_'+band] = np.full(len(planet_table), np.nan)
        planet_table['sigma_syst_'+band] = np.full(len(planet_table), np.nan)
        planet_table['DIT_'+band]        = np.full(len(planet_table), np.nan)
    
    # K-band    
    R_K    = 10_000 # only for photometric purposes (does not need more resolution)
    dl_K   = ((lmin_K+lmax_K)/2)/(2*R_K)
    wave_K = np.arange(lmin_K, lmax_K, dl_K)
    
    # instru-band    
    lmin_instru = config_data["lambda_range"]["lambda_min"] # in µm
    lmax_instru = config_data["lambda_range"]["lambda_max"] # in µm
    R_instru    = R0_max # abritrary resolution (needs to be high enough)
    dl_instru   = ((lmin_instru+lmax_instru)/2)/(2*R_instru)
    wave_instru = np.arange(0.98*lmin_instru, 1.02*lmax_instru, dl_instru)
    
    # vega spectrum on K-band and instru-band
    vega_spectrum   = load_vega_spectrum()
    vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm = False)
    vega_spectrum   = vega_spectrum.interpolate_wavelength(wave_instru, renorm = False)
    
    if systematic:
        if PCA:
            print("\n "+instru+" ("+apodizer+" & "+strehl+") with systematics+PCA ("+thermal_model+" & "+reflected_model+")")
        else:
            print("\n "+instru+" ("+apodizer+" & "+strehl+") with systematics ("+thermal_model+" & "+reflected_model+")")
    else:
        print("\n "+instru+" ("+apodizer+" & "+strehl+") without systematics ("+thermal_model+" & "+reflected_model+")")
    
    band0 = "instru"
    
    if PCA: # if PCA, no multiprocessing (otherwise it crashes ?)
        for idx in tqdm(range(len(planet_table))):
            args = (idx, planet_table[idx]["StarINSTRUmag("+instru+")"], make_table_serializable(planet_table[idx]), instru, thermal_model, reflected_model, wave_instru, wave_K, vega_spectrum, vega_spectrum_K, lmin_instru, lmax_instru, band0, exposure_time, name_model, systematic, apodizer, strehl, PCA, Nc)
            idx, planet_spectrum, planet_thermal, planet_reflected, star_spectrum, mag_p, name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = process_SNR_table(args)
            # Recalculates the magnitude in case the thermal model is no longer BT-Settl or the reflected model is no longer PICASO (the mag changes with regards to the raw archive table with the estimated magnbitudes)
            planet_table[idx]['PlanetINSTRUmag('+instru+')('+spectrum_contributions+')'] = mag_p
            for band in bands:
                if lmin_instru < globals()["lmin_"+band] and globals()["lmax_"+band] < lmax_instru:
                    planet_table[idx]['Planet'+band+'mag('+spectrum_contributions+')'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave_instru>globals()["lmin_"+band])&(wave_instru<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+band]) & (wave_instru<globals()["lmax_"+band])]))
            SNR     = np.sqrt(exposure_time/DIT_band) * signal_planet / np.sqrt( sigma_fund_planet**2 + (exposure_time/DIT_band)*sigma_syst_planet**2 )
            idx_max = SNR.argmax() # all quantities are saved in e-/DIT
            planet_table[idx]['signal_INSTRU']     = signal_planet[idx_max]
            planet_table[idx]['sigma_fund_INSTRU'] = sigma_fund_planet[idx_max]
            planet_table[idx]['sigma_syst_INSTRU'] = sigma_syst_planet[idx_max]
            planet_table[idx]['DIT_INSTRU']        = DIT_band[idx_max]
            for nb, band in enumerate(name_band):
                planet_table[idx]['signal_'+band]     = signal_planet[nb]
                planet_table[idx]['sigma_fund_'+band] = sigma_fund_planet[nb]
                planet_table[idx]['sigma_syst_'+band] = sigma_syst_planet[nb]
                planet_table[idx]['DIT_'+band]        = DIT_band[nb]
    
    else: # if no PCA, uses multiprocessing
        with Pool(processes=cpu_count()//2) as pool: # Utilisation de multiprocessing pour paralléliser les combinaisons i, j
            results = list(tqdm(pool.imap(process_SNR_table, [(idx, planet_table[idx]["StarINSTRUmag("+instru+")"], make_table_serializable(planet_table[idx]), instru, thermal_model, reflected_model, wave_instru, wave_K, vega_spectrum, vega_spectrum_K, lmin_instru, lmax_instru, band0, exposure_time, name_model, systematic, apodizer, strehl, PCA, Nc) for idx in range(len(planet_table))]), total=len(planet_table)))
            for (idx, planet_spectrum, planet_thermal, planet_reflected, star_spectrum, mag_p, name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band) in results:
                # Recalculates the magnitude in case the thermal model is no longer BT-Settl or the reflected model is no longer PICASO (the mag changes with regards to the raw archive table with the estimated magnbitudes)
                planet_table[idx]['PlanetINSTRUmag('+instru+')('+spectrum_contributions+')'] = mag_p
                for band in bands:
                    if lmin_instru < globals()["lmin_"+band] and globals()["lmax_"+band] < lmax_instru:
                        planet_table[idx]['Planet'+band+'mag('+spectrum_contributions+')'] = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave_instru>globals()["lmin_"+band])&(wave_instru<globals()["lmax_"+band])])/np.nanmean(vega_spectrum.flux[(wave_instru>globals()["lmin_"+band]) & (wave_instru<globals()["lmax_"+band])]))
                SNR     = np.sqrt(exposure_time/DIT_band) * signal_planet / np.sqrt( sigma_fund_planet**2 + (exposure_time/DIT_band)*sigma_syst_planet**2 )
                idx_max = SNR.argmax() # all quantities are saved in e-/DIT
                planet_table[idx]['signal_INSTRU']     = signal_planet[idx_max]
                planet_table[idx]['sigma_fund_INSTRU'] = sigma_fund_planet[idx_max]
                planet_table[idx]['sigma_syst_INSTRU'] = sigma_syst_planet[idx_max]
                planet_table[idx]['DIT_INSTRU']        = DIT_band[idx_max]
                for nb, band in enumerate(name_band):
                    planet_table[idx]['signal_'+band]     = signal_planet[nb]
                    planet_table[idx]['sigma_fund_'+band] = sigma_fund_planet[nb]
                    planet_table[idx]['sigma_syst_'+band] = sigma_syst_planet[nb]
                    planet_table[idx]['DIT_'+band]        = DIT_band[nb]
        
    print('\n Calculating SNR took {0:.3f} s'.format(time.time()-time1))
    if systematic:
        if PCA:
            planet_table.write(path+table+"_Pull_"+instru+"_"+apodizer+"_"+strehl+"_with_systematics+PCA_"+name_model+".ecsv", format='ascii.ecsv', overwrite=True)
        else:
            planet_table.write(path+table+"_Pull_"+instru+"_"+apodizer+"_"+strehl+"_with_systematics_"+name_model+".ecsv", format='ascii.ecsv', overwrite=True)
    else:
        planet_table.write(path+table+"_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv", format='ascii.ecsv', overwrite=True)

def process_SNR_table(args):
    idx, mag_s, planet, instru, thermal_model, reflected_model, wave_instru, wave_K, vega_spectrum, vega_spectrum_K, lmin_instru, lmax_instru, band0, exposure_time, name_model, systematic, apodizer, strehl, PCA, Nc = args
    planet_spectrum, planet_thermal, planet_reflected, star_spectrum = thermal_reflected_spectrum(planet=planet, instru=instru, thermal_model=thermal_model, reflected_model=reflected_model, wave_instru=wave_instru, wave_K=wave_K, vega_spectrum_K=vega_spectrum_K, show=False)
    mag_p = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave_instru>lmin_instru)&(wave_instru<lmax_instru)])/np.nanmean(vega_spectrum.flux[(wave_instru>lmin_instru) & (wave_instru<lmax_instru)]))
    name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band = FastCurves(instru=instru, calculation="SNR", systematic=systematic, T_planet=float(planet["PlanetTeq"].value), lg_planet=float(planet["PlanetLogg"].value), mag_star=mag_s, band0=band0, T_star=float(planet["StarTeff"].value), lg_star=float(planet["StarLogg"].value), exposure_time=exposure_time, model_planet="None", mag_planet=mag_p, separation_planet=float(planet["AngSep"].value/1000), planet_name="None", return_SNR_planet=True, show_plot=False, verbose=False, planet_spectrum=planet_spectrum.copy(), star_spectrum=star_spectrum.copy(), apodizer=apodizer, strehl=strehl, PCA=PCA, Nc=Nc)
    return idx, planet_spectrum, planet_thermal, planet_reflected, star_spectrum, mag_p, name_band, SNR_planet, signal_planet, sigma_fund_planet, sigma_syst_planet, DIT_band



def all_SNR_table(table="Archive"): # takes ~ 13 hours
    """
    To compute SNR for every instruments with different model spectra
    """

    time0 = time.time()
    for instru in instru_name_list:
        if instru not in ["HARMONI", "ANDES"]:
            config_data = get_config_data(instru)
            if config_data["lambda_range"]["lambda_max"] > 6:
                thermal_models   = ["None", "BT-Settl", "Exo-REM"]
                reflected_models = ["None"]
            else:
                thermal_models   = ["None", "BT-Settl", "Exo-REM", "PICASO"]
                reflected_models = ["None", "tellurics", "flat", "PICASO"]
            apodizers = [apodizer for apodizer in config_data["apodizers"]]
            if config_data["base"] == "ground":
                strehls = config_data["strehls"]
            elif config_data["base"] == "space":
                strehls = ["NO_JQ"]
            for apodizer in apodizers:
                for strehl in strehls:
                    for thermal_model in thermal_models:
                        for reflected_model in reflected_models:
                            if thermal_model == "None" and reflected_model == "None":
                                continue
                            else:
                                calculate_SNR_table(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, systematic=False)
                                if instru in instru_with_systematics:
                                    calculate_SNR_table(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, systematic=True)
                                    calculate_SNR_table(instru=instru, table=table, thermal_model=thermal_model, reflected_model=reflected_model, apodizer=apodizer, strehl=strehl, systematic=True, PCA=True, Nc=20)
        
    print('\n Calculating all SNR took {0:.3f} s'.format(time.time()-time0))



def all_simulated_SNR_table():
    calculate_SNR_table("HARMONI", table="Simulated", thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", systematic=False)
    calculate_SNR_table("ERIS", table="Simulated", thermal_model="BT-Settl", reflected_model="PICASO", systematic=False)
    calculate_SNR_table("MIRIMRS", table="Simulated", thermal_model="BT-Settl", systematic=False) # without systematics
    calculate_SNR_table("MIRIMRS", table="Simulated", thermal_model="BT-Settl", systematic=True, exposure_time=1e9) # systematic limit
    calculate_SNR_table("NIRCam", table="Simulated", thermal_model="BT-Settl", reflected_model="PICASO", systematic=False)
    calculate_SNR_table("NIRSpec", table="Simulated", thermal_model="BT-Settl", reflected_model="PICASO", systematic=False)



#######################################################################################################################
################################################# PLOTS: ##############################################################
#######################################################################################################################



def archive_yield_instrus_plot_texp(thermal_model="BT-Settl", reflected_model="PICASO", band="INSTRU", fraction=False):
    
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table_harmoni          = load_planet_table("Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_andes            = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_without_systematics_"+name_model+".ecsv")
    planet_table_eris             = load_planet_table("Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_"+name_model+".ecsv")
    planet_table_mirimrs_non_syst = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_"+thermal_model+".ecsv")
    planet_table_mirimrs_syst     = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_"+thermal_model+".ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_"+thermal_model+".ecsv")
    planet_table_nircam           = load_planet_table("Archive_Pull_NIRCam_NO_SP_NO_JQ_without_systematics_"+name_model+".ecsv")
    planet_table_nirspec_non_syst = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_"+name_model+".ecsv")
    planet_table_nirspec_syst     = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_"+name_model+".ecsv")
    planet_table_nirspec_syst_pca = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_"+name_model+".ecsv") # à modifier
        
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
        ratio = 100 ; norm_harmoni = len(planet_table_harmoni) ; norm_andes = len(planet_table_andes) ; norm_eris = len(planet_table_eris) ; norm_mirimrs = len(planet_table_mirimrs_non_syst) ; norm_nircam = len(planet_table_nircam) ; norm_nirspec = len(planet_table_nirspec_non_syst)
    else:
        ratio = 1 ; norm_harmoni = 1 ; norm_andes = 1 ; norm_eris = 1 ; norm_mirimrs = 1 ; norm_nircam = 1 ; norm_nirspec = 1
    
    for i in range(len(exposure_time)):
        SNR_harmoni          = np.sqrt(exposure_time[i]/planet_table_harmoni['DIT_'+band]) * planet_table_harmoni['signal_'+band] / np.sqrt(  planet_table_harmoni['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_harmoni['DIT_'+band])*planet_table_harmoni['sigma_syst_'+band]**2 )
        SNR_andes            = np.sqrt(exposure_time[i]/planet_table_andes['DIT_'+band]) * planet_table_andes['signal_'+band] / np.sqrt(  planet_table_andes['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_andes['DIT_'+band])*planet_table_andes['sigma_syst_'+band]**2 )
        SNR_eris             = np.sqrt(exposure_time[i]/planet_table_eris['DIT_'+band]) * planet_table_eris['signal_'+band] / np.sqrt(  planet_table_eris['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_eris['DIT_'+band])*planet_table_eris['sigma_syst_'+band]**2 )
        SNR_mirimrs_non_syst = np.sqrt(exposure_time[i]/planet_table_mirimrs_non_syst['DIT_'+band]) * planet_table_mirimrs_non_syst['signal_'+band] / np.sqrt(  planet_table_mirimrs_non_syst['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_mirimrs_non_syst['DIT_'+band])*planet_table_mirimrs_non_syst['sigma_syst_'+band]**2 )
        SNR_mirimrs_syst     = np.sqrt(exposure_time[i]/planet_table_mirimrs_syst['DIT_'+band]) * planet_table_mirimrs_syst['signal_'+band] / np.sqrt(  planet_table_mirimrs_syst['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_mirimrs_syst['DIT_'+band])*planet_table_mirimrs_syst['sigma_syst_'+band]**2 )
        SNR_mirimrs_syst_pca = np.sqrt(exposure_time[i]/planet_table_mirimrs_syst_pca['DIT_'+band]) * planet_table_mirimrs_syst_pca['signal_'+band] / np.sqrt(  planet_table_mirimrs_syst_pca['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_mirimrs_syst_pca['DIT_'+band])*planet_table_mirimrs_syst_pca['sigma_syst_'+band]**2 )
        SNR_nircam           = np.sqrt(exposure_time[i]/planet_table_nircam['DIT_'+band]) * planet_table_nircam['signal_'+band] / np.sqrt(  planet_table_nircam['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_nircam['DIT_'+band])*planet_table_nircam['sigma_syst_'+band]**2 )
        SNR_nirspec_non_syst = np.sqrt(exposure_time[i]/planet_table_nirspec_non_syst['DIT_'+band]) * planet_table_nirspec_non_syst['signal_'+band] / np.sqrt(  planet_table_nirspec_non_syst['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_nirspec_non_syst['DIT_'+band])*planet_table_nirspec_non_syst['sigma_syst_'+band]**2 )
        SNR_nirspec_syst     = np.sqrt(exposure_time[i]/planet_table_nirspec_syst['DIT_'+band]) * planet_table_nirspec_syst['signal_'+band] / np.sqrt(  planet_table_nirspec_syst['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_nirspec_syst['DIT_'+band])*planet_table_nirspec_syst['sigma_syst_'+band]**2 )
        SNR_nirspec_syst_pca = np.sqrt(exposure_time[i]/planet_table_nirspec_syst_pca['DIT_'+band]) * planet_table_nirspec_syst_pca['signal_'+band] / np.sqrt(  planet_table_nirspec_syst_pca['sigma_fund_'+band]**2 + (exposure_time[i]/planet_table_nirspec_syst_pca['DIT_'+band])*planet_table_nirspec_syst_pca['sigma_syst_'+band]**2 )
        
        yield_harmoni[i]          = ratio*len(planet_table_harmoni[SNR_harmoni>5]) / norm_harmoni
        yield_andes[i]            = ratio*len(planet_table_andes[SNR_andes>5]) / norm_andes
        yield_eris[i]             = ratio*len(planet_table_eris[SNR_eris>5]) / norm_eris
        yield_mirimrs_non_syst[i] = ratio*len(planet_table_mirimrs_non_syst[SNR_mirimrs_non_syst>5]) / norm_mirimrs
        yield_mirimrs_syst[i]     = ratio*len(planet_table_mirimrs_syst[SNR_mirimrs_syst>5]) / norm_mirimrs
        yield_mirimrs_syst_pca[i] = ratio*len(planet_table_mirimrs_syst_pca[SNR_mirimrs_syst_pca>5]) / norm_mirimrs
        yield_nircam[i]           = ratio*len(planet_table_nircam[SNR_nircam>5]) / norm_nircam
        yield_nirspec_non_syst[i] = ratio*len(planet_table_nirspec_non_syst[SNR_nirspec_non_syst>5]) / norm_nirspec
        yield_nirspec_syst[i]     = ratio*len(planet_table_nirspec_syst[SNR_nirspec_syst>5]) / norm_nirspec
        yield_nirspec_syst_pca[i] = ratio*len(planet_table_nirspec_syst_pca[SNR_nirspec_syst_pca>5]) / norm_nirspec

    plt.figure(dpi=300, figsize=(9, 6))
    plt.plot(exposure_time, yield_harmoni, 'b', label="ELT/HARMONI")
    plt.plot(exposure_time, yield_andes, 'gray', label="ELT/ANDES")
    plt.plot(exposure_time, yield_eris, 'r', label="VLT/ERIS")
    plt.plot(exposure_time, yield_mirimrs_non_syst, 'g', label="JWST/MIRI/MRS")
    plt.plot(exposure_time, yield_mirimrs_syst, 'g--')
    plt.plot(exposure_time, yield_mirimrs_syst_pca, 'g:')
    plt.plot(exposure_time, yield_nircam, 'm', label="JWST/NIRCam")
    plt.plot(exposure_time, yield_nirspec_non_syst, 'c', label="JWST/NIRSpec/IFU")
    plt.plot(exposure_time, yield_nirspec_syst, 'c--')
    plt.plot(exposure_time, yield_nirspec_syst_pca, 'c:')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure Time per Target [mn]', fontsize=14)
    plt.xlim(exposure_time[0], exposure_time[-1])
    if fraction:
        plt.ylabel('Fraction of Planets Re-detected [%]', fontsize=14)
    else:
        plt.ylabel('Number of Planets Re-detected', fontsize=14)
        plt.yscale('log')
    plt.title('Known Exoplanets Detection Yield', fontsize=16)
    plt.legend(loc="upper left", fontsize=12)
    plt.tick_params(axis='both', labelsize=14)
    ax = plt.gca()
    ax_legend = ax.twinx()
    ax_legend.plot([], [], 'k-', label='Without Systematics', linewidth=1.5)
    ax_legend.plot([], [], 'k--', label='With Systematics', linewidth=1.5)
    ax_legend.plot([], [], 'k:', label='With Systematics + PCA', linewidth=1.5)
    ax_legend.legend(loc='lower right', fontsize=12)
    ax_legend.tick_params(axis='y', colors='w')  # Masquer l'axe secondaire
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(dpi=300, figsize=(9, 6))
    plt.plot(exposure_time, yield_mirimrs_non_syst, 'g', label="JWST/MIRI/MRS")
    plt.plot(exposure_time, yield_mirimrs_syst, 'g--')
    plt.plot(exposure_time, yield_mirimrs_syst_pca, 'g:')
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure Time per Target [mn]', fontsize=14)
    plt.xlim(exposure_time[0], exposure_time[-1])
    plt.ylabel('Number of Planets Re-detected', fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.tick_params(axis='both', labelsize=14)
    ax = plt.gca()
    ax_legend = ax.twinx()
    ax_legend.plot([], [], 'k-', label='Without Systematics', linewidth=1.5)
    ax_legend.plot([], [], 'k--', label='With Systematics', linewidth=1.5)
    ax_legend.plot([], [], 'k:', label='With Systematics + PCA', linewidth=1.5)
    ax_legend.legend(loc='lower right', fontsize=12)
    ax_legend.tick_params(axis='y', colors='w')  # Masquer l'axe secondaire
    plt.tight_layout()
    plt.show()

def archive_yield_instrus_plot_ptypes(exposure_time=120, thermal_model="BT-Settl", reflected_model="tellurics", band="INSTRU", fraction=False):
    
    snr_threshold = 5
    
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    planet_table_harmoni          = load_planet_table("Archive_Pull_HARMONI_SP_Prox_JQ1_without_systematics_"+name_model+".ecsv")
    planet_table_andes            = load_planet_table("Archive_Pull_ANDES_NO_SP_MED_without_systematics_"+name_model+".ecsv")
    planet_table_eris             = load_planet_table("Archive_Pull_ERIS_NO_SP_JQ0_without_systematics_"+name_model+".ecsv")
    planet_table_mirimrs_non_syst = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_without_systematics_"+thermal_model+".ecsv")
    planet_table_mirimrs_syst     = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics_"+thermal_model+".ecsv")
    planet_table_mirimrs_syst_pca = load_planet_table("Archive_Pull_MIRIMRS_NO_SP_NO_JQ_with_systematics+PCA_"+thermal_model+".ecsv")
    planet_table_nircam           = load_planet_table("Archive_Pull_NIRCam_NO_SP_NO_JQ_without_systematics_"+name_model+".ecsv")
    planet_table_nirspec_non_syst = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_without_systematics_"+name_model+".ecsv")
    planet_table_nirspec_syst     = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics_"+name_model+".ecsv")
    planet_table_nirspec_syst_pca = load_planet_table("Archive_Pull_NIRSpec_NO_SP_NO_JQ_with_systematics+PCA_"+name_model+".ecsv") # à modifier
    
    planet_table_harmoni["SNR"]          = np.sqrt(exposure_time/planet_table_harmoni['DIT_'+band]) * planet_table_harmoni['signal_'+band] / np.sqrt(  planet_table_harmoni['sigma_fund_'+band]**2 + (exposure_time/planet_table_harmoni['DIT_'+band])*planet_table_harmoni['sigma_syst_'+band]**2 )
    planet_table_andes["SNR"]            = np.sqrt(exposure_time/planet_table_andes['DIT_'+band]) * planet_table_andes['signal_'+band] / np.sqrt(  planet_table_andes['sigma_fund_'+band]**2 + (exposure_time/planet_table_andes['DIT_'+band])*planet_table_andes['sigma_syst_'+band]**2 )
    planet_table_eris["SNR"]             = np.sqrt(exposure_time/planet_table_eris['DIT_'+band]) * planet_table_eris['signal_'+band] / np.sqrt(  planet_table_eris['sigma_fund_'+band]**2 + (exposure_time/planet_table_eris['DIT_'+band])*planet_table_eris['sigma_syst_'+band]**2 )
    planet_table_mirimrs_non_syst["SNR"] = np.sqrt(exposure_time/planet_table_mirimrs_non_syst['DIT_'+band]) * planet_table_mirimrs_non_syst['signal_'+band] / np.sqrt(  planet_table_mirimrs_non_syst['sigma_fund_'+band]**2 + (exposure_time/planet_table_mirimrs_non_syst['DIT_'+band])*planet_table_mirimrs_non_syst['sigma_syst_'+band]**2 )
    planet_table_mirimrs_syst["SNR"]     = np.sqrt(exposure_time/planet_table_mirimrs_syst['DIT_'+band]) * planet_table_mirimrs_syst['signal_'+band] / np.sqrt(  planet_table_mirimrs_syst['sigma_fund_'+band]**2 + (exposure_time/planet_table_mirimrs_syst['DIT_'+band])*planet_table_mirimrs_syst['sigma_syst_'+band]**2 )
    planet_table_mirimrs_syst_pca["SNR"] = np.sqrt(exposure_time/planet_table_mirimrs_syst_pca['DIT_'+band]) * planet_table_mirimrs_syst_pca['signal_'+band] / np.sqrt(  planet_table_mirimrs_syst_pca['sigma_fund_'+band]**2 + (exposure_time/planet_table_mirimrs_syst_pca['DIT_'+band])*planet_table_mirimrs_syst_pca['sigma_syst_'+band]**2 )
    planet_table_nircam["SNR"]           = np.sqrt(exposure_time/planet_table_nircam['DIT_'+band]) * planet_table_nircam['signal_'+band] / np.sqrt(  planet_table_nircam['sigma_fund_'+band]**2 + (exposure_time/planet_table_nircam['DIT_'+band])*planet_table_nircam['sigma_syst_'+band]**2 )
    planet_table_nirspec_non_syst["SNR"] = np.sqrt(exposure_time/planet_table_nirspec_non_syst['DIT_'+band]) * planet_table_nirspec_non_syst['signal_'+band] / np.sqrt(  planet_table_nirspec_non_syst['sigma_fund_'+band]**2 + (exposure_time/planet_table_nirspec_non_syst['DIT_'+band])*planet_table_nirspec_non_syst['sigma_syst_'+band]**2 )
    planet_table_nirspec_syst["SNR"]     = np.sqrt(exposure_time/planet_table_nirspec_syst['DIT_'+band]) * planet_table_nirspec_syst['signal_'+band] / np.sqrt(  planet_table_nirspec_syst['sigma_fund_'+band]**2 + (exposure_time/planet_table_nirspec_syst['DIT_'+band])*planet_table_nirspec_syst['sigma_syst_'+band]**2 )
    planet_table_nirspec_syst_pca["SNR"] = np.sqrt(exposure_time/planet_table_nirspec_syst_pca['DIT_'+band]) * planet_table_nirspec_syst_pca['signal_'+band] / np.sqrt(  planet_table_nirspec_syst_pca['sigma_fund_'+band]**2 + (exposure_time/planet_table_nirspec_syst_pca['DIT_'+band])*planet_table_nirspec_syst_pca['sigma_syst_'+band]**2 )

    mp_harmoni          = {ptype: find_matching_planets(criteria, planet_table_harmoni.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_andes            = {ptype: find_matching_planets(criteria, planet_table_andes.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_eris             = {ptype: find_matching_planets(criteria, planet_table_eris.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_mirimrs_non_syst = {ptype: find_matching_planets(criteria, planet_table_mirimrs_non_syst.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_mirimrs_syst     = {ptype: find_matching_planets(criteria, planet_table_mirimrs_syst.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_mirimrs_syst_pca = {ptype: find_matching_planets(criteria, planet_table_mirimrs_syst_pca.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_nircam           = {ptype: find_matching_planets(criteria, planet_table_nircam.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_nirspec_non_syst = {ptype: find_matching_planets(criteria, planet_table_nirspec_non_syst.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_nirspec_syst     = {ptype: find_matching_planets(criteria, planet_table_nirspec_syst.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    mp_nirspec_syst_pca = {ptype: find_matching_planets(criteria, planet_table_nirspec_syst_pca.to_pandas(), "multi") for ptype, criteria in planet_types.items()}
    
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
        
        yield_harmoni[i]          = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_harmoni[ptype]) / norm_harmoni if norm_harmoni > 0 else 0
        yield_andes[i]            = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_andes[ptype]) / norm_andes if norm_andes > 0 else 0
        yield_eris[i]             = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_eris[ptype]) / norm_eris if norm_eris > 0 else 0
        yield_mirimrs_non_syst[i] = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_mirimrs_non_syst[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst[i]     = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_mirimrs_syst[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_mirimrs_syst_pca[i] = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_mirimrs_syst_pca[ptype]) / norm_mirimrs if norm_mirimrs > 0 else 0
        yield_nircam[i]           = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_nircam[ptype]) / norm_nircam if norm_nircam > 0 else 0
        yield_nirspec_non_syst[i] = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_nirspec_non_syst[ptype]) / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst[i]     = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_nirspec_syst[ptype]) / norm_nirspec if norm_nirspec > 0 else 0
        yield_nirspec_syst_pca[i] = ratio * sum(planet["SNR"] > snr_threshold for planet in mp_nirspec_syst_pca[ptype]) / norm_nirspec if norm_nirspec > 0 else 0

    # Définition des couleurs et des styles (hachures pour différencier les cas avec systématiques)
    colors = {"ELT/HARMONI":      "blue",
              "ELT/ANDES":        "gray",
              "VLT/ERIS":         "red",
              "JWST/MIRI/MRS":    "green",
              "JWST/NIRCam":      "magenta",
              "JWST/NIRSpec/IFU": "cyan"}
    
    # Configuration du diagramme en barres groupées
    n_series  = 10  # 3 instruments (ELT/HARMONI, ELT/ANDES, VLT/ERIS) + 3 (JWST/MIRI/MRS) + 1 (JWST/NIRCam) + 3 (JWST/NIRSpec/IFU)
    bar_width = 0.08
    indices   = np.arange(len(planet_types_array))
    
    plt.figure(figsize=(14, 8), dpi=300)
    
    if not fraction:
        plt.bar(indices - 4.5*bar_width, N_harmoni, bar_width, edgecolor="black", color="white")
        plt.bar(indices - 3.5*bar_width, N_andes,   bar_width, edgecolor="black", color="white")
        plt.bar(indices - 2.5*bar_width, N_eris,    bar_width, edgecolor="black", color="white")
        plt.bar(indices - 1.5*bar_width, N_mirimrs, bar_width, edgecolor="black", color="white")
        plt.bar(indices - 0.5*bar_width, N_mirimrs, bar_width, edgecolor="black", color="white")
        plt.bar(indices + 0.5*bar_width, N_mirimrs, bar_width, edgecolor="black", color="white")
        plt.bar(indices + 1.5*bar_width, N_nircam,  bar_width, edgecolor="black", color="white")
        plt.bar(indices + 2.5*bar_width, N_nirspec, bar_width, edgecolor="black", color="white")
        plt.bar(indices + 3.5*bar_width, N_nirspec, bar_width, edgecolor="black", color="white")
        plt.bar(indices + 4.5*bar_width, N_nirspec, bar_width, edgecolor="black", color="white")
    
    plt.bar(indices - 4.5*bar_width, yield_harmoni,          bar_width, edgecolor="black", color=colors["ELT/HARMONI"],      label="ELT/HARMONI")
    plt.bar(indices - 3.5*bar_width, yield_andes,            bar_width, edgecolor="black", color=colors["ELT/ANDES"],        label="ELT/ANDES")
    plt.bar(indices - 2.5*bar_width, yield_eris,             bar_width, edgecolor="black", color=colors["VLT/ERIS"],         label="VLT/ERIS")
    plt.bar(indices - 1.5*bar_width, yield_mirimrs_non_syst, bar_width, edgecolor="black", color=colors["JWST/MIRI/MRS"],    label="JWST/MIRI/MRS")
    plt.bar(indices - 0.5*bar_width, yield_mirimrs_syst,     bar_width, edgecolor="black", color=colors["JWST/MIRI/MRS"],    label="JWST/MIRI/MRS (with syst)",        hatch='//')
    plt.bar(indices + 0.5*bar_width, yield_mirimrs_syst_pca, bar_width, edgecolor="black", color=colors["JWST/MIRI/MRS"],    label="JWST/MIRI/MRS (with syst+PCA)",    hatch='xx')
    plt.bar(indices + 1.5*bar_width, yield_nircam,           bar_width, edgecolor="black", color=colors["JWST/NIRCam"],      label="JWST/NIRCam")
    plt.bar(indices + 2.5*bar_width, yield_nirspec_non_syst, bar_width, edgecolor="black", color=colors["JWST/NIRSpec/IFU"], label="JWST/NIRSpec/IFU")
    plt.bar(indices + 3.5*bar_width, yield_nirspec_syst,     bar_width, edgecolor="black", color=colors["JWST/NIRSpec/IFU"], label="JWST/NIRSpec/IFU (with syst)",     hatch='//')
    plt.bar(indices + 4.5*bar_width, yield_nirspec_syst_pca, bar_width, edgecolor="black", color=colors["JWST/NIRSpec/IFU"], label="JWST/NIRSpec/IFU (with syst+PCA)", hatch='xx')
    
    
    
    plt.xticks(indices, planet_types_array, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    if fraction:
        plt.ylabel('Fraction of Planets Re-detected [%]', fontsize=14)
    else:
        plt.ylabel('Number of Planets Re-detected', fontsize=14)
        plt.yscale('log')
    plt.title(f'Known Exoplanets Detection Yield for {int(round(exposure_time/60))} h per target', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc="upper center", fontsize=10, frameon=True, edgecolor="black", ncol=2)
    plt.tight_layout()
    plt.show()
    
    
    
    

    
    

def archive_yield_bands_plot_texp(instru="HARMONI", strehl="JQ1", thermal_model="BT-Settl", reflected_model="PICASO",
                             systematic=False, PCA=False, fraction=False):
    config_data = get_config_data(instru)
    # WORKING ANGLE
    try:
        pxscale = min(config_data["pxscale"].values()) * 1000 # mas
    except:
        pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    iwa = pxscale

    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    apodizer = "NO_SP"
    if systematic:
        if PCA:
            planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_with_systematics+PCA_"+name_model+".ecsv")
        else:
            planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_with_systematics_"+name_model+".ecsv")
    else:
        if instru=="HARMONI":
            apodizers    = ["NO_SP", "SP1", "SP_Prox"]
            ls_apodizers = ["-", "--", ":"]
            planet_table = []
            iwa = 30 # mas
            NbPlanet = np.zeros((len(apodizers)))
            for na, apodizer in enumerate(apodizers):
                pt = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
                pt = pt[pt["AngSep"]>iwa*u.mas]
                planet_table.append(pt)
                NbPlanet[na] = len(pt)
        else:
            planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
            NbPlanet = len(planet_table)
    if len(config_data["gratings"])%2 != 0: # for plot colors purposes
        cmap = plt.get_cmap("Spectral", len(config_data["gratings"])+1)
    else:
        cmap = plt.get_cmap("Spectral", len(config_data["gratings"]))
    exposure_time = np.logspace(np.log10(0.1), np.log10(1000), 100)
    if instru == "HARMONI":
        yields = np.zeros((len(config_data["gratings"]), len(apodizers), len(exposure_time)))
    else:
        yields = np.zeros((len(config_data["gratings"]), len(exposure_time)))
    for i, band in enumerate(config_data["gratings"]):
        if instru == "HARMONI":
            for na in range(len(apodizers)):
                for j in range(len(exposure_time)):
                    SNR = np.sqrt(exposure_time[j]/planet_table[na]['DIT_'+band]) * planet_table[na]['signal_'+band] / np.sqrt(  planet_table[na]['sigma_fund_'+band]**2 + (exposure_time[j]/planet_table[na]['DIT_'+band])*planet_table[na]['sigma_syst_'+band]**2 )
                    if fraction :
                        yields[i, na, j] = len(planet_table[na][SNR>5]) * 100/NbPlanet[na]
                    else:
                        yields[i, na, j] = len(planet_table[na][SNR>5])
        else:
            for j in range(len(exposure_time)):
                SNR = np.sqrt(exposure_time[j]/planet_table['DIT_'+band]) * planet_table['signal_'+band] / np.sqrt(  planet_table['sigma_fund_'+band]**2 + (exposure_time[j]/planet_table['DIT_'+band])*planet_table['sigma_syst_'+band]**2 )
                if fraction:
                    yields[i, j] = len(planet_table[SNR>5]) * 100/NbPlanet
                else:
                    yields[i, j] = len(planet_table[SNR>5])


    plt.figure(dpi=300, figsize=(10, 6))
    # Tracer les courbes pour chaque bande de données
    for i, band in enumerate(config_data["gratings"]):
        if instru=="HARMONI":
            for na in range(len(apodizers)):
                if not (yields[i, na]==0).all():
                    if na == 0:
                        plt.plot(exposure_time, yields[i, na], color=cmap(i), label=band, ls=ls_apodizers[na], lw=3)
                    else:
                        plt.plot(exposure_time, yields[i, na], color=cmap(i), ls=ls_apodizers[na], lw=3)
        else:
            plt.plot(exposure_time, yields[i], color=cmap(i), label=band, lw=3)
    # Paramètres du graphique
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.xscale('log')
    plt.xlabel('Exposure Time per Target [mn]', fontsize=14)
    if fraction:
        plt.ylabel('Fraction of Planets Re-detected [%]', fontsize=14)
    else:
        plt.ylabel('Number of Planets Re-detected', fontsize=14)
        plt.yscale('log')
    plt.title('Known Exoplanets Detection Yield with {instru} for {strehl} strehl', fontsize=16)
    plt.title(f"{instru} re-detections statistics with {int(np.max(NbPlanet))} known planets above {iwa} mas for {strehl} strehl \n (with {name_model} model in {spectrum_contributions} light)", fontsize=16)

    plt.legend(loc="upper left", fontsize=10)
    if instru == "HARMONI": # Légende supplémentaire sur l'axe secondaire
        ax = plt.gca()
        ax_legend = ax.twinx()
        for na, apodizer in enumerate(apodizers):
            ax_legend.plot([], [], c="k", ls=ls_apodizers[na], label=apodizer, lw=3)
        ax_legend.legend(loc='lower right', fontsize=10)
        ax_legend.tick_params(axis='y', colors='w')  # Masquer l'axe secondaire
    # Afficher le graphique
    plt.tight_layout()
    plt.show()




#######################################################################################################################

# CORNER PLOTS


    
def detections_corner(instru="HARMONI", exposure_time=600, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", band="INSTRU"):
    smooth_corner = 1
    ndim = 6 # Mp, Rp, Tp, d, a, sep
    config_data = get_config_data(instru)

    # WORKING ANGLE
    if band == "INSTRU":
        try:
            pxscale = min(config_data["pxscale"].values()) * 1000 # mas
        except:
            pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    else:
        try:
            pxscale = config_data["pxscale"][band] * 1000 # mas
        except:
            pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    iwa = max(pxscale, config_data["apodizers"][apodizer].sep)
    if config_data["type"] == "IFU_fiber":
        if band == "INSTRU":
            owa = config_data["FOV_fiber"]/2 * max(config_data["pxscale"].values()) * 1000 # en mas
        else:
            owa = config_data["FOV_fiber"]/2 * config_data["pxscale"][band] * 1000 # en mas
    else:
        owa = config_data["spec"]["FOV"]/2 * 1000 # en mas

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    data_raw = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data_raw,
        bins = 20,
        labels = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles = [0.16, 0.5, 0.84], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = True,
        title_kwargs = {"fontsize": 14, "pad": 10},
        top_ticks = False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        smooth = smooth_corner,
        smooth1d = smooth_corner,
        label_kwargs={"fontsize": 16})

    # DETECTIONS TABLE
    planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
    planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
    SNR = np.sqrt(exposure_time/planet_table['DIT_'+band]) * planet_table['signal_'+band] / np.sqrt(  planet_table['sigma_fund_'+band]**2 + (exposure_time/planet_table['DIT_'+band])*planet_table['sigma_syst_'+band]**2 )
    planet_table = planet_table[SNR>5]
    data = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeq"].value)
    data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
    data[:, 4] = np.log10(np.array(planet_table["Distance"].value))
    data[:, 5] = np.log10(np.array(planet_table["AngSep"].value))
    corner.corner(
        data,
        fig = figure,
        bins = 20,
        quantiles = [0.5], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = False,
        top_ticks = False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        color='r',
        smooth = smooth_corner,
        smooth1d = smooth_corner)

    figure.suptitle(f"{instru} re-detections statistics with {len(planet_table)} / {len(planet_table_raw)} detections between {round(iwa)} and {round(owa)} mas for {round(exposure_time/60)} hours per target \n (with {spectrum_contributions} light with {name_model})", fontsize=18, y=1.05)
    plt.gcf().set_dpi(300)
    plt.show()
    


def detections_corner_instrus_comparison(instru1="HARMONI", instru2="ANDES", apodizer1="NO_SP", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", 
                          exposure_time=600, thermal_model="BT-Settl", reflected_model="PICASO", band="INSTRU"):
    instrus = [instru1, instru2]
    apodizers = [apodizer1, apodizer2]
    strehls = [strehl1, strehl2]
    c_instrus = ["r", "b"]
    smooth_corner = 1
    ndim = 6 # Mp, Rp, Tp, d, a, sep
    alpha_earth = 0.5
    color_earth = "g"
    earth_values = np.array([np.log10(1), np.log10(1), 300, np.log10(1), None, None])

    # WORKING ANGLE
    IWA = np.zeros((len(instrus))) ; OWA = np.zeros((len(instrus)))
    for ni, instru in enumerate(instrus):
        config_data = get_config_data(instru)
        if band == "INSTRU":
            try:
                pxscale = min(config_data["pxscale"].values()) * 1000 # mas
            except:
                pxscale = config_data["spec"]["pxscale"] * 1000 # mas
        else:
            try:
                pxscale = config_data["pxscale"][band] * 1000 # mas
            except:
                pxscale = config_data["spec"]["pxscale"] * 1000 # mas
        iwa = max(pxscale, config_data["apodizers"][apodizers[ni]].sep)
        if config_data["type"] == "IFU_fiber":
            if band == "INSTRU":
                owa = config_data["FOV_fiber"]/2 * max(config_data["pxscale"].values()) * 1000 # en mas
            else:
                owa = config_data["FOV_fiber"]/2 * config_data["pxscale"][band] * 1000 # en mas
        else:
            owa = config_data["spec"]["FOV"]/2 * 1000 # en mas
        IWA[ni] = iwa ; OWA[ni] = owa
    iwa = np.nanmin(IWA) ; owa = np.nanmax(OWA)
    
    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    data_raw = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data_raw,
        bins = 20,
        labels = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles = [0.16, 0.5, 0.84], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = True,
        title_kwargs = {"fontsize": 14, "pad": 10},
        top_ticks = False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        smooth = smooth_corner,
        smooth1d = smooth_corner,
        label_kwargs={"fontsize": 16})
    
    # DETECTIONS TABLES
    yields = np.zeros((len(instrus)))
    for ni, instru in enumerate(instrus):
        apodizer = apodizers[ni]
        strehl = strehls[ni]
        planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
        planet_table = planet_table[(planet_table["AngSep"]>IWA[ni]*u.mas)&(planet_table["AngSep"]<OWA[ni]*u.mas)]
        SNR = np.sqrt(exposure_time/planet_table['DIT_'+band]) * planet_table['signal_'+band] / np.sqrt(  planet_table['sigma_fund_'+band]**2 + (exposure_time/planet_table['DIT_'+band])*planet_table['sigma_syst_'+band]**2 )
        planet_table = planet_table[SNR>5]
        yields[ni] = len(planet_table)
        data = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeq"].value)
        data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
        data[:, 4] = np.log10(np.array(planet_table["Distance"].value))
        data[:, 5] = np.log10(np.array(planet_table["AngSep"].value))
        corner.corner(
            data,
            fig = figure,
            bins = 20,
            quantiles = [0.5], # below -+1 sigma 
            levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
            show_titles = False,
            top_ticks = False,
            plot_density = True,
            plot_contours = True,
            fill_contours = True,
            color=c_instrus[ni],
            smooth = smooth_corner,
            smooth1d = smooth_corner)
            
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
    for ni, instru in enumerate(instrus):
        handles.append(mlines.Line2D([], [], color="w", label=""))
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=c_instrus[ni], label=instru+f" ({round(yields[ni])} / {len(planet_table_raw)})"))
    plt.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=14, title="Instruments:", title_fontsize=16)
    handles = [mlines.Line2D([], [], linestyle="-", marker="s", color=color_earth, label="Earth")]
    ax_legend = plt.gca().twinx()
    ax_legend.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="lower right", fontsize=14)
    figure.suptitle(f"{instru1} VS {instru2} re-detections statistics with {len(planet_table_raw)} known planets between {round(iwa)} and {round(owa)} mas for {round(exposure_time/60)} hours per target \n (with {name_model} model in {spectrum_contributions} light)", fontsize=18, y=1.06)
    plt.gcf().set_dpi(300)
    plt.show()



def detections_corner_models_comparison(model1="tellurics", model2="flat", 
                                       instru="ANDES", apodizer="NO_SP", strehl="MED", exposure_time=600, band="INSTRU"):
    models = [model1, model2]
    c_models = ["r", "b"]
    smooth_corner = 1
    ndim = 6 # Mp, Rp, Tp, d, a, sep
    alpha_earth = 0.5
    color_earth = "g"
    earth_values = np.array([np.log10(1), np.log10(1), 300, np.log10(1), None, None])
    config_data = get_config_data(instru)
    
    # WORKING ANGLE
    if band == "INSTRU":
        try:
            pxscale = min(config_data["pxscale"].values()) * 1000 # mas
        except:
            pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    else:
        try:
            pxscale = config_data["pxscale"][band] * 1000 # mas
        except:
            pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    iwa = max(pxscale, config_data["apodizers"][apodizer].sep)
    if config_data["type"] == "IFU_fiber":
        if band == "INSTRU":
            owa = config_data["FOV_fiber"]/2 * max(config_data["pxscale"].values()) * 1000 # en mas
        else:
            owa = config_data["FOV_fiber"]/2 * config_data["pxscale"][band] * 1000 # en mas
    else:
        owa = config_data["spec"]["FOV"]/2 * 1000 # en mas
    
    # MODELS NAME
    if model1 in thermal_models and model2 in thermal_models:
        spectrum_contributions = "thermal"
    elif model1 in reflected_models and model2 in reflected_models:
        spectrum_contributions = "reflected"
    else:
        raise KeyError("WRONG MODEL1 OR MODEL2")

    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    data_raw = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data_raw,
        bins = 20,
        labels = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles = [0.16, 0.5, 0.84], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = True,
        title_kwargs = {"fontsize": 14, "pad": 10},
        top_ticks = False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        smooth = smooth_corner,
        smooth1d = smooth_corner,
        label_kwargs={"fontsize": 16})
    
    # DETECTIONS TABLES
    yields = np.zeros((len(models)))
    for nm, model in enumerate(models):
        if model == "PICASO":
            name_model = "PICASO_"+spectrum_contributions+"_only"
        else:
            name_model = model
        planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
        planet_table = planet_table[(planet_table["AngSep"]>iwa*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
        SNR = np.sqrt(exposure_time/planet_table['DIT_'+band]) * planet_table['signal_'+band] / np.sqrt(  planet_table['sigma_fund_'+band]**2 + (exposure_time/planet_table['DIT_'+band])*planet_table['sigma_syst_'+band]**2 )
        planet_table = planet_table[SNR>5]
        yields[nm] = len(planet_table)
        data = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeq"].value)
        data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
        data[:, 4] = np.log10(np.array(planet_table["Distance"].value))
        data[:, 5] = np.log10(np.array(planet_table["AngSep"].value))
        corner.corner(
            data,
            fig = figure,
            bins = 20,
            quantiles = [0.5], # below -+1 sigma 
            levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
            show_titles = False,
            top_ticks = False,
            plot_density = True,
            plot_contours = True,
            fill_contours = True,
            color=c_models[nm],
            smooth = smooth_corner,
            smooth1d = smooth_corner)

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
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=c_models[nm], label=model+f" ({round(yields[nm])} / {len(planet_table_raw)})"))
    plt.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=14, title="Models:", title_fontsize=16)
    handles = [mlines.Line2D([], [], linestyle="-", marker="s", color=color_earth, label="Earth")]
    ax_legend = plt.gca().twinx()
    ax_legend.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="lower right", fontsize=14)
    figure.suptitle(f"{instru} re-detections statistics with {len(planet_table_raw)} known planets between {round(iwa)} and {round(owa)} mas for {round(exposure_time/60)} hours per target \n (with {spectrum_contributions} light)", fontsize=18, y=1.05)
    plt.gcf().set_dpi(300)
    plt.show()



def detections_corner_apodizers_comparison(exposure_time=600, thermal_model="BT-Settl", reflected_model="tellurics", band="INSTRU"):
    instru = "HARMONI"
    apodizers = ["SP_Prox", "NO_SP", "SP1"]
    strehls = ["JQ1", "JQ1", "JQ1"]
    c_apodizers = ["r", "b", "g"]
    smooth_corner = 1
    ndim = 6 # Mp, Rp, Tp, d, a, sep
    alpha_earth = 0.5
    color_earth = "k"
    earth_values = np.array([np.log10(1), np.log10(1), 300, np.log10(1), None, None])

    # WORKING ANGLE
    config_data = get_config_data(instru)    
    IWA = np.zeros((len(apodizers)))
    for ni, apodizer in enumerate(apodizers):
        IWA[ni] = config_data["apodizers"][apodizers[ni]].sep
    iwa = np.nanmin(IWA) # mas
    owa = config_data["spec"]["FOV"]/2 * 1000 # mas

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)
    
    # RAW TABLE
    planet_table_raw = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    planet_table_raw = planet_table_raw[(planet_table_raw["AngSep"]>iwa*u.mas)&(planet_table_raw["AngSep"]<owa*u.mas)]
    data_raw = np.zeros((len(planet_table_raw), ndim))
    data_raw[:, 0] = np.log10(np.array(planet_table_raw["PlanetMass"].value))
    data_raw[:, 1] = np.log10(np.array(planet_table_raw["PlanetRadius"].value))
    data_raw[:, 2] = np.array(planet_table_raw["PlanetTeq"].value)
    data_raw[:, 3] = np.log10(np.array(planet_table_raw["SMA"].value))
    data_raw[:, 4] = np.log10(np.array(planet_table_raw["Distance"].value))
    data_raw[:, 5] = np.log10(np.array(planet_table_raw["AngSep"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data_raw,
        bins = 20,
        labels = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$", r"$log(\frac{sep}{mas})$"],
        quantiles = [0.16, 0.5, 0.84], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = True,
        title_kwargs = {"fontsize": 14, "pad": 10},
        top_ticks = False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        smooth = smooth_corner,
        smooth1d = smooth_corner,
        label_kwargs={"fontsize": 16})
    
    # DETECTIONS TABLES
    yields = np.zeros((len(apodizers)))
    for ni, apodizer in enumerate(apodizers):
        strehl = strehls[ni]
        planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
        planet_table = planet_table[(planet_table["AngSep"]>IWA[ni]*u.mas)&(planet_table["AngSep"]<owa*u.mas)]
        SNR = np.sqrt(exposure_time/planet_table['DIT_'+band]) * planet_table['signal_'+band] / np.sqrt(  planet_table['sigma_fund_'+band]**2 + (exposure_time/planet_table['DIT_'+band])*planet_table['sigma_syst_'+band]**2 )
        planet_table = planet_table[SNR>5]
        yields[ni] = len(planet_table)
        data = np.zeros((len(planet_table), ndim))
        data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
        data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
        data[:, 2] = np.array(planet_table["PlanetTeq"].value)
        data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
        data[:, 4] = np.log10(np.array(planet_table["Distance"].value))
        data[:, 5] = np.log10(np.array(planet_table["AngSep"].value))
        corner.corner(
            data,
            fig = figure,
            bins = 20,
            quantiles = [0.5], # below -+1 sigma 
            levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
            show_titles = False,
            top_ticks = False,
            plot_density = True,
            plot_contours = True,
            fill_contours = True,
            color = c_apodizers[ni],
            smooth = smooth_corner,
            smooth1d = smooth_corner)
            
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
    for ni, apodizer in enumerate(apodizers):
        handles.append(mlines.Line2D([], [], color="w", label=""))
        handles.append(mlines.Line2D([], [], linestyle="-", marker="s", color=c_apodizers[ni], label=apodizer.replace('_', ' ')+f" ({round(yields[ni])} / {len(planet_table_raw)})"))
    plt.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="upper right", fontsize=14, title="Instruments:", title_fontsize=16)
    handles = [mlines.Line2D([], [], linestyle="-", marker="s", color=color_earth, label="Earth")]
    ax_legend = plt.gca().twinx()
    ax_legend.legend(handles=handles, frameon=True, bbox_to_anchor=(1, ndim), loc="lower right", fontsize=14)
    figure.suptitle(f"{apodizers[0].replace('_', ' ')} VS {apodizers[1].replace('_', ' ')} VS {apodizers[2].replace('_', ' ')} re-detections statistics with {len(planet_table_raw)} known planets between {round(iwa)} and {round(owa)} mas for {round(exposure_time/60)} hours per target \n (with {name_model} model in {spectrum_contributions} light)", fontsize=18, y=1.06)
    plt.gcf().set_dpi(300)
    plt.show()



#######################################################################################################################

# OTHERS



def archive_corner_plot():
    import corner
    smooth_corner = 0
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    mp_mask = get_mask(planet_table, "PlanetMass")
    planet_table = planet_table[~mp_mask]
    ndim = 5 # Mp, Rp, Tp, d, a
    data = np.zeros((len(planet_table), ndim))
    data[:, 0] = np.log10(np.array(planet_table["PlanetMass"].value))
    data[:, 1] = np.log10(np.array(planet_table["PlanetRadius"].value))
    data[:, 2] = np.array(planet_table["PlanetTeq"].value)
    data[:, 3] = np.log10(np.array(planet_table["SMA"].value))
    data[:, 4] = np.log10(np.array(planet_table["Distance"].value))
    figure = corner.corner( # https://corner.readthedocs.io/en/latest/api/
        data,
        bins = 20,
        labels = [r"$log(\frac{M_p}{M_{\oplus}})$", r"$log(\frac{R_p}{R_{\oplus}})$", r"$T \, [\mathrm{K}]$", r"$log(\frac{SMA}{AU})$", r"$log(\frac{d}{pc})$"],
        quantiles = [0.16, 0.5, 0.84], # below -+1 sigma 
        levels = [0.68, 0.95, 0.997], # 1, 2 and 3 sigma contour
        show_titles = True,
        title_kwargs = {"fontsize": 14, "pad": 10},
        top_ticks = False,
        plot_density = True,
        plot_contours = True,
        fill_contours = True,
        smooth = smooth_corner,
        smooth1d = smooth_corner,
        label_kwargs={"fontsize": 16})
    figure.suptitle(f"Archive statistics (with {len(planet_table)} planets)", fontsize=16, y=1.05)
    plt.gcf().set_dpi(300)
    plt.show()



def contrast_detection_plot(instru="ANDES", exposure_time=600, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", band="INSTRU"):
    config_data = get_config_data(instru)

    # WORKING ANGLE
    lambda_c = (config_data["lambda_range"]["lambda_min"]+config_data["lambda_range"]["lambda_max"])/2 * 1e-6
    diameter = config_data['telescope']['diameter']
    if band == "INSTRU":
        try:
            pxscale = min(config_data["pxscale"].values()) * 1000 # mas
        except:
            pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    else:
        try:
            pxscale = config_data["pxscale"][band] * 1000 # mas
        except:
            pxscale = config_data["spec"]["pxscale"] * 1000 # mas
    iwa = max(pxscale, config_data["apodizers"][apodizer].sep)
    if config_data["type"] == "IFU_fiber":
        if band == "INSTRU":
            owa = config_data["FOV_fiber"]/2 * max(config_data["pxscale"].values()) * 1000 # en mas
        else:
            owa = config_data["FOV_fiber"]/2 * config_data["pxscale"][band] * 1000 # en mas
    else:
        owa = config_data["spec"]["FOV"]/2 * 1000 # en mas

    # params spectro de l'instru
    lmin = config_data["lambda_range"]["lambda_min"] ; lmax = config_data["lambda_range"]["lambda_max"]
    R = 0. ; N = len(config_data["gratings"]) 
    for b in config_data["gratings"]:
        R += config_data["gratings"][b].R/N # mean resolution

    # MODELS NAME
    spectrum_contributions, name_model = get_spectrum_contribution_name_model(thermal_model, reflected_model)

    planet_table = load_planet_table("Archive_Pull_"+instru+"_"+apodizer+"_"+strehl+"_without_systematics_"+name_model+".ecsv")
    SNR = np.sqrt(exposure_time/planet_table['DIT_'+band]) * planet_table['signal_'+band] / np.sqrt(  planet_table['sigma_fund_'+band]**2 + (exposure_time/planet_table['DIT_'+band])*planet_table['sigma_syst_'+band]**2 )
    planet_table = planet_table[SNR>5]
    x = np.array(planet_table["AngSep"].value) # sep axis [mas]
    if config_data["sep_unit"]=="arcsec":
        x /= 1000
    mag_p = planet_table['PlanetINSTRUmag('+instru+')('+spectrum_contributions+')']
    mag_s = planet_table['StarINSTRUmag('+instru+')']
    y = 10**(-0.4*(mag_p-mag_s))
    z = np.array(planet_table["PlanetTeq"].value) # color axis [K]

    im_mask = planet_table["DiscoveryMethod"]=="Imaging"
    tr_mask = planet_table["DiscoveryMethod"]=="Transit"
    rv_mask = planet_table["DiscoveryMethod"]=="Radial Velocity"
    ot_mask = (planet_table["DiscoveryMethod"]!="Imaging") & (planet_table["DiscoveryMethod"]!="Radial Velocity") & (planet_table["DiscoveryMethod"]!="Transit")

    s0 = 50 ; ds = 200
    s = s0 + ds*np.array(planet_table["AngSep"].value)/2000
    s[s>s0+ds] = s0+ds
    cmap = plt.get_cmap("rainbow")
    fig = plt.figure(dpi=300, figsize=(10,5)) ; ax1 = plt.gca() ; ax1.set_yscale('log') ; ax1.set_xscale('log') ; ax1.grid(True, which='both', linestyle='--', linewidth=0.5) ; ax1.minorticks_on()
    ax1.set_title(f'Re-detections with {instru} ({spectrum_contributions} light with {name_model})'+'\n (from '+str(round(lmin, 1))+' to '+str(round(lmax, 1))+f' µm with R ~ {int(round(R, -2))}) with '+'$t_{exp}$=' + str(round(exposure_time/60)) + 'h ', fontsize = 14)
    ax1.set_xlabel("angular separation ["+config_data["sep_unit"]+"]", fontsize=12)
    ax1.set_ylabel("contrast on the instrumental bandwidth", fontsize=12)
    ax1.axvspan(iwa, owa, color='k', alpha=0.5, lw=0, label="Working angle", zorder=2)
    norm = LogNorm(vmin=100, vmax=3000) ; sm =  ScalarMappable(norm=norm, cmap=cmap) ; sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.1, shrink=0.8) ; cbar.set_label('$T_{eff}$ [K]', fontsize=12, labelpad=20, rotation=270)
    ax1.scatter(x[rv_mask], y[rv_mask], s=s[rv_mask], c=z[rv_mask], ec="k", marker="o", cmap=cmap, norm=norm, zorder=3)
    ax1.scatter(x[tr_mask], y[tr_mask], s=s[tr_mask], c=z[tr_mask], ec="k", marker="v", cmap=cmap, norm=norm, zorder=3)
    ax1.scatter(x[im_mask], y[im_mask], s=s[im_mask], c=z[im_mask], ec="k", marker="s", cmap=cmap, norm=norm, zorder=3)
    ax1.scatter(x[ot_mask], y[ot_mask], s=s[ot_mask], c=z[ot_mask], ec="k", marker="+", cmap=cmap, norm=norm, zorder=3)
    ax1.plot([], [], 'ko', label="Radial Velocity")
    ax1.plot([], [], 'kv', label="Transit")
    ax1.plot([], [], 'ks', label="Direct Imaging")
    ax1.plot([], [], 'k+', label="Other")
    ax2 = ax1.twinx() ; ax2.invert_yaxis() ; ax2.set_ylabel(r'$\Delta$mag', fontsize=12, labelpad=17, rotation=270) ; ax2.tick_params(axis='y')   
    ymin, ymax = ax1.get_ylim() ; ax2.set_ylim(-2.5*np.log10(ymin), -2.5*np.log10(ymax))        
    ax1.legend(loc="lower right") ; ax1.set_zorder(1) ; plt.xlim(iwa) ; plt.show()



def Vsini_plots():
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    
    nbins = 50
    pm_mask = np.logical_not(get_mask(planet_table, "PlanetMass"))
    plt.figure(dpi=300) ; plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on()
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
    plt.show()
    
    st_mask = np.logical_not(get_mask(planet_table, "StarTeff"))
    plt.figure(dpi=300)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; plt.minorticks_on()
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
    plt.show()




















