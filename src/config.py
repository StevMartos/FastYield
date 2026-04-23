# import astropy modules
from astropy import constants as const

# import numpy modules
import numpy as np

# import other modules
import collections
import os



# -------------------------------------------------------------------------
# Defining constants
# -------------------------------------------------------------------------

h          = const.h.value     # [J.s]
c          = const.c.value     # [m/s]
kB         = const.k_B.value   # [J/K]
rad2arcsec = 180/np.pi*3600    # [arcsec/rad] (to convert [rad] => [arcsec])
sr2arcsec2 = rad2arcsec**2     # [arcsec2/sr] (to convert [steradians] => [arcsec2])
G          = 6.67430e-11       # [m3/kg/s2]
G_cgs      = const.G.cgs.value # [cm3/g/s2]
m_u        = 1.66053906660e-27 # [kg] (atomic mass unit)
AU         = 149_597_870_700   # [m]
pc         = 3.085677581e16    # [m]

T_earth       = 288             # [K]
lg_earth      = 2.99            # [dex(cm/s2)]
R_earth       = 6.371e6         # [m]
M_earth       = 5.9722e24       # [kg]
drv_earth     = 29.8            # [km/s]
vrot_earth    = 0.465           # [km/s]
vesc_earth    = 11.186          # [km/s]
airmass_earth = 2.0             # [dimensionless]

vrot_jupiter = 12.6         # [km/s]
M_jupiter    = 1.8981246e27 # [kg]

T_sun    = 5778        # [K]
M_sun    = 1.98841e+30 # [kg]
R_sun    = 695_700_000 # [m]
lg_sun   = 4.44        # [dex(cm/s2)]
vrot_sun = 2.0         # [km/s]



# -------------------------------------------------------------------------
# Global resolving-power cap (lower threshold)
# -------------------------------------------------------------------------

R0_min = 1_000   # For photometry purposes only



# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

file_path = os.path.dirname(__file__)

# Sim_data path
sim_data_path = os.path.join(os.path.dirname(file_path), "sim_data/")

# Atmospheric models paths
spectra_path = os.path.join(os.path.dirname(sim_data_path), "Spectra/")
vega_path    = os.path.join(os.path.dirname(spectra_path),  "star_spectrum/VEGA_Fnu.fits")

# FastYield paths
archive_path   = os.path.join(os.path.dirname(sim_data_path), "Archive_table/")
simulated_path = os.path.join(os.path.dirname(sim_data_path), "Simulated_table/")

# Colormaps plot paths
colormaps_path = os.path.join(os.path.dirname(file_path), "plots/colormaps/")



# -------------------------------------------------------------------------
# Config data
# -------------------------------------------------------------------------

# Define namedtuples for instrument properties
GratingInfo  = collections.namedtuple("GratingInfo",  ["lmin", "lmax", "R"])
ApodizerInfo = collections.namedtuple("ApodizerInfo", ["transmission"])

#------------------------------------------------------------------------------#
#                                  ELT:                                        #
#------------------------------------------------------------------------------#

# config_data_HARMONI = {
#                        'name':           "HARMONI",                                    # Instrument name
#                        'telescope_name': "ELT",                                        # Telescope name
#                        'type':           "IFU",                                        # Instrument type
#                        'gratings':      {"J":       GratingInfo(1.046, 1.324,  7555),  # Spectral range [µm], resolving power
#                                          "H":       GratingInfo(1.435, 1.815,  7555),  # Spectral range [µm], resolving power
#                                          "H_high":  GratingInfo(1.538, 1.678, 17385),  # Spectral range [µm], resolving power
#                                          "HK":      GratingInfo(1.450, 2.450,  3355),  # Spectral range [µm], resolving power
#                                          "K":       GratingInfo(1.951, 2.469,  7555),  # Spectral range [µm], resolving power
#                                          "K1_high": GratingInfo(2.017, 2.200, 17385),  # Spectral range [µm], resolving power
#                                          "K2_high": GratingInfo(2.199, 2.400, 17385)}, # Spectral range [µm], resolving power
#                        'lambda_range':  {"lambda_min": 1.046, "lambda_max": 2.469},    # Minimum and maximum instrumental wavelength range [µm]
#                        'size_core':      3,                                            # Size in pixels of the FWHM  (here: side of the PSF core box)
#                        'apodizers':     {"NO_SP":   ApodizerInfo(0.84),                # Available apodizers (transmission, IWA [mas])
#                                          "SP1":     ApodizerInfo(0.45),                # Available apodizers (transmission, IWA [mas])
#                                          "SP2":     ApodizerInfo(0.35),                # Available apodizers (transmission, IWA [mas])
#                                          "SP_Prox": ApodizerInfo(0.68)},               # Available apodizers (transmission, IWA [mas])
#                        'strehls':       ["JQ1"],                                       # Available strehls
#                        'coronagraphs':  [None],                                        # Available coronagraphs
#                        'pxscale':        0.004,                                        # Pixel scale [arcsec/px]
#                        'FOV':            0.8,                                          # Field Of View [arcsec]
#                        'sep_unit':       "mas",                                        # Unit for angular separation
#                        'detector':      {"RON":          10.0,                         # Read-Out Noise [e-/px/read]
#                                          "RON_lim":      1.,                           # Read-Out Noise limit [e-/px/read]
#                                          "dark_current": 0.0053,                       # Dark current [e-/px/s]
#                                          "minDIT":       1.3/60,                       # Minimum Detector Integration Time [mn]
#                                          "maxDIT":       5,                            # Maximum Detector Integration Time [mn]
#                                          "saturation_e": 80_000},                      # Full well capacity [e-] 
#                      }



config_data_HARMONI = {
                       'name':           "HARMONI",                                 # Instrument name
                       'telescope_name': "ELT",                                     # Telescope name
                       'type':           "IFU",                                     # Instrument type
                       'gratings':      {"J":  GratingInfo(1.046, 1.324,  7555),    # Spectral range [µm], resolving power
                                         "H":  GratingInfo(1.435, 1.815,  7555),    # Spectral range [µm], resolving power
                                         "HK": GratingInfo(1.450, 2.450,  3355),    # Spectral range [µm], resolving power
                                         "K":  GratingInfo(1.951, 2.469,  7555)},   # Spectral range [µm], resolving power
                       'lambda_range':  {"lambda_min": 1.046, "lambda_max": 2.469}, # Minimum and maximum instrumental wavelength range [µm]
                       'size_core':      3,                                         # Size in pixels of the FWHM  (here: side of the PSF core box)
                       'apodizers':     {"NO_SP":   ApodizerInfo(0.84),             # Available apodizers (transmission, IWA [mas])
                                         "SP1":     ApodizerInfo(0.45),             # Available apodizers (transmission, IWA [mas])
                                         "SP2":     ApodizerInfo(0.35),             # Available apodizers (transmission, IWA [mas])
                                         "SP_Prox": ApodizerInfo(0.68)},            # Available apodizers (transmission, IWA [mas])
                       'FPMs':          [30, 50, 70, 100],                          # FPM IWA available values in [mas]
                       #'FPMs':          [10, 20, 30, 50, 70, 100],                  # FPM IWA available values in [mas]
                       #'FPMs':          [30],                                       # FPM IWA available values in [mas]
                       'strehls':       ["JQ1"],                                    # Available strehls
                       'coronagraphs':  [None],                                     # Available coronagraphs
                       'pxscale':        0.004,                                     # Pixel scale [arcsec/px]
                       'FOV':            0.8,                                       # Field Of View [arcsec]
                       'sep_unit':       "mas",                                     # Unit for angular separation
                       'detector':      {"RON":          10.0,                      # Read-Out Noise [e-/px/read]
                                         "RON_lim":      1.,                        # Read-Out Noise limit [e-/px/read]
                                         "dark_current": 0.0053,                    # Dark current [e-/px/s]
                                         "minDIT":       1.3/60,                    # Minimum Detector Integration Time [mn]
                                         "maxDIT":       5,                         # Maximum Detector Integration Time [mn]
                                         "saturation_e": 80_000},                   # Full well capacity [e-] 
                     }



config_data_ANDES = {
                     'name':                       "ANDES",                                          # Instrument name
                     'telescope_name':             "ELT",                                            # Telescope name
                     'type':                       "IFU_fiber",                                      # Instrument type
                     'gratings':                  {"YJH_5mas_100":  GratingInfo(0.95, 1.8, 100000),  # Spectral range [µm], resolving power
                                                   "YJH_10mas_100": GratingInfo(0.95, 1.8, 100000),  # Spectral range [µm], resolving power
                                                   "YJH_16mas_100": GratingInfo(0.95, 1.8, 100000)}, # Spectral range [µm], resolving power
                     'lambda_range':              {"lambda_min": 0.95, "lambda_max": 1.8},           # Minimum and maximum instrumental wavelength range [µm]
                     'size_core':                  1,                                                # Size in pixels of the FWHM  (here: 1 fiber on the planet)
                     'injection':                 {"YJH_5mas_100":  0.842,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
                                                   "YJH_10mas_100": 0.671,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
                                                   "YJH_16mas_100": 0.620},                          # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
                     'pixel_detector_projection':  3,                                                # Effective number of pixels per spectral channel (here: first guess)
                     'apodizers':                 {"NO_SP": ApodizerInfo(1)},                        # Available apodizers
                     'strehls':                   ["MED"],                                           # Available strehls
                     'coronagraphs':              [None, "LYOT"],                                    # Available coronagraphs
                     'pxscale':                   {"YJH_5mas_100":  0.005,                           # Pixel scale [arcsec/px]
                                                   "YJH_10mas_100": 0.010,                           # Pixel scale [arcsec/px]
                                                   "YJH_16mas_100": 0.016},                          # Pixel scale [arcsec/px]
                     'FOV':                        0.144,                                            # Field Of View [arcsec], 16 mas * 9 fibers
                     'sep_unit':                   "mas",                                            # Unit for angular separation
                     'detector':                  {"RON":          10.0,                             # Read-Out Noise [e-/px/read]
                                                   "RON_lim":      1.,                               # Read-Out Noise limit [e-/px/read]
                                                   "dark_current": 0.0053,                           # Dark current [e-/px/s]
                                                   "minDIT":       1.3/60,                           # Minimum Detector Integration Time [mn]
                                                   "maxDIT":       5,                                # Maximum Detector Integration Time [mn]
                                                   "saturation_e": 80_000},                          # Full well capacity [e-] 
                   }


#------------------------------------------------------------------------------#
#                                  VLT:                                        #
#------------------------------------------------------------------------------#

config_data_ERIS = {
                    'name':           "ERIS",                                       # Instrument name
                    'telescope_name': "VLT",                                        # Telescope name
                    'type':           "IFU",                                        # Instrument type
                    'gratings':      {"J_low":    GratingInfo(1.09, 1.42,  5000.),  # Spectral range [µm], resolving power
                                      "H_low":    GratingInfo(1.45, 1.87,  5200.),  # Spectral range [µm], resolving power
                                      "K_low":    GratingInfo(1.93, 2.48,  5600.),  # Spectral range [µm], resolving power
                                      "J_short":  GratingInfo(1.10, 1.27, 10000.),  # Spectral range [µm], resolving power
                                      "J_middle": GratingInfo(1.18, 1.35, 10000.),  # Spectral range [µm], resolving power
                                      "J_long":   GratingInfo(1.26, 1.43, 10000.),  # Spectral range [µm], resolving power
                                      "H_short":  GratingInfo(1.46, 1.67, 10400.),  # Spectral range [µm], resolving power
                                      "H_middle": GratingInfo(1.56, 1.77, 10400.),  # Spectral range [µm], resolving power
                                      "H_long":   GratingInfo(1.66, 1.87, 10400.),  # Spectral range [µm], resolving power
                                      "K_short":  GratingInfo(1.93, 2.22, 11200.),  # Spectral range [µm], resolving power
                                      "K_middle": GratingInfo(2.06, 2.34, 11200.),  # Spectral range [µm], resolving power
                                      "K_long":   GratingInfo(2.19, 2.47, 11200.)}, # Spectral range [µm], resolving power
                    'lambda_range':  {"lambda_min": 1.08, "lambda_max": 2.48},      # Minimum and maximum instrumental wavelength range [µm]
                    'size_core':      3,                                            # Size in pixels of the FWHM  (here: side of the PSF core box)
                    'apodizers':     {"NO_SP": ApodizerInfo(1)},                    # Available apodizers
                    'strehls':       ["JQ0"],                                       # Available strehls
                    'coronagraphs':  [None],                                        # Available coronagraphs
                    'pxscale':        0.025,                                        # Pixel scale [arcsec/px]
                    'FOV':            0.8,                                          # Field Of View [arcsec]
                    'sep_unit':       "mas",                                        # Unit for angular separation
                    'detector':      {"RON":          12.0,                         # Read-Out Noise [e-/px/read]
                                      "RON_lim":      7.,                           # Read-Out Noise limit [e-/px/read]
                                      "dark_current": 0.19,                         # Dark current [e-/px/s]
                                      "minDIT":       1.6/60,                       # Minimum Detector Integration Time [mn]
                                      "maxDIT":       2,                            # Maximum Detector Integration Time [mn]
                                      "saturation_e": 80_000},                      # Full well capacity [e-] 
                   }




config_data_HiRISE = {
                      'name':                       "HiRISE",                                # Instrument name
                      'telescope_name':             "VLT",                                   # Telescope name
                      'type':                       "fiber_injection_HRS",                   # Instrument type
                      'gratings':                  {"H": GratingInfo(1.43, 1.78, 140_000)},  # Spectral range [µm], resolving power
                      'lambda_range':              {"lambda_min": 1.43, "lambda_max": 1.78}, # Minimum and maximum instrumental wavelength range [µm]
                      'size_core':                 1,                                        # Size in pixels of the FWHM                      (here: 1 fiber on the planet)
                      'pixel_detector_projection': 2 * 2.04,                                 # Effective number of pixels per spectral channel (here: 2 × Npx_x × Npx_y)
                      'apodizers':                 {"NO_SP": ApodizerInfo(1)},               # Available apodizers
                      'strehls':                   ["MED"],                                  # Available strehls
                      'coronagraphs':              [None],                                   # Available coronagraphs
                      'pxscale':                    0.01225,                                 # Pixel scale [arcsec/px]
                      'FOV':                        4,                                       # Field Of View [arcsec]
                      'sep_unit':                   "arcsec",                                # Unit for angular separation
                      'detector':                  {"RON":          12,                      # Read-Out Noise [e-/px/read]
                                                    "RON_lim":      1.,                      # Read-Out Noise limit [e-/px/read]
                                                    "dark_current": 0.03,                    # Dark current [e-/px/s]
                                                    "pxscale":      0.01225,                 # Pixel scale [arcsec/px]
                                                    "minDIT":       1.4725/60,               # Minimum Detector Integration Time [mn]
                                                    "maxDIT":       20,                      # Maximum Detector Integration Time [mn]
                                                    "saturation_e": 80_000},                 # Full well capacity [e-] 
                     }



#------------------------------------------------------------------------------#
#                                  JWST:                                       #
#------------------------------------------------------------------------------#

config_data_MIRIMRS = {
                       'name':           "MIRIMRS",                                               # Instrument name
                       'telescope_name': "JWST",                                                  # Telescope name
                       'type':           "IFU",                                                   # Instrument type
                       'gratings':      {"1SHORT":  GratingInfo(4.90,  5.741,  (3320.+3710.)/2),  # Spectral range [µm], resolving power
                                         "1MEDIUM": GratingInfo(5.66,  6.63,   (3750.+3190.)/2),  # Spectral range [µm], resolving power
                                         "1LONG":   GratingInfo(6.53,  7.65,   (3610.+3100.)/2),  # Spectral range [µm], resolving power
                                         "2SHORT":  GratingInfo(7.51,  8.77,   (3110.+2990.)/2),  # Spectral range [µm], resolving power
                                         "2MEDIUM": GratingInfo(8.67,  10.13,  (2750.+3170.)/2),  # Spectral range [µm], resolving power
                                         "2LONG":   GratingInfo(10.02, 11.70,  (2860.+3300.)/2),  # Spectral range [µm], resolving power
                                         "3SHORT":  GratingInfo(11.55, 13.47,  (2530.+2880.)/2),  # Spectral range [µm], resolving power
                                         "3MEDIUM": GratingInfo(13.34, 15.57,  (1790.+2640.)/2),  # Spectral range [µm], resolving power
                                         "3LONG":   GratingInfo(15.42, 17.98,  (1980.+2790.)/2),  # Spectral range [µm], resolving power
                                         "4SHORT":  GratingInfo(17.70, 20.915, (1460.+1930.)/2),  # Spectral range [µm], resolving power
                                         "4MEDIUM": GratingInfo(20.69, 24.385, (1680.+1770.)/2),  # Spectral range [µm], resolving power
                                         "4LONG":   GratingInfo(24.19, 27.90,  (1630.+1330.)/2)}, # Spectral range [µm], resolving power
                      'lambda_range':   {"lambda_min": 4.90, "lambda_max": 27.90},                # Minimum and maximum instrumental wavelength range [µm]
                      'size_core':       3,                                                       # Size in pixels of the FWHM  (here: side of the PSF core box)
                      'R_cov':           2.4,                                                     # Spatial covariance factor   (here: for size_core = 3)
                      'apodizers':      {"NO_SP": ApodizerInfo(1)},                               # Available apodizers
                      'strehls':        ["NO_JQ"],                                                # Available strehls
                      'coronagraphs':   [None],                                                   # Available coronagraphs
                      'pxscale':        {"1SHORT":0.13, "1MEDIUM":0.13, "1LONG":0.13,             # Pixel scale [arcsec/px] (channel 1 with dithering)
                                         "2SHORT":0.17, "2MEDIUM":0.17, "2LONG":0.17,             # Pixel scale [arcsec/px] (channel 2 with dithering)
                                         "3SHORT":0.20, "3MEDIUM":0.20, "3LONG":0.20,             # Pixel scale [arcsec/px] (channel 3 with dithering)
                                         "4SHORT":0.35, "4MEDIUM":0.35, "4LONG":0.35},            # Pixel scale [arcsec/px] (channel 4 with dithering)
                      'pxscale0':       {"1SHORT":0.196, "1MEDIUM":0.196, "1LONG":0.196,          # Pixel scale [arcsec/px] (channel 1 without dithering)
                                         "2SHORT":0.196, "2MEDIUM":0.196, "2LONG":0.196,          # Pixel scale [arcsec/px] (channel 2 without dithering)
                                         "3SHORT":0.245, "3MEDIUM":0.245, "3LONG":0.245,          # Pixel scale [arcsec/px] (channel 3 without dithering)
                                         "4SHORT":0.273, "4MEDIUM":0.273, "4LONG":0.273},         # Pixel scale [arcsec/px] (channel 4 without dithering)
                      'FOV':             5.2,                                                     # Field Of View [arcsec]
                      'sep_unit':        "arcsec",                                                # Unit for angular separation
                      'detector':       {"RON":          19,                                      # Read-Out Noise [e-/px/read]
                                         "RON_lim":      1.,                                      # Read-Out Noise limit [e-/px/read]
                                         "dark_current": 0.2,                                     # Dark current [e-/px/s]
                                         "minDIT":       2.775/60,                                # Minimum Detector Integration Time [mn]
                                         "maxDIT":       5,                                       # Maximum Detector Integration Time [mn]
                                         "saturation_e": 200_000},                                # Full well capacity [e-] 
                     }



config_data_NIRCam = {
                      'name':           "NIRCam",                                          # Instrument name
                      'telescope_name': "JWST",                                            # Telescope name
                      'type':           "imager",                                          # Instrument type
                      'gratings':      {"F250M": GratingInfo(2.35, 2.75, R0_min),          # Spectral range [µm], resolving power (R=R0_min abritrary, used for interpolation)
                                        "F300M": GratingInfo(2.65, 3.35, R0_min),          # Spectral range [µm], resolving power
                                        "F410M": GratingInfo(3.70, 4.60, R0_min),          # Spectral range [µm], resolving power
                                        #"F480M": GratingInfo(4.58, 5.09, R0_min),          # Spectral range [µm], resolving power
                                        #"F150W": GratingInfo(1.33, 1.67, R0_min),          # Spectral range [µm], resolving power
                                        "F356W": GratingInfo(3.00, 4.25, R0_min),          # Spectral range [µm], resolving power
                                        "F444W": GratingInfo(3.65, 5.19, R0_min)},         # Spectral range [µm], resolving power
                      'lambda_range':  {"lambda_min": 1.33, "lambda_max": 5.19},           # Minimum and maximum instrumental wavelength range [µm]
                      'size_core':      3,                                                 # Size in pixels of the FWHM  (here: side of the PSF core box)
                      'apodizers':     {"NO_SP": ApodizerInfo(1)},                         # Available apodizers
                      'strehls':       ["NO_JQ"],                                          # Available strehls
                      'coronagraphs':  ["MASK335R"],                                       # Available coronagraphs
                      'pxscale':        0.063,                                             # Pixel scale [arcsec/px]
                      'FOV':            10,                                                # Field Of View [arcsec]
                      'sep_unit':       "arcsec",                                          # Unit for angular separation
                      'detector':      {"RON":          9.4,                               # Read-Out Noise [e-/px/read]
                                        "RON_lim":      1.,                                # Read-Out Noise limit [e-/px/read]
                                        "dark_current": 34.2/1000,                         # Dark current [e-/px/s]
                                        "minDIT":       10.737/60,                         # Minimum Detector Integration Time [mn]
                                        "maxDIT":       5,                                 # Maximum Detector Integration Time [mn]
                                        "saturation_e": 83_300},                           # Full well capacity [e-] 
                     }



config_data_NIRSpec = {
                       'name':           "NIRSpec",                                         # Instrument name
                       'telescope_name': "JWST",                                            # Telescope name
                       'type':           "IFU",                                             # Instrument type
                       'gratings':      {"G140H_F100LP": GratingInfo(0.98,  1.89, 2700),    # Spectral range [µm], resolving power
                                         "G235H_F170LP": GratingInfo(1.66,  3.17, 2700),    # Spectral range [µm], resolving power
                                         "G395H_F290LP": GratingInfo(2.87,  5.27, 2700)},   # Spectral range [µm], resolving power
                       'lambda_range':  {"lambda_min": 0.90, "lambda_max": 5.27},           # Minimum and maximum instrumental wavelength range [µm]
                       'size_core':      3,                                                 # Size in pixels of the FWHM  (here: side of the PSF core box)
                       'R_cov':          1.7,                                               # Spatial covariance factor   (here: for size_core = 3)
                       'apodizers':     {"NO_SP": ApodizerInfo(1)},                         # Available apodizers
                       'strehls':       ["NO_JQ"],                                          # Available strehls
                       'coronagraphs':  [None],                                             # Available coronagraphs
                       'pxscale':        0.1045,                                            # Pixel scale [arcsec/px]
                       'FOV':            5,                                                 # Field Of View [arcsec]
                       'sep_unit':       "arcsec",                                          # Unit for angular separation
                       'detector':      {"RON":          10,                                # Read-Out Noise [e-/px/read]
                                         "RON_lim":      1.,                                # Read-Out Noise limit [e-/px/read]
                                         "dark_current": 0.008,                             # Dark current [e-/px/s]
                                         "minDIT":       10.737/60,                         # Minimum Detector Integration Time [mn]
                                         "maxDIT":       5,                                 # Maximum Detector Integration Time [mn]
                                         "saturation_e": 57_500},                           # Full well capacity [e-] 
                      }



#------------------------------------------------------------------------------#
#                               Test bench:                                    #
#------------------------------------------------------------------------------#

# VIPA spectrometer at 152cm telescope of the OHP (with K-band)
config_data_VIPAPYRUS = {
                         'name':                      "VIPAPYRUS",                                 # Instrument name
                         'telescope_name':            "T152",                                      # Telescope name
                         'type':                      "fiber_injection_HRS",                       # Instrument type
                         'gratings':                 {"H": GratingInfo(1.5525, 1.760, 60_000),     # Spectral range [µm], resolving power
                                                      "K": GratingInfo(2.1,    2.32,  60_000)},    # Spectral range [µm], resolving power
                         'lambda_range':             {"lambda_min": 1.5525, "lambda_max": 2.320},  # Minimum and maximum instrumental wavelength range [µm]
                         'size_core':                 1,                                           # Size in pixels of the FWHM                      (here: 1 fiber on the planet)
                         'R_corr':                    0.65,                                        # Corrective factor on photon noise               (here: R_corr = scale_poisson**2)
                         'pixel_detector_projection': 8.655,                                       # Effective number of pixels per spectral channel (here: scale_flat**2 / scale_poisson**2)
                         'apodizers':                {"NO_SP": ApodizerInfo(1)},                   # Available apodizers
                         'strehls':                  ["MED"],                                      # Available strehls
                         'coronagraphs':             [None],                                       # Available coronagraphs
                         'pxscale':                   0.15, #0.24,                                 # Pixel scale [arcsec/px]                         (here: pxscale ~ 1.03*lambda/D, i.e. size of the fiber/FWHM)
                         'FOV':                       4,                                           # Field Of View [arcsec]
                         'sep_unit':                  "arcsec",                                    # Unit for angular separation
                         'detector':                 {"RON":          11.4,                        # Read-Out Noise [e-/px/read]
                                                      "RON_lim":      4.1,                         # Read-Out Noise limit [e-/px/read]
                                                      "dark_current": 0.0053,                      # Dark current [e-/px/s]
                                                      "minDIT":       1.4725/60,                   # Minimum Detector Integration Time [mn]
                                                      "maxDIT":       5,                           # Maximum Detector Integration Time [mn]
                                                      "saturation_e": 40_000},                     # Full well capacity [e-] 
                        }


# # HARMONI (HC Bench) demonstrator
# config_data_HARMONI = {
#                        'name':        "HARMONI",                                   # Instrument name
#                        'type':         "IFU",                                      # Instrument type
#                        'base':         "ground",                                   # "ground" or "space" based
#                        'bench':        True,                                       # HC bench (demonstrator)
#                        'latitude':     -24.627,                                    # Geographic latitude [°]
#                        'longitude':    -70.404,                                    # Geographic longitude [°]
#                        'altitude':     2635,                                       # Altitude [m]
#                        'sep_unit':     "mas",                                      # Unit for angular separation
#                        'telescope':    {"diameter": 38.452, "area": 980.},         # All-glass diameter [m], Effective collecting area [m²] (accounting for central hole, secondary mirror, and spider obscuration), eps = geometric central obstruction (diameter of the secondary / diameter of the primary)
#                        'gratings':     {"H": GratingInfo(1.5601, 1.69, 7555)},     # Spectral range [µm], resolving power
#                        'lambda_range': {"lambda_min": 1.5601, "lambda_max": 1.69}, # Minimum and maximum instrumental wavelength range [µm]
#                        'size_core':    3,                                          # Size in pixels of the FWHM  (here: side of the PSF core box)
#                        'apodizers':    {"NO_SP":   ApodizerInfo(0.84),             # Available apodizers (transmission, IWA [mas])
#                                         "SP1":     ApodizerInfo(0.45),             # Available apodizers (transmission, IWA [mas])
#                                         "SP2":     ApodizerInfo(0.35),             # Available apodizers (transmission, IWA [mas])
#                                         "SP_Prox": ApodizerInfo(0.68)},            # Available apodizers (transmission, IWA [mas])
#                        'strehls':      ["JQ1"],                                    # Available strehls
#                        'coronagraphs': [None],                                     # Available coronagraphs
#                        'spec':         {"RON":          10.0,                      # Read-Out Noise [e-/px/read]
#                                         "RON_lim":      1.,                        # Read-Out Noise limit [e-/px/read]
#                                         "dark_current": 0.0053,                    # Dark current [e-/px/s]
#                                         "FOV":          0.8,                       # Field Of View [arcsec]
#                                         "pxscale":      0.004,                     # Pixel scale [arcsec/px]
#                                         "minDIT":       2.62/60,                   # Minimum Detector Integration Time [mn]
#                                         "maxDIT":       1,                         # Maximum Detector Integration Time [mn]
#                                         "saturation_e": 40_000},                   # Full well capacity [e-] 
#                      }




# -------------------------------------------------------------------------
# Instrument registry
# -------------------------------------------------------------------------

config_data_list = [
                    config_data_HARMONI,
                    config_data_ANDES,
                    config_data_ERIS,
                    config_data_MIRIMRS,
                    config_data_NIRCam,
                    config_data_NIRSpec,
                    config_data_HiRISE,
                    config_data_VIPAPYRUS,
                   ]


for config_data in config_data_list:
    
    if config_data["telescope_name"] == "ELT":
        config_data["telescope"] = {"diameter": 38.54, "area": 978, "eps": 0.108} # All-glass diameter [m], Effective collecting area [m²] (accounting for central hole, secondary mirror, and spider obscuration), eps = geometric central obstruction (diameter of the secondary / diameter of the primary)
        config_data["base"]      = "ground"                                        # "ground" or "space" based
        config_data["latitude"]  = -24.627                                         # Geographic latitude [°]
        config_data["longitude"] = -70.404                                         # Geographic longitude [°]
        config_data["altitude"]  = 2635                                            # Altitude [m]
    
    elif config_data["telescope_name"] == "VLT":
        config_data["telescope"] = {"diameter": 8, "area": 49.3, "eps": 0.137} # All-glass diameter [m], Effective collecting area [m²] (accounting for central hole, secondary mirror, and spider obscuration), eps = geometric central obstruction (diameter of the secondary / diameter of the primary)
        config_data["base"]      = "ground"                                    # "ground" or "space" based
        config_data["latitude"]  = -24.627                                     # Geographic latitude [°]
        config_data["longitude"] = -70.404                                     # Geographic longitude [°]
        config_data["altitude"]  = 2635                                        # Altitude [m]

    elif config_data["telescope_name"] == "JWST":
        config_data["telescope"] = {"diameter":  6.6052, "area": 25.032, "eps": 0.108} # All-glass diameter [m], Effective collecting area [m²] (accounting for central hole, secondary mirror, and spider obscuration), eps = geometric central obstruction (diameter of the secondary / diameter of the primary)
        config_data["base"]      = "space"                                             # "ground" or "space" based
    
    elif config_data["telescope_name"] == "T152":
        config_data["telescope"] = {"diameter": 1.52, "area": 1.81, "eps": 0.12} # All-glass diameter [m], Effective collecting area [m²] (accounting for central hole, secondary mirror, and spider obscuration), eps = geometric central obstruction (diameter of the secondary / diameter of the primary)
        config_data["base"]      = "ground"                                      # "ground" or "space" based
        config_data["latitude"]  = 43.92                                         # Geographic latitude [°]
        config_data["longitude"] = 5.712                                         # Geographic longitude [°]
        config_data["altitude"]  = 650                                           # Altitude [m]




instrus_with_systematics = ["MIRIMRS", "NIRSpec"]

colors_instru = {"HARMONI": "royalblue",
                 "ANDES":   "gray",
                 "ERIS":    "crimson",
                 "MIRIMRS": "seagreen",
                 "NIRCam":  "darkviolet",
                 "NIRSpec": "rosybrown",
                 "HiRISE":  "darkorange"}



# -----------------------------------------------
# Aggregate wavelength coverage and band registry
# -----------------------------------------------

LMIN    = np.inf
LMAX    = -np.inf
bands   = []     # band/grating names
instrus = []     # instrument names (order of config_data_list)

# dictionaries for cut-on/off (um)
lmin_bands = {}
lmax_bands = {}

for cfg in config_data_list:
    instru = cfg["name"]
    instrus.append(instru)

    # store instrument global range as entries too
    lmin_bands[instru] = cfg["lambda_range"]["lambda_min"]
    lmax_bands[instru] = cfg["lambda_range"]["lambda_max"]

    # update global coverage
    LMIN = min(LMIN, cfg["lambda_range"]["lambda_min"])
    LMAX = max(LMAX, cfg["lambda_range"]["lambda_max"])

    # register gratings / bands
    for band_name, g in cfg["gratings"].items():
        if band_name not in lmin_bands:  # first time we see it
            bands.append(band_name)
            lmin_bands[band_name] = float(g.lmin)
            lmax_bands[band_name] = float(g.lmax)

# --- Add extra (non-instrument) filters/bands here directly into the dicts ---
extra_bands = {
    # --- MIRI Imaging filters ---
    "F560W":  (4.86,  6.43),
    "F770W":  (6.62,  8.74),
    "F1000W": (8.57,  11.30),
    "F1130W": (10.95, 12.66),
    "F1280W": (11.52, 14.05),
    "F1500W": (13.29, 16.84),
    "F1800W": (16.24, 20.05),
    "F2100W": (18.55, 24.50),
    "F2550W": (22.47, 29.89),

    # --- Coronagraphic filters ---
    "F1065C": (10.0, 11.0),
    "F1140C": (11.1, 11.9),
    "F1550C": (14.9, 15.8),
    "F2300C": (22.5, 23.9),

    # --- WISE ---
    "W1": (2.8, 3.8),
    "W2": (4.1, 5.1),

    # --- MKO ---
    "L": (3.42, 4.12),
    "M": (4.47, 4.79),

    # --- PaBeta ---
    "PaB": (1.274, 1.290),

    # --- extra NIRCam-like bands (if you want them exposed even if not in gratings) ---
    "F150W": (1.3313, 1.6689),
    "F480M": (4.5820, 5.0919),
}

for b, (l0, l1) in extra_bands.items():
    # keep the first definition if already present
    if b not in lmin_bands:
        lmin_bands[b] = float(l0)
        lmax_bands[b] = float(l1)
        bands.append(b)

# convenience aliases if you really need them
lmin_VIPA = lmin_bands["VIPAPYRUS"]
lmax_VIPA = lmax_bands["VIPAPYRUS"]



# -------------------------------------------------------------------------
# FastYield config
# -------------------------------------------------------------------------

# Models list
thermal_models   = ["None", "BT-Settl", "Exo-REM", "PICASO"]
reflected_models = ["None", "tellurics", "flat", "PICASO"]

# Ignore reflected contribution if lmin > 6 µm
ignore_reflected_thresh_um = 6 # [µm]

# Detection threshold for yields
SNR_thresh = 5 

# Planet types list [M_earth, R_earth, K]
planet_types = {
    # ---- Cold (Teff < 500 K)
    "Cold Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teff_min": 0, "teff_max": 500},
    "Cold Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teff_min": 0, "teff_max": 500},
    "Cold Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teff_min": 0, "teff_max": 500},
    "Cold Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teff_min": 0, "teff_max": 500},
    "Cold Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teff_min": 0, "teff_max": 500},
    "Cold Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": 500},
    "Cold Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": 500},
    "Cold Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": 500},
    # ---- Warm (500 ≤ Teff < 1000 K)
    "Warm Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teff_min": 500, "teff_max": 1000},
    "Warm Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teff_min": 500, "teff_max": 1000},
    "Warm Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teff_min": 500, "teff_max": 1000},
    "Warm Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teff_min": 500, "teff_max": 1000},
    "Warm Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teff_min": 500, "teff_max": 1000},
    "Warm Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 500, "teff_max": 1000},
    "Warm Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 500, "teff_max": 1000},
    "Warm Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teff_min": 500, "teff_max": 1000},
    # ---- Hot (Teff ≥ 1000 K)
    "Hot Sub-Earth":       {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teff_min": 1000, "teff_max": np.inf},
    "Hot Earth-like":      {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teff_min": 1000, "teff_max": np.inf},
    "Hot Super-Earth":     {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teff_min": 1000, "teff_max": np.inf},
    "Hot Sub-Neptune":     {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teff_min": 1000, "teff_max": np.inf},
    "Hot Neptune-like":    {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teff_min": 1000, "teff_max": np.inf},
    "Hot Saturn-like":     {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 1000, "teff_max": np.inf},
    "Hot Jupiter-like":    {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 1000, "teff_max": np.inf},
    "Hot Super-Jupiter":   {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teff_min": 1000, "teff_max": np.inf},
}

# Planet types list [M_earth, R_earth, K]
planet_types_semireduced = {
    # ---- Cold (Teff < 500 K)
    "Cold Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teff_min": 0, "teff_max": 500},
    "Cold Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teff_min": 0, "teff_max": 500},
    "Cold Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teff_min": 0, "teff_max": 500},
    "Cold Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teff_min": 0, "teff_max": 500},
    "Cold Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teff_min": 0, "teff_max": 500},
    "Cold Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": 500},
    "Cold Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": 500},
    "Cold Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": 500},
    # ---- Hot (Teff ≥ 1000 K)
    "Hot Sub-Earth":       {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teff_min": 500, "teff_max": np.inf},
    "Hot Earth-like":      {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teff_min": 500, "teff_max": np.inf},
    "Hot Super-Earth":     {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teff_min": 500, "teff_max": np.inf},
    "Hot Sub-Neptune":     {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teff_min": 500, "teff_max": np.inf},
    "Hot Neptune-like":    {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teff_min": 500, "teff_max": np.inf},
    "Hot Saturn-like":     {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 500, "teff_max": np.inf},
    "Hot Jupiter-like":    {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 500, "teff_max": np.inf},
    "Hot Super-Jupiter":   {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teff_min": 500, "teff_max": np.inf},
}

# Planet types list  [M_earth, R_earth]
planet_types_reduced = {
    "Sub-Earth":      {"mass_min": 0,   "mass_max": 0.6,    "radius_min": 0,   "radius_max": 1.0,    "teff_min": 0, "teff_max": np.inf},
    "Earth-like":     {"mass_min": 0.6, "mass_max": 2.0,    "radius_min": 0.8, "radius_max": 1.6,    "teff_min": 0, "teff_max": np.inf},
    "Super-Earth":    {"mass_min": 2.0, "mass_max": 10,     "radius_min": 1.0, "radius_max": 2.0,    "teff_min": 0, "teff_max": np.inf},
    "Sub-Neptune":    {"mass_min": 2.0, "mass_max": 40,     "radius_min": 2.0, "radius_max": 4.0,    "teff_min": 0, "teff_max": np.inf},
    "Neptune-like":   {"mass_min": 10,  "mass_max": 80,     "radius_min": 4.0, "radius_max": 8.0,    "teff_min": 0, "teff_max": np.inf},
    "Saturn-like":    {"mass_min": 40,  "mass_max": 200,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": np.inf},
    "Jupiter-like":   {"mass_min": 200, "mass_max": 600,    "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": np.inf},
    "Super-Jupiter":  {"mass_min": 600, "mass_max": np.inf, "radius_min": 8.0, "radius_max": np.inf, "teff_min": 0, "teff_max": np.inf},
}


# -------------------------------------------------------------------------
# Global resolving-power cap (upper threshold)
# -------------------------------------------------------------------------

R_max = -np.inf # Max resolution considered over all the instruments
for config_data in config_data_list:
    for band in config_data["gratings"]:
        R_band = config_data["gratings"][band].R
        if R_band > R_max:
            R_max = R_band
R0_max = 2*R_max # Max resolution limit considered (No need for more)
