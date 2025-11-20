import collections
import numpy as np

# Global resolving-power cap
R0_max = 300_000
R0_min = 1_000

# Define namedtuples for instrument properties
GratingInfo  = collections.namedtuple("GratingInfo",  ["lmin", "lmax", "R"])
ApodizerInfo = collections.namedtuple("ApodizerInfo", ["transmission", "sep"])

def get_config_data(instrument_name):
    """
    Retrieve the specifications of a given instrument.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument.

    Returns
    -------
    dict
        Configuration parameters of the instrument.

    Raises
    ------
    NameError
        If the instrument name is not defined in config_data_list.
    """
    for cfg in config_data_list:
        if cfg["name"] == instrument_name:
            return cfg
    raise NameError(f"Undefined instrument name: {instrument_name}")



def get_band_lims(band):
    return globals()[f"lmin_{band}"], globals()[f"lmax_{band}"]



#------------------------------------------------------------------------------#
#                                  ELT:                                        #
#------------------------------------------------------------------------------#

config_data_HARMONI = {
                       'name':        "HARMONI",                                      # Instrument name
                       'type':         "IFU",                                         # Instrument type
                       'base':         "ground",                                      # "ground" or "space" based
                       'latitude':     -24.627,                                       # Geographic latitude [°]
                       'longitude':    -70.404,                                       # Geographic longitude [°]
                       'altitude':     2635,                                          # Altitude [m]
                       'sep_unit':     "mas",                                         # Unit for angular separation
                       'telescope':    {"diameter": 38.452, "area": 980.},            # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                       'gratings':     {"J":       GratingInfo(1.046, 1.324,  7555),  # Spectral range [µm], resolving power
                                        "H":       GratingInfo(1.435, 1.815,  7555),  # Spectral range [µm], resolving power
                                        "H_high":  GratingInfo(1.538, 1.678, 17385),  # Spectral range [µm], resolving power
                                        "HK":      GratingInfo(1.450, 2.450,  3355),  # Spectral range [µm], resolving power
                                        "K":       GratingInfo(1.951, 2.469,  7555),  # Spectral range [µm], resolving power
                                        "K1_high": GratingInfo(2.017, 2.200, 17385),  # Spectral range [µm], resolving power
                                        "K2_high": GratingInfo(2.199, 2.400, 17385)}, # Spectral range [µm], resolving power
                       'lambda_range': {"lambda_min": 1.046, "lambda_max": 2.470},    # Minimum and maximum instrumental wavelength range [µm]
                       'size_core':    3,                                             # Size in pixels of the FWHM  (here : side of the PSF core box)
                       'apodizers':    {"NO_SP":   ApodizerInfo(0.84, 50),            # Available apodizers (transmission, IWA [mas])
                                        "SP1":     ApodizerInfo(0.45, 70),            # Available apodizers (transmission, IWA [mas])
                                        "SP2":     ApodizerInfo(0.35, 100),           # Available apodizers (transmission, IWA [mas])
                                        "SP_Prox": ApodizerInfo(0.68, 10)},           # Available apodizers (transmission, IWA [mas])
                       'strehls':      ["JQ1"],                                # Available strehls
                       'coronagraphs': [None],                                        # Available coronagraphs
                       'spec':         {"RON":          10.0,                         # Read-Out Noise [e-/px/DIT]
                                        "dark_current": 0.0053,                       # Dark current [e-/px/s]
                                        "FOV":          0.8,                          # Field Of View [arcsec]
                                        "pxscale":      0.004,                        # Pixel scale [arcsec/px]
                                        "minDIT":       1.3/60,                      # Minimum Detector Integration Time [mn]
                                        "maxDIT":       3,                            # Maximum Detector Integration Time [mn]
                                        "saturation_e": 40_000},                      # Full well capacity [e-] 
                     }




# config_data_ANDES = {
#                      'name':                      "ANDES",                                           # Instrument name
#                      'type':                      "IFU_fiber",                                       # Instrument type
#                      'base':                      "ground",                                          # "ground" or "space" based
#                      'latitude':                  -24.627,                                           # Geographic latitude [°]
#                      'longitude':                 -70.404,                                           # Geographic longitude [°]
#                      'altitude':                  2635,                                              # Altitude [m]
#                      'sep_unit':                  "mas",                                             # Unit for angular separation
#                      'telescope':                 {"diameter": 38.452, "area": 980.},                # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
#                      'gratings':                  {"YJH_10mas_50":  GratingInfo(0.95, 1.8,  50000),  # Spectral range [µm], resolving power
#                                                    "YJH_10mas_75":  GratingInfo(0.95, 1.8,  75000),  # Spectral range [µm], resolving power
#                                                    "YJH_10mas_100": GratingInfo(0.95, 1.8, 100000),  # Spectral range [µm], resolving power
#                                                    "YJH_10mas_125": GratingInfo(0.95, 1.8, 125000),  # Spectral range [µm], resolving power
#                                                    "YJH_10mas_150": GratingInfo(0.95, 1.8, 150000)}, # Spectral range [µm], resolving power
#                      'lambda_range':              {"lambda_min": 0.95, "lambda_max": 1.8},           # Minimum and maximum instrumental wavelength range [µm]
#                      'pxscale':                   {"YJH_10mas_50":  0.010,                           # Pixel scale [arcsec/px]
#                                                    "YJH_10mas_75":  0.010,                           # Pixel scale [arcsec/px]
#                                                    "YJH_10mas_100": 0.010,                           # Pixel scale [arcsec/px]
#                                                    "YJH_10mas_125": 0.010,                           # Pixel scale [arcsec/px]
#                                                    "YJH_10mas_150": 0.010},                          # Pixel scale [arcsec/px]
#                      'size_core':                 1,                                                 # Size in pixels of the FWHM  (here : 1 fiber on the planet)
#                      'injection':                 {"YJH_10mas_50":  0.671,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
#                                                    "YJH_10mas_75":  0.671,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
#                                                    "YJH_10mas_100": 0.671,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
#                                                    "YJH_10mas_125": 0.671,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
#                                                    "YJH_10mas_150": 0.671},                          # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
#                      'pixel_detector_projection': 3,                                                 # Effective number of pixels per spectral channel (here : first guess)
#                      'apodizers':                 {"NO_SP": ApodizerInfo(1, 0)},                     # Available apodizers
#                      'strehls':                   ["MED"],                                           # Available strehls
#                      'coronagraphs':              [None, "LYOT"],                                    # Available coronagraphs
#                      'spec':                      {"RON":          4.5,                              # Read-Out Noise [e-/px/DIT]
#                                                    "dark_current": 0.0053,                           # Dark current [e-/px/s]
#                                                    "FOV":          0.144,                            # Field Of View [arcsec] (16 mas * 9 fibers)
#                                                    "minDIT":       1.3/60,                          # Minimum Detector Integration Time [mn]
#                                                    "maxDIT":       5,                                # Maximum Detector Integration Time [mn]
#                                                    "saturation_e": 40_000},                          # Full well capacity [e-] 
#                    }



config_data_ANDES = {
                     'name':                      "ANDES",                                           # Instrument name
                     'type':                      "IFU_fiber",                                       # Instrument type
                     'base':                      "ground",                                          # "ground" or "space" based
                     'latitude':                  -24.627,                                           # Geographic latitude [°]
                     'longitude':                 -70.404,                                           # Geographic longitude [°]
                     'altitude':                  2635,                                              # Altitude [m]
                     'sep_unit':                  "mas",                                             # Unit for angular separation
                     'telescope':                 {"diameter": 38.452, "area": 980.},                # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                     'gratings':                  {"YJH_5mas_100":  GratingInfo(0.95, 1.8, 100000),  # Spectral range [µm], resolving power
                                                   "YJH_10mas_100": GratingInfo(0.95, 1.8, 100000),  # Spectral range [µm], resolving power
                                                   "YJH_16mas_100": GratingInfo(0.95, 1.8, 100000)}, # Spectral range [µm], resolving power
                     'lambda_range':              {"lambda_min": 0.95, "lambda_max": 1.8},           # Minimum and maximum instrumental wavelength range [µm]
                     'pxscale':                   {"YJH_5mas_100":  0.005,                           # Pixel scale [arcsec/px]
                                                   "YJH_10mas_100": 0.010,                           # Pixel scale [arcsec/px]
                                                   "YJH_16mas_100": 0.016},                          # Pixel scale [arcsec/px]
                     'size_core':                 1,                                                 # Size in pixels of the FWHM  (here : 1 fiber on the planet)
                     'injection':                 {"YJH_5mas_100":  0.842,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
                                                   "YJH_10mas_100": 0.671,                           # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
                                                   "YJH_16mas_100": 0.620},                          # Mean injection efficiency in the fiber (the fact that the planet could not be perfectly centered in the fiber)
                     'pixel_detector_projection': 3,                                                 # Effective number of pixels per spectral channel (here : first guess)
                     'apodizers':                 {"NO_SP": ApodizerInfo(1, 0)},                     # Available apodizers
                     'strehls':                   ["MED"],                                           # Available strehls
                     'coronagraphs':              [None, "LYOT"],                                    # Available coronagraphs
                     'spec':                      {"RON":          4.5,                              # Read-Out Noise [e-/px/DIT]
                                                   "dark_current": 0.0053,                           # Dark current [e-/px/s]
                                                   "FOV":          0.144,                            # Field Of View [arcsec] (16 mas * 9 fibers)
                                                   "minDIT":       1.3/60,                           # Minimum Detector Integration Time [mn]
                                                   "maxDIT":       5,                                # Maximum Detector Integration Time [mn]
                                                   "saturation_e": 40_000},                          # Full well capacity [e-] 
                   }

#------------------------------------------------------------------------------#
#                                  VLT:                                        #
#------------------------------------------------------------------------------#

config_data_ERIS = {
                    'name':         "ERIS",                                        # Instrument name
                    'type':         "IFU",                                         # Instrument type
                    'base':         "ground",                                      # "ground" or "space" based
                    'latitude':     -24.627,                                       # Geographic latitude [°]
                    'longitude':    -70.404,                                       # Geographic longitude [°]
                    'altitude':     2635,                                          # Altitude [m]
                    'sep_unit':     "mas",                                         # Unit for angular separation
                    'telescope':    {"diameter": 8, "area": 49.3},                 # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                    'gratings':     {"J_low":    GratingInfo(1.09, 1.42,  5000.),  # Spectral range [µm], resolving power
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
                    'lambda_range': {"lambda_min": 1.08, "lambda_max": 2.48},      # Minimum and maximum instrumental wavelength range [µm]
                    'size_core':    3,                                             # Size in pixels of the FWHM  (here : side of the PSF core box)
                    'apodizers':    {"NO_SP": ApodizerInfo(1, 0)},                 # Available apodizers
                    'strehls':      ["JQ0"],                                       # Available strehls
                    'coronagraphs': [None],                                        # Available coronagraphs
                    'spec':         {"RON":          12.0,                         # Read-Out Noise [e-/px/DIT]
                                     "dark_current": 0.1,                          # Dark current [e-/px/s]
                                     "FOV":          0.8,                          # Field Of View [arcsec]
                                     "pxscale":      0.025,                        # Pixel scale [arcsec/px]
                                     "minDIT":       2.62/60,                        # Minimum Detector Integration Time [mn]
                                     "maxDIT":       2,                            # Maximum Detector Integration Time [mn]
                                     "saturation_e": 40_000},                      # Full well capacity [e-] 
                   }




config_data_HiRISE = {
                      'name':                      "HiRISE",                                 # Instrument name
                      'type':                      "IFU_fiber",                              # Instrument type
                      'base':                      "ground",                                 # "ground" or "space" based
                      'latitude':                  -24.627,                                  # Geographic latitude [°]
                      'longitude':                 -70.404,                                  # Geographic longitude [°]
                      'altitude':                  2635,                                     # Altitude [m]
                      'sep_unit':                  "arcsec",                                 # Unit for angular separation
                      'telescope':                 {"diameter": 8, "area": 49.3},            # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                      'gratings':                  {"H": GratingInfo(1.43, 1.78, 140_000)},  # Spectral range [µm], resolving power
                      'lambda_range':              {"lambda_min": 1.43, "lambda_max": 1.78}, # Minimum and maximum instrumental wavelength range [µm]
                      'size_core':                 1,                                        # Size in pixels of the FWHM                      (here : 1 fiber on the planet)
                      'pixel_detector_projection': 2 * 2.04,                                 # Effective number of pixels per spectral channel (here : 2 × Npx_x × Npx_y)
                      'apodizers':                 {"NO_SP": ApodizerInfo(1, 0)},            # Available apodizers
                      'strehls':                   ["MED"],                                  # Available strehls
                      'coronagraphs':              [None],                                   # Available coronagraphs
                      'spec':                      {"RON":          12,                      # Read-Out Noise [e-/px/DIT]
                                                    "dark_current": 0.0053,                  # Dark current [e-/px/s]
                                                    "FOV":          4,                       # Field Of View [arcsec]
                                                    "pxscale":      0.01225,                 # Pixel scale [arcsec/px]
                                                    "minDIT":       1.4725/60,               # Minimum Detector Integration Time [mn]
                                                    "maxDIT":       20,                      # Maximum Detector Integration Time [mn]
                                                    "saturation_e": 64_000},                 # Full well capacity [e-] 
                     }



#------------------------------------------------------------------------------#
#                                  JWST:                                       #
#------------------------------------------------------------------------------#

config_data_MIRIMRS = {
                       'name':        "MIRIMRS",                                                # Instrument name
                       'type':        "IFU",                                                    # Instrument type
                       'base':        "space",                                                  # "ground" or "space" based
                       'sep_unit':    "arcsec",                                                 # Unit for angular separation
                       'telescope':   {"diameter": 6.6052, "area": 25.032},                     # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                       'gratings':    {"1SHORT":  GratingInfo(4.90,  5.741,  (3320.+3710.)/2),  # Spectral range [µm], resolving power
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
                      'lambda_range': {"lambda_min": 4.90, "lambda_max": 27.90},                # Minimum and maximum instrumental wavelength range [µm]
                      'pxscale':      {"1SHORT":0.13, "1MEDIUM":0.13, "1LONG":0.13,             # Pixel scale [arcsec/px] (channel 1 with dithering)
                                       "2SHORT":0.17, "2MEDIUM":0.17, "2LONG":0.17,             # Pixel scale [arcsec/px] (channel 2 with dithering)
                                       "3SHORT":0.20, "3MEDIUM":0.20, "3LONG":0.20,             # Pixel scale [arcsec/px] (channel 3 with dithering)
                                       "4SHORT":0.35, "4MEDIUM":0.35, "4LONG":0.35},            # Pixel scale [arcsec/px] (channel 4 with dithering)
                      'pxscale0':     {"1SHORT":0.196, "1MEDIUM":0.196, "1LONG":0.196,          # Pixel scale [arcsec/px] (channel 1 without dithering)
                                       "2SHORT":0.196, "2MEDIUM":0.196, "2LONG":0.196,          # Pixel scale [arcsec/px] (channel 2 without dithering)
                                       "3SHORT":0.245, "3MEDIUM":0.245, "3LONG":0.245,          # Pixel scale [arcsec/px] (channel 3 without dithering)
                                       "4SHORT":0.273, "4MEDIUM":0.273, "4LONG":0.273},         # Pixel scale [arcsec/px] (channel 4 without dithering)
                      'size_core':    3,                                                        # Size in pixels of the FWHM  (here : side of the PSF core box)
                      'R_cov':        2.4,                                                      # Spatial covariance factor   (here : for size_core = 3)
                      'apodizers':    {"NO_SP": ApodizerInfo(1, 0)},                            # Available apodizers
                      'strehls':      ["NO_JQ"],                                                # Available strehls
                      'coronagraphs': [None],                                                   # Available coronagraphs
                      'spec':         {"RON":          14.0*np.sqrt(8),                         # Read-Out Noise [e-/px/DIT]
                                       "dark_current": 0.2,                                     # Dark current [e-/px/s]
                                       "FOV":          5.2,                                     # Field Of View [arcsec]
                                       "minDIT":       2.775/60,                                # Minimum Detector Integration Time [mn]
                                       "maxDIT":       5,                                       # Maximum Detector Integration Time [mn]
                                       "saturation_e": 210_000},                                # Full well capacity [e-] 
                     }



config_data_NIRCam = {
                      'name':         "NIRCam",                                   # Instrument name
                      'type':         "imager",                                   # Instrument type
                      'base':         "space",                                    # "ground" or "space" based
                      'sep_unit':     "arcsec",                                   # Unit for angular separation
                      'telescope':    {"diameter": 6.6052, "area": 25.032},       # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                      'gratings':     {"F250M": GratingInfo(2.35, 2.75, R0_min),  # Spectral range [µm], resolving power (R=R0_min abritrary, used for interpolation)
                                       "F300M": GratingInfo(2.65, 3.35, R0_min),  # Spectral range [µm], resolving power
                                       "F410M": GratingInfo(3.70, 4.60, R0_min),  # Spectral range [µm], resolving power
                                       #"F480M": GratingInfo(4.58, 5.09, R0_min),  # Spectral range [µm], resolving power
                                       #"F150W": GratingInfo(1.33, 1.67, R0_min),  # Spectral range [µm], resolving power
                                       "F356W": GratingInfo(3.00, 4.25, R0_min),  # Spectral range [µm], resolving power
                                       "F444W": GratingInfo(3.65, 5.19, R0_min)}, # Spectral range [µm], resolving power
                      'lambda_range': {"lambda_min": 1.33, "lambda_max": 5.19},   # Minimum and maximum instrumental wavelength range [µm]
                      'size_core':    3,                                          # Size in pixels of the FWHM  (here : side of the PSF core box)
                      'apodizers':    {"NO_SP": ApodizerInfo(1, 0)},              # Available apodizers
                      'strehls':      ["NO_JQ"],                                  # Available strehls
                      'coronagraphs': ["MASK335R"],                               # Available coronagraphs
                      'spec':         {"RON":          13.17*np.sqrt(2),          # Read-Out Noise [e-/px/DIT]
                                       "dark_current": 34.2/1000,                 # Dark current [e-/px/s]
                                       "FOV":          10,                        # Field Of View [arcsec]
                                       "pxscale":      0.063,                     # Pixel scale [arcsec/px]
                                       "minDIT":       20.155/60,                 # Minimum Detector Integration Time [mn]
                                       "maxDIT":       5,                         # Maximum Detector Integration Time [mn]
                                       "saturation_e": 62000},                    # Full well capacity [e-] 
                     }



config_data_NIRSpec = {
                       'name':         "NIRSpec",                                        # Instrument name
                       'type':         "IFU",                                            # Instrument type
                       'base':         "space",                                          # "ground" or "space" based
                       'sep_unit':     "arcsec",                                         # Unit for angular separation
                       'telescope':    {"diameter": 6.6052, "area": 25.032},             # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                       'gratings':     {"G140H_F100LP": GratingInfo(0.98,  1.89, 2700),  # Spectral range [µm], resolving power
                                        "G235H_F170LP": GratingInfo(1.66,  3.17, 2700),  # Spectral range [µm], resolving power
                                        "G395H_F290LP": GratingInfo(2.87,  5.27, 2700)}, # Spectral range [µm], resolving power
                       'lambda_range': {"lambda_min": 0.90, "lambda_max": 5.27},         # Minimum and maximum instrumental wavelength range [µm]
                       'size_core':    3,                                                # Size in pixels of the FWHM  (here : side of the PSF core box)
                       'R_cov':        1.7,                                              # Spatial covariance factor   (here : for size_core = 3)
                       'apodizers':    {"NO_SP": ApodizerInfo(1, 0)},                    # Available apodizers
                       'strehls':      ["NO_JQ"],                                        # Available strehls
                       'coronagraphs': [None],                                           # Available coronagraphs
                       'spec':         {"RON":          14*np.sqrt(2),                   # Read-Out Noise [e-/px/DIT]
                                        "dark_current": 0.008,                           # Dark current [e-/px/s]
                                        "FOV":          5,                               # Field Of View [arcsec]
                                        "pxscale":      0.1045,                          # Pixel scale [arcsec/px]
                                        "minDIT":       14.58889/60,                     # Minimum Detector Integration Time [mn]
                                        "maxDIT":       5,                               # Maximum Detector Integration Time [mn]
                                        "saturation_e": 200_000},                        # Full well capacity [e-] 
                      }



#------------------------------------------------------------------------------#
#                               Test bench:                                    #
#------------------------------------------------------------------------------#

# VIPA spectrometer at 152cm telescope of the OHP
config_data_VIPAPYRUS = {
                         'name':                      "VIPAPYRUS",                                 # Instrument name
                         'type':                      "IFU_fiber",                                 # Instrument type
                         'base':                      "ground",                                    # "ground" or "space" based
                         'latitude':                  43.92,                                       # Geographic latitude [°N]
                         'longitude':                 5.712,                                       # Geographic longitude [°E]
                         'altitude':                  650,                                         # Altitude [m]
                         'sep_unit':                  "arcsec",                                    # Unit for angular separation
                         'telescope':                 {"diameter": 1.52, "area": 1.81},            # All-glass diameter [m] and effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
                         'gratings':                  {"H": GratingInfo(1.5525, 1.760, 75_000)},   # Spectral range [µm], resolving power
                         'lambda_range':              {"lambda_min": 1.5525, "lambda_max": 1.760}, # Minimum and maximum instrumental wavelength range [µm]
                         'size_core':                 1,                                           # Size in pixels of the FWHM                      (here: 1 fiber on the planet)
                         'R_corr':                    0.65,                                        # Corrective factor on photon noise               (here: R_corr = scale_poisson**2)
                         'pixel_detector_projection': 8.655,                                       # Effective number of pixels per spectral channel (here: scale_flat**2 / scale_poisson**2)
                         'apodizers':                 {"NO_SP": ApodizerInfo(1, 0)},               # Available apodizers
                         'strehls':                   ["MED"],                                     # Available strehls
                         'coronagraphs':              [None],                                      # Available coronagraphs
                         'spec':                      {"RON":          13,                         # Read-Out Noise [e-/px/DIT]
                                                       "dark_current": 0.0053,                     # Dark current [e-/px/s]
                                                       "FOV":          4,                          # Field Of View [arcsec]
                                                       "pxscale":      0.15,#0.24,                 # Pixel scale [arcsec/px]                         (here : pxscale ~ 1.03*lambda/D, i.e. size of the fiber/FWHM)
                                                       "minDIT":       1.4725/60,                  # Minimum Detector Integration Time [mn]
                                                       "maxDIT":       5,                          # Maximum Detector Integration Time [mn]
                                                       "saturation_e": 64_000},                    # Full well capacity [e-] 
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
#                        'telescope':    {"diameter": 38.452, "area": 980.},         # All-glass diameter [m] and Effective collecting area [m²], accounting for central hole, secondary mirror, and spider obscuration
#                        'gratings':     {"H": GratingInfo(1.5601, 1.69, 7555)},     # Spectral range [µm], resolving power
#                        'lambda_range': {"lambda_min": 1.5601, "lambda_max": 1.69}, # Minimum and maximum instrumental wavelength range [µm]
#                        'size_core':    3,                                          # Size in pixels of the FWHM  (here : side of the PSF core box)
#                        'apodizers':    {"NO_SP":   ApodizerInfo(0.84, 30),         # Available apodizers (transmission, IWA [mas])
#                                         "SP1":     ApodizerInfo(0.45, 30),         # Available apodizers (transmission, IWA [mas])
#                                         "SP2":     ApodizerInfo(0.35, 30),         # Available apodizers (transmission, IWA [mas])
#                                         "SP_Prox": ApodizerInfo(0.68, 30)},        # Available apodizers (transmission, IWA [mas])
#                        'strehls':      ["JQ1"],                                    # Available strehls
#                        'coronagraphs': [None],                                     # Available coronagraphs
#                        'spec':         {"RON":          10.0,                      # Read-Out Noise [e-/px/DIT]
#                                         "dark_current": 0.0053,                    # Dark current [e-/px/s]
#                                         "FOV":          0.8,                       # Field Of View [arcsec]
#                                         "pxscale":      0.004,                     # Pixel scale [arcsec/px]
#                                         "minDIT":       2.62/60,                     # Minimum Detector Integration Time [mn]
#                                         "maxDIT":       1,                         # Maximum Detector Integration Time [mn]
#                                         "saturation_e": 40_000},                   # Full well capacity [e-] 
#                      }


###############################################################################

# -------------------------
# Instrument registry
# -------------------------
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

instru_name_list = [config_data["name"] for config_data in config_data_list]

instru_with_systematics = ["MIRIMRS", "NIRSpec"]

colors_instru = {"HARMONI": "royalblue",
                 "ANDES":   "gray",
                 "ERIS":    "crimson",
                 "MIRIMRS": "seagreen",
                 "NIRCam":  "darkviolet",
                 "NIRSpec": "rosybrown"}

# -------------------------
# Aggregate wavelength coverage and band registry
# -------------------------
LMIN    = np.inf   # Global min λ [µm] over all instruments
LMAX    = -np.inf  # Global max λ [µm] over all instruments
bands   = []       # Unique list of band/grating names across instruments
instrus = []       # Instrument names (order of config_data_list)

for config_data in config_data_list:
    instru = config_data["name"]
    instrus.append(instru)
    
    # Expose per-instrument λ-range as globals: lmin_<INSTRU>, lmax_<INSTRU>
    globals()[f"lmin_{instru}"] = config_data["lambda_range"]["lambda_min"]
    globals()[f"lmax_{instru}"] = config_data["lambda_range"]["lambda_max"]
    
    # Update global λ coverage
    if LMIN > config_data["lambda_range"]["lambda_min"]:
        LMIN = config_data["lambda_range"]["lambda_min"]
    if LMAX < config_data["lambda_range"]["lambda_max"]:
        LMAX = config_data["lambda_range"]["lambda_max"]
        
    # Register bands and expose cut-on/off as globals: lmin_<BAND>, lmax_<BAND>
    for name_band in config_data["gratings"]:
        if name_band not in bands:
            bands.append(name_band)
            globals()[f"lmin_{name_band}"] = config_data['gratings'][name_band].lmin
            globals()[f"lmax_{name_band}"] = config_data['gratings'][name_band].lmax

lmin_VIPA = globals()["lmin_VIPAPYRUS"]
lmax_VIPA = globals()["lmax_VIPAPYRUS"]

# --- MIRI Imaging filters (cut-on / cut-off) [µm] ---
lmin_F560W  = 4.86   ; lmax_F560W  = 6.43
lmin_F770W  = 6.62   ; lmax_F770W  = 8.74
lmin_F1000W = 8.57   ; lmax_F1000W = 11.30
lmin_F1130W = 10.95  ; lmax_F1130W = 12.66
lmin_F1280W = 11.52  ; lmax_F1280W = 14.05
lmin_F1500W = 13.29  ; lmax_F1500W = 16.84
lmin_F1800W = 16.24  ; lmax_F1800W = 20.05
lmin_F2100W = 18.55  ; lmax_F2100W = 24.50
lmin_F2550W = 22.47  ; lmax_F2550W = 29.89

bands += ["F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"]

# Coronagraphic filters (Lyot / 4QPM)
lmin_F1065C = 10.0   ; lmax_F1065C = 11.0
lmin_F1140C = 11.1   ; lmax_F1140C = 11.9
lmin_F1550C = 14.9   ; lmax_F1550C = 15.8
lmin_F2300C = 22.5   ; lmax_F2300C = 23.9

bands += ["F1065C", "F1140C", "F1550C", "F2300C"]

# WISE bands [µm]
lmin_W1 = 2.8 ; lmax_W1 = 3.8
lmin_W2 = 4.1 ; lmax_W2 = 5.1

bands += ["W1", "W2"]

# MKO bands [µm]
lmin_L = 3.42 ; lmax_L = 4.12
lmin_M = 4.47 ; lmax_M = 4.79

bands += ["L", "M"]

# PaBeta band [µm]
lmin_PaB = 1.274 ; lmax_PaB = 1.290

bands += ["PaB"]

# NIRCam bands [µm]
lmin_F150W = 1.3313 ; lmax_F150W = 1.6689
lmin_F480M = 4.5820 ; lmax_F480M = 5.0919

bands += ["F150W", "F480M"]


