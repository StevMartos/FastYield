import collections
import numpy as np

def get_config_data(instrument_name):
    """
    Get the specifications of an instrument

    Parameters
    ----------
    instrument_name : str
        name of the considered instrument

    Returns : collections
        config parameters of the instrument
    """
    for dict in config_data_list:
        if dict['name'] == instrument_name:
            return(dict)
    raise NameError('Undefined Instrument Name')

GratingInfo  = collections.namedtuple('GratingInfo', 'lmin, lmax, R')
ApodizerInfo = collections.namedtuple('ApodizerInfo', 'transmission, sep')

#------------------------------------------------------------------------------#
#                                  ELT:                                        #
#------------------------------------------------------------------------------#

config_data_HARMONI = {
    'name': "HARMONI", 
    'type': "IFU", 
    'base': "ground", 
    'latitude': -24.627,  # °N Latitude of Paranal
    'longitude': -70.404, # °E Longitude of Paranal
    'altitude': 2635,     # Altitude of Paranal (in m)
    'sep_unit': "mas", 
    'telescope': {"diameter": 38.452, "area": 980.}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"J":       GratingInfo(1.046, 1.324, 7555), 
                 "H":       GratingInfo(1.435, 1.815, 7555), 
                 "H_high":  GratingInfo(1.538, 1.678, 17385), 
                 "HK":      GratingInfo(1.450, 2.450, 3355), 
                 "K":       GratingInfo(1.951, 2.469, 7555), 
                 "K1_high": GratingInfo(2.017, 2.20, 17385), 
                 "K2_high": GratingInfo(2.199, 2.40, 17385)}, 
    'lambda_range': {"lambda_min":1.046, "lambda_max": 2.470}, #HARMONI
    'size_core': 3, # size of box side corresponding to FWHM
    'apodizers': {"NO_SP": ApodizerInfo(0.84, 50), "SP1": ApodizerInfo(0.45, 70), "SP2": ApodizerInfo(0.35, 100), "SP_Prox": ApodizerInfo(0.68, 30)}, # (transmission, iwa)     
    #'apodizers': {"NO_SP": ApodizerInfo(0.84, 30), "SP1": ApodizerInfo(0.45, 30), "SP2": ApodizerInfo(0.35, 30), "SP_Prox": ApodizerInfo(0.68, 30)}, # (transmission, iwa)
    'strehls': ["JQ1"], 
    'spec': {"RON": 10.0, "dark_current": 0.0053, "FOV": 0.8, "pxscale": 0.004, "minDIT": 0.026, "maxDIT": 1, "saturation_e": 40000.}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-/ph;
}

config_data_ANDES = {
    'name': "ANDES", 
    'type': "IFU_fiber", 
    'base': "ground", 
    'latitude': -24.627,  # °N Latitude of Paranal
    'longitude': -70.404, # °E Longitude of Paranal
    'altitude': 2635,     # Altitude of Paranal (in m)
    'sep_unit': "mas", 
    'telescope': {"diameter": 38.452, "area": 980.}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"YJH_5mas":  GratingInfo(0.95, 1.8, 100000), 
                 "YJH_10mas": GratingInfo(0.95, 1.8, 100000),}, 
    'lambda_range': {"lambda_min":0.95, "lambda_max": 1.8}, #ANDES
    'pxscale': {"YJH_5mas":0.005, "YJH_10mas":0.010}, # in arcsec/px 
    'size_core': 1, # 1 fiber on the planet
    'FOV_fiber': 10, # nb of fiber/spaxel across the FOV (the FOV is then given by FOV_fiber*pxscale)
    'pixel_detector_projection': 10, # nb of pixels on which the fiber's signal is projected along the direction perpendicular to spectral dispersion
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["MED"], 
    'spec': {"RON": 4.5, "dark_current": 0.0053, "minDIT": 0.026, "maxDIT": 1, "saturation_e": 40000.}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-/ph;
}

#------------------------------------------------------------------------------#
#                                  VLT:                                        #
#------------------------------------------------------------------------------#

config_data_ERIS = {
    'name': "ERIS", 
    'type': "IFU", 
    'base': "ground", 
    'latitude': -24.627,  # °N Latitude of Paranal
    'longitude': -70.404, # °E Longitude of Paranal
    'altitude': 2635,     # Altitude of Paranal (in m)
    'sep_unit': "mas", 
    'telescope': {"diameter": 8, "area": 49.3}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"J_low":    GratingInfo(1.09, 1.42, 5000.), # J_low
                 "H_low":    GratingInfo(1.45, 1.87, 5200.), 
                 "K_low":    GratingInfo(1.93, 2.48, 5600.), 
                 "J_short":  GratingInfo(1.10, 1.27, 10000.), 
                 "J_middle": GratingInfo(1.18, 1.35, 10000.), 
                 "J_long":   GratingInfo(1.26, 1.43, 10000.), 
                 "H_short":  GratingInfo(1.46, 1.67, 10400.), 
                 "H_middle": GratingInfo(1.56, 1.77, 10400.), 
                 "H_long":   GratingInfo(1.66, 1.87, 10400.), 
                 "K_short":  GratingInfo(1.93, 2.22, 11200.), 
                 "K_middle": GratingInfo(2.06, 2.34, 11200.), 
                 "K_long":   GratingInfo(2.19, 2.47, 11200.)}, 
    'lambda_range': {"lambda_min": 1.08, "lambda_max": 2.48}, #ERIS
    'size_core': 3, # size of box side corresponding to FWHM
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["JQ0"], 
    'spec': {"RON": 12.0, "dark_current": 0.1, "FOV": 0.8, "pxscale": 0.025, "minDIT": 0.026, "maxDIT": 2, "saturation_e": 40000.}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-, e-/ph;
}

config_data_HiRISE = {
    'name': "HiRISE", 
    'type': "IFU_fiber", 
    'base': "ground", 
    'latitude': -24.627,  # °N Latitude of Paranal
    'longitude': -70.404, # °E Longitude of Paranal
    'altitude': 2635,     # Altitude of Paranal (in m)
    'sep_unit': "arcsec", 
    'telescope': {"diameter": 8, "area": 49.3}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"H": GratingInfo(1.43, 1.78, 140000.)}, 
    'lambda_range': {"lambda_min": 1.43, "lambda_max": 1.78}, # µm
    'size_core': 1, # 1 fiber on the planet
    'pixel_detector_projection': 2 * 2.04 , # ~ 2 * Npx_x * Npx_y (2x for the darks: assuming the same number of bkg as sci) = nb of pixels on which the fiber's signal is projected 
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["MED"], 
    'spec': {"RON": 12, "dark_current": 0.0053, "FOV":4, "pxscale":0.01225, "minDIT": 1.4725/60, "maxDIT": 20, "saturation_e": 64000.}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-/ph;
}

#------------------------------------------------------------------------------#
#                                  JWST:                                       #
#------------------------------------------------------------------------------#

config_data_MIRIMRS = {
    'name': "MIRIMRS", 
    'type': "IFU", 
    'base': "space", 
    'sep_unit': "arcsec", 
    'telescope': {"diameter": 6.6052, "area": 25.032}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"1SHORT":   GratingInfo(4.90, 5.741, (3320+3710)/2),
                  "1MEDIUM": GratingInfo(5.66, 6.63, (3750.+3190)/2), 
                  "1LONG":   GratingInfo(6.53, 7.65, (3610.+3100.)/2), 
                  "2SHORT":  GratingInfo(7.51, 8.77, (3110.+2990.)/2), 
                  "2MEDIUM": GratingInfo(8.67, 10.13, (2750.+3170.)/2), 
                  "2LONG":   GratingInfo(10.02, 11.70, (2860.+3300.)/2), 
                  "3SHORT":  GratingInfo(11.55, 13.47, (2530.+2880.)/2), 
                  "3MEDIUM": GratingInfo(13.34, 15.57, (1790.+2640.)/2), 
                  "3LONG":   GratingInfo(15.42, 17.98, (1980.+2790.)/2), 
                  "4SHORT":  GratingInfo(17.70, 20.915, (1460.+1930.)/2), 
                  "4MEDIUM": GratingInfo(20.69, 24.385, (1680.+1770.)/2), 
                  "4LONG":   GratingInfo(24.19, 27.9, (1630.+1330.)/2)}, 
    'lambda_range': {"lambda_min": 4.90, "lambda_max": 27.9},
    'pxscale': {"1SHORT":0.13, "1MEDIUM":0.13, "1LONG":0.13, "2SHORT":0.17, "2MEDIUM":0.17, "2LONG":0.17, "3SHORT":0.20, "3MEDIUM":0.20, "3LONG":0.20, "4SHORT":0.35, "4MEDIUM":0.35, "4LONG":0.35}, # en arcsec/px (avec dithering)
    'pxscale0': {"1":0.196, "2":0.196, "3":0.245, "4":0.273}, # en arcsec/px (withtout dithering)
    'size_core': 3, # size of box side corresponding to FWHM
    'R_cov': 2.4, # spatial covariance factor with size_core = 3 (1.55**2 = R_corr/R_dith (R_dith = R_corr_perpx))
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["NO_JQ"], 
    'spec': {"RON": 14.0*np.sqrt(8), "dark_current": 0.2, "FOV":5.2, "minDIT": 2.775/60, "maxDIT": 5, "saturation_e": 210000.},
    # en SLOW mode on a RON = 14 e- # e-, e-/s, arcsec, min, min, e-, e-/ph ;
}

config_data_NIRCam = {
    'name': "NIRCam", 
    'type': "imager", 
    'base': "space", 
    'sep_unit': "arcsec", 
    'telescope': {"diameter": 6.6052, "area": 25.032}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"F250M": GratingInfo(2.35, 2.75, 10000), # R = 10 000 (arbitrary) just to “interpolate” the spectra on the bands
                 "F300M": GratingInfo(2.65, 3.35, 10000), 
                 "F410M": GratingInfo(3.70, 4.60, 10000), 
                 "F356W": GratingInfo(3.00, 4.25, 10000), 
                 "F444W": GratingInfo(3.65, 5.19, 10000), }, 
    'lambda_range': {"lambda_min": 2.35, "lambda_max": 5.19}, 
    'size_core': 3, # size of box side corresponding to FWHM
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["NO_JQ"], 
    'spec': {"RON": 13.17*np.sqrt(2), "dark_current": 34.2/1000, "FOV":10, "pxscale":  0.063, "minDIT": 20.155/60, "maxDIT": 5, "saturation_e": 62000}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-, e-/ph ; # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance#gsc.tab=0
    'lambda_pivot': {"F250M": 2.503, 
                     "F300M": 2.996, 
                     "F410M": 4.092, 
                     "F356W": 3.563, 
                     "F444W": 4.421}, # en µm 
    'bandwidth': {"F250M": 0.181, 
                  "F300M": 0.318, 
                  "F410M": 0.436, 
                  "F356W": 0.787, 
                  "F444W": 1.024}, # en µm 
}

config_data_NIRSpec = {
    'name': "NIRSpec", 
    'type': "IFU", 
    'base': "space", 
    'sep_unit': "arcsec", 
    'telescope': {"diameter": 6.6052, "area": 25.032}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"G140H_F100LP": GratingInfo(0.98, 1.89, 2700), 
                 "G235H_F170LP": GratingInfo(1.66, 3.17, 2700), 
                 "G395H_F290LP": GratingInfo(2.87, 5.27, 2700)}, 
    'lambda_range': {"lambda_min": 0.90, "lambda_max": 5.27},
    'size_core': 3, # size of box side corresponding to FWHM
    'R_cov': 1.7, # spatial covariance factor with size_core = 3 
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["NO_JQ"], 
    'spec': {"RON": 14*np.sqrt(2), "dark_current": 0.008, "FOV": 5, "pxscale":0.1045, "minDIT": 14.58889/60, "maxDIT": 5, "saturation_e": 200000.}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-, e-/ph ;
}

#------------------------------------------------------------------------------#
#                               Test bench:                                    #
#------------------------------------------------------------------------------#

# VIPA spectrometer at 152cm telescope of the OHP
config_data_VIPAPYRUS = {
    'name': "VIPAPYRUS", 
    'type': "IFU_fiber", 
    'base': "ground", 
    'latitude': 43.92,  # °N Latitude of OHP
    'longitude': 5.712, # °E Longitude of OHP
    'altitude': 650,    # Altitude of OHP (in m)
    'sep_unit': "arcsec", 
    'telescope': {"diameter": 1.52, "area": 1.81}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
    'gratings': {"H": GratingInfo(1.552, 1.760, 70000), }, # µm
    'lambda_range': {"lambda_min": 1.552, "lambda_max": 1.760}, # VIPA
    'size_core': 1, # 1 fiber on the planet
    'pixel_detector_projection': 1.55 * 10 , # ~ 2 * k * Npx_x * Npx_y * 10 (2x for the darks) (10x because Npx_y has been calculated for 10x 70 000 of R) = nb of pixels on which the fiber's signal is projected 
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)}, 
    'strehls': ["MED"], 
    'spec': {"RON": 12, "dark_current": 0.0053, "FOV":4, "pxscale":0.24, "minDIT": 1.4725/60, "maxDIT": 5, "saturation_e": 64000.}, 
    # e-, e-/s, arcsec, arcsec/px, min, min, e-/ph;
} # pxscale ~ 1.03*lambda/D

# # HARMONI (HC Bench) demonstrator
# config_data_HARMONI = {
#     'name': "HARMONI", 
#     'type': "IFU", 
#     'base': "ground", 
#     'latitude': -24.627, # °N °N Latitude of Paranal
#     'longitude': -70.404, # °E °E Longitude of Paranal
#     'altitude': 2635, # Altitude of Paranal (in m)
#     'sep_unit': "mas", 
#     'telescope': {"diameter": 38.452, "area": 980.}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m2
#     'gratings': {"H": GratingInfo(1.5601, 1.69, 7555)}, 
#     'lambda_range': {"lambda_min":1.5601, "lambda_max": 1.69}, #HARMONI
#     'size_core': 3, # size of box side corresponding to FWHM
#     'apodizers': {"NO_SP": ApodizerInfo(0.84, 50), "SP1": ApodizerInfo(0.45, 70), "SP2": ApodizerInfo(0.35, 100), "SP3": ApodizerInfo(0.53, 50), "SP4": ApodizerInfo(0.59, 30), "SP_Prox": ApodizerInfo(0.68, 30)}, # (transmission, iwa)     
#     'strehls': ["JQ1", "MED"], 
#     'spec': {"RON": 10.0, "dark_current": 0.0053, "FOV": 0.8, "pxscale": 0.004, "minDIT": 0.026, "maxDIT": 1, "saturation_e": 40000.}, 
#     # e-, e-/s, arcsec, arcsec/px, min, min, e-/ph;
# }

###############################################################################

config_data_list = [config_data_HARMONI, config_data_ANDES, config_data_ERIS, config_data_MIRIMRS, config_data_NIRCam, config_data_NIRSpec, config_data_HiRISE, config_data_VIPAPYRUS]
instru_name_list = [config_data["name"] for config_data in config_data_list]

instru_with_systematics = ["MIRIMRS", "NIRSpec"]

R0_max = 300_000

# ALL BANDS
LMIN    = 1 # µm
LMAX    = 1 # µm
bands   = []
instrus = []
for config_data in config_data_list:
    instru = config_data["name"]
    instrus.append(instru)
    globals()["lmin_"+instru] = config_data["lambda_range"]["lambda_min"]
    globals()["lmax_"+instru] = config_data["lambda_range"]["lambda_max"]
    if LMIN > config_data["lambda_range"]["lambda_min"]:
        LMIN = config_data["lambda_range"]["lambda_min"]
    if LMAX < config_data["lambda_range"]["lambda_max"]:
        LMAX = config_data["lambda_range"]["lambda_max"]
    for name_band in config_data["gratings"]:
        if name_band not in bands:
            bands.append(name_band)
            globals()["lmin_"+name_band] = config_data['gratings'][name_band].lmin
            globals()["lmax_"+name_band] = config_data['gratings'][name_band].lmax
