import collections
import numpy as np

"""
File that stores hardwired data for use in HARMONI
pipeline. Data stored in dictionary format
with keywords.
"""

def get_config_data(instrument_name):
    """
    Get the parameters of an instrument

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

GratingInfo = collections.namedtuple('GratingInfo', 'lmin, lmax, R')
ApodizerInfo = collections.namedtuple('ApodizerInfo', 'transmission, sep')

config_data_HARMONI = {
    'name': "HARMONI",
    'telescope': {"diameter": 37, "area": 980.}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'gratings': {"HK": GratingInfo(1.450, 2.450, 3355.),
                 "H": GratingInfo(1.435, 1.815, 7104.),
                 "H_high": GratingInfo(1.538, 1.678, 17385.),
                 "K": GratingInfo(1.951, 2.469, 7104.),
                 "K1_high": GratingInfo(2.017, 2.20, 17385.),
                 "K2_high": GratingInfo(2.199, 2.40, 17385.)},
    'lambda_range': {"lambda_min":1.434, "lambda_max": 2.470}, #HARMONI
    'apodizers': {"SP1": ApodizerInfo(0.45, 70), "SP2": ApodizerInfo(0.35, 100), "SP3": ApodizerInfo(0.53, 50), "SP4": ApodizerInfo(0.59, 30), "NO_SP": ApodizerInfo(0.84, 50) , "SP_Prox": ApodizerInfo(0.7615, 30)}, # (transmission, iwa)     
    #'apodizers': {"SP1": ApodizerInfo(0.45, 30), "SP2": ApodizerInfo(0.35, 30), "SP3": ApodizerInfo(0.53, 30), "SP4": ApodizerInfo(0.59, 30), "NO_SP": ApodizerInfo(0.84, 30), "SP_Prox": ApodizerInfo(0.7615, 30)}, # (transmission, iwa)
    'strehl': {"JQ1", "JQ2", "MED"},
    'spec': {"RON": 10.0, "dark_current": 0.0053, "FOV": 0.8, "pixscale": 0.004, "minDIT": 0.026, "maxDIT": 1, "saturation_e": 40000., "Q_eff": 0.90},
    # e-, e-/s,arcsec,arcsec/px,min,min, e-/ph;
}

config_data_ERIS = {
    'name': "ERIS",
    'telescope': {"diameter": 8, "area": 49.3}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'gratings': {"J_low": GratingInfo(1.09, 1.42, 5000.), # J_low
                 "H_low": GratingInfo(1.45, 1.87, 5200.),
                 "K_low": GratingInfo(1.93, 2.48, 5600.),
                 "J_short": GratingInfo(1.10, 1.27, 10000.),
                 "J_middle": GratingInfo(1.18, 1.35, 10000.),
                 "J_long": GratingInfo(1.26, 1.43, 10000.),
                 "H_short": GratingInfo(1.46, 1.67, 10400.),
                 "H_middle": GratingInfo(1.56, 1.77, 10400.),
                 "H_long": GratingInfo(1.66, 1.87, 10400.),
                 "K_short": GratingInfo(1.93, 2.22, 11200.),
                 "K_middle": GratingInfo(2.06, 2.34, 11200.),
                 "K_long": GratingInfo(2.19, 2.47, 11200.)},
    'lambda_range': {"lambda_min": 1.08, "lambda_max": 2.48}, #ERIS
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)},
    'strehl': {"JQ1"},
    'spec': {"RON": 12.0, "dark_current": 0.1, "FOV": 0.8, "pixscale": 0.025, "minDIT": 0.026, "maxDIT": 2, "saturation_e": 40000., "Q_eff": 0.85},
    # e-, e-/s, arcsec, arcsec/px, min, min, e-, e-/ph;
}

config_data_MIRIMRS = {
    'name': "MIRIMRS",
    'telescope': {"diameter": 6.5, "area": 25.4}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'gratings': {"1SHORT": GratingInfo(4.90, 5.74, (3320+3710)/2), # R calculé en fonction du step des datas MAST
                 "1MEDIUM": GratingInfo(5.66, 6.63, (3750.+3190)/2),
                 "1LONG": GratingInfo(6.53, 7.65, (3610.+3100.)/2),
                 "2SHORT": GratingInfo(7.51, 8.77, (3110.+2990.)/2),
                 "2MEDIUM": GratingInfo(8.67, 10.13, (2750.+3170.)/2),
                 "2LONG": GratingInfo(10.02, 11.70, (2860.+3300.)/2)},
    'pixscale': {"1SHORT":0.13,"1MEDIUM":0.13,"1LONG":0.13,"2SHORT":0.170001,"2MEDIUM":0.170001,"2LONG":0.170001}, # en arcsec/px (avec dithering)
    'pixscale0': 0.196, # en arcsec/px (sans dithering)
    'lambda_range': {"lambda_min": 4.90, "lambda_max": 11.70}, #pas plus car le spectre des planètes va jusqu'à 12 µm
    'spec': {"RON": 14.0*np.sqrt(8), "dark_current": 0.1,"FOV":5.2, "minDIT": 2.775/60, "maxDIT": 5*60/60, "saturation_e": 200000., "Q_eff": 1}, # en réalité Q_eff ~ 0.54 mais elle est déjà comprise dans les transmissions
    #'spec': {"RON": 14.0, "dark_current": 0.1,"FOV":5.2, "minDIT": 2.775/60, "maxDIT": 250/60, "saturation_e": 200000., "Q_eff": 1}, # en réalité Q_eff ~ 0.54 mais elle est déjà comprise dans les transmissions
    # en SLOW mode on a RON = 14 e- # e-, e-/s, arcsec, min, min, e-, e-/ph ;
}

config_data_NIRCam = {
    'name': "NIRCam",
    'telescope': {"diameter": 6.5, "area": 25.4}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'gratings': {"F250M": GratingInfo(2.35, 2.75, 10000), # R = 10 000 (arbitraire) juste pour "interpoler" les spectres sur les bandes
                 "F300M": GratingInfo(2.65, 3.35, 10000),
                 "F410M": GratingInfo(3.70, 4.60, 10000),
                 "F356W": GratingInfo(3.00, 4.25, 10000),
                 "F444W": GratingInfo(3.65, 5.19, 10000),},
    'lambda_range': {"lambda_min": 2.35, "lambda_max": 5.19}, #pas plus car le spectre des planètes va jusqu'à 12 µm
    'spec': {"RON": 13.25, "dark_current": 34.2/1000, "FOV":10, "pixscale":  0.063, "minDIT": 20.155/60, "maxDIT": 308/60, "saturation_e": 83300, "Q_eff": 1}, # en réalité Q_eff ~ 0.54 mais elle est déjà comprise dans les transmissions
    # e-, e-/s, arcsec, arcsec/px, min, min, e-, e-/ph ;
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
    'telescope': {"diameter": 6.5, "area": 25.4}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'gratings': {"G140M_F070LP": GratingInfo(0.90, 1.27, 1000), # R calculé en fonction du step des datas MAST
                 "G140M_F100LP": GratingInfo(0.97, 1.89, 1000),
                 "G235M_F170LP": GratingInfo(1.66, 3.17, 1000),
                 "G395M_F290LP": GratingInfo(2.87, 5.27, 1000),
                 "G140H_F070LP": GratingInfo(0.95, 1.27, 2700),
                 "G140H_F100LP": GratingInfo(0.98, 1.89, 2700),
                 "G235H_F170LP": GratingInfo(1.66, 3.17, 2700),
                 "G395H_F290LP": GratingInfo(2.87, 5.27, 2700)},
    'lambda_range': {"lambda_min": 0.90, "lambda_max": 5.27}, #pas plus car le spectre des planètes va jusqu'à 12 µm
    'spec': {"RON": 10, "dark_current": 0.008,"FOV":3.15, "pixscale":0.1045, "minDIT": 2.775/60, "maxDIT": 5*60/60, "saturation_e": 200000., "Q_eff": 1}, # en réalité Q_eff ~ 0.54 mais elle est déjà comprise dans les transmissions
    # e-, e-/s, arcsec, arcsec/px, min, min, e-, e-/ph ;
}

config_data_ANDES = {
    'name': "ANDES",
    'telescope': {"diameter": 37, "area": 980.}, # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'gratings': {"Y_small": GratingInfo(0.95, 1.15, 100000),
                 "J_small": GratingInfo(1.1, 1.4, 100000),
                 "H_small": GratingInfo(1.435, 1.835, 100000),
                 "Y_large": GratingInfo(0.95, 1.15, 100000),
                 "J_large": GratingInfo(1.1, 1.4, 100000),
                 "H_large": GratingInfo(1.435, 1.835, 100000),},
    'lambda_range': {"lambda_min":0.95, "lambda_max": 1.835}, #ANDES
    'pixscale': {"Y_small":0.01,"J_small":0.01,"H_small":0.01,"Y_large":0.1,"J_large":0.1,"H_large":0.1},
    'apodizers': {"NO_SP": ApodizerInfo(1, 0)},
    'strehl': {"JQ1", "JQ2", "MED"},
    'spec': {"RON": 4.5, "dark_current": 0.0053, "FOV": 0.2, "minDIT": 0.026, "maxDIT": 1, "saturation_e": 40000., "Q_eff": 0.90},
    # e-, e-/s,arcsec,arcsec/px,min,min, e-/ph;
}


config_data_list = [config_data_HARMONI,config_data_ERIS,config_data_MIRIMRS,config_data_NIRCam,config_data_NIRSpec,config_data_ANDES]



