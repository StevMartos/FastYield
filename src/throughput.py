from src.spectrum import *
from src.config import config_data_HARMONI

from astropy.io import ascii
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy import constants as const
from astropy import units as u
import numpy as np
import os
from scipy.optimize import curve_fit

path_file = os.path.dirname(__file__)
tppath = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/HARMONI/Instrumental_transmission/")

def telescope_throughput(waveobs, ao=True):
    """
    Compute the telescope transmission at the observed wavelengths
    Args:
        waveobs: numpy 1d array of N elements, input wavelengths of interest [micron]
        ao: Boolean, consider the transmission of the AO dichroic into the total optical path [default=True]
    Returns:
        tel_tr_interpolated: numpy 1d array of N elements, telescope transmission at the observed wavelengths
    """

    # Load the mirror transmission curve
    tel_tr = ascii.read(os.path.join(tppath, 'ELT_mirror_reflectivity.txt'))

    # Interpolate onto the input wavelength array
    f = interp1d(tel_tr['col1'], tel_tr['col2'],  bounds_error=False, fill_value=0)
    tel_tr_interpolated = f(waveobs)

    if ao is True:
        # Load the dichroic emission curve which is used to feed the AO system
        ao_tr = ascii.read(os.path.join(tppath, 'ao_dichroic.txt'))
        # Interpolate onto the input wavelength array
        f = interp1d(ao_tr['col1'], ao_tr['col2'],  bounds_error=False, fill_value=0)
        ao_tr_interpolated = 1. - f(waveobs)  # transmission
        tel_tr_interpolated = np.multiply(tel_tr_interpolated, ao_tr_interpolated)  # Combined both transmission curves
    return tel_tr_interpolated


def fprs_throughput(waveobs):
    """
    Compute the FPRS transmission at the observed wavelengths
    Args:
        waveobs: numpy 1d array of N elements, input wavelengths of interest [micron]
    Returns:
        fprs_tr_interpolated: numpy 1d array of N elements, FPRS transmission at the observed wavelengths
    """
    # Load the Focal plane relay system, not taken into account into the instrument grating transmission curve
    fprs_tr = ascii.read(os.path.join(tppath, 'FPRS.txt'))

    # Interpolate onto the input wavelength array
    f = interp1d(fprs_tr['col1'], fprs_tr['col2'],  bounds_error=False, fill_value=0)
    fprs_tr_interpolated = f(waveobs)

    return fprs_tr_interpolated


def instrument_throughput(waveobs, filter, CRYOSTAT=True):
    """
    Compute the instrument transmission at the observed wavelengths
    Args:
        waveobs: numpy 1d array of N elements, input wavelengths of interest [micron]
        filter: string, name of the filter of the observations
        CRYOSTAT: boolean, if set, consider the transmission of the HARMONI pre-IFU optics, IFU, and spectrograph [default=True]
    Returns:
        instrument_tr_interpolated: numpy 1d array of N elements, instrument transmission at the observed wavelengths
    """

    # Load the specific grating emission curve
    l_grating, emi_grating = np.loadtxt(os.path.join(tppath, filter + '_grating.txt'), unpack=True,
                                        comments="#", delimiter=",")
    emi_grating *= ((config_data_HARMONI['telescope']['diameter'] / 2.) ** 2. * np.pi / config_data_HARMONI['telescope'][
        'area'])  # scaling with effective surface

    # Interpolate onto the input wavelength array
    f = interp1d(l_grating, emi_grating, bounds_error=False, fill_value=0)
    emi_grating_interpolated = f(waveobs)

    instrument_tr_interpolated = 1. - emi_grating_interpolated

    if CRYOSTAT is True:
        # Load the lenses transmission profiles, not taken into account into the instrument grating transmission curve
        l_lens, emi_lens = np.loadtxt(os.path.join(tppath, 'HARMONI_lens_emissivity.txt'), unpack=True,
                                      comments="#", delimiter=",")
        cryo_lens_emi = 1. - (1. - emi_lens) ** 8  # 8 lenses in the cryostat, hard coded from HSIM
        cryo_lens_emi *= ((config_data_HARMONI['telescope']['diameter'] / 2.) ** 2. * np.pi / config_data_HARMONI['telescope'][
            'area'])  # scaling with effective surface
        # Interpolate over the input wavelength array
        f = interp1d(l_lens, cryo_lens_emi, bounds_error=False, fill_value=0)
        cryo_lens_emi_interpolated = f(waveobs)
        # Load the mirrors transmission profiles, not taken into account into the instrument grating transmission curve
        l_mirror, emi_mirror = np.loadtxt(os.path.join(tppath, 'HARMONI_mirror_emissivity.txt'),
                                          unpack=True, comments="#", delimiter=",")
        cryo_mirror_emi = 1. - (1. - emi_mirror) ** 19  # 19 mirrors in the cryostat, hard coded from HSIM
        cryo_mirror_emi *= ((config_data_HARMONI['telescope']['diameter'] / 2.) ** 2. * np.pi / config_data_HARMONI['telescope'][
            'area'])  # scaling with effective surface
        # Interpolate over the input wavelength array
        f = interp1d(l_mirror, cryo_mirror_emi, bounds_error=False, fill_value=0)
        cryo_mirror_emi_interpolated = f(waveobs)
        # Compute combined transmission profile
        cryo_emi_interpolated = 1. - ((1. - cryo_mirror_emi_interpolated) * (1. - cryo_lens_emi_interpolated))
        cryo_tr_interpolated = 1. - cryo_emi_interpolated
        instrument_tr_interpolated = np.multiply(instrument_tr_interpolated,
                                                 cryo_tr_interpolated)  # Combined both transmission curves

    return instrument_tr_interpolated

