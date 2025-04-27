# import instru configs
from src.config import *

# import astropy modules
from astropy import constants as const
from astropy import units as u
from astropy.io import fits, ascii
from astropy.coordinates import EarthLocation, AltAz, get_body, SkyCoord
from astropy.convolution import Gaussian1DKernel
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy.table import QTable, Table, Column, MaskedColumn
from astropy.utils.masked.core import Masked
from astropy.units import Quantity

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# import scipy modules
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy.ndimage import shift, zoom
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.special import comb
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

# import other modules
import time
import copy
import emcee
import os
import pandas as pd
import pyvo as vo
import statsmodels.api as sm
import warnings
from PyAstronomy import pyasl
from tqdm import tqdm
from itertools import combinations
from numba import njit, prange
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore', category=UserWarning, append=True)

# Defining constants
h  = const.h.value # J.s
c  = const.c.value # m/s
kB = const.k_B.value # J/K
rad2arcsec = 180/np.pi*3600 # arcsec  / rad # rad => arcsec
sr2arcsec2 = 4.25e10        # arcsec2 / sr  # steradians => arcsec2


############################# Basic functions #################################


def linear_interpolate(y1, y2, x1, x2, x):
    """Interpolate between y1 and y2 given x1 and x2, targeting x"""
    if x2==x1:
        raise KeyError("Error: x1==x2 = ", x1)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a * x + b


def improved_power_law_extrapolation(x, x0, y0, alpha, rc):
    """ Hybrid model for PSF profile extrapolation : power law + exponential decrease """
    return y0 * (x / x0) ** (-abs(alpha)) * np.exp(-x / rc)


def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def gaussian0(x, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x)/sig, 2.)/2)


def lorentzian(x, x0, L):
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)


def chi2(x, x0, L):
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)


def smoothstep(x, Rc=None, N=10, x_min=None, x_max=None, filtering=True):
    if filtering :
        x_min = 0 ; x_max = 2*Rc
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    if filtering :
        result += result[::-1]
        result = np.abs(result-1)
    return result


def get_logL(flux_observed, flux_model, sigma_l, method="classic"): # see https://github.com/exoAtmospheres/ForMoSA/blob/activ_dev/ForMoSA/nested_sampling/nested_logL_functions.py
    N = len(flux_observed[(~np.isnan(flux_observed))&(~np.isnan(flux_model))&(~np.isnan(sigma_l))])    
    p = 4 # nb de paramètres : T, lg, Vsini, rv
    if method=="classic":
        R = np.nansum( flux_observed*flux_model/sigma_l**2 ) / np.nansum( flux_model**2/sigma_l**2 )
        chi2 = np.nansum(((flux_observed - R*flux_model) / sigma_l )**2)
        logL = - chi2 / 2
    if method=="classic_bis": # same thing as the CCF but weighted with the noise
        flux_observed = flux_observed / np.sqrt(np.nansum(flux_observed**2/sigma_l**2))
        flux_model = flux_model / np.sqrt(np.nansum(flux_model**2/sigma_l**2))
        chi2 = np.nansum(((flux_observed - flux_model) / sigma_l )**2)
        logL = - chi2 / 2
    elif method=="extended":
        R = np.nansum( flux_observed*flux_model/sigma_l**2 ) / np.nansum( flux_model**2/sigma_l**2 )
        chi2 = np.nansum(((flux_observed - R*flux_model) / sigma_l )**2)
        s2 = 1/N * chi2
        logL = - (chi2 / (2*s2) + N/2 * np.log(2*np.pi*s2) + 1/2 * np.log(np.nansum(sigma_l**2)))
    elif method=="extended_bis":
        flux_observed = flux_observed / np.sqrt(np.nansum(flux_observed**2/sigma_l**2))
        flux_model = flux_model / np.sqrt(np.nansum(flux_model**2/sigma_l**2))
        chi2 = np.nansum(((flux_observed - flux_model) / sigma_l )**2)
        s2 = 1/N * chi2
        logL = - (chi2 / (2*s2) + N/2 * np.log(2*np.pi*s2) + 1/2 * np.log(np.nansum(sigma_l**2)))
    elif method=="Brogi":
        flux_observed = flux_observed / np.sqrt(np.nansum(flux_observed**2))
        flux_model = flux_model / np.sqrt(np.nansum(flux_model**2))
        Sf2 = 1/N * np.nansum(flux_observed**2)
        Sg2 = 1/N * np.nansum(flux_model**2)
        R = 1/N * np.nansum(flux_observed * flux_model)
        logL = -N/2 * np.log(Sf2 - 2*R + Sg2)
    elif method=="Zucker":
        flux_observed = flux_observed / np.sqrt(np.nansum(flux_observed**2))
        flux_model = flux_model / np.sqrt(np.nansum(flux_model**2))
        Sf2 = 1/N * np.nansum(flux_observed**2)
        Sg2 = 1/N * np.nansum(flux_model**2)
        R = 1/N * np.nansum(flux_observed * flux_model)
        C2 = (R**2)/(Sf2 * Sg2)
        logL = -N/2 * np.log(1-C2)
    elif method=="custom":
        flux_observed = flux_observed / np.sqrt(np.nansum(flux_observed**2))
        flux_model = flux_model / np.sqrt(np.nansum(flux_model**2))
        Sf2 = 1/N * np.nansum(flux_observed**2)
        Sg2 = 1/N * np.nansum(flux_model**2)
        R = 1/N * np.nansum(flux_observed * flux_model)
        sigma2_weight = 1/(1/N * np.nansum(1/sigma_l**2))
        logL = -N/(2*sigma2_weight) * (Sf2 + Sg2 - 2*R)
    return logL


#################### Image processing/analysis functions ######################


def annular_mask(r_in, r_out, size, value=np.nan):
    """
    Creates a 2D annular mask with value inside the annulus [r_in, r_out],
    and `value` elsewhere.

    Parameters
    ----------
    r_in : float
        Inner radius of the annulus (in pixels).
    r_out : float
        Outer radius of the annulus (in pixels).
    size : tuple of int
        (ny, nx) shape of the output mask.
    value : float or np.nan, optional
        Value to assign outside the annulus.

    Returns
    -------
    mask : 2D ndarray
        Mask with 1s inside the annulus and `value` elsewhere.
    """
    if r_out < r_in:
        raise ValueError("`r_out` must be greater than `r_in`.")
    ny, nx = size
    y0, x0 = ny // 2, nx // 2
    Y, X = np.ogrid[:ny, :nx]
    r2 = (Y - y0)**2 + (X - x0)**2
    mask = np.full((ny, nx), value, dtype=float)
    mask[(r2 >= r_in**2) & (r2 <= r_out**2)] = 1.0
    return mask


def crop(data, Y0=None, X0=None, R_crop=None, return_center=False):
    # Calcul du centre global sur la médiane du cube si non fourni
    if Y0 is None or X0 is None:
        if data.ndim == 3:
            A = np.nanmedian(np.nan_to_num(data), axis=0)
        else:
            A = np.nan_to_num(data)
        Y0, X0 = np.unravel_index(np.argmax(A, axis=None), A.shape)
    # Calcul du rayon de recadrage si non fourni
    if R_crop is None:
        R_crop = max(data.shape[-2] - Y0, data.shape[-1] - X0, Y0, X0)
    # Détermination des limites de recadrage
    y_min = max(0, Y0 - R_crop)
    y_max = min(data.shape[-2], Y0 + R_crop + 1)
    x_min = max(0, X0 - R_crop)
    x_max = min(data.shape[-1], X0 + R_crop + 1)
    # Découpage des indices pour l'image recadrée
    crop_y_min = R_crop - (Y0 - y_min)
    crop_y_max = crop_y_min + (y_max - y_min)
    crop_x_min = R_crop - (X0 - x_min)
    crop_x_max = crop_x_min + (x_max - x_min)
    # Création d'un tableau de sortie initialisé avec des NaN
    if data.ndim == 3:
        B = np.full((data.shape[0], 2 * R_crop + 1, 2 * R_crop + 1), np.nan)
        B[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max] = data[:, y_min:y_max, x_min:x_max]
    else:
        B = np.full((2 * R_crop + 1, 2 * R_crop + 1), np.nan)
        B[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = data[y_min:y_max, x_min:x_max]
    if return_center:
        return B, Y0, X0
    else:
        return B
    
    
def crop_both(data1, data2, Y0=None, X0=None, R_crop=None, return_center=False):
    # Calcul du centre global sur la médiane du cube si non fourni
    if Y0 is None or X0 is None:
        if data1.ndim == 3:
            A = np.nanmedian(np.nan_to_num(data1), axis=0)
        else:
            A = np.nan_to_num(data1)
        Y0, X0 = np.unravel_index(np.argmax(A, axis=None), A.shape)
    # Calcul du rayon de recadrage si non fourni
    if R_crop is None:
        R_crop = max(data1.shape[-2] - Y0, data1.shape[-1] - X0, Y0, X0)
    # Détermination des limites de recadrage
    y_min = max(0, Y0 - R_crop)
    y_max = min(data1.shape[-2], Y0 + R_crop + 1)
    x_min = max(0, X0 - R_crop)
    x_max = min(data1.shape[-1], X0 + R_crop + 1)
    # Découpage des indices pour l'image recadrée
    crop_y_min = R_crop - (Y0 - y_min)
    crop_y_max = crop_y_min + (y_max - y_min)
    crop_x_min = R_crop - (X0 - x_min)
    crop_x_max = crop_x_min + (x_max - x_min)
    # Création des tableaux de sortie initialisés avec des NaN
    if data1.ndim == 3:
        B = np.full((data1.shape[0], 2 * R_crop + 1, 2 * R_crop + 1), np.nan)
        C = np.full((data2.shape[0], 2 * R_crop + 1, 2 * R_crop + 1), np.nan)
        B[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max] = data1[:, y_min:y_max, x_min:x_max]
        C[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max] = data2[:, y_min:y_max, x_min:x_max]
    else:
        B = np.full((2 * R_crop + 1, 2 * R_crop + 1), np.nan)
        C = np.full((2 * R_crop + 1, 2 * R_crop + 1), np.nan)
        B[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = data1[y_min:y_max, x_min:x_max]
        C[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = data2[y_min:y_max, x_min:x_max]
    if return_center:
        return B, C, Y0, X0
    else:
        return B, C


def PSF_profile_ratio(PSF, pxscale, size_core, show=True):
    PSF = np.copy(PSF)
    NbLine, NbColumn = PSF.shape
    y0, x0           = NbLine // 2, NbColumn // 2
    N                = max(y0, x0) + 2*size_core
    PSF_flux = np.nansum(PSF)
    PSF     /= PSF_flux
    if size_core != 1:
        psf = np.zeros_like(PSF) + np.nan 
        for i in range(NbLine):
            for j in range(NbColumn): 
                if not (np.isnan(PSF[max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)]).all()):
                    psf[i, j] = np.nanmean(PSF[max(i-size_core//2, 0):min(i+size_core//2+1, NbLine-1), max(j-size_core//2, 0):min(j+size_core//2+1, NbColumn-1)])
        psf[psf == 0] = np.nan ; PSF = psf
    fraction_core = size_core**2 * PSF[y0, x0]
    profile = np.zeros((2, N))
    for r in range(N):
        if r == 0 or r == 1:
            r_int = r
            r_ext = r
        else:
            r_int = max(1, r - 1)
            r_ext = r
        profile[0, r] = (r_int + r_ext)/2 * pxscale
        profile[1, r] = np.nanmean(PSF * annular_mask(r_int, r_ext, size=(NbLine, NbColumn)))
    profile[1, :] /= pxscale**2
    # # PSF profile sep sanity check
    # NbLine, NbColumn = PSF.shape
    # y0, x0 = NbLine // 2, NbColumn // 2
    # separation = np.zeros((max(y0, x0)+1))
    # sep_mean = np.zeros((max(y0, x0)+1))
    # # Grille de coordonnées
    # Y, X = np.indices(PSF.shape)
    # distances_from_center = np.sqrt((X - x0)**2 + (Y - y0)**2) * pxscale
    # for R in range(max(y0, x0)+1):
    #     if R == 0 or R == 1:
    #         R_int = R
    #         R_ext = R
    #     else:
    #         R_int = max(1, R - 1)
    #         R_ext = R
    #     separation[R] = (R_int + R_ext)/2 * pxscale
    #     PSF_R         = PSF * annular_mask(R_int, R_ext, size=(NbLine, NbColumn))
    #     sep_mean[R] = np.nanmean(distances_from_center[~np.isnan(PSF_R)])
    #     # # Visualisation si besoin
    #     # plt.figure(dpi=300)
    #     # plt.imshow(PSF_R)
    #     # plt.title(f"Annulus R={R}")
    #     # plt.xlim(-10+NbLine/2, 10+NbLine/2)
    #     # plt.ylim(-10+NbLine/2, 10+NbLine/2)
    #     # plt.show()
    # plt.figure(dpi=300)
    # plt.plot(separation-sep_mean)
    # plt.show()
    return profile, fraction_core


def register_PSF_ratio(instru, profile, fraction_core, aper_corr, band, strehl, apodizer, coronagraph=None):
    hdr = fits.Header()
    hdr['FC'] = fraction_core
    hdr['AC'] = aper_corr
    if coronagraph is None:
        fits.writeto("sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_"+strehl+"_"+apodizer+".fits", profile, header=hdr, overwrite=True)
    else:
        fits.writeto("sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_"+coronagraph+"_"+strehl+"_"+apodizer+".fits", profile, header=hdr, overwrite=True)


def fitting_PSF(instru, data, wave, pxscale, size_core, sep_unit="arcsec", model="gaussian", Y0=None, X0=None, sigfactor=5, debug=False):
    import vip_hci as vip
    D        = get_config_data(instru)["telescope"]["diameter"]  # diameter  in m
    lambda0  = (wave[0] + wave[-1]) / 2 * 1e-6  # wavelength in m
    FWHM_ang = lambda0 / D * rad2arcsec  # FWHM [arcsec]
    if sep_unit == "mas":
        FWHM_ang *= 1000  # FWHM [mas]
    FWHM_px = FWHM_ang / pxscale  # FWHM [pixels]
    dpx = size_core

    if len(data.shape) == 2:
        NbLine, NbColumn = data.shape
        if Y0 is None or X0 is None: # First guess
            Y0, X0 = np.unravel_index(np.nanargmax(data, axis=None), data.shape)
        img = np.nan_to_num(data)
        ymin = max(Y0 - 2*dpx  , 0)
        ymax = min(Y0 + 2*dpx+1, NbLine-1)
        xmin = max(X0 - 2*dpx  , 0)
        xmax = min(X0 + 2*dpx+1, NbColumn-1)
        img_cropped = img[ymin:ymax, xmin:xmax]
        if model == "gaussian":
            results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=2*FWHM_px, fwhmy=2*FWHM_px, sigfactor=sigfactor, threshold=True, full_output=True, debug=debug)
            fwhm_y     = results["fwhm_y"][0]
            fwhm_x     = results["fwhm_x"][0]
            fwhm_y_err = results["fwhm_y_err"][0]
            fwhm_x_err = results["fwhm_x_err"][0]
        elif model == "airy_disk":
            results = vip.var.fit_2d.fit_2dairydisk(img_cropped, fwhm=2*FWHM_px, sigfactor=sigfactor, threshold=True, full_output=True, debug=debug)
            fwhm_x     = fwhm_y     = results["fwhm"][0]
            fwhm_x_err = fwhm_y_err = results["fwhm_err"][0]
        elif model == "moffat":
            results = vip.var.fit_2d.fit_2dmoffat(img_cropped, fwhm=2*FWHM_px, sigfactor=sigfactor, threshold=True, full_output=True, debug=debug)
            fwhm_x     = fwhm_y     = results["fwhm"][0]
            fwhm_x_err = fwhm_y_err = results["fwhm_err"][0]
        y0_err  = results["centroid_y_err"][0]
        x0_err  = results["centroid_x_err"][0]
        centroid_y = results["centroid_y"][0]
        centroid_x = results["centroid_x"][0]
        y0 = centroid_y + ymin
        x0 = centroid_x + xmin
        return y0, x0, fwhm_y, fwhm_x
    
    elif len(data.shape) == 3:
        NbChannel, NbLine, NbColumn = data.shape
        if Y0 is None or X0 is None: # First guess
            Y0, X0 = np.unravel_index(np.nanargmax(np.nanmedian(data, axis=0), axis=None), data[0].shape)
        y0         = np.zeros((NbChannel))
        x0         = np.zeros((NbChannel))
        y0_err     = np.zeros((NbChannel))
        x0_err     = np.zeros((NbChannel))
        fwhm_y     = np.zeros((NbChannel))
        fwhm_x     = np.zeros((NbChannel))
        fwhm_y_err = np.zeros((NbChannel))
        fwhm_x_err = np.zeros((NbChannel))
        for i in range(NbChannel):
            img = np.nan_to_num(data[i])
            ymin = max(Y0-2*dpx, 0)
            ymax = min(Y0+2*dpx+1, NbLine-1)
            xmin = max(X0-2*dpx, 0)
            xmax = min(X0+2*dpx+1, NbColumn-1)
            img_cropped = img[ymin:ymax, xmin:xmax]
            if model == "gaussian":
                results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=2*FWHM_px, fwhmy=2*FWHM_px, sigfactor=sigfactor, threshold=True, full_output=True, debug=(i == NbChannel//2) and debug)
                fwhm_y[i]     = results["fwhm_y"][0]
                fwhm_x[i]     = results["fwhm_x"][0]
                fwhm_y_err[i] = results["fwhm_y_err"][0]
                fwhm_x_err[i] = results["fwhm_x_err"][0]
            elif model == "airy_disk":
                results = vip.var.fit_2d.fit_2dairydisk(img_cropped, fwhm=2*FWHM_px, sigfactor=sigfactor, threshold=True, full_output=True, debug=(i == NbChannel//2) and debug)
                fwhm_y[i]     = fwhm_x[i]     = results["fwhm"][0]
                fwhm_y_err[i] = fwhm_x_err[i] = results["fwhm_err"][0]
            elif model == "moffat":
                results = vip.var.fit_2d.fit_2dmoffat(img_cropped, fwhm=2*FWHM_px, sigfactor=sigfactor, threshold=True, full_output=True, debug=(i == NbChannel//2) and debug)
                fwhm_y[i]     = fwhm_x[i]     = results["fwhm"][0]
                fwhm_y_err[i] = fwhm_x_err[i] = results["fwhm_err"][0]
            y0_err[i]  = results["centroid_y_err"][0]
            x0_err[i]  = results["centroid_x_err"][0]
            centroid_y = results["centroid_y"][0]
            centroid_x = results["centroid_x"][0]
            y0[i] = centroid_y + ymin
            x0[i] = centroid_x + xmin
            dy = Y0 - y0[i]
            dx = X0 - x0[i]
        if wave is not None:        
            plt.figure(figsize=(14, 7), dpi=300)
            plt.suptitle("PSF Fitting", fontsize=22, fontweight="bold", color="#333")                
            plt.subplot(1, 2, 1)
            plt.fill_between(wave, y0 - y0_err, y0 + y0_err, color="#1f77b4", alpha=0.3)
            plt.fill_between(wave, x0 - x0_err, x0 + x0_err, color="#ff7f0e", alpha=0.3)
            plt.plot(wave, y0, label=f"y ("r"$\mu$"f" = {round(np.nanmean(y0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(y0_err), 3)} px)", linewidth=2.5, color="#1f77b4")
            plt.plot(wave, x0, label=f"x ("r"$\mu$"f" = {round(np.nanmean(x0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(x0_err), 3)} px)", linewidth=2.5, color="#ff7f0e")
            plt.ylabel("PSF Centroid Position [px]", fontsize=16)
            plt.xlabel("Wavelength [µm]", fontsize=16)
            plt.title("Centroid Position as a Function of Wavelength", fontsize=18, pad=10)
            plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.subplot(1, 2, 2)
            if model == "gaussian":
                plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="#1f77b4", alpha=0.3)
                plt.fill_between(wave, fwhm_x - fwhm_x_err, fwhm_x + fwhm_x_err, color="#ff7f0e", alpha=0.3)
                plt.plot(wave, fwhm_y, label=f"y FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="#1f77b4")
                plt.plot(wave, fwhm_x, label=f"x FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_x), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_x_err), 2)} px)", linewidth=2.5, color="#ff7f0e")
            else:
                plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="black", alpha=0.3)
                plt.plot(wave, fwhm_y, label=f"FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="black")
            plt.ylabel("PSF FWHM [px]", fontsize=16)
            plt.xlabel("Wavelength [µm]", fontsize=16)
            plt.title("FWHM as a Function of Wavelength", fontsize=18, pad=10)
            plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return np.nanmean(y0), np.nanmean(x0), np.nanmean(fwhm_y), np.nanmean(fwhm_x)



def align_HC_bench_psf(cube_desat, cube, model="airy_disk", dpx=10, wave=None):
    """
    Align each slice of the HC bench cubes. 
    """
    import vip_hci as vip
    cube_desat = np.nan_to_num(cube_desat)
    cube       = np.nan_to_num(cube)
    cube_desat_med = np.nanmedian(cube_desat, axis=0)
    Y0, X0         = np.unravel_index(np.argmax(np.nan_to_num(cube_desat_med), axis=None), cube_desat_med.shape)
    aligned_cube       = np.zeros_like(cube)
    aligned_cube_desat = np.zeros_like(cube_desat)
    NbChannel, NbLine, NbColumn = cube_desat.shape
    y0         = np.zeros((NbChannel))
    x0         = np.zeros((NbChannel))
    y0_err     = np.zeros((NbChannel))
    x0_err     = np.zeros((NbChannel))
    fwhm_y     = np.zeros((NbChannel))
    fwhm_x     = np.zeros((NbChannel))
    fwhm_y_err = np.zeros((NbChannel))
    fwhm_x_err = np.zeros((NbChannel))
    for i in range(NbChannel):
        img = cube_desat[i]
        ymin = max(Y0-2*dpx, 0)
        ymax = min(Y0+2*dpx+1, NbLine-1)
        xmin = max(X0-2*dpx, 0)
        xmax = min(X0+2*dpx+1, NbColumn-1)
        img_cropped = img[ymin:ymax, xmin:xmax]
        if model == "gaussian":
            results = vip.var.fit_2d.fit_2dgaussian(img_cropped, fwhmx=2.77, fwhmy=2.69, threshold=True, full_output=True, debug=(i == 0))
            fwhm_y[i]     = results["fwhm_y"][0]
            fwhm_x[i]     = results["fwhm_x"][0]
            fwhm_y_err[i] = results["fwhm_y_err"][0]
            fwhm_x_err[i] = results["fwhm_x_err"][0]
        elif model == "airy_disk":
            results = vip.var.fit_2d.fit_2dairydisk(img_cropped, fwhm=2.81, threshold=True, full_output=True, debug=(i == 0))
            fwhm_y[i]     = results["fwhm"][0]
            fwhm_y_err[i] = results["fwhm_err"][0]
        elif model == "moffat":
            results = vip.var.fit_2d.fit_2dmoffat(img_cropped, fwhm=2.81, threshold=True, full_output=True, debug=(i == 0))
            fwhm_y[i]     = results["fwhm"][0]
            fwhm_y_err[i] = results["fwhm_err"][0]
        y0_err[i]  = results["centroid_y_err"][0]
        x0_err[i]  = results["centroid_x_err"][0]
        centroid_y = results["centroid_y"][0]
        centroid_x = results["centroid_x"][0]
        y0[i] = centroid_y + ymin
        x0[i] = centroid_x + xmin
        dy = Y0 - y0[i]
        dx = X0 - x0[i]
        aligned_cube[i]       = shift(cube[i], shift=(dy, dx), mode='constant', cval=0)
        aligned_cube_desat[i] = shift(cube_desat[i], shift=(dy, dx), mode='constant', cval=0)
    if wave is not None:        
        plt.figure(figsize=(14, 7), dpi=300)
        plt.suptitle("PSF Fitting and Alignment", fontsize=22, fontweight="bold", color="#333")                
        plt.subplot(1, 2, 1)
        plt.fill_between(wave, y0 - y0_err, y0 + y0_err, color="#1f77b4", alpha=0.3)
        plt.fill_between(wave, x0 - x0_err, x0 + x0_err, color="#ff7f0e", alpha=0.3)
        plt.plot(wave, y0, label=f"y ("r"$\mu$"f" = {round(np.nanmean(y0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(y0_err), 3)} px)", linewidth=2.5, color="#1f77b4")
        plt.plot(wave, x0, label=f"x ("r"$\mu$"f" = {round(np.nanmean(x0), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(x0_err), 3)} px)", linewidth=2.5, color="#ff7f0e")
        plt.ylabel("PSF Centroid Position [px]", fontsize=16)
        plt.xlabel("Wavelength [µm]", fontsize=16)
        plt.title("Centroid Position as a Function of Wavelength", fontsize=18, pad=10)
        plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.subplot(1, 2, 2)
        if model == "gaussian":
            plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="#1f77b4", alpha=0.3)
            plt.fill_between(wave, fwhm_x - fwhm_x_err, fwhm_x + fwhm_x_err, color="#ff7f0e", alpha=0.3)
            plt.plot(wave, fwhm_y, label=f"y FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="#1f77b4")
            plt.plot(wave, fwhm_x, label=f"x FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_x), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_x_err), 2)} px)", linewidth=2.5, color="#ff7f0e")
        else:
            plt.fill_between(wave, fwhm_y - fwhm_y_err, fwhm_y + fwhm_y_err, color="black", alpha=0.3)
            plt.plot(wave, fwhm_y, label=f"FWHM ("r"$\mu$"f" = {round(np.nanmean(fwhm_y), 2)} px, "r"$\sigma$"f" = {round(np.nanmean(fwhm_y_err), 2)} px)", linewidth=2.5, color="black")
        plt.ylabel("PSF FWHM [px]", fontsize=16)
        plt.xlabel("Wavelength [µm]", fontsize=16)
        plt.title("FWHM as a Function of Wavelength", fontsize=18, pad=10)
        plt.legend(loc="best", fontsize=14, fancybox=True, shadow=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return aligned_cube_desat, aligned_cube



#################### Spectral processing/analysis functions ###################


def downbin_spec(specHR, lamHR, lamLR, dlam=None):
    """
    from : https://github.com/jlustigy/coronagraph/blob/master/coronagraph/degrade_spec.py
    Re-bin spectum to lower resolution using :py:obj:`scipy.binned_statistic`
    with ``statistic = 'mean'``. This is a "top-hat" convolution.

    Parameters
    ----------
    specHR : array-like
        Spectrum to be degraded
    lamHR : array-like
        High-res wavelength grid
    lamLR : array-like
        Low-res wavelength grid
    dlam : array-like, optional
        Low-res wavelength width grid

    Returns
    -------
    specLR : :py:obj:`numpy.ndarray`
        Low-res spectrum
    """
    if dlam is None:
        ValueError("Please supply dlam in downbin_spec()")
    # Reverse ordering if wl vector is decreasing with index
    if len(lamLR) > 1:
        if lamHR[0] > lamHR[1]:
            lamHI = np.array(lamHR[::-1])
            spec = np.array(specHR[::-1])
        if lamLR[0] > lamLR[1]:
            lamLO = np.array(lamLR[::-1])
            dlamLO = np.array(dlam[::-1])
    # Calculate bin edges
    LRedges = np.hstack([lamLR - 0.5*dlam, lamLR[-1]+0.5*dlam[-1]])
    # Call scipy.stats.binned_statistic()
    specLR = binned_statistic(lamHR, specHR, statistic="mean", bins=LRedges)[0]
    return specLR


def air2vacuum(wavelength):
    """
    Convert wavelength from air to vacuum (wavelength in µm)
    """
    s = 1e4 / (wavelength * 1e4) # self.wavelength in Angstrom
    n =  1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wavelength * n


def PCA_subtraction(Sres, n_comp_sub, y0=None, x0=None, size_core=None, PCA_annular=False, PCA_mask=False, scree_plot=False, PCA_plots=False, PCA_plots_PDF=False, path_PDF=None, wave=None, R=None, pxscale=None):
    """
    Perform PCA subtraction on the input data cube.

    This function applies Principal Component Analysis (PCA) to subtract signal
    components from the input data cube (Sres). Optionally, it masks out regions 
    around a specified planet location before performing the PCA. It can also display 
    plots for the first few PCA components interactively, and if a file path is provided, 
    it saves plots of all PCA components to a PDF (one per page).

    Parameters
    ----------
    Sres : numpy.ndarray
        Input data cube with dimensions (NbChannel, NbColumn, NbLine).
    n_comp_sub : int
        Number of PCA components to subtract. If 0, no PCA subtraction is performed.
    y0 : int, optional
        Y-coordinate of the planet location.
    x0 : int, optional
        X-coordinate of the planet location.
    size_core : int, optional
        Size parameter for masking around the planet (FWHM size in px).
    PCA_annular : bool, optional
        If True, applies PCA on an annular mask around the planet location.
    PCA_mask : bool, optional
        If True, applies a rectangular mask (of size_core x size_core) around the planet location.
    scree_plot : bool, optional
        If True, displays a scree plot of the eigenvalues.
    PCA_plots : bool, optional
        If True, displays plots for up to the first 5 PCA components.
    PCA_plots_PDF : bool, optional
        If True, saves plots of all PCA components into a PDF.
    path_PDF : str, optional
        If provided, saves plots of all PCA components into a PDF at this path.
    wave : numpy.ndarray, optional
        Array of wavelengths used for plotting the PCA component curves.
    R : int or float, optional
        Resolution parameter used for plotting the Power Spectral Density (PSD).
    pxscale : float, optional
        Pixel scale used for plotting the correlation maps.

    Returns
    -------
    Sres_sub : numpy.ndarray
        Data cube after PCA subtraction.
    pca : PCA object or None
        The fitted PCA object if n_comp_sub is not 0; otherwise, None.
    """
    
    if n_comp_sub != 0:
        from sklearn.decomposition import PCA
        NbChannel, NbColumn, NbLine = Sres.shape # Retrieve the shape of the data cube
        pca = PCA(n_components=n_comp_sub)
        Sres_wo_planet = np.copy(Sres) # Create a copy of the input data for masking purposes
        
        # If planet coordinates are provided, apply the masks if specified
        if y0 is not None and x0 is not None:
            if PCA_annular:
                # Calculate the separation from the center to the planet
                planet_sep = int(round(np.sqrt((y0 - NbLine // 2) ** 2 + (x0 - NbColumn // 2) ** 2)))
                # Apply an annular mask with a given core size
                Sres_wo_planet *= annular_mask(max(1, planet_sep - size_core - 1), planet_sep + size_core, value=np.nan, size=(NbLine, NbColumn))
            if PCA_mask:
                # Apply a rectangular mask around the planet location
                Sres_wo_planet[:, y0 - size_core // 2:y0 + size_core // 2 + 1, x0 - size_core // 2:x0 + size_core // 2 + 10] = np.nan 
        
        # Reshape the cube to 2D (pixels x channels) and replace NaNs with 0
        Sres_wo_planet = np.reshape(Sres_wo_planet, (NbChannel, NbColumn * NbLine)).transpose()
        Sres_wo_planet[np.isnan(Sres_wo_planet)] = 0
        
        # Fit the PCA on the masked data
        pca.fit(Sres_wo_planet)
        
        # Prepare the data for subtraction (reshape and replace NaNs)
        Sres_sub = np.reshape(np.copy(Sres), (NbChannel, NbColumn * NbLine)).transpose()
        Sres_sub[np.isnan(Sres_sub)] = 0
        
        # Transform and inverse transform to obtain the PCA model reconstruction
        X = pca.transform(Sres_sub)
        X = pca.inverse_transform(X)
        
        # Subtract the PCA reconstruction from the original data
        Sres_sub = (Sres_sub - X).transpose()
        Sres_sub = np.reshape(Sres_sub, (NbChannel, NbColumn, NbLine))
        Sres_sub[Sres_sub == 0] = np.nan
        
        # ---------------------------
        # PCA plots
        # ---------------------------
        if PCA_plots:
            # Display up to the first 5 components interactively
            Nk = min(n_comp_sub, 5)
            from src.spectrum import Spectrum
            cmap = plt.get_cmap("Spectral", Nk)
            fig, ax = plt.subplots(Nk, 3, figsize=(16, Nk * 3), sharex='col', sharey='col',
                                   layout="constrained", gridspec_kw={'wspace': 0.05, 'hspace': 0}, dpi=300)
            for k in range(Nk):
                # Retrieve the k-th PCA component and replace zeros with NaN for plotting
                pca_comp = pca.components_[k]
                pca_comp[pca_comp == 0] = np.nan
                
                # First column: PCA component curve
                ax[k, 0].plot(wave, pca_comp, c=cmap(k), label=f"$n_k$ = {k+1}")
                ax[k, 0].legend(fontsize=14, loc="upper center")
                if k == Nk - 1:
                    ax[k, 0].set_xlim(wave[0], wave[-1])
                    ax[k, 0].set_xlabel("wavelength (in µm)", fontsize=14)
                ax[k, 0].set_ylabel("modulation (normalized)", fontsize=14)
                ax[k, 0].grid(True)
                
                # Second column: Plot the Power Spectral Density (PSD)
                m_HF_spectrum = Spectrum(wave, pca_comp, R, None)
                m_HF_spectrum.wavelength = m_HF_spectrum.wavelength[~np.isnan(m_HF_spectrum.flux)]
                m_HF_spectrum.flux = m_HF_spectrum.flux[~np.isnan(m_HF_spectrum.flux)]
                res, psd = m_HF_spectrum.get_psd(smooth=0)
                ax[k, 1].plot(res, psd, c=cmap(k))
                if k == Nk - 1:
                    ax[k, 1].set_xlim(10, R)
                    ax[k, 1].set_xlabel("resolution", fontsize=14)
                    ax[k, 1].set_xscale('log')
                    ax[k, 1].set_yscale('log')
                ax[k, 1].set_ylabel("PSD", fontsize=14)
                ax[k, 1].grid(True)
                
                # Third column: Vectorized computation and plot of the correlation map
                numerator = np.nansum(Sres * pca_comp[:, None, None], axis=0)
                denom = np.sqrt(np.nansum(Sres ** 2, axis=0))
                mask = ~np.all(np.isnan(Sres), axis=0)
                CCF = np.full(Sres.shape[1:], np.nan)
                CCF[mask] = np.nan_to_num(numerator[mask]) / denom[mask]
                cax = ax[k, 2].imshow(CCF, extent=[-(CCF.shape[0] + 1) // 2 * pxscale, (CCF.shape[0]) // 2 * pxscale,
                                                    -(CCF.shape[1] - 2) // 2 * pxscale, (CCF.shape[1]) // 2 * pxscale],
                                      zorder=3)
                cbar = fig.colorbar(cax, ax=ax[k, 2], orientation='vertical', shrink=0.8)
                cbar.set_label("correlation", fontsize=14, labelpad=20, rotation=270)
                if k == Nk - 1:
                    ax[k, 2].set_xlabel('x offset (in ")', fontsize=14)
                ax[k, 2].set_ylabel('y offset (in ")', fontsize=14)
                ax[k, 2].grid(True)
            plt.show()
        
        # ---------------------------
        # Save PCA component plots to a PDF
        # ---------------------------
        if PCA_plots_PDF and path_PDF is not None:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf = PdfPages(path_PDF)
            # Create a colormap with n_comp_sub distinct colors
            N = min(n_comp_sub, 100)
            cmap_pdf = get_cmap("Spectral", N)
            from src.spectrum import Spectrum
            for k in tqdm(range(N), desc="Saving PCA components in PDF"):
                pca_comp = pca.components_[k]
                pca_comp[pca_comp == 0] = np.nan
                
                fig, ax = plt.subplots(1, 3, figsize=(16, 3), dpi=100)
                
                # First column: PCA component curve
                ax[0].plot(wave, pca_comp, c=cmap_pdf(k), label=f"$n_k$ = {k+1}")
                ax[0].legend(fontsize=14, loc="upper center")
                ax[0].set_xlim(wave[0], wave[-1])
                ax[0].set_xlabel("wavelength (in µm)", fontsize=14)
                ax[0].set_ylabel("modulation (normalized)", fontsize=14)
                ax[0].grid(True)
                
                # Second column: PSD plot
                m_HF_spectrum = Spectrum(wave, pca_comp, R, None)
                m_HF_spectrum.wavelength = m_HF_spectrum.wavelength[~np.isnan(m_HF_spectrum.flux)]
                m_HF_spectrum.flux = m_HF_spectrum.flux[~np.isnan(m_HF_spectrum.flux)]
                res, psd = m_HF_spectrum.get_psd(smooth=0)
                ax[1].plot(res, psd, c=cmap_pdf(k))
                ax[1].set_xlim(10, R)
                ax[1].set_xlabel("resolution", fontsize=14)
                ax[1].set_xscale('log')
                ax[1].set_yscale('log')
                ax[1].set_ylabel("PSD", fontsize=14)
                ax[1].grid(True)
                
                # Third column: correlation map
                numerator = np.nansum(Sres * pca_comp[:, None, None], axis=0)
                denom = np.sqrt(np.nansum(Sres ** 2, axis=0))
                mask = ~np.all(np.isnan(Sres), axis=0)
                CCF = np.full(Sres.shape[1:], np.nan)
                CCF[mask] = np.nan_to_num(numerator[mask]) / denom[mask]
                cax = ax[2].imshow(CCF, extent=[-(CCF.shape[0] + 1) // 2 * pxscale, (CCF.shape[0]) // 2 * pxscale, -(CCF.shape[1] - 2) // 2 * pxscale, (CCF.shape[1]) // 2 * pxscale], zorder=3)
                cbar = fig.colorbar(cax, ax=ax[2], orientation='vertical', shrink=0.8)
                cbar.set_label("correlation", fontsize=14, labelpad=20, rotation=270)
                ax[2].set_xlabel('x offset (in ")', fontsize=14)
                ax[2].set_ylabel('y offset (in ")', fontsize=14)
                ax[2].grid(True)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
            pdf.close()
            print(f"PCA PDF saved in {path_PDF}")

        # ---------------------------
        # Scree plot (Eigenvalues)
        # ---------------------------
        if scree_plot:
            X = np.reshape(Sres, (NbChannel, NbColumn * NbLine)).transpose()
            X[np.isnan(X)] = 0
            pca = PCA(n_components=n_comp_sub)
            pca.fit_transform(X)
            eigenvalues = pca.explained_variance_
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
            plt.xlabel('Principal Components')
            plt.ylabel('Eigenvalues')
            plt.title('Scree Plot')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
            
    elif n_comp_sub == 0:
        Sres_sub = np.copy(Sres)
        pca      = None
        
    return Sres_sub, pca


def cut_spectral_frequencies(input_flux, R, Rmin, Rmax, filter_type='empirical', show=False, target_name=None, force_new_calc=False):
    """
    Removes spectral fringes from the input flux by setting specific frequency components to zero or applying a smoother filter.

    Parameters:
    input_flux (array-like): The flux values to be filtered, which may contain NaN values.
    R (float): The spectral resolution of the data.
    Rmin (float): The lower bound of the fringe domain in resolution units.
    Rmax (float): The upper bound of the fringe domain in resolution units.
    filter_type (str): The type of smoother filter to apply, either 'gaussian', 'step' or 'empirical'.
    data (bool): In order to estimate the empirical filter response profile from data.

    Returns:
    array-like: The filtered flux with spectral fringes removed or smoothed.
    """
    # Ensure there are no NaN values in the input flux
    valid_data = ~np.isnan(input_flux)
    # Perform FFT on the valid part of the input flux
    fft_values = np.fft.fft(input_flux[valid_data])
    frequencies = np.fft.fftfreq(len(input_flux[valid_data]))
    # Convert frequencies to resolution scale
    res_values = frequencies * R * 2
    if filter_type == 'gaussian': # Apply a Gaussian filter to smoothly reduce the amplitudes in the fringe domain
        sigma = (Rmax - Rmin) / (2 * np.sqrt(2 * np.log(2)))  # assuming FWHM = Rmax - Rmin
        n = 1 # Control the sharpness of the filter, n > 1 for super-gaussian
        gaussian_filter = (1 - np.exp(-0.5 * ((res_values - (Rmin + Rmax) / 2) / sigma) ** (2*n))) / 2 + (1 - np.exp(-0.5 * ((res_values + (Rmin + Rmax) / 2) / sigma) ** (2*n))) / 2
        filter_response = gaussian_filter
    elif filter_type == 'step': # Apply a window filter for a sharp cutoff of the fringe domain
        step_filter = np.ones_like(res_values)
        step_filter[(Rmax > np.abs(res_values)) & (np.abs(res_values) > Rmin)] = 0
        filter_response = step_filter
    elif filter_type == 'empirical': 
        try : # Opening existing filter response profile
            if force_new_calc:
                raise ValueError("force_new_calc = True")
            empirical_res_values, empirical_filter_response = fits.getdata(f"utils/empirical_filter_response/{target_name}_{R}_{Rmin}_{Rmax}_empirical_filter_response.fits")
            f = interp1d(empirical_res_values, empirical_filter_response, bounds_error=False, fill_value="extrapolate")
            filter_response = f(res_values)
        except Exception as e: # Calculating empirical filter response profile (needs to be done once on data!!! and not on models)
            print(f"empirical cut_spectral_frequencies(): {e}")
            # Calcul de la PSD des données
            psd_values = np.abs(fft_values)**2
            # Lissage de la PSD des données
            psd_values_LF = gaussian_filter1d(psd_values, sigma=100)
            # Interpolation de la PSD en masquant le pic des franges
            psd_values_LF_interp = np.copy(psd_values_LF)
            psd_values_LF_interp[(Rmax > np.abs(res_values)) & (np.abs(res_values) > Rmin)] = np.nan
            f = interp1d(res_values[~np.isnan(psd_values_LF_interp)], psd_values_LF_interp[~np.isnan(psd_values_LF_interp)], bounds_error=False, fill_value=np.nan)
            psd_values_LF_interp = f(res_values)
            # Calcul de la forme du pic des franges (psd_values_LF = PSD lissée avec le pic et psd_values_LF_interp PSD lissée sans le pic)
            peak = np.abs(psd_values_LF - psd_values_LF_interp)
            # Normalisation du pic (on s'intéresse uniquement à la forme)
            peak /= np.nanmax(peak)
            # Lissage de la forme du pic trouvé (semble mieux marcher), sigma=400 est une valeur qui fonctionne bien pour les données de Beta Pic c, mais il faudra surement modifier cette valeur pour d'autres données
            if "beta_Pic_c" in target_name:
                sigma = 666
            else:
                sigma = 1000
            filter_response = gaussian_filter1d(1 - peak, sigma=sigma)
            # Sauvegarde du profil trouvé
            empirical_filter_response = np.zeros((2, len(filter_response)))
            empirical_filter_response[0] = res_values ; empirical_filter_response[1] = filter_response
            fits.writeto(f"utils/empirical_filter_response/{target_name}_{R}_{Rmin}_{Rmax}_empirical_filter_response.fits", empirical_filter_response, overwrite=True)
            plt.figure(dpi=300)
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(psd_values), label="PSD des données")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(psd_values_LF), label="PSD lissée (avec le pic)")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(psd_values_LF_interp), label="PSD lissée (sans le pic)")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(np.abs(psd_values_LF - psd_values_LF_interp)), label="Estimation du pic")
            plt.plot(np.fft.fftshift(res_values), np.fft.fftshift(filter_response), label="normalisation, lissage et inversion du pic")
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.xlabel("resolution")
            plt.ylabel("PSD")
            plt.show()
    else:
        raise ValueError("Invalid filter_type. Use 'gaussian', 'lorentzian', or 'step'.")
    if show: # Plot the filter and the effect on FFT values
        plt.figure(dpi=300)
        plt.plot(np.fft.fftshift(res_values), abs(np.fft.fftshift(fft_values))**2, label='Original PSD')
        plt.plot(np.fft.fftshift(res_values), abs(np.fft.fftshift(filter_response))**2, label=f'{filter_type.capitalize()} Filter Response')
        plt.plot(np.fft.fftshift(res_values), abs(np.fft.fftshift(fft_values * filter_response))**2, label='Filtered PSD')
        plt.xscale('log') ; plt.yscale('log')
        plt.xlabel("resolution") ; plt.ylabel("PSD")
        plt.legend()
        plt.show()
    fft_values *= filter_response
    filtered_flux = np.copy(input_flux)
    filtered_flux[valid_data] = np.real(np.fft.ifft(fft_values))
    return filtered_flux


def keep_true_chunks(nan_values, N):
    # Convertir nan_values en une array numpy (si ce n'est pas déjà fait)
    nan_values = np.array(nan_values)
    # Initialisation des listes pour stocker les indices des débuts et fins de séquences de True
    start_indices = []
    end_indices   = []
    # Parcourir l'array pour identifier les séquences de True
    in_chunk = False
    for i in range(len(nan_values)):
        if nan_values[i] and not in_chunk:
            # Début d'une nouvelle séquence de True
            in_chunk = True
            start_idx = i
        elif not nan_values[i] and in_chunk:
            # Fin de la séquence en cours
            in_chunk = False
            if i - start_idx >= N:
                start_indices.append(start_idx)
                end_indices.append(i)
    # Pour la dernière séquence si elle continue jusqu'à la fin de l'array
    if in_chunk and (len(nan_values) - start_idx >= N):
        start_indices.append(start_idx)
        end_indices.append(len(nan_values))
    # Créer une nouvelle array ne contenant que les chunks de True de longueur >= N
    filtered_nan_values = np.zeros_like(nan_values, dtype=bool)
    for start, end in zip(start_indices, end_indices):
        filtered_nan_values[start:end] = True
    return filtered_nan_values


########################### Q-Q plot functions ################################


def qqplot(map_w, show=True):
    list_corr_w = map_w.reshape(map_w.shape) # annular mask CCF = map_w
    list_c_w = list_corr_w[np.isnan(list_corr_w)!=1] # filtrage nan
    # create Q-Q plot with 45-degree line added to plot
    list_cn_w = (list_c_w-np.mean(list_c_w))/np.std(list_c_w) #loi normale centrée (std=1)
    plt.figure(dpi=300) ; ax = plt.gca()
    sm.qqplot(list_cn_w, ax=ax, line='45')
    plt.title('Q-Q plots of the CCF', fontsize = 14) ; plt.ylabel("sample quantiles", fontsize = 14) ; plt.xlabel("theoritical quantiles", fontsize = 14) ; plt.grid(True) ; plt.show()

    
def qqplot2(CCF_map, sep_lim, sep_unit, pxscale, band, target_name):
    NbLine, NbColumn = CCF_map.shape
    map1 = CCF_map*annular_mask(0, int(round(sep_lim/pxscale)), size=(NbLine, NbColumn))
    map1 = map1[~np.isnan(map1)] # filtrage nan
    map1 = (map1-np.mean(map1))/np.std(map1) #loi normale centrée (std=1)
    map2 = CCF_map*annular_mask(int(round(sep_lim/pxscale))+1, max(NbLine//2, NbColumn//2)-3, size=(NbLine, NbColumn))
    map2 = map2[~np.isnan(map2)] # filtrage nan
    map2 = (map2-np.mean(map2))/np.std(map2) #loi normale centrée (std=1)
    
    plt.figure(dpi=300, figsize=(6, 6))
    ax = plt.gca()    
    sm.qqplot(map1, line=None, ax=ax, marker='o', markerfacecolor='royalblue', markeredgecolor='royalblue', alpha=0.6, label=f'sep < {sep_lim} {sep_unit}', lw=1)
    sm.qqplot(map2, line=None, ax=ax, marker='o', markerfacecolor='crimson', markeredgecolor='crimson', alpha=0.6, label=f'sep > {sep_lim} {sep_unit}', lw=1)    
    sm.qqline(ax=ax, line='45', fmt='k--', lw=2)    
    plt.title(f"Q-Q plot of the CCF of {target_name} on {band}", fontsize=15, pad=12)
    plt.xlabel("Theoretical quantiles", fontsize=14)
    plt.ylabel("Sample quantiles", fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    
def qqplot2_hirise(CCF_signal, CCF_bkgd, band, target_name):
    plt.figure(dpi=300) ; ax = plt.gca()
    CCF_signal = CCF_signal[~np.isnan(CCF_signal)] # filtrage nan
    CCF_signal = (CCF_signal-np.mean(CCF_signal))/np.std(CCF_signal) #loi normale centrée (std=1)
    plt.plot([], [], 'o', c="gray", alpha=0.5, label='noise')
    for i in range(len(CCF_bkgd)):
        map_bkgd = CCF_bkgd[i][~np.isnan(CCF_bkgd[i])] # filtrage nan
        map_bkgd = (map_bkgd-np.mean(map_bkgd))/np.std(map_bkgd) # loi normale centrée (normalisée => std=1)
        sm.qqplot(map_bkgd, ax=ax, markerfacecolor='gray', markeredgecolor='gray', alpha=0.1)
    sm.qqline(ax=ax, line='45', fmt='k')
    sm.qqplot(CCF_signal, ax=ax, alpha=1, label='planet')
    plt.legend()
    plt.title(f'Q-Q plots of the CCF of {target_name} on {band}', fontsize = 14) ; plt.ylabel("sample quantiles", fontsize = 14) ; plt.xlabel("theoritical quantiles", fontsize = 14) ; plt.grid(True) ; plt.show()


########################## Extracting data functions ##########################


def extract_jwst_data(instru, target_name, band, crop_band=True, outliers=False, sigma_outliers=5, file=None, X0=None, Y0=None, R_crop=None, verbose=True):
    
    # Instrument specs
    config_data = get_config_data(instru)
    area        = config_data['telescope']['area'] # collective area m2

    # Opening file
    if file is None :
        if instru=="MIRIMRS" : 
            if "sim" in target_name.lower():
                file = f"data/MIRIMRS/MIRISim/{target_name}_{band}_s3d.fits"
            else :
                file = f"data/MIRIMRS/MAST/{target_name}_ch{band[0]}-shortmediumlong_s3d.fits"
        elif instru=="NIRSpec":
            file = f"data/NIRSpec/MAST/{target_name}_nirspec_{band}_s3d.fits"
        else:
            raise KeyError(f"Unknown instrument {instru}")
    f = fits.open(file)
    
    # Retrieving header values
    hdr0 = f[0].header
    hdr  = f[1].header
    if instru=="MIRIMRS" :
        if "sim" in target_name.lower() or "shortmediumlong" not in file: # MIRISIM are already per band
            exposure_time = f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
        elif crop_band:
            target_name   = hdr0['TARGNAME']
            exposure_time = f[0].header['EFFEXPTM']/3/60 # in mn / Effective exposure time
        else:
            target_name   = hdr0['TARGNAME']
            exposure_time = f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
    elif instru == "NIRSpec":
        target_name   = hdr0['TARGNAME']
        exposure_time = f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
    DIT         = f[0].header['EFFINTTM']/60 # in mn
    pxsteradian = hdr['PIXAR_SR'] # Nominal pixel area in steradians
    pxscale     = hdr['CDELT1']*3600 # data pixel scale in "/px
    step        = hdr['CDELT3'] # delta_lambda in µm
    wave        = (np.arange(hdr['NAXIS3'])+hdr['CRPIX3']-1)*hdr['CDELT3']+hdr['CRVAL3'] # axe de longueur d'onde des données en µm

    # Retrieving data
    cube = f[1].data # en MJy/Steradian (densité "angulaire" de flux mesurée dans chaque pixel)
    err  = f[2].data # erreur sur le flux en MJy/Sr
    cube, err = crop_both(cube, err, X0=X0, Y0=Y0, R_crop=R_crop)
    NbChannel, NbLine, NbColumn = cube.shape
    
    # Converting data in total e- (or ph if crop_band) / px
    cube *= pxsteradian*1e6              # MJy/Steradian => Jy/px
    err  *= pxsteradian*1e6
    cube *= 1e-26                        # Jy/pixel => J/s/m²/Hz/px
    err  *= 1e-26
    for i in range(NbChannel):
        cube[i] *= c/((wave[i]*1e-6)**2) # J/s/m²/Hz/px => J/s/m²/m/px
        err[i]  *= c/((wave[i]*1e-6)**2)
    cube *= step*1e-6                    # J/s/m²/m/px => J/s/m²/px
    err  *= step*1e-6
    for i in range(NbChannel):
        cube[i] *= wave[i]*1e-6/(h*c)    # J/s/m²/px => ph/s/m²/px
        err[i] *= wave[i]*1e-6/(h*c)
    cube *= area                         # ph/s/m²/px => photons/s/px
    err  *= area
    if crop_band: 
        lmin = config_data['gratings'][band].lmin
        lmax = config_data['gratings'][band].lmax
        band_mask = (wave >= config_data['gratings'][band].lmin) & (wave <= config_data['gratings'][band].lmax)
        cube = cube[band_mask]
        err  = err[band_mask]
        wave = wave[band_mask]
        NbChannel = cube.shape[0]
        from src.signal_noise_estimate import get_transmission
        trans = get_transmission(instru, wave, band, tellurics=False, apodizer="NO_SP") / fits.getheader("sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_NO_JQ_NO_SP.fits")['AC'] # AC is obviously already taken into account
        for i in range(NbChannel):
            cube[i] *= trans[i]
            err[i]  *= trans[i] # e-/s/pixel
    else:
        trans = 1
    cube *= exposure_time*60
    err  *= exposure_time*60 # e-/px or ph/px
    
    # Cropping first and last empty slices
    valid_slices = np.array([np.any(np.isfinite(cube[i]) & (cube[i] != 0)) for i in range(NbChannel)]) # Identify valid slices (not all zeros or NaNs)    
    start = np.argmax(valid_slices)  # first True -> first valid slice
    end   = len(valid_slices) - np.argmax(valid_slices[::-1])  # last True + 1 -> last valid slice
    wave = wave[start:end] # Crop only the empty slices at the beginning and the end
    cube = cube[start:end]
    err  = err[start:end]
    if crop_band:
        trans = trans[start:end]    
    NbChannel = cube.shape[0] # Update the number of channels

    # Flagging bad pixels
    if instru == "MIRIMRS": # flagging edge effects
        bad_pixels  = np.sum(cube, axis=(0))
        bad_pixels *= 0
    else:
        bad_pixels = np.zeros((NbLine, NbColumn))        
    cube += bad_pixels
    err  += bad_pixels
    cube[cube==0] = np.nan
    err[err==0]   = np.nan
    
    # Flagging outliers
    if outliers :
        NbChannel, NbLine, NbColumn = cube.shape
        Y = np.reshape(cube, (NbChannel, NbLine*NbColumn))
        Z = np.reshape(err, (NbChannel, NbLine*NbColumn))
        for k in range(Y.shape[1]):
            if not all(np.isnan(Y[:, k])):
                sg = sigma_clip(Y[:, k], sigma=sigma_outliers)
                Y[:, k] = np.array(np.ma.masked_array(Y[:, k], mask=sg.mask).filled(np.nan))
                Z[:, k] = np.array(np.ma.masked_array(Z[:, k], mask=sg.mask).filled(np.nan))
                sg = sigma_clip(Z[:, k], sigma=sigma_outliers)
                Y[:, k] = np.array(np.ma.masked_array(Y[:, k], mask=sg.mask).filled(np.nan))
                Z[:, k] = np.array(np.ma.masked_array(Z[:, k], mask=sg.mask).filled(np.nan))
        cube = Y.reshape((NbChannel, NbLine, NbColumn))
        err  = Z.reshape((NbChannel, NbLine, NbColumn))

    if verbose :
        dl = wave - np.roll(wave, 1) ; dl[0] = dl[1] ; dl[dl==0] = np.nanmean(dl) # delta lambda array
        R  = np.nanmean(wave/(2*dl)) # calculating the resolution
        print("\n"+"\033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(wave.min(), 2)} to {round(wave.max(), 2)} µm with R={R:.0f}):"+"\033[0m")
        print(f" Pixel scale  : {pxscale:.2f} arcsec/px")
        print(f" Exposure time: {exposure_time:.1f} mn (with DIT of {DIT*60:.1f} s)") 
        fields = [  ("TITLE", "{}"),
                    ("CATEGORY", "{}"),
                    ("SCICAT", "{}"),
                    ("DATE-OBS", "{}"),
                    ("TARGPROP", "{}"),
                    ("TARGNAME", "{}"),
                    ("TARGTYPE", "{}"),
                    ("TARGCAT", "{}"),
                    ("TARGDESC", "{}"),
                    ("TARG_RA", "{:.2f}"),
                    ("TARG_DEC", "{:.2f}"),
                    ("PATTTYPE", "{}")]
        for key, fmt in fields:
            try:
                print(f" {key:<13}: {fmt.format(hdr0[key])}")
            except KeyError:
                pass

    return cube, wave, pxscale, err, trans, exposure_time, DIT



def extract_vipa_data(instru, target_name, gain, label_fiber, interpolate=True, degrade_resolution=True, outliers=False, sigma_outliers=5, use_weight=True, mask_nan_values=False, R=70000, Rc=100, filter_type="gaussian", wave_input=None, only_high_pass=False):
    from src.spectrum import Spectrum, filtered_flux
    f                                     = fits.open("data/"+instru+"/VIPA_Final_Spectrum_"+target_name+"_gain_"+str(gain)+"_fiber_"+label_fiber+".fits")
    wave0, flux0, sigma0, weight0, trans0 = f[0].data
    header                                = f[0].header
    exposure_time                         = header["INTTIME"] # [s] 
    
    print(f"exposure time = {round(exposure_time/60, 1)} mn")
    
    wave0      *= 1e-3 # nm => µm
    lmin        = wave0[0]
    lmax        = wave0[-1]
    dwl0        = wave0 - np.roll(wave0, 1) ; dwl0[0] = dwl0[1] ; dwl0[dwl0==0] = np.nanmedian(dwl0) # delta lambda array
    R0          = np.nanmedian(wave0/(2*dwl0)) # calculating the resolution => assuming a Nyquist sampling
    dl0         = np.nanmedian(dwl0)
    nan_values0 = np.isnan(flux0)|np.isnan(trans0) # missing values
    
    # (first) OUTLIERS FILTERING (if wanted)
    if outliers:
        trans0_HF       = filtered_flux(trans0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(trans0_HF, sigma=2*sigma_outliers)
        trans0          = np.array(np.ma.masked_array(trans0, mask=sg.mask).filled(np.nan))
        flux0_HF = filtered_flux(flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg       = sigma_clip(flux0_HF, sigma=2*sigma_outliers)
        flux0    = np.array(np.ma.masked_array(flux0, mask=sg.mask).filled(np.nan))

    # For the flux renormalization (conservation needed) at the end 
    norm_flux0  = np.nansum(flux0)
    norm_sigma0 = np.sqrt(np.nansum(sigma0**2))
    
    # Interpolation of the data (should not really degrade data) (if wanted)
    if interpolate: # in order to have a regular wavelength axis
        if wave_input is not None and not degrade_resolution:
            wave = wave_input
        else:
            wave = np.arange(0.98*lmin, 1.02*lmax+dl0, dl0) # constant and regular wavelength array : 0.01 µm ~ doppler shift at few thousands of km/s
        f = interp1d(wave0[~np.isnan(flux0)], flux0[~np.isnan(flux0)], bounds_error=False, fill_value=np.nan)        ; flux0       = f(wave)
        f = interp1d(wave0[~np.isnan(weight0)], weight0[~np.isnan(weight0)], bounds_error=False, fill_value=np.nan)  ; weight0     = f(wave)
        f = interp1d(wave0[~np.isnan(sigma0)], sigma0[~np.isnan(sigma0)], bounds_error=False, fill_value=np.nan)     ; sigma0      = f(wave)
        f = interp1d(wave0[~np.isnan(trans0)], trans0[~np.isnan(trans0)], bounds_error=False, fill_value=np.nan)     ; trans0      = f(wave)
        f = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan)                                      ; nan_values0 = f(wave) != 0
        wave0 = wave

    # Artificially degrating the data to an arbitrary resolution R (if wanted)
    if degrade_resolution:
        dl = np.nanmedian(wave0/(2*R)) # 2*R => Nyquist sampling (Shannon)
        if wave_input is not None:
            wave = wave_input
        else:
            wave = np.arange(0.98*lmin, 1.02*lmax+dl, dl) # new wavelength array (with degrated resolution)
        flux       = Spectrum(wave0, flux0, R0, None)      ; flux   = flux.degrade_resolution(wave, renorm=False, R_output=R).flux
        weight     = Spectrum(wave0, weight0, R0, None)    ; weight = weight.degrade_resolution(wave, renorm=False, R_output=R).flux
        sigma      = Spectrum(wave0, sigma0, R0, None)     ; sigma  = sigma.degrade_resolution(wave, renorm=False, R_output=R).flux
        trans      = Spectrum(wave0, trans0, R0, None)     ; trans  = trans.degrade_resolution(wave, renorm=False, R_output=R).flux
        f          = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan) 
        nan_values = f(wave) != 0

    else: # otherwise, takes the raw data
        R = R0 ; wave = wave0 ; dl = dl0 ; flux = flux0 ; weight = weight0 ; sigma = sigma0 ; trans = trans0 ; nan_values = nan_values0
        
    # MM post-processing
    if only_high_pass:
        flux_HF = trans * filtered_flux(flux/trans, R=R, Rc=Rc, filter_type=filter_type)[0]
            
    # Removing the flagged NaN values
    if mask_nan_values:
        flux[nan_values]    = np.nan
        flux_HF[nan_values] = np.nan
        sigma[nan_values]   = np.nan
        weight[nan_values]  = np.nan
        trans[nan_values]   = np.nan

    # Renormalization (flux and noise conservation)
    valid    = (wave>lmin) & (wave<lmax)
    flux    *= norm_flux0/np.nansum(flux[valid])
    flux_HF *= norm_flux0/np.nansum(flux[valid])
    sigma   *= norm_sigma0/np.sqrt(np.nansum(sigma[valid]**2)) # for the noise, this is the power (total variance) that is conserved
    
    # (second final) OUTLIERS FILTERING (if wanted)
    if outliers: 
        sg      = sigma_clip(flux_HF, sigma=sigma_outliers)
        flux_HF = np.array(np.ma.masked_array(flux_HF, mask=sg.mask).filled(np.nan))
    
    # Weight function (if wanted)
    if use_weight:
        if 1==1:
            weight /= sigma
        weight /= np.nanmax(weight)
        sigma  *= weight # since the signals will be multiplied by the weight, the noise needs also to be multiplied by it
    else:
        weight = None
        
    return wave, flux, flux_HF, sigma, weight, trans, R


def extract_hirise_data(target_name, interpolate, degrade_resolution, R, Rc, filter_type, order_by_order, outliers, sigma_outliers, only_high_pass=False, cut_fringes=False, Rmin=None, Rmax=None, use_weight=True, mask_nan_values=False, keep_only_good=False, wave_input=None, reference_fibers=True, crop_tell_orders=False, verbose=True): # OPENING DATA AND DEGRADATING THE RESOLUTION (if wanted)
    from src.spectrum import Spectrum, filtered_flux
    
    # hard coded values
    GAIN     = np.nanmean([2.28, 2.19, 2.00]) # in e-/ADU
    noffsets = 1
    nrefs    = 3
    
    # OPENING DATA 
    file       = "data/HiRISE/"+target_name+".fits"
    f          = fits.open(file)
    hdr        = f[0].header
    T_star     = hdr["HIERARCH STAR TEFF"] 
    lg_star    = hdr["HIERARCH STAR LOGG"]
    rv_star    = hdr["HIERARCH STAR RV"] - hdr["HIERARCH STAR HELCORR MEAN"]
    vsini_star = hdr["HIERARCH STAR VSINI"]
    t_exp_comp = hdr["HIERARCH COMP DIT"]*hdr["HIERARCH COMP NEXP"]
    t_exp_star = hdr["HIERARCH STAR DIT"]*hdr["HIERARCH STAR NEXP"]
    if verbose:
        print("OBSERVATION DATE: ", mjd_to_date(hdr["MJD-OBS"]))
        print(f" star t_exp = {round(t_exp_star/60)} mn")
        print(f" comp t_exp = {round(t_exp_comp/60)} mn")
        print(f" DELTA RV HELCORR = ", round(abs(hdr["HIERARCH STAR HELCORR MEAN"] - hdr["HIERARCH COMP HELCORR MEAN"]), 3), "km/s")

    bkg_flux0 = []
    wave0     = f[1].data["pipeline"]*1e-3     # in µm
    wave0     = f[1].data["recalibrated"]*1e-3 # in µm
    #wave0     = f[7].data["wave"]*1e-3         # in µm
    for ioff in range(noffsets):
        # Response data
        data_response = f[f.index_of(f'RESPONSE,OFFSET{ioff}')].data
        trans0        = data_response["response"]       # no unit
        trans_model0  = data_response["response_model"] # no unit
        # Star data
        data_star       = f[f.index_of(f'STAR,OFFSET{ioff},SCI')].data
        star_flux0      = data_star["signal"]                        # in e-
        star_wave0      = data_star["wave"]*1e-3                     # in µm
        star_weight0    = data_star["weight"]                        # no unit
        star_sigma_tot0 = data_star["noise"]                         # e-
        star_sigma_bkg0 = data_star["noise_background"]              # e-
        #star_sigma_bkg0 = np.sqrt(star_sigma_tot0**2 - star_flux0)   # e-
        for iref in range(nrefs):
            data_star_ref = f[f.index_of(f'STAR,OFFSET{ioff},REF{iref}')].data
            bkg_flux0.append(data_star_ref["signal"])             # e-
        # Comp data
        data_planet       = f[f.index_of(f'COMP,OFFSET{ioff},SCI')].data
        planet_flux0      = data_planet["signal"]                          # in e-
        planet_wave0      = data_planet["wave"]*1e-3                       # in µm
        planet_weight0    = data_planet["weight"]                          # no unit
        planet_sigma_tot0 = data_planet["noise"]                           # e-
        planet_sigma_bkg0 = data_planet["noise_background"]                # e-
        #planet_sigma_bkg0 = np.sqrt(planet_sigma_tot0**2 - planet_flux0)   # e-
        for iref in range(nrefs):
            data_planet_ref = f[f.index_of(f'COMP,OFFSET{ioff},REF{iref}')].data
            bkg_flux0.append(data_planet_ref["signal"]) # e-
                
    wave_raw = wave0
        
    # sky_transmission_path = os.path.join("sim_data/Transmission/sky_transmission_airmass_1.0.fits")
    # sky_trans             = fits.getdata(sky_transmission_path)
    # trans_tell_band       = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
    # planet_flux0          = trans_tell_band.interpolate_wavelength(wave0, renorm=False).flux # degraded tellurics transmission on the considered band
    # #planet_flux0          = trans_model0

    # calib_name = "HD_26820"
    # wave0      = fits.open("data/HiRISE/"+calib_name+".fits")[1].data["recalibrated"]*1e-3      # in µm

    # wavelength axis properties
    if crop_tell_orders:
        lmin = 1.505/0.98
        lmax = 1.73/1.02 # Cropping two firsts and one last order
    else:
        lmin = wave0[0]
        lmax = wave0[-1]
    dl0  = wave0 - np.roll(wave0, 1) ; dl0[0] = dl0[1] ; dl0[dl0==0] = np.nanmedian(dl0) # delta lambda array
    R0   = np.nanmedian(wave0/(2*dl0)) # calculating the resolution => assuming a Nyquist sampling
    dl0  = np.nanmedian(dl0)
    
    # Flagging NaN values
    nan_values0 = np.isnan(star_flux0)|np.isnan(planet_flux0)|np.isnan(trans0) # missing values
        
    # flagging the orders limits (for "order_by_order" filtering method)      
    if order_by_order:          
        transitions = np.where(np.diff(wave0) > 1000 * dl0)[0] + 1  # Indexes where the order changes
        lmin_orders = wave0[transitions - 1]
        lmax_orders = wave0[transitions]
    
    # (first) OUTLIERS FILTERING (if wanted)
    if outliers:
        trans0_HF       = filtered_flux(trans0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(trans0_HF, sigma=2*sigma_outliers)
        trans0          = np.array(np.ma.masked_array(trans0, mask=sg.mask).filled(np.nan))
        trans_model0_HF = filtered_flux(trans_model0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(trans_model0_HF, sigma=2*sigma_outliers)
        trans_model0    = np.array(np.ma.masked_array(trans_model0, mask=sg.mask).filled(np.nan))
        star_flux0_HF   = filtered_flux(star_flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(star_flux0_HF, sigma=2*sigma_outliers)
        star_flux0      = np.array(np.ma.masked_array(star_flux0, mask=sg.mask).filled(np.nan))
        planet_flux0_HF = filtered_flux(planet_flux0, R=R0, Rc=Rc, filter_type=filter_type)[0]
        sg              = sigma_clip(planet_flux0_HF, sigma=2*sigma_outliers)
        planet_flux0    = np.array(np.ma.masked_array(planet_flux0, mask=sg.mask).filled(np.nan))
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                bkg_flux0_HF = filtered_flux(bkg_flux0[i], R=R0, Rc=Rc, filter_type=filter_type)[0]
                sg           = sigma_clip(bkg_flux0_HF, sigma=2*sigma_outliers)
                bkg_flux0[i] = np.array(np.ma.masked_array(bkg_flux0[i], mask=sg.mask).filled(np.nan))
        
    # For the flux renormalization (conservation needed) at the end 
    norm_star_flux0        = np.nansum(star_flux0)
    norm_star_sigma_tot0   = np.sqrt(np.nansum(star_sigma_tot0**2))
    norm_star_sigma_bkg0   = np.sqrt(np.nansum(star_sigma_bkg0**2))
    norm_planet_flux0      = np.nansum(planet_flux0)
    norm_planet_sigma_tot0 = np.sqrt(np.nansum(planet_sigma_tot0**2))
    norm_planet_sigma_bkg0 = np.sqrt(np.nansum(planet_sigma_bkg0**2))
    if reference_fibers:
        norm_bkg_flux0 = []
        for i in range(len(bkg_flux0)):
            norm_bkg_flux0.append(np.nansum(bkg_flux0[i]))
        
    # Interpolation of the data (should not really degrade data) (if wanted)
    if interpolate: # in order to have a regular wavelength axis
        # new wavelength axis
        if wave_input is not None and not degrade_resolution:
            wave = wave_input
        else:
            wave = np.arange(0.98*lmin, 1.02*lmax+dl0, dl0) # constant and regular wavelength array : 0.01 µm ~ doppler shift at few thousands of km/s
        # star data interpolation
        f = interp1d(wave0[~np.isnan(star_flux0)], star_flux0[~np.isnan(star_flux0)], bounds_error=False, fill_value=np.nan)                      ; star_flux0        = f(wave)
        f = interp1d(wave0[~np.isnan(star_weight0)], star_weight0[~np.isnan(star_weight0)], bounds_error=False, fill_value=np.nan)                ; star_weight0      = f(wave)
        f = interp1d(wave0[~np.isnan(star_sigma_tot0)], star_sigma_tot0[~np.isnan(star_sigma_tot0)], bounds_error=False, fill_value=np.nan)       ; star_sigma_tot0   = f(wave)
        f = interp1d(wave0[~np.isnan(star_sigma_bkg0)], star_sigma_bkg0[~np.isnan(star_sigma_bkg0)], bounds_error=False, fill_value=np.nan)       ; star_sigma_bkg0   = f(wave)
        # planet data interpolation 
        f = interp1d(wave0[~np.isnan(planet_flux0)], planet_flux0[~np.isnan(planet_flux0)], bounds_error=False, fill_value=np.nan)                ; planet_flux0      = f(wave)
        f = interp1d(wave0[~np.isnan(planet_weight0)], planet_weight0[~np.isnan(planet_weight0)], bounds_error=False, fill_value=np.nan)          ; planet_weight0    = f(wave)
        f = interp1d(wave0[~np.isnan(planet_sigma_tot0)], planet_sigma_tot0[~np.isnan(planet_sigma_tot0)], bounds_error=False, fill_value=np.nan) ; planet_sigma_tot0 = f(wave)
        f = interp1d(wave0[~np.isnan(planet_sigma_bkg0)], planet_sigma_bkg0[~np.isnan(planet_sigma_bkg0)], bounds_error=False, fill_value=np.nan) ; planet_sigma_bkg0 = f(wave)
        # trans data interpolation
        f = interp1d(wave0[~np.isnan(trans0)], trans0[~np.isnan(trans0)], bounds_error=False, fill_value=np.nan)                                  ; trans0            = f(wave)
        f = interp1d(wave0[~np.isnan(trans_model0)], trans_model0[~np.isnan(trans_model0)], bounds_error=False, fill_value=np.nan)                ; trans_model0      = f(wave)
        # NaN values interpolation
        f = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan)                                                                   ; nan_values0       = f(wave) != 0
        # bkg data interpolation
        if reference_fibers:
            for i in range(len(bkg_flux0)):
                f = interp1d(wave0[~np.isnan(bkg_flux0[i])], bkg_flux0[i][~np.isnan(bkg_flux0[i])], bounds_error=False, fill_value=np.nan)        ; bkg_flux0[i]      = f(wave)
        # new wavelength axis
        wave0 = wave

    # Artificially degrating the data to an arbitrary resolution R (if wanted)
    if degrade_resolution:
        # new wavelength axis
        if wave_input is not None:
            wave = wave_input
        else:
            dl   = np.nanmedian(wave0/(2*R))              # 2*R => Nyquist sampling (Shannon)
            wave = np.arange(0.98*lmin, 1.02*lmax+dl, dl) # new wavelength array (with degrated resolution)
        # Degrade star data
        star_flux        = Spectrum(wave0, star_flux0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        star_weight      = Spectrum(wave0, star_weight0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        star_sigma_tot   = Spectrum(wave0, star_sigma_tot0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        star_sigma_bkg   = Spectrum(wave0, star_sigma_bkg0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        # Degrade planet data
        planet_flux      = Spectrum(wave0, planet_flux0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        planet_weight    = Spectrum(wave0, planet_weight0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        planet_sigma_tot = Spectrum(wave0, planet_sigma_tot0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        planet_sigma_bkg = Spectrum(wave0, planet_sigma_bkg0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        # Degrade trans data
        trans            = Spectrum(wave0, trans0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        trans_model      = Spectrum(wave0, trans_model0, R0, None).degrade_resolution(wave, renorm=False, R_output=R).flux
        # Interpolate star data
        f = interp1d(wave0, nan_values0, bounds_error=False, fill_value=np.nan) ; nan_values = f(wave) != 0
        # Degrade bkg data
        if reference_fibers:
            bkg_flux = [0] * len(bkg_flux0)
            for i in range(len(bkg_flux0)):
                bkg_flux[i] = Spectrum(wave0, bkg_flux0[i], R0, None).degrade_resolution(wave, renorm=False).flux
            
    else: # otherwise, takes the raw data
        R = R0 ; wave = wave0 ; dl = dl0 ; star_flux = star_flux0 ; star_weight = star_weight0 ; star_sigma_tot = star_sigma_tot0 ; star_sigma_bkg = star_sigma_bkg0 ; planet_flux = planet_flux0 ; planet_weight = planet_weight0 ; planet_sigma_tot = planet_sigma_tot0 ; planet_sigma_bkg = planet_sigma_bkg0 ; trans = trans0 ; trans_model = trans_model0 ; nan_values = nan_values0 ; bkg_flux = bkg_flux0
        
    # HIGH PASS FILTERING
    if order_by_order:
        star_flux_HF, star_flux_LF     = np.full_like(wave, np.nan), np.full_like(wave, np.nan)
        planet_flux_HF, planet_flux_LF = np.full_like(wave, np.nan), np.full_like(wave, np.nan)
        sf_HF, sf_LF                   = np.full_like(wave, np.nan), np.full_like(wave, np.nan)
        # Process each spectral order separately
        for i in range(len(lmin_orders) + 1):
            if i == 0:                  # First order
                mask = wave < lmin_orders[i]
            elif i == len(lmin_orders): # Last order
                mask = wave > lmax_orders[i - 1]
            else:                       # Intermediate orders
                mask = (wave > lmax_orders[i - 1]) & (wave < lmin_orders[i])
            if np.any(mask): # Apply filtering if the mask is not empty
                star_flux_HF[mask], star_flux_LF[mask]     = filtered_flux(star_flux[mask], R=R, Rc=Rc, filter_type=filter_type)
                planet_flux_HF[mask], planet_flux_LF[mask] = filtered_flux(planet_flux[mask], R=R, Rc=Rc, filter_type=filter_type)
                sf_HF[mask], sf_LF[mask]                   = filtered_flux(star_flux[mask]/trans[mask], R=R, Rc=Rc, filter_type=filter_type)

    else:
        # Handling LF filtering edge effects due to the gaps bewteen the orders
        star_flux_HF, star_flux_LF     = filtered_flux(star_flux, R=R, Rc=Rc, filter_type=filter_type)
        planet_flux_HF, planet_flux_LF = filtered_flux(planet_flux, R=R, Rc=Rc, filter_type=filter_type)
        _, trans_LF                    = filtered_flux(trans, R=R, Rc=Rc, filter_type=filter_type)
        f              = interp1d(wave[~nan_values], star_flux_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
        star_flux_LF   = f(wave)
        f              = interp1d(wave[~nan_values], planet_flux_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
        planet_flux_LF = f(wave)
        f              = interp1d(wave[~nan_values], trans_LF[~nan_values], bounds_error=False, fill_value=np.nan) 
        trans_LF       = f(wave)
        # masking inter order regions
        NV              = keep_true_chunks(nan_values, N=0.005/np.nanmean(np.diff(wave))) # 0.005 µm ~ size of the gap between orders
        star_flux[NV]   = star_flux_LF[NV]
        planet_flux[NV] = planet_flux_LF[NV]
        trans[NV]       = trans_LF[NV]
        # HF / LF calculations
        star_flux_HF, star_flux_LF     = filtered_flux(star_flux, R=R, Rc=Rc, filter_type=filter_type)
        planet_flux_HF, planet_flux_LF = filtered_flux(planet_flux, R=R, Rc=Rc, filter_type=filter_type)
        sf_HF, sf_LF                   = filtered_flux(star_flux/trans, R=R, Rc=Rc, filter_type=filter_type)

    if reference_fibers:
        bkg_flux_HF = [0] * len(bkg_flux)
        bkg_flux_LF = [0] * len(bkg_flux)
        for i in range(len(bkg_flux)):
            bkg_flux_HF[i], bkg_flux_LF[i] = filtered_flux(bkg_flux[i], R=R, Rc=Rc, filter_type=filter_type)
        
    # Only apply a high pass filter to the data (no stellar subtraction) (if wanted)
    if only_high_pass: 
        planet_flux_HF, planet_flux_LF = filtered_flux(planet_flux/trans, R=R, Rc=Rc, filter_type=filter_type)
        d_planet                       = trans*planet_flux_HF
        if reference_fibers:
            bkg_flux_HF = [0] * len(bkg_flux)
            bkg_flux_LF = [0] * len(bkg_flux)
            d_bkg       = [0] * len(bkg_flux)
            for i in range(len(bkg_flux)):
                bkg_flux_HF[i], bkg_flux_LF[i] = filtered_flux(bkg_flux[i]/trans, R=R, Rc=Rc, filter_type=filter_type)
                d_bkg[i]                       = trans*bkg_flux_HF[i]
                
    # Standard molecular mapping post-processing
    else:
        d_planet = planet_flux - star_flux * planet_flux_LF / star_flux_LF # high pass planet spectrum extracted = trans*[Sp]_HF
        if reference_fibers:
            d_bkg = [0] * len(bkg_flux)
            for i in range(len(bkg_flux)):
                d_bkg[i] = bkg_flux[i] - star_flux * bkg_flux_LF[i] / star_flux_LF
                
    # Star high-pass filtered data
    d_star       = trans*sf_HF # Considered as noise / background flux
        
    # Removing the flagged NaN values
    if mask_nan_values:
        mask = nan_values
    else: # If not masking raw NaN values, it is still needed to mask the gaps between the orders
        mask = keep_true_chunks(nan_values, N=50)
    if keep_only_good: # keeping only very good data (i.e. with weight == 1)
        mask = mask|(planet_weight<1)
    trans[mask]            = np.nan
    star_flux[mask]        = np.nan
    star_flux_LF[mask]     = np.nan
    star_flux_HF[mask]     = np.nan
    star_weight[mask]      = np.nan 
    star_sigma_tot[mask]   = np.nan
    star_sigma_bkg[mask]   = np.nan
    d_star[mask]           = np.nan
    planet_flux[mask]      = np.nan
    planet_flux_LF[mask]   = np.nan
    planet_flux_HF[mask]   = np.nan
    planet_weight[mask]    = np.nan
    planet_sigma_tot[mask] = np.nan
    planet_sigma_bkg[mask] = np.nan
    d_planet[mask]         = np.nan
    if reference_fibers:
        for i in range(len(d_bkg)):
            d_bkg[i][mask]         = np.nan
        
    # Renormalization (flux and noise conservation)
    valid = (wave>lmin) & (wave<lmax)
    d_star         *= norm_star_flux0/np.nansum(star_flux[valid])
    star_flux_LF   *= norm_star_flux0/np.nansum(star_flux[valid])
    star_flux_HF   *= norm_star_flux0/np.nansum(star_flux[valid])
    star_flux      *= norm_star_flux0/np.nansum(star_flux[valid])
    star_sigma_tot *= norm_star_sigma_tot0/np.sqrt(np.nansum(star_sigma_tot[valid]**2)) # for the noise, this is the power (total variance) that is conserved
    star_sigma_bkg *= norm_star_sigma_bkg0/np.sqrt(np.nansum(star_sigma_bkg[valid]**2)) # for the noise, this is the power (total variance) that is conserved
    d_planet         *= norm_planet_flux0/np.nansum(planet_flux[valid])
    planet_flux_LF   *= norm_planet_flux0/np.nansum(planet_flux[valid])
    planet_flux_HF   *= norm_planet_flux0/np.nansum(planet_flux[valid])
    planet_flux      *= norm_planet_flux0/np.nansum(planet_flux[valid])
    planet_sigma_tot *= norm_planet_sigma_tot0/np.sqrt(np.nansum(planet_sigma_tot[valid]**2)) # for the noise, this is the power (total variance) that is conserved
    planet_sigma_bkg *= norm_planet_sigma_bkg0/np.sqrt(np.nansum(planet_sigma_bkg[valid]**2)) # for the noise, this is the power (total variance) that is conserved
    if reference_fibers:
        for i in range(len(d_bkg)):
            d_bkg[i] *= norm_bkg_flux0[i]/np.nansum(bkg_flux[i][valid])
    
    # (second final) OUTLIERS FILTERING (if wanted)
    if outliers: 
        sg             = sigma_clip(d_star, sigma=sigma_outliers)
        d_star         = np.array(np.ma.masked_array(d_star, mask=sg.mask).filled(np.nan))
        sg             = sigma_clip(star_sigma_tot, sigma=sigma_outliers)
        star_sigma_tot = np.array(np.ma.masked_array(star_sigma_tot, mask=sg.mask).filled(np.nan))
        sg             = sigma_clip(star_sigma_bkg, sigma=sigma_outliers)
        star_sigma_bkg = np.array(np.ma.masked_array(star_sigma_bkg, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(d_planet, sigma=sigma_outliers)
        d_planet         = np.array(np.ma.masked_array(d_planet, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(planet_sigma_tot, sigma=sigma_outliers)
        planet_sigma_tot = np.array(np.ma.masked_array(planet_sigma_tot, mask=sg.mask).filled(np.nan))
        sg               = sigma_clip(planet_sigma_bkg, sigma=sigma_outliers)
        planet_sigma_bkg = np.array(np.ma.masked_array(planet_sigma_bkg, mask=sg.mask).filled(np.nan))
        if reference_fibers:
            for i in range(len(d_bkg)):
                sg       = sigma_clip(d_bkg[i], sigma=sigma_outliers)
                d_bkg[i] = np.array(np.ma.masked_array(d_bkg[i], mask=sg.mask).filled(np.nan))

    # Plots
    if verbose and "fiber" not in target_name:
        # Plot of the filtered data
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), dpi=300, sharex=True)        
        axes[0].set_title("Companion's Signal", fontsize=14, fontweight="bold")
        axes[0].plot(wave, planet_flux, "r-", linewidth=2, label="LF+HF (Total Flux)")
        axes[0].plot(wave, planet_flux_LF, "g-", linewidth=2, label="LF (Low-Frequency)")
        axes[0].plot(wave, planet_flux_HF, "b-", linewidth=2, label="HF (High-Frequency)")
        axes[0].plot(wave, filtered_flux(planet_flux_HF, R=R, Rc=Rc, filter_type=filter_type)[1], "k:", linewidth=2, label="[HF]_LF (Filtered HF)")
        axes[0].set_ylabel("Flux", fontsize=12)
        axes[0].legend(fontsize=10, loc="best", frameon=True)
        axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axes[1].set_title("Star's Signal", fontsize=14, fontweight="bold")
        axes[1].plot(wave, star_flux, "r-", linewidth=2, label="LF+HF (Total Flux)")
        axes[1].plot(wave, star_flux_LF, "g-", linewidth=2, label="LF (Low-Frequency)")
        axes[1].plot(wave, star_flux_HF, "b-", linewidth=2, label="HF (High-Frequency)")
        axes[1].plot(wave, filtered_flux(star_flux_HF, R=R, Rc=Rc, filter_type=filter_type)[1], "k:", linewidth=2, label="[HF]_LF (Filtered HF)")        
        axes[1].set_xlabel("Wavelength [µm]", fontsize=12)
        axes[1].set_ylabel("Flux", fontsize=12)
        axes[1].legend(fontsize=10, loc="best", frameon=True)
        axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)        
        fig.suptitle("Comparison of Companion & Star Signals", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()
        
        # Plot of the noise budget
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), dpi=300, sharex=True)    
        axes[0].plot(wave, planet_flux, color="gray", linestyle="-", linewidth=2, alpha=0.8, label=r"$S$ (Signal)")
        axes[0].plot(wave, np.sqrt(planet_flux), color="red", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{ph}$ (Photon Noise)")
        axes[0].plot(wave, planet_sigma_bkg, color="blue", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{bkg}$ (Background Noise)")
        axes[0].plot(wave, planet_sigma_tot, color="green", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{total}$ (Total Noise)")
        axes[0].set_yscale('log')
        axes[0].set_ylabel("Signal [e-]", fontsize=14)
        axes[0].set_title("Planet - Spectral Signal and Noise Components", fontsize=16, fontweight="bold")
        axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        axes[0].legend(fontsize=12, loc="best", frameon=True)
        axes[1].plot(wave, star_flux, color="gray", linestyle="-", linewidth=2, alpha=0.8, label=r"$S$ (Signal)")
        axes[1].plot(wave, np.sqrt(star_flux), color="red", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{ph}$ (Photon Noise)")
        axes[1].plot(wave, star_sigma_bkg, color="blue", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{bkg}$ (Background Noise)")
        axes[1].plot(wave, star_sigma_tot, color="green", linestyle="-", linewidth=2, alpha=0.8, label=r"$\sigma_{total}$ (Total Noise)")
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Wavelength [µm]", fontsize=14)
        axes[1].set_ylabel("Signal [e-]", fontsize=14)
        axes[1].set_title("Star - Spectral Signal and Noise Components", fontsize=16, fontweight="bold")
        axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        axes[1].legend(fontsize=12, loc="best", frameon=True)    
        plt.tight_layout()    
        plt.show()
        
    # NOISE ESTIMATION
    planet_sigma = np.sqrt( planet_sigma_tot**2 + star_sigma_tot**2 * (planet_flux_LF/star_flux_LF)**2 ) 
    
    # Weight function (if wanted)
    if use_weight:
        planet_weight  = (planet_weight + star_weight) / 2
        planet_weight /= np.nanmax(planet_weight)
        planet_sigma  *= planet_weight # since the signals will be multiplied by the weight, the noise needs also to be multiplied by it
    else:
        planet_weight = None
        
    # Filtering fringes frequencies (if wanted)
    if cut_fringes:
        if "fiber" not in target_name:
            d_planet = cut_spectral_frequencies(input_flux=d_planet, R=R, Rmin=Rmin, Rmax=Rmax, show=verbose, target_name=target_name, force_new_calc=True)
        else:
            d_planet = cut_spectral_frequencies(input_flux=d_planet, R=R, Rmin=Rmin, Rmax=Rmax, show=False, target_name=target_name[:-7], force_new_calc=False)
    
    d_bkg.append(d_star)
    
    return wave, star_flux, d_star, planet_flux, d_planet, trans, trans_model, R, planet_sigma, planet_weight, T_star, lg_star, rv_star, vsini_star, hdr, d_bkg, wave_raw


############################## Other functions ################################


def mjd_to_date(mjd):
    t = Time(mjd, format='mjd')
    date_str = t.datetime.strftime("%d/%m/%Y")
    return date_str


def propagate_coordinates_at_epoch(targetname, date, verbose=True):
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord, Distance
    from astropy.time import Time
    """Get coordinates at an epoch for some target, taking into account proper motions.
    Retrieves the SIMBAD coordinates, applies proper motion, returns the result as an
    astropy coordinates object
    from : https://github.com/jruffio/breads/blob/main/breads/utils.py
    """
    # Configure Simbad query to retrieve some extra fields
    if 'pmra' not in Simbad._VOTABLE_FIELDS:
        Simbad.add_votable_fields("pmra")  # Retrieve proper motion in RA
    if 'pmdec' not in Simbad._VOTABLE_FIELDS:
        Simbad.add_votable_fields("pmdec")  # Retrieve proper motion in Dec.
    if 'plx' not in Simbad._VOTABLE_FIELDS:
        Simbad.add_votable_fields("plx")  # Retrieve parallax
    if verbose:
        print(f"Retrieving SIMBAD coordinates for {targetname}")
    result_table = Simbad.query_object(targetname)
    # Get the coordinates and proper motion from the result table
    ra = result_table["RA"][0]
    dec = result_table["DEC"][0]
    pm_ra = result_table["PMRA"][0]
    pm_dec = result_table["PMDEC"][0]
    plx = result_table["PLX_VALUE"][0]
    # Create a SkyCoord object with the coordinates and proper motion
    target_coord_j2000 = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), pm_ra_cosdec=pm_ra * u.mas / u.year, pm_dec=pm_dec * u.mas / u.year, distance=Distance(parallax=plx * u.mas), frame='icrs', obstime='J2000.0')
    # Convert the desired date to an astropy Time object
    t = Time(date)
    # Calculate the updated SkyCoord object for the desired date
    host_coord_at_date = target_coord_j2000.apply_space_motion(new_obstime=t)
    if verbose:
        print(f"Coordinates at J2000:  {target_coord_j2000.icrs.to_string('hmsdms')}")
        print(f"Coordinates at {date}:  {host_coord_at_date.icrs.to_string('hmsdms')}")
    return host_coord_at_date


def get_coordinates_arrays(filename) :
    """ Determine the relative coordinates in the focal plane relative to the target.
        Compute the coordinates {wavelen, delta_ra, delta_dec, area} for each pixel in a 2D image
        Parameters
        ----------
        save_utils : bool
            Save the computed coordinates into the utils directory
        Returns
        -------
        wavelen_array: in microns
        dra_as_array: in arcsec
        ddec_as_array: in arcsec
        area2d: in arcsec^2
        from : https://github.com/jruffio/breads/blob/main/breads/instruments/jwstnirspec_cal.py#L338
        """
    try :
        wavelen_array = fits.getdata(filename.replace(".fits", "_wavelen_array.fits")) # µm
        dra_as_array = fits.getdata(filename.replace(".fits", "_dra_as_array.fits")) # arcsec
        ddec_as_array = fits.getdata(filename.replace(".fits", "_ddec_as_array.fits")) # arcsec
        area2d = fits.getdata(filename.replace(".fits", "_area2d.fits")) # arcsec^2
    except :
        import jwst.datamodels, jwst.assign_wcs
        from jwst.photom.photom import DataSet
        hdulist = fits.open(filename) #open file
        hdr0 = hdulist[0].header
        host_coord = propagate_coordinates_at_epoch(hdr0["TARGNAME"], hdr0["DATE-OBS"])
        host_ra_deg = host_coord.ra.deg
        host_dec_deg = host_coord.dec.deg
        shape = hdulist[1].data.shape #obtain generic shape of data
        calfile = jwst.datamodels.open(hdulist) #save time opening by passing the already opened file
        photom_dataset = DataSet(calfile)
        ## Determine pixel areas for each pixel, retrieved from a CRDS reference file
        area_fname = hdr0["R_AREA"].replace("crds://", os.path.join("/home/martoss/crds_cache", "references", "jwst", "nirspec") + os.path.sep)
        # Load the pixel area table for the IFU slices
        area_model = jwst.datamodels.open(area_fname)
        area_data = area_model.area_table
        wave2d, area2d, dqmap = photom_dataset.calc_nrs_ifu_sens2d(area_data)
        area2d[np.where(area2d == 1)] = np.nan
        wcses = jwst.assign_wcs.nrs_ifu_wcs(calfile)  # returns a list of 30 WCSes, one per slice. This is slow.
        #change this hardcoding?
        ra_array = np.zeros((2048, 2048)) + np.nan
        dec_array = np.zeros((2048, 2048)) + np.nan
        wavelen_array = np.zeros((2048, 2048)) + np.nan
        for i in range(len(wcses)):
                    print(f"Computing coords for slice {i}")
                    # Set up 2D X, Y index arrays spanning across the full area of the slice WCS
                    xmin = max(int(np.round(wcses[i].bounding_box.intervals[0][0])), 0)
                    xmax = int(np.round(wcses[i].bounding_box.intervals[0][1]))
                    ymin = max(int(np.round(wcses[i].bounding_box.intervals[1][0])), 0)
                    ymax = int(np.round(wcses[i].bounding_box.intervals[1][1]))
                    # print(xmax, xmin, ymax, ymin, ymax - ymin, xmax - xmin)
                    x = np.arange(xmin, xmax)
                    x = x.reshape(1, x.shape[0]) * np.ones((ymax - ymin, 1))
                    y = np.arange(ymin, ymax)
                    y = y.reshape(y.shape[0], 1) * np.ones((1, xmax - xmin))
                    # Transform all those pixels to RA, Dec, wavelength
                    skycoords, speccoord = wcses[i](x, y, with_units=True)
                    # print(skycoords.ra)
                    ra_array[ymin:ymax, xmin:xmax] = skycoords.ra
                    dec_array[ymin:ymax, xmin:xmax] = skycoords.dec
                    wavelen_array[ymin:ymax, xmin:xmax] = speccoord
        dra_as_array = (ra_array - host_ra_deg) * 3600 * np.cos(np.radians(dec_array)) # in arcsec
        ddec_as_array = (dec_array - host_dec_deg) * 3600 # in arcsec
        fits.writeto(filename.replace(".fits", "_wavelen_array.fits"), wavelen_array, overwrite=True)
        fits.writeto(filename.replace(".fits", "_dra_as_array.fits"), dra_as_array, overwrite=True)
        fits.writeto(filename.replace(".fits", "_ddec_as_array.fits"), ddec_as_array, overwrite=True)
        fits.writeto(filename.replace(".fits", "_area2d.fits"), area2d, overwrite=True)
    return wavelen_array, dra_as_array, ddec_as_array, area2d


def instru_thermal_background(temperature, emissivity, wavelengths_um):
    """
    Calcule le flux thermique en ph/s/µm/arcsec².

    Parameters:
        temperature (float): Température du télescope (K).
        emissivity (float): Émissivité du télescope.
        wavelengths_um (array): Longueurs d'onde (µm).
    
    Returns:
        flux_photon (array): Flux thermique en ph/s/µm/arcsec².
    """
    wavelengths_m = wavelengths_um * 1e-6  # Conversion µm -> m
    B_lambda = (2 * h * c**2) / (wavelengths_m**5) / (np.exp((h * c) / (wavelengths_m * kB * temperature)) - 1)  # W/m²/m/sr
    # Conversion en ph/s/µm/sr
    energy_per_photon = (h * c) / wavelengths_m
    B_lambda_ph       = B_lambda / energy_per_photon * 1e-6  # ph/s/m²/µm/sr

    flux = emissivity * B_lambda_ph / sr2arcsec2  # ph/s/µm/arcsec²
    return flux

