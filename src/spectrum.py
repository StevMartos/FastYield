from src.config import *
from src import throughput
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d,interp2d
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import coronagraph as cg
from astropy.convolution import Gaussian1DKernel
from astropy.stats import sigma_clip
import warnings
import os
import time
import ssl
import wget
import lzma
from scipy.special import comb

path_file = os.path.dirname(__file__)
load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")
vega_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/star_spectrum/VEGA_Fnu.fits")

h=6.6260701e-34 # J.s
c=2.9979246e8 # m/s
kB=1.38065e-23 # J/K



class Spectrum:

    def __init__(self,wavelength,flux,R,T,lg=None,model=None,syst_rv=0,delta_rv=0,high_pass_flux=None):
        """
        Parameters
        ----------
        wavelength : 1d array
            wavelength axis
        flux : 1d array
            flux axis
        R : float
            spectral resolution of the spectrum
        T : float
            temperature of the spectrum
        """
        self.wavelength = wavelength
        self.flux = flux
        self.R = R
        self.T = T
        self.lg = lg
        self.model = model  # model of the spectrum
        self.syst_rv = syst_rv # radial velocity of the system (star)
        self.delta_rv = delta_rv # Doppler shift between the planet and the star
        self.high_pass_flux = high_pass_flux # high-pass filtered flux
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def crop(self,lmin,lmax):
        """
        Crop the spectrum between lmin and lmax (and calculate the new spectral resolution)

        Parameters
        ----------
        lmin : float (in µm)
            lambda min value
        lmax : float (in µm)
            lambda max value
        """
        self.flux = self.flux[(self.wavelength >= lmin) & (self.wavelength <= lmax)]
        self.wavelength = self.wavelength[(self.wavelength >= lmin) & (self.wavelength <= lmax)]
        dwl = self.wavelength - np.roll(self.wavelength, 1) ; dwl[0] = dwl[1] # array de delta Lambda
        Rnew = np.nanmean(self.wavelength/(2*dwl)) # calcule de la nouvelle résolution
        self.R = Rnew
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def set_flux(self, nbPhotons):
        """
        Renormalize / Convert flux to photon number (or other)
        The flux can be in density but must have a constant delta_lambda.
        
        Parameters
        ----------
        nbPhotons : float/int
            quantity in a certain unit with which to renormalize the flux
        """
        self.flux = nbPhotons * self.flux / np.nansum(self.flux) # self.flux / np.sum(self.flux) = fraction du flux dans le canal spectral correspondant

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_nbphotons_min(self, config_data, wave_band):
        """
        Converts a spectrum initially in density (J/s/m²/µm) into nb of photons/min received by the telescope over the wave_band range
        !!! wave_band MUST HAVE A delta_lambda CONSTANT !!!  
            
        Parameters
        ----------
        config_data : collections
            gives the parameters of the considered instrument
        wave_band : array (must be in µm)
            wavelength axis on which the spectrum is converted

        Returns
        -------
        spectrum_band : class Spectrum
            spectrum converted into nb of photons/min
        """
        delta_lambda_band = wave_band - np.roll(wave_band, 1) ; delta_lambda_band[0] = delta_lambda_band[1]  # delta lambda en µm
        Rnew = np.nanmean(wave_band/(2*delta_lambda_band)) # calcul de la nouvelle résolution
        spectrum_band_flux = self.interpolate_wavelength(self.flux, self.wavelength, wave_band , renorm=False).flux # on réinterpole le flux (en densité) sur wave_band
        spectrum_band = Spectrum(wave_band,spectrum_band_flux,Rnew,self.T,self.lg,self.model,syst_rv=self.syst_rv,delta_rv=self.delta_rv,high_pass_flux=self.high_pass_flux) # on définit le nouveau spectre comme étant un class Spectrum
        spectrum_band.flux = spectrum_band.flux*wave_band*1e-6/(h*c) # J/s/m²/µm => photons/s/m2/µm
        S = config_data["telescope"]["area"] # surface collectrice du télescope en m2
        spectrum_band.flux = spectrum_band.flux*S*delta_lambda_band*60 # en photons/min en fonction du canal spectral
        return spectrum_band

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def degrade_resolution(self, wavelength_output, renorm=True, gaussian_filtering=True):
        """
        Degrade the spectral resolution at the new resolution (of wavelength_output) with a convolution
        ! Does not work if the new resolution is higher than the basic resolution !

        Parameters
        ----------
        wavelength_output : array
            new wavelength axis (new resolution)
        renorm : bool, optional
            for renormalisation to conserve the flux (True => flux must not be in density eg. J/s/m²/µm) . The default is True.

        Returns
        -------
        spectrum_band : class Spectrum
            degrated spectrum
        """
        valid = np.where((self.wavelength >= wavelength_output[0]) & (self.wavelength <= wavelength_output[-1])) # flux[valid] => retourne une array (plus petite) qui garde les valeurs du flux pour une longueur d'onde comprise entre wavelength_output[0] et wavelength_output[-1]
        nbPhot = np.nansum(self.flux[valid]) # nb total de photon / Ã©nergie totale dans le spectre
        if gaussian_filtering :
            dl = self.wavelength[valid] - np.roll(self.wavelength[valid], 1) ; dl[0] = dl[1] # array de delta Lambda
            Rold = np.nanmax(self.wavelength[valid]/(2*dl))
            if Rold > 200000 : 
                Rold = 200000
            dl = np.nanmean(self.wavelength[valid]/(2*Rold)) # np.nanmin(dl) # 2*R => Nyquist samplé (Shannon)
            wave_inter = np.arange(0.98*wavelength_output[0],1.02*wavelength_output[-1],dl)
            flr = self.interpolate_wavelength(self.flux, self.wavelength, wave_inter, renorm=renorm).flux
            dwl = wavelength_output - np.roll(wavelength_output, 1) ; dwl[0] = dwl[1] # array de delta Lambda
            Rnew = np.nanmean(wavelength_output/(2*dwl)) # calcul de la nouvelle rÃ©solution*
            fwhm = Rold/Rnew
            sigma_conv = fwhm # fwhm/(2*np.sqrt(2*np.log(2)))
            flr_conv = gaussian_filter(flr[~np.isnan(flr)],sigma=sigma_conv)
            flr[~np.isnan(flr)] = flr_conv
            flr = cg.downbin_spec(flr, wave_inter, wavelength_output, dlam=dwl) # convolution de la courbe du flux avec la nouvelle rÃ©solution/largeur des bins (nouvel axe de lambda)
        else :
            dwl = wavelength_output - np.roll(wavelength_output, 1) ; dwl[0] = dwl[1] # array de delta Lambda
            Rnew = np.nanmean(wavelength_output/(2*dwl)) # calcul de la nouvelle résolution
            flr = (cg.downbin_spec(self.flux, self.wavelength, wavelength_output, dlam=dwl)) # convolution de la courbe du flux avec la nouvelle résolution/largeur des bins (nouvel axe de lambda)    
        if renorm:
            return Spectrum(wavelength_output, nbPhot*flr/np.nansum(flr), Rnew, self.T, lg=self.lg, model=self.model,syst_rv=self.syst_rv,delta_rv=self.delta_rv,high_pass_flux=self.high_pass_flux) # conserve le nb de photon
        else:
            return Spectrum(wavelength_output, flr, Rnew, self.T, lg=self.lg, model=self.model,syst_rv=self.syst_rv,delta_rv=self.delta_rv,high_pass_flux=self.high_pass_flux) # ne conserve pas le nb de photons (pour la transmission et spectre en densitÃ©)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
    def interpolate_wavelength(self, influx, wavelength_input, wavelength_output, renorm=True, fill_value=np.nan):
        """
        Re-interpolates the flux on a new wavelength axis
        
        Parameters
        ----------
        influx : array (can be in density or not)
            actual flux (non-interpolated)
        wavelength_input : array 
            actual wavelength axis of the flux
        wavelength_output : array
            new wavelength axis for the interpolation
        renorm : bool, optional
            for renormalisation (True => flux must not be in density eg. J/s/m²/µm) . The default is True.

        Returns : class Spectrum
            Spectrum with the interpolated flux on the new wavelength axis
        """
        valid = np.where((wavelength_input >= wavelength_output[0]) & (wavelength_input <= wavelength_output[-1])) # influx[valid] => retourne une array (plus petite) qui garde les valeurs du flux pour une longueur d'onde comprise entre wavelength_output[0] et wavelength_output[-1]
        nbPhot = np.nansum(influx[valid]) # nombre total de photon dans la gamme de longueur d'onde output (pour la renormalisation si renorm = True)
        #f = interp1d(wavelength_input, influx, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        f = interp1d(wavelength_input, influx, bounds_error=False, fill_value=fill_value) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        flux_interp = f(wavelength_output) # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
        dwl = wavelength_output - np.roll(wavelength_output, 1) ; dwl[0] = dwl[1] # array de delta Lambda
        Rnew = np.nanmean(wavelength_output/(2*dwl)) # calcule de la nouvelle résolution
        if renorm:
            spec = Spectrum(wavelength_output, nbPhot*flux_interp/np.nansum(flux_interp), Rnew, self.T,self.lg,self.model,syst_rv=self.syst_rv,delta_rv=self.delta_rv,high_pass_flux=self.high_pass_flux)
        else :
            spec = Spectrum(wavelength_output, flux_interp, Rnew, self.T,self.lg,self.model,syst_rv=self.syst_rv,delta_rv=self.delta_rv,high_pass_flux=self.high_pass_flux)
        return spec
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def doppler_shift(self, rv, renorm=True, fill_value=np.nan):
        """
        Doppler shift of a spectrum as a function of radial velocity rv
        
        Parameters
        ----------
        rv : float/int (in km/s)
            radial velocity

        Returns : class Spectrum
            shifted spectrum
        """
        rv = rv * (u.km / u.s) # définit rv comme étant en km/s avec astropy
        rv = rv.to(u.m / u.s) # convertit rv en m/s avec astropy 
        wshift = self.wavelength * (1 + (rv / const.c)) # const.c = vitesse de la lumière (via astropy) / axe de longueur d'onde décalé
        spec_rv = self.interpolate_wavelength(self.flux, wshift, self.wavelength, renorm=renorm,fill_value=fill_value)
        spec_rv.R = self.R        
        return spec_rv

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def broad(self, broad):
        """
        Broadens spectrum lines as a function of rotation speed (in km/s)

        Parameters
        ----------
        broad : float
            rotation speed (in km/s)

        Returns : class Spectrum
            broadened spectrum
        """
        flux = pyasl.fastRotBroad(self.wavelength * 1e4, self.flux, 0.8, broad) # Pourquoi *1e4 ? (pour passer en Angstrom ?)
        return Spectrum(self.wavelength, flux, self.R, self.T,self.lg,self.model,syst_rv=self.syst_rv,delta_rv=self.delta_rv,high_pass_flux=self.high_pass_flux) # pas besoin de "conserver le nb de photons" car c'est un effet intrinsèque (on ne change pas largeur des bins)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def plot_spectrum(self):
        """
        Plot of a spectrum
        """
        #plt.figure()
        plt.plot(self.wavelength, self.flux) ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("flux")


    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def plot_psd(self,smooth=1, color='b',show=False,ret=False,ymax=False,xlim=True,area=True,crop=True):
        if len(self.flux)%2 != 0 and crop :
            signal = np.copy(self.flux)[:-1]
        else :
            signal = np.copy(self.flux)
        N = len(signal)
        ffreq = np.fft.fftfreq(N)
        fft = np.abs(np.fft.fft(signal))**2
        if crop :
            PSD = (fft[:N//2] + np.roll(np.flip(fft[N//2:]),1)) * (1/N)
            #PSD = 2*fft[:N//2] * (1/N)
            res = ffreq[:N//2]*self.R*2
            #plt.figure() ; plt.plot(res,np.sqrt(fft[:N//2])) ; plt.plot(res,np.sqrt(np.roll(np.flip(fft[N//2:]),1))) ; plt.xscale('log') ; plt.yscale('log') ; plt.show()
        else :
            PSD = fft * (1/N)
            res = ffreq*self.R*2
        if smooth==0:
            PSD_smooth = PSD
        else:
            PSD_smooth = gaussian_filter(PSD, sigma=smooth)
        if show:
            plt.plot(res, PSD_smooth, color=color)
            if area:
                plt.fill_between(res, PSD_smooth, 0, color="none", hatch="//", edgecolor=color)
            plt.title(f"PSD of planet spectrum at {self.T} K", fontsize=14)
            plt.xlabel("Resolution ", fontsize=14)
            plt.ylabel(r'PSD{$S_p(\lambda$)}(R)', fontsize=14)
            if xlim:
                plt.xlim([10, self.R*2])
            plt.xscale('log')
            if ymax:
                plt.ylim(0,np.max(PSD_smooth))
        if ret:
            return res, PSD_smooth


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_planet_spectrum(T, lg, model="BT-Settl", version="new", met=1, CO=0.5, load_path=load_path):
    """
    To read and retrieve planet spectra from models

    Parameters
    ----------
    T : float/int
        planet temperature
    lg : float/int
        planet gravity surface
    model : str
        spectrum model ("BT-Settl" or "Exo-REM")
    load_path : str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns : class Spectrum
        loaded planet spectrum in J/s/m²/µm
    """
    
    if model=="BT-Settl":
        if T < 200:
            print("Changing the input temperature to the minimal temperature : 200K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature : 3000K.")
        T0 = np.append(np.arange(500,1000,50),np.arange(1000,3100,100))
        T0 = np.append([200,220,240,260,280,300,320,340,360,380,400,450],T0)
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        if T < 260 :
            lg0 = np.array([4.0])
        elif T == 260 :
            lg0 = np.array([3.5])
        elif T <= 300 :
            lg0 = np.array([3.0,3.5])
        elif T < 500 :
            lg0 = np.array([2.5,3.0,3.5])
        else :
            lg0 = np.array([3.0,3.5,4.0,4.5,5.0])
        idx = (np.abs(lg0 - lg)).argmin()
        lg = lg0[idx]
        if T >= 1000:
            str_T = "0"+str(T)[:2] ; str_lg = str(lg)
        else :
            str_T = "00"+str(T)[0] ; str_lg = str(lg)
        if str(T)[-2]!="0":
            str_T += "."+str(T)[-2]
        wave,flux = fits.getdata(load_path+'/planet_spectrum/'+model+"/lte"+str_T+"-"+str_lg+"-0.0a+0.0.BT-Settl.fits")
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; R = np.nanmean(wave/(2*dwl))
        spec = Spectrum(wave,flux,R,T,lg,"BT-Settl")
    
    elif model=="BT-Dusty":
        if T < 1400:
            print("Changing the input temperature to the minimal temperature : 1400K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature : 3000K.")
        T0 = np.arange(1400,3100,100)
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        lg0 = np.array([4.5,5.0])
        idx = (np.abs(lg0 - lg)).argmin()
        lg = lg0[idx]
        if T >= 1000:
            str_T = "0"+str(T)[:2] ; str_lg = str(lg)
        else :
            str_T = "00"+str(T)[0] ; str_lg = str(lg)
        if str(T)[-2]!="0":
            str_T += "."+str(T)[-2]
        wave,flux = fits.getdata(load_path+'/planet_spectrum/'+model+"/lte"+str_T+"-"+str_lg+"-0.0a+0.0.BT-Dusty.fits")
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; dwl[dwl==0] = np.nan ; R = np.nanmean(wave/(2*dwl))
        spec = Spectrum(wave,flux,R,T,lg,"BT-Dusty")
        
    elif model=="Exo-REM":
        if T < 400:
            print("Changing the input temperature to the minimal temperature : 400K.")
        elif T > 2000:
            print("Changing the input temperature to the maximal temperature : 2000K.")
        if version == "old": # high res (mais commmence à 4 µm)
            lg0 = np.array([3.5, 4.0]) # valeur possible
            idx = (np.abs(lg0 - lg)).argmin()
            lg = lg0[idx]
            T0 = np.array([400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
            idx = (np.abs(T0 - T)).argmin()
            T = T0[idx]
            load_path += '/planet_spectrum/'+model+'/lte-g' + str(float(lg)) + '/'
            load_path+="spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
        elif version == "new": # low res
            lg0 = np.arange(3.0, 5.5,0.5) # valeur possible
            idx = (np.abs(lg0 - lg)).argmin()
            lg = lg0[idx]
            T0 = np.arange(400,2050,50)
            idx = (np.abs(T0 - T)).argmin()
            T = T0[idx]
            load_path += '/planet_spectrum/'+model+'/'
            load_path += "spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
        wave,flux = fits.getdata(load_path)
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; R = np.nanmean(wave/(2*dwl))
        spec = Spectrum(wave,flux,R,T,lg,"Exo-REM")
        
    elif model=="PICASO":
        if T < 200:
            print("Changing the input temperature to the minimal temperature : 200K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature : 3000K.")
        T0 = np.append(np.arange(500,1000,50),np.arange(1000,3100,100))
        T0 = np.append([200,220,240,260,280,300,320,340,360,380,400,450],T0)
        idx = (np.abs(T0 - T)).argmin()
        T=T0[idx]
        if T < 260 :
            lg0 = np.array([4.0])
        elif T == 260 :
            lg0 = np.array([3.5])
        elif T <= 300 :
            lg0 = np.array([3.0,3.5])
        elif T < 500 :
            lg0 = np.array([2.5,3.0,3.5])
        else :
            lg0 = np.array([3.0,3.5,4.0,4.5,5.0])
        idx = (np.abs(lg0 - lg)).argmin()
        lg=lg0[idx]
        wave,flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{T}K_lg{lg}.fits")
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; R = np.nanmean(wave/(2*dwl)) # calcule de la nouvelle résolution
        spec = Spectrum(wave,flux,R,T,lg,"PICASO")
    
    elif model=="Morley": # 2012 + 2014 avec nuage
        if T < 200:
            print("Changing the input temperature to the minimal temperature : 200K.")
        if T > 1300:
            print("Changing the input temperature to the maximal temperature : 1300K.")
        T0 = np.array([200,225,250,275,300,325,350,375,400,450,500,550,600,700,800,900,1000,1100,1200,1300]) # K
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        if T < 500 : 
            g0 = np.array([10,30,100,300,1000]) # m/s²
            lg0 = np.array([3,3.5,4,4.5,5])
        else :
            g0 = np.array([100,300,1000,3000]) # m/s² 
            lg0 = np.array([4,4.5,5,5.5])
        g = 10**lg*1e-2 # m/s²
        idx = (np.abs(g0 - g)).argmin()
        g = g0[idx]
        lg = lg0[idx]
        wave,flux = fits.getdata("sim_data/Spectra/planet_spectrum/Morley/sp_t"+str(T)+"g"+str(g)+".fits")
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] # array de delta Lambda
        R = np.nanmean(wave/(2*dwl)) # calcule de la nouvelle résolution
        spec = Spectrum(wave,flux,R,T,lg,"Morley")
        
    elif model == "Saumon":
        if T < 400:
            print("Changing the input temperature to the minimal temperature : 400K.")
        if T > 1200:
            print("Changing the input temperature to the maximal temperature : 1200K.")
        T0 = np.arange(400,1250,50)
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        g0 = np.array([10,30,100,300,1000]) # m/s²
        lg0 = np.array([3,3.5,4,4.5,5])
        g = 10**lg*1e-2 # m/s²
        idx = (np.abs(g0 - g)).argmin()
        g = g0[idx]
        lg = lg0[idx]
        wave , flux = fits.getdata("sim_data/Spectra/planet_spectrum/Saumon/sp_t"+str(T)+"g"+str(g)+"nc.fits")
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; R = np.nanmean(wave/(2*dwl)) # calcule de la nouvelle résolution
        spec = Spectrum(wave,flux,R,T,lg,"Saumon")
    
    elif model == "SONORA":
        if T < 200:
            print("Changing the input temperature to the minimal temperature : 200K.")
        if T > 2400:
            print("Changing the input temperature to the maximal temperature : 2400K.")
        T0 = np.append(np.arange(200,1050,50),np.arange(1100,2500,100))
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        g0 = np.array([10,31,100,316,1000,3160]) # m/s²
        lg0 = np.array([3,3.5,4,4.5,5,5.5])
        g = 10**lg*1e-2 # m/s²
        idx = (np.abs(g0 - g)).argmin()
        g = g0[idx]
        lg = lg0[idx]
        wave , flux = fits.getdata("sim_data/Spectra/planet_spectrum/SONORA/sp_t"+str(T)+"g"+str(g)+"nc_m0.0.fits")
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; R = np.nanmean(wave/(2*dwl)) # calcule de la nouvelle résolution
        spec = Spectrum(wave,flux,R,T,lg,"Saumon")
        
    return spec # en J/s/m²/µm

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_star_spectrum(T,lg,model="BT-NextGen",load_path=load_path):
    """
    load AGSS2009 star spectrum model (in J/s/m2/µm)
    Parameters
    ----------
    spectral_type : str
        spectral type of the star
        
    Returns : class Spectrum
        loaded star spectrum in J/s/m²/µm
    """
    if model=="BT-NextGen":
        lg0 = np.array([3.0,3.5,4.0,4.5])
        T0 = np.append(np.arange(3000,10000,200),np.arange(10000,41000,1000))
        idx = (np.abs(lg0 - lg)).argmin()
        lg = lg0[idx]
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        if T >= 10000 :
            str_T = str(T)[:3] ; str_lg = str(lg)
        elif T >= 1000:
            str_T = "0"+str(T)[:2] ; str_lg = str(lg)
        else :
            str_T = "00"+str(T)[0] ; str_lg = str(lg)
        spectrum = fits.getdata(load_path+'/star_spectrum/'+model+"/lte"+str_T+"-"+str_lg+"-0.0a+0.0."+model+".fits")
        wave = spectrum[0] ; dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; R = np.nanmean(wave/(2*dwl))
        spec = Spectrum(spectrum[0],spectrum[1],R,T,lg,model)
        
    return spec

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_bz2(filepath):
    """
    load a star spectrum (in J/s/m2/µm)
    
    Returns : class Spectrum
        loaded star spectrum in J/s/m²/µm
    """
    dataf = pd.pandas.read_csv(filepath,usecols=[0, 1],names=['wavelength', 'flux'],header=None,dtype={'wavelength': str, 'flux': str},delim_whitespace=True,compression='bz2')
    dataf['wavelength'] = dataf['wavelength'].str.replace('D', 'E')
    dataf['flux'] = dataf['flux'].str.replace('D', 'E')
    dataf = dataf.apply(pd.to_numeric)
    data = dataf.values
    star_wavel = data[:, 0] * 1e-4  # Angstrom => µm
    star_spect = 10. ** (data[:, 1] - 8.)  # erg/s/cm2/A
    star_spect = star_spect*10  # erg/s/cm2/A => J/s/m2/µm
    index_sort = np.argsort(star_wavel) #donne la position des valeurs de lambda dans l'ordre croissant 
    spectrum = np.array([star_wavel[index_sort], star_spect[index_sort]]) #remet les données dans l'ordre croissant de lambda
    spec = Spectrum(spectrum[0, :], spectrum[1, :], 100000, None) # +0.001 pour ne pas avoir de 0 dans l'interpolation (log10(0) => -inf)
    return spectrum #spec

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_vega_spectrum(vega_path=vega_path):
    f = fits.getdata(os.path.join(vega_path))
    wave=f[:,0]*1e-3 # nm => µm
    flux=f[:,1]*10 # 10 = 1e4 * 1e4 * 1e-7 => erg/s/cm2/A -> erg/s/cm2/µm -> erg/s/m2/µm -> J/s/m2/µm
    vega_spec = Spectrum(wave,flux,None,None)
    return vega_spec


def zeropoint(waveobs, config_data, vega_path=vega_path,return_spectrum=False):
    """
    Compute the Zero point (flux) of the instrument with the vega spectrum in order to ajust spectra magnitude with it
    
    Parameters
    ----------
    waveobs : 1d array
        input wavelengths of interest ( in µm)
    config_data : collections
        gives the parameters of the considered instrument

    Returns : 1d array
        vega flux in J/s/m²/µm
    """
    vega_spec = load_vega_spectrum()
    vega_spec=vega_spec.set_nbphotons_min(config_data, waveobs) # in µm and photons/min
    if return_spectrum :
        return vega_spec
    else :
        return vega_spec.flux
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def spectrum_instru(band0,R,config_data,mag,spectrum):
    """
    restricts the spectra to the instrument's wavelength range and adjusts it to the input magnitude
    
    Parameters
    ----------
    band0 : str
        wavelength range in which magnitude is entered ("J" or "H" etc.)
    R : float
        spectral resolution of the input spectrum (it can be arbitrary but must be well above the instrumental spectral resolution)
    config_data : collections
        gives the parameters of the considered instrument
    mag : float
        input magnitude
    spectrum : class Spectrum
        spectrum to restrict and adjust (! must be in J/s/m2/µm !)

    Returns
    -------
    spectrum_instru : class Spectrum
        instrumental-wavelength-range-restricted and magnitude-adjusted spectrum in photons/min received
    """
    # Calcul du nombre de photons/min pour une mag dans la bande donnée et du nombre de photons/min initialement dans cette même bande
    if band0=="J":
        lambda_c=1.215 ; Dlambda=0.26 # en µm
    elif band0=="H":
        lambda_c=1.654 ; Dlambda=0.29 # en µm
    elif band0=='Ks':
        lambda_c=2.157 ; Dlambda=0.32 # en µm
    elif band0=='K':
        lambda_c=2.179 ; Dlambda=0.41 # en µm
    elif band0=="L": 
        lambda_c=3.547 ; Dlambda=0.57 # en µm
    elif band0=="L'": 
        lambda_c=3.761 ; Dlambda=0.65 # en µm
    elif band0=="instru":
        lambda_c=(config_data["lambda_range"]["lambda_max"]+config_data["lambda_range"]["lambda_min"])/2 ; Dlambda=config_data["lambda_range"]["lambda_max"]-config_data["lambda_range"]["lambda_min"] # en µm
    elif band0=="NIR": 
        lambda_c=2.75 ; Dlambda=4.5 # en µm
    lambda_min_band0 = lambda_c-Dlambda/2 ; lambda_max_band0 = lambda_c+Dlambda/2 # en µm
    delta_lamb_band0 = ((lambda_max_band0+lambda_min_band0)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave_band0 = np.arange(lambda_min_band0, lambda_max_band0, delta_lamb_band0) # en µm
    spec = spectrum.interpolate_wavelength(spectrum.flux, spectrum.wavelength, wave_band0, renorm=False)
    vega_spec = load_vega_spectrum()
    vega_spec = vega_spec.interpolate_wavelength(vega_spec.flux, vega_spec.wavelength, wave_band0, renorm=False)
    ratio = np.nanmean(vega_spec.flux)*10**(-0.4*mag) / np.nanmean(spec.flux)
    # Conversion en photons/min + restriction des spectres à la gamme instrumentale + ajustement des spectres à la bonne magnitude
    lambda_min_instru = config_data["lambda_range"]["lambda_min"] ; lambda_max_instru = config_data["lambda_range"]["lambda_max"] # en µm
    delta_lamb_instru = ((lambda_max_instru+lambda_min_instru)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave_instru = np.arange(0.98*lambda_min_instru, 1.02*lambda_max_instru, delta_lamb_instru) # en µm
    spectrum.flux *= ratio # ajustement du spectre à la bonne magnitude
    spectrum_density = spectrum.interpolate_wavelength(spectrum.flux, spectrum.wavelength, wave_instru, renorm = False)        
    spectrum_instru = spectrum.set_nbphotons_min(config_data, wave_instru) # J/s/m²/µm => photons/min sur la gamme instrumentale
    return spectrum_instru , spectrum_density

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def spectrum_inter(config_data,band,spectrum_instru):
    """
    Degradation of the spectrum resolution and restriction to the considered spectral band of the instrument

    Parameters
    ----------
    config_data : collections
        gives the parameters of the considered instrument
    band : str
        considered spectral band of the instrument
    spectrum_instru : class Spectrum
        instrumental-wavelength-range-restricted and magnitude-adjusted spectrum in photons/min received
    renorm : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    spectrum_instru : class Spectrum
        band-restricted and resolution-degrated spectrum in photons/min received
    """
    lmin = config_data['gratings'][band].lmin ; lmax = config_data['gratings'][band].lmax # lambda_min/lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    if R is None : # dans le cas où il ne s'agit pas d'un spectro-imageur (eg NIRCAM)
        R = spectrum_instru.R
    delta_lambda = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave_inter = np.arange(lmin,lmax,delta_lambda) # axe de longueur d'onde de la bande considérée
    spectrum_inter = spectrum_instru.degrade_resolution(wave_inter,renorm=True) # dégradation de la résolution du spectre à la résolution spectrale de la bande considérée
    return spectrum_inter # spectre dégradé

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def transmission(instru,wave_inter,band,tellurics,apodizer):
    """
    To read and retrieve instrumental and sky (if needed) transmission

    Parameters
    ----------
    instru : str
        considered instrument
    wave_inter : 1d array
        wavelength axis of the considered spectral band
    band : str
        considered spectral band
    R : float
        spectral resolution of the considered spectral band
    tellurics : bool (True or False)
        considering (or not) the earth atmosphere (ground or space observations)

    Returns
    -------
    transmission : 1d array (same size as wave_inter)
        total system transmission 
    """
    if instru == "HARMONI" or instru == "ANDES" :
        trans_telescope = throughput.telescope_throughput(wave_inter)
        trans_instrumental = throughput.instrument_throughput(wave_inter, str(band))
        trans_fprs = throughput.fprs_throughput(wave_inter)
        apo_trans = config_data_HARMONI["apodizers"][str(apodizer)].transmission
        trans = trans_telescope * trans_instrumental * trans_fprs * apo_trans
    else :
        wave, trans = fits.getdata("sim_data/Transmission/"+instru+"/Instrumental_transmission/transmission_" + band + ".fits") # transmission instrumentale de la bande considérée
        f = interp1d(wave, trans, bounds_error=False, fill_value=np.nan)
        trans = f(wave_inter) # transmission instrumentale de la bande considérée en fonction du canal spectral
        if instru == "MIRIMRS":
            trans *= fits.getheader("sim_data/PSF/star_center/PSF_MIRIMRS/PSF_"+band+".fits")['AC']
    if tellurics: # si on considère l'atmosphère terrestre 
        sky_transmission_path = os.path.join("sim_data/Transmission/sky_transmission_airmass_1.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        trans_tell_band = Spectrum(sky_trans[0, :], sky_trans[1, :], 100000, None)
        trans_tell_band = trans_tell_band.degrade_resolution(wave_inter, renorm=False).flux # transmission atmosphérique de la bande considérée en fonction du canal spectral
    else:
        trans_tell_band = 1
    transmission = trans*trans_tell_band
    transmission[transmission==0] = np.nan
    return transmission

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def PSF_profile_fraction_separation(band,strehl,apodizer,coronagraph,instru,config_data,sep_unit,star_pos):
    """
    Gives the PSF profile, the fraction of flux contained in the PSF core of interest (or coronagraphic transmission) and the separation vector.
    
    Parameters
    ----------
    band : str
        considered spectral band
    strehl : str
        strehl ratio
    apodizer : str
        apodizer
    coronagraph : str
        coronagraph
    instru : str
        instrument
    config_data : collections
        gives the parameters of the considered instrument

    Returns
    -------
    PSF_profile : 1d-array
        PSF profile
    fraction_PSF : float
        fraction core or coronagraphic transmission
    separation : 1d-array
        separation vector (in arcsec or mas)
    """
    if instru=="HARMONI" or instru=="ERIS" or instru=="ANDES" : # observation depuis le sol 
        file = "sim_data/PSF/star_"+star_pos+"/PSF_"+instru+"/PSF_"+band+"_"+strehl+"_"+apodizer+".fits"
        fraction_PSF = fits.getheader("sim_data/PSF/star_center/PSF_"+instru+"/PSF_"+band+"_"+strehl+"_"+apodizer+".fits")['FC'] # fraction du flux dans une boîte de size_core²
    elif instru=="MIRIMRS" or instru=="NIRCam" or instru=="NIRSpec" : # observation depuis l'espace
        if coronagraph is not None : # Profil de la PSF (stellaire) coronographique
            file = "sim_data/PSF/star_"+star_pos+"/PSF_"+instru+"/PSF_"+band+"_"+coronagraph+".fits"
            fraction_PSF = fits.getheader("sim_data/PSF/star_center/PSF_"+instru+"/PSF_"+band+"_"+coronagraph+".fits")['FC'] # transmission du flux stellaire à travers les masques
        else :
            file = "sim_data/PSF/star_"+star_pos+"/PSF_"+instru+"/PSF_"+band+".fits"
            fraction_PSF = fits.getheader("sim_data/PSF/star_center/PSF_"+instru+"/PSF_"+band+".fits")['FC'] # fraction du flux dans une boîte de size_core²
    if instru=="MIRIMRS":
        pixscale = config_data["pixscale"][band] # en arcsec/px (pixscale dithered)    
    else :
        pixscale = config_data["spec"]["pixscale"] # en arcsec/px
    FOV = config_data["spec"]["FOV"] # arcsec
    if sep_unit=="mas":
        pixscale *= 1e3 ; FOV *= 1e3
    profile = fits.getdata(file) # en fraction/arcsec ou fraction/mas
    #separation = np.arange(pixscale, FOV/2, pixscale) # en arcsec (/4 ne change pas le résultat mais rend la courbe plus "smooth")
    separation = np.arange(pixscale, FOV/2, pixscale/4) # en arcsec (/4 ne change pas le résultat mais rend la courbe plus "smooth")
    f = interp1d(profile[0], profile[1], bounds_error=False, fill_value=np.nan)
    if instru=="MIRIMRS":
        PSF_profile = f(separation) * config_data["pixscale0"] # pxscale non dithered
    else :
        PSF_profile = f(separation) * pixscale
    return PSF_profile , fraction_PSF , separation

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def smoothstep(x, Rc, N=15):
    x_min = 0
    x_max = 2*Rc
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    result += result[::-1]
    result = np.abs(result-1)
    return result

def filtered_flux(flux,R,Rc,used_filter):
    if Rc is None :
        flux_BF = 0
    else :
        flux_BF_valid = np.copy(flux)
        if used_filter == "gaussian" :
            sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2)
            flux_BF = gaussian_filter(flux[~np.isnan(flux)],sigma=sigma)
        elif used_filter == "step" :
            fft = np.fft.fft(flux[~np.isnan(flux)]) ; ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)])) ; res = ffreq*R*2 ; fft[np.abs(res)>Rc] = 0 ; flux_BF = np.fft.ifft(fft)
        elif used_filter == "smoothstep" :
            fft = np.fft.fft(flux[~np.isnan(flux)])
            ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)]))
            res = ffreq*R*2
            fft *= smoothstep(res,Rc)
            flux_BF = np.real(np.fft.ifft(fft))
        flux_BF_valid[~np.isnan(flux)] = flux_BF
    flux_HF = flux - flux_BF_valid
    return flux_HF,flux_BF_valid

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def beta_calc(star_spectrum_inter,planet_spectrum_inter,template,Rc,R,fraction_PSF,trans,separation,used_filter):
    """
    Gives the value of the self-subtraction term β

    Parameters
    ----------
    star_spectrum_inter : class Spectrum
        band-restricted and resolution-degrated star spectrum in photons/min received
    planet_spectrum_inter : class Spectrum
        band-restricted and resolution-degrated planet spectrum in photons/min received
    sigma : float
        cut-off frequency = parameter for the high-pass filter (plus sigma est grand, plus la fréquence de coupure est petite => moins on coupera les basses fréquences)
    fraction_PSF : float
        fraction of flux contained in the PSF core of interest
    trans : 1d-array
        total-systemm transmission
    return_template : bool, optional
        True for returning template. The default is True.
    """
    star_HF,star_BF = filtered_flux(star_spectrum_inter.flux,R,Rc,used_filter)
    planet_HF,planet_BF = filtered_flux(planet_spectrum_inter.flux*fraction_PSF,R,Rc,used_filter)
    beta = np.nansum(trans*star_HF*planet_BF/star_BF * template)
    beta = np.zeros_like(separation) + beta
    return beta

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def alpha_calc(planet_spectrum_inter,template,Rc,R,fraction_PSF,transmission,separation,used_filter):
    """
    Compute the total amount of useful photons/min for molecular mapping detection
    
    Parameters
    ----------
    fraction_PSF : float
        fraction of flux of the PSF in size_core**2 pixels
    transmission : array
        instrumental + sky transmission
    sigma : float
        cut-off frequency = parameter for the high-pass filter
        (plus sigma est grand, plus la fréquence de coupure est petite => moins on coupera les basses fréquences)
    
    Returns
    -------
    alpha : float/int
        amount of useful photons
    """
    flux = planet_spectrum_inter.flux*fraction_PSF
    flux_HF,_ = filtered_flux(flux,R,Rc,used_filter)
    flux_HF *= transmission
    alpha = np.nansum(flux_HF*template) # alpha cos theta lim
    alpha = np.zeros_like(separation) + alpha
    return alpha

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DIT_RON(instru,config_data,apodizer,PSF_profile,separation,star_spectrum_inter,time,min_DIT,max_DIT,trans,quantum_efficiency,RON,saturation_e,print_value=True):
    """
    Gives DIT and effective reading noise

    Parameters
    ----------
    instru : str
        considered instrument
    config_data : collections
        gives the parameters of the considered instrument
    apodizer : str
        apodizer
    PSF_profile : 1d-array
        PSF profile
    star_spectrum_inter : class Spectrum
        band-restricted and resolution-degrated star spectrum in photons/min received
    time : float (in min)
        total exposure time
    min_DIT : float (in min)
        minimum integration time
    max_DIT : float (in min)
        maximum integration time
    quantum_efficiency : float
        quantum efficiency (photon => electron)
    RON : float
        Read-Out Noise (in e-)
    saturation_e : float
        full well capacity (in e-)

    Returns
    -------
    DIT : float (in min)
        integration time
    RON_eff : TYPE
        effective Read-Out Noise (e-)
    """
    # Calcul du du DIT et du saturing DIT
    if instru == "NIRCam":
        max_flux_e = np.nanmax(PSF_profile) * np.nansum(star_spectrum_inter.flux) * trans * quantum_efficiency # nombre max d'e-/min en fonction du canal spectral dans la bande spectrale considérée
    else :
        if apodizer is not None: # Pour HARMONI, le flux max n'est pas au centre de l'image à cause de l'apodiseur
            sep_min = config_data_HARMONI["apodizers"][apodizer].sep # séparation où la PSF est maximale 
        else :
            sep_min = 0 
        n = np.argmin(np.abs(separation - sep_min)) # donne l'index qui correspond à la séparation où la valeur de la PSF est maximale
        max_flux_e = PSF_profile[n] * star_spectrum_inter.flux * trans * quantum_efficiency # nombre max d'e-/min en fonction du canal spectral dans la bande spectrale considérée  
        
    saturating_DIT = saturation_e/np.nanmax(max_flux_e) # ici DIT = saturating DIT (en min)
    if print_value :
        print(" Saturating DIT =", round(saturating_DIT,3), " minutes") #print("Saturating  DIT = ", DIT*60, " minutes")
    # Le max DIT est déterminée par la saturation ou bien par le "smearing" (le fait que le planète ne doit pas trop bouger pendant la pose)
    if saturating_DIT > max_DIT :
        DIT = max_DIT
    elif saturating_DIT < min_DIT :
        DIT = min_DIT
        if print_value:
            print("Saturated detector even with the shortest integration time")
    else :
        DIT = saturating_DIT
    if DIT > time :
        DIT = time
    # Mode de lecture en "rampe", on séquence la pose en plusieurs lectures non destructives afin de réduire le bruit de lecture (voir wiki average signal)
    if DIT > 4*min_DIT and instru != "MIRIMRS":  # on choisit 4 min_DIT car si les lectures intermittentes sont trop courtes, le détecteur va chauffer trop vite => + de dark current
        N_i = np.round(DIT/(4*min_DIT)) # nombre de lectures intermittentes
        RON_eff = RON/np.sqrt(N_i) # bruit de lecture effectif
    elif instru=='MIRIMRS' and DIT > min_DIT:
        N_i = np.round(DIT/(min_DIT)) # nombre de lectures intermittentes
        RON_eff = RON/np.sqrt(N_i) # bruit de lecture effectif
    else:
        RON_eff = RON
    if instru == 'ERIS' and RON_eff < 7:
        RON_eff = 7
    if print_value :
        print(" DIT =", round(DIT*60,2)," seconds / ", "RON =",round(RON_eff,3),"e-")
    return DIT, RON_eff


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
############################### CALCUL DU BRUIT SYSTEMATIQUE POUR MIRIMRS : ###########################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


def systematic_profile(config_data,band,sigma,Rc,R,star_spectrum_instru,planet_spectrum_instru,wave_inter,size_core,star_pos,used_filter):
    from src.molecular_mapping import crop, annular_mask, molecular_mapping_rv, stellar_high_filtering
    if star_pos=="center":
        file="data/MIRIMRS/MIRISim/star_center/star_center_s3d_"+band+"_all_corrected.fits"
        #file="data/MIRIMRS/MIRISim/star_center/star_center_s3d_"+band+"_with_fringes_straylight.fits"
    elif star_pos=="edge":
        file="data/MIRIMRS/MIRISim/star_edge/star_edge_s3d_"+band+"_with_fringes.fits" 
        file="data/MIRIMRS/MIRISim/star_edge/star_edge_s3d_"+band+"_all_corrected.fits"
    
    data = False
    #file = 'data/MIRIMRS/MAST/HD159222_ch'+band[0]+'-shortmediumlong_s3d.fits' ; data = True


    f = fits.open(file)
    S_noiseless = crop(f[1].data) # en MJy/str
    hdr = f[1].header
    delta_lambda = hdr['CDELT3']
    wave = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']
    S_noiseless = S_noiseless[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
    wave = wave[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
    pixscale = hdr['CDELT1']*3600 # pixel scale in "/px
    pxsteradian = hdr['PIXAR_SR'] # Nominal pixel area in steradians 
    S_noiseless *= pxsteradian*1e6 # en Jy
    S_noiseless *= 1e-26 # J/s/m²/Hz
    for i in range(S_noiseless.shape[0]):
        S_noiseless[i] *= c/((wave[i]*1e-6)**2)# J/s/m²/m
    S_noiseless *= delta_lambda*1e-6 # J/s/m²
    for i in range(S_noiseless.shape[0]):
        S_noiseless[i] *= wave[i]*1e-6/(h*c) # photons/s/m²
    S_noiseless *= config_data['telescope']['area'] # photons/s
    wave_trans,trans = fits.getdata("sim_data/Transmission/MIRIMRS/Instrumental_transmission/transmission_"+band+".fits")
    f = interp1d(wave_trans, trans, bounds_error=False, fill_value=np.nan) 
    trans = f(wave)
    for i in range(S_noiseless.shape[0]):
        S_noiseless[i] *= trans[i] # e-/s
    S_noiseless *= 60 # e-/mn
    NbChannel, NbLine, NbColumn = S_noiseless.shape # => donne (taille de l'axe lambda, " de l'axe y , " de l'axe x)
    
    star_spectrum_inter = star_spectrum_instru.degrade_resolution(wave,renorm=True)
    star_flux = trans*star_spectrum_inter.flux # gammaS* en e-/mn
    sep = np.zeros((int(round(config_data["spec"]["FOV"]/(2*pixscale)))+1))
    sigma_syst_2 = np.zeros((len(sep),len(wave_inter))) # array 2D de taille 2x Nbline (ou Column) /2
    sigma_syst_prime_2 = np.zeros((len(sep)))
    M_HF = np.zeros((len(sep),len(wave_inter))) # array 2D de taille 2x Nbline (ou Column) /2
    input_flux = np.nansum(S_noiseless,(1,2)) # m_S* en e-/mn du cube simulé sans bruits
    
    if not data and 1==1 : 
        star_data = np.loadtxt('/home/martoss/mirisim/spectra/star_6000_mag7_J.txt',skiprows=1) ; spectrum = Spectrum(star_data[:,0],star_data[:,1],None,None) ; spectrum = spectrum.degrade_resolution(wave,renorm=False) ; spectrum = spectrum.set_nbphotons_min(config_data, wave) ; spectrum.flux *= trans # en e-/mn
        input_flux = spectrum.flux*np.nanmean(input_flux)/np.nanmean(spectrum.flux) # S_star
        
    M_m_S = np.zeros_like(S_noiseless)+np.nan
    for i in range(NbChannel):
        M_m_S[i] = S_noiseless[i]/input_flux[i] * star_flux[i]
    
    if Rc is not None :
        S_res,_ = stellar_high_filtering(c=M_m_S,calculation="contrast",renorm_cube_res=False,R=R,Rc=Rc,used_filter=used_filter,print_value=False,cosmic=True,sigma_cosmic=6)
    else :
        S_res = M_m_S
    s_res = np.zeros_like(S_res)+np.nan
    for i in range(NbLine):
        for j in range(NbColumn): # pour chaque spaxel
            s_res[:,i,j] = np.nansum(S_res[:,i-size_core//2:i+size_core//2+1,j-size_core//2:j+size_core//2+1],axis=(1,2))
    s_res[s_res==0]=np.nan ; S_res = s_res
    CCF,_ = molecular_mapping_rv(instru=config_data["name"],band=band,cube_high_filtered=S_res,T=planet_spectrum_instru.T,lg=planet_spectrum_instru.lg,model=planet_spectrum_instru.model,wave=wave,trans=trans,calculation="contrast",R=R,Rc=Rc,used_filter=used_filter,broad=0,rv=planet_spectrum_instru.delta_rv,print_value=False)
    for r in range(1, len(sep)+1):
        sep[r-1] = r*pixscale
        ccf = np.copy(CCF)*annular_mask(r-1,r,size=(NbLine, NbColumn)) # anneau du cube M à la séparation r
        if not all(np.isnan(ccf.flatten())):
            m_HF_S = np.copy(S_res)*annular_mask(r-1,r,size=(NbLine, NbColumn)) # anneau du cube M à la séparation r
            f = interp1d(wave, np.nanvar(m_HF_S,axis=(1,2))/np.nansum(star_flux)**2, bounds_error=False, fill_value=np.nan)
            sigma_syst_2[r-1,:] = f(wave_inter)
            sigma_syst_prime_2[r-1] = np.nanvar(ccf)/np.nansum(star_flux)**2 # bruit systématique en e-/flux stellaire total 
            f = interp1d(wave, S_res[:,NbLine//2+1-r,NbColumn//2+1-r]/star_flux, bounds_error=False, fill_value=np.nan)
            M_HF[r-1,:] = f(wave_inter)
    Mp = np.nanmean(M_m_S[:,NbLine//2-1:NbLine//2+2,NbColumn//2-1:NbColumn//2+2],axis=(1,2))/star_flux # car son somme le signal sur la FWHM
    Mp /= np.nanmean(Mp)
    #Mp = fits.getdata("utils/Mp/Mp_"+band+"_"+str(planet_spectrum_instru.T)+"K.fits")
    f = interp1d(wave, Mp, bounds_error=False, fill_value=np.nan)
    Mp = f(wave_inter)
    #plt.figure() ; plt.plot(wave_inter,Mp) ; plt.title(f"{band}") ; plt.show()
    
    
    #sig = np.zeros_like(sigma_syst_prime_2) + np.nan
    #f = interp1d(wave, template, bounds_error=False, fill_value=np.nan)
    #template = f(wave_inter)
    #for i in range(len(sep)):
        #sig[i] = np.nansum(sigma_syst_2[i,:]*template**2)
    #plt.figure() ; plt.plot(sep,sigma_syst_prime_2) ; plt.plot(sep,sig) ; plt.yscale('log') ; plt.show()
    #plt.figure() ; plt.plot(sep,sigma_syst_prime_2) ; plt.plot(sep,np.nanmean(sigma_syst_2,1)) ; plt.yscale('log') ; plt.show()
    #sigma_syst_prime_2 = sig
    
    
    return sigma_syst_prime_2,sigma_syst_2,sep,M_HF,Mp,wave



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
########################################### PARTIE SPECTRE PICASO : ###################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



def import_picaso():
    try: 
        os.environ['picaso_refdata'] = '/home/martoss/picaso/reference/'
        os.environ['PYSYN_CDBS'] = '/home/martoss/picaso/grp/redcat/trds/'
        import picaso
        from picaso import justdoit as jdi
    except ImportError:
        print("Tried importing picaso, but couldn't do it")
    return picaso,jdi



def simulate_picaso_spectrum(instru,planet_table_entry,spectrum_contributions='thermal+reflected',opacity=None,wave=None,vega_spectrum=None,planet_type="gas",clouds=True,planet_mh=1,stellar_mh=0.0122,plot=False,save=False,in_im_mag=True):
    '''
    A function that returns the required inputs for picaso, 
    given a row from a universe planet table

    Inputs:
    planet_table_entry - a single row, corresponding to a single planet from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice", or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    planet_mh - planetary metalicity. 1 = 1x Solar
    stellar_mh - stellar metalicity

Opacity class from justdoit.opannection
    
    NOTE: this assumes a planet phase of 0. You can change the phase in the resulting params object afterwards.
    '''
    
    planet_table_entry["Phase"] = 0.0 * u.rad # Albedo géométrique
    picaso,jdi=import_picaso()

    config_data = get_config_data(instru)
    lmin_instru = config_data["lambda_range"]["lambda_min"] # en µm
    lmax_instru = config_data["lambda_range"]["lambda_max"] # en µm
    # For K band
    lmin_K = 1.974 ; lmax_K = 2.384
    
    if opacity is None:
        wvrng = [0.98*min(lmin_K,lmin_instru),1.02*max(lmax_K,lmax_instru)]
        #wvrng = [0.6,6]
        # opacity file to load
        opacity_folder = os.path.join(os.getenv("picaso_refdata"),'opacities')
        dbname = 'all_opacities_0.6_6_R60000.db' # lambda va de 0.6 à 6µm (mais indiqué 0.3 à 15µm)
        dbname = os.path.join(opacity_folder,dbname)
        # molecules, pt_pairs = opa.molecular_avail(dbname) ; print("\n molecules considérées : \n ", molecules)
        opacity = jdi.opannection(filename_db=dbname,wave_range=wvrng)
    
    #-- Define the grids that phoenix / ckmodel models like
    host_temp_list=np.hstack([np.arange(3500,13000,250),np.arange(13000,50000,1000)])
    host_logg_list=[5.00,4.50,4.00,3.50,3.00,2.50,2.00,1.50,1.00,0.50,0.0]
    f_teff_grid=interp1d(host_temp_list,host_temp_list,kind='nearest',bounds_error=False,fill_value='extrapolate')
    f_logg_grid=interp1d(host_logg_list,host_logg_list,kind='nearest',bounds_error=False,fill_value='extrapolate')
    planet_table_entry['StarTeff'] = f_teff_grid(planet_table_entry['StarTeff']) *planet_table_entry['StarTeff'].unit
    planet_table_entry['StarLogg'] = f_logg_grid(planet_table_entry['StarLogg']) *planet_table_entry['StarLogg'].unit
    if planet_type != "gas":
        print("Only planet_type='gas' spectra are currently implemented")
        print("Generating a gas-like spectrum")
        planet_type = 'gas'
    params = jdi.inputs() ; params.approx(raman='none') # voir justdoit.py => class inputs():
    # Note: non-0 phase in reflectance requires a different geometry so we'll deal with that in the simulate_spectrum() call
    params.phase_angle(planet_table_entry['Phase'].value)
    # NOTE: picaso gravity() won't use the "gravity" input if mass and radius are provided
    params.gravity(gravity=planet_table_entry['PlanetLogg'].value,gravity_unit=planet_table_entry['PlanetLogg'].physical.unit)#,mass=planet_table_entry['PlanetMass'].value,mass_unit=planet_table_entry['PlanetMass'].unit,radius=planet_table_entry['PlanetRadius'].value,radius_unit=planet_table_entry['PlanetRadius'].unit)
    # The current stellar models do not like log g > 5, so we'll force it here for now. 
    star_logG = planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value
    if star_logG > 5.0:
        star_logG = 5.0
    #The current stellar models do not like Teff < 3000, so we'll force it here for now. 
    star_Teff = planet_table_entry['StarTeff'].to(u.K).value
    if star_Teff < 3000:
        star_Teff = 3000   
    params.star(opacity, star_Teff, stellar_mh, star_logG,radius=planet_table_entry['StarRad'].value,semi_major=planet_table_entry['SMA'].value, semi_major_unit=planet_table_entry['SMA'].unit, radius_unit=planet_table_entry['StarRad'].unit) 
    #-- Define atmosphere PT profile, mixing ratios, and clouds
    if planet_type == 'gas':
        params.guillot_pt(planet_table_entry['PlanetTeq'].value, T_int=150, logg1=-0.5, logKir=-1)
        # T_int = Internal temperature / logg1,logKir = see parameterization Guillot 2010
        params.channon_grid_high() # get chemistry via chemical equillibrium
        if clouds : # may need to consider tweaking these for reflected light
            params.clouds(g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5])
            # g0 = Asymmetry factor / w0 = Single Scattering Albedo / opd = Total Extinction in `dp` / p = Bottom location of cloud deck (LOG10 bars) / dp = Total thickness cloud deck above p (LOG10 bars)
    elif planet_type == 'terrestrial':
        # TODO: add Terrestrial type
        pass
    elif planet_type == 'ice':
        # TODO: add ice type
        pass
    # Make sure that picaso wavelengths are within requested wavelength range
    op_wv = opacity.wave # this is identical to the model_wvs we compute below  
    # non-0 phases require special geometry which takes longer to run.
      # To improve runtime, we always run thermal with phase=0 and simple geom.
      # and then for non-0 phase, we run reflected with the costly geometry
    phase = planet_table_entry['Phase'].value
    if phase == 0:
        # Perform the simple simulation since 0-phase allows simple geometry
        df = params.spectrum(opacity,full_output=True,calculation=spectrum_contributions,plot_opacity=False)
    else:
        # Perform the thermal simulation as usual with simple geometry
        df1 = params.spectrum(opacity,full_output=True,calculation='thermal')
        # Apply the true phase and change geometry for the reflected simulation
        params.phase_angle(phase, num_tangle=8, num_gangle=8)
        df2 = params.spectrum(opacity,full_output=True,calculation='reflected')
        # Combine the output dfs into one df to be returned
        df = df1.copy(); df.update(df2)
        df['full_output_therm'] = df1.pop('full_output')
        df['full_output_ref'] = df2.pop('full_output')
        
    model_wvs = 1./df['wavenumber'] * 1e4 *u.micron ; argsort = np.argsort(model_wvs) ; model_wvs = model_wvs[argsort]
    model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1)) ; model_dwvs[0] = model_dwvs[1] ; model_R = np.nanmean(model_wvs/(2*model_dwvs)) ; model_R = model_R.value # = 30000
    
    if not save :
        R = 200000 # environ la résolution des spectres BT-Settl / BT-NextGen
        if wave is None :
            delta_lambda = ((max(lmax_K,lmax_instru)+min(lmin_K,lmin_instru))/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
            wave = np.arange(0.98*min(lmin_K,lmin_instru),1.02*max(lmax_K,lmax_instru), delta_lambda)
        
        if vega_spectrum is None :
            vega_spectrum = load_vega_spectrum()
            vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave, renorm = False)
            
        star_spectrum = load_star_spectrum(float(planet_table_entry["StarTeff"].value),float(planet_table_entry["StarLogg"].value))
        star_spectrum = star_spectrum.interpolate_wavelength(star_spectrum.flux, star_spectrum.wavelength, wave, renorm = False)
        star_spectrum.flux *= np.nanmean(vega_spectrum.flux[(wave>lmin_instru)&(wave<lmax_instru)])*10**(-0.4*float(planet_table_entry["StarINSTRUmag("+instru+")"])) / np.nanmean(star_spectrum.flux[(wave>lmin_instru) & (wave<lmax_instru)])
        
        if "thermal" in spectrum_contributions :
            planet_thermal = df["thermal"][argsort] * u.erg/u.s/u.cm**2/u.cm # On trouve un facteur entre le flux thermique de PICASO et de BT-Settl ou Exo-REM, (je ne sais pas vraiment d'où il vient mais je suppose que c'est du au fait qu'il s'agit d'un flux thermique "géométrique")
            planet_thermal = planet_thermal.to(u.J/u.s/u.m**2/u.micron)
            planet_thermal = Spectrum(np.array(model_wvs.value),np.array(planet_thermal.value),model_R,float(planet_table_entry["PlanetTeq"].value),float(planet_table_entry["PlanetLogg"].value),"PICASO")
            planet_thermal = planet_thermal.interpolate_wavelength(planet_thermal.flux, planet_thermal.wavelength, wave, renorm = False)
            planet_thermal.flux *= float((planet_table_entry['PlanetRadius']/planet_table_entry['Distance']).decompose().value)**2
    
        if "reflected" in spectrum_contributions :
            model_alb = df['albedo'][argsort]
            f = interp1d(model_wvs.value, model_alb,bounds_error=False,fill_value=np.nan)
            model_alb = f(wave)
            planet_reflected = star_spectrum.flux * model_alb * planet_table_entry["g_alpha"] * (planet_table_entry['PlanetRadius']/planet_table_entry['SMA']).decompose()**2
            planet_reflected = Spectrum(wave,np.array(planet_reflected.value),R,float(planet_table_entry["PlanetTeq"].value),float(planet_table_entry["PlanetLogg"].value),"PICASO")
        
        if spectrum_contributions == "thermal+reflected":
            planet_spectrum = Spectrum(wave,planet_thermal.flux+planet_reflected.flux,R,planet_thermal.T,planet_thermal.lg,"PICASO")
        elif spectrum_contributions == "thermal":
            planet_spectrum = Spectrum(wave,planet_thermal.flux,R,planet_thermal.T,planet_thermal.lg,"PICASO")
        elif spectrum_contributions == "reflected":
            planet_spectrum = Spectrum(wave,planet_reflected.flux,R,float(planet_table_entry["PlanetTeq"].value),float(planet_table_entry["PlanetLogg"].value),"PICASO")
        
        if in_im_mag and planet_table_entry["DiscoveryMethod"]=="Imaging" and not np.isnan(planet_table_entry["PlanetKmag("+spectrum_contributions+")"]):
            ratio = np.nanmean(vega_spectrum.flux[(wave>lmin_K)&(wave<lmax_K)])*10**(-0.4*float(planet_table_entry["PlanetKmag("+spectrum_contributions+")"])) / np.nanmean(planet_spectrum.flux[(wave>lmin_K)&(wave<lmax_K)])
            planet_spectrum.flux = np.copy(planet_spectrum.flux) * ratio
            if "thermal" in spectrum_contributions :
                planet_thermal.flux = np.copy(planet_thermal.flux) * ratio
            if "reflected" in spectrum_contributions :
                planet_reflected.flux = np.copy(planet_reflected.flux) * ratio
        
        mag_p_total = -2.5*np.log10(np.nanmean(planet_spectrum.flux[(wave>(lmin_K))&(wave<(lmax_K))])/np.nanmean(vega_spectrum.flux[(wave>(lmin_K))&(wave<(lmax_K))]))
        if "thermal" in spectrum_contributions :
            mag_p_thermal = -2.5*np.log10(np.nanmean(planet_thermal.flux[(wave>(lmin_K))&(wave<(lmax_K))])/np.nanmean(vega_spectrum.flux[(wave>(lmin_K))&(wave<(lmax_K))]))
        if "reflected" in spectrum_contributions :
            mag_p_reflected = -2.5*np.log10(np.nanmean(planet_reflected.flux[(wave>(lmin_K))&(wave<(lmax_K))])/np.nanmean(vega_spectrum.flux[(wave>(lmin_K))&(wave<(lmax_K))]))
        if plot :
            plt.figure() ; plt.grid(True) ; plt.xlabel("wavelength (in µm)",fontsize=14) ; plt.ylabel(f"flux (in J/s/m²/µm)",fontsize=14) ; plt.yscale('log') ; plt.title(f"The different spectrum contributions \n for {planet_table_entry['PlanetName']} at {round(float(planet_spectrum.T))}K (on the same spectral resolution)",fontsize=16)
            if spectrum_contributions=="thermal+reflected" :
                plt.plot(wave,planet_spectrum.flux,'g',label=f"thermal+reflected (PICASO), mag(K) = {round(float(mag_p_total),2)}")
            if "thermal" in spectrum_contributions :
                plt.plot(planet_thermal.wavelength,planet_thermal.flux,'r',label=f"thermal (PICASO), mag(K) = {round(float(mag_p_thermal),2)}")
            if "reflected" in spectrum_contributions :
                plt.plot(wave,planet_reflected.flux,'b',label=f"reflected (PICASO), mag(K) = {round(float(mag_p_reflected),2)}")
            plt.axvspan(lmin_K, lmax_K, color='k', alpha=0.5, lw=0,label="K-band")
            plt.legend()
            
        return star_spectrum , planet_spectrum , mag_p_total

    else :
        if spectrum_contributions == "thermal":
            planet_thermal = np.zeros((2, len(model_wvs))) ; planet_thermal[0] = model_wvs
            thermal_flux = df["thermal"][argsort] * u.erg/u.s/u.cm**2/u.cm
            thermal_flux = thermal_flux.to(u.J/u.s/u.m**2/u.micron)
            planet_thermal[1] = np.array(thermal_flux.value)
            fits.writeto(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{round(float(planet_table_entry['PlanetTeq'].value))}K_lg{round(float(planet_table_entry['PlanetLogg'].value),1)}.fits",planet_thermal,overwrite=True)
            plt.figure() ; plt.plot(planet_thermal[0],planet_thermal[1]) ; plt.title(f'{round(float(planet_table_entry["PlanetTeq"].value))}K and lg = {round(float(planet_table_entry["PlanetLogg"].value),1)}') ; plt.xlabel('wavelength (in µm)') ; plt.ylabel("flux (in J/s/µm/m²)") ; plt.yscale('log') ; plt.show()
        elif spectrum_contributions == "reflected":
            albedo = np.zeros((2, len(model_wvs))) ; albedo[0] = model_wvs ; albedo[1] = df['albedo'][argsort]
            fits.writeto(f"sim_data/Spectra/planet_spectrum/albedo/albedo_gas_giant_{round(float(planet_table_entry['PlanetTeq'].value))}K_lg{round(float(planet_table_entry['PlanetLogg'].value),1)}.fits",albedo,overwrite=True)
            plt.figure() ; plt.plot(albedo[0],albedo[1]) ; plt.title(f'{round(float(planet_table_entry["PlanetTeq"].value))}K and lg = {round(float(planet_table_entry["PlanetLogg"].value),1)}') ; plt.xlabel('wavelength (in µm)') ; plt.ylabel("albedo") ; plt.show()
       



def get_picasso_thermal(name_planet="HR 8799 b"):
    from src.FastYield import load_planet_table,planet_index
    picaso,jdi=import_picaso()
    wvrng = [0.6,6]
    # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"),'opacities')
    dbname = 'all_opacities_0.6_6_R60000.db' # lambda va de 0.6 à 6µm (mais indiqué 0.3 à 15µm)
    dbname = os.path.join(opacity_folder,dbname)
    # molecules, pt_pairs = opa.molecular_avail(dbname) ; print("\n molecules considérées : \n ", molecules)
    opacity = jdi.opannection(filename_db=dbname,wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx = planet_index(planet_table,name_planet)
    T0 = np.append(np.arange(500,1000,50),np.arange(1000,3100,100))
    T0 = np.append([200,220,240,260,280,300,320,340,360,380,400,450],T0)
    k=0
    for T in T0 :
        if T < 260 :
            lg0 = np.array([4.0])
        elif T == 260 :
            lg0 = np.array([3.5])
        elif T <= 300 :
            lg0 = np.array([3.0,3.5])
        elif T < 500 :
            lg0 = np.array([2.5,3.0,3.5])
        else :
            lg0 = np.array([3.0,3.5,4.0,4.5,5.0])
        for lg in lg0 :
            k += 1
            
            planet_table[idx]["PlanetTeq"] = T * planet_table[idx]["PlanetTeq"].unit # 
            planet_table[idx]["PlanetLogg"] = lg * planet_table[idx]["PlanetLogg"].unit # 
            simulate_picaso_spectrum("HARMONI",planet_table[idx],spectrum_contributions="thermal",opacity=opacity,plot=False,save=True)
            print(round(100*(k+1)/177,2),"%")
            
def get_picasso_albedo(name_planet="HR 8799 b"):
    from src.FastYield import load_archive_table,planet_index
    picaso,jdi=import_picaso()
    wvrng = [0.6,6]
    # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"),'opacities')
    dbname = 'all_opacities_0.6_6_R60000.db' # lambda va de 0.6 à 6µm (mais indiqué 0.3 à 15µm)
    dbname = os.path.join(opacity_folder,dbname)
    # molecules, pt_pairs = opa.molecular_avail(dbname) ; print("\n molecules considérées : \n ", molecules)
    opacity = jdi.opannection(filename_db=dbname,wave_range=wvrng)
    planet_table = load_archive_table("Archive_Pull_for_FastCurves.ecsv")
    idx = planet_index(planet_table,name_planet)
    T0 = np.append(np.arange(500,1000,50),np.arange(1000,3100,100))
    T0 = np.append([200,220,240,260,280,300,320,340,360,380,400,450],T0)
    k=0
    for T in T0 :
        if T < 260 :
            lg0 = np.array([4.0])
        elif T == 260 :
            lg0 = np.array([3.5])
        elif T <= 300 :
            lg0 = np.array([3.0,3.5])
        elif T < 500 :
            lg0 = np.array([2.5,3.0,3.5])
        else :
            lg0 = np.array([3.0,3.5,4.0,4.5,5.0])
        for lg in lg0 :
            k += 1
            planet_table[idx]["PlanetTeq"] = T * planet_table[idx]["PlanetTeq"].unit # 
            planet_table[idx]["PlanetLogg"] = lg * planet_table[idx]["PlanetLogg"].unit # 
            simulate_picaso_spectrum("HARMONI",planet_table[idx],spectrum_contributions="reflected",opacity=opacity,plot=False,save=True)
            print(round(100*(k+1)/177,2),"%")
            
def load_albedo(T,lg):
    T0 = np.append(np.arange(500,1000,50),np.arange(1000,3100,100))
    T0 = np.append([200,220,240,260,280,300,320,340,360,380,400,450],T0)
    idx = (np.abs(T0 - T)).argmin()
    T=T0[idx]
    if T < 260 :
        lg0 = np.array([4.0])
    elif T == 260 :
        lg0 = np.array([3.5])
    elif T <= 300 :
        lg0 = np.array([3.0,3.5])
    elif T < 500 :
        lg0 = np.array([2.5,3.0,3.5])
    else :
        lg0 = np.array([3.0,3.5,4.0,4.5,5.0])
    idx = (np.abs(T0 - lg)).argmin()
    lg=lg0[idx]
    wave,albedo = fits.getdata(f"sim_data/Spectra/planet_spectrum/albedo/albedo_gas_giant_{T}K_lg{lg}.fits")
    dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] # array de delta Lambda
    R = np.nanmean(wave/(2*dwl)) # calcule de la nouvelle résolution
    albedo = Spectrum(wave,albedo,R,T,lg,"PICASO")
    return albedo



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
########################################### PARTIE FASTYIELD : ###########################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################



def thermal_reflected_spectrum(planet,instru=None,thermal_model="BT-Settl",reflected_model="PICASO",wave=None,vega_spectrum=None,show=True,in_im_mag=True):
    lmin_K = 1.951 ; lmax_K = 2.469
    if thermal_model == "None" and reflected_model == "None" :
        raise KeyError("PLEASE DEFINE A MODEL FOR THE THERMAL OR THE REFLECTED COMPONENT !")
    if instru is not None :
        config_data = get_config_data(instru)
        lmin_instru = config_data["lambda_range"]["lambda_min"] # en µm
        lmax_instru = config_data["lambda_range"]["lambda_max"] # en µm
        if wave is None :
            config_data = get_config_data(instru)
            R = 200000 # environ la résolution des spectres BT-Settl / BT-NextGen
            delta_lamb_instru = ((max(lmax_K,lmax_instru)+min(lmin_K,lmin_instru))/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
            wave = np.arange(0.98*min(lmin_K,lmin_instru), 1.02*max(lmax_K,lmax_instru), delta_lamb_instru)
    if vega_spectrum is None :
        vega_spectrum = load_vega_spectrum()
        vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave, renorm = False)
    
    star_spectrum = load_star_spectrum(float(planet["StarTeff"].value),float(planet["StarLogg"].value))
    star_spectrum = star_spectrum.interpolate_wavelength(star_spectrum.flux, star_spectrum.wavelength, wave, renorm = False)
    
    star_spectrum.flux *= np.nanmean(vega_spectrum.flux[(wave>lmin_K)&(wave<lmax_K)])*10**(-0.4*float(planet["StarKmag"])) / np.nanmean(star_spectrum.flux[(wave>lmin_K) & (wave<lmax_K)])

    if thermal_model != "None":
        planet_thermal = load_planet_spectrum(float(planet["PlanetTeq"].value),float(planet["PlanetLogg"].value),model=thermal_model)
        planet_thermal = planet_thermal.interpolate_wavelength(planet_thermal.flux, planet_thermal.wavelength, wave, renorm = False)
        planet_thermal.flux *= float((planet['PlanetRadius']/planet['Distance']).decompose()**2)
    elif thermal_model == "None":
        planet_thermal = Spectrum(wave,np.zeros_like(wave),star_spectrum.R,float(planet["PlanetTeq"].value),float(planet["PlanetLogg"].value),thermal_model)

    albedo = load_albedo(planet_thermal.T,planet_thermal.lg)
    albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, wave, renorm = False)
    if reflected_model == "PICASO":
        planet_reflected = star_spectrum.flux * albedo.flux * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "flat":
        planet_reflected = star_spectrum.flux * np.nanmean(albedo.flux) * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "tellurics":
        wave_tell,tell = fits.getdata("sim_data/Transmission/sky_transmission_airmass_2_5.fits")
        f = interp1d(wave_tell,tell,bounds_error=False,fill_value=np.nan)
        tell = f(wave)
        planet_reflected = star_spectrum.flux * np.nanmean(albedo.flux)/np.nanmean(tell)*tell * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "None":
        planet_reflected = np.zeros_like(wave)*u.dimensionless_unscaled
    planet_reflected = Spectrum(wave,np.array(planet_reflected.value),max(star_spectrum.R,albedo.R),albedo.T,float(planet["PlanetLogg"].value),reflected_model)
    
    planet_spectrum = Spectrum(wave,planet_thermal.flux+planet_reflected.flux,max(planet_thermal.R,planet_reflected.R),planet_thermal.T,planet_thermal.lg,thermal_model+"+"+reflected_model)
    
    if in_im_mag and planet["DiscoveryMethod"]=="Imaging" :
        if not np.isnan(planet["PlanetKmag(thermal+reflected)"]) : # On connait déjà leur magnitude par définition
            ratio = np.nanmean(vega_spectrum.flux[(vega_spectrum.wavelength>lmin_K)&(vega_spectrum.wavelength<lmax_K)])*10**(-0.4*float(planet["PlanetKmag(thermal+reflected)"])) / np.nanmean(planet_spectrum.flux[(planet_spectrum.wavelength>lmin_K) & (planet_spectrum.wavelength<lmax_K)])
            planet_spectrum.flux = np.copy(planet_spectrum.flux) * ratio
            planet_thermal.flux = np.copy(planet_thermal.flux) * ratio
            planet_reflected.flux = np.copy(planet_reflected.flux) * ratio

        
    if show : 
        lmin = lmin_K ; lmax = lmax_K ; band0 = "K"
        mag_p_total = -2.5*np.log10(np.mean(planet_spectrum.flux[(wave>lmin)&(wave<lmax)])/np.nanmean(vega_spectrum.flux[(wave>lmin) & (wave<lmax)]))
        plt.figure() ; plt.xlabel("wavelength (in µm)",fontsize=14) ; plt.ylabel(f"flux (in J/s/m²/µm)",fontsize=14) ; plt.yscale('log') ; plt.title(f"The different spectrum contributions \n for {planet['PlanetName']} at {round(float(planet_spectrum.T))}K (on the same spectral resolution)",fontsize=16)
        if thermal_model!="None" and reflected_model!="None":
            plt.plot(wave,planet_spectrum.flux,'g',label=f"thermal+reflected ({planet_spectrum.model}), mag("+band0+f") = {round(mag_p_total,2)}")
        if thermal_model != "None":
            mag_p_thermal = -2.5*np.log10(np.mean(planet_thermal.flux[(wave>lmin)&(wave<lmax)])/np.nanmean(vega_spectrum.flux[(wave>lmin) & (wave<lmax)])) 
            plt.plot(wave,planet_thermal.flux,'r',label=f"thermal ({thermal_model}), mag("+band0+f") = {round(mag_p_thermal,2)}")
        if reflected_model != "None":
            mag_p_reflected = -2.5*np.log10(np.mean(planet_reflected.flux[(wave>(lmin))&(wave<(lmax))])/np.nanmean(vega_spectrum.flux[(wave>(lmin))&(wave<(lmax))]))
            plt.plot(wave,planet_reflected.flux,'b',label=f"reflected ({reflected_model}+BT-NextGen), mag("+band0+f") = {round(mag_p_reflected,2)}")
        if np.nanmin(wave) < lmin_K:
            plt.axvspan(lmin_K, lmax_K, color='k', alpha=0.5, lw=0,label="K-band")
        plt.legend()
        #plt.ylim(np.nanmin(planet_spectrum.flux),np.nanmax(planet_spectrum.flux))
    
    return planet_spectrum , planet_thermal , planet_reflected , star_spectrum









