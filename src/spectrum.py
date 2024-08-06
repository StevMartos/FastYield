from src.utils import *

path_file = os.path.dirname(__file__)
load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")
vega_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/star_spectrum/VEGA_Fnu.fits")



class Spectrum:

    def __init__(self, wavelength, flux, R, T, lg=None, model=None, star_rv=0, delta_rv=0, high_pass_flux=None):
        """
        Parameters
        ----------
        wavelength: 1d array
            wavelength axis
        flux: 1d array
            flux axis
        R: float
            spectral resolution of the spectrum
        T: float
            temperature of the spectrum
        """
        self.wavelength = wavelength # wavelength axis of the spectrum
        self.flux = flux # flux of the spectrum
        self.R = R # resolution of the spectrum
        self.T = T # temperature of the spectrum
        self.lg = lg # surface gravity of the spectrum
        self.model = model  # model of the spectrum
        self.star_rv = star_rv # radial velocity of the system (star)
        self.delta_rv = delta_rv # Doppler shift between the planet and the star
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def crop(self, lmin, lmax):
        """
        Crop the spectrum between lmin and lmax (and calculate the new spectral resolution)

        Parameters
        ----------
        lmin: float (in µm)
            lambda min value
        lmax: float (in µm)
            lambda max value
        """
        self.flux = self.flux[(self.wavelength >= lmin) & (self.wavelength <= lmax)]
        self.wavelength = self.wavelength[(self.wavelength >= lmin) & (self.wavelength <= lmax)]
        dwl = self.wavelength - np.roll(self.wavelength, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda array
        Rnew = np.nanmean(self.wavelength/(2*dwl)) # calculating the new resolution (2*R => Nyquist sampling / Shannon)
        self.R = Rnew
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def set_flux(self, nbPhotons):
        """
        Renormalize / Convert flux to photon number (or other)
        The flux can be in density but must have a constant delta_lambda.
        
        Parameters
        ----------
        nbPhotons: float/int
            quantity in a certain unit with which to renormalize the flux
        """
        self.flux = nbPhotons * self.flux / np.nansum(self.flux)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_nbphotons_min(self, config_data, wave):
        """
        Converts a spectrum initially in density (J/s/m²/µm) into nb of photons/min received by the telescope over the wave range
        !!! wave MUST HAVE A CONSTANT delta_lambda !!!  
            
        Parameters
        ----------
        config_data: collections
            gives the parameters of the considered instrument
        wave: array (must be in µm)
            wavelength axis on which the spectrum is converted

        Returns
        -------
        spectrum: class Spectrum
            spectrum converted into nb of photons/min
        """
        dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda en µm
        Rnew = np.nanmean(wave/(2*dwl)) # calculating the new resolution
        spectrum_flux = self.interpolate_wavelength(self.flux, self.wavelength, wave, renorm=False).flux # reinterpolating the flux (in density) on wave
        spectrum = Spectrum(wave, spectrum_flux, Rnew, self.T, self.lg, self.model, star_rv=self.star_rv, delta_rv=self.delta_rv)
        spectrum.flux = spectrum.flux*wave*1e-6/(h*c) # J/s/m²/µm => photons/s/m2/µm
        S = config_data["telescope"]["area"] # telescope collector area in m2
        spectrum.flux = spectrum.flux*S*dwl*60 # in photons/mn
        return spectrum

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def degrade_resolution(self, wave_output, renorm=True, gaussian_filtering=True):
        """
        Degrade the spectral resolution at the new resolution (of wave_output) with a convolution
        ! Does not work if the new resolution is higher than the basic resolution !

        Parameters
        ----------
        wave_output: array
            new wavelength axis (new resolution)
        renorm: bool, optional
            for renormalisation to conserve the flux (True => flux must not be in density eg. J/s/m²/µm) . The default is True.

        Returns
        -------
        spectrum: class Spectrum
            degrated spectrum
        """
        valid = np.where((self.wavelength >= wave_output[0]) & (self.wavelength <= wave_output[-1])) # flux[valid] => returns a (smaller) array that stores flux values for a wavelength between wave_output[0] and wave_output[-1].
        flux_tot = np.nansum(self.flux[valid]) # total flux in the spectrum (for the renormalization, if wanted)
        if gaussian_filtering: # convolution + down binning
            dwl = self.wavelength[valid] - np.roll(self.wavelength[valid], 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda array
            Rold = np.nanmax(self.wavelength[valid]/(2*dwl)) # old Resolution
            if Rold > 200000: # fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
                Rold = 200000
            dl = np.nanmean(self.wavelength[valid]/(2*Rold)) 
            wave_band = np.arange(0.98*wave_output[0], 1.02*wave_output[-1], dl) # constant and linear input wavelength array
            flr = self.interpolate_wavelength(self.flux, self.wavelength, wave_band, renorm=renorm).flux # reinterpolate the flux on wave_band
            dwl = wave_output - np.roll(wave_output, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda array
            Rnew = np.nanmean(wave_output/(2*dwl)) # calculating the new resolution
            fwhm = Rold/Rnew # https://github.com/spacetelescope/pysynphot/issues/78
            sigma_conv = fwhm # sigma width of the gaussian kernel for the convolution
            flr_conv = gaussian_filter(flr[~np.isnan(flr)], sigma=sigma_conv) # convoluted flux
            flr[~np.isnan(flr)] = flr_conv # ignoring the NaN values
            flr = cg.downbin_spec(flr, wave_band, wave_output, dlam=dwl) # down binned flux
        else: # only down binning
            dwl = wave_output - np.roll(wave_output, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda array
            Rnew = np.nanmean(wave_output/(2*dwl)) # calculating the new resolution
            flr = (cg.downbin_spec(self.flux, self.wavelength, wave_output, dlam=dwl)) # down binned flux
        if renorm: # conserve the flux
            return Spectrum(wave_output, flux_tot*flr/np.nansum(flr), Rnew, self.T, lg=self.lg, model=self.model, star_rv=self.star_rv, delta_rv=self.delta_rv)
        else: # does not conserve the flux (for transmission and density spectrum)
            return Spectrum(wave_output, flr, Rnew, self.T, lg=self.lg, model=self.model, star_rv=self.star_rv, delta_rv=self.delta_rv)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
    def interpolate_wavelength(self, influx, wave_input, wave_output, renorm=True, fill_value=np.nan):
        """
        Re-interpolates the flux on a new wavelength axis
        
        Parameters
        ----------
        influx: array (can be in density or not)
            actual flux (non-interpolated)
        wave_input: array 
            actual wavelength axis of the flux
        wave_output: array
            new wavelength axis for the interpolation
        renorm: bool, optional
            for renormalisation (True => flux must not be in density eg. J/s/m²/µm) . The default is True.

        Returns: class Spectrum
            Spectrum with the interpolated flux on the new wavelength axis
        """
        valid = np.where((self.wavelength >= wave_output[0]) & (self.wavelength <= wave_output[-1])) # flux[valid] => returns a (smaller) array that stores flux values for a wavelength between wave_output[0] and wave_output[-1].
        flux_tot = np.nansum(self.flux[valid]) # total flux in the spectrum (for the renormalization, if wanted)
        f = interp1d(wave_input, influx, bounds_error=False, fill_value=fill_value)
        #f = interp1d(wave_input[~np.isnan(influx)], influx[~np.isnan(influx)], bounds_error=False, fill_value=fill_value)
        flux_interp = f(wave_output) # interpolates flux values on the new axis (wave_output)
        dwl = wave_output - np.roll(wave_output, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda array
        Rnew = np.nanmean(wave_output/(2*dwl)) # calculating the new resolution
        if renorm: # conserve the flux
            spec = Spectrum(wave_output, flux_tot*flux_interp/np.nansum(flux_interp), Rnew, self.T, self.lg, self.model, star_rv=self.star_rv, delta_rv=self.delta_rv)
        else: # does not conserve the flux (for transmission and density spectrum)
            spec = Spectrum(wave_output, flux_interp, Rnew, self.T, self.lg, self.model, star_rv=self.star_rv, delta_rv=self.delta_rv)
        return spec
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def doppler_shift(self, rv, renorm=True, fill_value=np.nan):
        """
        Doppler shift of a spectrum as a function of radial velocity rv
        
        Parameters
        ----------
        rv: float/int (in km/s)
            radial velocity

        Returns: class Spectrum
            shifted spectrum
        """
        rv = rv * (u.km / u.s) # defines rv in km/s with astropy
        rv = rv.to(u.m / u.s) # convert rv to m/s with astropy 
        wshift = self.wavelength * (1 + (rv / const.c)) # const.c = speed of light (via astropy) / offset wavelength axis
        spec_rv = self.interpolate_wavelength(self.flux, wshift, self.wavelength, renorm=renorm, fill_value=fill_value)
        spec_rv.R = self.R        
        return spec_rv

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def broad(self, vsini, epsilon=0.8, fastbroad=True):
        """
        Broadens spectrum lines as a function of rotation speed (in km/s)

        Parameters
        ----------
        vsini: float
            Projected rotational velocity (in km/s).
        epsilon: float
            Linear limb-darkening coefficient (0-1).

        Returns: class Spectrum
            broadened spectrum
        """
        if fastbroad: # fast spectral broadening (but less accurate)
            flux = pyasl.fastRotBroad(self.wavelength*1e4, self.flux, epsilon=epsilon, vsini=vsini)
        else: # slow spectral broadening (but more accurate)
            flux = pyasl.rotBroad(self.wavelength[~np.isnan(self.flux)]*1e4, self.flux[~np.isnan(self.flux)], epsilon=epsilon, vsini=vsini) # ignoring NaN values at the same time
            f = interp1d(self.wavelength[~np.isnan(self.flux)], flux, bounds_error=False, fill_value=np.nan) 
            flux = f(self.wavelength) 
        return Spectrum(self.wavelength, flux, self.R, self.T, self.lg, self.model, star_rv=self.star_rv, delta_rv=self.delta_rv) # pas besoin de "conserver le nb de photons" car c'est un effet intrinsèque (on ne change pas largeur des bins)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def get_psd(self, smooth=1, crop=True):
        """
        Gives the PSD of the Spectrum

        Parameters
        ----------
        smooth: TYPE, optional
            Smooth the PSD by convolution with a gaussian. The default is 1.
        crop: TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        res: TYPE
            resolution array.
        PSD_smooth: TYPE
            DESCRIPTION.

        """
        if len(self.flux)%2!=0 and crop: # in order to have an odd value for the len of the resolution
            signal = np.copy(self.flux)[:-1]
        else:
            signal = np.copy(self.flux)
        N = len(signal)
        ffreq = np.fft.fftfreq(N)
        fft = np.abs(np.fft.fft(signal))**2
        if crop:
            PSD = (fft[:N//2] + np.roll(np.flip(fft[N//2:]), 1)) * (1/N)
            res = ffreq[:N//2]*self.R*2
            #plt.figure() ; plt.plot(res, np.sqrt(fft[:N//2])) ; plt.plot(res, np.sqrt(np.roll(np.flip(fft[N//2:]), 1))) ; plt.xscale('log') ; plt.yscale('log') ; plt.show()
        else:
            PSD = fft * (1/N)
            res = ffreq*self.R*2
        if smooth == 0:
            PSD = PSD
        else:
            PSD = gaussian_filter(PSD, sigma=smooth)
        return res, PSD


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_planet_spectrum(T, lg, model="BT-Settl", instru=None, load_path=load_path):
    """
    To read and retrieve planet spectra from models (see http://svo2.cab.inta-csic.es/theory/newov2/)

    Parameters
    ----------
    T: float/int
        planet temperature (in K)
    lg: float/int
        planet gravity surface (in dex(cm/s2))
    model: str
        spectrum mode
    load_path: str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns: class Spectrum
        loaded planet spectrum in J/s/m²/µm
    """
    
    if model == "BT-Settl": # https://articles.adsabs.harvard.edu/pdf/2013MSAIS..24..128A
        if T < 200:
            print("Changing the input temperature to the minimal temperature: 200K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature: 3000K.")
        T0 = np.append([200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 450], np.append(np.arange(500, 1000, 50), np.arange(1000, 3100, 100))) # available values
        idx = (np.abs(T0 - T)).argmin() 
        T = T0[idx] # closest available value 
        if T < 260: # lg available values as regard to the temperature value
            lg0 = np.array([4.0])
        elif T == 260:
            lg0 = np.array([3.5])
        elif T <= 300:
            lg0 = np.array([3.0, 3.5])
        elif T < 500:
            lg0 = np.array([2.5, 3.0, 3.5])
        else:
            lg0 = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        idx = (np.abs(lg0 - lg)).argmin()
        lg = lg0[idx] # closest lg value
        if T >= 1000: # convert to file name compatible value
            str_T = "0"+str(T)[:2] ; str_lg = str(lg) 
        else:
            str_T = "00"+str(T)[0] ; str_lg = str(lg)
        if str(T)[-2]!="0":
            str_T += "."+str(T)[-2]
        wave, flux = fits.getdata(load_path+'/planet_spectrum/'+model+"/lte"+str_T+"-"+str_lg+"-0.0a+0.0.BT-Settl.fits")
    
    elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
        if T < 1400:
            print("Changing the input temperature to the minimal temperature: 1400K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature: 3000K.")
        T0 = np.arange(1400, 3100, 100)
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        lg0 = np.array([4.5, 5.0])
        idx = (np.abs(lg0 - lg)).argmin()
        lg = lg0[idx]
        if T >= 1000:
            str_T = "0"+str(T)[:2] ; str_lg = str(lg)
        else:
            str_T = "00"+str(T)[0] ; str_lg = str(lg)
        if str(T)[-2]!="0":
            str_T += "."+str(T)[-2]
        wave, flux = fits.getdata(load_path+'/planet_spectrum/'+model+"/lte"+str_T+"-"+str_lg+"-0.0a+0.0.BT-Dusty.fits")
        
    elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
        if T < 400:
            print("Changing the input temperature to the minimal temperature: 400K.")
        elif T > 2000:
            print("Changing the input temperature to the maximal temperature: 2000K.")
        if instru is None or get_config_data(instru)["lambda_range"]["lambda_min"] < 4: # low res
            lg0 = np.arange(3.0, 5.5, 0.5) # valeur possible # PUBLIC
            idx = (np.abs(lg0 - lg)).argmin()
            lg = lg0[idx]
            T0 = np.arange(400, 2050, 50)
            idx = (np.abs(T0 - T)).argmin()
            T = T0[idx]
            load_path += '/planet_spectrum/'+model+'/'
            load_path += "spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
        else: # high res (mais commmence à 4 µm) # NOT PUBLIC
            lg0 = np.array([3.5, 4.0]) # valeur possible
            idx = (np.abs(lg0 - lg)).argmin()
            lg = lg0[idx]
            T0 = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
            idx = (np.abs(T0 - T)).argmin()
            T = T0[idx]
            load_path += '/planet_spectrum/'+model+'/lte-g' + str(float(lg)) + '/'
            load_path+="spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
        wave, flux = fits.getdata(load_path)
        
    elif model == "PICASO": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
        if T < 200:
            print("Changing the input temperature to the minimal temperature: 200K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature: 3000K.")
        T0 = np.append(np.arange(500, 1000, 50), np.arange(1000, 3100, 100))
        T0 = np.append([200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 450], T0)
        idx = (np.abs(T0 - T)).argmin()
        T=T0[idx]
        if T < 260:
            lg0 = np.array([4.0])
        elif T == 260:
            lg0 = np.array([3.5])
        elif T <= 300:
            lg0 = np.array([3.0, 3.5])
        elif T < 500:
            lg0 = np.array([2.5, 3.0, 3.5])
        else:
            lg0 = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        idx = (np.abs(lg0 - lg)).argmin()
        lg=lg0[idx]
        wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{T}K_lg{lg}.fits")

    elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
        if T < 200:
            print("Changing the input temperature to the minimal temperature: 200K.")
        if T > 1300:
            print("Changing the input temperature to the maximal temperature: 1300K.")
        T0 = np.array([200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300]) # K
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        if T < 500: 
            g0 = np.array([10, 30, 100, 300, 1000]) # m/s²
            lg0 = np.array([3, 3.5, 4, 4.5, 5])
        else:
            g0 = np.array([100, 300, 1000, 3000]) # m/s² 
            lg0 = np.array([4, 4.5, 5, 5.5])
        g = 10**lg*1e-2 # m/s²
        idx = (np.abs(g0 - g)).argmin()
        g = g0[idx]
        lg = lg0[idx]
        wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/Morley/sp_t"+str(T)+"g"+str(g)+".fits")
        
    elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
        if T < 400:
            print("Changing the input temperature to the minimal temperature: 400K.")
        if T > 1200:
            print("Changing the input temperature to the maximal temperature: 1200K.")
        T0 = np.arange(400, 1250, 50)
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        g0 = np.array([10, 30, 100, 300, 1000]) # m/s²
        lg0 = np.array([3, 3.5, 4, 4.5, 5])
        g = 10**lg*1e-2 # m/s²
        idx = (np.abs(g0 - g)).argmin()
        g = g0[idx]
        lg = lg0[idx]
        wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/Saumon/sp_t"+str(T)+"g"+str(g)+"nc.fits")
    
    elif model == "SONORA": # https://zenodo.org/records/5063476
        if T < 200:
            print("Changing the input temperature to the minimal temperature: 200K.")
        if T > 2400:
            print("Changing the input temperature to the maximal temperature: 2400K.")
        T0 = np.append(np.arange(200, 1050, 50), np.arange(1100, 2500, 100))
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        g0 = np.array([10, 31, 100, 316, 1000, 3160]) # m/s²
        lg0 = np.array([3, 3.5, 4, 4.5, 5, 5.5])
        g = 10**lg*1e-2 # m/s²
        idx = (np.abs(g0 - g)).argmin()
        g = g0[idx]
        lg = lg0[idx]
        wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/SONORA/sp_t"+str(T)+"g"+str(g)+"nc_m0.0.fits")
        
    elif model[:4] == "mol_": # https://hitran.org/lbl/
        molecule = model[4:]
        if T < 200:
            print("Changing the input temperature to the minimal temperature: 200K.")
        if T > 3000:
            print("Changing the input temperature to the maximal temperature: 3000K.")
        T0 = np.append(np.arange(200, 1000, 50), np.arange(1000, 3100, 100))
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        wave, flux = fits.getdata(load_path+"/planet_spectrum/molecular/"+molecule+"_T"+str(T)+"K.fits")
    
        
    elif model == "Jupiter" or model == "Saturn" or model == "Uranus" or model == "Neptune": # private ?
        wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/solar system/plnt_"+model+".fits")
        if model == "Jupiter":
            T = 88 ; lg = 3.4 # np.log10(24.79*100) # https://en.wikipedia.org/wiki/Jupiter
        elif model == "Saturn":
            T = 81  ; lg = 3.0 # np.log10(10.44*100) # https://en.wikipedia.org/wiki/Saturn
        elif model == "Uranus":
            T = 49  ; lg = 2.9 # np.log10(8.69*100) # https://en.wikipedia.org/wiki/Uranus
        elif model == "Neptune":
            T = 47  ; lg = 3.0 # np.log10(11.15*100) # https://en.wikipedia.org/wiki/Neptune
        
    else:
        raise KeyError(model+" IS NOT A VALID THERMAL MODEL: BT-Settl, BT-Dusty, Exo-REM, PICASO, Morley, Saumon or SONORA")
        
    dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # delta lambda array
    R = np.nanmean(wave/(2*dwl)) # calculating the resolution of the raw spectrum
    spec = Spectrum(wave, flux, R, T, lg, model)
    return spec # in J/s/m²/µm

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_star_spectrum(T, lg, model="BT-NextGen", load_path=load_path):
    """
    load AGSS2009 star spectrum model (in J/s/m2/µm)
    Parameters
    ----------
    T: (float)
        star temperature (in K).
    lg: (float)
        star surface gravity (in dex(cm/s2).
    model: str, optional
        star model. The default is "BT-NextGen".

    Returns: class Spectrum
        loaded star spectrum in J/s/m²/µm
    """
    if model == "BT-NextGen":
        lg0 = np.array([3.0, 3.5, 4.0, 4.5])
        T0 = np.append(np.arange(3000, 10000, 200), np.arange(10000, 41000, 1000))
        idx = (np.abs(lg0 - lg)).argmin()
        lg = lg0[idx]
        idx = (np.abs(T0 - T)).argmin()
        T = T0[idx]
        if T >= 10000:
            str_T = str(T)[:3] ; str_lg = str(lg)
        elif T >= 1000:
            str_T = "0"+str(T)[:2] ; str_lg = str(lg)
        else:
            str_T = "00"+str(T)[0] ; str_lg = str(lg)
        spectrum = fits.getdata(load_path+'/star_spectrum/'+model+"/lte"+str_T+"-"+str_lg+"-0.0a+0.0."+model+".fits")
        wave = spectrum[0] ; dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) ; R = np.nanmean(wave/(2*dwl))
        spec = Spectrum(spectrum[0], spectrum[1], R, T, lg, model)
        
    return spec

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_vega_spectrum(vega_path=vega_path):
    """
    Load and retrieve vega spectrum (from magnitude calculation purposes)
    
    Returns
    -------
    vega_spec: Spectrum()
        vega spectrum.

    """
    f = fits.getdata(os.path.join(vega_path))
    wave=f[:, 0]*1e-3 # nm => µm
    flux=f[:, 1]*10 # 10 = 1e4 * 1e4 * 1e-7: erg/s/cm2/A -> erg/s/cm2/µm -> erg/s/m2/µm -> J/s/m2/µm
    vega_spec = Spectrum(wave, flux, None, None)
    return vega_spec
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def spectrum_instru(band0, R, config_data, mag, spectrum):
    """
    restricts the spectra to the instrument's wavelength range and adjusts it to the input magnitude on band0
    
    Parameters
    ----------
    band0: str
        wavelength range in which magnitude is entered ("J", "H", etc.)
    R: float
        spectral resolution of the input spectrum (it can be arbitrary but must be well above the instrumental spectral resolution)
    config_data: collections
        gives the parameters of the considered instrument
    mag: float
        input magnitude
    spectrum: class Spectrum
        spectrum to restrict and adjust (! must be in J/s/m2/µm !)

    Returns
    -------
    spectrum_instru: class Spectrum
        instrumental-wavelength-range-restricted and magnitude-adjusted spectrum in photons/min received
    """
    if band0 == "J": # Calculation of the number of photons/mn for a mag in the given band and the number of photons/mnn initially in the same band
        lambda_min_band0 = 1.085 ; lambda_max_band0 = 1.345 # in µm
    elif band0 == "H":
        lambda_min_band0 = 1.509 ; lambda_max_band0 = 1.799 # in µm
    elif band0 == 'Ks':
        lambda_min_band0 = 1.997 ; lambda_max_band0 = 2.317 # in µm
    elif band0 == 'K':
        lambda_min_band0 = 1.974 ; lambda_max_band0 = 2.384 # in µm
    elif band0 == "L":
        lambda_min_band0 = 3.262 ; lambda_max_band0 = 3.832 # in µm
    elif band0 == "L'":
        lambda_min_band0 = 3.436 ; lambda_max_band0 = 4.086 # in µm
    elif band0 == "instru":
        lambda_min_band0 = config_data["lambda_range"]["lambda_min"] ; lambda_max_band0 = config_data["lambda_range"]["lambda_max"] # in µm
    else:
        raise KeyError(f"{band0} is not considered band to define the magnitude")
    dl_band0 = ((lambda_max_band0+lambda_min_band0)/2)/(2*R)
    wave_band0 = np.arange(lambda_min_band0, lambda_max_band0, dl_band0) # wavelength array on band0 (in µm)
    spec = spectrum.interpolate_wavelength(spectrum.flux, spectrum.wavelength, wave_band0, renorm=False) # interpolating the input spectrum on band0
    vega_spec = load_vega_spectrum() # getting the vega spectrum
    vega_spec = vega_spec.interpolate_wavelength(vega_spec.flux, vega_spec.wavelength, wave_band0, renorm=False) # interpolating the vega spectrum on band0
    ratio = np.nanmean(vega_spec.flux)*10**(-0.4*mag) / np.nanmean(spec.flux) # ratio by which to adjust the spectrum flux in order to have the input magnitude
    # Conversion to photons/mn + restriction of spectra to instrumental range + adjustment of spectra to the input magnitude
    lambda_min_instru = config_data["lambda_range"]["lambda_min"] ; lambda_max_instru = config_data["lambda_range"]["lambda_max"] # (in µm)
    dl_instru = ((lambda_max_instru+lambda_min_instru)/2)/(2*R) 
    wave_instru = np.arange(0.98*lambda_min_instru, 1.02*lambda_max_instru, dl_instru) # constant and linear wavelength array on the instrumental bandwidth with equivalent resolution than the raw one
    spectrum.flux *= ratio # adjusting the spectrum to the input magnitude
    spectrum_density = spectrum.interpolate_wavelength(spectrum.flux, spectrum.wavelength, wave_instru, renorm = False) # in order to have a spectrum in density (i.e. J/s/m2/µm)     
    spectrum_instru = spectrum.set_nbphotons_min(config_data, wave_instru) # J/s/m²/µm => photons/mn on the instrumental bandwidth
    return spectrum_instru, spectrum_density # in ph/mn and J/s/m2/µm respectrively

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def spectrum_band(config_data, band, spectrum_instru):
    """
    Degradation of the spectrum resolution and restriction to the considered spectral band of the instrument

    Parameters
    ----------
    config_data: collections
        gives the parameters of the considered instrument
    band: str
        considered spectral band of the instrument
    spectrum_instru: class Spectrum
        instrumental-wavelength-range-restricted and magnitude-adjusted spectrum in photons/min received
    renorm: TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    spectrum_instru: class Spectrum
        band-restricted and resolution-degrated spectrum in photons/min received
    """
    lmin = config_data['gratings'][band].lmin ; lmax = config_data['gratings'][band].lmax # lambda_min/lambda_max of the considered band
    R = config_data['gratings'][band].R # spectral resolution of the band
    if R is None: # if not a spectro-imager (e.g. NIRCam)
        R = spectrum_instru.R # leaves resolution at native resolution
    dl_band = ((lmin+lmax)/2)/(2*R)
    wave_band = np.arange(lmin, lmax, dl_band) # constant and linear wavelength array on the considered band
    spectrum_band = spectrum_instru.degrade_resolution(wave_band, renorm=True) # degradation from spectrum resolution to spectral resolution of the considered band
    
    return spectrum_band # degrated spectrum

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def transmission(instru, wave_band, band, tellurics, apodizer):
    """
    To read and retrieve instrumental and sky (if needed) transmission

    Parameters
    ----------
    instru: str
        considered instrument
    wave_band: 1d array
        wavelength axis of the considered spectral band
    band: str
        considered spectral band
    R: float
        spectral resolution of the considered spectral band
    tellurics: bool (True or False)
        considering (or not) the earth atmosphere (ground or space observations)

    Returns
    -------
    transmission: 1d array (same size as wave_band)
        total system transmission 
    """
    wave, trans = fits.getdata("sim_data/Transmission/"+instru+"/transmission_" + band + ".fits") # instrumental transmission on the considered band
    f = interp1d(wave, trans, bounds_error=False, fill_value=np.nan)
    trans = f(wave_band) # interpolated instrumental transmission on the considered band
    config_data = get_config_data(instru) # get instrument specs
    apodizer_trans = config_data["apodizers"][str(apodizer)].transmission # get apodizer transmission, if any
    trans *= apodizer_trans
    if instru == "MIRIMRS" or instru == "NIRSpec":
        trans *= fits.getheader("sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_NO_JQ_NO_SP.fits")['AC'] # aperture corrective factor (the fact that not all the incident flux reaches the FOV)
    if tellurics: # if ground-based observation
        sky_transmission_path = os.path.join("sim_data/Transmission/sky_transmission_airmass_1.0.fits")
        sky_trans = fits.getdata(sky_transmission_path)
        trans_tell_band = Spectrum(sky_trans[0, :], sky_trans[1, :], None, None)
        trans_tell_band = trans_tell_band.degrade_resolution(wave_band, renorm=False).flux # degraded tellurics transmission on the considered band
        trans *= trans_tell_band # total system throughput (instru x atmo)
    trans[trans<=0] = np.nan # ignoring negatives values (if any)
    return trans

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def PSF_profile_fraction_separation(band, strehl, apodizer, coronagraph, instru, config_data, sep_unit):
    """
    Gives the PSF profile, the fraction of flux contained in the PSF core (or coronagraphic transmission) and the separation array.
    
    Parameters
    ----------
    band: str
        considered spectral band
    strehl: str
        strehl ratio
    apodizer: str
        apodizer
    coronagraph: str
        coronagraph
    instru: str
        instrument
    config_data: collections
        gives the parameters of the considered instrument

    Returns
    -------
    PSF_profile: 1d-array
        PSF profile
    fraction_PSF: float
        fraction core or coronagraphic transmission
    separation: 1d-array
        separation vector (in arcsec or mas)
    """
    if coronagraph is None: # Coronographic (stellar) PSF profile
        file = "sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_"+strehl+"_"+apodizer+".fits"
    else:
        file = "sim_data/PSF/PSF_"+instru+"/PSF_"+band+"_"+coronagraph+"_"+strehl+"_"+apodizer+".fits"
    fraction_PSF = fits.getheader(file)['FC'] # fraction of flux contained in the FWHM (or the coronagraphic stellar transmission)
    if instru == "MIRIMRS" or instru == "ANDES":
        pxscale = config_data["pxscale"][band] # in arcsec/px (dithered effective pixelscale)
    else:
        pxscale = config_data["spec"]["pxscale"] # in arcsec/px
    if instru == "ANDES":
        FOV = config_data["FOV_fiber"]*pxscale # in arcsec
    else:
        FOV = config_data["spec"]["FOV"] # in arcsec
    if sep_unit == "mas":
        pxscale *= 1e3 ; FOV *= 1e3 # arcsec => mas (if wanted)
    profile = fits.getdata(file) # in fraction/arcsec or fraction/mas
    separation = np.arange(pxscale/4, FOV/2+pxscale/4, pxscale/4) # in arcsec or mas (/4 doesn't change the result but gives smoother curves)
    separation = np.arange(pxscale/2, FOV/2+pxscale/2, pxscale/2) # in arcsec or mas (/4 doesn't change the result but gives smoother curves)
    f = interp1d(profile[0], profile[1], bounds_error=False, fill_value=np.nan)
    if instru == "MIRIMRS":
        PSF_profile = f(separation) * config_data["pxscale0"]**2 # pxscale (non-dithered) (because all the values are considered in the detector in the first place, then multiplied by R_corr, to take into account the transformation into 3D cube sapce)
    else:
        PSF_profile = f(separation) * pxscale**2
    return PSF_profile, fraction_PSF, separation, pxscale

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filtered_flux(flux, R, Rc, used_filter="gaussian"):
    """
    Gives low-pass and high-pass filtered flux as function of the cut-of resolution Rc

    Parameters
    ----------
    flux: 1d-array
        input flux.
    R: float
        spectral resolution of the input flux.
    Rc: TYPE
        cut-off resolution of the filter.
    used_filter: str, optional
        filter method considered. The default is "gaussian".

    Returns
    -------
    flux_HF: 1d-array
        high-pass filtered flux.
    flux_LF_valid: 1d-array
        low-pass filtered flux.
    """
    flux_LF_valid = np.copy(flux)
    if Rc is None:
        flux_LF = 0 # No filter applied
    else:
        if used_filter == "gaussian":
            sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2) # see Appendix A of Marots et al. (2024)
            flux_LF = gaussian_filter(flux[~np.isnan(flux)], sigma=sigma) # + ignoring NaN values
        elif used_filter == "step": # step filter function in the Fourier space
            fft = np.fft.fft(flux[~np.isnan(flux)]) ; ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)])) ; res = ffreq*R*2 ; fft[np.abs(res)>Rc] = 0 ; flux_LF = np.fft.ifft(fft)
        elif used_filter == "smoothstep":# smooth filter function in the Fourier space
            fft = np.fft.fft(flux[~np.isnan(flux)])
            ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)]))
            res = ffreq*R*2
            fft *= smoothstep(res, Rc)
            flux_LF = np.real(np.fft.ifft(fft))
    flux_LF_valid[~np.isnan(flux)] = flux_LF
    flux_HF = flux - flux_LF_valid
    return flux_HF, flux_LF_valid



def get_fraction_noise_filtered(wave, R, Rc, used_filter, empirical=False):
    """
    Gives the fraction power of noise that is filtered

    Parameters
    ----------
    wave: 1d-array
        input wavelength.
    R: float
        spectral resolution of wave.
    Rc: float
        cut-off resolution.
    used_filter: TYPE
        used method for the filter.
    empirical: TYPE, optional
        To estimate the fractions empirically or analytically. The default is False.
    """
    if Rc is None: # No filter applied
        fn_HF = 1. ; fn_LF = 1.
    else:
        if empirical:
            N = 1000 ; fn_HF = 0. ; fn_LF = 0.
            for i in range(N):
                n = np.random.normal(0, 1, len(wave))
                n_HF, n_LF = filtered_flux(n, R=R, Rc=Rc, used_filter=used_filter)
                fn_HF += np.nansum(n_HF**2) / np.nansum(n**2) / N
                fn_LF += np.nansum(n_LF**2) / np.nansum(n**2) / N
        else:
            ffreq = np.fft.fftfreq(len(wave)) ; res = ffreq*R*2
            K = np.zeros_like(res) + 1.
            K_LF = np.copy(K)
            K_HF = np.copy(K)
            if used_filter == "gaussian":
                sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2) 
                K_LF *= np.abs( np.exp( - 2*np.pi**2 * (res/(2*R))**2 * sigma**2 ) )**2
                K_HF *=  np.abs( 1 - np.exp( - 2*np.pi**2 * (res/(2*R))**2 * sigma**2 ) )**2
            elif used_filter == "step":
                K_LF[np.abs(res)>Rc] = 0 
                K_HF[np.abs(res)<Rc] = 0 
            elif used_filter == "smoothstep":
                K_LF *= smoothstep(res, Rc)
                K_HF *= (1-smoothstep(res, Rc))
            fn_LF = np.nansum(K_LF)/np.nansum(K) # power fraction of the noise being filtered
            fn_HF = np.nansum(K_HF)/np.nansum(K)
    return fn_HF, fn_LF

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_beta(star_spectrum_band, planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, used_filter):
    """
    Gives the value of the self-subtraction term β

    Parameters
    ----------
    star_spectrum_band: class Spectrum
        band-restricted and resolution-degrated star spectrum in photons/min received
    planet_spectrum_band: class Spectrum
        band-restricted and resolution-degrated planet spectrum in photons/min received
    sigma: float
        cut-off frequency = parameter for the high-pass filter (plus sigma est grand, plus la fréquence de coupure est petite => moins on coupera les basses fréquences)
    fraction_PSF: float
        fraction of flux contained in the PSF core of interest
    trans: 1d-array
        total-systemm transmission
    """
    star_HF, star_LF = filtered_flux(star_spectrum_band.flux, R, Rc, used_filter) # star filtered spectra
    planet_HF, planet_LF = filtered_flux(planet_spectrum_band.flux*fraction_PSF, R, Rc, used_filter) # planet filtered spectra
    beta = np.nansum(trans*star_HF*planet_LF/star_LF * template) # self-subtraction term
    beta = np.zeros_like(separation) + beta # constant as function of the separation
    return beta # in ph/mn

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def get_alpha(planet_spectrum_band, template, Rc, R, fraction_PSF, trans, separation, used_filter):
    """
    Compute the total amount of useful photons/min for molecular mapping detection
    
    Parameters
    ----------
    fraction_PSF: float
        fraction of flux of the PSF in size_core**2 pixels
    transmission: array
        instrumental + sky transmission
    sigma: float
        cut-off frequency = parameter for the high-pass filter
        (plus sigma est grand, plus la fréquence de coupure est petite => moins on coupera les basses fréquences)
    
    Returns
    -------
    alpha: float/int
        amount of useful photons
    """
    Sp = planet_spectrum_band.flux*fraction_PSF # planetary flux integrated in the FWHM (in ph/mn)
    Sp_HF, _ = filtered_flux(Sp, R, Rc, used_filter) # high_pass filtered planetary flux
    Sp_HF *= trans # gamma x [Sp]_HF
    alpha = np.nansum(Sp_HF*template) # alpha x cos theta lim (if systematic)
    alpha = np.zeros_like(separation) + alpha # constant as function of the separation
    return alpha # in ph/mn

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DIT_RON(instru, config_data, apodizer, PSF_profile, separation, star_spectrum_band, exposure_time, min_DIT, max_DIT, trans, quantum_efficiency, RON, saturation_e, input_DIT, print_value=True):
    """
    Gives DIT and effective reading noise

    Parameters
    ----------
    instru: str
        considered instrument
    config_data: collections
        gives the parameters of the considered instrument
    apodizer: str
        apodizer
    PSF_profile: 1d-array
        PSF profile
    star_spectrum_band: class Spectrum
        band-restricted and resolution-degrated star spectrum in photons/min received
    exposure_time: float (in min)
        total exposure_time
    min_DIT: float (in min)
        minimum integration exposure_time
    max_DIT: float (in min)
        maximum integration time
    quantum_efficiency: float
        quantum efficiency (in e-/ph)
    RON: float
        Read-Out Noise (in e-)
    saturation_e: float
        full well capacity (in e-)

    Returns
    -------
    DIT: float (in min)
        integration time
    RON_eff: TYPE
        effective Read-Out Noise (e-)
    """
    if instru == "NIRCam": # Calculating the DIT and the saturing DIT
        max_flux_e = np.nanmax(PSF_profile) * np.nansum(star_spectrum_band.flux) * trans * quantum_efficiency
    else: # maximum number of e-/mn in the considered band
        sep_min = config_data["apodizers"][apodizer].sep # separation where the PSF starts
        max_flux_e = np.nanmax(PSF_profile[separation>sep_min]) * star_spectrum_band.flux * trans * quantum_efficiency  
    saturating_DIT = saturation_e/np.nanmax(max_flux_e) # in mn
    if saturating_DIT > max_DIT: # The max DIT is determined by saturation or smearing DIT (the fact that the planet must not move too much angularly during the integration) or by the maximum detector integration time
        DIT = max_DIT
    elif saturating_DIT < min_DIT: 
        DIT = min_DIT
        if print_value:
            print(" Saturated detector even with the shortest integration time")
    else: # otherwise the DIT is given by the saturating DIT
        DIT = saturating_DIT
    if input_DIT is not None: # except if a DIT is input
        DIT = input_DIT
    if DIT > exposure_time: # The DIT cannot be longer than the total exposure time
        DIT = exposure_time
    nb_min_DIT = 1 # "Up the ramp" reading mode: the pose is sequenced in several non-destructive readings to reduce reading noise (see https://en.wikipedia.org/wiki/Signal_averaging).
    if DIT > nb_min_DIT*min_DIT: # choose 4 min_DIT because if intermittent readings are too short, the detector will heat up too quickly => + dark current
        N_i = DIT/(nb_min_DIT*min_DIT) # number of intermittent readings
        RON_eff = RON/np.sqrt(N_i) # effective read out noise (in e-/DIT)
    else:
        RON_eff = RON
    if instru == 'ERIS' and RON_eff < 7: # effective RON low limit for ERIS
        RON_eff = 7
    if print_value:
        print(" Saturating DIT =", round(saturating_DIT, 3), " mn")
        print(" DIT =", round(DIT*60, 2), "s / ", "RON =", round(RON_eff, 3), "e-")
    return DIT, RON_eff



#######################################################################################################################
##################################### SYSTEMATIC NOISE PROFILE CALCULATION: ###########################################
#######################################################################################################################

def systematic_profile(config_data, band, trans, Rc, R, star_spectrum_instru, planet_spectrum_instru, wave_band, size_core, used_filter, show_cos_theta_est=False, PCA=False, PCA_mask=False, Nc=20, mag_planet=None, band0=None, separation_planet=None, mag_star=None):
    """
    Estimates the systematic noise profile projected into the CCF

    Parameters
    ----------
    config_data: dictonnary
        instrument specs
    band: str
        spectral band considered.
    trans: 1d-array
        total system throughput
    Rc: float
        cut-off resolution
    R: float
        spectral resolution of the band
    star_spectrum_instru: class Spectrum()
        star spectrum considered in FastCurves on the instrumental bandwidth
    planet_spectrum_instru: class Spectrum()
        raw planet spectrum considered in FastCurves
    wave_band: 1d-array
        wavelength array of the considered band
    size_core: float
        size of the FWHM boxes
    used_filter: str
        used filter method
    show_cos_theta_est: bool, optional
        to estimate the correlation that would be measured in the data. The default is False.
    PCA: bool, optional
        to apply PCA to reduce the systematic noise level. The default is False.
    PCA_mask: bool, optional
        to mask the planet while applying the PCA. The default is False.
    Nc: int, optional
        number of principal components of the PCA subtracted. The default is 20.
    mag_planet: float, optional
        planet magnitude (for fake planet injection to estimate the PCA performances). The default is None.
    band0: str, optional
        band where the planet magnitude is given. The default is None.
    separation_planet: TYPE, optional
        planet separation (for fake planet injection to estimate the PCA performances). The default is None.
    """
    instru = config_data["name"] ; pca = None ; data = False ; warnings.simplefilter("ignore", category=RuntimeWarning)
    from src.molecular_mapping import crop, annular_mask, molecular_mapping_rv, stellar_high_filtering
    if instru == "MIRIMRS":
        correction = "all_corrected" # correction = "with_fringes_straylight" # correction applied to the simulated MIRISim noiseless data
        T_star_sim_arr = np.array([4000, 6000, 8000]) ; T_star_sim = T_star_sim_arr[np.abs(star_spectrum_instru.T-T_star_sim_arr).argmin()] # available values for the star temperature for MIRSim noiseless data
        file="data/MIRIMRS/MIRISim/star_center/star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction ; sigma_cosmic = None # simulated MIRISim noiseless data file
        #file = 'data/MIRIMRS/MAST/HD 159222_ch'+band[0]+'-shortmediumlong_s3d' ; data = True ; sigma_cosmic = 3 # CALIBRATION DATA => High S/N per spectral channel => M_data 
    elif instru == "NIRSpec":
        file = 'data/NIRSpec/MAST/HD 163466_nirspec_'+band+'_s3d' ; data = True ; sigma_cosmic = 3 # CALIBRATION DATA => High S/N per spectral channel => M_data: see Section 2.3 + 3.3 of Martos et al (2024)
    try: # if the files already exist
        if data: 
            S_noiseless = fits.getdata("sim_data/Systematics/"+instru+"/S_data_star_center_s3d_"+band+".fits") # on-sky data cube used to estimate the modulations (in e-/mn)
            pxscale = fits.getheader("sim_data/Systematics/"+instru+"/S_data_star_center_s3d_"+band+".fits")['pxscale'] # in arcsec/px
            wave = fits.getdata("sim_data/Systematics/"+instru+"/wave_data_star_center_s3d_"+band+".fits") # wavelength array of the data
        else:
            S_noiseless = fits.getdata("sim_data/Systematics/"+instru+"/S_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits")  # MIRISim noiseless data cube used to estimate the modulations (in e-/mn)
            pxscale = fits.getheader("sim_data/Systematics/"+instru+"/S_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits")['pxscale'] # in arcsec/px
            wave = fits.getdata("sim_data/Systematics/"+instru+"/wave_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits") # wavelength array of the data
    except: # in case they don't, create them (but the raw data are needed)
        file += ".fits"
        S_noiseless, wave, pxscale, _, _, exposure_time, _ = open_jwst_data(instru, "sim", band, cosmic=data, sigma_cosmic=sigma_cosmic, crop_band=True, file=file, print_value=False)
        S_noiseless /= exposure_time # in e-/mn
        hdr = fits.Header() ; hdr['pxscale'] = pxscale
        if data: # writing the data for systematics estimation purpose
            fits.writeto("sim_data/Systematics/"+instru+"/S_data_star_center_s3d_"+band+".fits", S_noiseless, header=hdr, overwrite=True)
            fits.writeto("sim_data/Systematics/"+instru+"/wave_data_star_center_s3d_"+band+".fits", wave, overwrite=True)
        else:
            fits.writeto("sim_data/Systematics/"+instru+"/S_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits", S_noiseless, header=hdr, overwrite=True)
            fits.writeto("sim_data/Systematics/"+instru+"/wave_noiseless_star_center_T"+str(T_star_sim)+"K_mag7_s3d_"+band+"_"+correction+".fits", wave, overwrite=True)

    f = interp1d(wave_band, trans, bounds_error=False, fill_value=np.nan) ; trans = f(wave) # interpolating the transmission on the wavelength array of the data
    NbChannel, NbLine, NbColumn = S_noiseless.shape # size of the cube
    y_center = NbLine//2 ; x_center = NbColumn//2 # spatial center position of the cube
    S_noiseless *= annular_mask(0, int(round(config_data["spec"]["FOV"]/2/pxscale)), size=(NbLine, NbColumn)) # ignoring the outter regions
    star_flux_FC = star_spectrum_instru.degrade_resolution(wave, renorm=True).flux*trans # star spectrum considered in FastCurves (in e-/mn)
    total_flux = np.nansum(star_flux_FC) # total stellar flux (in e-/mn)
    sep = np.zeros((int(round(config_data["spec"]["FOV"]/(2*pxscale)))+1)) # separation array for the data
    sigma_syst_prime_2 = np.zeros_like(sep) # systematic noise profile array estimated
    M_HF = np.zeros((len(sep), len(wave_band))) # high frequency systematic modulations as function of the separation [M(lambda, rho)]_HF
    star_flux_data = np.nansum(S_noiseless, (1, 2)) # gamma x S* of the data cube (in e-/mn) 
    
    if instru == "MIRIMRS" and not data: # for MIRISim data, the injected star spectra is known
        input_flux = np.loadtxt('sim_data/Systematics/MIRIMRS/star_'+str(T_star_sim)+'_mag7_J.txt', skiprows=1) ; input_flux = Spectrum(input_flux[:, 0], input_flux[:, 1], None, None) ; input_flux = input_flux.degrade_resolution(wave, renorm=False) ; input_flux = input_flux.set_nbphotons_min(config_data, wave) ; input_flux.flux *= trans # en e-/mn
        star_flux_data = input_flux.flux*np.nanmean(star_flux_data)/np.nanmean(input_flux.flux) # S_*
    
    cube_wo_planet = np.zeros_like(S_noiseless) + np.nan
    for i in range(NbChannel): # renormalizing the cube with the star spectra considered
        cube_wo_planet[i] = S_noiseless[i]/star_flux_data[i] * star_flux_FC[i] # M x gamma x S_*
        
    Sres_wo_planet, _ = stellar_high_filtering(cube_wo_planet, renorm_cube_res=False, R=R, Rc=Rc, used_filter=used_filter, cosmic=data, sigma_cosmic=sigma_cosmic, print_value=False) # stellar subtracted data
        
    if PCA: # applying PCA as it would be applied on real data
        T_star_t_syst_arr = np.array([3000,6000,9000]) ; T_star_t_syst = T_star_t_syst_arr[np.abs(star_spectrum_instru.T-T_star_t_syst_arr).argmin()] 
        T_planet_t_syst_arr = np.arange(500,3000+100,100) ; T_planet_t_syst = T_planet_t_syst_arr[np.abs(planet_spectrum_instru.T-T_planet_t_syst_arr).argmin()] 
        t_syst = fits.getdata("sim_data/Systematics/"+instru+"/t_syst/t_syst_"+instru+"_"+band+"_Tp"+str(T_planet_t_syst)+"K_Ts"+str(T_star_t_syst)+"K_Rc"+str(Rc))
        separation_t_syst = fits.getdata("sim_data/Systematics/"+instru+"/t_syst/separation_"+instru+"_"+band+"_Tp"+str(T_planet_t_syst)+"K_Ts"+str(T_star_t_syst)+"K_Rc"+str(Rc))
        mag_star_t_syst = fits.getdata("sim_data/Systematics/"+instru+"/t_syst/mag_star_"+instru+"_"+band+"_Tp"+str(T_planet_t_syst)+"K_Ts"+str(T_star_t_syst)+"K_Rc"+str(Rc))
        idx_mag_star_t_syst = np.abs(mag_star_t_syst - mag_star).argmin()
        idx_separation_t_syst = np.abs(separation_t_syst - separation_planet).argmin()
        if t_syst[idx_mag_star_t_syst,idx_separation_t_syst] < 120: # If the systematics are not dominating for an exoposure time of about 2 hours (~ order of magnitude of the observations generally made), PCA is not necessary
            mask = np.copy(Sres_wo_planet) ; mask[~np.isnan(mask)] = 0
            if mag_planet is not None: # fake injection of the planet in order to estimate components that would be estimated on real data and thus estimating the systematic noise and signal reduction 
                planet_spectrum_instru, _ = spectrum_instru(band0, 200000, config_data, mag_planet, planet_spectrum_instru) # planet spectrum on the instrumental bandwidth
                planet_flux_FC = trans*planet_spectrum_instru.degrade_resolution(wave, renorm=True).flux # gamma x Sp
                planet_HF, planet_LF = filtered_flux(planet_flux_FC/trans, R=R, Rc=Rc, used_filter=used_filter) # [Sp]_HF, [Sp]_LF
                star_HF, star_LF = filtered_flux(star_flux_FC/trans, R=R, Rc=Rc, used_filter=used_filter) # [S_*]_HF, [S_*]_LF
                y0 = int(round(NbColumn/2+min(separation_planet, config_data["spec"]["FOV"]/2-pxscale)/pxscale)) ; x0 = int(round(NbLine/2)) # planet's position according its separation supposing a position along the vertical spatial axis
                shift = int(round(min(separation_planet, config_data["spec"]["FOV"]/2-pxscale)/pxscale)) # estimating the shift in spaxel for the injection
                cube_shift = np.roll(np.copy(cube_wo_planet)*annular_mask(0, int(round(config_data["spec"]["FOV"]/4/pxscale)), size=(NbLine, NbColumn)), shift, 1) # planet = star PSF shifted
                for i in range(NbChannel):
                    cube_shift[i] *= planet_flux_FC[i]/star_flux_FC[i] # renormalize the star PSF to simulate the planet PSF
                cube = np.copy(cube_wo_planet) + np.nan_to_num(cube_shift) # add the planet PSF to the cube
                Sres, _ = stellar_high_filtering(cube, renorm_cube_res=False, R=R, Rc=Rc, used_filter=used_filter, cosmic=data, sigma_cosmic=sigma_cosmic, print_value=False) # stellar subtracted data with the fake planet injected
                Sres_pca, pca = PCA_subtraction(np.copy(Sres), n_comp_sub=Nc, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=PCA_mask, PCA_plots=False, wave=wave, R=R) # apply PCA to it
                Sres_pca += mask # retrieve the NaN values
                Sres_wo_planet = np.copy(Sres) ; Sres_wo_planet_pca = np.copy(Sres_pca)
                Sres_wo_planet[:, y0-size_core:y0+size_core+1, x0-size_core:x0+size_core+1] = np.nan ; Sres_wo_planet_pca[:, y0-size_core:y0+size_core+1, x0-size_core:x0+size_core+1] = np.nan # hiding the planet as it would be done on real data
                sres = np.zeros_like(Sres)+np.nan ; sres_pca = np.zeros_like(Sres_pca)+np.nan ; sres_wo_planet = np.zeros_like(Sres_wo_planet)+np.nan ; sres_wo_planet_pca = np.zeros_like(Sres_wo_planet_pca)+np.nan 
                for i in range(NbLine): # convolution by boxes of FWHM: see Martos et al. (2024)
                    for j in range(NbColumn): # for each spaxel
                        sres[:, i, j] = np.nansum(Sres[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2)) ; sres_pca[:, i, j] = np.nansum(Sres_pca[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2)) ; sres_wo_planet[:, i, j] = np.nansum(Sres_wo_planet[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2)) ; sres_wo_planet_pca[:, i, j] = np.nansum(Sres_wo_planet_pca[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2))
                sres[sres == 0] = np.nan ; Sres = sres ; sres_pca[sres_pca == 0] = np.nan ; Sres_pca = sres_pca ; sres_wo_planet[sres_wo_planet == 0] = np.nan ; Sres_wo_planet = sres_wo_planet ; sres_wo_planet_pca[sres_wo_planet_pca == 0] = np.nan ; Sres_wo_planet_pca = sres_wo_planet_pca
                CCF, _ = molecular_mapping_rv(instru=instru, S_res=Sres, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=None) ; CCF_pca, _ = molecular_mapping_rv(instru=instru, S_res=Sres_pca, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=pca) ; CCF_wo_planet, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=None) ; CCF_wo_planet_pca, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet_pca, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=pca) # the planet's parameters are not needed since the planet spectrum is injected
                from src.molecular_mapping import SNR_calculation
                _, _, CCF_signal, CCF_noise = SNR_calculation(CCF, CCF_wo_planet, y0, x0, size_core, print_value=False, snr_calc=False) ; _, _, CCF_signal_pca, CCF_noise_pca = SNR_calculation(CCF_pca, CCF_wo_planet_pca, y0, x0, size_core, print_value=False, snr_calc=False)
                signal = CCF_signal - np.nanmean(CCF_noise) ; signal_pca = CCF_signal_pca - np.nanmean(CCF_noise_pca) # estimating the planet signal at its location (with and without pca)
                M_pca = abs(signal_pca / signal) # signal loss measured
            else: # no fake injection of the planet 
                planet_spectrum_instru, _ = spectrum_instru(band0, 200000, config_data, 0, planet_spectrum_instru)
                planet_flux_FC = trans*planet_spectrum_instru.degrade_resolution(wave, renorm=True).flux
                planet_HF, planet_LF = filtered_flux(planet_flux_FC/trans, R=R, Rc=Rc, used_filter=used_filter)
                star_HF, star_LF = filtered_flux(star_flux_FC/trans, R=R, Rc=Rc, used_filter=used_filter)
                y0 = None ; x0 = None
                Sres_wo_planet_pca, pca = PCA_subtraction(np.copy(Sres_wo_planet), n_comp_sub=Nc, y0=y0, x0=x0, size_core=size_core, PCA_annular=False, scree_plot=False, PCA_mask=False, PCA_plots=False, wave=wave, R=R)
                Sres_wo_planet_pca += mask
                sres_wo_planet_pca = np.zeros_like(Sres_wo_planet_pca)+np.nan 
                for i in range(NbLine): # convolution by boxes of FWHM: see Martos et al. (2024)
                    for j in range(NbColumn): # for each spaxel
                        sres_wo_planet_pca[:, i, j] = np.nansum(Sres_wo_planet_pca[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2))
                sres_wo_planet_pca[sres_wo_planet_pca == 0] = np.nan ; Sres_wo_planet_pca = sres_wo_planet_pca
                CCF_wo_planet_pca, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet_pca, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=pca)
                M_pca = 1
        
            if Rc is None: # another way to estimate the signal loss due to the PCA: substract the PCA components to the planetary spectrum
                d = trans*planet_HF
            else:
                d = trans*planet_HF - trans*star_HF*planet_LF/star_LF # spectrum at the planet's location: see Eq.(18) of Martos et al. 2024
            template = trans*planet_HF
            template /= np.sqrt(np.nansum(template**2))
            d_sub = np.copy(d) ; template_sub = np.copy(template)
            for nk in range(Nc): # subtracting the components 
                d_sub -= np.nan_to_num(np.nansum(d*pca.components_[nk])*pca.components_[nk])
                template_sub -= np.nan_to_num(np.nansum(template*pca.components_[nk])*pca.components_[nk])
            m_pca = abs(np.nansum(d_sub*template_sub) / np.nansum(d*template)) # analytical signal loss
            
            M_pca = min(M_pca, m_pca, 1) # taking the minimal value between the two methods (and knowing that the signal loss ratio must be lower than 1)
            CCF_wo_planet = CCF_wo_planet_pca
        else:
            M_pca = 1. # no PCA
            sres_wo_planet = np.zeros_like(Sres_wo_planet) + np.nan 
            for i in range(NbLine): # convolution by boxes of FWHM: see Martos et al. (2024)
                for j in range(NbColumn): # for each spaxel
                    sres_wo_planet[:, i, j] = np.nansum(Sres_wo_planet[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2))
            sres_wo_planet[sres_wo_planet == 0] = np.nan ; Sres_wo_planet = sres_wo_planet
            CCF_wo_planet, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=pca)
    else:
        M_pca = 1. # no PCA
        sres_wo_planet = np.zeros_like(Sres_wo_planet) + np.nan 
        for i in range(NbLine): # convolution by boxes of FWHM: see Martos et al. (2024)
            for j in range(NbColumn): # for each spaxel
                sres_wo_planet[:, i, j] = np.nansum(Sres_wo_planet[:, i-size_core//2:i+size_core//2+1, j-size_core//2:j+size_core//2+1], axis=(1, 2))
        sres_wo_planet[sres_wo_planet == 0] = np.nan ; Sres_wo_planet = sres_wo_planet
        CCF_wo_planet, _ = molecular_mapping_rv(instru=instru, S_res=Sres_wo_planet, T=None, lg=None, rv=0, vsini=0, model=None, wave=wave, trans=trans, R=R, Rc=Rc, used_filter=used_filter, print_value=False, planet_spectrum=planet_spectrum_instru, pca=pca)

    for r in range(1, len(sep)+1):
        sep[r-1] = r*pxscale
        ccf = np.copy(CCF_wo_planet)*annular_mask(max(1, r-1), r, size=(NbLine, NbColumn)) # ring of the cube at separation r
        if not all(np.isnan(ccf.flatten())):
            sigma_syst_prime_2[r-1] = np.nanvar(ccf)/total_flux**2 # systematic noise at separation r (in e-/total stellar flux) 
            if show_cos_theta_est: # to estimate the correlation that would be measured, the high frequency modulation is needed at each separation
                f = interp1d(wave, Sres_wo_planet[:, y_center+1-r, x_center+1]/star_flux_FC, bounds_error=False, fill_value=np.nan)
                M_HF[r-1, :] = f(wave_band)
    
    Mp = np.nanmean(cube_wo_planet[:, y_center-size_core//2:y_center+size_core//2+1, x_center-size_core//2:x_center+size_core//2+1], axis=(1, 2))/star_flux_FC 
    Mp /= np.nanmean(Mp) # estimating the planet modulation function on the FWHM of the star (it's actually a bad estimation, but it's not significant for the performance estimates, only for the estimation of the correlation that would be measured cos_theta_est)
    #Mp = fits.getdata("utils/Mp/Mp_"+band+"_"+str(planet_spectrum_instru.T)+"K.fits")
    f = interp1d(wave, Mp, bounds_error=False, fill_value=np.nan)
    Mp = f(wave_band)
    #plt.figure() ; plt.plot(wave_band, Mp) ; plt.title(f"{band}") ; plt.show()
            
    return sigma_syst_prime_2, sep, M_HF, Mp, M_pca, wave



#######################################################################################################################
############################################# FastYield part: #########################################################
#######################################################################################################################

def thermal_reflected_spectrum(planet, instru=None, thermal_model="BT-Settl", reflected_model="PICASO", wave=None, vega_spectrum=None, show=True, in_im_mag=True):
    """
    Compute a thermal and reflected contributions for a given planet for FastYield table

    Parameters
    ----------
    planet: dictionnay
        planet caracteristics from FastYield planet.
    instru: str, optional
        instrument name (in order to have a wavelength array if it is not given). The default is None.
    thermal_model: str, optional
        thermal contribution model. The default is "BT-Settl".
    reflected_model: str, optional
        reflected contribution model. The default is "PICASO".
    wave: 1d-array, optional
        wavelength array on which the spectra will be calculated. The default is None.
    vega_spectrum: class Spectrum, optional
        vega spectrum. The default is None. If None, it will be retrieved
    show: bool, optional
        To plots the spectra contributions. The default is True.
    in_im_mag: bool, optional
        To renormalize the imaged planets in the K-band by the measured one. The default is True.
    """
    lmin_K = 1.951 ; lmax_K = 2.469 # K-band
    if thermal_model == "None" and reflected_model == "None":
        raise KeyError("PLEASE DEFINE A MODEL FOR THE THERMAL OR THE REFLECTED COMPONENT !")
    if wave is None and instru is not None: # in case a wavelength array is not input, create one
        config_data = get_config_data(instru)
        lmin_instru = config_data["lambda_range"]["lambda_min"] # in µm
        lmax_instru = config_data["lambda_range"]["lambda_max"] # in µm
        R = 200000 # abritrary resolution (need to be high enough)
        dl_instru = ((max(lmax_K, lmax_instru)+min(lmin_K, lmin_instru))/2)/(2*R)
        wave = np.arange(0.98*min(lmin_K, lmin_instru), 1.02*max(lmax_K, lmax_instru), dl_instru)
    if vega_spectrum is None: # in case a vega spectrum is not input, create one
        vega_spectrum = load_vega_spectrum()
        vega_spectrum = vega_spectrum.interpolate_wavelength(vega_spectrum.flux, vega_spectrum.wavelength, wave, renorm = False)
    
    star_spectrum = load_star_spectrum(float(planet["StarTeff"].value), float(planet["StarLogg"].value)) # load the star spectrum
    star_spectrum = star_spectrum.interpolate_wavelength(star_spectrum.flux, star_spectrum.wavelength, wave, renorm = False) # interpolating it to the wavelength array
    star_spectrum.flux *= np.nanmean(vega_spectrum.flux[(wave>lmin_K)&(wave<lmax_K)])*10**(-0.4*float(planet["StarKmag"])) / np.nanmean(star_spectrum.flux[(wave>lmin_K) & (wave<lmax_K)]) # renormalizing the star spectrum to the correct magnitude
    if float(planet["StarVsini"].value)!=0: # broadening th star spectrum
        star_spectrum = star_spectrum.broad(float(planet["StarVsini"].value))
    
    if thermal_model!="None": # load, reinterpolates and renormalizes the thermal contribution of the planet spectrum
        planet_thermal = load_planet_spectrum(float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), model=thermal_model)
        planet_thermal = planet_thermal.interpolate_wavelength(planet_thermal.flux, planet_thermal.wavelength, wave, renorm = False)
        planet_thermal.flux *= float((planet['PlanetRadius']/planet['Distance']).decompose()**2) # scaling factor
    elif thermal_model == "None":
        planet_thermal = Spectrum(wave, np.zeros_like(wave), star_spectrum.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), thermal_model)

    albedo = load_albedo(planet_thermal.T, planet_thermal.lg)
    albedo_geo = np.nanmean(albedo.flux) # mean value of the geometric albedo given by PICASO
    if reflected_model == "PICASO": # see Eq.(1) of Lovis et al. (2017): https://arxiv.org/pdf/1609.03082
        albedo = albedo.interpolate_wavelength(albedo.flux, albedo.wavelength, wave, renorm = False)
        planet_reflected = star_spectrum.flux * albedo.flux * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "flat":
        planet_reflected = star_spectrum.flux * albedo_geo * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "tellurics":
        wave_tell, tell = fits.getdata("sim_data/Transmission/sky_transmission_airmass_2.5.fits")
        f = interp1d(wave_tell, tell, bounds_error=False, fill_value=np.nan)
        albedo_tell = albedo_geo/np.nanmean(tell) * f(wave)
        planet_reflected = star_spectrum.flux * albedo_tell * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "None":
        planet_reflected = np.zeros_like(wave)*u.dimensionless_unscaled
    else:
        raise KeyError(reflected_model+" IS NOT A VALID REFLECTED MODEL: tellurics, flat, PICASO or None")
    planet_reflected = Spectrum(wave, np.nan_to_num(np.array(planet_reflected.value)), max(star_spectrum.R, albedo.R), albedo.T, float(planet["PlanetLogg"].value), reflected_model)
    
    planet_spectrum = Spectrum(wave, planet_thermal.flux+planet_reflected.flux, max(planet_thermal.R, planet_reflected.R), planet_thermal.T, planet_thermal.lg, thermal_model+"+"+reflected_model)
    
    if float(planet["PlanetVsini"].value)!=0: # broadening th planet spectrum
        if thermal_model!="None":
            planet_thermal = planet_thermal.broad(float(planet["PlanetVsini"].value))
        if reflected_model!="None":
            planet_reflected = planet_reflected.broad(float(planet["PlanetVsini"].value))
        planet_spectrum = planet_spectrum.broad(float(planet["PlanetVsini"].value))
        
    if float(planet["StarRadialVelocity"].value)!=0: # Doppler shifting the star spectrum
        star_spectrum = star_spectrum.doppler_shift(float(planet["StarRadialVelocity"].value))
        star_spectrum.star_rv = float(planet["StarRadialVelocity"].value)
    
    if float(planet["StarRadialVelocity"].value)!=0 or float(planet["DeltaRadialVelocity"].value)!=0: # Doppler shifting the planet spectrum
        if thermal_model!="None":
            planet_thermal = planet_thermal.doppler_shift(float(planet["StarRadialVelocity"].value)+float(planet["DeltaRadialVelocity"].value))
        if reflected_model!="None":
            planet_reflected = planet_reflected.doppler_shift(float(planet["StarRadialVelocity"].value)+float(planet["DeltaRadialVelocity"].value))
        planet_spectrum = planet_spectrum.doppler_shift(float(planet["StarRadialVelocity"].value)+float(planet["DeltaRadialVelocity"].value))
        planet_thermal.delta_rv = float(planet["DeltaRadialVelocity"].value)
        planet_reflected.delta_rv = float(planet["DeltaRadialVelocity"].value)
        planet_spectrum.delta_rv = float(planet["DeltaRadialVelocity"].value)

    if in_im_mag and planet["DiscoveryMethod"] == "Imaging" and thermal_model!="None": # To inject the known magnitudes of planets detected by direct imaging
        if not np.isnan(planet["PlanetKmag(thermal+reflected)"]): # the magnitude is already known by definition => renormalization in K-band
            ratio = np.nanmean(vega_spectrum.flux[(vega_spectrum.wavelength>lmin_K)&(vega_spectrum.wavelength<lmax_K)])*10**(-0.4*float(planet["PlanetKmag(thermal+reflected)"])) / np.nanmean(planet_spectrum.flux[(planet_spectrum.wavelength>lmin_K) & (planet_spectrum.wavelength<lmax_K)])
            planet_spectrum.flux = np.copy(planet_spectrum.flux) * ratio
            planet_thermal.flux = np.copy(planet_thermal.flux) * ratio
            planet_reflected.flux = np.copy(planet_reflected.flux) * ratio

    if show: # plotting the contributions
        lmin = lmin_K ; lmax = lmax_K ; band0 = "K"
        mag_p_total = -2.5*np.log10(np.mean(planet_spectrum.flux[(wave>lmin)&(wave<lmax)])/np.nanmean(vega_spectrum.flux[(wave>lmin) & (wave<lmax)]))
        plt.figure(dpi=300) ; plt.xlabel("wavelength (in µm)", fontsize=14) ; plt.ylabel(f"flux (in J/s/m²/µm)", fontsize=14) ; plt.yscale('log') ; plt.title(f"The different spectrum contributions \n for {planet['PlanetName']} at {round(float(planet_spectrum.T))}K (on the same spectral resolution)", fontsize=16)
        if thermal_model!="None" and reflected_model!="None":
            plt.plot(wave, planet_spectrum.flux, 'g', label=f"thermal+reflected ({planet_spectrum.model}), mag("+band0+f") = {round(mag_p_total, 2)}")
        if thermal_model!="None":
            mag_p_thermal = -2.5*np.log10(np.mean(planet_thermal.flux[(wave>lmin)&(wave<lmax)])/np.nanmean(vega_spectrum.flux[(wave>lmin) & (wave<lmax)])) 
            plt.plot(wave, planet_thermal.flux, 'r', label=f"thermal ({thermal_model}), mag("+band0+f") = {round(mag_p_thermal, 2)}")
        if reflected_model!="None":
            mag_p_reflected = -2.5*np.log10(np.mean(planet_reflected.flux[(wave>(lmin))&(wave<(lmax))])/np.nanmean(vega_spectrum.flux[(wave>(lmin))&(wave<(lmax))]))
            plt.plot(wave, planet_reflected.flux, 'b', label=f"reflected ({reflected_model}+BT-NextGen), mag("+band0+f") = {round(mag_p_reflected, 2)}")
        if np.nanmin(wave) < lmin_K:
            plt.axvspan(lmin_K, lmax_K, color='k', alpha=0.5, lw=0, label="K-band")
        plt.legend()
    
    return planet_spectrum, planet_thermal, planet_reflected, star_spectrum



#######################################################################################################################
###################################################### PICASO PART: ###################################################
#######################################################################################################################

def import_picaso():
    """
    Imports the picaso packages
    """
    try: 
        os.environ['picaso_refdata'] = '/home/martoss/picaso/reference/'
        os.environ['PYSYN_CDBS'] = '/home/martoss/picaso/grp/redcat/trds/'
        import picaso
        from picaso import justdoit as jdi
    except ImportError:
        print("Tried importing picaso, but couldn't do it")
    return picaso, jdi



def simulate_picaso_spectrum(instru, planet_table_entry, spectrum_contributions='thermal+reflected', opacity=None, planet_type="gas", clouds=True, stellar_mh=0.0122):
    '''
    TAKEN FROM: https://github.com/planetarysystemsimager/psisim/tree/main
    A function that returns the required inputs for picaso, given a row from a universe planet table. 
    
    Inputs:
    planet_table_entry - a single row, corresponding to a single planet from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice", or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    stellar_mh - stellar metalicity
    Opacity class from justdoit.opannection
    NOTE: this assumes a planet phase of 0. You can change the phase in the resulting params object afterwards.
    '''
    planet_table_entry["Phase"] = 0.0 * u.rad # in order to have the geometric Albedo (by definition)
    picaso, jdi=import_picaso()
    config_data = get_config_data(instru)
    lmin_instru = config_data["lambda_range"]["lambda_min"] # en µm
    lmax_instru = config_data["lambda_range"]["lambda_max"] # en µm
    lmin_K = 1.974 ; lmax_K = 2.384
    if opacity is None: # opacity file to load
        wvrng = [0.98*min(lmin_K, lmin_instru), 1.02*max(lmax_K, lmax_instru)] # opacity file goes from 0.6 to 6 µm with R ~ 30 000
        opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
        dbname = 'all_opacities_0.6_6_R60000.db' # lambda va de 0.6 à 6µm (mais indiqué 0.3 à 15µm)
        dbname = os.path.join(opacity_folder, dbname)
        opacity = jdi.opannection(filename_db=dbname, wave_range=wvrng) # molecules, pt_pairs = opa.molecular_avail(dbname) ; print("\n molecules considérées: \n ", molecules)
    host_temp_list=np.hstack([np.arange(3500, 13000, 250), np.arange(13000, 50000, 1000)])
    host_logg_list=[5.00, 4.50, 4.00, 3.50, 3.00, 2.50, 2.00, 1.50, 1.00, 0.50, 0.0] # Define the grids that phoenix / ckmodel models like
    f_teff_grid=interp1d(host_temp_list, host_temp_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    f_logg_grid=interp1d(host_logg_list, host_logg_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    planet_table_entry['StarTeff'] = f_teff_grid(planet_table_entry['StarTeff']) *planet_table_entry['StarTeff'].unit
    planet_table_entry['StarLogg'] = f_logg_grid(planet_table_entry['StarLogg']) *planet_table_entry['StarLogg'].unit
    params = jdi.inputs() ; params.approx(raman='none') # see justdoit.py => class inputs():
    params.phase_angle(planet_table_entry['Phase'].value)
    params.gravity(gravity=planet_table_entry['PlanetLogg'].value, gravity_unit=planet_table_entry['PlanetLogg'].physical.unit) # NOTE: picaso gravity() won't use the "gravity" input if mass and radius are provided
    star_logG = planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value
    if star_logG > 5.0: # The current stellar models do not like log g > 5, so we'll force it here for now. 
        star_logG = 5.0
    star_Teff = planet_table_entry['StarTeff'].to(u.K).value
    if star_Teff < 3000: # The current stellar models do not like Teff < 3000, so we'll force it here for now. 
        star_Teff = 3000   
    params.star(opacity, star_Teff, stellar_mh, star_logG, radius=planet_table_entry['StarRad'].value, semi_major=planet_table_entry['SMA'].value, semi_major_unit=planet_table_entry['SMA'].unit, radius_unit=planet_table_entry['StarRad'].unit) 
    if planet_type == 'gas': #-- Define atmosphere PT profile, mixing ratios, and clouds
        params.guillot_pt(planet_table_entry['PlanetTeq'].value, T_int=150, logg1=-0.5, logKir=-1) # T_int = Internal temperature / logg1, logKir = see parameterization Guillot 2010
        params.channon_grid_high() # get chemistry via chemical equillibrium
        if clouds: # may need to consider tweaking these for reflected light
            params.clouds(g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5]) # g0 = Asymmetry factor / w0 = Single Scattering Albedo / opd = Total Extinction in `dp` / p = Bottom location of cloud deck (LOG10 bars) / dp = Total thickness cloud deck above p (LOG10 bars)
    elif planet_type == 'terrestrial':
        pass # TODO: add Terrestrial type
    elif planet_type == 'ice':
        pass # TODO: add ice type
    op_wv = opacity.wave # this is identical to the model_wvs we compute below  
    phase = planet_table_entry['Phase'].value
    if phase == 0: # non-0 phases require special geometry which takes longer to run.
        df = params.spectrum(opacity, full_output=True, calculation=spectrum_contributions, plot_opacity=False) # Perform the simple simulation since 0-phase allows simple geometry
    else:
        df1 = params.spectrum(opacity, full_output=True, calculation='thermal') # Perform the thermal simulation as usual with simple geometry
        params.phase_angle(phase, num_tangle=8, num_gangle=8) # Apply the true phase and change geometry for the reflected simulation
        df2 = params.spectrum(opacity, full_output=True, calculation='reflected')
        df = df1.copy() ; df.update(df2) # Combine the output dfs into one df to be returned
        df['full_output_therm'] = df1.pop('full_output')
        df['full_output_ref'] = df2.pop('full_output')
    model_wvs = 1./df['wavenumber'] * 1e4 *u.micron ; argsort = np.argsort(model_wvs) ; model_wvs = model_wvs[argsort]
    model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1)) ; model_dwvs[0] = model_dwvs[1] ; model_R = np.nanmean(model_wvs/(2*model_dwvs)) ; model_R = model_R.value # = 30000
    if spectrum_contributions == "thermal":
        planet_thermal = np.zeros((2, len(model_wvs))) ; planet_thermal[0] = model_wvs
        thermal_flux = df["thermal"][argsort] * u.erg/u.s/u.cm**2/u.cm
        thermal_flux = thermal_flux.to(u.J/u.s/u.m**2/u.micron)
        planet_thermal[1] = np.array(thermal_flux.value)
        fits.writeto(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{round(float(planet_table_entry['PlanetTeq'].value))}K_lg{round(float(planet_table_entry['PlanetLogg'].value), 1)}.fits", planet_thermal, overwrite=True)
        plt.figure() ; plt.plot(planet_thermal[0], planet_thermal[1]) ; plt.title(f'{round(float(planet_table_entry["PlanetTeq"].value))}K and lg = {round(float(planet_table_entry["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength (in µm)') ; plt.ylabel("flux (in J/s/µm/m²)") ; plt.yscale('log') ; plt.show()
    elif spectrum_contributions == "reflected":
        albedo = np.zeros((2, len(model_wvs))) ; albedo[0] = model_wvs ; albedo[1] = df['albedo'][argsort]
        fits.writeto(f"sim_data/Spectra/planet_spectrum/albedo/albedo_gas_giant_{round(float(planet_table_entry['PlanetTeq'].value))}K_lg{round(float(planet_table_entry['PlanetLogg'].value), 1)}.fits", albedo, overwrite=True)
        plt.figure() ; plt.plot(albedo[0], albedo[1]) ; plt.title(f'{round(float(planet_table_entry["PlanetTeq"].value))}K and lg = {round(float(planet_table_entry["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength (in µm)') ; plt.ylabel("albedo") ; plt.show()
   
def get_picasso_thermal(name_planet="HR 8799 b"):
    from src.FastYield import load_planet_table, planet_index
    picaso, jdi=import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname = 'all_opacities_0.6_6_R60000.db'
    dbname = os.path.join(opacity_folder, dbname)
    opacity = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx = planet_index(planet_table, name_planet)
    T0 = np.append(np.arange(500, 1000, 50), np.arange(1000, 3100, 100))
    T0 = np.append([200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 450], T0)
    k=0
    for T in T0:
        if T < 260:
            lg0 = np.array([4.0])
        elif T == 260:
            lg0 = np.array([3.5])
        elif T <= 300:
            lg0 = np.array([3.0, 3.5])
        elif T < 500:
            lg0 = np.array([2.5, 3.0, 3.5])
        else:
            lg0 = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        for lg in lg0:
            k += 1
            planet_table[idx]["PlanetTeq"] = T * planet_table[idx]["PlanetTeq"].unit # 
            planet_table[idx]["PlanetLogg"] = lg * planet_table[idx]["PlanetLogg"].unit # 
            simulate_picaso_spectrum("HARMONI", planet_table[idx], spectrum_contributions="thermal", opacity=opacity)
            print(round(100*(k+1)/177, 2), "%")
            
def get_picasso_albedo(name_planet="HR 8799 b"):
    from src.FastYield import load_planet_table, planet_index
    picaso, jdi=import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname = 'all_opacities_0.6_6_R60000.db'
    dbname = os.path.join(opacity_folder, dbname)
    opacity = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx = planet_index(planet_table, name_planet)
    T0 = np.append(np.arange(500, 1000, 50), np.arange(1000, 3100, 100))
    T0 = np.append([200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 450], T0)
    k=0
    for T in T0:
        if T < 260:
            lg0 = np.array([4.0])
        elif T == 260:
            lg0 = np.array([3.5])
        elif T <= 300:
            lg0 = np.array([3.0, 3.5])
        elif T < 500:
            lg0 = np.array([2.5, 3.0, 3.5])
        else:
            lg0 = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        for lg in lg0:
            k += 1
            planet_table[idx]["PlanetTeq"] = T * planet_table[idx]["PlanetTeq"].unit # 
            planet_table[idx]["PlanetLogg"] = lg * planet_table[idx]["PlanetLogg"].unit # 
            simulate_picaso_spectrum("HARMONI", planet_table[idx], spectrum_contributions="reflected", opacity=opacity)
            print(round(100*(k+1)/177, 2), "%")
            
def load_albedo(T, lg, grid=True):
    if grid: 
        T0 = np.append(np.arange(500, 1000, 50), np.arange(1000, 3100, 100))
        T0 = np.append([200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 450], T0)
        idx = (np.abs(T0 - T)).argmin()
        T=T0[idx]
        if T < 260:
            lg0 = np.array([4.0])
        elif T == 260:
            lg0 = np.array([3.5])
        elif T <= 300:
            lg0 = np.array([3.0, 3.5])
        elif T < 500:
            lg0 = np.array([2.5, 3.0, 3.5])
        else:
            lg0 = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
        idx = (np.abs(T0 - lg)).argmin()
        lg=lg0[idx]
    wave, albedo = fits.getdata(f"sim_data/Spectra/planet_spectrum/albedo/albedo_gas_giant_{T}K_lg{lg}.fits")
    dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] # array de delta Lambda
    R = np.nanmean(wave/(2*dwl)) # calcule de la nouvelle résolution
    albedo = Spectrum(wave, albedo, R, T, lg, "PICASO")
    return albedo










