from src.utils import *

path_file = os.path.dirname(__file__)
load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")
vega_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/star_spectrum/VEGA_Fnu.fits")



class Spectrum:

    def __init__(self, wavelength, flux, R=None, T=None, lg=None, model=None, rv=0, vsini=0):
        """
        Parameters
        ----------
        wavelength: 1d array
            Wavelength axis (generally in µm)
        flux: 1d array
            Flux axis
        R: float, optional
            Spectral resolution of the spectrum
        T: float, optional
            Temperature of the spectrum (in K)
        lg: float, optional
            Surface gravity of the spectrum (in dex(cm/s2))
        model: str, optional
            Family model of the spectrum
        rv: float, optional
            Radial velocity of the spectrum (in km/s)
        vsini: float, optional
            Rotationnal velocity of the spectrum (in km/s)
        """
        self.wavelength = wavelength # wavelength axis of the spectrum
        self.flux       = flux       # flux of the spectrum
        self.R          = R          # resolution of the spectrum
        self.T          = T          # temperature of the spectrum
        self.lg         = lg         # surface gravity of the spectrum
        self.model      = model      # model of the spectrum
        self.rv         = rv         # radial velocity 
        self.vsini      = vsini      # rotationnal velocity
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def copy(self):
        """
        To copy a Spectrum object.
        """
        spectrum_copy = copy.deepcopy(self)
        return spectrum_copy
        
    def crop(self, lmin, lmax):
        """
        Crop the spectrum between lmin and lmax (and calculate the new spectral resolution).

        Parameters
        ----------
        lmin: float [µm]
            Lambda min value
        lmax: float [µm]
            Lambda max value
        """
        self.flux       = self.flux[(self.wavelength >= lmin) & (self.wavelength <= lmax)]
        self.wavelength = self.wavelength[(self.wavelength >= lmin) & (self.wavelength <= lmax)]
        dl     = self.wavelength - np.roll(self.wavelength, 1) ; dl[0] = dl[1] ; dl[dl == 0] = np.nanmean(dl) # delta lambda array
        Rnew   = np.nanmean(self.wavelength/(2*dl)) # calculating the new resolution (2*R => Nyquist sampling / Shannon)
        self.R = Rnew
    
    def crop_nan(self):
        """
        Crop nan values from the spectrum.

        Parameters
        ----------
        lmin: float [µm]
            Lambda min value
        lmax: float [µm]
            Lambda max value
        """
        self.wavelength = self.wavelength[~np.isnan(self.flux)]
        self.flux       = self.flux[~np.isnan(self.flux)]
        dl              = self.wavelength - np.roll(self.wavelength, 1) ; dl[0] = dl[1] ; dl[dl == 0] = np.nanmean(dl) # delta lambda array
        Rnew            = np.nanmean(self.wavelength/(2*dl)) # calculating the new resolution (2*R => Nyquist sampling / Shannon)
        self.R          = Rnew
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def set_flux(self, flux_tot):
        """
        Renormalize / Convert flux to photon number (or other).
        The flux can be in density but must have a constant delta_lambda.
        
        Parameters
        ----------
        flux_tot: float/int
            Quantity in a certain unit with which to renormalize the flux
        """
        self.flux = flux_tot * self.flux / np.nansum(self.flux)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_nbphotons_min(self, config_data, wave):
        """
        Converts a spectrum initially in density (J/s/m2/µm) into nb of photons/min received by the telescope over the wave range.
        (!wave MUST HAVE A CONSTANT delta_lambda!)  
            
        Parameters
        ----------
        config_data: collections
            Gives the parameters of the considered instrument
        wave: array (must be in µm)
            Wavelength axis on which the spectrum is converted

        Returns
        -------
        spectrum: class Spectrum
            Spectrum converted into nb of photons/min
        """
        dl                  = wave - np.roll(wave, 1) ; dl[0] = dl[1] ; dl[dl == 0] = np.nanmean(dl) # delta lambda en µm
        Rnew                = np.nanmean(wave/(2*dl)) # calculating the new resolution
        spectrum_flux       = self.interpolate_wavelength(wave, renorm=False).flux # reinterpolating the flux (in density) on wave
        spectrum            = self.copy()
        spectrum.wavelength = wave
        spectrum.flux       = spectrum_flux
        spectrum.flux       = spectrum.flux * wave*1e-6 / (h*c) # J/s/m2/µm => photons/s/m2/µm
        spectrum.flux       = spectrum.flux*config_data["telescope"]["area"]*dl*60 # photons/s/m2/µm => photons/mn
        return spectrum

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def degrade_resolution(self, wave_output, renorm=True, gaussian_filtering=True, R_output=None):
        """
        Degrade the spectral resolution at the new resolution (of wave_output) with a convolution.
        (!Does not work if the new resolution is higher than the original resolution!)

        Parameters
        ----------
        wave_output: array
            New wavelength axis (new resolution)
        renorm: bool, optional
            For renormalisation to conserve the flux (True => flux must not be in density eg. J/s/m2/µm). The default is True.
        gaussian_filtering: bool, optional
            To convole with a gaussian to mimic the convolution with the LSF of the instrument. The default is True.
        R_output: float, optional
            Resolution of the ouput spectrum (if None, assumed to be the Nyquist resolution of wave_output)
        
        Returns
        -------
        spectrum: class Spectrum
            Degrated spectrum
        """
        valid = (self.wavelength >= wave_output[0]) & (self.wavelength <= wave_output[-1]) # flux[valid] => returns a (smaller) array that stores flux values for a wavelength
        
        dl_old = self.wavelength[valid] - np.roll(self.wavelength[valid], 1) ; dl_old[0] = dl_old[1] ; dl_old[dl_old == 0] = np.nanmean(dl_old) # delta lambda array
        R_old  = np.nanmean(self.wavelength[valid]/(2*dl_old))   # old Resolution (assuming Nyquist sampling)
        
        dl_new = wave_output - np.roll(wave_output, 1) ; dl_new[0] = dl_new[1] ; dl_new[dl_new == 0] = np.nanmean(dl_new) # delta lambda array
        R_new  = np.nanmean(wave_output/(2*dl_new)) # calculating the new resolution
        
        spectrum_interp = self.evenly_spaced(renorm=renorm, wave_output=wave_output) # reinterpolate the flux on regular and linear wavelength array
        flux_interp     = spectrum_interp.flux
        wave_interp     = spectrum_interp.wavelength
        
        if gaussian_filtering: # convolution + down binning
            if R_output is None:
                fwhm = R_old / R_new # https://github.com/spacetelescope/pysynphot/issues/78
            elif R_output is not None:
                fwhm = R_old / R_output # https://github.com/spacetelescope/pysynphot/issues/78
            flux_conv = gaussian_filter(flux_interp[~np.isnan(flux_interp)], sigma=fwhm) # convoluted flux
            flux_interp[~np.isnan(flux_interp)] = flux_conv # ignoring the NaN values
        
        flux_lr = downbin_spec(flux_interp, wave_interp, wave_output, dlam=dl_new) # down binned flux
        
        spectrum            = self.copy()
        spectrum.wavelength = wave_output
        spectrum.R          = R_new
        if renorm:
            # conserving the flux
            flux_tot      = np.nansum(self.flux[valid])
            spectrum.flux = flux_tot * flux_lr / np.nansum(flux_lr)
        else:
            # not convserving the flux (e.g. for spectra in density or for transmissions)
            spectrum.flux = flux_lr
        return spectrum
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
    def interpolate_wavelength(self, wave_output, renorm=True, fill_value=np.nan, wave_input=None):
        """
        Re-interpolates the flux on a new wavelength axis.
        
        Parameters
        ----------
        wave_output: array
            New wavelength axis for the interpolation.
        renorm: bool, optional
            For renormalisation (True => flux must not be in density eg. J/s/m2/µm). The default is True.
        fill_value: optional
            Values to fill the extrapolations. The default is np.nan.
        wave_input: array, optional
            Inputed wavelength axis for the doppler_shift() function. The default is None.
        
        Returns
        -------
        spectrum: class Spectrum
            Spectrum with the interpolated flux on the new wavelength axis
        """
        if wave_input is None: # wave_input is only usefull for the doppler_shift() function
            wave_input = np.copy(self.wavelength)
        
        f           = interp1d(wave_input, self.flux, bounds_error=False, fill_value=fill_value)
        flux_interp = f(wave_output) # interpolates flux values on the new axis (wave_output)
        dl_new      = wave_output - np.roll(wave_output, 1) ; dl_new[0] = dl_new[1] ; dl_new[dl_new == 0] = np.nanmean(dl_new) # delta lambda array
        R_new       = np.nanmean(wave_output/(2*dl_new)) # calculating the new resolution
        
        spectrum            = self.copy()
        spectrum.wavelength = wave_output
        spectrum.R          = R_new
        if renorm:
            # conserving the flux
            flux_tot      = np.nansum(self.flux[(wave_input >= wave_output[0]) & (wave_input <= wave_output[-1])])
            spectrum.flux = flux_tot * flux_interp / np.nansum(flux_interp)
        else:
            # not convserving the flux (e.g. for spectra in density or for transmissions)
            spectrum.flux = flux_interp
        return spectrum
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def evenly_spaced(self, renorm=True, fill_value=np.nan, wave_output=None):
        """
        Re-interpolates the flux on a regular and linear wavelength axis.
        
        Parameters
        ----------
        renorm: bool, optional
            For renormalisation (True => flux must not be in density eg. J/s/m2/µm). The default is True.
        fill_value: optional
            Values to fill the extrapolations. The default is np.nan.
        wave_output: array, optional
            In order to have similar limits of wave_ouput array. The default is None.
        
        Returns
        -------
        spectrum: class Spectrum
            Spectrum with the regular and linear wavelength axis.
        """
        if wave_output is not None:
            valid = (self.wavelength >= wave_output[0]) & (self.wavelength <= wave_output[-1]) # flux[valid] => returns a (smaller) array that stores flux values for a wavelength
        else:
            valid = np.full(len(self.wavelength), True)
        
        dl_old   = self.wavelength[valid] - np.roll(self.wavelength[valid], 1) ; dl_old[0] = dl_old[1] ; dl_old[dl_old == 0] = np.nanmean(dl_old) # delta lambda array
        R_interp = np.nanmax(self.wavelength[valid]/(2*dl_old)) # interpolation Resolution (need to be the max res to avoid nan with cg.downbin)
        if R_interp > R0_max: # fixing the upper limit of resolution in order to speeds up the calculation (it also need to be high enough for instruments with very high resolution)
            R_interp = R0_max
        dl = np.nanmean((self.wavelength[valid][0]+self.wavelength[valid][-1])/2/(2*R_interp)) 
        if wave_output is not None:
            wave_interp = np.arange(0.98*wave_output[0], 1.02*wave_output[-1], dl) # constant and linear input wavelength array
        else:
            wave_interp = np.arange(self.wavelength[valid][0], self.wavelength[valid][-1], dl) # constant and linear input wavelength array
        return self.interpolate_wavelength(wave_interp, renorm=renorm, fill_value=fill_value) 
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def doppler_shift(self, rv, renorm=False, fill_value=np.nan):
        """
        Doppler shift of a spectrum as a function of radial velocity rv
        
        Parameters
        ----------
        rv: float/int (in km/s)
            Radial velocity
        fill_value: optional
            Values to fill the extrapolations. The default is np.nan.

        Returns
        -------
        spectrum: class Spectrum
            Shifted spectrum
        """
        if rv != 0:
            wshift  = self.wavelength * (1 + (1000*rv / c)) # offset wavelength axis
            spec_rv = self.interpolate_wavelength(wave_input=wshift, wave_output=self.wavelength, renorm=renorm, fill_value=fill_value)
        else:
            spec_rv = self.copy()
        spec_rv.rv += rv
        return spec_rv

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def broad(self, vsini, epsilon=0.8, fastbroad=True):
        """
        Broadens spectrum lines as a function of rotation speed (in km/s)

        Parameters
        ----------
        vsini: float
            Projected rotational velocity (in km/s).
        epsilon: float, optional
            Linear limb-darkening coefficient (0-1).
        fastbroad: bool, optional
            True for faster (but less precise) computations. The default is True.

        Returns
        -------
        spectrum: class Spectrum
            Broadened spectrum
        """
        if vsini > 0 :
            if fastbroad: # fast spectral broadening (but less accurate)
                flux = pyasl.fastRotBroad(self.wavelength*1e4, self.flux, epsilon=epsilon, vsini=vsini)
            else: # slow spectral broadening (but more accurate)
                flux = pyasl.rotBroad(self.wavelength[~np.isnan(self.flux)]*1e4, self.flux[~np.isnan(self.flux)], epsilon=epsilon, vsini=vsini) # ignoring NaN values at the same time
                f    = interp1d(self.wavelength[~np.isnan(self.flux)], flux, bounds_error=False, fill_value=np.nan) 
                flux = f(self.wavelength) # pas besoin de "conserver le nb de photons" car c'est un effet intrinsèque (on ne change pas largeur des bins)
            spec_vsini       = self.copy()
            spec_vsini.flux  = flux
        elif vsini < 0 :
            raise KeyError("Vsini can not be negative.")
        elif vsini == 0:
            spec_vsini = self.copy()
        spec_vsini.vsini += vsini
        return spec_vsini 

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def set_snr(self, snr_canal): 
        """
        setting the flux with snr_canal assuming noise only given by the photon noise of the flux

        """
        self.flux = snr_canal ** 2 * self.flux / np.nanmean(self.flux)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def air2vacuum(self):
        """
        Convert wavelength from air to vacuum
        """
        s = 1e4 / (self.wavelength * 1e4) # self.wavelength in Angstrom
        n =  1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
        self.wavelength *= n
    
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def get_psd(self, smooth=0, crop=True):
        """
        Gives the PSD of the Spectrum

        Parameters
        ----------
        smooth: float, optional
            Smooth the PSD by convolution with a gaussian. The default is 1.
        crop: TYPE, optional
            To crop negative resolution values. The default is True.

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
        else:
            PSD = fft * (1/N)
            res = ffreq*self.R*2
        if smooth == 0:
            PSD = PSD
        else:
            PSD = gaussian_filter(PSD, sigma=smooth)
        return res, PSD

def calc_psd(wave, flux, R, smooth=0):
    """
    Calculate the psd of the inpux flux.

    Parameters
    ----------
    wave : array 1d
        wavelength axis.
    flux : array 1d
        Flux axis.
    R : float
        Sampling resolution.
    smooth : float, optional
        Smoothing parameters of the PSF. The default is 0.

    Returns
    -------
    res : array 1d
        resolution axis.
    psd : array 1d
        PSD axis.
    """
    flux_spectrum = Spectrum(wave, flux, R, None)
    flux_spectrum.wavelength = flux_spectrum.wavelength[~np.isnan(flux_spectrum.flux)]
    flux_spectrum.flux = flux_spectrum.flux[~np.isnan(flux_spectrum.flux)]
    res, psd = flux_spectrum.get_psd(smooth=smooth)
    return res, psd



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_T_lg_valid(T, lg, model, instru=None):
    """
    Retrieve the closest valid values of T and lg in the model-grid.

    Parameters
    ----------
    T: float/int
        Temperature (in K)
    lg: float/int
        Surface gravity (in dex(cm/s2))
    model: str
        Spectrum model
    instru: str, optional
        Considered instrument (the grid can depend on the considered intrument). The default is None.
    
    Returns
    -------
    T_valid, lg_valid: float (tuple)
        Closest valid values of T and lg in the model-grid

    """
    T_grid, lg_grid = get_model_grid(model, instru=instru)
    T_valid         = T_grid[(np.abs(T_grid-T)).argmin()] # closest available value
    if model[:4] == "mol_":
        if model[4:] not in lg_grid:
            raise KeyError(f"{lg} is not a valid molecule, please choose among: {lg_grid}")
        lg_valid = lg
    else:
        lg_valid = lg_grid[(np.abs(lg_grid-lg)).argmin()] # closest lg value
    return T_valid, lg_valid

def get_model_grid(model, instru=None):
    """
    Retrieve the grid model parameters

    Parameters
    ----------
    model : str
        Name's model.
    instru: str, optional
        Considered instrument (the grid can depend on the considered intrument). The default is None.
        
    Returns
    -------
    T : 1d-array
        Temperature values of the model grid (in K).
    lg : 1d-array
        Surface gravity values of the model grid (in dex(cm/s2)).
    """
    if model == "BT-Settl" or model == "PICASO":
        T_grid  = np.append([200, 220, 240, 250, 260, 280, 300, 320, 340, 360, 380, 400, 450], np.append(np.arange(500, 1000, 50), np.arange(1000, 3100, 100)))
        lg_grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

    elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
        T_grid  = np.arange(1400, 3100, 100)
        lg_grid = np.array([4.5, 5.0])

    elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
        if instru is None: # low res
            T_grid  = np.arange(400, 2050, 50)
            lg_grid = np.arange(3.0, 5.5, 0.5)
        else:
            if globals()["lmin_"+instru] >= 1 and globals()["lmax_"+instru] <= 5.3: # very high res
                T_grid  = np.arange(200, 1950, 50)
                lg_grid = np.arange(3.0, 5.5, 0.5)
            elif globals()["lmin_"+instru] <= 4: # low res
                T_grid  = np.arange(400, 2050, 50)
                lg_grid = np.arange(3.0, 5.5, 0.5) 
            elif globals()["lmin_"+instru] >= 4: # high res (mais commmence à 4 µm) # NOT PUBLIC
                T_grid  = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
                lg_grid = np.array([3.5, 4.0]) 

    elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
        T_grid  = np.array([200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300]) # K
        g_grid  = np.array([10, 30, 100, 300, 1000, 3000]) # m/s2 
        g_grid  = np.array([100, 300, 1000, 3000]) # m/s2 
        lg_grid = np.round(np.log10(g_grid*1e2), 4) # dex(cm/s2)

    elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
        T_grid  = np.arange(400, 1250, 50)
        g_grid  = np.array([10, 30, 100, 300, 1000]) # m/s2
        lg_grid = np.round(np.log10(g_grid*1e2), 4) # dex(cm/s2)

    elif model == "SONORA": # https://zenodo.org/records/5063476
        T_grid  = np.append(np.arange(200, 1050, 50), np.arange(1100, 2500, 100))
        g_grid  = np.array([10, 31, 100, 316, 1000, 3160]) # m/s2
        lg_grid = np.round(np.log10(g_grid*1e2), 4) # dex(cm/s2)

    elif model[:4] == "mol_": # https://hitran.org/lbl/
        T_grid  = np.append(np.arange(200, 1000, 50), np.arange(1000, 3100, 100))
        lg_grid = np.array(["H2O","CO2","O3","N2O","CO","CH4","O2","NO","SO2","NO2","NH3"])
        
    elif model == "BT-NextGen":
        T_grid  = np.append(np.arange(3000, 10000, 200), np.arange(10000, 41000, 1000))
        lg_grid = np.array([3.0, 3.5, 4.0, 4.5])
    
    elif model == "Husser":
        T_grid  = np.append(np.arange(2300, 7100, 100), np.arange(7200, 12200, 200))
        lg_grid = np.arange(0.00, 6.50, 0.50)
        
    elif model == "Jupiter" or model == "Saturn" or model == "Uranus" or model == "Neptune": # private ?
        if model == "Jupiter":
            T_grid = np.array([88]) ; lg_grid = np.array([3.4]) # np.log10(24.79*100) # https://en.wikipedia.org/wiki/Jupiter
        elif model == "Saturn":
            T_grid = np.array([81])  ; lg_grid = np.array([3.0]) # np.log10(10.44*100) # https://en.wikipedia.org/wiki/Saturn
        elif model == "Uranus":
            T_grid = np.array([49])  ; lg_grid = np.array([2.9]) # np.log10(8.69*100) # https://en.wikipedia.org/wiki/Uranus
        elif model == "Neptune":
            T_grid = np.array([47])  ; lg_grid = np.array([3.0]) # np.log10(11.15*100) # https://en.wikipedia.org/wiki/Neptune
    
    else:
        raise KeyError(model+" IS NOT A VALID THERMAL MODEL: BT-NextGen, BT-Settl, BT-Dusty, Exo-REM, PICASO, Morley, Saumon or SONORA.")
    
    return T_grid, lg_grid



def load_spectrum(T, lg, model, albedo=False, instru=None, load_path=load_path):
    """
    To read and retrieve planet spectra from models grid with the exact input parameters (see http://svo2.cab.inta-csic.es/theory/newov2/)

    Parameters
    ----------
    T: float
        Temperature (in K)
    lg: float
        Surface gravity (in dex(cm/s2))
    model: str
        model's name
    albedo: bool, optional
        True if the spectrum is an albedo spectrum. The default is False.
    instru: str, optional
        Considered instrument (the grid can depend on the considered intrument). The default is None.
    load_path: str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns
    -------
    spectrum: class Spectrum
        Loaded spectrum (in J/s/m2/µm)
    """
    try:
        if albedo:
            if model == "PICASO": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/albedo/albedo_gas_giant_{T}K_lg{lg}.fits")
            else:
                raise KeyError(model+" IS NOT A VALID ALBEDO MODEL: PICASO.")
                
        elif not albedo:
            if model == "BT-Settl": # https://articles.adsabs.harvard.edu/pdf/2013MSAIS..24..128A              
                wave, flux = fits.getdata(load_path+f"/planet_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
            
            elif model == "BT-Dusty": # https://arxiv.org/pdf/1112.3591
                wave, flux = fits.getdata(load_path+f"/planet_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
                
            elif model == "Exo-REM": # https://iopscience.iop.org/article/10.3847/1538-4357/aaac7d/pdf
                if instru is None: # low res
                    load_path += '/planet_spectrum/'+model+'/low_res/'
                    load_path += "spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
                else:
                    if globals()["lmin_"+instru] >= 1 and globals()["lmax_"+instru] <= 5.3: # very high res
                        FeH = 0.0  # Métallicité
                        CO  = 0.65 # Ratio C/O
                        load_path += '/planet_spectrum/'+model+'/very_high_res/'
                        load_path += f"spect_Teff={T:04.0f}K_logg={lg:.1f}_FeH={FeH:+.1f}_CO={CO:.2f}.fits"
                    elif globals()["lmin_"+instru] <= 4: # low res
                        load_path += '/planet_spectrum/'+model+'/low_res/'
                        load_path += "spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
                    elif globals()["lmin_"+instru] >= 4: # high res (mais commmence à 4 µm) # NOT PUBLIC
                        load_path += '/planet_spectrum/'+model+'/high_res/'
                        load_path +="spectra_YGP_"+str(T)+"K_logg"+str(float(lg))+"_met1.00_CO0.50.fits"
                wave, flux = fits.getdata(load_path)
                
            elif model == "PICASO": # https://iopscience.iop.org/article/10.3847/1538-4357/ab1b51/pdf + https://github.com/natashabatalha/picaso
                wave, flux = fits.getdata(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{T}K_lg{lg}.fits")
        
            elif model == "Morley": # 2012 + 2014 with clouds (https://www.carolinemorley.com/models)
                g_planet = round(10**lg*1e-2) # m/s2
                wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/Morley/sp_t"+str(T)+"g"+str(g_planet)+".fits")
                
            elif model == "Saumon": # https://www.ucolick.org/~cmorley/cmorley/Models.html
                g_planet = round(10**lg*1e-2) # m/s2
                wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/Saumon/sp_t"+str(T)+"g"+str(g_planet)+"nc.fits")
            
            elif model == "SONORA": # https://zenodo.org/records/5063476
                g_planet = round(10**lg*1e-2) # m/s2
                wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/SONORA/sp_t"+str(T)+"g"+str(g_planet)+"nc_m0.0.fits")
                
            elif model[:4] == "mol_": # https://hitran.org/lbl/
                molecule = model[4:]
                wave, flux = fits.getdata(load_path+"/planet_spectrum/molecular/"+molecule+"_T"+str(T)+"K.fits")
            
            elif model == "Jupiter" or model == "Saturn" or model == "Uranus" or model == "Neptune": # private ?
                wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/solar system/plnt_"+model+".fits")
                wave, flux = fits.getdata("sim_data/Spectra/planet_spectrum/solar system/psg_"+model+"_rad.fits")
                
            elif model == "BT-NextGen":
                wave, flux = fits.getdata(load_path+f"/star_spectrum/{model}/lte{T/100:03.0f}-{lg:.1f}-0.0a+0.0.{model}.fits")
            
            elif model == "Husser":
                wave = fits.getdata(load_path+f"/star_spectrum/{model}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                flux = fits.getdata(load_path+f"/star_spectrum/{model}/lte{T:05.0f}-{lg:4.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
            
            else:
                raise KeyError(model+" IS NOT A VALID THERMAL MODEL: BT-NextGen, Husser, BT-Settl, BT-Dusty, Exo-REM, PICASO, Morley, Saumon or SONORA.")
                
        dl       = wave - np.roll(wave, 1) ; dl[0] = dl[1] ; dl[dl == 0] = np.nanmean(dl) # delta lambda array
        R        = np.nanmean(wave/(2*dl)) # calculating the resolution of the raw spectrum
        spectrum = Spectrum(wave, flux, R, T, lg, model)
        return spectrum # in J/s/m2/µm or no unit if albedo
    
    except Exception as e:
        raise KeyError(f"{T}K or {lg} are not valid parameters of the {model} grid: {e}")



def interpolate_T_lg_spectrum(T_valid, lg_valid, T, lg, model, albedo=False, instru=None, load_path=load_path):
    """
    Interpolates a spectrum at T and lg with spectra from the model grid

    Parameters
    ----------
    T_valid : float
        Closest valid planet's temperature in the model grid.
    lg_valid : float
        Closest valid planet's surface gravity in the model grid.
    T : float
        temperature at which interpolate.
    lg : float
        surface gravity at which interpolate.
    model : str
        model's name.
    albedo: bool, optional
        True if the spectrum is an albedo spectrum. The default is False.
    instru: str, optional
        Considered instrument (the grid can depend on the considered intrument). The default is None.
    load_path: str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns
    -------
    spectrum: class Spectrum
        Interpolated spectrum (in J/s/m2/µm)
    """
    T_grid, lg_grid = get_model_grid(model, instru=instru) # Retrieve the model grid
    if (T_valid >= T and T_valid == T_grid[0]) or (T_valid <= T and T_valid == T_grid[-1]) or (T_valid == T): # Handling T interpolation
        T_sup, T_inf = T_valid, T_valid
    else:
        T_sup, T_inf = (T_valid, T_grid[np.abs(T_grid-T_valid).argmin()-1]) if T_valid > T else (T_grid[np.abs(T_grid-T_valid).argmin()+1], T_valid)
    if (lg_valid >= lg and lg_valid == lg_grid[0]) or (lg_valid <= lg and lg_valid == lg_grid[-1]) or (lg_valid == lg): # Handling lg interpolation
        lg_sup, lg_inf = lg_valid, lg_valid
    else:
        lg_sup, lg_inf = (lg_valid, lg_grid[np.abs(lg_grid-lg_valid).argmin()-1]) if lg_valid > lg else (lg_grid[np.abs(lg_grid-lg_valid).argmin()+1], lg_valid)
        _, lg_inf = get_T_lg_valid(T=T, lg=lg_inf, model=model, instru=instru) # Adjust lg values with T (if interpolation along lg)
        _, lg_sup = get_T_lg_valid(T=T, lg=lg_sup, model=model, instru=instru)
    if lg_inf != lg_sup:  # Interpolation along lg
        if T_inf != T_sup:  # Interpolation along lg and T
            spec_T_inf_lg_inf = load_spectrum(T_inf, lg_inf, model=model, albedo=albedo, instru=instru, load_path=load_path)
            spec_T_inf_lg_sup = load_spectrum(T_inf, lg_sup, model=model, albedo=albedo, instru=instru, load_path=load_path)
            spec_T_sup_lg_inf = load_spectrum(T_sup, lg_inf, model=model, albedo=albedo, instru=instru, load_path=load_path)
            spec_T_sup_lg_sup = load_spectrum(T_sup, lg_sup, model=model, albedo=albedo, instru=instru, load_path=load_path)
            wave = spec_T_inf_lg_inf.wavelength
            if len(spec_T_inf_lg_sup.wavelength) != len(wave):
                spec_T_inf_lg_sup = spec_T_inf_lg_sup.interpolate_wavelength(wave, renorm = False)
            if len(spec_T_sup_lg_inf.wavelength) != len(wave):
                spec_T_sup_lg_inf = spec_T_sup_lg_inf.interpolate_wavelength(wave, renorm = False)
            if len(spec_T_sup_lg_sup.wavelength) != len(wave):
                spec_T_sup_lg_sup = spec_T_sup_lg_sup.interpolate_wavelength(wave, renorm = False)
            flux_T_inf = linear_interpolate(spec_T_inf_lg_inf.flux, spec_T_inf_lg_sup.flux, lg_inf, lg_sup, lg)
            flux_T_sup = linear_interpolate(spec_T_sup_lg_inf.flux, spec_T_sup_lg_sup.flux, lg_inf, lg_sup, lg)
            flux       = linear_interpolate(flux_T_inf, flux_T_sup, T_inf, T_sup, T)
        else:  # No interpolation along T, only along lg
            spec_lg_inf = load_spectrum(T_valid, lg_inf, model=model, albedo=albedo, instru=instru, load_path=load_path)
            spec_lg_sup = load_spectrum(T_valid, lg_sup, model=model, albedo=albedo, instru=instru, load_path=load_path)
            wave        = spec_lg_inf.wavelength
            if len(spec_lg_sup.wavelength) != len(wave):
                spec_lg_sup = spec_lg_sup.interpolate_wavelength(wave, renorm = False)
            flux = linear_interpolate(spec_lg_inf.flux, spec_lg_sup.flux, lg_inf, lg_sup, lg)
    else:  # No interpolation along lg
        if T_inf != T_sup:  # Interpolation along T
            spec_T_inf = load_spectrum(T_inf, lg_valid, model=model, albedo=albedo, instru=instru, load_path=load_path)
            spec_T_sup = load_spectrum(T_sup, lg_valid, model=model, albedo=albedo, instru=instru, load_path=load_path)
            wave = spec_T_inf.wavelength
            if len(spec_T_sup.wavelength) != len(wave):
                spec_T_sup = spec_T_sup.interpolate_wavelength(wave, renorm = False)
            flux = linear_interpolate(spec_T_inf.flux, spec_T_sup.flux, T_inf, T_sup, T)
        else:  # No interpolation along T and lg
            spec = load_spectrum(T_valid, lg_valid, model=model, albedo=albedo, instru=instru, load_path=load_path)
            wave = spec.wavelength ; flux = spec.flux
    dl       = wave - np.roll(wave, 1) ; dl[0] = dl[1] ; dl[dl==0] = np.nanmean(dl) # delta lambda array
    R        = np.nanmean(wave/(2*dl)) # calculating the resolution of the raw spectrum
    spectrum = Spectrum(wave, flux, R, T, lg, model)
    return spectrum



def load_planet_spectrum(T_planet=1000, lg_planet=4.0, model="BT-Settl", albedo=False, interpolated_spectrum=True, instru=None, load_path=load_path):
    """
    To read and retrieve planet spectra from models grid with the closest input parameters or compute an interpolated spectrum (see http://svo2.cab.inta-csic.es/theory/newov2/)

    Parameters
    ----------
    T_planet: float
        Planet's temperature (in K)
    lg_planet: float
        Planet's surface gravity (in dex(cm/s2))
    model: str
        model's name
    albedo: bool, optional
        True if the spectrum is an albedo spectrum. The default is False.
    interpolated_spectrum: bool, optional
        If true, interpolates a spectrum at T and lg with spectra from the model grid. The default is True.
    instru: str, optional
        Considered instrument (the grid can depend on the considered intrument). The default is None.
    load_path: str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns
    -------
    spectrum: class Spectrum
        Planet spectrum (in J/s/m2/µm)
    """
    
    T_valid, lg_valid = get_T_lg_valid(T=T_planet, lg=lg_planet, model=model, instru=instru) # closest valid values parameters in the model grid.
    if interpolated_spectrum and (T_valid != T_planet or lg_valid != lg_planet): # interpolates the grid in order to have the precise T_planet and lg_planet values.
        spectrum = interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_planet, lg=lg_planet, model=model, albedo=albedo, instru=instru, load_path=load_path)
    else: # load the spectrum with the closest parameters values in the model grid.
        T_planet  = T_valid
        lg_planet = lg_valid
        spectrum  = load_spectrum(T=T_planet, lg=lg_planet, model=model, albedo=albedo, instru=instru, load_path=load_path)
    return spectrum # in J/s/m2/µm



def load_albedo(T_planet, lg_planet, model="PICASO", interpolated_spectrum=True, instru=None, load_path=load_path):
    """
    To read and retrieve planet albedos from models grid with the closest input parameters or compute an interpolated albedo.

    Parameters
    ----------
    T_planet: float
        Planet's temperature (in K)
    lg_planet: float
        Planet's surface gravity (in dex(cm/s2))
    model: str
        model's name
    interpolated_spectrum: bool, optional
        If true, interpolates a spectrum at T and lg with spectra from the model grid. The default is True.
    instru: str, optional
        Considered instrument (the grid can depend on the considered intrument). The default is None.
    load_path: str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns
    -------
    spectrum: class Spectrum
        Planet albedo (no unit)
    """
    
    albedo = load_planet_spectrum(T_planet=T_planet, lg_planet=lg_planet, model="PICASO", albedo=True, interpolated_spectrum=interpolated_spectrum, instru=instru, load_path=load_path)
    if model == "PICASO": # see Eq.(1) of Lovis et al. (2017): https://arxiv.org/pdf/1609.03082
        pass
    elif model == "flat":
        albedo_geo  = np.nanmean(albedo.flux) # mean value of the geometric albedo given by PICASO
        albedo.flux = np.zeros_like(albedo.flux) + albedo_geo
    elif model == "tellurics":
        wave_tell, tell   = fits.getdata("sim_data/Transmission/sky_transmission_airmass_2.5.fits")
        albedo_geo        = np.nanmean(albedo.flux[(albedo.wavelength >= wave_tell[0]) & (albedo.wavelength <= wave_tell[-1])]) # mean value of the geometric albedo given by PICASO
        albedo.flux       = albedo_geo / np.nanmean(tell) * tell
        albedo.wavelength = wave_tell
        dl                = albedo.wavelength - np.roll(albedo.wavelength, 1) ; dl[0] = dl[1] ; dl[dl==0] = np.nanmean(dl) # delta lambda array
        albedo.R          = np.nanmean(albedo.wavelength/(2*dl)) # calculating the resolution of the raw spectrum
    else:
        raise KeyError(model+" IS NOT A VALID REFLECTED MODEL: tellurics, flat, PICASO")
    return albedo # no unit



def load_star_spectrum(T_star, lg_star, model="BT-NextGen", interpolated_spectrum=True, load_path=load_path):
    """
    Load star spectrum model (in J/s/m2/µm).
    
    Parameters
    ----------
    T_star: (float)
        star temperature (in K).
    lg_star: (float)
        star surface gravity (in dex(cm/s2).
    model: str, optional
        star model. The default is "BT-NextGen".
    interpolated_spectrum: bool, optional
        If true, interpolates a spectrum at T and lg with spectra from the model grid. The default is True.
    load_path: str, optional
        loading path of the files (load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")). The default is load_path.

    Returns
    -------
    spectrum: class Spectrum
        Star spectrum (in J/s/m2/µm)
    """
    
    T_valid, lg_valid = get_T_lg_valid(T=T_star, lg=lg_star, model=model) # closest valid values parameters in the model grid.
    if interpolated_spectrum and (T_valid != T_star or lg_valid != lg_star): # interpolates the grid in order to have the precise T_star and lg_star values.
        spectrum = interpolate_T_lg_spectrum(T_valid=T_valid, lg_valid=lg_valid, T=T_star, lg=lg_star, model=model, load_path=load_path)
    else: # load the spectrum with the closest parameters values in the model grid
        T_star   = T_valid
        lg_star  = lg_valid
        spectrum = load_spectrum(T=T_star, lg=lg_star, model=model, load_path=load_path)
    return spectrum # in J/s/m2/µm
    


def load_vega_spectrum(vega_path=vega_path):
    """
    Load and retrieve vega spectrum (for magnitude calculation purposes)
    
    Returns
    -------
    vega_spectrum: class Spectrum
        Vega spectrum (in J/s/m2/µm)
    """
    f             = fits.getdata(os.path.join(vega_path))
    wave          = f[:, 0]*1e-3 # nm => µm
    flux          = f[:, 1]*10   # 10 = 1e4 * 1e4 * 1e-7: erg/s/cm2/A -> erg/s/cm2/µm -> erg/s/m2/µm -> J/s/m2/µm
    vega_spectrum = Spectrum(wave, flux, None, None)
    return vega_spectrum
        


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def spectrum_instru(band0, R, config_data, mag, spectrum):
    """
    restricts the spectra to the instrument's wavelength range and adjusts it to the input magnitude on band0
    
    Parameters
    ----------
    band0: str
        Spectral band in which the magnitude is entered ("J", "H", etc.)
    R: float
        Spectral resolution of the input spectrum (it can be arbitrary but still must be well above the instrumental spectral resolution)
    config_data: collections
        Gives the specifications of the considered instrument
    mag: float
        Input magnitude
    spectrum: class Spectrum
        Spectrum to restrict and adjust (!must be in J/s/m2/µm!)

    Returns
    -------
    spectrum_instru: class Spectrum
        instrumental-wavelength-range-restricted and magnitude-adjusted spectrum in photons/min received
    """
    try:
        if band0 == "instru":
            lmin_band0 = globals()["lmin_"+config_data["name"]] ; lmax_band0 = globals()["lmax_"+config_data["name"]]
        else:
            lmin_band0 = globals()["lmin_"+band0] ; lmax_band0 = globals()["lmax_"+band0]
    except:
        raise KeyError(f"{band0} is not a considered band to define the magnitude, please choose among: {bands}, {instrus}")
    
    dl_band0   = ((lmax_band0+lmin_band0)/2)/(2*R)
    wave_band0 = np.arange(lmin_band0, lmax_band0, dl_band0)                       # wavelength array on band0 [µm]
    spec       = spectrum.interpolate_wavelength(wave_band0, renorm=False)         # interpolating the input spectrum on band0
    vega_spec  = load_vega_spectrum()                                              # getting the vega spectrum
    vega_spec  = vega_spec.interpolate_wavelength(wave_band0, renorm=False)        # interpolating the vega spectrum on band0
    ratio      = np.nanmean(vega_spec.flux)*10**(-0.4*mag) / np.nanmean(spec.flux) # ratio by which to adjust the spectrum flux in order to have the input magnitude
    
    # Conversion to photons/mn + restriction of spectra to instrumental range + adjustment of spectra to the input magnitude
    lmin_instru      = config_data["lambda_range"]["lambda_min"]
    lmax_instru      = config_data["lambda_range"]["lambda_max"]                    # [µm]
    dl_instru        = ((lmax_instru+lmin_instru)/2)/(2*R)
    wave_instru      = np.arange(lmin_instru, lmax_instru, dl_instru)               # constant and linear wavelength array on the instrumental bandwidth with equivalent resolution than the raw one
    spectrum.flux   *= ratio                                                        # adjusting the spectrum to the input magnitude
    spectrum_density = spectrum.interpolate_wavelength(wave_instru, renorm = False) # in order to have a spectrum in density (i.e. J/s/m2/µm)     
    spectrum_instru  = spectrum.set_nbphotons_min(config_data, wave_instru)         # J/s/m2/µm => photons/mn on the instrumental bandwidth
    
    return spectrum_instru, spectrum_density # in ph/mn and J/s/m2/µm respectively

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

    Returns
    -------
    spectrum_instru: class Spectrum
        band-restricted and resolution-degrated spectrum in photons/min received
    """
    lmin = config_data['gratings'][band].lmin # lambda_min of the considered band
    lmax = config_data['gratings'][band].lmax # lambda_max of the considered band
    R    = config_data['gratings'][band].R    # spectral resolution of the band
    if R is None: # if not a spectro-imager (e.g. NIRCam)
        R = spectrum_instru.R # leaves resolution at native resolution
    dl_band       = ((lmin+lmax)/2)/(2*R)
    wave_band     = np.arange(lmin, lmax, dl_band)                             # constant and linear wavelength array on the considered band
    spectrum_band = spectrum_instru.degrade_resolution(wave_band, renorm=True) # degradation from spectrum resolution to spectral resolution of the considered band
    
    return spectrum_band # degrated spectrum in ph /mn

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filtered_flux(flux, R, Rc, filter_type="gaussian", show=False):
    """
    Gives low-pass and high-pass filtered flux as function of the cut-off resolution Rc

    Parameters
    ----------
    flux: 1d-array
        input flux.
    R: float
        spectral resolution of the input flux.
    Rc: float
        cut-off resolution of the filter.
    filter_type: str, optional
        filter method considered. The default is "gaussian".
    show: bool, optional
        whether to show the plot of original, low-pass, and high-pass filtered flux.

    Returns
    -------
    flux_HF: 1d-array
        high-pass filtered flux.
    flux_LF_valid: 1d-array
        low-pass filtered flux.
    """
    if Rc is None:
        flux_LF_valid = 0 # No filter applied
    else:
        if filter_type == "gaussian":
            sigma = 2 * R / (np.pi * Rc) * np.sqrt(np.log(2) / 2) # see Appendix A of Martos et al. (2024)
            flux_LF_valid = gaussian_filter1d(flux[~np.isnan(flux)], sigma=sigma) # + ignoring NaN values
        elif filter_type == "gaussian_bis": # see Appendix A of Martos et al. (2024)
            fft = np.fft.fft(flux[~np.isnan(flux)])
            ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)]))
            res = ffreq * R * 2
            G = gaussian0(x=res, sig=Rc / (np.sqrt(2 * np.log(2))))
            G /= np.nanmax(G)
            fft *= G
            flux_LF_valid = np.real(np.fft.ifft(fft))
        elif filter_type == "step": # step filter function in the Fourier space
            fft = np.fft.fft(flux[~np.isnan(flux)])
            ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)]))
            res = ffreq * R * 2
            fft[np.abs(res) > Rc] = 0
            flux_LF_valid = np.real(np.fft.ifft(fft))
        elif filter_type == "smoothstep": # smooth filter function in the Fourier space
            fft = np.fft.fft(flux[~np.isnan(flux)])
            ffreq = np.fft.fftfreq(len(flux[~np.isnan(flux)]))
            res = ffreq * R * 2
            fft *= smoothstep(res, Rc)
            flux_LF_valid = np.real(np.fft.ifft(fft))
        elif filter_type == "savitzky_golay": # Savitzky-Golay filter
            sigma = 2 * R / (np.pi * Rc) * np.sqrt(np.log(2) / 2) # see Appendix A of Martos et al. (2024)
            N = 4 * sigma # gaussian approximation (sigma = N/4 is an empirical approximation) 
            window_length = min(len(flux[~np.isnan(flux)]), int(round(N)) + 1)  # set window length based on Rc
            if window_length % 2 == 0:
                window_length += 1  # window length must be odd
            flux_LF_valid = savgol_filter(flux[~np.isnan(flux)], window_length=window_length, polyorder=3)
        else:
            raise KeyError(f'{filter_type} is not a valid filter type: "gaussian", "gaussian_bis", "step", "smoothstep", or "savitzky_golay".')
    flux_LF = np.copy(flux)
    flux_LF[~np.isnan(flux_LF)] = flux_LF_valid
    flux_HF = flux - flux_LF
    if show:
        plt.figure(dpi=300)
        plt.plot(flux, 'r', label="Original Flux")
        plt.plot(flux_LF, 'g', label="Low-Pass Filtered Flux")
        plt.plot(flux_HF, 'b', label="High-Pass Filtered Flux")
        plt.xlabel("Wavelength Axis")
        plt.ylabel("Flux Axis")
        plt.legend()
        plt.show()
    return flux_HF, flux_LF



#######################################################################################################################
############################################# FastYield part: #########################################################
#######################################################################################################################



def thermal_reflected_spectrum(planet, instru=None, thermal_model="BT-Settl", reflected_model="PICASO", wave_instru=None, wave_K=None, vega_spectrum_K=None, show=True, in_im_mag=True):
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
    wave_instru: 1d-array, optional
        wavelength array on which the spectra will be calculated. The default is None.
    wave_K: 1d-array, optional
        wavelength array on which the magnitudes in K-band will be computed. The default is None.
    vega_spectrum: class Spectrum, optional
        vega spectrum on K-band. The default is None. If None, it will be retrieved
    show: bool, optional
        To plots the spectra contributions. The default is True.
    in_im_mag: bool, optional
        To renormalize the imaged planets in the K-band by the measured one. The default is True.
    """
    
    if thermal_model == "None" and reflected_model == "None":
        raise KeyError("PLEASE DEFINE A MODEL FOR THE THERMAL OR THE REFLECTED COMPONENT !")
    
    # K-band
    if wave_K is None:
        R_K    = 10_000 # only for photometric purposes (does not need more resolution)
        dl_K   = ((lmin_K+lmax_K)/2)/(2*R_K)
        wave_K = np.arange(lmin_K, lmax_K, dl_K)

    # instru-band
    if wave_instru is None and instru is not None: # in case a wavelength array is not input, create one
        config_data = get_config_data(instru)
        lmin_instru = config_data["lambda_range"]["lambda_min"] # in µm
        lmax_instru = config_data["lambda_range"]["lambda_max"] # in µm
        R_instru    = R0_max # abritrary resolution (needs to be high enough)
        dl_instru   = ((lmin_instru+lmax_instru)/2)/(2*R_instru)
        wave_instru = np.arange(0.98*lmin_instru, 1.02*lmax_instru, dl_instru)
        
    if vega_spectrum_K is None: # in case a vega spectrum is not input, create one
        vega_spectrum   = load_vega_spectrum()
        vega_spectrum_K = vega_spectrum.interpolate_wavelength(wave_K, renorm = False)
                
    star_spectrum         = load_star_spectrum(float(planet["StarTeff"].value), float(planet["StarLogg"].value)) # load the star spectrum
    star_spectrum_K       = star_spectrum.interpolate_wavelength(wave_K, renorm = False)                         # interpolating it to the wavelength array
    star_spectrum         = star_spectrum.interpolate_wavelength(wave_instru, renorm = False)                    # interpolating it to the wavelength array
    star_spectrum         = star_spectrum.broad(float(planet["StarVsini"].value))
    ratio_star            = np.nanmean(vega_spectrum_K.flux)*10**(-0.4*float(planet["StarKmag"])) / np.nanmean(star_spectrum_K.flux) # renormalizing the star spectrum to the correct magnitude
    star_spectrum.flux   *= ratio_star
    star_spectrum_K.flux *= ratio_star
        
    if thermal_model != "None": # load, reinterpolates and renormalizes the thermal contribution of the planet spectrum
        planet_thermal         = load_planet_spectrum(float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), model=thermal_model, interpolated_spectrum=True)
        planet_thermal_K       = planet_thermal.interpolate_wavelength(wave_K, renorm = False)
        planet_thermal         = planet_thermal.interpolate_wavelength(wave_instru, renorm = False)
        planet_scaling_factor  = float((planet['PlanetRadius']/planet['Distance']).decompose()**2)
        planet_thermal.flux   *= planet_scaling_factor
        planet_thermal_K.flux *= planet_scaling_factor
    elif thermal_model == "None":
        planet_thermal   = Spectrum(wave_instru, np.zeros_like(wave_instru), star_spectrum.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), thermal_model)
        planet_thermal_K = Spectrum(wave_K, np.zeros_like(wave_K), star_spectrum_K.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), thermal_model)
    
    if reflected_model != "None":
        albedo             = load_albedo(float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), model=reflected_model, interpolated_spectrum=True)
        albedo_K           = albedo.interpolate_wavelength(wave_K, renorm=False)
        albedo             = albedo.interpolate_wavelength(wave_instru, renorm=False)
        planet_reflected_K = star_spectrum_K.flux * albedo_K.flux * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
        planet_reflected   = star_spectrum.flux * albedo.flux * planet["g_alpha"] * (planet['PlanetRadius']/planet['SMA']).decompose()**2
    elif reflected_model == "None":
        planet_reflected   = np.zeros_like(wave_instru)*u.dimensionless_unscaled
        planet_reflected_K = np.zeros_like(wave_K)*u.dimensionless_unscaled
    planet_reflected   = Spectrum(wave_instru, np.nan_to_num(np.array(planet_reflected.value)), star_spectrum.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), reflected_model)
    planet_reflected_K = Spectrum(wave_K, np.nan_to_num(np.array(planet_reflected_K.value)), star_spectrum_K.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), reflected_model)
    
    planet_spectrum   = Spectrum(wave_instru, planet_thermal.flux+planet_reflected.flux, star_spectrum.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), thermal_model+"+"+reflected_model)
    planet_spectrum_K = Spectrum(wave_K, planet_thermal_K.flux+planet_reflected_K.flux, star_spectrum_K.R, float(planet["PlanetTeq"].value), float(planet["PlanetLogg"].value), thermal_model+"+"+reflected_model)

    # broadening the planet spectrum
    if thermal_model!="None":
        planet_thermal = planet_thermal.broad(float(planet["PlanetVsini"].value))
    if reflected_model!="None":
        planet_reflected = planet_reflected.broad(float(planet["PlanetVsini"].value))
    planet_spectrum = planet_spectrum.broad(float(planet["PlanetVsini"].value))
    
    # Doppler shifting the star spectrum
    star_spectrum = star_spectrum.doppler_shift(float(planet["StarRadialVelocity"].value))
    
    # Doppler shifting the planet spectrum
    if thermal_model!="None":
        planet_thermal   = planet_thermal.doppler_shift(float(planet["PlanetRadialVelocity"].value))
    if reflected_model!="None":
        planet_reflected = planet_reflected.doppler_shift(float(planet["PlanetRadialVelocity"].value))
    planet_spectrum = planet_spectrum.doppler_shift(float(planet["PlanetRadialVelocity"].value))

    if in_im_mag and planet["DiscoveryMethod"] == "Imaging" and thermal_model != "None": # To inject the known magnitudes of planets detected by direct imaging
        if not np.isnan(planet["PlanetKmag(thermal+reflected)"]): # the magnitude is already known by definition => renormalization in K-band
            ratio                 = np.nanmean(vega_spectrum_K.flux)*10**(-0.4*float(planet["PlanetKmag(thermal+reflected)"])) / np.nanmean(planet_spectrum_K.flux)
            planet_spectrum.flux  = np.copy(planet_spectrum.flux) * ratio
            planet_thermal.flux   = np.copy(planet_thermal.flux) * ratio
            planet_reflected.flux = np.copy(planet_reflected.flux) * ratio

    # plotting the contributions
    if show:
        band0       = "K"
        mag_p_total = -2.5 * np.log10(np.nanmean(planet_spectrum_K.flux) / np.nanmean(vega_spectrum_K.flux))
        plt.figure(figsize=(10, 7), dpi=300)
        plt.xlabel("wavelength [µm]", fontsize=18, labelpad=10)
        plt.ylabel("flux [contrast unit]", fontsize=18, labelpad=10)
        plt.yscale('log')
        plt.xlim(wave_instru[0], wave_instru[-1])
        plt.title(f"Spectrum Contributions for {planet['PlanetName']} \n"f"at {round(float(planet_spectrum.T))}K", fontsize=20, pad=20)            
        plt.plot(wave_instru, planet_spectrum.flux / star_spectrum.flux, color='green', lw=2, linestyle='-', label=f"Thermal+Reflected ({planet_spectrum.model}), mag({band0}) = {round(mag_p_total, 2)}")
        if thermal_model != "None":
            mag_p_thermal = -2.5 * np.log10(np.mean(planet_thermal_K.flux) / np.nanmean(vega_spectrum_K.flux))
            plt.plot(wave_instru, planet_thermal.flux / star_spectrum.flux, color='red', lw=2, linestyle='--', label=f"Thermal ({thermal_model}), mag({band0}) = {round(mag_p_thermal, 2)}")
        if reflected_model != "None":
            mag_p_reflected = -2.5 * np.log10(np.mean(planet_reflected_K.flux) / np.nanmean(vega_spectrum_K.flux))
            plt.plot(wave_instru, planet_reflected.flux / star_spectrum.flux, color='blue', lw=2, linestyle='--', label=f"Reflected ({reflected_model}+BT-NextGen), mag({band0}) = {round(mag_p_reflected, 2)}")        
        if np.nanmax(wave_instru) > lmin_K and np.nanmin(wave_instru) < lmax_K:
            plt.axvspan(lmin_K, lmax_K, color='gray', alpha=0.2, lw=0, label="K-band")        
        plt.ylim(np.nanmin(planet_spectrum.flux / star_spectrum.flux) / 10, np.nanmax(planet_spectrum.flux / star_spectrum.flux) * 10)        
        plt.ylim(max(1e-15, np.nanmin(planet_spectrum.flux / star_spectrum.flux) / 10))        
        plt.legend(fontsize=14, loc='best', frameon=True, facecolor='white', framealpha=0.9)        
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()

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
        os.environ['PYSYN_CDBS']     = '/home/martoss/picaso/grp/redcat/trds/'
        import picaso
        from picaso import justdoit as jdi
    except ImportError:
        print("Tried importing picaso, but couldn't do it")
    return picaso, jdi

def simulate_picaso_spectrum(planet, spectrum_contributions='thermal+reflected', planet_type="gas", clouds=True, stellar_mh=0.0122, opacity=None):
    '''
    TAKEN FROM: https://github.com/planetarysystemsimager/psisim/tree/main
    A function that returns the required inputs for picaso, given a row from a universe planet table. 
    
    Inputs:
    planet - a single row, corresponding to a single planet from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice", or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    stellar_mh - stellar metalicity
    Opacity class from justdoit.opannection
    NOTE: this assumes a planet phase of 0. You can change the phase in the resulting params object afterwards.
    '''
    
    picaso, jdi = import_picaso()
    
    # Retrieve star-planet configuration
    phase =  0. # float(planet['Phase'].value) # in order to have the geometric Albedo (by definition)
    SMA = float(planet['SMA'].value)
    R_star = float(planet['StarRadius'].value)
    host_temp_list = np.hstack([np.arange(3500, 13000, 250), np.arange(13000, 50000, 1000)])
    host_logg_list = [5.00, 4.50, 4.00, 3.50, 3.00, 2.50, 2.00, 1.50, 1.00, 0.50, 0.0] # Define the grids that phoenix / ckmodel models like
    f_teff_grid    = interp1d(host_temp_list, host_temp_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    f_logg_grid    = interp1d(host_logg_list, host_logg_list, kind='nearest', bounds_error=False, fill_value='extrapolate')
    T_star         = f_teff_grid(float(planet['StarTeff'].to(u.K).value))
    lg_star        = f_logg_grid(float(planet['StarLogg'].to(u.dex(u.cm/ u.s**2)).value))
    T_planet  = float(planet['PlanetTeq'].value)
    lg_planet = float(planet['PlanetLogg'].value)
    
    params = jdi.inputs()
    params.approx(raman='none') # see justdoit.py => class inputs():
    params.phase_angle(phase)
    params.gravity(gravity=float(planet['PlanetLogg'].value), gravity_unit=planet['PlanetLogg'].physical.unit) # NOTE: picaso gravity() won't use the "gravity" input if mass and radius are provided
    if lg_star > 5.0: # The current stellar models do not like log g > 5, so we'll force it here for now. 
        lg_star = 5.0
    if T_star < 3000: # The current stellar models do not like Teff < 3000, so we'll force it here for now. 
        T_star = 3000
    params.star(opacity, T_star, stellar_mh, lg_star, radius=R_star, semi_major=SMA, semi_major_unit=planet['SMA'].unit, radius_unit=planet['StarRadius'].unit) 
    if planet_type == 'gas': #-- Define atmosphere PT profile, mixing ratios, and clouds
        params.guillot_pt(T_planet, T_int=150, logg1=-0.5, logKir=-1) # T_int = Internal temperature / logg1, logKir = see parameterization Guillot 2010
        params.channon_grid_high() # get chemistry via chemical equillibrium
        if clouds: # may need to consider tweaking these for reflected light
            params.clouds(g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5]) # g0 = Asymmetry factor / w0 = Single Scattering Albedo / opd = Total Extinction in `dp` / p = Bottom location of cloud deck (LOG10 bars) / dp = Total thickness cloud deck above p (LOG10 bars)
    elif planet_type == 'terrestrial':
        pass # TODO: add Terrestrial type
    elif planet_type == 'ice':
        pass # TODO: add ice type
    
    if phase == 0: # non-0 phases require special geometry which takes longer to run.
        df = params.spectrum(opacity, full_output=True, calculation=spectrum_contributions, plot_opacity=False) # Perform the simple simulation since 0-phase allows simple geometry
    else:
        df1 = params.spectrum(opacity, full_output=True, calculation='thermal') # Perform the thermal simulation as usual with simple geometry
        params.phase_angle(phase, num_tangle=8, num_gangle=8) # Apply the true phase and change geometry for the reflected simulation
        df2 = params.spectrum(opacity, full_output=True, calculation='reflected')
        df = df1.copy() ; df.update(df2) # Combine the output dfs into one df to be returned
        df['full_output_therm'] = df1.pop('full_output')
        df['full_output_ref']   = df2.pop('full_output')
    
    model_wvs  = 1./df['wavenumber'] * 1e4 *u.micron
    argsort    = np.argsort(model_wvs)
    model_wvs  = model_wvs[argsort]
    model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1)) ; model_dwvs[0] = model_dwvs[1]
    model_R    = np.nanmean(model_wvs/(2*model_dwvs))
    model_R    = model_R.value # = 30000
    
    if spectrum_contributions == "thermal":
        planet_thermal    = np.zeros((2, len(model_wvs)))
        planet_thermal[0] = model_wvs
        thermal_flux      = df["thermal"][argsort] * u.erg/u.s/u.cm**2/u.cm
        thermal_flux      = thermal_flux.to(u.J/u.s/u.m**2/u.micron)
        planet_thermal[1] = np.array(thermal_flux.value)
        fits.writeto(f"sim_data/Spectra/planet_spectrum/PICASO/thermal_gas_giant_{round(float(planet['PlanetTeq'].value))}K_lg{round(float(planet['PlanetLogg'].value), 1)}.fits", planet_thermal, overwrite=True)
        plt.figure(dpi=300) ; plt.plot(planet_thermal[0], planet_thermal[1]) ; plt.title(f'Thermal: T = {round(float(planet["PlanetTeq"].value))}K and lg = {round(float(planet["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength [µm]') ; plt.ylabel("flux (in J/s/µm/m2)") ; plt.yscale('log') ; plt.show()
    
    elif spectrum_contributions == "reflected":
        albedo    = np.zeros((2, len(model_wvs)))
        albedo[0] = model_wvs
        albedo[1] = df['albedo'][argsort]
        fits.writeto(f"sim_data/Spectra/planet_spectrum/albedo/albedo_gas_giant_{round(float(planet['PlanetTeq'].value))}K_lg{round(float(planet['PlanetLogg'].value), 1)}.fits", albedo, overwrite=True)
        plt.figure(dpi=300) ; plt.plot(albedo[0], albedo[1]) ; plt.title(f'Albedo: T = {round(float(planet["PlanetTeq"].value))}K and lg = {round(float(planet["PlanetLogg"].value), 1)}') ; plt.xlabel('wavelength [µm]') ; plt.ylabel("albedo") ; plt.yscale('log') ; plt.show()
   
def get_picasso_thermal():
    from src.FastYield import load_planet_table, get_planet_index
    picaso, jdi = import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname         = 'all_opacities_0.6_6_R60000.db'
    dbname         = os.path.join(opacity_folder, dbname)
    opacity        = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx          = get_planet_index(planet_table, "HR 8799 b") # "HR 8799 b" => does not change anything
    planet       = planet_table[idx]
    T0, lg0 = get_model_grid("PICASO")
    for i in tqdm(range(len(T0))):
        T_planet = T0[i]
        for lg_planet in lg0:
            planet["PlanetTeq"]  = T_planet * planet["PlanetTeq"].unit # 
            planet["PlanetLogg"] = lg_planet * planet["PlanetLogg"].unit # 
            simulate_picaso_spectrum(planet, spectrum_contributions="thermal", opacity=opacity)
            
def get_picasso_albedo():
    from src.FastYield import load_planet_table, get_planet_index
    picaso, jdi = import_picaso()
    wvrng = [0.6, 6] # opacity file to load
    opacity_folder = os.path.join(os.getenv("picaso_refdata"), 'opacities')
    dbname         = 'all_opacities_0.6_6_R60000.db'
    dbname         = os.path.join(opacity_folder, dbname)
    opacity        = jdi.opannection(filename_db=dbname, wave_range=wvrng)
    planet_table = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    idx          = get_planet_index(planet_table, "HR 8799 b") # "HR 8799 b" => does not change anything
    planet       = planet_table[idx]
    T0, lg0 = get_model_grid("PICASO")
    for i in tqdm(range(len(T0))):
        T_planet = T0[i]
        for lg_planet in lg0:
            planet["PlanetTeq"]  = T_planet * planet["PlanetTeq"].unit # 
            planet["PlanetLogg"] = lg_planet * planet["PlanetLogg"].unit # 
            simulate_picaso_spectrum(planet, spectrum_contributions="reflected", opacity=opacity)
            









