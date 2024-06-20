from src.config import * # + numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import matplotlib
from scipy.interpolate import interp1d,interp2d
from scipy.ndimage.filters import gaussian_filter
from scipy.special import comb
from PyAstronomy import pyasl
import pandas as pd
import coronagraph as cg
from astropy.io import fits,ascii
from astropy import constants as const
from astropy import units as u
from astropy.convolution import Gaussian1DKernel
from astropy.stats import sigma_clip
import warnings
import os
import time
import ssl
import wget
import lzma
import statsmodels.api as sm
warnings.filterwarnings('ignore', category=UserWarning, append=True)

h = 6.6260701e-34 # J.s
c = 2.9979246e8 # m/s
kB = 1.38065e-23 # J/K



def gaussian(x,mu,sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def gaussian0(x,sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x)/sig, 2.)/2)

def lorentzian(x,x0,L):
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)

def chi2(x,x0,L):
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)

def smoothstep(x, Rc, N=10):
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



def annular_mask(R_int, R_ext, size, value=np.nan): 
    mask = np.zeros(size) + value
    i0, j0 = size[0]//2, size[1]//2
    for i in range(size[0]):    
        for j in range(size[1]):
            if R_ext ** 2 >= ((i0 - i) ** 2 + (j0 - j) ** 2) >= R_int ** 2:
                mask[i, j] = 1
    return mask

def crop(data,Y0=None,X0=None,R_crop=None):
    if data.ndim == 3 :
        cube = np.copy(data)
        cube[np.isnan(cube)] = 0
        A = np.nanmedian(cube,axis=0)
    elif data.ndim == 2 :
        A = np.copy(data)
    maxvalue = np.nanmax(A)
    if Y0 is None and X0 is None :
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j]==maxvalue:
                    Y0 = i ; X0 = j
    if R_crop is None :
        R_crop = max(A.shape[0]-Y0,A.shape[1]-X0,Y0,X0)
    #print("X0 = ",X0 , " / Y0 = ",Y0, " / R_crop = ",R_crop)
    if data.ndim == 3 :
        B = np.zeros((data.shape[0],2*R_crop,2*R_crop)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if  R_crop-Y0+i < 2*R_crop and R_crop-X0+j < 2*R_crop :
                    B[:,R_crop-Y0+i,R_crop-X0+j] = data[:,i,j]
    elif data.ndim == 2 :
        B = np.zeros((2*R_crop,2*R_crop)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[R_crop-Y0+i,R_crop-X0+j] = data[i,j]
    return B

def crop_both(data1,data2,Y0=None,X0=None,R_crop=None):
    if data1.ndim == 3 :
        cube = np.copy(data1)
        cube[np.isnan(cube)] = 0
        A = np.nanmedian(cube,axis=0)
    elif data1.ndim == 2 :
        A = np.copy(data1)
    maxvalue = np.nanmax(A)
    if Y0 is None and X0 is None :
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j]==maxvalue:
                    Y0 = i ; X0 = j
    if R_crop is None:
        R_crop = max(A.shape[0]-Y0,A.shape[1]-X0,Y0,X0)
    #print("X0 = ",X0 , " / Y0 = ",Y0, " / R_crop = ",R_crop)
    if data1.ndim == 3 :
        B = np.zeros((data1.shape[0],2*R_crop,2*R_crop)) + np.nan
        C = np.zeros((data1.shape[0],2*R_crop,2*R_crop)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[:,R_crop-Y0+i,R_crop-X0+j] = data1[:,i,j]
                C[:,R_crop-Y0+i,R_crop-X0+j] = data2[:,i,j]
    elif data1.ndim == 2 :
        B = np.zeros((2*R_crop,2*R_crop)) + np.nan
        C = np.zeros((2*R_crop,2*R_crop)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[R_crop-Y0+i,R_crop-X0+j] = data1[i,j]
                c[R_crop-Y0+i,R_crop-X0+j] = data2[i,j]
    return B,C
    
def dither(cube,factor=10):
    size = cube.shape
    cube_dither=np.zeros((int(size[0]*factor),int(size[1]*factor)))
    for i in range(size[0]):    
        for j in range(size[1]):
            cube_dither[int(i*factor):int((i+1)*factor),int(j*factor):int((j+1)*factor)]=cube[i,j]
    return cube_dither



def PSF_profile_ratio(PSF, pxscale, size_core, sep_unit="arcsec"):
    NbLine, NbColumn = PSF.shape # => donne (taille de l'axe lambda, " de l'axe y , " de l'axe x)
    y0, x0 = NbLine // 2, NbColumn // 2
    PSF_core = PSF[y0-size_core//2:y0+size_core//2+1,x0-size_core//2:x0+size_core//2+1]
    plt.figure() ; plt.imshow(PSF_core) ; plt.show()
    PSF_flux = np.nansum(PSF)
    print(' pxscale =', round(pxscale,4), f" {sep_unit}/px => size core = ", PSF_core.shape[0] , "px")
    fraction_core = np.nansum(PSF_core) / PSF_flux
    profile = np.zeros((2,max(y0,x0))) # array 2D de taille 2x Nbline (ou Column) /2
    for R in range(1, max(y0+1, x0+1)):
        profile[0, R - 1] = R * pxscale
        profile[1, R - 1] = np.nanmean(PSF * annular_mask(max(1,R-1), R, size=(NbLine, NbColumn))) / PSF_flux
    profile[1, :] /= pxscale
    return profile, fraction_core

def register_PSF_ratio(instru, profile, fraction_core, aper_corr, band, star_pos, strehl, apodizer):
    hdr = fits.Header()
    hdr['FC'] = fraction_core
    hdr['AC'] = aper_corr
    fits.writeto("sim_data/PSF/star_"+star_pos+"/PSF_"+instru+"/PSF_"+band+"_"+strehl+"_"+apodizer+".fits", profile, header=hdr,overwrite=True)




def qqplot(map_w):
    list_corr_w = map_w.reshape(map_w.shape) # annular mask CCF = map_w
    list_c_w = list_corr_w[np.isnan(list_corr_w)!=1] # filtrage nan
    # create Q-Q plot with 45-degree line added to plot
    list_cn_w = (list_c_w-np.mean(list_c_w))/np.std(list_c_w) #loi normale centrée (std=1)
    sm.qqplot(list_cn_w, line='45') ; plt.xlim(-5,5) ; plt.ylim(-5,5)
    plt.title('Q-Q plots of the CCF', fontsize = 14) ; plt.ylabel("sample quantiles", fontsize = 14) ; plt.xlabel("theoritical quantiles", fontsize = 14) ; plt.grid(True) ; plt.show()

def qqplot2(map1,map2,band,target_name):
    plt.figure()
    ax = plt.gca()
    map1 = map1[~np.isnan(map1)] # filtrage nan
    map1 = (map1-np.mean(map1))/np.std(map1) #loi normale centrée (std=1)
    sm.qqplot(map1,ax=ax,marker='o', markerfacecolor='b', markeredgecolor='b', alpha=1,label='below 1"')
    map2 = map2[~np.isnan(map2)] # filtrage nan
    map2 = (map2-np.mean(map2))/np.std(map2) #loi normale centrée (std=1)
    sm.qqplot(map2,ax=ax,marker='o', markerfacecolor='r', markeredgecolor='r', alpha=1,label='above 1"')
    sm.qqline(ax=ax, line='45', fmt='k')
    plt.legend()
    plt.title(f'Q-Q plots of the CCF of {target_name} on {band}', fontsize = 14) ; plt.ylabel("sample quantiles", fontsize = 14) ; plt.xlabel("theoritical quantiles", fontsize = 14) ; plt.grid(True) ; plt.show()



def open_data(instru,target_name,band,crop_band=True,cosmic=False,sigma_cosmic=5,file=None,X0=None,Y0=None,R_crop=None,print_value=True):
    config_data = get_config_data(instru)
    if file is None :
        if instru=="MIRIMRS" : 
            if target_name=="HR8799" or target_name=="Bpic" or "SIM Bpic" in target_name or target_name=="GJ504":
                file = 'data/MIRIMRS/MIRISim/'+target_name+'_'+band+'_s3d.fits'
            else :
                file = 'data/MIRIMRS/MAST/'+target_name+'_ch'+band[0]+'-shortmediumlong_s3d.fits'
        elif instru=="NIRSpec":
            file = 'data/NIRSpec/MAST/'+target_name+'_nirspec_'+band+'_s3d.fits'
    f = fits.open(file)
    hdr = f[1].header ; hdr0 = f[0].header
    pxsteradian = hdr['PIXAR_SR'] # Nominal pixel area in steradians
    if instru=="MIRIMRS" :
        if target_name == "HR8799" or target_name == "GJ504" or target_name == "Bpic" or "SIM Bpic" in target_name or "sim" in target_name : # les données MIRISIM sont déjà par bande (et non par channel)
            exposure_time=f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
        elif crop_band :
            target_name = hdr0['TARGNAME']
            exposure_time=f[0].header['EFFEXPTM']/3/60 # in mn / Effective exposure time
        else:
            target_name = hdr0['TARGNAME']
            exposure_time=f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
    elif instru == "NIRSpec":
        target_name = hdr0['TARGNAME']
        exposure_time=f[0].header['EFFEXPTM']/60 # in mn / Effective exposure time
    DIT = f[0].header['EFFINTTM']/60 # in mn
    pxscale = hdr['CDELT1']*3600 # data pixel scale in "/px
    step = hdr['CDELT3'] # delta_lambda in µm
    cube = f[1].data # en MJy/Steradian (densité "angulaire" de flux mesurée dans chaque pixel)
    err = f[2].data # erreur sur le flux en MJy/Sr
    wave = (np.arange(hdr['NAXIS3'])+hdr['CRPIX3']-1)*hdr['CDELT3']+hdr['CRVAL3'] # axe de longueur d'onde des données en µm
    area = config_data['telescope']['area'] # aire collectrice m²
    cube *= pxsteradian*1e6 ; err *= pxsteradian*1e6 # MJy/Steradian => Jy/px
    cube *= 1e-26 ; err *= 1e-26 # Jy/pixel => J/s/m²/Hz/px
    for i in range(cube.shape[0]):
        cube[i] *= c/((wave[i]*1e-6)**2) ; err[i] *= c/((wave[i]*1e-6)**2) # J/s/m²/Hz/px => J/s/m²/m/px
    cube *= step*1e-6 ; err *= step*1e-6 # J/s/m²/m/px => J/s/m²/px
    for i in range(cube.shape[0]):
        cube[i] *= wave[i]*1e-6/(h*c) ; err[i] *= wave[i]*1e-6/(h*c) # J/s/m²/px => ph/s/m²/px
    cube *= area ; err *= area # ph/s/m²/px => photons/s/px
    if crop_band : 
        cube = cube[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
        err = err[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
        wave = wave[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
        wave_trans,trans=fits.getdata("sim_data/Transmission/"+instru+"/Instrumental_transmission/transmission_"+band+".fits")
        f = interp1d(wave_trans, trans, bounds_error=False, fill_value=0) 
        trans = f(wave)
        for i in range(cube.shape[0]):
            cube[i] *= trans[i] ; err[i] *= trans[i] # e-/s/pixel
    else :
        trans = 1
    cube *= exposure_time*60 ; err *= exposure_time*60 # e-/pixel or ph/pixel
    cube,err = crop_both(cube,err,X0=X0,Y0=Y0,R_crop=R_crop) # on met l'étoile au centre du cube
    cube[cube==0] = np.nan ; err[err==0] = np.nan
    if cosmic :
        NbChannel, NbLine, NbColumn = cube.shape
        Y = np.reshape(cube,(NbChannel, NbLine*NbColumn))
        Z = np.reshape(err,(NbChannel, NbLine*NbColumn))
        for k in range(Y.shape[1]):
            if not all(np.isnan(Y[:,k])):
                sg = sigma_clip(Y[:,k],sigma=sigma_cosmic)
                Y[:,k] = np.array(np.ma.masked_array(Y[:,k],mask=sg.mask).filled(np.nan))
                Z[:,k] = np.array(np.ma.masked_array(Z[:,k],mask=sg.mask).filled(np.nan))
                sg = sigma_clip(Z[:,k],sigma=sigma_cosmic)
                Y[:,k] = np.array(np.ma.masked_array(Y[:,k],mask=sg.mask).filled(np.nan))
                Z[:,k] = np.array(np.ma.masked_array(Z[:,k],mask=sg.mask).filled(np.nan))
        cube = Y.reshape((NbChannel, NbLine, NbColumn))
        err = Z.reshape((NbChannel, NbLine, NbColumn))
    if print_value :
        print("\n exposure time = ",round(exposure_time,3), "mn and DIT = ", round(DIT*60,3) , 's') 
        print(' target name =', target_name, " / pixelscale =",round(pxscale,3),' "/px')
        if instru=="MIRIMRS":
            print(" DATE :",hdr0["DATE-OBS"])
        try :
            print(" TITLE :", hdr0["TITLE"])
        except : 
            pass
        dl = wave - np.roll(wave, 1) ; dl[0] = dl[1] # array de delta Lambda
        R = np.nanmean(wave/(2*dl)) # calcule de la nouvelle résolution
        print(' R = ', round(R))        
    return cube,wave,pxscale,err,trans,exposure_time,DIT



def PCA_subtraction(Sres, n_comp_sub,y0=None,x0=None,size_core=None,PCA_annular=False,PCA_mask=False,scree_plot=False,PCA_plots=False,wave=None,R=None):
    if n_comp_sub != 0 :
        NbChannel, NbColumn, NbLine = Sres.shape
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_comp_sub)
        X_mask = np.copy(Sres)
        if y0 is not None and x0 is not None : 
            if PCA_annular :
                planet_sep = int(round(np.sqrt((y0-NbLine//2)**2+(x0-NbColumn//2)**2)))
                X_mask *= annular_mask(max(1,planet_sep-size_core-1),planet_sep+size_core,value=np.nan,size=(NbLine,NbColumn))
            if PCA_mask :
                X_mask[:,y0-1:y0+2,x0-1:x0+2] = np.nan 
        X_mask = np.reshape(X_mask, (NbChannel, NbColumn * NbLine)).transpose()
        X_mask[np.isnan(X_mask)] = 0
        X_r = pca.fit_transform(X_mask)
        X_restored = pca.inverse_transform(X_r)
        X = np.reshape(Sres, (NbChannel, NbColumn * NbLine)).transpose()
        X[np.isnan(X)] = 0
        X_sub = (X - X_restored).transpose()
        X_sub = np.reshape(X_sub, (NbChannel, NbColumn, NbLine))
        X_sub[X_sub == 0] = np.nan
        if y0 is not None and x0 is not None and PCA_mask :
            for k in range(n_comp_sub):     
                for i in range(-1,2):
                    for j in range(-1,2) :
                        X_sub[:, y0+i, x0+j] -= np.nan_to_num(np.nansum(X_sub[:, y0+i, x0+i]*pca.components_[k])*pca.components_[k])
        
        if PCA_plots :
            Nk = 4 # nb de modes à plot
            plt.figure() ; plt.title(f"{Nk} first modes of the PCA") ; cmap = get_cmap("Spectral", Nk) ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("flux (normalized)")
            for k in range(0,Nk):
                plt.plot(wave,pca.components_[k],c=cmap(k),label=f"$n_k$ = {k+1}")
            plt.legend() ; plt.show()
            from src.spectrum import Spectrum
            plt.figure() ; plt.title(f"PSD of the {Nk} first modes of the PCA") ; cmap = get_cmap("Spectral", Nk) ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("PSD")
            for k in range(0,Nk):
                m_HF_spectrum = Spectrum(wave, pca.components_[k], R, None)
                res,psd = m_HF_spectrum.plot_psd(smooth=1, color='b',show=False,ret=True)
                plt.plot(res,psd,c=cmap(k),label=f"$n_k$ = {k+1}")
            plt.legend() ; plt.xscale('log') ; plt.yscale('log') ; plt.show()
            # for k in range(0,Nk):
            #     CCF = np.zeros((Sres.shape[1],Sres.shape[2]))
            #     for i in range(Sres.shape[1]):
            #         for j in range(Sres.shape[2]):
            #             CCF[i,j] = np.nan_to_num(np.nansum(Sres[:,i,j]*pca.components_[k]))/np.sqrt(np.nansum(Sres[:,i,j]**2))
            #     plt.figure()
            #     plt.imshow(CCF)
            #     cbar = plt.colorbar() ; cbar.set_label(r"cos $\theta_{est}$", fontsize=14, labelpad=20, rotation=270)
            #     plt.show()
                
        if scree_plot :
            X = np.reshape(Sres, (NbChannel, NbColumn * NbLine)).transpose()
            X[np.isnan(X)] = 0
            # Effectuer l'ACP
            pca = PCA(n_components=n_comp_sub)
            pca.fit_transform(X)
            # Obtenir les valeurs propres
            eigenvalues = pca.explained_variance_
            # Calculer laproportion de variance expliquée par chaque composante
            explained_variance_ratio = pca.explained_variance_ratio_
            # Tracer le scree plot
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
            plt.xlabel('Composantes principales')
            plt.ylabel('Valeurs propres')
            plt.title('Scree Plot')
            #plt.xticks(np.arange(1, len(eigenvalues) + 1))
            plt.yscale('log')
            plt.grid(True)
            plt.show()
    elif n_comp_sub == 0:
        X_sub = np.copy(Sres)
        pca = None
    return np.copy(X_sub) , pca



def model_T_lg_array(model):
    if model=="Exo-REM":
        lg = np.array([3.5,4.0])
        T = np.arange(500,2100,100) # K
    elif model=="BT-Settl" or model=="PICASO":
        lg = np.array([3.0,3.5,4.0,4.5,5.0])
        T = np.append(np.arange(500,1050,50),np.arange(1000,3100,100))
    elif model=="Morley":
        lg = np.array([4,4.5,5,5.5])
        T = np.arange(500,1400,100) # K
    elif model=="SONORA":
        lg = np.array([3,3.5,4,4.5,5,5.5])
        T = np.arange(500,2500,100)
    elif model=="BT-Dusty":
        lg = np.array([4.5,5])
        T = np.arange(1400,3100,100)
    elif model=="Saumon":
        lg = np.array([3,3.5,4,4.5,5])
        T = np.arange(400,1250,50)
    elif model[:4] == "mol_" :
        lg = np.array(["H2O","CO2","O3","N2O","CO","CH4","O2","NO","SO2","NO2","NH3"])
        T = np.append(np.arange(400,1050,50),np.arange(1000,3100,100))
    return T , lg



def propagate_coordinates_at_epoch(targetname, date, verbose=True):
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord,Distance
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
    target_coord_j2000 = SkyCoord(ra, dec, unit=(u.hourangle, u.deg),pm_ra_cosdec=pm_ra * u.mas / u.year,pm_dec=pm_dec * u.mas / u.year,distance=Distance(parallax=plx * u.mas),frame='icrs', obstime='J2000.0')
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
        wavelen_array = fits.getdata(filename.replace(".fits","_wavelen_array.fits")) # µm
        dra_as_array = fits.getdata(filename.replace(".fits","_dra_as_array.fits")) # arcsec
        ddec_as_array = fits.getdata(filename.replace(".fits","_ddec_as_array.fits")) # arcsec
        area2d = fits.getdata(filename.replace(".fits","_area2d.fits")) # arcsec^2
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
        area_fname = hdr0["R_AREA"].replace("crds://", os.path.join("/home/martoss/crds_cache", "references", "jwst","nirspec") + os.path.sep)
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
                    # print(xmax, xmin,ymax, ymin,ymax - ymin,xmax - xmin)
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
        fits.writeto(filename.replace(".fits","_wavelen_array.fits"),wavelen_array,overwrite=True)
        fits.writeto(filename.replace(".fits","_dra_as_array.fits"),dra_as_array,overwrite=True)
        fits.writeto(filename.replace(".fits","_ddec_as_array.fits"),ddec_as_array,overwrite=True)
        fits.writeto(filename.replace(".fits","_area2d.fits"),area2d,overwrite=True)
    return wavelen_array, dra_as_array, ddec_as_array, area2d









