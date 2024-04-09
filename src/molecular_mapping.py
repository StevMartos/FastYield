from src.spectrum import *
import numpy as np
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
from src.FastCurves import *
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.mlab as mlab

c=299792458 # m/s
h=6.626e-34 # J.s
warnings.filterwarnings('ignore', category=UserWarning, append=True)




def flux_shift(wave,flux,rv) :
    rv = rv * (u.km / u.s) # définit rv comme étant en km/s avec astropy
    rv = rv.to(u.m / u.s) # convertit rv en m/s avec astropy 
    R = 100000
    dl = np.nanmean(wave/(2*R))
    wave_hr = np.arange(wave[0],wave[-1]+dl,dl)
    wshift = wave_hr * (1 + (rv / const.c)).value # const.c = vitesse de la lumière (via astropy) / axe de longueur d'onde décalé
    f = interp1d(wave, flux, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
    flux = f(wave_hr) # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
    f = interp1d(wshift, flux, bounds_error=False, fill_value=np.nan,) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
    flux_shift = f(wave_hr) # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
    f = interp1d(wshift, flux, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
    flux_shift = f(wave) # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
    return flux_shift



def qqplot(map_w):
    list_corr_w = map_w.reshape(map_w.shape) # annular mask CCF = map_w
    list_c_w = list_corr_w[np.isnan(list_corr_w)!=1] # filtrage nan
    # create Q-Q plot with 45-degree line added to plot
    list_cn_w = (list_c_w-np.mean(list_c_w))/np.std(list_c_w) #loi normale centrée (std=1)
    plt.figure()
    sm.qqplot(list_cn_w, line='45') ; plt.xlim(-5,5) ; plt.ylim(-5,5)
    plt.title('Q-Q plots of the CCF', fontsize = 14) ; plt.ylabel("sample quantiles", fontsize = 14) ; plt.xlabel("theoritical quantiles", fontsize = 14) ; plt.grid(True) ; plt.show()


def qqplot2(map1,map2):
    plt.figure()
    ax = plt.gca()
    map1 = map1[~np.isnan(map1)] # filtrage nan
    map1 = (map1-np.mean(map1))/np.std(map1) #loi normale centrée (std=1)
    sm.qqplot(map1,ax=ax,marker='o', markerfacecolor='b', markeredgecolor='b', alpha=1,label='below 0.8"')
    map2 = map2[~np.isnan(map2)] # filtrage nan
    map2 = (map2-np.mean(map2))/np.std(map2) #loi normale centrée (std=1)
    sm.qqplot(map2,ax=ax,marker='o', markerfacecolor='r', markeredgecolor='r', alpha=1,label='above 0.8"')
    sm.qqline(ax=ax, line='45', fmt='k')
    plt.legend()
    plt.title('Q-Q plots of the CCF of CT Cha b on 1SHORT', fontsize = 14) ; plt.ylabel("sample quantiles", fontsize = 14) ; plt.xlabel("theoritical quantiles", fontsize = 14) ; plt.grid(True) ; plt.show()


def annular_mask(R_int, R_ext, value=np.nan, size=(215,215)): 
    mask = np.zeros(size) + value
    i0, j0 = size[0]//2, size[1]//2
    for i in range(size[0]):    
        for j in range(size[1]):
            if R_ext ** 2 >= ((i0 - i) ** 2 + (j0 - j) ** 2) >= R_int ** 2:
                mask[i, j] = 1
    return mask



def crop(data,Y0=None,X0=None):
    if data.ndim == 3 :
        cube = np.copy(data)
        cube[np.isnan(cube)] = 0
        A = np.nanmedian(cube,axis=0)
    elif data.ndim == 2 :
        A = np.copy(data)
    maxvalue = np.nanmax(A)
    if Y0 is None or X0 is None :
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j]==maxvalue:
                    Y0=i ; X0=j
    R = max(A.shape[0]-Y0,A.shape[1]-X0,Y0,X0)
    #print("HERE =" , R) ; R=30
    if data.ndim == 3 :
        B = np.zeros((data.shape[0],2*R,2*R)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[:,R-Y0+i,R-X0+j] = data[:,i,j]
    elif data.ndim == 2 :
        B = np.zeros((2*R,2*R)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[R-Y0+i,R-X0+j] = data[i,j]
    return B

def crop(data1,data2,Y0=None,X0=None):
    if data1.ndim == 3 :
        cube = np.copy(data1)
        cube[np.isnan(cube)] = 0
        A = np.nanmedian(cube,axis=0)
    elif data1.ndim == 2 :
        A = np.copy(data1)
    maxvalue = np.nanmax(A)
    if Y0 is None or X0 is None :
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j]==maxvalue:
                    Y0=i ; X0=j
    R = max(A.shape[0]-Y0,A.shape[1]-X0,Y0,X0)
    #print("HERE =" , R) ; R=30
    if data1.ndim == 3 :
        B = np.zeros((data1.shape[0],2*R,2*R)) + np.nan
        C = np.zeros((data1.shape[0],2*R,2*R)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[:,R-Y0+i,R-X0+j] = data1[:,i,j]
                C[:,R-Y0+i,R-X0+j] = data2[:,i,j]

    elif data1.ndim == 2 :
        B = np.zeros((2*R,2*R)) + np.nan
        C = np.zeros((2*R,2*R)) + np.nan
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B[R-Y0+i,R-X0+j] = data1[i,j]
                c[R-Y0+i,R-X0+j] = data2[i,j]

    return B,C
    
    

def dither(cube,factor=10):
    size = cube.shape
    cube_dither=np.zeros((int(size[0]*factor),int(size[1]*factor)))
    for i in range(size[0]):    
        for j in range(size[1]):
            cube_dither[int(i*factor):int((i+1)*factor),int(j*factor):int((j+1)*factor)]=cube[i,j]
    return cube_dither



def open_data(instru,target_name,band,R,crop_band=False,cosmic=False,sigma_cosmic=5):
    config_data = get_config_data(instru)
    if instru=="MIRIMRS" : 
        if target_name=="HR8799" or target_name=="Bpic" or target_name=="GJ504":
            file = 'data/MIRIMRS/MIRISim/'+target_name+'_'+band+'_s3d.fits'
        else :
            file = 'data/MIRIMRS/MAST/'+target_name+'_ch'+band[0]+'-shortmediumlong_s3d.fits'
    elif instru=="NIRSpec":
        file = 'data/NIRSpec/MAST/'+target_name+'_nirspec_'+band+'_s3d.fits'
    f = fits.open(file)
    hdr = f[1].header ; hdr0 = f[0].header
    pxsteradian = hdr['PIXAR_SR'] # Nominal pixel area in steradians
    if instru == "MIRIMRS" :
        if target_name == "HR8799" or target_name == "GJ504" or target_name == "Bpic" : # les données MIRISIM sont déjà par bande (et non par channel)
            exposure_time=f[0].header['EFFEXPTM'] # in s / Effective exposure time
        elif crop_band :
            target_name = hdr0['TARGNAME']
            exposure_time=f[0].header['EFFEXPTM']/3 # in s / Effective exposure time
        else:
            target_name = hdr0['TARGNAME']
            exposure_time=f[0].header['EFFEXPTM'] # in s / Effective exposure time
    elif instru == "NIRSpec":
        target_name = hdr0['TARGNAME']
        exposure_time=f[0].header['EFFEXPTM'] # in s / Effective exposure time
    DIT = f[0].header['EFFINTTM']
    pxscale = hdr['CDELT1']*3600 # data pixel scale in "/px
    step = hdr['CDELT3'] # delta_lambda in µm
    wave_0 = hdr['CRVAL3'] # first lambda in µm
    S = f[1].data # en MJy/Steradian (densité "angulaire" de flux mesurée dans chaque pixel)
    err = f[2].data # erreur sur le flux en MJy/Sr
    wave = (np.arange(hdr['NAXIS3'])+hdr['CRPIX3']-1)*hdr['CDELT3']+hdr['CRVAL3'] # axe de longueur d'onde des données en µm
    area = config_data['telescope']['area'] # aire collectrice m²
    S *= pxsteradian*1e6 ; err *= pxsteradian*1e6 # MJy/Steradian => Jy/px
    S *= 1e-26 ; err *= 1e-26 # Jy/pixel => J/s/m²/Hz/px
    for i in range(S.shape[0]):
        S[i] *= c/((wave[i]*1e-6)**2) ; err[i] *= c/((wave[i]*1e-6)**2) # J/s/m²/Hz/px => J/s/m²/m/px
    S *= step*1e-6 ; err *= step*1e-6 # J/s/m²/m/px => J/s/m²/px
    for i in range(S.shape[0]):
        S[i] *= wave[i]*1e-6/(h*c) ; err[i] *= wave[i]*1e-6/(h*c) # J/s/m²/px => ph/s/m²/px
    S *= area ; err *= area # ph/s/m²/px => photons/s/px
    if instru=="MIRIMRS" : 
        if crop_band: 
            S = S[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
            err = err[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
            wave = wave[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
            wave_trans,trans=fits.getdata("sim_data/Transmission/MIRIMRS/Instrumental_transmission/transmission_"+band+".fits")
            f = interp1d(wave_trans, trans, bounds_error=False, fill_value=0) 
            trans = f(wave)
            for i in range(S.shape[0]):
                S[i] *= trans[i] ; err[i] *= trans[i] # e-/s/pixel
        else :
            trans = 1
    elif instru=="NIRSpec":
        S = S[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
        err = err[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
        wave = wave[(wave>config_data['gratings'][band].lmin) & (wave<config_data['gratings'][band].lmax)]
        wave_trans,trans=fits.getdata("sim_data/Transmission/NIRSpec/Instrumental_transmission/transmission_"+band+".fits")
        f = interp1d(wave_trans, trans, bounds_error=False, fill_value=0) 
        trans = f(wave)
        for i in range(S.shape[0]):
            S[i] *= trans[i] ; err[i] *= trans[i] # e-/s/pixel
    S *= exposure_time ; err *= exposure_time # e-/pixel or ph/pixel
    cube = crop(S) ; err = crop(err) # on met l'étoile au centre du cube
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
    star_flux = np.nansum(cube,axis=(1,2))/(exposure_time/60) # en e-/mn
    plt.figure() ; plt.plot(wave,star_flux,'b') ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("flux (in e-/mn)") ; plt.title(f"Star flux estimated from data on {band} of {instru} for {target_name}") ; plt.grid(True) ; plt.show()
    B = np.zeros((2,np.size(wave))) ; B[0,:] = wave ; B[1,:] = star_flux
    fits.writeto("utils/star_spectrum/star_spectrum_"+band+"_data.fits",B,overwrite=True)
    print("\n exposure time = ",round(exposure_time/60,3), "min et DIT = ", round(DIT,3) , 's') 
    print(' target name =', target_name, " / pixelscale =",round(pxscale,3),' "/px & pxsteradian =', pxsteradian,"Sr/px")
    if target_name != "HR8799" or target_name != "GJ504" or target_name != "Bpic" :
        print(" DATE :",hdr0["DATE-OBS"])
    dl = wave - np.roll(wave, 1) ; dl[0] = dl[1] # array de delta Lambda
    R = np.nanmean(wave/(2*dl)) # calcule de la nouvelle résolution
    print(' R = ', R)
    return cube,wave,pxscale,err,trans,target_name,exposure_time



def stellar_high_filtering(c,calculation,renorm_cube_res,R,Rc,used_filter,cosmic=False,sigma_cosmic=5,print_value=True):
    cube = np.copy(c)
    stell = np.nansum(cube,(1,2)) # spectre stellaire estimé
    NbChannel, NbLine, NbColumn = cube.shape
    Y = np.reshape(cube,(NbChannel, NbLine*NbColumn))
    cube_M = np.copy(Y) ; m = 0
    for k in range(Y.shape[1]):
        if not all(np.isnan(Y[:,k])):
            _,M = filtered_flux(Y[:,k]/stell,R,Rc,used_filter)
            cube_M[:,k] = Y[:,k]/stell #  "vraie" modulation avec bruits (en supposant stell comme étant le vrai spectre de l'étoile)
            if cosmic :
                #plt.figure() ; plt.plot(Y[:,k] - stell*M,"r",label="without cosmic filtering")
                sg = sigma_clip(Y[:,k]-stell*M,sigma=sigma_cosmic)
                Y[:,k] = np.array(np.ma.masked_array(sg,mask=sg.mask).filled(np.nan))
                #plt.plot(Y[:,k],"b",label="with cosmic filtering") ; plt.ylim(np.nanmin(Y[:,k]),np.nanmax(Y[:,k])) ; plt.show()
            else :
                Y[:,k] = Y[:,k] - stell*M
                

            m += np.nansum(M) # pour chaque channel, sum(M) doit être égale à 1
    if print_value :
        print("\n norme de M =", round(m/(NbChannel),3))
    cube_res =  Y.reshape((NbChannel, NbLine, NbColumn))
    cube_M = cube_M.reshape((NbChannel, NbLine, NbColumn))
    cube_res[cube_res==0] = np.nan ; cube_M[cube_M==0] = np.nan
    if renorm_cube_res: # on peut renormaliser le spectre de chaque spaxel afin d'avoir directement cos theta
        for i in range(NbLine):
            for j in range(NbColumn): # pour chaque spaxel
                if not all(np.isnan(cube_res[:,i,j])): # on ignore les NaN
                    cube_res[:,i,j] = cube_res[:,i,j]/np.sqrt(np.nansum(cube_res[:,i,j]**2))
    return cube_res , cube_M



def molecular_mapping_rv(instru,band,cube_high_filtered,T,lg,model,wave,trans,calculation,R,Rc,used_filter,broad=0,rv=0,print_value=True):
    config_data = get_config_data(instru)
    if instru=="MIRIMRS" and model=="Exo-REM":
        planet_spectrum = load_planet_spectrum(T,lg,model=model,version="old")
    else:
        planet_spectrum = load_planet_spectrum(T,lg,model=model)
    planet_spectrum.crop(0.98*wave[0],1.02*wave[-1])
    if broad > 0:
        planet_spectrum = planet_spectrum.broad_r(broad)
    planet_spectrum = planet_spectrum.degrade_resolution(wave,renorm=False)
    planet_spectrum = planet_spectrum.set_nbphotons_min(config_data,wave)
    NbChannel, NbLine, NbColumn = cube_high_filtered.shape
    if calculation=="SNR_rv":
        rv = np.linspace(-50,50,101)
    else :
        rv = np.array([rv])
    CCF = np.zeros((len(rv), NbLine, NbColumn))
    for k in range(len(rv)):
        if calculation != "SNR_rv" and print_value:
            print(" CCF for : rv = ", rv[k], " km/s & Tp =",T,"K & lg = ",lg, " & band = ",band)
        planet_shift = planet_spectrum.doppler_shift(rv[k]) # en ph/mn
        valid = ~np.isnan(planet_shift.flux)
        planet_shift.flux = planet_shift.flux[valid] ; planet_shift.wavelength = planet_shift.wavelength[valid]
        template,_ = filtered_flux(planet_shift.flux,R,Rc,used_filter)
        f = interp1d(planet_shift.wavelength, template, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        template = f(wave)*trans # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
        for i in range(NbLine):
            for j in range(NbColumn): # pour chaque spaxel
                if not all(np.isnan(cube_high_filtered[:,i,j])): # on ignore les NaN
                    t = np.copy(template)
                    t[np.isnan(cube_high_filtered[:,i,j])] = np.nan
                    t = t/np.sqrt(np.nansum(t**2)) # on normalise le template à 1
                    S_res = np.copy(cube_high_filtered[:,i,j])
                    S_res[np.isnan(t)] = np.nan
                    CCF[k,i,j] = np.nansum((S_res*t)) # correlation entre les données et le template
    CCF[CCF==0]=np.nan
    if calculation=="SNR_rv":
        return CCF,rv,t
    else :
        return CCF[0],t



def correlation_rv(instru,sd_HF,wave,trans,T,lg,R,Rc,used_filter,model="BT-Settl",show=True,large_rv=False,broad=0):
    config_data = get_config_data(instru)
    if instru=="MIRIMRS" and model=="Exo-REM":
        planet_spectrum = load_planet_spectrum(T,lg,model=model,version="old")
    else:
        planet_spectrum = load_planet_spectrum(T,lg,model=model)
    planet_spectrum.crop(0.98*wave[0],1.02*wave[-1])
    if broad > 0:
        planet_spectrum = planet_spectrum.broad_r(broad)
    planet_spectrum = planet_spectrum.degrade_resolution(wave,renorm=False)
    planet_spectrum = planet_spectrum.set_nbphotons_min(config_data,wave)
    rv = np.linspace(-50,50,401)
    if large_rv :
        rv = np.linspace(-2000,2000,200)
    cos_theta = np.zeros_like(rv) ; CCF_rv = np.zeros_like(rv)
    for i in range(len(rv)) :
        planet_shift = planet_spectrum.doppler_shift(rv[i]) # en ph/mn
        valid = ~np.isnan(planet_shift.flux)
        planet_shift.flux = planet_shift.flux[valid] ; planet_shift.wavelength = planet_shift.wavelength[valid]
        Sp_HF,_ = filtered_flux(planet_shift.flux,R,Rc,used_filter)
        f = interp1d(planet_shift.wavelength, Sp_HF, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        Sp_HF = f(wave)*trans # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
        Sp_HF[np.isnan(sd_HF)] = np.nan
        S_res = np.copy(sd_HF)
        S_res[np.isnan(Sp_HF)] = np.nan
        cos_theta[i] = np.nansum(S_res*Sp_HF) / np.sqrt( np.nansum(S_res**2) * np.nansum(Sp_HF**2))
    if show :     
        plt.figure() ; plt.plot(wave,Sp_HF/np.sqrt(np.nansum(Sp_HF**2)),'b',label="template") ; plt.plot(wave,S_res/np.sqrt(np.nansum(S_res**2)),'r',label="data") ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("high-pass flux (normalized") ; plt.legend() ; plt.show()
    return cos_theta , rv



def correlation_rv_corr(instru,sd_HF,wave,wave_hr,trans_hr,T,lg,sigma,model="BT-Settl",show=True,large_rv=False,broad=0):
    config_data = get_config_data(instru)
    if instru=="MIRIMRS" and model=="Exo-REM":
        planet_spectrum = load_planet_spectrum(T,lg,model=model,version="old")
    else:
        planet_spectrum = load_planet_spectrum(T,lg,model=model)
    planet_spectrum.crop(0.98*wave[0],1.02*wave[-1])
    if broad > 0:
        planet_spectrum = planet_spectrum.broad_r(broad)
    planet_spectrum = planet_spectrum.degrade_resolution(wave,renorm=False)
    planet_spectrum = planet_spectrum.set_nbphotons_min(config_data,wave)
    rv = np.linspace(-50,50,401)
    if large_rv :
        rv = np.linspace(-2000,2000,200)
    cos_theta = np.zeros_like(rv) ; CCF_rv = np.zeros_like(rv)
    for i in range(len(rv)) :
        planet_shift = planet_spectrum.doppler_shift(rv[i]) # en ph/mn
        valid = ~np.isnan(planet_shift.flux)
        planet_shift.flux = planet_shift.flux[valid] ; planet_shift.wavelength = planet_shift.wavelength[valid]
        template = planet_shift.flux - gaussian_filter(planet_shift.flux,sigma=sigma) # en ph
        f = interp1d(planet_shift.wavelength, template, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        template = f(wave)*trans # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
        template[np.isnan(sd_HF)] = np.nan
        template = template / np.sqrt(np.nansum(template**2)) # On suppose qu'on a le template "parfait"
        template_data = np.copy(sd_HF)
        template_data[np.isnan(template)] = np.nan
        template_data /= np.sqrt(np.nansum(template_data**2))
        cos_theta[i] = np.nansum(template_data*template)
    if show :     
        plt.figure() ; plt.plot(wave,template,'b',label="template") ; plt.plot(wave,template_data,'r',label="data") ; plt.xlabel("wavelength (in µm)") ; plt.ylabel("high-pass flux (normalized") ; plt.legend() ; plt.show()
    return cos_theta , rv



def correlation_PSF(cube_M,CCF):
    PSF = np.nanmean(cube_M,axis=0)
    PSF = PSF/np.sqrt(np.nansum(PSF**2))
    idx_PSF_centroid = np.unravel_index(np.nanargmax(PSF,axis=None),PSF.shape)    
    i_PSF_centroid = idx_PSF_centroid[0] ; j_PSF_centroid = idx_PSF_centroid[1]
    PSF_shift = np.copy(PSF)*0
    CCF_conv = np.copy(CCF)*np.nan
    for i_shift in range(CCF_conv.shape[0]):
        for j_shift in range(CCF_conv.shape[1]): 
            if not np.isnan(CCF[i_shift,j_shift]):
                for i in range(PSF_shift.shape[0]):
                    for j in range(PSF_shift.shape[1]):
                        if i+i_PSF_centroid-i_shift>=0 and j+j_PSF_centroid-j_shift>=0 and i+i_PSF_centroid-i_shift<PSF_shift.shape[0] and j+j_PSF_centroid-j_shift<PSF_shift.shape[1] :
                            PSF_shift[i,j] = PSF[i+i_PSF_centroid-i_shift,j+j_PSF_centroid-j_shift]
                        else :
                            PSF_shift[i,j] = np.nan
                #plt.figure() ; plt.imshow(PSF_shift,extent=[-(cube.shape[2]+1)//2*pxscale,(cube.shape[2]+1)//2*pxscale,-(cube.shape[2]+1)//2*pxscale,(cube.shape[2]+1)//2*pxscale]) ; plt.title(f'MIRIMRS PSF of {target_name} \n on {band}', fontsize=14) ; plt.ylabel('y offset (in ")', fontsize=14) ; plt.xlabel('x offset (in ")', fontsize=14) ; plt.show()
                #plt.figure() ; plt.imshow(CCF,extent=[-(cube.shape[2]+1)//2*pxscale,(cube.shape[2]+1)//2*pxscale,-(cube.shape[2]+1)//2*pxscale,(cube.shape[2]+1)//2*pxscale]) ; plt.title(f'MIRIMRS PSF of {target_name} \n on {band}', fontsize=14) ; plt.ylabel('y offset (in ")', fontsize=14) ; plt.xlabel('x offset (in ")', fontsize=14) ; plt.show()
                CCF_conv[i_shift,j_shift] = np.nansum(PSF_shift*CCF)
    return CCF_conv



def SNR_calculation(CCF,y0,x0,size_core,print_value=True):
        planet_sep = int(np.sqrt((y0-CCF.shape[0]//2)**2+(x0-CCF.shape[1]//2)**2))
        CCF = CCF*annular_mask(planet_sep-6,planet_sep+7,value=np.nan,size=CCF.shape)
        CCF_signal = CCF[y0,x0]
        CCF_noise = np.copy(CCF)*annular_mask(planet_sep-1,planet_sep+2,value=np.nan,size=CCF.shape)
        CCF_noise[y0-size_core:y0+size_core+1,x0-size_core:x0+size_core+1] = np.nan # on masque la planète
        noise = np.sqrt(np.nanvar(CCF_noise))
        if print_value :
            print(" E[<n,t>]/Std[<n,t>] = ",round(100*np.nanmean(CCF_noise)/np.nanstd(CCF_noise),2),"%")
        signal = CCF_signal-np.nanmean(CCF_noise)
        SNR = signal/noise
        return SNR,CCF,CCF_signal,CCF_noise



def plot_CCF(instru,sd_HF,bs_HF,wave,trans,R,Rc,used_filter,target_name,band,model="BT-Settl",broad=0,large_rv=True): 
    T0 = np.array([500,1500,2500]) ; lg = 3.5
    plt.figure() ; plt.xlabel("Doppler velocity (in km/s)",fontsize=14) ; plt.ylabel(r"cos $\theta_{est}$",fontsize=14) ; plt.grid(True)
    for i in range(len(T0)):
        cos , rv = correlation_rv(instru,sd_HF,wave,trans,T=T0[i],lg=lg,R=R,Rc=Rc,used_filter=used_filter,model=model,show=False,large_rv=large_rv,broad=broad)
        plt.plot(rv,cos,label=f"T = {T0[i]} K")
        if i==int(len(T0)-1):
            cos , rv = correlation_rv(instru,bs_HF,wave,trans,T=T0[i],lg=lg,R=R,Rc=Rc,used_filter=used_filter,model=model,show=False,large_rv=large_rv,broad=broad)
            plt.plot(rv,cos,"k",label=f"off-planet")
    plt.legend(fontsize=12)
    plt.title(f'Correlation between {model} spectra and {target_name} data spectrum \n on {band} of {instru} with $R_c$ = {Rc}',fontsize=16)



def correlation_T_rv(instru,sd_HF,wave,trans,R,Rc,used_filter,target_name,band,model="BT-Settl",broad=0):
    if model=="Exo-REM":
        lg = np.array([3.5,4.0])
        T = np.arange(500,2100,100) # K
    elif model=="BT-Settl" or model=="PICASO":
        lg = np.array([3.0,3.5,4.0,4.5,5.0])
        T = np.append(np.arange(400,1050,50),np.arange(1000,3100,100))
    elif model=="Morley":
        lg = np.array([4,4.5,5,5.5])
        T = np.arange(500,1400,100) # K
    elif model=="SONORA":
        lg = np.array([3,3.5,4,4.5,5,5.5])
        T = np.arange(500,2500,100)
    elif model=="BT-Dusty":
        lg = np.array([4.5,5])
        T = np.arange(1400,3100,100)
    cos_2d = np.zeros((len(lg),len(T)))
    rv_2d = np.zeros((len(lg),len(T)))
    k=0
    for i in range(len(lg)):
        for j in range(len(T)):
            print(round(100*(k+1)/(len(T)*len(lg)),2),"%")
            cos_rv,rv = correlation_rv(instru,sd_HF,wave,trans,T=T[j],lg=lg[i],R=R,Rc=Rc,used_filter=used_filter,model=model,show=False,large_rv=False,broad=broad)
            cos_2d[i,j] = np.nanmax(cos_rv)
            rv_2d[i,j] = rv[cos_rv.argmax()]
            k+=1
    idx_max_corr = np.unravel_index(np.argmax(cos_2d, axis=None), cos_2d.shape)
    print(f"maximum correlation value of {round(np.nanmax(cos_2d),2)} for T = {T[idx_max_corr[1]]} K, lg = {lg[idx_max_corr[0]]} rv = {rv_2d[idx_max_corr[0],idx_max_corr[1]]} km/s")
    plt.figure()
    plt.pcolormesh(T,lg,cos_2d,cmap=plt.get_cmap('rainbow'),vmin=np.nanmin(cos_2d),vmax=np.nanmax(cos_2d))
    plt.xlabel("planet's temperature (in K)",fontsize=12) ; plt.ylabel("planet's gravity surface" , fontsize=12) ; plt.title(f'Correlation between {model} spectra and {target_name} \n data spectrum on {band} of {instru} with $R_c$ = {Rc}',fontsize=14)
    cbar = plt.colorbar() ; cbar.set_label(r"cos $\theta_{est}$", fontsize=14, labelpad=20, rotation=270)
    plt.plot([T[idx_max_corr[1]],T[idx_max_corr[1]]],[lg[idx_max_corr[0]],lg[idx_max_corr[0]]],'kX',ms=10,label=r"cos $\theta_{max}$ = "+f"{round(np.nanmax(cos_2d),2)} for T = {T[idx_max_corr[1]]} K, \n lg = {lg[idx_max_corr[0]]} and rv = {round(rv_2d[idx_max_corr[0],idx_max_corr[1]],2)} km/s")
    plt.contour(T,lg,cos_2d, linewidths=0.1,colors='k') ; plt.ylim(lg[0],lg[-1])
    plt.legend(fontsize=12) ; plt.show()



def SNR_T_rv(instru,cube_high_filtered,wave,trans,R,Rc,used_filter,target_name,band,model,broad,y0,x0,size_core):
    if target_name=="Chamaeleon" : # on masque les bords
        cube_high_filtered[:,:cube_high_filtered.shape[1]//2,:]=np.nan ; cube_high_filtered[:,:,:cube_high_filtered.shape[1]//2]=np.nan # on masque les bords
    elif target_name == "HR8799": # on masque les autres planètes et les bords
        cube_high_filtered[:,34:,:]=np.nan ;  cube_high_filtered[:,27-4:27+5,22-4:22+5] = np.nan ;  cube_high_filtered[:,17-4:17+5,24-4:24+5] = np.nan
    planet_sep = int(np.sqrt((y0-cube_high_filtered.shape[1]//2)**2+(x0-cube_high_filtered.shape[2]//2)**2))
    S_res = np.copy(cube_high_filtered)
    for l in range(S_res.shape[0]):
        S_res[l] = S_res[l]*annular_mask(planet_sep-6,planet_sep+6,value=np.nan,size=S_res[l].shape)
    if model=="Exo-REM":
        lg = np.array([3.5,4.0])
        T = np.arange(500,2100,100) # K
    elif model=="BT-Settl" or model=="PICASO":
        lg = np.append(np.arange(500,1050,50),np.arange(1000,3100,100))
        T = np.arange(500,3100,100)
    elif model=="Morley":
        lg = np.array([4,4.5,5,5.5])
        T = np.arange(500,1400,100) # K
    elif model=="SONORA":
        lg = np.array([3,3.5,4,4.5,5,5.5])
        T = np.arange(500,2500,100)
    SNR_2d = np.zeros((len(lg),len(T)))
    rv_2d =  np.zeros_like(SNR_2d)
    noise_2d = np.zeros_like(SNR_2d)
    signal_2d = np.zeros_like(SNR_2d)
    k=0
    for i in range(len(lg)):
        for j in range(len(T)):
            print(round(100*(k+1)/(len(T)*len(lg)),2),"%")
            CCF_rv,v_doppler,t = molecular_mapping_rv(instru,band,S_res,T[j],lg[i],model,wave,trans,"SNR_rv",R=R,Rc=Rc,used_filter=used_filter,broad=broad) 
            SNR_rv = np.zeros_like(v_doppler)
            noise_rv = np.zeros_like(v_doppler)
            signal_rv = np.zeros_like(v_doppler)
            for m in range(len(v_doppler)):
                ccf_rv = CCF_rv[m]
                snr,_,CCF_signal,CCF_noise = SNR_calculation(ccf_rv,y0,x0,size_core,print_value=False)
                SNR_rv[m] = snr
                noise_rv[m] = np.sqrt(np.nanvar(CCF_noise))
                signal_rv[m] = np.nansum(CCF_signal) - np.nanmean(CCF_noise)
            SNR_2d[i,j] = np.nanmax(SNR_rv)
            rv_2d[i,j] = v_doppler[SNR_rv.argmax()]
            noise_2d[i,j] = noise_rv[SNR_rv.argmax()]
            signal_2d[i,j] = signal_rv[SNR_rv.argmax()] # np.nanmax(signal_rv)
            k+=1
    idx_max_corr = np.unravel_index(np.argmax(SNR_2d, axis=None),SNR_2d.shape)
    print(f"maximum SNR value of {round(np.nanmax(SNR_2d),2)} for T = {T[idx_max_corr[1]]} K, lg = {lg[idx_max_corr[0]]} rv = {rv_2d[idx_max_corr[0],idx_max_corr[1]]} km/s")
    plt.figure() ; plt.pcolormesh(T,lg,SNR_2d,cmap=plt.get_cmap('rainbow'))
    plt.xlabel("planet's temperature (in K)",fontsize=12) ; plt.ylabel("planet's gravity surface" , fontsize=12) ; plt.title(f'SNR value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}',fontsize=14)
    cbar = plt.colorbar() ; cbar.set_label("SNR", fontsize=14, labelpad=20, rotation=270)
    plt.plot([T[idx_max_corr[1]],T[idx_max_corr[1]]],[lg[idx_max_corr[0]],lg[idx_max_corr[0]]],'kX',ms=10,label=r"$SNR_{max}$ = "+f"{round(np.nanmax(SNR_2d),1)} for T = {T[idx_max_corr[1]]} K, \n lg = {lg[idx_max_corr[0]]} and rv = {round(rv_2d[idx_max_corr[0],idx_max_corr[1]],2)} km/s")
    plt.legend() ; plt.show()
    plt.figure() ; plt.pcolormesh(T,lg,noise_2d,cmap=plt.get_cmap('rainbow'))
    plt.xlabel("planet's temperature (in K)",fontsize=12) ; plt.ylabel("planet's gravity surface" , fontsize=12) ; plt.title(f'Noise value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}',fontsize=14)
    cbar = plt.colorbar() ; cbar.set_label("noise (in e-)", fontsize=14, labelpad=20, rotation=270) ; plt.show()
    plt.figure() ; plt.pcolormesh(T,lg,signal_2d,cmap=plt.get_cmap('rainbow'),vmin=np.nanmin(signal_2d),vmax=np.nanmax(signal_2d))
    plt.xlabel("planet's temperature (in K)",fontsize=12) ; plt.ylabel("planet's gravity surface" , fontsize=12) ; plt.title(f'Signal value for different {model} spectra for {target_name} \n on {band} of {instru} with $R_c$ = {Rc}',fontsize=14)
    cbar = plt.colorbar() ; cbar.set_label("signal (in e-)", fontsize=14, labelpad=20, rotation=270) ; plt.show()



def chi_2_rv(instru,sd_HF,wave,trans,T,lg,sigma,sigma_l,model="BT-Settl",show=True,large_rv=False,broad=0):
    config_data = get_config_data(instru)
    if model=="Exo-REM":
        planet_spectrum = load_planet_spectrum(T,lg,model=model,version="old")
    else:
        planet_spectrum = load_planet_spectrum(T,lg,model=model)
    planet_spectrum = planet_spectrum.degrade_resolution(wave,renorm=True)
    if broad > 0:
        planet_spectrum = planet_spectrum.broad_r(broad)
    planet_spectrum = planet_spectrum.set_nbphotons_min(config_data,planet_spectrum.wavelength)
    rv = np.linspace(-50,50,101)
    if large_rv :
        rv = np.linspace(-2000,2000,200)
    chi2 = np.zeros_like(rv)
    for i in range(len(rv)) :
        planet_shift = planet_spectrum.doppler_shift(rv[i]) # en ph/mn
        valid = ~np.isnan(planet_shift.flux)
        planet_shift.flux = planet_shift.flux[valid] ; planet_shift.wavelength = planet_shift.wavelength[valid]
        template = planet_shift.flux - gaussian_filter(planet_shift.flux,sigma=sigma) # en ph
        f = interp1d(planet_shift.wavelength, template, bounds_error=False, fill_value=np.nan) # créer une interpolation avec l'axe lambda décalé / le spectre décalé
        template = f(wave)*trans # interpole les valeurs du flux sur le nouvel axe (wavelength_output)
        template[np.isnan(sd_HF)] = np.nan
        template = template / np.sqrt(np.nansum(template**2)) # On suppose qu'on a le template "parfait"
        R = np.nansum(sd_HF*template/sigma_l**2) / np.nansum(template**2/sigma_l**2)
        chi2[i] = np.nansum(((sd_HF-R*template)/sigma_l)**2)
    if show :     
        plt.figure() ; plt.plot(rv,cos_theta,'k') ; plt.xlabel("Doppler velocity (in km/s)",fontsize=14) ; plt.ylabel(r"cos($\theta$)",fontsize=14) ; plt.grid(True)
        plt.figure() ; plt.plot(wave,planet_spectrum_model.high_pass_flux,'r') ; plt.plot(wave,sd_HF,'b')
    return chi2 , rv



def chi2_T_rv(instru,sd_HF,wave,trans,sigma_l,sigma,target_name,band,Rc,model="BT-Settl",broad=0):
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
    chi2_2d = np.zeros((len(lg),len(T)))
    rv_2d = np.zeros((len(lg),len(T)))
    k=0
    for i in range(len(lg)):
        for j in range(len(T)):
            print(round(100*(k+1)/(len(T)*len(lg)),2),"%")
            chi2_rv,rv = chi_2_rv(instru,sd_HF,wave,trans,T=T[j],lg=lg[i],sigma=sigma,sigma_l=sigma_l,model=model,show=False,large_rv=False,broad=broad)
            chi2_2d[i,j] = np.nanmin(chi2_rv)
            rv_2d[i,j] = rv[chi2_rv.argmin()]
            k+=1
    idx_min_chi2 = np.unravel_index(np.argmin(chi2_2d, axis=None), chi2_2d.shape) ; chi2_min = np.nanmin(chi2_2d)
    chi2_2d = chi2_2d-chi2_min
    print(f"minimum chi2 value is given for T = {T[idx_min_chi2[1]]} K, lg = {lg[idx_min_chi2[0]]} rv = {rv_2d[idx_min_chi2[0],idx_min_chi2[1]]} km/s")
    plt.figure() ; plt.pcolormesh(T,lg,chi2_2d,cmap=plt.get_cmap('rainbow_r'))
    plt.xlabel("planet's temperature (in K)",fontsize=12) ; plt.ylabel("planet's gravity surface" , fontsize=12) ; plt.title(r'$\chi^2$'+f' between {model} spectra and {target_name} \n data spectrum on {band} of {instru} with $R_c$ = {Rc}',fontsize=14)
    cbar = plt.colorbar() ; cbar.set_label(r"$\chi^2$ - $\chi^2_{min}$", fontsize=14, labelpad=20, rotation=270)
    plt.plot([T[idx_min_chi2[1]],T[idx_min_chi2[1]]],[lg[idx_min_chi2[0]],lg[idx_min_chi2[0]]],'kX',ms=10,label=r"$\chi^2_{min}$ "+f"for T = {T[idx_min_chi2[1]]} K, \n lg = {lg[idx_min_chi2[0]]} and rv = {round(rv_2d[idx_min_chi2[0],idx_min_chi2[1]],2)} km/s")
    plt.legend() ; plt.show()




def gaussian(x,mu,sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def gaussian0(x,sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x)/sig, 2.)/2)

def lorentzian(x,x0,L):
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)

def chi2(x,x0,L):
    return L/(2*np.pi) * 1 / (L**2/4 + (x-x0)**2)
