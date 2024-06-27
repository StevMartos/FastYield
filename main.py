from src.FastCurves import *
from src.colormaps import *
from src.FastCurves_interface import *
from src.FastYield_interface import *
from src.DataAnalysis_interface import *
#get_picasso_thermal()
#get_picasso_albedo()



if 1==0 : # ouvrir table de planète synthétique
    planet_number,system_id,a,M,R,ecc,inc,stellar,Mcore,Menve,astart,fice,Lint,Ttau = np.loadtxt("syntheticpopJ39_978_ext.dat",unpack=True,skiprows=1)
    plt.figure()
    plt.scatter(M,R,c='k',alpha=0.1) ; plt.xscale('log') ; plt.yscale('log')



if 1==0: # ouvrir spectre Morley (R~10 000 => pas assez...)
    spec = load_planet_spectrum(400,1,"BT-Settl")
    albedo = 0.0 # 0.0 / 0.3 / 0.7
    path = "Morley2017_models/emission_spectra/alb"+str(albedo)+"/"
    list_files = os.listdir(path)
    planet = "gj1132b"
    planet = "lhs1140b"
    #planet = "trappist1c"
    chem_type = "earth"
    psurf = "psurf1.0"
    for name in list_files :
        if planet in name :
            if chem_type in name :
                if psurf in name :
                    print(name)
                    wave , flux = np.loadtxt(path+name,unpack=True, skiprows=3) # in µm and J/s/m2/m
                    flux /= 1e6 # in J/s/m2/µm
                    argsort = np.argsort(wave)
                    wave = wave[argsort]
                    flux = flux[argsort]
                    plt.figure()
                    plt.plot(spec.wavelength,spec.flux)
                    plt.plot(wave,flux)
                    plt.title(name)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.show()
                    dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] # array de delta Lambda
                    R = wave/(2*dwl) # calcule de la nouvelle résolution
                    plt.figure()
                    plt.plot(wave,R)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.show()
    


if 1==0 :  # ouvrir spectre moléculaire 
    #sigma_cosmic = 1
    config_data = get_config_data("MIRIMRS")
    band = '1SHORT'
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    Rc = 100 ; used_filter = "gaussian"
    mol = load_planet_spectrum(200,None,"mol_O2")
    #mol_HF,mol_LF = filtered_flux(mol.flux,R=mol.R,Rc=Rc,used_filter=used_filter)
    #sg = sigma_clip(mol_LF,sigma=sigma_cosmic)
    #mol.flux[~sg.mask] = np.nan
    plt.figure()
    plt.plot(mol.wavelength,mol.flux)
    #plt.plot(mol.wavelength,mol_HF)
    #plt.plot(mol.wavelength,mol_LF)
    plt.show()
    
    mol = spectrum_inter(config_data,band,mol)
    planet = load_planet_spectrum(1600,4,"BT-Settl")
    planet = spectrum_inter(config_data,band,planet)
    mol_HF,_ = filtered_flux(mol.flux,R,Rc,used_filter)
    planet_HF,_ = filtered_flux(planet.flux,R,Rc,used_filter)
    plt.figure()
    plt.plot(mol.wavelength,mol_HF/np.sqrt(np.nansum(mol_HF**2))) ; plt.plot(planet.wavelength,planet_HF/np.sqrt(np.nansum(planet_HF**2)))
    plt.show()
    cos = np.nansum(planet_HF*mol_HF)/(np.sqrt(np.nansum(mol_HF**2))*np.sqrt(np.nansum(planet_HF**2)))



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Graphic Interface :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FastCurves_interface() # need an update
FastYield_interface()
#DataAnalysis_interface()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# update FastYield :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#create_fastcurves_table(table="Archive")
#create_fastcurves_table(table="Simulated")

#all_SNR_table(table="Archive")
#all_SNR_table(table="Simulated")

#archive_yield(exposure_time=120,contrast=True,save=False)
#archive_yield_plot(fraction=False)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves (theoritical cases) :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pour comparer avec FastCurves v0
#FastCurves(instru="HARMONI",model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='instru',T_star=9600,lg_star=4.0,exposure_time=120,apodizer="SP1",strehl="MED",plot_mag=True)

# plots HARMONI Exo5
#FastCurves(instru="HARMONI",model="BT-Settl",calculation="contrast",T_planet=700,lg_planet=4.0,mag_star=6,band0='K',T_star=6000,lg_star=4.0,exposure_time=120,apodizer="SP1",strehl="MED",plot_mag=True)


#FastCurves(instru="HARMONI",model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='K',T_star=6000,lg_star=4.0,exposure_time=120,apodizer="NO_SP",strehl="JQ1")

#FastCurves(instru="ERIS",model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='K',T_star=6000,lg_star=4.0,exposure_time=120,apodizer="NO_SP",strehl="JQ0")

#FastCurves(instru="ANDES",model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='K',T_star=6000,lg_star=4.0,exposure_time=120,apodizer="NO_SP",strehl="JQ1")



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves (real data cases) :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Chameleon  (DIT = 138.75 s) / SNR(1SHORT) = 15.22
#FastCurves(instru="MIRIMRS",systematic=True,input_DIT=138.75/60,star_pos="edge",bkgd="high",model="BT-Settl",calculation="SNR",separation_planet=2.5,T_planet=2600,lg_planet=3.5,name_planet="CT Cha b",mag_star=8.66,mag_planet=14.9,band0='K',T_star=4400,lg_star=3.5,exposure_time=56.426,star_rv=-2.9,channel=False)

# HD 19467 (G1V / DIT = 218.8 s )
#FastCurves(instru="NIRSpec",systematic=True,separation_planet=1.5,band_only="G395H_F290LP",input_DIT=218.8/60,calculation="contrast",model="BT-Settl",T_planet=950,lg_planet=5.0,name_planet='HD 19467 b',mag_star=5.4,band0='K',T_star=5680,lg_star=4.0,exposure_time=0.665)
#FastCurves(instru="MIRIMRS",systematic=True,separation_planet=1.5,band_only="1SHORT",systematic=True,input_DIT=218.8/60,calculation="contrast",model="BT-Settl",T_planet=950,lg_planet=5.0,name_planet='HD 19467 b',mag_star=5.4,band0='K',T_star=5680,lg_star=4.0,exposure_time=0.665)

# HIP 65426 b (DIT = 308 s)
#FastCurves(instru="NIRCam",input_DIT=308/60,calculation="SNR",T_planet=1600,lg_planet=4.0,separation_planet=0.8,name_planet="HIP 65426 b",mag_planet=16.756,mag_star=6.771,band0='K',T_star=8000,lg_star=4.0,exposure_time = 20.3)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COLORMAPS :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#colormap_bandwidth_resolution(T_planet=1000,T_star=6000,delta_rv=30,spectrum_contributions="thermal",model="BT-Settl",instru="HiRISE",Rc=100,used_filter="gaussian")
#colormap_bandwidth_Tp(instru="HARMONI",T_star=5000,delta_rv=25,spectrum_contributions="reflected",model="PICASO",Rc=100,used_filter="gaussian")
#colormap_rv(T_planet=300,T_star=9000,spectrum_contributions="reflected",model="flat",instru="HARMONI",band="H",Rc=100)
#colormap_vsini(T_planet=1000,T_star=3000,spectrum_contributions="reflected",model="flat",instru="HARMONI",band="H",Rc=100)
#colormap_maxsep_phase(instru="HARMONI",band="H",inc=90)

#colormap_best_parameters_earth()




if 1==0: # PLOT QUASI-TOUTES LES COLORMAPS
    Tp = np.arange(500,3100,100)
    for T in Tp : # stop at 900K
        colormap_bandwidth_resolution(T_planet=T,T_star=5000,delta_rv=25,spectrum_contributions="reflected",model="PICASO",instru="all",Rc=100,used_filter="gaussian")
        colormap_bandwidth_resolution(T_planet=T,T_star=5000,delta_rv=25,spectrum_contributions="thermal",model="BT-Settl",instru="all",Rc=100,used_filter="gaussian")
        colormap_rv(T_planet=T,T_star=5000,spectrum_contributions="reflected",model="PICASO",instru="ANDES",band="YJH",Rc=100)
        colormap_rv(T_planet=T,T_star=5000,spectrum_contributions="reflected",model="PICASO",instru="HARMONI",band="H",Rc=100)
        colormap_rv(T_planet=T,T_star=5000,spectrum_contributions="thermal",model="BT-Settl",instru="ANDES",band="YJH",Rc=100)
        colormap_rv(T_planet=T,T_star=5000,spectrum_contributions="thermal",model="BT-Settl",instru="HARMONI",band="H",Rc=100)
        colormap_vsini(T_planet=T,T_star=5000,spectrum_contributions="reflected",model="PICASO",instru="ANDES",band="YJH",Rc=100)
        colormap_vsini(T_planet=T,T_star=5000,spectrum_contributions="reflected",model="PICASO",instru="HARMONI",band="H",Rc=100)
        colormap_vsini(T_planet=T,T_star=5000,spectrum_contributions="thermal",model="BT-Settl",instru="ANDES",band="YJH",Rc=100)
        colormap_vsini(T_planet=T,T_star=5000,spectrum_contributions="thermal",model="BT-Settl",instru="HARMONI",band="H",Rc=100)
    T_star = np.arange(3000,41000,1000)
    for T in T_star :
        colormap_rv(T_planet=300,T_star=T,spectrum_contributions="reflected",model="tellurics",instru="ANDES",band="YJH",Rc=100)
        colormap_rv(T_planet=300,T_star=T,spectrum_contributions="reflected",model="tellurics",instru="HARMONI",band="H",Rc=100)
        colormap_vsini(T_planet=300,T_star=T,spectrum_contributions="reflected",model="tellurics",instru="ANDES",band="YJH",Rc=100)
        colormap_vsini(T_planet=300,T_star=T,spectrum_contributions="reflected",model="tellurics",instru="HARMONI",band="H",Rc=100)
    colormap_bandwidth_Tp(instru="HARMONI",T_star=5000,delta_rv=25,spectrum_contributions="reflected",model="PICASO",Rc=100,used_filter="gaussian")
    colormap_bandwidth_Tp(instru="HARMONI",T_star=5000,delta_rv=25,spectrum_contributions="thermal",model="BT-Settl",Rc=100,used_filter="gaussian")
    colormap_bandwidth_Tp(instru="ANDES",T_star=5000,delta_rv=25,spectrum_contributions="reflected",model="PICASO",Rc=100,used_filter="gaussian")
    colormap_bandwidth_Tp(instru="ANDES",T_star=5000,delta_rv=25,spectrum_contributions="thermal",model="BT-Settl",Rc=100,used_filter="gaussian")
    inc = np.arange(0,181,1)
    for i in range(len(inc)) :
        colormap_maxsep_phase(instru="ANDES",band="YJH",inc=inc[i])
        colormap_maxsep_phase(instru="HARMONI",band="H",inc=inc[i])



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PSD : 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if 1==0:
    band="1SHORT" ; config_data=get_config_data('MIRIMRS') ; band0="instru" ; mag_planet=15
    planet_spectrum = load_planet_spectrum(500,3.5,"BT-Settl") ; R_planet = planet_spectrum.R
    planet_spectrum , density = spectrum_instru(band0,R_planet,config_data,mag_planet,planet_spectrum)
    planet_spectrum = spectrum_inter(config_data,band,planet_spectrum)
    
    l0 = (config_data['lambda_range']['lambda_min']+config_data['lambda_range']['lambda_max'])/2
    R = config_data['gratings'][band].R
    Rc=100
    sigma = 2*R/(np.pi*Rc)*np.sqrt(np.log(2)/2)
    
    wave=planet_spectrum.wavelength ; flux=planet_spectrum.flux ; flux_LF=gaussian_filter(flux,sigma) ; flux_HF=flux-flux_LF
    plt.figure() ; plt.plot(wave,flux,'r') ; plt.xlabel('wavelength (in µm)',fontsize=14) ; plt.xscale('log')
    plt.plot(wave,flux_LF,'g')  ; plt.ylabel("flux (in photons/s)",fontsize=14) ; plt.title(f'Spectrum of a planet at 500K on {band}',fontsize=14)
    plt.plot(wave,flux_HF,'b') ; plt.legend(["$S_p$","[$S_p$]$_{BF}$","[$S_p$]$_{HF}$"],fontsize=14)
    spectrum=Spectrum(wave,flux,planet_spectrum.R,None)
    spectrum_LF=Spectrum(wave,flux_LF,planet_spectrum.R,None)
    spectrum_HF=Spectrum(wave,flux_HF,planet_spectrum.R,None)
    plt.figure()
    res,psd=spectrum.plot_psd(smooth=0, color='r',show=True,ret=True,area=False)
    #res,psd_LF=spectrum_LF.plot_psd(smooth=0, color='g',show=True,ret=True)
    res,psd_HF=spectrum_HF.plot_psd(smooth=0, color='b',show=True,ret=True)
    plt.ylim(1,1e7);plt.yscale('log')
    plt.plot([R,R],[1,1e7],'k-')
    plt.plot([Rc,Rc],[1,1e7],'k:')

    plt.legend(["PSD{$S_p$}","PSD{[$S_p$]$_{HF}$}",r"$\alpha^2$","$R_{inst}$","$R_c$"],fontsize=11,loc="upper right")




if 1==0 : # permet de trouver la planète ayant les paramètres les plus proches de T et lg
    planet_table = load_archive_table("Archive_Pull_for_FastCurves.ecsv")
    T = 1500
    lg = 4.0
    diff_T = (planet_table["PlanetTeq"].value-T)/T
    diff_lg = (planet_table["PlanetLogg"].value-lg)/lg
    distance = np.sqrt(diff_T**2+diff_lg**2)
    idx = np.argmin(distance)
    print(planet_table[idx]["PlanetTeq"])
    print(planet_table[idx]["PlanetLogg"])

if 1==0 : # Missing values for K-band mag of Direct Imaging planets
    planet_table = load_planet_table("Archive_Pull_raw.ecsv")
    planet_table = planet_table[planet_table["DiscoveryMethod"]=="Imaging"]
    planet_table['PlanetKmag(thermal+reflected)'] = np.full((len(planet_table),), np.nan)
    planet_table = input_mag_imaging_planets(planet_table) # on rentre les valeurs connues des magnitudes (en bande K) des planètes détectées par imagerie directe
    print(f" {len(planet_table[np.isnan(planet_table['PlanetKmag(thermal+reflected)'])])}/{len(planet_table)} K-band value are missing for Imaging planets :")
    for i in range(len(planet_table)):
        if np.isnan(planet_table[i]['PlanetKmag(thermal+reflected)']) : 
            print(f"\n {planet_table[i]['PlanetName']} : {planet_table[i]['DiscoveryRef']}")
    




















if 1==0:
    band="1SHORT" ; config_data=get_config_data('MIRIMRS')
    
    
    
    
    lmin = config_data['gratings'][band].lmin ; lmax = config_data['gratings'][band].lmax # lambda_min/lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    if R is None : # dans le cas où il ne s'agit pas d'un spectro-imageur (eg NIRCAM)
        R = spectrum_instru.R
    delta_lambda = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave = np.arange(lmin,lmax,delta_lambda) # axe de longueur d'onde de la bande considérée
    
    Rc = 3000 ; used_filter = "smoothstep"
    
    N = 10000 ; mean_psd = 0. ; mean_psd_HF = 0. ; mean_psd_LF = 0.
    
    mean_fn_HF = 0. ; mean_fn_LF = 0.
    
    for i in range(N):
        n = np.random.normal(0,1,len(wave))
        n_HF,n_LF = filtered_flux(n,R=R,Rc=Rc,used_filter=used_filter)
        mean_fn_HF += np.nansum(n_HF**2) / np.nansum(n**2) / N
        mean_fn_LF += np.nansum(n_LF**2) / np.nansum(n**2) / N
        if i == 0 :
            plt.figure() ; plt.xlabel('wavelength (in µm)',fontsize=14) ; plt.ylabel("flux",fontsize=14)
            plt.plot(wave,n,'r',label="n")
            plt.plot(wave,n_HF,'b',label="$[n]_{HF}$")
            plt.plot(wave,n_LF,'g',label="$[n]_{LF}$")
            plt.legend()
        spectrum=Spectrum(wave,n,R,None)
        spectrum_LF=Spectrum(wave,n_LF,R,None)
        spectrum_HF=Spectrum(wave,n_HF,R,None)
        res,psd = spectrum.plot_psd(smooth=0, color='b',show=False,ret=True)
        res,psd_HF = spectrum_HF.plot_psd(smooth=0, color='b',show=False,ret=True)
        res,psd_LF = spectrum_LF.plot_psd(smooth=0, color='b',show=False,ret=True)
        mean_psd += psd / N
        mean_psd_HF += psd_HF / N
        mean_psd_LF += psd_LF / N
        print(round(100*(i+1)/N),"%")
    plt.figure()
    plt.plot(res,mean_psd,'r',label="n")
    plt.plot(res,mean_psd_HF,'b',label="$[n]_{HF}$")
    plt.plot(res,mean_psd_LF,'g',label="$[n]_{LF}$")
    plt.xlabel("resolution frequency R")
    plt.ylabel("PSD")
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([R,R],[np.nanmin(mean_psd_HF)/10,10*np.nanmax(mean_psd)],'k-',label="$R_c$")
    plt.plot([Rc,Rc],[np.nanmin(mean_psd_HF)/10,10*np.nanmax(mean_psd)],'k:',label="$R_{instru}$")
    plt.ylim(np.nanmin(mean_psd_HF)/10,10*np.nanmax(mean_psd))
    plt.legend()
    print("||[n]_HF||^2/||n||^2 = ",round(100*(np.nansum(mean_psd_HF)/np.nansum(mean_psd)),1), " & ",round(100*mean_fn_HF,1))
    print("||[n]_LF||^2/||n||^2 = ",round(100*(np.nansum(mean_psd_LF)/np.nansum(mean_psd)),1), " & ",round(100*mean_fn_LF,1))

    fn_HF,fn_LF = get_fraction_noise_filtered(wave,R,Rc,used_filter)
    print("fn_LF = ", round(100*fn_LF,1))
    print("fn_HF = ", round(100*fn_HF,1))

    
    
    






