from src.FastCurves import *
from src.colormaps import *
from src.FastCurves_interface import *
from src.FastYield_interface import *
#get_picasso_thermal()
#get_picasso_albedo()


if 1==0 :
    planet_number,system_id,a,M,R,ecc,inc,stellar,Mcore,Menve,astart,fice,Lint,Ttau = np.loadtxt("syntheticpopJ39_978_ext.dat",unpack=True,skiprows=1)
    plt.figure()
    plt.scatter(M,R,c='k',alpha=0.1) ; plt.xscale('log') ; plt.yscale('log')











#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Graphic Interface :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FastCurves_interface()
FastYield_interface()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# update FastYield :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#create_fastcurves_table(table="Archive")
#create_fastcurves_table(table="Simulated")

#all_SNR_table(table="Archive")
#all_SNR_table(table="Simulated")

#archive_yield(exposure_time=120,contrast=True,save=False)

#archive_yield_plot()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#n,sep,c=harmoni(model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='instru',T_star=8000,lg_star=4.0,exposure_time = 120,apodizer="SP1",strehl="MED",plot_mag=False)


#n,sep,c=eris(model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='instru',T_star=8000,lg_star=4.0,exposure_time = 120,plot_mag=False)


#n,sep,c=andes(model="BT-Settl",calculation="contrast",T_planet=1000,lg_planet=4.0,mag_star=6,band0='instru',T_star=8000,lg_star=4.0,exposure_time = 120,plot_mag=False)


#n,sep,c=nircam(calculation="contrast",T_planet=1600,lg_planet=4.0,separation_planet=0.8,name_planet="HIP 65426 b",mag_planet=16.756,mag_star=6.771,band0='K',T_star=8000,lg_star=4.0,exposure_time = 20.3)
#n,sep,c=nircam(calculation="SNR",T_planet=1600,lg_planet=4.0,separation_planet=0.8,name_planet="HIP 65426 b",mag_planet=16.756,mag_star=6.771,band0='K',T_star=8000,lg_star=4.0,exposure_time = 20.3)


#n,sep,c=nirspec(delta_radial_velocity=-25,model="BT-Settl",calculation="contrast",T_planet=1100,lg_planet=4.0,name_planet="HR 8799 b",mag_star=5.241,band0='instru',T_star=8000,lg_star=4.0,exposure_time=171)
#n,sep,SNR=nirspec(delta_radial_velocity=-25,model="Exo-REM",calculation="SNR",T_planet=1100,lg_planet=4.0,separation_planet=1.72,name_planet="HR 8799 b",mag_planet=15.25,mag_star=5.241,band0='instru',T_star=8000,lg_star=4.0,exposure_time=171)


# β Pictoris b (A6V / DIT = 13.875 s / mag*(Ks)=3.48 / mag_p(instru)= 12):
    
#n,sep,c=mirimrs(delta_radial_velocity=-25,model="BT-Settl",calculation="contrast",T_planet=1700,lg_planet=4.0,name_planet="β Pictoris b",mag_star=3.081,band0='instru',T_star=8000,lg_star=4.0,exposure_time=92.5)
#n,sep,SNR=mirimrs(channel=True,delta_radial_velocity=-25,model="Exo-REM",calculation="SNR",T_planet=1700,lg_planet=4.0,separation_planet=0.55,name_planet="β Pictoris b",mag_planet = 11,mag_star=3.481,band0='instru',T_star=8000,lg_star=4.0,exposure_time=92.5)

#n,sep,c=harmoni(calculation="contrast",T_planet=1700,lg_planet=4.0,mag_star=3.1,band0='instru',T_star=8000,lg_star=4.0,exposure_time = 92.5)
#n,sep,SNR=harmoni(calculation="SNR",T_planet=1700,lg_planet=4.0,separation_planet=0.55,name_planet="β Pictoris b",mag_planet=11.94,mag_star=3.1,band0='instru',T_star=8000,lg_star=4.0,exposure_time=92.5)



# HR8799b (A5V /  DIT = 58.3 s / mag*(K)=5.24 / mag_p(instru) = 14) : 

#n,sep,c=mirimrs(systematic=False,bkgd="low",delta_radial_velocity=-25,model="BT-Settl",calculation="contrast",T_planet=1100,lg_planet=4.0,name_planet="HR 8799",mag_star=5.241,band0='instru',T_star=8000,lg_star=4.0,exposure_time=171)
#n,sep,SNR=mirimrs(Rc=100,systematic=True,delta_radial_velocity=-25,channel=False,model="Exo-REM",calculation="SNR",T_planet=1100,lg_planet=4.0,separation_planet=1.72,name_planet="HR 8799 b",mag_planet=14,mag_star=5.241,band0='instru',T_star=8000,lg_star=4.0,exposure_time=171)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATA :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Chameleon  (DIT = 138.75 s) ( SNR(1SHORT) = 16.61 ) n,sep,c=mirimrs(star_pos="edge",bkgd="high",model="BT-Settl",calculation="SNR",separation_planet=2.5,T_planet=2600,lg_planet=3.5,name_planet="CT Cha b",mag_star=8.66,mag_planet=14.9,band0='K',T_star=4600,lg_star=4.0,exposure_time=56.426,channel=False)

#n,sep,c=mirimrs(star_pos="edge",bkgd="high",calculation="contrast",model="BT-Settl",separation_planet=2.5,T_planet=2600,lg_planet=4,name_planet="V* CT Cha B",mag_star=8.66,band0='K',T_star=4400,lg_star=4.0,exposure_time=56.426,plot_mag=False)
#n,sep,c=mirimrs(star_pos="edge",bkgd="high",model="BT-Settl",calculation="SNR",separation_planet=2.5,T_planet=2600,lg_planet=3.5,name_planet="CT Cha b",mag_star=8.66,mag_planet=14.9,band0='K',T_star=4600,lg_star=4.0,exposure_time=56.426,channel=False)



if 1==0 : 
    name_band,separation,signal_band,sigma_s_2_band,sigma_ns_2_band,DIT_band = mirimrs(return_quantity=True,star_pos="edge",delta_radial_velocity=25,bkgd="high",model="BT-Settl",calculation="contrast",separation_planet=2.5,T_planet=2600,lg_planet=4,name_planet="CT Cha b",mag_star=8.66,mag_planet=14.9,band0='K',T_star=4600,lg_star=4.0,exposure_time=56.426,channel=False)
    time = 56.426
    Nint = time/DIT_band[0]
    SNR = np.sqrt(Nint) * signal_band[0] / np.sqrt(sigma_ns_2_band[0] + Nint * sigma_s_2_band[0])
    t_syst = DIT_band[0] * sigma_ns_2_band[0] / sigma_s_2_band[0]
    plt.figure() ; plt.plot(separation[0],t_syst) ; plt.yscale('log') ; plt.show() 



# HD 159222 (G1V / DIT = 41.63 s )
#n,sep,c=mirimrs(bkgd="high",calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='HD 159222',mag_star=5,band0='K',T_star=5800,lg_star=4.5,exposure_time=17.6)


# delUMi (G1V / DIT = 22 s )
#n,sep,c=mirimrs(calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='delUMi',mag_star=4.258,band0='K',T_star=9000,lg_star=4.0,exposure_time=6.455)

# HD2811 (G1V / DIT = 177.603  s )
#n,sep,c=mirimrs(bkgd="high",calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='HD 2811',mag_star=7.057,band0='K',T_star=9000,lg_star=4.0,exposure_time=23.865)

# GO Tau (G1V / DIT = 55.8 s )
#n,sep,c=mirimrs(bkgd=None,calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='GO Tau',mag_star= 9.,band0='K',T_star=3680,lg_star=4.0,exposure_time=38.65)

# TW Cha (G1V / DIT = 97 s )
#n,sep,c=mirimrs(bkgd=None,calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='TW Cha',mag_star=8.616,band0='K',T_star=4680,lg_star=4.0,exposure_time=39.776)

# IRAS-04385 (G1V / DIT = 38.851 s )
#n,sep,c=mirimrs(bkgd=None,calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='IRAS-04385',mag_star=9.2,band0='K',T_star=3680,lg_star=4.0,exposure_time=16.776)

# HD 163296 (G1V / DIT = 13.875 s )
#n,sep,c=mirimrs(calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='HD 163296',mag_star=4.78,band0='K',T_star=8680,lg_star=4.0,exposure_time=6.5)

# HD 167060 (G1V / DIT = 250 s )
#n,sep,c=mirimrs(calculation="contrast",model="BT-Settl",T_planet=1000,lg_planet=4.0,name_planet='HD 167060',mag_star=7.43,band0='K',T_star=6080,lg_star=4.0,exposure_time=33.5)











# HD 19467 (G1V / DIT = 218.8 s )
#n,sep,c=nirspec(calculation="contrast",model="BT-Settl",T_planet=950,lg_planet=5.0,name_planet='HD 19467 b',mag_star=5.4,band0='K',T_star=5680,lg_star=4.0,exposure_time=65.65)




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COLORMAPS :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



def run_colormap_tradeoff_band(T_planet,spectrum_contributions,model,instru,Rc=100):
    if instru=="MIRIMRS" or instru=="NIRSpec":
        tellurics = False ; nbPixels = 1024 ; step_l0=0.1
    elif instru=="HARMONI" or instru=="ERIS" or instru=="ANDES" :
        tellurics = True ; nbPixels = 3330 ; step_l0=0.01
    else : 
        tellurics = False ; nbPixels = 2048 ; step_l0=0.1
    T_star = 4000
    delta_radial_velocity = 25
    lost_signal = False
    broadening = 0
    lg_planet = 4.0 ; lg_star = 4.0
    colormap_tradeoff_band(T_planet,T_star,lg_planet,lg_star, step_l0, nbPixels, tellurics,delta_radial_velocity,broadening,
                 spectrum_contributions, model, instru, Rc, lost_signal)

#run_colormap_tradeoff_band(T_planet=200,spectrum_contributions="thermal",model="BT-Settl",instru="all")

#colormap_tradeoff_rv(T_planet=3000,T_star=3000,lg_planet=4.0,lg_star=5.0, tellurics=True,spectrum_contributions="reflected", model="tellurics", instru="HARMONI",band="H",Rc=100)






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
    #sigma = l0/(np.pi*Rc)*np.sqrt(np.log(2)/2)
    
    wave=planet_spectrum.wavelength ; flux=planet_spectrum.flux ; flux_BF=gaussian_filter(flux,sigma) ; flux_HF=flux-flux_BF
    plt.figure() ; plt.plot(wave,flux,'r') ; plt.xlabel('wavelength (in µm)',fontsize=14) ; plt.xscale('log')
    plt.plot(wave,flux_BF,'g')  ; plt.ylabel("flux (in photons/s)",fontsize=14) ; plt.title(f'Spectrum of a planet at 500K on {band}',fontsize=14)
    plt.plot(wave,flux_HF,'b') ; plt.legend(["$S_p$","[$S_p$]$_{BF}$","[$S_p$]$_{HF}$"],fontsize=14)
    spectrum=Spectrum(wave,flux,planet_spectrum.R,None)
    spectrum_BF=Spectrum(wave,flux_BF,planet_spectrum.R,None)
    spectrum_HF=Spectrum(wave,flux_HF,planet_spectrum.R,None)
    plt.figure()
    res,psd=spectrum.plot_psd(smooth=0, color='r',show=True,ret=True,area=False)
    #res,psd_BF=spectrum_BF.plot_psd(smooth=0, color='g',show=True,ret=True)
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






