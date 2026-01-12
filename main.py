from src.colormaps import *
from src.FastYield_interface import *


#FastCurves(return_quantity=True, band_only="H", instru="HARMONI", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ1")



#------------------------------------------------------------------------------#
#                           Brown dwarfs with VIPAPYRUS:                       #
#------------------------------------------------------------------------------#

# # κ And b
# FastCurves(instru="VIPAPYRUS", planet_name="κ And b", model_planet="BT-Settl", calculation="SNR", separation_planet=1.05, T_planet=1900, lg_planet=4.0, vsini_planet=0, mag_planet=15.01, T_star=10900, lg_star=4.0, mag_star=4.595, vsini_star=190, band0='H', exposure_time=120, apodizer="NO_SP", strehl="MED")

# # HR 7672 B
# FastCurves(instru="VIPAPYRUS", planet_name="HR 7672 B", model_planet="BT-Settl", calculation="SNR", separation_planet=0.788, T_planet=1600, lg_planet=4.0, vsini_planet=0, mag_planet=13.04, T_star=5880, lg_star=4.0, mag_star=4.388, vsini_star=2, band0='K', exposure_time=120, apodizer="NO_SP", strehl="MED")

# # HD 49197 B (possible ?)
# FastCurves(instru="VIPAPYRUS", planet_name="HD 49197 B", model_planet="BT-Settl", calculation="SNR", separation_planet=0.95, T_planet=1700, lg_planet=4.0, vsini_planet=0, mag_planet=14.61, T_star=6500, lg_star=4.0, mag_star=6.095, vsini_star=25, band0='H', exposure_time=120, apodizer="NO_SP", strehl="MED")

# # HD 130948 B
# FastCurves(instru="VIPAPYRUS", planet_name="HD 130948 B", model_planet="BT-Settl", calculation="SNR", separation_planet=2.56, T_planet=1900, lg_planet=5.0, vsini_planet=62, mag_planet=11.8, T_star=5780, lg_star=4.18, mag_star=4.69, vsini_star=6.8, band0='H', exposure_time=120, apodizer="NO_SP", strehl="MED")



#------------------------------------------------------------------------------#
#                            Graphic interface:                                #
#------------------------------------------------------------------------------#

FastYield_interface()



#------------------------------------------------------------------------------#
#                                FastYield plots:                              #
#------------------------------------------------------------------------------#

# planet_table_classification()
# planet_table_statistics()

# yield_plot_instrus_texp(thermal_model="BT-Settl", reflected_model="PICASO", fraction=False)
# yield_plot_bands_texp(instru="ANDES", thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False, fraction=False)

# yield_hist_instrus_ptypes(exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", planet_types=planet_types_reduced, fraction=False)
# yield_hist_instrus_ptypes_ELT(exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", planet_types=planet_types_reduced, fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox", "ANDES", "ANDES+LYOT"])
# yield_hist_instrus_ptypes_ELT(exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", planet_types=planet_types_reduced, fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox"])

# yield_corner_instru(instru="HARMONI", exposure_time=600, thermal_model="BT-Settl", reflected_model="tellurics", apodizer="NO_SP", strehl="JQ1", coronagraph=None, band="INSTRU", systematics=False, PCA=False)
# yield_corner_instrus(instru1="HARMONI", instru2="ANDES", apodizer1="SP_Prox", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", coronagraph1=None, coronagraph2=None, exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False)
# yield_corner_instrus(instru1="HARMONI", instru2="HARMONI", apodizer1="SP_Prox", apodizer2="NO_SP", strehl1="JQ1", strehl2="JQ1", coronagraph1=None, coronagraph2=None, exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False)
# yield_corner_instrus(instru1="ANDES", instru2="ANDES", apodizer1="NO_SP", apodizer2="NO_SP", strehl1="MED", strehl2="MED", coronagraph1=None, coronagraph2="LYOT", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False)
# yield_corner_models(model1="tellurics", model2="PICASO", instru="ANDES", apodizer="NO_SP", strehl="MED", exposure_time=6*60, band="INSTRU")

# yield_contrast_instru(instru="ANDES", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", band="INSTRU", coronagraph="LYOT")
# yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", force_table_calc=False, exposure_time=6*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3)

# yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", spectrum_contributions="thermal+reflected", force_table_calc=True, exposure_time=10*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3)


#------------------------------------------------------------------------------#
#                           Update FastYield:                                  #
#------------------------------------------------------------------------------#

# get_archive_table()                                               # ~ 7 mn
# all_SNR_table(table="Archive", instru_name_list=instru_name_list) # ~ 15 hours

# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="BT-Settl", reflected_model="PICASO",    apodizer="NO_SP",   strehl="JQ1", coronagraph=None, systematics=False) # (60 s)
# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="BT-Settl", reflected_model="PICASO",    apodizer="SP_Prox", strehl="JQ1", coronagraph=None, systematics=False) # (60 s)
# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="BT-Settl", reflected_model="tellurics", apodizer="NO_SP",   strehl="JQ1", coronagraph=None, systematics=False)
# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="BT-Settl", reflected_model="tellurics", apodizer="SP_Prox", strehl="JQ1", coronagraph=None, systematics=False)

# get_planet_table_SNR(instru="ANDES", table="Archive", thermal_model="BT-Settl", reflected_model="PICASO",    apodizer="NO_SP", strehl="MED", coronagraph=None,   systematics=False)
# get_planet_table_SNR(instru="ANDES", table="Archive", thermal_model="BT-Settl", reflected_model="PICASO",    apodizer="NO_SP", strehl="MED", coronagraph="LYOT", systematics=False)
# get_planet_table_SNR(instru="ANDES", table="Archive", thermal_model="BT-Settl", reflected_model="tellurics", apodizer="NO_SP", strehl="MED", coronagraph=None,   systematics=False)
# get_planet_table_SNR(instru="ANDES", table="Archive", thermal_model="BT-Settl", reflected_model="tellurics", apodizer="NO_SP", strehl="MED", coronagraph="LYOT", systematics=False)

# get_planet_table_SNR(instru="VIPAPYRUS", table="Archive", thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", coronagraph=None, systematics=False) # (10 s)



#------------------------------------------------------------------------------#
#                     FastCurves (theoritical cases):                          #
#------------------------------------------------------------------------------#

# FastCurves(instru="HARMONI", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ1")

# FastCurves(instru="ANDES", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

# FastCurves(instru="ANDES", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED", coronagraph="LYOT")

# FastCurves(instru="ERIS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ0")

# FastCurves(instru="HiRISE", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

# FastCurves(instru="MIRIMRS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="NIRSpec", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="NIRCam", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ", coronagraph="MASK335R")

# FastCurves(instru="VIPAPYRUS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

 
 
#------------------------------------------------------------------------------#
#                      FastCurves (real data cases):                           #
#------------------------------------------------------------------------------#

# ### CT Cha b / SNR(1MEDIUM) = 15.51 (6.8 s)
# FastCurves(calculation="SNR", instru="MIRIMRS", systematics=True, input_DIT=138.75/60, model_planet="BT-Settl", separation_planet=2.5, T_planet=2600, lg_planet=3.5, planet_name="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, rv_star=-2.9, rv_planet=15, vsini_star=10, vsini_planet=10, channel=False)

# ### HD 19467 b / SNR(G395H F290LP) = 17.46
# FastCurves(calculation="SNR", instru="NIRSpec", systematics=True, separation_planet=1.5, input_DIT=218.8/60, model_planet="BT-Settl", T_planet=950, lg_planet=5.0, planet_name='HD 19467 b', mag_star=5.4, band0='K', mag_planet=17.97, T_star=5680, lg_star=4.0, exposure_time=65.65)

# ### HIP 65426 b / SNR(F356W) = 764.63
# FastCurves(instru="NIRCam", input_DIT=308/60, calculation="SNR", model_planet="BT-Settl", T_planet=1600, lg_planet=4.0, separation_planet=0.8, planet_name="HIP 65426 b", mag_planet=6.771+9.85, mag_star=6.771, band0='K', T_star=8000, lg_star=4.0, exposure_time=20.3, coronagraph="MASK335R")



#------------------------------------------------------------------------------#
#                                Colormaps:                                    #
#------------------------------------------------------------------------------#

# colormap_bandwidth_resolution_with_constant_Nlambda(num=200, instru="PCS", Nlambda=10_000,  photon_noise_limited=True, T_planet=300, T_star=3000, delta_rv=30, spectrum_contributions="reflected", model="tellurics", Rc=100, filter_type="gaussian")
# colormap_bandwidth_resolution_with_constant_Nlambda(num=200, instru="PCS", Nlambda=100_000, photon_noise_limited=True, T_planet=300, T_star=3000, delta_rv=30, spectrum_contributions="reflected", model="tellurics", Rc=100, filter_type="gaussian")

# colormap_bandwidth_resolution_with_constant_Nlambda(instru="HARMONI", T_planet=1200, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True)
# colormap_bandwidth_resolution_with_constant_Dlambda(instru="HARMONI", T_planet=1400, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True)

# colormap_bandwidth_Tp(instru="HARMONI", T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True)
# colormap_bands_Tp(instru="HARMONI", T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", photon_noise_limited=True)
# colormap_bands_planets_SNR(mode="multi", instru="HARMONI", thermal_model="BT-Settl", reflected_model="tellurics", exposure_time = 6*60, strehl="JQ1", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_planets_SNR(mode="multi", instru="ANDES",   thermal_model="BT-Settl", reflected_model="tellurics", exposure_time = 6*60, strehl="MED", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_planets_parameters(Nmax=1, mode="multi", instru="HARMONI", thermal_model="BT-Settl", reflected_model="tellurics", exposure_time = 2*60, apodizer="SP_Prox", strehl="JQ1", systematics=False, PCA=False)

# colormap_rv(instru="HARMONI", band="K2_high", T_planet=300, T_star=3000, spectrum_contributions="reflected", model="tellurics", photon_noise_limited=False)
# colormap_vsini(instru="HARMONI", band="J", T_planet=300, T_star=3000, delta_rv=0, spectrum_contributions="reflected", model="flat", photon_noise_limited=False)

# colormap_maxsep_phase(instru="HARMONI", band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None, inc=90)
# colormap_maxsep_inc(instru="HARMONI",   band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None)

# colormap_best_parameters_earth(Npx=10000, T_planet=288, T_star=5800, lg_planet=3.0, lg_star=4.4, delta_rv=30, vsini_planet=0.5, vsini_star=2, SMA=1, planet_radius=1, star_radius=1, distance=1, thermal_model="BT-Settl", reflected_model="tellurics", Rc=100, filter_type="gaussian", photon_noise_limited=True, norm_plot="star")



#------------------------------------------------------------------------------#
#                                   Utils:                                     #
#------------------------------------------------------------------------------#

#plot_dark_noise_budget(instru="HARMONI", noise_level=28)

if 1==0:
    N      = 1_000
    NbRead = np.linspace(2, 400, N)
    RON0   = 12 # e-/px/DIT
    RON_FC = RON0 / np.sqrt(NbRead)
    RON_eff = np.zeros((N)) + np.nan
    for n in range(N):
        RON_eff[n] = estimate_RON_up_the_ramp(N=NbRead[n], RON0=RON0, A=1.0, B= 0.0)
    
    plt.figure(dpi=300, figsize=(10, 6))
    plt.plot(NbRead, RON_FC, c="crimson", label="FastCurves")
    plt.plot(NbRead, RON_eff, c="royalblue", label="Pipeline")
    plt.xlim(NbRead[0], NbRead[-1])
    plt.ylim(0, RON0)
    plt.xlabel("Number of read")
    plt.ylabel("Effective RON [e-/px/DIT]")
    plt.axhline(0.5, c="black", ls=":", label="Lab limit (0.5 e-/px/DIT)")
    plt.legend()
    plt.show()
    
