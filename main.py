from src.utils_imports import *



FastCurves(band_only="J", instru="HARMONI", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="SP_Prox", strehl="JQ1")

#FastCurves(band_only="YJH_10mas_100", instru="ANDES", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

FastCurves(band_only="YJH_10mas_100", instru="ANDES", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED", coronagraph="LYOT")



#------------------------------------------------------------------------------#
#                       Graphic User Interface (GUI):                          #
#------------------------------------------------------------------------------#

# FastYield_interface()



#------------------------------------------------------------------------------#
#                                FastYield plots:                              #
#------------------------------------------------------------------------------#

# planet_table_classification()
# planet_table_statistics()

# yield_plot_instrus_texp(thermal_model="auto", reflected_model="auto", fraction=False)
# yield_plot_bands_texp(table="Archive", instru="HARMONI", thermal_model="auto", reflected_model="auto", systematics=False, PCA=False, fraction=False)

# yield_hist_instrus_ptypes(exposure_time=10*60,     thermal_model="auto", reflected_model="auto", planet_types=planet_types_reduced, fraction=False)
# yield_hist_instrus_ptypes_ELT(exposure_time=10*60, thermal_model="auto", reflected_model="auto", planet_types=planet_types,         fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox", "ANDES", "ANDES+LYOT"])



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

# get_archive_table()                             # ~ 7 mn
# all_SNR_table(table="Archive", instrus=instrus) # ~ 15 hours

# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="auto", reflected_model="auto",  apodizer="NO_SP",   strehl="JQ1", coronagraph=None, systematics=False) # ~ 2mn
# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="auto", reflected_model="auto",  apodizer="SP_Prox", strehl="JQ1", coronagraph=None, systematics=False) # ~ 2mn

# get_planet_table_SNR(instru="ANDES", table="Archive", thermal_model="auto", reflected_model="auto",    apodizer="NO_SP", strehl="MED", coronagraph=None,   systematics=False) # ~ 30 mn
# get_planet_table_SNR(instru="ANDES", table="Archive", thermal_model="auto", reflected_model="auto",    apodizer="NO_SP", strehl="MED", coronagraph="LYOT", systematics=False) # ~ 30 mn




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

# ### CT Cha b / SNR(1MEDIUM) = 12.43 (2.5 s)
# FastCurves(calculation="SNR", instru="MIRIMRS", systematics=True, input_DIT=138.75/60, model_planet="BT-Settl", separation_planet=2.5, T_planet=2600, lg_planet=3.5, planet_name="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, rv_star=-2.9, rv_planet=15, vsini_star=10, vsini_planet=10, channel=False)

# ### HD 19467 b / SNR(G395H F290LP) = 17.21 (5 s)
# FastCurves(calculation="SNR", instru="NIRSpec", systematics=True, separation_planet=1.5, input_DIT=218.8/60, model_planet="BT-Settl", T_planet=950, lg_planet=5.0, planet_name='HD 19467 b', mag_star=5.4, band0='K', mag_planet=17.97, T_star=5680, lg_star=4.0, exposure_time=65.65)

# ### HIP 65426 b / SNR(F356W) = 765.99
# FastCurves(calculation="SNR", instru="NIRCam", input_DIT=308/60, model_planet="BT-Settl", T_planet=1600, lg_planet=4.0, separation_planet=0.8, planet_name="HIP 65426 b", mag_planet=6.771+9.85, mag_star=6.771, band0='K', T_star=8000, lg_star=4.0, exposure_time=20.3, coronagraph="MASK335R")



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
# FastCurves(calculation="SNR", instru="VIPAPYRUS", planet_name="HD 130948 B", model_planet="BT-Settl", separation_planet=2.56, T_planet=1900, lg_planet=5.0, vsini_planet=62, mag_planet=11.8, T_star=5780, lg_star=4.18, mag_star=4.69, vsini_star=6.8, rv_star=10, rv_planet=30, band0='H', exposure_time=120, apodizer="NO_SP", strehl="MED")



#------------------------------------------------------------------------------#
#                                Colormaps:                                    #
#------------------------------------------------------------------------------#

# colormap_bandwidth_resolution_with_constant_Nlambda(instru="HARMONI", T_planet=1200, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bandwidth_resolution_with_constant_Dlambda(instru="HARMONI", T_planet=1400, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)

# colormap_bandwidth_Tp(instru="HARMONI", T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bands_Tp(instru="HARMONI", T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bands_ptypes_SNR(mode="multi",  instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="JQ1", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_ptypes_SNR(mode="unique", instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="JQ1", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_ptypes_SNR(mode="multi",  instru="ANDES,    thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="MED", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_ptypes_parameters(mode="unique", Nmax=1, instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, apodizer="NO_SP", strehl="JQ1", coronagraph=None, systematics=False, PCA=False, PCA_mask=False, N_PCA=20, Rc=100, filter_type="gaussian", planet_types=planet_types_reduced)

# colormap_rv(instru="HARMONI",    band="J", T_planet=300, T_star=3000, spectrum_contributions="reflected", model="flat", airmass=2.0, stellar_halo_photon_noise_limited=False)
# colormap_vrot(instru="HARMONI", band="J", T_planet=300, T_star=3000, delta_rv=30, inc=0, spectrum_contributions="reflected", model="tellurics", airmass=2.0, stellar_halo_photon_noise_limited=False)

# colormap_maxsep_phase_inc(instru="HARMONI", band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None)




#------------------------------------------------------------------------------#
#                                   Utils:                                     #
#------------------------------------------------------------------------------#

# plot_dark_noise_budget(instru="HARMONI", noise_level=28)

# if 1==0:
#     N      = 1_000
#     NbRead = np.linspace(2, 400, N)
#     RON0   = 12 # e-/px/DIT
#     RON_FC = RON0 / np.sqrt(NbRead)
#     RON_eff = np.zeros((N)) + np.nan
#     for n in range(N):
#         RON_eff[n] = estimate_RON_up_the_ramp(N=NbRead[n], RON0=RON0, A=1.0, B= 0.0)
    
#     plt.figure(dpi=300, figsize=(10, 6))
#     plt.plot(NbRead, RON_FC, c="crimson", label="FastCurves")
#     plt.plot(NbRead, RON_eff, c="royalblue", label="Pipeline")
#     plt.xlim(NbRead[0], NbRead[-1])
#     plt.ylim(0, RON0)
#     plt.xlabel("Number of read")
#     plt.ylabel("Effective RON [e-/px/DIT]")
#     plt.axhline(0.5, c="black", ls=":", label="Lab limit (0.5 e-/px/DIT)")
#     plt.legend()
#     plt.show()
    




