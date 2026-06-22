# import FastYield modules
from fastyield.config import set_sim_data_path, instrus
from fastyield.FastCurves import FastCurves
from fastyield.FastYield_interface import FastYield_interface
from fastyield.FastYield import planet_table_classification, planet_table_classification_histogram, planet_table_statistics, yield_plot_instrus_texp, yield_plot_bands_texp, yield_hist_instrus_ptypes, yield_hist_instrus_ptypes_ELT, yield_corner_instru, yield_corner_instrus, yield_corner_models, yield_contrast_instru, yield_contrast_ELT_earthlike, get_archive_table, all_SNR_table, get_planet_table_SNR
from fastyield.colormaps import colormap_bandwidth_resolution_with_constant_Nlambda, colormap_bandwidth_resolution_with_constant_Dlambda, colormap_bandwidth_Tp, colormap_bands_Tp, colormap_bands_ptypes_SNR, colormap_bands_ptypes_parameters, colormap_rv, colormap_vrot, colormap_maxsep_phase_inc

# Required if FASTYIELD_SIM_DATA_PATH is not defined:
# set the path to your local FastYield sim_data directory.
# This directory should contain the instrumental/simulation data folders
# and the Spectra/ directory.
# set_sim_data_path("/path/to/sim_data")



#------------------------------------------------------------------------------#
#                       Graphic User Interface (GUI):                          #
#------------------------------------------------------------------------------#

#FastYield_interface()



#------------------------------------------------------------------------------#
#                                FastYield plots:                              #
#------------------------------------------------------------------------------#

# planet_table_classification()
# planet_table_classification_histogram()
# planet_table_statistics()

# yield_plot_instrus_texp(thermal_model="auto", reflected_model="auto", fraction=False)
# yield_plot_bands_texp(table="Archive", instru="HARMONI", thermal_model="auto", reflected_model="auto", systematics=False, PCA=False, fraction=False)

# yield_hist_instrus_ptypes(exposure_time=10*60,     thermal_model="auto", reflected_model="auto", planet_types=planet_types_reduced, fraction=False)
# yield_hist_instrus_ptypes_ELT(exposure_time=10*60, thermal_model="auto", reflected_model="auto", planet_types=planet_types,         fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox", "ANDES", "ANDES+LYOT"])



# yield_hist_instrus_ptypes_ELT(exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", planet_types=planet_types_reduced, fraction=False, instrus=["HARMONI", "HARMONI+SP_Prox"])

# yield_corner_instru(instru="HARMONI", exposure_time=600, thermal_model="BT-Settl", reflected_model="tellurics", apodizer="NO_SP", strehl="JQ1", coronagraph=None, band="INSTRU", systematics=False, PCA=False)
# yield_corner_instrus(instru1="HARMONI", instru2="ANDES",   apodizer1="SP_Prox", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", coronagraph1=None, coronagraph2=None,   exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False)
# yield_corner_instrus(instru1="HARMONI", instru2="HARMONI", apodizer1="SP_Prox", apodizer2="NO_SP", strehl1="JQ1", strehl2="JQ1", coronagraph1=None, coronagraph2=None,   exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False)
# yield_corner_instrus(instru1="ANDES",   instru2="ANDES",   apodizer1="NO_SP",   apodizer2="NO_SP", strehl1="MED", strehl2="MED", coronagraph1=None, coronagraph2="LYOT", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="tellurics", systematics=False, PCA=False)
# yield_corner_models(model1="tellurics", model2="PICASO", instru="ANDES", apodizer="NO_SP", strehl="MED", exposure_time=6*60, band="INSTRU")

# yield_contrast_instru(instru="ANDES", exposure_time=6*60, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="MED", band="INSTRU", coronagraph="LYOT")
# yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", force_table_calc=False, exposure_time=6*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3)

# yield_contrast_ELT_earthlike(thermal_model="BT-Settl", reflected_model="tellurics", spectrum_contributions="thermal+reflected", force_table_calc=True, exposure_time=10*60, Rc=100, sep_max=100, s0=50, ds=10*50, alpha_sig=0.3)



#------------------------------------------------------------------------------#
#                           Update FastYield:                                  #
#------------------------------------------------------------------------------#

# get_archive_table()                             # ~ 7 mn
# all_SNR_table(table="Archive", instrus=instrus) # ~ 15 hours
# all_SNR_table(table="Archive", instrus=["MIRIMRS", "NIRCam", "NIRSpec", "VIPAPYRUS"])


# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="auto", reflected_model="auto",  apodizer="NO_SP",   strehl="JQ1", coronagraph=None, systematics=False) # ~ 3mn
# get_planet_table_SNR(instru="HARMONI", table="Archive", thermal_model="auto", reflected_model="auto",  apodizer="SP_Prox", strehl="JQ1", coronagraph=None, systematics=False) # ~ 3mn

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

# FastCurves(instru="CRIRES+", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

# FastCurves(instru="MIRIMRS", model_planet=" BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="NIRSpec", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="NIRCam", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ", coronagraph="MASK335R")

# FastCurves(instru="VIPAPYRUS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

 
 
#------------------------------------------------------------------------------#
#                  FastCurves (comparison with real data cases):               #
#------------------------------------------------------------------------------#

# ### CT Cha b / SNR(1MEDIUM) = 11.72 (3.5 s)
# FastCurves(calculation="SNR", instru="MIRIMRS", systematics=True, input_DIT=138.75/60, model_planet="BT-Settl", separation_planet=2.5, T_planet=2600, lg_planet=3.5, planet_name="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, rv_star=-2.9, rv_planet=15, vsini_star=10, vsini_planet=10, channel=False)

# ### HD 19467 b / SNR(G395H F290LP) = 19.69 (8.5 s)
# FastCurves(calculation="SNR", instru="NIRSpec", systematics=True, separation_planet=1.5, input_DIT=218.8/60, model_planet="BT-Settl", T_planet=950, lg_planet=5.0, planet_name='HD 19467 b', mag_star=5.4, band0='K', mag_planet=17.97, T_star=5680, lg_star=4.0, exposure_time=65.65)

# ### HIP 65426 b / SNR(F356W) = 779.17
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

# colormap_bandwidth_resolution_with_constant_Nlambda(instru="HARMONI", T_planet=1200, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", noise_regime="photon")
# colormap_bandwidth_resolution_with_constant_Dlambda(instru="HARMONI", T_planet=1400, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", noise_regime="photon")

# colormap_bandwidth_Tp(instru="HARMONI", T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bands_Tp(instru="HARMONI", T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bands_ptypes_SNR(mode="multi",  instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="JQ1", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_ptypes_SNR(mode="unique", instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="JQ1", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_ptypes_SNR(mode="multi",  instru="ANDES,    thermal_model="auto", reflected_model="auto", exposure_time=10*60, strehl="MED", systematics=False, PCA=False, planet_types=planet_types)
# colormap_bands_ptypes_parameters(mode="unique", Nmax=1, instru="HARMONI", thermal_model="auto", reflected_model="auto", exposure_time=10*60, apodizer="NO_SP", strehl="JQ1", coronagraph=None, systematics=False, PCA=False, PCA_mask=False, N_PCA=20, Rc=100, filter_type="gaussian", planet_types=planet_types_reduced)

# colormap_rv(instru="HARMONI",    band="J", T_planet=300, T_star=3000, spectrum_contributions="reflected", model="flat", airmass=2.0, stellar_halo_photon_noise_limited=False)
# colormap_vrot(instru="HARMONI", band="J", T_planet=300, T_star=3000, delta_rv=30, inc=0, spectrum_contributions="reflected", model="tellurics", airmass=2.0, stellar_halo_photon_noise_limited=False)

# colormap_maxsep_phase_inc(instru="HARMONI", band="H", apodizer="NO_SP", strehl="JQ1", coronagraph=None)







