from src.colormaps import *
from src.FastYield_interface import *



#------------------------------------------------------------------------------#
#                            Graphic interface:                                #
#------------------------------------------------------------------------------#

FastYield_interface()



#------------------------------------------------------------------------------#
#                           Update FastYield:                                  #
#------------------------------------------------------------------------------#

# create_fastcurves_table(table="Archive") # ~ 5 mn
# all_SNR_table(table="Archive")           # ~ 15 hours

# create_fastcurves_table(table="Simulated") # needs an update
# all_SNR_table(table="Simulated")           # needs an update



#------------------------------------------------------------------------------#
#                                Yield plots:                                  #
#------------------------------------------------------------------------------#

# archive_yield_instrus_plot_texp(fraction=False)
# archive_yield_instrus_plot_ptypes(exposure_time=5*60, fraction=False)

# archive_yield_bands_plot_texp(fraction=False, instru="HARMONI", strehl="JQ1", thermal_model="BT-Settl", reflected_model="tellurics", systematic=False, PCA=False)
# detections_corner(instru="HARMONI", exposure_time=600, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", band="INSTRU")
# detections_corner_instrus_comparison(exposure_time=600, instru1="HARMONI", instru2="ANDES", apodizer1="SP_Prox", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", thermal_model="BT-Settl", reflected_model="tellurics")
# detections_corner_models_comparison(model1="tellurics", model2="PICASO", instru="ANDES", apodizer="NO_SP", strehl="MED", exposure_time=600, band="INSTRU")
# detections_corner_apodizers_comparison(exposure_time=600, thermal_model="BT-Settl", reflected_model="tellurics", band="INSTRU")



#------------------------------------------------------------------------------#
#                     FastCurves (theoritical cases):                          #
#------------------------------------------------------------------------------#

# FastCurves(instru="HARMONI", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ1")

# FastCurves(instru="ANDES", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

# FastCurves(instru="ERIS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ0")

# FastCurves(instru="HiRISE", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

# FastCurves(instru="MIRIMRS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="NIRSpec", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="NIRCam", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")

# FastCurves(instru="VIPAPYRUS", model_planet="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")



#------------------------------------------------------------------------------#
#                      FastCurves (real data cases):                           #
#------------------------------------------------------------------------------#

### CT Cha b / SNR(1MEDIUM) = 14.66
# FastCurves(calculation="SNR", instru="MIRIMRS", systematic=True, input_DIT=138.75/60, model_planet="BT-Settl", separation_planet=2.5, T_planet=2600, lg_planet=3.5, planet_name="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, rv_star=-2.9, rv_planet=15, vsini_star=10, vsini_planet=10, channel=False)

### HD 19467 b / SNR(G395H F290LP) = 10.5
# FastCurves(calculation="SNR", instru="NIRSpec", systematic=True, separation_planet=1.5, input_DIT=218.8/60, model_planet="BT-Settl", T_planet=950, lg_planet=5.0, planet_name='HD 19467 b', mag_star=5.4, band0='K', mag_planet=17.97, T_star=5680, lg_star=4.0, exposure_time=65.65)

### HIP 65426 b / SNR(F356W) = 764.83
# FastCurves(instru="NIRCam", input_DIT=308/60, calculation="SNR", T_planet=1600, lg_planet=4.0, separation_planet=0.8, planet_name="HIP 65426 b", mag_planet=6.771+9.85, mag_star=6.771, band0='K', T_star=8000, lg_star=4.0, exposure_time=20.3)



#------------------------------------------------------------------------------#
#                                Colormaps:                                    #
#------------------------------------------------------------------------------#

# colormap_bandwidth_resolution_with_constant_Nlambda(T_planet=500, T_star=5000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bandwidth_resolution_with_constant_Dlambda(T_planet=1400, T_star=6000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", instru="HiRISE", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bandwidth_Tp(instru="HiRISE", T_star=6000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bands_Tp(instru="HARMONI", T_star=6000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
# colormap_bands_planets_SNR(mode="multi", instru="HARMONI", thermal_model="BT-Settl", reflected_model="tellurics", exposure_time = 2*60, apodizer="SP_Prox", strehl="JQ1", systematic=False, PCA=False, PCA_mask=False, Nc=20, Rc=100, filter_type="gaussian")
# colormap_bands_planets_parameters(Nmax=10, mode="unique", instru="HARMONI", thermal_model="BT-Settl", reflected_model="tellurics", exposure_time = 2*60, apodizer="NO_SP", strehl="JQ1", systematic=False, PCA=False, PCA_mask=False, Nc=20, Rc=100, filter_type="gaussian")
# colormap_rv(T_planet=300, T_star=3000, spectrum_contributions="reflected", model="flat", instru="HARMONI", band="J", Rc=100)
# colormap_vsini(T_planet=300, T_star=3000, spectrum_contributions="reflected", model="flat", instru="HARMONI", band="J", Rc=100)
# colormap_maxsep_phase(instru="HARMONI", band="H", inc=90)
# colormap_best_parameters_earth(norm_plot="1", thermal_model="BT-Settl", reflected_model="tellurics", T_planet=288, rv_planet=30, R_star=1, SMA=1, Npx=10000, stellar_halo_photon_noise_limited=True)



#------------------------------------------------------------------------------#
#                                   Utils:                                     #
#------------------------------------------------------------------------------#

#plot_dark_noise_budget(instru="HARMONI", noise_level=28)


