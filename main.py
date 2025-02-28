from src.FastCurves import *
from src.colormaps import *
from src.FastCurves_interface import *
from src.FastYield_interface import *
from src.DataAnalysis_interface import *



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

#archive_yield(exposure_time=120, contrast=True, save=False)
#archive_yield_plot(fraction=False)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves (theoritical cases) :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FastCurves(instru="HARMONI", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='instru', T_star=9600, lg_star=4.0, exposure_time=120, apodizer="SP1", strehl="MED")

#FastCurves(instru="HARMONI", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ1")

#FastCurves(instru="ERIS", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ0")

#FastCurves(instru="ANDES", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves (real data cases) :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Chameleon  (DIT = 138.75 s) / SNR(1SHORT) = 13.3
#FastCurves(instru="MIRIMRS", systematic=True, input_DIT=138.75/60, model="BT-Settl", calculation="SNR", separation_planet=2.5, T_planet=2600, lg_planet=3.5, name_planet="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, star_rv=-2.9, channel=False)

# HD 19467 (G1V / DIT = 218.8 s )
#FastCurves(calculation="SNR", instru="NIRSpec", systematic=True, separation_planet=1.5, input_DIT=218.8/60, model="BT-Settl", T_planet=950, lg_planet=5.0, name_planet='HD 19467 b', mag_star=5.4, band0='K', mag_planet=17.97, T_star=5680, lg_star=4.0, exposure_time=65.65)
#FastCurves(instru="MIRIMRS", systematic=True, separation_planet=1.5, band_only="1SHORT", systematic=True, input_DIT=218.8/60, calculation="contrast", model="BT-Settl", T_planet=950, lg_planet=5.0, name_planet='HD 19467 b', mag_star=5.4, band0='K', T_star=5680, lg_star=4.0, exposure_time=0.665)

# HIP 65426 b (DIT = 308 s)
#FastCurves(instru="NIRCam", input_DIT=308/60, calculation="contrast", T_planet=1600, lg_planet=4.0, separation_planet=0.8, name_planet="HIP 65426 b", mag_planet=6.771+9.85, mag_star=6.771, band0='K', T_star=8000, lg_star=4.0, exposure_time=20.3)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COLORMAPS :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#colormap_bandwidth_resolution(T_planet=1000, T_star=6000, delta_rv=30, spectrum_contributions="thermal", model="BT-Settl", instru="HiRISE", Rc=100, used_filter="gaussian")
#colormap_bandwidth_Tp(instru="HARMONI", T_star=5000, delta_rv=25, spectrum_contributions="reflected", model="PICASO", Rc=100, used_filter="gaussian")
#colormap_rv(T_planet=300, T_star=9000, spectrum_contributions="reflected", model="flat", instru="HARMONI", band="H", Rc=100)
#colormap_vsini(T_planet=1000, T_star=3000, spectrum_contributions="reflected", model="flat", instru="HARMONI", band="H", Rc=100)
#colormap_maxsep_phase(instru="HARMONI", band="H", inc=90)






