# import FastYield modules
from fastyield.config import set_sim_data_path
from fastyield.spectrum import load_star_spectrum

# import matplotlib modules
import matplotlib.pyplot as plt

# import numpy
import numpy as np

# set_sim_data_path("/path/to/sim_data")
set_sim_data_path("/home/martoss/Documents/PhD/work/sim_data")

# Loading spectra in [J/s/m2/µm]
star_spectrum_BTNextGen = load_star_spectrum(T_star=3_000, lg_star=4.0, model="BT-NextGen")
star_spectrum_Husser    = load_star_spectrum(T_star=3_000, lg_star=4.0, model="Husser")

# Plot
plt.figure(dpi=300)
plt.plot(star_spectrum_BTNextGen.wavelength, star_spectrum_BTNextGen.flux, label=f"{star_spectrum_BTNextGen.model} (R = {np.nanmedian(star_spectrum_BTNextGen.R):.0f}):\n{star_spectrum_BTNextGen.T:.0f}K and logg = {star_spectrum_BTNextGen.lg:.1f}dex")
plt.plot(star_spectrum_Husser.wavelength,    star_spectrum_Husser.flux,    label=f"{star_spectrum_Husser.model} (R = {np.nanmedian(star_spectrum_Husser.R):.0f}):\n{star_spectrum_Husser.T:.0f}K and logg = {star_spectrum_Husser.lg:.1f}dex")
plt.xlim(0.1, 30)
plt.ylabel("Flux in [J/s/m2/µm]")
plt.xlabel("Wavelength in [µm]")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="lower right")
plt.show()