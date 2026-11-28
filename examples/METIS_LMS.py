"""METIS LMS | FastCurves / FastYield hands-on

Run the cells (# %%) independently in an interactive Python session.
Requires FastYield and its sim_data directory (spectral models, Archive table,
METIS LMS transmissions, backgrounds and CVC PSF profiles).
All performance estimates below are fundamental-noise-limited predictions
with the current provisional METIS CVC PSFs, not on-sky predictions.
"""

# %% Imports and common observing setup

import numpy as np
from fastyield.FastCurves import FastCurves
from fastyield.FastYield_helpers import load_planet_table, get_planet_index
from fastyield.get_specs import get_config_data, get_R_instru
from fastyield.spectrum import get_thermal_reflected_spectrum, get_wave_model, load_vega_spectrum, get_mag

instru        = "METIS_LMS"
exposure_time = 60.     # Total on-source integration time [min], not wall-clock time
band_only     = None#"O29_Ep00"   # LMS setting ID; None evaluates all 107 settings
config_data   = get_config_data(instru)
# Available LMS settings: print(list(config_data["gratings"]))



# %% 1. FastYield graphical interface (optional)

# Uncomment and run this cell to explore precomputed planet populations.
# The GUI blocks execution until closed and requires a graphical session.
from fastyield.FastYield_interface import FastYield_interface
FastYield_interface()



# %% 2. FastCurves: 5-sigma contrast for a user-defined star and planet

results = FastCurves(
    instru=instru,                       # Instrument identifier [-]
    exposure_time=exposure_time,         # Total integration time [min]
    apodizer="NO_SP",                    # No additional apodizer
    strehl="MED",                        # Available PSF label (HEEPS Q2, not a measured Strehl value)
    coronagraph="CVC",                   # Classical vortex coronagraph
    post_processing="molecular mapping", # Or "differential imaging"
    Rc=100,                              # High-pass cutoff resolving power lambda/dlambda [-]; None: no filter
    filter_type="gaussian_fast",         # Spectral filtering kernel (package default)
    background="medium",                 # Thermal background scenario: "low", "medium", "high"
    systematics=False,                   # No calibrated METIS systematic noise model yet
    speckles=False,                      # No calibrated METIS speckle noise model yet
    PCA=False,                           # No PCA subtraction / associated signal loss
    input_DIT=None,                      # Individual integration time [min]; None: automatic DIT
    calculation="SNR",                   # "contrast" (5 sigma) or "SNR"
    mag_star=5.,                         # Apparent stellar Vega magnitude in band0 [mag]
    mag_planet=15.,                      # Apparent planet Vega magnitude in band0 [mag]
    band0="K",                           # Photometric normalization band, not the LMS setting
    model_star="BT-NextGen",             # Stellar spectral model
    T_star=5800.,                        # Stellar effective temperature [K]
    lg_star=4.5,                         # Stellar log10(g / (cm/s^2)) [dex]
    model_planet="BT-Settl",             # Planet thermal spectral model
    T_planet=1000.,                      # Planet effective temperature [K]
    lg_planet=4.,                        # Planet log10(g / (cm/s^2)) [dex]
    rv_star=0.,                          # Stellar radial velocity [km/s]
    rv_planet=0.,                        # Planet radial velocity [km/s]
    vsini_star=0.,                       # Stellar projected rotation velocity [km/s]
    vsini_planet=0.,                     # Planet projected rotation velocity [km/s]
    separation_planet=0.1,               # Angular separation [arcsec], (0.1 arcsec = 100 mas)
    band_only=band_only,                 # LMS setting ID or None
    show_plot=True,                      # Display spectra, contrast curves and noise contributions
    verbose=True,                        # Print observing setup and intermediate quantities
    return_quantity=True,                # Detailed signal/noise/spectral outputs
)



# Each list contains one entry per computed setting (in name_bands order).
# Radial grids and wavelength grids may differ between settings.
# "core" below means the 3 x 3 spaxel extraction aperture used for METIS LMS.
# Signal and noise are for ONE DIT, after spatial extraction and spectral
# projection onto the normalized template (CCF for molecular mapping).
(
    name_bands,           # LMS setting IDs [str]
    separation_bands,     # Radial separation arrays [mas for METIS LMS]
    signal_bands,         # Useful projected signal [e- per core per DIT]; see contrast note below
    sigma_syst_2_bands,   # Systematic variance [e-^2], per core / DIT (zero here)
    sigma_fund_2_bands,   # Total fundamental variance [e-^2], per core / DIT
    sigma_halo_2_bands,   # Stellar halo photon-noise variance [e-^2], per core / DIT
    sigma_det_2_bands,    # Read-noise + dark-current variance [e-^2], per core / DIT
    sigma_bkg_2_bands,    # Optical/thermal background variance [e-^2], per core / DIT
    DIT_bands,            # Individual detector integration times [min]
    planet_flux_bands,    # Planet spectrum after spectral throughput [e- per bin per DIT]
    star_flux_bands,      # Star spectrum after spectral throughput [e- per bin per DIT]
    wave_bands,           # Spectral bin centres [um]
    trans_bands,          # Spectral throughput [e- per incident photon], including atmosphere and QE
    sigma_budget_bands,   # Optional stability budgets; see below
) = results

# planet_flux_bands and star_flux_bands are spatially unallocated spectra:
# they do NOT include the radial CVC transmission or core/halo PSF losses.
# NaNs in transmission/flux arrays include unobserved LMS detector-gap bins.
# sigma_*_2 quantities are variances, not standard deviations: use sqrt().
# sigma_budget_bands = [sigma_m_syst_bands, sigma_m_speck_bands,
#                       sigma_phi_speck_bands] (fractions, fractions, nm RMS).
# These three lists are empty by default (internal show_syst_budget=False).

# IMPORTANT: calculation="contrast" normalizes signal_bands to a unit
# planet/star photon-flux ratio over the full instrumental wavelength range.
# This is NOT the signal of the specified 15-mag planet itself.
# return_quantity=True does not return the final contrast curves directly.
# Recover them from the detailed outputs, using the FastCurves convention:
contrast_bands = []
for signal, var_fund, var_syst, DIT in zip(signal_bands, sigma_fund_2_bands, sigma_syst_2_bands, DIT_bands):
    N_DIT    = exposure_time / DIT  # Effective number of integrations [-]
    contrast = 5 * np.sqrt(var_fund + N_DIT * var_syst) / (np.sqrt(N_DIT) * signal)
    contrast_bands.append(contrast)  # 5-sigma planet/star photon-flux ratio [-]

# Example: contrast and DIT at 100 mas for the first computed setting.
idx_sep = np.nanargmin(np.abs(separation_bands[0] - 100.))
print(f"\n{name_bands[0]}: 5-sigma contrast at {separation_bands[0][idx_sep]:.2f} mas = " f"{contrast_bands[0][idx_sep]:.3e}; DIT = {60 * DIT_bands[0]:.3f} s")

# For contrast or SNR curves only, use return_quantity=False instead.
# The standard return is (name_bands, separation_bands, results_bands),
# where results_bands contains the 5-sigma contrast or S/N (both dimensionless).



# %% 3. FastYield catalogue: select a real planet

planet_table = load_planet_table("Archive_Pull_For_FastYield.ecsv", table="Archive")
planet_name  = "GJ 504 b"  # Exact PlanetName in the catalogue; change as needed
planet_index = get_planet_index(planet_table, planet_name)
planet       = planet_table[planet_index]  # One QTable row, with physical units

# FastYield AngSep is the modelled projected separation, not necessarily the
# instantaneous measured separation at the date of an actual observation.
separation_planet = planet["AngSep"].to_value("arcsec")  # FastCurves input [arcsec]
print(f"\n{planet_name}: separation = {1000 * separation_planet:.2f} mas")



# %% 4. Generate the star and planet spectra (same approach as FastYield)

thermal_model   = "auto"  # Select model from PlanetType / HasAtmosphere; "None" disables thermal emission
reflected_model = "auto"  # Select albedo model automatically; "None" disables reflected light

# High-resolution model grid covering the instrument with a 2% margin.
lmin_instru = config_data["lambda_range"]["lambda_min"]  # [um]
lmax_instru = config_data["lambda_range"]["lambda_max"]  # [um]
R_model     = get_R_instru(instru=instru)                # Model resolving power [-], capped by R0_max
wave_model  = get_wave_model(lmin=lmin_instru, lmax=lmax_instru, R=R_model)  # [um], constant dlambda

# Spectra at Earth: wavelengths [um], spectral energy flux densities
# F_lambda [W m^-2 um^-1], BEFORE Earth's atmosphere and the instrument.
# The planet is scaled using its physical properties and the catalogue
# phase function; the star is normalized to its K magnitude. A measured
# planet K magnitude is used when available under the package's prescription.
# Rotational broadening and radial velocities are already applied here.
planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(
    planet=planet,                      # Catalogue row (physical quantities with units)
    thermal_model=thermal_model,        # Thermal-emission model name
    reflected_model=reflected_model,    # Reflected-light model name
    instru=None,                        # Explicit model grid below, as in FastYield's worker
    wave_model=wave_model,              # Wavelength grid [um]
    show=True,                          # Plot stellar, thermal and reflected spectra
    in_planet_mag=True,                 # Apply available measured planet K normalization
    in_star_mag=True,                   # Normalize star to catalogue K magnitude
    apply_kinematics=True,              # Apply rotational broadening and Doppler shifts
)
print(f"Thermal model: {planet_thermal.model}; reflected model: {planet_reflected.model}")

# Recompute Vega magnitudes from the generated spectra, not from potentially
# outdated model-dependent magnitude columns in the catalogue.
# The reference band is the FULL nominal LMS wavelength interval, not one
# grating setting or a filter. No atmospheric/instrument transmission here.
# This reproduces FastYield's photon-integrated instrumental magnitudes.
band0 = "instru"
mask  = (wave_model >= lmin_instru) & (wave_model <= lmax_instru)
wave  = wave_model[mask]  # [um]
vega_spectrum = load_vega_spectrum().interpolate_wavelength(wave, renorm=False)
mag_star = get_mag(wave=wave, density_obs=star_spectrum.flux[mask], density_vega=vega_spectrum.flux)    # [mag]
mag_planet = get_mag(wave=wave, density_obs=planet_spectrum.flux[mask], density_vega=vega_spectrum.flux) # [mag]
if not np.all(np.isfinite([mag_star, mag_planet, separation_planet])):
    raise ValueError("Non-finite magnitude or separation; check the catalogue row and model spectra.")
print(f"LMS Vega magnitudes: star = {mag_star:.3f}, planet = {mag_planet:.3f}")


# %% 5. FastCurves: S/N curves for the selected catalogue planet

# Unlike example 2, use the actual predicted planet brightness and spectra.
# No model_star/model_planet, temperatures, gravities or velocities are needed
# here: they are already encoded in the two Spectrum objects.

name_bands_planet, separation_bands_planet, SNR_bands_planet = FastCurves(
    instru=instru,                       # Instrument identifier [-]
    exposure_time=exposure_time,         # Total integration time [min]
    apodizer="NO_SP",                    # No additional apodizer
    strehl="MED",                        # Available PSF label (HEEPS Q2, not a measured Strehl value)
    coronagraph="CVC",                   # Classical vortex coronagraph
    post_processing="molecular mapping", # Or "differential imaging"
    Rc=100,                              # High-pass cutoff resolving power lambda/dlambda [-]; None: no filter
    filter_type="gaussian_fast",         # Spectral filtering kernel (package default)
    background="medium",                 # Thermal background scenario: "low", "medium", "high"
    systematics=False,                   # No calibrated METIS systematic noise model yet
    speckles=False,                      # No calibrated METIS speckle noise model yet
    PCA=False,                           # No PCA subtraction / associated signal loss
    input_DIT=None,                      # Individual integration time [min]; None: automatic DIT
    planet_name=planet_name,             # Label for plots/prints
    mag_star=mag_star,                   # Apparent stellar Vega magnitude [mag]
    mag_planet=mag_planet,               # Apparent planet Vega magnitude [mag]
    band0=band0,                         # Reference interval for BOTH magnitudes
    star_spectrum=star_spectrum,         # Spectrum: um, W m^-2 um^-1
    planet_spectrum=planet_spectrum,     # Total thermal + reflected Spectrum, same units
    separation_planet=separation_planet, # [arcsec]
    calculation="SNR",                   # S/N of the selected planet
    band_only=band_only,                 # None for all 107 settings (many plots if enabled)
    show_plot=True,
    verbose=True,
    return_quantity=False,               # Standard return: setting IDs, separations, S/N curves
)
# separation_bands_planet: [mas]; SNR_bands_planet: dimensionless, for the
# full exposure_time (not per DIT). Each entry corresponds to one setting.

