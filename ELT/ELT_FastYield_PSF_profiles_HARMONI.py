# import FastYield modules
from fastyield.config import rad2arcsec, get_sim_data_path
from fastyield.utils import box_convolution, annular_mask, interp_extrap_wave, interp_extrap_sep, build_PSF_grid, crop, fill_invalid_by_nearest, fill_invalid_by_symmetry, rebin_flux_conserving, estimate_SR, plot_PSF, plot_profiles, add_if_necessary
from fastyield.get_specs import get_config_data

# import astropy modules
from astropy.io import fits

# import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import numpy modules
import numpy as np

# import scipy modules
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import logit, expit

data_path = "/home/martoss/Documents/PhD/work/MAIN/data"



#
# Parameters
#
instru        = "HARMONI"
strehl        = "JQ1"
bands         = ["J", "H", "K"]
apodizers     = ["NO_SP", "SP_Prox"]
coronagraph   = None
order_PSF     = 1 # 1 is the best choice
model_PSF     = "gaussian"
sigfactor     = None
debug         = False

# Retrieving specs
config_data = get_config_data(instru)
size_core   = config_data["size_core"]             # [px]
sep_unit    = config_data["sep_unit"]              # [mas]
FOV         = 800                                  # [mas] = FOV size in the simulated data
D           = config_data["telescope"]["diameter"] # [m]
WFE_ref     = (174.7 + 160.5) / 2                  # [nm]
IWA_ref     = 1.0                                  # [mas] Dummy value
 
# WFE, IWA, wavelength and separation axis
lmin       = 0.6   # [µm]
lmax       = 3.0   # [µm]
sep_max    = 1_000 # [mas]
sep_max_DL = 1_000 # [mas]
WFE_min    = 10    # [nm]
WFE_max    = 500   # [nm]
IWA_min    = 1.0   # [mas]
IWA_max    = 1.0   # [mas]

wave       = np.linspace(lmin,                lmax,              num=100)
WFE        = np.linspace(WFE_min,             WFE_max,           num=100)
IWA        = np.logspace(np.log10(IWA_min),   np.log10(IWA_max), num=100)
separation = np.logspace(np.log10(1),         np.log10(sep_max), num=1_000)

WFE        = add_if_necessary(array=WFE,        value=WFE_ref)
IWA        = add_if_necessary(array=IWA,        value=IWA_ref)
separation = add_if_necessary(array=separation, value=0.0)

r_WFE = WFE / WFE_ref
r_IWA = IWA / IWA_ref

N_apod    = len(apodizers)
N_wave    = len(wave)
N_sep     = len(separation)

# Scales
pxscale      = config_data["pxscale"] * 1000        # [mas/px]
R_max        = int(round(FOV/2/pxscale))            # [px] FoV radius (FoV/2 in pixels)
NbLine       = 2*R_max+1
NbColumn     = 2*R_max+1
N_sep_raw    = int(round(np.sqrt((NbLine/2)**2+(NbColumn/2)**2)))
y0, x0       = NbLine//2, NbColumn//2                      # Spatial center
r_core       = 1.029 * wave*1e-6 / D * rad2arcsec*1000 / 2 # FWHM radius [mas]

R_max_DL     = int(round(sep_max_DL/pxscale))     # [px] FoV radius (FoV/2 in pixels)
NbLine_DL    = 2*R_max_DL+1
NbColumn_DL  = 2*R_max_DL+1
N_sep_DL     = int(round(np.sqrt((NbLine_DL/2)**2+(NbColumn_DL/2)**2)))
y0_DL, x0_DL = NbLine_DL//2, NbColumn_DL//2                      # Spatial center



PSF_profile_density_DL_3D = np.zeros((N_apod, N_wave, N_sep)) # fraction of flux inside the px at the considered separation
fraction_core_DL_3D       = np.zeros((N_apod, N_wave, N_sep)) # fraction of flux inside the FWHM at the considered separation
aperture_correction_DL    = np.zeros((N_apod, N_wave))        # fraction of flux inside the FoV 
for ia, apodizer in enumerate(apodizers):
    print("\n\n\n"+"\033[4m"+f"{apodizer.replace('_', ' ')} apodizer:"+"\033[0m")



    # %%
    # Computing 2D DL profiles (wave, separation)
    #
    
    # --- Getting data ---
    pupil            = fits.getdata(f"{data_path}/{instru}/PSF_simulations/Pupil_{instru}_{apodizer}.fits")
    N_pupil          = pupil.shape[0]                      # Nb of points of the original pupil
    pxscale_pupil    = D / N_pupil                         # dx = D/N_pupil [m/px]
    zeropadding      = 5                                   # Nyquist sampling >= 2 (zeropadding pixels by lambda over D) (must be odd for accurate fraction core calculation)
    N_focal          = zeropadding * N_pupil               # Nb of points of the PSF (focal)
    pxscale_focal_lD = 1 / zeropadding                     # PSF focal [lambda/D/px] (zeropadding pixels by lambda over D): df = TF(dx) = 1 / (N_focal*dx) = 1 / (zeropadding * D). In optic, natural frequency unit = dtheta = lambda * df 
    
    # Pupil plot
    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(pupil, cmap='gray', origin='lower', extent=[-D/2, D/2, -D/2, D/2])
    plt.plot(0, 0, ms=100, marker="+", c="white")
    if apodizer == "NO_SP":
        plt.title(f"ELT/{instru} pupil without apodizer", fontsize=14)
    else:
        plt.title(f"ELT/{instru} pupil with {apodizer.replace('_', ' ')}-apodizer", fontsize=14)
    plt.xlabel("X offset [m]", fontsize=12)
    plt.ylabel("Y offset [m]", fontsize=12)
    plt.show()
    
    # DL PSF computing
    pupil_zeropadded                     = np.zeros((N_focal, N_focal))
    pupil_zeropadded[:N_pupil, :N_pupil] = pupil
    E                                    = np.fft.fftshift(np.fft.fft2(pupil_zeropadded))
    PSF_DL_2D                            = np.abs(E)**2
    PSF_DL_2D                            = crop(PSF_DL_2D) # crop to center the maximum
    PSF_DL_2D                            = fill_invalid_by_symmetry(PSF_DL_2D)
    PSF_DL_2D                            = fill_invalid_by_nearest(PSF_DL_2D)
    NbLine_raw, NbColumn_raw             = PSF_DL_2D.shape
    PSF_DL_2D                           /= np.nansum(PSF_DL_2D) # normalizing to the total flux

    # PSF_DL plot
    plt.figure(figsize=(8, 8), dpi=300)
    extent = [-pxscale_focal_lD*NbColumn_raw/2, pxscale_focal_lD*NbColumn_raw/2, -pxscale_focal_lD*NbLine_raw/2, pxscale_focal_lD*NbLine_raw/2]
    plt.imshow(PSF_DL_2D/np.nanmax(PSF_DL_2D), cmap="inferno", extent=extent, norm=mcolors.LogNorm(vmin=1e-6, vmax=1))
    plt.plot(0, 0, ms=50, marker="+", c="k")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    if apodizer == "NO_SP":
        plt.title(f"ELT/{instru} DL PSF \n without apodizer", fontsize=14)
    else:
        plt.title(f"ELT/{instru} DL PSF \n with {apodizer.replace('_', ' ')}-apodizer", fontsize=14)
    plt.xlabel(r'x offset [$\lambda / D$]',  fontsize=12)
    plt.ylabel(r'y offset [$\lambda / D$]',  fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('PSF [raw contrast]', fontsize=14, labelpad=20, rotation=270)
    circle = plt.Circle((0, 0), 1.22, color='red', fill=False, linewidth=1, label = r"$1.22 \lambda / D$ ")
    plt.gca().add_patch(circle) # The central obstruction reduce the PSF core (first zero < 1.22)
    plt.legend()
    plt.show()
    
    # --- Estimating apperture correction factor ---
    # The post-AO / coronagraphic PSF cubes are only available over a finite FoV,
    # so the true total flux is unknown. We therefore approximate the aperture
    # correction using the diffraction-limited PSF at the same wavelength, assuming
    # that the fraction of flux enclosed in the simulated FoV is close enough to the
    # DL case for normalization purposes.
    for i, l0 in enumerate(wave):
        l0D               = l0*1e-6 / D * 1000*rad2arcsec # lambda/D [mas/(lambda/D)]
        pxscale_focal_mas = pxscale_focal_lD * l0D        # Focal pxscale [mas/px]
        
        # Cropping in order to have right FOV
        R_FOV         = int(round(FOV/2/pxscale_focal_mas))
        PSF_DL_2D_FOV = crop(PSF_DL_2D, Y0=PSF_DL_2D.shape[0]//2, X0=PSF_DL_2D.shape[1]//2, R_crop=R_FOV)
        
        # Estimating the apertuce correction factor (total flux in FOV / total flux)
        aperture_correction_DL[ia, i] = np.nansum(PSF_DL_2D_FOV)
    
    # --- Rebinning and retrieving the PSF for each wavelength ---
    PSF_DL_3D = np.zeros((N_wave, NbLine_DL, NbColumn_DL))
    for i, l0 in enumerate(wave):
        l0D               = l0*1e-6 / D * 1000*rad2arcsec # lambda/D [mas/(lambda/D)]
        pxscale_focal_mas = pxscale_focal_lD * l0D        # Focal pxscale [mas/px]
        
        # Cropping to the FoV (in order to make computation faster)
        R_sep_max    = int(round(sep_max_DL/pxscale_focal_mas))
        PSF_DL_2D_l0 = crop(PSF_DL_2D, Y0=PSF_DL_2D.shape[0]//2, X0=PSF_DL_2D.shape[1]//2, R_crop=R_sep_max)
        PSF_DL_2D_l0 = fill_invalid_by_symmetry(PSF_DL_2D_l0)
        PSF_DL_2D_l0 = fill_invalid_by_nearest(PSF_DL_2D_l0)
        
        # Rebinning the PSF at the band pxscale
        PSF_DL_2D_l0 = rebin_flux_conserving(data=PSF_DL_2D_l0, pxscale_in=pxscale_focal_mas, pxscale_out=pxscale)
        
        # Cropping in order to have same PSF sizes
        PSF_DL_2D_l0 = crop(PSF_DL_2D_l0, Y0=PSF_DL_2D_l0.shape[0]//2, X0=PSF_DL_2D_l0.shape[1]//2, R_crop=R_max_DL)
        
        # Saving the array
        PSF_DL_3D[i] = PSF_DL_2D_l0



    # --- Computing PSF profiles and fraction core (already normalized) ---
        
    # Optional local NaN-aware box average (size_core x size_core)
    if size_core > 1:
        PSF_DL_3D_conv = box_convolution(data=PSF_DL_3D, size_core=size_core, mode="mean")
    else:
        PSF_DL_3D_conv = np.copy(PSF_DL_3D)
    
    # PSF radial profile using 1-pixel-wide annuli out to N rings
    separation_DL     = np.zeros((N_sep_DL))
    PSF_profile_DL_2D = np.zeros((N_wave, N_sep_DL))
    for r in range(N_sep_DL):
        r_int                   = max(1, r - 1) if r > 1 else r
        r_ext                   = r
        separation_DL[r]        = (r_int + r_ext)/2 * pxscale
        amask                   = (annular_mask(r_int, r_ext, size=(NbLine_DL, NbColumn_DL)) == 1)
        PSF_profile_DL_2D[:, r] = np.nanmean(PSF_DL_3D_conv[:, amask], axis=1)
    PSF_profile_density_DL_2D = PSF_profile_DL_2D / pxscale**2
    
    # Fraction in the "core" (approximate central pixel block): box_convolution("mean") => PSF[y0, x0] = sum PSF(i, j) / A_FWHM (with (i, j) in FWHM and A_FWHM = size_core**2)
    fraction_core_DL_1D = size_core**2 * PSF_DL_3D_conv[:, y0_DL, x0_DL] # (wave)
    fraction_core_DL_2D = fraction_core_DL_1D[:, None] * np.ones((1, len(separation_DL)))
    


    # --- INTERPOLATION AND EXTRAPOLATION ALONG SEPARATION ---
    PSF_profile_density_DL_2D_new = np.zeros((N_wave, N_sep))
    fraction_core_DL_2D_new       = np.zeros((N_wave, N_sep))
    for iw in range(N_wave):
        PSF_profile_density_DL_2D_new[iw] = interp_extrap_sep(separation_ref=separation_DL, separation_new=separation, y_ref=PSF_profile_density_DL_2D[iw], mode="log",   tail_model="powerlaw")
        fraction_core_DL_2D_new[iw]       = interp_extrap_sep(separation_ref=separation_DL, separation_new=separation, y_ref=fraction_core_DL_2D[iw],       mode="logit", tail_model="flat")
    PSF_profile_density_DL_2D = PSF_profile_density_DL_2D_new
    fraction_core_DL_2D       = fraction_core_DL_2D_new
    
    
    
    # --- Plots ---
    
    # PSF
    plot_PSF(instru=instru, coronagraph=None, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, wave=wave, PSF_3D=PSF_DL_3D, type_PSF="DL", model_PSF=model_PSF, sigfactor=sigfactor, debug=debug)
    
    # PROFILES
    plot_profiles(instru=instru, coronagraph=None, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=None, separation=separation, PSF_profile_density_2D=PSF_profile_density_DL_2D, fraction_core_2D=fraction_core_DL_2D, radial_transmission_2D=None, type_PSF="DL", title_suffix="\n")
    

    
    # --- Saving arrays ---
    PSF_profile_density_DL_3D[ia] = PSF_profile_density_DL_2D
    fraction_core_DL_3D[ia]       = fraction_core_DL_2D

    

    # %%
    # Computing 2D post AO profiles (wave, separation)
    #
    
    # --- Getting data ---
    PSF_3D_raw = None
    wave_raw   = None
    for band in bands:
        cube                                       = fits.getdata(f"{data_path}/{instru}/PSF_simulations/PSF_3D_{band}_{strehl}_{apodizer}.fits")
        NbChannel_data, NbLine_data, NbColumn_data = cube.shape
        lmin_data                                  = config_data['gratings'][band].lmin 
        lmax_data                                  = config_data['gratings'][band].lmax
        if band == "J":
            lmin_data = 1.2
            lmax_data = 1.324 # en µm infos fournies par Alexis par mail le 17 juin 2024
        wave_data = np.linspace(lmin_data, lmax_data, NbChannel_data)
        if PSF_3D_raw is None:
            PSF_3D_raw = cube
            wave_raw   = wave_data
        else:
            PSF_3D_raw = np.concatenate((PSF_3D_raw,  cube),   axis=0)
            wave_raw   = np.concatenate((wave_raw, wave_data), axis=0)
    N_wave_raw, NbLine_raw, NbColumn_raw = PSF_3D_raw.shape
    pxscale_raw                          = 800/NbLine_raw # hard coded 
    
    

    # --- Estimating the SR and the WFE (with the DL profiles) ---
    mask_wave           = (wave >= wave_raw[0]) & (wave <= wave_raw[-1])
    PSF_3D              = interp1d(wave_raw, PSF_3D_raw, axis=0, bounds_error=True, assume_sorted=True)(wave[mask_wave])
    aperture_correction = interp1d(wave, aperture_correction_DL[ia], bounds_error=True)(wave[mask_wave])
    SR, WFE_nm          = estimate_SR(instru=instru, apodizer=apodizer, strehl=strehl, PSF_3D=PSF_3D, PSF_DL_2D_lD=PSF_DL_2D, pxscale_data=pxscale_raw, pxscale_DL_lD=pxscale_focal_lD, FOV=FOV, D=D, wave=wave[mask_wave], aperture_correction=aperture_correction, wave_output=wave)


    
    # --- Rebinning data ---
    
    # Rebinning the PSF_3D
    PSF_3D = rebin_flux_conserving(data=PSF_3D_raw, pxscale_in=pxscale_raw, pxscale_out=pxscale)
    
    # Cropping in order to have same PSF sizes
    PSF_3D = crop(PSF_3D, Y0=PSF_3D.shape[1]//2, X0=PSF_3D.shape[2]//2, R_crop=R_max)
    
    # Normalizing with total flux
    aperture_correction_raw = interp1d(wave, aperture_correction_DL[ia], bounds_error=True)(wave_raw)
    PSF_3D                 /= np.nansum(PSF_3D, axis=(1, 2))[:, None, None] / aperture_correction_raw[:, None, None]
    
    
    
    # --- Computing PSF_3D profiles and fraction core ---
    
    # Optional local NaN-aware box average (size_core x size_core)
    if size_core > 1:
        PSF_3D_conv = box_convolution(data=PSF_3D, size_core=size_core, mode="mean")
    else:
        PSF_3D_conv = np.copy(PSF_3D)
    
    # PSF_3D radial profile using 1-pixel-wide annuli out to N rings
    separation_raw = np.zeros((N_sep_raw))
    PSF_profile_2D = np.zeros((N_wave_raw, N_sep_raw))
    for r in range(N_sep_raw):
        r_int                = max(1, r - 1) if r > 1 else r
        r_ext                = r
        separation_raw[r]    = (r_int + r_ext)/2 * pxscale
        amask                = (annular_mask(r_int, r_ext, size=(NbLine, NbColumn)) == 1)
        PSF_profile_2D[:, r] = np.nanmean(PSF_3D_conv[:, amask], axis=1)
    PSF_profile_density_2D = PSF_profile_2D / pxscale**2
    
    # Fraction in the "core" (approximate central pixel block): box_convolution("mean") => PSF_3D[y0, x0] = sum PSF_3D(i, j) / A_FWHM (with (i, j) in FWHM and A_FWHM = size_core**2)
    fraction_core_1D = size_core**2 * PSF_3D_conv[:, y0, x0] # (wave_raw)
    fraction_core_2D = fraction_core_1D[:, None] * np.ones((1, len(separation_raw)))
    
    
    
    # --- INTERPOLATION AND EXTRAPOLATION ALONG SEPARATION ---
    PSF_profile_density_2D_new = np.zeros((N_wave_raw, N_sep))
    fraction_core_2D_new       = np.zeros((N_wave_raw, N_sep))
    for iw in range(N_wave_raw):
        PSF_profile_density_2D_new[iw] = interp_extrap_sep(separation_ref=separation_raw, separation_new=separation, y_ref=PSF_profile_density_2D[iw], mode="log",   tail_model="powerlaw")
        fraction_core_2D_new[iw]       = interp_extrap_sep(separation_ref=separation_raw, separation_new=separation, y_ref=fraction_core_2D[iw],       mode="logit", tail_model="flat")
    PSF_profile_density_2D = PSF_profile_density_2D_new
    fraction_core_2D       = fraction_core_2D_new
        
    
    
    # --- INTERPOLATION AND EXTRAPOLATION ALONG WAVELENGTH ---
    PSF_profile_density_2D_new, _, _ = interp_extrap_wave(wave_ref=wave_raw, wave_new=wave, separation=separation, PSF_profile_density_2D=PSF_profile_density_2D, fraction_core_2D=fraction_core_2D, radial_transmission_2D=None)
    # Because here fraction core does not depend with the separation (w/o coronagraph)
    fraction_core_2D_new             = expit(interp1d(wave_raw, logit(fraction_core_2D), bounds_error=False, axis=0, assume_sorted=True, fill_value="extrapolate")(wave))
    PSF_profile_density_2D = PSF_profile_density_2D_new
    fraction_core_2D       = fraction_core_2D_new
        
    
    
    # --- Post-AO fractions core should always be <= to the DL fractions core --- 
    fraction_core_2D = np.minimum(fraction_core_2D, fraction_core_DL_2D)
    
    
    
    # --- Giving the post-AO profiles extrapolation the same shapes of DL profiles ---
    mask_tail = separation >= separation_raw[-1]
    if coronagraph is None:
        scale                                = (PSF_profile_density_2D[:, ~mask_tail][:, -1] / PSF_profile_density_DL_2D[:, mask_tail][:, 0])
        PSF_profile_density_2D[:, mask_tail] = PSF_profile_density_DL_2D[:, mask_tail] * scale[:, None]
    # Sanity check plots
    if debug:
        for iw in range(N_wave):
            I_speck = PSF_profile_density_2D[iw] - SR[iw]*PSF_profile_density_DL_2D[iw]
            plt.figure(dpi=300)
            plt.plot(separation, PSF_profile_density_2D[iw],              c="black", ls="-",  label="I_PSF")
            plt.plot(separation, SR[iw]*PSF_profile_density_DL_2D[iw], c="black", ls="--", label="SR*I_DL")
            plt.plot(separation, I_speck,                                    c="black", ls=":",  label="I_speck = I_PSF - SR*I_DL")
            plt.legend()
            plt.title(f"wave = {wave[iw]:.2f} µm")
            plt.xlabel("Separation [mas]")
            plt.ylabel("PSF")
            plt.yscale('log')
            plt.xscale('log')
            plt.show()
    
    
    
    # --- Plots ---
    
    # PSF
    plot_PSF(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, wave=wave_raw, PSF_3D=PSF_3D, type_PSF="post-AO", model_PSF=model_PSF, sigfactor=sigfactor, debug=debug)
    
    # PROFILES
    plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=wave_raw, separation=separation, PSF_profile_density_2D=PSF_profile_density_2D, fraction_core_2D=fraction_core_2D, radial_transmission_2D=None, type_PSF="post-AO", title_suffix="\n")
    

    
    # %%
    # Extrapolating 2D post-AO profile to 4D (wave, WFE, IWA, separation)
    #
        
    # --- VARYING AO PERFORMANCE (r_WFE) AND CORONAGRAPH IWA (r_IWA) (wave, r_WFE, r_IWA, separation) ---
    PSF_profile_density_4D, fraction_core_4D, _, _, _ = build_PSF_grid(SR=SR, D=D, wave=wave, r_WFE=r_WFE, r_IWA=r_IWA, separation=separation, PSF_profile_density_no_coro_2D=PSF_profile_density_2D, fraction_core_no_coro_2D=fraction_core_2D, PSF_profile_density_DL_2D=PSF_profile_density_DL_2D, fraction_core_DL_2D=fraction_core_DL_2D, coronagraph=coronagraph, IWA_ref=None, PSF_profile_density_coro_2D=None, fraction_core_coro_2D=None, radial_transmission_2D=None)
    
    
    
    # --- Forcing nominal equality ---
    
    # On PSF_profile_density
    PSF_profile_density_OG_2D = np.copy(PSF_profile_density_2D)
    PSF_profile_density_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), PSF_profile_density_4D, bounds_error=True, fill_value=np.nan)(np.stack((np.meshgrid(wave, WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1))[:, 0, 0, :]
    corr_factor_PSF           = PSF_profile_density_OG_2D / PSF_profile_density_EX_2D # (wave, separation)
    PSF_profile_density_4D   *= corr_factor_PSF[:, None, None, :]
    
    # On faction_core
    fraction_core_OG_2D = np.copy(fraction_core_2D)
    fraction_core_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), fraction_core_4D, bounds_error=True, fill_value=np.nan)(np.stack((np.meshgrid(wave, WFE_ref, IWA_ref, separation, indexing="ij")), axis=-1))[:, 0, 0, :]
    corr_factor_FC      = fraction_core_OG_2D / fraction_core_EX_2D # (wave, separation)
    fraction_core_4D   *= corr_factor_FC[:, None, None, :]
    
    
    
    # --- Empirical sanity check ---
    fraction_core_3D_from_PSF = PSF_profile_density_4D[:, :, :, 0]*pxscale**2*size_core**2
    fraction_core_4D          = np.minimum(fraction_core_4D, fraction_core_3D_from_PSF[:, :, :, None])
    
    
    
    # --- PLOT ---
            
    idx_IWA = np.abs(r_IWA - 1).argmin() # at r_IWA = 1.0
    #for idx_WFE in range(len(r_WFE)):
    for idx_WFE in [0, np.abs(r_WFE - 1).argmin(), -1]:
        psf = PSF_profile_density_4D[:, idx_WFE, idx_IWA]
        fc  = fraction_core_4D[:, idx_WFE, idx_IWA]
        rt  = None
        plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=wave_raw, separation=separation, PSF_profile_density_2D=psf, fraction_core_2D=fc, radial_transmission_2D=rt, type_PSF="post-AO", title_suffix=f" \n r_WFE = {r_WFE[idx_WFE]:.1f} & r_IWA = {r_IWA[idx_IWA]:.1f}")
    
    if coronagraph is not None:
        idx_WFE = np.abs(r_WFE - 1).argmin() # at r_WFE = 1.0
        #for idx_IWA in range(len(r_IWA)):
        for idx_IWA in [0, np.abs(r_IWA - 1).argmin(), -1]:
            psf = PSF_profile_density_4D[:, idx_WFE, idx_IWA]
            fc  = fraction_core_4D[:, idx_WFE, idx_IWA]
            rt  = None
            plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=wave_raw, separation=separation, PSF_profile_density_2D=psf, fraction_core_2D=fc, radial_transmission_2D=rt, type_PSF="post-AO", title_suffix=f" \n r_WFE = {r_WFE[idx_WFE]:.1f} & r_IWA = {r_IWA[idx_IWA]:.1f}")


    # --- SAVING ---
    suffix_PSF = f"{apodizer}_{coronagraph}_{strehl}_{pxscale}_{lmin}_{lmax}_{WFE_min}_{WFE_max}_{IWA_min}_{IWA_max}_{sep_max}"
    output_dir = get_sim_data_path() / "ELT" / instru / "PSF_inputs"
    fits.writeto(f"{output_dir}/{instru}_WFE_{suffix_PSF}.fits",                    WFE,                    overwrite=True)
    fits.writeto(f"{output_dir}/{instru}_IWA_{suffix_PSF}.fits",                    IWA,                    overwrite=True)
    fits.writeto(f"{output_dir}/{instru}_wave_{suffix_PSF}.fits",                   wave,                   overwrite=True)
    fits.writeto(f"{output_dir}/{instru}_separation_{suffix_PSF}.fits",             separation,             overwrite=True)
    fits.writeto(f"{output_dir}/{instru}_PSF_profile_density_4D_{suffix_PSF}.fits", PSF_profile_density_4D, overwrite=True)
    fits.writeto(f"{output_dir}/{instru}_fraction_core_4D_{suffix_PSF}.fits",       fraction_core_4D,       overwrite=True)
    


