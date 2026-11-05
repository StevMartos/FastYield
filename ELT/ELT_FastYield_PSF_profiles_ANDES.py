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
from scipy.ndimage import shift

# import other modules
from tqdm import tqdm

data_path = "/home/martoss/Documents/PhD/work/MAIN/data"



#
# Parameters
#
instru        = "ANDES"
strehl        = "MED"
bands         = ["YJH_5mas_100", "YJH_10mas_100", "YJH_16mas_100"]
apodizer      = "NO_SP"
coronagraphs  = [None, "LYOT"] 
order_PSF     = 1 # 1 is the best choice
model_PSF     = "gaussian"
sigfactor     = None
debug         = False



# Retrieving specs
config_data = get_config_data(instru)
size_core   = config_data["size_core"]             # [px]
sep_unit    = config_data["sep_unit"]              # [mas]
FOV         = 125                                  # [mas] = FOV size in the simulated data
D           = config_data["telescope"]["diameter"] # [m]
WFE_ref     = 212.2                                # [nm]
IWA_ref     = 33.4/2                               # [mas] ANDES Lyot FPM radius (cf. plot de Mamadou N'Diaye)


# WFE, IWA, wavelength and separation axis
lmin       = 0.6   # [µm]
lmax       = 3.0   # [µm]
sep_max    = 1_000 # [mas]
sep_max_DL = 1_000 # [mas]
WFE_min    = 10    # [nm] RMS
WFE_max    = 500   # [nm] RMS
IWA_min    = 1e-1  # [mas]
IWA_max    = 100   # [mas]

wave       = np.linspace(lmin,              lmax,              num=50)
WFE        = np.linspace(WFE_min,           WFE_max,           num=50)
IWA        = np.logspace(np.log10(IWA_min), np.log10(IWA_max), num=50)
separation = np.logspace(np.log10(1),       np.log10(sep_max), num=1_000)

WFE        = add_if_necessary(array=WFE,        value=WFE_ref)
IWA        = add_if_necessary(array=IWA,        value=IWA_ref)
separation = add_if_necessary(array=separation, value=0.0)

r_WFE = WFE / WFE_ref
r_IWA = IWA / IWA_ref

N_band = len(bands)
N_coro = len(coronagraphs)
N_wave = len(wave)
N_sep  = len(separation)
N_WFE  = len(r_WFE)
N_IWA  = len(r_IWA)




# %%
# Computing 2D DL PSF in [lambda/D]
#

print("\n\n\n"+"\033[4m"+"DL PSF without coronagraph:"+"\033[0m")

# --- Getting data ---
pupil            = fits.getdata(f"{data_path}/{instru}/PSF_simulations/Pupil_{instru}_{apodizer}.fits")
N_pupil          = pupil.shape[0]        # Nb of points of the original pupil
pxscale_pupil    = D / N_pupil           # dx = D/N_pupil [m/px]
zeropadding      = 5                     # Nyquist sampling >= 2 (zeropadding pixels by lambda over D) (must be odd for accurate fraction core calculation)
N_focal          = zeropadding * N_pupil # Nb of points of the PSF (focal)
pxscale_focal_lD = 1 / zeropadding       # PSF focal [lambda/D/px] (zeropadding pixels by lambda over D): df = TF(dx) = 1 / (N_focal*dx) = 1 / (zeropadding * D). In optic, natural frequency unit = dtheta = lambda * df 

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
PSF_DL_2D                           /= np.nansum(PSF_DL_2D)

# PSF_DL plot
plt.figure(figsize=(8, 8), dpi=300)
extent = [-pxscale_focal_lD*NbColumn_raw/2, pxscale_focal_lD*NbColumn_raw/2, -pxscale_focal_lD*NbLine_raw/2, pxscale_focal_lD*NbLine_raw/2]
plt.imshow(PSF_DL_2D/np.nanmax(PSF_DL_2D), cmap="inferno", extent=extent, norm=mcolors.LogNorm(vmin=1e-6, vmax=1))
plt.plot(0, 0, ms=50, marker="+", c="k")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title(f"ELT/{instru} DL PSF", fontsize=14)
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
aperture_correction_DL = np.zeros((len(wave)))
for i, l0 in enumerate(wave):
    l0D               = l0*1e-6 / D * 1000*rad2arcsec # lambda/D [mas/(lambda/D)]
    pxscale_focal_mas = pxscale_focal_lD * l0D        # Focal pxscale [mas/px]
    
    # Cropping in order to have right FOV
    R_FOV         = int(round(FOV/2/pxscale_focal_mas))
    PSF_DL_2D_FOV = crop(PSF_DL_2D, Y0=PSF_DL_2D.shape[0]//2, X0=PSF_DL_2D.shape[1]//2, R_crop=R_FOV)
    
    # Estimating the apertuce correction factor (total flux in FOV / total flux)
    aperture_correction_DL[i] = np.nansum(PSF_DL_2D_FOV)



# %%
# Computing 2D DL profiles (band, wave, separation) for each band/pxscale
#

PSF_profile_density_DL_3D = np.zeros((N_band, N_wave, N_sep)) # fraction of flux inside the px at the considered separation
fraction_core_DL_3D       = np.zeros((N_band, N_wave, N_sep)) # fraction of flux inside the FWHM at the considered separation
    
for ib, band in enumerate(bands):
    lmin_band = config_data['gratings'][band].lmin 
    lmax_band = config_data['gratings'][band].lmax
    R_band    = config_data['gratings'][band].R
    FWHM_band = 1.029*((lmin_band+lmax_band)*1e-6/2)/D * rad2arcsec*1000 # [mas]
    print("\n"+" \033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(lmin_band, 2)} to {round(lmax_band, 2)} µm with R={R_band:.0f}):"+"\033[0m")

    # Scales
    pxscale      = config_data["pxscale"][band] * 1000 # [mas/px]
    R_max_DL     = int(round(sep_max_DL/pxscale))     # [px] FoV radius (FoV/2 in pixels)
    NbLine_DL    = 2*R_max_DL+1
    NbColumn_DL  = 2*R_max_DL+1
    N_sep_DL     = int(round(np.sqrt((NbLine_DL/2)**2+(NbColumn_DL/2)**2)))
    y0_DL, x0_DL = NbLine_DL//2, NbColumn_DL//2                      # Spatial center
    
    # --- Rebinning and retrieving the PSF for each wavelength ---
    PSF_DL_3D = np.zeros((N_wave, NbLine_DL, NbColumn_DL))
    for i, l0 in enumerate(wave):
        l0D               = l0*1e-6 / D * 1000*rad2arcsec # lambda/D [mas/(lambda/D)]
        pxscale_focal_mas = pxscale_focal_lD * l0D        # Focal pxscale [mas/px]
        
        # Cropping to the FoV (in order to make computation faster)
        R_sep_max   = int(round(sep_max_DL/pxscale_focal_mas))
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
    PSF_profile_density_DL_3D[ib] = PSF_profile_density_DL_2D
    fraction_core_DL_3D[ib]       = fraction_core_DL_2D


    
# %%
# Computing 4D post-AO profiles (wave, WFE, IWA, separation)
#

PSF_profile_density_no_coro_3D = np.zeros((N_band, N_wave, N_sep))
fraction_core_no_coro_3D       = np.zeros((N_band, N_wave, N_sep))
PSF_profile_density_coro_3D    = np.zeros((N_band, N_wave, N_sep))
fraction_core_coro_3D          = np.zeros((N_band, N_wave, N_sep))
radial_transmission_3D         = np.zeros((N_band, N_wave, N_sep))
wave_raw_2D                    = {}

for coronagraph in coronagraphs:
    print("\n\n\n"+"\033[4m"+f"Post-AO PSF with {str(coronagraph).replace('_', ' ')} coronagraph:"+"\033[0m")
    
    # Opening data
    suffix_data = f"{coronagraph}_{strehl}_{apodizer}"
    if coronagraph is not None:
        max_trans                                       = 0.756  # cf plot de Mamadou N'Diaye
        wave_raw                                        = np.array([1000, 1200, 1400, 1600, 1800]) * 1e-3 # µm
        PSF_4D_raw                                      = fits.getdata(f"{data_path}/ANDES/PSF_simulations/PSF_4D_YJH_{suffix_data}.fits")
        N_wave_raw, NbOff_raw, NbLine_raw, NbColumn_raw = PSF_4D_raw.shape
        offset_raw                                      = np.linspace(0, 62.5, NbOff_raw) # mas
        pxscale_raw                                     = 130 / NbLine_raw # [mas/px]
        PSF_3D_raw                                      = np.copy(PSF_4D_raw[:, 0]) # Star PSF given at sep = 0
        # Computing the radial transmission
        cmap_wave = plt.get_cmap("rainbow", N_wave_raw)
        plt.figure(dpi=300, figsize=(10, 6))
        radial_transmission_2D_raw = np.zeros((N_wave_raw, NbOff_raw))
        for iw in range(N_wave_raw):
            # Calcul de la transmission radiale
            radial_transmission_2D_raw[iw]  = np.nansum(PSF_4D_raw[iw], axis=(1, 2))
            radial_transmission_2D_raw[iw] *= max_trans / np.nanmax(radial_transmission_2D_raw[iw])
            # Écrêtage à partir du maximum
            idx_max                                                             = np.argmax(radial_transmission_2D_raw[iw])
            radial_transmission_2D_raw[iw][offset_raw > offset_raw[idx_max]] = np.nanmax(radial_transmission_2D_raw[iw])
            plt.plot(offset_raw, 100*radial_transmission_2D_raw[iw], color=cmap_wave(iw), linewidth=2, label=f"{wave_raw[iw]:.1f} µm")
        plt.xlabel(f"Separation [{sep_unit}]", fontsize=14, labelpad=10)
        plt.ylabel("Transmission [%]", fontsize=14, labelpad=10)
        plt.title(f"Radial Transmission Profile – {instru} with coronagraph", fontsize=16, fontweight='bold', pad=15)
        plt.legend(title="Wavelength", fontsize=11, title_fontsize=12, loc="upper left", frameon=True, edgecolor="gray", facecolor="whitesmoke")
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.xlim(offset_raw[0], offset_raw[-1])
        plt.ylim(0, 100)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', length=4, color='gray')
        plt.tight_layout()
        plt.show()
        
    
    else:
        PSF_3D_raw  = fits.getdata(f"{data_path}/ANDES/PSF_simulations/PSF_3D_YJH_{suffix_data}.fits")
        pxscale_raw = fits.getheader(f"{data_path}/ANDES/PSF_simulations/PSF_3D_YJH_{suffix_data}.fits")["pxscale"]
        wave_raw    = fits.getdata(f"{data_path}/ANDES/PSF_simulations/wave_data_YJH_{suffix_data}.fits")
        N_wave_raw, NbLine_raw, NbColumn_raw = PSF_3D_raw.shape
        # No radial transsmision (w/o coronagraph)
        radial_transmission_2D_raw = None
        
        # --- Estimating the SR and the WFE (with the DL profiles) ---
        aperture_correction_raw = interp1d(wave, aperture_correction_DL, bounds_error=True)(wave_raw)
        SR, WFE_nm              = estimate_SR(instru=instru, apodizer=apodizer, strehl=strehl, PSF_3D=PSF_3D_raw, PSF_DL_2D_lD=PSF_DL_2D, pxscale_data=pxscale_raw, pxscale_DL_lD=pxscale_focal_lD, FOV=FOV, D=D, wave=wave_raw, aperture_correction=aperture_correction_raw, wave_output=wave)
        


    # --- For each band / pxscale
    for ib, band in enumerate(bands):
        lmin_band = config_data['gratings'][band].lmin 
        lmax_band = config_data['gratings'][band].lmax
        R_band    = config_data['gratings'][band].R
        FWHM_band = 1.028*((lmin_band+lmax_band)*1e-6/2)/D * rad2arcsec * 1000 # [mas]
        print("\n"+" \033[4m"+f"{band.replace('_', ' ')}-BAND (from {round(lmin_band, 2)} to {round(lmax_band, 2)} µm with R={R_band:.0f}):"+"\033[0m")

        # --- Scales ---
        pxscale   = config_data["pxscale"][band] * 1000 # [mas/px]
        R_max     = int(round(FOV/2/pxscale))           # [px] FoV radius (FoV/2 in pixels)
        NbLine    = 2*R_max+1
        NbColumn  = 2*R_max+1
        y0, x0    = NbLine//2, NbColumn//2
        N_sep_raw = int(round(np.sqrt((NbLine/2)**2+(NbColumn/2)**2)))

        PSF_profile_density_DL_2D = PSF_profile_density_DL_3D[ib]
        fraction_core_DL_2D       = fraction_core_DL_3D[ib]
        radial_transmission_2D    = radial_transmission_2D_raw
        
        
        
        # --- Rebinning data ---

        # Rebinning the PSF_3D
        PSF_3D = rebin_flux_conserving(data=PSF_3D_raw, pxscale_in=pxscale_raw, pxscale_out=pxscale)
        
        # Cropping in order to have same PSF sizes
        PSF_3D = crop(PSF_3D, Y0=PSF_3D.shape[1]//2, X0=PSF_3D.shape[2]//2, R_crop=R_max)
        
        # Normalizing with total flux
        aperture_correction_raw = interp1d(wave, aperture_correction_DL, bounds_error=True)(wave_raw)
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
        
        
        
        # --- WITH CORONAGRAPH: computing fraction core for each offset ---
        if coronagraph is not None:
                        
            fraction_core_2D = np.zeros((N_wave_raw, NbOff_raw))
            for ioff in tqdm(range(NbOff_raw), desc=f"{band}-band"):

                PSF_3D_off  = np.copy(PSF_4D_raw[:, ioff])
                
                # Cancellation of the offset (shifting)
                dy_shift   = 0
                dx_shift   = - offset_raw[ioff] / pxscale_raw # separation shift
                PSF_3D_off = shift(PSF_3D_off, shift=(0, dy_shift, dx_shift), order=order_PSF, mode="constant", cval=np.nan)
                PSF_3D_off = fill_invalid_by_symmetry(PSF_3D_off)
                PSF_3D_off = fill_invalid_by_nearest(PSF_3D_off)
                
                # Rebinning the PSF_3D
                PSF_3D_off = rebin_flux_conserving(data=PSF_3D_off, pxscale_in=pxscale_raw, pxscale_out=pxscale)
                
                # Cropping in order to have same PSF sizes
                PSF_3D_off = crop(PSF_3D_off, Y0=PSF_3D_off.shape[1]//2, X0=PSF_3D_off.shape[2]//2, R_crop=R_max)
                
                # Normalizing with total flux
                PSF_3D_off /= np.nansum(PSF_3D_off, axis=(1, 2))[:, None, None] / aperture_correction_raw[:, None, None]
                
                # Optional local NaN-aware box average (size_core x size_core)
                if size_core > 1:
                    PSF_3D_off = box_convolution(data=PSF_3D_off, size_core=size_core, mode="mean")
                else:
                    PSF_3D_off = np.copy(PSF_3D_off)
                
                # Fraction in the "core" (approximate central pixel block): box_convolution("mean") => PSF_3D[y0, x0] = sum PSF_3D(i, j) / A_FWHM (with (i, j) in FWHM and A_FWHM = size_core**2)
                fraction_core_2D[:, ioff] = size_core**2 * PSF_3D_off[:, y0, x0] # (wave_raw)
                
                if ioff == 0:
                    if not np.allclose(fraction_core_1D, fraction_core_2D[:, ioff], rtol=1e-6, atol=1e-10):
                        raise ValueError("This should not happen!")
            
            # Interpolating over separation_raw
            fraction_core_2D_new       = np.zeros((N_wave_raw, N_sep_raw))
            radial_transmission_2D_new = np.zeros((N_wave_raw, N_sep_raw))
            for iw in range(N_wave_raw):
                fraction_core_2D_new[iw]       = interp_extrap_sep(separation_ref=offset_raw, separation_new=separation_raw, y_ref=fraction_core_2D[iw],       mode="logit", tail_model="flat")
                radial_transmission_2D_new[iw] = interp_extrap_sep(separation_ref=offset_raw, separation_new=separation_raw, y_ref=radial_transmission_2D[iw], mode="logit", tail_model="flat")
            fraction_core_2D       = fraction_core_2D_new
            radial_transmission_2D = radial_transmission_2D_new
        
        
        
        # --- WITHOUT CORONAGRAPH: fraction core does not depend with the separation (wave, separation) ---
        else:
            fraction_core_2D = fraction_core_1D[:, None] * np.ones((1, len(separation_raw)))
        
        
        
        # --- INTERPOLATION AND EXTRAPOLATION ALONG SEPARATION ---
        PSF_profile_density_2D_new = np.zeros((N_wave_raw, N_sep))
        fraction_core_2D_new       = np.zeros((N_wave_raw, N_sep))
        if coronagraph is not None:
            radial_transmission_2D_new = np.zeros((N_wave_raw, N_sep))
        else:
            radial_transmission_2D_new = None
        for iw in range(N_wave_raw):
            PSF_profile_density_2D_new[iw]     = interp_extrap_sep(separation_ref=separation_raw, separation_new=separation, y_ref=PSF_profile_density_2D[iw], mode="log",   tail_model="powerlaw")
            fraction_core_2D_new[iw]           = interp_extrap_sep(separation_ref=separation_raw, separation_new=separation, y_ref=fraction_core_2D[iw],       mode="logit", tail_model="flat")
            if coronagraph is not None:
                radial_transmission_2D_new[iw] = interp_extrap_sep(separation_ref=separation_raw, separation_new=separation, y_ref=radial_transmission_2D[iw], mode="logit", tail_model="flat")
        PSF_profile_density_2D = PSF_profile_density_2D_new
        fraction_core_2D       = fraction_core_2D_new
        radial_transmission_2D = radial_transmission_2D_new
        
        
        
        # --- INTERPOLATION AND EXTRAPOLATION ALONG WAVELENGTH ---
        PSF_profile_density_2D_new, fraction_core_2D_new, radial_transmission_2D_new = interp_extrap_wave(wave_ref=wave_raw, wave_new=wave, separation=separation, PSF_profile_density_2D=PSF_profile_density_2D, fraction_core_2D=fraction_core_2D, radial_transmission_2D=radial_transmission_2D)
        if coronagraph is None: # Because here fraction core does not depend with the separation (w/o coronagraph)
            fraction_core_2D_new = expit(interp1d(wave_raw, logit(fraction_core_2D), bounds_error=False, axis=0, assume_sorted=True, fill_value="extrapolate")(wave))
        PSF_profile_density_2D = PSF_profile_density_2D_new
        fraction_core_2D       = fraction_core_2D_new
        radial_transmission_2D = radial_transmission_2D_new



        # --- Post-AO fractions core should always be <= to the DL fractions core --- 
        fraction_core_2D = np.minimum(fraction_core_2D, fraction_core_DL_2D)
        
        
        
        # --- Post-AO fractions core 'off-axis' should be equal with and without coronagraph ---
        if coronagraph is not None:
            fraction_core_no_coro_2D          = fraction_core_no_coro_3D[ib]
            fraction_core_coro_2D             = fraction_core_2D
            fraction_core_no_coro_off_axis_1D = fraction_core_no_coro_2D[:, -1] # Without coronagraph, this is always 'off-axis' by definition, so this is constant with separation, but for homogenety we also take sep = -1
            fraction_core_coro_off_axis_1D    = fraction_core_coro_2D[:, -1]    # With coronagraph, the off-axis fraction core is assumed to be given at the largest offset, i.e. sep = -1
            scale_fraction_core_off_axis_1D   = fraction_core_no_coro_off_axis_1D / fraction_core_coro_off_axis_1D
            fraction_core_2D                 *= scale_fraction_core_off_axis_1D[:, None]
            
        
        
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
        plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=wave_raw, separation=separation, PSF_profile_density_2D=PSF_profile_density_2D, fraction_core_2D=fraction_core_2D, radial_transmission_2D=radial_transmission_2D, type_PSF="post-AO", title_suffix="\n")
        
        
        
        # --- Saving for later ---
        if coronagraph is None:
            PSF_profile_density_no_coro_3D[ib] = np.copy(PSF_profile_density_2D)        
            fraction_core_no_coro_3D[ib]       = np.copy(fraction_core_2D)  
        else:
            PSF_profile_density_coro_3D[ib] = np.copy(PSF_profile_density_2D)        
            fraction_core_coro_3D[ib]       = np.copy(fraction_core_2D)   
            radial_transmission_3D[ib]      = np.copy(radial_transmission_2D)   
        wave_raw_2D[ib] = wave_raw


        
    
# %%
# Extrapolating 2D post-AO profile to 4D (wave, WFE, IWA, separation)
#

# =============================================================================
# Build ANDES PSF grids over AO performance and coronagraph IWA
# =============================================================================
# Output convention:
#   4D arrays: (wave, WFE, IWA, separation)
#
# For each spectral band, build_PSF_grid() first extrapolates the reference PSF
# products over WFE and IWA. The resulting 4D products are then corrected so that
# they exactly recover the original nominal inputs at (WFE_ref, IWA_ref), before
# applying empirical consistency checks, monotonicity constraints, and smoothing.

eps        = 1e-8
safe01     = lambda x: np.clip(x, eps, 1 - eps)
points_ref = np.stack(np.meshgrid(wave, WFE_ref, IWA_ref, separation, indexing="ij"), axis=-1)

for ib, band in enumerate(bands):

    pxscale = config_data["pxscale"][band] * 1000  # [mas/px]

    # -------------------------------------------------------------------------
    # 1) Build 4D PSF grid for the current ANDES band
    # -------------------------------------------------------------------------
    PSF_profile_density_no_coro_4D, fraction_core_no_coro_4D, PSF_profile_density_coro_4D, fraction_core_coro_4D, radial_transmission_4D = build_PSF_grid(SR=SR, D=D, wave=wave, r_WFE=r_WFE, r_IWA=r_IWA, separation=separation, PSF_profile_density_no_coro_2D=PSF_profile_density_no_coro_3D[ib], fraction_core_no_coro_2D=fraction_core_no_coro_3D[ib], PSF_profile_density_DL_2D=PSF_profile_density_DL_3D[ib], fraction_core_DL_2D=fraction_core_DL_3D[ib], coronagraph="LYOT", IWA_ref=IWA_ref, PSF_profile_density_coro_2D=PSF_profile_density_coro_3D[ib], fraction_core_coro_2D=fraction_core_coro_3D[ib], radial_transmission_2D=radial_transmission_3D[ib])

    # -------------------------------------------------------------------------
    # 2) Forcing nominal equality
    # -------------------------------------------------------------------------
    # The generated 4D grid must recover exactly the original 2D input products
    # at the nominal reference point (WFE_ref, IWA_ref).

    # On PSF_profile_density (no_coro)
    PSF_profile_density_no_coro_OG_2D = np.copy(PSF_profile_density_no_coro_3D[ib])
    PSF_profile_density_no_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), PSF_profile_density_no_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_PSF_no_coro           = PSF_profile_density_no_coro_OG_2D / PSF_profile_density_no_coro_EX_2D
    PSF_profile_density_no_coro_4D   *= corr_factor_PSF_no_coro[:, None, None, :]

    # On fraction_core (no_coro)
    fraction_core_no_coro_OG_2D = np.copy(fraction_core_no_coro_3D[ib])
    fraction_core_no_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), fraction_core_no_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_FC_no_coro      = fraction_core_no_coro_OG_2D / fraction_core_no_coro_EX_2D
    fraction_core_no_coro_4D   *= corr_factor_FC_no_coro[:, None, None, :]

    # On PSF_profile_density (coro)
    PSF_profile_density_coro_OG_2D = np.copy(PSF_profile_density_coro_3D[ib])
    PSF_profile_density_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), PSF_profile_density_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_PSF_coro           = PSF_profile_density_coro_OG_2D / PSF_profile_density_coro_EX_2D
    PSF_profile_density_coro_4D   *= corr_factor_PSF_coro[:, None, None, :]

    # On fraction_core (coro)
    fraction_core_coro_OG_2D = np.copy(fraction_core_coro_3D[ib])
    fraction_core_coro_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), fraction_core_coro_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_FC_coro      = fraction_core_coro_OG_2D / fraction_core_coro_EX_2D
    fraction_core_coro_4D   *= corr_factor_FC_coro[:, None, None, :]

    # On radial_transmission (coro)
    radial_transmission_OG_2D = np.copy(radial_transmission_3D[ib])
    radial_transmission_EX_2D = RegularGridInterpolator((wave, WFE, IWA, separation), radial_transmission_4D, bounds_error=True, fill_value=np.nan)(points_ref)[:, 0, 0, :]
    corr_factor_RT_coro       = radial_transmission_OG_2D / radial_transmission_EX_2D
    radial_transmission_4D   *= corr_factor_RT_coro[:, None, None, :]

    # # -------------------------------------------------------------------------
    # # 3) Empirical consistency checks and monotonicity constraints
    # # -------------------------------------------------------------------------

    # # 3.1) The non-coronagraphic core fraction must not exceed the empirical
    # #      estimate derived from the PSF profile.
    # fraction_core_no_coro_3D_from_PSF = PSF_profile_density_no_coro_4D[:, :, :, 0] * pxscale**2 * size_core**2
    # fraction_core_no_coro_4D         = safe01(np.minimum(fraction_core_no_coro_4D, fraction_core_no_coro_3D_from_PSF[:, :, :, None]))

    # # 3.2) The coronagraphic core fraction must not exceed the non-coronagraphic one.
    # fraction_core_coro_4D = safe01(np.minimum(fraction_core_coro_4D, fraction_core_no_coro_4D))

    # # 3.3) Monotonicity with IWA:
    # #      for a larger IWA, both the coronagraphic core fraction and the radial
    # #      transmission are assumed to be no better than for a smaller IWA.
    # fraction_core_coro_4D  = safe01(np.minimum.accumulate(fraction_core_coro_4D,  axis=2))
    # radial_transmission_4D = safe01(np.minimum.accumulate(radial_transmission_4D, axis=2))

    # # 3.4) Monotonicity with separation:
    # #      both the coronagraphic core fraction and the radial transmission are
    # #      assumed to increase monotonically with angular separation.
    # fraction_core_coro_4D  = safe01(np.maximum.accumulate(fraction_core_coro_4D, axis=3))
    # radial_transmission_4D = safe01(np.maximum.accumulate(radial_transmission_4D, axis=3))

    # # -------------------------------------------------------------------------
    # # 4) On-axis coronagraphic core-fraction consistency check
    # # -------------------------------------------------------------------------
    # # Apply an affine transform in logit space so that the corrected profile
    # # matches the target on-axis value while preserving the off-axis asymptote.

    # fraction_core_coro_3D_from_PSF = safe01(PSF_profile_density_coro_4D[:, :, :, 0] * pxscale**2 * size_core**2)
    # y0_3D                          = safe01(fraction_core_coro_4D[:, :, :, 0])
    # y0_new_3D                      = safe01(np.minimum(y0_3D, fraction_core_coro_3D_from_PSF))
    # y1_3D                          = safe01(fraction_core_coro_4D[:, :, :, -1])
    # for idx_WFE in range(N_WFE):
    #     for idx_IWA in range(N_IWA):
    #         y0     = y0_3D[:, idx_WFE, idx_IWA]
    #         y0_new = y0_new_3D[:, idx_WFE, idx_IWA]
    #         y1     = y1_3D[:, idx_WFE, idx_IWA]
    #         L0     = logit(y0)
    #         L1     = logit(y1)
    #         L0new  = logit(y0_new)
    #         den_a  = L1 - L0
    #         valid  = np.isfinite(L0) & np.isfinite(L1) & np.isfinite(L0new) & (np.abs(den_a) > eps)
    #         if not np.any(valid):
    #             continue
    #         prof = safe01(fraction_core_coro_4D[:, idx_WFE, idx_IWA, :])
    #         a    = (L1[valid] - L0new[valid]) / den_a[valid]
    #         b    = L0new[valid] - a * L0[valid]
    #         fraction_core_coro_4D[valid, idx_WFE, idx_IWA, :] = expit(a[:, None] * logit(prof[valid]) + b[:, None])

    # fraction_core_coro_4D = safe01(np.minimum(fraction_core_coro_4D, fraction_core_no_coro_4D))
    # fraction_core_coro_4D = safe01(np.minimum.accumulate(fraction_core_coro_4D, axis=2))
    # fraction_core_coro_4D = safe01(np.maximum.accumulate(fraction_core_coro_4D, axis=3))
    
    # # -------------------------------------------------------------------------
    # # 5) Smooth the coronagraphic core fraction
    # # -------------------------------------------------------------------------
    # # The smoothing scale is set by the diffraction-limited FWHM converted into
    # # bins along the separation axis.
    # r          = separation                                   # (separation) [mas]
    # dr         = np.gradient(r)                               # (separation) [mas/bin]
    # FWHM       = 1.029 * wave * 1e-6 / D * rad2arcsec * 1000  # (wave) [mas]
    # sigma      = FWHM / (2 * np.sqrt(2 * np.log(2)))          # (wave) [mas]
    # sigma_bins = np.nanmedian(sigma[:, None] / dr[None, :], axis=1)
    # for iw in range(N_wave):
    #     fraction_core_coro_4D[iw] = gaussian_filter1d(fraction_core_coro_4D[iw], sigma=sigma_bins[iw], axis=2)
    # fraction_core_coro_4D = safe01(np.minimum(fraction_core_coro_4D, fraction_core_no_coro_4D))
    # fraction_core_coro_4D = safe01(np.minimum.accumulate(fraction_core_coro_4D, axis=2))
    # fraction_core_coro_4D = safe01(np.maximum.accumulate(fraction_core_coro_4D, axis=3))
    

    # --- Plotting + saving ---
    for coronagraph in coronagraphs:
    
        # --- Assigning the profiles ---
        if coronagraph is None:
            PSF_profile_density_4D = PSF_profile_density_no_coro_4D
            fraction_core_4D       = fraction_core_no_coro_4D
        else:
            PSF_profile_density_4D = PSF_profile_density_coro_4D
            fraction_core_4D       = fraction_core_coro_4D
        
        
        
        # --- PLOT ---
        idx_IWA = np.abs(r_IWA - 1).argmin() # at r_IWA = 1.0
        #for idx_WFE in range(len(r_WFE)):
        for idx_WFE in [0, np.abs(r_WFE - 1).argmin(), -1]:
            psf    = PSF_profile_density_4D[:, idx_WFE, idx_IWA]
            fc     = fraction_core_4D[:, idx_WFE, idx_IWA]
            if coronagraph is not None:
                rt = radial_transmission_4D[:, idx_WFE, idx_IWA]
            else:
                rt = None
            plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=wave_raw_2D[ib], separation=separation, PSF_profile_density_2D=psf, fraction_core_2D=fc, radial_transmission_2D=rt, type_PSF="post-AO", title_suffix=f" \n r_WFE = {r_WFE[idx_WFE]:.1f} & r_IWA = {r_IWA[idx_IWA]:.1f}")
        
        if coronagraph is not None:
            idx_WFE = np.abs(r_WFE - 1).argmin() # at r_WFE = 1.0
            #for idx_IWA in range(len(r_IWA)):
            for idx_IWA in [0, np.abs(r_IWA - 1).argmin(), -1]:
                psf    = PSF_profile_density_4D[:, idx_WFE, idx_IWA]
                fc     = fraction_core_4D[:, idx_WFE, idx_IWA]
                if coronagraph is not None:
                    rt = radial_transmission_4D[:, idx_WFE, idx_IWA]
                else:
                    rt = None
                plot_profiles(instru=instru, coronagraph=coronagraph, apodizer=apodizer, strehl=strehl, pxscale=pxscale, sep_unit=sep_unit, size_core=size_core, wave=wave, wave_raw=wave_raw_2D[ib], separation=separation, PSF_profile_density_2D=psf, fraction_core_2D=fc, radial_transmission_2D=rt, type_PSF="post-AO", title_suffix=f" \n r_WFE = {r_WFE[idx_WFE]:.1f} & r_IWA = {r_IWA[idx_IWA]:.1f}")
        
        
        
        # --- SAVING ---
        suffix_PSF = f"{apodizer}_{coronagraph}_{strehl}_{pxscale}_{lmin}_{lmax}_{WFE_min}_{WFE_max}_{IWA_min}_{IWA_max}_{sep_max}"
        output_dir = get_sim_data_path() / "ELT" / instru / "PSF_inputs"
        fits.writeto(f"{output_dir}/{instru}_WFE_{suffix_PSF}.fits",                    WFE,                    overwrite=True)
        fits.writeto(f"{output_dir}/{instru}_IWA_{suffix_PSF}.fits",                    IWA,                    overwrite=True)
        fits.writeto(f"{output_dir}/{instru}_wave_{suffix_PSF}.fits",                   wave,                   overwrite=True)
        fits.writeto(f"{output_dir}/{instru}_separation_{suffix_PSF}.fits",             separation,             overwrite=True)
        fits.writeto(f"{output_dir}/{instru}_PSF_profile_density_4D_{suffix_PSF}.fits", PSF_profile_density_4D, overwrite=True)
        fits.writeto(f"{output_dir}/{instru}_fraction_core_4D_{suffix_PSF}.fits",       fraction_core_4D,       overwrite=True)
        if coronagraph is not None:
            fits.writeto(f"{output_dir}/{instru}_radial_transmission_4D_{suffix_PSF}.fits", radial_transmission_4D, overwrite=True)




