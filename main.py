from src.FastCurves import *
from src.colormaps import *
from src.FastCurves_interface import *
from src.FastYield_interface import *
from src.DataAnalysis_interface import *






# HWO ESTIMATIONS
if 1==0:
    D                   = 8      # [m]
    trans_instru        = 0.2    # [no unit]
    central_obstruction = 0.2    # [no unit]
    spiders_thickness   = 0.1    # [m]
    detector            = "H2RG" # detector considered
    g_a                 = 0.32   # elongation max phase function
    FOV_max             = 1000   # max separation considered by the PSF profile
    exposure_time       = 120    # [mn]
    
    distance = 10   # [pc]
    SMA      = 1    # [AU]
    
    R_planet        = 0.5          # [Earth_rad]
    T_planet        = 300        # [K]
    lg_planet       = 3.0        # [dex]
    rv_planet       = 30         # [km/s]
    vsini_planet    = 0.5        # [km/s]
    thermal_model   = "BT-Settl"
    reflected_model = "tellurics"
    
    R_star     = 1    # [Sun_rad]
    T_star     = 3500 # [K]
    lg_star    = 4.0  # [dex]
    rv_star    = 0    # [km/s]
    vsini_star = 2    # [km/s]
    
    Rc          = 100        # MM cut-off resolution
    filter_type = "gaussian" # MM filter type
    
    # General parameters for the simulation
    N        = 100
    R        = np.logspace(np.log10(1000), np.log10(100000), num=N)
    lmin     = 0.6
    lmax     = 6   # µm
    lambda_0 = np.linspace(lmin, lmax, N)
    dl       = np.nanmean(np.diff(lambda_0))
    
    #
    # DETECTORS CARACTERISTICS
    #
    
    if detector == "H2RG":
        RON          = 10.0      # [e-/px/DIT]
        DC           = 0.0053    # [e-/px/s]
        saturation_e = 64_000    # [e-]
        Npx          = 2048      # [px]
        min_DIT      = 1.4725/60 # [mn]
    else:
        raise KeyError(f"Please define caracteristics for the {detector} detector.")
    
    #
    # PSF SIMULATION
    #
    
    N_pupil       = 200                                               # image N_pupil
    pxscale_pupil = D / N_pupil                                       # [m/px]
    zeropadding   = 5                                                 # Nyquist sampling >= 2
    N_focal       = zeropadding * N_pupil                             # Nb of points of the PSF
    FOV_focal     = (lambda_0 * 1e-6 / D * 1000*rad2arcsec) * N_pupil # [mas]
    pxscale_focal = FOV_focal / N_focal                               # PSF_DL pxscale [mas/px]
    pxscale_lD    = 1 / zeropadding                                   # PSF_DL [lambda/D]
    pxscale_real  = zeropadding * pxscale_focal / 2                   # Effective Nyquist Sampled pxscale [mas/px]
    
    # Primary and secondary mirrors
    pupil                   = np.zeros((N_pupil, N_pupil))
    center                  = N_pupil // 2
    mirror_radius           = N_pupil // 2
    y, x                    = np.ogrid[:N_pupil, :N_pupil]
    mask_mirror             = (x - center)**2 + (y - center)**2 <= mirror_radius**2
    pupil[mask_mirror]      = 1
    obstruction_radius      = mirror_radius * central_obstruction
    mask_obstruction        = (x - center)**2 + (y - center)**2 <= obstruction_radius**2
    pupil[mask_obstruction] = 0
    
    # Spiders
    spiders_thickness_px = int(round(spiders_thickness/pxscale_pupil))
    pupil[center - spiders_thickness_px//2 : center + spiders_thickness_px//2, :] = 0  # Vertical spider
    pupil[:, center - spiders_thickness_px//2 : center + spiders_thickness_px//2] = 0  # Horizontal spider
    
    # Effective collective area
    S = np.sum(pupil == 1) * (pxscale_pupil) ** 2  # effective collective area [m2]
    
    # Plot
    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(pupil, cmap='gray', origin='lower', extent=[-D / 2, D / 2, -D / 2, D / 2])
    plt.title(f"8m Pupil with {int(central_obstruction * 100)}% Central Obstruction and spiders")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.show()
    
    # PSF computing
    pupil_zeropadded = np.zeros((N_focal, N_focal))
    pupil_zeropadded[:N_pupil, :N_pupil] = pupil
    PSF_DL  = np.fft.fftshift(np.fft.fft2(pupil_zeropadded))
    PSF_DL  = np.abs(PSF_DL)**2
    PSF_DL  = crop(PSF_DL)
    PSF_DL /= np.nansum(PSF_DL)
    
    # Plot
    plt.figure(dpi=300)
    plt.imshow(PSF_DL/np.nanmax(PSF_DL), cmap="inferno", extent = [-pxscale_lD*N_focal/2, pxscale_lD*N_focal/2, -pxscale_lD*N_focal/2, pxscale_lD*N_focal/2], norm=LogNorm(vmin=1e-6, vmax=1))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel(r'x offset [$\lambda / D$]')
    plt.ylabel(r'y offset [$\lambda / D$]')
    plt.title(f"Diffraction-Limited PSF")
    cbar = plt.colorbar() ; cbar.set_label('PSF [raw contrast]', fontsize=14, labelpad=20, rotation=270)
    circle = plt.Circle((0, 0), 1.22, color='red', fill=False, linewidth=1, label = r"$1.22 \lambda / D$ ")
    plt.gca().add_patch(circle)  # Ajout du cercle à l'axe actuel
    plt.legend()
    plt.show()
    
    # Profile computing
    NbLine, NbColumn = PSF_DL.shape
    y0, x0           = NbLine // 2, NbColumn // 2
    size_core_focal  = zeropadding # 1 lambda/D / pxscale_lD = zeropadding
    r_max            = int(round(FOV_max/pxscale_focal[0]))
    profile          = np.zeros((2, r_max+1)) # array 2D de taille 2x Nbline (ou Column) /2
    for r in range(r_max+1):
        profile[0, r] = r
        if r==0:
            profile[1, r] = np.nanmean(PSF_DL * annular_mask(0, 0, size=(NbLine, NbColumn)))
        else:
            profile[1, r] = np.nanmean(PSF_DL * annular_mask(max(1, r-1), r, size=(NbLine, NbColumn)))
    
    if size_core_focal==1:
        PSF_core = PSF_DL[y0, x0]
    else:
        PSF_core = PSF_DL[y0-size_core_focal//2:y0+size_core_focal//2+1, x0-size_core_focal//2:x0+size_core_focal//2+1]
    fraction_core = np.nansum(PSF_core)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(profile[0]*pxscale_lD, profile[1], label=f"Diffraction-limited profile \nFraction in core = {round(100*fraction_core)} %", color='tab:blue', linewidth=2)
    plt.xlabel(r"Separation [$\lambda / D$]", fontsize=14, labelpad=10)
    plt.ylabel("Profile [Fraction Flux / px]", fontsize=14, labelpad=10)
    plt.yscale('log')
    plt.title("Radial Profile of the Diffraction-Limited PSF", fontsize=16, weight='bold')
    plt.legend(fontsize=12, loc='upper right', frameon=False, labelspacing=1.2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)
    plt.xlim(0)
    plt.tight_layout()
    plt.show()
    
    # Profile interpolation
    profile_interp = interp1d(profile[0], profile[1], bounds_error=True, fill_value=np.nan)
    
    #
    # SPECTRA SIMULATIONS
    #
    
    res_model    = 2e5
    delta_lambda = (lmin+lmax)/2 / (2*res_model)
    wave         = np.arange(0.9*lmin, 1.1*lmax, delta_lambda)
    distance     = distance * u.pc       
    SMA          = SMA * u.AU           
    R_planet     = R_planet * u.earthRad
    R_star       = R_star * u.solRad
    separation_planet = 1000*(SMA/distance).value # [mas]
    
    # Star spectrum
    star       = load_star_spectrum(T_star, lg_star)             # loading star spectrum [J/s/µm/m2]
    star       = star.interpolate_wavelength(wave, renorm=False) # interpolating star spectrum
    star.flux *= float((R_star/distance).decompose()**2)         # dilution factor
    star       = star.doppler_shift(rv_star)                     # Doppler shifting
    star       = star.broad(vsini_star)                          # Rotationnal broadening
        
    # Planet thermal spectrum
    planet_thermal       = load_planet_spectrum(T_planet, lg_planet, model=thermal_model)
    planet_thermal       = planet_thermal.interpolate_wavelength(wave, renorm = False)
    planet_thermal.flux *= float((R_planet/distance).decompose()**2)
    planet_thermal       = planet_thermal.doppler_shift(rv_planet)
    planet_thermal       = planet_thermal.broad(vsini_planet)
    
    # Planet reflected spectrum
    albedo = load_albedo(planet_thermal.T, planet_thermal.lg)
    albedo = albedo.interpolate_wavelength(wave, renorm = False)
    if reflected_model == "PICASO":
        planet_reflected = star.flux * albedo.flux * g_a * (R_planet/SMA).decompose()**2
    elif reflected_model == "flat":
        planet_reflected = star.flux * np.nanmean(albedo.flux) * g_a * (R_planet/SMA).decompose()**2
    elif reflected_model == "tellurics":
        wave_tell, tell = fits.getdata("sim_data/Transmission/sky_transmission_airmass_2.5.fits")
        f = interp1d(wave_tell, tell, bounds_error=False, fill_value=np.nan)
        tell = f(wave)
        planet_reflected = star.flux * np.nanmean(albedo.flux)/np.nanmean(tell)*tell * g_a * (R_planet/SMA).decompose()**2
    else :
        raise KeyError(reflected_model+" IS NOT A VALID REFLECTED MODEL : tellurics, flat, or PICASO")
    planet_reflected = Spectrum(wave, np.nan_to_num(np.array(planet_reflected.value)), max(star.R, albedo.R), albedo.T, lg_planet, reflected_model)
    planet_reflected = planet_reflected.doppler_shift(rv_planet)
    planet_reflected = planet_reflected.broad(vsini_planet)
    
    # Planet spectrum
    planet = Spectrum(wave, planet_thermal.flux+planet_reflected.flux, max(planet_thermal.R, planet_reflected.R), planet_thermal.T, planet_thermal.lg, thermal_model+"+"+reflected_model)
        
    # Blackbody spectra
    bb_star             = (2*const.h * const.c**2 / (wave*u.micron)**5).decompose() / np.expm1((const.h * const.c/(wave * u.micron * const.k_B * T_star * u.K)).decompose())
    bb_star             = bb_star.to(u.J/u.s/u.m**2/u.micron)
    bb_star             = np.pi * bb_star.value * float((R_star/distance).decompose()**2) # facteur de dilution = (R/distance)^2
    bb_planet_thermal   = (2*const.h * const.c**2 / (wave*u.micron)**5).decompose() / np.expm1((const.h * const.c/(wave * u.micron * const.k_B * T_planet * u.K)).decompose())
    bb_planet_thermal   = bb_planet_thermal.to(u.J/u.s/u.m**2/u.micron)
    bb_planet_thermal   = np.pi * bb_planet_thermal.value * float((R_planet/distance).decompose()**2) # facteur de dilution = (R/distance)^2
    bb_planet_reflected = bb_star * np.nanmean(albedo.flux) * g_a * float((R_planet/SMA).decompose()**2)
    bb_planet           = bb_planet_thermal + bb_planet_reflected
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=300)
    axes[0].plot(wave, planet_thermal.flux, 'r-', label=f"thermal ({thermal_model})")
    axes[0].plot(wave, planet_reflected.flux, 'b-', label=f"reflected ({reflected_model})")
    axes[0].plot(wave, bb_planet, 'k-', label=f"blackbody")
    axes[0].set_title("Planet spectrum", fontsize=16, weight='bold')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel("wavelength [µm]", fontsize=14)
    axes[0].set_ylabel("flux [J/s/µm/m2]", fontsize=14)
    axes[0].set_xlim(lmin, lmax)
    axes[0].set_ylim(1e-30, 10 * np.nanmax(planet_thermal.flux))
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    axes[0].minorticks_on()
    axes[0].legend(fontsize=12)
    axes[1].plot(wave, star.flux, 'r-', label=f"star ({star.model})")
    axes[1].plot(wave, bb_star, 'k-', label=f"blackbody")
    axes[1].set_title("Star spectrum", fontsize=16, weight='bold')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel("wavelength [µm]", fontsize=14)
    axes[1].set_ylabel("flux [J/s/µm/m2]", fontsize=14)
    axes[1].set_xlim(lmin, lmax)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    axes[1].minorticks_on()
    axes[1].legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Converting spectra into total photo-electrons
    dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] ; dwl[dwl == 0] = np.nanmean(dwl) # [µm]
    star.flux   *= wave*1e-6 / (h*c)             # [J/s/m²/µm] => [ph/s/m2/µm]
    star.flux   *= S*dwl                         # [ph/s/m2/µm] => [ph/s]
    star.flux   *= 60*exposure_time*trans_instru # [ph/s] => [e-]
    planet.flux *= wave*1e-6 / (h*c)             # [J/s/m²/µm] => [ph/s/m2/µm]
    planet.flux *= S*dwl                         # [ph/s/m2/µm] => [ph/s]
    planet.flux *= 60*exposure_time*trans_instru # [ph/s] => [e-]
    
    # 
    # SNR CALCULATIONS
    #
    
    size_core  = 2                                      # [px] by definition, since we consider Nyquist sampled data
    sigma_dc_2 = size_core**2 * DC * exposure_time * 60 # [e-/FWHM]
    
    SNR_total = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_halo  = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_ron   = np.zeros((len(R), len(lambda_0)), dtype=float)
    SNR_dc    = np.zeros((len(R), len(lambda_0)), dtype=float)
    
    for ires in tqdm(range(len(R))):
        res      = R[ires]
        dl       = (lmin+lmax)/2 / (2 * res)
        wav      = np.arange(lmin, lmax, dl)
        planet_R = planet.degrade_resolution(wav, renorm=True)
        star_R   = star.degrade_resolution(wav, renorm=True)
        for ilam, l0 in enumerate(lambda_0):
            umin  = l0-(Npx/2)*dl
            umax  = l0+(Npx/2)*dl
            valid = np.where(((wav<umax)&(wav>umin)))
            f_psf = profile_interp(separation_planet/pxscale_focal[ilam]) * pxscale_real[ilam]**2 / pxscale_focal[ilam]**2
            
            star_R_crop   = Spectrum(wav[valid], star_R.flux[valid], res, None)
            planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, None)   
            
            star_R_crop.flux   *= f_psf
            planet_R_crop.flux *= fraction_core
            
            star_HF, star_BF     = filtered_flux(star_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
            planet_HF, planet_BF = filtered_flux(planet_R_crop.flux, R=res, Rc=Rc, filter_type=filter_type)
            
            # RON computation
            saturating_DIT = saturation_e/np.nanmax(star_R_crop.flux/exposure_time) # [mn]
            if saturating_DIT < min_DIT: 
                DIT = min_DIT
                print(" Saturated detector even with the shortest integration time")
            else: # otherwise the DIT is given by the saturating DIT
                DIT = saturating_DIT # mn
            if DIT > exposure_time: # The DIT cannot be longer than the total exposure time
                DIT = exposure_time # mn
            nb_min_DIT = 1 # "Up the ramp" reading mode: the pose is sequenced in several non-destructive readings to reduce reading noise (see https://en.wikipedia.org/wiki/Signal_averaging).
            if DIT > nb_min_DIT*min_DIT: # choose 4 min_DIT because if intermittent readings are too short, the detector will heat up too quickly => + dark current
                N_i = DIT/(nb_min_DIT*min_DIT) # number of intermittent readings
                RON_eff = RON/np.sqrt(N_i) # effective read out noise (in e-/DIT)
            else:
                RON_eff = RON
            if RON_eff < 0.5:
                RON_eff = 0.5
            N_DIT       = exposure_time / DIT                # [no unit]
            sigma_ron_2 = size_core**2 * N_DIT * RON_eff**2  # [e-/FWHM]
            
            # template computation
            template = planet_HF / np.sqrt(np.nansum((planet_HF)**2)) # [no unit]
            
            # Stellar halo photon computation
            sigma_halo_2 = size_core**2 * np.nansum(star_R_crop.flux * template**2) # [e-/FWHM]
            
            # alpha computation
            alpha = np.sqrt(np.nansum((planet_HF)**2)) 
            
            # beta computation 
            beta = np.nansum(star_HF*planet_BF/star_BF * template)
            
            SNR_total[ires, ilam] = (alpha - beta) / np.sqrt(sigma_halo_2 + sigma_ron_2 + sigma_dc_2)
            SNR_halo[ires, ilam]  = (alpha - beta) / np.sqrt(sigma_halo_2)
            SNR_ron[ires, ilam]   = (alpha - beta) / np.sqrt(sigma_ron_2)
            SNR_dc[ires, ilam]    = (alpha - beta) / np.sqrt(sigma_dc_2)
            
            plt.figure(figsize=(10, 6), dpi=300)        
            plt.plot(wav[valid], planet_R_crop.flux, label="Planet signal", color='blue', linewidth=2)
            plt.plot(wav[valid], np.sqrt(size_core**2 * star_R_crop.flux), label=f"Stellar halo photon noise (S/N = {round(SNR_halo[ires, ilam], 1)})", color='red', linestyle='--', linewidth=2)
            plt.plot(wav[valid], np.sqrt(sigma_ron_2) * wav[valid] / wav[valid], label=f"RON (S/N = {round(SNR_ron[ires, ilam], 1)})", color='green', linestyle='--', linewidth=2)
            plt.plot(wav[valid], np.sqrt(sigma_dc_2) * wav[valid] / wav[valid], label=f"Dark current photon noise (S/N = {round(SNR_dc[ires, ilam], 1)})", color='orange', linestyle='--', linewidth=2)
            plt.title("Planet and Noise Components", fontsize=16, weight='bold')
            plt.xlabel("Wavelength [µm]", fontsize=14, labelpad=10)
            plt.ylabel("Flux [e-]", fontsize=14, labelpad=10)        
            plt.legend(fontsize=12, frameon=False, labelspacing=1.2)        
            plt.yscale('log')        
            plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)
            plt.minorticks_on()        
            plt.tight_layout()
            plt.show()
               
            
    plt.figure(dpi=300)
    plt.yscale('log')
    plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14)
    plt.ylabel("instrumental resolution R", fontsize=14)
    plt.ylim([R[0], R[-1]])
    plt.xlim(lmin, lmax)
    plt.contour(lambda_0, R, 100*SNR_total/np.nanmax(SNR_total), linewidths=0.333, colors='k')
    plt.pcolormesh(lambda_0, R, 100*SNR_total/np.nanmax(SNR_total), cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    plt.show()
    
    
    plt.figure(dpi=300)
    plt.yscale('log')
    plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14)
    plt.ylabel("instrumental resolution R", fontsize=14)
    plt.ylim([R[0], R[-1]])
    plt.xlim(lmin, lmax)
    plt.contour(lambda_0, R, 100*SNR_halo/np.nanmax(SNR_halo), linewidths=0.333, colors='k')
    plt.pcolormesh(lambda_0, R, 100*SNR_halo/np.nanmax(SNR_halo), cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    plt.show()
    
    
    plt.figure(dpi=300)
    plt.yscale('log')
    plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14)
    plt.ylabel("instrumental resolution R", fontsize=14)
    plt.ylim([R[0], R[-1]])
    plt.xlim(lmin, lmax)
    plt.contour(lambda_0, R, 100*SNR_ron/np.nanmax(SNR_ron), linewidths=0.333, colors='k')
    plt.pcolormesh(lambda_0, R, 100*SNR_ron/np.nanmax(SNR_ron), cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    plt.show()
    
    
    plt.figure(dpi=300)
    plt.yscale('log')
    plt.xlabel("central wavelength range $\lambda_0$ [µm]", fontsize=14)
    plt.ylabel("instrumental resolution R", fontsize=14)
    plt.ylim([R[0], R[-1]])
    plt.xlim(lmin, lmax)
    plt.contour(lambda_0, R, 100*SNR_dc/np.nanmax(SNR_dc), linewidths=0.333, colors='k')
    plt.pcolormesh(lambda_0, R, 100*SNR_dc/np.nanmax(SNR_dc), cmap=plt.get_cmap('rainbow'), vmin=0, vmax=100)
    cbar = plt.colorbar() ; cbar.set_label('$GAIN_{S/N}$ [%]', fontsize=14, labelpad=20, rotation=270)
    plt.show()
    
    




















# SYSTEMATICS MODELS ERRORS
if 1==0:
    force_new_est = False ; show = False
    
    R_arr = np.logspace(np.log10(1000), np.log10(100000), 10)
    Rc = 100 ; filter_type = "gaussian"
    lmin = 1
    lmax = 2.5
    trans = 1
    
    model_template = "BT-Settl"
    DT = 200 ; Dlg = 1

    R_template = load_planet_spectrum(model=model_template).R
    T_template_arr, lg_template_arr = get_model_grid(model_template, instru=None)   
    models_planet = ["BT-Settl", "Exo-REM", "PICASO", "Morley", "SONORA", "Saumon"]
    models_planet = [model for model in models_planet if model != model_template]
    vsini_planet = 0 ; vsini_arr = np.array([0])
    rv_planet = 25 ; rv_arr = np.arange(rv_planet-10, rv_planet+10, 1)
    
    sigma_T = np.zeros((len(models_planet), len(R_arr))) + np.nan
    sigma_lg = np.zeros((len(models_planet), len(R_arr))) + np.nan
    sigma_rv = np.zeros((len(models_planet), len(R_arr))) + np.nan
    for l in tqdm(range(len(models_planet)), desc="model loop"):
        model_planet = models_planet[l]
        T_planet_arr, lg_planet_arr = get_model_grid(model_planet, instru=None)    
        T_error = np.zeros((len(R_arr), len(T_planet_arr), len(lg_planet_arr))) + np.nan
        lg_error = np.zeros((len(R_arr), len(T_planet_arr), len(lg_planet_arr))) + np.nan
        rv_error = np.zeros((len(R_arr), len(T_planet_arr), len(lg_planet_arr))) + np.nan
        R_planet = load_planet_spectrum(model=model_planet).R
        for k in tqdm(range(len(R_arr)), desc="R loop"):
            R = R_arr[k]
            if R < R_planet and R < R_template:
                dl = ((lmin+lmax)/2)/(2*R)
                wave = np.arange(lmin, lmax, dl) # constant and linear wavelength array on the considered band
                for i in range(len(T_planet_arr)):
                    T_planet = T_planet_arr[i]
                    T_arr = np.arange(max(np.nanmin(T_planet_arr), np.nanmin(T_template_arr), T_planet-DT/2), min(np.nanmax(T_planet_arr), np.nanmax(T_template_arr), T_planet+DT/2) + 25, 25).astype(np.float32) # dT = 5 K
                    for j in range(len(lg_planet_arr)):
                        lg_planet = lg_planet_arr[j]
                        lg_arr = np.arange(max(np.nanmin(lg_planet_arr), np.nanmin(lg_template_arr), lg_planet-Dlg/2), min(np.nanmax(lg_planet_arr), np.nanmax(lg_template_arr), lg_planet+Dlg/2) + 0.1, 0.1).astype(np.float32) # dlg = 2
                        if force_new_est:
                            planet = get_template(instru=None, wave=wave, model=model_planet, T_planet=T_planet, lg_planet=lg_planet, vsini_planet=vsini_planet, rv_planet=rv_planet, R=R, Rc=Rc, filter_type=filter_type, epsilon=0.8, fastbroad=True)
                            planet = planet.degrade_resolution(wave, renorm=False) # degrating the planet to the instrumental resolution
                            planet_HF, planet_LF = filtered_flux(planet.flux, R, Rc, filter_type) # high pass filtering
                            d_planet = trans*planet_HF 
                        else:
                            d_planet = None
                        _, _, _, _, corr_4D, _, _, _ = parameters_estimation(instru=str(T_planet), band=str(lg_planet), target_name=model_planet, wave=wave, d_planet=d_planet, star_flux=None, trans=trans, R=R, Rc=Rc, filter_type=filter_type, model=model_template, logL=False, method_logL="classic", sigma_l=None, weight=None, pca=None, precise_estimate=False, SNR_estimate=False, T_planet=None, lg_planet=None, vsini_planet=None, rv_planet=None, T_arr=T_arr, lg_arr=lg_arr, vsini_arr=vsini_arr, rv_arr=rv_arr, show=show, verbose=False, stellar_component=False, force_new_est=force_new_est, d_planet_sim=False)
                        idx_max_corr = np.unravel_index(np.argmax(corr_4D, axis=None), corr_4D.shape)
                        T_error[k, i, j] = T_arr[idx_max_corr[0]] - T_planet
                        lg_error[k, i, j] = lg_arr[idx_max_corr[1]] - lg_planet
                        rv_error[k, i, j] = rv_arr[idx_max_corr[3]] - rv_planet
                        
        sigma_T[l] = np.nanstd(T_error, (1,2))
        sigma_lg[l] = np.nanstd(lg_error, (1,2))
        sigma_rv[l] = np.nanstd(rv_error, (1,2))
    
    
    plt.figure(dpi=300)
    for l in range(len(models_planet)):
        plt.plot(R_arr, sigma_T[l], label=models_planet[l])
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    plt.figure(dpi=300)
    for l in range(len(models_planet)):
        plt.plot(R_arr, sigma_lg[l], label=models_planet[l])
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.figure(dpi=300)
    for l in range(len(models_planet)):
        plt.plot(R_arr, sigma_rv[l], label=models_planet[l])
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    sigma_T = np.nanmean(sigma_T, 0)
    sigma_lg = np.nanmean(sigma_lg, 0)
    sigma_rv = np.nanmean(sigma_rv, 0)
    
    # Plot 1: Temperature systematics errors
    plt.figure(dpi=300, figsize=(6, 4))
    plt.plot(R_arr, sigma_T, "s-", label=r'$\sigma_T$', color='blue', linewidth=1.5, markersize=5)
    plt.xscale('log')
    plt.xlabel(r'$R$', fontsize=12)
    plt.ylabel(r'$\sigma_T$ [K]', fontsize=12)
    plt.title(f'Temperature Retrieval Systematic Errors\nfrom {model_template} Model', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Surface gravity systematics errors
    plt.figure(dpi=300, figsize=(6, 4))
    plt.plot(R_arr, sigma_lg, "s-", label=r'$\sigma_{lg}$', color='green', linewidth=1.5, markersize=5)
    plt.xscale('log')
    plt.xlabel(r'$R$', fontsize=12)
    plt.ylabel(r'$\sigma_{lg}$', fontsize=12)
    plt.title(f'Surface Gravity Retrieval Systematic Errors\nfrom {model_template} Model', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Radial velocity systematics errors
    plt.figure(dpi=300, figsize=(6, 4))
    plt.plot(R_arr, sigma_rv, "s-", label=r'$\sigma_{rv}$', color='red', linewidth=1.5, markersize=5)
    plt.xscale('log')
    plt.xlabel(r'$R$', fontsize=12)
    plt.ylabel(r'$\sigma_{rv}$ [km/s]', fontsize=12)
    plt.title(f'Radial Velocity Retrieval Systematic Errors\nfrom {model_template} Model', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()







#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Graphic Interface :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

FastYield_interface()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# update FastYield :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#create_fastcurves_table(table="Archive") # ~ 5 mn
#create_fastcurves_table(table="Simulated")

#all_SNR_table(table="Archive") # ~ 15 hours
#all_SNR_table(table="Simulated")



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Yield plots :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# archive_yield_instrus_plot(fraction=False)
# archive_yield_bands_plot(fraction=False, instru="HARMONI", strehl="JQ1", thermal_model="BT-Settl", reflected_model="tellurics", systematic=False, PCA=False)
# detections_corner(instru="HARMONI", exposure_time=600, thermal_model="BT-Settl", reflected_model="PICASO", apodizer="NO_SP", strehl="JQ1", band="INSTRU")
# detections_corner_instrus_comparison(exposure_time=600, instru1="HARMONI", instru2="ANDES", apodizer1="SP_Prox", apodizer2="NO_SP", strehl1="JQ1", strehl2="MED", thermal_model="BT-Settl", reflected_model="tellurics")
# detections_corner_models_comparison(model1="tellurics", model2="PICASO", instru="ANDES", apodizer="NO_SP", strehl="MED", exposure_time=600, band="INSTRU")
# detections_corner_apodizers_comparison(exposure_time=600, thermal_model="BT-Settl", reflected_model="tellurics", band="INSTRU")



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves (theoritical cases) :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FastCurves(instru="HARMONI", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='instru', T_star=9600, lg_star=4.0, exposure_time=120, apodizer="SP1", strehl="MED")

#FastCurves(instru="ERIS", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="JQ0")

#FastCurves(instru="ANDES", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

#FastCurves(instru="VIPAPYRUS", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=6, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

#FastCurves(instru="HiRISE", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="MED")

#FastCurves(instru="NIRSpec", model="BT-Settl", calculation="contrast", T_planet=1000, lg_planet=4.0, mag_star=3, band0='K', T_star=6000, lg_star=4.0, exposure_time=120, apodizer="NO_SP", strehl="NO_JQ")



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FastCurves (real data cases) :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Chameleon  (DIT = 138.75 s) / SNR(1MEDIUM) = 11.97
#FastCurves(instru="MIRIMRS", systematic=True, input_DIT=138.75/60, model="BT-Settl", calculation="SNR", separation_planet=2.5, T_planet=2600, lg_planet=3.5, planet_name="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, rv_star=-2.9, rv_planet=15, vsini_star=10, vsini_planet=10, channel=False)
#FastCurves(band_only="1SHORT", instru="MIRIMRS", systematic=False, input_DIT=138.75/60, model="BT-Settl", calculation="corner plot", separation_planet=2.5, T_planet=2600, lg_planet=3.5, planet_name="CT Cha b", mag_star=8.66, mag_planet=14.9, band0='K', T_star=4400, lg_star=3.5, exposure_time=56.426, rv_star=-2.9, rv_planet=15, channel=False)


# HD 19467 (G1V / DIT = 218.8 s )
#FastCurves(calculation="SNR", instru="NIRSpec", systematic=True, separation_planet=1.5, input_DIT=218.8/60, model="BT-Settl", T_planet=950, lg_planet=5.0, planet_name='HD 19467 b', mag_star=5.4, band0='K', mag_planet=17.97, T_star=5680, lg_star=4.0, exposure_time=65.65)
#FastCurves(instru="MIRIMRS", systematic=True, separation_planet=1.5, band_only="1SHORT", systematic=True, input_DIT=218.8/60, calculation="contrast", model="BT-Settl", T_planet=950, lg_planet=5.0, planet_name='HD 19467 b', mag_star=5.4, band0='K', T_star=5680, lg_star=4.0, exposure_time=0.665)

# HIP 65426 b (DIT = 308 s)
#FastCurves(instru="NIRCam", input_DIT=308/60, calculation="contrast", T_planet=1600, lg_planet=4.0, separation_planet=0.8, planet_name="HIP 65426 b", mag_planet=6.771+9.85, mag_star=6.771, band0='K', T_star=8000, lg_star=4.0, exposure_time=20.3)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COLORMAPS :
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#colormap_bandwidth_resolution_with_constant_Nlambda(T_planet=500, T_star=5000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", instru="HARMONI", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
#colormap_bandwidth_resolution_with_constant_Dlambda(T_planet=1400, T_star=6000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", instru="HiRISE", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
#colormap_bandwidth_Tp(instru="HiRISE", T_star=6000, rv_planet=30, spectrum_contributions="thermal", model="BT-Settl", Rc=100, filter_type="gaussian", stellar_halo_photon_noise_limited=True)
#colormap_rv(T_planet=300, T_star=3000, spectrum_contributions="reflected", model="flat", instru="HARMONI", band="J", Rc=100)
#colormap_vsini(T_planet=300, T_star=3000, spectrum_contributions="reflected", model="flat", instru="HARMONI", band="J", Rc=100)
#colormap_maxsep_phase(instru="HARMONI", band="H", inc=90)
#colormap_best_parameters_earth(norm_plot="1", thermal_model="BT-Settl", reflected_model="tellurics", T_planet=288, rv_planet=30, R_star=1, SMA=1, Npx=10000, stellar_halo_photon_noise_limited=True)






if 1==0 :
    for npx in np.logspace(np.log10(1000), np.log10(100000), 10):
        colormap_best_parameters_earth(Npx=npx, stellar_halo_photon_noise_limited=True)
    
    












#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# UTILS : 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------




if 1==0: # alpha / beta 
    Rc = 100 ; R = 10000 ; lmin = 0.1 ; lmax = 3.
    dl = ((lmin+lmax)/2)/(2*R) ; wave = np.arange(lmin, lmax, dl) # constant and linear wavelength array on the considered band
    T_arr = np.arange(500.,3100.,100.) ; T_star = 3000
    alpha = np.zeros_like(T_arr) ; beta = np.zeros_like(T_arr) ; cos_star = np.zeros_like(T_arr)
    star = load_star_spectrum(T_star,4)
    star = star.degrade_resolution(wave,renorm=False)
    star_HF, star_LF = filtered_flux(star.flux, R, Rc)
    for i in tqdm(range(len(T_arr))):
        planet = load_planet_spectrum(T_arr[i], 4.0, "BT-Settl")
        planet = planet.degrade_resolution(wave,renorm=False)
        planet_HF, planet_LF = filtered_flux(planet.flux, R, Rc)
        alpha[i] = np.sqrt(np.nansum(planet_HF**2))
        beta[i] = np.sqrt(np.nansum(star_HF**2 * planet_LF**2/star_LF**2))
        cos_star[i] = np.nansum(planet_HF * star_HF * planet_LF/star_LF) / (alpha[i]*beta[i])
    plt.figure(dpi=300) ; plt.plot(T_arr, alpha/beta) ; plt.xlabel("planet's temperature (in K)") ; plt.ylabel(r"$\alpha / \beta$") ; plt.title(f"Star VS planet spectral content \n with $T_*$ = {T_star}K") ; plt.grid(True) ; plt.show()

if 1==0: # error on signal regarding the template considered : t1 = [Sp]_HF or t2 = [Sp]_HF - [S*]_HF
    N = 1000
    alpha_beta = np.linspace(2, 10, N) # 
    cos_star = np.linspace(-1, 1, N)
    signal1 = np.zeros((len(alpha_beta),len(cos_star)))
    signal2 = np.zeros((len(alpha_beta),len(cos_star)))

    error = np.zeros((len(alpha_beta),len(cos_star)))
    beta = 1
    for i in range(len(alpha_beta)):
        for j in range(len(cos_star)):
            signal1[i,j] = alpha_beta[i]*beta - beta*cos_star[j] # t = [Sp]_HF
            signal2[i,j] = np.sqrt( (alpha_beta[i]*beta)**2 + beta**2 - 2 * alpha_beta[i]*beta*beta*cos_star[j]) # t = [Sp]_HF - [S*]_HF
            error[i,j] = 100*(signal2[i,j]-signal1[i,j])/signal2[i,j]
    plt.figure(dpi=300) ; plt.ylabel(r"$\alpha / \beta$", fontsize=12) ; plt.xlabel(r"$cos\theta_*$",fontsize=12) ; plt.title("Error on signal regarding the template considered:\n "+r"$t = [S_p]_{HF}$ or $t = [S_p]_{HF} - [S_*]_{HF}$",fontsize=14) ; plt.pcolormesh(cos_star, alpha_beta, error, cmap=plt.get_cmap('rainbow'), vmin=np.nanmin(error), vmax=np.nanmax(error)) ; cbar = plt.colorbar() ; cbar.set_label("error (in %)", fontsize=12, labelpad=20, rotation=270) ; plt.contour(cos_star, alpha_beta, error, linewidths=0.1,colors='k') ; plt.show()
    plt.figure(dpi=300) ; plt.ylabel("signal", fontsize=12) ; plt.xlabel(r"$cos\theta_*$",fontsize=12) ; plt.title("Error on signal regarding the template considered:\n "+r"$t_1 = [S_p]_{HF}$ or $t_2 = [S_p]_{HF} - [S_*]_{HF}$",fontsize=14) ; plt.plot(cos_star, signal1[np.abs(alpha_beta-2).argmin(),:]/np.nanmax(signal1[np.abs(alpha_beta-2).argmin(),:]),'r-', label=r"$\alpha/\beta = 2$") ; plt.plot(cos_star, signal2[np.abs(alpha_beta-2).argmin(),:]/np.nanmax(signal2[np.abs(alpha_beta-2).argmin(),:]),'r--') ; plt.plot(cos_star, signal1[np.abs(alpha_beta-5).argmin(),:]/np.nanmax(signal1[np.abs(alpha_beta-5).argmin(),:]),'g-', label=r"$\alpha/\beta = 5$") ; plt.plot(cos_star, signal2[np.abs(alpha_beta-5).argmin(),:]/np.nanmax(signal2[np.abs(alpha_beta-5).argmin(),:]),'g--') ; plt.plot(cos_star, signal1[np.abs(alpha_beta-10).argmin(),:]/np.nanmax(signal1[np.abs(alpha_beta-10).argmin(),:]),'b-', label=r"$\alpha/\beta = 10$") ; plt.plot(cos_star, signal2[np.abs(alpha_beta-10).argmin(),:]/np.nanmax(signal2[np.abs(alpha_beta-10).argmin(),:]),'b--') ; plt.plot([],[],'k-',label=r"$t_1$") ; plt.plot([],[],'k--',label=r"$t_2$") ; plt.legend() ; plt.show()






if 1==0: # PSD of high pass filtered planet spectrum
    band="1SHORT" ; config_data=get_config_data('MIRIMRS') ; band0="instru" ; mag_planet=15
    planet_spectrum = load_planet_spectrum(500, 3.5, "BT-Settl") ; R_planet = planet_spectrum.R
    planet_spectrum, density = spectrum_instru(band0, R_planet, config_data, mag_planet, planet_spectrum)
    planet_spectrum = spectrum_band(config_data, band, planet_spectrum)
    l0 = (config_data['lambda_range']['lambda_min']+config_data['lambda_range']['lambda_max'])/2
    R = config_data['gratings'][band].R
    Rc = 100 ; filter_type = "savitzky_golay"
    wave = planet_spectrum.wavelength
    flux = planet_spectrum.flux
    flux_HF, flux_LF = filtered_flux(flux,R=R,Rc=Rc,filter_type=filter_type)
    plt.figure(dpi=300) ; plt.plot(wave, flux, 'r') ; plt.xlabel('wavelength (in µm)', fontsize=14) ; plt.xscale('log')
    plt.plot(wave, flux_LF, 'g')  ; plt.ylabel("flux (in photons/s)", fontsize=14) ; plt.title(f'Spectrum of a planet at {planet_spectrum.T}K on {band}', fontsize=14)
    plt.plot(wave, flux_HF, 'b') ; plt.legend(["$S_p$", "[$S_p$]$_{BF}$", "[$S_p$]$_{HF}$"], fontsize=14)
    spectrum = Spectrum(wave, flux, planet_spectrum.R, None)
    spectrum_LF = Spectrum(wave, flux_LF, planet_spectrum.R, None)
    spectrum_HF = Spectrum(wave, flux_HF, planet_spectrum.R, None)
    plt.figure(dpi=300) ; smooth = 1
    plt.yscale('log') ; plt.xscale('log')
    res, psd = spectrum.get_psd(smooth=smooth)
    res, psd_HF = spectrum_HF.get_psd(smooth=smooth)
    res, psd_LF = spectrum_LF.get_psd(smooth=smooth)
    plt.plot(res,psd,label="$PSD(S_p)$")
    plt.plot(res,psd_HF,label="$PSD([S_p]_{HF})$")
    plt.plot(res,psd_LF,label="$PSD([S_p]_{LF})$")
    ymin, ymax = plt.gca().get_ylim() ; plt.ylim(ymin,ymax)
    plt.plot([R, R], [ymin, ymax], 'k-',label="R")
    plt.plot([Rc, Rc], [ymin, ymax], 'k:',label="$R_c$")
    plt.legend()
    plt.title(f"F = {round(psd_LF[np.abs(res-Rc).argmin()]/psd[np.abs(res-Rc).argmin()],2)}")
    plt.show()
    
if 1==0:
    Rc = 100 ; filter_type= "savitzky_golay"
    model = "BT-Settl" ; lg = 4.0
    T_arr, _ = get_model_grid(model)
    F = np.zeros((len(T_arr)))
    lmin = 1.4 ; lmax = 1.8 ; R = 100000
    dl = ((lmin+lmax)/2)/(2*R)
    wave = np.arange(lmin, lmax, dl) # constant and linear wavelength array on the considered band
    R_arr = wave / (2*dl)
    smooth = 1
    cmap = plt.get_cmap("Spectral_r", len(T_arr))
    plt.figure(dpi=300,figsize=(10,5)) ; plt.yscale('log') ; plt.xscale('log') ; plt.xlabel("resolution frequency") ; plt.ylabel("PSD") ; plt.title(f"{filter_type}")
    from matplotlib.cm import ScalarMappable
    for t in tqdm(range(len(T_arr))):
        T = T_arr[t]
        planet_spectrum = load_planet_spectrum(T, lg, model) 
        planet_spectrum.flux *= planet_spectrum.wavelength
        planet_spectrum = planet_spectrum.degrade_resolution(wave, renorm=True) # degradation from spectrum resolution to spectral resolution of the considered band
        planet_HF, planet_LF = filtered_flux(planet_spectrum.flux, R=R, Rc=Rc, filter_type=filter_type)
        res, psd = calc_psd(wave, planet_spectrum.flux, R, smooth=smooth)
        res_LF, psd_LF = calc_psd(wave, planet_LF, R, smooth=smooth)
        res_HF, psd_HF = calc_psd(wave, planet_HF, R, smooth=smooth)
        plt.plot(res_HF, psd_HF, c=cmap(t), alpha=0.5)
        F[t] = psd_LF[np.abs(res_LF-Rc).argmin()]/psd[np.abs(res-Rc).argmin()]
    ymin, ymax = plt.gca().get_ylim() ; plt.ylim(ymin,ymax) ; plt.plot([R, R], [ymin, ymax], 'k-',label="$R_{inst}$") ; plt.plot([Rc, Rc], [ymin, ymax], 'k:',label="$R_c$") ; plt.legend() ; norm = plt.Normalize(vmin=np.nanmin(T_arr), vmax=np.nanmax(T_arr)) ; sm = ScalarMappable(cmap=cmap, norm=norm) ; sm.set_array([]) ; cbar = plt.colorbar(sm, ax=plt.gca()) ; cbar.set_label("planet's temperature (in K)", fontsize=14, labelpad=20, rotation=270) ; plt.show()
    plt.figure(dpi=300) ; plt.title(f"{filter_type}") ; plt.plot([T_arr[0], T_arr[-1]], [0.25, 0.25], "k--") ; plt.plot(T_arr, F, label=f"{round(np.nanmean(F),2)}") ; plt.grid(True) ; plt.ylim(0,1) ; plt.ylabel("F") ; plt.xlabel("planet's temperature") ; plt.legend() ; plt.show()

    


if 1==0: # effet du filtrage sur le bruit (fraction du signal filtré)
    band="1SHORT" ; config_data=get_config_data('MIRIMRS')
    lmin = config_data['gratings'][band].lmin ; lmax = config_data['gratings'][band].lmax # lambda_min/lambda_max de la bande considérée
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    if R is None : # dans le cas où il ne s'agit pas d'un spectro-imageur (eg NIRCAM)
        R = spectrum_instru.R
    delta_lambda = ((lmin+lmax)/2)/(2*R) # 2*R => Nyquist samplé (Shannon)
    wave = np.arange(lmin, lmax, delta_lambda) # axe de longueur d'onde de la bande considérée
    Rc = 100 ; filter_type = "gaussian"
    N = 1000 ; mean_psd = 0. ; mean_psd_HF = 0. ; mean_psd_LF = 0.
    mean_fn_HF = 0. ; mean_fn_LF = 0.
    for i in tqdm(range(N)):
        n = np.random.normal(0, 1, len(wave))
        n_HF, n_LF = filtered_flux(n, R=R, Rc=Rc, filter_type=filter_type)
        mean_fn_HF += np.nansum(n_HF**2) / np.nansum(n**2) / N
        mean_fn_LF += np.nansum(n_LF**2) / np.nansum(n**2) / N
        if i == 0 :
            plt.figure(dpi=300) ; plt.xlabel('wavelength (in µm)', fontsize=14) ; plt.ylabel("flux", fontsize=14)
            plt.plot(wave, n, 'r', label="n")
            plt.plot(wave, n_HF, 'b', label="$[n]_{HF}$")
            plt.plot(wave, n_LF, 'g', label="$[n]_{LF}$")
            plt.legend()
        spectrum=Spectrum(wave, n, R, None)
        spectrum_LF=Spectrum(wave, n_LF, R, None)
        spectrum_HF=Spectrum(wave, n_HF, R, None)
        res, psd = spectrum.get_psd(smooth=0)
        res, psd_HF = spectrum_HF.get_psd(smooth=0)
        res, psd_LF = spectrum_LF.get_psd(smooth=0)
        mean_psd += psd / N
        mean_psd_HF += psd_HF / N
        mean_psd_LF += psd_LF / N
    plt.figure(dpi=300) ; plt.plot(res, mean_psd, 'r', label="n") ; plt.plot(res, mean_psd_HF, 'b', label="$[n]_{HF}$") ; plt.plot(res, mean_psd_LF, 'g', label="$[n]_{LF}$") ; plt.xlabel("resolution frequency R") ; plt.ylabel("PSD") ; plt.xscale('log') ; plt.yscale('log') ; plt.plot([R, R], [np.nanmin(mean_psd_HF)/10, 10*np.nanmax(mean_psd)], 'k-', label="$R_c$") ; plt.plot([Rc, Rc], [np.nanmin(mean_psd_HF)/10, 10*np.nanmax(mean_psd)], 'k:', label="$R_{instru}$") ; plt.ylim(np.nanmin(mean_psd_HF)/10, 10*np.nanmax(mean_psd)) ; plt.legend() ; plt.title(f"F = {round(mean_psd_LF[np.abs(res-Rc).argmin()]/mean_psd[np.abs(res-Rc).argmin()],2)}") ; plt.show()
    print("||[n]_HF||^2/||n||^2 = ", round(100*(np.nansum(mean_psd_HF)/np.nansum(mean_psd)), 1), " & ", round(100*mean_fn_HF, 1))
    print("||[n]_LF||^2/||n||^2 = ", round(100*(np.nansum(mean_psd_LF)/np.nansum(mean_psd)), 1), " & ", round(100*mean_fn_LF, 1))
    fn_HF, fn_LF = get_fraction_noise_filtered(wave, R, Rc, filter_type)
    print("fn_LF = ", round(100*fn_LF, 1))
    print("fn_HF = ", round(100*fn_HF, 1))








if 1==0 : # ouvrir table de planète synthétique
    planet_number, system_id, a, M, R, ecc, inc, stellar, Mcore, Menve, astart, fice, Lint, Ttau = np.loadtxt("syntheticpopJ39_978_ext.dat", unpack=True, skiprows=1)
    plt.figure(dpi=300)
    plt.scatter(M, R, c='k', alpha=0.1) ; plt.xscale('log') ; plt.yscale('log')






if 1==0: # ouvrir spectre Morley (R~10 000 => pas assez...)
    spec = load_planet_spectrum(400, 1, "BT-Settl")
    albedo = 0.0 # 0.0 / 0.3 / 0.7
    path = "Morley2017_models/emission_spectra/alb"+str(albedo)+"/"
    list_files = os.listdir(path)
    planet = "gj1132b"
    planet = "lhs1140b"
    #planet = "trappist1c"
    chem_type = "earth"
    psurf = "psurf1.0"
    for name in list_files :
        if planet in name :
            if chem_type in name :
                if psurf in name :
                    print(name)
                    wave, flux = np.loadtxt(path+name, unpack=True, skiprows=3) # in µm and J/s/m2/m
                    flux /= 1e6 # in J/s/m2/µm
                    argsort = np.argsort(wave)
                    wave = wave[argsort]
                    flux = flux[argsort]
                    plt.figure(dpi=300)
                    plt.plot(spec.wavelength, spec.flux)
                    plt.plot(wave, flux)
                    plt.title(name)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.show()
                    dwl = wave - np.roll(wave, 1) ; dwl[0] = dwl[1] # array de delta Lambda
                    R = wave/(2*dwl) # calcule de la nouvelle résolution
                    plt.figure(dpi=300)
                    plt.plot(wave, R)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.show()
    



if 1==0 :  # ouvrir spectre moléculaire 
    #sigma_cosmic = 1
    config_data = get_config_data("MIRIMRS")
    band = '1SHORT'
    R = config_data['gratings'][band].R # Résolution spectrale de la bande considérée 
    Rc = 100 ; filter_type = "gaussian"
    mol = load_planet_spectrum(200, None, "mol_O2")
    #mol_HF, mol_LF = filtered_flux(mol.flux, R=mol.R, Rc=Rc, filter_type=filter_type)
    #sg = sigma_clip(mol_LF, sigma=sigma_cosmic)
    #mol.flux[~sg.mask] = np.nan
    plt.figure(dpi=300)
    plt.plot(mol.wavelength, mol.flux)
    #plt.plot(mol.wavelength, mol_HF)
    #plt.plot(mol.wavelength, mol_LF)
    plt.show()
    
    mol = spectrum_band(config_data, band, mol)
    planet = load_planet_spectrum(1600, 4, "BT-Settl")
    planet = spectrum_band(config_data, band, planet)
    mol_HF, _ = filtered_flux(mol.flux, R, Rc, filter_type)
    planet_HF, _ = filtered_flux(planet.flux, R, Rc, filter_type)
    plt.figure(dpi=300)
    plt.plot(mol.wavelength, mol_HF/np.sqrt(np.nansum(mol_HF**2))) ; plt.plot(planet.wavelength, planet_HF/np.sqrt(np.nansum(planet_HF**2)))
    plt.show()
    cos = np.nansum(planet_HF*mol_HF)/(np.sqrt(np.nansum(mol_HF**2))*np.sqrt(np.nansum(planet_HF**2)))








