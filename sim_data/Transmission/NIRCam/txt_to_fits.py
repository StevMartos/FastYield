import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from astropy.io import fits
import matplotlib.pyplot as plt


band="F444W"
file="mean_throughputs/"+band+"_mean_system_throughput.txt"

with open(file) as f:
    contents = f.readlines()

wave=np.array([])
trans=np.array([])
for i in range(1,len(contents)):
    wave=np.append(wave,float(contents[i].split()[0]))
    trans=np.append(trans,float(contents[i].split()[1]))

plt.plot(wave,trans,"r") ; plt.ylabel('throughput') ; plt.xlabel("wavelength (in µm)") ; plt.title("Total system throughput of "+band)



if 1==1:
    file="ModA_COM_Substrate_Transmission_20151028_JKrist.dat"
    with open(file) as f:
        contents = f.readlines()
    wave_COM=np.array([])
    trans_COM=np.array([])
    for i in range(1,len(contents)):
        wave_COM=np.append(wave_COM,float(contents[i].split()[0]))
        trans_COM=np.append(trans_COM,float(contents[i].split()[1]))
        
    #plt.figure() ; plt.plot(wave_COM,trans_COM)
    
    f = interp1d(wave_COM, trans_COM, bounds_error=False, fill_value=0)
    trans_COM = f(wave)
    trans *= trans_COM
    
    plt.plot(wave,trans,"b") ; plt.legend(["without COM","with COM"])



A=np.zeros((2,np.size(wave)))
A[0,:]=wave
A[1,:]=trans

fits.writeto("transmission_"+band+".fits",A,overwrite=True)





