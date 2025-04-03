import numpy as np 
import pandas as pd
from scipy import signal
import scipy.fft as sf 


from seidart.routines import sourcefunction, prjrun
from seidart.routines.arraybuild import Array 
from seidart.routines.definitions import * 
from seidart.visualization.im2anim import build_animation

def rmawhiten(ts, rma_window = 8):
    # Compute the running mean average whitening routine as referenced in 
    # 
    # INPUTS
    #   ts (FLOAT) - the m-by-1 time series that will be whitened
    #   k (INT) - the normalization window 
    #   freq (BOOL) - whether to return the values in the frequency domain or the 
    m = len(ts)
    
    # Taper 
    taperwin = signal.windows.tukey(m, 0.3)
    ts = taperwin * ts 
    
    # Pad the timeseries before taking the fft
    pad = np.zeros(ts.shape)
    ts = np.concat([np.zeros(ts.shape), ts])
    
    # Compute the fft 
    U = np.fft.fft(ts)
    Uw = U[:]
    # Normalize using the running mean
    Uw.real = agc(Uw.real, rma_window, 'mean')
    
    return(U, Uw)

# -----------------------------------------------------------------------------
def acf(timeseries, agc_window = 16, window_length = 2048, noverlap = 512):
    # AGC 
    ts = agc(timeseries, agc_window, 'mean')
    n = len(ts)
    k = 0
    ind1 = 0 
    ind2 = ind1 + window_length
    # Split, taper, pad, then whiten
    while ind2 < n:
        k += 1
        ind1 += noverlap 
        ind2 += noverlap
    
    if ind2 > n:
        k -= 1
    
    autocoher = np.zeros([window_length])
    # padding = split_ts.copy() 
    for ind in range(k):
        shift = noverlap*k
        U, Uw = rmawhiten( ts[shift:(window_length+shift)] )
        # Compute the auto cross-coherence
        C = (np.conj(U)*U)/(Uw*Uw)
        autocoher += (np.real(np.fft.ifft(C))[window_length:])[::-1]
    
    autocoher = autocoher / k 
    autocoher = np.real(np.fft.ifft(autocoher))[::-1]
    return autocoher

# -----------------------------------------------------------------------------
def mwca(ts1, ts2, dt, smoothwin = 1+2**2, freqmin = 5e7, freqmax = 2e8):
    if len(ts1) != len(ts2):
        return
    
    n = len(ts1)
    k = 0
    ind1 = 0 
    ind2 = ind1 + window_length
    # Split, taper, pad, then whiten
    while ind2 < n:
        k += 1
        ind1 += noverlap 
        ind2 += noverlap
    
    if ind2 > n:
        k -= 1
    
    for jj in range(0,k):
    shift = noverlap*jj,
    deltat[jj], dterr[jj], dcoh[jj] = cross_spec(
            ts1[shift:(window_length+shift)],
            ts2[shift:(window_length+shift)],
            smoothwin,
            freqmin,freqmax,
            dt
        )
    
    # Compute the statistics for each day
    sortind = np.argsort(rttime)
    dt_median = np.median(deltat)
    dt_mean = np.mean(deltat)
    dt_std = np.std(deltat)
    
    # remove values that don't satisfy what the coherency and dt error
    t_dt, t_dterr, t_dcoh, twin = higherr_lowcoh(
        deltat, dterr, dcoh
    )
    
    # Change to dvv and save outputs
    dv = -t_dt[:]
    dv_std = -tdt_std[:]
    dv_mean = -tdt_mean[:]
    dv_median = -tdt_median[:]
    return dv, dv_mean, dv_median, dv_std


# -----------------------------------------------------------------------------
# Define some necessary values 
prjfile = 'single_medium.prj' 
receiverfile = 'receivers.xyz'

domain, material, seismic, electromag = prjrun.domain_initialization(prjfile)
timevec, fx, fy, fz, srcfn = sourcefunction(electromag, 1e8, 'gaus1')

prjrun.status_check(
    electromag, material, domain, prjfile, append_to_prjfile = True
)

# wavenumber bandlimited value must be checked after computing the tensor
kband_check(electromag, domain)

# -----------------------------------------------------------------------------
prjrun.runelectromag(
    electromag, 
    material, 
    domain, 
    use_complex_equations = False
)

# frame_delay = 10
# frame_interval = 20
# alpha_value = 0.3 

# build_animation(
#         prjfile, 
#         'Ex', frame_delay, frame_interval, alpha_value, 
#         is_complex = False, 
#         is_single_precision = True
# )

ref_receiver = Array('Ex', prjfile, receiverfile, is_complex = False)
ref_coher = acf(ref_receiver.timeseries.copy())
# -----------------------------------------------------------------------------
lwc = np.arange(0,100)
porosity = np.arange(0,100)*0.7

m = len(lwc)
n = len(porosity)

snow_ind = 1 

coher = np.zeros([ref_coher.shape, m, n])
ts = np.zeros([ref_receiver.shape, m, n])

for ii in range(2): #range(m):
    material.lwc[snow_ind] = lwc[ii] 
    for jj in range(2): #range(n):
        print([ii,jj])
        material.porosity[snow_ind] = lwc[jj]
        prjrun.status_check(
            electromag, material, domain, prjfile, append_to_prjfile = False
        )
        prjrun.runelectromag(
            electromag, 
            material, 
            domain, 
            use_complex_equations = False
        )
        arr = Array('Ex', prjfile, rcxfile, is_complex = False)
        mwca(
            ref_receiver.timeseries.copy(), arr.timeseries.copy(),
            electromag.dt
        )
        # ts[:,ii,jj]
        # coher[:,ii,jj] = acf(ts[:,ii,jj])
         


# -----------------------------------------------------------------------------