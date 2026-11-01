#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import matplotlib
import h5py
import glob
import stuffr

fl_csp=glob.glob("results3/csp_acf*.h5")
fl_clp=glob.glob("results3/clp_acf*.h5")
fl_csp.sort()
fl_clp.sort()
fl=fl_csp+fl_clp
nf=len(fl)
tv=n.zeros(nf)
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    t0 = h["t0"].value
    tv[fi]=t0
    h.close()


tv_idx=n.argsort(tv)
fl2=[]
for fi in range(len(tv_idx)):
    fl2.append(fl[tv_idx[fi]])
    

n_avg=8*4

rv=n.zeros(5000)

ai=0

lags=n.array([])


fl = fl2

for fi,f in enumerate(fl[0:20]):
    h=h5py.File(f,"r")
    file_lags=h["lags"].value
    if len(file_lags) == 6:
        lags = n.concatenate([lags,h["lags"].value[0:5]])
    else:
        lags = n.concatenate([lags,h["lags"].value])
    h.close()
    
lags=n.concatenate([lags,n.arange(430,1000,10)])    
lags=n.array(n.unique(lags),dtype=n.int)

plot_lags=n.copy(lags)
plot_lags[0]=4.0


print("Unique lags",lags)
n_lags=len(lags)

S=n.zeros([10,5000,n_lags],dtype=n.complex)
count=n.zeros([10,5000,n_lags],dtype=n.float)

rgs=0.3*n.arange(5000)
#e_idx0=n.argmin(n.abs(rgs-90))
#e_idx1=n.argmin(n.abs(rgs-94))
#e_idx2=n.argmin(n.abs(rgs-98))
#e_idx3=n.argmin(n.abs(rgs-100))

ai=0
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(h.keys())
    flags=n.array(h["lags"].value,dtype=n.int)
    if len(flags)==6:
        flags=flags[0:5]
        
    for fli,flag in enumerate(flags):
        lagi = n.where(lags == flag)[0]
        print("fi %d flag %f idx %d"%(fli,flag,lagi))
        acf_lag=h["acf"].value[:,fli]
        if (flag > 0) and (flag < 500):
            acf_lag[0:int(flag*0.5+190*0.5)]=n.nan
        S[ai,:,lagi]+=acf_lag
        count[ai,:,lagi]+=1.0
 #       plt.plot(acf_lag.real)
  #      plt.show()

    h.close()
    
acfm=n.copy(S[ai,:,:]/count[ai,:,:])
acfmn=n.copy(acfm)
clp_lag_idx=n.where( (lags > 0) & (lags < 500))[0]
for ri in range(acfm.shape[0]):
    acfmn[ri,:]=acfm[ri,:]/n.real(acfm[ri,0])
    if ri*0.3 < 94:
        acfmn[ri,clp_lag_idx]=n.nan

plotacfmn = n.ma.masked_where(n.isnan(acfmn), acfmn)    
plt.pcolormesh(n.log10(1e-6*plot_lags),0.3*n.arange(5000),plotacfmn.real,vmin=-0.2,vmax=1.2,cmap="gist_ncar")
plt.title("Normalized ACF (real component)")
plt.xlabel("Lag ($\log_{10}(s)$)")
plt.ylabel("Altitude (km)")
plt.ylim([50,140])
#plt.xlim([0,4000])        
plt.colorbar()
plt.show()

for pri in range(85,140):
    avg_acf=n.copy(acfmn[0,:])
    avg_acf[:]=0.0
    pri_idx=n.where(n.abs(rgs-pri)<1.0)[0]
    for pri2 in pri_idx:
        avg_acf+=acfmn[pri2,:]
    if pri < 94:
        avg_acf[clp_lag_idx]=n.nan
   # clp_factor=1.0
  #  if pri < 100:
#        clp_factor=avg_acf[0].real/avg_acf[1].real
 #       avg_acf[clp_lag_idx]=avg_acf[clp_lag_idx]*clp_factor
    
    plt.plot(plot_lags,avg_acf.real/float(len(pri_idx))+pri,".",color="blue")
#    plt.plot(plot_lags,avg_acf.imag/float(len(pri_idx))+pri,".",color="green")    
    plt.axhline(pri,color="gray")
plt.xlim([0,4000])
plt.title("Normalized ACF (real component)")
plt.xlabel("Lag ($\mu s$)")
plt.ylabel("Altitude (km)")

plt.show()

print(lags)
