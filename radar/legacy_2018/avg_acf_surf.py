#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import matplotlib
import h5py
import glob
import stuffr
import matplotlib.dates as mdates

fl_csp=glob.glob("results3/csp_acf*.h5")
fl_clp=[]#glob.glob("results3/clp_acf*.h5")
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
    

n_avg=7

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
    
lags=n.array(n.unique(lags),dtype=n.int)
fit_lags=n.copy(lags)
fit_lags[0]=22.0
print(fit_lags)


print("Unique lags",lags)
n_lags=len(lags)

n_t=int(len(fl)/float(n_avg))+1
S=n.zeros([n_t,5000,n_lags],dtype=n.complex)
P=n.zeros([n_t,5000],dtype=n.float)
count=n.zeros([n_t,5000,n_lags],dtype=n.float)

rgs=0.3*n.arange(5000)
tv=tv[tv_idx]
tv2=n.zeros(n_t)
ai=0
t0=tv[0]
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    print(h.keys())
    flags=n.array(h["lags"].value,dtype=n.int)
    if len(flags)==6:
        flags=flags[0:5]
        
    for fli,flag in enumerate(flags):
        lagi = n.where(lags == flag)[0][0]
        print("fi %d flag %f idx %d"%(fli,flag,lagi))
        acf_lag=h["acf"].value[:,fli]
        if (flag > 0) and (flag < 500):
            acf_lag[0:int(flag*0.5+190*0.5)]=n.nan
            
        S[ai,:,lagi]+=acf_lag
        count[ai,:,lagi]+=1.0
    tv2[ai]=h["t0"].value
    h.close()
    if (fi+1)%n_avg==0:
        acfm=n.copy(S[ai,:,:]/count[ai,:,:])


        
        for ri in range(500):
            acf_est=acfm[ri,:]
            if ri < 266 and ri > 2:
                acf_est=1.0/3.0*(acfm[ri-1,:]+acfm[ri,:]+acfm[ri+1,:])
            
            if ri < 500:
                polyord=1
                if ri > 266:
                    polyord=2

                sigma2=2.0*n.median(n.abs(acfm[ri,(len(lags)/2):len(lags)]))                
                fit_sigma=sigma2*n.sqrt(1.0/count[ai,0,:])#+sigma2

                fit_weight=1.0/fit_sigma

                p=n.polyfit(n.log10(fit_lags),acf_est.real,polyord,w=fit_weight)
                pf=n.poly1d(p)
                P[ai,ri]=pf(n.log10(fit_lags[0]))
#                if ri > 220:
 #                   plt.plot(n.log10(fit_lags),pf(n.log10(fit_lags)))
  #                  plt.plot(n.log10(fit_lags),acf_est.real,"x",color="black")
   #                 plt.plot(n.log10(fit_lags),acf_est.real+fit_sigma,color="grey")
    #                plt.plot(n.log10(fit_lags),acf_est.real-fit_sigma,color="grey")
     #               plt.title("R %1.2f %d"%(ri*0.3,ri))
      #              plt.show()
            else:
                P[ai,ri]=acfm[ri,0].real
#            if ri < 266:
 #               P[ai,ri]=n.mean(acfm[ri,:].real)
            

 #       plt.plot(n.log10(P[ai,0:500]),n.arange(500)*0.3)
 #       plt.show()
        t0=tv[fi]
#        tv2[ai]=t0
        
        ai+=1


Ne=n.copy(P[:,0:500])
alpha=42859649122.80702
rv=n.arange(500)*0.3
for ri in range(500):
    Ne[:,ri]=alpha*P[:,ri]*rv[ri]**2.0
#for ti in range(n_t):
 #   plt.plot(n.log10(Ne[ti,:]),rv,".")
  #  plt.xlim([5,12])
   # plt.show()

    
tv2=tv2[0:(n_t-1)]
Ne=Ne[0:(n_t-1),:]
dts=[]
dts_utc=[]
for t in tv2:
    print(t)
    dts.append(stuffr.unix2date(t-4*3600))
    dts_utc.append(stuffr.unix2datestr(t))
        
#plt.pcolormesh(tv2[0:(n_t-1)],rv,n.transpose(n.log10(Ne[0:(n_t-1),:])),vmin=8,vmax=12,cmap="viridis")
#plt.ylim([50,140])
#plt.colorbar()
#plt.show()
ho=h5py.File("ne_result.h5","w")
ho["Ne_m3"]=Ne
#ho["dates_AST"]=dts
ho["dates_UTC"]=dts_utc
ho["unix_seconds"]=tv2
ho["range_km"]=rv
ho.close()



fig, ax = plt.subplots()
nedB=n.transpose(n.log10(Ne))
c=ax.pcolormesh(dts,rv,nedB,vmin=8.0,vmax=12.0,cmap="nipy_spectral")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#ax.fmt_xdata = mdates.DateFormatter("%H:%M")
fig.autofmt_xdate()
ax.set_title("Arecibo D- and E-region electron density\n Starting %s"%(stuffr.unix2datestr(n.min(tv2)-4*3600)))
ax.set_ylabel("Altitude (km)")


ax.set_xlabel("Time (AST)")
#plt.pcolormesh(n.transpose(dB),vmin=-60.0,vmax=-28,cmap="jet")

ax.set_ylim([50,140])
fig.colorbar(c,ax=ax,label="$\log_{10}(N_e)$")
plt.savefig("arecibo_d_e.png")
plt.show()

