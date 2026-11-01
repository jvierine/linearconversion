#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import matplotlib.dates as mdates

fl=glob.glob("results3/cs*.h5")
fl.sort()
nf=len(fl)
S=n.zeros([nf,5000],dtype=n.complex)
P=n.zeros([nf,5000],dtype=n.float)
count=n.zeros([nf,5000],dtype=n.float)
tv=n.zeros(nf)
rv=n.zeros(5000)
tv2=n.arange(nf)
rv=n.arange(5000)*0.3
low_idx=n.where(rv < 80.0)[0]
high_idx=n.where(rv >= 80.0)[0]
ai=0
for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")

    S[fi,:]=h["acf"].value[:,0]
    lags=h["lags"].value[0:5]
    for ri in range(200,300):
        acf=(1.0/3.0)*(h["acf"].value[ri-1,0:5]+h["acf"].value[ri,0:5]+h["acf"].value[ri+1,0:5])
        lags[0]=22.0
        p=n.polyfit(n.log10(lags),acf.real,1)
        pf=n.poly1d(p)
        S[fi,ri]=pf(n.log10(lags[0]))
#        plt.plot(n.log10(lags),acf.real)
 #       plt.plot(n.log10(lags),pf(n.log10(lags)))
  #      plt.title(ri*0.3)
   #     plt.show()
        

    
    tv[fi]=h["t0"].value
    h.close()

#tv=tv/float(n_avg)
#plt.plot(tv,".")
#plt.show()
idx=n.argsort(tv)

S=S[idx,:]
tv=tv[idx]

alpha=42859649122.80702
for ri in range(S.shape[1]):
    S[:,ri]=alpha*S[:,ri]*rv[ri]**2.0

dB=n.log10(n.abs(S.real))
import stuffr
dts=[]
for t in tv:
    print(t)
    dts.append(stuffr.unix2date(t-4*3600))
print(len(dts))
print(len(tv))
fig, ax = plt.subplots()

c=ax.pcolormesh(dts,rv,n.transpose(dB),vmin=8,vmax=12,cmap="plasma")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#ax.fmt_xdata = mdates.DateFormatter("%H:%M")
fig.autofmt_xdate()
ax.set_title("Arecibo D- and E-region electron density\n Starting %s"%(stuffr.unix2datestr(n.min(tv)-4*3600)))
ax.set_ylabel("Altitude (km)")

#ax.set_title('fig.autofmt_xdate fixes the labels')

ax.set_xlabel("Time (AST)")
#plt.pcolormesh(n.transpose(dB),vmin=-60.0,vmax=-28,cmap="jet")

ax.set_ylim([50,140])
fig.colorbar(c,ax=ax,label="$\log_{10}(N_e)$")
plt.savefig("arecibo_d_e.png")
plt.show()
