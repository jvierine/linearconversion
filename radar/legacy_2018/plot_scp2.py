#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob


lags=n.array([0,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500],dtype=n.int)

fl=glob.glob("results3/cs*.h5")
fl.sort()
nf=len(fl)
S=n.zeros([nf,5000],dtype=n.complex)
tv=n.zeros(nf)
rv=n.zeros(5000)
tv2=n.arange(nf)

for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
    print(h.keys())
    print(h["lags"].value)
#    S[fi,:]=h["acf"].value[:,1]/n.real(h["acf"].value[:,0])

 #   tv[fi]=h["t0"].value

    h.close()

#plt.plot(tv,".")
#plt.show()
idx=n.argsort(tv)[::-1]
S=S[idx,:]
rv=n.arange(5000)*0.3
#scale=n.nanmean(n.abs(S))

plt.pcolormesh(tv,rv,n.transpose(S.real),vmin=0,vmax=1.0,cmap="jet")
plt.ylim([30,150])
plt.colorbar()
plt.show()
