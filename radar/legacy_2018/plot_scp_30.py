#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob


fl=glob.glob("/data2/processings/results_30m/cs*.h5")
fl.sort()
nf=len(fl)
S=n.zeros([nf,5120],dtype=n.complex)
tv=n.zeros(nf)
rv=n.zeros(5120)
tv2=n.arange(nf)

for fi,f in enumerate(fl):
    print(f)
    h=h5py.File(f,"r")
    print(h.keys())
    S[fi,:]=h["acf"].value[:,0]

    tv[fi]=h["t0"].value

    h.close()

#plt.plot(tv,".")
#plt.show()
idx=n.argsort(tv)[::-1]
S=S[idx,:]
tv=tv[idx]
rv=n.arange(5120)*0.15/2.5
dB=10.0*n.log10(n.abs(S.real))

plt.pcolormesh(tv,rv,n.transpose(dB),vmin=-60.0,vmax=-30,cmap="jet")
plt.ylim([30,140])
plt.colorbar()
plt.show()
