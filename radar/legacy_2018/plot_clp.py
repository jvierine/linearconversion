#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import stuffr

fl=glob.glob("results3/clp*.h5")
fl.sort()
nf=len(fl)
S=n.zeros([nf,5000],dtype=n.complex)
tv=n.zeros(nf)
rv=n.zeros(5000)
tv2=n.arange(nf)
for fi,f in enumerate(fl):
    print(fi)
    h=h5py.File(f,"r")
    print(h.keys())
    S[fi,:]=h["acf"].value[:,0]
    tv[fi]=h["t0"].value
    h.close()

idx=n.argsort(tv)[::-1]
S=S[idx,:]
tv=tv[idx]

#plt.plot(tv,".")
#plt.show()
print(n.diff(tv))
#tv2=n.arange(nf)
rv=n.arange(5000)*0.3
dB=10.0*n.log10(n.abs(S.real))
#dB=dB-n.median(dB)
plt.pcolormesh(tv,rv,n.transpose(dB),vmin=-60.0,vmax=-28)

plt.ylim([30,700])
plt.colorbar()
plt.show()
