#!/usr/bin/env

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob


fl=glob.glob("results3/cs*.h5")

nf=len(fl)
S=n.zeros([nf,5000],dtype=n.complex)
tv=n.zeros(nf)
rv=n.zeros(5000)
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
rv=n.arange(5000)*0.3
dB=10.0*n.log10(n.abs(S.real))

Z=n.copy(S)
for ri in range(S.shape[1]):
    Z[:,ri]=Z[:,ri]-n.median(Z[:,ri])
plt.pcolormesh(tv,rv,n.transpose(10.0*n.log10(Z)),vmin=-60,vmax=-30)
plt.colorbar()
plt.ylim([20,150])
plt.show()

plt.pcolormesh(tv,rv,n.transpose(dB),vmin=-60.0,vmax=-28,cmap="jet")

plt.ylim([30,700])
plt.colorbar()
plt.show()
